from _input_utils import TextChunkSpliter
from alive_inference_status import ALRealtimeIOAndStatus
import logging
from pathlib import Path
import sys
import threading
import os
from openai import OpenAI
import wave
import array
import time
import requests
import json
# import pyaudio
from cartesia import Cartesia
import psutil
import numpy as np
import queue
import threading
import socket

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class AudioPlayer:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.pyaudio_obj = pyaudio.PyAudio()
        self.pyaudio_out_stream = None

    def start_stream(self):
        if not self.pyaudio_out_stream:
            self.pyaudio_out_stream = self.pyaudio_obj.open(
                format=pyaudio.paInt16, channels=1, rate=self.sample_rate, output=True
            )

    def play_buffer(self, buffer):
        if not self.pyaudio_out_stream:
            self.start_stream()
        self.pyaudio_out_stream.write(buffer)

    def close(self):
        if self.pyaudio_out_stream:
            self.pyaudio_out_stream.stop_stream()
            self.pyaudio_out_stream.close()
        self.pyaudio_obj.terminate()


class TTS_Manager:
    def __init__(
        self,
        status: ALRealtimeIOAndStatus = None,
        listen_locally=False,
        audio_play_locally=False,
        tts_voice_id_cartesia: str = None,
    ):
        # Initialize Cartesia client
        self.cartesia = Cartesia(api_key=os.getenv("CARTESIA_API_KEY"))
        if not self.cartesia:
            raise ValueError("CARTESIA_API_KEY environment variable is not set")
        self.context_id = "local_test"
        self.voice_id = tts_voice_id_cartesia

        self.status = status
        self.audio_play_locally = audio_play_locally
        if self.audio_play_locally:
            self.audio_player = AudioPlayer()
        else:
            self.audio_player = None

        # Initialize text chunk splitter
        self.text_chunk_spliter = TextChunkSpliter()

        # Timing information
        self.timing = {
            "speech_end_time": -1,
            "llm_first_token_time": -1,
            "tts_first_chunk_time": -1,
        }
        self.user_interrupting_flag = False

        self.listen_locally = listen_locally
        if self.listen_locally:
            self.local_port = 2000
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind(("localhost", self.local_port))
                self.server_socket.listen(
                    1
                )  # Listen for one incoming connection (e.g., from ASR/LLM module)
                logger.info(
                    f"TTS_Manager server listening for text input on localhost:{self.local_port}"
                )

                self.local_listener_thread = threading.Thread(
                    target=self.listen_text_input_locally, daemon=True
                )
                self.local_listener_thread.start()
            except Exception as e:
                logger.error(
                    f"Failed to start local text input server: {e}", exc_info=True
                )
                self.server_socket = None  # Ensure it's None if setup fails
        else:
            # add webrtc logic here
            pass

        # TTS worker
        self.ws = self.cartesia.tts.websocket()
        self.tts_queue = queue.Queue()
        self.tts_thread = None

    def listen_text(self):
        if self.listen_locally:
            self.listen_text_input_locally()
        else:
            self.listen_text_input_webrtc()

    def listen_text_input_webrtc(self):
        pass

    def listen_text_input_locally(self):
        if not self.server_socket:
            logger.error(
                "Local text input server socket not initialized. Cannot listen."
            )
            return

        try:
            while True:  # Loop to continuously accept new connections
                client_socket, addr = (
                    self.server_socket.accept()
                )  # no need for sleep, blocking call
                buffer = ""
                try:
                    while True:  # Loop to receive data from the connected client
                        data = client_socket.recv(1024)  # Read 1024 bytes
                        if not data:
                            # Current transmission is done
                            break

                        buffer += data.decode("utf-8")

                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if line:
                                self._handle_incoming_text(line)
                except socket.error as e:
                    # Log common socket errors like connection reset
                    logger.warning(f"TTS Manager: Socket error with client {addr}: {e}")
                except Exception as e_client:
                    logger.error(
                        f"TTS Manager: Error handling client {addr}: {e_client}",
                        exc_info=True,
                    )
                finally:
                    if buffer.strip():  # Process any remaining data in buffer
                        remaining_line = buffer.strip()
                        self._handle_incoming_text(remaining_line)
                    client_socket.close()

        except (
            OSError
        ) as e:  # Handle errors like socket closed if server_socket is closed by __del__
            if (
                e.errno == 9
            ):  # [Errno 9] Bad file descriptor - typically means socket was closed
                logger.info(
                    "TTS Manager: Local text input server socket was closed. Stopping listener."
                )
            else:
                logger.error(
                    f"TTS Manager: Local text input server loop OS error: {e}",
                    exc_info=True,
                )
        except Exception as e_server:
            logger.error(
                f"TTS Manager: Local text input server loop crashed: {e_server}",
                exc_info=True,
            )

    def _handle_incoming_text(self, text):
        text = text.strip()
        if text == "[START]":
            # Stop last tts thread
            if self.tts_thread:
                while not self.tts_queue.empty():
                    self.tts_queue.get()
                self.tts_queue.put(None)  # force thread quit
                self.tts_thread.join()
                self.tts_thread = None
            # Initialize tts for this call
            self.tts_queue = queue.Queue()
            self.tts_thread = None
            self.user_interrupting_flag = False
            # Start TTS worker thread
            self.tts_thread = threading.Thread(
                target=self._tts_worker, args=(), daemon=True
            )
            self.tts_thread.start()
        elif text == "[INTERRUPTED]":
            self.user_interrupting_flag = True
        elif text == "[DONE]":
            # --- Wait for TTS Worker ---
            self.tts_queue.put(None)  # Send sentinel to worker thread
            if self.tts_thread:
                self.tts_thread.join()  # Wait for the worker thread to exit
                self.tts_thread = None
        elif text.startswith("[speech_end_time]"):
            self.timing["speech_end_time"] = float(text.split(":")[1].strip())
        elif text.startswith("[llm_first_token_time]"):
            self.timing["llm_first_token_time"] = float(text.split(":")[1].strip())
        else:
            assert (
                self.tts_thread is not None and self.tts_queue is not None
            ), "TTS thread or queue not initialized"
            self.tts_queue.put(text)

    def _tts_worker(self):
        """Worker thread function to process TTS queue."""
        if not self.status.get_is_voice_transmitting():
            self.status.set_voice_transmit_started()
        while not self.user_interrupting_flag:
            try:
                text_chunk = self.tts_queue.get(
                    block=True  # Blocks until an item is available
                )
                if text_chunk is None:  # Sentinel value received
                    self.tts_queue.task_done()
                    break
            except Exception as e_queue:
                logger.error(
                    f"TTS worker: Error getting item from queue: {e_queue}",
                    exc_info=True,
                )
                break

            # --- Core TTS Streaming Logic ---
            try:
                # This loop blocks until audio for *this* chunk is received
                for output in self.ws.send(
                    model_id="sonic-2",
                    transcript=text_chunk,
                    voice={"id": self.voice_id},
                    stream=True,
                    context_id=self.context_id,
                    output_format={
                        "container": "raw",
                        "encoding": "pcm_s16le",
                        "sample_rate": 16000,
                    },
                ):
                    if not output.audio:
                        continue

                    # First chunk timing (thread-safe enough for logging)
                    if self.timing["tts_first_chunk_time"] == -1:
                        self.timing["tts_first_chunk_time"] = time.time()
                        if self.timing["llm_first_token_time"] != -1:
                            logger.info(
                                f"Time from LLM first token to TTS first chunk: {self.timing['tts_first_chunk_time'] - self.timing['llm_first_token_time']:.2f} seconds"
                            )
                        if self.timing["speech_end_time"] != -1:
                            logger.info(
                                f"Time from user speech end to first audio: {self.timing['tts_first_chunk_time'] - self.timing['speech_end_time']:.2f} seconds"
                            )

                    # Process audio
                    audio_np = np.frombuffer(output.audio, dtype=np.int16).reshape(
                        1, -1
                    )
                    self.status.audio_source_wavform_queue.put(audio_np)
                    if self.audio_play_locally and self.audio_player:
                        self.audio_player.play_buffer(output.audio)

            except Exception as e_tts:
                logger.error(
                    f"TTS worker: Error during Cartesia send/recv for chunk '{text_chunk}...': {e_tts}",
                    exc_info=True,
                )
                continue

            except:
                logger.error(
                    f"TTS worker: Skipping this chunk due to Error. Error: {sys.exc_info()[0]}",
                    exc_info=True,
                )
                continue

            finally:
                self.tts_queue.task_done()  # Signal that this task is done

        # Set status after all processing and cleanup attempts
        self.status.set_voice_transmit_completed()
        self.timing["tts_first_chunk_time"] = -1
        while not self.tts_queue.empty():
            self.tts_queue.get()

    def __del__(self):
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.error(f"Error closing Cartesia websocket: {e}")

        if (
            hasattr(self, "audio_player")
            and self.audio_play_locally
            and self.audio_player
        ):
            try:
                self.audio_player.close()
            except Exception as e:
                logger.error(f"Error closing audio player: {e}")

        if hasattr(self, "server_socket") and self.server_socket:
            try:
                logger.info("Closing TTS_Manager server socket.")
                self.server_socket.close()
                self.server_socket = None
            except Exception as e:
                logger.error(f"Error closing server socket in TTS_Manager: {e}")
