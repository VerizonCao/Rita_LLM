from _input_utils import TextChunkSpliter
from alive_inference_status import ALRealtimeIOAndStatus
import logging
from pathlib import Path
import sys
import threading
import os
import time
import numpy as np
import queue
from cartesia import Cartesia
import pyaudio

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
        self._lock = threading.Lock()

    def start_stream(self):
        with self._lock:
            if not self.pyaudio_out_stream:
                self.pyaudio_out_stream = self.pyaudio_obj.open(
                    format=pyaudio.paInt16, channels=1, rate=self.sample_rate, output=True
                )

    def play_buffer(self, buffer):
        with self._lock:
            if not self.pyaudio_out_stream:
                self.start_stream()
            self.pyaudio_out_stream.write(buffer)

    def close(self):
        with self._lock:
            if self.pyaudio_out_stream:
                try:
                    self.pyaudio_out_stream.stop_stream()
                    self.pyaudio_out_stream.close()
                except Exception as e:
                    logger.error(f"Error closing audio stream: {e}")
            try:
                self.pyaudio_obj.terminate()
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")

class TTS_Manager:
    def __init__(
        self,
        status: ALRealtimeIOAndStatus = None,
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

        # TTS worker
        self.ws = None
        self.tts_queue = queue.Queue()
        self.tts_thread = None
        self.is_processing = False
        self._lock = threading.Lock()

    def _initialize_websocket(self):
        """Initialize the websocket connection if not already initialized"""
        if self.ws is None:
            try:
                self.ws = self.cartesia.tts.websocket()
            except Exception as e:
                logger.error(f"Failed to initialize Cartesia websocket: {e}")
                raise

    def _handle_incoming_text(self, text):
        text = text.strip()
        if text == "[START]":
            logger.info("Starting new TTS stream")
            with self._lock:
                # Stop last tts thread if it exists
                if self.tts_thread and self.tts_thread.is_alive():
                    while not self.tts_queue.empty():
                        self.tts_queue.get()
                    self.tts_queue.put(None)  # force thread quit
                    self.tts_thread.join()
                    self.tts_thread = None
                
                # Initialize new stream
                self.tts_queue = queue.Queue()
                self.tts_thread = None
                self.user_interrupting_flag = False
                self.is_processing = True
                
                # Initialize websocket if needed
                try:
                    self._initialize_websocket()
                except Exception as e:
                    logger.error(f"Failed to initialize websocket: {e}")
                    self.is_processing = False
                    return
                
                # Start TTS worker thread
                self.tts_thread = threading.Thread(
                    target=self._tts_worker, args=(), daemon=True
                )
                self.tts_thread.start()
                logger.info("TTS worker thread started")
            
        elif text == "[INTERRUPTED]":
            logger.info("TTS stream interrupted")
            with self._lock:
                self.user_interrupting_flag = True
            
        elif text == "[DONE]":
            logger.info("TTS stream completed")
            with self._lock:
                if self.is_processing:
                    self.is_processing = False
                    # Send sentinel to worker thread
                    self.tts_queue.put(None)
                    if self.tts_thread:
                        self.tts_thread.join()
                        self.tts_thread = None
                    
        elif text.startswith("[speech_end_time]"):
            try:
                self.timing["speech_end_time"] = float(text.split(":")[1].strip())
            except (IndexError, ValueError) as e:
                logger.error(f"Error parsing speech end time: {e}")
            
        elif text.startswith("[llm_first_token_time]"):
            try:
                self.timing["llm_first_token_time"] = float(text.split(":")[1].strip())
            except (IndexError, ValueError) as e:
                logger.error(f"Error parsing LLM first token time: {e}")
            
        else:
            with self._lock:
                if not self.is_processing:
                    logger.warning("Received text chunk but TTS stream not started")
                    return
                    
                if not self.tts_thread or not self.tts_thread.is_alive():
                    logger.error("TTS thread not running")
                    return
                    
                logger.debug(f"Queueing text chunk: {text}...")
                self.tts_queue.put(text)

    def _tts_worker(self):
        """Worker thread function to process TTS queue."""
        if not self.status.get_is_voice_transmitting():
            self.status.set_voice_transmit_started()
            
        while not self.user_interrupting_flag:
            try:
                text_chunk = self.tts_queue.get(block=True)
                if text_chunk is None:  # Sentinel value received
                    self.tts_queue.task_done()
                    break
                    
            except Exception as e_queue:
                logger.error(f"TTS worker: Error getting item from queue: {e_queue}")
                break

            # Process text chunk through Cartesia
            try:
                if not self.ws:
                    logger.error("Websocket not initialized")
                    break

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

                    # First chunk timing
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
                    audio_np = np.frombuffer(output.audio, dtype=np.int16).reshape(1, -1)
                    self.status.audio_source_wavform_queue.put(audio_np)
                    
                    if self.audio_play_locally and self.audio_player:
                        self.audio_player.play_buffer(output.audio)

            except Exception as e_tts:
                logger.error(f"TTS worker: Error during Cartesia processing: {e_tts}")
                continue

            finally:
                self.tts_queue.task_done()

        # Cleanup
        self.status.set_voice_transmit_completed()
        self.timing["tts_first_chunk_time"] = -1
        while not self.tts_queue.empty():
            self.tts_queue.get()
        logger.info("TTS worker thread completed")

    def __del__(self):
        # Clean up websocket
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.error(f"Error closing Cartesia websocket: {e}")

        # Clean up audio player
        if hasattr(self, "audio_player") and self.audio_play_locally and self.audio_player:
            try:
                self.audio_player.close()
            except Exception as e:
                logger.error(f"Error closing audio player: {e}")

        # Clean up thread
        if self.tts_thread and self.tts_thread.is_alive():
            try:
                self.tts_queue.put(None)  # Send sentinel
                self.tts_thread.join(timeout=1.0)  # Wait with timeout
            except Exception as e:
                logger.error(f"Error cleaning up TTS thread: {e}")

        # Clear queue
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except queue.Empty:
                break
