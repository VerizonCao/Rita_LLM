import logging
import threading
import os
import wave
import array
import time
import tempfile
from pathlib import Path
import asyncio

from alive_inference_status import ALRealtimeIOAndStatus
from audio_capture_livekit import (
    AudioCaptureLiveKit,
    AudioCaptureEventHandler,
)
from asr_llm_standalone import ASR_LLM_Manager
from alive_inference_config import AliveInferenceConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = SCRIPT_DIR / "pretrained_weights" / "audio_capture"

class SimpleAudioCaptureHandler(AudioCaptureEventHandler):
    def __init__(
        self,
        status: ALRealtimeIOAndStatus,
        config: AliveInferenceConfig,
        audio_play_locally=False,
        audio_track=None,
        room=None,
        loop=None,
        image_swap=False,
        image_url=None,
    ):
        self.is_speaking = False
        self.current_audio = []
        self.sample_rate = 24000
        self.status = status
        self.config = config  # Store config reference
        if image_swap:
            self.asr_llm_manager = ASR_LLM_Manager(
                llm_data=config.get_llm_tuple(),
                room=room,
                loop=loop,
                mcp_server_path='chat_server.py',
                image_url=image_url,
            )
        else:
            self.asr_llm_manager = ASR_LLM_Manager(
                llm_data=config.get_llm_tuple(),
                room=room,
                loop=loop,
            )
        self.processing_thread = None
        self.stop_processing = threading.Event()
        self.speech_end_time = 0
        self.last_speech_detected_time = 0
        self.audio_track = audio_track
        self.room = room
        self.loop = loop
        self.current_temp_file = None  # Track the current temporary file
        self.image_url = image_url

        self.text_input = None
        self.text_input_voice = []

    def update_llm_manager(self):
        """Update the ASR_LLM_Manager with new LLM data"""
        if self.asr_llm_manager:
            # Create a new ASR_LLM_Manager with updated config
            self.asr_llm_manager = ASR_LLM_Manager(
                llm_data=self.config.get_llm_tuple(),
                room=self.room,
                loop=self.loop,
            )

    def interrupt_processing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            logger.info("Interrupting previous processing")
            self.stop_processing.set()
            self.processing_thread.join(timeout=0.1)
            self.stop_processing.clear()
        if self.asr_llm_manager:
            self.asr_llm_manager.user_interrupting_flag = True

    def save_audio_to_temp_file(self):
        """Save accumulated audio to a temporary WAV file"""
        try:
            # Clean up any existing temp file
            if self.current_temp_file and os.path.exists(self.current_temp_file):
                try:
                    os.remove(self.current_temp_file)
                    logger.info(f"Cleaned up previous temp file: {self.current_temp_file}")
                except Exception as e:
                    logger.error(f"Error cleaning up previous temp file: {e}")

            time_stamp = time.time()
            temp_file = os.path.join(tempfile.gettempdir(), f"temp_audio_{time_stamp}.wav")
            logger.info(f"Creating new temp file: {temp_file}")

            with wave.open(temp_file, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(array.array("h", self.current_audio).tobytes())

            self.current_temp_file = temp_file
            return temp_file
        except Exception as e:
            logger.error(f"Error creating temp file: {e}")
            return None

    def send_audio_data(self, audio_data: bytes):
        if self.is_speaking:
            int_data = array.array("h", audio_data)
            self.current_audio.extend(int_data)
        logger.debug(f"Received audio data of length: {len(audio_data)} bytes")
        self.last_speech_detected_time = time.time()

    def on_speech_start(self):
        self.is_speaking = True
        self.current_audio = []
        self.status.set_user_talking()
        logger.info("Local VAD: User speech started")
        self.interrupt_processing()

    def on_text_received(self, text: str):
        print("SimpleAudioCaptureHandler, text received:", text)
        self.is_speaking = False
        try:
            self.text_input = text
        except Exception as e:
            logger.error(f"Error setting user text input: {e}")

    def has_text_input(self):
        return self.text_input is not None

    def on_text_handled(self):
        try:
            if not self.text_input:
                return
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    self.asr_llm_manager.send_to_openrouter(self.text_input),
                    self.loop
                )
            else:
                logger.error("No event loop available for async operation")
            self.text_input = None
        except Exception as e:
            logger.error(f"Error processing user text input: {e}")

    def has_text_input_voice(self):
        return len(self.text_input_voice) > 0

    def on_speech_end(self):
        self.is_speaking = False
        logger.info("Local VAD: User speech ended")
        self.speech_end_time = time.time()
        logger.info(
            f"Time for detecting speech-end: {self.speech_end_time - self.last_speech_detected_time:.2f} seconds"
        )

        if len(self.current_audio) > 0:
            temp_file = self.save_audio_to_temp_file()
            if not temp_file:
                logger.error("Failed to create temp file for speech processing")
                return

            def process_audio_pipeline():
                try:
                    if self.stop_processing.is_set():
                        logger.info("Processing interrupted before transcription")
                        return

                    transcription_text = self.asr_llm_manager.speech_to_text(
                        temp_file, self.speech_end_time
                    )
                    
                    # Remove temp file after successful transcription
                    try:
                        os.remove(temp_file)
                        logger.info(f"Removed temp file after transcription: {temp_file}")
                        self.current_temp_file = None
                    except Exception as e:
                        logger.error(f"Error removing temp file after transcription: {e}")

                    self.text_input_voice.append(transcription_text)

                    if self.stop_processing.is_set():
                        logger.info("Processing interrupted before LLM")
                        return

                    if self.loop:
                        asyncio.run_coroutine_threadsafe(
                            self.asr_llm_manager.send_to_openrouter(transcription_text),
                            self.loop
                        )
                    else:
                        logger.error("No event loop available for async operation")

                except Exception as e:
                    logger.error(f"Error during async processing: {e}")
                finally:
                    # Ensure temp file is cleaned up
                    if temp_file and os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                            logger.info(f"Removed temp file in finally block: {temp_file}")
                            self.current_temp_file = None
                        except Exception as e:
                            logger.error(f"Error removing temp file in finally block: {e}")

            self.interrupt_processing()
            self.processing_thread = threading.Thread(target=process_audio_pipeline)
            self.processing_thread.daemon = True
            self.processing_thread.start()

            self.current_audio = []

    def on_keyword_detected(self, result):
        pass

    def __del__(self):
        """Cleanup when the handler is destroyed"""
        self.interrupt_processing()
        # Clean up any remaining temp file
        if self.current_temp_file and os.path.exists(self.current_temp_file):
            try:
                os.remove(self.current_temp_file)
                logger.info(f"Cleaned up temp file in destructor: {self.current_temp_file}")
            except Exception as e:
                logger.error(f"Error cleaning up temp file in destructor: {e}")
        if hasattr(self, "asr_llm_manager"):
            del self.asr_llm_manager

class AudioCaptureWrapper:
    def __init__(self):
        self.audio_capture = None
        self.agent_messages = []
        self.text_input_voice = []  # Array to store voice transcriptions
        self.audio_track = None

def run_audio_capture_test(
    status: ALRealtimeIOAndStatus,
    config: AliveInferenceConfig,
    use_silero_vad=True,
    audio_play_locally=False,
    audio_capture_wrapper: AudioCaptureWrapper = None,
    room = None,
    loop = None,
    image_swap=False,
    image_url=None,
):
    audio_capture = None

    try:
        event_handler = SimpleAudioCaptureHandler(
            status=status, 
            config=config, 
            audio_play_locally=audio_play_locally,
            audio_track=audio_capture_wrapper.audio_track if audio_capture_wrapper else None,
            room=room,
            loop=loop,
            image_swap=image_swap,
            image_url=image_url,
        )

        event_handler.asr_llm_manager.image_swap = image_swap
        event_handler.asr_llm_manager.image_url = image_url

        if use_silero_vad:
            logger.info("Using Silero VAD...")
            vad_parameters = {
                "sample_rate": 24000,
                "chunk_size": 1024,
                "window_size_samples": 512,
                "threshold": 0.5,
                "min_speech_duration": 0.3,
                "min_silence_duration": 0.3,
                "model_path": str(RESOURCES_DIR / "silero_vad.onnx"),
            }
        else:
            logger.info("Using basic VoiceActivityDetector...")
            vad_parameters = {
                "sample_rate": 24000,
                "chunk_size": 1024,
                "window_duration": 1.5,
                "silence_ratio": 1.5,
                "min_speech_duration": 0.3,
                "min_silence_duration": 0.2,
            }

        audio_capture = AudioCaptureLiveKit(
            event_handler=event_handler,
            sample_rate=24000,
            channels=1,
            frames_per_buffer=1024,
            buffer_duration_sec=1.0,
            cross_fade_duration_ms=20,
            vad_parameters=vad_parameters,
            enable_wave_capture=False,
            audio_track=audio_capture_wrapper.audio_track if audio_capture_wrapper else None,
        )

        if audio_capture_wrapper:
            audio_capture_wrapper.audio_capture = audio_capture
            audio_capture_wrapper.agent_messages = event_handler.asr_llm_manager.messages
            print(f"Debug: Synchronized agent_messages. Total messages: {len(audio_capture_wrapper.agent_messages)}")
            print("finish setup the audio capture !!")

        logger.info("Starting audio capture test... Press Ctrl+C to stop.")
        audio_capture.start()

        stop_event = threading.Event()
        while not stop_event.is_set():
            try:
                stop_event.wait(timeout=0.1)
            except KeyboardInterrupt:
                logger.info("Test stopped by user.")
                stop_event.set()

    except Exception as e:
        logger.error(f"Error during audio capture test: {e}")

    finally:
        if audio_capture:
            try:
                audio_capture.stop()
                audio_capture.close()
                logger.info("Audio capture cleaned up successfully")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")


def run_audio2audio_in_thread(
    status: ALRealtimeIOAndStatus,
    config: AliveInferenceConfig,
    audio_play_locally=False,
    audio_capture_wrapper: AudioCaptureWrapper = None,
    room = None,
    loop = None,
    image_swap=False,
    image_url=None,
):
    thread = threading.Thread(
        target=run_audio_capture_test,
        args=(status, config, True, audio_play_locally, audio_capture_wrapper, room, loop, image_swap, image_url),
        daemon=True,
    )
    thread.start()
    return thread


if __name__ == "__main__":
    status = ALRealtimeIOAndStatus()
    config = AliveInferenceConfig()
    audio_thread = run_audio2audio_in_thread(status, config, audio_play_locally=True)
    try:
        audio_thread.join()
    except KeyboardInterrupt:
        logger.info("Main thread received interrupt signal")
