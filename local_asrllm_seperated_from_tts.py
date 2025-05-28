import logging
from pathlib import Path
import sys
import threading
import os
import wave
import array
import time
import psutil
import numpy as np

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from alive_inference_status import ALRealtimeIOAndStatus
from audio_capture import (
    AudioCapture,
    AudioCaptureEventHandler,
)

from asr_llm_standalone import ASR_LLM_Manager
from tts_standalone import TTS_Manager
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
    ):
        self.is_speaking = False
        self.current_audio = []
        self.sample_rate = 24000
        self.asr_llm_manager = ASR_LLM_Manager(
            publish_locally=True,
            llm_data=config.get_llm_tuple(),
        )
        self.status = status
        # Add thread management
        self.processing_thread = None
        self.stop_processing = threading.Event()
        self.speech_end_time = 0
        self.last_speech_detected_time = 0

    def interrupt_processing(self):
        """Interrupt the current processing thread if it exists"""
        if self.processing_thread and self.processing_thread.is_alive():
            logger.info("Interrupting previous processing")
            self.stop_processing.set()
            self.processing_thread.join(timeout=0.1)  # Give it a short time to clean up
            self.stop_processing.clear()
        if self.asr_llm_manager:
            self.asr_llm_manager.user_interrupting_flag = True

    def save_audio_to_temp_file(self):
        """Save accumulated audio to a temporary WAV file"""
        time_stamp = time.time()
        temp_file = f"temp_audio_{time_stamp}.wav"

        with wave.open(temp_file, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(array.array("h", self.current_audio).tobytes())

        return temp_file

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
        # Interrupt any ongoing processing when new speech starts
        self.interrupt_processing()

    def on_speech_end(self):
        self.is_speaking = False
        logger.info("Local VAD: User speech ended")
        self.speech_end_time = time.time()
        logger.info(
            f"Time for detecting speech-end: {self.speech_end_time - self.last_speech_detected_time:.2f} seconds"
        )

        if len(self.current_audio) > 0:
            # Save audio to temporary file
            temp_file = self.save_audio_to_temp_file()

            # Process through AI pipeline in a separate thread
            def process_audio_pipeline():
                try:
                    # Check if we should stop at each major step
                    if self.stop_processing.is_set():
                        logger.info("Processing interrupted before transcription")
                        return

                    transcription_text = self.asr_llm_manager.speech_to_text(
                        temp_file, self.speech_end_time
                    )
                    os.remove(temp_file)

                    if self.stop_processing.is_set():
                        logger.info("Processing interrupted before LLM")
                        return

                    llm_response = self.asr_llm_manager.send_to_openrouter(
                        transcription_text
                    )

                    if self.stop_processing.is_set():
                        logger.info("Processing interrupted before TTS")
                        return

                except Exception as e:
                    logger.error(f"Error during async processing: {e}")
                finally:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

            # Interrupt any existing processing before starting new thread
            self.interrupt_processing()

            # Start new processing thread
            self.processing_thread = threading.Thread(target=process_audio_pipeline)
            self.processing_thread.daemon = True
            self.processing_thread.start()

            self.current_audio = []

    def on_keyword_detected(self, result):
        pass

    def __del__(self):
        self.interrupt_processing()
        if hasattr(self, "asr_llm_manager"):
            del self.asr_llm_manager


# audio capture to llm to tts
def run_audio_capture_test(
    status: ALRealtimeIOAndStatus,
    config: AliveInferenceConfig,
    use_silero_vad=True,
    audio_play_locally=False,
):
    """
    Run audio capture test with specified VAD model

    Args:
        use_silero_vad: If True, use Silero VAD, otherwise use basic VAD
    """
    audio_capture = None

    try:
        # initialize tts
        tts_manager = TTS_Manager(
            status=status,
            listen_locally=True,
            audio_play_locally=audio_play_locally,
            tts_voice_id_cartesia=config.tts_voice_id_cartesia,
        )

        # Initialize the event handler
        event_handler = SimpleAudioCaptureHandler(
            status=status, config=config, audio_play_locally=audio_play_locally
        )

        # Configure VAD parameters based on the selected model
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

        # Initialize AudioCapture
        audio_capture = AudioCapture(
            event_handler=event_handler,
            sample_rate=24000,
            channels=1,
            frames_per_buffer=1024,
            buffer_duration_sec=1.0,
            cross_fade_duration_ms=20,
            vad_parameters=vad_parameters,
            enable_wave_capture=False,
        )

        logger.info("Starting audio capture test... Press Ctrl+C to stop.")
        audio_capture.start()

        # Keep the script running until interrupted
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


# launch in a thread. Call this from the main thread or local __main__
def run_audio2audio_in_thread(
    status: ALRealtimeIOAndStatus,
    config: AliveInferenceConfig,
    audio_play_locally=False,
):
    """
    Start the audio capture test in a separate thread with CPU affinity set to CPU 1
    """
    thread = threading.Thread(
        target=run_audio_capture_test,
        args=(status, config, True, audio_play_locally),
        daemon=True,
    )
    thread.start()
    return thread


if __name__ == "__main__":
    status = ALRealtimeIOAndStatus()
    config = AliveInferenceConfig()
    # Run directly with thread and CPU affinity
    audio_thread = run_audio2audio_in_thread(status, config, audio_play_locally=True)
    try:
        audio_thread.join()
    except KeyboardInterrupt:
        logger.info("Main thread received interrupt signal")
