# local tts livekit
import logging
from pathlib import Path
import sys
import threading
import os
import time
import psutil
import numpy as np

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from alive_inference_status import ALRealtimeIOAndStatus
from alive_inference_config import AliveInferenceConfig
from tts_standalone_livekit import TTS_Manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

class TTSWrapper:
    def __init__(self):
        self.tts_manager = None
        self.last_message_id = None  # Track last message ID to avoid duplicates
        self.current_stream_buffer = ""  # Buffer for text chunks
        self.is_streaming = False  # Track streaming state
        self.pending_text = []  # Buffer for text that arrives before [START]

    def handle_text_stream(self, text: str):
        """
        Handle incoming text stream data from LiveKit
        Args:
            text: The text content to process
        """
        if not self.tts_manager:
            logger.error("TTS manager not initialized")
            return

        try:
            # Log full text without truncation
            logger.info(f"Received text: {text}")
            
            if text == "[START]":
                logger.info("=== RECEIVED [START] MARKER ===")
                self.is_streaming = True
                self.current_stream_buffer = ""
                self.tts_manager._handle_incoming_text("[START]")
                logger.info("=== TTS STREAM STARTED ===")
                
                # Process any pending text that arrived before [START]
                if self.pending_text:
                    logger.info(f"=== PROCESSING {len(self.pending_text)} BUFFERED CHUNKS ===")
                    for i, pending in enumerate(self.pending_text, 1):
                        logger.info(f"Processing buffered chunk {i}/{len(self.pending_text)}: {pending}")
                        self.tts_manager._handle_incoming_text(pending)
                    self.pending_text = []
                    logger.info("=== ALL BUFFERED CHUNKS PROCESSED ===")
                else:
                    logger.info("=== NO BUFFERED CHUNKS TO PROCESS ===")
            
            elif text == "[DONE]" or text == "[INTERRUPTED]":
                self.is_streaming = False
                self.tts_manager._handle_incoming_text(text)
                logger.info(f"=== TTS STREAM ENDED WITH: {text} ===")
                if self.pending_text:
                    logger.warning(f"=== CLEARING {len(self.pending_text)} UNPROCESSED BUFFERED CHUNKS ===")
                self.pending_text = []  # Clear any pending text
            
            elif text.startswith("[speech_end_time]:") or text.startswith("[llm_first_token_time]:"):
                # Handle timing information
                logger.info(f"Received timing info: {text}")
            
            elif self.is_streaming:
                # Process streaming text chunk
                self.tts_manager._handle_incoming_text(text)
                logger.debug(f"Processed text chunk: {text}")
            
            else:
                # Buffer text that arrives before [START]
                logger.info(f"Buffering text before stream start: {text}")
                self.pending_text.append(text)
                logger.info(f"Current buffer size: {len(self.pending_text)} chunks")

        except Exception as e:
            logger.error(f"Error handling text stream: {e}")

def run_text2audio_test(
    status: ALRealtimeIOAndStatus,
    config: AliveInferenceConfig,
    audio_play_locally=False,
    tts_wrapper: TTSWrapper = None,
):
    """
    Run text to audio conversion test
    """
    try:
        # Initialize TTS manager
        tts_manager = TTS_Manager(
            status=status,
            audio_play_locally=audio_play_locally,
            tts_voice_id_cartesia=config.tts_voice_id_cartesia,
        )

        tts_wrapper.tts_manager = tts_manager
        print("TTS manager initialized!")

        # Keep the script running until interrupted
        stop_event = threading.Event()
        while not stop_event.is_set():
            try:
                stop_event.wait(timeout=0.1)
            except KeyboardInterrupt:
                logger.info("Test stopped by user.")
                stop_event.set()

    except Exception as e:
        logger.error(f"Error during text to audio test: {e}")

    finally:
        if tts_manager:
            try:
                del tts_manager
                logger.info("TTS manager cleaned up successfully")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

def run_text2audio_in_thread(
    status: ALRealtimeIOAndStatus,
    config: AliveInferenceConfig,
    audio_play_locally=False,
    tts_wrapper: TTSWrapper = None,
):
    """
    Start the text to audio conversion in a separate thread
    """
    thread = threading.Thread(
        target=run_text2audio_test,
        args=(status, config, audio_play_locally, tts_wrapper),
        daemon=True,
    )
    thread.start()
    return thread

if __name__ == "__main__":
    status = ALRealtimeIOAndStatus()
    config = AliveInferenceConfig()
    # Run directly with thread
    tts_wrapper = TTSWrapper()
    tts_thread = run_text2audio_in_thread(status, config, audio_play_locally=True, tts_wrapper=tts_wrapper)
    try:
        tts_thread.join()
    except KeyboardInterrupt:
        logger.info("Main thread received interrupt signal")