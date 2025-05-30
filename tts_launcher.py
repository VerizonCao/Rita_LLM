import time
import asyncio
import logging
import os
import uuid
import threading
from typing import Union
from dotenv import load_dotenv
from livekit import rtc, api
import argparse
import json

from alive_inference_status import ALRealtimeIOAndStatus
from alive_inference_config import AliveInferenceConfig
from local_tts_livekit import run_text2audio_in_thread, TTSWrapper

# Load environment variables
load_dotenv(dotenv_path=".env.local")

# Disable aiohttp access logs
logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# Initialize config and status
config = AliveInferenceConfig()
status = ALRealtimeIOAndStatus()

# Constants for text stream
LLM_DATA_TOPIC = "llm_data"

# Store active tasks to prevent garbage collection
_active_tasks = set()

async def main_room(room: rtc.Room, room_name: str):
    # Create a class to hold our state
    class RoomState:
        def __init__(self):
            self.user_left = False
            self.tts_wrapper = None
            self.last_received_text = None
            self.current_stream_buffer = ""
            self.is_streaming = False
            self.is_initialized = False

    state = RoomState()

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        logging.info(f"participant connected: {participant.sid} {participant.identity}")

    async def async_handle_text_stream(reader, participant_identity):
        try:
            info = reader.info
            logger.info(
                f'Text stream received from {participant_identity}\n'
                f'  Topic: {info.topic}\n'
                f'  Timestamp: {info.timestamp}'
            )

            # Read all text at once
            full_text = await reader.read_all()
            
            if not state.tts_wrapper or not state.tts_wrapper.tts_manager:
                logger.warning("TTS wrapper or manager not initialized")
                return

            # Pass the text directly to the TTS wrapper
            state.tts_wrapper.handle_text_stream(full_text)

        except Exception as e:
            logger.error(f"Error processing text stream: {e}", exc_info=True)

    def handle_text_stream(reader, participant_identity):
        task = asyncio.create_task(async_handle_text_stream(reader, participant_identity))
        _active_tasks.add(task)
        task.add_done_callback(lambda t: _active_tasks.remove(t))

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logging.info(f"participant disconnected: {participant.sid} {participant.identity}")

    def initialize_tts():
        """Initialize TTS system synchronously"""
        try:
            state.tts_wrapper = TTSWrapper()
            tts_thread = run_text2audio_in_thread(
                status=status,
                config=config,
                audio_play_locally=False,
                tts_wrapper=state.tts_wrapper
            )
            
            # Wait a moment to ensure thread is running
            time.sleep(1)
            
            # Verify initialization
            if state.tts_wrapper and state.tts_wrapper.tts_manager:
                state.is_initialized = True
            else:
                state.is_initialized = False
                
        except Exception as e:
            logger.error(f"Error initializing TTS: {e}", exc_info=True)
            state.is_initialized = False

    @room.on("connected")
    def on_connected() -> None:
        logging.info("Room connected event received")
        initialize_tts()

    @room.on("disconnected")
    def on_disconnected() -> None:
        logging.info("Room disconnected event received")
        state.is_initialized = False

    @room.on("reconnecting")
    def on_reconnecting() -> None:
        logging.info("Room reconnecting event received")
        state.is_initialized = False

    @room.on("reconnected")
    def on_reconnected() -> None:
        logging.info("Room reconnected event received")
        # Re-register text stream handler after reconnection
        room.register_text_stream_handler(LLM_DATA_TOPIC, handle_text_stream)
        # Re-initialize TTS synchronously
        initialize_tts()

    # Register text stream handler before connecting
    room.register_text_stream_handler(LLM_DATA_TOPIC, handle_text_stream)

    # Generate token for room connection
    token = (
        api.AccessToken()
        .with_identity(f"TTS-Agent-{uuid.uuid4()}")
        .with_name("TTS-Agent")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )

    # Connect to room
    await room.connect(
        os.getenv("LIVEKIT_URL"), token, rtc.RoomOptions(auto_subscribe=True)
    )

    print("TTS agent connected to room!")
    
    # Initialize TTS immediately after connection
    initialize_tts()

    # Main loop to keep the connection alive
    while True:
        try:
            if state.user_left:
                print("user left the room")
                break

            await asyncio.sleep(0.1)  # Small delay to prevent CPU overuse

        except Exception as e:
            print(f"Error in main loop: {e}")
            break

def run_async_room_connection(room_name: str):
    """Wrapper function to run async function in a thread"""
    print("Starting TTS agent room connection!")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    room: rtc.Room = rtc.Room(loop)

    async def cleanup():
        print("Starting cleanup...")
        try:
            await room.disconnect()
            print("Room disconnected")
            loop.stop()
            print("Cleanup completed successfully")
        except Exception as e:
            print(f"Error during cleanup: {e}")
            raise

    async def run_with_cleanup():
        try:
            await main_room(room, room_name)
        finally:
            print("Starting final cleanup...")
            await cleanup()
            print("Final cleanup completed")

    # Create and run the main task
    main_task = loop.create_task(run_with_cleanup())

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("TTS agent is leaving the room")
        loop.run_until_complete(cleanup())
        loop.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start the TTS agent with a specified room name')
    parser.add_argument('--room', type=str, default='test-room', help='Name of the room to connect to')
    args = parser.parse_args()
    
    run_async_room_connection(args.room)
