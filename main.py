import time
START_TIME = time.time()

import asyncio
import logging
import os
import uuid
import threading
from typing import Union
from dotenv import load_dotenv
from livekit import rtc, api
import argparse

from alive_inference_status import ALRealtimeIOAndStatus
from alive_inference_config import AliveInferenceConfig
from local_asrllm_seperated_from_tts_livekit import (
    run_audio2audio_in_thread,
    SimpleAudioCaptureHandler,
    AudioCaptureWrapper,
)

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

async def main_room(room: rtc.Room, room_name: str):
    # Create a class to hold our state
    class RoomState:
        def __init__(self):
            self.user_left = False
            self.audio_capture_wrapper = None

    state = RoomState()

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        logging.info(f"participant connected: {participant.sid} {participant.identity}")

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logging.info(f"participant disconnected: {participant.sid} {participant.identity}")

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logging.info(f"track subscribed: {publication.sid}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print("Subscribed to an Audio Track!")
            # Initialize audio capture if not already done
            if not state.audio_capture_wrapper:
                state.audio_capture_wrapper = AudioCaptureWrapper()
                # Start audio processing thread
                audio_thread = run_audio2audio_in_thread(
                    status=status,
                    config=config,
                    audio_play_locally=False,
                    audio_capture_wrapper=state.audio_capture_wrapper
                )
                # Wait for audio capture to be ready
                while not state.audio_capture_wrapper.audio_capture:
                    time.sleep(0.1)
                print("Audio capture is ready!")
            
            # Set the audio track in the wrapper
            if state.audio_capture_wrapper.audio_capture:
                state.audio_capture_wrapper.audio_capture.audio_track = publication.track
                print("Audio track set in wrapper")

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logging.info(f"track unsubscribed: {publication.sid}")

    @room.on("track_muted")
    def on_track_muted(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        logging.info(f"track muted: {publication.sid}")

    @room.on("track_unmuted")
    def on_track_unmuted(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        logging.info(f"track unmuted: {publication.sid}")

    @room.on("connection_quality_changed")
    def on_connection_quality_changed(
        participant: rtc.Participant, quality: rtc.ConnectionQuality
    ):
        logging.info(f"connection quality changed for {participant.identity}")

    @room.on("track_subscription_failed")
    def on_track_subscription_failed(
        participant: rtc.RemoteParticipant, track_sid: str, error: str
    ):
        logging.info(f"track subscription failed: {participant.identity} {error}")

    @room.on("connection_state_changed")
    def on_connection_state_changed(state: rtc.ConnectionState):
        logging.info(f"connection state changed: {state}")

    @room.on("connected")
    def on_connected() -> None:
        logging.info("connected")
        # Check if connection took too long
        elapsed_time = time.time() - START_TIME
        if elapsed_time > 60:
            print(f"Room connection took too long ({elapsed_time:.1f}s), marking as left")
            state.user_left = True
            return

    @room.on("disconnected")
    def on_disconnected() -> None:
        logging.info("disconnected")

    @room.on("reconnecting")
    def on_reconnecting() -> None:
        logging.info("reconnecting")
        print("reconnecting")

    @room.on("reconnected")
    def on_reconnected() -> None:
        logging.info("reconnected")
        print("reconnected")

    # Generate token for room connection
    token = (
        api.AccessToken()
        .with_identity(f"Avatar-{uuid.uuid4()}")
        .with_name("Avatar")
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

    print("room connected!")

    # Main loop to keep the connection alive
    loop_count = 0
    user_left_loop_count = 0
    user_left_confirm_number = 5  # roughly 15s

    while True:
        try:
            if state.user_left:
                print("user left the room")
                break

            loop_count += 1

            # check if user left every 300 loops
            if loop_count % 300 == 0:
                if not room.remote_participants or room.connection_state == rtc.ConnectionState.CONN_DISCONNECTED:
                    user_left_loop_count += 1
                    print(f"user not in room or room disconnected count + 1, total {user_left_loop_count}")
                    if user_left_loop_count > user_left_confirm_number:
                        state.user_left = True
                else:
                    user_left_loop_count = 0

            await asyncio.sleep(0.1)  # Small delay to prevent CPU overuse

        except Exception as e:
            print(f"Error in main loop: {e}")
            break

def run_async_room_connection(room_name: str):
    """Wrapper function to run async function in a thread"""
    print("start running the room connection!")

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
        print("agent is leaving the room")
        loop.run_until_complete(cleanup())
        loop.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start the room connection with a specified room name')
    parser.add_argument('--room', type=str, default='test-room', help='Name of the room to connect to')
    args = parser.parse_args()
    
    run_async_room_connection(args.room)
