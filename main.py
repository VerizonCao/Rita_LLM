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
import json

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
            self.last_sent_voice_transcription = None  # Track last sent voice transcription
            self.agent_message_count = 0  # Track number of agent messages processed
            self.text_stream_writer = None  # Store the text stream writer

    state = RoomState()
    loop = asyncio.get_event_loop()  # Get the current event loop

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        logging.info(f"participant connected: {participant.sid} {participant.identity}")

    @room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        async def process_data():
            try:
                raw_json = data.data.decode("utf-8")
                json_data = json.loads(raw_json)

                message = json_data.get("message")
                if message:
                    # Check if this is a duplicate of our last sent voice transcription
                    if message == state.last_sent_voice_transcription:
                        print("Received duplicate voice transcription, ignoring")
                        state.last_sent_voice_transcription = None  # Reset after processing
                        return
                    # also remove it. 
                    state.last_sent_voice_transcription = None  # Reset after processing
                    print("ready to send text input to the agent: ", message)
                    if state.audio_capture_wrapper and state.audio_capture_wrapper.audio_capture:
                        state.audio_capture_wrapper.audio_capture.on_text_received(message)
            except Exception as e:
                print(f"Error processing data: {e}")

        asyncio.create_task(process_data())

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
        # Check if participant is not an agent-avatar
        if participant.attributes and participant.attributes.get("role") == "agent-avatar":
            logging.info("Ignoring audio track from agent-avatar")
            return
            
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
                    audio_capture_wrapper=state.audio_capture_wrapper,
                    room=room,  # Pass the room to the wrapper
                    loop=loop,  # Pass the event loop to the wrapper
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
        .with_attributes({"role": "agent-asr"})
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                hidden=True,
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
    user_left_confirm_number = 2  # roughly 5s
    hasPublished = False

    while True:
        try:
            if state.user_left:
                print("user left the room")
                break

            loop_count += 1

            # check if user left every 300 loops
            if loop_count % 30 == 0:
                # Check if there are no remote participants or if the only participant is an agent-avatar
                has_valid_participant = False
                if room.remote_participants:
                    for participant in room.remote_participants.values():
                        if not (participant.attributes and participant.attributes.get("role") == "agent-avatar"):
                            has_valid_participant = True
                            break

                if not has_valid_participant or room.connection_state == rtc.ConnectionState.CONN_DISCONNECTED:
                    user_left_loop_count += 1
                    print(f"user not in room or room disconnected count + 1, total {user_left_loop_count}")
                    if user_left_loop_count > user_left_confirm_number:
                        state.user_left = True
                else:
                    user_left_loop_count = 0

            # Handle complete assistant messages (for backward compatibility)
            if (
                hasPublished
                and state.audio_capture_wrapper
                and len(state.audio_capture_wrapper.agent_messages) > state.agent_message_count
            ):
                start_time = time.time()
                message = state.audio_capture_wrapper.agent_messages[state.agent_message_count]
                
                if message.get("role") == "assistant":
                    try:
                        if not room.remote_participants:
                            print("No remote participants available to send message to")
                            continue
                            
                        response_content = message["content"]
                        if len(response_content) > 0:
                            print("send agent message: ", response_content)
                            await room.local_participant.send_text(
                                text=response_content,
                                topic="lk.chat",
                            )
                    except Exception as e:
                        print(f"Error sending message: {e}")
                        continue

                state.agent_message_count += 1
                end_time = time.time()
                processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
                print(f"Message processing time: {processing_time:.2f}ms")

            await asyncio.sleep(0.1)  # Small delay to prevent CPU overuse

        except Exception as e:
            print(f"Error in main loop: {e}")
            break

    return state  # Return the state object

def run_async_room_connection(room_name: str):
    """Wrapper function to run async function in a thread"""
    print("start running the room connection!")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    room: rtc.Room = rtc.Room(loop)
    state = None  # Initialize state variable

    async def cleanup():
        print("Starting cleanup...")
        try:
            # Stop audio capture if it exists
            if state and state.audio_capture_wrapper and state.audio_capture_wrapper.audio_capture:
                try:
                    state.audio_capture_wrapper.audio_capture.stop()
                except Exception as e:
                    print(f"Error stopping audio capture: {e}")

            # Disconnect from room
            try:
                if room and room.connection_state != rtc.ConnectionState.CONN_DISCONNECTED:
                    await room.disconnect()
            except Exception as e:
                print(f"Error disconnecting from room: {e}")

            print("Cleanup completed")
        except Exception as e:
            print(f"Error during cleanup: {e}")

    async def run_with_cleanup():
        nonlocal state
        try:
            state = await main_room(room, room_name)
        finally:
            await cleanup()
            # Stop the event loop when we're done
            loop.stop()

    # Create and run the main task
    main_task = loop.create_task(run_with_cleanup())

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt")
    finally:
        print("agent is leaving the room")
        try:
            if not loop.is_closed():
                loop.run_until_complete(cleanup())
        except Exception as e:
            print(f"Error during final cleanup: {e}")
        finally:
            if not loop.is_closed():
                loop.close()

def handler(event, context):
    """
    AWS Lambda handler function to process incoming events and start room connections.
    
    Args:
        event (dict): The event data containing room information
        context (object): The Lambda context object
    
    Returns:
        dict: Response containing status code and message
    """
    try:
        # Handle direct invocation (not through SQS)
        if "Records" not in event:
            room_name = event.get("room_name", "test-room")
            print(f"Starting room connection for room: {room_name}")
            # Direct call since run_async_room_connection manages its own event loop
            run_async_room_connection(room_name)
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "Room connection started successfully",
                    "room_name": room_name
                })
            }

        # Handle SQS events
        for record in event["Records"]:
            try:
                body = json.loads(record["body"])
                room_name = body.get("room_name", "test-room")
                print(f"Starting room connection for room: {room_name}")
                # Direct call since run_async_room_connection manages its own event loop
                run_async_room_connection(room_name)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from record: {str(e)}")
                continue
            except Exception as e:
                print(f"Error processing record: {str(e)}")
                continue
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Room connections started successfully"
            })
        }
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start the room connection with a specified room name')
    parser.add_argument('--room', type=str, default='test-room', help='Name of the room to connect to')
    args = parser.parse_args()
    
    run_async_room_connection(args.room)
