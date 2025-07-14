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
import shutil
import aiohttp

# force push

from alive_inference_config import AliveInferenceConfig
from local_asrllm_seperated_from_tts_livekit import (
    run_audio_capture_in_thread,
    AudioCaptureWrapper,
)
from asr_llm_standalone import ASR_LLM_Manager
from telemetry import setup_telemetry, shutdown_telemetry

# Load environment variables
if os.path.exists(".env"):
    load_dotenv(dotenv_path=".env")
elif os.path.exists(".env.local"):
    load_dotenv(dotenv_path=".env.local")
else:
    raise ValueError("No environment file found, did you forget to add .env or .env.local?")

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

# Initialize metrics
metrics = setup_telemetry()

def override_llm_properties(config: AliveInferenceConfig, override_config: dict):
    """
    Override LLM properties in the configuration
    
    Args:
        config: AliveInferenceConfig instance
        override_config: Dictionary containing LLM properties to override
    """
    # Create a temporary copy of file_paths.json in /tmp
    import shutil
    import os
    
    # Get the original file path
    current_dir = os.path.dirname(__file__)
    original_file_path = os.path.join(current_dir, "file_paths.json")
    temp_file_path = "/tmp/file_paths.json"
    
    # Copy the file to /tmp
    shutil.copy2(original_file_path, temp_file_path)
    
    # Update the config to use the temporary file
    config.file_path = temp_file_path
    
    # Now perform the override
    old_llm_config, old_io_source_portrait_folder = config.replace_all_llm(override_config)
    
    # We don't try to copy back to the original location since it's read-only
    # Instead, we'll just use the temporary file for the rest of the session
    
    return old_llm_config, old_io_source_portrait_folder

async def main_room(room: rtc.Room, room_name: str, llm_overrides: dict = None, image_swap: bool = False, image_url: str = None):
    # Create a class to hold our state
    class RoomState:
        def __init__(self):
            self.user_left = False
            self.audio_capture_wrapper : AudioCaptureWrapper = None
            self.last_sent_voice_transcription = None  # Track last sent voice transcription
            self.agent_message_count = 0  # Track number of agent messages processed
            self.text_stream_writer = None  # Store the text stream writer
            self.voice_transcription_count = 0  # Track number of voice transcriptions processed
            self.old_llm_config = None  # Store old LLM config for cleanup
            self.old_io_source_portrait_folder = None  # Store old portrait folder for cleanup
            self.temp_file_path = None  # Store temporary file path for cleanup
            self.serve_start_time = None  # Track when we start serving
            
            self.asr_llm_manager : ASR_LLM_Manager = None
            self.current_user_id = None  # Track current user ID
            self.current_avatar_id = None  # Track current avatar ID
            self.image_swap = False  # Track image swap setting
            self.image_url = None  # Track image URL

    state = RoomState()
    state.image_swap = image_swap
    state.image_url = image_url
    loop = asyncio.get_event_loop()  # Get the current event loop

    # Override LLM properties if provided
    if llm_overrides:
        state.old_llm_config, state.old_io_source_portrait_folder = override_llm_properties(config, llm_overrides)
        # Re-initialize the config to load the new values
        config.io_init(config.file_path)
        # Store user and avatar IDs if provided
        state.current_user_id = llm_overrides.get("user_id", "unknown")
        state.current_avatar_id = llm_overrides.get("avatar_id", "unknown")

    state.asr_llm_manager = ASR_LLM_Manager(
        user_id=state.current_user_id,
        avatar_id=state.current_avatar_id,
        room=room,
        loop=loop,
        image_swap=state.image_swap,
        image_url=state.image_url,
    )
    # Initialize audio capture wrapper and start audio thread before room connection
    state.audio_capture_wrapper = AudioCaptureWrapper()
    audio_thread = run_audio_capture_in_thread(
        audio_capture_wrapper=state.audio_capture_wrapper,
        asr_llm_manager=state.asr_llm_manager,
        loop=loop,
    )

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        logging.info(f"participant connected: {participant.sid} {participant.identity}")
        # Start tracking serve time when a non-agent participant connects
        if not (participant.attributes and participant.attributes.get("role") == "agent-avatar"):
            if not state.serve_start_time:
                state.serve_start_time = time.time()

    @room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        async def process_data():
            try:
                raw_json = data.data.decode("utf-8")
                json_data = json.loads(raw_json)

                # Check for IMAGE_MSG_REMOVE message type
                message_type = json_data.get("type")
                if message_type == "IMAGE_MSG_REMOVE":
                    message_id = json_data.get("messageId")
                    image_url = json_data.get("imageUrl")
                    
                    if message_id and image_url:
                        logger.info(f"Received IMAGE_MSG_REMOVE request for messageId: {message_id}, imageUrl: {image_url}")
                        
                        # Access the ASR_LLM_Manager
                        if  state.asr_llm_manager :
                            
                            # Call the remove_image_message method
                            success = state.asr_llm_manager.remove_image_message(
                                message_id=message_id,
                                image_url=image_url
                            )
                            
                            if success:
                                logger.info(f"Successfully removed image message with ID: {message_id}")
                            else:
                                logger.warning(f"Failed to remove image message with ID: {message_id}")
                        else:
                            logger.error("ASR_LLM_Manager not available for image message removal")
                    else:
                        logger.error(f"Invalid IMAGE_MSG_REMOVE message: missing messageId or imageUrl")
                    return

                # Handle regular text messages (existing logic)
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
        # Check if the disconnected participant has the role "avatar-agent"
        if participant.attributes and participant.attributes.get("role") == "agent-avatar":
            state.user_left = True
            logging.info("Avatar agent disconnected, marking user_left as True")

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
            # Set the audio track in the wrapper
            if state.audio_capture_wrapper and state.audio_capture_wrapper.audio_capture:
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
    
    # Send Discord webhook notification
    await send_discord_webhook(room_name, state.current_user_id, state.current_avatar_id)

    # local test logic. comment me in production
    # state.serve_start_time = time.time()

    # Main loop to keep the connection alive
    loop_count = 0
    user_left_loop_count = 0
    user_left_confirm_number = 2  # roughly 2s

    # for debug why user is not enter the room.
    user_not_enter_loop_count = 0
    user_not_enter_log_number = 500

    while True:
        try:
            if state.user_left:
                print("user left the room")
                
                # Send Discord webhook notification for user leaving
                await send_discord_webhook(room_name, state.current_user_id, state.current_avatar_id, "leaving")
                
                # Report final serve time when user leaves
                if state.serve_start_time:
                    serve_time = round(time.time() - state.serve_start_time)
                    print(f"going to report serve time: {serve_time}, avatar_id: {state.current_avatar_id}, user_id: {state.current_user_id}")
                    metrics["avatar_serve_time"].add(serve_time, {
                        "avatar_id": state.current_avatar_id or "unknown",
                        "user_id": state.current_user_id or "unknown"
                    })

                    # here, let's create the play session and save into db
                    try:
                        from data import PlaySessionManager
                        session_manager = PlaySessionManager()
                        
                        # Get usage information from LLM manager
                        has_llm_manager = state.asr_llm_manager
                        
                        usage = None
                        if has_llm_manager:
                            usage = state.asr_llm_manager.current_usage
                            # Add serve_time to usage information
                            usage["session_time"] = serve_time
                            
                            # Report total tokens to metrics
                            if usage["total_tokens"] > 0:
                                print(f"going to report total token usage: {usage['total_tokens']}, avatar_id: {state.current_avatar_id}, user_id: {state.current_user_id}")
                                metrics["token_usage_counter"].add(usage["total_tokens"], {
                                    "avatar_id": state.current_avatar_id or "unknown",
                                    "user_id": state.current_user_id or "unknown"
                                })
                        
                        # Create new play session with usage information
                        session_id = session_manager.create_session(
                            user_id=state.current_user_id or "unknown",
                            avatar_id=state.current_avatar_id or "unknown",
                            usage=usage
                        )
                        
                        if session_id:
                            print(f"Created play session {session_id} with metrics:")
                            if usage:
                                print(f"- Total tokens: {usage['total_tokens']}")
                                print(f"- Prompt tokens: {usage['prompt_tokens']}")
                                print(f"- Completion tokens: {usage['completion_tokens']}")
                                print(f"- TTS tokens: {usage.get('tts_tokens', 0)}")
                                print(f"- Cost: {usage['cost']}")
                                print(f"- Session time: {usage['session_time']}")
                    except Exception as e:
                        print(f"Error creating play session: {e}")
                    
                    # Force immediate export of metrics and shutdown telemetry
                    if "provider" in metrics:
                        metrics["provider"].force_flush()
                        shutdown_telemetry(metrics["provider"])
                    
                    time.sleep(1)

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

                if has_valid_participant and not state.serve_start_time:
                    state.serve_start_time = time.time()

                # if not has_valid_participant or room.connection_state == rtc.ConnectionState.CONN_DISCONNECTED:
                if room.connection_state == rtc.ConnectionState.CONN_DISCONNECTED:
                    user_left_loop_count += 1
                    print(f"user not in room or room disconnected count + 1, total {user_left_loop_count}")
                    if user_left_loop_count > user_left_confirm_number:
                        state.user_left = True
                else:
                    user_left_loop_count = 0

            # Handle complete assistant messages (for backward compatibility)
            if (
                state.audio_capture_wrapper
                and len(state.audio_capture_wrapper.agent_messages) > state.agent_message_count
            ):
                start_time = time.time()
                print(f"Debug: Found {len(state.audio_capture_wrapper.agent_messages)} messages, current count: {state.agent_message_count}")
                message = state.audio_capture_wrapper.agent_messages[state.agent_message_count]
                
                if message.get("role") == "assistant":
                    try:
                        if not room.remote_participants:
                            user_not_enter_loop_count += 1
                            if user_not_enter_loop_count == user_not_enter_log_number:
                                # Send webhook notification for user not entering the room
                                await send_user_not_entered_webhook(room_name, state.current_user_id, state.current_avatar_id, user_not_enter_loop_count)

                        #     # add a 0.2 sec interval to prevent log overload. 
                        #     time.sleep(0.2)
                        #     continue
                            
                        response_content = message["content"]
                        if len(response_content) > 0:
                            print("send agent message: ", response_content)
                            # replaced this with asrllm manager directly streaming text to frontend
                            # await room.local_participant.send_text(
                            #     text=response_content,
                            #     topic="lk.chat",
                            # )
                    except Exception as e:
                        print(f"Error sending message: {e}")
                        time.sleep(0.2)
                        continue

                state.agent_message_count += 1
                end_time = time.time()
                processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
                print(f"Message processing time: {processing_time:.2f}ms")

            # Check for voice transcriptions
            # if (False):
            if (
                state.audio_capture_wrapper 
                and state.audio_capture_wrapper.audio_capture 
                and state.audio_capture_wrapper.audio_capture.event_handler
                and len(state.audio_capture_wrapper.audio_capture.event_handler.text_input_voice) > state.voice_transcription_count
                # and room.remote_participants
            ):
                try:
                    # Get the next transcription to send
                    transcription = state.audio_capture_wrapper.audio_capture.event_handler.text_input_voice[state.voice_transcription_count]
                    print("Sending transcription: ", transcription)
                    
                    # Store the transcription before sending
                    state.last_sent_voice_transcription = transcription

                    # let's use livkit data channel to send.
                    response_data = {}
                    response_data["text"] = transcription
                    response_data["index"] = state.voice_transcription_count
                    await room.local_participant.publish_data(
                        json.dumps(
                            {
                                "type": "voice_transcription",
                                "resp": response_data,
                            }
                        )
                    )

                    state.voice_transcription_count += 1
                    
                except Exception as e:
                    print(f"Error sending transcription: {e}")
                    time.sleep(0.2)
                    continue

            await asyncio.sleep(0.1)  # Small delay to prevent CPU overuse

        except Exception as e:
            print(f"Error in main loop: {e}")
            break

    # Restore original LLM configuration if it was overridden
    if state.old_llm_config:
        config.replace_all_llm(state.old_llm_config, io_source_portrait_folder_reset=state.old_io_source_portrait_folder)
        # Clean up temporary portrait folder if it exists
        if state.old_io_source_portrait_folder and os.path.exists(state.old_io_source_portrait_folder):
            try:
                shutil.rmtree(state.old_io_source_portrait_folder)
            except Exception as e:
                print(f"Error cleaning up portrait folder: {e}")
        
        # Clean up temporary file if it exists
        if os.path.exists("/tmp/file_paths.json"):
            try:
                os.remove("/tmp/file_paths.json")
            except Exception as e:
                print(f"Error cleaning up temporary file: {e}")

    return state  # Return the state object

def run_async_room_connection(room_name: str, llm_overrides: dict = None, image_swap: bool = False, image_url: str = None):
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
            state = await main_room(room, room_name, llm_overrides, image_swap, image_url)
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
            # Get LLM properties and additional fields, filter out empty values
            llm_properties = {k: v for k, v in event.items() if k.startswith("llm_") or k == "tts_voice_id_cartesia" or k == "user_id" or k == "avatar_id"}
            if llm_properties:
                llm_properties = {k: v for k, v in llm_properties.items() if v and str(v).strip()}
                if not llm_properties:  # If all values were empty, set to None
                    llm_properties = None
                else:
                    print(f"Applying LLM properties: {llm_properties}")
                    print(f"user_id: {llm_properties['user_id']}, avatar_id: {llm_properties['avatar_id']}")
            
            # Get image_swap parameter
            image_swap = event.get("image_swap", False)
            if isinstance(image_swap, str):
                image_swap = image_swap.lower() == "true"
            
            # Get image_url parameter
            image_url = event.get("image_url", None)
            
            print(f"Starting room connection for room: {room_name}")
            print(f"Image swap setting: {image_swap}")
            print(f"Image URL: {image_url}")
            # Direct call since run_async_room_connection manages its own event loop
            run_async_room_connection(room_name, llm_properties, image_swap, image_url)
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "Room connection started successfully",
                    "room_name": room_name,
                    "applied_properties": llm_properties if llm_properties else None,
                    "image_swap": image_swap,
                    "image_url": image_url
                })
            }

        # Handle SQS events
        for record in event["Records"]:
            try:
                body = json.loads(record["body"])
                room_name = body.get("room_name", "test-room")
                # Get LLM properties and additional fields, filter out empty values
                llm_properties = {k: v for k, v in body.items() if k.startswith("llm_") or k == "tts_voice_id_cartesia" or k == "user_id" or k == "avatar_id"}
                if llm_properties:
                    llm_properties = {k: v for k, v in llm_properties.items() if v and str(v).strip()}
                    if not llm_properties:  # If all values were empty, set to None
                        llm_properties = None
                    else:
                        print(f"Applying LLM properties: {llm_properties}")
                        print(f"user_id: {llm_properties['user_id']}, avatar_id: {llm_properties['avatar_id']}")
                
                # Get image_swap parameter
                image_swap = body.get("image_swap", False)
                if isinstance(image_swap, str):
                    image_swap = image_swap.lower() == "true"
                
                # Get image_url parameter
                image_url = body.get("image_url", None)
                
                print(f"Starting room connection for room: {room_name}")
                print(f"Image swap setting: {image_swap}")
                print(f"Image URL: {image_url}")
                # Direct call since run_async_room_connection manages its own event loop
                run_async_room_connection(room_name, llm_properties, image_swap, image_url)
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

async def send_discord_webhook(room_name: str, user_id: str = None, avatar_id: str = None, message_type: str = "joining"):
    """
    Send a Discord webhook notification when a user joins or leaves a room
    
    Args:
        room_name: Name of the room
        user_id: User ID (optional)
        avatar_id: Avatar ID (optional)
        message_type: Type of message - "joining" or "leaving"
    """
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("DISCORD_WEBHOOK_URL not found in environment variables")
        return
    
    try:
        # Prepare the message
        user_info = ""
        if user_id and avatar_id:
            user_info = f" (User: {user_id}, Avatar: {avatar_id})"
        elif user_id:
            user_info = f" (User: {user_id})"
        elif avatar_id:
            user_info = f" (Avatar: {avatar_id})"
        
        action = "joining" if message_type == "joining" else "leaving"
        # Add emojis for better visibility
        if message_type == "joining":
            emoji_prefix = "🟢🤖"
        else:  # leaving
            emoji_prefix = "🔴🤖"
        
        message = f"{emoji_prefix} Rita:LLM: user is {action} the room: {room_name}{user_info}"
        
        # Prepare the webhook payload
        payload = {
            "content": message
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status == 204:
                    print(f"Discord webhook sent successfully: {message}")
                else:
                    print(f"Failed to send Discord webhook. Status: {response.status}")
                    
    except Exception as e:
        print(f"Error sending Discord webhook: {e}")

async def send_user_not_entered_webhook(room_name: str, user_id: str = None, avatar_id: str = None, loop_count: int = 0):
    """
    Send a Discord webhook notification when a user doesn't enter the room after multiple attempts
    
    Args:
        room_name: Name of the room
        user_id: User ID (optional)
        avatar_id: Avatar ID (optional)
        loop_count: Number of loops without remote participants
    """
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("DISCORD_WEBHOOK_URL not found in environment variables")
        return
    
    try:
        # Prepare the message
        user_info = ""
        if user_id and avatar_id:
            user_info = f" (User: {user_id}, Avatar: {avatar_id})"
        elif user_id:
            user_info = f" (User: {user_id})"
        elif avatar_id:
            user_info = f" (Avatar: {avatar_id})"
        
        # Add emojis for better visibility
        emoji_prefix = "⚠️🤖"
        
        message = f"{emoji_prefix} Rita:LLM: No remote participants available in room: {room_name}{user_info} (Loop count: {loop_count})"
        
        # Prepare the webhook payload
        payload = {
            "content": message
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status == 204:
                    print(f"User not entered webhook sent successfully: {message}")
                else:
                    print(f"Failed to send user not entered webhook. Status: {response.status}")
                    
    except Exception as e:
        print(f"Error sending user not entered webhook: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start the room connection with a specified room name')
    parser.add_argument('--room', type=str, default='test-room', help='Name of the room to connect to')
    parser.add_argument('--properties', type=str, help='JSON string containing LLM properties')
    parser.add_argument('--image-swap', action='store_true', help='Enable image swap functionality')
    parser.add_argument('--image-url', type=str, help='URL of the image to use')
    args = parser.parse_args()
    
    llm_properties = None
    if args.properties:
        try:
            # Parse the properties directly
            properties = json.loads(args.properties)
            # Filter for LLM properties and additional fields
            llm_properties = {k: v for k, v in properties.items() if k.startswith("llm_") or k == "tts_voice_id_cartesia" or k == "user_id" or k == "avatar_id"}
            # Filter out empty values
            if llm_properties:
                llm_properties = {k: v for k, v in llm_properties.items() if v and str(v).strip()}
                if not llm_properties:  # If all values were empty, set to None
                    llm_properties = None
                else:
                    print(f"Applying LLM properties: {llm_properties}")
                    print(f"user_id: {llm_properties['user_id']}, avatar_id: {llm_properties['avatar_id']}")
        except json.JSONDecodeError as e:
            print(f"Error parsing properties JSON: {e}")
            exit(1)
    
    print(f"Image swap setting: {args.image_swap}")
    print(f"Image URL: {args.image_url}")
    run_async_room_connection(args.room, llm_properties, args.image_swap, args.image_url)
