import logging
from pathlib import Path
import sys
import os
from openai import OpenAI
import time
import requests
import json
from livekit import rtc  # Add LiveKit import
from livekit.rtc import TextStreamWriter  # Add TextStreamWriter import
import asyncio
from dotenv import load_dotenv
from typing import Optional, List
import glob
import random
from typing import Dict

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import config

# Import Chat utils
from generation.generation_util.text_chunk_spliter import TextChunkSpliter
from generation.generation_util.system_prompt import LLM_System_Prompt

# Import chat db system
from data import ChatSessionManager, DatabaseManager

# Import world agent
from generation.world_agent import WorldAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class ASR_LLM_Manager:
    def __init__(
        self,
        user_id: str = None,
        avatar_id: str = None,
        room: rtc.Room = None,  # Add type hint for LiveKit room
        loop: asyncio.AbstractEventLoop = None,  # Add event loop parameter
        image_swap: bool = False,  # Enable image swap and world agent
        image_url: str = None,  # Base image URL for world agent
    ):
        # Initialize Deepinfra client for Whisper ASR
        self.openai_client = OpenAI(
            api_key=os.getenv("DEEPINFRA_API_KEY"),
            base_url="https://api.deepinfra.com/v1/openai",
        )
        if not self.openai_client.api_key:
            raise ValueError("DEEPINFRA_API_KEY environment variable is not set")

        self.avatar_id = avatar_id
        self.user_id = user_id
        # Store room reference and event loop
        self.room = room
        self.loop = loop  # Store the event loop reference
        self.text_stream_writer: TextStreamWriter | None = None  # Add type hint for text stream writer
        self._is_shutting_down = False  # Add shutdown flag

        # Text publishing queue for non-blocking operation
        self.tts_text_queue = asyncio.Queue()
        self.tts_text_publisher_task = None
        # Image swap functionality and world agent integration
        self.image_swap = image_swap  # Track image swap setting
        
        # World agent integration (enabled when image_swap is True)
        self.world_agent = None

        # Token usage tracking
        self.current_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0,
            "tts_tokens": 0  # Track tokens for TTS (dialogue only, excluding narrative)
        }

        # OpenRouter configuration
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.openrouter_headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        }
        # Initialize simplified chat session manager
        self.chat_session_manager = ChatSessionManager()
        # chat metadata and session config
        self._load_llm_data() # load name, prompt, greeting, user nickname

        system_prompt_obj: LLM_System_Prompt = LLM_System_Prompt(
            character_name=self.character_name,
            character_prompt=self.character_prompt,
            user_preferred_name=self.user_preferred_name,
        )
        self.system_prompt = system_prompt_obj.get_system_prompt()

        # World agent integration (enabled when image_swap is True)
        if self.image_swap:
            try:
                self.world_agent = WorldAgent(
                    character_system_prompt=self.system_prompt,
                    mcp_server_path="generation/chat_server.py",  # Fixed MCP server path
                    image_url=image_url,
                    room=room,
                    loop=loop,
                    chat_session_manager=self.chat_session_manager,
                    user_id=self.user_id,
                    avatar_id=self.avatar_id,
                    assistant_name=self.character_name or "Assistant"
                )
                logger.info("World agent initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize world agent: {e}")
                self.world_agent = None

        # Initialize messages with system prompt (always first, not stored in DB)
        self.messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        # Load conversation history from database and populate messages
        self._load_conversation_history()

        # Initialize text chunk splitter
        self.text_chunk_spliter = TextChunkSpliter()

        # Timing information
        self.timing = {
            "speech_end_time": -1,
            "whisper_end_time": -1,
            "llm_first_token_time": -1,
        }
        self.user_interrupting_flag = False
        
        # Schedule the async task on the proper event loop
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.start_tts_text_publisher(), self.loop)
        else:
            logger.warning("No event loop provided, TTS text publisher not started")

        # Image sending functionality, remove me when we are actually have images from genai. 
        self.test_images = self._load_test_images()
        self.current_image_index = 0

    async def start_tts_text_publisher(self):
        """Start the background text publisher task"""
        if self.tts_text_publisher_task is None:
            self.tts_text_publisher_task = asyncio.create_task(self._tts_text_publisher_worker())
            print("Text publisher worker initialized")

    async def _tts_text_publisher_worker(self):
        """Background worker that processes text publishing queue"""
        print("Text publisher worker thread started")
        while not self._is_shutting_down:
            try:
                # Wait for text with short timeout to check shutdown flag
                text = await asyncio.wait_for(self.tts_text_queue.get(), timeout=0.001)  # 1ms timeout
                print(f"TTS Text Publisher: Processing text: {text}")
                # Process the text based on type
                if text.startswith("[SLEEP] "):
                    # Handle sleep command
                    try:
                        parts = text.split(" ", 1)
                        if len(parts) > 1:
                            sleep_duration = float(parts[1])
                            print(f"Sleeping for {sleep_duration} seconds")
                            await asyncio.sleep(sleep_duration)
                        else:
                            logger.warning("Sleep command missing duration")
                    except ValueError:
                        logger.error(f"Invalid sleep duration in: {text}")
                else:
                    # Handle all other text (control flags and regular content)
                    if self._is_shutting_down:
                        logger.warning("Skipping text publish during shutdown")
                    elif not self.room:
                        logger.error("No LiveKit room available for publishing")
                    else:
                        try:
                            if text == "[START]":
                                logger.info("=== SENDING [START] MARKER ===")
                                # Start a new text stream
                                self.text_stream_writer = await self.room.local_participant.stream_text(
                                    topic="llm_data"
                                )
                                # Send the [START] marker first
                                await self.text_stream_writer.write("[START]")
                                logger.info("=== [START] MARKER SENT AND STREAM INITIALIZED ===")
                            
                            elif text == "[DONE]" or text == "[INTERRUPTED]":
                                if self.text_stream_writer:
                                    await self.text_stream_writer.write(text)  # Send the marker before closing
                                    await self.text_stream_writer.aclose()
                                    logger.info(f"Closed text stream with marker: {text}")
                                    self.text_stream_writer = None
                            
                            else:
                                # Regular text chunk - add to TTS token count
                                self.add_tts_tokens(text)
                                if self.text_stream_writer:
                                    await self.text_stream_writer.write(text)
                                else:
                                    logger.error("Attempted to write text chunk but no active stream")
                                    
                        except Exception as e:
                            logger.error(f"Error publishing text to LiveKit: {e}")
                            # Try to clean up the stream if there was an error
                            if self.text_stream_writer:
                                try:
                                    await self.text_stream_writer.aclose()
                                except:
                                    pass
                                self.text_stream_writer = None
                
                # Mark task as done
                self.tts_text_queue.task_done()
                
            except asyncio.TimeoutError:
                # Timeout is normal, just continue to check shutdown flag
                continue
            except Exception as e:
                logger.error(f"Error in text publisher worker: {e}")
                # Continue running even on errors
                continue
        
        logger.info("Text publisher worker thread stopped")
        
    def _load_llm_data(self):
        "Load character name, prompt, opening prompt(greeting), user nickname "
        try:
            # Load avatar data from avatars table
            avatar_data = self.chat_session_manager.load_avatar_metadata(self.avatar_id)
            if avatar_data:
                self.character_name = avatar_data.get('avatar_name', '')
                self.character_prompt = avatar_data.get('prompt', '')
                self.avatar_opening_prompt = avatar_data.get('opening_prompt', '')
                logger.info(f"Loaded avatar data: name={self.character_name}, prompt_length={len(self.character_prompt) if self.character_prompt else 0}, opening_prompt_length={len(self.avatar_opening_prompt) if self.avatar_opening_prompt else 0}")
            else:
                logger.warning(f"Failed to load avatar data for avatar_id: {self.avatar_id}")
                self.character_name = ''
                self.character_prompt = ''
                self.avatar_opening_prompt = ''
            
            # Load user preferred name from users table
            preferred_name = self.chat_session_manager.load_user_preferred_name(self.user_id)
            if preferred_name:
                self.user_preferred_name = preferred_name
                logger.info(f"Loaded user preferred name: {self.user_preferred_name}")
            else:
                logger.warning(f"Failed to load preferred name for user_id: {self.user_id}")
                self.user_preferred_name = ''
                
        except Exception as e:
            logger.error(f"Error in _load_llm_data: {e}")
            # Set default values on error
            self.character_name = ''
            self.character_prompt = ''
            self.avatar_opening_prompt = ''
            self.user_preferred_name = ''

    def _load_test_images(self) -> list:
        """Load all test images from the test folder"""
        try:
            test_folder = Path(__file__).parent / "util" / "test_images"
            if not test_folder.exists():
                logger.warning(f"Test folder not found: {test_folder}")
                return []
            
            # Get all image files (png, jpg, jpeg, gif, etc.)
            image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.webp"]
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(glob.glob(str(test_folder / ext)))
                image_files.extend(glob.glob(str(test_folder / ext.upper())))
            
            if not image_files:
                logger.warning(f"No image files found in test folder: {test_folder}")
                return []
            
            logger.info(f"Loaded {len(image_files)} test images: {[Path(f).name for f in image_files]}")
            return image_files
            
        except Exception as e:
            logger.error(f"Failed to load test images: {e}")
            return []

    def _load_conversation_history(self):
        """Load conversation history from database and populate self.messages"""
        try:
            # Read history using simplified interface
            history = self.chat_session_manager.read_history(
                user_id=self.user_id,
                avatar_id=self.avatar_id,
                max_messages=50  # Load recent 50 messages
            )
            
            # Add history to messages (system prompt is already first)
            if history:
                self.messages.extend(history)
                logger.info(f"Loaded {len(history)} messages from conversation history")
            else:
                # No history exists, try to load opening_prompt from avatars table
                opening_prompt = self.avatar_opening_prompt
                if opening_prompt:
                    # Add opening prompt as first assistant message
                    opening_message = {"role": "assistant", "content": opening_prompt}
                    self.messages.append(opening_message)
                    
                    # Save opening prompt to database as first assistant message
                    try:
                        self.chat_session_manager.write_assistant_message(
                            user_id=self.user_id,
                            avatar_id=self.avatar_id,
                            content=opening_prompt,
                            assistant_name=self.character_name or "Assistant",
                            model="opening_prompt"  # Special model name to indicate this is an opening
                        )
                        logger.info(f"Added opening prompt as first assistant message")
                    except Exception as e:
                        logger.error(f"Failed to save opening prompt to database: {e}")
            
        except Exception as e:
            logger.error(f"Failed to load conversation history: {e}")

    def speech_to_text(self, audio_file_path, speech_end_time):
        """Convert speech to text using Whisper API"""
        with open(audio_file_path, "rb") as audio_file:
            transcription = self.openai_client.audio.transcriptions.create(
                model="openai/whisper-large-v3-turbo",
                file=audio_file,
                language="en",  # Optional: Specify language
            )
        self.timing["speech_end_time"] = speech_end_time
        self.timing["whisper_end_time"] = time.time()
        logger.info(
            f"Time to transcribe: {self.timing['whisper_end_time'] - speech_end_time:.2f} seconds"
        )
        logger.info(f"Transcription: {transcription.text}")

        return transcription.text

    async def cleanup(self):
        """Gracefully cleanup resources before shutdown"""
        self._is_shutting_down = True
        try:
            # Stop the text publisher worker
            if self.tts_text_publisher_task:
                try:
                    self.tts_text_publisher_task.cancel()
                    await self.tts_text_publisher_task
                except asyncio.CancelledError:
                    pass  # Expected when cancelling
                except Exception as e:
                    logger.warning(f"Error stopping text publisher task: {e}")
                finally:
                    self.tts_text_publisher_task = None

            # Wait for any remaining queue items to be processed
            try:
                await asyncio.wait_for(self.tts_text_queue.join(), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for text queue to finish")
            except Exception as e:
                logger.warning(f"Error waiting for text queue: {e}")

            # Close text stream if it exists
            if self.text_stream_writer:
                try:
                    await self.text_stream_writer.aclose()
                except Exception as e:
                    logger.warning(f"Error closing text stream: {e}")
                finally:
                    self.text_stream_writer = None

            # Disconnect from room if it exists
            if self.room:
                try:
                    await self.room.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting from room: {e}")
                finally:
                    self.room = None

            # Cleanup world agent
            if self.world_agent:
                try:
                    await self.world_agent.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up world agent: {e}")
                finally:
                    self.world_agent = None

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            self._is_shutting_down = False

    @staticmethod
    async def shutdown_event_loop():
        """Safely shutdown the event loop"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Cancel all running tasks
                for task in asyncio.all_tasks(loop):
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            logger.warning(f"Error cancelling task: {e}")
                
                # Stop the loop
                loop.stop()
        except Exception as e:
            logger.error(f"Error during event loop shutdown: {e}")

    async def publish_tts_text(self, text):
        """
        Queue text for publishing (non-blocking)
        """
        if not self._is_shutting_down:
            await self.tts_text_queue.put(text)
            print(f"Queued text for publishing: {text[:50] if text else 'N/A'}")
        else:
            print("Skipping text queuing during shutdown")

    async def publish_frontend_stream_livekit(self, stream_type, content):
        """
        Publish frontend streaming content directly to LiveKit data channel
        """
        if self._is_shutting_down:
            logger.warning("Skipping frontend stream publish during shutdown")
            return

        if not self.room:
            logger.error("No LiveKit room available for frontend streaming")
            return

        try:
            await self.room.local_participant.publish_data(
                json.dumps({
                    "topic": "frontend_stream",
                    "type": stream_type,
                    "text": content
                })
            )
            logger.debug(f"Published frontend stream: {stream_type} - {content[:50] if content else 'N/A'}")
        except Exception as e:
            logger.error(f"Error publishing frontend stream to LiveKit: {e}")

    async def send_image_to_livekit(self):
        """
        Send a test image to the LiveKit data channel with topic 'image_file'
        Cycles through available test images
        """
        if self._is_shutting_down:
            logger.warning("Skipping image send during shutdown")
            return

        if not self.room:
            logger.error("No LiveKit room available for image sending")
            return

        if not self.test_images:
            logger.warning("No test images available to send")
            return

        try:
            # Get the current image file
            image_path = self.test_images[self.current_image_index]
            image_name = Path(image_path).name
            
            logger.info(f"Sending image: {image_name}")
            
            # Send the image file using LiveKit's send_file method
            info = await self.room.local_participant.send_file(
                file_path=image_path,
                topic="image_file",
            )
            
            logger.info(f"Successfully sent image '{image_name}' with stream ID: {info.stream_id}")
            
            # Move to next image (cycle through available images)
            self.current_image_index = (self.current_image_index + 1) % len(self.test_images)
            
        except Exception as e:
            logger.error(f"Error sending image to LiveKit: {e}")
            # Try to move to next image even if current one failed
            self.current_image_index = (self.current_image_index + 1) % len(self.test_images)

    def final_response_format_check(self, text):
        response = text.strip()        
        return response

    def add_tts_tokens(self, segment: str):
        """
        Add TTS tokens based on character count of the segment.
        
        Args:
            segment: The text segment being sent to TTS
        """
        if segment and not segment.startswith('[') and not segment.endswith(']'):
            # Only count actual text content, ignore control flags like [START], [DONE], [INTERRUPTED]
            char_count = len(segment)
            self.current_usage["tts_tokens"] += char_count
            logger.debug(f"Added {char_count} TTS characters: '{segment[:50]}{'...' if len(segment) > 50 else ''}'")
        else:
            logger.debug(f"Skipping TTS character count for control flag: {segment}")

    def get_tts_tokens(self) -> int:
        """
        Get the current TTS token usage (dialogue only, excluding narrative).
        
        Returns:
            int: Current TTS token count
        """
        return self.current_usage.get("tts_tokens", 0)

    def get_token_usage(self) -> dict:
        """
        Get the complete token usage information.
        
        Returns:
            dict: Complete token usage including prompt, completion, total, TTS, and cost
        """
        return self.current_usage.copy()

    def get_recent_messages(self, count: int = 6) -> List[Dict[str, str]]:
        """
        Get the last `count` non-image messages (user/assistant dialogue only).
        Exclude messages that are image bubbles (i.e., those with an 'imageUrl' field or model == 'world_agent_image_generation').
        """
        def is_dialogue_message(msg):
            # Exclude if message has an imageUrl (image bubble)
            if 'imageUrl' in msg and msg['imageUrl']:
                return False
            # Exclude if model is world_agent_image_generation (image bubble)
            if msg.get('model') == 'world_agent_image_generation':
                return False
            return True

        filtered = [msg for msg in self.messages if is_dialogue_message(msg)]
        return filtered[-count:]

    def get_filtered_messages_for_llm(self) -> List[Dict[str, str]]:
        """
        Get messages filtered for the character LLM (exclude image bubbles).
        This prevents the LLM from learning to mimic image generation prompts.
        """
        def is_dialogue_message(msg):
            # Exclude if message has an imageUrl (image bubble)
            if 'imageUrl' in msg and msg['imageUrl']:
                return False
            # Exclude if model is world_agent_image_generation (image bubble)
            if msg.get('model') == 'world_agent_image_generation':
                return False
            return True

        return [msg for msg in self.messages if is_dialogue_message(msg)]

    async def trigger_world_agent_analysis(self):
        """
        Trigger world agent analysis after a new message is added.
        This should be called after the assistant message is added to self.messages.
        """
        if not self.image_swap or not self.world_agent:
            logger.debug("World agent analysis skipped - image_swap disabled or world_agent not available")
            return
            
        try:
            logger.info("World agent analysis starting...")
            # Get the last 6 messages for context
            recent_messages = self.get_recent_messages(6)
            
            if recent_messages:
                logger.info(f"Triggering world agent analysis with {len(recent_messages)} recent messages")
                
                # Process the conversation update with world agent
                image_generated = await self.world_agent.process_conversation_update(recent_messages)
                
                if image_generated:
                    logger.info("World agent successfully generated and sent an image")
                else:
                    logger.debug("World agent decided no image generation was needed")
            else:
                logger.warning("No recent messages found for world agent analysis")
                    
        except Exception as e:
            logger.error(f"Error triggering world agent analysis: {e}")
        finally:
            logger.info("World agent analysis completed")

    def trigger_world_agent_analysis_sync(self):
        """
        Synchronous wrapper that runs the async world agent analysis in a new event loop.
        This method runs in a separate thread to avoid blocking the main event loop.
        """
        if not self.image_swap or not self.world_agent:
            logger.debug("World agent analysis skipped - image_swap disabled or world_agent not available")
            return
            
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info("World agent analysis starting in separate thread...")
            # Get the last 6 messages for context
            recent_messages = self.get_recent_messages(6)
            
            if recent_messages:
                logger.info(f"Triggering world agent analysis with {len(recent_messages)} recent messages")
                
                # Process the conversation update with world agent
                image_generated = loop.run_until_complete(
                    self.world_agent.process_conversation_update(recent_messages)
                )
                
                if image_generated:
                    logger.info("World agent successfully generated and sent an image")
                else:
                    logger.debug("World agent decided no image generation was needed")
            else:
                logger.warning("No recent messages found for world agent analysis")
                    
        except Exception as e:
            logger.error(f"Error triggering world agent analysis: {e}")
        finally:
            logger.info("World agent analysis completed in separate thread")
            loop.close()
    
    def replace_special_quotes_to_straight_quotes(self, input_prompt: str) -> str:
        """Replace special quotes to straight quotes"""
        if not input_prompt:
            return input_prompt
            
        # Replace various types of curly quotes with straight quotes
        replacements = {
            '“': '"',  # Left double quotation mark
            '”': '"',  # Right double quotation mark         
        }
        
        result = input_prompt
        for special_quote, straight_quote in replacements.items():
            if special_quote in result:
                logger.info(f"Replacing {special_quote} with {straight_quote}")
                result = result.replace(special_quote, straight_quote)
        return result

    async def send_to_openrouter(self, text):
        """
        1. receives text from ASR or user input.
        2. send text to LLM and get streaming response.
        3. print the response and publish to LiveKit.
        """
        self.user_interrupting_flag = False
        first_llm_token_received = False
        current_response = ""
        current_token_usage = 0  # Track tokens for this interaction
        # State tracking for dialogue/narrative detection
        is_on_dialogue = False    # True when inside " dialogue "
        is_tts_started = False
        past_quote_count = 0
        track_char_index = 0
        consecutive_narrative_chars = 0 
        consecutive_dialogue_chars = 0  # "Prev dialogue speak time" + "prev Narrative read time" = "New prev-dialogue sleep time"

        # Add user message to self.messages (single source of truth)
        self.messages.append({"role": "user", "content": text})
        
        # Save user message to database
        try:
            self.chat_session_manager.write_user_message(
                user_id=self.user_id,
                avatar_id=self.avatar_id,
                content=text,
                user_name=self.user_preferred_name or "User"
            )
        except Exception as e:
            logger.error(f"Failed to save user message to database: {e}")

        payload = {
            "model": "deepseek/deepseek-chat-v3-0324",
            "messages": self.get_filtered_messages_for_llm(),
            "stream": True,
            "provider": {
                'order': 
                [
                    'deepinfra/fp4',
                    'lambda/fp8',
                    'baseten/fp8',
                ]
                },
            "usage": {
                "include": True
            }
        }

        with requests.post(
            self.openrouter_url,
            headers=self.openrouter_headers,
            json=payload,
            stream=True,
        ) as r:
            r.raise_for_status()
            # Ensure proper UTF-8 encoding
            r.encoding = "utf-8"
            for chunk in r.iter_lines(chunk_size=1024, decode_unicode=True):
                if chunk:  # filter out keep-alive new chunks
                    line = chunk.strip()
                    if not first_llm_token_received:
                        first_llm_token_received = True
                        
                        # Send START marker to frontend immediately via LiveKit
                        await self.publish_frontend_stream_livekit("START", "")
                        
                        self.timing["llm_first_token_time"] = time.time()
                        if self.timing["whisper_end_time"] != -1:
                            logger.info(
                                f"Time from whisper end to LLM first token: {self.timing['llm_first_token_time'] - self.timing['whisper_end_time']:.2f} seconds"
                            )
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            # Process any remaining tts buffer content
                            remaining_segments = (
                                self.text_chunk_spliter.get_remaining_buffer()
                            )
                            for segment in remaining_segments:
                                if not is_tts_started:
                                    is_tts_started = True
                                    await self.publish_tts_text("[START]")
                                logger.info(f"sending segment in final: {segment} \n")
                                await self.publish_tts_text(segment)

                            # Send [DONE] to TTS to close the text stream
                            await self.publish_tts_text("[DONE]")

                            # TTS tokens are now tracked in real-time via publish_text_livekit
                            logger.info(f"Total TTS characters sent: {self.current_usage['tts_tokens']}")

                            # Add assistant message (single source of truth)
                            current_response = self.final_response_format_check(current_response)
                            self.messages.append(
                                {"role": "assistant", "content": current_response}
                            )
                            logger.info(f"Current response: {current_response}")
                            print(f"Debug: Added assistant message to messages list. Total messages: {len(self.messages)}")

                            # Trigger world agent analysis after assistant message is added (non-blocking)
                            logger.info("Starting world agent analysis in background task")
                            asyncio.create_task(asyncio.to_thread(self.trigger_world_agent_analysis_sync))
                            logger.info("World agent analysis task created, continuing with TTS flow")
                            
                            # Save LLM response to database
                            try:
                                self.chat_session_manager.write_assistant_message(
                                    user_id=self.user_id,
                                    avatar_id=self.avatar_id,
                                    content=current_response,
                                    assistant_name=self.character_name or "Assistant",
                                    model="google/gemini-2.5-flash-preview-05-20"
                                )
                                
                            except Exception as e:
                                logger.error(f"Failed to save LLM response to database: {e}")
                            
                            # Send DONE marker to frontend immediately via LiveKit
                            await self.publish_frontend_stream_livekit("DONE", "")
                            
                                                        
                            break

                        try:
                            data_obj = json.loads(data)
                            
                            # Check for usage information
                            if "usage" in data_obj:
                                usage = data_obj["usage"]
                                # Accumulate all usage values
                                self.current_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                                self.current_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                                self.current_usage["total_tokens"] += usage.get("total_tokens", 0)
                                self.current_usage["cost"] += usage.get("cost", 0)
                                
                                logger.info(f"Token usage - Prompt: {self.current_usage['prompt_tokens']}, "
                                          f"Completion: {self.current_usage['completion_tokens']}, "
                                          f"Total: {self.current_usage['total_tokens']}, "
                                          f"Cost: {self.current_usage['cost']}")
                            
                            content = data_obj["choices"][0]["delta"].get("content")
                            content = self.replace_special_quotes_to_straight_quotes(content)
                            if content:
                                current_response += content
                                
                                # Send delta content to frontend immediately via LiveKit
                                await self.publish_frontend_stream_livekit("CONTENT", content)
                                
                                # Track quotation marks for dialogue/narrative detection
                                while track_char_index < len(current_response):
                                    checking_char = current_response[track_char_index]
                                    if checking_char == '"':
                                        past_quote_count += 1
                                        was_on_dialogue = is_on_dialogue
                                        is_on_dialogue = (past_quote_count % 2 == 1)
                                        
                                        if not was_on_dialogue and is_on_dialogue:
                                            # We're switching from narrative to dialogue, send sleep before dialogue
                                            if consecutive_narrative_chars > 0 or consecutive_dialogue_chars > 0:
                                                sleep_duration = consecutive_narrative_chars * 0.01 + consecutive_dialogue_chars * 0.065
                                                print(f"Sending sleep command: {consecutive_narrative_chars} chars = {sleep_duration:.3f}s")
                                                await self.publish_tts_text(f"[SLEEP] {sleep_duration}")
                                                consecutive_narrative_chars = 0
                                                consecutive_dialogue_chars = 0  # Reset counter
                                        
                                        elif was_on_dialogue and not is_on_dialogue:
                                            # We're switching from dialogue to narrative, Process any remaining tts buffer content
                                            if len(self.text_chunk_spliter.buffer) > 0:
                                                remaining_segments = (
                                                    self.text_chunk_spliter.get_remaining_buffer()
                                                )
                                                for segment in remaining_segments:
                                                    logger.info(f"clearing tts buffer when switching to narrative: {segment}")
                                                    if not is_tts_started:
                                                        is_tts_started = True
                                                        await self.publish_tts_text("[START]")
                                                    await self.publish_tts_text(segment)
                                            # Reset narrative counter when exiting dialogue
                                            consecutive_narrative_chars = 0
                                            is_tts_started = False
                                            await self.publish_tts_text("[DONE]")
                                        
                                        # Don't send [DONE] when switching modes - only send it when LLM response is complete
                                        track_char_index += 1
                                        
                                    elif is_on_dialogue:
                                        # We're in dialogue mode, send to TTS buffer
                                        segments = self.text_chunk_spliter.process_chunk(checking_char)
                                        for segment in segments:
                                            if not is_tts_started:
                                                is_tts_started = True
                                                await self.publish_tts_text("[START]")
                                            await self.publish_tts_text(segment)
                                        
                                        track_char_index += 1
                                        consecutive_dialogue_chars += 1
                                    else:
                                        # We're in narrative mode, count characters for timing
                                        consecutive_narrative_chars += 1
                                        track_char_index += 1
                                        if is_tts_started:
                                            is_tts_started = False
                                            await self.publish_tts_text("[DONE]")

                        except json.JSONDecodeError:
                            logger.warning(f"Could not decode JSON data: {data}")
                            pass  # Ignore malformed JSON lines
                        except Exception as e:
                            logger.error(f"Error processing stream data chunk: {e}")
                            break  # Safer to break the inner loop on unexpected errors

                    if self.user_interrupting_flag:
                        logger.warning("Stopping LLM stream due to user interruption.")
                        self.user_interrupting_flag = False
                        print("\n[INTERRUPTED]")
                        await self.publish_tts_text("[INTERRUPTED]")
                        
                        # Send INTERRUPTED marker to frontend immediately via LiveKit
                        await self.publish_frontend_stream_livekit("INTERRUPTED", "")
                        
                        # World agent handles image generation and sending (no legacy image swap)
                        
                        break
        if self.user_interrupting_flag:
            logger.warning(f"Skipping history appending due to user interruption")
        
        return self.current_usage["total_tokens"]  # Return the total tokens for backward compatibility

    def _extract_s3_key_from_url(self, image_url: str) -> str:
        """
        Extract S3 key from a full S3 URL.
        
        Args:
            image_url: Full S3 URL with query parameters
            
        Returns:
            S3 key (e.g., 'rita-swap-images/user_id/avatar_id/filename.jpg')
        """
        try:
            from urllib.parse import urlparse
            
            # Parse the URL
            parsed = urlparse(image_url)
            
            # Extract the path and remove leading slash
            s3_key = parsed.path.lstrip('/')
            
            logger.debug(f"Extracted S3 key '{s3_key}' from URL: {image_url}")
            return s3_key
            
        except Exception as e:
            logger.error(f"Error extracting S3 key from URL {image_url}: {e}")
            return image_url  # Return original URL if extraction fails

    def remove_image_message(self, message_id: str, image_url: str) -> bool:
        """
        Remove an image message from database.
        Loads full message data only when needed for removal.
        Matches by imageUrl since it's a unique identifier.
        Handles both full S3 URLs and S3 keys.
        
        Args:
            message_id: The ID of the message to remove (from frontend, kept for logging)
            image_url: The image URL to identify the message to remove (can be full S3 URL or S3 key)
            
        Returns:
            True if message was found and removed, False otherwise
        """
        try:
            logger.info(f"Attempting to remove image message with URL: {image_url}")
            
            # Load full messages from database (only when needed)
            if not self.chat_session_manager or not self.user_id or not self.avatar_id:
                logger.error("Chat session manager not available for image message removal")
                return False
            
            current_messages = self.chat_session_manager.read_full_history(
                user_id=self.user_id,
                avatar_id=self.avatar_id
            )
            
            # Extract S3 key if the URL is a full S3 URL
            search_key = image_url
            if 'amazonaws.com' in image_url and '?' in image_url:
                search_key = self._extract_s3_key_from_url(image_url)
                logger.info(f"Extracted S3 key for search: {search_key}")
            
            # Find and remove the message from the database messages
            # Match by imageUrl or origin_image_url since either could be used for removal
            db_message_found = False
            for i, db_message in enumerate(current_messages):
                # Check if the message matches by imageUrl or origin_image_url
                matches_image_url = (hasattr(db_message, 'imageUrl') and 
                                   db_message.imageUrl == search_key)
                matches_origin_url = (hasattr(db_message, 'origin_image_url') and 
                                    db_message.origin_image_url == search_key)
                
                if ((matches_image_url or matches_origin_url) and
                    db_message.role == 'assistant'):
                    
                    logger.info(f"Found image message in database to remove at index {i}: {db_message.to_dict()}")
                    logger.info(f"Matched by imageUrl: {matches_image_url}, origin_image_url: {matches_origin_url}")
                    
                    # Remove the message from the list
                    current_messages.pop(i)
                    db_message_found = True
                    break
            
            if not db_message_found:
                logger.warning(f"Image message not found in database with URL/key: {search_key}")
                return False
            
            # Update the database session with the modified messages list
            session_id = self.chat_session_manager.update_session_messages(
                user_id=self.user_id,
                avatar_id=self.avatar_id,
                messages=current_messages
            )
            
            if session_id:
                logger.info(f"Successfully updated database session {session_id} after removing image message")
                return True
            else:
                logger.error("Failed to update database session after removing image message")
                return False
                
        except Exception as e:
            logger.error(f"Error removing image message: {e}")
            return False
