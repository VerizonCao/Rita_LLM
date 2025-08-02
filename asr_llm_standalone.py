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
from generation.generation_util.text_format import replace_special_quotes_to_straight_quotes

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
        self.hint_prompt = system_prompt_obj.get_hint_prompt()

        # World agent integration (enabled when image_swap is True)
        if self.image_swap:
            try:
                self.world_agent = WorldAgent(
                    character_system_prompt=self.system_prompt,
                    mcp_server_path="generation/chat_server.py",  # Fixed MCP server path
                    room=room,
                    loop=loop,
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
        self.current_llm_model = "deepseek/deepseek-chat-v3-0324"
        self.current_image_gen_model = "black-forest-labs/flux-kontext-dev"

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

    # =========== Event Loop Related Functions ===========
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

    # =========== TTS Worker Related Functions ===========
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
        
    # =========== Livekit Publishing Related Functions ===========
    async def publish_tts_text(self, text):
        """
        Queue text for publishing (non-blocking)
        """
        if not self._is_shutting_down:
            await self.tts_text_queue.put(text)
            print(f"Queued text for publishing: {text[:50] if text else 'N/A'}")
        else:
            print("Skipping text queuing during shutdown")

    async def publish_frontend_stream_livekit(self, stream_type, content, message_id=''):
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
                    "text": content,
                    "message_id": message_id
                })
            )
            logger.debug(f"Published frontend stream: {stream_type} - {content[:50] if content else 'N/A'}")
        except Exception as e:
            logger.error(f"Error publishing frontend stream to LiveKit: {e}")

    # =========== Core Generation Functions ===========
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
            recent_messages = self.get_recent_messages_with_default_image(count=6)
            
            if recent_messages:
                logger.info(f"Triggering world agent analysis with {len(recent_messages)} recent messages")
                
                # Process the conversation update with world agent
                image_generated, image_prompt, s3_key, image_gen_message_id = loop.run_until_complete(
                    self.world_agent.process_conversation_update(recent_messages, 
                                                                 self.get_last_image_url(use_default_image=True))
                )
                
                if image_generated:
                    logger.info("World agent successfully generated and sent an image")
                    self.messages.append({"role": "assistant", "content": image_prompt, "imageUrl": s3_key, "message_id": image_gen_message_id})
                         
                    # 3. Save to database with s3_key as imageUrl
                    if self.chat_session_manager and self.user_id and self.avatar_id:
                        try:
                            # Create a message with the S3 key
                            self.chat_session_manager.write_assistant_message_with_image(
                                user_id=self.user_id,
                                avatar_id=self.avatar_id,
                                content=f"{image_prompt}",
                                imageUrl=s3_key,  # Store S3 key for permanent reference
                                assistant_name=self.character_name + ", image update",
                                model=self.current_image_gen_model,
                                message_id=image_gen_message_id
                            )
                        except Exception as e:
                            logger.error(f"Failed to save image to database: {e}")
                    else:
                        logger.warning("Chat session manager not available, skipping database save")
                else:
                    logger.debug("World agent decided no image generation was needed")
            else:
                logger.warning("No recent messages found for world agent analysis")
                    
        except Exception as e:
            logger.error(f"Error triggering world agent analysis: {e}")
        finally:
            logger.info("World agent analysis completed in separate thread")
            loop.close()
    
    async def send_to_openrouter(self, text, user_message_id):
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
        self.messages.append({"role": "user", "content": text, 'message_id': user_message_id})
        
        assistant_message_id = str(int(time.time() * 1000))
        # Save user message to database
        try:
            self.chat_session_manager.write_user_message(
                user_id=self.user_id,
                avatar_id=self.avatar_id,
                content=text,
                user_name=self.user_preferred_name or "User",
                message_id=user_message_id
            )
        except Exception as e:
            logger.error(f"Failed to save user message to database: {e}")

        payload = {
            "model": self.current_llm_model,
            "messages": self.messages,
            "stream": True,
            "provider": {
                'order': 
                [
                    'lambda/fp8',
                    'deepinfra/fp4',
                    'baseten/fp8',
                ]
                # 'sort': 'latency',
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
                        await self.publish_frontend_stream_livekit("START", content='', message_id=assistant_message_id)
                        
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
                            current_response = current_response.strip()
                            self.messages.append(
                                {"role": "assistant", "content": current_response, "message_id": assistant_message_id}
                            )
                            logger.info(f"Current response: {current_response}")
                            print(f"Debug: Added assistant message to messages list. Total messages: {len(self.messages)}")
                            
                            # Save LLM response to database
                            try:
                                self.chat_session_manager.write_assistant_message(
                                    user_id=self.user_id,
                                    avatar_id=self.avatar_id,
                                    content=current_response,
                                    assistant_name=self.character_name or "Assistant",
                                    model=self.current_llm_model,
                                    message_id=assistant_message_id
                                )
                                
                            except Exception as e:
                                logger.error(f"Failed to save LLM response to database: {e}")
                            
                            # Send DONE marker to frontend immediately via LiveKit
                            await self.publish_frontend_stream_livekit("DONE", content='', message_id=assistant_message_id)
                                          
                            # Trigger world agent analysis after assistant message is added (non-blocking)
                            logger.info("Starting world agent analysis in background task")
                            asyncio.create_task(asyncio.to_thread(self.trigger_world_agent_analysis_sync))
                            logger.info("World agent analysis task created, continuing with TTS flow")
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
                            content = replace_special_quotes_to_straight_quotes(content)
                            if content:
                                current_response += content
                                
                                # Send delta content to frontend immediately via LiveKit
                                await self.publish_frontend_stream_livekit("CONTENT", content, message_id=assistant_message_id)
                                
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
                        await self.publish_frontend_stream_livekit("INTERRUPTED", content='', message_id=assistant_message_id)
                        
                        # World agent handles image generation and sending (no legacy image swap)
                        
                        break
        if self.user_interrupting_flag:
            logger.warning(f"Skipping history appending due to user interruption")
        
        return self.current_usage["total_tokens"]  # Return the total tokens for backward compatibility

    async def generate_hint(self):
        """
        Generate hints for the user based on recent conversation history.
        Uses the same recent messages pattern as image generation.
        """
        try:
            # Get recent messages using the same method as image generation
            recent_messages = self.get_recent_messages_with_default_image(count=6)
            hint_prompt_msg = {
                "role": "system",
                "content": self.hint_prompt
            }
            hint_context_messsages = [hint_prompt_msg] + recent_messages
            
            # Create payload similar to send_to_openrouter but for hint generation
            payload = {
                "model": "deepseek/deepseek-chat-v3-0324",
                "messages": hint_context_messsages,
                "stream": False,  # Don't stream, wait for full response
                "provider": {
                    'sort': 'latency',
                },
                "usage": {
                    "include": True
                }
            }

            # Send request to OpenRouter
            response = requests.post(
                self.openrouter_url,
                headers=self.openrouter_headers,
                json=payload,
                stream=False  # Don't stream
            )
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            hint_content = response_data["choices"][0]["message"]["content"]
            
            # Update token usage
            if "usage" in response_data:
                usage = response_data["usage"]
                self.current_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                self.current_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                self.current_usage["total_tokens"] += usage.get("total_tokens", 0)
                self.current_usage["cost"] += usage.get("cost", 0)
            
            logger.info(f"Generated hints: {hint_content}...")
            
            # Publish to LiveKit with frontend_hint topic
            if self.room:
                try:
                    await self.room.local_participant.publish_data(
                        json.dumps({
                            "topic": "frontend_hint",
                            "type": "hint",
                            "text": hint_content
                        })
                    )
                    logger.info("Published hints to LiveKit frontend_hint topic")
                except Exception as e:
                    logger.error(f"Error publishing hints to LiveKit: {e}")
            else:
                logger.warning("No LiveKit room available for hint publishing")
                
        except Exception as e:
            logger.error(f"Error generating hints: {e}")

    # =========== Character/Messages Init & Get Recent Messages ===========
    def _load_llm_data(self):
        "Load character name, prompt, opening prompt(greeting), user nickname "
        try:
            # Load avatar data from avatars table
            avatar_data = self.chat_session_manager.load_avatar_metadata(self.avatar_id)
            if avatar_data:
                self.character_name = avatar_data.get('avatar_name', '')
                self.character_prompt = avatar_data.get('prompt', '')
                self.avatar_opening_prompt = avatar_data.get('opening_prompt', '')
                self.avatar_image_uri = avatar_data.get('image_uri', '')
                self.avatar_img_caption = avatar_data.get('img_caption', '')
                logger.info(f"Loaded avatar data: name={self.character_name}, prompt_length={len(self.character_prompt) if self.character_prompt else 0}, opening_prompt_length={len(self.avatar_opening_prompt) if self.avatar_opening_prompt else 0}, img_caption_length={len(self.avatar_img_caption) if self.avatar_img_caption else 0}")
            else:
                logger.warning(f"Failed to load avatar data for avatar_id: {self.avatar_id}")
                self.character_name = ''
                self.character_prompt = ''
                self.avatar_opening_prompt = ''
                self.avatar_image_uri = ''
                self.avatar_img_caption = ''
            
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
            self.avatar_image_uri = ''
            self.avatar_img_caption = ''
            self.user_preferred_name = ''

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
                    # Add opening prompt as first assistant message, default image as first image message
                    opening_message = {"role": "assistant", "content": opening_prompt, "message_id": ''}
                    self.messages.append(opening_message)
                    default_image_message = {"role": "assistant", 
                                             "content": "Default Image:\n" + self.avatar_img_caption, 
                                             "imageUrl": self.avatar_image_uri,
                                             "message_id": ''}
                    self.messages.append(default_image_message)
                    # Save opening prompt & first image to database as first assistant message
                    try:
                        self.chat_session_manager.write_assistant_message(
                            user_id=self.user_id,
                            avatar_id=self.avatar_id,
                            content=opening_prompt,
                            assistant_name=self.character_name or "Assistant",
                            model="opening_prompt",  # Special model name to indicate this is an opening
                            message_id='default_greeting' # use default_greeting for default greeting
                        )
                        self.chat_session_manager.write_assistant_message_with_image(
                            user_id=self.user_id,
                            avatar_id=self.avatar_id,
                            content=self.avatar_img_caption,
                            imageUrl=self.avatar_image_uri,
                            assistant_name=self.character_name or "Assistant",
                            model="character_default_image",  # Special model name to indicate this is an opening
                            message_id='default_image' # use default_image for default image
                        )
                        logger.info(f"Added opening prompt as first assistant message")
                    except Exception as e:
                        logger.error(f"Failed to save opening prompt to database: {e}")
            
        except Exception as e:
            logger.error(f"Failed to load conversation history: {e}")
    '''
    def get_recent_messages(self, count: int = 6, exclude_image_bubbles: bool = True) -> List[Dict[str, str]]:
        """
        Get the last `count` non-image messages (user/assistant dialogue only).
        Exclude messages that are image bubbles (i.e., those with an 'imageUrl' field).
        Exclude system prompt.
        """
        def is_dialogue_message(msg):
            # Exclude if message has an imageUrl (image bubble)
            if 'imageUrl' in msg and msg['imageUrl']:
                return False
            return True
        if exclude_image_bubbles:
            filtered = [msg for msg in self.messages if is_dialogue_message(msg)]
        else:
            filtered = self.messages
        return filtered[-count:]
    '''
    
    def get_recent_messages_with_default_image(self, count: int = 6) -> List[Dict[str, str]]:
        """
        Get the last `count` non-image messages (user/assistant dialogue only), 
        last image bubble, and default image, excluding system prompt.

        Args:
            count (int, optional): number of messages to return. Defaults to 6.

        Returns:
            List[Dict[str, str]]: list of messages
        """
        def is_dialogue_message(msg):
            # Exclude if message has an imageUrl (image bubble)
            if 'imageUrl' in msg and msg['imageUrl']:
                return False
            return True
        messages_reversed = list(reversed(self.messages[1:])) # exclude system prompt
        filtered = []
        text_msg_count = 0
        default_image_message = self.messages[2] # system prompt, opening prompt, default image
        last_image_bubble_msg = None
        for msg in messages_reversed:
            if is_dialogue_message(msg):
                if text_msg_count < count:
                    filtered.append(msg)
                    text_msg_count += 1
            else:
                if last_image_bubble_msg is None:
                    last_image_bubble_msg = msg
        if last_image_bubble_msg is not None and \
            last_image_bubble_msg['imageUrl'] != default_image_message['imageUrl']: # skip if last is the default image
            filtered = [default_image_message] + filtered # will be last, once reversed
        filtered.append(last_image_bubble_msg) # will be first, once reversed
        return list(reversed(filtered))
    
    def get_last_image_url(self, use_default_image: bool = True) -> str:
        """
        Get the last image URL from the messages.
        """
        if use_default_image:
            return self.avatar_image_uri
        found_image_url = False
        for msg in reversed(self.messages):
            if 'imageUrl' in msg and msg['imageUrl']:
                print(f"Found Last image URL: {msg['imageUrl']}")
                return msg['imageUrl']
        if not found_image_url:
            return self.avatar_image_uri

    # =========== Usage Tracking Functions ===========
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

    def remove_message(self, message_id: str) -> bool:
        """
        Remove a message by its ID from both database and local messages list.
        
        Args:
            message_id: The ID of the message to remove
            
        Returns:
            True if message was found and removed from local list, False otherwise
        """
        try:
            logger.info(f"Attempting to remove message with ID: {message_id}")
            
            # Step 1: Try to remove from database (no matter what result)
            db_removed = False
            if self.chat_session_manager and self.user_id and self.avatar_id:
                db_removed = self.chat_session_manager.remove_message_by_id(
                    user_id=self.user_id,
                    avatar_id=self.avatar_id,
                    message_id=message_id
                )
                if db_removed:
                    logger.info(f"Successfully removed message {message_id} from database")
                else:
                    logger.warning(f"Message {message_id} not found in database or failed to remove")
            else:
                logger.warning("Chat session manager not available for database removal")
            
            # Step 2: Remove from local messages list (no matter what database result was)
            local_removed = False
            for i, msg in enumerate(self.messages):
                if msg.get('message_id') == message_id:
                    logger.info(f"Found message in local list at index {i}: {msg}")
                    self.messages.pop(i)
                    local_removed = True
                    break
            
            if not local_removed:
                logger.warning(f"Message {message_id} not found in local messages list")
                return False
            
            logger.info(f"Successfully removed message {message_id} from local list")
            return True
                
        except Exception as e:
            logger.error(f"Error removing message: {e}")
            return False
