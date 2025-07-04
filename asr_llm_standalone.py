from _input_utils import TextChunkSpliter
import logging
from pathlib import Path
import sys
import os
from openai import OpenAI
import time
import requests
import json
from system_prompt import LLM_System_Prompt
from livekit import rtc  # Add LiveKit import
from livekit.rtc import TextStreamWriter  # Add TextStreamWriter import
import asyncio
from dotenv import load_dotenv
from typing import Optional
import glob
import random
from image_gen import ImageGenerator, run_flux_model  # Add ImageGenerator import

# MCP Client imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import chat system
from data import ChatSessionManager

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
        llm_data: tuple = None,
        room: rtc.Room = None,  # Add type hint for LiveKit room
        loop: asyncio.AbstractEventLoop = None,  # Add event loop parameter
        mcp_server_path: str = None,  # Add MCP server path parameter
        image_url: str = None,
    ):
        

        # Initialize Deepinfra client for Whisper ASR
        self.openai_client = OpenAI(
            api_key=os.getenv("DEEPINFRA_API_KEY"),
            base_url="https://api.deepinfra.com/v1/openai",
        )
        if not self.openai_client.api_key:
            raise ValueError("DEEPINFRA_API_KEY environment variable is not set")

        # Store room reference and event loop
        self.room = room
        self.loop = loop  # Store the event loop reference
        self.text_stream_writer: TextStreamWriter | None = None  # Add type hint for text stream writer
        self._is_shutting_down = False  # Add shutdown flag

        # MCP Client setup
        self.mcp_server_path = mcp_server_path  # Always initialize, even if None
        if mcp_server_path:
            self.session: Optional[ClientSession] = None
            self.exit_stack = AsyncExitStack()
            self.stdio = None
            self.write = None
            self.available_tools = []
        else:
            self.session = None
            self.exit_stack = None
            self.stdio = None
            self.write = None
            self.available_tools = []

        # Text publishing queue for non-blocking operation
        self.tts_text_queue = asyncio.Queue()
        self.tts_text_publisher_task = None
        # Image swap functionality
        self.image_swap = False  # Track image swap setting
        self.image_url = None  # Track image URL

        # Token usage tracking
        self.current_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0,
            "tts_tokens": 0  # Track tokens for TTS (dialogue only, excluding narrative)
        }

        # OpenRouter configuration (kept for fallback and token tracking)
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.openrouter_headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        }
        self.model = "google/gemini-2.5-flash-preview-05-20"
        
        (
            user_nickname,
            user_bio,
            assistant_nickname,
            assistant_bio,
            assistant_additional_characteristics,
            conversation_context,
            expression_list,
            user_id,
            avatar_id,
        ) = llm_data
        self.user_nickname = user_nickname
        self.user_bio = user_bio
        self.assistant_nickname = assistant_nickname
        self.assistant_bio = assistant_bio
        self.assistant_additional_characteristics = assistant_additional_characteristics
        self.conversation_context = conversation_context
        self.expression_list = expression_list
        self.user_id = user_id
        self.avatar_id = avatar_id

        # Initialize simplified chat session manager
        self.chat_session_manager = ChatSessionManager()

        self.expression_list = expression_list
        system_prompt_obj: LLM_System_Prompt = LLM_System_Prompt(
            assistant_name=assistant_nickname,
            assistant_bio=assistant_bio,
            assistant_additional_characteristics=assistant_additional_characteristics,
            user_name=user_nickname,
            conversation_context=conversation_context,
        )
        self.system_prompt = system_prompt_obj.get_system_prompt()

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
            if self.mcp_server_path:
                # Also initialize MCP connection since we have a hardcoded path
                asyncio.run_coroutine_threadsafe(self.initialize_mcp_connection(), self.loop)
        else:
            logger.warning("No event loop provided, TTS text publisher not started")
            logger.warning("No event loop provided, MCP connection not initialized")

        # Image sending functionality, remove me when we are actually have images from genai. 
        self.test_images = self._load_test_images()
        self.current_image_index = 0
        
        # Initialize image generator
        try:
            if self.mcp_server_path:
                self.image_generator = ImageGenerator()
                logger.info("Image generator initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize image generator: {e}")
            self.image_generator = None

    async def connect_to_mcp_server(self):
        """Connect to the MCP server"""
        if not self.mcp_server_path:
            logger.warning("No MCP server path provided, skipping MCP connection")
            return False
            
        try:
            is_python = self.mcp_server_path.endswith('.py')
            is_js = self.mcp_server_path.endswith('.js')
            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")

            command = "python" if is_python else "node"
            server_params = StdioServerParameters(
                command=command,
                args=[self.mcp_server_path],
                env=None
            )

            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

            await self.session.initialize()

            # List available tools and cache them
            response = await self.session.list_tools()
            tools = response.tools
            self.available_tools = [{
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in tools]
            logger.info(f"Connected to MCP server with tools: {[tool.name for tool in tools]}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return False

    def _convert_mcp_tools_to_openai_format(self):
        """Convert MCP tools to OpenAI/OpenRouter format"""
        openai_tools = []
        for tool in self.available_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            }
            openai_tools.append(openai_tool)
        return openai_tools

    async def initialize_mcp_connection(self):
        """Initialize MCP connection if server path is provided"""
        if self.mcp_server_path:
            success = await self.connect_to_mcp_server()
            if success:
                logger.info("MCP connection established successfully")
            else:
                logger.warning("Failed to establish MCP connection, will use fallback mode")
        else:
            logger.info("No MCP server path provided, using fallback mode")

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

    def _load_test_images(self) -> list:
        """Load all test images from the test folder"""
        try:
            test_folder = Path(__file__).parent / "test"
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
                opening_prompt = self._get_opening_prompt()
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
                            assistant_name=self.assistant_nickname or "Assistant",
                            model="opening_prompt"  # Special model name to indicate this is an opening
                        )
                        logger.info(f"Added opening prompt as first assistant message")
                    except Exception as e:
                        logger.error(f"Failed to save opening prompt to database: {e}")
            
        except Exception as e:
            logger.error(f"Failed to load conversation history: {e}")

    def _get_opening_prompt(self) -> Optional[str]:
        """Get opening_prompt from avatars table for this avatar"""
        try:
            # Use the database manager from chat session manager
            if not self.chat_session_manager.db_manager.ensure_connection():
                logger.error("Failed to establish database connection for opening prompt")
                return None
            
            query = "SELECT opening_prompt FROM avatars WHERE avatar_id = %s"
            result = self.chat_session_manager.db_manager.execute_query(query, (self.avatar_id,))
            
            if result and len(result) > 0:
                opening_prompt = result[0].get('opening_prompt')
                if opening_prompt and opening_prompt.strip():
                    logger.info(f"Retrieved opening prompt for avatar {self.avatar_id}")
                    return opening_prompt.strip()
            
            logger.info(f"No opening prompt found for avatar {self.avatar_id}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get opening prompt: {e}")
            return None

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

            # Cleanup MCP client resources
            if self.exit_stack:
                try:
                    await self.exit_stack.aclose()
                except Exception as e:
                    logger.warning(f"Error closing MCP client: {e}")
                finally:
                    self.session = None
                    self.stdio = None
                    self.write = None

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



    async def send_specific_image_to_livekit(self, image_path: str):
        """
        Send a specific image to the LiveKit data channel with topic 'image_file'
        
        Args:
            image_path: Path to the image file to send
        """
        if self._is_shutting_down:
            logger.warning("Skipping image send during shutdown")
            return

        if not self.room:
            logger.error("No LiveKit room available for image sending")
            return

        try:
            image_name = Path(image_path).name
            logger.info(f"Sending image: {image_name}")
            
            # Send the image file using LiveKit's send_file method
            info = await self.room.local_participant.send_file(
                file_path=image_path,
                topic="image_file",
            )
            
            logger.info(f"Successfully sent image '{image_name}' with stream ID: {info.stream_id}")
            
        except Exception as e:
            logger.error(f"Error sending image to LiveKit: {e}")

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
        2. send text to LLM and get streaming response with MCP tool support, if MCP enabled. 
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
                user_name=self.user_nickname or "User"
            )
        except Exception as e:
            logger.error(f"Failed to save user message to database: {e}")

        # Check if MCP server is available
        if self.session and self.available_tools:
            # Use MCP client with tool support
            print("Using MCP tools")
            await self._process_with_mcp_tools(text)
        else:
            # Fallback to direct OpenRouter call without tools
            print("Using direct OpenRouter call without tools")
            await self._process_with_direct_openrouter(text)

    async def _process_with_mcp_tools(self, text):
        """Process query using MCP tools with streaming support"""
        try:
            # Convert MCP tools to OpenAI format
            openai_tools = self._convert_mcp_tools_to_openai_format()

            # Prepare messages with system prompt included
            messages = [{"role": "system", "content": self.system_prompt}] + self.messages[1:]  # Skip system prompt from self.messages

            # Initial OpenRouter API call with tools
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "max_tokens": 1000,
                "tools": openai_tools,
                "provider": {"sort": "latency"},
                "usage": {"include": True}
            }

            # Process streaming response with tool handling
            await self._handle_streaming_response_common(payload, text, with_tools=True)

        except Exception as e:
            logger.error(f"Error in MCP tool processing: {e}")
            # Fallback to direct processing
            await self._process_with_direct_openrouter(text)

    async def _process_with_direct_openrouter(self, text):
        """Process query using direct OpenRouter call (fallback)"""
        try:
            payload = {
                "model": self.model,
                "messages": self.messages,
                "stream": True,
                "provider": {"sort": "latency"},
                "usage": {"include": True}
            }

            # Process streaming response without tools
            await self._handle_streaming_response_common(payload, text, with_tools=False)

        except Exception as e:
            logger.error(f"Error in direct OpenRouter processing: {e}")

    async def _handle_streaming_response_common(self, payload, text, with_tools=False):
        """Handle streaming response, optionally with tool calls"""
        first_llm_token_received = False
        current_response = ""
        is_on_dialogue = False
        is_tts_started = False
        past_quote_count = 0
        track_char_index = 0
        consecutive_narrative_chars = 0 
        consecutive_dialogue_chars = 0

        with requests.post(
            self.openrouter_url,
            headers=self.openrouter_headers,
            json=payload,
            stream=True,
        ) as r:
            r.raise_for_status()
            r.encoding = "utf-8"

            for chunk in r.iter_lines(chunk_size=1024, decode_unicode=True):
                if chunk:
                    line = chunk.strip()
                    if not first_llm_token_received:
                        first_llm_token_received = True
                        await self.publish_frontend_stream_livekit("START", "")
                        self.timing["llm_first_token_time"] = time.time()
                        if self.timing["whisper_end_time"] != -1:
                            logger.info(
                                f"Time from whisper end to LLM first token: {self.timing['llm_first_token_time'] - self.timing['whisper_end_time']:.2f} seconds"
                            )

                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            await self._finalize_response(current_response, text)
                            break

                        try:
                            data_obj = json.loads(data)

                            # Token usage
                            if "usage" in data_obj:
                                usage = data_obj["usage"]
                                self.current_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                                self.current_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                                self.current_usage["total_tokens"] += usage.get("total_tokens", 0)
                                self.current_usage["cost"] += usage.get("cost", 0)

                                logger.info(f"Token usage - Prompt: {self.current_usage['prompt_tokens']}, "
                                            f"Completion: {self.current_usage['completion_tokens']}, "
                                            f"Total: {self.current_usage['total_tokens']}, "
                                            f"Cost: {self.current_usage['cost']}")

                            # Process choices
                            if "choices" in data_obj and data_obj["choices"]:
                                delta = data_obj["choices"][0].get("delta", {})

                                if with_tools and "tool_calls" in delta:
                                    await self._handle_tool_calls(delta["tool_calls"])
                                    continue

                                content = delta.get("content")
                                if content:
                                    content = self.replace_special_quotes_to_straight_quotes(content)
                                    current_response += content
                                    await self.publish_frontend_stream_livekit("CONTENT", content)
                                    (is_on_dialogue, is_tts_started, past_quote_count, track_char_index, 
                                     consecutive_narrative_chars, consecutive_dialogue_chars) = await self._process_content_for_tts(
                                        content, is_on_dialogue, is_tts_started, 
                                        past_quote_count, track_char_index, 
                                        consecutive_narrative_chars, consecutive_dialogue_chars)

                        except json.JSONDecodeError:
                            logger.warning(f"Could not decode JSON data: {data}")
                        except Exception as e:
                            logger.error(f"Error processing stream data chunk: {e}")
                            break

                    if self.user_interrupting_flag:
                        await self._handle_interruption()
                        break

    async def _handle_tool_calls(self, tool_calls):
        """Handle tool calls from the LLM"""
        try:
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])

                # Execute tool call via MCP
                result = await self.session.call_tool(tool_name, tool_args)
                logger.info(f"Tool {tool_name} executed with result: {result}")

                # Extract text content from MCP response
                if hasattr(result, 'content') and result.content:
                    if isinstance(result.content, list) and len(result.content) > 0:
                        content_text = ""
                        for content_item in result.content:
                            if hasattr(content_item, 'text'):
                                content_text += content_item.text
                            elif isinstance(content_item, str):
                                content_text += content_item
                        tool_result_content = content_text
                    elif hasattr(result.content, 'text'):
                        tool_result_content = result.content.text
                    else:
                        tool_result_content = str(result.content)
                elif isinstance(result, str):
                    tool_result_content = result
                else:
                    tool_result_content = "Tool executed successfully"

                # append some explanation to the tool result
                image_gen_prompt = None
                if tool_name == "generate_character_image":
                    image_gen_prompt = tool_result_content
                    tool_result_content = f"Image generation requested with prompt: '{tool_result_content}'. Image generation would be triggered here."
                
                # Add tool result to conversation history
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result_content
                })

                if image_gen_prompt:
                    if self.image_url:
                        print("sending image gen request")
                        image_gen_prompt += ", [important]: never show more than 1 person in the image, and keep the character's face consistent with the original, matching facial features, and proportions as closely as possible"
                        input_params = {
                            "prompt": image_gen_prompt,
                            "input_image": self.image_url,
                            "aspect_ratio": "match_input_image",
                            "output_format": "jpg",
                            "go_fast": True
                        }
                        image_path = "image_gen/" + time.strftime("%Y%m%d%H%M%S") + ".jpg"
                        # Create test directory if it doesn't exist
                        os.makedirs("image_gen", exist_ok=True)
                        # Use the existing image generator instance
                        try:
                            run_flux_model(input_params, image_path, self.image_generator)
                            # after it's done, let's swap the image.
                            await self.send_specific_image_to_livekit(image_path)
                        except Exception as e:
                            logger.error(f"Image generation failed: {e}")
                            print(f"Image generation failed: {e}")
                    else:
                        print("no image url provided, skipping image gen")
                
                    

        except Exception as e:
            logger.error(f"Error handling tool calls: {e}")

    async def _process_content_for_tts(self, content, is_on_dialogue, is_tts_started, 
                                     past_quote_count, track_char_index, 
                                     consecutive_narrative_chars, consecutive_dialogue_chars):
        """Process content for TTS with dialogue/narrative detection"""
        # Track quotation marks for dialogue/narrative detection
        for char in content:
            if char == '"':
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
                segments = self.text_chunk_spliter.process_chunk(char)
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
        
        # Return updated state variables
        return is_on_dialogue, is_tts_started, past_quote_count, track_char_index, consecutive_narrative_chars, consecutive_dialogue_chars

    async def _finalize_response(self, current_response, text):
        """Finalize the response processing"""
        # Process any remaining tts buffer content
        remaining_segments = (
            self.text_chunk_spliter.get_remaining_buffer()
        )
        for segment in remaining_segments:
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
        
        # Save LLM response to database
        try:
            self.chat_session_manager.write_assistant_message(
                user_id=self.user_id,
                avatar_id=self.avatar_id,
                content=current_response,
                assistant_name=self.assistant_nickname or "Assistant",
                model=self.model
            )
        except Exception as e:
            logger.error(f"Failed to save LLM response to database: {e}")
        
        # Send DONE marker to frontend immediately via LiveKit
        await self.publish_frontend_stream_livekit("DONE", "")
        
        # Send an image after the response is complete
        # if self.image_swap:
            # await self.send_image_to_livekit()

    async def _handle_interruption(self):
        """Handle user interruption"""
        logger.warning("Stopping LLM stream due to user interruption.")
        self.user_interrupting_flag = False
        print("\n[INTERRUPTED]")
        await self.publish_tts_text("[INTERRUPTED]")
        
        # Send INTERRUPTED marker to frontend immediately via LiveKit
        await self.publish_frontend_stream_livekit("INTERRUPTED", "")
        
        # Send an image even when interrupted
        if self.image_swap:
            await self.send_image_to_livekit()

        if self.user_interrupting_flag:
            logger.warning(f"Skipping history appending due to user interruption")
        
        return self.current_usage["total_tokens"]  # Return the total tokens for backward compatibility
