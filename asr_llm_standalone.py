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
    ):
        # Initialize Deepinfra client for Whisper ASR
        self.openai_client = OpenAI(
            api_key=os.getenv("DEEPINFRA_API_KEY"),
            base_url="https://api.deepinfra.com/v1/openai",
        )
        if not self.openai_client.api_key:
            raise ValueError("DEEPINFRA_API_KEY environment variable is not set")

        # Store room reference
        self.room = room
        self.text_stream_writer: TextStreamWriter | None = None  # Add type hint for text stream writer
        self._is_shutting_down = False  # Add shutdown flag

        # OpenRouter configuration
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.openrouter_headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        }
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

    async def publish_text_livekit(self, text):
        """
        Publish text to LiveKit room using the streaming format
        """
        if self._is_shutting_down:
            logger.warning("Skipping text publish during shutdown")
            return

        if not self.room:
            logger.error("No LiveKit room available for publishing")
            return

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
                # Regular text chunk
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

    async def send_to_openrouter(self, text):
        """
        1. receives text from ASR or user input.
        2. send text to LLM and get streaming response.
        3. print the response and publish to LiveKit.
        """
        self.user_interrupting_flag = False
        first_llm_token_received = False
        buffer = ""
        current_response = ""

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

        payload = {
            "model": "deepseek/deepseek-chat-v3-0324",
            "messages": self.messages,
            "stream": True,
            "provider": {"sort": "latency"},
        }

        # Print system prompt and messages for debugging
        # print("\n=== System Prompt ===")
        # print(self.system_prompt)
        # print("\n=== Message History ===")
        # for msg in self.messages:
        #     print(f"\nRole: {msg['role']}")
        #     print(f"Content: {msg['content']}")
        # print("\n=== End Message History ===\n")

        # Send stream start at the beginning
        await self.publish_text_livekit("[START]")

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
                        self.timing["llm_first_token_time"] = time.time()
                        if self.timing["whisper_end_time"] != -1:
                            logger.info(
                                f"Time from whisper end to LLM first token: {self.timing['llm_first_token_time'] - self.timing['whisper_end_time']:.2f} seconds"
                            )
                        # Send timing information after first token
                        # await self.publish_text_livekit(f"[speech_end_time]: {self.timing['speech_end_time']}")
                        # await self.publish_text_livekit(f"[llm_first_token_time]: {self.timing['llm_first_token_time']}")

                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            # Process any remaining buffer content
                            remaining_segments = (
                                self.text_chunk_spliter.get_remaining_buffer()
                            )
                            for segment in remaining_segments:
                                print("sending segment in final: ", segment, '\n', flush=True)
                                logger.info(f"sending segment in final: {segment} \n")
                                await self.publish_text_livekit(segment)

                            # Add assistant message to self.messages (single source of truth)
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
                                    model="deepseek/deepseek-chat-v3-0324"
                                )
                            except Exception as e:
                                logger.error(f"Failed to save LLM response to database: {e}")
                            
                            # Send stream end
                            await self.publish_text_livekit("[DONE]")
                            break

                        try:
                            data_obj = json.loads(data)
                            content = data_obj["choices"][0]["delta"].get("content")
                            if content:
                                current_response += content
                                # Process and print each segment immediately
                                segments = self.text_chunk_spliter.process_chunk(
                                    content
                                )
                                for segment in segments:
                                    print("sending segment: ", segment, '\n', flush=True)
                                    logger.info(f"sending segment: {segment} \n")
                                    await self.publish_text_livekit(segment)

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
                        await self.publish_text_livekit("[INTERRUPTED]")
                        break
        if self.user_interrupting_flag:
            logger.warning(f"Skipping history appending due to user interruption")
