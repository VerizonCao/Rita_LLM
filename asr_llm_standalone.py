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

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


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
    ):
        # Initialize Deepinfra client for Whisper ASR
        self.openai_client = OpenAI(
            api_key=os.getenv("DEEPINFRA_API_KEY"),
            base_url="https://api.deepinfra.com/v1/openai",
        )
        if not self.openai_client.api_key:
            raise ValueError("DEEPINFRA_API_KEY environment variable is not set")

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
        ) = llm_data
        self.user_nickname = user_nickname
        self.user_bio = user_bio
        self.assistant_nickname = assistant_nickname
        self.assistant_bio = assistant_bio
        self.assistant_additional_characteristics = assistant_additional_characteristics
        self.conversation_context = conversation_context
        self.expression_list = expression_list

        self.expression_list = expression_list
        system_prompt_obj: LLM_System_Prompt = LLM_System_Prompt(
            assistant_name=assistant_nickname,
            assistant_bio=assistant_bio,
            assistant_additional_characteristics=assistant_additional_characteristics,
            user_name=user_nickname,
            conversation_context=conversation_context,
        )
        self.system_prompt = system_prompt_obj.get_system_prompt()

        self.messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        # Initialize text chunk splitter
        self.text_chunk_spliter = TextChunkSpliter()

        # Timing information
        self.timing = {
            "speech_end_time": -1,
            "whisper_end_time": -1,
            "llm_first_token_time": -1,
        }
        self.user_interrupting_flag = False

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

    def send_to_openrouter(self, text):
        """
        1. receives text from ASR or user input.
        2. send text to LLM and get streaming response.
        3. print the response and publish to WebRTC.
        """
        self.user_interrupting_flag = False
        first_llm_token_received = False
        buffer = ""
        current_response = ""

        self.messages.append({"role": "user", "content": text})
        payload = {
            "model": "google/gemini-2.0-flash-001",
            "messages": self.messages,
            "stream": True,
            "provider": {"sort": "latency"},
        }
        with requests.post(
            self.openrouter_url,
            headers=self.openrouter_headers,
            json=payload,
            stream=True,
        ) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=512, decode_unicode=True):
                if not first_llm_token_received:
                    first_llm_token_received = True
                    self.timing["llm_first_token_time"] = time.time()
                    if self.timing["whisper_end_time"] != -1:
                        logger.info(
                            f"Time from whisper end to LLM first token: {self.timing['llm_first_token_time'] - self.timing['whisper_end_time']:.2f} seconds"
                        )

                buffer += chunk
                while True:
                    try:
                        line_end = buffer.find("\n")
                        if line_end == -1:
                            break

                        line = buffer[:line_end].strip()
                        buffer = buffer[line_end + 1 :]

                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                # Process any remaining buffer content
                                remaining_segments = (
                                    self.text_chunk_spliter.get_remaining_buffer()
                                )
                                for segment in remaining_segments:
                                    print(segment, end="", flush=True)
                                    self.publish_text_webrtc(segment)

                                self.messages.append(
                                    {"role": "assistant", "content": current_response}
                                )
                                logger.info(f"Current response: {current_response}")
                                print()  # New line after response
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
                                        print(segment, end="", flush=True)
                                        self.publish_text_webrtc(segment)

                            except json.JSONDecodeError:
                                logger.warning(f"Could not decode JSON data: {data}")
                                pass  # Ignore malformed JSON lines
                            except Exception as e:
                                logger.error(f"Error processing stream data chunk: {e}")
                                break  # Safer to break the inner loop on unexpected errors
                    except Exception as e:
                        logger.error(f"Error processing received buffer line: {e}")
                        break  # Break inner loop on buffer processing error

                    if self.user_interrupting_flag:
                        logger.warning("Stopping LLM stream due to user interruption.")
                        self.user_interrupting_flag = False
                        print("\n[INTERRUPTED]")
                        self.publish_text_webrtc("[INTERRUPTED]")
                        break
        if self.user_interrupting_flag:
            logger.warning(f"Skipping history appending due to user interruption")

    def publish_text_webrtc(self, text):
        # publish to webrtc
        pass
