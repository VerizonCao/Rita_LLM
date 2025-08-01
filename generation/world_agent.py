# mcp agent for chat 

# World Agent for handling image generation and scene management
import logging
import asyncio
import os
import time
import json
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
# MCP Client imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

# Image generation imports
from generation.image_gen import ImageGenerator, run_flux_model

# S3 client import
from data import s3_manager, ChatSessionManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class WorldAgent:
    """
    World Agent that handles image generation and scene management.
    Receives the last 6 messages from character chat to understand context
    and decides when to generate images based on scene changes.
    """
    
    def __init__(
        self,
        character_system_prompt: str,
        mcp_server_path: str = "generation/chat_server.py",
        room=None,  # LiveKit room for sending images
        loop: asyncio.AbstractEventLoop = None,
        user_id: str = None,  # User ID for database operations
        avatar_id: str = None,  # Avatar ID for database operations
        assistant_name: str = "Assistant",  # Assistant name for database operations
    ):
        self.character_system_prompt = character_system_prompt
        self.mcp_server_path = mcp_server_path
        self.room = room
        self.loop = loop
        self._is_shutting_down = False  # Add shutdown flag
        
        # Database integration for saving image URLs
        self.user_id = user_id
        self.avatar_id = avatar_id
        self.assistant_name = assistant_name
        
        # Character appearance tracking
        self.current_character_appearance = None  # Stores the last image generation prompt
        
        # MCP Client setup
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.stdio = None
        self.write = None
        self.available_tools = []
        
        # Image generator
        self.image_generator = None
        try:
            self.image_generator = ImageGenerator()
            logger.info("Image generator initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize image generator: {e}")
        
        # World agent system prompt
        self.world_system_prompt = self._create_world_system_prompt()
        
        # OpenRouter configuration for world agent
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.openrouter_headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        }
        
        # Initialize MCP connection if event loop is available
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.initialize_mcp_connection(), self.loop)
        else:
            logger.warning("No event loop provided, MCP connection not initialized")

    def _create_world_system_prompt(self) -> str:
        """Create the world agent system prompt that includes character context"""
        world_prompt = """You are a World Agent that analyzes conversation context and decides when to generate character images.

CHARACTER CONTEXT:
"""
        world_prompt += self.character_system_prompt
        
        # Add current character appearance if available
        if self.current_character_appearance:
            world_prompt += f"""

CURRENT CHARACTER APPEARANCE:
{self.current_character_appearance}

Note: The character's current appearance is based on the last image generation. Use this information to maintain consistency when deciding if a new image should be generated."""
        
        world_prompt += """

TASK:
Analyze the recent conversation and decide if the tools should be used.

If no tool is needed, respond normally without using any tools.

Keep your analysis brief and focused."""
        return world_prompt

    def _update_character_appearance(self, prompt: str):
        """
        Update the current character appearance with the new prompt and refresh the system prompt.
        
        Args:
            prompt: The image generation prompt that describes the character's appearance
        """
        self.current_character_appearance = prompt
        # Refresh the world system prompt to include the new appearance
        self.world_system_prompt = self._create_world_system_prompt()
        logger.info(f"Image editing prompt: {prompt}")

    def get_current_character_appearance(self) -> Optional[str]:
        """
        Get the current character appearance description.
        
        Returns:
            The current character appearance prompt, or None if not set
        """
        return self.current_character_appearance

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

    async def analyze_conversation_context(self, recent_messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Analyze the recent conversation context to decide if an image should be generated.
        
        Args:
            recent_messages: List of the last 6 messages (user and assistant)
            
        Returns:
            Image generation prompt if needed, None otherwise
        """
        try:
            # Prepare messages for the world agent
            messages = [
                {"role": "system", "content": self.world_system_prompt},
                {"role": "user", "content": f"Analyze these recent conversation messages and decide if an image should be generated:\n\n{json.dumps(recent_messages, indent=2)}"}
            ]

            # Check if MCP tools are available
            if self.session and self.available_tools:
                # Use MCP client with tool support
                logger.info("Using MCP tools for world agent analysis")
                return await self._analyze_with_mcp_tools(messages)
            else:
                # Fallback to direct OpenRouter call
                logger.info("Using direct OpenRouter call for world agent analysis")
                return await self._analyze_with_direct_openrouter(messages)

        except Exception as e:
            logger.error(f"Error analyzing conversation context: {e}")
            return None

    async def _analyze_with_mcp_tools(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Analyze context using MCP tools"""
        try:
            # Convert MCP tools to OpenAI format
            openai_tools = self._convert_mcp_tools_to_openai_format()

            payload = {
                "model": "google/gemini-2.5-flash",
                "messages": messages,
                "stream": False,
                "max_tokens": 500,
                "tools": openai_tools,
                "provider": {
                    'require_parameters': True,
                }
            }

            response = requests.post(
                self.openrouter_url,
                headers=self.openrouter_headers,
                json=payload
            )
            response.raise_for_status()
            response_data = response.json()

            # Process response
            if "choices" in response_data and response_data["choices"]:
                choice = response_data["choices"][0]
                
                # Check for tool calls
                if "message" in choice and "tool_calls" in choice["message"]:
                    tool_calls = choice["message"]["tool_calls"]
                    for tool_call in tool_calls:
                        if tool_call["function"]["name"] == "generate_character_image":
                            # Extract the prompt from the tool call
                            tool_args = json.loads(tool_call["function"]["arguments"])
                            return tool_args.get("prompt")
                
                # If no tool call, check the content
                content = choice["message"].get("content", "")
                if content.startswith("GENERATE_IMAGE:"):
                    return content.replace("GENERATE_IMAGE:", "").strip()
                elif content.startswith("NO_IMAGE:"):
                    return None

            return None

        except Exception as e:
            logger.error(f"Error in MCP tool analysis: {e}")
            return None

    async def _analyze_with_direct_openrouter(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Analyze context using direct OpenRouter call"""
        try:
            payload = {
                "model": "deepseek/deepseek-chat-v3-0324",
                "messages": messages,
                "stream": False,
                "max_tokens": 500,
                "provider": {
                    'order': ['deepinfra/fp4', 'lambda/fp8', 'baseten/fp8']
                }
            }

            response = requests.post(
                self.openrouter_url,
                headers=self.openrouter_headers,
                json=payload
            )
            response.raise_for_status()
            response_data = response.json()

            if "choices" in response_data and response_data["choices"]:
                content = response_data["choices"][0]["message"].get("content", "")
                if content.startswith("GENERATE_IMAGE:"):
                    return content.replace("GENERATE_IMAGE:", "").strip()
                elif content.startswith("NO_IMAGE:"):
                    return None

            return None

        except Exception as e:
            logger.error(f"Error in direct OpenRouter analysis: {e}")
            return None

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

    async def publish_image_url_livekit(self, image_url: str, content: str = None, message_id=''):
        """
        Publish image URL and content via frontend_stream data channel
        
        Args:
            image_url: The URL of the generated image
            content: The text content to display with the image
        """
        if self._is_shutting_down:
            logger.warning("Skipping image URL publish during shutdown")
            return

        if not self.room:
            logger.error("No LiveKit room available for image URL publishing")
            return

        try:
            data = {
                "topic": "frontend_stream",
                "type": "IMAGE_URL",
                "imageUrl": image_url,
                "message_id": message_id
            }
            
            # Add content if provided
            if content:
                data["text"] = content
                
            await self.room.local_participant.publish_data(json.dumps(data))
            logger.info(f"Published image URL to frontend: {image_url}")
        except Exception as e:
            logger.error(f"Error publishing image URL to LiveKit: {e}")

    async def generate_and_send_image(self, prompt: str, image_url: str = None) -> bool:
        """
        Generate an image using the provided prompt and send it via LiveKit.
        
        Args:
            prompt: The image generation prompt
            
        Returns:
            True if successful, False otherwise
            Image edit prompt, str
            Image url, str ( s3 key )
        """
        if not self.image_generator:
            logger.error("Image generator not available")
            return False, "", ""

        if not image_url:
            logger.warning("No base image URL provided, skipping image generation")
            return False, "", ""

        image_gen_message_id = str(int(time.time() * 1000))
        try:
            logger.info(f"Generating image with prompt: {prompt}")
            
            # Send IMAGE_START signal to frontend
            await self.publish_frontend_stream_livekit("IMAGE_START", content='', message_id=image_gen_message_id)
            logger.info("Sent IMAGE_START signal to frontend")
            
            input_params = {
                "prompt": prompt,
                "input_image": image_url,
                "aspect_ratio": "match_input_image",
                "output_format": "jpg",
                "go_fast": True
            }
            
            logger.info(f"Image generation input params: {input_params}")
            
            # Generate image
            output_url = run_flux_model(
                input_params = input_params, 
                download_to=None, 
                ImageGenerator_PassIn=self.image_generator
            )
            
            if output_url:
                logger.info(f"Image generated successfully: {output_url}")
                
                # 1. Immediately upload to S3 to get s3_key
                s3_key = self.upload_image_to_s3(output_url, prompt)
                if not s3_key:
                    logger.error("Failed to upload image to S3, aborting")
                    # Send IMAGE_END signal even if S3 upload failed
                    await self.publish_frontend_stream_livekit("IMAGE_END", content='', message_id=image_gen_message_id)
                    return False, "", ""
                
                logger.info(f"Successfully uploaded image to S3: {s3_key}")
                
                # # 2. Get public URL from S3 manager
                # s3_public_url = s3_manager.get_public_url_with_cache_check(s3_key, expires_in=3600)
                # if not s3_public_url:
                #     logger.error("Failed to generate public URL for S3 key")
                #     # Send IMAGE_END signal even if public URL generation failed
                #     await self.publish_frontend_stream_livekit("IMAGE_END", "")
                #     return False, "", ""
                
                # logger.info(f"Generated public URL for S3 key: {s3_public_url}")
                
                # 3. Send the public URL via LiveKit
                await self.publish_image_url_livekit(s3_key, f"Generated image: {prompt}", message_id=image_gen_message_id)
                logger.info("Sent public image URL to frontend via frontend_stream")   

                send_end_success = False
                # Send IMAGE_END signal to frontend after successful image generation and sending
                if self.room:
                    await self.publish_frontend_stream_livekit("IMAGE_END", content='', message_id=image_gen_message_id)
                    logger.info("Sent IMAGE_END signal to frontend")
                    send_end_success = True
                else:
                    logger.warning("No LiveKit room available for sending image file")
                    # Send IMAGE_END signal even if room is not available
                    await self.publish_frontend_stream_livekit("IMAGE_END", content='', message_id=image_gen_message_id)
                    logger.info("Sent IMAGE_END signal to frontend (no room available)")
                    send_end_success = False

                # 4. Update character appearance with the successful prompt
                self._update_character_appearance(prompt)
                return send_end_success, prompt, s3_key, image_gen_message_id

                
            else:
                logger.error("Image generation failed - no output URL received")
                # Send IMAGE_END signal even if generation failed
                await self.publish_frontend_stream_livekit("IMAGE_END", content='', message_id=image_gen_message_id)
                logger.info("Sent IMAGE_END signal to frontend (generation failed)")
                return False, "", ""
                
        except Exception as e:
            logger.error(f"Error generating and sending image: {e}")
            # Send IMAGE_END signal even if there was an exception
            try:
                await self.publish_frontend_stream_livekit("IMAGE_END", content='', message_id=image_gen_message_id)
                logger.info("Sent IMAGE_END signal to frontend (exception occurred)")
            except Exception as signal_error:
                logger.error(f"Error sending IMAGE_END signal: {signal_error}")
            return False, "", ""

    async def _send_image_to_livekit(self, image_path: str):
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
            print(f"Sending image: {image_name}")

            # Send the image file using LiveKit's send_file method
            info = await self.room.local_participant.send_file(
                file_path=image_path,
                topic="image_file",
            )

            logger.info(f"Successfully sent image '{image_name}' with stream ID: {info.stream_id}")
            print(f"Successfully sent image '{image_name}' with stream ID: {info.stream_id}")

        except Exception as e:
            logger.error(f"Error sending image to LiveKit: {e}")

    def upload_image_to_s3(self, image_url: str, prompt: str) -> Optional[str]:
        """
        Download image from replicate URL and upload to S3.
        
        Args:
            image_url: The replicate URL of the generated image
            prompt: The image generation prompt for logging
            
        Returns:
            S3 object key if successful, None otherwise
        """
        if not s3_manager.is_available():
            logger.warning("S3 client not available, skipping upload")
            return None
            
        try:
            logger.info(f"Starting S3 upload for image: {prompt[:100]}...")
            
            # Download image from replicate URL
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image_content = response.content
            
            # Generate S3 object key
            timestamp = int(time.time())
            file_extension = '.jpg'  # Default extension
            if '.' in image_url:
                file_extension = '.' + image_url.split('.')[-1]
            elif 'content-type' in response.headers:
                content_type = response.headers['content-type']
                if 'png' in content_type:
                    file_extension = '.png'
                elif 'gif' in content_type:
                    file_extension = '.gif'
                elif 'webp' in content_type:
                    file_extension = '.webp'
            
            # Create S3 object key with user and avatar context
            # Format: rita-swap-images/{user_id}/a-{avatar_id}/{timestamp}.{extension}
            s3_key = f"rita-swap-images/{self.user_id}/{self.avatar_id}/{timestamp}{file_extension}"
            
            # Upload to S3 using the centralized S3 manager
            success = s3_manager.upload_file(
                file_content=image_content,
                s3_key=s3_key,
                content_type=response.headers.get('content-type', 'image/jpeg'),
                metadata={
                    'user_id': self.user_id,
                    'avatar_id': self.avatar_id,
                    'generated_at': str(timestamp)
                }
            )
            
            if success:
                logger.info(f"Successfully uploaded image to S3: {s3_key}")
                return s3_key
            else:
                logger.error("Failed to upload image to S3")
                return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download image from {image_url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during S3 upload: {e}")
            return None

    async def process_conversation_update(self, recent_messages: List[Dict[str, str]], image_url: str = None) -> bool:
        """
        Process a conversation update and decide if an image should be generated.
        
        Args:
            recent_messages: List of the last 6 messages from character chat
            
        Returns:
            True if image was generated, False otherwise.
            Image edit prompt, str
            Image url, str ( s3 key )
        """
        try:
            # Analyze the conversation context
            image_prompt = await self.analyze_conversation_context(recent_messages)
            
            if image_prompt:
                logger.info(f"World agent decided to generate image: {image_prompt}")
                # Generate and send the image
                input_s3_public_url = s3_manager.get_public_url_with_cache_check(image_url, expires_in=3600)
                success, image_prompt, s3_key, image_gen_message_id = await self.generate_and_send_image(image_prompt, input_s3_public_url)
                return success, image_prompt, s3_key, image_gen_message_id
            else:
                logger.debug("World agent decided no image generation needed")
                return False, "", ""
                
        except Exception as e:
            logger.error(f"Error processing conversation update: {e}")
            return False, "", ""

    async def cleanup(self):
        """Cleanup resources"""
        try:
            self._is_shutting_down = True
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