from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("character-image-generator")

@mcp.tool()
async def generate_character_image(prompt: str) -> str:
    # """Generate a character image based on the provided prompt.
    
    # This function should be called when:
    # 1. User explicitly requests a photo/picture of the character
    # 2. The conversation scene/location changes (e.g., moving to a park, different setting)
    
    # Args:
    #     prompt: Description of the character and scene to generate (e.g., "character in a park", "character at the beach")
    # """

    # version2 
    """Generate a new character image based on the described scene or location.

    This tool should be called in the following situations:
    1. When the user directly requests a new image or photo of the character.
    2. When the conversation setting or location changes (even implicitly) — for example, if the user says "let's go to the bar" or "we're walking along the beach," generate a new image that reflects the new scene.
    3. When the character's appearance is expected to change due to context (e.g., lighting, outfit, background, activity).

    Always generate a fresh image that reflects the latest described scene, setting, or activity.
    Never generate more than 1 person in the image.

    Args:
        prompt: A description of the character and the current setting or scene (e.g., "the character is sitting in a dimly lit bar", "the character is walking along the beach at sunset").
    """



    # This is a placeholder implementation
    # In a real implementation, this would call an image generation API
    # such as DALL-E, Midjourney, or Stable Diffusion
    print(f"Generating character image with prompt: {prompt}")
    
    # For now, just acknowledge the request
    # return f"Character image generation requested with prompt: '{prompt}'. Image generation would be triggered here."
    return prompt  # we just return prompt, so it's easy to send to image generator. But for llm resp, we will add the above info. 

if __name__ == "__main__":
    # Initialize and run the server
    print("Starting the MCP character image generator server...")
    mcp.run(transport='stdio')
