from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("character-image-generator")

@mcp.tool()
async def generate_character_image(prompt: str) -> str:
 
    """
    Generate a new character image based on noticeable changes in scene, setting, or character action.

    This tool should be called when a new visual representation of the character is helpful due to a change in the context of the conversation.

    Trigger this function in the following situations:
    1. The user explicitly requests a new image or photo of the character.
    2. The scene or location changes — either explicitly or implicitly. For example: 
    - “Let's go to the house.”
    - “We're walking in the park now.”
    In these cases, generate an image that reflects the new setting, unless the change is extremely minor or does not affect the visual scene.
    3. The character begins or changes an activity or gesture that visually affects the scene. For example:
    - “She's eating something.”
    - “He waves goodbye.”
    Skip generation if the change is very subtle and doesn't significantly alter the visual.
    4. The character's appearance is expected to change based on context — including pose, background, lighting, outfit, or emotional expression.

    Always:
    - Generate a fresh image that reflects the current setting or activity, when it visually impacts the scene.
    - Ensure the image contains **only one** character — never include more than one person.

    Args:
        prompt: A detailed description of the character and the current setting, action, or scene 
                (e.g., “the character is sitting at a cafe table in the park,” 
                “the character is eating ramen in a cozy indoor kitchen,” 
                “the character is walking along the beach at sunset”).
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
