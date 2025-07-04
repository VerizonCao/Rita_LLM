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
    # """Generate a new character image based on the described scene or location or character change motion

    # This tool should be called in the following situations:
    # 1. When the user directly requests a new image or photo of the character.
    # 2. When the conversation setting or location changes (even implicitly) — for example, if the user says "let's go to the bar" or "we're walking along the beach,", generate a new image that reflects the new scene.
    # 3. When the character's appearance is expected to change due to context (e.g., lighting, outfit, background, activity).
    # 4. When the character is showing a new thing or it's changing it's pose.

    # Always generate a fresh image that reflects the latest described scene, setting, or activity.
    # Never generate more than 1 person in the image.

    # Args:
    #     prompt: A description of the character and the current setting or scene (e.g., "the character is sitting in a dimly lit bar", "the character is walking along the beach at sunset").
    # """

    # version3
    """Generate a new character image based on changes in scene, setting, or character action.

    This tool should be called whenever a new visual representation of the character is needed due to contextual changes in the conversation.

    Trigger this function in the following situations:
    1. The user explicitly requests a new image or photo of the character.
    2. The scene or location changes — either explicitly or implicitly. For example: 
    - “Let's go to the house.”
    - “We're walking in the park now.”
    In these cases, generate an image that reflects the new setting.
    3. The character begins or changes an activity or gesture. For example:
    - “She's eating something.”
    - “He waves goodbye.”
    4. The character's appearance is expected to change based on context — including pose, background, lighting, outfit, or emotional expression.

    Always:
    - Generate a fresh image that reflects the most recent scene, setting, or activity described.
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
