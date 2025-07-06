from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("character-image-generator")


@mcp.tool()
async def generate_character_image(prompt: str) -> str:
    """
    Generate a new character image when there is a meaningful visual change in the scene, setting, or the character's appearance or action — but only when appropriate.

    This tool should only be called when a new image would *clearly enhance* the user's visual understanding of the current situation.

    **Trigger this function only in the following cases:**

    1. **User explicitly requests a photo or image of the character.**
    - If the character has refused or declined to show an image, do **not** call this tool. Always respect the character's stated consent.

    2. **The scene or location changes in a way that would visually alter the background or setting.**
    - Examples: “Let’s go to the house.” / “We’re walking in the park now.”
    - *Skip generation if the change is minimal or does not noticeably impact the visual scene.*

    3. **The character begins a new action, gesture, or pose that meaningfully alters the visual.**
    - Examples: “She’s eating noodles.” / “He’s dancing under the streetlight.”
    - Do not trigger for small or subtle changes like blinking, turning slightly, or changing tone of voice.

    4. **The character’s appearance is clearly affected by context.**
    - This includes major pose shifts, emotional expressions, outfit changes, lighting conditions, or scene mood shifts.

    **Always:**
    - Generate only if the new image meaningfully improves the visual representation.
    - Never show more than one person in the image.
    - Avoid redundant or overly frequent generations — especially when the scene hasn’t changed.

    Args:
        prompt: A clear, detailed description of the character and the current visual context — including location, action, expression, and relevant atmosphere.
        Example prompts:
            - “The character is sitting at a sunny cafe patio, sipping tea.”
            - “The character is cooking ramen in a small cozy kitchen
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
    mcp.run(transport="stdio")
