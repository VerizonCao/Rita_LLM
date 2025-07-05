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
    - "Let's go to the house."
    - "We're walking in the park now."
    In these cases, generate an image that reflects the new setting, unless the change is extremely minor or does not affect the visual scene.
    3. The character begins or changes an activity or gesture that visually affects the scene. For example:
    - "She's eating something."
    - "He waves goodbye."
    Skip generation if the change is very subtle and doesn't significantly alter the visual.
    4. The character's appearance is expected to change based on context — including pose, background, lighting, outfit, or emotional expression.

    PROMPT ENGINEERING GUIDELINES:
    - Be specific with colors, styles, and descriptions: Use precise color names (e.g., "deep burgundy sweater" not just "red")
    - Start simple and iterate on successful edits: Begin with core elements, then add details
    - Preserve intentionally by stating what to keep unchanged: "Keep the character's blue eyes and blonde hair, but change the outfit to..."
    - Use quotation marks for exact text replacements when needed
    - Control composition by specifying camera angles and framing: "medium shot from waist up", "close-up portrait", "wide angle showing full scene"
    - Choose verbs carefully: "sitting" vs "perched", "walking" vs "strolling" give different visual results
    - Include lighting and mood: "warm afternoon sunlight", "soft indoor lighting", "dramatic shadows"
    - Specify art style when relevant: "photorealistic", "anime style", "watercolor painting"
    - Mention background details: "cozy cafe with exposed brick walls", "sunset beach with palm trees"

    Always:
    - Generate a fresh image that reflects the current setting or activity, when it visually impacts the scene.
    - Ensure the image contains **only one** character — never include more than one person.
    - Include specific visual details that will help the AI generate a more accurate and appealing image.

    Args:
        prompt: A detailed description of the character and the current setting, action, or scene 
                with specific visual details. Examples:
                - "A young woman with long auburn hair wearing a cream cable-knit sweater, sitting at a rustic wooden table in a cozy cafe with exposed brick walls, warm afternoon sunlight streaming through large windows, photorealistic style, medium shot"
                - "A man with short dark hair in a navy blue polo shirt, walking along a sandy beach at sunset with palm trees silhouetted against orange sky, casual pose, wide angle shot showing the full coastal scene"
                - "A character with bright green eyes and curly brown hair, eating ramen at a traditional Japanese restaurant with bamboo walls and paper lanterns, soft ambient lighting, close-up shot focusing on the character's expression"
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
