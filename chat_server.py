from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("character-image-generator")


@mcp.tool()
async def generate_character_image(prompt: str) -> str:
    """
    Generate a new image of the character with carefully constrained visual logic.

    This tool should only be called when there is a meaningful change in the visual scene,
    setting, outfit, pose, or background context.

    **Trigger this function in these cases:**

    1. The user explicitly asks for a new image or picture of the character.
    2. The scene or location clearly changes (e.g., "They arrive at the beach", "We’re in the garden now").
    3. The character’s outfit or hair has changed.
    4. The character begins an action or gesture that visually alters their pose.
    5. The emotional tone or lighting of the environment significantly shifts.

    **Never call this if:**
    - The change is only verbal or minor (e.g., blinking, looking sideways).
    - The scene remains visually similar.
    - The character has not consented to be shown.

    **Prompt Template (LLM must follow this format strictly):**

    Generate an image of the character with the following constraints:

    - **Clothing & Outfit**: [Unchanged or describe new outfit in detail, e.g., "A pair of loose jeans, a tank top, and a pink hat."]
    - **Haircut and Color**: [Unchanged or describe new hairstyle and/or color, e.g., "Curly hair, pink-colored."]
    - **Facial Features**: Always keep facial features unchanged. Match original face structure, proportions, and identity.
    - **Facial Expression**: [Typically unchanged, but very subtle changes like "a subtle smile" or "slightly raised eyebrow" are allowed.]
    - **Body Size**: Unchanged.
    - **Background**: [Context-driven. Vividly describe if changed, e.g., "A flourishing garden lies behind her."]
    - **Color Tone**: Keep unchanged.
    - **Image Style**: Keep unchanged.
    - **Action and Props**: [Only if visually relevant, e.g., "Standing barefoot on the sand, one hand shielding eyes from the sun."]

    **⚠ Rules:**
    - Do NOT include more than one person in the image.
    - DO NOT exaggerate expressions or change the face.
    - Only generate if the scene meaningfully benefits from it.

    📸 Example 1:
    Clothing & Outfit: A pair of loose jeans, a tank top, and a pink hat.
    Haircut and Color: Pink-colored hair, styled in soft curls.
    Facial Features: Unchanged — match original facial structure and identity.
    Facial Expression: A subtle smile.
    Body Size: Unchanged.
    Background: A flourishing garden lies behind her, filled with vibrant flowers and winding stone paths.
    Color Tone: Unchanged.
    Image Style: Unchanged.
    Action and Props: Sitting on a stone bench, gently holding a teacup with both hands.

    📸 Example 2:
    Clothing & Outfit: Unchanged.
    Haircut and Color: Curly hair, color unchanged.
    Facial Features: Unchanged — retain the original look.
    Facial Expression: Unchanged.
    Body Size: Unchanged.
    Background: The waves washed the beach, and there were scattered shells along the shoreline.
    Color Tone: Unchanged.
    Image Style: Unchanged.
    Action and Props: Standing barefoot on the sand, one hand shielding eyes from the sun, looking out toward the ocean.
    """

    # This is a placeholder implementation
    print(f"Generating character image with prompt: {prompt}")
    return prompt


if __name__ == "__main__":
    # Initialize and run the server
    print("Starting the MCP character image generator server...")
    mcp.run(transport="stdio")
