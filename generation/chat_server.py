from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("character-image-generator")


@mcp.tool()
async def generate_character_image(prompt: str) -> str:
    """
    The World Agent has already determined that a background change is needed.
    **Your task is to generate the specific background description for the image generation.**    
    If image gen is triggered, use the last text world states and last image edit prompts to generate the new image.
    If the criteria is not met, you should just return nothing.
    **What constitutes an image background change ?**
    - Scene & location change
    - Relative position inside the scene changed, causing image background showing different details 
      ( moving from bedroom entrance to the bed or closet)
    - You should never, ever repeat the same image background from previous image edit prompts.
    - Each image edit prompt should produce very visually different image details.

    **Structual Output Guidelines: **
    You should output in this format:
    '''
    Change the image background to <image background description>
    '''
    
    **Guidelines for image background:**
    Consider specific locations interior or exterior visual, with very specific details. 
    Take bedroom as an example, if the location is a bedroom, is there a bed ?
    What is the color of the bedframe and bedsheet ? Is there a visible window / a lamp / a closet ?
    Consider time of the day or weather. and its effect on lighting.
    If interior, always mention uses in-door lighting, no windows. You may add specific details about the lighting.
      
    **What NOT to include as image background:**
    - Do not mention the main character or user. 
      DO NOT include more than one person ( like user's nickname, or other characters or personel ) in the descriptions.
    - DO not mention any interaction between character and other characters or personel, 
      like moving closer, staring into their eyes, or any pose change, etc.
    - DO not mention any interaction between character and environment, like sitting on a chair, or standing in a room, etc.
      Solely focus on visible background details, which shouldn't has any interaction with character.
    - DO not mention any internal thoughts of the character, like "I'm feeling nervous", "I'm feeling excited", etc.
    - DO not mention any facial expressions, like smiling, frowning, rolling eyes, eyes, mouth, 
      smirking, blushing,etc.

    **Writing Rules:**
    - Use intuitive, straightforward, descriptive language. Don't use anything conceptual, 
      abstract, vague, poetic, complex.
      The input messages may contain complex and non-descriptive language, you should simplify it.
    - DO NOT use any emojis, special characters, or any markdown formatting. Use plain English text.
    - DO NOT use words like "Close up" which implies camera angle. Image background should be seen from a distance.
    """

    # This is a placeholder implementation
    print(f"Generating character image with prompt: {prompt}")
    return prompt


if __name__ == "__main__":
    # Initialize and run the server
    print("Starting the MCP character image generator server...")
    mcp.run(transport="stdio")
