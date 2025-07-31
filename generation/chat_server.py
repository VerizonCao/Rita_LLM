from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("character-image-generator")


@mcp.tool()
async def generate_character_image(prompt: str) -> str:
    """
    This function is ONLY triggered when 
    a. background of the image changed OR
    b. character is undressed ( Not triggered by ordinary clothing)
    Only when at least one of the two conditions are met
    If image gen is triggered, use the last text world states and last image edit prompts to generate the new image.
    If the criteria is not met, you should just return nothing.
    
    **Change necessary visual details only.**
    **Always use 'Character' as the subject of the description.**
    **DO NOT include more than one person ( like user's nickname, or other characters or personel ) in the descriptions.**
    
    **Structual Output Guidelines: **
    At the end of each response, you should output each category in the following format according to the world state:
    <head accessories , optional>
    <tops , optional>
    <bottoms , optional>
    <footwear , optional>
    <neck accessories , optional>
    <leg accessories , optional>
    <general accessories , optional>
    Change the image background to <image background , optional>
    
    **Per class of world state guidelines:**
    
    Image background ( Most commonly triggered):
    What constitutes an image background change ?
    - Scene & location change
    - Relative position inside the scene changed, causing image background showing different details 
      ( moving from bedroom entrance to the bed or closet)
    
    Consider specific locations interior or exterior visual, with very specific details and description. 
    If the location is a bedroom, is there a bed ? Is character sitting or lying on the bed, in what pose, where on the bed ?
    What is the color of the bedframe and bedsheet ? Is there a visible window / a lamp / a closet ?
    Consider time of the day or weather. and its effect on lighting.
    If interior, always mention uses in-door lighting, no windows. You may add specific details about the lighting.
    
    Character clothing:
    Character clothing is only ever changed, when the conversation involves undressing the character sexually, female or male.
    DO NOT INVOLVE ORDINARY CLOTHING IN ANY OTHER SCENARIO. Including but not limited to top/bottom/legs/feet/neck/hands/etc.
    If not clothing is not involved, add "Character clothing remains the same" in your response.
    Be specific about the color, style, and other details for each item. Describe the item for the visual details. 
    Do mention sexy left and right legs if there is no legwear. 
    Here are some examples of sexy clothes:
    tops sexy clothes, like bra, bustier top, corset top, and may come in various styles and textures like
    lace, satin/silk, mesh. 
    bottoms, panties, skirt, shorts, lingerie,
    legwear, like thighhighs, stockings, garter, or bare legs, if any. Skip if not present.
    ( If the clothing display both lowerbody clothing like shorts, also legs, you should mention the cloth showing off legs)
    footwear, shoes, boots, high heels, or barefoot, if any. Skip if not present.
    neck accessories, necklace, choker, if any. Skip if not present.
    Clothing state shouldn't add facial wear like glasses or face mask.
    
    Undressing scenario:
    If the conversation invovling undressing the character, you should reflect the underneath clothing, that could like this:
     Character hair spread out on a white bed. 
     Character upperbody dressed with nothing but white lace short cut bra. 
     Character wears short cut tight underpant, showing off long legs, and legs wear white lace stocking, from foot to thigh. 
     Camera angle and position is the same, framing is the same, and perspective is the same.
    Clothing always involve the full body, from head to foot, whether we are describing outfit or underneath clothing.
    
    **What to include:**
    Each image generation prompt should include all details that are visible in the image, relevant to the two classes of world state(background, clothing).
    - Include all relevant world state, either on image background, or clothing, to the new image generation prompt.
    - If changing image background, each item must begin or end with "change the image background to" or "add/remove/change"
      in image background details.
      
    **What NOT to include:**
    - Any character's nickname, or any other characters or personel.
    - The interaction between character and other characters or personel, like moving closer, staring into their eyes, etc.
    - The internal thoughts of the character, like "I'm feeling nervous", "I'm feeling excited", etc.
    - Facial expressions, like smiling, frowning, rolling eyes, etc.

    **Format Rules:**
    - Always use 'Character' as the subject of the description.
    - Your prompt should only describe the character, either under different image background, or different clothing. 
       DO NOT include more than one person 
      ( like user's nickname, or other characters or personel ) in the descriptions.
    - DO NOT mention any facial expressions or change the facical appearance details like eyes, mouth, 
      smirking, blushing, rolling eyes, etc.
    - DO NOT use any emojis, special characters, or any markdown formatting. Use plain English text.

    **Writing Rules:**
    - Use intuitive, straightforward, descriptive language. Don't use anything conceptual, 
      abstract, vague, poetic, complex.
      The input messages may contain complex and non-descriptive language, you should simplify it.
    - Careful to not use words that have influence across different categories. 
      For example, when describing a long female dress, you shouldn't use words like 'floor-length', 
      as it may affect the image background setting and camera framing to focus on 'floor' which is unnecessary.
    - Hard rules about pose:
      1. Never let the character turn their back to the camera.
      2. The character should always be facing the camera, never turn their heads to the back nor the side of camera.
    
    **Always include at the end of your response:**
    Camera angle and position is the same, framing is the same, and perspective is the same.
    Character pose stays the same, same relative position to the camera. Character framing and perspective remain the same.
    (If image background uses in-door setting) Uses in-door lighting, no windows.
    (If character clothing doesn't change, which is almost always the case) Character clothing should remain the same.
    """

    # This is a placeholder implementation
    print(f"Generating character image with prompt: {prompt}")
    return prompt


if __name__ == "__main__":
    # Initialize and run the server
    print("Starting the MCP character image generator server...")
    mcp.run(transport="stdio")
