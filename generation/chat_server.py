from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("character-image-generator")


@mcp.tool()
async def generate_character_image(prompt: str) -> str:
    """
    Summarize how the character and setting should change, from user's ( camera's ) first person perspective.
    
    **This function is always triggered. You task is too mention how to change the background and character details shown in the image.**
    You should carefully describe the change, even if the change is subtle (e.g., different hand position, slighly moving their head).

    **Response Requirements**
    We consider three types of changes:
    A. User(Camera) and character stay at the same positions, make appearance or small pose changes.
       Same camera, same character framing. Editing either background, character pose, or character apparel.
       Camera and character framing stays the same, we maintain identical camera angle, framing, and perspective. 
       Only update the background, character pose, or character apparel that is visibly changed in the image.
    B. One of user(camera) or character physically moved inside the scene. 
       Here we would keep character apparel details and location the same. We may update background objects or pov-based changed details due to orientation change, or update character pose 
       due to movement and new position.       
       You should add or remove more details that isn't available in the old image but is visible in the new image.
       In terms of conceptual scene-level details, we don't change any background and character aesthetic details. 
       However this case may ask for more details to display in the image that is consistent with the old image. 
       For example if camera is zooming out, we would be describing what isn't shown in the original image, but visible in the new image, 
       for example you may say "camera zooms out, showing the full body of the character. 
       Character has long and attractive legs, wearing a dark stocking. The background shows a bedroom with a bed and a closet."
       Or for example, if User(Camera) moved, the pov might be showing a different side of the character and background. You should provide such details update too.
       Another example, if user(Camera) moved from a remote view into a close up view ( across the restaurant table for example, or side by side with character ), 
       The pov-related change should be reflected in the description.
    C. A complete scene change.
       Here you are allowed to interpret the new scene, character details however you like. However, in most cases, you should make minimum changes to the 
       character apparel and pose details. You may focus on changing the background and camera-related pov details.

    In any cases, we would focus either scene-level and character-level changes in specific categories, and provide a descriptive new set of details:
    Examples of valid item to change:
    Catergory Background:
    * background/scene setting 
       Consider specific locations change with very specific details and description. 
       If the new location is a bedroom, is there a bed ? Is character sitting or lying on the bed, in what pose, where on the bed ?
       What is the color of the bedframe and bedsheet ? Is there a window / a lamp / a closet ?
       If the location changed, should lighting change ? What is the lighting like ? How does it shown on character and background ?
       If the scene doesn't change, are we moving to a slightly different location inside the same scene, like moving from bed to closet ?
    * background props addition or removal 
       Whether we are switchign to a new scene, or moving to a slightly different location inside the same scene, 
       what should be added or removed from background ? 
       If the location changed, what should added or changed in the background ?
    Catergory Character Pose:
    * face facing direction 
      ( Consider slightly change facing direction or slightly tilting the head.
        You may use words like slightly to left, front, slightly to right, etc.
        If mentioned, also mention eyes directed at front facing camera.
    * pose (e.g standing, sitting, kneeling, or simply changing facing direction, etc, you may connect this with hand gesture or background)
    * hand gesture and position, mention specfic hand gesture and position. Is the hand holding the face, the necklace, or any prop ?
    Catergory Character Apparel:
    * upperbody clothing, mention specfic style, color, and designs.
    * lowerbody clothing, mention specfic style, color, and designs.
    * shoes & socks, mention specfic style, color, and designs.
    * hand prop (e.g what should character be holding in hand, or wearing like a wristband)
    * facial apparels addition or removal (e.g glasses, hat, earrings, necklace, etc)
    
    Category Camera POV ( For Type B and C )
    You should also consider the POV-based details related to background and character pose.
    What is the camera looking at, ( for example looking through the window ?) 
    What is the character position, and is the camera looking at from the front, side, top, or what angles ? What does that particular angle show regarding background ?
    You are not limited to these categories, you may add any other item that is relevant to the change and IS VISIBLE IN THE NEW IMAGE.

    **Format Rules:**
    - DO NOT mention the existing / old description of the item, focus on the new details.
    - Do NOT include more than one person in the new descriptions.
    - DO NOT mention any facial expressions or change the facical appearance details like eyes, mouth, etc.
    - DO NOT use any emojis, special characters, or any markdown formatting. Use plain English text.

    **Coherence Rules:**
    - Mention connection between the changes, your final response should be coherent. 
      For example, if the character is driving a car, it is probably mentioned in the background to show car seat and window details.
    - Use intuitive, straightforward, descriptive language. Don't use anything conceptual, abstract, vague, poetic, complex.
    - Careful to not use words that have influence across different categories. 
      For example, when describing a long female dress, you shouldn't use words like 'floor-length', 
      as it may affect the background setting and camera framing to focus on 'floor' which is unnecessary.
    - Subtle changes are allowed and encouraged:
        for example if character and user are sitting down chatting, the only change could be just moving their hand to a different position,  
        or changing their poses a bit, or small interaction with props. In this case, you should not mention background or clothing change.
    - Always think about POV related changes.
      The pov should always mimick how the user would see the scene with the character in it. If the character and user are sitting down at the 
      restaurant table, the pov should be looking at the character from the front, showing off the booth/sofa etc. It shouldn't be looking at the side
      of the character nor the sofa booth, unless the user is physically moved to that kind of pov.
        
    Consider our initial three types of changes: 
    A. Camera and character framing stays the same, change either background , character apparel , or character pose.
    B. Either user(camera) or character physically moved inside the scene, keep the same general background, character apparel.
    C. A complete scene change.
    
    **Content Rules:**
    - If under Type A, you should focus on only one major category to change ( background, character apparel, or character pose ). 
      You are not allowed, for example, to mention both background and character pose change in the same response. One at a time.
      The clothing & apparel change should only occur if 
      a. User specifically asked for it.
      b. The story line allowed it, like the character took off their clothes, or put on new clothes.
      Basically, you shouldn't update the clothing & apparel in a consistent conversation occuring in the same location.
      If the conversation doesn't invovle scene change or apparal change, simply make sutble hand position / pose change that is common within conversations.
    - If under Type B, you should focus on the new details displayed in the image.
      What has been added to the pov given the new relative position.
    - If under Type C, you should focus on the new details regarding background. You may update character apparel and pose details, but you should focus on the background.
    - Each item should be described in only one sentence. Don't give too much unnecessary details.
    
    **Maintain Consistency Rules:**
    The ending prompt for what not to change:
    You shall add the following to the end of your response:
    For type A change, you shall 
        Always add "Maintain identical camera angle, framing, and perspective"
        And for keeping the same background, you shall add
            "Maintain the same background, lighting, and subject placement."
        And for keeping the same character pose, you shall add
            "Maintain character pose, keep the person in the exact same position, scale, and pose."
        And for keeping the same character apparel, you shall add
            "Maintain the same cloth, apparels, accessories, hair styles, and other character appearance details."
    For type B change, you should mention similar things, choose one or more from the following:
        "Maintain the same background, lighting, and subject placement."
        "Maintain the same character pose, keep the person in the exact same position, scale, and pose."
        "Maintain the same cloth, apparels, accessories, hair styles, and other character appearance details."
    For type C change, you should mention part of the following:
        If no cloth change: "Maintain the same cloth, apparels, accessories, hair styles, and other character appearance details."
        If no pose change: "Maintain the character pose, keep the person in the exact same position, scale, and pose."

    Example final response, of a Type A change focusing on character apparel, 
    (do not copy verbatim, do not incude Type A or Type B in your response):
    
    Change upper clothing to a high-collar red military coat with gold trim to match official design.
    Maintain identical camera angle, framing, and perspective.
    Maintain the same background, lighting, and subject placement.
    Maintain character pose, keep the person in the exact same position, scale, and pose. 
    
    Example final response, of a Type B/C change focusing on background & pov change:
    
    Camera move back, showing the full body of the character. 
    The character has long and sexy legs and thighs, very sexy and curvy, a perferct body ratio.
    As the camera zooms out, the character's is kneeling on the ground, with her legs visible.
    camera angle is the same, framing shows more of the character's body, and perspective is the same.
    Maintain the same character pose, keep the person in the exact same position, scale, and pose. 
    Maintain the same background, lighting, and subject placement.
    Maintain the same cloth, apparels, accessories, hair styles, and other character appearance details.
    """

    # This is a placeholder implementation
    print(f"Generating character image with prompt: {prompt}")
    return prompt


if __name__ == "__main__":
    # Initialize and run the server
    print("Starting the MCP character image generator server...")
    mcp.run(transport="stdio")
