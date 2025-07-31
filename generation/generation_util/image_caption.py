import os
import base64
from typing import Union
import dotenv
from openai import OpenAI

if os.path.exists(".env.local"):
    dotenv.load_dotenv(".env.local")
else:
    dotenv.load_dotenv()

image_caption_prompt = """
You are an image captioning model specialized in analyzing character images to extract world state information. Your task is to examine a character image and provide a detailed description organized into three main classes: Background, Character Clothing, and Character Pose.
Core Requirements

Always use 'Character' as the subject in all descriptions
Focus on a single character only - do not describe multiple people
Provide objective, descriptive language - avoid abstract, poetic, or conceptual terms
Use plain English text - no emojis, special characters, or markdown formatting
Do not include facial expressions - focus on clothing, pose, and environment only

Output Structure
Organize your response into the following categories:
1. BACKGROUND
Describe the setting and environment:

Location type: interior/exterior, specific room or place
Environmental details: furniture, objects, architectural elements
Lighting conditions: time of day, artificial/natural lighting, shadows
Spatial elements: walls, floors, windows, doors, decorative items
Color scheme: dominant colors of the background elements

2. CHARACTER CLOTHING
Describe each clothing category that is visible (skip categories if not present):

Head accessories: hats, hair accessories, head coverings
Tops: shirts, jackets, coats, blouses, sweaters, or bare chest
Bottoms: pants, skirts, shorts, dresses, hosiery, or bare legs
Footwear: shoes, boots, heels, or barefoot
Neck accessories: scarves, ties, necklaces, chokers
Leg accessories: thigh-high stockings, garters, leg warmers
General accessories: jewelry, watches, belts, bags

For each item, specify:

Color and material
Style and cut
Fit and coverage
Any distinctive features or patterns.

3. CHARACTER POSE
Describe the character's body position and stance:

Overall body position: standing, sitting, lying, kneeling
Head orientation: facing forward (should always face camera/viewer)
Arm and hand positions: specific placement and gestures
Leg positioning: stance, crossing, positioning
Interaction with environment: touching objects, using furniture
General posture: upright, relaxed, formal, casual

Special Guidelines
Pose Requirements

Character must always be front-facing toward the camera/viewer
Never describe the character turning their back or side to the viewer
Head should be straight and facing forward
If describing hand gestures, mention both left and right hands specifically

Clothing Details

Be highly specific about colors, styles, and visual details
If legs are visible without legwear, explicitly mention "bare legs"
If clothing shows skin (like shorts showing legs), mention both the clothing and exposed areas
For undressed scenarios, describe visible undergarments and exposed areas accurately

Background Context

Consider time of day and how it affects lighting
Describe specific furniture and room elements if indoors
Mention weather conditions if outdoors
Include spatial relationships (character's position relative to background elements)

Output Format
Structure your response as:
BACKGROUND:
[Detailed description of setting and environment]
CHARACTER CLOTHING:

Head accessories: [description or "None visible"]
Tops: [description or "None visible"]
Bottoms: [description or "None visible"]
Footwear: [description or "None visible"]
Neck accessories: [description or "None visible"]
Leg accessories: [description or "None visible"]
General accessories: [description or "None visible"]

CHARACTER POSE:
[Detailed description of body position, stance, and positioning]
Remember: Focus on what is visually present in the image. Provide clear, descriptive language that would allow someone to understand the scene without seeing the image themselves.

"""

def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def caption_image(image_input: Union[str, bytes], api_key: str = None) -> str:
    """
    Caption an image using llama-4 maverick model via OpenRouter
    
    Args:
        image_input: Either a file path (str) or base64 encoded image data (bytes/str)
        api_key: OpenRouter API key (if None, will try to get from environment)
    
    Returns:
        str: The text caption of the image
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("API key must be provided or set in OPENROUTER_API_KEY environment variable")
    
    # Handle different input types
    if isinstance(image_input, str):
        if os.path.isfile(image_input):
            # It's a file path
            base64_image = encode_image_to_base64(image_input)
            image_url = f"data:image/jpeg;base64,{base64_image}"
        elif image_input.startswith(('http://', 'https://')):
            # It's already a URL
            image_url = image_input
        elif image_input.startswith('data:image'):
            # It's already a data URL
            image_url = image_input
        else:
            # Assume it's base64 data
            image_url = f"data:image/jpeg;base64,{image_input}"
    elif isinstance(image_input, bytes):
        # Convert bytes to base64
        base64_image = base64.b64encode(image_input).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_image}"
    else:
        raise ValueError("image_input must be a file path, URL, base64 string, or bytes")
    
    # Initialize OpenAI client with OpenRouter
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    try:
        # Make the API request using OpenAI client
        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": image_caption_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ]
        )
        
        # Extract text from response
        return response.choices[0].message.content
        
    except Exception as e:
        raise Exception(f"API request failed: {e}")


def caption_image_file(image_path: str, api_key: str = None) -> str:
    """
    Convenience function to caption an image file
    
    Args:
        image_path: Path to the image file
        api_key: OpenRouter API key for the service
    
    Returns:
        str: The text caption of the image
    """
    return caption_image(image_path, api_key)


def caption_image_url(image_url: str, api_key: str = None) -> str:
    """
    Convenience function to caption an image from URL
    
    Args:
        image_url: URL of the image
        api_key: OpenRouter API key for the service
    
    Returns:
        str: The text caption of the image
    """
    return caption_image(image_url, api_key)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python image_caption.py <image_path_or_url>")
        sys.exit(1)
    
    image_input = sys.argv[1]
    
    try:
        caption = caption_image(image_input)
        print(caption)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 