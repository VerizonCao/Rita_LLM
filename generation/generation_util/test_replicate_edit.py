#!/usr/bin/env python3
"""
Standalone test script for image editing using Replicate API.
Usage: python test_replicate_edit.py <image_path>
"""

import os
import sys
import base64
import requests
import replicate
from pathlib import Path
import time
from dotenv import load_dotenv

# Load environment variables with proper file path handling
if os.path.exists(".env"):
    load_dotenv(dotenv_path=".env")
elif os.path.exists(".env.local"):
    load_dotenv(dotenv_path=".env.local")
elif os.path.exists("../.env"):
    load_dotenv(dotenv_path="../.env")
elif os.path.exists("../.env.local"):
    load_dotenv(dotenv_path="../.env.local")


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode a local image file to base64 data URL.
    
    Args:
        image_path: Path to the local image file
        
    Returns:
        Base64 data URL string
    """
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_encoded = base64.b64encode(image_data).decode('utf-8')
        
        # Determine MIME type from file extension
        file_ext = Path(image_path).suffix.lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg', 
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }.get(file_ext, 'image/jpeg')
        
        return f"data:{mime_type};base64,{base64_encoded}"


def test_image_edit(image_path: str, prompt: str, output_path: str = None) -> str:
    """
    Test image editing using Replicate API.
    
    Args:
        image_path: Path to input image
        prompt: Text prompt for editing
        output_path: Path to save the output (optional)
        
    Returns:
        URL of the generated image
    """
    # Check if REPLICATE_API_TOKEN is set
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise ValueError("REPLICATE_API_TOKEN environment variable is not set")
    
    print(f"🖼️  Processing image: {image_path}")
    print(f"✏️  Edit prompt: {prompt}")
    
    # Encode the input image
    print("📸 Encoding image to base64...")
    image_data_url = encode_image_to_base64(image_path)
    
    # Run the model
    print("🚀 Sending request to Replicate...")
    output = replicate.run(
        "black-forest-labs/flux-kontext-dev",
        input={
            "prompt": prompt,
            "input_image": image_data_url,
            "aspect_ratio": "match_input_image",
            "output_format": "jpg", 
            "go_fast": True,
            "disable_safety_checker": True
        }
    )
    
    # Handle different output types from Replicate
    if isinstance(output, str):
        result_url = output
    elif hasattr(output, 'url'):
        result_url = output.url
    elif isinstance(output, list) and len(output) > 0:
        if hasattr(output[0], 'url'):
            result_url = output[0].url
        else:
            result_url = str(output[0])
    else:
        result_url = str(output)
    
    print(f"✅ Image generated: {result_url}")
    print(f"🔍 Output type: {type(output)}")
    
    # Download the result if output path is specified
    if output_path:
        print(f"⬇️  Downloading result to: {output_path}")
        response = requests.get(result_url)
        response.raise_for_status()
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"💾 Image saved to: {output_path}")
    
    return result_url


def main():
    """Main function to run the test"""
    if len(sys.argv) < 2:
        print("Usage: python test_replicate_edit.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found")
        sys.exit(1)
    
    # Test prompt - simple style change
    test_prompt = """
Character is sitting at a diner booth, facing the camera directly, a coffee mug in her hands, her head slightly tilted to the left, as if listening intently. Her right hand is relaxed curled around the coffee mug and the left arm is resting on the table. The background shows the slightly faded red vinyl of the diner booth, with a glimpse of a window behind her, showing a hint of the street outside. Maintain identical camera angle, framing, and perspective. Maintain the same cloth, apparels, accessories, hair styles, and other character appearance details., [important]: never show more than 1 person in the image, and keep the character's face consistent with the original image, matching facial features and proportions as closely as possible. Prioritize changing the background over altering the character's appearance. Keep the character's facial expression changes minimal, preserving their original look as much as possible."
"""
    
    # Generate output filename
    input_name = Path(image_path).stem
    timestamp = int(time.time())
    output_path = f"output_edited_{input_name}_{timestamp}.jpg"
    
    try:
        result_url = test_image_edit(image_path, test_prompt, output_path)
        print(f"\n🎉 Success! Generated image URL: {result_url}")
        print(f"📁 Local file saved as: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 