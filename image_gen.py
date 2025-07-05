# image gen tool
import os
import requests
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageGenerator:
    """
    Simple image generation and editing tool using FLUX.1 Kontext model via Replicate API.
    """
    
    def __init__(self):
        """Initialize the image generator with Replicate API configuration."""
        self.api_key = os.getenv("REPLICATE_API_TOKEN")
        if not self.api_key:
            raise ValueError("REPLICATE_API_TOKEN environment variable is not set")
        
        self.model_version = "black-forest-labs/flux-kontext-dev"
    
    def run(self, input_params: Dict[str, Any]) -> str:
        """
        Run the FLUX.1 Kontext model with the given input parameters.
        
        Args:
            input_params: Dictionary containing input parameters like:
                - prompt: Text description or editing instruction
                - input_image: URL of input image (optional)
                - output_format: Output format (jpg, png, etc.)
                - aspect_ratio: Aspect ratio (optional)
                - prompt_upsampling: Boolean (optional)
                - seed: Random seed (optional)
                - safety_tolerance: Safety tolerance (optional)
                - disable_safety_checker: Disable NSFW safety checker (optional)
        
        Returns:
            URL of the generated image
        """
        try:
            # Enable NSFW content by default
            if 'disable_safety_checker' not in input_params:
                input_params['disable_safety_checker'] = True
            
            # Create the prediction request
            payload = {
                "version": self.model_version,
                "input": input_params
            }
            
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json"
            }
            
            logger.info(f"Starting image generation with prompt: {input_params.get('prompt', '')[:100]}...")
            
            # Make the API request
            response = requests.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            prediction = response.json()
            prediction_id = prediction["id"]
            
            logger.info(f"Prediction created with ID: {prediction_id}")
            print(f"Prediction created with ID: {prediction_id}")
            
            # Poll for completion
            result_url = self._poll_prediction(prediction_id)
            
            if result_url:
                logger.info(f"Image generation completed: {result_url}")
                print(f"Image generation completed: {result_url}")
                return result_url
            else:
                raise Exception("Image generation failed or timed out")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            raise
    
    def download_image(self, image_url: str, output_path: str) -> str:
        """
        Download an image from a URL and save it to a file.
        
        Args:
            image_url: URL of the image to download
            output_path: Path where to save the image file
            
        Returns:
            Path to the saved image file
        """
        try:
            logger.info(f"Downloading image from: {image_url}")
            
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Save the image
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Image saved to: {output_path}")
            print(f"Image saved to: {output_path}")
            return output_path
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download image: {e}")
            raise
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            raise
    
    def _poll_prediction(self, prediction_id: str, max_wait_time: int = 300) -> Optional[str]:
        """
        Poll the prediction until it completes or times out.
        
        Args:
            prediction_id: The prediction ID to poll
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            The result URL if successful, None otherwise
        """
        import time
        
        start_time = time.time()
        poll_interval = 2
        
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
        
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers=headers
                )
                response.raise_for_status()
                
                prediction = response.json()
                status = prediction.get("status")
                
                if status == "succeeded":
                    # Extract the result URL
                    result = prediction.get("output")
                    if isinstance(result, str):
                        return result
                    elif isinstance(result, list) and len(result) > 0:
                        return result[0]
                    else:
                        logger.error(f"Unexpected result format: {result}")
                        return None
                        
                elif status == "failed":
                    error = prediction.get("error", "Unknown error")
                    logger.error(f"Prediction failed: {error}")
                    return None
                    
                elif status in ["starting", "processing"]:
                    logger.debug(f"Prediction status: {status}")
                    time.sleep(poll_interval)
                    poll_interval = min(poll_interval * 1.5, 10)
                    continue
                    
                else:
                    logger.warning(f"Unknown prediction status: {status}")
                    time.sleep(poll_interval)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Error polling prediction: {e}")
                time.sleep(poll_interval)
                
        logger.error(f"Prediction timed out after {max_wait_time} seconds")
        return None
    
    def validate_input_image(self, image_url: str) -> bool:
        """
        Validate that the input image URL points to a supported format.
        
        Args:
            image_url: URL of the image to validate
            
        Returns:
            True if valid, False otherwise
        """
        supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        image_url_lower = image_url.lower()
        
        return any(image_url_lower.endswith(fmt) for fmt in supported_formats)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the FLUX.1 Kontext model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "name": "FLUX.1 Kontext Dev",
            "description": "Open-weight version of FLUX.1 Kontext for text-based image editing",
            "capabilities": [
                "Style transfer",
                "Object/clothing changes", 
                "Text editing",
                "Background swapping",
                "Character consistency"
            ],
            "supported_formats": ["jpg", "jpeg", "png", "gif", "webp"],
            "safety_tolerance_range": "0-6 (max 2 with input images)"
        }


# Convenience function for simple usage
def run_flux_model(input_params: Dict[str, Any], download_to: Optional[str] = None, ImageGenerator_PassIn: ImageGenerator = None) -> str:
    """
    Simple function to run FLUX.1 Kontext model.
    
    Args:
        input_params: Dictionary containing input parameters
        download_to: Optional path to save the generated image (e.g., "output.jpg")
    
    Example:
        input_params = {
            "prompt": "Make this a 90s cartoon",
            "input_image": "https://example.com/image.png",
            "output_format": "jpg"
        }
        # Just get the URL
        output_url = run_flux_model(input_params)
        
        # Get URL and download the image
        output_url = run_flux_model(input_params, download_to="output.jpg")
    """

    if ImageGenerator_PassIn:
        output_url = ImageGenerator_PassIn.run(input_params)
    
        if download_to:
            ImageGenerator_PassIn.download_image(output_url, download_to)
        
        return output_url
    else:
        generator = ImageGenerator()
        output_url = generator.run(input_params)
        
        if download_to:
            generator.download_image(output_url, download_to)
        
        return output_url

