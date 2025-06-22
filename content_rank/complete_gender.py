import os
import json
import base64
import requests
import psycopg2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_env_file(env_path: str = "Rita_LLM/.env.local"):
    """Load environment variables from .env.local file"""
    env_vars = {}
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")
        logger.info(f"Loaded {len(env_vars)} environment variables from {env_path}")
        return env_vars
    except Exception as e:
        logger.error(f"Failed to load environment file {env_path}: {e}")
        return {}

class GenderClassifier:
    def __init__(self):
        # Load credentials from .env.local
        env_vars = load_env_file()
        self.openrouter_api_key = env_vars.get('OPENROUTER_API_KEY')
        self.postgres_url = env_vars.get('POSTGRES_URL')
        
        # AWS credentials for S3
        self.aws_access_key_id = env_vars.get('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = env_vars.get('AWS_SECRET_ACCESS_KEY')
        self.aws_region = env_vars.get('AWS_REGION', 'us-west-2')
        self.aws_bucket_name = env_vars.get('AWS_BUCKET_NAME', 'rita-avatar-image')
        
        self.output_dir = Path('Rita_LLM/content_rank/output')
        
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in Rita_LLM/.env.local")
        if not self.postgres_url:
            raise ValueError("POSTGRES_URL not found in Rita_LLM/.env.local")
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError("AWS credentials not found in Rita_LLM/.env.local")
            
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            region_name=self.aws_region,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key
        )
        
        # OpenRouter API configuration
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "meta-llama/llama-4-maverick"
        
        # System prompt for gender classification
        self.system_prompt = """You are a gender classification system for avatar characters. 

Your task is to analyze the provided avatar image and character information (name, bio, prompt) to determine the character's gender presentation.

IMPORTANT: You must respond with EXACTLY ONE of these three options:
- male
- female  
- non-binary

Classification guidelines:
- "male": Characters that clearly present as male
- "female": Characters that clearly present as female
- "non-binary": Characters that are ambiguous, non-human (furry, alien, robot), or explicitly non-binary

Base your classification primarily on:
1. Visual appearance in the image (facial features, body shape, clothing style)
2. Character name (if clearly gendered)
3. Character bio/prompt content (pronouns, descriptions)

If the character is:
- Non-human (furry, alien, robot, etc.): classify as "non-binary"
- Ambiguous or androgynous: classify as "non-binary"
- Clearly presenting as one gender: classify as that gender

Respond with ONLY the gender classification word. Do not include any explanation or additional text."""

        # JSON Schema for structured gender classification
        self.gender_schema = {
            "name": "gender_classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "gender": {
                        "type": "string",
                        "enum": ["male", "female", "non-binary"],
                        "description": "Gender classification of the avatar"
                    }
                },
                "required": ["gender"],
                "additionalProperties": False
            }
        }

        # Thread-safe counter for progress tracking
        self._lock = threading.Lock()
        self._processed_count = 0
        self._total_count = 0

    def get_avatars_without_gender(self) -> List[Tuple[str, str, str, str, str]]:
        """Get avatar data for avatars with NULL gender"""
        try:
            conn = psycopg2.connect(self.postgres_url)
            cursor = conn.cursor()
            
            query = """
            SELECT avatar_id, avatar_name, agent_bio, prompt, image_uri
            FROM avatars 
            WHERE gender IS NULL 
            AND image_uri IS NOT NULL 
            AND image_uri != ''
            ORDER BY create_time DESC
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            logger.info(f"Found {len(results)} avatars without gender classification")
            return results
            
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return []

    def get_presigned_url(self, s3_key: str) -> Optional[str]:
        """Generate presigned URL from S3 object key"""
        try:
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.aws_bucket_name, 'Key': s3_key},
                ExpiresIn=3600  # 1 hour expiration
            )
            logger.info(f"Generated presigned URL for {s3_key}")
            return presigned_url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL for {s3_key}: {e}")
            return None

    def get_or_download_image(self, image_uri: str, avatar_id: str) -> Optional[bytes]:
        """Get image from local cache or download from S3"""
        try:
            # Check if image already exists locally in output directory
            avatar_dir = self.output_dir / avatar_id
            existing_image = None
            
            if avatar_dir.exists():
                for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                    image_path = avatar_dir / f"image{ext}"
                    if image_path.exists():
                        existing_image = image_path
                        break
            
            # If image exists locally, use it
            if existing_image:
                logger.info(f"Using existing image for {avatar_id}: {existing_image}")
                with open(existing_image, 'rb') as f:
                    return f.read()
            
            # Download image from S3
            logger.info(f"Downloading image for {avatar_id} from S3")
            presigned_url = self.get_presigned_url(image_uri)
            if not presigned_url:
                return None
            
            response = requests.get(presigned_url, timeout=30)
            response.raise_for_status()
            image_content = response.content
            
            # Create avatar directory and save image locally for future use
            avatar_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine file extension from original S3 key or response headers
            file_extension = '.jpg'  # default
            if '.' in image_uri:
                file_extension = '.' + image_uri.split('.')[-1]
            elif 'content-type' in response.headers:
                content_type = response.headers['content-type']
                if 'png' in content_type:
                    file_extension = '.png'
                elif 'gif' in content_type:
                    file_extension = '.gif'
                elif 'webp' in content_type:
                    file_extension = '.webp'
            
            # Save image locally
            image_path = avatar_dir / f"image{file_extension}"
            with open(image_path, 'wb') as f:
                f.write(image_content)
            logger.info(f"Saved image locally: {image_path}")
            
            return image_content
            
        except Exception as e:
            logger.error(f"Failed to get/download image for {avatar_id}: {e}")
            return None

    def classify_gender(self, avatar_data: Dict, image_content: bytes) -> Optional[str]:
        """Classify gender using OpenRouter API"""
        try:
            # Prepare character information
            char_info = f"""Character Name: {avatar_data['name']}
Character Bio: {avatar_data['bio']}
Character Prompt: {avatar_data['prompt']}"""
            
            # Encode image for API
            encoded_image = base64.b64encode(image_content).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{encoded_image}"
            
            # API request with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    headers = {
                        "Authorization": f"Bearer {self.openrouter_api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    payload = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {
                                "role": "user", 
                                "content": [
                                    {"type": "text", "text": f"Please classify the gender of this character:\n\n{char_info}"},
                                    {"type": "image_url", "image_url": {"url": data_url}}
                                ]
                            }
                        ],
                        "provider": {
                            "require_parameters": True
                        },
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": self.gender_schema
                        },
                        "max_tokens": 100,
                        "temperature": 0.1
                    }
                    
                    response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                    response.raise_for_status()
                    response_data = response.json()
                    break  # Success, exit retry loop
                    
                except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}")
                        logger.info(f"Retrying in {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"API request failed after {max_retries} attempts: {e}")
                        return None
                except Exception as e:
                    logger.error(f"Unexpected API error: {e}")
                    return None
            
            if 'choices' in response_data and len(response_data['choices']) > 0:
                content = response_data['choices'][0]['message']['content']
                try:
                    # Parse the JSON response
                    result = json.loads(content)
                    gender = result.get('gender', '').lower()
                    
                    # Validate gender is one of the allowed values
                    valid_genders = ['male', 'female', 'non-binary']
                    if gender in valid_genders:
                        logger.info(f"Successfully classified gender as: {gender}")
                        return gender
                    else:
                        logger.error(f"Invalid gender classification: {gender}")
                        return None
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse gender classification JSON: {e}")
                    logger.error(f"Content: {content}")
                    
                    # Fallback: try to extract gender from raw text
                    content_lower = content.lower()
                    if 'male' in content_lower and 'female' not in content_lower:
                        return 'male'
                    elif 'female' in content_lower:
                        return 'female'
                    elif 'non-binary' in content_lower:
                        return 'non-binary'
                    else:
                        logger.error(f"Could not extract valid gender from content: {content}")
                        return None
            else:
                logger.error("No choices in API response")
                return None
                
        except Exception as e:
            logger.error(f"Gender classification failed: {e}")
            return None

    def update_avatar_gender(self, avatar_id: str, gender: str) -> bool:
        """Update avatar gender in database"""
        try:
            conn = psycopg2.connect(self.postgres_url)
            cursor = conn.cursor()
            
            update_query = """
            UPDATE avatars 
            SET gender = %s
            WHERE avatar_id = %s
            """
            
            cursor.execute(update_query, (gender, avatar_id))
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Updated gender for {avatar_id}: {gender}")
            return True
            
        except Exception as e:
            logger.error(f"Database update failed for {avatar_id}: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
            return False

    def process_single_avatar(self, avatar_data: Tuple[str, str, str, str, str]) -> Tuple[str, bool, str]:
        """Process a single avatar for gender classification"""
        avatar_id, name, bio, prompt, image_uri = avatar_data
        
        try:
            # Prepare avatar data dict
            avatar_info = {
                'avatar_id': avatar_id,
                'name': name or '',
                'bio': bio or '',
                'prompt': prompt or ''
            }
            
            # Get or download image
            image_content = self.get_or_download_image(image_uri, avatar_id)
            if not image_content:
                logger.error(f"Failed to get image for avatar {avatar_id}")
                return avatar_id, False, ''
            
            # Classify gender
            gender = self.classify_gender(avatar_info, image_content)
            if not gender:
                logger.error(f"Failed to classify gender for avatar {avatar_id}")
                return avatar_id, False, ''
            
            # Update database
            success = self.update_avatar_gender(avatar_id, gender)
            
            return avatar_id, success, gender
            
        except Exception as e:
            logger.error(f"Error processing {avatar_id}: {e}")
            return avatar_id, False, ''

    def _update_progress(self, avatar_id: str, success: bool, gender: str):
        """Thread-safe progress update"""
        with self._lock:
            self._processed_count += 1
            status = "✅" if success else "❌"
            gender_display = f"({gender})" if gender else ""
            print(f"{status} [{self._processed_count}/{self._total_count}] {avatar_id} {gender_display}")

    def run_batch_classification(self, max_workers: int = 16):
        """Run batch gender classification with multithreading"""
        logger.info(f"Starting gender classification with {max_workers} workers")
        print(f"🎯 Gender Classifier - Batch Mode")
        print("=" * 50)
        print(f"🧵 Using {max_workers} worker threads")
        
        # Get avatars to process
        avatars = self.get_avatars_without_gender()
        if not avatars:
            print("✅ No avatars found without gender classification")
            return
        
        print(f"📊 Found {len(avatars)} avatars to classify")
        
        # Reset counters
        with self._lock:
            self._processed_count = 0
            self._total_count = len(avatars)
        
        successful_classifications = []
        failed_classifications = []
        gender_counts = {'male': 0, 'female': 0, 'non-binary': 0}
        
        # Process avatars with multithreading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_avatar = {
                executor.submit(self.process_single_avatar, avatar_data): avatar_data[0] 
                for avatar_data in avatars
            }
            
            # Process completed tasks
            for future in as_completed(future_to_avatar):
                avatar_id = future_to_avatar[future]
                try:
                    processed_avatar_id, success, gender = future.result()
                    self._update_progress(processed_avatar_id, success, gender)
                    
                    if success:
                        successful_classifications.append((processed_avatar_id, gender))
                        if gender in gender_counts:
                            gender_counts[gender] += 1
                    else:
                        failed_classifications.append(processed_avatar_id)
                        
                except Exception as e:
                    logger.error(f"Exception in thread for {avatar_id}: {e}")
                    failed_classifications.append(avatar_id)
                    self._update_progress(avatar_id, False, '')
        
        # Final summary
        success_count = len(successful_classifications)
        total_count = len(avatars)
        
        print(f"\n🎉 Batch gender classification complete!")
        print(f"📈 Successfully classified: {success_count}/{total_count} avatars")
        
        if success_count > 0:
            print(f"👥 Gender distribution:")
            print(f"   Male: {gender_counts['male']}")
            print(f"   Female: {gender_counts['female']}")
            print(f"   Non-binary: {gender_counts['non-binary']}")
        
        if failed_classifications:
            print(f"❌ Failed classifications: {len(failed_classifications)}")
            print(f"   {', '.join(failed_classifications[:10])}{' ...' if len(failed_classifications) > 10 else ''}")
        
        logger.info(f"Batch complete: {success_count}/{total_count} genders classified successfully")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Avatar Gender Classification System')
    parser.add_argument('--max-workers', type=int, default=16,
                       help='Number of worker threads for batch processing (default: 16)')
    
    args = parser.parse_args()
    
    try:
        classifier = GenderClassifier()
        classifier.run_batch_classification(max_workers=args.max_workers)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
