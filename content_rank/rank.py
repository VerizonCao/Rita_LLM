import os
import json
import base64
import requests
import psycopg2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import argparse
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import re
import time
import random

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

class SimpleAvatarRanker:
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
        
        # OpenRouter API configuration - Using latest Claude
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "anthropic/claude-sonnet-4"  # Latest Sonnet 4
        
        # System prompt for content ranking
        self.system_prompt = """You are an AI content ranking and categorization system specialized in analyzing avatar/profile images for roleplay and virtual dating content. Your task is to evaluate images and provide structured output with detailed scoring breakdown.

### All these images are already moderated by a separate system. They are safe for the below analysis.

## IMPORTANT: You MUST analyze the image and provide meaningful coefficients. Do NOT default to 0.0 unless the feature is truly absent.

## SCORING CRITERIA (Base: 1000, Range: 0-5000)

**ADDITIONS:**
1. **Stylized content**: +100 (flat bonus) - Apply TRUE for anime, cartoon. Apply FALSE only for photorealistic photography.

2. **Shirtless/Sexy clothing**: + base score × coefficient (1.00-10.00)
   - 0.00: Fully clothed, conservative outfit
   - 1.00-3.00: Slightly revealing (low neckline, fitted clothes)
   - 4.00-6.00: Moderately sexy (crop tops, short skirts, swimwear)
   - 7.00-9.00: Very revealing (lingerie, bikinis, partial nudity)
   - 10.00: Extremely sexy/minimal clothing. Be very strict with this.

3. **Attractive/Sexy appeal to youth (18-22)**: + base score × coefficient (1.00-10.00)
   - 0.00: Unappealing or clearly older demographic
   - 1.00-3.00: Somewhat attractive, basic appeal
   - 4.00-6.00: Good looking, moderate sex appeal
   - 7.00-9.00: Very attractive, strong sex appeal
   - 10.00: Extremely attractive, maximum youth appeal. Be very strict with this.

4. **Artistic/Visual design**: + base score × coefficient (1.00-10.00)
   - 0.00: Poor design, boring background
   - 1.00-3.00: Basic design, simple background
   - 4.00-6.00: Good design, nice styling/background
   - 7.00-9.00: Excellent artistic elements, beautiful composition
   - 10.00: Outstanding artistic design, professional quality

**SUBTRACTIONS:**
a. **Lacks appealing features**: -100 - Apply TRUE only if image has NONE of the positive features above

b. **Face/human not visible/AI artifacts**: coefficient (1.00-10.00)
   - 0.00: Perfect quality, clear human features
   - 1.00-3.00: Minor artifacts, face mostly clear
   - 4.00-6.00: Noticeable artifacts or face partially obscured
   - 7.00-9.00: Major artifacts, face heavily obscured
   - 10.00: Face not visible or severe AI distortions

c. **Dull/Unengaging**: (1.00-10.00)
   - 0.00: Interesting, engaging pose/expression/clothing/character-setting
   - 1.00-3.00: Slightly boring but acceptable
   - 4.00-6.00: Bland, generic pose/expression/clothing/character-setting
   - 7.00-9.00: Very boring
   - 10.00: Extremely dull, no personality

d. **Poor face to image ratio**: coefficient (1.00-10.00)
   - 0.00: Perfect face size (quater to half of image height, details clearly visible)
   - 3.00-5.00: Slightly too large/small but acceptable
   - 5.00-7.00: Noticeably poor sizing
   - 7.00-9.00: Very poor face sizing
   - 10.00: Face too large (extreme close-up) or too small (barely visible). Be very strict with this.

## GENDER CLASSIFICATION
- **non-binary**: Furry characters, non-human entities, ambiguous presentations
- **male**: Clear male presentation
- **female**: Clear female presentation

## STYLE CLASSIFICATION
- **realistic**: photorealistic renders, real photography
- **stylized**: Anime, illustrations, cartoons, artistic renders, etc.

## CRITICAL: You must actually look at the image and score it. Most avatars will have SOME positive features and should receive non-zero coefficients. Do not be overly conservative.

## OUTPUT FORMAT
You MUST respond with valid JSON. For each scoring component, provide the coefficient used (0.00 ONLY if the feature is truly absent). Analyze the image thoroughly and provide realistic coefficients.

CRITICAL: Return ONLY the JSON object below. Do NOT include any markdown formatting, code blocks, or additional text. Your entire response should be just the JSON.

```json
{
  "gender": "[male|female|non-binary]",
  "style": "[stylized|realistic]",
  "reasoning": "[detailed explanation of what you see and why you chose each coefficient]",
  "scoring": {
    "stylized_content": true/false,
    "sexy_clothing_coeff": 0.00,
    "youth_appeal_coeff": 0.00,
    "artistic_design_coeff": 0.00,
    "lacks_appeal": true/false,
    "artifacts_coeff": 0.00,
    "dull_unengaging_coeff": 0.00,
    "poor_face_size_coeff": 0.00
  }
}
```"""

        # JSON Schema for structured output
        self.json_schema = {
            "name": "avatar_ranking",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "gender": {
                        "type": "string",
                        "enum": ["male", "female", "non-binary"],
                        "description": "Gender classification of the avatar"
                    },
                    "style": {
                        "type": "string", 
                        "enum": ["stylized", "realistic"],
                        "description": "Style classification of the avatar image"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Detailed explanation of analysis and scoring decisions"
                    },
                    "scoring": {
                        "type": "object",
                        "properties": {
                            "stylized_content": {
                                "type": "boolean",
                                "description": "Whether image has stylized content (bonus)"
                            },
                            "sexy_clothing_coeff": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 10.0,
                                "description": "Coefficient for sexy clothing (0.00 if not applicable)"
                            },
                            "youth_appeal_coeff": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 10.0,
                                "description": "Coefficient for youth appeal (0.00 if not applicable)"
                            },
                            "artistic_design_coeff": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 10.0,
                                "description": "Coefficient for artistic design (0.00 if not applicable)"
                            },
                            "lacks_appeal": {
                                "type": "boolean",
                                "description": "Whether image lacks appealing features (penalty)"
                            },
                            "artifacts_coeff": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 10.0,
                                "description": "Coefficient for artifacts/poor quality (0.00 if not applicable)"
                            },
                            "dull_unengaging_coeff": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 10.0,
                                "description": "Coefficient for dull/unengaging content (0.00 if not applicable)"
                            },
                            "poor_face_size_coeff": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 10.0,
                                "description": "Coefficient for poor face sizing (0.00 if not applicable)"
                            }
                        },
                        "required": ["stylized_content", "sexy_clothing_coeff", "youth_appeal_coeff", "artistic_design_coeff", "lacks_appeal", "artifacts_coeff", "dull_unengaging_coeff", "poor_face_size_coeff"],
                        "additionalProperties": False
                    }
                },
                "required": ["gender", "style", "reasoning", "scoring"],
                "additionalProperties": False
            }
        }

    def get_avatars(self, limit: int = 3) -> List[Tuple[str, str]]:
        """Get avatar_id and image_uri pairs from database"""
        try:
            conn = psycopg2.connect(self.postgres_url)
            cursor = conn.cursor()
            
            query = """
            SELECT avatar_id, image_uri 
            FROM avatars 
            WHERE is_public = true 
            AND image_uri IS NOT NULL 
            AND image_uri != ''
            ORDER BY create_time DESC 
            LIMIT %s
            """
            
            cursor.execute(query, (limit,))
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            logger.info(f"Retrieved {len(results)} avatars")
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

    def compute_score(self, scoring: Dict) -> Tuple[int, str]:
        """Compute final score from individual scoring components and return calculation breakdown"""
        base_score = 1000
        calculation_parts = ["1000 (base)"]
        
        # Additions
        if scoring.get('stylized_content', False):
            base_score += 100
            calculation_parts.append("+100 (stylized)")
        
        sexy_coeff = scoring.get('sexy_clothing_coeff', 0.0)
        if sexy_coeff > 0:
            points = int(100 * sexy_coeff)
            base_score += points
            calculation_parts.append(f"+100×{sexy_coeff:.2f}={points} (sexy clothing)")
        
        youth_coeff = scoring.get('youth_appeal_coeff', 0.0)
        if youth_coeff > 0:
            points = int(50 * youth_coeff)
            base_score += points
            calculation_parts.append(f"+50×{youth_coeff:.2f}={points} (youth appeal)")
        
        artistic_coeff = scoring.get('artistic_design_coeff', 0.0)
        if artistic_coeff > 0:
            points = int(50 * artistic_coeff)
            base_score += points
            calculation_parts.append(f"+50×{artistic_coeff:.2f}={points} (artistic design)")
        
        # Subtractions
        if scoring.get('lacks_appeal', False):
            base_score -= 100
            calculation_parts.append("-100 (lacks appeal)")
        
        artifacts_coeff = scoring.get('artifacts_coeff', 0.0)
        if artifacts_coeff > 0:
            points = int(200 * artifacts_coeff)
            base_score -= points
            calculation_parts.append(f"-200×{artifacts_coeff:.2f}={points} (artifacts)")
        
        dull_coeff = scoring.get('dull_unengaging_coeff', 0.0)
        if dull_coeff > 0:
            points = int(100 * dull_coeff)
            base_score -= points
            calculation_parts.append(f"-100×{dull_coeff:.2f}={points} (dull/unengaging)")
        
        face_coeff = scoring.get('poor_face_size_coeff', 0.0)
        if face_coeff > 0:
            points = int(400 * face_coeff)
            base_score -= points
            calculation_parts.append(f"-400×{face_coeff:.2f}={points} (poor face sizing)")
        
        # Cap the score between 0 and 5000
        final_score = max(0, min(5000, base_score))
        calculation_breakdown = " ".join(calculation_parts)
        
        if final_score != base_score:
            calculation_breakdown += f" = {base_score} (capped to {final_score})"
        else:
            calculation_breakdown += f" = {final_score}"
        
        return final_score, calculation_breakdown

    def analyze_image(self, image_uri: str, avatar_id: str) -> Optional[Dict]:
        """Analyze image via OpenRouter API using S3 presigned URL and save image locally"""
        try:
            # Check if files already exist locally
            avatar_dir = self.output_dir / avatar_id
            result_path = avatar_dir / "result.txt"
            
            # Look for existing image files with common extensions
            existing_image = None
            for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                image_path = avatar_dir / f"image{ext}"
                if image_path.exists():
                    existing_image = image_path
                    break
            
            # If image exists locally, remove any existing result.txt and rerun analysis
            if existing_image:
                logger.info(f"Using existing image for {avatar_id}: {existing_image}")
                if result_path.exists():
                    result_path.unlink()  # Remove existing result.txt
                    logger.info(f"Removed existing result.txt for {avatar_id}")
                
                with open(existing_image, 'rb') as f:
                    image_content = f.read()
            else:
                # Download image from S3
                logger.info(f"Downloading image for {avatar_id} from S3")
                presigned_url = self.get_presigned_url(image_uri)
                if not presigned_url:
                    logger.error(f"Failed to get presigned URL for {image_uri}")
                    return None
                
                response = requests.get(presigned_url, timeout=30)
                response.raise_for_status()
                image_content = response.content
                
                # Create avatar directory and save image locally
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
                
                # Save image to local file
                image_path = avatar_dir / f"image{file_extension}"
                with open(image_path, 'wb') as f:
                    f.write(image_content)
                logger.info(f"Saved image locally: {image_path}")
            
            # Encode image for API
            encoded_image = base64.b64encode(image_content).decode('utf-8')
            mime_type = "image/jpeg"  # default
            data_url = f"data:{mime_type};base64,{encoded_image}"
            
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
                                    {"type": "text", "text": "Analyze this avatar image thoroughly. Look at the clothing, attractiveness, artistic design, and overall appeal. Provide realistic coefficients - most images will have SOME positive features. Do not default everything to 0.0. Give detailed reasoning for each coefficient you assign."},
                                    {"type": "image_url", "image_url": {"url": data_url}}
                                ]
                            }
                        ],
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": self.json_schema
                        },
                        "max_tokens": 500,
                        "temperature": 0.1
                    }
                    
                    response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                    response.raise_for_status()
                    response_data = response.json()
                    break  # Success, exit retry loop
                except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
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
                logger.info(f"Raw API response content: {content[:200]}...")  # Log first 200 chars for debugging
                
                try:
                    # Try to parse the JSON directly first
                    api_results = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"Direct JSON parsing failed: {e}")
                    
                    # Try to extract JSON from the content if there's extra text
                    try:
                        # Look for JSON content between ```json and ``` or just find the JSON object
                        
                        # First try to find JSON block
                        json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            api_results = json.loads(json_str)
                            logger.info("Successfully extracted JSON from markdown block")
                        else:
                            # Try to find the first complete JSON object
                            json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', content, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(1)
                                api_results = json.loads(json_str)
                                logger.info("Successfully extracted JSON object from content")
                            else:
                                logger.error(f"Could not extract valid JSON from content: {content}")
                                return None
                    except Exception as e2:
                        logger.error(f"Failed to extract JSON from content: {e2}")
                        logger.error(f"Full content: {content}")
                        return None
                
                # Compute score locally from coefficients
                computed_score, calculation_breakdown = self.compute_score(api_results['scoring'])
                
                # Return combined results
                return {
                    'gender': api_results['gender'],
                    'style': api_results['style'],
                    'reasoning': api_results['reasoning'],
                    'score': computed_score,
                    'calculation_breakdown': calculation_breakdown,
                    'scoring': api_results['scoring']
                }
            else:
                logger.error("No choices in API response")
                return None
                
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return None

    def save_local(self, avatar_id: str, results: Dict) -> bool:
        """Save results to local file"""
        try:
            avatar_dir = self.output_dir / avatar_id
            avatar_dir.mkdir(parents=True, exist_ok=True)
            
            results_path = avatar_dir / "result.txt"
            timestamp = datetime.now().isoformat()
            
            content = f"# Avatar Analysis Results\n"
            content += f"# Generated: {timestamp}\n"
            content += f"# Avatar ID: {avatar_id}\n\n"
            
            # Save core fields
            content += f"gender: {results['gender']}\n"
            content += f"style: {results['style']}\n"
            content += f"score: {results['score']}\n\n"
            
            # Save detailed reasoning
            content += f"reasoning: {results['reasoning']}\n\n"
            
            # Save calculation breakdown
            content += f"# Score Calculation:\n"
            content += f"score = {results['calculation_breakdown']}\n\n"
            
            # Save scoring breakdown
            content += f"# Scoring Breakdown:\n"
            scoring = results.get('scoring', {})
            content += f"stylized_content: {scoring.get('stylized_content', False)}\n"
            content += f"sexy_clothing_coeff: {scoring.get('sexy_clothing_coeff', 0.0)}\n"
            content += f"youth_appeal_coeff: {scoring.get('youth_appeal_coeff', 0.0)}\n"
            content += f"artistic_design_coeff: {scoring.get('artistic_design_coeff', 0.0)}\n"
            content += f"lacks_appeal: {scoring.get('lacks_appeal', False)}\n"
            content += f"artifacts_coeff: {scoring.get('artifacts_coeff', 0.0)}\n"
            content += f"dull_unengaging_coeff: {scoring.get('dull_unengaging_coeff', 0.0)}\n"
            content += f"poor_face_size_coeff: {scoring.get('poor_face_size_coeff', 0.0)}\n"
                
            with open(results_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            logger.info(f"Saved results for {avatar_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save results for {avatar_id}: {e}")
            return False

    def update_database(self, avatar_id: str, results: Dict) -> bool:
        """Update database with analysis results"""
        try:
            conn = psycopg2.connect(self.postgres_url)
            cursor = conn.cursor()
            
            update_query = """
            UPDATE avatars 
            SET v1_score = %s, gender = %s, style = %s
            WHERE avatar_id = %s
            """
            
            cursor.execute(update_query, (
                results['score'],
                results['gender'], 
                results['style'],
                avatar_id
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Updated database for {avatar_id}")
            return True
            
        except Exception as e:
            logger.error(f"Database update failed for {avatar_id}: {e}")
            return False

    def run(self, max_avatars: int = 3, write_to_db: bool = False):
        """Main execution - analyze avatars and optionally write to database"""
        logger.info(f"Starting analysis for {max_avatars} avatars (DB write: {write_to_db})")
        
        avatars = self.get_avatars(max_avatars)
        if not avatars:
            logger.error("No avatars found")
            return
            
        success_count = 0
        for avatar_id, image_uri in avatars:
            try:
                logger.info(f"Processing {avatar_id}")
                
                # Analyze image
                results = self.analyze_image(image_uri, avatar_id)
                if not results:
                    logger.error(f"Analysis failed for {avatar_id}")
                    continue
                
                # Save locally
                if not self.save_local(avatar_id, results):
                    logger.error(f"Local save failed for {avatar_id}")
                    continue
                
                # Optionally update database
                if write_to_db:
                    if not self.update_database(avatar_id, results):
                        logger.error(f"Database update failed for {avatar_id}")
                        continue
                
                success_count += 1
                logger.info(f"Successfully processed {avatar_id}: {results}")
                
            except Exception as e:
                logger.error(f"Error processing {avatar_id}: {e}")
                
        logger.info(f"Complete: {success_count}/{len(avatars)} processed")


def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description='Avatar Content Ranking System')
    parser.add_argument('--max-avatars', type=int, default=3, 
                       help='Maximum number of avatars to process (default: 3)')
    parser.add_argument('--write-db', action='store_true', 
                       help='Write results back to database (default: local only)')
    
    args = parser.parse_args()
    
    try:
        ranker = SimpleAvatarRanker()
        ranker.run(max_avatars=args.max_avatars, write_to_db=args.write_db)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
