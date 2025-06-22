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

class CharacterCardRanker:
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
        self.gemini_model = "google/gemini-2.5-pro-preview"  # For structured output
        
        # System prompt for character card ranking
        self.system_prompt = """Instruction:

You will act like a character-card content ranker for an ai roleplay, romance-simulation platform.
You will be provided with a character card metadata, that includes the character name, bio, prompt, any related text, and the image. You should provide your evaluation on question. The target audience is very young ( 18 to 22 ), driven by lonliness, desire, and curiosity. You should act like their age and judge the card.

For each question, aside from your answer, provide a simple explanation, on what do you see in the image / text and why you picked your answer

Your response format should be

[ heading ]

Question number

Output response:

Output reasons

[ newline]

Score creteria:

# **Text plus Image ( character-card ). Reference name, prompt, bio, and the given image**

## **Emotional Response and Quality**

### Q1.1 emotion_tag

Does the character and setting offer a potentially engaging, sexually desirable, parody-related, wholesome, or iconic roleplay experience?
A typical user's emotional response may include, but is not limited to, the following:

- fantasizing (e.g., Victoria England, medieval, futuristic sci-fi, etc.)
- wholesome (warm, helpful, positive, etc.)
- desirable (e.g., an exciting encounter with a beautiful potential partner, a Japanese yandere-type experience, etc.)
- parody (funny, or with a parody-like contrast)
- iconic (references a famous book, movie, or anime)
- or any single word or phrase that best describes it. If not applicable, use words like 'dull,' 'plain,' 'boring,' or similar.

**Output:** one distinct word or phrase

### Q1.2 text_quality

Measure how interesting and engaging the given prompt and image are, based on the emotional response provided.
On a scale of 1 to 5:

- 1: poor quality. The character and its plot are not only plain but also of bad quality, containing a nonsensical plot. Content should be avoided.
- 2: baseline acceptable. The plot is plain and may not be very engaging for a young audience. It can be displayed but should be assigned low priority.
- 3: baseline good, above average interesting.
- 4: the character text is very interesting overall, and is at least **iconic** or **desirable**.
- 5: the character and plot are very interesting, matching more than two emotional responses (e.g., both wholesome, fantasizing, and iconic, relating to a famous anime or movie aired in 2017).

You are required to be very strict about this criteria.

**Output:** a number between 1 to 5

## **Image, Plot & Character Setting**

### Q1.3 age

Describe the age category of the character.

**Output:** one of the three: young, middle-aged, old

### Q1.4 ethnicity

Describe the regional ethnicity:

- white
- hispanic
- middle-eastern
- south-asian (india, pakistan, sri lanka, etc.)
- southeast-asian (filipino, vietnam, indonesia, etc.)
- east-asian (china, japan, korea)
- native (american native, pacific islander)
- black
- none (alien character, furry, or difficult-to-discern stylized character)

**Output:** one of the specific race or ethnicity of the character

### Q1.5 character_card_tag

Describe the character and/or background setting.

Background description, including but not limited to general terms like: steampunk, vintage, medieval, Victorian England, magical (or any western/eastern typical setting), or simply modern places and encounters (e.g., school, office, bar encounter, beach, pool, shopping, gym, conference).

Character type: yandere, goth, mommy, friend, daddy, dominant, coach, teacher, mysterious, kawaii, senpai, cute, girl-next-door, etc. You are allowed to be creative on wording.

**Output:** you may use at least one, at most three distinct words or phrases to describe the character and its setting reflected in plot and image

## **Image and Character Coherence**

### Q1.6 image_coherence

Character image setting coherence: Do the character's outfit and look match the image background and ambient setting?

**Output: one of: average, good, excellent**

### Q1.7 character_card_coherence

Character card setting coherence: Specifically does the prompt match what is shown in the image?

**Output: one of: average, good, excellent**

# **Image Only Instruction**

For these questions regarding characters in the image: if there are multiple characters, pick the most dominant one. If there is no character, output the lowest score or 'no,' depending on the question's choices.

## **Image Style and Technical Quality**

### Q2.1 image_style

Pick one general word that best summarizes the image's style. If the image is stylized, be very specific about its style. If it is photorealistic, whether it is photography or realistic 3D rendering ), simply mark as realistic.

Common styles may include: painting, pencil sketch, 3D catoon, anime, etc.

**Output:** one word or phrase

### Q2.2 image_quality

Is the image overall good quality? Is there excessive blurring or low resolution, causing details to be obscured or artifacts to appear?

**Output: one of: average, good, excellent**

### Q2.3 is_single_character

Does the image contain precisely one character? Images with more than one character should be marked as 'multiple.' Images that are landscape-only, text-only, graph-only, object-only, or completely empty should be marked as 'none.'

**Output:** one of: yes, multiple, none

## **Appearance Facial Rating**

### Q3.1 face

Measure facial appeal to a young audience: Is it good-looking or desirable to a young audience, or does the character's appearance resemble any popular characters from video games or anime?

- average: average looking. Also include those with inconsistent looking (e.g., appears too artificial, missing facial details, or face partially/fully covered)
- good: above average appealing
- excellent: very appealing and/or features familiar faces to the audience

**Output:** one of: average, good, excellent

### Q3.2 hair

Does the character have appealing hair? If the character does not have hair, output 'average'. If the hair is not appealing, or does not fit the character, output 'average'.

**Output:** one of: average, good, excellent

### Q3.3 is_neutral

Does the character show a mostly neutral, neutral-positive, or neutral-negative facial expression (i.e., not a strong or extreme expression)?

neutral: the character doesn't show much emotion or expression

medium: the character may show emotion, but the expression doesn't deviate much from neutral

strong: the character shows strong emotion and expression ( big laugh, sad, frown, etc )

**Output:** one of: neutral, medium, strong

## **Appearance Body Rating**

### Q3.4 body

Does the character display desirable body features? Body appeal: Does the character exhibit dominant features that are considered desirable (e.g., larger than average chest area, desirable legs or thighs for females, muscular body for males)? If the body feature is not visible, output 'average'. Be strict about what makes good and desiring body shape.

**Output:** one of: average, good, excellent

### Q3.5 headpose

Is the character mostly facing front with a mostly upright headpose? Slight rotation should be marked as tilted . More rotation ( when the oritentaion is more than half way to the other side ) would be marked as 'offset'.

neutral: the character head and face is facing front, with neglible offset.

tilted : the character face and head is a little tilted, by not much

offet: more tilted, needs some correction.

**Output:** one of: neutral, tilted, offset

### Q3.6 gesture

Is the character showing a fantasized or desirable body pose or gesture? Examples include, but are not limited to: seducing (e.g., on one knee), funny gestures, etc. Only really interesting or seducive gesture shall be marked as 'excellent'.

**Output:** one of: average, good, excellent

## **Appearance Outfit, Cloth, and Decoration**

### Q3.7 outerwear_quality

Is the outerwear good-looking? If not applicable, output 'average'. Be strict.

**Output:** one of: average, good, excellent

### Q3.8 outerwear_desirability

Does the outerwear improve the character's desirability for virtual dating and roleplay? If not applicable, output 'average'. If they wear ordinary cloth, also mark as average. 

if the outerwear is interesting or mystic or reflects the setting well, return 'good'.

If the outerwear explicitly show sextually-attractiveness and seducing, then return 'excellent'.

**Output:** one of: average, good, excellent

### Q3.9 apparels_quality

Are the apparels good-looking or desirable? If not applicable, output 'no.'

**Output:** one of: average, good, excellent"""

        # JSON Schema for structured character card analysis
        self.character_analysis_schema = {
            "name": "character_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "emotion_tag": {
                        "type": "string",
                        "description": "One distinct word or phrase describing emotional response"
                    },
                    "text_quality": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Text quality rating from 1-5"
                    },
                    "age": {
                        "type": "string",
                        "enum": ["young", "middle-aged", "old"],
                        "description": "Age category of the character"
                    },
                    "ethnicity": {
                        "type": "string",
                        "enum": ["white", "hispanic", "middle-eastern", "south-asian", "southeast-asian", "east-asian", "native", "black", "none"],
                        "description": "Regional ethnicity of the character"
                    },
                    "character_card_tag": {
                        "type": "string",
                        "description": "1-3 words or phrases describing character and setting"
                    },
                    "image_coherence": {
                        "type": "string",
                        "enum": ["average", "good", "excellent"],
                        "description": "Whether outfit and look match image background"
                    },
                    "character_card_coherence": {
                        "type": "string",
                        "enum": ["yes", "no"],
                        "description": "Whether prompt matches what is shown in image"
                    },
                    "image_style": {
                        "type": "string",
                        "description": "One word or phrase describing image style."
                    },
                    "style_is_realistic": {
                        "type": "string",
                        "enum": ["yes", "no"],
                        "description": "Whether image style is realistic"
                    },
                    "image_quality": {
                        "type": "string",
                        "enum": ["average", "good", "excellent"],
                        "description": "Whether image is overall good quality"
                    },
                    "is_single_character": {
                        "type": "string",
                        "enum": ["yes", "multiple", "none"],
                        "description": "Whether image contains precisely one character"
                    },
                    "face": {
                        "type": "string",
                        "enum": ["average", "good", "excellent"],
                        "description": "Facial appeal rating to young audience"
                    },
                    "hair": {
                        "type": "string",
                        "enum": ["yes", "no"],
                        "description": "Whether character has appealing hair"
                    },
                    "is_neutral": {
                        "type": "string",
                        "enum": ["neutral", "medium", "strong"],
                        "description": "Character facial expression level"
                    },
                    "body": {
                        "type": "string",
                        "enum": ["average", "good", "excellent"],
                        "description": "Whether character displays desirable body features"
                    },
                    "headpose": {
                        "type": "string",
                        "enum": ["neutral", "tilted", "offset"],
                        "description": "Character head pose and orientation"
                    },
                    "gesture": {
                        "type": "string",
                        "enum": ["average", "good", "excellent"],
                        "description": "Whether character shows fantasized or desirable pose"
                    },
                    "outerwear_quality": {
                        "type": "string",
                        "enum": ["average", "good", "excellent"],
                        "description": "Whether outerwear is good-looking"
                    },
                    "outerwear_desirability": {
                        "type": "string",
                        "enum": ["average", "good", "excellent"],
                        "description": "Whether outerwear improves character desirability"
                    },
                    "apparels_quality": {
                        "type": "string",
                        "enum": ["average", "good", "excellent"],
                        "description": "Whether apparels are good-looking or desirable"
                    },
                },
                "required": [
                    "emotion_tag", "text_quality", "age", "ethnicity",
                    "character_card_tag", "image_coherence", "character_card_coherence",
                    "image_style", "style_is_realistic", "image_quality", "is_single_character",
                    "face", "hair", "is_neutral", "body", "headpose",
                    "gesture", "outerwear_quality", "outerwear_desirability",
                    "apparels_quality",
                ],
                "additionalProperties": False
            }
        }

        # Thread-safe counter for progress tracking
        self._lock = threading.Lock()
        self._processed_count = 0
        self._total_count = 0

    def get_avatar_data(self, avatar_id: str) -> Optional[Dict]:
        """Get avatar data from database including avatar_name, tagline, prompt, and image_uri"""
        try:
            conn = psycopg2.connect(self.postgres_url)
            cursor = conn.cursor()
            
            query = """
            SELECT avatar_id, avatar_name, agent_bio, prompt, image_uri, gender
            FROM avatars 
            WHERE avatar_id = %s
            """
            
            cursor.execute(query, (avatar_id,))
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                return {
                    'avatar_id': result[0],
                    'name': result[1] or '',
                    'tagline': result[2] or '',
                    'description': result[3] or '',
                    'image_uri': result[4] or '',
                    'gender': result[5] or ''
                }
            else:
                logger.error(f"Avatar {avatar_id} not found in database")
                return None
                
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return None

    def get_avatars(self, overwrite: bool = False) -> List[Tuple[str, str]]:
        """Get avatar_id and image_uri pairs from database for batch processing"""
        try:
            conn = psycopg2.connect(self.postgres_url)
            cursor = conn.cursor()
            
            if overwrite:
                # Process ALL avatars regardless of whether they're already processed
                query = """
                SELECT avatar_id, image_uri 
                FROM avatars 
                WHERE is_public = true 
                AND image_uri IS NOT NULL 
                AND image_uri != ''
                ORDER BY create_time DESC
                """
                cursor.execute(query)
                logger.info("Fetching ALL avatars (overwrite mode)")
            else:
                # Process only avatars that exist in avatars but NOT in avatar_discovery
                query = """
                SELECT a.avatar_id, a.image_uri 
                FROM avatars a
                LEFT JOIN avatar_discovery ad ON a.avatar_id = ad.avatar_id
                WHERE a.is_public = true 
                AND a.image_uri IS NOT NULL 
                AND a.image_uri != ''
                AND ad.avatar_id IS NULL
                ORDER BY a.create_time DESC
                """
                cursor.execute(query)
                logger.info("Fetching unprocessed avatars only (default mode)")
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            mode = "ALL" if overwrite else "unprocessed"
            logger.info(f"Retrieved {len(results)} {mode} avatars for batch processing")
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

    def download_and_save_image(self, image_uri: str, avatar_id: str) -> Optional[bytes]:
        """Download image from S3 and save locally"""
        try:
            # Create avatar directory
            avatar_dir = self.output_dir / avatar_id
            avatar_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if image already exists locally
            for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                image_path = avatar_dir / f"image{ext}"
                if image_path.exists():
                    logger.info(f"Using existing image for {avatar_id}: {image_path}")
                    with open(image_path, 'rb') as f:
                        return f.read()
            
            # Download from S3
            logger.info(f"Downloading image for {avatar_id} from S3")
            presigned_url = self.get_presigned_url(image_uri)
            if not presigned_url:
                return None
            
            response = requests.get(presigned_url, timeout=30)
            response.raise_for_status()
            image_content = response.content
            
            # Determine file extension
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
            logger.error(f"Failed to download image: {e}")
            return None

    def analyze_character_card(self, avatar_data: Dict, image_content: bytes) -> Optional[str]:
        """Send character card data to OpenRouter API for analysis"""
        try:
            # Prepare character card text
            card_text = f"""Name: {avatar_data['name']}
Gender: {avatar_data['gender']}
Tagline: {avatar_data['tagline']}
Description: {avatar_data['description']}"""
            
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
                                    {"type": "text", "text": f"Please analyze this character card:\n\n{card_text}"},
                                    {"type": "image_url", "image_url": {"url": data_url}}
                                ]
                            }
                        ],
                        "max_tokens": 5000,
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
                return response_data['choices'][0]['message']['content']
            else:
                logger.error("No choices in API response")
                return None
                
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return None

    def save_results(self, avatar_id: str, results: str, avatar_data: Dict) -> bool:
        """Save analysis results to local file"""
        try:
            avatar_dir = self.output_dir / avatar_id
            avatar_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = avatar_dir / "out.txt"
            timestamp = datetime.now().isoformat()
            
            content = f"# Character Card Analysis Results\n"
            content += f"# Generated: {timestamp}\n"
            content += f"# Avatar ID: {avatar_id}\n\n"
            content += f"# Character Card Input:\n"
            content += f"Name: {avatar_data['name']}\n"
            content += f"Gender: {avatar_data['gender']}\n"
            content += f"Tagline: {avatar_data['tagline']}\n"
            content += f"Description: {avatar_data['description']}\n\n"
            content += f"# Analysis Results:\n\n"
            content += results
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            logger.info(f"Saved results for {avatar_id} to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save results for {avatar_id}: {e}")
            return False

    def update_database(self, avatar_id: str, results: str) -> bool:
        """Update database with analysis results (placeholder for future implementation)"""
        # For now, just log that we would update the database
        logger.info(f"Would update database for {avatar_id} (not implemented yet)")
        return True

    def extract_structured_analysis(self, initial_analysis: str, avatar_data: Dict, image_content: bytes) -> Optional[Dict]:
        """Extract structured analysis using Gemini Flash with enforced JSON schema"""
        try:
            # Prepare character card text
            card_text = f"""Name: {avatar_data['name']}
Gender: {avatar_data['gender']}
Tagline: {avatar_data['tagline']}
Description: {avatar_data['description']}"""
            
            # Encode image for API
            encoded_image = base64.b64encode(image_content).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{encoded_image}"
            
            # Create a structured prompt for Gemini Flash
            structured_prompt = f"""Based on the following character card analysis, extract structured answers for each question.

Original Analysis:
{initial_analysis}

Please analyze the character card and image to answer ALL questions with the exact required format and constraints. For questions with enumerated options, you MUST choose from the provided options only."""

            # API request with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    headers = {
                        "Authorization": f"Bearer {self.openrouter_api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    payload = {
                        "model": self.gemini_model,
                        "messages": [
                            {
                                "role": "system", 
                                "content": [
                                    {"type": "text", "text": structured_prompt},
                                ]
                            }
                        ],
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": self.character_analysis_schema
                        },
                        "max_tokens": 5000,
                        "temperature": 0.1
                    }
                    
                    response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                    response.raise_for_status()
                    response_data = response.json()
                    break  # Success, exit retry loop
                    
                except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Structured analysis API request failed (attempt {attempt + 1}/{max_retries}): {e}")
                        logger.info(f"Retrying in {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Structured analysis API request failed after {max_retries} attempts: {e}")
                        return None
                except Exception as e:
                    logger.error(f"Unexpected structured analysis API error: {e}")
                    return None
            
            if 'choices' in response_data and len(response_data['choices']) > 0:
                content = response_data['choices'][0]['message']['content']
                try:
                    # Parse the structured JSON response
                    structured_results = json.loads(content)
                    logger.info("Successfully parsed structured analysis results")
                    return structured_results
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse structured analysis JSON: {e}")
                    logger.error(f"Content: {content}")
                    return None
            else:
                logger.error("No choices in structured analysis API response")
                return None
                
        except Exception as e:
            logger.error(f"Structured analysis failed: {e}")
            return None

    def save_structured_results(self, avatar_id: str, structured_results: Dict, avatar_data: Dict, initial_analysis: str) -> bool:
        """Save structured analysis results to JSON file"""
        try:
            avatar_dir = self.output_dir / avatar_id
            avatar_dir.mkdir(parents=True, exist_ok=True)
            
            json_output_path = avatar_dir / "analysis.json"
            timestamp = datetime.now().isoformat()
            
            # Create comprehensive JSON output
            complete_results = {
                "metadata": {
                    "avatar_id": avatar_id,
                    "generated": timestamp,
                    "character_card": {
                        "name": avatar_data['name'],
                        "gender": avatar_data['gender'],
                        "tagline": avatar_data['tagline'],
                        "description": avatar_data['description']
                    }
                },
                "initial_analysis": initial_analysis,
                "structured_analysis": structured_results
            }
            
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(complete_results, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved structured results for {avatar_id} to {json_output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save structured results for {avatar_id}: {e}")
            return False

    def map_quality_to_int(self, value: str) -> int:
        """Map quality values (average/good/excellent) to integers (1/2/3)"""
        mapping = {
            "average": 1,
            "good": 2,
            "excellent": 3
        }
        return mapping.get(value.lower(), 1)  # Default to 1 if not found

    def map_boolean(self, value: str) -> bool:
        """Map yes/no values to boolean"""
        return value.lower() == "yes"

    def update_database_structured(self, avatar_id: str, structured_results: Dict) -> bool:
        """Update database with structured analysis results"""
        try:
            conn = psycopg2.connect(self.postgres_url)
            cursor = conn.cursor()
            
            # Check if record exists
            check_query = "SELECT avatar_id FROM avatar_discovery WHERE avatar_id = %s"
            cursor.execute(check_query, (avatar_id,))
            exists = cursor.fetchone() is not None
            
            # Map JSON results to SQL schema
            mapped_data = {
                'avatar_id': avatar_id,
                'emotion_tag': structured_results.get('emotion_tag', ''),
                'text_quality': structured_results.get('text_quality', 1),
                'age': structured_results.get('age', 'young'),
                'ethnicity': structured_results.get('ethnicity', 'none'),
                'character_card_tag': structured_results.get('character_card_tag', ''),
                'image_coherence': self.map_quality_to_int(structured_results.get('image_coherence', 'average')),
                'character_card_coherence': self.map_boolean(structured_results.get('character_card_coherence', 'no')),
                'img_style': structured_results.get('image_style', ''),
                'style_is_realistic': self.map_boolean(structured_results.get('style_is_realistic', 'no')),
                'image_quality': self.map_quality_to_int(structured_results.get('image_quality', 'average')),
                'is_single_character': structured_results.get('is_single_character', 'none'),
                'face': self.map_quality_to_int(structured_results.get('face', 'average')),
                'hair': self.map_boolean(structured_results.get('hair', 'no')),
                'is_neutral': structured_results.get('is_neutral', 'neutral'),
                'body': self.map_quality_to_int(structured_results.get('body', 'average')),
                'headpose': structured_results.get('headpose', 'neutral'),
                'gesture': self.map_quality_to_int(structured_results.get('gesture', 'average')),
                'outerwear_quality': self.map_quality_to_int(structured_results.get('outerwear_quality', 'average')),
                'outerwear_desirability': self.map_quality_to_int(structured_results.get('outerwear_desirability', 'average')),
                'apparels_quality': self.map_quality_to_int(structured_results.get('apparels_quality', 'average'))
            }
            
            if exists:
                # Update existing record
                update_query = """
                UPDATE avatar_discovery SET
                    emotion_tag = %s,
                    text_quality = %s,
                    age = %s,
                    ethnicity = %s,
                    character_card_tag = %s,
                    image_coherence = %s,
                    character_card_coherence = %s,
                    img_style = %s,
                    style_is_realistic = %s,
                    image_quality = %s,
                    is_single_character = %s,
                    face = %s,
                    hair = %s,
                    is_neutral = %s,
                    body = %s,
                    headpose = %s,
                    gesture = %s,
                    outerwear_quality = %s,
                    outerwear_desirability = %s,
                    apparels_quality = %s
                WHERE avatar_id = %s
                """
                
                cursor.execute(update_query, (
                    mapped_data['emotion_tag'],
                    mapped_data['text_quality'],
                    mapped_data['age'],
                    mapped_data['ethnicity'],
                    mapped_data['character_card_tag'],
                    mapped_data['image_coherence'],
                    mapped_data['character_card_coherence'],
                    mapped_data['img_style'],
                    mapped_data['style_is_realistic'],
                    mapped_data['image_quality'],
                    mapped_data['is_single_character'],
                    mapped_data['face'],
                    mapped_data['hair'],
                    mapped_data['is_neutral'],
                    mapped_data['body'],
                    mapped_data['headpose'],
                    mapped_data['gesture'],
                    mapped_data['outerwear_quality'],
                    mapped_data['outerwear_desirability'],
                    mapped_data['apparels_quality'],
                    avatar_id
                ))
                logger.info(f"Updated existing record for avatar {avatar_id}")
            else:
                # Insert new record
                insert_query = """
                INSERT INTO avatar_discovery (
                    avatar_id, emotion_tag, text_quality, age, ethnicity, character_card_tag,
                    image_coherence, character_card_coherence, img_style, style_is_realistic,
                    image_quality, is_single_character, face, hair, is_neutral, body,
                    headpose, gesture, outerwear_quality, outerwear_desirability, apparels_quality
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """
                
                cursor.execute(insert_query, (
                    mapped_data['avatar_id'],
                    mapped_data['emotion_tag'],
                    mapped_data['text_quality'],
                    mapped_data['age'],
                    mapped_data['ethnicity'],
                    mapped_data['character_card_tag'],
                    mapped_data['image_coherence'],
                    mapped_data['character_card_coherence'],
                    mapped_data['img_style'],
                    mapped_data['style_is_realistic'],
                    mapped_data['image_quality'],
                    mapped_data['is_single_character'],
                    mapped_data['face'],
                    mapped_data['hair'],
                    mapped_data['is_neutral'],
                    mapped_data['body'],
                    mapped_data['headpose'],
                    mapped_data['gesture'],
                    mapped_data['outerwear_quality'],
                    mapped_data['outerwear_desirability'],
                    mapped_data['apparels_quality']
                ))
                logger.info(f"Inserted new record for avatar {avatar_id}")
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Successfully updated database for avatar {avatar_id}")
            return True
            
        except Exception as e:
            logger.error(f"Database update failed for {avatar_id}: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
            return False

    def process_avatar(self, avatar_id: str) -> bool:
        """Process a single avatar through the complete pipeline"""
        try:
            print(f"\n🔍 Processing avatar: {avatar_id}")
            
            # Get avatar data from database
            avatar_data = self.get_avatar_data(avatar_id)
            if not avatar_data:
                print(f"❌ Avatar {avatar_id} not found in database")
                return False
            
            print(f"📝 Found avatar: {avatar_data['name']}")
            
            # Download and save image
            if not avatar_data['image_uri']:
                print(f"❌ No image URI found for avatar {avatar_id}")
                return False
                
            image_content = self.download_and_save_image(avatar_data['image_uri'], avatar_id)
            if not image_content:
                print(f"❌ Failed to download image for avatar {avatar_id}")
                return False
            
            print(f"🖼️  Image downloaded and saved")
            
            # First analysis: Character card analysis
            print(f"🤖 Analyzing character card...")
            initial_results = self.analyze_character_card(avatar_data, image_content)
            if not initial_results:
                print(f"❌ Initial analysis failed for avatar {avatar_id}")
                return False
            
            # Print initial results
            print(f"\n📊 Initial Analysis Results:")
            print("=" * 50)
            print(initial_results)
            print("=" * 50)
            
            # Save initial results locally
            if self.save_results(avatar_id, initial_results, avatar_data):
                print(f"💾 Initial results saved to output/{avatar_id}/out.txt")
            else:
                print(f"❌ Failed to save initial results")
            
            # Second analysis: Structured extraction using Gemini Flash
            print(f"🔬 Extracting structured analysis...")
            structured_results = self.extract_structured_analysis(initial_results, avatar_data, image_content)
            if not structured_results:
                print(f"❌ Structured analysis failed for avatar {avatar_id}")
                return False
            
            # Print structured results
            print(f"\n📋 Structured Analysis Results:")
            print("=" * 50)
            for key, value in structured_results.items():
                print(f"{key}: {value}")
            print("=" * 50)
            
            # Save structured results as JSON
            if self.save_structured_results(avatar_id, structured_results, avatar_data, initial_results):
                print(f"💾 Structured results saved to output/{avatar_id}/analysis.json")
            else:
                print(f"❌ Failed to save structured results")
            
            # Update database with structured analysis results
            if self.update_database_structured(avatar_id, structured_results):
                print(f"🗄️  Successfully updated database for avatar {avatar_id}")
            else:
                print(f"❌ Failed to update database for avatar {avatar_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {avatar_id}: {e}")
            print(f"❌ Error processing avatar {avatar_id}: {e}")
            return False

    def process_single_avatar(self, avatar_data_tuple: Tuple[str, str]) -> Tuple[str, bool]:
        """Thread-safe method to process a single avatar for batch processing"""
        avatar_id, image_uri = avatar_data_tuple
        
        try:
            # Get full avatar data
            avatar_data = self.get_avatar_data(avatar_id)
            if not avatar_data:
                logger.error(f"Avatar {avatar_id} not found in database")
                return avatar_id, False
            
            # Use the image_uri from the batch query
            avatar_data['image_uri'] = image_uri
            
            # Download and save image
            if not avatar_data['image_uri']:
                logger.error(f"No image URI found for avatar {avatar_id}")
                return avatar_id, False
                
            image_content = self.download_and_save_image(avatar_data['image_uri'], avatar_id)
            if not image_content:
                logger.error(f"Failed to download image for avatar {avatar_id}")
                return avatar_id, False
            
            logger.info(f"Image downloaded for {avatar_id}")
            
            # First analysis: Character card analysis
            logger.info(f"Analyzing character card for {avatar_id}")
            initial_results = self.analyze_character_card(avatar_data, image_content)
            if not initial_results:
                logger.error(f"Initial analysis failed for avatar {avatar_id}")
                return avatar_id, False
            
            # Save initial results locally
            if not self.save_results(avatar_id, initial_results, avatar_data):
                logger.error(f"Failed to save initial results for {avatar_id}")
                return avatar_id, False
            
            # Second analysis: Structured extraction using Gemini Flash
            logger.info(f"Extracting structured analysis for {avatar_id}")
            structured_results = self.extract_structured_analysis(initial_results, avatar_data, image_content)
            if not structured_results:
                logger.error(f"Structured analysis failed for avatar {avatar_id}")
                return avatar_id, False
            
            # Save structured results as JSON
            if not self.save_structured_results(avatar_id, structured_results, avatar_data, initial_results):
                logger.error(f"Failed to save structured results for {avatar_id}")
                return avatar_id, False
            
            # Update database with structured analysis results
            if not self.update_database_structured(avatar_id, structured_results):
                logger.error(f"Failed to update database for {avatar_id}")
                return avatar_id, False
            
            logger.info(f"Successfully processed {avatar_id}")
            return avatar_id, True
            
        except Exception as e:
            logger.error(f"Error processing {avatar_id}: {e}")
            return avatar_id, False

    def _update_progress(self, avatar_id: str, success: bool):
        """Thread-safe progress update"""
        with self._lock:
            self._processed_count += 1
            status = "✅" if success else "❌"
            print(f"{status} [{self._processed_count}/{self._total_count}] {avatar_id}")

    def run_batch(self, overwrite: bool = False, max_workers: int = 16):
        """Run batch processing mode with multithreading"""
        mode_desc = "ALL avatars (overwrite mode)" if overwrite else "unprocessed avatars only"
        logger.info(f"Starting batch analysis for {mode_desc} with {max_workers} workers")
        print(f"🎭 Character Card Ranker v2 - Batch Mode (Multithreaded)")
        print("=" * 60)
        print(f"Mode: {'🔄 Overwrite - reprocessing ALL avatars' if overwrite else '🆕 Default - processing unprocessed avatars only'}")
        print(f"🧵 Using {max_workers} worker threads")
        
        avatars = self.get_avatars(overwrite)
        if not avatars:
            logger.error("No avatars found")
            if overwrite:
                print("❌ No avatars found in database")
            else:
                print("✅ No unprocessed avatars found - all avatars have been processed!")
                print("💡 Use --overwrite to reprocess all avatars")
            return
        
        print(f"📊 Found {len(avatars)} {'total' if overwrite else 'unprocessed'} avatars to process")
        
        # Reset counters
        with self._lock:
            self._processed_count = 0
            self._total_count = len(avatars)
        
        successful_avatars = []
        failed_avatars = []
        
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
                    processed_avatar_id, success = future.result()
                    self._update_progress(processed_avatar_id, success)
                    
                    if success:
                        successful_avatars.append(processed_avatar_id)
                    else:
                        failed_avatars.append(processed_avatar_id)
                        
                except Exception as e:
                    logger.error(f"Exception in thread for {avatar_id}: {e}")
                    failed_avatars.append(avatar_id)
                    self._update_progress(avatar_id, False)
        
        # Final summary
        success_count = len(successful_avatars)
        total_count = len(avatars)
        
        print(f"\n🎉 Batch processing complete!")
        print(f"📈 Successfully processed: {success_count}/{total_count} avatars")
        
        if failed_avatars:
            print(f"❌ Failed avatars: {len(failed_avatars)}")
            print(f"   {', '.join(failed_avatars[:10])}{' ...' if len(failed_avatars) > 10 else ''}")
        
        logger.info(f"Batch complete: {success_count}/{total_count} processed successfully")

    def run_interactive(self):
        """Run interactive CLI mode"""
        print("🎭 Character Card Ranker v2")
        print("=" * 40)
        print("Enter avatar IDs to analyze (type 'quit' to exit)")
        
        while True:
            try:
                avatar_id = input("\n🆔 Enter avatar ID: ").strip()
                
                if avatar_id.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not avatar_id:
                    print("⚠️  Please enter a valid avatar ID")
                    continue
                
                success = self.process_avatar(avatar_id)
                if success:
                    print(f"✅ Successfully processed {avatar_id}")
                else:
                    print(f"❌ Failed to process {avatar_id}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")


def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description='Character Card Ranking System v2')
    parser.add_argument('--batch', action='store_true', 
                       help='Run in batch mode (process avatars from database automatically)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Reprocess ALL avatars (default: only process unprocessed avatars)')
    parser.add_argument('--max-workers', type=int, default=16,
                       help='Number of worker threads for batch processing (default: 16)')
    
    args = parser.parse_args()
    
    try:
        ranker = CharacterCardRanker()
        
        if args.batch:
            # Batch processing mode
            ranker.run_batch(overwrite=args.overwrite, max_workers=args.max_workers)
        else:
            # Interactive mode
            ranker.run_interactive()
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())