import os
import json
import requests
import psycopg2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import argparse
from datetime import datetime
import time
import random
from tqdm import tqdm
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

class AvatarOpeningPromptFiller:
    def __init__(self):
        # Load credentials from .env.local
        env_vars = load_env_file()
        self.openrouter_api_key = env_vars.get('OPENROUTER_API_KEY')
        self.postgres_url = env_vars.get('POSTGRES_URL')
        
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in Rita_LLM/.env.local")
        if not self.postgres_url:
            raise ValueError("POSTGRES_URL not found in Rita_LLM/.env.local")
        
        # OpenRouter API configuration
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "google/gemini-2.5-flash-preview-05-20"
        
        # Rate limiting
        self._api_lock = threading.Lock()
        self._last_api_call = 0
        self._min_api_interval = 0.1  # Minimum 100ms between API calls
        
        # System prompt for generating opening messages
        self.base_system_prompt = """You are a creative writer specializing in roleplay character interactions. Your task is to generate an engaging opening message for a character meeting someone for the first time.

You must respond in this EXACT format – ONLY these two lines:
Line 1: EXACTLY ONE dialogue line enclosed in quotation marks "..."
Line 2: EXACTLY ONE narrative line enclosed in double asterisks **...**
STRICT FORMAT REQUIREMENTS:
ONLY ONE dialogue line – do not write multiple dialogue sentences
ONLY ONE narrative line – do not write multiple narrative sentences
EXACTLY TWO LINES TOTAL – no more, no less
Line 1 must be dialogue in quotation marks: "single dialogue here"
Line 2 must be narrative in double asterisks: **single narrative here**
Write narrative in third person (he/she/they, not I/me)
Do not break dialogue with narrative insertions
Do not use emojis, emoticons, or special characters
GREETINGS MUST:
Be around 100 words total across both lines 
Reflect the bot’s unique background, personality, and past
Establish their prior connection or intrigue with the user
Immediately set up an engaging story hook or scenario
Stay natural and immersive in tone, like a real person speaking
Example:
"Did you really think you could capture me? Now, tell me... why are you really here?"

**The room is dimly lit, casting long shadows as you regain consciousness. Your wrists are bound to the chair, and a dull ache reminds you of the struggle before. You, a policewoman, had come here to capture him—to bring down Alexander Chelsea, the notorious mafia boss. But now, you're the one at his mercy. Alexander stands before you, dressed impeccably in his black suit, a leather whip coiled in his hand. His dark eyes glint with cruel amusement as he steps closer, towering over you. His voice cold and smooth. He trails the whip along your arm, his gaze never leaving yours.**
CRITICAL: Your response must contain EXACTLY one dialogue line and EXACTLY one narrative line. Do not deviate from this format.."""

    def get_avatars_for_processing(self, limit: int = None) -> List[Tuple[str, str, str, str]]:
        """Get avatar data for opening prompt generation (overwriting existing prompts)"""
        try:
            conn = psycopg2.connect(self.postgres_url)
            cursor = conn.cursor()
            
            # Get all public avatars for opening prompt generation/overwriting
            query = """
            SELECT avatar_id, prompt, scene_prompt, agent_bio 
            FROM avatars 
            WHERE is_public = true 
            ORDER BY create_time DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            logger.info(f"Retrieved {len(results)} avatars for opening prompt generation")
            return results
            
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return []

    def build_character_context(self, prompt: str, scene_prompt: str, bio: str) -> str:
        """Build character context for the system prompt"""
        context = "CHARACTER INFORMATION:\n"
        
        if bio and bio.strip():
            context += f"Bio: {bio.strip()}\n"
        
        if prompt and prompt.strip():
            context += f"Character Description: {prompt.strip()}\n"
        
        if scene_prompt and scene_prompt.strip():
            context += f"Scene/Setting: {scene_prompt.strip()}\n"
            context += "\nIMPORTANT: Focus heavily on the scene/setting provided above for the opening interaction.\n"
        else:
            context += "\nNo specific scene provided - create a general friendly greeting appropriate for the character.\n"
        
        context += "\nGenerate an opening message where this character greets someone they're meeting. Make it feel natural and true to their personality."
        
        return context

    def generate_opening_prompt(self, prompt: str, scene_prompt: str, bio: str) -> Optional[str]:
        """Generate opening prompt via OpenRouter API"""
        try:
            # Build character context
            character_context = self.build_character_context(prompt, scene_prompt, bio)
            
            # Rate limiting for API calls in multithreaded environment
            with self._api_lock:
                current_time = time.time()
                time_since_last_call = current_time - self._last_api_call
                if time_since_last_call < self._min_api_interval:
                    time.sleep(self._min_api_interval - time_since_last_call)
                self._last_api_call = time.time()
            
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
                            {"role": "system", "content": self.base_system_prompt},
                            {"role": "user", "content": character_context}
                        ],
                        "max_tokens": 200,
                        "temperature": 0.8,
                        "provider": {"sort": "latency"},
                        "usage": {"include": True}
                    }
                    
                    response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
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
                content = response_data['choices'][0]['message']['content'].strip()
                logger.debug(f"Generated opening prompt: {content[:100]}...")
                return content
            else:
                logger.error("No choices in API response")
                return None
                
        except Exception as e:
            logger.error(f"Opening prompt generation failed: {e}")
            return None

    def update_opening_prompt(self, avatar_id: str, opening_prompt: str) -> bool:
        """Update database with generated opening prompt"""
        try:
            conn = psycopg2.connect(self.postgres_url)
            cursor = conn.cursor()
            
            update_query = """
            UPDATE avatars 
            SET opening_prompt = %s
            WHERE avatar_id = %s
            """
            
            cursor.execute(update_query, (opening_prompt, avatar_id))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.debug(f"Updated opening_prompt for {avatar_id}")
            return True
            
        except Exception as e:
            logger.error(f"Database update failed for {avatar_id}: {e}")
            return False

    def process_single_avatar(self, avatar_data: Tuple[str, str, str, str]) -> bool:
        """Process a single avatar - generate and update opening prompt"""
        avatar_id, prompt, scene_prompt, bio = avatar_data
        
        try:
            # Generate opening prompt
            opening_prompt = self.generate_opening_prompt(prompt, scene_prompt, bio)
            if not opening_prompt:
                logger.error(f"Failed to generate opening prompt for {avatar_id}")
                return False
            
            # Update database
            if not self.update_opening_prompt(avatar_id, opening_prompt):
                logger.error(f"Failed to update database for {avatar_id}")
                return False
            
            logger.info(f"Successfully generated opening prompt for {avatar_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {avatar_id}: {e}")
            return False

    def run(self, max_avatars: int = None, max_workers: int = 4):
        """Main execution - generate opening prompts for avatars using multithreading"""
        logger.info(f"Starting opening prompt generation (overwriting mode) for up to {max_avatars or 'all'} avatars with {max_workers} workers")
        
        avatars = self.get_avatars_for_processing(max_avatars)
        if not avatars:
            logger.info("No avatars found for opening prompt generation")
            return
            
        success_count = 0
        
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_avatar = {
                executor.submit(self.process_single_avatar, avatar_data): avatar_data[0] 
                for avatar_data in avatars
            }
            
            # Use tqdm for progress tracking
            with tqdm(total=len(avatars), desc="Generating opening prompts", unit="avatar") as pbar:
                for future in as_completed(future_to_avatar):
                    avatar_id = future_to_avatar[future]
                    try:
                        success = future.result()
                        if success:
                            success_count += 1
                        
                        pbar.set_description(f"Processed {avatar_id[:8]}...")
                        
                    except Exception as e:
                        logger.error(f"Thread error for {avatar_id}: {e}")
                    
                    pbar.update(1)
                    
        logger.info(f"Complete: {success_count}/{len(avatars)} opening prompts generated")

def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description='Avatar Opening Prompt Generator')
    parser.add_argument('--max-avatars', type=int, default=None, 
                       help='Maximum number of avatars to process (default: all)')
    parser.add_argument('--max-workers', type=int, default=16,
                       help='Maximum number of concurrent threads (default: 4)')
    
    args = parser.parse_args()
    
    try:
        filler = AvatarOpeningPromptFiller()
        filler.run(max_avatars=args.max_avatars, max_workers=args.max_workers)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())