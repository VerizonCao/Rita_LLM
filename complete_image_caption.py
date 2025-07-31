#!/usr/bin/env python3
"""
Complete Image Caption Script for Rita LLM

This script processes avatars that have missing image captions (img_caption column is NULL or empty).
It retrieves the avatar's image URI, generates a presigned AWS URL, uses the image caption function
to generate a caption, and updates the database with the caption.

Usage:
    python complete_image_caption.py [--max-avatars N] [--max-workers N]

Args:
    --max-avatars: Maximum number of avatars to process (default: all)
    --max-workers: Maximum number of concurrent threads (default: 4)
"""

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
import sys
import dotenv

if os.path.exists(".env.local"):
    dotenv.load_dotenv(".env.local")
else:
    dotenv.load_dotenv()

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import required modules
from generation.generation_util.image_caption import caption_image_url
from data.s3 import s3_manager
from data.database import DatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageCaptionCompleter:
    def __init__(self):
        # Load credentials from .env.local
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.postgres_url = os.getenv('POSTGRES_URL')
        
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in Rita_LLM/.env.local")
        if not self.postgres_url:
            raise ValueError("POSTGRES_URL not found in Rita_LLM/.env.local")
        
        # Initialize database manager
        self.db_manager = DatabaseManager()
        
        # Rate limiting
        self._api_lock = threading.Lock()
        self._last_api_call = 0
        self._min_api_interval = 0.5  # Minimum 500ms between API calls to avoid rate limits
        
        # Check if img_caption column exists
        self._check_img_caption_column()

    def _check_img_caption_column(self):
        """Check if img_caption column exists in avatars table"""
        try:
            if not self.db_manager.ensure_connection():
                logger.error("Failed to establish database connection for column check")
                return
            
            # Try to query the img_caption column
            query = """
            SELECT img_caption FROM avatars LIMIT 1
            """
            
            result = self.db_manager.execute_query(query)
            logger.info("img_caption column exists in avatars table")
            
        except Exception as e:
            if "column \"img_caption\" does not exist" in str(e).lower():
                logger.error("img_caption column does not exist in avatars table!")
                logger.error("Please add the column first with:")
                logger.error("ALTER TABLE avatars ADD COLUMN img_caption TEXT;")
                raise ValueError("img_caption column missing from avatars table")
            else:
                logger.error(f"Error checking img_caption column: {e}")
                raise

    def get_avatars_for_processing(self, limit: int = None) -> List[Tuple[str, str, str]]:
        """Get avatar data for image caption completion (only those with missing captions)"""
        try:
            if not self.db_manager.ensure_connection():
                logger.error("Failed to establish database connection")
                return []
            
            # Query for avatars with missing img_caption
            query = """
            SELECT avatar_id, image_uri, avatar_name
            FROM avatars 
            WHERE (img_caption IS NULL OR img_caption = '' OR img_caption = '')
            AND image_uri IS NOT NULL 
            AND image_uri != ''
            ORDER BY create_time DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            result = self.db_manager.execute_query(query)
            
            if result:
                logger.info(f"Retrieved {len(result)} avatars for image caption completion")
                return [(row['avatar_id'], row['image_uri'], row['avatar_name']) for row in result]
            else:
                logger.info("No avatars found with missing image captions")
                return []
                
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            # Check if the error is due to missing column
            if "column \"img_caption\" does not exist" in str(e).lower():
                logger.error("img_caption column does not exist in avatars table. Please add it first.")
                logger.error("SQL to add column: ALTER TABLE avatars ADD COLUMN img_caption TEXT;")
            return []

    def get_presigned_url(self, image_uri: str) -> Optional[str]:
        """Get presigned URL for the image URI"""
        try:
            # Extract S3 key from image_uri if it's a full URL
            if 'amazonaws.com' in image_uri and '?' in image_uri:
                # It's already a presigned URL, return as is
                return image_uri
            else:
                # It's an S3 key, generate presigned URL
                if s3_manager.is_available():
                    presigned_url = s3_manager.get_public_url_with_cache_check(image_uri)
                    if presigned_url:
                        logger.debug(f"Generated presigned URL for S3 key: {image_uri}")
                        return presigned_url
                    else:
                        logger.warning(f"Failed to generate presigned URL for S3 key: {image_uri}")
                        return None
                else:
                    logger.error("S3 manager not available")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting presigned URL for {image_uri}: {e}")
            return None

    def generate_image_caption(self, image_url: str) -> Optional[str]:
        """Generate image caption using the image caption function"""
        try:
            # Rate limiting for API calls in multithreaded environment
            with self._api_lock:
                current_time = time.time()
                time_since_last_call = current_time - self._last_api_call
                if time_since_last_call < self._min_api_interval:
                    time.sleep(self._min_api_interval - time_since_last_call)
                self._last_api_call = time.time()
            
            # Generate caption using the image caption function
            caption = caption_image_url(image_url, self.openrouter_api_key)
            
            if caption:
                logger.debug(f"Generated caption: {caption[:100]}...")
                return caption
            else:
                logger.warning("No caption generated")
                return None
                
        except Exception as e:
            logger.error(f"Error generating image caption: {e}")
            # Check for specific API errors
            if "rate limit" in str(e).lower():
                logger.error("Rate limit exceeded, consider reducing max_workers or increasing min_api_interval")
            elif "api key" in str(e).lower():
                logger.error("Invalid API key or authentication error")
            elif "timeout" in str(e).lower():
                logger.error("Request timeout, consider increasing timeout or checking network")
            return None

    def update_image_caption(self, avatar_id: str, caption: str) -> bool:
        """Update database with generated image caption"""
        try:
            if not self.db_manager.ensure_connection():
                logger.error("Failed to establish database connection for update")
                return False
            
            update_query = """
            UPDATE avatars 
            SET img_caption = %s, update_time = NOW()
            WHERE avatar_id = %s
            """
            
            result = self.db_manager.execute_query(update_query, (caption, avatar_id))
            
            if result:
                logger.debug(f"Updated img_caption for {avatar_id}")
                return True
            else:
                logger.error(f"Failed to update img_caption for {avatar_id}")
                return False
                
        except Exception as e:
            logger.error(f"Database update failed for {avatar_id}: {e}")
            return False

    def process_single_avatar(self, avatar_data: Tuple[str, str, str]) -> bool:
        """Process a single avatar - generate and update image caption"""
        avatar_id, image_uri, avatar_name = avatar_data
        
        try:
            logger.info(f"Processing avatar: {avatar_name} ({avatar_id[:8]}...)")
            
            # Validate image_uri
            if not image_uri or image_uri.strip() == '':
                logger.error(f"Empty image_uri for {avatar_id}")
                return False
            
            # Get presigned URL
            presigned_url = self.get_presigned_url(image_uri)
            if not presigned_url:
                logger.error(f"Failed to get presigned URL for {avatar_id}")
                return False
            
            # Generate image caption
            caption = self.generate_image_caption(presigned_url)
            if not caption:
                logger.error(f"Failed to generate caption for {avatar_id}")
                return False
            
            # Validate caption length
            if len(caption.strip()) < 10:
                logger.warning(f"Generated caption too short for {avatar_id}: {caption}")
                return False
            
            # Update database
            if not self.update_image_caption(avatar_id, caption):
                logger.error(f"Failed to update database for {avatar_id}")
                return False
            
            logger.info(f"Successfully generated and updated caption for {avatar_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {avatar_id}: {e}")
            return False

    def run(self, max_avatars: int = None, max_workers: int = 4):
        """Main execution - complete image captions for avatars using multithreading"""
        logger.info(f"Starting image caption completion for up to {max_avatars or 'all'} avatars with {max_workers} workers")
        
        avatars = self.get_avatars_for_processing(max_avatars)
        if not avatars:
            logger.info("No avatars found for image caption completion")
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
            with tqdm(total=len(avatars), desc="Completing image captions", unit="avatar") as pbar:
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
                    
        logger.info(f"Complete: {success_count}/{len(avatars)} image captions completed")

def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description='Avatar Image Caption Completer')
    parser.add_argument('--max-avatars', type=int, default=None, 
                       help='Maximum number of avatars to process (default: all)')
    parser.add_argument('--max-workers', type=int, default=16,
                       help='Maximum number of concurrent threads (default: 4)')
    
    args = parser.parse_args()
    
    try:
        completer = ImageCaptionCompleter()
        completer.run(max_avatars=args.max_avatars, max_workers=args.max_workers)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main()) 