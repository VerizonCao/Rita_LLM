import os
import psycopg2
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Dict, List, Tuple, Optional

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

class V2ScoreCalculator:
    def __init__(self):
        # Load credentials from .env.local
        env_vars = load_env_file()
        self.postgres_url = env_vars.get('POSTGRES_URL')
        
        if not self.postgres_url:
            raise ValueError("POSTGRES_URL not found in Rita_LLM/.env.local")
        
        # Thread-safe counter for progress tracking
        self._lock = threading.Lock()
        self._processed_count = 0
        self._total_count = 0

    def ensure_int(self, value) -> int:
        """Ensure value is an integer (handles both int and None)"""
        if value is None:
            return 1
        if isinstance(value, int):
            return value
        return 1

    def map_enum_to_int(self, value, enum_mapping: dict) -> int:
        """Map enum string values to integers"""
        if not value:
            return 1
        if isinstance(value, str):
            return enum_mapping.get(value.lower(), 1)
        return 1

    def map_age_to_points(self, value: str) -> int:
        """Map age values to direct points"""
        if not value:
            return 0
        mapping = {
            "old": 0,
            "middle-aged": 20,
            "young": 30
        }
        if isinstance(value, str):
            return mapping.get(value.lower(), 0)
        return 0

    def compute_v2_score(self, avatar_data: Dict) -> int:
        """Compute v2 score based on the scoring logic"""
        base_score = 1000
        
        # text_quality (1-5): coefficient * 50
        text_quality = self.ensure_int(avatar_data.get('text_quality', 1))
        base_score += text_quality * 50
        
        # age: direct points
        age_points = self.map_age_to_points(avatar_data.get('age', ''))
        base_score += age_points
        
        # image_coherence (1-3): coefficient * 20 - already INTEGER in DB
        image_coherence = self.ensure_int(avatar_data.get('image_coherence', 1))
        base_score += image_coherence * 20
        
        # character_card_coherence: true=20, false=0
        if avatar_data.get('character_card_coherence', False):
            base_score += 20
        
        # image_quality (1-3): coefficient * 20 - already INTEGER in DB
        image_quality = self.ensure_int(avatar_data.get('image_quality', 1))
        base_score += image_quality * 20
        
        # face (1-3): coefficient * 50 - already INTEGER in DB
        face = self.ensure_int(avatar_data.get('face', 1))
        base_score += face * 50
        
        # hair: true=20, false=0 - already BOOLEAN in DB
        if avatar_data.get('hair', False):
            base_score += 20
        
        # is_neutral (1-3): coefficient * 10 - enum in DB
        is_neutral_mapping = {"neutral": 1, "medium": 2, "strong": 3}
        is_neutral = self.map_enum_to_int(avatar_data.get('is_neutral', ''), is_neutral_mapping)
        base_score += is_neutral * 10
        
        # body (1-3): coefficient * 20 - already INTEGER in DB
        body = self.ensure_int(avatar_data.get('body', 1))
        base_score += body * 20
        
        # headpose (1-3): coefficient * 20 - enum in DB  
        headpose_mapping = {"neutral": 1, "tilted": 2, "offset": 3}
        headpose = self.map_enum_to_int(avatar_data.get('headpose', ''), headpose_mapping)
        base_score += headpose * 20
        
        # gesture (1-3): coefficient * 10 - already INTEGER in DB
        gesture = self.ensure_int(avatar_data.get('gesture', 1))
        base_score += gesture * 10
        
        # outerwear_quality (1-3): coefficient * 30 - already INTEGER in DB
        outerwear_quality = self.ensure_int(avatar_data.get('outerwear_quality', 1))
        base_score += outerwear_quality * 30
        
        # outerwear_desirability (1-3): coefficient * 50 - already INTEGER in DB
        outerwear_desirability = self.ensure_int(avatar_data.get('outerwear_desirability', 1))
        base_score += outerwear_desirability * 50
        
        # apparels_quality (1-3): coefficient * 10 - already INTEGER in DB
        apparels_quality = self.ensure_int(avatar_data.get('apparels_quality', 1))
        base_score += apparels_quality * 10
        
        return base_score

    def get_avatars_to_process(self) -> List[str]:
        """Get avatar_ids that need v2_score computation (where v2_score = 1000)"""
        try:
            conn = psycopg2.connect(self.postgres_url)
            cursor = conn.cursor()
            
            query = """
            SELECT avatar_id 
            FROM avatar_discovery 
            WHERE v2_score = 1000
            ORDER BY avatar_id
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            avatar_ids = [row[0] for row in results]
            logger.info(f"Found {len(avatar_ids)} avatars with default v2_score (1000)")
            return avatar_ids
            
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return []

    def get_avatar_data(self, avatar_id: str) -> Optional[Dict]:
        """Get avatar data from avatar_discovery table"""
        try:
            conn = psycopg2.connect(self.postgres_url)
            cursor = conn.cursor()
            
            query = """
            SELECT avatar_id, text_quality, age, image_coherence, character_card_coherence,
                   image_quality, face, hair, is_neutral, body, headpose, gesture,
                   outerwear_quality, outerwear_desirability, apparels_quality
            FROM avatar_discovery 
            WHERE avatar_id = %s
            """
            
            cursor.execute(query, (avatar_id,))
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                return {
                    'avatar_id': result[0],
                    'text_quality': result[1],
                    'age': result[2],
                    'image_coherence': result[3],
                    'character_card_coherence': result[4],
                    'image_quality': result[5],
                    'face': result[6],
                    'hair': result[7],
                    'is_neutral': result[8],
                    'body': result[9],
                    'headpose': result[10],
                    'gesture': result[11],
                    'outerwear_quality': result[12],
                    'outerwear_desirability': result[13],
                    'apparels_quality': result[14]
                }
            else:
                logger.error(f"Avatar {avatar_id} not found in avatar_discovery table")
                return None
                
        except Exception as e:
            logger.error(f"Database query failed for {avatar_id}: {e}")
            return None

    def update_v2_score(self, avatar_id: str, v2_score: int) -> bool:
        """Update v2_score in database"""
        try:
            conn = psycopg2.connect(self.postgres_url)
            cursor = conn.cursor()
            
            update_query = """
            UPDATE avatar_discovery 
            SET v2_score = %s
            WHERE avatar_id = %s
            """
            
            cursor.execute(update_query, (v2_score, avatar_id))
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Updated v2_score for {avatar_id}: {v2_score}")
            return True
            
        except Exception as e:
            logger.error(f"Database update failed for {avatar_id}: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
            return False

    def process_single_avatar(self, avatar_id: str) -> Tuple[str, bool, int]:
        """Process a single avatar and return result"""
        try:
            # Get avatar data
            avatar_data = self.get_avatar_data(avatar_id)
            if not avatar_data:
                return avatar_id, False, 0
            
            # Compute v2 score
            v2_score = self.compute_v2_score(avatar_data)
            
            # Update database
            success = self.update_v2_score(avatar_id, v2_score)
            
            return avatar_id, success, v2_score
            
        except Exception as e:
            logger.error(f"Error processing {avatar_id}: {e}")
            return avatar_id, False, 0

    def _update_progress(self, avatar_id: str, success: bool, score: int):
        """Thread-safe progress update"""
        with self._lock:
            self._processed_count += 1
            status = "✅" if success else "❌"
            print(f"{status} [{self._processed_count}/{self._total_count}] {avatar_id}: {score}")

    def run_batch_update(self, max_workers: int = 16):
        """Run batch v2 score update with multithreading"""
        logger.info(f"Starting v2 score batch update with {max_workers} workers")
        print(f"🎯 V2 Score Calculator - Batch Mode")
        print("=" * 50)
        print(f"🧵 Using {max_workers} worker threads")
        
        # Get avatars to process
        avatar_ids = self.get_avatars_to_process()
        if not avatar_ids:
            print("✅ No avatars found with default v2_score (1000)")
            return
        
        print(f"📊 Found {len(avatar_ids)} avatars to process")
        
        # Reset counters
        with self._lock:
            self._processed_count = 0
            self._total_count = len(avatar_ids)
        
        successful_updates = []
        failed_updates = []
        total_score_sum = 0
        
        # Process avatars with multithreading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_avatar = {
                executor.submit(self.process_single_avatar, avatar_id): avatar_id 
                for avatar_id in avatar_ids
            }
            
            # Process completed tasks
            for future in as_completed(future_to_avatar):
                avatar_id = future_to_avatar[future]
                try:
                    processed_avatar_id, success, score = future.result()
                    self._update_progress(processed_avatar_id, success, score)
                    
                    if success:
                        successful_updates.append((processed_avatar_id, score))
                        total_score_sum += score
                    else:
                        failed_updates.append(processed_avatar_id)
                        
                except Exception as e:
                    logger.error(f"Exception in thread for {avatar_id}: {e}")
                    failed_updates.append(avatar_id)
                    self._update_progress(avatar_id, False, 0)
        
        # Final summary
        success_count = len(successful_updates)
        total_count = len(avatar_ids)
        
        print(f"\n🎉 Batch v2 score update complete!")
        print(f"📈 Successfully updated: {success_count}/{total_count} avatars")
        
        if success_count > 0:
            avg_score = total_score_sum / success_count
            print(f"📊 Average v2 score: {avg_score:.1f}")
            
            # Show score distribution
            scores = [score for _, score in successful_updates]
            min_score = min(scores)
            max_score = max(scores)
            print(f"📊 Score range: {min_score} - {max_score}")
        
        if failed_updates:
            print(f"❌ Failed updates: {len(failed_updates)}")
            print(f"   {', '.join(failed_updates[:10])}{' ...' if len(failed_updates) > 10 else ''}")
        
        logger.info(f"Batch complete: {success_count}/{total_count} v2 scores updated successfully")


def main():
    """Main entry point"""
    try:
        calculator = V2ScoreCalculator()
        calculator.run_batch_update(max_workers=16)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
