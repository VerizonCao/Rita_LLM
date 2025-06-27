import os
import requests
import psycopg2
import logging
from pathlib import Path
import time

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

def detect_gender_from_path(audio_path: str) -> str:
    """Detect gender from parent directory name"""
    path_obj = Path(audio_path)
    parent_dir = path_obj.parent.name.lower()
    
    if parent_dir == "female":
        return "female"
    elif parent_dir == "male":
        return "male"
    else:
        # Default to non-binary if not clearly male/female
        return "non-binary"

def get_relative_filename(audio_path: str) -> str:
    """Get relative filename starting from gender directory"""
    path_obj = Path(audio_path)
    parent_dir = path_obj.parent.name
    filename = path_obj.name
    return f"{parent_dir}/{filename}"

def find_audio_files(base_directory: str) -> list:
    """Find all audio files in the directory structure"""
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
    audio_files = []
    
    base_path = Path(base_directory)
    if not base_path.exists():
        print(f"❌ Directory not found: {base_directory}")
        return []
    
    # Look for female and male directories
    for gender_dir in ['female', 'male']:
        gender_path = base_path / gender_dir
        if gender_path.exists() and gender_path.is_dir():
            print(f"📁 Scanning {gender_dir} directory...")
            for file_path in gender_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                    audio_files.append(str(file_path))
                    print(f"   Found: {file_path.name}")
    
    return audio_files

def clone_voice(audio_path, description="Voice cloned using Cartesia API"):
    """
    Clone a voice using Cartesia API
    
    Args:
        audio_path (str): Path to the audio file to clone
        description (str): Description for the cloned voice
    
    Returns:
        str: The voice_id of the newly cloned voice, or None if failed
    """
    # Get API key from environment
    env_vars = load_env_file()
    CARTESIA_API_KEY = env_vars.get("CARTESIA_API_KEY")
    
    if not CARTESIA_API_KEY:
        print("Error: CARTESIA_API_KEY environment variable not set.")
        return None
    
    # Check if audio file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return None
    
    # Extract filename without directory and extension for voice name
    filename = os.path.basename(audio_path)
    voice_name = os.path.splitext(filename)[0]
    
    # API endpoint and headers
    url = "https://api.cartesia.ai/voices/clone"
    headers = {
        "Cartesia-Version": "2024-11-13",
        "X-API-Key": CARTESIA_API_KEY
    }
    
    # Prepare the payload
    payload = {
        "name": voice_name,
        "description": description,
        "language": "en",
        "mode": "stability",
        "enhance": False,  # Set to True if you want AI enhancement to reduce background noise
    }
    
    try:
        # Open and send the file
        with open(audio_path, 'rb') as audio_file:
            files = {"clip": audio_file}
            
            print(f"   🔄 Uploading: {os.path.basename(audio_path)}")
            
            response = requests.post(url, data=payload, files=files, headers=headers)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            voice_id = result.get("id")
            print(f"   ✅ Cloned successfully! ID: {voice_id}")
            return voice_id
        else:
            print(f"   ❌ Cloning failed. Status: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"   ❌ Error occurred: {str(e)}")
        return None

def save_voice_to_database(cartesia_voice_id: str, audio_path: str, voice_name: str):
    """
    Save voice entry to voice_library database table
    
    Args:
        cartesia_voice_id (str): The Cartesia voice ID
        audio_path (str): Path to the audio file
        voice_name (str): Name of the voice
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load database credentials
        env_vars = load_env_file()
        postgres_url = env_vars.get('POSTGRES_URL')
        
        if not postgres_url:
            print("   ❌ POSTGRES_URL not found in Rita_LLM/.env.local")
            return False
        
        # Detect gender and get relative filename
        gender = detect_gender_from_path(audio_path)
        file_name = get_relative_filename(audio_path)
        
        # Connect to database
        conn = psycopg2.connect(postgres_url)
        cursor = conn.cursor()
        
        # Insert into voice_library table
        insert_query = """
        INSERT INTO voice_library (
            cartesia_voice_id,
            gender,
            is_public,
            voice_name,
            file_name
        ) VALUES (%s, %s, %s, %s, %s)
        RETURNING rita_voice_id
        """
        
        cursor.execute(insert_query, (
            cartesia_voice_id,
            gender,
            False,  # is_public = False
            voice_name,
            file_name
        ))
        
        # Get the generated rita_voice_id
        rita_voice_id = cursor.fetchone()[0]
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"   💾 Saved to DB: {gender} voice '{voice_name}'")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Database error: {str(e)}")
        logger.error(f"Database save failed: {e}")
        if 'conn' in locals():
            conn.rollback()
            cursor.close()
            conn.close()
        return False

def process_single_audio_file(audio_path: str) -> tuple:
    """Process a single audio file: clone and save to database"""
    filename = os.path.basename(audio_path)
    voice_name = os.path.splitext(filename)[0]
    gender = detect_gender_from_path(audio_path)
    
    print(f"\n🎤 Processing: {voice_name} ({gender})")
    
    # Clone the voice
    voice_id = clone_voice(audio_path)
    if not voice_id:
        return False, voice_name, gender
    
    # Save to database
    db_success = save_voice_to_database(voice_id, audio_path, voice_name)
    
    return db_success, voice_name, gender

def process_all_voices():
    """Process all audio files in the directory structure"""
    base_directory = "/mnt/c/Users/mjh/Downloads/tts_used/"
    
    print("🎯 Cartesia Batch Voice Cloner")
    print("=" * 50)
    print(f"📂 Scanning directory: {base_directory}")
    
    # Find all audio files
    audio_files = find_audio_files(base_directory)
    
    if not audio_files:
        print("❌ No audio files found!")
        return
    
    print(f"\n📊 Found {len(audio_files)} audio files to process")
    
    # Process each file
    successful = []
    failed = []
    gender_counts = {'male': 0, 'female': 0, 'non-binary': 0}
    
    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] " + "=" * 30)
        
        try:
            success, voice_name, gender = process_single_audio_file(audio_path)
            
            if success:
                successful.append((voice_name, gender))
                gender_counts[gender] += 1
                print(f"   🎉 SUCCESS: {voice_name}")
            else:
                failed.append(voice_name)
                print(f"   💔 FAILED: {voice_name}")
                
            # Small delay between requests to be respectful to the API
            time.sleep(1)
            
        except Exception as e:
            print(f"   ❌ Unexpected error processing {audio_path}: {e}")
            failed.append(os.path.splitext(os.path.basename(audio_path))[0])
    
    # Final summary
    print(f"\n🎊 BATCH PROCESSING COMPLETE!")
    print("=" * 50)
    print(f"✅ Successfully processed: {len(successful)}/{len(audio_files)} voices")
    print(f"❌ Failed: {len(failed)} voices")
    
    if successful:
        print(f"\n👥 Gender distribution:")
        print(f"   🚺 Female: {gender_counts['female']}")
        print(f"   🚹 Male: {gender_counts['male']}")
        print(f"   ⚧ Non-binary: {gender_counts['non-binary']}")
        
        print(f"\n🎤 Successfully cloned voices:")
        for voice_name, gender in successful:
            print(f"   • {voice_name} ({gender})")
    
    if failed:
        print(f"\n💔 Failed voices:")
        for voice_name in failed:
            print(f"   • {voice_name}")

# Run the batch processing
if __name__ == "__main__":
    process_all_voices()