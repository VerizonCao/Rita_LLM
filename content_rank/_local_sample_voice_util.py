import os
import requests
import psycopg2
from pathlib import Path

# List of TTS prompts to cycle through
TTS_PROMPTS = [
    "There's always that one class, honestly, that you'd love to skip forever.",
    "Sometimes, I think bosses should try doing our jobs, just for one day.",
    "First dates get weird when someone orders the strangest thing on the menu.",
    "It would be amazing, suddenly, to wake up with incredible artistic talent.",
    "In the future, robots will probably cook everything, and I'm here for it.",
    "Living in the 1800s might be cool, but I'd definitely need my phone.",
    "Your phone contacts say a lot, honestly, about who really matters.",
    "Ten-minute shopping sprees sound stressful, but also kind of exciting.",
    "Being famous for a week, then disappearing, actually sounds perfect.",
    "Pets would probably complain about everything, if they could talk.",
    "Teaching a class about something you love, honestly, sounds pretty fun.",
    "Sending texts to the wrong person, especially parents, is pure nightmare fuel.",
    "Redecorating with just one color, surprisingly, might look really cool.",
    "Working at a bakery, with free treats, would be dangerous for me.",
    "Meeting your future self, honestly, would be terrifying but fascinating.",
    "Having your best friend as a neighbor, actually, sounds pretty great.",
    "Mastering any sport instantly, even for a day, would be incredible.",
    "Finding random cash, with no owner, creates such a moral dilemma.",
    "Fifty dollars can go surprisingly far, if you're creative about it.",
    "Being inside your favorite game, honestly, would be the ultimate escape."
]


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
        return env_vars
    except Exception as e:
        print(f"Failed to load environment file {env_path}: {e}")
        return {}

def get_voice_mappings():
    """Query database for voices with NULL provider"""
    env_vars = load_env_file()
    postgres_url = env_vars.get('POSTGRES_URL')
    
    if not postgres_url:
        print("❌ POSTGRES_URL not found in Rita_LLM/.env.local")
        return {}
    
    try:
        conn = psycopg2.connect(postgres_url)
        cursor = conn.cursor()
        
        query = """
        SELECT rita_voice_id, cartesia_voice_id, voice_name
        FROM voice_library 
        WHERE provider IS NULL
        ORDER BY voice_name
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Create mapping from rita_voice_id to cartesia_voice_id
        mapping = {}
        for rita_id, cartesia_id, voice_name in results:
            mapping[rita_id] = cartesia_id
        
        print(f"📊 Loaded {len(mapping)} voice mappings from database")
        return mapping
        
    except Exception as e:
        print(f"❌ Database query failed: {e}")
        return {}

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

def extract_rita_voice_id_from_filename(filename: str) -> str:
    """Extract rita voice ID from filename by splitting on underscore"""
    # Filename format: VoiceName_rita-voice-id.ext
    # Example: Sweet_4616dc78-2a44-41cf-bbd6-e25118f03763.mp3
    name_without_ext = Path(filename).stem
    parts = name_without_ext.split('_')
    
    if len(parts) >= 2:
        # The rita voice ID should be the last part after splitting by underscore
        return parts[-1]
    else:
        print(f"   ⚠️  Could not extract rita voice ID from: {filename}")
        return None

def generate_tts_audio(cartesia_voice_id: str, transcript: str) -> bytes:
    """Generate TTS audio using Cartesia API"""
    env_vars = load_env_file()
    CARTESIA_API_KEY = env_vars.get("CARTESIA_API_KEY")
    
    if not CARTESIA_API_KEY:
        print("❌ CARTESIA_API_KEY environment variable not set.")
        return None
    
    url = "https://api.cartesia.ai/tts/bytes"
    headers = {
        "Cartesia-Version": "2024-11-13",
        "X-API-Key": CARTESIA_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "model_id": "sonic-2",
        "transcript": transcript,
        "voice": {
            "mode": "id",
            "id": cartesia_voice_id
        },
        "output_format": {
            "container": "mp3",
            "bit_rate": 128000,
            "sample_rate": 44100
        },
        "language": "en"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.content
        else:
            print(f"   ❌ TTS API failed. Status: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"   ❌ TTS API error: {str(e)}")
        return None

def process_audio_file(file_path: str, voice_mapping: dict, voice_index: int) -> bool:
    """Process a single audio file: extract ID, generate new TTS, replace file"""
    filename = os.path.basename(file_path)
    voice_name = filename.split('_')[0]
    
    # Get the prompt for this voice using rolling index
    prompt = TTS_PROMPTS[voice_index % len(TTS_PROMPTS)]
    
    print(f"\n🎤 Processing: {voice_name}")
    print(f"   📄 File: {filename}")
    
    # Extract rita voice ID from filename
    rita_voice_id = extract_rita_voice_id_from_filename(filename)
    if not rita_voice_id:
        return False
    
    print(f"   🆔 Rita Voice ID: {rita_voice_id}")
    
    # Map rita voice ID to cartesia voice ID
    cartesia_voice_id = voice_mapping.get(rita_voice_id)
    if not cartesia_voice_id:
        print(f"   ❌ No cartesia voice ID mapping found for rita ID: {rita_voice_id}")
        return False
    
    print(f"   🎵 Cartesia Voice ID: {cartesia_voice_id}")
    print(f"   💬 Text: {prompt}")
    
    # Generate new TTS audio
    print(f"   🔄 Generating TTS...")
    audio_content = generate_tts_audio(cartesia_voice_id, prompt)
    
    if not audio_content:
        print(f"   ❌ Failed to generate TTS")
        return False
    
    try:
        # Remove the old file
        os.remove(file_path)
        print(f"   🗑️  Removed old file")
        
        # Save the new audio content with the same filename
        with open(file_path, 'wb') as f:
            f.write(audio_content)
        
        print(f"   ✅ SUCCESS: Generated new audio sample")
        return True
        
    except Exception as e:
        print(f"   ❌ File operation error: {str(e)}")
        return False

def regenerate_voice_samples():
    """Main function to regenerate all voice sample files"""
    base_directory = "/home/mjh/wsl_project_root/NextJS_Example/public/audio_samples"
    
    print("🎵 Voice Sample Regenerator")
    print("=" * 50)
    print(f"📂 Base directory: {base_directory}")
    
    # Get voice mappings from database
    voice_mapping = get_voice_mappings()
    if not voice_mapping:
        print("❌ No voice mappings found in database!")
        return
    
    # Find all audio files
    audio_files = find_audio_files(base_directory)
    
    if not audio_files:
        print("❌ No audio files found!")
        return
    
    print(f"\n📊 Found {len(audio_files)} audio files to regenerate")
    
    # Process each file
    successful = []
    failed = []
    
    for i, file_path in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] " + "=" * 30)
        
        success = process_audio_file(file_path, voice_mapping, i - 1)  # Use 0-based index for prompt selection
        
        if success:
            successful.append(os.path.basename(file_path))
        else:
            failed.append(os.path.basename(file_path))
    
    # Final summary
    print(f"\n🎊 REGENERATION COMPLETE!")
    print("=" * 50)
    print(f"✅ Successfully regenerated: {len(successful)}/{len(audio_files)} files")
    print(f"❌ Failed: {len(failed)} files")
    
    if successful:
        print(f"\n🎤 Successfully regenerated:")
        for filename in successful:
            print(f"   • {filename}")
    
    if failed:
        print(f"\n💔 Failed regenerations:")
        for filename in failed:
            print(f"   • {filename}")

if __name__ == "__main__":
    regenerate_voice_samples()
