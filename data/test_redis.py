#!/usr/bin/env python3
"""
Test script to list all keys in the Upstash Redis database.
"""

import sys
import os
from redis import redis_manager
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
    
def test_redis_keys():
    """Test Redis connection and list all keys."""
    print("Testing Upstash Redis connection...")
    
    # Check if Redis is available
    if not redis_manager.is_available():
        print("❌ Redis is not available. Check your connection and environment variables.")
        print("Required env vars: KV_REST_API_URL (or KV_URL) and KV_REST_API_TOKEN")
        return
    
    print("✅ Redis connection successful!")
    
    try:
        # Get all keys
        print("\nFetching all keys from Redis...")
        all_keys = redis_manager.redis_client.keys("*")
        
        if not all_keys:
            print("📭 No keys found in Redis database.")
        else:
            print(f"📋 Found {len(all_keys)} key(s) in Redis:")
            print("-" * 50)
            
            for i, key in enumerate(all_keys, 1):
                # Get TTL for each key
                try:
                    ttl = redis_manager.redis_client.ttl(key)
                    if ttl == -1:
                        ttl_info = "(no expiration)"
                    elif ttl == -2:
                        ttl_info = "(key doesn't exist)"
                    else:
                        ttl_info = f"(expires in {ttl}s)"
                except Exception as e:
                    ttl_info = f"(TTL error: {e})"
                
                print(f"{i:3d}. {key} {ttl_info}")
        
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error fetching keys: {e}")

if __name__ == "__main__":
    test_redis_keys()