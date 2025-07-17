#!/usr/bin/env python3
"""
Redis client manager for Rita LLM backend.
Provides centralized Redis functionality for caching presigned URLs.
"""

import os
import logging
from upstash_redis import Redis
from typing import Optional
from dotenv import load_dotenv
logger = logging.getLogger(__name__)

# Load environment variables with proper file path handling
if os.path.exists(".env"):
    load_dotenv(dotenv_path=".env")
elif os.path.exists(".env.local"):
    load_dotenv(dotenv_path=".env.local")
elif os.path.exists("../.env"):
    load_dotenv(dotenv_path="../.env")
elif os.path.exists("../.env.local"):
    load_dotenv(dotenv_path="../.env.local")


class RedisManager:
    """Centralized Redis client manager for Rita LLM using Upstash."""
    
    def __init__(self):
        """Initialize Upstash Redis client with environment configuration."""
        # Upstash Redis configuration - map your env vars to what Upstash expects
        self.redis_url = os.getenv('KV_REST_API_URL') or os.getenv('KV_URL')
        self.redis_token = os.getenv('KV_REST_API_TOKEN')
        
        # Fallback to standard Upstash env vars if above aren't found
        if not self.redis_url:
            self.redis_url = os.getenv('UPSTASH_REDIS_REST_URL')
        if not self.redis_token:
            self.redis_token = os.getenv('UPSTASH_REDIS_REST_TOKEN')
        
        # Initialize Redis client
        self.redis_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Upstash Redis client."""
        try:
            if self.redis_url and self.redis_token:
                # Direct instantiation with URL and token
                self.redis_client = Redis(
                    url=self.redis_url,
                    token=self.redis_token
                )
                logger.info("Upstash Redis client initialized with URL and token")
            else:
                # Try to use from_env() as fallback if standard env vars are set
                try:
                    self.redis_client = Redis.from_env()
                    logger.info("Upstash Redis client initialized from environment")
                except Exception as env_error:
                    logger.error(f"Failed to initialize from environment: {env_error}")
                    raise
                
            # Test the connection
            self.redis_client.ping()
            logger.info("Upstash Redis connection established successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Upstash Redis client: {e}")
            logger.info("Available env vars: KV_REST_API_URL=%s, KV_URL=%s, KV_REST_API_TOKEN=%s", 
                       bool(os.getenv('KV_REST_API_URL')), 
                       bool(os.getenv('KV_URL')), 
                       bool(os.getenv('KV_REST_API_TOKEN')))
            self.redis_client = None
    
    def is_available(self) -> bool:
        """Check if Redis client is available and connected."""
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False
    
    def get(self, uri: str) -> Optional[str]:
        """
        Get a presigned URL from Redis cache.
        
        Args:
            uri: The S3 object key/URI
            
        Returns:
            Cached presigned URL if found, None otherwise
        """
        if not self.is_available():
            logger.debug("Redis client not available, cannot get cached URL")
            return None
            
        try:
            # Wrap the URI with the same format as frontend: uri:${uri}
            key = f"uri:{uri}"
            cached_url = self.redis_client.get(key)
            
            if cached_url:
                logger.debug(f"Retrieved cached presigned URL for key: {key}")
                return cached_url
            else:
                logger.debug(f"No cached URL found for key: {key}")
                return None
                
        except Exception as e:
            logger.warning(f"Error retrieving cached URL for {uri}: {e}")
            return None
    
    def set(self, uri: str, presigned_url: str, expires_in: int = 3600) -> bool:
        """
        Set a presigned URL in Redis cache.
        
        Args:
            uri: The S3 object key/URI
            presigned_url: The presigned URL to cache
            expires_in: TTL for the cache entry in seconds (default: 1 hour)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.debug("Redis client not available, cannot cache URL")
            return False
            
        try:
            # Wrap the URI with the same format as frontend: uri:${uri}
            key = f"uri:{uri}"
            
            # Set with TTL (slightly less than S3 URL expiration to avoid serving expired URLs)
            cache_ttl = max(expires_in - 60, 60)  # At least 1 minute, but 1 minute less than S3 expiration
            
            self.redis_client.setex(key, cache_ttl, presigned_url)
            logger.debug(f"Cached presigned URL for key: {key} with TTL: {cache_ttl}")
            return True
            
        except Exception as e:
            logger.warning(f"Error caching URL for {uri}: {e}")
            return False
    
    def get_with_min_ttl(self, uri: str, min_ttl_seconds: int = 60) -> Optional[str]:
        """
        Get a presigned URL from Redis cache only if it has sufficient time to live.
        
        Args:
            uri: The S3 object key/URI
            min_ttl_seconds: Minimum TTL required to return the cached URL
            
        Returns:
            Cached presigned URL if found with sufficient TTL, None otherwise
        """
        if not self.is_available():
            logger.debug("Redis client not available, cannot get cached URL")
            return None
            
        try:
            # Wrap the URI with the same format as frontend: uri:${uri}
            key = f"uri:{uri}"
            
            # Check if key exists and get TTL
            ttl = self.redis_client.ttl(key)
            if ttl < min_ttl_seconds:
                logger.debug(f"Cached URL for key {key} has insufficient TTL: {ttl}s < {min_ttl_seconds}s")
                return None
            
            # Get the cached URL
            cached_url = self.redis_client.get(key)
            
            if cached_url:
                logger.debug(f"Retrieved cached presigned URL for key: {key} with TTL: {ttl}s")
                return cached_url
            else:
                logger.debug(f"No cached URL found for key: {key}")
                return None
                
        except Exception as e:
            logger.warning(f"Error retrieving cached URL with TTL check for {uri}: {e}")
            return None


# Create a global instance that can be imported
redis_manager: RedisManager = RedisManager() 