#!/usr/bin/env python3
"""
S3 client manager for Rita LLM backend.
Provides centralized S3 client functionality for image storage and URL generation.
"""

import os
import logging
import boto3
from botocore.exceptions import ClientError
from typing import Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

# Import Redis manager for caching (import after logger to avoid circular imports)
try:
    from .redis import redis_manager
except ImportError:
    # If Redis is not available, we'll handle it gracefully
    redis_manager = None
    logger.warning("Redis manager not available, S3 URLs will not be cached")


class S3Manager:
    """Centralized S3 client manager for Rita LLM."""
    
    def __init__(self):
        """Initialize S3 client with environment configuration."""
        # AWS S3 configuration
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_REGION', 'us-west-2')
        self.aws_bucket_name = os.getenv('AWS_BUCKET_NAME', 'rita-avatar-image')
        
        # Initialize S3 client - support both explicit credentials and IAM role
        self.s3_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the S3 client based on available credentials."""
        try:
            # Check if we're in Lambda environment (has session token)
            aws_session_token = os.getenv('AWS_SESSION_TOKEN')
            
            if aws_session_token:
                # Lambda environment - use IAM role with session token
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    aws_session_token=aws_session_token,
                    region_name=self.aws_region
                )
                logger.info("S3 client initialized with Lambda IAM role")
            elif self.aws_access_key_id and self.aws_secret_access_key:
                # Local development - use explicit credentials
                self.s3_client = boto3.client(
                    's3',
                    region_name=self.aws_region,
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key
                )
                logger.info("S3 client initialized with explicit credentials")
            else:
                # Try to use default credentials (IAM role without session token)
                self.s3_client = boto3.client('s3', region_name=self.aws_region)
                logger.info("S3 client initialized with default credentials")
                
        except Exception as e:
            logger.warning(f"Failed to initialize S3 client: {e}")
            self.s3_client = None
    
    def is_available(self) -> bool:
        """Check if S3 client is available."""
        return self.s3_client is not None
    
    def get_public_url_from_s3(self, s3_key: str, expires_in: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL for public access to an S3 object directly from S3.
        
        Args:
            s3_key: The S3 object key (e.g., 'rita-swap-images/user_id/avatar_id/filename.jpg')
            expires_in: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            Presigned URL if successful, None otherwise
        """
        if not self.s3_client:
            logger.warning("S3 client not available, cannot generate public URL")
            return None
            
        try:
            # Generate presigned URL for GET operation
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.aws_bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expires_in
            )
            
            logger.debug(f"Generated presigned URL for S3 key: {s3_key}")
            return presigned_url
            
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL for {s3_key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error generating presigned URL for {s3_key}: {e}")
            return None
    
    def get_public_url_with_cache_check(self, s3_key: str, expires_in: int = 3600) -> Optional[str]:
        """
        Get a presigned URL for public access to an S3 object with Redis caching.
        First checks Redis cache, then generates a new URL from S3 if not cached.
        
        Args:
            s3_key: The S3 object key (e.g., 'rita-swap-images/user_id/avatar_id/filename.jpg')
            expires_in: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            Presigned URL if successful, None otherwise
        """
        # First check Redis cache with minimum TTL requirement (1 minute)
        if redis_manager and redis_manager.is_available():
            try:
                cached_url = redis_manager.get_with_min_ttl(s3_key, min_ttl_seconds=60)
                if cached_url:
                    logger.debug(f"Retrieved cached presigned URL for S3 key: {s3_key}")
                    return cached_url
            except Exception as e:
                # Gracefully handle Redis errors - don't fail the entire operation
                logger.warning(f"Redis lookup failed for presigned URL, proceeding with S3 generation: {e}")
        
        # If not in cache or Redis failed, generate new URL from S3
        presigned_url = self.get_public_url_from_s3(s3_key, expires_in)
        
        if presigned_url:
            # Try to cache the new presigned URL in Redis (fire-and-forget)
            if redis_manager and redis_manager.is_available():
                try:
                    redis_manager.set(s3_key, presigned_url, expires_in)
                    logger.debug(f"Cached new presigned URL for S3 key: {s3_key}")
                except Exception as e:
                    # Log but don't fail - the URL is still valid even if we can't cache it
                    logger.warning(f"Failed to cache presigned URL in Redis, but URL is still valid: {e}")
        
        return presigned_url
    
    def upload_file(self, file_content: bytes, s3_key: str, content_type: str = 'image/jpeg', metadata: Optional[dict] = None) -> bool:
        """
        Upload file content to S3.
        
        Args:
            file_content: Binary file content to upload
            s3_key: S3 object key where file will be stored
            content_type: MIME type of the file
            metadata: Optional metadata to attach to the object
            
        Returns:
            True if successful, False otherwise
        """
        if not self.s3_client:
            logger.warning("S3 client not available, cannot upload file")
            return False
            
        try:
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.aws_bucket_name,
                Key=s3_key,
                Body=file_content,
                ContentType=content_type,
                Metadata=metadata or {}
            )
            
            logger.info(f"Successfully uploaded file to S3: {s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to upload file to S3: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading file to S3: {e}")
            return False
    
    def object_exists(self, s3_key: str) -> bool:
        """
        Check if an object exists in S3.
        
        Args:
            s3_key: S3 object key to check
            
        Returns:
            True if object exists, False otherwise
        """
        if not self.s3_client:
            logger.warning("S3 client not available, cannot check object existence")
            return False
            
        try:
            self.s3_client.head_object(Bucket=self.aws_bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                logger.error(f"Error checking object existence for {s3_key}: {e}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error checking object existence for {s3_key}: {e}")
            return False


# Create a global instance that can be imported
s3_manager: S3Manager = S3Manager() 