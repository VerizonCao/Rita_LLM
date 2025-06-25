#!/usr/bin/env python3
"""
Play Session Management Module

Handles operations for play sessions in the Rita LLM system.
Manages session creation, updates, and token/cost tracking.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from .database import DatabaseManager

class PlaySessionManager:
    """Manages play session operations in the database."""
    
    def __init__(self):
        """Initialize the play session manager."""
        self.db = DatabaseManager()
    
    def create_session(self, user_id: str, avatar_id: str, usage: Dict[str, Any] = None) -> Optional[str]:
        """
        Create a new play session with optional usage information.
        
        Args:
            user_id: The ID of the user (NOT NULL)
            avatar_id: The ID of the avatar (NOT NULL)
            usage: Optional dictionary containing usage information with:
                - prompt_tokens: int (nullable)
                - completion_tokens: int (nullable)
                - total_tokens: int (nullable)
                - tts_tokens: int (nullable)
                - cost: int (nullable)
                - session_time: int (defaults to 0)
            
        Returns:
            The session_id if successful, None otherwise
        """
        if usage:
            query = """
                INSERT INTO play_sessions (
                    user_id, avatar_id,
                    prompt_tokens, completion_tokens, total_tokens, tts_tokens, cost,
                    session_time
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING session_id
            """
            params = (
                user_id, avatar_id,
                usage.get('prompt_tokens'),
                usage.get('completion_tokens'),
                usage.get('total_tokens'),
                usage.get('tts_tokens'),
                usage.get('cost'),
                usage.get('session_time', 0)  # Default to 0 if not provided
            )
        else:
            query = """
                INSERT INTO play_sessions (user_id, avatar_id)
                VALUES (%s, %s)
                RETURNING session_id
            """
            params = (user_id, avatar_id)
        
        if not self.db.ensure_connection():
            return None
            
        result = self.db.execute_query(query, params)
        if result and len(result) > 0:
            return result[0]['session_id']
        return None
    
    def update_session_time(self, session_id: str, additional_time: int) -> bool:
        """
        Update the session time by adding the specified amount.
        
        Args:
            session_id: The session ID to update
            additional_time: Time in seconds to add to the session
            
        Returns:
            True if successful, False otherwise
        """
        query = """
            UPDATE play_sessions
            SET session_time = session_time + %s,
                update_time = now()
            WHERE session_id = %s
        """
        
        if not self.db.ensure_connection():
            return False
            
        result = self.db.execute_query(query, (additional_time, session_id))
        return result is not None and result[0]['affected_rows'] > 0
    
    def update_token_usage(self, session_id: str, prompt_tokens: int, 
                          completion_tokens: int, cost: int, tts_tokens: int = None) -> bool:
        """
        Update token usage and cost for a session.
        
        Args:
            session_id: The session ID to update
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
            cost: Cost in smallest currency unit (e.g., cents)
            tts_tokens: Number of TTS tokens used (dialogue only, optional)
            
        Returns:
            True if successful, False otherwise
        """
        query = """
            UPDATE play_sessions
            SET prompt_tokens = COALESCE(prompt_tokens, 0) + %s,
                completion_tokens = COALESCE(completion_tokens, 0) + %s,
                total_tokens = COALESCE(total_tokens, 0) + %s,
                tts_tokens = COALESCE(tts_tokens, 0) + %s,
                cost = COALESCE(cost, 0) + %s,
                update_time = now()
            WHERE session_id = %s
        """
        
        total_tokens = prompt_tokens + completion_tokens
        tts_tokens_value = tts_tokens if tts_tokens is not None else 0
        
        params = (prompt_tokens, completion_tokens, total_tokens, tts_tokens_value, cost, session_id)
        
        if not self.db.ensure_connection():
            return False
            
        result = self.db.execute_query(query, params)
        return result is not None and result[0]['affected_rows'] > 0
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a play session by ID.
        
        Args:
            session_id: The session ID to retrieve
            
        Returns:
            Dictionary containing session data if found, None otherwise
        """
        query = """
            SELECT *
            FROM play_sessions
            WHERE session_id = %s
        """
        
        if not self.db.ensure_connection():
            return None
            
        result = self.db.execute_query(query, (session_id,))
        if result and len(result) > 0:
            return result[0]
        return None
    
    def get_user_sessions(self, user_id: str) -> list:
        """
        Retrieve all play sessions for a user.
        
        Args:
            user_id: The user ID to get sessions for
            
        Returns:
            List of session dictionaries
        """
        query = """
            SELECT *
            FROM play_sessions
            WHERE user_id = %s
            ORDER BY creation_time DESC
        """
        
        if not self.db.ensure_connection():
            return []
            
        result = self.db.execute_query(query, (user_id,))
        return result if result else [] 