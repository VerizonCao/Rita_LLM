#!/usr/bin/env python3
"""
Simplified chat session management for Rita LLM backend.
Contains ChatSession class for data structure and ChatSessionManager for operations.
"""

import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from .database import DatabaseManager
from .message import ChatMessage, UserMessage, AssistantMessage, parse_messages_from_jsonb

logger = logging.getLogger(__name__)

class ChatSessionManager:
    """
    Simplified chat session manager with only two operations:
    1. read_history() - Load conversation history 
    2. write_message() - Write/append messages (creates session if needed)
    """
    
    def __init__(self):
        """Initialize with database manager."""
        self.db_manager = DatabaseManager()
    
    def read_history(self, user_id: str, avatar_id: str, max_messages: int = 50) -> List[Dict[str, str]]:
        """
        Read conversation history formatted for LLM.
        
        Args:
            user_id: User identifier
            avatar_id: Avatar identifier  
            max_messages: Maximum number of recent messages
            
        Returns:
            List of dicts with 'role' and 'content' keys for LLM
        """
        try:
            if not self.db_manager.ensure_connection():
                logger.error("Failed to establish database connection")
                return []
            
            query = """
            SELECT messages
            FROM chat_sessions
            WHERE user_id = %s AND avatar_id = %s
            ORDER BY updated_time DESC
            LIMIT 1
            """
            
            result = self.db_manager.execute_query(query, (user_id, avatar_id))
            
            if not result or len(result) == 0:
                return []
            
            # Parse messages from JSONB
            messages_data = result[0].get('messages', [])
            messages = parse_messages_from_jsonb(messages_data)
            
            # Sort by creation time and limit
            messages = sorted(messages, key=lambda msg: msg.created_at)
            if max_messages:
                messages = messages[-max_messages:]
            
            # Convert to LLM format (role + content only)
            llm_messages = []
            for msg in messages:
                message_dict = {
                    'role': msg.role,
                    'content': msg.content
                }
                # Include imageUrl if present
                if hasattr(msg, 'imageUrl') and msg.imageUrl:
                    message_dict['imageUrl'] = msg.imageUrl
                llm_messages.append(message_dict)
            
            return llm_messages
            
        except Exception as e:
            logger.error(f"Error reading conversation history: {e}")
            return []

    def read_full_history(self, user_id: str, avatar_id: str) -> List[ChatMessage]:
        """
        Read full conversation history with all message fields.
        
        Args:
            user_id: User identifier
            avatar_id: Avatar identifier
            
        Returns:
            List of ChatMessage objects with all fields
        """
        try:
            if not self.db_manager.ensure_connection():
                logger.error("Failed to establish database connection")
                return []
            
            query = """
            SELECT messages
            FROM chat_sessions
            WHERE user_id = %s AND avatar_id = %s
            ORDER BY updated_time DESC
            LIMIT 1
            """
            
            result = self.db_manager.execute_query(query, (user_id, avatar_id))
            
            if not result or len(result) == 0:
                return []
            
            # Parse messages from JSONB
            messages_data = result[0].get('messages', [])
            messages = parse_messages_from_jsonb(messages_data)
            
            # Sort by creation time
            messages = sorted(messages, key=lambda msg: msg.created_at)
            
            return messages
            
        except Exception as e:
            logger.error(f"Error reading full conversation history: {e}")
            return []
    
    def write_message(
        self, 
        user_id: str, 
        avatar_id: str, 
        message: ChatMessage
    ) -> Optional[str]:
        """
        Write/append a message to the chat session.
        Creates session if it doesn't exist.
        
        Args:
            user_id: User identifier
            avatar_id: Avatar identifier
            message: ChatMessage to append
            
        Returns:
            Session ID if successful, None otherwise
        """
        try:
            if not self.db_manager.ensure_connection():
                logger.error("Failed to establish database connection")
                return None
            
            # Prepare message as JSONB array with single message
            message_jsonb = json.dumps([message.to_dict()])
            
            # Use INSERT with ON CONFLICT to append message
            query = """
            INSERT INTO chat_sessions (user_id, avatar_id, messages)
            VALUES (%s, %s, %s::jsonb)
            ON CONFLICT (user_id, avatar_id) DO
              UPDATE
                SET
                  messages     = chat_sessions.messages || EXCLUDED.messages,
                  updated_time = CURRENT_TIMESTAMP
            RETURNING chat_session_id;
            """
            
            result = self.db_manager.execute_query(
                query, 
                (user_id, avatar_id, message_jsonb)
            )
            
            if result and len(result) > 0:
                return result[0]['chat_session_id']
            
            return None
            
        except Exception as e:
            logger.error(f"Error writing message to session: {e}")
            return None
    
    def write_user_message(
        self, 
        user_id: str, 
        avatar_id: str, 
        content: str, 
        user_name: str = "User"
    ) -> Optional[str]:
        """
        Convenience method to write a user message.
        
        Args:
            user_id: User identifier
            avatar_id: Avatar identifier
            content: Message content
            user_name: User display name
            
        Returns:
            Session ID if successful, None otherwise
        """
        user_message = UserMessage(
            content=content,
            sender_id=user_id,
            sender_name=user_name
        )
        return self.write_message(user_id, avatar_id, user_message)
    
    def write_assistant_message(
        self, 
        user_id: str, 
        avatar_id: str, 
        content: str, 
        assistant_name: str = "Assistant",
        model: Optional[str] = None
    ) -> Optional[str]:
        """
        Convenience method to write an assistant message.
        
        Args:
            user_id: User identifier
            avatar_id: Avatar identifier
            content: Message content
            assistant_name: Assistant display name
            model: LLM model used
            
        Returns:
            Session ID if successful, None otherwise
        """
        assistant_message = AssistantMessage(
            content=content,
            sender_id=avatar_id,
            sender_name=assistant_name,
            model=model
        )
        return self.write_message(user_id, avatar_id, assistant_message)
    
    def write_assistant_message_with_image(
        self, 
        user_id: str, 
        avatar_id: str, 
        content: str, 
        imageUrl: str,
        assistant_name: str = "Assistant",
        model: Optional[str] = None
    ) -> Optional[str]:
        """
        Convenience method to write an assistant message with an image URL.
        
        Args:
            user_id: User identifier
            avatar_id: Avatar identifier
            content: Message content
            imageUrl: URL of the generated image
            assistant_name: Assistant display name
            model: LLM model used
            
        Returns:
            Session ID if successful, None otherwise
        """
        assistant_message = AssistantMessage(
            content=content,
            sender_id=avatar_id,
            sender_name=assistant_name,
            model=model,
            imageUrl=imageUrl
        )
        return self.write_message(user_id, avatar_id, assistant_message)

    def update_session_messages(
        self, 
        user_id: str, 
        avatar_id: str, 
        messages: List[ChatMessage]
    ) -> Optional[str]:
        """
        Update the entire messages list for a chat session.
        
        Args:
            user_id: User identifier
            avatar_id: Avatar identifier
            messages: List of ChatMessage objects to replace the current messages
            
        Returns:
            Session ID if successful, None otherwise
        """
        try:
            if not self.db_manager.ensure_connection():
                logger.error("Failed to establish database connection")
                return None
            
            # Convert messages to JSONB format
            from .message import messages_to_jsonb
            messages_jsonb = messages_to_jsonb(messages)
            
            # Use INSERT with ON CONFLICT to update the entire messages list
            query = """
            INSERT INTO chat_sessions (user_id, avatar_id, messages)
            VALUES (%s, %s, %s::jsonb)
            ON CONFLICT (user_id, avatar_id) DO
              UPDATE
                SET
                  messages     = EXCLUDED.messages,
                  updated_time = CURRENT_TIMESTAMP
            RETURNING chat_session_id;
            """
            
            result = self.db_manager.execute_query(
                query, 
                (user_id, avatar_id, messages_jsonb)
            )
            
            if result and len(result) > 0:
                return result[0]['chat_session_id']
            
            return None
            
        except Exception as e:
            logger.error(f"Error updating session messages: {e}")
            return None 