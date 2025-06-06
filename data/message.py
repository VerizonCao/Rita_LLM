#!/usr/bin/env python3
"""
Chat message data structures for Rita LLM backend.
Defines message classes for user and assistant roles with JSON serialization.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union, List
from abc import ABC, abstractmethod

def generate_short_id() -> str:
    """Generate a short unique ID for messages."""
    return str(uuid.uuid4()).replace('-', '')[:12]

class ChatMessage(ABC):
    """Abstract base class for chat messages."""
    
    def __init__(
        self,
        content: str,
        sender_id: str,
        sender_name: str,
        role: str,
        message_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        deleted: bool = False,
        filtered: bool = False,
        context: int = -1
    ):
        """
        Initialize chat message.
        
        Args:
            content: The message content
            sender_id: ID of the sender (user_id or avatar_id)
            sender_name: Display name of the sender
            role: Message role ('user' or 'assistant')
            message_id: Unique message ID (auto-generated if None)
            created_at: Message creation timestamp (auto-generated if None)
            deleted: Whether message is deleted
            filtered: Whether message is filtered
            context: Context index for the message
        """
        self.id = message_id or generate_short_id()
        self.content = content
        self.role = role
        self.sender_id = sender_id
        self.sender_name = sender_name
        self.created_at = created_at or datetime.now(timezone.utc)
        self.deleted = deleted
        self.filtered = filtered
        self.context = context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "role": self.role,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "created_at": self.created_at.isoformat(),
            "deleted": self.deleted,
            "filtered": self.filtered,
            "context": self.context
        }
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create message instance from dictionary."""
        # Parse created_at if it's a string
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        
        # Determine the correct subclass based on role
        role = data.get('role', 'user')
        if role == 'user':
            return UserMessage.from_dict(data)
        elif role == 'assistant':
            return AssistantMessage.from_dict(data)
        else:
            raise ValueError(f"Unknown message role: {role}")

class UserMessage(ChatMessage):
    """Message from a user."""
    
    def __init__(
        self,
        content: str,
        sender_id: str,  # user_id
        sender_name: str,  # user display name
        message_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        deleted: bool = False,
        filtered: bool = False,
        context: int = -1
    ):
        """Initialize user message."""
        super().__init__(
            content=content,
            sender_id=sender_id,
            sender_name=sender_name,
            role="user",
            message_id=message_id,
            created_at=created_at,
            deleted=deleted,
            filtered=filtered,
            context=context
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserMessage':
        """Create UserMessage from dictionary."""
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        
        return cls(
            content=data['content'],
            sender_id=data['sender_id'],
            sender_name=data['sender_name'],
            message_id=data.get('id'),
            created_at=created_at,
            deleted=data.get('deleted', False),
            filtered=data.get('filtered', False),
            context=data.get('context', -1)
        )

class AssistantMessage(ChatMessage):
    """Message from an AI assistant/avatar."""
    
    def __init__(
        self,
        content: str,
        sender_id: str,  # avatar_id
        sender_name: str,  # avatar display name
        model: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        out_tokens: Optional[int] = None,
        message_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        deleted: bool = False,
        filtered: bool = False,
        context: int = -1
    ):
        """
        Initialize assistant message.
        
        Args:
            content: The message content
            sender_id: Avatar ID
            sender_name: Avatar display name
            model: LLM model used
            prompt_tokens: Number of input tokens
            out_tokens: Number of output tokens
            message_id: Unique message ID
            created_at: Message creation timestamp
            deleted: Whether message is deleted
            filtered: Whether message is filtered
            context: Context index for the message
        """
        super().__init__(
            content=content,
            sender_id=sender_id,
            sender_name=sender_name,
            role="assistant",
            message_id=message_id,
            created_at=created_at,
            deleted=deleted,
            filtered=filtered,
            context=context
        )
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.out_tokens = out_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert assistant message to dictionary with LLM-specific fields."""
        base_dict = super().to_dict()
        base_dict.update({
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "out_tokens": self.out_tokens
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssistantMessage':
        """Create AssistantMessage from dictionary."""
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        
        return cls(
            content=data['content'],
            sender_id=data['sender_id'],
            sender_name=data['sender_name'],
            model=data.get('model'),
            prompt_tokens=data.get('prompt_tokens'),
            out_tokens=data.get('out_tokens'),
            message_id=data.get('id'),
            created_at=created_at,
            deleted=data.get('deleted', False),
            filtered=data.get('filtered', False),
            context=data.get('context', -1)
        )

def parse_messages_from_jsonb(jsonb_data: Union[str, list]) -> List[ChatMessage]:
    """
    Parse messages from JSONB data.
    
    Args:
        jsonb_data: JSONB data as string or list
        
    Returns:
        List of ChatMessage instances
    """
    if isinstance(jsonb_data, str):
        messages_data = json.loads(jsonb_data)
    else:
        messages_data = jsonb_data
    
    messages = []
    for msg_data in messages_data:
        try:
            message = ChatMessage.from_dict(msg_data)
            messages.append(message)
        except Exception as e:
            print(f"Error parsing message: {e}, data: {msg_data}")
            continue
    
    return messages

def messages_to_jsonb(messages: List[ChatMessage]) -> str:
    """
    Convert list of messages to JSONB string.
    
    Args:
        messages: List of ChatMessage instances
        
    Returns:
        JSON string for JSONB storage
    """
    return json.dumps([msg.to_dict() for msg in messages]) 