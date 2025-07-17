#!/usr/bin/env python3
"""
Rita LLM Chat System Data Package

Simplified chat persistence system with two core operations:
1. read_history() - Load conversation history for LLM
2. write_message() - Write/append messages (creates session if needed)

Example usage:
    from data import ChatSessionManager
    
    chat_manager = ChatSessionManager()
    
    # Load history for LLM
    history = chat_manager.read_history(user_id, avatar_id)
    
    # Write user message  
    chat_manager.write_user_message(user_id, avatar_id, "Hello!")
    
    # Write assistant message
    chat_manager.write_assistant_message(user_id, avatar_id, "Hi there!", model="gpt-4")
"""

from .database import DatabaseManager
from .message import ChatMessage, UserMessage, AssistantMessage
from .chat_session import ChatSessionManager
from .play_session import PlaySessionManager
from .s3 import s3_manager, S3Manager
from .redis import redis_manager, RedisManager

__all__ = [
    'ChatSessionManager',
    'ChatMessage',
    'UserMessage',
    'AssistantMessage',
    'DatabaseManager',
    'PlaySessionManager',
    's3_manager',
    'S3Manager',
    'redis_manager',
    'RedisManager'
] 