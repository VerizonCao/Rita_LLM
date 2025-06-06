#!/usr/bin/env python3
"""
Database access layer for chat sessions in Rita LLM backend.
Handles PostgreSQL operations for chat session management.
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
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

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database connection and management class for chat sessions."""
    
    def __init__(self):
        """Initialize database manager with connection parameters."""
        self.connection: Optional[psycopg2.extensions.connection] = None
        self.postgres_url = os.getenv('POSTGRES_URL')
        
        if not self.postgres_url:
            raise ValueError("POSTGRES_URL environment variable is required")
    
    def connect(self) -> bool:
        """Establish database connection with automatic retry."""
        try:
            self.connection = psycopg2.connect(
                self.postgres_url,
                cursor_factory=RealDictCursor
            )
            return True
        except psycopg2.Error as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close database connection safely."""
        if self.connection:
            try:
                self.connection.close()
            except Exception as e:
                logger.warning(f"Error closing database connection: {e}")
            finally:
                self.connection = None
    
    def execute_query(self, query: str, params: tuple = None) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a SQL query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters tuple
            
        Returns:
            List of dictionaries for SELECT queries, or affected row count for others
        """
        if not self.connection:
            logger.error("No database connection available")
            return None
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                
                # Check if this is a data modification query (INSERT, UPDATE, DELETE)
                query_upper = query.strip().upper()
                is_modification = any(query_upper.startswith(cmd) for cmd in ['INSERT', 'UPDATE', 'DELETE'])
                
                if query_upper.startswith('SELECT'):
                    # Pure SELECT query - just return results
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
                elif is_modification:
                    # INSERT/UPDATE/DELETE - always commit the transaction
                    self.connection.commit()
                    
                    # If the query has RETURNING, fetch and return results
                    if 'RETURNING' in query.upper():
                        results = cursor.fetchall()
                        return [dict(row) for row in results]
                    else:
                        # No RETURNING clause - return affected row count
                        return [{"affected_rows": cursor.rowcount}]
                else:
                    # Other types of queries (CREATE, DROP, etc.)
                    self.connection.commit()
                    return [{"affected_rows": cursor.rowcount}]
                    
        except psycopg2.Error as e:
            logger.error(f"Query execution failed: {e}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def ensure_connection(self) -> bool:
        """Ensure database connection is active, reconnect if needed."""
        if not self.connection or self.connection.closed:
            return self.connect()
        return True
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect() 