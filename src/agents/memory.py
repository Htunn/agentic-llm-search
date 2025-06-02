"""
Conversation memory module for the Agentic LLM Agent

Manages conversation history to support follow-up questions and maintain context
across multiple interactions.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ConversationMessage:
    """Represents a message in the conversation history"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConversationMemory:
    """Manages conversation history for an agent"""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation memory
        
        Args:
            max_history: Maximum number of exchanges to keep in memory
        """
        self.messages: List[ConversationMessage] = []
        self.max_history = max_history
        logger.info(f"Initialized conversation memory with max_history={max_history}")
    
    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a user message to the conversation history
        
        Args:
            content: User's message content
            metadata: Optional metadata associated with the message
        """
        self.messages.append(
            ConversationMessage(
                role="user",
                content=content,
                metadata=metadata or {}
            )
        )
        self._trim_history()
        logger.debug(f"Added user message to memory: {content[:50]}...")
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add an assistant message to the conversation history
        
        Args:
            content: Assistant's message content
            metadata: Optional metadata associated with the message
        """
        self.messages.append(
            ConversationMessage(
                role="assistant",
                content=content,
                metadata=metadata or {}
            )
        )
        self._trim_history()
        logger.debug(f"Added assistant message to memory: {content[:50]}...")
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[ConversationMessage]:
        """
        Get the conversation history
        
        Args:
            limit: Optional limit on the number of messages to return
                  (most recent messages are returned)
        
        Returns:
            List of conversation messages
        """
        if limit is None or limit >= len(self.messages):
            return self.messages
        return self.messages[-limit:]
    
    def format_for_context(self, max_tokens: Optional[int] = 1500) -> str:
        """
        Format the conversation history as context for the LLM
        
        Args:
            max_tokens: Approximate maximum number of tokens to include
                       (Each message roughly counts as content length / 4 tokens)
        
        Returns:
            Formatted conversation history for model context
        """
        if not self.messages:
            return ""
        
        context = "Conversation history:\n\n"
        token_count = 0
        messages_to_include = []
        
        # Start from most recent messages and go backwards
        for message in reversed(self.messages):
            # Rough estimate of tokens (characters / 4)
            estimated_tokens = len(message.content) // 4
            
            if token_count + estimated_tokens > max_tokens:
                break
                
            messages_to_include.append(message)
            token_count += estimated_tokens
        
        # Reverse back to chronological order
        messages_to_include.reverse()
        
        # Format messages
        for i, message in enumerate(messages_to_include):
            role_display = "User" if message.role == "user" else "Assistant"
            context += f"{role_display}: {message.content}\n\n"
        
        return context
    
    def clear(self):
        """Clear the conversation history"""
        self.messages = []
        logger.info("Cleared conversation memory")
    
    def _trim_history(self):
        """Trim history to max_history exchanges"""
        if len(self.messages) > self.max_history * 2:  # Each exchange has 2 messages
            # Keep the most recent messages
            self.messages = self.messages[-self.max_history * 2:]
            logger.debug(f"Trimmed conversation memory to {len(self.messages)} messages")
    
    def detect_follow_up(self, query: str) -> bool:
        """
        Detect if the user query is a follow-up question
        
        Args:
            query: User's query
        
        Returns:
            Boolean indicating if the query appears to be a follow-up
        """
        if not self.messages:
            return False
            
        # Simple heuristics for follow-up detection
        follow_up_indicators = [
            "what about", "and", "what else", "tell me more", "continue", 
            "furthermore", "additionally", "also", "why is that", "how come",
            "what is", "how does", "can you explain", "elaborate", "go on",
            "what if", "anything else", "more details", "follow up", 
            "related to", "on that note", "speaking of", "regarding that",
            "to that point", "based on that"
        ]
        
        query_lower = query.lower()
        
        # Check for pronouns that might indicate referencing previous content
        has_pronouns = any(p in query_lower.split() for p in [
            "it", "that", "this", "these", "those", "they", "them", "their", "its"
        ])
        
        # Check for follow-up phrases
        has_follow_up_phrase = any(phrase in query_lower for phrase in follow_up_indicators)
        
        # Check if the query is very short (likely a follow-up)
        is_short_query = len(query.split()) <= 5
        
        return has_pronouns or has_follow_up_phrase or is_short_query
