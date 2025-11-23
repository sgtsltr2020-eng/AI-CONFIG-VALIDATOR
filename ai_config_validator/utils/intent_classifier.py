"""Intent classification for smart routing."""

from enum import Enum
from typing import Tuple
import re


class UserIntent(Enum):
    """User intent types for intelligent routing."""
    
    STORAGE = "storage"          # User wants to save/remember something
    RETRIEVAL = "retrieval"      # User asking about past info
    QUESTION = "question"        # General question/conversation
    ACTION = "action"            # Command (start game, schedule, etc)
    META = "meta"                # Talking about Vesper itself


class IntentClassifier:
    """Fast intent classification using pattern matching."""
    
    def __init__(self):
        self.storage_patterns = [
            "save", "remember", "store", "keep track", "note that",
            "write down", "record", "log", "file", "add to"
        ]
        
        self.retrieval_patterns = [
            "what did", "do you remember", "tell me about", "when did",
            "show me", "find", "search for", "look up", "retrieve",
            "did i", "have i", "who is", "what was"
        ]
        
        self.action_patterns = [
            "start", "begin", "schedule", "set reminder", "create",
            "launch", "open", "run", "execute"
        ]
        
        self.meta_patterns = [
            "how do you", "what can you", "your capabilities", "about you",
            "who are you", "what are you", "how does vesper"
        ]
    
    def classify_fast(self, message: str) -> Tuple[UserIntent, float]:
        """
        Quick classification using pattern matching (no LLM call).
        
        Args:
            message: User message to classify
            
        Returns:
            (intent, confidence) tuple
        """
        msg_lower = message.lower()
        
        # Check storage patterns
        if any(p in msg_lower for p in self.storage_patterns):
            return UserIntent.STORAGE, 0.9
        
        # Check retrieval patterns
        if any(p in msg_lower for p in self.retrieval_patterns):
            return UserIntent.RETRIEVAL, 0.9
        
        # Check action patterns
        if any(p in msg_lower for p in self.action_patterns):
            return UserIntent.ACTION, 0.8
        
        # Check meta patterns
        if any(p in msg_lower for p in self.meta_patterns):
            return UserIntent.META, 0.85
        
        # Default to question (most common)
        return UserIntent.QUESTION, 0.5
    
    def is_greeting(self, message: str) -> bool:
        """Check if message is a simple greeting."""
        greetings = {"hi", "hello", "hey", "yo", "sup", "hiya", "greetings"}
        return message.lower().strip() in greetings
    
    def is_goodbye(self, message: str) -> bool:
        """Check if message is a goodbye."""
        goodbyes = {"bye", "goodbye", "see you", "later", "cya", "farewell"}
        return message.lower().strip() in goodbyes
    
    def is_simple_acknowledgment(self, message: str) -> bool:
        """Check if message is a simple acknowledgment."""
        acks = {"ok", "okay", "k", "thanks", "thank you", "got it", "cool"}
        return message.lower().strip() in acks
