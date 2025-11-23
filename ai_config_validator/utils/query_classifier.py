"""Query complexity classification."""

from enum import Enum


class QueryComplexity(str, Enum):
    """Query complexity levels."""
    SIMPLE = "SIMPLE"
    COMPLEX = "COMPLEX"


class QueryClassifier:
    """Classifies query complexity for optimal routing."""
    
    # SIMPLE: Greetings and acknowledgments
    # EXACT COPY from classify_query_complexity() lines 1277-1281
    SIMPLE_PATTERNS = [
        "test", "hello", "hi", "hey", "thanks", "thank you", "ok", "okay",
        "yes", "no", "yep", "nope", "good", "bad", "cool", "nice", 
        "lol", "haha", "sure", "alright"
    ]
    
    # COMPLEX: Research/search queries (need web search)
    # EXACT COPY from classify_query_complexity() lines 1286-1291
    COMPLEX_KEYWORDS = [
        "explain", "analyze", "research", "compare", "latest", "news",
        "how does", "how do", "why does", "why do", "what is", "what are",
        "who is", "when did", "where is", "search for", "find", "look up",
        "tell me about", "what's the", "describe", "define"
    ]
    
    # COMPLEX: Questions about recent/current events
    # EXACT COPY from classify_query_complexity() line 1296
    TIME_KEYWORDS = ["today", "now", "latest", "current", "recent", "this week", "yesterday"]
    
    # COMPLEX: Code generation/debugging
    # EXACT COPY from classify_query_complexity() line 1301
    CODE_KEYWORDS = ["write code", "debug", "fix this", "error", "implement", "function", "class", "def "]
    
    @staticmethod
    def classify(message: str) -> QueryComplexity:
        """
        Classify query as SIMPLE or COMPLEX.
        
        SIMPLE → Fast models (Groq, GitHub, Gemini)
        COMPLEX → Smart models (Perplexity, Gemini, Groq, GitHub)
        
        This matches the EXACT logic from classify_query_complexity() in main.py (lines 1263-1314)
        
        Args:
            message: User's query message
        
        Returns:
            QueryComplexity enum (SIMPLE or COMPLEX)
        """
        message_lower = message.lower().strip()
        message_len = len(message_lower)
        
        # SIMPLE: Very short queries (< 15 chars, no question mark)
        # EXACT COPY from classify_query_complexity() lines 1272-1274
        if message_len < 15 and "?" not in message_lower:
            return QueryComplexity.SIMPLE
        
        # SIMPLE: Greetings and acknowledgments
        # EXACT COPY from classify_query_complexity() lines 1276-1283
        if any(pattern == message_lower or message_lower.startswith(f"{pattern} ") for pattern in QueryClassifier.SIMPLE_PATTERNS):
            return QueryComplexity.SIMPLE
        
        # COMPLEX: Research/search queries (need web search)
        # EXACT COPY from classify_query_complexity() lines 1285-1293
        if any(keyword in message_lower for keyword in QueryClassifier.COMPLEX_KEYWORDS):
            return QueryComplexity.COMPLEX
        
        # COMPLEX: Questions about recent/current events
        # EXACT COPY from classify_query_complexity() lines 1295-1298
        if "?" in message_lower and any(word in message_lower for word in QueryClassifier.TIME_KEYWORDS):
            return QueryComplexity.COMPLEX
        
        # COMPLEX: Code generation/debugging
        # EXACT COPY from classify_query_complexity() lines 1300-1303
        if any(keyword in message_lower for keyword in QueryClassifier.CODE_KEYWORDS):
            return QueryComplexity.COMPLEX
        
        # COMPLEX: Longer questions (likely need detailed answers)
        # EXACT COPY from classify_query_complexity() lines 1305-1307
        if "?" in message_lower and message_len > 30:
            return QueryComplexity.COMPLEX
        
        # COMPLEX: Long messages (> 60 chars) default to smart models
        # EXACT COPY from classify_query_complexity() lines 1309-1311
        if message_len > 60:
            return QueryComplexity.COMPLEX
        
        # DEFAULT: Medium queries default to SIMPLE (most things are simple)
        # EXACT COPY from classify_query_complexity() line 1314
        return QueryComplexity.SIMPLE

