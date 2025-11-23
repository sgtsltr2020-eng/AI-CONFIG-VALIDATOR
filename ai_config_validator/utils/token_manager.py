"""Token management utilities for smart prompt engineering."""

from typing import Dict, List
import tiktoken


class TokenManager:
    """Manages token budgets and estimation."""
    
    def __init__(self):
        # Use tiktoken for accurate token estimation
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback to approximate estimation
            self.encoder = None
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def get_provider_budget(self, provider: str) -> Dict[str, int]:
        """
        Get token budget configuration for provider.
        
        Args:
            provider: Provider name (Groq, Gemini, GitHub)
            
        Returns:
            Budget configuration dict
        """
        budgets = {
            "Groq": {
                "total_context": 8000,
                "reserved_for_response": 2000,
                "available": 6000,
                "system_prompt": 50,      # Ultra-minimal for Groq
                "context_injection": 200,
                "history": 800,
            },
            "Gemini 2.0 Flash": {
                "total_context": 2000000,  # 2M tokens!
                "reserved_for_response": 4000,
                "available": 1996000,
                "system_prompt": 500,      # Can afford more
                "context_injection": 2000,
                "history": 5000,
            },
            "GitHub Models": {
                "total_context": 128000,
                "reserved_for_response": 4000,
                "available": 124000,
                "system_prompt": 200,
                "context_injection": 1000,
                "history": 2000,
            }
        }
        
        # Default to most restrictive (Groq)
        return budgets.get(provider, budgets["Groq"])
    
    def trim_to_budget(self, items: List[str], budget: int) -> List[str]:
        """
        Trim list of text items to fit within token budget.
        
        Args:
            items: List of text items to trim
            budget: Maximum tokens allowed
            
        Returns:
            Trimmed list of items that fit in budget
        """
        selected = []
        used = 0
        
        for item in items:
            tokens = self.estimate_tokens(item)
            if used + tokens <= budget:
                selected.append(item)
                used += tokens
            else:
                break
        
        return selected
    
    def fits_in_budget(self, text: str, budget: int) -> bool:
        """Check if text fits within token budget."""
        return self.estimate_tokens(text) <= budget
