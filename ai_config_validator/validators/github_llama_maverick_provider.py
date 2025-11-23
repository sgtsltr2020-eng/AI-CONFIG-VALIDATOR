"""
Llama 4 Maverick 170B Provider (GitHub Models).
Best for: Conversational excellence, complex reasoning.
"""

from .github_model_provider import GitHubModelProvider
from .base import ProviderConfig

class GitHubLlamaMaverickProvider(GitHubModelProvider):
    """
    Provider for Llama 4 Maverick 170B.
    """
    
    def __init__(self):
        super().__init__(ProviderConfig(
            name="Llama 4 Maverick",
            model="meta/Llama-4-Maverick-17B-128E-Instruct-FP8",  # Exact model ID from user
            timeout=30.0,
            temperature=0.7,
            max_tokens=1024
        ))
