"""
Llama 4 Scout 17B 16E Instruct Provider (GitHub Models).
Best for: Multi-document summarization, extensive user activity parsing.
"""

from .github_model_provider import GitHubModelProvider
from .base import ProviderConfig

class GitHubLlamaScoutProvider(GitHubModelProvider):
    """
    Provider for Llama 4 Scout 17B 16E Instruct.
    """
    
    def __init__(self):
        super().__init__(ProviderConfig(
            name="Llama 4 Scout",
            model="meta/Llama-4-Scout-17B-16E-Instruct",  # Exact model ID from user
            timeout=30.0,
            temperature=0.7,
            max_tokens=1024
        ))
