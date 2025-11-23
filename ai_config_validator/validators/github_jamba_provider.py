"""
AI21 Jamba 1.5 Large Provider (GitHub Models).
Best for: Long-context tasks, document analysis, and complex synthesis.
"""

from .github_model_provider import GitHubModelProvider
from .base import ProviderConfig

class GitHubJambaProvider(GitHubModelProvider):
    """
    Provider for AI21 Jamba 1.5 Large.
    Specialized for long-context reasoning (up to 256k tokens).
    """
    
    def __init__(self):
        super().__init__(ProviderConfig(
            name="Jamba 1.5 Large",
            model="ai21-labs/AI21-Jamba-1.5-Large",
            timeout=90.0,  # Long context takes time
            temperature=0.4,
            max_tokens=4096
        ))
