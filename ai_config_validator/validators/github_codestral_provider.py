"""
Codestral 2501 Provider (GitHub Models).
Best for: Code generation, debugging, refactoring, and technical analysis.
"""

from .github_model_provider import GitHubModelProvider
from .base import ProviderConfig

class GitHubCodestralProvider(GitHubModelProvider):
    """
    Provider for Mistral Codestral 2501.
    Specialized for code-related tasks.
    """
    
    def __init__(self):
        super().__init__(ProviderConfig(
            name="Codestral",
            model="Codestral-2501",
            timeout=60.0,  # Code generation can be slow
            temperature=0.2,  # Lower temperature for precise code
            max_tokens=4096
        ))
