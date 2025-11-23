"""
Microsoft Phi-4 Provider (GitHub Models).
Best for: Efficient reasoning, context compression, and quick tasks.
"""

from .github_model_provider import GitHubModelProvider
from .base import ProviderConfig

class GitHubPhi4Provider(GitHubModelProvider):
    """
    Provider for Microsoft Phi-4.
    Specialized for efficiency and strong reasoning in a small package.
    """
    
    def __init__(self):
        super().__init__(ProviderConfig(
            name="Phi-4",
            model="Phi-4",
            timeout=30.0,
            temperature=0.6,
            max_tokens=2048
        ))
