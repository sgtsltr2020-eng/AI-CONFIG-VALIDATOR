"""
Llama 3.2 90B Vision Instruct Provider (GitHub Models).
Best for: Image analysis, screenshot understanding, OCR, visual reasoning.
"""

from .github_model_provider import GitHubModelProvider
from .base import ProviderConfig

class GitHubLlamaVisionProvider(GitHubModelProvider):
    """
    Provider for Llama 3.2 90B Vision Instruct.
    Supports both text and image inputs.
    """
    
    def __init__(self):
        super().__init__(ProviderConfig(
            name="Llama 3.2 Vision",
            model="meta/Llama-3.2-90B-Vision-Instruct",  # From user's screenshot
            timeout=45.0,  # Vision tasks may take longer
            temperature=0.7,
            max_tokens=2048  # Vision responses can be detailed
        ))
