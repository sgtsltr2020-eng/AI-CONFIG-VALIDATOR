"""
OpenAI Text Embedding 3 Provider (GitHub Models).
Best for: Semantic search, context retrieval, and memory management.
"""

import logging
from typing import List, Optional
from .base import BaseProvider, ProviderConfig, ProviderResponse

logger = logging.getLogger(__name__)

class GitHubEmbeddingProvider(BaseProvider):
    """
    Provider for OpenAI Text Embedding 3 (via GitHub Models).
    Note: This provider returns embeddings (list of floats), not text.
    """
    
    def __init__(self, client):
        """
        Initialize with GitHub Models client.
        
        Args:
            client: Initialized OpenAI client configured for GitHub Models
        """
        super().__init__(ProviderConfig(
            name="Text Embedding 3 Small",
            model="text-embedding-3-small",
            timeout=10.0,
            temperature=0.0,  # Not used for embeddings
            max_tokens=8191   # Max input tokens
        ))
        self.client = client

    async def generate(self, prompt: str) -> ProviderResponse:
        """
        Generate embeddings for the given text.
        """
        if not self.client:
            return ProviderResponse(
                content="",
                raw_response={},
                usage={"error": "Client not initialized"}
            )

        try:
            # Create embedding
            response = await self.client.embeddings.create(
                input=prompt,
                model="text-embedding-3-small"
            )
            
            # Extract embedding vector
            embedding = response.data[0].embedding
            
            return ProviderResponse(
                content="",  # No text content for embeddings
                model_name="text-embedding-3-small",
                raw_response={"embedding": embedding},
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"GitHub Embedding error: {e}")
            return ProviderResponse(
                content="",
                raw_response={},
                usage={"error": str(e)}
            )

    async def _generate_impl(self, messages: List[dict]) -> ProviderResponse:
        """
        Dummy implementation to satisfy BaseProvider abstract method.
        Not used because generate() is overridden.
        """
        raise NotImplementedError("Use generate(prompt: str) for embeddings")
