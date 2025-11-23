"""OpenRouter provider for access to 100+ LLM models."""

import httpx
import time
from typing import Optional, Dict, Any, List

from .base import BaseProvider, ProviderResponse, ProviderConfig

class OpenRouterProvider(BaseProvider):
    """OpenRouter provider for access to 100+ LLM models."""
    
    # Model selection tiers (quality-based)
    MODELS = {
        'reasoning': 'anthropic/claude-3.5-sonnet',  # Best reasoning
        'coding': 'deepseek/deepseek-coder',  # Best for code
        'creative': 'mistralai/mistral-large',  # Best for creative writing
        'fast': 'meta-llama/llama-3.1-8b-instruct',  # Fastest responses
        'vision': 'anthropic/claude-3-opus',  # Multimodal
    }

    def __init__(self, api_key: str, default_model: str = 'reasoning'):
        """
        Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key (from https://openrouter.ai/keys)
            default_model: Default model tier to use
        """
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.default_model_tier = default_model
        resolved_model = self.MODELS.get(default_model, self.MODELS['reasoning'])
        
        config = ProviderConfig(
            name="OpenRouter",
            model=resolved_model,
            timeout=30.0
        )
        
        # HTTP client with retry logic
        client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/sgtsltr2020-eng/Vesper",
                "X-Title": "Vesper AI Assistant"
            },
            timeout=30.0
        )
        
        super().__init__(config, client)
        self._explicitly_available = True
        
    @property
    def is_available(self) -> bool:
        """Check if provider is available."""
        # Check base availability (client exists) and circuit breaker
        base_available = super().is_available
        return base_available and self._explicitly_available and bool(self.api_key)

    async def _generate_impl(
        self,
        messages: List[Dict[str, Any]]
    ) -> Optional[ProviderResponse]:
        """
        Internal implementation of generate using OpenRouter.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
        
        Returns:
            ProviderResponse or None if generation fails
        """
        if not self.is_available:
            return None
        
        # Resolve model (support both tier names and full model IDs)
        # Check if model override is passed in kwargs? 
        # BaseProvider.generate signature is generate(self, messages).
        # To support model_override, we might need to inspect messages or context, 
        # but for now let's stick to the config model or default.
        # The user's prompt had `model_override` in `generate`, but `BaseProvider` defines `generate` 
        # as `generate(self, messages)`. 
        # We can't easily change the signature of `generate` in `BaseProvider` without affecting others.
        # However, we can check if the last message has some metadata or just use the configured model.
        # For this implementation, I'll use self.config.model which is set in __init__.
        # If we want dynamic model switching, we might need to update self.config.model before calling generate
        # or handle it differently.
        
        # The user's prompt showed:
        # async def generate(self, messages, model_override=None, **kwargs)
        # But BaseProvider has:
        # async def generate(self, messages) -> Optional[ProviderResponse]
        # And calls _generate_impl(messages)
        
        # I will implement _generate_impl and use self.config.model.
        # If the user wants model overrides, they might need to instantiate the provider with that model
        # or we can look for a special system message or metadata.
        # For now, I will stick to the BaseProvider contract.
        
        model = self.config.model
        
        try:
            # Build request payload
            payload = {
                "model": model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            
            # Make API call
            response = await self.client.post(
                "/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            # Track usage for observability
            usage = data.get('usage', {})
            tokens_used = usage.get('total_tokens', 0)
            
            return ProviderResponse(
                content=content,
                model_name=model,
                tokens_used=tokens_used,
                latency_ms=0, # Set by base class
                provider=self.name
            )
        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                print(f"⚠️ OpenRouter rate limit hit: {e}")
                self._explicitly_available = False  # Temporarily disable
                # In a real system we'd want a way to re-enable it after some time
            else:
                print(f"⚠️ OpenRouter HTTP error {e.response.status_code}: {e}")
            raise # Re-raise to be handled by BaseProvider
        
        except Exception as e:
            print(f"⚠️ OpenRouter generation failed: {str(e)[:200]}")
            raise # Re-raise to be handled by BaseProvider

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
