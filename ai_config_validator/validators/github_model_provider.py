"""
Base provider for GitHub Models.
Handles authentication, quota tracking, and common error handling.
"""

import time
import logging
from typing import List, Dict, Optional, Any
from .base import BaseProvider, ProviderResponse, ProviderConfig
from config import config
from src.utils.quota_tracker import quota_tracker

logger = logging.getLogger(__name__)

class GitHubModelProvider(BaseProvider):
    """
    Base class for all GitHub Models providers.
    """
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = self._get_client()
        self._available = self.client is not None
        self.quota_tracker = quota_tracker
        
    def _get_client(self):
        """Get authenticated OpenAI client for GitHub Models"""
        try:
            return config.get_github_models_client()
        except Exception as e:
            logger.error(f"Failed to initialize GitHub Models client: {e}")
            return None

    @property
    def model_name(self) -> str:
        """Get model name (ID)"""
        return self.config.model

    @property
    def is_available(self) -> bool:
        """Check if model is available (quota & circuit breaker)"""
        # 1. Check Circuit Breaker (via base class logic if needed, but base class uses this property too)
        # BaseProvider.is_available checks self._available (client exists) and circuit_breaker.can_attempt()
        if not super().is_available:
            return False
            
        # 2. Check Feature Flag
        if not config.ENABLE_GITHUB_MODELS_ROUTING:
            return False
            
        # 3. Check Quota
        available, reason = self.quota_tracker.check_availability(self.model_name)
        if not available:
            # We don't log warning here to avoid spamming logs on every check
            return False
            
        return True

    async def _generate_impl(self, messages: List[Dict[str, Any]]) -> Optional[ProviderResponse]:
        """
        Internal implementation of generate.
        """
        if not self.client:
            raise RuntimeError("GitHub Models client not initialized")
            
        # Quota check (double check before spending tokens)
        available, reason = self.quota_tracker.check_availability(self.model_name)
        if not available:
            raise RuntimeError(f"Quota exceeded: {reason}")

        # Call API
        # Note: timeouts are handled by BaseProvider.generate wrapping this in wait_for
        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout # Pass timeout to client just in case, though base handles it too
        )
        
        # Extract content
        content = response.choices[0].message.content
        if not content:
            return None
            
        # Calculate metrics
        tokens_used = response.usage.total_tokens if response.usage else 0
        
        # Track Quota Usage
        self.quota_tracker.increment_usage(self.model_name, tokens_used)
        
        return ProviderResponse(
            content=content,
            latency_ms=0, # Base class calculates this
            tokens_used=tokens_used,
            model_name=self.model_name,
            provider=self.config.name
        )
