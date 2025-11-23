"""
Anthropic (Claude) API validator with auto-discovery support.

Validates API keys, discovers available models, and fetches account info.
"""

import re
import requests
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Pattern

from ..errors import (
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
)
from ..logging_config import get_logger
from ..models import (
    ProviderCapabilities,
    ProviderType,
    ValidationResult,
    ValidationStatus,
)
from .base import BaseValidator

logger = get_logger(__name__)


# Static Anthropic Model Catalog (for cost data and capabilities)
ANTHROPIC_MODELS: Dict[str, ProviderCapabilities] = {
    # Claude Sonnet 4 (Latest - Feb 2025)
    "claude-sonnet-4-20250514": ProviderCapabilities(
        provider=ProviderType.ANTHROPIC,
        model="claude-sonnet-4-20250514",
        max_tokens=200000,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.015,
        rate_limit_rpm=4000,
        rate_limit_tpm=400000,
    ),
    # Claude 3.5 Sonnet (Oct 2024)
    "claude-3-5-sonnet-20241022": ProviderCapabilities(
        provider=ProviderType.ANTHROPIC,
        model="claude-3-5-sonnet-20241022",
        max_tokens=200000,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.015,
        rate_limit_rpm=4000,
        rate_limit_tpm=400000,
    ),
    # Claude 3.5 Haiku
    "claude-3-5-haiku-20241022": ProviderCapabilities(
        provider=ProviderType.ANTHROPIC,
        model="claude-3-5-haiku-20241022",
        max_tokens=200000,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=False,
        cost_per_1k_input_tokens=0.001,
        cost_per_1k_output_tokens=0.005,
        rate_limit_rpm=4000,
        rate_limit_tpm=400000,
    ),
    # Claude 3 Opus
    "claude-3-opus-20240229": ProviderCapabilities(
        provider=ProviderType.ANTHROPIC,
        model="claude-3-opus-20240229",
        max_tokens=200000,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        cost_per_1k_input_tokens=0.015,
        cost_per_1k_output_tokens=0.075,
        rate_limit_rpm=4000,
        rate_limit_tpm=400000,
    ),
    # Claude 3 Sonnet
    "claude-3-sonnet-20240229": ProviderCapabilities(
        provider=ProviderType.ANTHROPIC,
        model="claude-3-sonnet-20240229",
        max_tokens=200000,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.015,
        rate_limit_rpm=4000,
        rate_limit_tpm=400000,
    ),
    # Claude 3 Haiku
    "claude-3-haiku-20240307": ProviderCapabilities(
        provider=ProviderType.ANTHROPIC,
        model="claude-3-haiku-20240307",
        max_tokens=200000,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        cost_per_1k_input_tokens=0.00025,
        cost_per_1k_output_tokens=0.00125,
        rate_limit_rpm=4000,
        rate_limit_tpm=400000,
    ),
}

FEATURED_MODELS = {
    "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
}


class AnthropicValidator(BaseValidator):
    """
    Validator for Anthropic (Claude) API configurations.
    
    Features:
    - Complete model catalog (6 models)
    - Auto-discovery of user's accessible models
    - Cost estimation and rate limits
    - Intelligent typo suggestions
    """

    @classmethod
    def provider_type(cls) -> ProviderType:
        """Return Anthropic as the provider type."""
        return ProviderType.ANTHROPIC

    @classmethod
    def api_key_pattern(cls) -> Pattern[str]:
        """
        Anthropic API key pattern.
        Format: sk-ant-api03-<95+ characters>
        """
        return re.compile(r"^sk-ant-api\d{2}-[a-zA-Z0-9_-]{95,}$")

    def discover_available_models(self) -> List[str]:
        """
        Fetch models available to this API key via Anthropic API.
        
        Uses the /v1/models endpoint to get user's accessible models.
        
        Returns:
            List of model IDs the user can access
            
        Raises:
            AuthenticationError: If API key is invalid
            APIConnectionError: If cannot connect
            RateLimitError: If rate limit exceeded
        """
        logger.info(
            "Discovering available models from Anthropic API",
            extra={"provider": "anthropic"}
        )
        
        try:
            response = requests.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": self.config.api_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=10,
            )
            
            if response.status_code == 401:
                logger.error("API key authentication failed", extra={"provider": "anthropic"})
                raise AuthenticationError(provider="anthropic", status_code=401)
            
            if response.status_code == 429:
                retry_after = int(response.headers.get("retry-after", 60))
                logger.warning(
                    "Rate limit hit during discovery",
                    extra={"provider": "anthropic", "retry_after": retry_after}
                )
                raise RateLimitError(provider="anthropic", retry_after=retry_after)
            
            response.raise_for_status()
            data = response.json()
            
            # Extract model IDs from response
            models = [item["id"] for item in data.get("data", [])]
            
            logger.info(
                f"Discovered {len(models)} models",
                extra={"provider": "anthropic", "count": len(models)}
            )
            
            return sorted(models)
            
        except requests.RequestException as e:
            logger.error(f"Model discovery failed: {e}", extra={"provider": "anthropic"})
            raise APIConnectionError(provider="anthropic", reason=str(e))

    def get_enriched_model_info(self, discovered_models: List[str]) -> List[Dict[str, Any]]:
        """
        Enrich discovered models with metadata from static catalog.
        
        Args:
            discovered_models: Model IDs from API
            
        Returns:
            List of dicts with full model metadata
        """
        enriched = []
        
        for model_id in discovered_models:
            info = {
                "id": model_id,
                "name": model_id,
                "featured": model_id in FEATURED_MODELS,
            }
            
            if model_id in ANTHROPIC_MODELS:
                cap = ANTHROPIC_MODELS[model_id]
                info.update({
                    "max_tokens": cap.max_tokens,
                    "supports_streaming": cap.supports_streaming,
                    "supports_function_calling": cap.supports_function_calling,
                    "supports_vision": cap.supports_vision,
                    "cost_per_1k_input": cap.cost_per_1k_input_tokens,
                    "cost_per_1k_output": cap.cost_per_1k_output_tokens,
                    "rate_limit_rpm": cap.rate_limit_rpm,
                })
            else:
                info["unknown"] = True
            
            enriched.append(info)
        
        enriched.sort(key=lambda x: (not x.get("featured", False), x["name"]))
        return enriched

    def get_account_info(self) -> Dict[str, Any]:
        """
        Fetch account information.
        
        Note: Anthropic doesn't expose billing info via API yet,
        so we return basic info.
        
        Returns:
            Dict with organization and tier information
        """
        logger.info("Fetching account information", extra={"provider": "anthropic"})
        
        # Anthropic doesn't have a billing endpoint yet
        # Return basic info that we can get
        return {
            "organization": "Anthropic Account",
            "tier": "API Access",
            "note": "Anthropic does not expose billing information via API",
            "balance": {
                "available_credits_usd": None,
                "usage_this_month_usd": None,
                "usage_percent": None,
            },
            "rate_limits": {
                "requests_per_minute": 4000,
                "tokens_per_minute": 400000,
            },
            "warnings": [],
        }

    def validate_model_local(self, model: str) -> ValidationResult:
        """Validate model name using static catalog."""
        model_clean = model.strip().lower()

        if model_clean in ANTHROPIC_MODELS:
            cap = ANTHROPIC_MODELS[model_clean]
            return ValidationResult(
                status=ValidationStatus.VALID,
                provider=ProviderType.ANTHROPIC,
                model=model_clean,
                message=f"Valid Anthropic model: {model_clean}",
                details={
                    "check_type": "local",
                    "max_tokens": cap.max_tokens,
                    "cost_per_1k_input": cap.cost_per_1k_input_tokens,
                    "cost_per_1k_output": cap.cost_per_1k_output_tokens,
                    "supports_streaming": cap.supports_streaming,
                    "supports_function_calling": cap.supports_function_calling,
                    "supports_vision": cap.supports_vision,
                },
            )

        # Model not found - suggest alternatives
        suggestion = self._suggest_model(model_clean)

        return ValidationResult(
            status=ValidationStatus.INVALID,
            provider=ProviderType.ANTHROPIC,
            model=model_clean,
            message=f"Model '{model_clean}' is not valid",
            suggestion=f"Did you mean '{suggestion}'?" if suggestion else None,
            details={
                "check_type": "local",
                "suggested_model": suggestion,
                "available_models": list(ANTHROPIC_MODELS.keys()),
            },
        )

    def _suggest_model(self, model: str) -> Optional[str]:
        """Fuzzy match to suggest closest valid model."""
        matches = get_close_matches(model, ANTHROPIC_MODELS.keys(), n=1, cutoff=0.6)
        if matches:
            return matches[0]

        # Fallback: keyword matching
        if "sonnet-4" in model or "claude-4" in model or ("claude" in model and "4" in model):
            return "claude-sonnet-4-20250514"
        elif "sonnet" in model and "3.5" in model:
            return "claude-3-5-sonnet-20241022"
        elif "haiku" in model:
            return "claude-3-5-haiku-20241022"
        elif "opus" in model:
            return "claude-3-opus-20240229"

        return None

    @classmethod
    def get_featured_models(cls) -> List[str]:
        """Get featured models for UI."""
        return sorted(FEATURED_MODELS)

    @classmethod
    def get_all_models(cls) -> List[str]:
        """Get all models."""
        return sorted(ANTHROPIC_MODELS.keys())
