"""
Google Gemini API validator with complete model catalog.

Validates API keys and model names for Google's Gemini models.
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


# Google Gemini Model Catalog (Updated Nov 2025)
GOOGLE_MODELS: Dict[str, ProviderCapabilities] = {
    # Gemini 3 Pro (Latest - Nov 2025)
    "gemini-3-pro-preview": ProviderCapabilities(
        provider=ProviderType.GOOGLE,
        model="gemini-3-pro-preview",
        max_tokens=1048576,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        cost_per_1k_input_tokens=2.5,
        cost_per_1k_output_tokens=10.0,
        rate_limit_rpm=360,
        rate_limit_tpm=1000000,
    ),
    # Gemini 2.5 Pro (Advanced Thinking - June 2025)
    "gemini-2.5-pro": ProviderCapabilities(
        provider=ProviderType.GOOGLE,
        model="gemini-2.5-pro",
        max_tokens=1048576,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        cost_per_1k_input_tokens=1.25,
        cost_per_1k_output_tokens=5.0,
        rate_limit_rpm=360,
        rate_limit_tpm=1000000,
    ),
    # Gemini 2.5 Flash (Fast & Balanced - June 2025)
    "gemini-2.5-flash": ProviderCapabilities(
        provider=ProviderType.GOOGLE,
        model="gemini-2.5-flash",
        max_tokens=1048576,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        cost_per_1k_input_tokens=0.075,
        cost_per_1k_output_tokens=0.3,
        rate_limit_rpm=1000,
        rate_limit_tpm=4000000,
    ),
    "gemini-2.5-flash-preview-09-2025": ProviderCapabilities(
        provider=ProviderType.GOOGLE,
        model="gemini-2.5-flash-preview-09-2025",
        max_tokens=1048576,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        cost_per_1k_input_tokens=0.075,
        cost_per_1k_output_tokens=0.3,
        rate_limit_rpm=1000,
        rate_limit_tpm=4000000,
    ),
    # Gemini 2.5 Flash-Lite (Ultra Fast - July 2025)
    "gemini-2.5-flash-lite": ProviderCapabilities(
        provider=ProviderType.GOOGLE,
        model="gemini-2.5-flash-lite",
        max_tokens=1048576,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        cost_per_1k_input_tokens=0.0375,
        cost_per_1k_output_tokens=0.15,
        rate_limit_rpm=1000,
        rate_limit_tpm=4000000,
    ),
    # Gemini 2.5 Flash Image (Image Generation - Oct 2025)
    "gemini-2.5-flash-image": ProviderCapabilities(
        provider=ProviderType.GOOGLE,
        model="gemini-2.5-flash-image",
        max_tokens=65536,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        cost_per_1k_input_tokens=0.075,
        cost_per_1k_output_tokens=0.3,
        rate_limit_rpm=1000,
        rate_limit_tpm=4000000,
    ),
    # Gemini 2.0 Flash (Second Gen - Feb 2025)
    "gemini-2.0-flash": ProviderCapabilities(
        provider=ProviderType.GOOGLE,
        model="gemini-2.0-flash",
        max_tokens=1048576,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        cost_per_1k_input_tokens=0.075,
        cost_per_1k_output_tokens=0.3,
        rate_limit_rpm=1000,
        rate_limit_tpm=4000000,
    ),
    # Gemini 2.0 Flash-Lite (Cost-Efficient - Feb 2025)
    "gemini-2.0-flash-lite": ProviderCapabilities(
        provider=ProviderType.GOOGLE,
        model="gemini-2.0-flash-lite",
        max_tokens=1048576,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=False,
        cost_per_1k_input_tokens=0.0375,
        cost_per_1k_output_tokens=0.15,
        rate_limit_rpm=1000,
        rate_limit_tpm=4000000,
    ),
}

FEATURED_MODELS = {
    "gemini-3-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
}


class GoogleValidator(BaseValidator):
    """
    Validator for Google Gemini API configurations.
    
    Features:
    - Complete model catalog (8 Gemini models including Gemini 3)
    - Auto-discovery of user's accessible models
    - Cost estimation and rate limits
    - Intelligent typo suggestions
    """

    @classmethod
    def provider_type(cls) -> ProviderType:
        """Return Google as the provider type."""
        return ProviderType.GOOGLE

    @classmethod
    def api_key_pattern(cls) -> Pattern[str]:
        """
        Google Gemini API key pattern.
        Format: AIza[a-zA-Z0-9_-]{35}
        """
        return re.compile(r"^AIza[a-zA-Z0-9_-]{35}$")

    def discover_available_models(self) -> List[str]:
        """
        Fetch models available to this API key via Google API.
        
        Returns:
            List of model IDs the user can access
            
        Raises:
            AuthenticationError: If API key is invalid
            APIConnectionError: If cannot connect
            RateLimitError: If rate limit exceeded
        """
        logger.info(
            "Discovering available models from Google API",
            extra={"provider": "google"}
        )
        
        try:
            response = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": self.config.api_key},
                timeout=10,
            )
            
            if response.status_code == 400:
                logger.error("API key authentication failed", extra={"provider": "google"})
                raise AuthenticationError(provider="google", status_code=400)
            
            if response.status_code == 429:
                retry_after = int(response.headers.get("retry-after", 60))
                logger.warning(
                    "Rate limit hit during discovery",
                    extra={"provider": "google", "retry_after": retry_after}
                )
                raise RateLimitError(provider="google", retry_after=retry_after)
            
            response.raise_for_status()
            data = response.json()
            
            models = []
            for item in data.get("models", []):
                model_id = item.get("name", "").replace("models/", "")
                if model_id and "gemini" in model_id.lower():
                    models.append(model_id)
            
            logger.info(
                f"Discovered {len(models)} models",
                extra={"provider": "google", "count": len(models)}
            )
            
            return sorted(models)
            
        except requests.RequestException as e:
            logger.error(f"Model discovery failed: {e}", extra={"provider": "google"})
            raise APIConnectionError(provider="google", reason=str(e))

    def get_enriched_model_info(self, discovered_models: List[str]) -> List[Dict[str, Any]]:
        """
        Enrich discovered models with metadata from static catalog.
        """
        enriched = []
        
        for model_id in discovered_models:
            info = {
                "id": model_id,
                "name": model_id,
                "featured": model_id in FEATURED_MODELS,
            }
            
            if model_id in GOOGLE_MODELS:
                cap = GOOGLE_MODELS[model_id]
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
        """Fetch account information."""
        logger.info("Fetching account information", extra={"provider": "google"})
        
        return {
            "organization": "Google Cloud",
            "tier": "API Access",
            "note": "Google does not expose billing information via public API",
            "balance": {
                "available_credits_usd": None,
                "usage_this_month_usd": None,
                "usage_percent": None,
            },
            "rate_limits": {
                "requests_per_minute": 1000,
                "tokens_per_minute": 4000000,
            },
            "warnings": [],
        }

    def validate_model_local(self, model: str) -> ValidationResult:
        """Validate model name using static catalog."""
        model_clean = model.strip().lower()

        if model_clean in GOOGLE_MODELS:
            cap = GOOGLE_MODELS[model_clean]
            return ValidationResult(
                status=ValidationStatus.VALID,
                provider=ProviderType.GOOGLE,
                model=model_clean,
                message=f"Valid Google model: {model_clean}",
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

        suggestion = self._suggest_model(model_clean)

        return ValidationResult(
            status=ValidationStatus.INVALID,
            provider=ProviderType.GOOGLE,
            model=model_clean,
            message=f"Model '{model_clean}' is not valid",
            suggestion=f"Did you mean '{suggestion}'?" if suggestion else None,
            details={
                "check_type": "local",
                "suggested_model": suggestion,
                "available_models": list(GOOGLE_MODELS.keys()),
            },
        )

    def _suggest_model(self, model: str) -> Optional[str]:
        """Fuzzy match to suggest closest valid model."""
        matches = get_close_matches(model, GOOGLE_MODELS.keys(), n=1, cutoff=0.6)
        if matches:
            return matches[0]

        if "3" in model or "gemini-3" in model:
            return "gemini-3-pro-preview"
        elif "2.5" in model and "pro" in model:
            return "gemini-2.5-pro"
        elif "2.5" in model and ("flash" in model or "lite" not in model):
            return "gemini-2.5-flash"
        elif "2.5" in model and "lite" in model:
            return "gemini-2.5-flash-lite"
        elif "2.0" in model and "flash" in model:
            return "gemini-2.0-flash"
        elif "flash" in model:
            return "gemini-2.5-flash"
        elif "pro" in model:
            return "gemini-2.5-pro"

        return None

    @classmethod
    def get_featured_models(cls) -> List[str]:
        """Get featured models for UI."""
        return sorted(FEATURED_MODELS)

    @classmethod
    def get_all_models(cls) -> List[str]:
        """Get all models."""
        return sorted(GOOGLE_MODELS.keys())
