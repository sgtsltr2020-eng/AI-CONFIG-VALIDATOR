"""
Groq API validator with auto-discovery support.

Validates API keys, discovers available models via OpenAI-compatible endpoint,
and exposes capabilities/costs for Groq-hosted models.
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


# Groq Model Catalog - Production Models Only
# Source: https://console.groq.com/docs/models
GROQ_MODELS: Dict[str, ProviderCapabilities] = {
    # Llama 3.1 8B Instant
    "llama-3.1-8b-instant": ProviderCapabilities(
        provider=ProviderType.GROQ,
        model="llama-3.1-8b-instant",
        max_tokens=131_072,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=False,
        cost_per_1k_input_tokens=0.00005,  # $0.05 per 1M tokens
        cost_per_1k_output_tokens=0.00008,  # $0.08 per 1M tokens
        rate_limit_rpm=1_000,
        rate_limit_tpm=250_000,
    ),
    # Llama 3.3 70B Versatile
    "llama-3.3-70b-versatile": ProviderCapabilities(
        provider=ProviderType.GROQ,
        model="llama-3.3-70b-versatile",
        max_tokens=131_072,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=False,
        cost_per_1k_input_tokens=0.00059,  # $0.59 per 1M tokens
        cost_per_1k_output_tokens=0.00079,  # $0.79 per 1M tokens
        rate_limit_rpm=1_000,
        rate_limit_tpm=300_000,
    ),
    # Llama Guard 4 12B (Safety/Moderation)
    "meta-llama/llama-guard-4-12b": ProviderCapabilities(
        provider=ProviderType.GROQ,
        model="meta-llama/llama-guard-4-12b",
        max_tokens=131_072,
        supports_streaming=False,
        supports_function_calling=False,
        supports_vision=False,
        cost_per_1k_input_tokens=0.00020,  # $0.20 per 1M tokens
        cost_per_1k_output_tokens=0.00020,  # $0.20 per 1M tokens
        rate_limit_rpm=100,
        rate_limit_tpm=30_000,
    ),
    # OpenAI GPT-OSS 120B
    "openai/gpt-oss-120b": ProviderCapabilities(
        provider=ProviderType.GROQ,
        model="openai/gpt-oss-120b",
        max_tokens=131_072,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=False,
        cost_per_1k_input_tokens=0.00015,  # $0.15 per 1M tokens
        cost_per_1k_output_tokens=0.00060,  # $0.60 per 1M tokens
        rate_limit_rpm=1_000,
        rate_limit_tpm=250_000,
    ),
    # OpenAI GPT-OSS 20B
    "openai/gpt-oss-20b": ProviderCapabilities(
        provider=ProviderType.GROQ,
        model="openai/gpt-oss-20b",
        max_tokens=131_072,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=False,
        cost_per_1k_input_tokens=0.000075,  # $0.075 per 1M tokens
        cost_per_1k_output_tokens=0.00030,  # $0.30 per 1M tokens
        rate_limit_rpm=1_000,
        rate_limit_tpm=250_000,
    ),
    # Whisper Large V3
    "whisper-large-v3": ProviderCapabilities(
        provider=ProviderType.GROQ,
        model="whisper-large-v3",
        max_tokens=448,
        supports_streaming=False,
        supports_function_calling=False,
        supports_vision=False,
        cost_per_1k_input_tokens=0.0,  # Priced per hour, not per token
        cost_per_1k_output_tokens=0.0,
        rate_limit_rpm=300,
        rate_limit_tpm=None,
    ),
    # Whisper Large V3 Turbo
    "whisper-large-v3-turbo": ProviderCapabilities(
        provider=ProviderType.GROQ,
        model="whisper-large-v3-turbo",
        max_tokens=448,
        supports_streaming=False,
        supports_function_calling=False,
        supports_vision=False,
        cost_per_1k_input_tokens=0.0,  # Priced per hour, not per token
        cost_per_1k_output_tokens=0.0,
        rate_limit_rpm=400,
        rate_limit_tpm=None,
    ),
}

FEATURED_MODELS = {
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
}


class GroqValidator(BaseValidator):
    """
    Validator for Groq API configurations.
    
    Features:
    - OpenAI-compatible /openai/v1/models endpoint for discovery
    - Static catalog for 7 production models
    - Fuzzy suggestions for model typos
    - Special handling for Whisper pricing
    """

    @classmethod
    def provider_type(cls) -> ProviderType:
        """Return Groq as the provider type."""
        return ProviderType.GROQ

    @classmethod
    def api_key_pattern(cls) -> Pattern[str]:
        """
        Groq API key pattern.
        
        Groq keys are opaque strings without a fixed prefix.
        Enforce: alphanumeric + underscore/hyphen, min 20 chars, no spaces.
        """
        return re.compile(r"^[A-Za-z0-9_\-]{20,}$")

    def discover_available_models(self) -> List[str]:
        """
        Fetch models available to this API key via Groq's OpenAI-compatible API.
        
        Endpoint: GET https://api.groq.com/openai/v1/models
        Auth: Bearer token in Authorization header
        
        Returns:
            List of model IDs accessible to this API key
            
        Raises:
            AuthenticationError: If API key is invalid (401/403)
            APIConnectionError: If request fails
            RateLimitError: If rate limit exceeded (429)
        """
        logger.info(
            "Discovering available models from Groq API",
            extra={"provider": "groq"},
        )

        try:
            response = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )

            if response.status_code in (401, 403):
                logger.error(
                    "Groq API key authentication failed",
                    extra={"provider": "groq", "status_code": response.status_code},
                )
                raise AuthenticationError(
                    provider="groq",
                    status_code=response.status_code,
                )

            if response.status_code == 429:
                retry_after = int(response.headers.get("retry-after", 60))
                logger.warning(
                    "Groq rate limit hit during discovery",
                    extra={"provider": "groq", "retry_after": retry_after},
                )
                raise RateLimitError(provider="groq", retry_after=retry_after)

            response.raise_for_status()
            data = response.json()

            # OpenAI-compatible format: {"object": "list", "data": [{"id": "...", "active": true}]}
            models = [
                item["id"]
                for item in data.get("data", [])
                if item.get("active", True)
            ]

            logger.info(
                f"Discovered {len(models)} Groq models",
                extra={"provider": "groq", "count": len(models)},
            )

            return sorted(models)

        except requests.RequestException as e:
            logger.error(
                f"Groq model discovery failed: {e}",
                extra={"provider": "groq"},
            )
            raise APIConnectionError(provider="groq", reason=str(e))

    def get_enriched_model_info(self, discovered_models: List[str]) -> List[Dict[str, Any]]:
        """
        Enrich discovered models with metadata from static catalog.
        
        Args:
            discovered_models: Model IDs from Groq API
            
        Returns:
            List of dicts with full model metadata
        """
        enriched: List[Dict[str, Any]] = []

        for model_id in discovered_models:
            info: Dict[str, Any] = {
                "id": model_id,
                "name": model_id,
                "featured": model_id in FEATURED_MODELS,
            }

            if model_id in GROQ_MODELS:
                cap = GROQ_MODELS[model_id]
                info.update({
                    "max_tokens": cap.max_tokens,
                    "supports_streaming": cap.supports_streaming,
                    "supports_function_calling": cap.supports_function_calling,
                    "supports_vision": cap.supports_vision,
                    "cost_per_1k_input": cap.cost_per_1k_input_tokens,
                    "cost_per_1k_output": cap.cost_per_1k_output_tokens,
                    "rate_limit_rpm": cap.rate_limit_rpm,
                })

                # Special handling for Whisper models (priced per hour)
                if "whisper" in model_id:
                    info["pricing_note"] = "Priced per audio hour, not per token"
                    if model_id == "whisper-large-v3":
                        info["cost_per_hour"] = 0.111
                    elif model_id == "whisper-large-v3-turbo":
                        info["cost_per_hour"] = 0.04
            else:
                info["unknown"] = True

            enriched.append(info)

        enriched.sort(key=lambda x: (not x.get("featured", False), x["name"]))
        return enriched

    def get_account_info(self) -> Dict[str, Any]:
        """
        Fetch account information.
        
        Note: Groq does not expose billing info via public API.
        Configuration is managed via GroqCloud console.
        
        Returns:
            Dict with basic account information
        """
        logger.info(
            "Fetching Groq account info (static)",
            extra={"provider": "groq"},
        )

        return {
            "organization": "GroqCloud Account",
            "tier": "API Access",
            "note": "Groq does not expose billing information via public API",
            "balance": {
                "available_credits_usd": None,
                "usage_this_month_usd": None,
                "usage_percent": None,
            },
            "rate_limits": {
                "requests_per_minute": 1_000,
                "tokens_per_minute": 300_000,
            },
            "warnings": [],
        }

    def validate_model_local(self, model: str) -> ValidationResult:
        """Validate model name using static catalog."""
        model_clean = model.strip().lower()

        if model_clean in GROQ_MODELS:
            cap = GROQ_MODELS[model_clean]
            return ValidationResult(
                status=ValidationStatus.VALID,
                provider=ProviderType.GROQ,
                model=model_clean,
                message=f"Valid Groq model: {model_clean}",
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
            provider=ProviderType.GROQ,
            model=model_clean,
            message=f"Model '{model_clean}' is not valid",
            suggestion=f"Did you mean '{suggestion}'?" if suggestion else None,
            details={
                "check_type": "local",
                "suggested_model": suggestion,
                "available_models": list(GROQ_MODELS.keys()),
            },
        )

    def _suggest_model(self, model: str) -> Optional[str]:
        """Fuzzy match to suggest closest valid model."""
        matches = get_close_matches(model, GROQ_MODELS.keys(), n=1, cutoff=0.6)
        if matches:
            return matches[0]

        # Fallback: keyword matching
        lower = model.lower()
        if "llama" in lower and "70" in lower:
            return "llama-3.3-70b-versatile"
        elif "llama" in lower and "8" in lower:
            return "llama-3.1-8b-instant"
        elif "gpt" in lower and "120" in lower:
            return "openai/gpt-oss-120b"
        elif "gpt" in lower and "20" in lower:
            return "openai/gpt-oss-20b"
        elif "whisper" in lower and "turbo" in lower:
            return "whisper-large-v3-turbo"
        elif "whisper" in lower:
            return "whisper-large-v3"

        return None

    @classmethod
    def get_featured_models(cls) -> List[str]:
        """Get featured models for UI display."""
        return sorted(FEATURED_MODELS)

    @classmethod
    def get_all_models(cls) -> List[str]:
        """Get all models in static catalog."""
        return sorted(GROQ_MODELS.keys())
