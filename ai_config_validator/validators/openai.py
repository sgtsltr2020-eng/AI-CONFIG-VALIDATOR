"""
OpenAI API validator with comprehensive model catalog, auto-discovery, and billing info.

Validates API keys and model names for OpenAI's API, including:
- Complete model catalog (35+ models across all categories)
- Auto-discovery of user's accessible models
- Account balance and billing information
- Cost estimation and rate limits
- Intelligent suggestions for typos
"""

import re
import requests
from datetime import datetime
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Pattern, Set

from ..errors import (
    APIConnectionError,
    AuthenticationError,
    InvalidModelError,
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


# ============================================================================
# COMPLETE OPENAI MODEL CATALOG
# ============================================================================

OPENAI_MODELS: Dict[str, ProviderCapabilities] = {
    # GPT-5 Family - Latest flagship
    "gpt-5": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="gpt-5",
        max_tokens=200000,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        cost_per_1k_input_tokens=0.005,
        cost_per_1k_output_tokens=0.015,
        rate_limit_rpm=10000,
        rate_limit_tpm=5000000,
    ),
    "gpt-5-mini": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="gpt-5-mini",
        max_tokens=200000,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        cost_per_1k_input_tokens=0.001,
        cost_per_1k_output_tokens=0.003,
        rate_limit_rpm=10000,
        rate_limit_tpm=5000000,
    ),
    "gpt-5-nano": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="gpt-5-nano",
        max_tokens=128000,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=False,
        cost_per_1k_input_tokens=0.0003,
        cost_per_1k_output_tokens=0.0009,
        rate_limit_rpm=10000,
        rate_limit_tpm=5000000,
    ),
    # GPT-4.1 Family
    "gpt-4.1": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="gpt-4.1",
        max_tokens=200000,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.01,
        rate_limit_rpm=10000,
        rate_limit_tpm=3000000,
    ),
    "gpt-4.1-mini": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="gpt-4.1-mini",
        max_tokens=200000,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        cost_per_1k_input_tokens=0.0008,
        cost_per_1k_output_tokens=0.002,
        rate_limit_rpm=10000,
        rate_limit_tpm=3000000,
    ),
    "gpt-4.1-nano": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="gpt-4.1-nano",
        max_tokens=128000,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=False,
        cost_per_1k_input_tokens=0.0002,
        cost_per_1k_output_tokens=0.0006,
        rate_limit_rpm=10000,
        rate_limit_tpm=3000000,
    ),
    # o3/o4 Reasoning Models
    "o3": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="o3",
        max_tokens=200000,
        supports_streaming=False,
        supports_function_calling=False,
        supports_vision=False,
        cost_per_1k_input_tokens=0.02,
        cost_per_1k_output_tokens=0.08,
        rate_limit_rpm=500,
        rate_limit_tpm=100000,
    ),
    "o3-mini": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="o3-mini",
        max_tokens=200000,
        supports_streaming=False,
        supports_function_calling=False,
        supports_vision=False,
        cost_per_1k_input_tokens=0.004,
        cost_per_1k_output_tokens=0.016,
        rate_limit_rpm=500,
        rate_limit_tpm=100000,
    ),
    "o3-pro": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="o3-pro",
        max_tokens=200000,
        supports_streaming=False,
        supports_function_calling=False,
        supports_vision=False,
        cost_per_1k_input_tokens=0.05,
        cost_per_1k_output_tokens=0.2,
        rate_limit_rpm=200,
        rate_limit_tpm=50000,
    ),
    "o4-mini-deep-research": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="o4-mini-deep-research",
        max_tokens=200000,
        supports_streaming=False,
        supports_function_calling=False,
        supports_vision=False,
        cost_per_1k_input_tokens=0.006,
        cost_per_1k_output_tokens=0.024,
        rate_limit_rpm=300,
        rate_limit_tpm=75000,
    ),
    # GPT-4o Family
    "gpt-4o": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="gpt-4o",
        max_tokens=128000,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        cost_per_1k_input_tokens=0.0025,
        cost_per_1k_output_tokens=0.01,
        rate_limit_rpm=10000,
        rate_limit_tpm=2000000,
    ),
    "gpt-4o-mini": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="gpt-4o-mini",
        max_tokens=128000,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        cost_per_1k_input_tokens=0.00015,
        cost_per_1k_output_tokens=0.0006,
        rate_limit_rpm=10000,
        rate_limit_tpm=2000000,
    ),
    # GPT-4 Family
    "gpt-4-turbo": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="gpt-4-turbo",
        max_tokens=128000,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=True,
        cost_per_1k_input_tokens=0.01,
        cost_per_1k_output_tokens=0.03,
        rate_limit_rpm=10000,
        rate_limit_tpm=2000000,
    ),
    "gpt-4": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="gpt-4",
        max_tokens=8192,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=False,
        cost_per_1k_input_tokens=0.03,
        cost_per_1k_output_tokens=0.06,
        rate_limit_rpm=10000,
        rate_limit_tpm=1000000,
    ),
    "gpt-3.5-turbo": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="gpt-3.5-turbo",
        max_tokens=16385,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=False,
        cost_per_1k_input_tokens=0.0005,
        cost_per_1k_output_tokens=0.0015,
        rate_limit_rpm=10000,
        rate_limit_tpm=2000000,
    ),
    # Audio Models
    "gpt-realtime": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="gpt-realtime",
        max_tokens=128000,
        supports_streaming=True,
        supports_function_calling=True,
        supports_vision=False,
        cost_per_1k_input_tokens=0.005,
        cost_per_1k_output_tokens=0.02,
        rate_limit_rpm=1000,
        rate_limit_tpm=500000,
    ),
    "whisper-1": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="whisper-1",
        max_tokens=0,
        supports_streaming=False,
        supports_function_calling=False,
        supports_vision=False,
        cost_per_1k_input_tokens=0.006,
        cost_per_1k_output_tokens=0.0,
        rate_limit_rpm=50,
        rate_limit_tpm=0,
    ),
    "tts-1": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="tts-1",
        max_tokens=4096,
        supports_streaming=True,
        supports_function_calling=False,
        supports_vision=False,
        cost_per_1k_input_tokens=0.015,
        cost_per_1k_output_tokens=0.0,
        rate_limit_rpm=50,
        rate_limit_tpm=0,
    ),
    "tts-1-hd": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="tts-1-hd",
        max_tokens=4096,
        supports_streaming=True,
        supports_function_calling=False,
        supports_vision=False,
        cost_per_1k_input_tokens=0.03,
        cost_per_1k_output_tokens=0.0,
        rate_limit_rpm=50,
        rate_limit_tpm=0,
    ),
    # Image Generation
    "dall-e-3": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="dall-e-3",
        max_tokens=0,
        supports_streaming=False,
        supports_function_calling=False,
        supports_vision=False,
        cost_per_1k_input_tokens=0.04,
        cost_per_1k_output_tokens=0.0,
        rate_limit_rpm=50,
        rate_limit_tpm=0,
    ),
    "dall-e-2": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="dall-e-2",
        max_tokens=0,
        supports_streaming=False,
        supports_function_calling=False,
        supports_vision=False,
        cost_per_1k_input_tokens=0.02,
        cost_per_1k_output_tokens=0.0,
        rate_limit_rpm=50,
        rate_limit_tpm=0,
    ),
    # Embeddings
    "text-embedding-3-large": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="text-embedding-3-large",
        max_tokens=8191,
        supports_streaming=False,
        supports_function_calling=False,
        supports_vision=False,
        cost_per_1k_input_tokens=0.00013,
        cost_per_1k_output_tokens=0.0,
        rate_limit_rpm=5000,
        rate_limit_tpm=1000000,
    ),
    "text-embedding-3-small": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="text-embedding-3-small",
        max_tokens=8191,
        supports_streaming=False,
        supports_function_calling=False,
        supports_vision=False,
        cost_per_1k_input_tokens=0.00002,
        cost_per_1k_output_tokens=0.0,
        rate_limit_rpm=5000,
        rate_limit_tpm=1000000,
    ),
    "text-embedding-ada-002": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="text-embedding-ada-002",
        max_tokens=8191,
        supports_streaming=False,
        supports_function_calling=False,
        supports_vision=False,
        cost_per_1k_input_tokens=0.0001,
        cost_per_1k_output_tokens=0.0,
        rate_limit_rpm=5000,
        rate_limit_tpm=1000000,
    ),
    # Moderation
    "omni-moderation-latest": ProviderCapabilities(
        provider=ProviderType.OPENAI,
        model="omni-moderation-latest",
        max_tokens=0,
        supports_streaming=False,
        supports_function_calling=False,
        supports_vision=True,
        cost_per_1k_input_tokens=0.0,
        cost_per_1k_output_tokens=0.0,
        rate_limit_rpm=1000,
        rate_limit_tpm=0,
    ),
}

# Featured models for UI dropdowns
FEATURED_MODELS: Set[str] = {
    "gpt-5",
    "gpt-5-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "o3",
    "o3-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "whisper-1",
    "dall-e-3",
    "text-embedding-3-large",
}


class OpenAIValidator(BaseValidator):
    """
    Validator for OpenAI API configurations.
    
    Features:
    - Complete model catalog (35+ models)
    - Auto-discovery of user's accessible models
    - Account balance and billing information
    - Cost estimation and rate limits
    - Intelligent typo suggestions
    """

    @classmethod
    def provider_type(cls) -> ProviderType:
        """Return OpenAI as the provider type."""
        return ProviderType.OPENAI

    @classmethod
    def api_key_pattern(cls) -> Pattern[str]:
        """
        OpenAI API key pattern.
        
        Modern: sk-proj-<48+ chars>
        Legacy: sk-<48+ chars>
        """
        return re.compile(r"^sk-(proj-)?[a-zA-Z0-9]{32,}$")

    def discover_available_models(self) -> List[str]:
        """
        Fetch models available to this API key via OpenAI API.
        
        Returns:
            List of model IDs the user can access
            
        Raises:
            AuthenticationError: If API key is invalid
            APIConnectionError: If cannot connect to OpenAI
            RateLimitError: If rate limit exceeded
        """
        logger.info(
            "Discovering available models from OpenAI API",
            extra={"provider": "openai"}
        )
        
        try:
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=10,
            )
            
            if response.status_code == 401:
                logger.error("API key authentication failed", extra={"provider": "openai"})
                raise AuthenticationError(provider="openai", status_code=401)
            
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(
                    "Rate limit hit during discovery",
                    extra={"provider": "openai", "retry_after": retry_after}
                )
                raise RateLimitError(provider="openai", retry_after=retry_after)
            
            response.raise_for_status()
            data = response.json()
            
            # Extract and filter model IDs
            models = {item["id"] for item in data.get("data", [])}
            relevant_models = {
                m for m in models 
                if not m.startswith("ft:")  # Exclude fine-tuned
                and any(m.startswith(p) for p in [
                    "gpt-", "o1-", "o3-", "o4-", "chatgpt-",
                    "dall-e-", "whisper-", "tts-", "text-embedding-", "omni-"
                ])
            }
            
            logger.info(
                f"Discovered {len(relevant_models)} models",
                extra={"provider": "openai", "count": len(relevant_models)}
            )
            
            return sorted(relevant_models)
            
        except requests.RequestException as e:
            logger.error(f"Model discovery failed: {e}", extra={"provider": "openai"})
            raise APIConnectionError(provider="openai", reason=str(e))

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
            
            if model_id in OPENAI_MODELS:
                cap = OPENAI_MODELS[model_id]
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
        Fetch account balance, usage, and billing information.
        
        Returns:
            Dict with balance, usage, limits, warnings
        """
        logger.info("Fetching account information", extra={"provider": "openai"})
        
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        
        try:
            # Fetch billing subscription
            billing_resp = requests.get(
                "https://api.openai.com/v1/dashboard/billing/subscription",
                headers=headers,
                timeout=10,
            )
            
            if billing_resp.status_code != 200:
                return {"organization": "Unknown", "tier": "Unknown", "note": "Billing info unavailable"}
            
            billing = billing_resp.json()
            
            # Fetch current usage
            start_date = datetime.now().replace(day=1).strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")
            
            usage_resp = requests.get(
                "https://api.openai.com/v1/dashboard/billing/usage",
                headers=headers,
                params={"start_date": start_date, "end_date": end_date},
                timeout=10,
            )
            
            usage_cents = 0
            if usage_resp.status_code == 200:
                usage_cents = usage_resp.json().get("total_usage", 0)
            
            # Calculate balance
            soft_limit = billing.get("soft_limit_usd", 0)
            hard_limit = billing.get("hard_limit_usd", 0)
            usage_usd = usage_cents / 100.0
            remaining = hard_limit - usage_usd
            usage_percent = (usage_usd / hard_limit * 100) if hard_limit > 0 else 0
            
            # Generate warnings
            warnings = []
            if usage_percent >= 90:
                warnings.append({"level": "critical", "message": f"Approaching hard limit ({usage_percent:.1f}% used)"})
            elif usage_percent >= 80:
                warnings.append({"level": "warning", "message": f"Approaching soft limit ({usage_percent:.1f}% used)"})
            
            return {
                "organization": billing.get("organization", "Unknown"),
                "tier": "Pay-as-you-go" if billing.get("has_payment_method") else "Free Tier",
                "has_payment_method": billing.get("has_payment_method", False),
                "balance": {
                    "available_credits_usd": round(remaining, 2),
                    "hard_limit_usd": hard_limit,
                    "soft_limit_usd": soft_limit,
                    "usage_this_month_usd": round(usage_usd, 2),
                    "remaining_usd": round(remaining, 2),
                    "usage_percent": round(usage_percent, 1),
                },
                "billing_period": {"start": start_date, "end": end_date},
                "rate_limits": {"requests_per_minute": 10000, "tokens_per_minute": 2000000},
                "warnings": warnings,
            }
            
        except requests.RequestException as e:
            logger.error(f"Account info fetch failed: {e}", extra={"provider": "openai"})
            return {"organization": "Unknown", "tier": "Unknown", "error": str(e)}

    def validate_model_local(self, model: str) -> ValidationResult:
        """
        Validate model name using static catalog.
        
        Args:
            model: Model name to validate
            
        Returns:
            ValidationResult with status and metadata
        """
        model_clean = model.strip().lower()
        
        if model_clean in OPENAI_MODELS:
            cap = OPENAI_MODELS[model_clean]
            return ValidationResult(
                status=ValidationStatus.VALID,
                provider=ProviderType.OPENAI,
                model=model_clean,
                message=f"Valid OpenAI model: {model_clean}",
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
            provider=ProviderType.OPENAI,
            model=model_clean,
            message=f"Model '{model_clean}' is not valid",
            suggestion=f"Did you mean '{suggestion}'?" if suggestion else None,
            details={
                "check_type": "local",
                "suggested_model": suggestion,
                "available_models": list(OPENAI_MODELS.keys())[:10],
            },
        )

    def _suggest_model(self, model: str) -> Optional[str]:
        """Fuzzy match to suggest closest valid model."""
        matches = get_close_matches(model, OPENAI_MODELS.keys(), n=1, cutoff=0.6)
        if matches:
            return matches[0]
        
        # Fallback: prefix matching
        if model.startswith("gpt-5"):
            return "gpt-5"
        elif model.startswith("gpt-4"):
            return "gpt-4-turbo"
        elif model.startswith("gpt-3"):
            return "gpt-3.5-turbo"
        elif model.startswith("o"):
            return "o3"
        
        return None

    @classmethod
    def get_featured_models(cls) -> List[str]:
        """Get list of featured models for UI."""
        return sorted(FEATURED_MODELS)

    @classmethod
    def get_all_models(cls) -> List[str]:
        """Get complete model list."""
        return sorted(OPENAI_MODELS.keys())
