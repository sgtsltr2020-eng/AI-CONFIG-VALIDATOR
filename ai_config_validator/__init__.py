"""
AI Config Validator - Enterprise-grade API configuration validation.

Public API for validating LLM provider credentials, discovering available models,
and fetching account information.
"""

from datetime import datetime, UTC
from typing import Any, Dict

from .errors import UnsupportedProviderError, ValidationError
from .logging_config import get_logger, setup_logging
from .models import (
    APIKeyConfig,
    ProviderCapabilities,
    ProviderType,
    ValidationResult,
    ValidationStatus,
)
from .validators.openai import OpenAIValidator
from .validators.anthropic import AnthropicValidator
from .validators.google import GoogleValidator
from .validators.groq import GroqValidator

__version__ = "0.1.0"
__all__ = [
    "validate_and_discover",
    "validate_llm_config",
    "setup_logging",
    "OpenAIValidator",
    "AnthropicValidator",
    "GoogleValidator",
    "GroqValidator",
    "ValidationResult",
    "APIKeyConfig",
    "ProviderType",
]

logger = get_logger(__name__)


def validate_and_discover(provider: str, api_key: str) -> Dict[str, Any]:
    """
    One-call API: Validate key + discover models + fetch account info.
    
    This is the main function your frontend calls for the "Validate & Auto-Configure" button.
    
    Args:
        provider: Provider name ("openai", "anthropic", etc.)
        api_key: User's API key
        
    Returns:
        Complete response with:
        - key_valid: bool
        - models: List of available models with metadata
        - account_info: Balance, usage, limits, warnings
        - error: Error details if validation failed
        
    Example:
        >>> result = validate_and_discover("openai", "sk-proj-...")
        >>> if result["success"]:
        ...     print(f"Found {result['model_count']} models")
        ...     print(f"Balance: ${result['account_info']['balance']['available_credits_usd']}")
    """
    provider_lower = provider.lower().strip()
    
    logger.info(
        f"Starting validation and discovery for {provider_lower}",
        extra={"provider": provider_lower}
    )
    
    try:
        # Step 1: Create validator instance
        if provider_lower == "openai":
            validator = OpenAIValidator(api_key=api_key)
        elif provider_lower == "anthropic":
            validator = AnthropicValidator(api_key=api_key)
        elif provider_lower == "google":
            validator = GoogleValidator(api_key=api_key)
        elif provider_lower == "groq":
            validator = GroqValidator(api_key=api_key)
        else:
            raise UnsupportedProviderError(
                provider=provider_lower,
                supported_providers=["openai", "anthropic", "google", "groq"],
            )
        
        # Step 2: Validate API key format (fast, local)
        logger.debug("Validating API key format", extra={"provider": provider_lower})
        validator.validate_api_key_format()
        
        # Step 3: Discover available models (API call)
        logger.info("Discovering available models", extra={"provider": provider_lower})
        discovered_models = validator.discover_available_models()
        
        # Step 4: Enrich with metadata from static catalog
        logger.debug("Enriching model metadata", extra={"provider": provider_lower})
        enriched_models = validator.get_enriched_model_info(discovered_models)
        
        # Step 5: Fetch account information (balance, usage, limits)
        logger.info("Fetching account information", extra={"provider": provider_lower})
        account_info = validator.get_account_info()
        
        logger.info(
            f"Validation successful for {provider_lower}",
            extra={
                "provider": provider_lower,
                "model_count": len(enriched_models),
                "balance": account_info.get("balance", {}).get("available_credits_usd"),
            }
        )
        
        return {
            "success": True,
            "key_valid": True,
            "provider": provider_lower,
            "account_info": account_info,
            "models": enriched_models,
            "model_count": len(enriched_models),
            "timestamp": datetime.utcnow().isoformat(),
            "error": None,
        }
        
    except ValidationError as e:
        logger.error(
            f"Validation failed for {provider_lower}",
            extra={"provider": provider_lower, "error": e.to_dict()}
        )
        
        return {
            "success": False,
            "key_valid": False,
            "provider": provider_lower,
            "account_info": {},
            "models": [],
            "model_count": 0,
            "timestamp": datetime.now(UTC).isoformat(),
            "error": e.to_dict(),
        }
    
    except Exception as e:
        logger.critical(
            f"Unexpected error during validation: {e}",
            extra={"provider": provider_lower, "error": str(e)},
            exc_info=True,
        )
        
        return {
            "success": False,
            "key_valid": False,
            "provider": provider_lower,
            "account_info": {},
            "models": [],
            "model_count": 0,
            "timestamp": datetime.utcnow().isoformat(),
            "error": {
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred during validation",
                "details": {"exception": str(e)},
                "suggestion": "Please try again or contact support",
            },
        }


def validate_llm_config(
    provider: str,
    api_key: str,
    model: str,
) -> ValidationResult:
    """
    Simple validation: Check if provider + API key + model combination is valid.
    
    This is a simpler API for basic validation without discovery.
    Use validate_and_discover() for the full auto-configuration flow.
    
    Args:
        provider: Provider name ("openai", "anthropic", etc.)
        api_key: User's API key
        model: Model name to validate
        
    Returns:
        ValidationResult with status and details
        
    Raises:
        UnsupportedProviderError: If provider not supported
        ValidationError: If validation fails
        
    Example:
        >>> result = validate_llm_config("openai", "sk-proj-...", "gpt-4o")
        >>> if result.is_valid():
        ...     print(f"Cost: ${result.details['cost_per_1k_input']}")
    """
    provider_lower = provider.lower().strip()
    
    logger.info(
        f"Validating config for {provider_lower}/{model}",
        extra={"provider": provider_lower, "model": model}
    )
    
    # Create validator
    if provider_lower == "openai":
        validator = OpenAIValidator(api_key=api_key)
    elif provider_lower == "anthropic":
        validator = AnthropicValidator(api_key=api_key)
    elif provider_lower == "google":
        validator = GoogleValidator(api_key=api_key)
    elif provider_lower == "groq":
        validator = GroqValidator(api_key=api_key)
    else:
        raise UnsupportedProviderError(
            provider=provider_lower,
            supported_providers=["openai", "anthropic", "google", "groq"],
        )
    
    # Validate (format check + model check)
    result = validator.validate(model, check_live=False)
    
    logger.info(
        f"Validation complete: {result.status.value}",
        extra={
            "provider": provider_lower,
            "model": model,
            "status": result.status.value,
        }
    )
    
    return result
