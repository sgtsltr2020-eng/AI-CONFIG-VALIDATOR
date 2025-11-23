"""
Exception hierarchy for API configuration validation.

All exceptions provide structured error information for logging,
user feedback, and programmatic error handling.
"""

from typing import Any, Dict, List, Optional


class ValidationError(Exception):
    """
    Base exception for all validation errors.
    
    All custom exceptions inherit from this to allow catch-all
    error handling while preserving specific error types.
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ) -> None:
        """
        Initialize validation error with structured context.
        
        Args:
            message: Human-readable error description
            error_code: Machine-readable error identifier (e.g., "INVALID_API_KEY")
            details: Additional context for debugging/logging
            suggestion: Actionable recommendation to fix the error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.suggestion = suggestion

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
            "suggestion": self.suggestion,
        }

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"{self.__class__.__name__}("
            f"error_code={self.error_code!r}, "
            f"message={self.message!r})"
        )


class UnsupportedProviderError(ValidationError):
    """Raised when the specified provider is not supported."""

    def __init__(
        self,
        provider: str,
        supported_providers: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize unsupported provider error.
        
        Args:
            provider: The provider that was requested
            supported_providers: List of valid provider names
        """
        suggestion = None
        if supported_providers:
            providers_str = ", ".join(supported_providers)
            suggestion = f"Supported providers: {providers_str}"

        super().__init__(
            message=f"Provider '{provider}' is not supported",
            error_code="UNSUPPORTED_PROVIDER",
            details={"provider": provider, "supported_providers": supported_providers},
            suggestion=suggestion,
        )
        self.provider = provider
        self.supported_providers = supported_providers


class InvalidAPIKeyError(ValidationError):
    """Raised when API key format is invalid (local check, no API call)."""

    def __init__(
        self,
        provider: str,
        key_preview: str,
        expected_format: str,
    ) -> None:
        """
        Initialize invalid API key error.
        
        Args:
            provider: Provider name (e.g., "openai")
            key_preview: Safe preview of key (first 8 chars + "...")
            expected_format: Human-readable format description
        """
        super().__init__(
            message=f"Invalid {provider} API key format",
            error_code="INVALID_API_KEY_FORMAT",
            details={
                "provider": provider,
                "key_preview": key_preview,
                "expected_format": expected_format,
            },
            suggestion=f"API key must match format: {expected_format}",
        )
        self.provider = provider
        self.key_preview = key_preview


class InvalidModelError(ValidationError):
    """Raised when model name is not recognized."""

    def __init__(
        self,
        provider: str,
        model: str,
        suggested_model: Optional[str] = None,
        available_models: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize invalid model error.
        
        Args:
            provider: Provider name
            model: Invalid model name that was provided
            suggested_model: Best matching valid model (if any)
            available_models: List of valid model names
        """
        suggestion = None
        if suggested_model:
            suggestion = f"Did you mean '{suggested_model}'?"
        elif available_models:
            models_preview = ", ".join(available_models[:5])
            suggestion = f"Available models: {models_preview}..."

        super().__init__(
            message=f"Model '{model}' is not valid for provider '{provider}'",
            error_code="INVALID_MODEL",
            details={
                "provider": provider,
                "model": model,
                "suggested_model": suggested_model,
                "available_models": available_models,
            },
            suggestion=suggestion,
        )
        self.provider = provider
        self.model = model
        self.suggested_model = suggested_model


class APIConnectionError(ValidationError):
    """Raised when unable to connect to provider API."""

    def __init__(
        self,
        provider: str,
        reason: str,
        retry_after: Optional[int] = None,
    ) -> None:
        """
        Initialize API connection error.
        
        Args:
            provider: Provider name
            reason: Why the connection failed
            retry_after: Seconds to wait before retrying (if applicable)
        """
        suggestion = "Check your internet connection and API endpoint"
        if retry_after:
            suggestion = f"Retry after {retry_after} seconds"

        super().__init__(
            message=f"Failed to connect to {provider} API: {reason}",
            error_code="API_CONNECTION_ERROR",
            details={
                "provider": provider,
                "reason": reason,
                "retry_after": retry_after,
            },
            suggestion=suggestion,
        )
        self.provider = provider
        self.retry_after = retry_after


class RateLimitError(ValidationError):
    """Raised when API rate limit is exceeded."""

    def __init__(
        self,
        provider: str,
        retry_after: Optional[int] = None,
        limit_type: Optional[str] = None,
    ) -> None:
        """
        Initialize rate limit error.
        
        Args:
            provider: Provider name
            retry_after: Seconds until rate limit resets
            limit_type: Type of limit hit (e.g., "requests_per_minute", "tokens_per_day")
        """
        suggestion = "Wait before retrying"
        if retry_after:
            suggestion = f"Retry after {retry_after} seconds"

        super().__init__(
            message=f"Rate limit exceeded for {provider}",
            error_code="RATE_LIMIT_EXCEEDED",
            details={
                "provider": provider,
                "retry_after": retry_after,
                "limit_type": limit_type,
            },
            suggestion=suggestion,
        )
        self.provider = provider
        self.retry_after = retry_after


class AuthenticationError(ValidationError):
    """Raised when API key is rejected by provider (live check)."""

    def __init__(
        self,
        provider: str,
        status_code: Optional[int] = None,
    ) -> None:
        """
        Initialize authentication error.
        
        Args:
            provider: Provider name
            status_code: HTTP status code from provider (if applicable)
        """
        super().__init__(
            message=f"Authentication failed for {provider}",
            error_code="AUTHENTICATION_FAILED",
            details={
                "provider": provider,
                "status_code": status_code,
            },
            suggestion="Verify your API key is correct and has not expired",
        )
        self.provider = provider
        self.status_code = status_code


class ConfigurationError(ValidationError):
    """Raised when configuration is invalid or incomplete."""

    def __init__(
        self,
        field: str,
        reason: str,
        expected: Optional[str] = None,
    ) -> None:
        """
        Initialize configuration error.
        
        Args:
            field: Configuration field that is invalid
            reason: Why it's invalid
            expected: What format/value is expected
        """
        suggestion = None
        if expected:
            suggestion = f"Expected: {expected}"

        super().__init__(
            message=f"Invalid configuration for field '{field}': {reason}",
            error_code="INVALID_CONFIGURATION",
            details={
                "field": field,
                "reason": reason,
                "expected": expected,
            },
            suggestion=suggestion,
        )
        self.field = field
