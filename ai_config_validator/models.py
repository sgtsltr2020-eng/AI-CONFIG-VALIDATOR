"""
Core data models for API configuration validation.

All models use Pydantic for runtime validation, serialization,
and automatic API documentation generation.
"""

from datetime import datetime, UTC
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProviderType(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    MISTRAL = "mistral"
    GROQ = "groq"
    TOGETHER = "together"
    OPENROUTER = "openrouter"
    GITHUB = "github"


class ValidationStatus(str, Enum):
    """Validation result status."""

    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    ERROR = "error"


class ValidationResult(BaseModel):
    """
    Result of API configuration validation.

    This is the primary response object returned by all validation operations.
    Immutable after creation to ensure consistency in logs and audit trails.
    """

    model_config = ConfigDict(frozen=True)

    status: ValidationStatus = Field(..., description="Overall validation status")
    provider: ProviderType = Field(..., description="LLM provider being validated")
    model: str = Field(..., description="Model name that was validated")
    message: str = Field(..., description="Human-readable validation message")
    suggestion: Optional[str] = Field(
        None, description="Suggested fix if validation failed"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (costs, limits, capabilities)",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When validation was performed (UTC)",
    )
    request_id: Optional[str] = Field(None, description="Request tracking ID")

    @field_validator("model")
    @classmethod
    def model_must_not_be_empty(cls, v: str) -> str:
        """Ensure model name is not empty or whitespace."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()

    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.status == ValidationStatus.VALID

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with ISO timestamp."""
        data = self.model_dump()
        data["timestamp"] = self.timestamp.isoformat()
        return data


class APIKeyConfig(BaseModel):
    """
    API key configuration with validation rules.

    Stores sensitive credentials securely and validates format
    before any external API calls.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider": "openai",
                "api_key": "sk-proj-...",
                "environment": "production",
                "metadata": {"team": "ml-platform", "cost_center": "engineering"},
            }
        }
    )

    provider: ProviderType = Field(..., description="Provider for this API key")
    api_key: str = Field(..., description="API key value", repr=False)  # Hide in logs
    environment: str = Field(
        default="production", description="Environment (production/staging/dev)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional configuration"
    )

    @field_validator("api_key")
    @classmethod
    def api_key_must_not_be_empty(cls, v: str) -> str:
        """Ensure API key is not empty."""
        if not v or not v.strip():
            raise ValueError("API key cannot be empty")
        if len(v) < 8:
            raise ValueError("API key too short (minimum 8 characters)")
        return v

    def __repr__(self) -> str:
        """Safe representation that doesn't expose full API key."""
        key_preview = f"{self.api_key[:8]}..." if len(self.api_key) > 8 else "***"
        return f"APIKeyConfig(provider={self.provider}, api_key={key_preview})"


class ProviderCapabilities(BaseModel):
    """
    Capabilities and limits for a specific provider/model combination.

    Used for cost estimation, rate limiting, and feature detection.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider": "openai",
                "model": "gpt-4o",
                "max_tokens": 128000,
                "supports_streaming": True,
                "supports_function_calling": True,
                "supports_vision": True,
                "cost_per_1k_input_tokens": 0.0025,
                "cost_per_1k_output_tokens": 0.01,
                "rate_limit_rpm": 10000,
                "rate_limit_tpm": 2000000,
            }
        }
    )

    provider: ProviderType
    model: str
    max_tokens: int = Field(..., description="Maximum context window size")
    supports_streaming: bool = Field(default=True)
    supports_function_calling: bool = Field(default=False)
    supports_vision: bool = Field(default=False)
    cost_per_1k_input_tokens: float = Field(..., description="USD per 1K input tokens")
    cost_per_1k_output_tokens: float = Field(..., description="USD per 1K output tokens")
    rate_limit_rpm: Optional[int] = Field(None, description="Requests per minute")
    rate_limit_tpm: Optional[int] = Field(None, description="Tokens per minute")
