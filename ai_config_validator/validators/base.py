"""
Abstract base validator that all provider validators must implement.

This defines the contract for validation operations and provides
common functionality for all providers.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Optional, Pattern

from ..errors import InvalidAPIKeyError
from ..logging_config import get_logger
from ..models import APIKeyConfig, ProviderType, ValidationResult, ValidationStatus

logger = get_logger(__name__)


class BaseValidator(ABC):
    """
    Abstract base class for all provider validators.
    
    Each provider (OpenAI, Anthropic, etc.) must implement this interface
    to provide consistent validation behavior across the system.
    
    Subclasses must implement:
    - provider_type: Return the ProviderType enum value
    - api_key_pattern: Return compiled regex for API key format
    - validate_model_local: Perform local model validation
    
    Optional to override:
    - validate_model_live: Perform live API validation
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[APIKeyConfig] = None,
    ) -> None:
        """
        Initialize validator with API key or configuration.
        
        Args:
            api_key: Raw API key string (convenience parameter)
            config: Full APIKeyConfig object (preferred for production)
            
        Raises:
            ValueError: If neither api_key nor config provided
        """
        if config:
            self.config = config
        elif api_key:
            self.config = APIKeyConfig(
                provider=self.provider_type(),
                api_key=api_key,
            )
        else:
            raise ValueError("Must provide either api_key or config")

        logger.debug(
            f"Initialized {self.__class__.__name__}",
            extra={"provider": self.config.provider.value},
        )

    @classmethod
    @abstractmethod
    def provider_type(cls) -> ProviderType:
        """
        Return the provider type this validator handles.
        
        Returns:
            ProviderType enum value
            
        Example:
            >>> OpenAIValidator.provider_type()
            ProviderType.OPENAI
        """
        pass

    @classmethod
    @abstractmethod
    def api_key_pattern(cls) -> Pattern[str]:
        """
        Return regex pattern for API key validation.
        
        This should match the provider's API key format.
        Must be a compiled regex pattern.
        
        Returns:
            Compiled regex pattern
            
        Example:
            >>> OpenAIValidator.api_key_pattern()
            re.compile(r'^sk-[a-zA-Z0-9]{32,}$')
        """
        pass

    @abstractmethod
    def validate_model_local(self, model: str) -> ValidationResult:
        """
        Validate model name using local checks only (no API calls).
        
        This should check if the model name is in the known list
        and provide suggestions if not found.
        
        Args:
            model: Model name to validate
            
        Returns:
            ValidationResult with status and details
            
        Example:
            >>> result = validator.validate_model_local("gpt-4o")
            >>> assert result.is_valid()
        """
        pass

    def validate_api_key_format(self) -> bool:
        """
        Validate API key format without making external calls.
        
        Returns:
            True if format is valid
            
        Raises:
            InvalidAPIKeyError: If format is invalid
            
        Example:
            >>> validator = OpenAIValidator(api_key="sk-proj-...")
            >>> validator.validate_api_key_format()
            True
        """
        key = self.config.api_key
        pattern = self.api_key_pattern()

        if not key:
            logger.error(
                "API key validation failed: key is empty",
                extra={"provider": self.config.provider.value},
            )
            raise InvalidAPIKeyError(
                provider=self.config.provider.value,
                key_preview="(empty)",
                expected_format=pattern.pattern,
            )

        if not isinstance(key, str):
            logger.error(
                f"API key validation failed: key is not a string (got {type(key).__name__})",
                extra={"provider": self.config.provider.value},
            )
            raise TypeError(f"API key must be a string, got {type(key).__name__}")

        if not pattern.match(key):
            key_preview = f"{key[:8]}..." if len(key) > 8 else key
            logger.warning(
                "API key format validation failed",
                extra={
                    "provider": self.config.provider.value,
                    "key_preview": key_preview,
                },
            )
            raise InvalidAPIKeyError(
                provider=self.config.provider.value,
                key_preview=key_preview,
                expected_format=pattern.pattern,
            )

        logger.debug(
            "API key format validated successfully",
            extra={"provider": self.config.provider.value},
        )
        return True

    def validate(
        self,
        model: str,
        check_live: bool = False,
    ) -> ValidationResult:
        """
        Perform complete validation: API key format + model name.
        
        This is the main entry point for validation operations.
        It orchestrates the validation workflow.
        
        Args:
            model: Model name to validate
            check_live: If True, make actual API call to verify (optional)
            
        Returns:
            ValidationResult with status and details
            
        Raises:
            InvalidAPIKeyError: If API key format is invalid
            Various ValidationErrors: Based on what fails
            
        Example:
            >>> result = validator.validate("gpt-4o")
            >>> if result.is_valid():
            ...     print("Configuration is valid!")
        """
        logger.info(
            f"Starting validation for {self.config.provider.value}",
            extra={
                "provider": self.config.provider.value,
                "model": model,
                "check_live": check_live,
            },
        )

        # Step 1: Validate API key format
        try:
            self.validate_api_key_format()
        except InvalidAPIKeyError:
            logger.error(
                "Validation failed at API key format check",
                extra={"provider": self.config.provider.value},
            )
            raise

        # Step 2: Validate model (local check)
        result = self.validate_model_local(model)

        # Step 3: Optional live validation
        if check_live and result.is_valid():
            logger.info(
                "Performing live API validation",
                extra={"provider": self.config.provider.value, "model": model},
            )
            # Subclasses can override validate_model_live if they support it
            # For now, we skip live checks in base implementation
            result.details["live_check"] = "skipped (not implemented)"

        logger.info(
            f"Validation complete: {result.status.value}",
            extra={
                "provider": self.config.provider.value,
                "model": model,
                "status": result.status.value,
            },
        )

        return result

    def __repr__(self) -> str:
        """Safe representation without exposing API key."""
        return f"{self.__class__.__name__}(provider={self.config.provider.value})"
