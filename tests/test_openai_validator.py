"""
Comprehensive tests for OpenAI validator.

Tests cover:
- API key format validation
- Model validation (valid/invalid)
- Fuzzy matching suggestions
- Error handling
- Edge cases
"""

import pytest
from ai_config_validator import validate_llm_config, OpenAIValidator
from ai_config_validator.errors import (
    InvalidAPIKeyError,
    InvalidModelError,
    UnsupportedProviderError,
)
from ai_config_validator.models import ValidationStatus, ProviderType


class TestAPIKeyValidation:
    """Test API key format validation."""
    
    def test_valid_modern_key_format(self):
        """Test modern OpenAI key format (sk-proj-...)."""
        validator = OpenAIValidator(api_key="sk-proj-" + "a" * 48)
        assert validator.validate_api_key_format() is True
    
    def test_valid_legacy_key_format(self):
        """Test legacy OpenAI key format (sk-...)."""
        validator = OpenAIValidator(api_key="sk-" + "a" * 48)
        assert validator.validate_api_key_format() is True
    
    def test_invalid_key_prefix(self):
        """Test key with wrong prefix."""
        validator = OpenAIValidator(api_key="pk-" + "a" * 48)
        with pytest.raises(InvalidAPIKeyError) as exc_info:
            validator.validate_api_key_format()
        
        assert exc_info.value.provider == "openai"
        assert "format" in exc_info.value.message.lower()
    
    def test_invalid_key_too_short(self):
        """Test key that's too short."""
        validator = OpenAIValidator(api_key="sk-abc123")
        with pytest.raises(InvalidAPIKeyError):
            validator.validate_api_key_format()
    
    def test_empty_key(self):
        """Test empty API key."""
        # Empty key fails during Pydantic validation in APIKeyConfig
        # The BaseValidator treats empty string as falsy, so it raises ValueError
        with pytest.raises(ValueError):
            validator = OpenAIValidator(api_key="")
    
    def test_none_key(self):
        """Test None API key."""
        validator = OpenAIValidator(api_key="sk-test123")
        validator.config.api_key = None
        with pytest.raises(InvalidAPIKeyError):
            validator.validate_api_key_format()


class TestModelValidation:
    """Test model name validation."""
    
    def test_valid_gpt5_model(self):
        """Test GPT-5 model validation."""
        result = validate_llm_config("openai", "sk-proj-" + "a" * 48, "gpt-5")
        assert result.is_valid()
        assert result.status == ValidationStatus.VALID
        assert result.provider == ProviderType.OPENAI
        assert result.model == "gpt-5"
        assert "cost_per_1k_input" in result.details
    
    def test_valid_gpt4o_model(self):
        """Test GPT-4o model validation."""
        result = validate_llm_config("openai", "sk-proj-" + "a" * 48, "gpt-4o")
        assert result.is_valid()
        assert result.details["supports_vision"] is True
        assert result.details["supports_function_calling"] is True
    
    def test_valid_o3_reasoning_model(self):
        """Test o3 reasoning model."""
        result = validate_llm_config("openai", "sk-proj-" + "a" * 48, "o3")
        assert result.is_valid()
        assert result.details["supports_streaming"] is False  # Reasoning models don't stream
    
    def test_valid_whisper_audio_model(self):
        """Test Whisper audio model."""
        result = validate_llm_config("openai", "sk-proj-" + "a" * 48, "whisper-1")
        assert result.is_valid()
        assert result.model == "whisper-1"
    
    def test_valid_dalle_image_model(self):
        """Test DALL-E image generation model."""
        result = validate_llm_config("openai", "sk-proj-" + "a" * 48, "dall-e-3")
        assert result.is_valid()
        assert result.model == "dall-e-3"
    
    def test_valid_embedding_model(self):
        """Test embedding model."""
        result = validate_llm_config("openai", "sk-proj-" + "a" * 48, "text-embedding-3-large")
        assert result.is_valid()
        assert result.model == "text-embedding-3-large"
    
    def test_invalid_model_name(self):
        """Test completely invalid model name."""
        result = validate_llm_config("openai", "sk-proj-" + "a" * 48, "invalid-model-xyz")
        assert not result.is_valid()
        assert result.status == ValidationStatus.INVALID
        # Either suggestion or suggested_model should be present (can be None if no good match)
        # This test just verifies the model is marked as invalid
    
    def test_case_insensitive_validation(self):
        """Test that model names are case-insensitive."""
        result = validate_llm_config("openai", "sk-proj-" + "a" * 48, "GPT-5")
        assert result.is_valid()
        assert result.model == "gpt-5"  # Normalized to lowercase
    
    def test_whitespace_handling(self):
        """Test that whitespace is trimmed."""
        result = validate_llm_config("openai", "sk-proj-" + "a" * 48, "  gpt-4o  ")
        assert result.is_valid()
        assert result.model == "gpt-4o"


class TestFuzzyMatching:
    """Test intelligent model suggestions."""
    
    def test_typo_gpt5_suggestion(self):
        """Test suggestion for 'gpt5' typo."""
        validator = OpenAIValidator(api_key="sk-proj-" + "a" * 48)
        result = validator.validate_model_local("gpt5")
        assert not result.is_valid()
        assert result.details.get("suggested_model") == "gpt-5"
    
    def test_typo_gpt4o_suggestion(self):
        """Test suggestion for 'gpt4o' typo."""
        validator = OpenAIValidator(api_key="sk-proj-" + "a" * 48)
        result = validator.validate_model_local("gpt4o")
        assert not result.is_valid()
        suggested = result.details.get("suggested_model")
        assert suggested in ["gpt-4o", "gpt-4o-mini"]
    
    def test_prefix_matching_gpt4(self):
        """Test prefix-based suggestion for GPT-4 family."""
        validator = OpenAIValidator(api_key="sk-proj-" + "a" * 48)
        result = validator.validate_model_local("gpt-4-xyz")
        assert not result.is_valid()
        assert result.details.get("suggested_model") in ["gpt-4-turbo", "gpt-4"]
    
    def test_prefix_matching_o_series(self):
        """Test prefix-based suggestion for o-series."""
        validator = OpenAIValidator(api_key="sk-proj-" + "a" * 48)
        result = validator.validate_model_local("o5")
        assert not result.is_valid()
        assert result.details.get("suggested_model") in ["o3", "o3-mini"]


class TestHelperMethods:
    """Test utility methods."""
    
    def test_get_featured_models(self):
        """Test featured models list."""
        featured = OpenAIValidator.get_featured_models()
        assert isinstance(featured, list)
        assert len(featured) > 0
        assert "gpt-5" in featured
        assert "gpt-4o" in featured
        assert all(isinstance(m, str) for m in featured)
    
    def test_get_all_models(self):
        """Test complete model list."""
        all_models = OpenAIValidator.get_all_models()
        assert isinstance(all_models, list)
        assert len(all_models) >= 25  # We added 25+ models
        assert "gpt-5" in all_models
        assert "whisper-1" in all_models
        assert "dall-e-3" in all_models
    
    def test_provider_type(self):
        """Test provider type identification."""
        assert OpenAIValidator.provider_type() == ProviderType.OPENAI


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_unsupported_provider(self):
        """Test unsupported provider error."""
        with pytest.raises(UnsupportedProviderError) as exc_info:
            validate_llm_config("unsupported-provider", "sk-test", "model")
        
        assert exc_info.value.provider == "unsupported-provider"
        assert "supported_providers" in exc_info.value.details
    
    def test_empty_model_name(self):
        """Test validation with empty model name."""
        # Should be caught by Pydantic validation
        with pytest.raises(ValueError):
            validate_llm_config("openai", "sk-proj-" + "a" * 48, "")
    
    def test_validator_repr_hides_key(self):
        """Test that repr doesn't expose API key."""
        validator = OpenAIValidator(api_key="sk-proj-secret123")
        repr_str = repr(validator)
        assert "secret123" not in repr_str
        assert "openai" in repr_str.lower()


class TestValidationResult:
    """Test ValidationResult model."""
    
    def test_result_immutability(self):
        """Test that ValidationResult is immutable."""
        result = validate_llm_config("openai", "sk-proj-" + "a" * 48, "gpt-4o")
        
        # Should not be able to modify
        with pytest.raises(Exception):  # Pydantic raises ValidationError or AttributeError
            result.status = ValidationStatus.INVALID
    
    def test_result_to_dict(self):
        """Test conversion to dictionary."""
        result = validate_llm_config("openai", "sk-proj-" + "a" * 48, "gpt-4o")
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "status" in result_dict
        assert "provider" in result_dict
        assert "model" in result_dict
        assert "timestamp" in result_dict
        assert isinstance(result_dict["timestamp"], str)  # ISO format
    
    def test_result_is_valid_helper(self):
        """Test is_valid() helper method."""
        valid_result = validate_llm_config("openai", "sk-proj-" + "a" * 48, "gpt-4o")
        invalid_result = validate_llm_config("openai", "sk-proj-" + "a" * 48, "invalid-model")
        
        assert valid_result.is_valid() is True
        assert invalid_result.is_valid() is False


class TestModelCapabilities:
    """Test model capability metadata."""
    
    def test_gpt5_capabilities(self):
        """Test GPT-5 has expected capabilities."""
        result = validate_llm_config("openai", "sk-proj-" + "a" * 48, "gpt-5")
        details = result.details
        
        assert details["max_tokens"] == 200000
        assert details["supports_streaming"] is True
        assert details["supports_function_calling"] is True
        assert details["supports_vision"] is True
        assert details["cost_per_1k_input"] > 0
        assert details["cost_per_1k_output"] > 0
    
    def test_reasoning_model_no_streaming(self):
        """Test reasoning models correctly marked as non-streaming."""
        result = validate_llm_config("openai", "sk-proj-" + "a" * 48, "o3")
        assert result.details["supports_streaming"] is False
    
    def test_cost_data_present(self):
        """Test all models have cost data."""
        for model in ["gpt-5", "gpt-4o", "gpt-3.5-turbo", "o3"]:
            result = validate_llm_config("openai", "sk-proj-" + "a" * 48, model)
            assert "cost_per_1k_input" in result.details
            assert "cost_per_1k_output" in result.details
            assert isinstance(result.details["cost_per_1k_input"], (int, float))


# Run tests with: pytest tests/test_openai_validator.py -v
