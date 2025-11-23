"""
Tests for Anthropic (Claude) validator.
"""

import pytest
from ai_config_validator import validate_llm_config
from ai_config_validator.validators.anthropic import AnthropicValidator
from ai_config_validator.errors import InvalidAPIKeyError
from ai_config_validator.models import ValidationStatus, ProviderType


class TestAnthropicAPIKeyValidation:
    """Test Anthropic API key format validation."""
    
    def test_valid_anthropic_key_format(self):
        """Test valid Anthropic key format."""
        validator = AnthropicValidator(api_key="sk-ant-api03-" + "a" * 95)
        assert validator.validate_api_key_format() is True
    
    def test_invalid_key_prefix(self):
        """Test key with wrong prefix."""
        validator = AnthropicValidator(api_key="sk-proj-" + "a" * 95)
        with pytest.raises(InvalidAPIKeyError):
            validator.validate_api_key_format()
    
    def test_invalid_key_too_short(self):
        """Test key that's too short."""
        validator = AnthropicValidator(api_key="sk-ant-api03-short")
        with pytest.raises(InvalidAPIKeyError):
            validator.validate_api_key_format()


class TestAnthropicModelValidation:
    """Test Anthropic model validation."""
    
    def test_valid_claude_sonnet_4(self):
        """Test Claude Sonnet 4 validation."""
        result = validate_llm_config(
            "anthropic",
            "sk-ant-api03-" + "a" * 95,
            "claude-sonnet-4-20250514"
        )
        assert result.is_valid()
        assert result.provider == ProviderType.ANTHROPIC
        assert "cost_per_1k_input" in result.details
    
    def test_valid_claude_3_5_sonnet(self):
        """Test Claude 3.5 Sonnet validation."""
        result = validate_llm_config(
            "anthropic",
            "sk-ant-api03-" + "a" * 95,
            "claude-3-5-sonnet-20241022"
        )
        assert result.is_valid()
        assert result.details["supports_vision"] is True
    
    def test_valid_claude_3_opus(self):
        """Test Claude 3 Opus validation."""
        result = validate_llm_config(
            "anthropic",
            "sk-ant-api03-" + "a" * 95,
            "claude-3-opus-20240229"
        )
        assert result.is_valid()
        assert result.details["supports_function_calling"] is True
    
    def test_invalid_model_name(self):
        """Test invalid model name."""
        result = validate_llm_config(
            "anthropic",
            "sk-ant-api03-" + "a" * 95,
            "invalid-model"
        )
        assert not result.is_valid()
        assert result.status == ValidationStatus.INVALID
    
    def test_case_insensitive_validation(self):
        """Test case-insensitive model names."""
        result = validate_llm_config(
            "anthropic",
            "sk-ant-api03-" + "a" * 95,
            "CLAUDE-3-OPUS-20240229"
        )
        assert result.is_valid()


class TestAnthropicFuzzyMatching:
    """Test Anthropic model suggestions."""
    
    def test_typo_claude4_suggestion(self):
        """Test suggestion for 'claude4' typo."""
        validator = AnthropicValidator(api_key="sk-ant-api03-" + "a" * 95)
        result = validator.validate_model_local("claude4")
        assert not result.is_valid()
        suggested = result.details.get("suggested_model")
        assert suggested is not None
        assert "claude-sonnet-4" in suggested
    
    def test_keyword_matching_opus(self):
        """Test keyword matching for Opus."""
        validator = AnthropicValidator(api_key="sk-ant-api03-" + "a" * 95)
        result = validator.validate_model_local("opus")
        assert not result.is_valid()
        assert "opus" in result.details.get("suggested_model", "")


class TestAnthropicHelperMethods:
    """Test Anthropic utility methods."""
    
    def test_get_featured_models(self):
        """Test featured models list."""
        featured = AnthropicValidator.get_featured_models()
        assert isinstance(featured, list)
        assert len(featured) >= 3
        assert "claude-3-5-sonnet-20241022" in featured
    
    def test_get_all_models(self):
        """Test complete model list."""
        all_models = AnthropicValidator.get_all_models()
        assert isinstance(all_models, list)
        assert len(all_models) >= 6
    
    def test_provider_type(self):
        """Test provider type identification."""
        assert AnthropicValidator.provider_type() == ProviderType.ANTHROPIC


# Run tests with: pytest tests/test_anthropic_validator.py -v
