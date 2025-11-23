"""
Tests for Groq validator.
"""

import pytest
from ai_config_validator import validate_llm_config, GroqValidator
from ai_config_validator.errors import InvalidAPIKeyError
from ai_config_validator.models import ValidationStatus, ProviderType


class TestGroqAPIKeyValidation:
    """Test Groq API key format validation."""
    
    def test_valid_groq_key_format(self):
        """Test valid Groq key format."""
        validator = GroqValidator(api_key="A" * 24 + "_-" + "b" * 8)
        assert validator.validate_api_key_format() is True
    
    def test_invalid_key_too_short(self):
        """Test key that's too short."""
        validator = GroqValidator(api_key="short_key")
        with pytest.raises(InvalidAPIKeyError):
            validator.validate_api_key_format()
    
    def test_invalid_key_with_space(self):
        """Test key with invalid space character."""
        validator = GroqValidator(api_key="invalid key with space")
        with pytest.raises(InvalidAPIKeyError):
            validator.validate_api_key_format()


class TestGroqModelValidation:
    """Test Groq model validation."""
    
    def test_valid_llama_3_1_8b(self):
        """Test Llama 3.1 8B validation."""
        result = validate_llm_config(
            "groq",
            "A" * 24 + "_-" + "b" * 8,
            "llama-3.1-8b-instant",
        )
        assert result.is_valid()
        assert result.provider == ProviderType.GROQ
        assert "cost_per_1k_input" in result.details
    
    def test_valid_llama_3_3_70b(self):
        """Test Llama 3.3 70B validation."""
        result = validate_llm_config(
            "groq",
            "A" * 24 + "_-" + "b" * 8,
            "llama-3.3-70b-versatile",
        )
        assert result.is_valid()
        assert result.details["supports_streaming"] is True
    
    def test_valid_gpt_oss_120b(self):
        """Test GPT-OSS 120B validation."""
        result = validate_llm_config(
            "groq",
            "A" * 24 + "_-" + "b" * 8,
            "openai/gpt-oss-120b",
        )
        assert result.is_valid()
    
    def test_valid_gpt_oss_20b(self):
        """Test GPT-OSS 20B validation."""
        result = validate_llm_config(
            "groq",
            "A" * 24 + "_-" + "b" * 8,
            "openai/gpt-oss-20b",
        )
        assert result.is_valid()
    
    def test_invalid_model_name(self):
        """Test invalid model name."""
        result = validate_llm_config(
            "groq",
            "A" * 24 + "_-" + "b" * 8,
            "non-existent-model",
        )
        assert not result.is_valid()
        assert result.status == ValidationStatus.INVALID
    
    def test_case_insensitive_validation(self):
        """Test case-insensitive model names."""
        result = validate_llm_config(
            "groq",
            "A" * 24 + "_-" + "b" * 8,
            "LLAMA-3.3-70B-VERSATILE",
        )
        assert result.is_valid()


class TestGroqFuzzyMatching:
    """Test Groq model suggestions."""
    
    def test_typo_llama70_suggestion(self):
        """Test suggestion for 'llama70b' typo."""
        validator = GroqValidator(api_key="A" * 24 + "_-" + "b" * 8)
        result = validator.validate_model_local("llama70b")
        assert not result.is_valid()
        assert result.details.get("suggested_model") is not None
    
    def test_keyword_matching_whisper(self):
        """Test keyword matching for Whisper."""
        validator = GroqValidator(api_key="A" * 24 + "_-" + "b" * 8)
        result = validator.validate_model_local("whisper")
        assert not result.is_valid()
        assert "whisper" in result.details.get("suggested_model", "")


class TestGroqHelperMethods:
    """Test Groq utility methods."""
    
    def test_get_featured_models(self):
        """Test featured models list."""
        featured = GroqValidator.get_featured_models()
        assert isinstance(featured, list)
        assert len(featured) >= 3
        assert "llama-3.1-8b-instant" in featured
        assert "llama-3.3-70b-versatile" in featured
    
    def test_get_all_models(self):
        """Test complete model list."""
        all_models = GroqValidator.get_all_models()
        assert isinstance(all_models, list)
        assert len(all_models) >= 7
        assert "llama-3.1-8b-instant" in all_models
        assert "openai/gpt-oss-120b" in all_models
    
    def test_provider_type(self):
        """Test provider type identification."""
        assert GroqValidator.provider_type() == ProviderType.GROQ
