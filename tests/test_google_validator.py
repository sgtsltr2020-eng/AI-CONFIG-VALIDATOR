"""
Tests for Google Gemini validator.
"""

import pytest
from ai_config_validator import validate_llm_config
from ai_config_validator.validators.google import GoogleValidator
from ai_config_validator.errors import InvalidAPIKeyError
from ai_config_validator.models import ValidationStatus, ProviderType


class TestGoogleAPIKeyValidation:
    """Test Google API key format validation."""
    
    def test_valid_google_key_format(self):
        """Test valid Google key format."""
        validator = GoogleValidator(api_key="AIza" + "a" * 35)
        assert validator.validate_api_key_format() is True
    
    def test_invalid_key_prefix(self):
        """Test key with wrong prefix."""
        validator = GoogleValidator(api_key="sk-proj-" + "a" * 35)
        with pytest.raises(InvalidAPIKeyError):
            validator.validate_api_key_format()
    
    def test_invalid_key_too_short(self):
        """Test key that's too short."""
        validator = GoogleValidator(api_key="AIza-short")
        with pytest.raises(InvalidAPIKeyError):
            validator.validate_api_key_format()


class TestGoogleModelValidation:
    """Test Google model validation."""
    
    def test_valid_gemini_3_pro(self):
        """Test Gemini 3 Pro validation."""
        result = validate_llm_config(
            "google",
            "AIza" + "a" * 35,
            "gemini-3-pro-preview"
        )
        assert result.is_valid()
        assert result.provider == ProviderType.GOOGLE
        assert "cost_per_1k_input" in result.details
    
    def test_valid_gemini_2_5_pro(self):
        """Test Gemini 2.5 Pro validation."""
        result = validate_llm_config(
            "google",
            "AIza" + "a" * 35,
            "gemini-2.5-pro"
        )
        assert result.is_valid()
        assert result.details["supports_vision"] is True
    
    def test_valid_gemini_2_5_flash(self):
        """Test Gemini 2.5 Flash validation."""
        result = validate_llm_config(
            "google",
            "AIza" + "a" * 35,
            "gemini-2.5-flash"
        )
        assert result.is_valid()
        assert result.details["supports_function_calling"] is True
    
    def test_valid_gemini_2_0_flash(self):
        """Test Gemini 2.0 Flash validation."""
        result = validate_llm_config(
            "google",
            "AIza" + "a" * 35,
            "gemini-2.0-flash"
        )
        assert result.is_valid()
    
    def test_invalid_model_name(self):
        """Test invalid model name."""
        result = validate_llm_config(
            "google",
            "AIza" + "a" * 35,
            "invalid-model"
        )
        assert not result.is_valid()
        assert result.status == ValidationStatus.INVALID
    
    def test_case_insensitive_validation(self):
        """Test case-insensitive model names."""
        result = validate_llm_config(
            "google",
            "AIza" + "a" * 35,
            "GEMINI-3-PRO-PREVIEW"
        )
        assert result.is_valid()


class TestGoogleFuzzyMatching:
    """Test Google model suggestions."""
    
    def test_typo_gemini3_suggestion(self):
        """Test suggestion for 'gemini3' typo."""
        validator = GoogleValidator(api_key="AIza" + "a" * 35)
        result = validator.validate_model_local("gemini3")
        assert not result.is_valid()
        assert result.details.get("suggested_model") is not None
    
    def test_keyword_matching_pro(self):
        """Test keyword matching for Pro."""
        validator = GoogleValidator(api_key="AIza" + "a" * 35)
        result = validator.validate_model_local("pro")
        assert not result.is_valid()
        assert "pro" in result.details.get("suggested_model", "")


class TestGoogleHelperMethods:
    """Test Google utility methods."""
    
    def test_get_featured_models(self):
        """Test featured models list."""
        featured = GoogleValidator.get_featured_models()
        assert isinstance(featured, list)
        assert len(featured) >= 3
        assert "gemini-3-pro-preview" in featured
        assert "gemini-2.5-flash" in featured
    
    def test_get_all_models(self):
        """Test complete model list."""
        all_models = GoogleValidator.get_all_models()
        assert isinstance(all_models, list)
        assert len(all_models) >= 8
        assert "gemini-3-pro-preview" in all_models
    
    def test_provider_type(self):
        """Test provider type identification."""
        assert GoogleValidator.provider_type() == ProviderType.GOOGLE


# Run tests with: pytest tests/test_google_validator.py -v
