"""
Integration tests for validate_and_discover API.

These tests require a real API key (set via environment variable).
They are marked as integration tests and can be skipped.
"""

import os
import pytest
from ai_config_validator import validate_and_discover


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def openai_api_key():
    """Get OpenAI API key from environment."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return key


class TestValidateAndDiscover:
    """Integration tests for validate_and_discover function."""
    
    def test_successful_validation(self, openai_api_key):
        """Test complete validation flow with real API key."""
        result = validate_and_discover("openai", openai_api_key)
        
        assert result["success"] is True
        assert result["key_valid"] is True
        assert result["provider"] == "openai"
        assert result["model_count"] > 0
        assert len(result["models"]) > 0
        assert "account_info" in result
        assert result["error"] is None
    
    def test_invalid_api_key(self):
        """Test validation with invalid API key."""
        result = validate_and_discover("openai", "sk-invalid-key")
        
        assert result["success"] is False
        assert result["key_valid"] is False
        assert result["error"] is not None
        assert "error" in result["error"]
        assert "message" in result["error"]
    
    def test_model_enrichment(self, openai_api_key):
        """Test that discovered models are enriched with metadata."""
        result = validate_and_discover("openai", openai_api_key)
        
        if result["success"] and len(result["models"]) > 0:
            first_model = result["models"][0]
            
            # Check required fields
            assert "id" in first_model
            assert "name" in first_model
            assert "featured" in first_model
            
            # If model is in catalog, should have rich metadata
            if not first_model.get("unknown"):
                assert "max_tokens" in first_model
                assert "cost_per_1k_input" in first_model
                assert "supports_streaming" in first_model
    
    def test_account_info_structure(self, openai_api_key):
        """Test account info has expected structure."""
        result = validate_and_discover("openai", openai_api_key)
        
        if result["success"]:
            acc = result["account_info"]
            
            # Should have these fields (even if some are "Unknown")
            assert "organization" in acc
            assert "tier" in acc


# Run with: pytest tests/test_integration.py -v -m integration
# Skip integration tests: pytest -v -m "not integration"
