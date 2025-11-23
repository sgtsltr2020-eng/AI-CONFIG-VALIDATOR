# AI Config Validator

Enterprise-grade API configuration validator and failover router for LLM providers.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

✅ **Universal Validation** - Validate API keys and model configurations  
✅ **Auto-Discovery** - Automatically detect available models for your API key  
✅ **Account Info** - Check balance, usage, and billing information  
✅ **25+ OpenAI Models** - Complete catalog with costs and capabilities  
✅ **Smart Suggestions** - Fuzzy matching for typos  
✅ **Enterprise-Ready** - Structured logging, error handling, type safety

## Quick Start

### Installation

```bash
pip install -e .
```

### Python API

```python
from ai_config_validator import validate_and_discover

# Validate API key and discover models
result = validate_and_discover("openai", "sk-proj-...")

if result["success"]:
    print(f"Found {result['model_count']} models")
    print(f"Balance: ${result['account_info']['balance']['available_credits_usd']}")
    
    for model in result['models']:
        print(f"  • {model['name']}")
```

### Command Line

```bash
# Discover available models
ai-validator discover openai sk-proj-...

# Validate specific model
ai-validator validate openai sk-proj-... gpt-4o

# List all models
ai-validator models openai
```

## API Reference

### `validate_and_discover(provider, api_key)`

Complete validation with auto-discovery.

**Returns:**
```python
{
    "success": bool,
    "key_valid": bool,
    "provider": str,
    "models": List[Dict],  # Available models with metadata
    "model_count": int,
    "account_info": {
        "organization": str,
        "tier": str,
        "balance": {
            "available_credits_usd": float,
            "usage_this_month_usd": float,
            "usage_percent": float
        },
        "warnings": List[Dict]
    }
}
```

### `validate_llm_config(provider, api_key, model)`

Simple validation for specific model.

**Returns:** `ValidationResult` object

## Supported Providers

- ✅ OpenAI (25+ models)
- 🚧 Anthropic (coming soon)
- 🚧 Google (coming soon)
- 🚧 Cohere (coming soon)

## Development

### Run Tests

```bash
# Unit tests
pytest tests/test_openai_validator.py -v

# Integration tests (requires API key)
export OPENAI_API_KEY="sk-proj-..."
pytest tests/test_integration.py -v -m integration

# With coverage
pytest --cov=ai_config_validator --cov-report=html
```

### Code Quality

```bash
# Format code
black ai_config_validator tests

# Sort imports
isort ai_config_validator tests

# Type checking
mypy ai_config_validator

# Linting
ruff check ai_config_validator
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or PR.

## Roadmap

- [ ] Anthropic Claude support
- [ ] Google Gemini support
- [ ] Cohere support
- [ ] FastAPI/Flask integration examples
- [ ] Async API support
- [ ] Rate limit tracking
- [ ] Cost calculator utilities
