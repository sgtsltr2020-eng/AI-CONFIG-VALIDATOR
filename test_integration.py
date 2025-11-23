"""Quick integration test."""
from ai_config_validator import validate_and_discover, setup_logging

# Enable logging
setup_logging(level="INFO")

# Test with your API key
result = validate_and_discover(
    provider="openai",
    api_key="sk-proj-YOUR_KEY_HERE"  # Replace with real key
)

print("\n" + "="*60)
print("VALIDATION RESULT")
print("="*60)
print(f"Success: {result['success']}")
print(f"Provider: {result['provider']}")
print(f"Models Found: {result['model_count']}")

if result['success']:
    print(f"\n📊 Account Info:")
    acc = result['account_info']
    print(f"  Organization: {acc['organization']}")
    print(f"  Tier: {acc['tier']}")
    print(f"  Balance: ${acc['balance']['available_credits_usd']}")
    print(f"  Usage: {acc['balance']['usage_percent']}%")
    
    print(f"\n⭐ Featured Models:")
    for model in result['models'][:5]:
        if model.get('featured'):
            print(f"  • {model['name']} - ${model.get('cost_per_1k_input', 'N/A')}/1K tokens")
else:
    print(f"\n❌ Error: {result['error']['message']}")
    print(f"💡 {result['error']['suggestion']}")
