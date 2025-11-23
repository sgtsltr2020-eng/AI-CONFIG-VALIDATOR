"""
Command-line interface for AI Config Validator.

Provides easy testing and validation from the terminal.
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from . import validate_and_discover, validate_llm_config, setup_logging, __version__
from .errors import ValidationError

console = Console()


@click.group()
@click.version_option(version=__version__)
@click.option('--debug', is_flag=True, help='Enable debug logging')
def cli(debug: bool) -> None:
    """AI Config Validator - Enterprise-grade LLM API validation."""
    if debug:
        setup_logging(level="DEBUG")
    else:
        setup_logging(level="INFO", use_color=True)


@cli.command()
@click.argument('provider')
@click.argument('api_key')
def discover(provider: str, api_key: str) -> None:
    """
    Validate API key and discover available models.
    
    Example:
        ai-validator discover openai sk-proj-...
    """
    console.print(f"\n🔍 Validating {provider} API key...\n", style="bold cyan")
    
    result = validate_and_discover(provider, api_key)
    
    if not result["success"]:
        console.print("❌ Validation Failed", style="bold red")
        error = result["error"]
        console.print(f"\n{error['message']}", style="red")
        if error.get("suggestion"):
            console.print(f"💡 {error['suggestion']}", style="yellow")
        sys.exit(1)
    
    # Success - display results
    console.print("✅ API Key Valid!\n", style="bold green")
    
    # Account Info Panel
    acc = result["account_info"]
    balance_info = acc.get("balance", {})
    
    account_text = f"""
[bold]Organization:[/bold] {acc.get('organization', 'N/A')}
[bold]Tier:[/bold] {acc.get('tier', 'N/A')}

[bold cyan]💰 Account Balance[/bold cyan]
Available Credits: ${balance_info.get('available_credits_usd', 0):.2f}
Usage This Month: ${balance_info.get('usage_this_month_usd', 0):.2f}
Usage: {balance_info.get('usage_percent', 0):.1f}%
"""
    
    # Add warnings if present
    warnings = acc.get("warnings", [])
    if warnings:
        account_text += "\n[bold yellow]⚠️  Warnings:[/bold yellow]\n"
        for warning in warnings:
            account_text += f"  • {warning['message']}\n"
    
    console.print(Panel(account_text.strip(), title="Account Information", box=box.ROUNDED))
    
    # Models Table
    console.print(f"\n[bold]Found {result['model_count']} available models[/bold]\n")
    
    table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
    table.add_column("Model", style="cyan")
    table.add_column("Featured", justify="center")
    table.add_column("Input Cost", justify="right")
    table.add_column("Output Cost", justify="right")
    table.add_column("Max Tokens", justify="right")
    
    # Show featured models first, then top 10 others
    featured = [m for m in result["models"] if m.get("featured")]
    others = [m for m in result["models"] if not m.get("featured")][:10]
    
    for model in featured + others:
        featured_mark = "⭐" if model.get("featured") else ""
        input_cost = f"${model.get('cost_per_1k_input', 0):.4f}" if not model.get("unknown") else "N/A"
        output_cost = f"${model.get('cost_per_1k_output', 0):.4f}" if not model.get("unknown") else "N/A"
        max_tokens = f"{model.get('max_tokens', 0):,}" if not model.get("unknown") else "N/A"
        
        table.add_row(
            model["name"],
            featured_mark,
            input_cost,
            output_cost,
            max_tokens
        )
    
    console.print(table)
    
    if len(result["models"]) > len(featured) + 10:
        remaining = len(result["models"]) - len(featured) - 10
        console.print(f"\n[dim]... and {remaining} more models[/dim]")


@cli.command()
@click.argument('provider')
@click.argument('api_key')
@click.argument('model')
def validate(provider: str, api_key: str, model: str) -> None:
    """
    Validate a specific provider/model combination.
    
    Example:
        ai-validator validate openai sk-proj-... gpt-4o
    """
    console.print(f"\n🔍 Validating {provider}/{model}...\n", style="bold cyan")
    
    try:
        result = validate_llm_config(provider, api_key, model)
        
        if result.is_valid():
            console.print("✅ Valid Configuration", style="bold green")
            
            # Display details
            details_text = f"""
[bold]Model:[/bold] {result.model}
[bold]Provider:[/bold] {result.provider.value}
[bold]Status:[/bold] {result.status.value}

[bold cyan]💰 Cost Estimate[/bold cyan]
Input: ${result.details.get('cost_per_1k_input', 0):.4f} per 1K tokens
Output: ${result.details.get('cost_per_1k_output', 0):.4f} per 1K tokens

[bold cyan]✨ Features[/bold cyan]
Streaming: {'✓' if result.details.get('supports_streaming') else '✗'}
Function Calling: {'✓' if result.details.get('supports_function_calling') else '✗'}
Vision: {'✓' if result.details.get('supports_vision') else '✗'}

[bold cyan]📊 Limits[/bold cyan]
Max Tokens: {result.details.get('max_tokens', 0):,}
"""
            console.print(Panel(details_text.strip(), title="Validation Result", box=box.ROUNDED))
            
        else:
            console.print("❌ Invalid Configuration", style="bold red")
            console.print(f"\n{result.message}", style="red")
            if result.suggestion:
                console.print(f"💡 {result.suggestion}", style="yellow")
            sys.exit(1)
            
    except ValidationError as e:
        console.print("❌ Validation Error", style="bold red")
        console.print(f"\n{e.message}", style="red")
        if e.suggestion:
            console.print(f"💡 {e.suggestion}", style="yellow")
        sys.exit(1)


@cli.command()
@click.argument('provider')
def models(provider: str) -> None:
    """
    List all supported models for a provider.
    
    Example:
        ai-validator models openai
        ai-validator models anthropic
    """
    from .validators.openai import OpenAIValidator
    from .validators.anthropic import AnthropicValidator
    from .validators.google import GoogleValidator
    from .validators.groq import GroqValidator
    
    provider_lower = provider.lower()
    
    if provider_lower == "openai":
        all_models = OpenAIValidator.get_all_models()
        featured = OpenAIValidator.get_featured_models()
        provider_display = "OpenAI"
        
    elif provider_lower == "anthropic":
        all_models = AnthropicValidator.get_all_models()
        featured = AnthropicValidator.get_featured_models()
        provider_display = "Anthropic (Claude)"
        
    elif provider_lower == "google":
        all_models = GoogleValidator.get_all_models()
        featured = GoogleValidator.get_featured_models()
        provider_display = "Google Gemini"
        
    elif provider_lower == "groq":
        all_models = GroqValidator.get_all_models()
        featured = GroqValidator.get_featured_models()
        provider_display = "Groq"
        
    else:
        console.print(f"❌ Provider '{provider}' not supported", style="red")
        console.print("\nSupported providers: openai, anthropic, google, groq", style="yellow")
        sys.exit(1)
    
    console.print(f"\n[bold]{provider_display} Models ({len(all_models)} total)[/bold]\n")
    
    console.print("[bold cyan]⭐ Featured Models:[/bold cyan]")
    for model in featured:
        console.print(f"  • {model}")
    
    console.print(f"\n[bold]All Models:[/bold]")
    for model in all_models:
        if model not in featured:
            console.print(f"  • {model}")


if __name__ == "__main__":
    cli()
