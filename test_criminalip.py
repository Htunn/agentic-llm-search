#!/usr/bin/env python3
"""
Test script for Criminal IP integration with agentic-llm-search

This script demonstrates how to use the Criminal IP integration
with the agentic LLM search.
"""

import os
import sys
import argparse
import asyncio
import logging
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our tools directly for API key verification
from src.tools.criminalip_tool import CriminalIPTool

# Import the agent with Criminal IP integration
from src.agents.agentic_llm_with_criminalip import AgenticLLM

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

console = Console()

def verify_api_key():
    """Verify that the Criminal IP API key is configured correctly"""
    api_key = os.getenv("CRIMINAL_IP_API_KEY")
    
    if not api_key:
        console.print("[bold red]ERROR: CRIMINAL_IP_API_KEY environment variable not set[/bold red]")
        console.print("To set up your API key, run: [bold]./criminalip_cli.py setup[/bold]")
        return False
    
    # Create a tool instance and verify the API key
    tool = CriminalIPTool(api_key)
    validation = tool.verify_api_key()
    
    if not validation["success"]:
        console.print(f"[bold red]API key validation failed: {validation.get('message', 'Unknown error')}[/bold red]")
        if "solution" in validation:
            console.print(f"[yellow]Suggestion: {validation['solution']}[/yellow]")
        return False
    
    console.print("[bold green]✓ Criminal IP API key verified successfully[/bold green]")
    return True

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Test Criminal IP integration with agentic-llm-search")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="command")
    
    # IP lookup command
    ip_parser = subparsers.add_parser("ip", help="Look up information about an IP address")
    ip_parser.add_argument("ip_address", help="IP address to look up")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search Criminal IP")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results (default: 5)")
    
    # Models command to show available models
    subparsers.add_parser("models", help="List available LLM models")
    
    # Options for all commands
    parser.add_argument("--model", help="Name of the LLM model to use")
    parser.add_argument("--provider", help="Provider of the LLM model (huggingface, openai, azure-openai)")
    
    return parser.parse_args()

def check_api_key():
    """Check if the Criminal IP API key is set"""
    # Use the more thorough verification function
    return verify_api_key()

def list_available_models():
    """List the available LLM models"""
    console.print("[bold]Available LLM Models:[/bold]")
    console.print("\n[bold cyan]Built-in Models:[/bold cyan]")
    console.print("  ./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (local)")
    
    console.print("\n[bold cyan]OpenAI Models:[/bold cyan]")
    console.print("  gpt-4")
    console.print("  gpt-4-turbo")
    console.print("  gpt-3.5-turbo")
    
    console.print("\n[bold cyan]HuggingFace Models:[/bold cyan]")
    console.print("  deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    console.print("  meta-llama/Llama-2-7b-chat-hf")
    console.print("  mistralai/Mistral-7B-Instruct-v0.2")
    
    console.print("\nTo specify a model, use the --model flag:")
    console.print("  python test_criminalip.py ip 8.8.8.8 --model gpt-4 --provider openai")
    console.print("  python test_criminalip.py search apache --model ./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

async def ip_lookup(args):
    """Look up information about an IP address"""
    if not check_api_key():
        return
    
    ip_address = args.ip_address
    console.print(f"[bold cyan]Looking up information for IP: {ip_address}[/bold cyan]")
    
    # Initialize the agent with optional model overrides
    agent = AgenticLLM(
        model_name=args.model,
        model_provider=args.provider
    )
    
    # Look up the IP address
    with console.status("[bold green]Analyzing IP address...[/bold green]"):
        response = agent.process_criminalip_ip(ip_address)
    
    # Display the answer
    console.print(Panel(
        Markdown(response.answer),
        title=f"Criminal IP Analysis for {ip_address}",
        border_style="cyan"
    ))
    
    # Display the sources
    if response.sources:
        console.print("\n[bold]Sources:[/bold]")
        for i, source in enumerate(response.sources, 1):
            console.print(f"{i}. [link={source.url}]{source.title}[/link]")

async def search(args):
    """Search Criminal IP"""
    if not check_api_key():
        return
    
    query = args.query
    limit = args.limit
    
    console.print(f"[bold cyan]Searching Criminal IP for: {query} (limit: {limit})[/bold cyan]")
    
    # Initialize the agent with optional model overrides
    agent = AgenticLLM(
        model_name=args.model,
        model_provider=args.provider
    )
    
    # Perform the search
    with console.status("[bold green]Searching and analyzing...[/bold green]"):
        response = agent.process_criminalip_search(query, limit)
    
    # Display the answer
    console.print(Panel(
        Markdown(response.answer),
        title=f"Criminal IP Search Results for '{query}'",
        border_style="cyan"
    ))
    
    # Display the sources
    if response.sources:
        console.print("\n[bold]Sources:[/bold]")
        for i, source in enumerate(response.sources, 1):
            console.print(f"{i}. [link={source.url}]{source.title}[/link]")

async def main():
    """Main function"""
    args = parse_args()
    
    if not args.command:
        console.print("[bold red]Error: No command specified[/bold red]")
        console.print("Run with --help for usage information")
        return
    
    if args.command == "ip":
        await ip_lookup(args)
    elif args.command == "search":
        await search(args)
    elif args.command == "models":
        list_available_models()
    else:
        console.print(f"[bold red]Unknown command: {args.command}[/bold red]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation cancelled[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
