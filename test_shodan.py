#!/usr/bin/env python3
"""
Test script for Shodan integration
This script tests basic Shodan API functionality
"""

import os
import sys
import logging
from dotenv import load_dotenv
from rich.console import Console
from src.tools.shodan_tool import ShodanSearchTool

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize console
console = Console()

def test_shodan_integration():
    """Test basic Shodan functionality"""
    
    console.print("[bold cyan]Testing Shodan Integration[/bold cyan]")
    
    # Load environment variables
    load_dotenv()
    
    # Check if Shodan API key is available
    shodan_api_key = os.getenv("SHODAN_API_KEY")
    if not shodan_api_key:
        console.print("[bold red]Error: SHODAN_API_KEY not found in environment variables[/bold red]")
        console.print("Please add your Shodan API key to the .env file with the key SHODAN_API_KEY")
        sys.exit(1)
    
    # Initialize Shodan tool
    try:
        console.print("Initializing Shodan tool...")
        shodan_tool = ShodanSearchTool(shodan_api_key)
    except Exception as e:
        console.print(f"[bold red]Error initializing Shodan tool: {str(e)}[/bold red]")
        sys.exit(1)
    
    # Test API info
    try:
        console.print("\n[bold cyan]Testing API Info:[/bold cyan]")
        api_info = shodan_tool.api_info()
        
        if "error" in api_info:
            console.print(f"[bold red]API Error: {api_info['error']}[/bold red]")
        else:
            console.print(f"API Plan: {api_info.get('plan', 'Unknown')}")
            console.print(f"Query Credits: {api_info.get('query_credits', 0)}")
            console.print(f"Scan Credits: {api_info.get('scan_credits', 0)}")
            console.print("[bold green]API Info test successful![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error testing API info: {str(e)}[/bold red]")
    
    # Test search functionality with a simple query
    try:
        console.print("\n[bold cyan]Testing Search Functionality:[/bold cyan]")
        search_results = shodan_tool.search("apache country:US", limit=1)
        
        if "error" in search_results:
            console.print(f"[bold red]Search Error: {search_results['error']}[/bold red]")
        else:
            total = search_results.get("total", 0)
            matches = len(search_results.get("matches", []))
            console.print(f"Total results: {total}")
            console.print(f"Returned matches: {matches}")
            
            if matches > 0:
                match = search_results["matches"][0]
                console.print(f"Sample result: {match.get('ip_str')}:{match.get('port')} - {match.get('org', 'Unknown')}")
                console.print("[bold green]Search test successful![/bold green]")
            else:
                console.print("[yellow]Search returned no results. This might be normal.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error testing search: {str(e)}[/bold red]")
    
    # Output success message if we got here
    console.print("\n[bold green]Shodan integration tests complete![/bold green]")
    console.print("If any errors occurred, please check your API key and network connection.")
    console.print("For more detailed testing, try the shodan_cli.py script.")

if __name__ == "__main__":
    test_shodan_integration()
