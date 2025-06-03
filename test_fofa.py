#!/usr/bin/env python3
"""
Test script for FOFA integration
This script tests basic FOFA API functionality
"""

import os
import sys
import logging
from dotenv import load_dotenv
from rich.console import Console
from src.tools.fofa_tool import FofaSearchTool

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize console
console = Console()

def test_fofa_integration():
    """Test basic FOFA functionality"""
    
    console.print("[bold cyan]Testing FOFA Integration[/bold cyan]")
    
    # Load environment variables
    load_dotenv()
    
    # Check if FOFA API credentials are available
    fofa_email = os.getenv("FOFA_EMAIL")
    fofa_api_key = os.getenv("FOFA_API_KEY")
    
    if not fofa_email or not fofa_api_key:
        console.print("[bold red]Error: FOFA credentials not found in environment variables[/bold red]")
        console.print("Please add your FOFA email and API key to the .env file:")
        console.print("FOFA_EMAIL=your-email@example.com")
        console.print("FOFA_API_KEY=your-fofa-api-key")
        sys.exit(1)
    
    # Initialize FOFA tool
    try:
        console.print("Initializing FOFA tool...")
        fofa_tool = FofaSearchTool(fofa_email, fofa_api_key)
    except Exception as e:
        console.print(f"[bold red]Error initializing FOFA tool: {str(e)}[/bold red]")
        sys.exit(1)
    
    # Test API info/account info
    try:
        console.print("\n[bold cyan]Testing Account Info:[/bold cyan]")
        account_info = fofa_tool.account_info()
        
        if "error" in account_info:
            console.print(f"[bold red]API Error: {account_info['error']}[/bold red]")
        else:
            user_info = account_info.get("data", {}).get("user", {})
            console.print(f"Username: {user_info.get('username', 'Unknown')}")
            console.print(f"Email: {user_info.get('email', 'Unknown')}")
            console.print(f"VIP Level: {user_info.get('vip_level', 0)}")
            console.print("[bold green]Account Info test successful![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error testing account info: {str(e)}[/bold red]")
    
    # Test search functionality with a simple query
    try:
        console.print("\n[bold cyan]Testing Search Functionality:[/bold cyan]")
        search_results = fofa_tool.search("domain=example.com", limit=1)
        
        if "error" in search_results:
            console.print(f"[bold red]Search Error: {search_results['error']}[/bold red]")
        else:
            total = search_results.get("total", 0)
            results = len(search_results.get("results", []))
            console.print(f"Total results: {total}")
            console.print(f"Returned results: {results}")
            
            if results > 0:
                item = search_results["results"][0]
                ip_port = f"{item.get('ip')}:{item.get('port')}" if "port" in item else item.get('ip', 'Unknown')
                console.print(f"Sample result: {ip_port} - {item.get('server', 'Unknown')}")
                console.print("[bold green]Search test successful![/bold green]")
            else:
                console.print("[yellow]Search returned no results. This might be normal.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error testing search: {str(e)}[/bold red]")
    
    # Test host info functionality with a well-known IP
    try:
        console.print("\n[bold cyan]Testing Host Info Functionality:[/bold cyan]")
        host_info = fofa_tool.host_info("8.8.8.8")  # Google's DNS
        
        if "error" in host_info:
            console.print(f"[bold red]Host Info Error: {host_info['error']}[/bold red]")
        else:
            services = len(host_info.get("services", []))
            console.print(f"IP: {host_info.get('ip', 'Unknown')}")
            console.print(f"Services found: {services}")
            console.print("[bold green]Host Info test successful![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error testing host info: {str(e)}[/bold red]")
    
    # Output success message if we got here
    console.print("\n[bold green]FOFA integration tests complete![/bold green]")
    console.print("If any errors occurred, please check your API credentials and network connection.")
    console.print("For more detailed testing, try the fofa_cli.py script.")

if __name__ == "__main__":
    test_fofa_integration()
