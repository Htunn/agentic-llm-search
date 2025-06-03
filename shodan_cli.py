#!/usr/bin/env python3
"""
Shodan Search CLI Tool
Perform Shodan searches from the command line.
"""

import os
import sys
import argparse
import logging
import asyncio
from dotenv import load_dotenv
import json
from rich.console import Console
from rich.table import Table

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from the project
from src.agents.agentic_llm import AgenticLLMAgent
from src.tools.shodan_tool import ShodanSearchTool

# Initialize console for rich output
console = Console()

def setup_argparse():
    """Configure command-line arguments"""
    parser = argparse.ArgumentParser(description="Shodan Search Tool")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for devices on Shodan")
    search_parser.add_argument("query", help="Shodan search query")
    search_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results")
    search_parser.add_argument("--agent", action="store_true", help="Use the LLM agent to analyze results")
    search_parser.add_argument("--raw", action="store_true", help="Display raw results as JSON")
    
    # Host lookup command
    host_parser = subparsers.add_parser("host", help="Get information about a specific IP address")
    host_parser.add_argument("ip", help="IP address to look up")
    host_parser.add_argument("--agent", action="store_true", help="Use the LLM agent to analyze results")
    host_parser.add_argument("--raw", action="store_true", help="Display raw results as JSON")
    
    # Count command
    count_parser = subparsers.add_parser("count", help="Count the number of devices matching a query")
    count_parser.add_argument("query", help="Shodan search query")
    count_parser.add_argument("--facets", nargs="*", help="Optional facets to count by (format: name:count)")
    
    # API info command
    subparsers.add_parser("info", help="Get information about the Shodan API plan and usage")
    
    return parser.parse_args()

async def run_search(args):
    """Run a Shodan search"""
    console.print(f"[bold cyan]Searching Shodan for: {args.query}[/bold cyan]")
    
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("SHODAN_API_KEY")
        
        if not api_key:
            console.print("[bold red]Error: SHODAN_API_KEY not found in environment variables[/bold red]")
            console.print("Please add your Shodan API key to the .env file with the key SHODAN_API_KEY")
            return
        
        # If using the agent for analysis
        if args.agent:
            # Initialize the agent
            agent = AgenticLLMAgent()
            
            # Process the search query
            response = await agent.process_shodan_query(args.query, args.limit)
            
            # Display the response
            console.print("\n[bold green]Agent Analysis:[/bold green]")
            console.print(response.answer)
            console.print("\n[bold cyan]Sources:[/bold cyan]")
            for source in response.sources:
                console.print(f"- {source.url}")
            
            return
        
        # Direct search with the Shodan tool
        shodan_tool = ShodanSearchTool(api_key)
        results = shodan_tool.search(args.query, limit=args.limit)
        
        # Handle potential error
        if "error" in results:
            error = results['error']
            console.print(f"[bold red]Error: {error}[/bold red]")
            
            # Provide helpful guidance for access denied errors
            if "Access denied" in error or "403 Forbidden" in error:
                console.print("\n[bold yellow]Troubleshooting Steps:[/bold yellow]")
                console.print("1. Verify your API key at https://account.shodan.io/")
                console.print("2. Check your subscription plan and query credits")
                console.print("3. The free 'oss' plan has very limited search capabilities")
                console.print("4. Consider upgrading your Shodan plan for more search features")
                console.print("5. Try simpler queries that use fewer query credits")
            
            return
        
        # Display results based on format preference
        if args.raw:
            # Display raw JSON results
            console.print(json.dumps(results, indent=2))
        else:
            # Display formatted results in a table
            console.print(f"\n[bold green]Found {results['total']} matching devices[/bold green]")
            
            table = Table(title=f"Shodan Search Results for: {args.query}")
            table.add_column("IP:Port", style="cyan")
            table.add_column("Organization", style="green")
            table.add_column("Location", style="yellow")
            table.add_column("Product", style="magenta")
            table.add_column("Hostname", style="blue")
            
            for match in results.get("matches", []):
                # Format the location
                location = f"{match.get('city', '')} {match.get('country', '')}"
                
                # Format the product
                product = f"{match.get('product', '')} {match.get('version', '')}"
                
                # Format hostnames
                hostnames = ", ".join(match.get('hostname', []))
                
                table.add_row(
                    f"{match.get('ip_str')}:{match.get('port')}",
                    match.get('org', 'Unknown'),
                    location.strip(),
                    product.strip(),
                    hostnames
                )
            
            console.print(table)
            
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")

async def run_host_lookup(args):
    """Look up information about a specific host by IP address"""
    console.print(f"[bold cyan]Looking up host information for IP: {args.ip}[/bold cyan]")
    
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("SHODAN_API_KEY")
        
        if not api_key:
            console.print("[bold red]Error: SHODAN_API_KEY not found in environment variables[/bold red]")
            console.print("Please add your Shodan API key to the .env file with the key SHODAN_API_KEY")
            return
        
        # If using the agent for analysis
        if args.agent:
            # Initialize the agent
            agent = AgenticLLMAgent()
            
            # Process the host lookup
            response = await agent.process_shodan_host(args.ip)
            
            # Display the response
            console.print("\n[bold green]Agent Analysis:[/bold green]")
            console.print(response.answer)
            console.print("\n[bold cyan]Sources:[/bold cyan]")
            for source in response.sources:
                console.print(f"- {source.url}")
            
            return
        
        # Direct lookup with the Shodan tool
        shodan_tool = ShodanSearchTool(api_key)
        results = shodan_tool.host_info(args.ip)
        
        # Handle potential error
        if "error" in results:
            console.print(f"[bold red]Error: {results['error']}[/bold red]")
            return
        
        # Display results based on format preference
        if args.raw:
            # Display raw JSON results
            console.print(json.dumps(results, indent=2))
        else:
            # Display formatted host information
            console.print(f"\n[bold green]Host Information: {args.ip}[/bold green]\n")
            
            console.print(f"[bold]IP:[/bold] {results.get('ip_str')}")
            console.print(f"[bold]Organization:[/bold] {results.get('organization', 'Unknown')}")
            console.print(f"[bold]Location:[/bold] {results.get('city', '')}, {results.get('country', '')}")
            console.print(f"[bold]Hostnames:[/bold] {', '.join(results.get('hostnames', []))}")
            console.print(f"[bold]Domains:[/bold] {', '.join(results.get('domains', []))}")
            console.print(f"[bold]Open Ports:[/bold] {', '.join(str(p) for p in results.get('ports', []))}")
            
            if results.get("vulns"):
                console.print(f"[bold red]Vulnerabilities:[/bold red] {', '.join(results.get('vulns', []))}")
            
            console.print(f"[bold]Last Update:[/bold] {results.get('last_update', 'Unknown')}")
            
            # Display services table
            console.print(f"\n[bold]Services ({results.get('services_count', 0)}):[/bold]")
            services_table = Table()
            services_table.add_column("Port", style="cyan")
            services_table.add_column("Protocol", style="green")
            services_table.add_column("Product", style="magenta")
            services_table.add_column("Version", style="yellow")
            
            for service in results.get("services", []):
                services_table.add_row(
                    str(service.get('port', '')),
                    service.get('protocol', ''),
                    service.get('product', ''),
                    service.get('version', '')
                )
                
            console.print(services_table)
            
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")

async def run_count(args):
    """Count devices matching a query"""
    console.print(f"[bold cyan]Counting devices matching: {args.query}[/bold cyan]")
    
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("SHODAN_API_KEY")
        
        if not api_key:
            console.print("[bold red]Error: SHODAN_API_KEY not found in environment variables[/bold red]")
            console.print("Please add your Shodan API key to the .env file with the key SHODAN_API_KEY")
            return
        
        # Parse facets if provided
        facets = []
        if args.facets:
            for facet in args.facets:
                facets.append(facet)
        
        # Count with the Shodan tool
        shodan_tool = ShodanSearchTool(api_key)
        results = shodan_tool.count(args.query, facets=facets)
        
        # Handle potential error
        if "error" in results:
            console.print(f"[bold red]Error: {results['error']}[/bold red]")
            return
        
        # Display results
        console.print(f"\n[bold green]Total Results: {results['total']:,}[/bold green]\n")
        
        # Display facets if available
        if "facets" in results:
            console.print("[bold]Facet Counts:[/bold]")
            for facet_name, facet_items in results["facets"].items():
                console.print(f"\n[bold cyan]{facet_name}:[/bold cyan]")
                
                facet_table = Table()
                facet_table.add_column("Value", style="green")
                facet_table.add_column("Count", style="yellow", justify="right")
                
                for item in facet_items:
                    facet_table.add_row(
                        str(item["value"]),
                        f"{item['count']:,}"
                    )
                
                console.print(facet_table)
            
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")

async def run_api_info():
    """Get information about the Shodan API plan"""
    console.print("[bold cyan]Getting Shodan API Information[/bold cyan]")
    
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("SHODAN_API_KEY")
        
        if not api_key:
            console.print("[bold red]Error: SHODAN_API_KEY not found in environment variables[/bold red]")
            console.print("Please add your Shodan API key to the .env file with the key SHODAN_API_KEY")
            return
        
        # Get info with the Shodan tool
        shodan_tool = ShodanSearchTool(api_key)
        results = shodan_tool.api_info()
        
        # Handle potential error
        if "error" in results:
            console.print(f"[bold red]Error: {results['error']}[/bold red]")
            return
        
        # Display results
        console.print("\n[bold green]Shodan API Information[/bold green]\n")
        console.print(f"[bold]Plan:[/bold] {results.get('plan', 'Unknown')}")
        console.print(f"[bold]Credits:[/bold] {results.get('unlocked_left', 0):,} credits remaining")
        console.print(f"[bold]HTTPS:[/bold] {results.get('https', False)}")
        console.print(f"[bold]Telnet:[/bold] {results.get('telnet', False)}")
        console.print(f"[bold]Scan Credits:[/bold] {results.get('scan_credits', 0):,}")
        console.print(f"[bold]Query Credits:[/bold] {results.get('query_credits', 0):,}")
        console.print(f"[bold]Monitor Credits:[/bold] {results.get('monitored_ips', 0):,}")
            
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")

async def main():
    """Main function"""
    args = setup_argparse()
    
    if args.command == "search":
        await run_search(args)
    elif args.command == "host":
        await run_host_lookup(args)
    elif args.command == "count":
        await run_count(args)
    elif args.command == "info":
        await run_api_info()
    else:
        console.print("[bold yellow]Please specify a command: search, host, count, or info[/bold yellow]")
        console.print("Use -h or --help for more information")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
