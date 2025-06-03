#!/usr/bin/env python3
"""
FOFA Search CLI Tool
Perform FOFA searches from the command line.
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
from src.tools.fofa_tool import FofaSearchTool

# Initialize console for rich output
console = Console()

def setup_argparse():
    """Configure command-line arguments"""
    parser = argparse.ArgumentParser(description="FOFA Search Tool")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for devices on FOFA")
    search_parser.add_argument("query", help="FOFA search query")
    search_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results")
    search_parser.add_argument("--agent", action="store_true", help="Use the LLM agent to analyze results")
    search_parser.add_argument("--raw", action="store_true", help="Display raw results as JSON")
    search_parser.add_argument("--fields", help="Comma-separated list of fields to include in results")
    
    # Host lookup command
    host_parser = subparsers.add_parser("host", help="Get information about a specific IP address")
    host_parser.add_argument("ip", help="IP address to look up")
    host_parser.add_argument("--agent", action="store_true", help="Use the LLM agent to analyze results")
    host_parser.add_argument("--raw", action="store_true", help="Display raw results as JSON")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get statistics for a query")
    stats_parser.add_argument("query", help="FOFA search query")
    
    # Account info command
    subparsers.add_parser("info", help="Get information about the FOFA account and API usage")
    
    # Points command (shorthand for checking F-points)
    subparsers.add_parser("points", help="Check your FOFA F-points balance")
    
    return parser.parse_args()

async def run_search(args):
    """Run a FOFA search"""
    console.print(f"[bold cyan]Searching FOFA for: {args.query}[/bold cyan]")
    
    try:
        # Load environment variables
        load_dotenv()
        email = os.getenv("FOFA_EMAIL")
        api_key = os.getenv("FOFA_API_KEY")
        
        if not email or not api_key:
            console.print("[bold red]Error: FOFA credentials not found in environment variables[/bold red]")
            console.print("Please add your FOFA email and API key to the .env file:")
            console.print("FOFA_EMAIL=your-email@example.com")
            console.print("FOFA_API_KEY=your-fofa-api-key")
            return
        
        # If using the agent for analysis
        if args.agent:
            # Initialize the agent
            agent = AgenticLLMAgent()
            
            # Process the search query
            response = await agent.process_fofa_query(args.query, args.limit)
            
            # Display the response
            console.print("\n[bold green]Agent Analysis:[/bold green]")
            console.print(response.answer)
            console.print("\n[bold cyan]Sources:[/bold cyan]")
            for source in response.sources:
                console.print(f"- {source.url}")
            
            return
        
        # Direct search with the FOFA tool
        fields = args.fields if args.fields else "ip,port,protocol,domain,host,os,server,title,country,city"
        fofa_tool = FofaSearchTool(email, api_key)
        results = fofa_tool.search(args.query, limit=args.limit, fields=fields)
        
        # Handle potential error
        if "error" in results:
            console.print(f"[bold red]Error: {results['error']}[/bold red]")
            
            # Provide additional guidance for F-points errors
            if "F点余额不足" in str(results.get("error")) or "Insufficient F-points" in str(results.get("error")):
                console.print("\n[bold yellow]F-Points Error Guidance:[/bold yellow]")
                console.print("Your search couldn't be completed due to insufficient F-points in your FOFA account.")
                console.print("Try these solutions:")
                console.print("1. Reduce the number of results (use --limit with a smaller value)")
                console.print("2. Use a simpler search query")
                console.print("3. Wait for your F-points to replenish (if you have a free account)")
                console.print("4. Check your current F-points balance with: python fofa_cli.py points")
                console.print("5. Upgrade your FOFA account at: https://fofa.info/static_pages/vip")
            return
        
        # Display results based on format preference
        if args.raw:
            # Display raw JSON results
            console.print(json.dumps(results, indent=2))
        else:
            # Display formatted results in a table
            console.print(f"\n[bold green]Found {results['total']} matching devices[/bold green]")
            
            table = Table(title=f"FOFA Search Results for: {args.query}")
            
            # Determine which fields we have in the results
            field_names = fields.split(",")
            
            # Add columns based on fields
            if "ip" in field_names:
                table.add_column("IP", style="cyan")
            if "port" in field_names:
                table.add_column("Port", style="green")
            if "protocol" in field_names:
                table.add_column("Protocol", style="yellow")
            if "server" in field_names:
                table.add_column("Server", style="magenta")
            if "country" in field_names or "city" in field_names:
                table.add_column("Location", style="blue")
            if "title" in field_names:
                table.add_column("Title", style="white", max_width=40)
            
            for item in results.get("results", []):
                row = []
                
                if "ip" in field_names:
                    row.append(item.get('ip', ''))
                if "port" in field_names:
                    row.append(item.get('port', ''))
                if "protocol" in field_names:
                    row.append(item.get('protocol', ''))
                if "server" in field_names:
                    row.append(item.get('server', ''))
                if "country" in field_names or "city" in field_names:
                    location = f"{item.get('city', '')} {item.get('country', '')}".strip()
                    row.append(location)
                if "title" in field_names:
                    title = item.get('title', '')
                    if len(title) > 40:
                        title = title[:37] + "..."
                    row.append(title)
                
                table.add_row(*row)
            
            console.print(table)
            
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")

async def run_host_lookup(args):
    """Look up information about a specific host by IP address"""
    console.print(f"[bold cyan]Looking up host information for IP: {args.ip}[/bold cyan]")
    
    try:
        # Load environment variables
        load_dotenv()
        email = os.getenv("FOFA_EMAIL")
        api_key = os.getenv("FOFA_API_KEY")
        
        if not email or not api_key:
            console.print("[bold red]Error: FOFA credentials not found in environment variables[/bold red]")
            console.print("Please add your FOFA email and API key to the .env file:")
            console.print("FOFA_EMAIL=your-email@example.com")
            console.print("FOFA_API_KEY=your-fofa-api-key")
            return
        
        # If using the agent for analysis
        if args.agent:
            # Initialize the agent
            agent = AgenticLLMAgent()
            
            # Process the host lookup
            response = await agent.process_fofa_host(args.ip)
            
            # Display the response
            console.print("\n[bold green]Agent Analysis:[/bold green]")
            console.print(response.answer)
            console.print("\n[bold cyan]Sources:[/bold cyan]")
            for source in response.sources:
                console.print(f"- {source.url}")
            
            return
        
        # Direct lookup with the FOFA tool
        fofa_tool = FofaSearchTool(email, api_key)
        results = fofa_tool.host_info(args.ip)
        
        # Handle potential error
        if "error" in results:
            console.print(f"[bold red]Error: {results['error']}[/bold red]")
            
            # Provide additional guidance for F-points errors
            if "F点余额不足" in str(results.get("error")) or "Insufficient F-points" in str(results.get("error")):
                console.print("\n[bold yellow]F-Points Error Guidance:[/bold yellow]")
                console.print("Your host lookup couldn't be completed due to insufficient F-points in your FOFA account.")
                console.print("Try these solutions:")
                console.print("1. Wait for your F-points to replenish (if you have a free account)")
                console.print("2. Check your current F-points balance with: python fofa_cli.py points")
                console.print("3. Upgrade your FOFA account at: https://fofa.info/static_pages/vip")
            return
        
        # Display results based on format preference
        if args.raw:
            # Display raw JSON results
            console.print(json.dumps(results, indent=2))
        else:
            # Display formatted host information
            console.print(f"\n[bold green]Host Information: {args.ip}[/bold green]\n")
            
            console.print(f"[bold]IP:[/bold] {results.get('ip')}")
            
            if results.get('os'):
                console.print(f"[bold]Operating System:[/bold] {results.get('os')}")
                
            if results.get('country') or results.get('city'):
                location = f"{results.get('city', '')}, {results.get('country', '')}".strip(', ')
                console.print(f"[bold]Location:[/bold] {location}")
                
            if results.get('domains'):
                console.print(f"[bold]Domains:[/bold] {', '.join(results.get('domains', []))}")
                
            if results.get('protocols'):
                console.print(f"[bold]Protocols:[/bold] {', '.join(results.get('protocols', []))}")
            
            # Display services table
            console.print(f"\n[bold]Services ({results.get('services_count', 0)}):[/bold]")
            services_table = Table()
            services_table.add_column("Port", style="cyan")
            services_table.add_column("Protocol", style="green")
            services_table.add_column("Server", style="magenta")
            services_table.add_column("Title", style="yellow")
            
            for service in results.get("services", []):
                services_table.add_row(
                    str(service.get('port', '')),
                    service.get('protocol', ''),
                    service.get('server', ''),
                    service.get('title', '')
                )
                
            console.print(services_table)
            
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")

async def run_stats(args):
    """Get statistics for a query"""
    console.print(f"[bold cyan]Getting statistics for query: {args.query}[/bold cyan]")
    
    try:
        # Load environment variables
        load_dotenv()
        email = os.getenv("FOFA_EMAIL")
        api_key = os.getenv("FOFA_API_KEY")
        
        if not email or not api_key:
            console.print("[bold red]Error: FOFA credentials not found in environment variables[/bold red]")
            console.print("Please add your FOFA email and API key to the .env file:")
            console.print("FOFA_EMAIL=your-email@example.com")
            console.print("FOFA_API_KEY=your-fofa-api-key")
            return
        
        # Get statistics with the FOFA tool
        fofa_tool = FofaSearchTool(email, api_key)
        results = fofa_tool.stats(args.query)
        
        # Handle potential error
        if "error" in results:
            console.print(f"[bold red]Error: {results['error']}[/bold red]")
            
            # Provide additional guidance for F-points errors
            if "F点余额不足" in str(results.get("error")) or "Insufficient F-points" in str(results.get("error")):
                console.print("\n[bold yellow]F-Points Error Guidance:[/bold yellow]")
                console.print("Your stats request couldn't be completed due to insufficient F-points in your FOFA account.")
                console.print("Try these solutions:")
                console.print("1. Use a simpler query")
                console.print("2. Wait for your F-points to replenish (if you have a free account)")
                console.print("3. Check your current F-points balance with: python fofa_cli.py points")
                console.print("4. Upgrade your FOFA account at: https://fofa.info/static_pages/vip")
            return
        
        # Display results
        console.print(json.dumps(results, indent=2))
            
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")

async def run_api_info():
    """Get information about the FOFA account and API usage"""
    console.print("[bold cyan]Getting FOFA Account Information[/bold cyan]")
    
    try:
        # Load environment variables
        load_dotenv()
        email = os.getenv("FOFA_EMAIL")
        api_key = os.getenv("FOFA_API_KEY")
        
        if not email or not api_key:
            console.print("[bold red]Error: FOFA credentials not found in environment variables[/bold red]")
            console.print("Please add your FOFA email and API key to the .env file:")
            console.print("FOFA_EMAIL=your-email@example.com")
            console.print("FOFA_API_KEY=your-fofa-api-key")
            return
        
        # Get info with the FOFA tool
        fofa_tool = FofaSearchTool(email, api_key)
        results = fofa_tool.account_info()
        
        # Handle potential error
        if "error" in results:
            console.print(f"[bold red]Error: {results['error']}[/bold red]")
            if "F点余额不足" in str(results.get("error")) or "Insufficient F-points" in str(results.get("error")):
                console.print("\n[bold yellow]FOFA F-Points System:[/bold yellow]")
                console.print("FOFA uses 'F-points' for API usage. Each API request consumes points.")
                console.print("Free accounts have limited points that replenish over time.")
                console.print("To increase your limit, consider upgrading your FOFA account.")
                console.print("Visit: https://fofa.info/static_pages/vip")
            return
        
        # Display results
        console.print("\n[bold green]FOFA Account Information[/bold green]\n")
        
        if "error" in results:
            console.print(f"[bold red]API Error: {results['error']}[/bold red]")
            return
            
        user_info = results.get("data", {}).get("user", {})
        
        console.print(f"[bold]Username:[/bold] {user_info.get('username', 'Unknown')}")
        console.print(f"[bold]Email:[/bold] {user_info.get('email', 'Unknown')}")
        console.print(f"[bold]VIP Level:[/bold] {user_info.get('vip_level', 0)}")
        console.print(f"[bold]Is VIP:[/bold] {user_info.get('is_vip', False)}")
        console.print(f"[bold]VIP Time:[/bold] {user_info.get('vip_expired_time', 'Not a VIP')}")
        console.print(f"[bold]F-Points:[/bold] {user_info.get('fpoints', 0)}")
        console.print(f"[bold]Avatar:[/bold] {user_info.get('avatar', 'None')}")
            
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")

async def check_fofa_points():
    """Check F-points balance in FOFA account"""
    console.print("[bold cyan]Checking FOFA F-Points Balance[/bold cyan]")
    
    try:
        # Load environment variables
        load_dotenv()
        email = os.getenv("FOFA_EMAIL")
        api_key = os.getenv("FOFA_API_KEY")
        
        if not email or not api_key:
            console.print("[bold red]Error: FOFA credentials not found in environment variables[/bold red]")
            console.print("Please add your FOFA email and API key to the .env file:")
            console.print("FOFA_EMAIL=your-email@example.com")
            console.print("FOFA_API_KEY=your-fofa-api-key")
            return
        
        # Get info with the FOFA tool
        fofa_tool = FofaSearchTool(email, api_key)
        results = fofa_tool.account_info()
        
        # Handle potential error
        if "error" in results:
            console.print(f"[bold red]Error: {results['error']}[/bold red]")
            return
        
        # Display F-points specifically
        user_info = results.get("data", {}).get("user", {})
        f_points = user_info.get("fpoints", 0)
        is_vip = user_info.get("is_vip", False)
        vip_level = user_info.get("vip_level", 0)
        
        console.print(f"\n[bold green]FOFA F-Points Balance: {f_points}[/bold green]")
        console.print(f"VIP Status: {'Active (Level ' + str(vip_level) + ')' if is_vip else 'Not a VIP'}")
        
        # Provide information about F-points
        console.print("\n[bold yellow]FOFA F-Points System:[/bold yellow]")
        console.print("• Each API request consumes F-points based on the operation")
        console.print("• Free accounts have limited points that replenish over time")
        console.print("• More complex queries and higher result limits use more points")
        console.print("• To increase your limit, consider upgrading your FOFA account")
        console.print("• Visit: https://fofa.info/static_pages/vip for more information")
        
    except Exception as e:
        console.print(f"[bold red]Error checking F-points: {str(e)}[/bold red]")

async def main():
    """Main function"""
    args = setup_argparse()
    
    if args.command == "search":
        await run_search(args)
    elif args.command == "host":
        await run_host_lookup(args)
    elif args.command == "stats":
        await run_stats(args)
    elif args.command == "info":
        await run_api_info()
    elif args.command == "points":
        await check_fofa_points()
    else:
        console.print("[bold yellow]Please specify a command: search, host, stats, info, or points[/bold yellow]")
        console.print("Use -h or --help for more information")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
