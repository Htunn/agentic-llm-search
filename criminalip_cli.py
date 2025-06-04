#!/usr/bin/env python3
"""
Criminal IP CLI - Command line interface for Criminal IP API

This tool provides command-line access to the Criminal IP API
for cybersecurity research and threat intell        if "error" in results:
            error_msg = results.get("error", "Unknown error")
            solution = results.get("solution", "")
            
            console.print(f"[bold red]IP lookup error: {error_msg}[/bold red]")
            if solution:
                console.print(f"[yellow]Suggestion: {solution}[/yellow]")
            returnnce.

Usage:
  ./criminalip_cli.py setup [--api-key=<api_key>]
  ./criminalip_cli.py search <query> [--limit=<number>]
  ./criminalip_cli.py ip <ip_address>
  ./criminalip_cli.py domain <domain> [--limit=<number>]
  ./criminalip_cli.py banner <query> [--limit=<number>]
  ./criminalip_cli.py info
  ./criminalip_cli.py help
"""

import os
import sys
import argparse
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

# Import our tooling
try:
    from src.tools.criminalip_tool import CriminalIPTool
except ImportError:
    print("Error: Could not import CriminalIPTool. Make sure you're running from the project root.")
    sys.exit(1)

# Load environment variables
load_dotenv()

console = Console()

def setup_args():
    """Set up command-line arguments"""
    parser = argparse.ArgumentParser(description="Criminal IP CLI - Command line interface for Criminal IP API")
    subparsers = parser.add_subparsers(dest="command", help="command")
    
    # Help command
    help_parser = subparsers.add_parser("help", help="Show help information for using Criminal IP")
    
    # Setup command for API key
    setup_parser = subparsers.add_parser("setup", help="Configure Criminal IP API key")
    setup_parser.add_argument("--api-key", help="Criminal IP API key to save in .env file")
    setup_parser.add_argument("--validate", action="store_true", help="Validate the current API key")
    
    # IP report command
    ip_parser = subparsers.add_parser("ip", help="Get detailed information about an IP address")
    ip_parser.add_argument("ip_address", help="IP address to look up")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search Criminal IP for assets matching a query")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results (default: 10)")
    
    # Domain command
    domain_parser = subparsers.add_parser("domain", help="Get information about a domain")
    domain_parser.add_argument("domain", help="Domain name to search for")
    domain_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results (default: 10)")
    
    # Banner search command
    banner_parser = subparsers.add_parser("banner", help="Search for banner information matching a query")
    banner_parser.add_argument("query", help="Search query for banner information")
    banner_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results (default: 10)")
    
    # API info command
    subparsers.add_parser("info", help="Show information about your Criminal IP API account")
    
    return parser.parse_args()

async def run_search(args):
    """Run a search query against Criminal IP"""
    console.print(f"[bold cyan]Criminal IP Asset Search:[/bold cyan] {args.query}")
    
    try:
        # Initialize the Criminal IP tool
        api_key = os.getenv("CRIMINAL_IP_API_KEY")
        if not api_key:
            console.print("[bold red]Error: CRIMINAL_IP_API_KEY environment variable not set[/bold red]")
            console.print("Please set this in your .env file")
            return
            
        criminalip = CriminalIPTool(api_key)
        
        # Perform the search
        results = criminalip.asset_search(args.query, args.limit)
        
        if "error" in results:
            error_msg = results.get("error", "Unknown error")
            solution = results.get("solution", "")
            
            console.print(f"[bold red]Search error: {error_msg}[/bold red]")
            if solution:
                console.print(f"[yellow]Suggestion: {solution}[/yellow]")
            return
            
        # Display the results
        if "data" in results and "result" in results["data"]:
            items = results["data"]["result"]
            count = results["data"].get("count", 0)
            
            if not items:
                console.print("[yellow]No results found[/yellow]")
                return
                
            console.print(f"\n[bold green]Found {count} results (showing {len(items)})[/bold green]\n")
            
            # Create a table for the results
            table = Table(show_header=True)
            table.add_column("Type", style="cyan")
            table.add_column("Target", style="green")
            table.add_column("Score", style="red")
            table.add_column("Country")
            table.add_column("Details")
            
            for item in items:
                # In the new API format, we need to determine the type from available fields
                if "ip_address" in item:
                    item_type = "ip"
                    target = item.get("ip_address", "Unknown")
                elif "domain" in item:
                    item_type = "domain"
                    target = item.get("domain", "Unknown")
                else:
                    item_type = "service"
                    target = item.get("title", "Unknown") or item.get("hostname", "Unknown")
                
                # Score is not directly provided in this response format
                score = "N/A"
                country = item.get("country", "Unknown")
                
                # Collect additional details
                details = []
                if "port" in item:
                    details.append(f"Port: {item['port']}")
                if "hostname" in item and item["hostname"]:
                    details.append(f"Host: {item['hostname']}")
                if "banner" in item and item["banner"]:
                    banner_preview = item["banner"][:30].replace('\n', ' ') + "..."
                    details.append(f"Banner: {banner_preview}")
                if "server" in item and item["server"]:
                    details.append(f"Server: {item['server']}")
                if "as_name" in item and item["as_name"]:
                    details.append(f"AS: {item['as_name']}")
                if "has_cve" in item:
                    has_cve = "Yes" if item["has_cve"] else "No"
                    details.append(f"CVEs: {has_cve}")
                    
                details_str = ", ".join(details)
                
                table.add_row(item_type, target, score, country, details_str)
                
            console.print(table)
            
            # Display URL for web view
            web_url = f"https://www.criminalip.io/search?query={args.query}"
            console.print(f"\n[bold]View in Criminal IP: [link={web_url}]{web_url}[/link][/bold]")
        else:
            console.print("[yellow]No results or unexpected response format[/yellow]")
            
    except Exception as e:
        console.print(f"[bold red]Error during search: {str(e)}[/bold red]")
        
async def run_ip_lookup(args):
    """Look up detailed information about an IP address"""
    console.print(f"[bold cyan]Criminal IP Lookup:[/bold cyan] {args.ip_address}")
    
    try:
        # Initialize the Criminal IP tool
        api_key = os.getenv("CRIMINAL_IP_API_KEY")
        if not api_key:
            console.print("[bold red]Error: CRIMINAL_IP_API_KEY environment variable not set[/bold red]")
            console.print("Please set this in your .env file")
            return
            
        criminalip = CriminalIPTool(api_key)
        
        # Perform the IP lookup
        results = criminalip.search_ip(args.ip_address)
        
        # Skip debug output for production use
        
        if "error" in results:
            console.print(f"[bold red]Lookup error: {results['error']}[/bold red]")
            return
            
        # Display the results
        # In the new API format, the data is directly in the results
        ip_data = results
        
        if ip_data:
            # Create a panel with the IP information
            ip_info = []
            ip_info.append(f"## IP: {args.ip_address}")
            
            # Security score
            if "score" in ip_data:
                inbound = ip_data["score"].get("inbound", "Unknown")
                outbound = ip_data["score"].get("outbound", "Unknown")
                
                score_text = f"Inbound: {inbound}, Outbound: {outbound}"
                score_color = "green" if "Safe" in [inbound, outbound] else "yellow"
                ip_info.append(f"**Security Score:** [bold {score_color}]{score_text}[/bold {score_color}]")
            
            # Security issues
            if "issues" in ip_data:
                issues = []
                for issue, value in ip_data["issues"].items():
                    if value:  # Only show true issues
                        issues.append(issue.replace("is_", "").title())
                        
                if issues:
                    ip_info.append(f"**Security Issues:** {', '.join(issues)}")
                
            # User search count
            if "user_search_count" in ip_data:
                ip_info.append(f"**Times Searched:** {ip_data['user_search_count']}")
            
            # Open ports
            if "open_ports" in ip_data and ip_data["open_ports"]:
                ip_info.append("\n### Open Ports")
                for port in ip_data["open_ports"][:10]:  # Limit to first 10 ports
                    port_num = port.get("port", "Unknown")
                    protocol = port.get("protocol", "").upper()
                    service = port.get("service_name", "Unknown")
                    ip_info.append(f"- **{port_num}/{protocol}:** {service}")
                    
                if len(ip_data["open_ports"]) > 10:
                    ip_info.append(f"... and {len(ip_data['open_ports']) - 10} more ports")
            
            # Malicious information
            malicious_info = None
            try:
                malicious_info = criminalip.check_ip_malicious(args.ip_address)
            except Exception:
                console.print("[yellow]Could not retrieve malicious information[/yellow]")
            
            if malicious_info and "data" in malicious_info and malicious_info["data"]:
                mal_data = malicious_info["data"]
                ip_info.append("\n### Security Information")
                
                if "is_malicious" in mal_data:
                    is_malicious = mal_data["is_malicious"]
                    status_color = "red" if is_malicious else "green"
                    status_text = "Yes" if is_malicious else "No"
                    ip_info.append(f"**Malicious:** [bold {status_color}]{status_text}[/bold {status_color}]")
                    
                if "is_proxy" in mal_data:
                    is_proxy = mal_data["is_proxy"]
                    proxy_text = "Yes" if is_proxy else "No"
                    ip_info.append(f"**Proxy/VPN:** {proxy_text}")
                    
                if "tags" in mal_data and mal_data["tags"]:
                    ip_info.append(f"**Tags:** {', '.join(mal_data['tags'])}")
            
            # Display information
            markdown_text = "\n".join(ip_info)
            console.print(Panel(Markdown(markdown_text), title=f"Criminal IP Analysis", border_style="cyan"))
            
            # Display URL for web view
            web_url = f"https://www.criminalip.io/asset/report?ip={args.ip_address}"
            console.print(f"\n[bold]View in Criminal IP: [link={web_url}]{web_url}[/link][/bold]")
        else:
            console.print("[yellow]No data found for this IP address[/yellow]")
            
    except Exception as e:
        console.print(f"[bold red]Error during IP lookup: {str(e)}[/bold red]")
        
async def run_domain_lookup(args):
    """Look up information about a domain"""
    console.print(f"[bold cyan]Criminal IP Domain Lookup:[/bold cyan] {args.domain}")
    
    try:
        # Initialize the Criminal IP tool
        api_key = os.getenv("CRIMINAL_IP_API_KEY")
        if not api_key:
            console.print("[bold red]Error: CRIMINAL_IP_API_KEY environment variable not set[/bold red]")
            console.print("Please set this in your .env file")
            return
            
        criminalip = CriminalIPTool(api_key)
        
        # Perform the domain lookup
        results = criminalip.search_domain(args.domain, args.limit)
        
        if "error" in results:
            error_msg = results.get("error", "Unknown error")
            solution = results.get("solution", "")
            
            console.print(f"[bold red]Domain lookup error: {error_msg}[/bold red]")
            if solution:
                console.print(f"[yellow]Suggestion: {solution}[/yellow]")
            return
            
        # Display the results
        if "data" in results and "reports" in results["data"]:  # Fixed: use "reports" instead of "items"
            items = results["data"]["reports"]
            count = results["data"].get("count", 0)
            
            if not items:
                console.print("[yellow]No domain information found[/yellow]")
                return
                
            console.print(f"\n[bold green]Found {count} results (showing {len(items)})[/bold green]\n")
            
            # Create a table for the results
            table = Table(show_header=True)
            table.add_column("Domain", style="cyan")
            table.add_column("Score", style="red")
            table.add_column("Scan Date")
            table.add_column("Details")
            
            for item in items:
                # In the new API format, domain name is the query itself
                domain = args.domain
                score = str(item.get("score", "N/A"))
                scan_date = item.get("reg_dtime", "Unknown")
                
                # Collect additional details
                details = []
                
                # Add connected IPs count
                if "connected_ip_cnt" in item:
                    details.append(f"IPs: {item['connected_ip_cnt']}")
                
                # Add issue flags if present
                if "issue" in item and isinstance(item["issue"], list):
                    issues = ", ".join(item["issue"])
                    details.append(f"Issues: {issues}")
                
                # Add country info if present
                if "country_code" in item and isinstance(item["country_code"], list) and item["country_code"]:
                    countries = ", ".join(code for code in item["country_code"] if code)
                    if countries:
                        details.append(f"Countries: {countries}")
                        
                details_str = ", ".join(details)
                
                table.add_row(domain, score, scan_date, details_str)
                
            console.print(table)
            
            # Display URL for web view
            web_url = f"https://www.criminalip.io/domain/report?query={args.domain}"
            console.print(f"\n[bold]View in Criminal IP: [link={web_url}]{web_url}[/link][/bold]")
        else:
            console.print("[yellow]No domain information or unexpected response format[/yellow]")
            
    except Exception as e:
        console.print(f"[bold red]Error during domain lookup: {str(e)}[/bold red]")
        
async def run_banner_search(args):
    """Search for banner information"""
    console.print(f"[bold cyan]Criminal IP Banner Search:[/bold cyan] {args.query}")
    
    try:
        # Initialize the Criminal IP tool
        api_key = os.getenv("CRIMINAL_IP_API_KEY")
        if not api_key:
            console.print("[bold red]Error: CRIMINAL_IP_API_KEY environment variable not set[/bold red]")
            console.print("Please set this in your .env file")
            return
            
        criminalip = CriminalIPTool(api_key)
        
        # Perform the banner search
        results = criminalip.banner_search(args.query, args.limit)
        
        if "error" in results:
            error_msg = results.get("error", "Unknown error")
            solution = results.get("solution", "")
            
            console.print(f"[bold red]Banner search error: {error_msg}[/bold red]")
            if solution:
                console.print(f"[yellow]Suggestion: {solution}[/yellow]")
            return
            
        # Display the results
        if "data" in results:
            # Check if "result" key exists (new API format)
            if "result" in results["data"]:
                items = results["data"]["result"]
                count = results["data"].get("count", 0)
            # Fall back to old API format if needed
            else:
                items = results["data"].get("items", [])
                count = results["data"].get("total", 0)
            
            if not items:
                console.print("[yellow]No banner information found[/yellow]")
                return
                
            console.print(f"\n[bold green]Found {count} results (showing {len(items)})[/bold green]\n")
            
            # Create a table for the results
            table = Table(show_header=True)
            table.add_column("IP", style="cyan")
            table.add_column("Port", style="green")
            table.add_column("Protocol")
            table.add_column("Banner/Service")
            
            for item in items:
                # Support both old and new API field names
                ip = item.get("ip", item.get("ip_address", "Unknown"))
                
                # Port might be in different fields or nested
                if "port" in item:
                    port = str(item.get("port", "Unknown"))
                elif "service_info" in item and "port" in item["service_info"]:
                    port = str(item["service_info"]["port"])
                else:
                    port = "Unknown"
                
                # Protocol handling with fallbacks
                if "protocol" in item:
                    protocol = item.get("protocol", "").upper()
                elif "service_info" in item and "protocol" in item["service_info"]:
                    protocol = item["service_info"]["protocol"].upper()
                elif "service" in item:
                    protocol = item.get("service", "").upper()
                else:
                    protocol = "UNKNOWN"
                
                # Banner data could be in different fields
                banner = item.get("banner", "")
                if not banner:
                    # Try other possible banner field locations
                    banner = item.get("data", item.get("snippet", ""))
                    if not banner and "http_response" in item:
                        banner = item["http_response"]
                    
                if len(banner) > 50:
                    banner = banner[:47] + "..."
                    
                table.add_row(ip, port, protocol, banner)
                
            console.print(table)
            
            # Display URL for web view
            web_url = f"https://www.criminalip.io/banner/search?query={args.query}"
            console.print(f"\n[bold]View in Criminal IP: [link={web_url}]{web_url}[/link][/bold]")
        else:
            console.print("[yellow]No banner information or unexpected response format[/yellow]")
            
    except Exception as e:
        console.print(f"[bold red]Error during banner search: {str(e)}[/bold red]")
        
async def run_api_info():
    """Show information about your Criminal IP API account"""
    console.print("[bold cyan]Criminal IP API Account Information[/bold cyan]")
    
    try:
        # Initialize the Criminal IP tool
        api_key = os.getenv("CRIMINAL_IP_API_KEY")
        if not api_key:
            console.print("[bold red]Error: CRIMINAL_IP_API_KEY environment variable not set[/bold red]")
            console.print("Please set this in your .env file")
            return
            
        criminalip = CriminalIPTool(api_key)
        
        # Get user information
        user_info = criminalip.get_user_info()
        
        if "error" in user_info:
            error_msg = user_info.get("error", "Unknown error")
            solution = user_info.get("solution", "")
            
            console.print(f"[bold red]API error: {error_msg}[/bold red]")
            if solution:
                console.print(f"[yellow]Suggestion: {solution}[/yellow]")
            return
            
        if "data" in user_info:
            user_data = user_info["data"]
            
            # Create a panel with the account information
            account_info = []
            
            if "email" in user_data:
                account_info.append(f"**Email:** {user_data['email']}")
                
            if "user_id" in user_data:
                account_info.append(f"**User ID:** {user_data['user_id']}")
                
            if "plan_name" in user_data:
                account_info.append(f"**Plan:** {user_data['plan_name']}")
                
            # API quota information
            if "api_quota" in user_data:
                quota = user_data["api_quota"]
                account_info.append("\n### API Quota")
                
                if "daily_limit" in quota and "daily_usage" in quota:
                    daily_limit = quota["daily_limit"]
                    daily_usage = quota["daily_usage"]
                    daily_percent = (daily_usage / daily_limit) * 100 if daily_limit > 0 else 0
                    
                    account_info.append(f"**Daily Usage:** {daily_usage}/{daily_limit} ({daily_percent:.1f}%)")
                    
                if "monthly_limit" in quota and "monthly_usage" in quota:
                    monthly_limit = quota["monthly_limit"]
                    monthly_usage = quota["monthly_usage"]
                    monthly_percent = (monthly_usage / monthly_limit) * 100 if monthly_limit > 0 else 0
                    
                    account_info.append(f"**Monthly Usage:** {monthly_usage}/{monthly_limit} ({monthly_percent:.1f}%)")
            
            # Display information
            markdown_text = "\n".join(account_info)
            console.print(Panel(Markdown(markdown_text), title="Criminal IP Account Information", border_style="cyan"))
        else:
            console.print("[yellow]Could not retrieve account information[/yellow]")
            
    except Exception as e:
        console.print(f"[bold red]Error retrieving account information: {str(e)}[/bold red]")

async def run_setup(args):
    """Set up or validate Criminal IP API key"""
    import getpass
    import os
    
    # Get the path to the .env file
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    
    # If API key was provided via command-line argument
    if hasattr(args, 'api_key') and args.api_key:
        new_api_key = args.api_key
    else:
        # Prompt for API key securely
        console.print("[bold]Setting up Criminal IP API Key[/bold]")
        console.print("You can get your API key from https://www.criminalip.io/developer/api")
        new_api_key = getpass.getpass("Enter your Criminal IP API Key: ")
        
    if not new_api_key:
        console.print("[bold red]Error: No API key provided[/bold red]")
        return
        
    # Validate the key before saving
    console.print("[yellow]Validating the API key...[/yellow]")
    tool = CriminalIPTool(new_api_key)
    validation = tool.verify_api_key()
    
    if not validation["success"]:
        console.print(f"[bold red]✗ API key validation failed: {validation.get('message', 'Unknown error')}[/bold red]")
        if "solution" in validation:
            console.print(f"[yellow]Suggestion: {validation['solution']}[/yellow]")
        
        save_anyway = input("Save this API key anyway? (y/n): ").lower() == 'y'
        if not save_anyway:
            console.print("[yellow]API key not saved[/yellow]")
            return
    else:
        console.print("[bold green]✓ API key is valid![/bold green]")
        if "user_info" in validation:
            user_info = validation["user_info"]
            try:
                if "data" in user_info:
                    user_data = user_info["data"]
                    console.print(Panel.fit(
                        f"[bold]Criminal IP Account Information[/bold]\n\n"
                        f"Email: {user_data.get('email', 'N/A')}\n"
                        f"API Limit: {user_data.get('api_limit', 'N/A')}\n"
                        f"API Usage: {user_data.get('api_usage', 'N/A')}\n"
                        f"Tier: {user_data.get('tier', 'N/A')}",
                        title="Account Details",
                        border_style="green"
                    ))
            except Exception as e:
                console.print(f"[yellow]Could not display account details: {str(e)}[/yellow]")
    
    # Save the API key to .env file
    try:
        # Check if .env file exists and contains API key
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                env_contents = f.read()
            
            # Update existing API key
            if "CRIMINAL_IP_API_KEY=" in env_contents:
                import re
                new_contents = re.sub(
                    r'CRIMINAL_IP_API_KEY=.*',
                    f'CRIMINAL_IP_API_KEY={new_api_key}',
                    env_contents
                )
                with open(env_path, "w") as f:
                    f.write(new_contents)
            else:
                # Append to existing file
                with open(env_path, "a") as f:
                    f.write(f"\nCRIMINAL_IP_API_KEY={new_api_key}\n")
        else:
            # Create a new .env file
            with open(env_path, "w") as f:
                f.write(f"CRIMINAL_IP_API_KEY={new_api_key}\n")
        
        console.print(f"[bold green]✓ API key saved to {env_path}[/bold green]")
        console.print("[yellow]Reload your environment or restart your application to use the new API key[/yellow]")
        
        # Update the current environment variable
        os.environ["CRIMINAL_IP_API_KEY"] = new_api_key
        
    except Exception as e:
        console.print(f"[bold red]Error saving API key to .env file: {str(e)}[/bold red]")
        console.print("[yellow]You can manually add your API key to the .env file:[/yellow]")
        console.print(f"CRIMINAL_IP_API_KEY={new_api_key}")

async def run_help():
    """Display help information for using Criminal IP"""
    console.print(Panel.fit(
        Markdown("""
        # Criminal IP CLI Tool Help

        This tool allows you to interact with the Criminal IP cybersecurity search engine from the command line.

        ## Setup

        Before using this tool, you need to set up your Criminal IP API key:

        ```
        ./criminalip_cli.py setup
        ```

        ## Available Commands

        - [bold]setup[/bold]: Configure your Criminal IP API key
        - [bold]ip[/bold]: Look up information about an IP address
        - [bold]search[/bold]: Search for assets matching a query
        - [bold]domain[/bold]: Look up information about a domain
        - [bold]banner[/bold]: Search for specific services based on banner information
        - [bold]info[/bold]: Show information about your Criminal IP account

        ## Examples

        Look up information about an IP address:
        ```
        ./criminalip_cli.py ip 8.8.8.8
        ```

        Search for assets:
        ```
        ./criminalip_cli.py search "apache 2.4"
        ```

        Look up a domain:
        ```
        ./criminalip_cli.py domain example.com
        ```

        ## More Information

        For more detailed information, see the [link=https://www.criminalip.io/developer/api]Criminal IP API Documentation[/link].
        """),
        title="Criminal IP CLI Help",
        border_style="blue"
    ))

async def main():
    """Main entry point for the CLI"""
    args = setup_args()
    
    if not args.command:
        # If no command is specified, show a brief help message
        console.print("[bold yellow]Welcome to the Criminal IP CLI tool![/bold yellow]")
        console.print("Run './criminalip_cli.py help' for detailed usage information")
        console.print("Or './criminalip_cli.py setup' to configure your API key")
        return
    
    if args.command == "search":
        await run_search(args)
    elif args.command == "ip":
        await run_ip_lookup(args)
    elif args.command == "domain":
        await run_domain_lookup(args)
    elif args.command == "banner":
        await run_banner_search(args)
    elif args.command == "info":
        await run_api_info()
    elif args.command == "setup":
        await run_setup(args)
    elif args.command == "help":
        await run_help()
    else:
        console.print(f"[bold red]Error: Unknown command: {args.command}[/bold red]")
        console.print("Run './criminalip_cli.py help' for usage information")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation cancelled by user[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")
