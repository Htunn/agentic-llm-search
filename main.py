"""
Command-line interface for the Agentic LLM Agent
"""

import asyncio
import argparse
import os
import sys
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from dotenv import load_dotenv

from src.agents.agentic_llm import AgenticLLMAgent, AgentConfig
from src import AgentResponse

# Load environment variables
load_dotenv()

console = Console()

class AgentCLI:
    """Command-line interface for the agent"""
    
    def __init__(self):
        self.config = AgentConfig()
        self.agent = None
        
    def initialize_agent(self):
        """Initialize the agent with current config"""
        try:
            # Log configuration before initializing
            console.print(f"[bold blue]Model configuration:[/bold blue]")
            console.print(f"- Model provider: {self.config.model_provider}")
            console.print(f"- Model name: {self.config.model_name}")
            
            # Special logging for Azure OpenAI
            if self.config.model_provider.lower() == 'azure-openai':
                console.print(f"- Azure OpenAI Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT', 'Not set')}")
                console.print(f"- Azure OpenAI Deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT', 'Not set')}")
            
            self.agent = AgenticLLMAgent(
                model_name=self.config.model_name,
                model_provider=self.config.model_provider,
                max_search_results=self.config.max_search_results,
                enable_memory=True,
                max_memory=int(os.getenv("MAX_MEMORY", "10"))
            )
            console.print("[green]✓ Agent initialized successfully![/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to initialize agent: {str(e)}[/red]")
            console.print(f"[yellow]Error details: {str(e)}[/yellow]")
            sys.exit(1)
    
    def display_response(self, response: AgentResponse):
        """Display the agent's response in a formatted way"""
        # Display main answer
        console.print(Panel(
            Markdown(response.answer),
            title="Answer",
            border_style="blue"
        ))
        
        # Display sources if available
        if response.sources:
            console.print("\n[bold]Sources:[/bold]")
            for i, source in enumerate(response.sources, 1):
                console.print(f"[{i}] {source.title}")
                console.print(f"    🔗 {source.url}")
                console.print(f"    📅 {source.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                console.print()
        
        # Display metadata
        console.print(f"[dim]Model: {response.model_used} | "
                     f"Query time: {response.timestamp.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
    
    def interactive_mode(self):
        """Run the agent in interactive mode"""
        console.print(Panel(
            "[bold blue]Agentic LLM Agent[/bold blue]\n"
            "Ask me anything and I'll search the internet for the most up-to-date information!\n\n"
            "Commands:\n"
            "• /help - Show help\n"
            "• /config - Show configuration\n"
            "• /search on/off - Enable/disable search\n"
            "• /memory on/off - Enable/disable conversation memory\n"
            "• /clear - Clear conversation history\n"
            "• /exit - Exit the application",
            title="Welcome",
            border_style="green"
        ))
        
        self.initialize_agent()
        
        while True:
            try:
                # Get user input
                query = Prompt.ask("\n[bold]Your question")
                
                # Handle commands
                if query.lower() in ['/exit', '/quit', 'exit', 'quit']:
                    console.print("[yellow]Goodbye! 👋[/yellow]")
                    break
                elif query.lower() == '/help':
                    self.show_help()
                    continue
                elif query.lower() == '/config':
                    self.show_config()
                    continue
                elif query.lower().startswith('/search'):
                    parts = query.lower().split()
                    if len(parts) > 1:
                        if parts[1] == 'on':
                            self.agent.set_search_enabled(True)
                            console.print("[green]✓ Search enabled[/green]")
                        elif parts[1] == 'off':
                            self.agent.set_search_enabled(False)
                            console.print("[yellow]⚠ Search disabled[/yellow]")
                    continue
                elif query.lower().startswith('/memory'):
                    parts = query.lower().split()
                    if len(parts) > 1:
                        if parts[1] == 'on':
                            self.agent.set_memory_enabled(True)
                            console.print("[green]✓ Conversation memory enabled[/green]")
                        elif parts[1] == 'off':
                            self.agent.set_memory_enabled(False)
                            console.print("[yellow]⚠ Conversation memory disabled[/yellow]")
                    continue
                elif query.lower() == '/clear':
                    self.agent.clear_memory()
                    console.print("[green]✓ Conversation memory cleared[/green]")
                    continue
                
                # Process the query
                console.print("[dim]🔍 Searching and analyzing...[/dim]")
                
                response = self.agent.process_query_sync(query)
                self.display_response(response)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
    
    def show_help(self):
        """Show help information"""
        help_text = """
        [bold]Available Commands:[/bold]
        
        • [cyan]/help[/cyan] - Show this help message
        • [cyan]/config[/cyan] - Show current configuration
        • [cyan]/search on/off[/cyan] - Enable or disable internet search
        • [cyan]/memory on/off[/cyan] - Enable or disable conversation memory
        • [cyan]/clear[/cyan] - Clear conversation memory
        • [cyan]/exit[/cyan] - Exit the application
        
        [bold]Usage:[/bold]
        Simply type your question and press Enter. The agent will search the internet
        for relevant information and provide a comprehensive answer with sources.
        
        [bold]Examples:[/bold]
        • "What are the latest developments in AI?"
        • "How does quantum computing work?"
        • "What happened in the news today?"
        """
        console.print(Panel(help_text, title="Help", border_style="cyan"))
    
    def show_config(self):
        """Show current configuration"""
        config_text = f"""
        [bold]Current Configuration:[/bold]
        
        • Model: {self.config.model_name}
        • Provider: {self.config.model_provider}
        • Temperature: {self.config.temperature}
        • Max Tokens: {self.config.max_tokens}
        • Max Search Results: {self.config.max_search_results}
        • Search Engine: {self.config.search_engine}
        • Memory Enabled: {getattr(self.agent, 'enable_memory', False)}
        • Debug Mode: {self.config.debug}
        """
        console.print(Panel(config_text, title="Configuration", border_style="yellow"))
    
    def single_query_mode(self, query: str):
        """Process a single query and exit"""
        self.initialize_agent()
        
        console.print(f"[dim]Processing: {query}[/dim]")
        response = self.agent.process_query_sync(query)
        self.display_response(response)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Agentic LLM Agent with Internet Search")
    parser.add_argument("query", nargs="?", help="Single query to process")
    parser.add_argument("--model", default="./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", help="LLM model to use")
    parser.add_argument("--provider", default="huggingface", choices=["huggingface", "openai"], help="Model provider to use")
    parser.add_argument("--no-search", action="store_true", help="Disable internet search")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum search results")
    
    args = parser.parse_args()
    
    cli = AgentCLI()
    
    # Override config with command line arguments
    if args.model:
        cli.config.model_name = args.model
    if args.provider:
        cli.config.model_provider = args.provider
    if args.max_results:
        cli.config.max_search_results = args.max_results
    
    if args.query:
        # Single query mode
        cli.single_query_mode(args.query)
    else:
        # Interactive mode
        cli.interactive_mode()

if __name__ == "__main__":
    main()
