#!/usr/bin/env python3
"""
Test script for the Agentic LLM Search using TinyLlama
This script allows you to test the agent interactively
"""

import asyncio
import os
import sys
from rich.console import Console
from rich.markdown import Markdown
from rich import print as rprint

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.agentic_llm import AgenticLLMAgent
from src import AgentResponse

console = Console()

async def main():
    console.print("[bold cyan]Initializing Agentic LLM Search with TinyLlama...[/bold cyan]")
    
    # Initialize the agent
    try:
        agent = AgenticLLMAgent(
            model_name="./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            model_provider="huggingface",
            max_search_results=3,
            enable_search=True
        )
        console.print("[bold green]Agent initialized successfully![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Failed to initialize agent: {str(e)}[/bold red]")
        return
    
    # Interactive loop
    while True:
        console.print("\n[bold yellow]Enter your query (or type 'exit' to quit):[/bold yellow]")
        query = input("> ")
        
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        if not query.strip():
            continue
            
        console.print("[cyan]Searching and processing...[/cyan]")
        
        try:
            # Process the query
            response: AgentResponse = await agent.process_query(query)
            
            console.print("\n[bold green]Response:[/bold green]")
            console.print(Markdown(response.answer))
            
            console.print("\n[bold blue]Sources:[/bold blue]")
            for i, source in enumerate(response.sources, 1):
                console.print(f"[{i}] [link={source.url}]{source.title}[/link] ({source.url})")
                
            console.print(f"\n[dim]Model used: {response.model_used}[/dim]")
            
        except Exception as e:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")
    
    console.print("[bold cyan]Thank you for using Agentic LLM Search![/bold cyan]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold red]Search interrupted. Exiting...[/bold red]")
