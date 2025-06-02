#!/usr/bin/env python3
"""
Test script for the conversation memory feature of the Agentic LLM Agent.

This script verifies that:
1. The agent correctly remembers previous interactions
2. Follow-up questions are correctly handled with context
3. Memory can be cleared and disabled/enabled as expected
"""

import os
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import time

# Add the project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the agent
from src.agents.agentic_llm import AgenticLLMAgent
from src.agents.memory import ConversationMemory

# Load environment variables
load_dotenv()

console = Console()

def print_section(title):
    """Print a section header"""
    console.print(f"\n[bold cyan]{'='*20} {title} {'='*20}[/bold cyan]\n")

def print_result(title, text):
    """Print a result in a panel"""
    console.print(Panel(
        Markdown(text),
        title=title,
        border_style="green"
    ))

def display_memory(memory):
    """Display conversation memory contents"""
    console.print("[bold]Current conversation memory:[/bold]")
    for i, msg in enumerate(memory.messages):
        prefix = "👤 User:" if msg.role == "user" else "🤖 Agent:"
        console.print(f"[{i+1}] {prefix} {msg.content[:100]}..." if len(msg.content) > 100 else f"[{i+1}] {prefix} {msg.content}")

def main():
    """Main test function"""
    # Initialize the agent with memory enabled
    print_section("INITIALIZING AGENT")
    
    try:
        # Use a smaller local model for faster testing if available
        model_name = "./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        model_provider = "huggingface"
        
        # Check if we have OpenAI key for better quality responses
        if os.getenv("OPENAI_API_KEY"):
            console.print("[yellow]OpenAI API key detected. Using OpenAI for better quality responses.[/yellow]")
            model_name = "gpt-3.5-turbo"
            model_provider = "openai"
            
        # Initialize agent
        agent = AgenticLLMAgent(
            model_name=model_name,
            model_provider=model_provider,
            enable_memory=True,
            max_memory=5  # Keep small for testing
        )
        
        console.print(f"[green]✓ Agent initialized with {model_provider} model: {model_name}[/green]")
        console.print(f"[green]✓ Memory enabled: {agent.enable_memory}, Max memory: {agent.memory.max_history}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to initialize agent: {str(e)}[/red]")
        return
    
    # Test 1: Basic memory functionality
    print_section("TEST 1: BASIC MEMORY FUNCTIONALITY")
    
    # First question
    console.print("[bold]Asking first question...[/bold]")
    query1 = "What is quantum computing?"
    console.print(f"Question: {query1}")
    
    response1 = agent.process_query_sync(query1)
    print_result("Response 1", response1.answer)
    
    # Check memory after first question
    console.print("\n[bold]Checking memory after first question:[/bold]")
    display_memory(agent.memory)
    
    # Second question (follow-up)
    console.print("\n[bold]Asking follow-up question...[/bold]")
    query2 = "What are its practical applications?"  # Follow-up question without explicit subject
    console.print(f"Question: {query2}")
    
    response2 = agent.process_query_sync(query2)
    print_result("Response 2", response2.answer)
    
    # Check memory after second question
    console.print("\n[bold]Checking memory after follow-up question:[/bold]")
    display_memory(agent.memory)
    
    # Test 2: Memory clearing
    print_section("TEST 2: MEMORY CLEARING")
    
    console.print("[bold]Clearing memory...[/bold]")
    agent.clear_memory()
    
    # Verify memory is empty
    console.print("[bold]Checking if memory is empty:[/bold]")
    display_memory(agent.memory)
    
    # Test 3: Memory disable/enable
    print_section("TEST 3: MEMORY DISABLE/ENABLE")
    
    # Ask a question with memory disabled
    console.print("[bold]Disabling memory and asking a question...[/bold]")
    agent.set_memory_enabled(False)
    
    query3 = "Who developed the theory of relativity?"
    console.print(f"Question: {query3}")
    
    response3 = agent.process_query_sync(query3)
    print_result("Response with memory disabled", response3.answer)
    
    # Check if memory was updated (shouldn't be)
    console.print("\n[bold]Checking if memory was updated (should be empty):[/bold]")
    display_memory(agent.memory)
    
    # Re-enable memory
    console.print("\n[bold]Re-enabling memory and asking a question...[/bold]")
    agent.set_memory_enabled(True)
    
    response4 = agent.process_query_sync(query3)
    print_result("Response with memory re-enabled", response4.answer)
    
    # Check if memory was updated (should be now)
    console.print("\n[bold]Checking if memory was updated (should have entries):[/bold]")
    display_memory(agent.memory)
    
    # Test 4: Complex follow-up
    print_section("TEST 4: COMPLEX FOLLOW-UP HANDLING")
    
    # First question establishing context
    console.print("[bold]Starting new conversation about AI...[/bold]")
    agent.clear_memory()
    
    query5 = "What are the main types of machine learning algorithms?"
    console.print(f"Question: {query5}")
    
    response5 = agent.process_query_sync(query5)
    print_result("Response about ML algorithms", response5.answer)
    
    # Complex follow-up with pronouns
    console.print("\n[bold]Asking complex follow-up with pronouns...[/bold]")
    query6 = "Which one of these is best for time series analysis and why?"
    console.print(f"Follow-up question: {query6}")
    
    response6 = agent.process_query_sync(query6)
    print_result("Response to follow-up", response6.answer)
    
    # Display final memory state
    console.print("\n[bold]Final memory state:[/bold]")
    display_memory(agent.memory)
    
    # Success message
    console.print("\n[bold green]Conversation memory testing completed![/bold green]")

if __name__ == "__main__":
    main()
