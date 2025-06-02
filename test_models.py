#!/usr/bin/env python3
"""
Test script for different LLM models
This script will test if different GGUF models can be loaded and used
"""

import os
import sys
import time
import argparse
import logging
from rich.console import Console
from rich.logging import RichHandler
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("test_models")

# Create console for rich output
console = Console()

# Load environment variables
load_dotenv()

def determine_model_type(model_path):
    """Try to determine the model type from the filename"""
    filename = os.path.basename(model_path).lower()
    
    if any(name in filename for name in ["llama-3", "llama3"]):
        return "Llama 3"
    elif any(name in filename for name in ["phi-3", "phi3"]):
        return "Phi-3"
    elif any(name in filename for name in ["llama-2", "llama2"]):
        return "Llama 2"
    elif "tinyllama" in filename:
        return "TinyLlama"
    else:
        return "Unknown"

def test_model(model_path):
    """Test a specific model by loading it and generating a response"""
    try:
        model_type = determine_model_type(model_path)
        console.print(f"[bold cyan]Testing model: {model_path}[/bold cyan]")
        console.print(f"[cyan]Detected model type: {model_type}[/cyan]")
        
        # Import here to avoid loading unnecessary modules
        from src.models.llm_models import HuggingFaceModel
        
        # Create the model instance
        console.print("[yellow]Loading model...[/yellow]")
        start_time = time.time()
        model = HuggingFaceModel(model_name=model_path)
        load_time = time.time() - start_time
        console.print(f"[green]Model loaded successfully in {load_time:.2f} seconds![/green]")
        
        # Try a simple generation
        console.print("[yellow]Generating a test response...[/yellow]")
        test_prompts = [
            "What is artificial intelligence?",
            "Write a short poem about technology."
        ]
        
        for prompt in test_prompts:
            console.print(f"[yellow]Testing prompt: '{prompt}'[/yellow]")
            start_time = time.time()
            response = model.generate_response(prompt)
            gen_time = time.time() - start_time
            
            console.print(f"[green]Response generated in {gen_time:.2f} seconds[/green]")
            console.print(f"[cyan]Response:[/cyan] {response[:200]}...")
            console.print("---")
        
        console.print("[bold green]Model test successful![/bold green]")
        return True
    except Exception as e:
        console.print(f"[bold red]Error testing model: {str(e)}[/bold red]")
        return False

def list_available_models():
    """List all available GGUF models in the models directory"""
    models_dir = "./src/models"
    if not os.path.exists(models_dir):
        console.print("[yellow]Models directory doesn't exist yet.[/yellow]")
        return []
    
    models = []
    for file in os.listdir(models_dir):
        if file.endswith(".gguf"):
            models.append(os.path.join(models_dir, file))
    
    return models

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test different LLM models")
    parser.add_argument("--model", help="Path to the model to test")
    parser.add_argument("--list", action="store_true", help="List all available models")
    args = parser.parse_args()
    
    # List available models
    if args.list or not args.model:
        console.print("[bold cyan]Available Models:[/bold cyan]")
        models = list_available_models()
        if models:
            for i, model in enumerate(models, 1):
                console.print(f"[{i}] {os.path.basename(model)}")
        else:
            console.print("[yellow]No GGUF models found. Use download_model.py to download models.[/yellow]")
        
        if not args.model:
            return
    
    # Test specific model if provided
    if args.model:
        test_model(args.model)

if __name__ == "__main__":
    main()
