#!/usr/bin/env python3
"""
Script to easily install HuggingFace CLI tools and authenticate
"""

import sys
import subprocess
from rich.console import Console
from rich.prompt import Confirm

console = Console()

def install_huggingface_cli():
    """Install or upgrade huggingface_hub with CLI support"""
    try:
        console.print("[bold cyan]Installing HuggingFace CLI tools...[/bold cyan]")
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub[cli]"], check=True)
        console.print("[green]Successfully installed HuggingFace CLI[/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error installing HuggingFace CLI: {str(e)}[/bold red]")
        return False

def login_to_huggingface():
    """Run the HuggingFace login command"""
    try:
        console.print("[bold cyan]Logging in to HuggingFace...[/bold cyan]")
        console.print("[yellow]You will be asked to provide your HuggingFace token from https://huggingface.co/settings/tokens[/yellow]")
        subprocess.run(["huggingface-cli", "login"], check=True)
        console.print("[green]Successfully logged in to HuggingFace![/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error logging in to HuggingFace: {str(e)}[/bold red]")
        return False
    except FileNotFoundError:
        console.print("[bold red]huggingface-cli command not found. Make sure it's properly installed.[/bold red]")
        return False

def install_hf_transfer():
    """Install the hf_transfer package for faster downloads"""
    try:
        console.print("[bold cyan]Installing hf_transfer for faster downloads...[/bold cyan]")
        subprocess.run([sys.executable, "-m", "pip", "install", "hf_transfer"], check=True)
        console.print("[green]Successfully installed hf_transfer[/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error installing hf_transfer: {str(e)}[/bold red]")
        return False

def main():
    console.print("[bold]===== HuggingFace Setup Helper =====[/bold]")
    
    # Install HuggingFace CLI
    if install_huggingface_cli():
        # Ask if the user wants to login now
        if Confirm.ask("Do you want to login to HuggingFace now?", default=True):
            login_to_huggingface()
    
    # Ask if they want to install hf_transfer for faster downloads
    if Confirm.ask("Install hf_transfer for faster downloads?", default=True):
        install_hf_transfer()
    
    console.print("\n[bold cyan]Next steps:[/bold cyan]")
    console.print("1. Run 'python download_model.py' to download a model")
    console.print("2. Update your .env file with the model path")
    console.print("3. Run 'streamlit run app.py' to start the application")

if __name__ == "__main__":
    main()
