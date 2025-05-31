#!/usr/bin/env python3
"""
Python 3.12 Compatibility Check for Agentic LLM Search
"""

import sys
import importlib.util
import subprocess
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

def check_python_version():
    version = sys.version_info
    console.print(f"[bold]Current Python version:[/bold] {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 12:
        console.print("[bold green]✓ Using Python 3.12 or newer[/bold green]")
    else:
        console.print("[bold yellow]⚠ Not using Python 3.12 (using {version.major}.{version.minor})[/bold yellow]")
        console.print("Consider upgrading to Python 3.12 for better performance.")

def check_package(package_name):
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return False, "Not installed"
        
        # Try to get version
        try:
            module = importlib.import_module(package_name)
            if hasattr(module, '__version__'):
                return True, module.__version__
            elif hasattr(module, 'VERSION'):
                return True, module.VERSION
            else:
                return True, "Unknown version"
        except Exception:
            return True, "Installed (version unknown)"
            
    except ImportError:
        return False, "Import error"

def check_dependencies():
    dependencies = [
        "torch", "transformers", "huggingface_hub", "openai", 
        "duckduckgo_search", "beautifulsoup4", "requests",
        "ctransformers", "hf_transfer"
    ]
    
    table = Table(title="Dependency Check")
    table.add_column("Package", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Version", style="yellow")
    
    all_installed = True
    
    for dep in dependencies:
        installed, version = check_package(dep)
        status = "[green]✓ Installed[/green]" if installed else "[red]✗ Not installed[/red]"
        table.add_row(dep, status, version)
        
        if not installed:
            all_installed = False
    
    console.print(table)
    return all_installed

def check_tinyllama_model():
    import os
    model_path = "./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        console.print(f"[bold green]✓ TinyLlama model found ({size_mb:.1f} MB)[/bold green]")
        return True
    else:
        console.print("[bold red]✗ TinyLlama model not found at expected path[/bold red]")
        console.print("  Expected path: " + model_path)
        return False

def main():
    console.print("[bold cyan]===== Agentic LLM Search Python 3.12 Compatibility Check =====[/bold cyan]\n")
    
    check_python_version()
    console.print("")
    
    deps_installed = check_dependencies()
    console.print("")
    
    model_exists = check_tinyllama_model()
    console.print("")
    
    if deps_installed and model_exists:
        console.print("[bold green]✅ All checks passed! Ready for testing.[/bold green]")
        console.print("\nTo run the agentic search test:")
        console.print("[bold]python test_agentic_search.py[/bold]")
    else:
        console.print("[bold yellow]⚠️ Some checks failed. Please resolve issues before testing.[/bold yellow]")
        
        if not deps_installed:
            console.print("\nInstall missing dependencies:")
            console.print("[bold]pip install -r requirements.txt[/bold]")
        
        if not model_exists:
            console.print("\nDownload the TinyLlama model:")
            console.print("Visit https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
            console.print("Download the Q4_K_M.gguf model and place it in src/models/")

if __name__ == "__main__":
    main()
