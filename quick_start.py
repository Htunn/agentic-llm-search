#!/usr/bin/env python3
"""
Quick Start Script for Agentic LLM Search
Sets up the environment and runs a test query
"""

import os
import sys
import subprocess
from pathlib import Path
import platform
from rich.console import Console

console = Console()

def check_python_version():
    """Check if Python version is 3.12 or newer"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 12:
        console.print("[bold green]✓ Using Python 3.12 or newer[/bold green]")
        return True
    else:
        console.print(f"[bold yellow]⚠ Using Python {version.major}.{version.minor}, but 3.12+ is recommended[/bold yellow]")
        if platform.system() == "Darwin":  # macOS
            console.print("To install Python 3.12 on macOS:")
            console.print("  brew install python@3.12")
        elif platform.system() == "Linux":
            console.print("To install Python 3.12 on Linux:")
            console.print("  Follow instructions at https://www.python.org/downloads/")
        return False

def setup_environment():
    """Set up the Python environment"""
    console.print("\n[bold cyan]Setting up environment...[/bold cyan]")
    
    venv_dir = Path("venv")
    if venv_dir.exists():
        console.print("[green]✓ Virtual environment exists[/green]")
    else:
        console.print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
        console.print("[green]✓ Virtual environment created[/green]")
    
    # Activate virtual environment in the current process
    activate_script = "venv/bin/activate" if platform.system() != "Windows" else "venv\\Scripts\\activate"
    console.print(f"To activate the virtual environment, run:")
    console.print(f"  source {activate_script}")
    
    # Install requirements
    pip_path = "venv/bin/pip" if platform.system() != "Windows" else "venv\\Scripts\\pip"
    console.print("\nInstalling requirements...")
    subprocess.run([pip_path, "install", "-r", "requirements.txt"])
    
    # Install additional required packages
    console.print("\nInstalling additional required packages...")
    python_path = str(venv_dir / "bin" / "python") if platform.system() != "Windows" else str(venv_dir / "Scripts" / "python.exe")
    subprocess.run([python_path, "install_hf_transfer.py"])
    subprocess.run([pip_path, "install", "huggingface_hub"])
    console.print("[green]✓ Requirements installed[/green]")

def download_model():
    """Download the TinyLlama model if not exists"""
    console.print("\n[bold cyan]Checking model...[/bold cyan]")
    model_path = Path("src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    
    if model_path.exists() and model_path.stat().st_size > 1000000:  # >1MB
        size_mb = model_path.stat().st_size / (1024 * 1024)
        console.print(f"[green]✓ Model exists ({size_mb:.1f} MB)[/green]")
    else:
        console.print("Downloading TinyLlama model...")
        python_path = "venv/bin/python" if platform.system() != "Windows" else "venv\\Scripts\\python"
        subprocess.run([python_path, "download_model.py"])

def run_test():
    """Run a simple test of the agent"""
    console.print("\n[bold cyan]Running test query...[/bold cyan]")
    console.print("This will execute a test query using the agent.")
    console.print("Please wait while the model loads and processes the query...\n")
    
    python_path = "venv/bin/python" if platform.system() != "Windows" else "venv\\Scripts\\python"
    cmd = [python_path, "-c", """
import asyncio
from src.agents.agentic_llm import AgenticLLMAgent

async def test_query():
    agent = AgenticLLMAgent()
    print("Query: What is Python programming?")
    response = await agent.process_query("What is Python programming?")
    print(f"\\nResponse:\\n{response.answer}\\n")
    print("Sources:")
    for i, source in enumerate(response.sources, 1):
        print(f"[{i}] {source.title} ({source.url})")

if __name__ == "__main__":
    asyncio.run(test_query())
"""]
    
    subprocess.run(cmd)

def main():
    console.print("[bold]===== Agentic LLM Search Quick Start =====[/bold]\n")
    
    check_python_version()
    setup_environment()
    download_model()
    
    console.print("\n[bold green]✅ Setup complete![/bold green]")
    
    run_test()
    
    console.print("\n[bold green]✅ Quick start complete![/bold green]")
    console.print("\nTo continue using the agent, run:")
    console.print("  python test_agentic_search.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Quick start interrupted. Exiting...[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
        console.print("Please check the error message above and try again.")
