#!/usr/bin/env python3
"""
Run tests for the Agentic LLM Search project
This script makes it easier to run pytest with the right configuration
"""

import os
import sys
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

def find_python():
    """Find the Python executable to use"""
    venv_path = Path("venv")
    if venv_path.exists():
        if os.name == "nt":  # Windows
            python_path = venv_path / "Scripts" / "python.exe"
        else:
            python_path = venv_path / "bin" / "python"
        
        if python_path.exists():
            return str(python_path)
    
    return sys.executable  # Default to the current Python

def run_test(test_file=None):
    """Run tests with pytest"""
    python_exe = find_python()
    
    # Build pytest command
    cmd = [python_exe, "-m", "pytest"]
    
    # Add verbosity flag
    cmd.append("-v")
    
    # Add specific test file if provided
    if test_file:
        cmd.append(test_file)
    
    console.print(Panel(f"Running tests with: {' '.join(cmd)}", title="Agentic LLM Search Tests"))
    
    # Run the command
    result = subprocess.run(cmd)
    
    # Check if tests were successful
    if result.returncode != 0:
        console.print("\n[bold red]❌ Tests failed![/bold red]")
        return False
    else:
        console.print("\n[bold green]✅ All tests passed![/bold green]")
        return True

def main():
    """Main entry point"""
    console.print("[bold]===== Agentic LLM Search Test Runner =====[/bold]\n")
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        console.print(f"Running specific test: {test_file}")
        run_test(test_file)
    else:
        console.print("Running all tests")
        run_test()

if __name__ == "__main__":
    main()
