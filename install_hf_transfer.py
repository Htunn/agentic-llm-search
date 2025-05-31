#!/usr/bin/env python3
"""
Utility script to install the hf_transfer package for faster downloads
"""

import os
import sys
import subprocess
from rich.console import Console

console = Console()

def install_hf_transfer():
    """Install the hf_transfer package"""
    console.print("[bold]Installing hf_transfer package for faster downloads...[/bold]")
    
    try:
        import hf_transfer
        console.print("[green]✓ hf_transfer is already installed[/green]")
        return True
    except ImportError:
        try:
            console.print("Installing hf_transfer...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "hf_transfer"])
            console.print("[green]✓ Successfully installed hf_transfer[/green]")
            
            # Set environment variable for current process
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            
            # Try to import to verify installation
            try:
                import hf_transfer
                console.print("[green]✓ hf_transfer imported successfully[/green]")
                return True
            except ImportError:
                console.print("[yellow]⚠ Installed hf_transfer but cannot import it[/yellow]")
                return False
                
        except subprocess.CalledProcessError as e:
            console.print(f"[red]✗ Failed to install hf_transfer: {e}[/red]")
            return False

if __name__ == "__main__":
    install_hf_transfer()
