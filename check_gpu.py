#!/usr/bin/env python3
"""
Check GPU Acceleration for Agentic LLM Search
This script verifies if GPU acceleration is available for the model
"""

import os
import platform
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import subprocess

# Set up console
console = Console()

def check_apple_silicon():
    """Check if running on Apple Silicon"""
    is_mac = platform.system() == "Darwin"
    is_arm = platform.machine().startswith(('arm', 'aarch'))
    
    if is_mac and is_arm:
        return True, "Apple Silicon detected (M-series chip)"
    elif is_mac:
        return False, f"Mac detected but using {platform.machine()} architecture"
    else:
        return False, f"Not running on Mac ({platform.system()} {platform.machine()})"

def check_torch_mps():
    """Check if PyTorch MPS (Metal Performance Shaders) is available"""
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                return True, f"PyTorch {torch.__version__} with MPS support"
            else:
                return False, f"PyTorch {torch.__version__} built without MPS support"
        else:
            return False, f"PyTorch {torch.__version__} without MPS support"
    except ImportError:
        return False, "PyTorch not installed"

def check_ctransformers_metal():
    """Check if ctransformers has Metal support"""
    try:
        import ctransformers
        version = getattr(ctransformers, '__version__', 'unknown')
        
        # This is a simplistic check - ctransformers doesn't expose Metal capability directly
        if version >= '0.2.24':
            return True, f"ctransformers {version} should support Metal acceleration"
        else:
            return False, f"ctransformers {version} may have limited Metal support"
    except ImportError:
        return False, "ctransformers not installed"

def run_torch_benchmark():
    """Run a simple PyTorch benchmark to test GPU performance"""
    try:
        import torch
        import time
        
        console.print("\n[bold]Running PyTorch benchmark to test GPU performance...[/bold]")
        
        # Create tensors
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            console.print("Using [bold green]MPS (Metal)[/bold green] device for benchmark")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            console.print("Using [bold green]CUDA[/bold green] device for benchmark")
        else:
            device = torch.device("cpu")
            console.print("Using [bold yellow]CPU[/bold yellow] device for benchmark")
        
        # Run matrix multiplication benchmark
        sizes = [1024, 2048, 4096]
        for size in sizes:
            console.print(f"Testing {size}x{size} matrix multiplication... ", end="")
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Warmup
            torch.matmul(a, b)
            torch.sync(device)
            
            # Benchmark
            start = time.time()
            for _ in range(3):
                torch.matmul(a, b)
            torch.sync(device)
            end = time.time()
            
            avg_time = (end - start) / 3
            console.print(f"[green]{avg_time:.4f}s[/green]")
        
        return True
    except Exception as e:
        console.print(f"[red]Benchmark failed: {str(e)}[/red]")
        return False

def main():
    """Main function"""
    console.print(Panel("GPU Acceleration Check for Agentic LLM Search", style="bold cyan"))
    
    # System information
    console.print("\n[bold]System Information:[/bold]")
    console.print(f"OS: {platform.system()} {platform.release()}")
    console.print(f"Python: {platform.python_version()}")
    
    # Hardware checks
    table = Table(title="Hardware Acceleration")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Details", style="yellow")
    
    # Check for Apple Silicon
    is_as, as_details = check_apple_silicon()
    table.add_row(
        "Apple Silicon", 
        "[green]Available[/green]" if is_as else "[yellow]Not available[/yellow]",
        as_details
    )
    
    # Check PyTorch MPS
    has_mps, mps_details = check_torch_mps()
    table.add_row(
        "PyTorch Metal", 
        "[green]Available[/green]" if has_mps else "[yellow]Not available[/yellow]",
        mps_details
    )
    
    # Check ctransformers Metal
    has_ct_metal, ct_details = check_ctransformers_metal()
    table.add_row(
        "CTransformers Metal", 
        "[green]Available[/green]" if has_ct_metal else "[yellow]Limited/Not available[/yellow]",
        ct_details
    )
    
    console.print(table)
    
    # Print recommendations
    console.print("\n[bold]Recommendations:[/bold]")
    
    if is_as:
        if has_mps:
            console.print("[green]✓ Your system is configured for Metal GPU acceleration![/green]")
            console.print("  Your model will use the M-series GPU for faster inference.")
        else:
            console.print("[yellow]⚠ You have Apple Silicon but PyTorch MPS is not available[/yellow]")
            console.print("  Try running: pip install --upgrade torch")
    else:
        console.print("[yellow]⚠ No Apple Silicon detected[/yellow]")
        console.print("  The system will use CPU for model inference.")
    
    # Benchmark if GPU is available
    if has_mps or torch.cuda.is_available() if 'torch' in sys.modules else False:
        run_torch_benchmark()
    
    # Explain context length issue
    console.print("\n[bold]Context Length Configuration:[/bold]")
    console.print("The default context length for ctransformers has been increased to 4096")
    console.print("to fix the 'Number of tokens exceeded maximum context length' warning.")
    console.print("You can adjust this in the .env file by changing the CONTEXT_LENGTH value.")

if __name__ == "__main__":
    import sys
    main()
