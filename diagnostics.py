#!/usr/bin/env python3
"""
Agentic LLM Search Diagnostics Tool

This script performs advanced diagnostics on the system setup for the Agentic LLM Search project.
It checks GPU configuration, model compatibility, and helps users troubleshoot common issues.
"""

import os
import sys
import platform
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    print("Note: 'rich' module not found. Install with 'pip install rich' for better formatting.")
    print("Continuing with standard output...")

# Define color outputs for non-rich environments
class Colors:
    GREEN = "\033[92m" if not HAS_RICH else ""
    RED = "\033[91m" if not HAS_RICH else ""
    YELLOW = "\033[93m" if not HAS_RICH else ""
    BLUE = "\033[94m" if not HAS_RICH else ""
    BOLD = "\033[1m" if not HAS_RICH else ""
    END = "\033[0m" if not HAS_RICH else ""


def print_header(title):
    """Print a formatted header"""
    if HAS_RICH:
        console.print(f"\n[bold blue]{'-' * 60}[/bold blue]")
        console.print(f"[bold blue]{title.center(60)}[/bold blue]")
        console.print(f"[bold blue]{'-' * 60}[/bold blue]\n")
    else:
        print(f"\n{Colors.BLUE}{Colors.BOLD}{'-' * 60}{Colors.END}")
        print(f"{Colors.BLUE}{Colors.BOLD}{title.center(60)}{Colors.END}")
        print(f"{Colors.BLUE}{Colors.BOLD}{'-' * 60}{Colors.END}\n")


def print_success(message):
    """Print a success message"""
    if HAS_RICH:
        console.print(f"[bold green]✓ {message}[/bold green]")
    else:
        print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_error(message):
    """Print an error message"""
    if HAS_RICH:
        console.print(f"[bold red]✗ {message}[/bold red]")
    else:
        print(f"{Colors.RED}✗ {message}{Colors.END}")


def print_warning(message):
    """Print a warning message"""
    if HAS_RICH:
        console.print(f"[bold yellow]⚠ {message}[/bold yellow]")
    else:
        print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")


def print_info(message):
    """Print an info message"""
    if HAS_RICH:
        console.print(f"[blue]ℹ {message}[/blue]")
    else:
        print(f"{Colors.BLUE}ℹ {message}{Colors.END}")


def check_system_info():
    """Check and display system information"""
    print_header("System Information")
    
    # Get system info
    system = platform.system()
    release = platform.release()
    version = platform.version()
    machine = platform.machine()
    processor = platform.processor()
    python_version = platform.python_version()
    
    if HAS_RICH:
        table = Table(title="System Information")
        table.add_column("Component", style="cyan")
        table.add_column("Details", style="green")
        
        table.add_row("Operating System", f"{system} {release}")
        table.add_row("OS Version", version)
        table.add_row("Architecture", machine)
        table.add_row("Processor", processor)
        table.add_row("Python Version", python_version)
        
        console.print(table)
    else:
        print(f"Operating System: {system} {release}")
        print(f"OS Version: {version}")
        print(f"Architecture: {machine}")
        print(f"Processor: {processor}")
        print(f"Python Version: {python_version}")
    
    # Check for Apple Silicon
    is_apple_silicon = (system == "Darwin" and machine == "arm64")
    if is_apple_silicon:
        print_success("Apple Silicon detected (M-series chip)")
    
    return is_apple_silicon


def check_gpu_configuration():
    """Check GPU configuration and support"""
    print_header("GPU Configuration")
    
    apple_silicon = (platform.system() == "Darwin" and platform.machine() == "arm64")
    gpu_available = False
    metal_support = False
    torch_mps = False
    
    # Check for Apple Silicon GPU (Metal)
    if apple_silicon:
        try:
            subprocess.check_output(["system_profiler", "SPDisplaysDataType"], stderr=subprocess.STDOUT, text=True)
            print_success("Metal-compatible GPU detected")
            metal_support = True
            gpu_available = True
        except (subprocess.SubprocessError, FileNotFoundError):
            print_error("Failed to detect Metal GPU support")
    
    # Check PyTorch MPS support
    try:
        import torch
        torch_version = torch.__version__
        print_info(f"PyTorch version: {torch_version}")
        
        if apple_silicon and hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available'):
            if torch.mps.is_available():
                print_success("PyTorch MPS (Metal) support is available")
                torch_mps = True
            else:
                print_error("PyTorch was not built with MPS support or MPS is not available")
        elif apple_silicon:
            print_warning("PyTorch version doesn't support MPS (Metal). Consider upgrading to PyTorch 2.0+")
    except ImportError:
        print_error("PyTorch not installed")
    
    # Check ctransformers support
    try:
        import ctransformers
        ct_version = getattr(ctransformers, '__version__', 'Unknown')
        print_info(f"CTransformers version: {ct_version}")
        
        # Check if GPU support is built in
        if hasattr(ctransformers, 'get_available_backends'):
            backends = ctransformers.get_available_backends()
            if 'metal' in backends:
                print_success("CTransformers has Metal acceleration support")
            else:
                print_warning("CTransformers doesn't have Metal backend support")
                print_info("Available backends: " + ", ".join(backends))
    except ImportError:
        print_error("CTransformers not installed")
    
    return {
        'gpu_available': gpu_available,
        'metal_support': metal_support,
        'torch_mps': torch_mps
    }


def check_environment_variables():
    """Check environment variables configuration"""
    print_header("Environment Variables")
    
    env_vars = {
        'USE_GPU': os.environ.get('USE_GPU', 'Not set'),
        'USE_METAL': os.environ.get('USE_METAL', 'Not set'),
        'CONTEXT_LENGTH': os.environ.get('CONTEXT_LENGTH', 'Not set'),
        'GPU_LAYERS': os.environ.get('GPU_LAYERS', 'Not set'),
        'OPENAI_API_KEY': 'Present' if os.environ.get('OPENAI_API_KEY') else 'Not set',
        'DEFAULT_MODEL': os.environ.get('DEFAULT_MODEL', 'Not set'),
        'MODEL_PROVIDER': os.environ.get('MODEL_PROVIDER', 'Not set')
    }
    
    if HAS_RICH:
        table = Table(title="Environment Variables")
        table.add_column("Variable", style="cyan")
        table.add_column("Value", style="green")
        
        for var, value in env_vars.items():
            table.add_row(var, value)
        
        console.print(table)
    else:
        for var, value in env_vars.items():
            print(f"{var}: {value}")
    
    # Check for .env file
    if os.path.exists('.env'):
        print_success(".env file exists")
    else:
        print_warning(".env file not found - environment variables are not being loaded from file")
        if os.path.exists('.env.example'):
            print_info("Found .env.example - consider copying to .env and configuring variables")
    
    return env_vars


def check_model_files():
    """Check for model files and their details"""
    print_header("Model Files")
    
    model_dir = os.path.join("src", "models")
    if not os.path.exists(model_dir):
        print_error(f"Model directory {model_dir} not found")
        return False
    
    model_files = [f for f in os.listdir(model_dir) 
                  if f.endswith(('.gguf', '.bin', '.safetensors'))]
    
    if not model_files:
        print_error("No model files found")
        return False
    
    if HAS_RICH:
        table = Table(title="Model Files")
        table.add_column("Model File", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Last Modified", style="yellow")
        
        for model_file in model_files:
            file_path = os.path.join(model_dir, model_file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M')
            table.add_row(model_file, f"{size_mb:.2f} MB", mod_time)
        
        console.print(table)
    else:
        for model_file in model_files:
            file_path = os.path.join(model_dir, model_file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M')
            print(f"Model: {model_file}, Size: {size_mb:.2f} MB, Modified: {mod_time}")
    
    # Check for TinyLlama model specifically
    tinyllama_path = os.path.join(model_dir, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    if os.path.exists(tinyllama_path):
        print_success("TinyLlama model found")
    else:
        print_warning("Default TinyLlama model not found")
    
    return True


def check_module_structure():
    """Check the project module structure"""
    print_header("Module Structure")
    
    modules_to_check = [
        ("src.agents.agentic_llm", "AgenticLLMAgent", "Agent implementation"),
        ("src.models.llm_models", "HuggingFaceModel", "Model implementation"),
        ("src.models.llm_models", "AgentModelOrchestrator", "Model orchestrator"),
        ("src.tools.search_tool", "InternetSearchTool", "Search functionality"),
        ("src.models.env_setup", "setup_huggingface_env", "Environment setup")
    ]
    
    if HAS_RICH:
        table = Table(title="Module Structure")
        table.add_column("Module", style="cyan")
        table.add_column("Class/Function", style="yellow")
        table.add_column("Status", style="green")
    
    for module_path, attribute, description in modules_to_check:
        try:
            module_parts = module_path.split('.')
            spec = importlib.util.find_spec(module_path)
            
            if spec is None:
                if HAS_RICH:
                    table.add_row(module_path, attribute, "[bold red]Not Found[/bold red]")
                else:
                    print(f"{Colors.RED}✗ {module_path}.{attribute} - Not Found{Colors.END}")
                continue
                
            module = importlib.import_module(module_path)
            
            if hasattr(module, attribute):
                if HAS_RICH:
                    table.add_row(module_path, attribute, "[bold green]✓ Found[/bold green]")
                else:
                    print(f"{Colors.GREEN}✓ {module_path}.{attribute} - Found{Colors.END}")
            else:
                if HAS_RICH:
                    table.add_row(module_path, attribute, "[bold red]Missing Attribute[/bold red]")
                else:
                    print(f"{Colors.RED}✗ {module_path}.{attribute} - Missing Attribute{Colors.END}")
                    
        except ImportError:
            if HAS_RICH:
                table.add_row(module_path, attribute, "[bold red]Import Error[/bold red]")
            else:
                print(f"{Colors.RED}✗ {module_path}.{attribute} - Import Error{Colors.END}")
    
    if HAS_RICH:
        console.print(table)


def run_quick_test():
    """Run a quick functionality test"""
    print_header("Quick Functionality Test")
    print_info("Attempting to initialize the LLM model (without inference)...")
    
    try:
        # Import necessary modules
        from src.models.llm_models import HuggingFaceModel
        
        # Create an instance of the model
        model = HuggingFaceModel(
            model_path="./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            context_length=2048,  # Use a smaller context for testing
            use_gpu=False  # Disable GPU for testing
        )
        print_success("Model initialization successful!")
    except ImportError as e:
        print_error(f"Import error: {e}")
    except Exception as e:
        print_error(f"Error initializing model: {str(e)}")


def provide_recommendations(gpu_config):
    """Provide recommendations based on the diagnostics"""
    print_header("Recommendations")
    
    issues = []
    recommendations = []
    
    # Python version
    if sys.version_info < (3, 12):
        issues.append("Python version below 3.12")
        recommendations.append("Upgrade to Python 3.12 for best performance")
    
    # GPU configuration
    if gpu_config['gpu_available'] and gpu_config['metal_support']:
        if not gpu_config['torch_mps']:
            issues.append("PyTorch MPS support not available")
            recommendations.append("Set USE_METAL=True in .env file")
            recommendations.append("Ensure PyTorch version >= 2.0.0 for MPS support")
    
    # Check for .env file
    if not os.path.exists('.env'):
        issues.append(".env file not found")
        recommendations.append("Create a .env file based on .env.example")
    
    # Model recommendations
    model_dir = os.path.join("src", "models")
    tinyllama_path = os.path.join(model_dir, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    if not os.path.exists(tinyllama_path):
        issues.append("TinyLlama model not found")
        recommendations.append("Run 'python download_model.py' to download the TinyLlama model")
    
    if not issues:
        if HAS_RICH:
            console.print(Panel.fit("[bold green]No issues found. Your system is ready![/bold green]"))
        else:
            print(f"\n{Colors.GREEN}{Colors.BOLD}No issues found. Your system is ready!{Colors.END}\n")
    else:
        if HAS_RICH:
            console.print("[bold yellow]Issues Found:[/bold yellow]")
            for i, issue in enumerate(issues, 1):
                console.print(f"[yellow]{i}. {issue}[/yellow]")
            
            console.print("\n[bold blue]Recommendations:[/bold blue]")
            for i, rec in enumerate(recommendations, 1):
                console.print(f"[blue]{i}. {rec}[/blue]")
        else:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}Issues Found:{Colors.END}")
            for i, issue in enumerate(issues, 1):
                print(f"{Colors.YELLOW}{i}. {issue}{Colors.END}")
            
            print(f"\n{Colors.BLUE}{Colors.BOLD}Recommendations:{Colors.END}")
            for i, rec in enumerate(recommendations, 1):
                print(f"{Colors.BLUE}{i}. {rec}{Colors.END}")


def main():
    """Run the diagnostics tool"""
    if HAS_RICH:
        console.print("[bold cyan]===== Agentic LLM Search Diagnostics Tool =====[/bold cyan]\n")
    else:
        print(f"{Colors.BLUE}{Colors.BOLD}===== Agentic LLM Search Diagnostics Tool ====={Colors.END}\n")
    
    check_system_info()
    gpu_config = check_gpu_configuration()
    check_environment_variables()
    check_model_files()
    check_module_structure()
    run_quick_test()
    provide_recommendations(gpu_config)
    
    if HAS_RICH:
        console.print("\n[bold cyan]Diagnostics complete![/bold cyan]")
    else:
        print(f"\n{Colors.BLUE}{Colors.BOLD}Diagnostics complete!{Colors.END}")


if __name__ == "__main__":
    main()
