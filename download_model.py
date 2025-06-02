#!/usr/bin/env python3
"""
Download script for various LLM GGUF models
This script will download models for local inference
"""

import os
import sys
import subprocess
from typing import Dict, List, Optional, Tuple
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn
from rich.prompt import Prompt
import requests
from pathlib import Path

# Try to import huggingface_hub to use its download functionality
try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Check for hf_transfer
try:
    import hf_transfer
    HF_TRANSFER_AVAILABLE = True
except ImportError:
    HF_TRANSFER_AVAILABLE = False

console = Console()

# Available models to download
AVAILABLE_MODELS = {
    "tinyllama": {
        "name": "TinyLlama 1.1B Chat",
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "type": "llama"
    },
    "llama3": {
        "name": "Llama 3 8B Instruct",
        "repo": "mlabonne/Llama-3-8B-Instruct-GGUF",
        "file": "Llama-3-8B-Instruct.Q4_K_M.gguf",
        "type": "llama"
    },
    "phi3": {
        "name": "Microsoft Phi-3 Mini",
        "repo": "microsoft/Phi-3-mini-4k-instruct-GGUF",
        "file": "phi-3-mini-4k-instruct-q4_k_m.gguf",
        "type": "mistral"
    },
    "llama2": {
        "name": "Llama 2 7B Chat",
        "repo": "TheBloke/Llama-2-7B-Chat-GGUF",
        "file": "llama-2-7b-chat.Q4_K_M.gguf",
        "type": "llama"
    }
}

def install_missing_packages():
    """Install missing required packages"""
    global HF_HUB_AVAILABLE
    global HF_TRANSFER_AVAILABLE
    
    packages_to_check = {
        "hf_transfer": HF_TRANSFER_AVAILABLE,
        "huggingface_hub": HF_HUB_AVAILABLE
    }
    
    missing_packages = [pkg for pkg, available in packages_to_check.items() if not available]
    
    if missing_packages:
        console.print(f"[bold yellow]Some required packages are missing: {', '.join(missing_packages)}[/bold yellow]")
        console.print("Attempting to install them now...")
        
        for package in missing_packages:
            try:
                console.print(f"Installing {package}...")
                subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
                console.print(f"[green]Successfully installed {package}[/green]")
            except subprocess.CalledProcessError:
                console.print(f"[red]Failed to install {package}. Continuing without it.[/red]")
        
        # Reimport if installation was successful
        if "huggingface_hub" in missing_packages:
            try:
                from huggingface_hub import hf_hub_download
                HF_HUB_AVAILABLE = True
            except ImportError:
                pass
                
        if "hf_transfer" in missing_packages:
            try:
                import hf_transfer
                HF_TRANSFER_AVAILABLE = True
            except ImportError:
                pass

def check_huggingface_login():
    """Check if user is logged in to HuggingFace"""
    if not HF_HUB_AVAILABLE:
        return False
    
    try:
        from huggingface_hub import HfApi
        
        hf_api = HfApi()
        try:
            # Try to get user info to check if logged in
            user_info = hf_api.whoami()
            console.print(f"[green]✓ Logged in to HuggingFace as: {user_info['name']}[/green]")
            return True
        except Exception:
            console.print("[yellow]Not logged in to HuggingFace. Some models might not be accessible.[/yellow]")
            console.print("[yellow]To login, run: huggingface-cli login[/yellow]")
            return False
    except Exception as e:
        console.print(f"[red]Error checking HuggingFace login: {str(e)}[/red]")
        return False

def ensure_directory_exists():
    """Make sure the models directory exists"""
    os.makedirs("./src/models", exist_ok=True)

def select_model():
    """Prompt user to select which model to download"""
    console.print("[bold cyan]Available Models:[/bold cyan]")
    
    # Display model options
    for i, (key, model) in enumerate(AVAILABLE_MODELS.items(), 1):
        console.print(f"[{i}] {model['name']} ({model['type']} architecture)")
    
    # Get user selection
    choice = Prompt.ask(
        "Select a model to download",
        choices=[str(i) for i in range(1, len(AVAILABLE_MODELS) + 1)],
        default="1"
    )
    
    # Convert choice to model key
    model_key = list(AVAILABLE_MODELS.keys())[int(choice) - 1]
    model_info = AVAILABLE_MODELS[model_key]
    
    console.print(f"\n[bold cyan]Selected: {model_info['name']}[/bold cyan]")
    
    return model_key, model_info

def download_model(model_key=None):
    """Download the selected model"""
    if model_key is None:
        model_key, model_info = select_model()
    else:
        model_info = AVAILABLE_MODELS.get(model_key)
        if not model_info:
            console.print(f"[bold red]Error: Model '{model_key}' not found in available models.[/bold red]")
            return False
    
    # Set model variables
    model_repo = model_info["repo"]
    model_file = model_info["file"]
    model_path = f"./src/models/{model_file}"
    model_url = f"https://huggingface.co/{model_repo}/resolve/main/{model_file}"
    
    # Check if model already exists
    if os.path.exists(model_path):
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        console.print(f"[bold green]Model already exists ({file_size_mb:.1f} MB)[/bold green]")
        return True
    
    # Install any missing packages
    install_missing_packages()
    
    # Make sure the models directory exists
    ensure_directory_exists()
    
    console.print(f"[bold cyan]Downloading {model_info['name']} from Hugging Face...[/bold cyan]")
    
    # Install any missing packages and check login
    install_missing_packages()
    is_logged_in = check_huggingface_login()
    
    # Try to download with huggingface_hub if available
    if HF_HUB_AVAILABLE:
        try:
            # Set environment variable to enable hf_transfer if available
            env_vars = os.environ.copy()
            if HF_TRANSFER_AVAILABLE:
                env_vars["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
                console.print("[green]Using fast download with hf_transfer[/green]")
            
            # For some models, suggest login if not logged in
            if not is_logged_in and (model_key == "llama3" or model_key == "phi3"):
                console.print(f"[yellow]Note: {model_info['name']} may require HuggingFace login for access[/yellow]")
                console.print("[yellow]If download fails, run 'huggingface-cli login' and try again[/yellow]")
            
            console.print(f"Downloading {model_file} from {model_repo}")
            # Download the model
            with console.status("[bold green]Downloading model (this may take a while)...[/bold green]"):
                local_path = hf_hub_download(
                    repo_id=model_repo,
                    filename=model_file,
                    local_dir="./src/models",
                    local_dir_use_symlinks=False
                )
            
            if os.path.exists(local_path):
                file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                console.print(f"[bold green]Downloaded to {local_path} ({file_size_mb:.1f} MB)[/bold green]")
                
                # Update the model filename if it's different from what we expected
                actual_filename = os.path.basename(local_path)
                
                # Provide instructions for updating the config
                console.print("\n[bold cyan]To use this model, update your .env file:[/bold cyan]")
                console.print(f"DEFAULT_MODEL=./src/models/{actual_filename}")
                console.print(f"MODEL_PROVIDER=huggingface")
                return True
            else:
                console.print("[bold red]Failed to download with huggingface_hub. Trying with requests.[/bold red]")
        except Exception as e:
            console.print(f"[bold yellow]Error using huggingface_hub: {str(e)}[/bold yellow]")
            console.print("[yellow]Trying with requests as fallback...[/yellow]")
    
    # Fall back to using requests for direct download
    try:
        console.print(f"Direct download from URL: {model_url}")
        
        # Start a streaming request
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        # Get the total size in bytes if available
        total_size = int(response.headers.get('content-length', 0))
        total_size_mb = total_size / (1024 * 1024)
        
        console.print(f"Total size: {total_size_mb:.1f} MB")
        
        # Download with progress bar
        with Progress(
            TextColumn("[bold blue]Downloading...", justify="right"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
        ) as progress:
            task = progress.add_task("[green]Downloading", total=total_size)
            
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))
        
        # Check if download was successful
        if os.path.exists(model_path):
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            console.print(f"[bold green]Download complete! ({file_size_mb:.1f} MB)[/bold green]")
            # Provide instructions for updating the config
            console.print("\n[bold cyan]To use this model, update your .env file:[/bold cyan]")
            console.print(f"DEFAULT_MODEL=./src/models/{model_file}")
            console.print(f"MODEL_PROVIDER=huggingface")
            return True
        else:
            console.print("[bold red]Download failed: File not found after download[/bold red]")
            return False
            
    except Exception as e:
        console.print(f"[bold red]Error downloading model: {str(e)}[/bold red]")
        return False

def show_usage():
    """Show script usage instructions"""
    console.print("[bold cyan]Usage:[/bold cyan]")
    console.print("  python download_model.py            # Interactive mode")
    console.print("  python download_model.py <model>    # Download specific model")
    console.print("  python download_model.py --help     # Show this help")
    console.print("\n[bold cyan]Available models:[/bold cyan]")
    
    for i, (key, model) in enumerate(AVAILABLE_MODELS.items(), 1):
        console.print(f"  {key:<10} - {model['name']} ({model['type']} architecture)")

def main():
    console.print("[bold]===== LLM Model Downloader =====[/bold]")
    
    # Show help if requested
    if len(sys.argv) > 1 and (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
        show_usage()
        return
    
    # Allow command-line selection of a specific model
    if len(sys.argv) > 1:
        if sys.argv[1] in AVAILABLE_MODELS:
            model_key = sys.argv[1]
            console.print(f"[bold]Using command line selected model: {AVAILABLE_MODELS[model_key]['name']}[/bold]")
            success = download_model(model_key)
        else:
            console.print(f"[bold red]Error: Unknown model '{sys.argv[1]}'[/bold red]")
            show_usage()
            return
    else:
        success = download_model()
    
    if success:
        console.print("\n[bold green]✅ Model is ready to use![/bold green]")
        console.print("[bold]To run the application with this model:[/bold]")
        console.print("1. Update your .env file with the settings shown above")
        console.print("2. Run the application: streamlit run app.py")
    else:
        console.print("\n[bold red]❌ Failed to download model[/bold red]")
        console.print("Please try one of the following:")
        console.print("1. Login to HuggingFace: huggingface-cli login")
        console.print("2. Try a different model")
        console.print("3. Download manually from HuggingFace:")
        
        for key, model in AVAILABLE_MODELS.items():
            console.print(f"   - {model['name']}: https://huggingface.co/{model['repo']}")

if __name__ == "__main__":
    main()
