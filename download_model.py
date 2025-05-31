#!/usr/bin/env python3
"""
Download script for TinyLlama GGUF model
This script will download the TinyLlama model for local inference
"""

import os
import sys
import subprocess
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn
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

MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"
MODEL_PATH = f"./src/models/{MODEL_FILE}"

def install_missing_packages():
    """Install missing required packages"""
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
                global HF_HUB_AVAILABLE
                HF_HUB_AVAILABLE = True
            except ImportError:
                pass
                
        if "hf_transfer" in missing_packages:
            try:
                import hf_transfer
                global HF_TRANSFER_AVAILABLE 
                HF_TRANSFER_AVAILABLE = True
            except ImportError:
                pass

def ensure_directory_exists():
    """Make sure the models directory exists"""
    os.makedirs("./src/models", exist_ok=True)

def download_model():
    """Download the TinyLlama model"""
    # Check if model already exists
    if os.path.exists(MODEL_PATH):
        file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        console.print(f"[bold green]Model already exists ({file_size_mb:.1f} MB)[/bold green]")
        return True
    
    # Install any missing packages
    install_missing_packages()
    
    # Make sure the models directory exists
    ensure_directory_exists()
    
    console.print(f"[bold cyan]Downloading TinyLlama model from Hugging Face...[/bold cyan]")
    
    # Try to download with huggingface_hub if available
    if HF_HUB_AVAILABLE:
        try:
            # Set environment variable to enable hf_transfer if available
            env_vars = os.environ.copy()
            if HF_TRANSFER_AVAILABLE:
                env_vars["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
                console.print("[green]Using fast download with hf_transfer[/green]")
            
            console.print(f"Downloading {MODEL_FILE} from {MODEL_REPO}")
            # Download the model
            with console.status("[bold green]Downloading model (this may take a while)...[/bold green]"):
                local_path = hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=MODEL_FILE,
                    local_dir="./src/models",
                    local_dir_use_symlinks=False
                )
            
            if os.path.exists(local_path):
                file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                console.print(f"[bold green]Downloaded to {local_path} ({file_size_mb:.1f} MB)[/bold green]")
                return True
            else:
                console.print("[bold red]Failed to download with huggingface_hub. Trying with requests.[/bold red]")
        except Exception as e:
            console.print(f"[bold yellow]Error using huggingface_hub: {str(e)}. Trying with requests.[/bold yellow]")
    
    # Fall back to using requests for direct download
    try:
        console.print(f"Direct download from URL: {MODEL_URL}")
        
        # Start a streaming request
        response = requests.get(MODEL_URL, stream=True)
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
            
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))
        
        # Check if download was successful
        if os.path.exists(MODEL_PATH):
            file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            console.print(f"[bold green]Download complete! ({file_size_mb:.1f} MB)[/bold green]")
            return True
        else:
            console.print("[bold red]Download failed: File not found after download[/bold red]")
            return False
            
    except Exception as e:
        console.print(f"[bold red]Error downloading model: {str(e)}[/bold red]")
        return False

def main():
    console.print("[bold]===== TinyLlama Model Downloader =====[/bold]")
    success = download_model()
    
    if success:
        console.print("\n[bold green]✅ Model is ready to use![/bold green]")
    else:
        console.print("\n[bold red]❌ Failed to download model[/bold red]")
        console.print("Please download the model manually from:")
        console.print("https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
        console.print(f"And place it at: {MODEL_PATH}")

if __name__ == "__main__":
    main()
