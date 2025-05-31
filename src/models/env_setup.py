"""
Environment setup utilities for model loading
Handles package availability checks and environment variable configuration
"""

import os
import sys
import logging
import importlib.util
import subprocess
import platform

logger = logging.getLogger(__name__)

def is_package_installed(package_name):
    """Check if a package is installed"""
    return importlib.util.find_spec(package_name) is not None

def setup_huggingface_env():
    """Set up environment for HuggingFace models"""
    
    # Check if hf_transfer is installed
    hf_transfer_available = is_package_installed("hf_transfer")
    
    # Set or unset HF_HUB_ENABLE_HF_TRANSFER based on package availability
    if "HF_HUB_ENABLE_HF_TRANSFER" in os.environ and os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "1":
        if not hf_transfer_available:
            logger.warning("HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer package is not installed")
            logger.warning("Disabling fast downloads. Install with: pip install hf_transfer")
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
            return False
        else:
            logger.info("hf_transfer is enabled for faster downloads")
            return True
    
    return hf_transfer_available

def detect_apple_silicon():
    """
    Detect if running on Apple Silicon (M-series chips) and configure environment
    
    Returns:
        bool: True if running on Apple Silicon with GPU support enabled
    """
    is_apple_silicon = (platform.system() == "Darwin" and 
                       platform.machine().startswith(('arm', 'aarch')))
    
    if is_apple_silicon:
        logger.info("Detected Apple Silicon (M-series) hardware")
        
        # Check for PyTorch MPS support
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("PyTorch MPS (Metal) acceleration is available")
                
                # Set environment variables for optimal Metal performance
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                return True
            else:
                logger.info("PyTorch MPS acceleration is not available")
        except ImportError:
            logger.info("PyTorch not imported, cannot check MPS support")
    
    return False

def setup_gpu_environment():
    """
    Set up environment for GPU acceleration based on detected hardware
    """
    # Check for Apple Silicon first
    if detect_apple_silicon():
        # Set optimal environment variables for Apple Silicon
        logger.info("Configuring environment for Apple Silicon M-series GPU")
        
        # Configure ctransformers for Metal
        os.environ["CT_METAL_LAYERS"] = "32"  # Use Metal for up to 32 layers
        
        # Configure PyTorch for Metal
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.8"  # Allow using up to 80% of GPU memory
        
        return "mps"  # Metal Performance Shaders
    
    # Check for CUDA (NVIDIA) support
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA acceleration available with {torch.cuda.device_count()} device(s)")
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            return "cuda"
    except ImportError:
        pass
    
    logger.info("No GPU acceleration detected, using CPU")
    return "cpu"

def install_package(package_name):
    """Install a package if not already installed"""
    if is_package_installed(package_name):
        logger.info(f"Package {package_name} is already installed")
        return True
    
    logger.info(f"Installing package {package_name}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        logger.info(f"Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_name}: {str(e)}")
        return False
