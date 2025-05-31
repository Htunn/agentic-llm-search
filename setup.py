#!/usr/bin/env python3
"""
Setup script for Agentic LLM Search project

This script will:
1. Check if pip is installed
2. Install required dependencies from requirements.txt
3. Verify critical dependencies are installed
"""

import subprocess
import sys
import os


def print_header(message):
    """Print a formatted header message."""
    border = "=" * (len(message) + 4)
    print(f"\n{border}")
    print(f"| {message} |")
    print(f"{border}\n")


def check_pip():
    """Check if pip is installed."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                       check=True, capture_output=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_dependencies():
    """Install dependencies from requirements.txt file."""
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    
    if not os.path.exists(requirements_path):
        print(f"⚠️  Error: requirements.txt not found at {requirements_path}")
        return False
    
    print("📦 Installing dependencies from requirements.txt...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_path], check=True)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Error installing dependencies: {e}")
        return False


def verify_critical_dependencies():
    """Verify that critical dependencies were installed."""
    critical_packages = [
        "rich",  # For formatting
        "torch",  # For model inference
        "transformers",  # For model handling
        "ctransformers",  # For GGUF model support
        "requests",  # For web requests
        "duckduckgo_search",  # For search capabilities
    ]
    
    missing = []
    for package in critical_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"⚠️  Warning: The following critical packages could not be imported: {', '.join(missing)}")
        print("   You might need to install them manually or check your Python environment.")
        return False
    
    print("✅ All critical dependencies are installed and importable!")
    return True


def main():
    """Main function to set up the environment."""
    print_header("Agentic LLM Search Setup")
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"🐍 Python version: {python_version}")
    
    if sys.version_info.major != 3 or sys.version_info.minor < 12:
        print("⚠️  Warning: This project is optimized for Python 3.12 or newer")
        print(f"   You're running Python {python_version}")
        response = input("Do you want to continue with setup anyway? (y/n): ")
        if response.lower() != 'y':
            print("Setup aborted.")
            sys.exit(1)
    
    # Check if pip is installed
    if not check_pip():
        print("⚠️  Error: pip is not installed or not working")
        print("   Please install pip and try again")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Failed to install dependencies")
        sys.exit(1)
    
    # Verify critical dependencies
    if not verify_critical_dependencies():
        print("⚠️  Some critical dependencies could not be verified")
        print("   You might need to install them manually")
        print("   Try: pip install torch transformers ctransformers rich")
    
    print_header("Setup Complete")
    print("You can now run the compatibility check:")
    print("python check_compatibility.py")
    print("\nIf all checks pass, you can run the test script:")
    print("python test_agentic_search.py")
    print("\nHappy searching! 🔍")


if __name__ == "__main__":
    main()
