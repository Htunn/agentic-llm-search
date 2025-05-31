#!/usr/bin/env bash
# Simple script to run the Agentic LLM Search application

# Function to check if Python 3.12+ is available
check_python_version() {
  python_version=$(python3 --version 2>&1)
  version_major=$(echo $python_version | cut -d. -f1 | grep -o "[0-9]\+")
  version_minor=$(echo $python_version | cut -d. -f2)
  
  if [ "$version_major" -lt 3 ] || [ "$version_major" -eq 3 -a "$version_minor" -lt 12 ]; then
    echo "Warning: Python 3.12+ is recommended but found $python_version"
    echo "The application may still work, but some features might be limited."
  else
    echo "Python version check passed: $python_version"
  fi
}

# Check for virtual environment
check_venv() {
  # Check for both venv and .venv directories
  if [ -d ".venv" ]; then
    echo "Virtual environment (.venv) found. Activating..."
    source .venv/bin/activate
  elif [ -d "venv" ]; then
    echo "Virtual environment (venv) found. Activating..."
    source venv/bin/activate
  else
    echo "Virtual environment not found. Creating one..."
    python3 -m venv .venv
    echo "Activating virtual environment and installing dependencies..."
    source .venv/bin/activate
    pip install -r requirements.txt
  fi
}

# Install required packages
install_packages() {
  echo "Checking for required packages..."
  echo "Installing hf_transfer and huggingface_hub for faster model downloads..."
  python install_hf_transfer.py
  pip install -q huggingface_hub
  echo "Required packages checked and installed if needed."
}

# Check for model
check_model() {
  if [ ! -f "src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" ]; then
    echo "TinyLlama model not found. Downloading..."
    install_packages
    python download_model.py
  else
    echo "TinyLlama model found."
  fi
}

# Main function
main() {
  echo "==== Agentic LLM Search ===="
  echo "Setting up environment..."
  
  check_python_version
  check_venv
  check_model
  
  echo ""
  echo "Starting the application..."
  echo "Choose an interface:"
  echo "1. Command Line Interface"
  echo "2. Web Interface (Streamlit)"
  read -p "Enter your choice (1-2): " choice
  
  case $choice in
    1)
      echo "Starting command line interface..."
      python3 test_agentic_search.py
      ;;
    2)
      echo "Starting web interface with Streamlit..."
      streamlit run app.py
      ;;
    *)
      echo "Invalid choice. Exiting."
      exit 1
      ;;
  esac
}

# Run main function
main
