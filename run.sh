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
  echo "Choose a model provider:"
  echo "1. Local TinyLlama (Default, no API key required)"
  echo "2. Azure OpenAI (Requires configuration)"
  read -p "Enter model provider (1-2): " model_choice
  
  # Set model provider based on choice
  if [ "$model_choice" = "2" ]; then
    echo "Using Azure OpenAI as model provider"
    export MODEL_PROVIDER=azure-openai
    export DEFAULT_MODEL=gpt-35-turbo
    
    # Ensure Azure OpenAI configuration is set
    if [ -z "$AZURE_OPENAI_API_KEY" ] || [ -z "$AZURE_OPENAI_ENDPOINT" ]; then
      echo "Azure OpenAI configuration not found in environment"
      echo "Using values from .env file if available"
    fi
  else
    echo "Using local TinyLlama as model provider"
    export MODEL_PROVIDER=huggingface
    export DEFAULT_MODEL=./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
  fi
  
  # Check for Criminal IP configuration without setting up
  if [ -n "$CRIMINAL_IP_API_KEY" ]; then
    echo "Criminal IP API key found in environment."
  fi
  
  echo ""
  echo "Choose an interface:"
  echo "1. Command Line Interface"
  echo "2. Web Interface (Streamlit)"
  echo "3. Criminal IP CLI (Cybersecurity Search)"
  read -p "Enter your choice (1-3): " interface_choice
  
  case $interface_choice in
    1)
      echo "Starting command line interface..."
      python3 test_agentic_search.py
      ;;
    2)
      echo "Starting web interface with Streamlit..."
      streamlit run app.py
      ;;
    3)
      if [ -z "$CRIMINAL_IP_API_KEY" ]; then
        echo "[Error] Criminal IP API key not found in environment."
        echo "Please set the CRIMINAL_IP_API_KEY in your .env file or environment."
        echo "For help setting up the API key, visit: https://www.criminalip.io/developer"
        exit 1
      fi
      
      echo "Criminal IP Cybersecurity Search"
      echo "---------------------------------"
      echo "Choose an operation:"
      echo "1. IP Address Lookup"
      echo "2. Domain Analysis"
      echo "3. Asset Search"
      echo "4. Banner Search"
      echo "5. Account Information"
      echo "6. Help / All Commands"
      read -p "Enter your choice (1-6): " criminalip_choice
      
      case $criminalip_choice in
        1)
          read -p "Enter an IP address to analyze: " ip_address
          python3 criminalip_cli.py ip "$ip_address"
          ;;
        2)
          read -p "Enter a domain to analyze: " domain
          python3 criminalip_cli.py domain "$domain"
          ;;
        3)
          read -p "Enter a search query: " search_query
          python3 criminalip_cli.py search "$search_query"
          ;;
        4)
          read -p "Enter a banner search query: " banner_query
          python3 criminalip_cli.py banner "$banner_query"
          ;;
        5)
          python3 criminalip_cli.py info
          ;;
        6|*)
          python3 criminalip_cli.py help
          ;;
      esac
      ;;
    *)
      echo "Invalid choice. Exiting."
      exit 1
      ;;
  esac
}

# Run main function
main
