#!/usr/bin/env python3
"""
Debug script for Azure OpenAI integration
"""

import os
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Print out all Azure OpenAI related environment variables
def print_azure_config():
    print("\n=== Azure OpenAI Configuration ===")
    print(f"AZURE_OPENAI_API_KEY: {'*****' + os.getenv('AZURE_OPENAI_API_KEY')[-4:] if os.getenv('AZURE_OPENAI_API_KEY') else 'Not set'}")
    print(f"AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT', 'Not set')}")
    print(f"AZURE_OPENAI_API_VERSION: {os.getenv('AZURE_OPENAI_API_VERSION', 'Not set')}")
    print(f"AZURE_OPENAI_DEPLOYMENT: {os.getenv('AZURE_OPENAI_DEPLOYMENT', 'Not set')}")
    print(f"DEFAULT_MODEL: {os.getenv('DEFAULT_MODEL', 'Not set')}")
    print(f"MODEL_PROVIDER: {os.getenv('MODEL_PROVIDER', 'Not set')}")
    print("\n")

def test_azure_openai():
    # Import after environment variables are loaded
    from src.models.llm_models import AzureOpenAIModel
    
    print("\n=== Testing Azure OpenAI Integration ===")
    
    # Create the model
    try:
        model = AzureOpenAIModel()
        print(f"✓ Successfully initialized Azure OpenAI model")
        print(f"  - Model name: {model.model_name}")
        print(f"  - Deployment name: {model.deployment_name}")
        
        # Test a simple completion
        print("\n=== Testing model completion ===")
        response = model.generate_response("Hello, world! Please respond with a simple greeting.")
        print(f"\nResponse from Azure OpenAI: {response}\n")
        
    except Exception as e:
        print(f"✗ Error initializing Azure OpenAI model: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    print_azure_config()
    success = test_azure_openai()
    
    if success:
        print("\n✓ Azure OpenAI integration test completed successfully!")
    else:
        print("\n✗ Azure OpenAI integration test failed. Please check the error messages above.")
