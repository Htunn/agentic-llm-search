#!/usr/bin/env python3
"""
Test script for Azure OpenAI integration
"""

import os
import time
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.models.llm_models import AzureOpenAIModel

def test_azure_openai_model():
    """Test the Azure OpenAI model"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Fixed values from the reference
        endpoint = "https://idaas-openai-poc.openai.azure.com/" 
        deployment_name = "IDaaS-OpenAI-GPT-35-Turbo-Service"
        model_name = "gpt-35-turbo"
        api_version = "2024-12-01-preview"  # Using the reference version
        
        # Get API key from environment
        api_key = os.getenv("AZURE_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        
        # Check for required configuration
        if not api_key:
            print("Error: AZURE_OPENAI_API_KEY or OPENAI_API_KEY is required")
            return
        
        print(f"\nTesting Azure OpenAI")
        print(f"Endpoint: {endpoint}")
        print(f"API Version: {api_version}")
        print(f"Model: {model_name}")
        print(f"Deployment: {deployment_name}")
        print("-" * 60)
        
        # Initialize model
        print("Initializing Azure OpenAI model...")
        start_time = time.time()
        
        model = AzureOpenAIModel(
            model_name=model_name,
            deployment_name=deployment_name,
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version
        )
        
        init_time = time.time() - start_time
        print(f"Model initialized in {init_time:.2f} seconds")
        
        # Test prompt
        prompt = "What is artificial intelligence?"
        print(f"\nGenerating response for: '{prompt}'")
        
        # Generate response
        start_time = time.time()
        response = model.generate_response(prompt)
        generation_time = time.time() - start_time
        
        print("\n" + "-" * 60)
        print("RESPONSE:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        print(f"Generated in {generation_time:.2f} seconds")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_azure_openai_model()
