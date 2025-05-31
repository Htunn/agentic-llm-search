#!/usr/bin/env python3
"""
Simple test script to verify the HuggingFace model is working
"""

import sys
import time
from dotenv import load_dotenv
from src.models.llm_models import HuggingFaceModel

def test_huggingface_model():
    """Test the HuggingFace TinyLlama model"""
    try:
        # Load environment variables
        load_dotenv()
        
        print("Loading HuggingFace TinyLlama model...")
        start_time = time.time()
        
        # Initialize model
        model = HuggingFaceModel(model_name="./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
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
        
        return True
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print("\nMake sure you have installed the required dependencies:")
        print("pip install torch transformers")
        return False

if __name__ == "__main__":
    success = test_huggingface_model()
    sys.exit(0 if success else 1)
