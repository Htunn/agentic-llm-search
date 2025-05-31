#!/usr/bin/env python3
"""
Simple script to fix the GGUF model loading issue
"""

import os

def fix_llm_models():
    # Path to the llm_models.py file
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm_models.py')
    
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove the problematic nested config dictionary
    content = content.replace(
        'gpu_config = {\n                        "gpu_layers": 32,  # Send most layers to GPU for M-series chips\n                        "config": {"max_batch_size": 512}  # Better batching for Metal\n                    }',
        'gpu_config = {\n                        "gpu_layers": 32,  # Send most layers to GPU for M-series chips\n                        "max_batch_size": 512  # Better batching for Metal\n                    }'
    )
    
    # Write the modified content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed the GGUF model loading issue in {file_path}")

if __name__ == "__main__":
    fix_llm_models()
