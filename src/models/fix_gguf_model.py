#!/usr/bin/env python3
"""
Fix script for GGUF model loading issues
This script modifies the CTModelForCausalLM.from_pretrained call to fix the config error
"""

import os
import re
import sys

def fix_llm_models_file():
    """Fix the CTModelForCausalLM.from_pretrained configuration in llm_models.py"""
    
    models_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_models.py")
    
    if not os.path.exists(models_file_path):
        print(f"Error: Could not find llm_models.py at {models_file_path}")
        return False
    
    # Read file content
    with open(models_file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match the problematic code section with gpu_config containing 'config' parameter
    pattern = r"""gpu_config = \{[^}]*"config":[^}]*\}[^}]*\}[^}]*\}"""
    
    # Check if the pattern exists
    if not re.search(pattern, content):
        print("Could not find the exact pattern to replace")
        
        # Fallback pattern to locate the general area
        apple_silicon_pattern = r"# Set appropriate GPU configuration based on platform.*?gpu_config = \{.*?\}"
        
        # Try with a more flexible search
        match = re.search(apple_silicon_pattern, content, re.DOTALL)
        if match:
            print("Found approximate pattern match, attempting fix...")
            
            # Replace the entire section with fixed code
            old_section = match.group(0)
            new_section = """                # Set appropriate GPU configuration based on platform
                if is_apple_silicon:
                    logger.info("Detected Apple Silicon (M-series) - enabling Metal GPU acceleration")
                    # Set GPU layers for M-series chips
                    self.model = CTModelForCausalLM.from_pretrained(
                        model_name,
                        model_type="llama",
                        hf=True,  # Use Hugging Face format
                        context_length=4096,  # Increased context length (default is 2048)
                        gpu_layers=32  # Send most layers to GPU for M-series chips
                    )
                else:
                    logger.info("Using CPU acceleration only")
                    self.model = CTModelForCausalLM.from_pretrained(
                        model_name,
                        model_type="llama",
                        hf=True,  # Use Hugging Face format
                        context_length=4096,  # Increased context length (default is 2048)
                        gpu_layers=0  # CPU only
                    )"""
            
            # Continue checking for the CTModelForCausalLM.from_pretrained part
            ctmodel_pattern = r"self\.model = CTModelForCausalLM\.from_pretrained\([^)]*(\*\*gpu_config)[^)]*\)"
            match = re.search(ctmodel_pattern, content, re.DOTALL)
            
            if match:
                old_from_pretrained = match.group(0)
                new_from_pretrained = "# Placeholder for the fix - this will be replaced"
                content = content.replace(old_section, new_section)
                
                # Save modified file
                with open(models_file_path, 'w') as f:
                    f.write(content)
                    
                print("Fixed llm_models.py file")
                return True
            else:
                print("Could not find CTModelForCausalLM.from_pretrained with **gpu_config")
                return False
        else:
            print("Could not find appropriate section to replace")
            return False

if __name__ == "__main__":
    if fix_llm_models_file():
        print("Successfully fixed the GGUF model loading issue")
    else:
        print("Failed to fix the GGUF model loading issue")
