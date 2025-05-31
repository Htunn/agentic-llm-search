#!/usr/bin/env python3
"""
Final fix script for the test_agentic_search.py issues
This script creates a complete fixed version of the llm_models.py file
"""

import os
import sys

def create_fixed_llm_models():
    # Path to the llm_models.py file
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm_models.py')
    
    # Backup the original file
    backup_path = file_path + '.bak'
    try:
        with open(file_path, 'r') as f:
            original_content = f.read()
            
        # Create backup
        with open(backup_path, 'w') as f:
            f.write(original_content)
            
        print(f"Created backup of the original file at {backup_path}")
    except Exception as e:
        print(f"Error creating backup: {str(e)}")
        return False
        
    # The fixed content for the HuggingFaceModel class initialization method
    fixed_content = """
    def __init__(self, model_name: str = "./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", **kwargs):
        # HuggingFace models don't need an API key for inference
        super().__init__(model_name, api_key=None, **kwargs)
        
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("HuggingFace transformers and torch libraries are required but not installed.")
            
        logger.info(f"Loading HuggingFace model: {model_name}")
        
        try:
            # Check if the model is a GGUF file
            is_gguf = model_name.endswith('.gguf')
            
            if is_gguf and CTRANSFORMERS_AVAILABLE:
                # Use CTransformers for GGUF models
                logger.info("Using CTransformers for GGUF model")
                
                # Check for Apple Silicon to use Metal GPU acceleration
                import platform
                is_apple_silicon = (platform.system() == "Darwin" and 
                                   platform.machine().startswith(('arm', 'aarch')))
                
                # Set appropriate GPU configuration based on platform
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
                    )
                
                # Load tokenizer from Hugging Face
                self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                
            else:
                # Use standard HuggingFace transformers
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Check for Apple Silicon to use MPS (Metal Performance Shaders)
                import platform
                is_apple_silicon = (platform.system() == "Darwin" and 
                                  platform.machine().startswith(('arm', 'aarch')))
                
                # Set the appropriate device for accelerated computing
                if is_apple_silicon and torch.backends.mps.is_available():
                    logger.info("Detected Apple Silicon (M-series) - enabling MPS acceleration")
                    compute_dtype = torch.float16
                    device_map = "mps"
                else:
                    compute_dtype = torch.float16
                    device_map = "auto"
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=compute_dtype,
                    low_cpu_mem_usage=True,
                    device_map=device_map
                )
                
                # Create text generation pipeline with appropriate device
                self.generator = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_length=self.max_tokens,
                    temperature=self.temperature,
                    top_p=0.95,
                    do_sample=True,
                    device_map=device_map
                )
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {str(e)}")
            raise
"""

    # The fixed content for the generate_response method
    fixed_generate_response = """
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        # Generate response using HuggingFace model
        try:
            # Format prompt based on model's expected format
            system_prompt = self._get_system_prompt()
            
            if context:
                full_prompt = f"{system_prompt}\\n\\nContext: {context}\\n\\nUser: {prompt}\\n\\nAssistant:"
            else:
                full_prompt = f"{system_prompt}\\n\\nUser: {prompt}\\n\\nAssistant:"
            
            # Different handling for GGUF models vs standard HF models
            if hasattr(self, 'generator'):
                # Use the pipeline for standard HF models
                response = self.generator(
                    full_prompt, 
                    max_length=len(self.tokenizer(full_prompt).input_ids) + self.max_tokens,
                    num_return_sequences=1
                )
                
                # Extract generated text from pipeline output
                if isinstance(response, list) and response:
                    if isinstance(response[0], dict) and 'generated_text' in response[0]:
                        generated_text = response[0]['generated_text']
                    else:
                        generated_text = str(response[0])
                else:
                    generated_text = str(response)
            else:
                # For GGUF models using ctransformers
                try:
                    # Try simple string call style
                    generated_text = self.model(full_prompt)
                except Exception as e:
                    logger.warning(f"Error generating with simple call: {str(e)}")
                    try:
                        # Try using specific generation parameters
                        generated_text = self.model(
                            full_prompt,
                            max_tokens=self.max_tokens,
                            temperature=float(self.temperature)
                        )
                    except Exception as e2:
                        logger.warning(f"Error generating with parameters: {str(e2)}")
                        # Last resort
                        from ctransformers import AutoConfig
                        config = AutoConfig.from_pretrained(self.model_name, context_length=2048)
                        from ctransformers import AutoModelForCausalLM as CTModelForCausalLM_direct
                        model = CTModelForCausalLM_direct.from_pretrained(self.model_name, config=config)
                        generated_text = model(full_prompt)
            
            # Extract just the assistant's response
            if "Assistant:" in generated_text:
                assistant_response = generated_text.split("Assistant:")[-1].strip()
            else:
                # If the model didn't follow the format, return the generated text after the prompt
                assistant_response = generated_text[len(full_prompt):].strip()
                
            # If empty or too short, return a fallback message
            if len(assistant_response) < 5:
                assistant_response = "I apologize, but I wasn't able to generate a proper response. Please try again with a different query."
                
            return assistant_response
            
        except Exception as e:
            logger.error(f"HuggingFace model error: {str(e)}")
            return f"Error generating response: {str(e)}"
"""

    try:
        # Replace the original init method in the file
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Find the HuggingFaceModel class definition and __init__ method
        class_start = content.find("class HuggingFaceModel(LLMModel):")
        if class_start == -1:
            print("Could not find HuggingFaceModel class definition")
            return False
            
        # Find the end of __init__ method (next def after __init__)
        init_start = content.find("    def __init__", class_start)
        if init_start == -1:
            print("Could not find __init__ method")
            return False
            
        next_def_start = content.find("    def ", init_start + 10)
        if next_def_start == -1:
            print("Could not find next method after __init__")
            return False
            
        # Replace the __init__ method
        content = content[:init_start] + fixed_content + content[next_def_start:]
        
        # Find the generate_response method
        gen_start = content.find("    def generate_response", class_start)
        if gen_start == -1:
            print("Could not find generate_response method")
            return False
            
        next_def_after_gen = content.find("    def ", gen_start + 10)
        if next_def_after_gen == -1:
            print("Could not find next method after generate_response")
            # If we didn't find another def, look for the next class
            next_def_after_gen = content.find("class ", gen_start + 10)
            if next_def_after_gen == -1:
                print("Could not find next class after generate_response")
                return False
        
        # Replace the generate_response method
        content = content[:gen_start] + fixed_generate_response + content[next_def_after_gen:]
        
        # Write the fixed content
        with open(file_path, 'w') as f:
            f.write(content)
            
        print(f"Successfully fixed {file_path}")
        return True
    
    except Exception as e:
        print(f"Error fixing file: {str(e)}")
        # Try to restore from backup
        try:
            if os.path.exists(backup_path):
                with open(backup_path, 'r') as f:
                    original = f.read()
                
                with open(file_path, 'w') as f:
                    f.write(original)
                
                print("Restored original from backup")
        except Exception as restore_error:
            print(f"Error restoring backup: {str(restore_error)}")
            
        return False

if __name__ == "__main__":
    if create_fixed_llm_models():
        print("Successfully fixed llm_models.py - test_agentic_search.py should now work correctly")
    else:
        print("Failed to fix llm_models.py")
