"""
LLM Models wrapper for different providers
Supports OpenAI, HuggingFace, and other providers
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import openai
from openai import OpenAI

# Import our environment setup utilities
from src.models.env_setup import setup_huggingface_env

# Import HuggingFace and CTransformers components conditionally to avoid errors if not installed
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    HUGGINGFACE_AVAILABLE = True
    
    # Try importing ctransformers for GGUF support
    try:
        from ctransformers import AutoModelForCausalLM as CTModelForCausalLM
        CTRANSFORMERS_AVAILABLE = True
    except ImportError:
        CTRANSFORMERS_AVAILABLE = False
        
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    CTRANSFORMERS_AVAILABLE = False

# Configure environment for HuggingFace (handles hf_transfer check)
HF_TRANSFER_AVAILABLE = setup_huggingface_env()

# Set up GPU environment
from src.models.env_setup import setup_gpu_environment
DEVICE = setup_gpu_environment()

from src import AgentResponse, SearchResult

logger = logging.getLogger(__name__)

class LLMModel:
    """Base class for LLM models"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 2000):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate response from the model"""
        raise NotImplementedError

class HuggingFaceModel(LLMModel):
    """HuggingFace model wrapper for local LLMs"""
    

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

    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        # Generate response using HuggingFace model
        try:
            # Format prompt based on model's expected format
            system_prompt = self._get_system_prompt()
            
            if context:
                full_prompt = f"{system_prompt}\n\nContext: {context}\n\nUser: {prompt}\n\nAssistant:"
            else:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            
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
                # For GGUF models using ctransformers, known issue with 'tolist' attribute
                # Just use a predefined answer for now as a workaround
                try:
                    # Create a basic summarization from the search results
                    generated_text = full_prompt
                    
                    # Add a default generation for now - in the future, this could be fixed with proper ctransformers integration
                    if context and "search results" in context:
                        # Pull the titles from the search results for a basic response
                        sources = []
                        try:
                            lines = context.split("\n")
                            for line in lines:
                                if line.startswith("[") and "] Title:" in line:
                                    sources.append(line)
                            
                            generated_text += "\n\nBased on the search results, I can provide the following information:\n\n"
                            generated_text += "I found several relevant sources that match your query. "
                            generated_text += "Please check the list of sources below for more information.\n\n"
                            
                            for i, source in enumerate(sources, 1):
                                generated_text += f"Source {i}: {source.split('] Title:')[1].strip()}\n"
                        except Exception as parse_error:
                            logger.warning(f"Error parsing context: {str(parse_error)}")
                    else:
                        # Generic response if no context
                        generated_text += "\n\nI apologize, but I don't have enough information to provide a detailed answer to your question."
                    
                except Exception as e:
                    # If all fails, use a very simple fallback
                    generated_text = full_prompt + "\nI apologize, but I'm having trouble processing your request right now."
                    logger.error(f"Final error in text generation: {str(e)}")
            
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
    def _get_system_prompt(self) -> str:
        """Get system prompt for the model"""
        return """You are an intelligent research assistant that provides accurate, well-researched answers with proper citations. 

When provided with search results, you should:
1. Analyze the information from multiple sources
2. Synthesize a comprehensive answer
3. Include proper citations with [1], [2], etc.
4. Mention if information is recent or dated
5. Acknowledge any limitations or uncertainties

Format your response clearly with:
- A direct answer to the question
- Supporting evidence from sources
- Proper citations
- Source reliability assessment when relevant"""

class OpenAIModel(LLMModel):
    """OpenAI GPT model wrapper"""
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        super().__init__(model_name, api_key, **kwargs)
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
            
        self.client = OpenAI(api_key=self.api_key)
        
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate response using OpenAI GPT"""
        try:
            from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
            
            messages = []
            # Add system message
            messages.append(ChatCompletionSystemMessageParam(
                role="system",
                content=self._get_system_prompt()
            ))
            
            # Add context if provided
            if context:
                messages.append(ChatCompletionSystemMessageParam(
                    role="system",
                    content=f"Context: {context}"
                ))
            
            # Add user message
            messages.append(ChatCompletionUserMessageParam(
                role="user",
                content=prompt
            ))
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            content = response.choices[0].message.content
            return content if content is not None else "No response generated"
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the model"""
        return """You are an intelligent research assistant that provides accurate, well-researched answers with proper citations. 

When provided with search results, you should:
1. Analyze the information from multiple sources
2. Synthesize a comprehensive answer
3. Include proper citations with [1], [2], etc.
4. Mention if information is recent or dated
5. Acknowledge any limitations or uncertainties

Format your response clearly with:
- A direct answer to the question
- Supporting evidence from sources
- Proper citations
- Source reliability assessment when relevant"""

class AzureOpenAIModel(OpenAIModel):
    """Azure OpenAI model wrapper"""
    
    def __init__(self, 
                 model_name: str = "gpt-35-turbo", 
                 deployment_name: Optional[str] = None,
                 api_key: Optional[str] = None, 
                 api_version: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 **kwargs):
        # Use environment variables if parameters not provided
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        # If deployment name not provided, try to get from env or use model_name
        deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT", model_name)
        
        # Initialize parent with the API key
        super().__init__(model_name, api_key, **kwargs)
        
        # Store the deployment name separately from the model name
        self.deployment_name = deployment_name
        
        if not endpoint:
            raise ValueError("Azure OpenAI endpoint is required")
        
        # Ensure the endpoint has the correct format (https://...)    
        if not endpoint.startswith("https://"):
            endpoint = f"https://{endpoint}"
        
        if not endpoint.endswith("/"):
            endpoint = f"{endpoint}/"
            
        # Create Azure OpenAI client instead of standard OpenAI client
        try:
            from openai import AzureOpenAI
            
            # Log the actual values being used for debugging
            logger.info(f"Initializing Azure OpenAI with API version: {api_version}, Endpoint: {endpoint}, Deployment: {deployment_name}")
            
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )
            logger.info(f"Successfully initialized Azure OpenAI client with endpoint {endpoint}")
            logger.info(f"Using deployment: {self.deployment_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            raise
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate response using Azure OpenAI"""
        try:
            from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
            
            messages = []
            # Add system message
            messages.append(ChatCompletionSystemMessageParam(
                role="system",
                content=self._get_system_prompt()
            ))
            
            # Add context if provided
            if context:
                messages.append(ChatCompletionSystemMessageParam(
                    role="system",
                    content=f"Context: {context}"
                ))
            
            # Add user message
            messages.append(ChatCompletionUserMessageParam(
                role="user",
                content=prompt
            ))
            
            logger.info(f"Sending request to Azure OpenAI with deployment: {self.deployment_name}")
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,  # Use the deployment name for Azure
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            content = response.choices[0].message.content
            return content if content is not None else "No response generated"
            
        except Exception as e:
            error_details = str(e)
            # Include more detailed debugging information
            client_info = f"Endpoint: {getattr(self.client, 'azure_endpoint', 'unknown')}, " \
                          f"API Version: {getattr(self.client, 'api_version', 'unknown')}, " \
                          f"Model/Deployment: {self.deployment_name}"
            
            logger.error(f"Azure OpenAI API error: {error_details}")
            logger.error(f"Client details: {client_info}")
            
            # Return a more informative error message
            return f"Error generating response: {error_details}\n\nThis might be due to:\n" \
                   f"1. Incorrect deployment name (current: {self.deployment_name})\n" \
                   f"2. Invalid API key or endpoint\n" \
                   f"3. Incompatible API version\n" \
                   f"Check the configuration in your .env file."

class AgentModelOrchestrator:
    """Orchestrates the LLM model responses with search results"""
    
    def __init__(self, llm_model: LLMModel):
        self.llm_model = llm_model
        
    def generate_research_response(self, query: str, search_results: List[SearchResult]) -> AgentResponse:
        """
        Generate a research response using the LLM model and search results
        
        Args:
            query: User's original query
            search_results: List of search results
            
        Returns:
            AgentResponse with answer and sources
        """
        try:
            # Format search results for context
            context = self._format_search_results(search_results)
            
            # Generate response
            answer = self.llm_model.generate_response(query, context)
            
            # Create response object
            response = AgentResponse(
                answer=answer,
                sources=search_results,
                query=query,
                timestamp=datetime.now(),
                model_used=self.llm_model.model_name
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating research response: {str(e)}")
            return AgentResponse(
                answer=f"I apologize, but I encountered an error while processing your request: {str(e)}",
                sources=[],
                query=query,
                timestamp=datetime.now(),
                model_used=self.llm_model.model_name
            )
    
    def _format_search_results(self, search_results: List[SearchResult]) -> str:
        """
        Format search results for model context
        
        Args:
            search_results: List of search results
            
        Returns:
            Formatted context string with search results
        """
        if not search_results:
            return ""
            
        context = "Here are some search results that might help answer the query:\n\n"
        
        for i, result in enumerate(search_results, 1):
            context += f"[{i}] Title: {result.title}\n"
            context += f"URL: {result.url}\n"
            context += f"Content: {result.content}\n\n"
            
        context += "Please use these search results to provide a comprehensive answer with proper citations."
        
        return context
