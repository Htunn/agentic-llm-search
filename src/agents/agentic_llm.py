"""
Main Agentic LLM Agent that coordinates search and response generation
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
from dotenv import load_dotenv

from src import AgentResponse, SearchResult
from src.tools.search_tool import InternetSearchTool, SearchQueryOptimizer
from src.models.llm_models import OpenAIModel, HuggingFaceModel, AzureOpenAIModel, AgentModelOrchestrator
from src.agents.memory import ConversationMemory

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class AgenticLLMAgent:
    """
    Main agent that coordinates internet search and LLM response generation
    """
    
    def __init__(
        self, 
        model_name: str = "./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        model_provider: str = "huggingface",
        max_search_results: int = 5,
        enable_search: bool = True,
        max_memory: int = 10,
        enable_memory: bool = True
    ):
        self.model_name = model_name
        self.model_provider = model_provider
        self.max_search_results = max_search_results
        self.enable_search = enable_search
        self.enable_memory = enable_memory
        
        # Initialize components
        self.search_tool = InternetSearchTool(max_results=max_search_results)
        self.query_optimizer = SearchQueryOptimizer()
        
        # Initialize conversation memory
        self.memory = ConversationMemory(max_history=max_memory)
        
        # Initialize LLM model
        try:
            if self.model_provider.lower() == "huggingface":
                self.llm_model = HuggingFaceModel(model_name=model_name)
                logger.info(f"Initialized HuggingFace model: {model_name}")
            elif self.model_provider.lower() == "azure-openai":
                self.llm_model = AzureOpenAIModel(model_name=model_name)
                logger.info(f"Initialized Azure OpenAI model: {model_name}")
            else:
                self.llm_model = OpenAIModel(model_name=model_name)
                logger.info(f"Initialized OpenAI model: {model_name}")
            
            self.orchestrator = AgentModelOrchestrator(self.llm_model)
            logger.info(f"Initialized agent with model: {model_name} from provider: {model_provider}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {str(e)}")
            raise
    
    async def process_query(self, query: str, search_type: str = "web") -> AgentResponse:
        """
        Process a user query with optional internet search
        
        Args:
            query: User's question or request
            search_type: Type of search to perform ("web", "news")
            
        Returns:
            AgentResponse with answer and sources
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Add user message to memory
            if self.enable_memory:
                self.memory.add_user_message(query)
                
                # Check if this is a follow-up question
                is_follow_up = self.memory.detect_follow_up(query)
                logger.info(f"Query detected as follow-up: {is_follow_up}")
            else:
                is_follow_up = False
            
            search_results = []
            
            # Perform internet search if enabled
            if self.enable_search:
                # If it's a follow-up question and we have context, use a more specific search query
                if is_follow_up and self.memory.messages:
                    # Include previous exchanges to provide context for query optimization
                    context_for_search = self.memory.format_for_context(max_tokens=500)
                    optimized_query = self.query_optimizer.optimize_query(query, additional_context=context_for_search)
                    logger.info(f"Optimized follow-up search query: {optimized_query}")
                else:
                    # Standard query optimization
                    optimized_query = self.query_optimizer.optimize_query(query)
                    logger.info(f"Optimized search query: {optimized_query}")
                
                search_results = await self.search_tool.async_search(
                    optimized_query, 
                    search_type
                )
                
                logger.info(f"Found {len(search_results)} search results")
            
            # Generate response using LLM with conversation history if enabled
            if self.enable_memory:
                response = self.orchestrator.generate_research_response(
                    query, 
                    search_results,
                    conversation_history=self.memory.format_for_context() if is_follow_up else None,
                    is_follow_up=is_follow_up
                )
                
                # Add assistant response to memory
                self.memory.add_assistant_message(
                    response.answer, 
                    metadata={"sources": [s.url for s in response.sources]}
                )
            else:
                response = self.orchestrator.generate_research_response(query, search_results)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return AgentResponse(
                answer=f"I apologize, but I encountered an error while processing your request: {str(e)}",
                sources=[],
                query=query,
                timestamp=datetime.now(),
                model_used=self.model_name
            )
    
    async def process_shodan_query(self, query: str, limit: int = 5) -> AgentResponse:
        """
        Process a query specifically for Shodan search
        
        Args:
            query: Search query for Shodan
            limit: Maximum number of results to return
            
        Returns:
            AgentResponse with answer and sources
        """
        logger.info(f"Processing Shodan query: {query}")
        
        try:
            # Check if Shodan is available
            if not self.search_tool.shodan:
                return AgentResponse(
                    answer="Shodan search is not available. Please check that the SHODAN_API_KEY is set in your .env file.",
                    sources=[],
                    query=query,
                    timestamp=datetime.now(),
                    model_used=self.model_name
                )
            
            # Add user message to memory if enabled
            if self.enable_memory:
                self.memory.add_user_message(f"Shodan search: {query}")
            
            # Perform Shodan search
            search_results = await self.search_tool.async_shodan_search(query, limit)
            
            logger.info(f"Found {len(search_results)} Shodan search results")
            
            # Generate response using LLM
            response_prompt = (
                "You are a cybersecurity analyst working with Shodan data. "
                "Analyze the following Shodan search results and provide a detailed, "
                "informative summary with security insights. Be specific about patterns, "
                "vulnerabilities, and notable findings.\n\n"
                f"Search query: {query}\n"
            )
            
            # Generate response using LLM with conversation history if enabled
            if self.enable_memory:
                response = self.orchestrator.generate_research_response(
                    response_prompt, 
                    search_results,
                    conversation_history=self.memory.format_for_context(),
                    is_follow_up=False
                )
                
                # Add assistant response to memory
                self.memory.add_assistant_message(
                    response.answer, 
                    metadata={"sources": [s.url for s in response.sources]}
                )
            else:
                response = self.orchestrator.generate_research_response(response_prompt, search_results)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing Shodan query: {str(e)}")
            return AgentResponse(
                answer=f"I apologize, but I encountered an error while processing your Shodan request: {str(e)}",
                sources=[],
                query=query,
                timestamp=datetime.now(),
                model_used=self.model_name
            )
    
    async def process_shodan_host(self, ip: str) -> AgentResponse:
        """
        Process a Shodan host lookup request
        
        Args:
            ip: IP address to look up in Shodan
            
        Returns:
            AgentResponse with answer and sources
        """
        logger.info(f"Processing Shodan host lookup for IP: {ip}")
        
        try:
            # Check if Shodan is available
            if not self.search_tool.shodan:
                return AgentResponse(
                    answer="Shodan search is not available. Please check that the SHODAN_API_KEY is set in your .env file.",
                    sources=[],
                    query=f"Shodan host lookup: {ip}",
                    timestamp=datetime.now(),
                    model_used=self.model_name
                )
            
            # Add user message to memory if enabled
            if self.enable_memory:
                self.memory.add_user_message(f"Shodan host lookup: {ip}")
            
            # Perform Shodan host lookup
            host_result = await self.search_tool.async_shodan_host_lookup(ip)
            
            if not host_result:
                return AgentResponse(
                    answer=f"No Shodan information found for IP: {ip}",
                    sources=[],
                    query=f"Shodan host lookup: {ip}",
                    timestamp=datetime.now(),
                    model_used=self.model_name
                )
            
            logger.info(f"Found Shodan information for IP: {ip}")
            
            # Generate response using LLM
            response_prompt = (
                "You are a cybersecurity analyst working with Shodan data. "
                "Analyze the following Shodan host information and provide a detailed, "
                "informative summary with security insights. Be specific about services, "
                "potential vulnerabilities, and notable findings. Include specific port information "
                "and service details that might be relevant from a security perspective.\n\n"
                f"Host IP: {ip}\n"
            )
            
            # Generate response using LLM with conversation history if enabled
            if self.enable_memory:
                response = self.orchestrator.generate_research_response(
                    response_prompt, 
                    [host_result],  # Pass as a list even though it's a single result
                    conversation_history=self.memory.format_for_context(),
                    is_follow_up=False
                )
                
                # Add assistant response to memory
                self.memory.add_assistant_message(
                    response.answer, 
                    metadata={"sources": [host_result.url]}
                )
            else:
                response = self.orchestrator.generate_research_response(response_prompt, [host_result])
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing Shodan host lookup: {str(e)}")
            return AgentResponse(
                answer=f"I apologize, but I encountered an error while looking up the IP address in Shodan: {str(e)}",
                sources=[],
                query=f"Shodan host lookup: {ip}",
                timestamp=datetime.now(),
                model_used=self.model_name
            )
    
    async def process_fofa_query(self, query: str, limit: int = 5) -> AgentResponse:
        """
        Process a query specifically for FOFA search
        
        Args:
            query: Search query for FOFA
            limit: Maximum number of results to return
            
        Returns:
            AgentResponse with answer and sources
        """
        logger.info(f"Processing FOFA query: {query}")
        
        try:
            # Check if FOFA is available
            if not self.search_tool.fofa:
                return AgentResponse(
                    answer="FOFA search is not available. Please check that the FOFA_EMAIL and FOFA_API_KEY are set in your .env file.",
                    sources=[],
                    query=query,
                    timestamp=datetime.now(),
                    model_used=self.model_name
                )
            
            # Add user message to memory if enabled
            if self.enable_memory:
                self.memory.add_user_message(f"FOFA search: {query}")
            
            # Perform FOFA search
            search_results = await self.search_tool.async_fofa_search(query, limit)
            
            logger.info(f"Found {len(search_results)} FOFA search results")
            
            # Generate response using LLM
            response_prompt = (
                "You are a cybersecurity analyst working with FOFA data. "
                "FOFA is a search engine for internet-connected devices, similar to Shodan. "
                "Analyze the following FOFA search results and provide a detailed, "
                "informative summary with security insights. Be specific about patterns, "
                "technologies, server versions, and notable findings.\n\n"
                f"Search query: {query}\n"
            )
            
            # Generate response using LLM with conversation history if enabled
            if self.enable_memory:
                response = self.orchestrator.generate_research_response(
                    response_prompt, 
                    search_results,
                    conversation_history=self.memory.format_for_context(),
                    is_follow_up=False
                )
                
                # Add assistant response to memory
                self.memory.add_assistant_message(
                    response.answer, 
                    metadata={"sources": [s.url for s in search_results]}
                )
            else:
                response = self.orchestrator.generate_research_response(response_prompt, search_results)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing FOFA query: {str(e)}")
            return AgentResponse(
                answer=f"I apologize, but I encountered an error while processing your FOFA request: {str(e)}",
                sources=[],
                query=query,
                timestamp=datetime.now(),
                model_used=self.model_name
            )
    
    async def process_fofa_host(self, ip: str) -> AgentResponse:
        """
        Process a FOFA host lookup request
        
        Args:
            ip: IP address to look up in FOFA
            
        Returns:
            AgentResponse with answer and sources
        """
        logger.info(f"Processing FOFA host lookup for IP: {ip}")
        
        try:
            # Check if FOFA is available
            if not self.search_tool.fofa:
                return AgentResponse(
                    answer="FOFA search is not available. Please check that the FOFA_EMAIL and FOFA_API_KEY are set in your .env file.",
                    sources=[],
                    query=f"FOFA host lookup: {ip}",
                    timestamp=datetime.now(),
                    model_used=self.model_name
                )
            
            # Add user message to memory if enabled
            if self.enable_memory:
                self.memory.add_user_message(f"FOFA host lookup: {ip}")
            
            # Perform FOFA host lookup
            host_result = await self.search_tool.async_fofa_host_lookup(ip)
            
            if not host_result:
                return AgentResponse(
                    answer=f"No FOFA information found for IP: {ip}",
                    sources=[],
                    query=f"FOFA host lookup: {ip}",
                    timestamp=datetime.now(),
                    model_used=self.model_name
                )
            
            logger.info(f"Found FOFA information for IP: {ip}")
            
            # Generate response using LLM
            response_prompt = (
                "You are a cybersecurity analyst working with FOFA data. "
                "FOFA is a search engine for internet-connected devices, similar to Shodan. "
                "Analyze the following FOFA host information and provide a detailed, "
                "informative summary with security insights. Be specific about services, "
                "server types, and notable findings. Include specific port information "
                "and service details that might be relevant from a security perspective.\n\n"
                f"Host IP: {ip}\n"
            )
            
            # Generate response using LLM with conversation history if enabled
            if self.enable_memory:
                response = self.orchestrator.generate_research_response(
                    response_prompt, 
                    [host_result],  # Pass as a list even though it's a single result
                    conversation_history=self.memory.format_for_context(),
                    is_follow_up=False
                )
                
                # Add assistant response to memory
                self.memory.add_assistant_message(
                    response.answer, 
                    metadata={"sources": [host_result.url]}
                )
            else:
                response = self.orchestrator.generate_research_response(response_prompt, [host_result])
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing FOFA host lookup: {str(e)}")
            return AgentResponse(
                answer=f"I apologize, but I encountered an error while looking up the IP address in FOFA: {str(e)}",
                sources=[],
                query=f"FOFA host lookup: {ip}",
                timestamp=datetime.now(),
                model_used=self.model_name
            )
    
    def process_query_sync(self, query: str, search_type: str = "web") -> AgentResponse:
        """Synchronous version of process_query"""
        return asyncio.run(self.process_query(query, search_type))
    
    def process_shodan_query_sync(self, query: str, limit: int = 5) -> AgentResponse:
        """Synchronous version of process_shodan_query"""
        return asyncio.run(self.process_shodan_query(query, limit))
        
    def process_shodan_host_sync(self, ip: str) -> AgentResponse:
        """Synchronous version of process_shodan_host"""
        return asyncio.run(self.process_shodan_host(ip))
    
    def process_fofa_query_sync(self, query: str, limit: int = 5) -> AgentResponse:
        """Synchronous version of process_fofa_query"""
        return asyncio.run(self.process_fofa_query(query, limit))
        
    def process_fofa_host_sync(self, ip: str) -> AgentResponse:
        """Synchronous version of process_fofa_host"""
        return asyncio.run(self.process_fofa_host(ip))
    
    def set_search_enabled(self, enabled: bool):
        """Enable or disable internet search"""
        self.enable_search = enabled
        logger.info(f"Search {'enabled' if enabled else 'disabled'}")
    
    def set_memory_enabled(self, enabled: bool):
        """Enable or disable conversation memory"""
        self.enable_memory = enabled
        logger.info(f"Conversation memory {'enabled' if enabled else 'disabled'}")
    
    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            logger.info("Cleared conversation memory")
    
    def get_conversation_history(self):
        """Get conversation history"""
        if not self.enable_memory:
            logger.warning("Conversation memory is disabled")
            return []
        return self.memory.get_conversation_history()
    
    def update_model(self, model_name: str, model_provider: Optional[str] = None):
        """Update the LLM model"""
        try:
            # If model_provider not provided, keep the current one
            if model_provider:
                self.model_provider = model_provider
                
            if self.model_provider.lower() == "huggingface":
                self.llm_model = HuggingFaceModel(model_name=model_name)
                logger.info(f"Updated to HuggingFace model: {model_name}")
            elif self.model_provider.lower() == "azure-openai":
                self.llm_model = AzureOpenAIModel(model_name=model_name)
                logger.info(f"Updated to Azure OpenAI model: {model_name}")
            else:
                self.llm_model = OpenAIModel(model_name=model_name)
                logger.info(f"Updated to OpenAI model: {model_name}")
                
            self.orchestrator = AgentModelOrchestrator(self.llm_model)
            self.model_name = model_name
            logger.info(f"Updated model to: {model_name}")
        except Exception as e:
            logger.error(f"Failed to update model: {str(e)}")
            # Re-raise the exception to be handled by the caller
            raise

class AgentConfig:
    """Configuration class for the agent"""
    
    def __init__(self):
        self.model_name = os.getenv("DEFAULT_MODEL", "./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        self.model_provider = os.getenv("MODEL_PROVIDER", "huggingface") # "openai", "huggingface", or "azure-openai"
        
        # Parse numeric values safely by stripping any comments
        try:
            temp_str = os.getenv("TEMPERATURE", "0.7")
            self.temperature = float(temp_str.split('#')[0].strip())
        except (ValueError, AttributeError):
            self.temperature = 0.7
            
        try:
            tokens_str = os.getenv("MAX_TOKENS", "2000")
            self.max_tokens = int(tokens_str.split('#')[0].strip())
        except (ValueError, AttributeError):
            self.max_tokens = 2000
            
        try:
            search_results_str = os.getenv("MAX_SEARCH_RESULTS", "5")
            self.max_search_results = int(search_results_str.split('#')[0].strip())
        except (ValueError, AttributeError):
            self.max_search_results = 5
            
        self.search_engine = os.getenv("SEARCH_ENGINE", "duckduckgo")
        
        try:
            content_length_str = os.getenv("MAX_CONTENT_LENGTH", "2000")
            self.max_content_length = int(content_length_str.split('#')[0].strip())
        except (ValueError, AttributeError):
            self.max_content_length = 2000
        
        # Memory configuration
        memory_str = os.getenv("ENABLE_MEMORY", "true")
        self.enable_memory = memory_str.lower().split('#')[0].strip() == "true"
        
        try:
            max_memory_str = os.getenv("MAX_MEMORY", "10")
            self.max_memory = int(max_memory_str.split('#')[0].strip())
        except (ValueError, AttributeError):
            self.max_memory = 10
        
        # Azure OpenAI specific configuration
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        
        # FOFA configuration
        self.fofa_email = os.getenv("FOFA_EMAIL", "")
        self.fofa_api_key = os.getenv("FOFA_API_KEY", "")
        self.enable_fofa = bool(self.fofa_email and self.fofa_api_key)
        
        # Shodan configuration (legacy)
        self.shodan_api_key = os.getenv("SHODAN_API_KEY", "")
        self.enable_shodan = bool(self.shodan_api_key)
        
        self.debug = os.getenv("DEBUG", "False").lower() == "true"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "model_name": self.model_name,
            "model_provider": self.model_provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_search_results": self.max_search_results,
            "search_engine": self.search_engine,
            "max_content_length": self.max_content_length,
            "enable_memory": self.enable_memory,
            "max_memory": self.max_memory,
            "enable_fofa": self.enable_fofa,
            "enable_shodan": self.enable_shodan,
            "debug": self.debug
        }
