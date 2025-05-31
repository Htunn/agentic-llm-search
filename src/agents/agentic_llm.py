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
from src.models.llm_models import OpenAIModel, HuggingFaceModel, AgentModelOrchestrator

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
        enable_search: bool = True
    ):
        self.model_name = model_name
        self.model_provider = model_provider
        self.max_search_results = max_search_results
        self.enable_search = enable_search
        
        # Initialize components
        self.search_tool = InternetSearchTool(max_results=max_search_results)
        self.query_optimizer = SearchQueryOptimizer()
        
        # Initialize LLM model
        try:
            if self.model_provider.lower() == "huggingface":
                self.llm_model = HuggingFaceModel(model_name=model_name)
                logger.info(f"Initialized HuggingFace model: {model_name}")
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
            search_results = []
            
            # Perform internet search if enabled
            if self.enable_search:
                optimized_query = self.query_optimizer.optimize_query(query)
                logger.info(f"Optimized search query: {optimized_query}")
                
                search_results = await self.search_tool.async_search(
                    optimized_query, 
                    search_type
                )
                
                logger.info(f"Found {len(search_results)} search results")
            
            # Generate response using LLM
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
    
    def process_query_sync(self, query: str, search_type: str = "web") -> AgentResponse:
        """Synchronous version of process_query"""
        return asyncio.run(self.process_query(query, search_type))
    
    def set_search_enabled(self, enabled: bool):
        """Enable or disable internet search"""
        self.enable_search = enabled
        logger.info(f"Search {'enabled' if enabled else 'disabled'}")
    
    def update_model(self, model_name: str, model_provider: Optional[str] = None):
        """Update the LLM model"""
        try:
            # If model_provider not provided, keep the current one
            if model_provider:
                self.model_provider = model_provider
                
            if self.model_provider.lower() == "huggingface":
                self.llm_model = HuggingFaceModel(model_name=model_name)
                logger.info(f"Updated to HuggingFace model: {model_name}")
            else:
                self.llm_model = OpenAIModel(model_name=model_name)
                logger.info(f"Updated to OpenAI model: {model_name}")
                
            self.orchestrator = AgentModelOrchestrator(self.llm_model)
            self.model_name = model_name
            logger.info(f"Updated model to: {model_name}")
        except Exception as e:
            logger.error(f"Failed to update model: {str(e)}")
            raise

class AgentConfig:
    """Configuration class for the agent"""
    
    def __init__(self):
        self.model_name = os.getenv("DEFAULT_MODEL", "./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        self.model_provider = os.getenv("MODEL_PROVIDER", "huggingface") # "openai" or "huggingface"
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "2000"))
        self.max_search_results = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
        self.search_engine = os.getenv("SEARCH_ENGINE", "duckduckgo")
        self.max_content_length = int(os.getenv("MAX_CONTENT_LENGTH", "2000"))
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
            "debug": self.debug
        }
