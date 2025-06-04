#!/usr/bin/env python3
"""
AgenticLLM class extended with Criminal IP integration

This module wraps the AgenticLLM class and adds Criminal IP methods.
"""

import os
import logging
import asyncio
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Import the original agent
from src.agents.agentic_llm import AgenticLLMAgent, AgentConfig
from src import AgentResponse
from src.models.add_criminalip import add_criminalip_methods, update_agent_config

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Update the AgentConfig class with Criminal IP configuration
AgentConfig = update_agent_config(AgentConfig)

# Extend AgenticLLMAgent with Criminal IP methods
AgenticLLMAgent = add_criminalip_methods(AgenticLLMAgent)

class AgenticLLM:
    """
    Main interface for the Agentic LLM agent
    
    This class wraps the AgenticLLMAgent to provide a simplified interface
    and ensures that the Criminal IP methods are available.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_provider: Optional[str] = None,
        max_search_results: int = 5,
    ):
        """
        Initialize the Agentic LLM agent
        
        Args:
            model_name: Name of the model to use (defaults to .env value)
            model_provider: Provider of the model (defaults to .env value)
            max_search_results: Maximum number of search results to return
        """
        # Use provided values or defaults from .env
        self.model_name = model_name or os.getenv("DEFAULT_MODEL", "./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        self.model_provider = model_provider or os.getenv("MODEL_PROVIDER", "huggingface")
        
        # Initialize the agent
        self.agent = AgenticLLMAgent(
            model_name=self.model_name,
            model_provider=self.model_provider,
            max_search_results=max_search_results,
        )
        
        # Get configuration settings from .env
        self.config = AgentConfig()
        
        # Log the available search tools
        self._log_available_tools()
    
    def _log_available_tools(self):
        """Log the available search tools based on configuration"""
        logger.info(f"Model: {self.model_name} (Provider: {self.model_provider})")
        
        available_tools = []
        
        if hasattr(self.agent.search_tool, "criminalip") and self.agent.search_tool.criminalip is not None:
            available_tools.append("Criminal IP")
            
        if hasattr(self.agent.search_tool, "shodan") and self.agent.search_tool.shodan is not None:
            available_tools.append("Shodan")
            
        if hasattr(self.agent.search_tool, "fofa") and self.agent.search_tool.fofa is not None:
            available_tools.append("FOFA")
        
        if available_tools:
            logger.info(f"Available security search tools: {', '.join(available_tools)}")
        else:
            logger.info("No security search tools available. Add API keys to .env file to enable.")
    
    def search_and_respond(self, query: str, search_type: str = "web") -> AgentResponse:
        """
        Process a user query with internet search and LLM response
        
        Args:
            query: User's question or query
            search_type: Type of search to perform ("web", "news")
            
        Returns:
            AgentResponse with answer and sources
        """
        return self.agent.process_query_sync(query, search_type)
    
    def process_criminalip_search(self, query: str, limit: int = 5) -> AgentResponse:
        """
        Search Criminal IP for the given query
        
        Args:
            query: Search query for Criminal IP
            limit: Maximum number of results
            
        Returns:
            AgentResponse with answer and sources
        """
        return self.agent.process_criminalip_query_sync(query, limit)
    
    def process_criminalip_ip(self, ip: str) -> AgentResponse:
        """
        Look up information about an IP address in Criminal IP
        
        Args:
            ip: IP address to look up
            
        Returns:
            AgentResponse with answer and sources
        """
        return self.agent.process_criminalip_host_sync(ip)
    
    def process_shodan_search(self, query: str, limit: int = 5) -> AgentResponse:
        """
        Search Shodan for the given query
        
        Args:
            query: Search query for Shodan
            limit: Maximum number of results
            
        Returns:
            AgentResponse with answer and sources
        """
        return self.agent.process_shodan_query_sync(query, limit)
    
    def process_shodan_ip(self, ip: str) -> AgentResponse:
        """
        Look up information about an IP address in Shodan
        
        Args:
            ip: IP address to look up
            
        Returns:
            AgentResponse with answer and sources
        """
        return self.agent.process_shodan_host_sync(ip)
    
    def process_fofa_search(self, query: str, limit: int = 5) -> AgentResponse:
        """
        Search FOFA for the given query
        
        Args:
            query: Search query for FOFA
            limit: Maximum number of results
            
        Returns:
            AgentResponse with answer and sources
        """
        return self.agent.process_fofa_query_sync(query, limit)
    
    def process_fofa_ip(self, ip: str) -> AgentResponse:
        """
        Look up information about an IP address in FOFA
        
        Args:
            ip: IP address to look up
            
        Returns:
            AgentResponse with answer and sources
        """
        return self.agent.process_fofa_host_sync(ip)
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.agent.clear_memory()
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration"""
        return self.config.to_dict()
    
async def main():
    """Example usage of the AgenticLLM class with Criminal IP"""
    agent = AgenticLLM()
    
    # Test Criminal IP IP lookup
    ip = "8.8.8.8"  # Example IP
    print(f"\n[Testing Criminal IP lookup for {ip}]")
    response = agent.process_criminalip_ip(ip)
    print(f"Answer: {response.answer}")
    
    # Test Criminal IP search
    query = "apache 2.4"  # Example query
    print(f"\n[Testing Criminal IP search for '{query}']")
    response = agent.process_criminalip_search(query)
    print(f"Answer: {response.answer}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the example
    asyncio.run(main())
