"""
This file adds the Criminal IP integration methods to the AgenticLLM class.
"""

import os
import logging
from typing import Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

# Add Criminal IP configuration to AgentConfig class
def update_agent_config(config_class):
    """Update the AgentConfig class with Criminal IP configuration"""
    # Add Criminal IP configuration
    original_init = config_class.__init__
    
    def new_init(self):
        original_init(self)
        # Add Criminal IP configuration
        self.criminalip_api_key = os.getenv("CRIMINAL_IP_API_KEY", "")
        self.enable_criminalip = bool(self.criminalip_api_key)
    
    config_class.__init__ = new_init
    
    # Update to_dict method
    original_to_dict = config_class.to_dict
    
    def new_to_dict(self) -> Dict[str, Any]:
        config_dict = original_to_dict(self)
        config_dict["enable_criminalip"] = self.enable_criminalip
        return config_dict
    
    config_class.to_dict = new_to_dict
    
    return config_class

# Add Criminal IP methods to AgenticLLM class
def add_criminalip_methods(agent_class):
    """Add Criminal IP methods to the AgenticLLM class"""
    
    # Add process_criminalip_query method
    async def process_criminalip_query(self, query: str, limit: int = 5):
        """
        Process a Criminal IP search query
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            AgentResponse with answer and sources
        """
        from datetime import datetime
        from src import AgentResponse
        
        logger.info(f"Processing Criminal IP search for: {query}")
        
        try:
            # Check if Criminal IP is available
            if not self.search_tool.criminalip:
                return AgentResponse(
                    answer="Criminal IP search is not available. Please check that the CRIMINAL_IP_API_KEY is set in your .env file.",
                    sources=[],
                    query=query,
                    timestamp=datetime.now(),
                    model_used=self.config.model_name
                )
            
            # Add user message to memory if enabled
            if self.config.enable_memory:
                self.memory.add_user_message(f"Criminal IP search: {query}")
            
            # Perform Criminal IP search
            search_results = await self.search_tool.async_criminalip_search(query, limit)
            
            if not search_results:
                return AgentResponse(
                    answer=f"No Criminal IP results found for '{query}'. Try a different search term or check that your API key is valid.",
                    sources=[],
                    query=query,
                    timestamp=datetime.now(),
                    model_used=self.config.model_name
                )
            
            logger.info(f"Found {len(search_results)} Criminal IP search results")
            
            # Generate response using LLM
            response_prompt = (
                "You are a cybersecurity analyst working with Criminal IP data. "
                "Criminal IP is a search engine for security information about internet assets. "
                "Analyze the following Criminal IP search results and provide a detailed, "
                "informative summary with security insights. Be specific about security issues, "
                "potential vulnerabilities, and notable findings.\n\n"
                f"Search query: {query}\n"
            )
            
            # Generate response using LLM with conversation history if enabled
            if self.config.enable_memory:
                response = self.orchestrator.generate_research_response(
                    response_prompt, 
                    search_results,
                    conversation_history=self._get_conversation_history(),
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
            logger.error(f"Error processing Criminal IP search: {str(e)}")
            return AgentResponse(
                answer=f"I apologize, but I encountered an error while processing your Criminal IP request: {str(e)}",
                sources=[],
                query=query,
                timestamp=datetime.now(),
                model_used=self.config.model_name
            )
    
    # Add process_criminalip_host method
    async def process_criminalip_host(self, ip: str):
        """
        Process a Criminal IP host lookup request
        
        Args:
            ip: IP address to look up
            
        Returns:
            AgentResponse with answer and sources
        """
        from datetime import datetime
        from src import AgentResponse
        
        logger.info(f"Processing Criminal IP lookup for IP: {ip}")
        
        try:
            # Check if Criminal IP is available
            if not self.search_tool.criminalip:
                return AgentResponse(
                    answer="Criminal IP search is not available. Please check that the CRIMINAL_IP_API_KEY is set in your .env file.",
                    sources=[],
                    query=f"Criminal IP lookup: {ip}",
                    timestamp=datetime.now(),
                    model_used=self.config.model_name
                )
            
            # Add user message to memory if enabled
            if self.config.enable_memory:
                self.memory.add_user_message(f"Criminal IP lookup: {ip}")
            
            # Get host information from Criminal IP
            host_result = await self.search_tool.async_criminalip_ip_lookup(ip)
            
            if not host_result:
                return AgentResponse(
                    answer=f"No Criminal IP information found for IP {ip}. This may not be a public IP or it may not have been indexed by Criminal IP.",
                    sources=[],
                    query=f"Criminal IP lookup: {ip}",
                    timestamp=datetime.now(),
                    model_used=self.config.model_name
                )
            
            logger.info(f"Found Criminal IP information for IP: {ip}")
            
            # Generate response using LLM
            response_prompt = (
                "You are a cybersecurity analyst working with Criminal IP data. "
                "Criminal IP is a search engine for security information about internet-connected devices. "
                "Analyze the following Criminal IP host information and provide a detailed, "
                "informative summary with security insights. Be specific about security issues, "
                "potential vulnerabilities, and notable findings. Include specific port information "
                "and service details that might be relevant from a security perspective.\n\n"
                f"Host IP: {ip}\n"
            )
            
            # Generate response using LLM with conversation history if enabled
            if self.config.enable_memory:
                response = self.orchestrator.generate_research_response(
                    response_prompt, 
                    [host_result],  # Pass as a list even though it's a single result
                    conversation_history=self._get_conversation_history(),
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
            logger.error(f"Error processing Criminal IP lookup: {str(e)}")
            return AgentResponse(
                answer=f"I apologize, but I encountered an error while looking up the IP address in Criminal IP: {str(e)}",
                sources=[],
                query=f"Criminal IP lookup: {ip}",
                timestamp=datetime.now(),
                model_used=self.config.model_name
            )
    
    # Add synchronous versions of the methods
    def process_criminalip_query_sync(self, query: str, limit: int = 5):
        """Synchronous version of process_criminalip_query"""
        import asyncio
        return asyncio.run(self.process_criminalip_query(query, limit))
    
    def process_criminalip_host_sync(self, ip: str):
        """Synchronous version of process_criminalip_host"""
        import asyncio
        return asyncio.run(self.process_criminalip_host(ip))
    
    # Add methods to the class
    agent_class.process_criminalip_query = process_criminalip_query
    agent_class.process_criminalip_host = process_criminalip_host
    agent_class.process_criminalip_query_sync = process_criminalip_query_sync
    agent_class.process_criminalip_host_sync = process_criminalip_host_sync
    
    return agent_class
