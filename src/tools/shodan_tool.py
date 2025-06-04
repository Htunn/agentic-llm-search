#!/usr/bin/env python3
"""
Shodan Search Tool for the Agentic LLM Search project
This tool uses the Shodan API to search for internet-connected devices and services
"""

import os
import json
from typing import Dict, List, Optional, Any
from shodan import Shodan
from shodan.exception import APIError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ShodanSearchTool:
    """
    A tool for searching and querying the Shodan API
    """

    def __init__(self, api_key: Optional[str] = None):
        # Try to get API key from parameters, environment, or .env file
        self.api_key = api_key or os.getenv("SHODAN_API_KEY")
        
        if not self.api_key:
            raise ValueError("Shodan API key not found. Please set the SHODAN_API_KEY environment variable.")
        
        self.client = Shodan(self.api_key)
        self.name = "shodan_search"
        self.description = "Search for internet-connected devices and services using Shodan."

    def search(self, query: str, limit: int = 10, facets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Search Shodan for devices matching the given query
        
        Args:
            query: The search query
            limit: Maximum number of results to return (default: 10)
            facets: List of facets to include in results
            
        Returns:
            Dict containing search results
        """
        try:
            # Perform the search
            results = self.client.search(query, limit=limit, facets=facets)
            
            # Format and return the results
            formatted_results = {
                "total": results.get("total", 0),
                "matches": []
            }
            
            # Process each result
            for item in results.get("matches", []):
                # Extract the most useful information from each match
                match = {
                    "ip_str": item.get("ip_str"),
                    "port": item.get("port"),
                    "org": item.get("org"),
                    "hostname": item.get("hostnames", []),
                    "country": item.get("location", {}).get("country_name"),
                    "city": item.get("location", {}).get("city"),
                    "timestamp": item.get("timestamp"),
                    "product": item.get("product"),
                    "version": item.get("version"),
                    "data": item.get("data", "")[:500]  # Limit data size
                }
                formatted_results["matches"].append(match)
            
            # Include facets if any
            if "facets" in results:
                formatted_results["facets"] = results["facets"]
                
            return formatted_results
            
        except APIError as e:
            error_message = str(e)
            
            # Provide more helpful error messages for common issues
            if "403 Forbidden" in error_message or "Access denied" in error_message:
                # Check API info to provide more context
                try:
                    api_info = self.client.info()
                    credits = api_info.get("query_credits", 0)
                    plan = api_info.get("plan", "unknown")
                    
                    if credits <= 0:
                        error_message = f"Access denied: You have 0 query credits remaining on your {plan} plan. " \
                                        f"Please upgrade your Shodan account or wait for credits to renew."
                    else:
                        error_message = f"Access denied (403 Forbidden): Your {plan} plan may not have permission " \
                                        f"for this search query. Available credits: {credits}."
                except:
                    error_message = "Access denied (403 Forbidden): Your Shodan plan may not allow this search or " \
                                    "you have exceeded your API limits. Please check your account at https://account.shodan.io/"
            
            return {
                "error": error_message,
                "total": 0,
                "matches": []
            }
    
    def host_info(self, ip: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific IP address
        
        Args:
            ip: The IP address to lookup
            
        Returns:
            Dict containing host information
        """
        try:
            results = self.client.host(ip)
            
            # Extract and return the most relevant information
            host_info = {
                "ip_str": results.get("ip_str"),
                "organization": results.get("org"),
                "country": results.get("country_name"),
                "city": results.get("city"),
                "hostnames": results.get("hostnames", []),
                "domains": results.get("domains", []),
                "ports": results.get("ports", []),
                "vulns": results.get("vulns", []),
                "last_update": results.get("last_update"),
                "services_count": len(results.get("data", [])),
                "services": []
            }
            
            # Process each service running on the host
            for service in results.get("data", []):
                service_info = {
                    "port": service.get("port"),
                    "protocol": service.get("transport", ""),
                    "product": service.get("product", ""),
                    "version": service.get("version", ""),
                }
                host_info["services"].append(service_info)
                
            return host_info
            
        except APIError as e:
            return {"error": str(e)}
    
    def count(self, query: str, facets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Count the number of devices matching a query
        
        Args:
            query: The search query
            facets: Optional facets to count by
            
        Returns:
            Dict containing the total count and facet statistics
        """
        try:
            results = self.client.count(query, facets=facets)
            return results
        except APIError as e:
            return {"error": str(e), "total": 0}

    def api_info(self) -> Dict[str, Any]:
        """
        Get information about the Shodan API plan and usage
        
        Returns:
            Dict containing API plan information
        """
        try:
            return self.client.info()
        except APIError as e:
            return {"error": str(e)}
