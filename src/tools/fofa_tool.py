#!/usr/bin/env python3
"""
FOFA Search Tool for the Agentic LLM Search project
This tool uses the FOFA API to search for internet-connected devices and services
"""

import os
import json
import base64
import logging
from typing import Dict, List, Optional, Any
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class FofaSearchTool:
    """
    A tool for searching and querying the FOFA API
    """

    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None):
        # Try to get API credentials from parameters, environment, or .env file
        self.email = email or os.getenv("FOFA_EMAIL")
        self.api_key = api_key or os.getenv("FOFA_API_KEY")
        
        if not self.email or not self.api_key:
            raise ValueError("FOFA credentials not found. Please set the FOFA_EMAIL and FOFA_API_KEY environment variables.")
        
        self.base_url = "https://fofa.info/api/v1"
        self.name = "fofa_search"
        self.description = "Search for internet-connected devices and services using FOFA."

    def search(self, query: str, limit: int = 10, fields: Optional[str] = None) -> Dict[str, Any]:
        """
        Search FOFA for devices matching the given query
        
        Args:
            query: The search query (will be base64 encoded)
            limit: Maximum number of results to return (default: 10)
            fields: Comma-separated list of fields to include in results
            
        Returns:
            Dict containing search results
        """
        try:
            # Default fields if not specified
            if not fields:
                fields = "ip,port,protocol,domain,host,os,server,title,country,city,cert"
            
            # Base64 encode the query
            encoded_query = base64.b64encode(query.encode()).decode()
            
            # Build the API URL
            url = f"{self.base_url}/search/all"
            params = {
                "email": self.email,
                "key": self.api_key,
                "qbase64": encoded_query,
                "size": limit,
                "fields": fields
            }
            
            logger.info(f"Searching FOFA for: {query}")
            logger.debug(f"Request URL: {url}")
            logger.debug(f"Request params: {params}")
            
            # Make the API request
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Check for API errors
            if data.get("error"):
                error_msg = data.get("errmsg", "Unknown API error")
                error_code = data.get("error", -1)
                
                # Translate common Chinese error messages
                if "F点余额不足" in error_msg:
                    user_friendly_msg = "Insufficient F-points in your FOFA account. Please upgrade your account or wait for points to replenish."
                    logger.error(f"FOFA API error: {error_code} - Insufficient F-points")
                else:
                    user_friendly_msg = error_msg
                    logger.error(f"FOFA API error: {error_msg}")
                
                return {
                    "error": user_friendly_msg,
                    "error_code": error_code,
                    "total": 0,
                    "results": []
                }
            
            # Format the results
            field_names = fields.split(",")
            formatted_results = []
            
            for result in data.get("results", []):
                result_dict = {}
                for i, field in enumerate(field_names):
                    if i < len(result):
                        result_dict[field] = result[i]
                formatted_results.append(result_dict)
            
            return {
                "total": data.get("size", 0),
                "page": 1,
                "mode": "extended",
                "query": query,
                "results": formatted_results
            }
            
        except requests.RequestException as e:
            logger.error(f"FOFA API request failed: {str(e)}")
            return {
                "error": f"Request failed: {str(e)}",
                "total": 0,
                "results": []
            }
        except json.JSONDecodeError:
            logger.error("Failed to parse FOFA API response")
            return {
                "error": "Failed to parse API response",
                "total": 0,
                "results": []
            }
        except Exception as e:
            logger.error(f"Unexpected error in FOFA search: {str(e)}")
            return {
                "error": f"Unexpected error: {str(e)}",
                "total": 0,
                "results": []
            }
    
    def host_info(self, ip: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific IP address
        
        Args:
            ip: The IP address to lookup
            
        Returns:
            Dict containing host information
        """
        # For IP info, we'll use a specific search query to get details about this IP
        query = f"ip=\"{ip}\""
        fields = "ip,port,protocol,domain,host,os,server,title,country,city,cert,icp"
        
        results = self.search(query, limit=20, fields=fields)
        
        # Check if there was an error
        if "error" in results:
            return results
        
        # If no results were found
        if not results.get("results"):
            return {
                "error": f"No information found for IP: {ip}",
                "total": 0
            }
        
        # Organize the data into a structured format
        services = []
        domains = set()
        protocols = set()
        country = None
        city = None
        os_type = None
        
        for item in results.get("results", []):
            # Extract domains if present
            if item.get("domain"):
                domains.add(item.get("domain"))
            
            # Extract protocols
            if item.get("protocol"):
                protocols.add(item.get("protocol"))
            
            # Extract location info from first result if not already set
            if not country and item.get("country"):
                country = item.get("country")
                
            if not city and item.get("city"):
                city = item.get("city")
                
            if not os_type and item.get("os"):
                os_type = item.get("os")
            
            # Add to services list
            service = {
                "port": item.get("port"),
                "protocol": item.get("protocol", ""),
                "server": item.get("server", ""),
                "title": item.get("title", "")
            }
            services.append(service)
        
        # Compile the host info
        host_info = {
            "ip": ip,
            "country": country,
            "city": city,
            "os": os_type,
            "domains": list(domains),
            "protocols": list(protocols),
            "services_count": len(services),
            "services": services
        }
        
        return host_info
    
    def stats(self, query: str) -> Dict[str, Any]:
        """
        Get statistics about the number of results for a query
        
        Args:
            query: The search query
            
        Returns:
            Dict containing statistics
        """
        try:
            # Base64 encode the query
            encoded_query = base64.b64encode(query.encode()).decode()
            
            # Build the API URL
            url = f"{self.base_url}/search/stats"
            params = {
                "email": self.email,
                "key": self.api_key,
                "qbase64": encoded_query,
            }
            
            # Make the API request
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Check for API errors
            if data.get("error"):
                error_msg = data.get("errmsg", "Unknown API error")
                error_code = data.get("error", -1)
                
                # Translate common Chinese error messages
                if "F点余额不足" in error_msg:
                    user_friendly_msg = "Insufficient F-points in your FOFA account. Please upgrade your account or wait for points to replenish."
                    logger.error(f"FOFA API error: {error_code} - Insufficient F-points")
                else:
                    user_friendly_msg = error_msg
                    logger.error(f"FOFA API error: {error_msg}")
                
                return {
                    "error": user_friendly_msg,
                    "error_code": error_code,
                    "total": 0
                }
            
            return data
            
        except Exception as e:
            return {"error": str(e), "total": 0}
    
    def account_info(self) -> Dict[str, Any]:
        """
        Get information about the FOFA account
        
        Returns:
            Dict containing account information
        """
        try:
            # Build the API URL
            url = f"{self.base_url}/info/my"
            params = {
                "email": self.email,
                "key": self.api_key
            }
            
            # Make the API request
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Check for API errors
            if data.get("error"):
                error_msg = data.get("errmsg", "Unknown API error")
                error_code = data.get("error", -1)
                
                # Translate common Chinese error messages
                if "F点余额不足" in error_msg:
                    user_friendly_msg = "Insufficient F-points in your FOFA account. Please upgrade your account or wait for points to replenish."
                    logger.error(f"FOFA API error: {error_code} - Insufficient F-points")
                else:
                    user_friendly_msg = error_msg
                    logger.error(f"FOFA API error: {error_msg}")
                
                return {
                    "error": user_friendly_msg,
                    "error_code": error_code
                }
            
            return data
            
        except Exception as e:
            return {"error": str(e)}
    
    def estimate_fofa_points_cost(self, operation: str, query: str = "", limit: int = 10, fields_count: int = 5) -> Dict[str, Any]:
        """
        Estimate the F-points cost for a FOFA operation
        This is an approximate estimation based on FOFA's typical pricing model
        
        Args:
            operation: Type of operation ('search', 'host', 'stats')
            query: The search query (for complexity estimation)
            limit: Maximum number of results (for search operations)
            fields_count: Number of fields requested in the response
            
        Returns:
            Dict with estimated cost information
        """
        # Base costs per operation (approximate)
        base_costs = {
            "search": 1,
            "host": 2,
            "stats": 3,
            "account_info": 0
        }
        
        if operation not in base_costs:
            return {
                "error": f"Unknown operation: {operation}",
                "estimated_cost": 0
            }
        
        # Start with base cost
        estimated_cost = base_costs.get(operation, 1)
        
        # Adjust for search parameters
        if operation == "search":
            # More results cost more points
            result_factor = max(1, limit // 10)
            estimated_cost *= result_factor
            
            # More fields cost more points
            fields_factor = max(1, fields_count // 5)
            estimated_cost *= fields_factor
            
            # Complex queries may cost more
            query_complexity = 1
            if query:
                # Check for complex operators
                if "&&" in query or "||" in query:
                    query_complexity += 0.5
                if query.count("=") > 2:
                    query_complexity += 0.5
                
            estimated_cost *= query_complexity
        
        return {
            "operation": operation,
            "estimated_cost": round(estimated_cost),
            "note": "This is an approximate estimation. Actual F-points cost may vary."
        }
