"""
Criminal IP Search Tool for Agentic LLM Search

This tool provides integration with Criminal IP's cybersecurity search engine API,
allowing the agent to search for information about IP addresses, domains,
and other security-related information.

API Documentation: https://www.criminalip.io/developer/api/
"""

import os
import logging
import requests
from typing import Dict, List, Optional, Any, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class CriminalIPTool:
    """Tool for interacting with Criminal IP API"""
    
    BASE_URL = "https://api.criminalip.io/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Criminal IP tool
        
        Args:
            api_key: Criminal IP API key (if not provided, will try to load from CRIMINAL_IP_API_KEY env var)
        """
        # Try to get API key from parameter, then environment variable, then config file
        self.api_key = api_key or os.getenv("CRIMINAL_IP_API_KEY")
        
        # If still no API key, try to read from .env file directly
        if not self.api_key:
            try:
                # Try to reload from .env file in case it was updated after program started
                load_dotenv(override=True)
                self.api_key = os.getenv("CRIMINAL_IP_API_KEY")
            except Exception as e:
                logger.debug(f"Error reloading .env file: {str(e)}")
        
        if not self.api_key:
            logger.warning("Criminal IP API key not found. Set CRIMINAL_IP_API_KEY environment variable.")
            logger.info("To obtain a Criminal IP API key, visit https://www.criminalip.io/developer and create an account")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
    
    def verify_api_key(self) -> Dict[str, Any]:
        """
        Verify that the API key is valid by making a simple request
        
        Returns:
            Dictionary with verification result:
            - success: True/False indicating if the API key is valid
            - message: Explanation of the verification result
            - user_info: User information if verification was successful
        """
        if not self.api_key:
            return {
                "success": False, 
                "message": "API key not configured. Please set CRIMINAL_IP_API_KEY in your environment or .env file.",
                "solution": "Visit https://www.criminalip.io/developer to create an account and obtain an API key."
            }
        
        # Try to get user information as a simple API test
        try:
            user_info = self.get_user_info()
            
            # Check if the request was successful
            if "error" in user_info:
                error_msg = user_info.get("error", "Unknown error")
                if "api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                    return {
                        "success": False,
                        "message": f"Invalid API key: {error_msg}",
                        "solution": "Check your API key or visit https://www.criminalip.io/developer to generate a new one."
                    }
                else:
                    return {
                        "success": False,
                        "message": f"API error: {error_msg}",
                        "solution": "The API key may be valid but there was an error with the request."
                    }
            
            # If we get here, the API key is valid
            return {
                "success": True,
                "message": "API key verified successfully",
                "user_info": user_info
            }
            
        except Exception as e:
            logger.error(f"API key verification error: {str(e)}")
            return {
                "success": False,
                "message": f"Error during verification: {str(e)}",
                "solution": "Check your internet connection or try again later."
            }
    
    def _make_request(self, endpoint: str, method: str = "GET", params: Optional[Dict[str, Any]] = None, 
                     data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the Criminal IP API
        
        Args:
            endpoint: API endpoint path
            method: HTTP method (GET, POST)
            params: URL parameters
            data: Request body data for POST requests
            
        Returns:
            API response as a dictionary
        """
        # Check for API key and try to reload if missing
        if not self.api_key:
            try:
                # Try to reload from .env file in case it was updated after program started
                load_dotenv(override=True)
                self.api_key = os.getenv("CRIMINAL_IP_API_KEY")
                
                if not self.api_key:
                    logger.error("Criminal IP API key not configured. Please set CRIMINAL_IP_API_KEY in your .env file")
                    return {
                        "error": "Criminal IP API key not configured", 
                        "solution": "Set CRIMINAL_IP_API_KEY in your .env file or run 'python criminalip_cli.py setup'"
                    }
            except Exception as e:
                logger.error(f"Error loading Criminal IP API key: {str(e)}")
                return {
                    "error": "Failed to load Criminal IP API key", 
                    "solution": "Set CRIMINAL_IP_API_KEY in your .env file or run 'python criminalip_cli.py setup'"
                }
        
        url = f"{self.BASE_URL}{endpoint}"
        headers = self._get_headers()
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data)
            else:
                return {"error": f"Unsupported method: {method}"}
            
            # First check HTTP-level errors
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if response.status_code == 401:
                    logger.error("Criminal IP API authentication failed: Invalid API key")
                    return {
                        "error": "Authentication failed", 
                        "solution": "Check your CRIMINAL_IP_API_KEY in .env file or run 'python criminalip_cli.py setup'"
                    }
                elif response.status_code == 429:
                    logger.error("Criminal IP API rate limit exceeded")
                    return {
                        "error": "API rate limit exceeded", 
                        "solution": "Wait before making additional requests or upgrade your plan"
                    }
                else:
                    logger.error(f"Criminal IP API HTTP error: {str(e)}")
                    return {"error": f"API request failed: {str(e)}"}
            
            # Now parse the response
            result = response.json()
            
            # Handle API-specific error responses
            if isinstance(result, dict):
                if "status" in result and result["status"] == "fail":
                    error_msg = result.get("message", "Unknown API error")
                    logger.error(f"Criminal IP API error: {error_msg}")
                    return {"error": f"API error: {error_msg}"}
                    
            return result
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Criminal IP API request error: {str(e)}")
            return {"error": f"API request failed: {str(e)}"}
        
        except ValueError as e:  # Includes JSONDecodeError
            logger.error(f"Criminal IP API response parsing error: {str(e)}")
            return {"error": "Failed to parse API response", "solution": "The API may be experiencing issues"}
    
    def get_user_info(self) -> Dict[str, Any]:
        """
        Get current user information
        
        Returns:
            User information including quota and plan details
        """
        return self._make_request("/user/me", method="POST")
    
    def search_ip(self, ip_address: str) -> Dict[str, Any]:
        """
        Search for information about an IP address
        
        Args:
            ip_address: IP address to search for
            
        Returns:
            IP information including security details and score
        """
        return self._make_request("/asset/ip/report", params={"ip": ip_address})
    
    def get_ip_summary(self, ip_address: str) -> Dict[str, Any]:
        """
        Get summary information about an IP address
        
        Args:
            ip_address: IP address to get summary for
            
        Returns:
            IP summary information
        """
        return self._make_request("/asset/ip/summary", params={"ip": ip_address})
    
    def check_ip_malicious(self, ip_address: str) -> Dict[str, Any]:
        """
        Check if an IP address is associated with malicious activity
        
        Args:
            ip_address: IP address to check
            
        Returns:
            Malicious activity information for the IP
        """
        return self._make_request("/ip/malicious-info", params={"ip": ip_address})
    
    def search_domain(self, domain: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for information about a domain
        
        Args:
            domain: Domain to search for
            limit: Maximum number of results (default: 10)
            
        Returns:
            Domain information including security details
        """
        return self._make_request("/domain/reports", params={"query": domain, "offset": 0, "limit": limit})
    
    def asset_search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for assets (IPs, domains, etc.) matching a query
        
        Args:
            query: Search query
            limit: Maximum number of results (default: 10)
            
        Returns:
            Asset search results
        """
        return self._make_request("/asset/search", params={"query": query, "limit": limit})
    
    def banner_search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for banner information matching a query
        
        Args:
            query: Search query
            limit: Maximum number of results (default: 10)
            
        Returns:
            Banner search results
        """
        return self._make_request("/banner/search", params={"query": query, "offset": 0, "limit": limit})
    
    def scan_domain(self, domain: str) -> Dict[str, Any]:
        """
        Request a domain scan
        
        Args:
            domain: Domain to scan
            
        Returns:
            Scan request information
        """
        return self._make_request("/domain/scan", method="POST", data={"domain": domain})
    
    def get_domain_status(self, scan_id: str) -> Dict[str, Any]:
        """
        Get domain scan status
        
        Args:
            scan_id: ID of the scan request
            
        Returns:
            Scan status information
        """
        return self._make_request(f"/domain/status/{scan_id}")
    
    def format_ip_results(self, results: Dict[str, Any]) -> str:
        """
        Format IP search results for the agent
        
        Args:
            results: IP search results
            
        Returns:
            Formatted results as a string
        """
        if "error" in results:
            return f"Error retrieving IP information: {results['error']}"
        
        try:
            data = results.get("data", {})
            
            if not data:
                return "No information found for this IP address."
            
            # Extract key information
            ip = data.get("ip", "Unknown")
            score = data.get("score", {}).get("total", "Unknown")
            country = data.get("first_seen_country", "Unknown")
            isp = data.get("isp", "Unknown")
            open_ports = data.get("open_ports", [])
            
            # Format the results
            formatted = f"IP: {ip}\n"
            formatted += f"Security Score: {score}\n"
            formatted += f"Country: {country}\n"
            formatted += f"ISP: {isp}\n"
            
            # Add open ports if available
            if open_ports:
                formatted += "Open Ports:\n"
                for port in open_ports[:5]:  # Show first 5 ports
                    port_number = port.get("port", "Unknown")
                    protocol = port.get("protocol", "Unknown")
                    service = port.get("service_name", "Unknown")
                    formatted += f"- Port {port_number} ({protocol}): {service}\n"
                
                if len(open_ports) > 5:
                    formatted += f"...and {len(open_ports) - 5} more open ports\n"
            
            # Add malicious activity if available
            malicious = data.get("malicious_info", {}).get("is_malicious")
            if malicious:
                formatted += "\nThis IP has been flagged for malicious activity.\n"
            
            return formatted
        
        except Exception as e:
            logger.error(f"Error formatting IP results: {str(e)}")
            return "Error formatting IP address information."
    
    def format_domain_results(self, results: Dict[str, Any]) -> str:
        """
        Format domain search results for the agent
        
        Args:
            results: Domain search results
            
        Returns:
            Formatted results as a string
        """
        if "error" in results:
            return f"Error retrieving domain information: {results['error']}"
        
        try:
            data = results.get("data", {})
            reports = data.get("reports", [])  # Updated from "items" to "reports"
            
            if not reports:
                return "No information found for this domain."
            
            # Format the results
            formatted = f"Found {data.get('count', 0)} results for this domain.\n\n"
            
            # Show the first 3 results
            for i, item in enumerate(reports[:3], 1):  # Updated from "items" to "reports"
                # In the new API format, domain name is in the query
                domain = item.get("domain", "Unknown")
                score = item.get("score", "Unknown")
                scan_date = item.get("reg_dtime", "Unknown")  # Updated from "scan_date" to "reg_dtime"
                
                formatted += f"Result {i}:\n"
                formatted += f"Domain: {domain}\n"
                formatted += f"Security Score: {score}\n"
                formatted += f"Scan Date: {scan_date}\n"
                
                # Add connected IPs count
                if "connected_ip_cnt" in item:
                    formatted += f"Connected IPs: {item['connected_ip_cnt']}\n"
                
                # Add issues if available
                if "issue" in item and isinstance(item["issue"], list) and item["issue"]:
                    formatted += "Issues:\n"
                    for issue in item["issue"]:
                        formatted += f"- {issue}\n"
                
                # Add countries if available
                if "country_code" in item and isinstance(item["country_code"], list) and item["country_code"]:
                    countries = [code for code in item["country_code"] if code]
                    if countries:
                        formatted += f"Countries: {', '.join(countries)}\n"
                
                formatted += "\n"
            
            if len(reports) > 3:
                formatted += f"...and {len(reports) - 3} more results\n"
                
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting domain results: {str(e)}")
            return f"Error processing domain information: {str(e)}"
    
    def validate_api_key(self) -> Dict[str, Any]:
        """
        Validate the Criminal IP API key
        
        Returns:
            Dict with validation results including:
            - is_valid: bool - True if API key is valid
            - message: str - Status message
            - error: str - Error message if any
        """
        if not self.api_key:
            return {
                "is_valid": False,
                "message": "API key is not configured",
                "error": "Missing API key",
                "solution": "Set CRIMINAL_IP_API_KEY in your .env file"
            }
            
        # Try to get user info as a simple validation check
        try:
            user_info = self.get_user_info()
            if "error" in user_info:
                return {
                    "is_valid": False,
                    "message": f"API key validation failed: {user_info.get('error', 'Unknown error')}",
                    "error": user_info.get("error"),
                    "solution": user_info.get("solution", "Check your API key configuration")
                }
            
            return {
                "is_valid": True,
                "message": "API key is valid",
                "user_info": user_info
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "message": f"API key validation failed with exception: {str(e)}",
                "error": str(e),
                "solution": "Check your API key and internet connection"
            }
    
    def verify_api_key(self) -> Dict[str, Any]:
        """
        Verify that the API key is valid by making a simple request
        
        Returns:
            Dictionary with verification result:
            - success: True/False indicating if the API key is valid
            - message: Explanation of the verification result
            - user_info: User information if verification was successful
        """
        if not self.api_key:
            return {
                "success": False, 
                "message": "API key not configured. Please set CRIMINAL_IP_API_KEY in your environment or .env file.",
                "solution": "Visit https://www.criminalip.io/developer to create an account and obtain an API key."
            }
        
        # Try to get user information as a simple API test
        try:
            user_info = self.get_user_info()
            
            # Check if the request was successful
            if "error" in user_info:
                error_msg = user_info.get("error", "Unknown error")
                if "api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                    return {
                        "success": False,
                        "message": f"Invalid API key: {error_msg}",
                        "solution": "Check your API key or visit https://www.criminalip.io/developer to generate a new one."
                    }
                else:
                    return {
                        "success": False,
                        "message": f"API error: {error_msg}",
                        "solution": "The API key may be valid but there was an error with the request."
                    }
            
            # If we get here, the API key is valid
            return {
                "success": True,
                "message": "API key verified successfully",
                "user_info": user_info
            }
            
        except Exception as e:
            logger.error(f"API key verification error: {str(e)}")
            return {
                "success": False,
                "message": f"Error during verification: {str(e)}",
                "solution": "Check your internet connection or try again later."
            }
