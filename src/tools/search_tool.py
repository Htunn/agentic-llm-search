"""
Internet search tool for the agentic LLM model
Supports multiple search engines with content extraction
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import time
import os
import re
from dotenv import load_dotenv

# Import the ShodanSearchTool
try:
    from src.tools.shodan_tool import ShodanSearchTool
    SHODAN_AVAILABLE = True
except ImportError:
    SHODAN_AVAILABLE = False

# Import the FofaSearchTool
try:
    from src.tools.fofa_tool import FofaSearchTool
    FOFA_AVAILABLE = True
except ImportError:
    FOFA_AVAILABLE = False

# Import the CriminalIPTool
try:
    from src.tools.criminalip_tool import CriminalIPTool
    CRIMINALIP_AVAILABLE = True
except ImportError:
    CRIMINALIP_AVAILABLE = False

from src import SearchResult

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class InternetSearchTool:
    """Tool for searching the internet and extracting content"""
    
    def __init__(self, max_results: int = 5, max_content_length: int = 2000):
        self.max_results = max_results
        self.max_content_length = max_content_length
        self.ddgs = DDGS()
        
        # Initialize Shodan search if available
        self.shodan = None
        if SHODAN_AVAILABLE:
            try:
                shodan_api_key = os.getenv("SHODAN_API_KEY")
                if shodan_api_key:
                    self.shodan = ShodanSearchTool(shodan_api_key)
                    logger.info("Shodan search tool initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Shodan search: {str(e)}")
                
        # Initialize FOFA search if available
        self.fofa = None
        if FOFA_AVAILABLE:
            try:
                fofa_email = os.getenv("FOFA_EMAIL")
                fofa_api_key = os.getenv("FOFA_API_KEY")
                if fofa_email and fofa_api_key:
                    self.fofa = FofaSearchTool(fofa_email, fofa_api_key)
                    logger.info("FOFA search tool initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize FOFA search: {str(e)}")
                
        # Initialize Criminal IP search if available
        self.criminalip = None
        if CRIMINALIP_AVAILABLE:
            try:
                # Try to get the API key from environment variables first
                criminalip_api_key = os.getenv("CRIMINAL_IP_API_KEY")
                
                # If not found, try to reload .env file (in case it was updated)
                if not criminalip_api_key:
                    load_dotenv(override=True)
                    criminalip_api_key = os.getenv("CRIMINAL_IP_API_KEY")
                
                if criminalip_api_key:
                    self.criminalip = CriminalIPTool(criminalip_api_key)
                    
                    # Test the API key by making a simple request
                    user_info = self.criminalip.get_user_info()
                    if "error" in user_info:
                        error_msg = user_info.get("error", "Unknown error")
                        solution = user_info.get("solution", "Check your API key configuration")
                        logger.warning(f"Criminal IP API key validation failed: {error_msg}. {solution}")
                        self.criminalip = None
                    else:
                        logger.info("Criminal IP search tool initialized successfully")
                else:
                    logger.warning("Criminal IP API key not found. Set CRIMINAL_IP_API_KEY in your .env file")
            except Exception as e:
                logger.warning(f"Failed to initialize Criminal IP search: {str(e)}")
                logger.info("To use Criminal IP search, set CRIMINAL_IP_API_KEY in your .env file")
    
    def search(self, query: str, search_type: str = "web") -> List[SearchResult]:
        """
        Search the internet using DuckDuckGo
        
        Args:
            query: Search query string
            search_type: Type of search ("web", "news", "images")
            
        Returns:
            List of SearchResult objects
        """
        try:
            logger.info(f"Searching for: {query}")
            
            search_results = []
            
            # Perform DuckDuckGo search
            if search_type == "news":
                results = list(self.ddgs.news(query, max_results=self.max_results))
            else:
                results = list(self.ddgs.text(query, max_results=self.max_results))
            
            for result in results:
                # Extract content from the webpage
                content = self._extract_content(result.get('href', ''))
                
                search_result = SearchResult(
                    title=result.get('title', ''),
                    url=result.get('href', ''),
                    content=content,
                    source='DuckDuckGo',
                    timestamp=datetime.now(),
                    relevance_score=1.0  # Could implement ranking algorithm
                )
                
                search_results.append(search_result)
                
            logger.info(f"Found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def _extract_content(self, url: str) -> str:
        """
        Extract text content from a webpage
        
        Args:
            url: URL to extract content from
            
        Returns:
            Extracted text content
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Truncate if too long
            if len(text) > self.max_content_length:
                text = text[:self.max_content_length] + "..."
                
            return text
            
        except Exception as e:
            logger.warning(f"Failed to extract content from {url}: {str(e)}")
            return ""
    
    async def async_search(self, query: str, search_type: str = "web") -> List[SearchResult]:
        """Async version of search method"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, query, search_type)

    def shodan_search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """
        Search for internet-connected devices using Shodan
        
        Args:
            query: The search query for Shodan
            limit: Maximum number of results to return
            
        Returns:
            List of SearchResult objects with Shodan data
        """
        if not self.shodan:
            logger.warning("Shodan search not available. Make sure SHODAN_API_KEY is set in .env")
            return []
        
        try:
            logger.info(f"Searching Shodan for: {query}")
            
            # Perform Shodan search
            results = self.shodan.search(query, limit=limit)
            
            search_results = []
            
            if "error" in results:
                logger.error(f"Shodan search error: {results['error']}")
                return []
                
            # Create search results from Shodan matches
            for match in results.get("matches", []):
                # Extract the most useful fields
                content = f"IP: {match.get('ip_str')}\n"
                content += f"Port: {match.get('port')}\n"
                
                if match.get("org"):
                    content += f"Organization: {match.get('org')}\n"
                    
                if match.get("hostname"):
                    content += f"Hostname: {', '.join(match.get('hostname', []))}\n"
                
                if match.get("country"):
                    content += f"Location: {match.get('city', 'Unknown')}, {match.get('country', 'Unknown')}\n"
                    
                if match.get("product"):
                    content += f"Product: {match.get('product')} {match.get('version', '')}\n"
                
                # Include a snippet of the data
                if match.get("data"):
                    content += f"\nData Sample:\n{match.get('data', '')[:500]}..."
                
                # Build the search result
                search_result = SearchResult(
                    title=f"Shodan: {match.get('ip_str')}:{match.get('port')} - {match.get('org', 'Unknown')}",
                    url=f"https://www.shodan.io/host/{match.get('ip_str')}",
                    content=content,
                    source='Shodan',
                    timestamp=datetime.now(),
                    relevance_score=1.0
                )
                
                search_results.append(search_result)
            
            logger.info(f"Found {len(search_results)} Shodan results")
            return search_results
            
        except Exception as e:
            logger.error(f"Shodan search failed: {str(e)}")
            return []
    
    def shodan_host_lookup(self, ip: str) -> Optional[SearchResult]:
        """
        Look up detailed information about an IP address in Shodan
        
        Args:
            ip: IP address to look up
            
        Returns:
            SearchResult with detailed host information or None
        """
        if not self.shodan:
            logger.warning("Shodan search not available. Make sure SHODAN_API_KEY is set in .env")
            return None
        
        try:
            logger.info(f"Looking up Shodan information for IP: {ip}")
            
            # Perform Shodan host lookup
            host_info = self.shodan.host_info(ip)
            
            if "error" in host_info:
                logger.error(f"Shodan lookup error: {host_info['error']}")
                return None
            
            # Build a detailed content string
            content = f"IP: {host_info.get('ip_str')}\n"
            content += f"Organization: {host_info.get('organization', 'Unknown')}\n"
            content += f"Location: {host_info.get('city', 'Unknown')}, {host_info.get('country', 'Unknown')}\n"
            content += f"Hostnames: {', '.join(host_info.get('hostnames', []))}\n"
            content += f"Domains: {', '.join(host_info.get('domains', []))}\n"
            content += f"Open Ports: {', '.join(str(port) for port in host_info.get('ports', []))}\n"
            
            if host_info.get("vulns"):
                content += f"Vulnerabilities: {', '.join(host_info.get('vulns', []))}\n"
            
            content += f"Last Update: {host_info.get('last_update', 'Unknown')}\n\n"
            
            # Add information about services
            content += f"Services ({host_info.get('services_count', 0)}):\n"
            for service in host_info.get("services", []):
                content += f"- Port {service.get('port')} ({service.get('protocol', '?')}): "
                content += f"{service.get('product', 'Unknown')} {service.get('version', '')}\n"
            
            # Create search result
            search_result = SearchResult(
                title=f"Shodan Host: {ip} - {host_info.get('organization', 'Unknown')}",
                url=f"https://www.shodan.io/host/{ip}",
                content=content,
                source='Shodan',
                timestamp=datetime.now(),
                relevance_score=1.0
            )
            
            return search_result
            
        except Exception as e:
            logger.error(f"Shodan host lookup failed: {str(e)}")
            return None
            
    async def async_shodan_search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Async version of shodan_search method"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.shodan_search, query, limit)
        
    async def async_shodan_host_lookup(self, ip: str) -> Optional[SearchResult]:
        """Async version of shodan_host_lookup method"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.shodan_host_lookup, ip)

    def fofa_search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """
        Search for internet-connected devices using FOFA
        
        Args:
            query: The search query for FOFA
            limit: Maximum number of results to return
            
        Returns:
            List of SearchResult objects with FOFA data
        """
        if not self.fofa:
            logger.warning("FOFA search not available. Make sure FOFA_EMAIL and FOFA_API_KEY are set in .env")
            return []
        
        try:
            logger.info(f"Searching FOFA for: {query}")
            
            # Define fields to retrieve for better results
            fields = "ip,port,protocol,domain,host,os,server,title,country,city"
            
            # Perform FOFA search
            results = self.fofa.search(query, limit=limit, fields=fields)
            
            search_results = []
            
            if "error" in results:
                logger.error(f"FOFA search error: {results['error']}")
                return []
                
            # Create search results from FOFA matches
            for item in results.get("results", []):
                # Extract the most useful fields
                content = f"IP: {item.get('ip')}\n"
                
                if item.get("port"):
                    content += f"Port: {item.get('port')}\n"
                
                if item.get("protocol"):
                    content += f"Protocol: {item.get('protocol')}\n"
                    
                if item.get("domain"):
                    content += f"Domain: {item.get('domain')}\n"
                
                if item.get("host"):
                    content += f"Host: {item.get('host')}\n"
                
                if item.get("server"):
                    content += f"Server: {item.get('server')}\n"
                
                if item.get("os"):
                    content += f"OS: {item.get('os')}\n"
                    
                if item.get("country") or item.get("city"):
                    location = f"{item.get('city', '')}, {item.get('country', '')}"
                    content += f"Location: {location.strip(', ')}\n"
                
                if item.get("title"):
                    content += f"\nTitle: {item.get('title')}\n"
                
                # Title for the result
                ip_port = f"{item.get('ip')}:{item.get('port')}" if item.get('port') else item.get('ip')
                server = item.get('server', '')
                title = f"FOFA: {ip_port} - {server}"
                
                # URL for viewing in FOFA
                url = f"https://fofa.info/result?qbase64={results.get('query', query)}"
                
                # Build the search result
                search_result = SearchResult(
                    title=title,
                    url=url,
                    content=content,
                    source='FOFA',
                    timestamp=datetime.now(),
                    relevance_score=1.0
                )
                
                search_results.append(search_result)
            
            logger.info(f"Found {len(search_results)} FOFA results")
            return search_results
            
        except Exception as e:
            logger.error(f"FOFA search failed: {str(e)}")
            return []
    
    def fofa_host_lookup(self, ip: str) -> Optional[SearchResult]:
        """
        Look up detailed information about an IP address in FOFA
        
        Args:
            ip: IP address to look up
            
        Returns:
            SearchResult with detailed host information or None
        """
        if not self.fofa:
            logger.warning("FOFA search not available. Make sure FOFA_EMAIL and FOFA_API_KEY are set in .env")
            return None
        
        try:
            logger.info(f"Looking up FOFA information for IP: {ip}")
            
            # Perform FOFA host lookup
            host_info = self.fofa.host_info(ip)
            
            if "error" in host_info:
                logger.error(f"FOFA lookup error: {host_info['error']}")
                return None
            
            # Build a detailed content string
            content = f"IP: {host_info.get('ip')}\n"
            
            if host_info.get("os"):
                content += f"Operating System: {host_info.get('os')}\n"
                
            if host_info.get("country") or host_info.get("city"):
                location = f"{host_info.get('city', '')}, {host_info.get('country', '')}"
                content += f"Location: {location.strip(', ')}\n"
            
            if host_info.get("domains"):
                content += f"Domains: {', '.join(host_info.get('domains', []))}\n"
                
            if host_info.get("protocols"):
                content += f"Protocols: {', '.join(host_info.get('protocols', []))}\n"
            
            # Add information about services
            content += f"\nServices ({host_info.get('services_count', 0)}):\n"
            for service in host_info.get("services", []):
                port_info = f"- Port {service.get('port')} ({service.get('protocol', '?')}): "
                server_info = service.get('server', '')
                title_info = service.get('title', '')
                
                service_details = f"{port_info}{server_info}"
                if title_info:
                    service_details += f" - {title_info}"
                    
                content += service_details + "\n"
            
            # Create search result
            search_result = SearchResult(
                title=f"FOFA Host: {ip}",
                url=f"https://fofa.info/hosts/{ip}",
                content=content,
                source='FOFA',
                timestamp=datetime.now(),
                relevance_score=1.0
            )
            
            return search_result
            
        except Exception as e:
            logger.error(f"FOFA host lookup failed: {str(e)}")
            return None
            
    async def async_fofa_search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Async version of fofa_search method"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.fofa_search, query, limit)
        
    async def async_fofa_host_lookup(self, ip: str) -> Optional[SearchResult]:
        """Async version of fofa_host_lookup method"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.fofa_host_lookup, ip)
    
    def criminalip_search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """
        Search for security information using Criminal IP
        
        Args:
            query: Search query
            limit: Maximum number of results (default: 5)
            
        Returns:
            List of SearchResult objects
        """
        if not self.criminalip:
            logger.warning("Criminal IP search not available. Make sure CRIMINAL_IP_API_KEY is set in .env")
            return []
        
        try:
            logger.info(f"Searching Criminal IP for: {query}")
            search_results = []
            
            # Check if the query is an IP address
            ip_pattern = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
            is_ip = bool(re.match(ip_pattern, query))
            
            if is_ip:
                # If it's an IP, do an IP report search
                result = self.criminalip_ip_lookup(query)
                if result:
                    search_results.append(result)
            else:
                # Check if it's a domain
                domain_pattern = r"^([a-z0-9]+(-[a-z0-9]+)*\.)+[a-z]{2,}$"
                is_domain = bool(re.match(domain_pattern, query.lower()))
                
                if is_domain:
                    # If it's a domain, do a domain search
                    domain_results = self.criminalip.search_domain(query, limit=limit)
                    formatted_results = self.criminalip.format_domain_results(domain_results)
                    
                    search_result = SearchResult(
                        title=f"Criminal IP Domain Analysis: {query}",
                        url=f"https://www.criminalip.io/domain/report?query={query}",
                        content=formatted_results,
                        source='Criminal IP',
                        timestamp=datetime.now(),
                        relevance_score=1.0
                    )
                    search_results.append(search_result)
                else:
                    # General asset search
                    asset_results = self.criminalip.asset_search(query, limit=limit)
                    
                    # Process the asset search results
                    if "data" in asset_results and "result" in asset_results["data"]:
                        items = asset_results["data"]["result"]
                        for item in items[:limit]:
                            # Determine the type from available fields
                            if "ip_address" in item:
                                result_type = "ip"
                                ip = item.get("ip_address", "")
                                domain = ""
                                title = ip
                            elif "domain" in item:
                                result_type = "domain"
                                domain = item.get("domain", "")
                                ip = ""
                                title = domain
                            else:
                                result_type = "service"
                                ip = ""
                                domain = ""
                                title = item.get("title", "") or item.get("hostname", "Unknown")
                            
                            # Create a URL based on the result type
                            if result_type == "ip":
                                url = f"https://www.criminalip.io/asset/report?ip={ip}"
                            elif result_type == "domain":
                                url = f"https://www.criminalip.io/domain/report?query={domain}"
                            else:
                                # For other types, use the search URL with the original query
                                url = f"https://www.criminalip.io/search?query={query}"
                            
                            # Build content from the item details
                            content = f"Type: {result_type}\n"
                            if ip:
                                content += f"IP: {ip}\n"
                            if domain:
                                content += f"Domain: {domain}\n"
                            if "hostname" in item and item["hostname"]:
                                content += f"Hostname: {item['hostname']}\n"
                            if "country" in item:
                                content += f"Country: {item['country']}\n"
                            if "city" in item and item["city"]:
                                content += f"City: {item['city']}\n"
                            if "as_name" in item and item["as_name"]:
                                content += f"ASN: {item['as_name']}\n"
                            if "server" in item and item["server"]:
                                content += f"Server: {item['server']}\n"
                            if "has_cve" in item:
                                has_cve = "Yes" if item["has_cve"] else "No"
                                content += f"Has CVEs: {has_cve}\n"
                                
                            search_result = SearchResult(
                                title=f"Criminal IP: {title}",
                                url=url,
                                content=content,
                                source='Criminal IP',
                                timestamp=datetime.now(),
                                relevance_score=0.9
                            )
                            search_results.append(search_result)
            
            logger.info(f"Found {len(search_results)} Criminal IP results")
            return search_results
            
        except Exception as e:
            logger.error(f"Criminal IP search failed: {str(e)}")
            return []
    
    def criminalip_ip_lookup(self, ip: str) -> Optional[SearchResult]:
        """
        Look up detailed information about an IP address in Criminal IP
        
        Args:
            ip: IP address to look up
            
        Returns:
            SearchResult with detailed IP information or None
        """
        if not self.criminalip:
            logger.warning("Criminal IP search not available. Make sure CRIMINAL_IP_API_KEY is set in .env")
            return None
        
        try:
            logger.info(f"Looking up Criminal IP information for IP: {ip}")
            
            # Get detailed IP report
            ip_report = self.criminalip.search_ip(ip)
            
            # Format the IP results
            content = self.criminalip.format_ip_results(ip_report)
            
            # Get additional security information
            try:
                malicious_info = self.criminalip.check_ip_malicious(ip)
                if "data" in malicious_info and malicious_info["data"]:
                    mal_data = malicious_info["data"]
                    content += "\nMalicious Activity Report:\n"
                    if "is_malicious" in mal_data:
                        mal_status = "Yes" if mal_data["is_malicious"] else "No"
                        content += f"Is Malicious: {mal_status}\n"
                    if "is_proxy" in mal_data:
                        proxy_status = "Yes" if mal_data["is_proxy"] else "No"
                        content += f"Is Proxy: {proxy_status}\n"
                    if "status" in mal_data:
                        content += f"Status: {mal_data['status']}\n"
            except Exception as mal_err:
                logger.warning(f"Could not retrieve malicious info: {str(mal_err)}")
            
            # Create search result
            search_result = SearchResult(
                title=f"Criminal IP Analysis: {ip}",
                url=f"https://www.criminalip.io/asset/report?ip={ip}",
                content=content,
                source='Criminal IP',
                timestamp=datetime.now(),
                relevance_score=1.0
            )
            
            return search_result
            
        except Exception as e:
            logger.error(f"Criminal IP lookup failed: {str(e)}")
            return None
    
    async def async_criminalip_search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Async version of criminalip_search method"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.criminalip_search, query, limit)
        
    async def async_criminalip_ip_lookup(self, ip: str) -> Optional[SearchResult]:
        """Async version of criminalip_ip_lookup method"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.criminalip_ip_lookup, ip)

class SearchQueryOptimizer:
    """Optimizes search queries for better results"""
    
    @staticmethod
    def optimize_query(original_query: str, additional_context: Optional[str] = None) -> str:
        """
        Optimize search query for better results
        
        Args:
            original_query: Original user query
            additional_context: Additional context for follow-up questions or clarification
            
        Returns:
            Optimized search query
        """
        # Add current year for recent information
        current_year = datetime.now().year
        
        # Basic query optimization
        optimized = original_query.strip()
        
        # Handle follow-up questions using additional context
        if additional_context:
            # This is likely a follow-up question, extract key terms from both
            # the original query and the additional context
            
            # Check for pronouns that need resolution
            pronouns = ["it", "this", "that", "these", "those", "they", "them", "their", "its"]
            has_pronouns = any(pronoun in optimized.lower().split() for pronoun in pronouns)
            
            # If we have pronouns and context, try to make the query more specific
            if has_pronouns:
                # Extract important terms from the context (simple approach)
                context_words = additional_context.lower().split()
                # Filter stop words and short words
                stop_words = ["a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", "of"]
                key_terms = [word for word in context_words if word not in stop_words and len(word) > 3]
                
                # Get most frequent important terms (simple approach)
                from collections import Counter
                term_counts = Counter(key_terms)
                top_terms = [term for term, count in term_counts.most_common(3)]
                
                # Create a more specific query by adding these terms
                if top_terms:
                    original_terms = optimized.split()
                    optimized = f"{optimized} {' '.join(term for term in top_terms if term not in original_terms)}"
                    logger.info(f"Enhanced follow-up query with context: {optimized}")
        
        # Add year for time-sensitive queries
        time_sensitive_keywords = ["latest", "recent", "current", "new", "today", "2024", "2025"]
        if any(keyword in optimized.lower() for keyword in time_sensitive_keywords):
            optimized += f" {current_year}"
        
        return optimized
