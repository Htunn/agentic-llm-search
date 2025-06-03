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
import dotenv

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

from src import SearchResult

# Load environment variables
dotenv.load_dotenv()

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
