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

from src import SearchResult

logger = logging.getLogger(__name__)

class InternetSearchTool:
    """Tool for searching the internet and extracting content"""
    
    def __init__(self, max_results: int = 5, max_content_length: int = 2000):
        self.max_results = max_results
        self.max_content_length = max_content_length
        self.ddgs = DDGS()
        
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
            
            # Perform DuckDuckGo search
            if search_type == "news":
                results = list(self.ddgs.news(query, max_results=self.max_results))
            else:
                results = list(self.ddgs.text(query, max_results=self.max_results))
            
            search_results = []
            
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
