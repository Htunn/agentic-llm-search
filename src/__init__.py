"""
Agentic LLM Model with Internet Search Capabilities

This project implements an intelligent agent that can search the internet
and provide responses with proper references and citations.
"""

from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SearchResult:
    """Represents a search result with metadata"""
    title: str
    url: str
    content: str
    source: str
    timestamp: datetime
    relevance_score: float = 0.0

@dataclass
class AgentResponse:
    """Represents the agent's response with references"""
    answer: str
    sources: List[SearchResult]
    query: str
    timestamp: datetime
    model_used: str

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
