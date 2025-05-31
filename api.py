"""
FastAPI web API for the Agentic LLM Agent
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import uvicorn
import os
from dotenv import load_dotenv

from src.agents.agentic_llm import AgenticLLMAgent, AgentConfig
from src import AgentResponse, SearchResult

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agentic LLM Agent API",
    description="An intelligent agent that searches the internet and provides responses with references",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent: Optional[AgenticLLMAgent] = None

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    search_type: str = "web"
    enable_search: bool = True
    max_results: Optional[int] = 5

class SearchResultResponse(BaseModel):
    title: str
    url: str
    content: str
    source: str
    timestamp: datetime
    relevance_score: float

class AgentResponseModel(BaseModel):
    answer: str
    sources: List[SearchResultResponse]
    query: str
    timestamp: datetime
    model_used: str

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    agent_initialized: bool

class ConfigResponse(BaseModel):
    model_name: str
    model_provider: str
    temperature: float
    max_tokens: int
    max_search_results: int
    search_engine: str
    debug: bool

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup"""
    global agent
    try:
        config = AgentConfig()
        agent = AgenticLLMAgent(
            model_name=config.model_name,
            model_provider=config.model_provider,
            max_search_results=config.max_search_results
        )
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {str(e)}")
        agent = None

# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Agentic LLM Agent API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if agent is not None else "unhealthy",
        timestamp=datetime.now(),
        version="1.0.0",
        agent_initialized=agent is not None
    )

@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    config = AgentConfig()
    return ConfigResponse(
        model_name=config.model_name,
        model_provider=config.model_provider,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        max_search_results=config.max_search_results,
        search_engine=config.search_engine,
        debug=config.debug
    )

@app.post("/query", response_model=AgentResponseModel)
async def process_query(request: QueryRequest):
    """Process a user query"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Update agent settings
        agent.set_search_enabled(request.enable_search)
        if request.max_results:
            agent.max_search_results = request.max_results
        
        # Process the query
        response = await agent.process_query(request.query, request.search_type)
        
        # Convert to response model
        sources_response = [
            SearchResultResponse(
                title=source.title,
                url=source.url,
                content=source.content,
                source=source.source,
                timestamp=source.timestamp,
                relevance_score=source.relevance_score
            )
            for source in response.sources
        ]
        
        return AgentResponseModel(
            answer=response.answer,
            sources=sources_response,
            query=response.query,
            timestamp=response.timestamp,
            model_used=response.model_used
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/search", response_model=List[SearchResultResponse])
async def search_internet(query: str, search_type: str = "web", max_results: int = 5):
    """Search the internet without generating a response"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Perform search
        search_results = await agent.search_tool.async_search(query, search_type)
        
        # Convert to response model
        return [
            SearchResultResponse(
                title=source.title,
                url=source.url,
                content=source.content,
                source=source.source,
                timestamp=source.timestamp,
                relevance_score=source.relevance_score
            )
            for source in search_results[:max_results]
        ]
        
    except Exception as e:
        logger.error(f"Error searching: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")

@app.put("/config/search")
async def update_search_config(enable_search: bool):
    """Enable or disable search"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    agent.set_search_enabled(enable_search)
    return {"message": f"Search {'enabled' if enable_search else 'disabled'}"}

@app.put("/config/model")
async def update_model(model_name: str, model_provider: Optional[str] = None):
    """Update the LLM model"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        agent.update_model(model_name, model_provider)
        return {
            "message": f"Model updated to {model_name} using provider {model_provider if model_provider else agent.model_provider}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error updating model: {str(e)}")

# Example usage endpoints
@app.get("/examples", response_model=List[str])
async def get_examples():
    """Get example queries"""
    return [
        "What are the latest developments in artificial intelligence?",
        "How does quantum computing work?",
        "What happened in tech news today?",
        "Explain the current state of renewable energy technology",
        "What are the latest COVID-19 research findings?",
        "How is climate change affecting global weather patterns?"
    ]

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )
