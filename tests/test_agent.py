"""
Test cases for the Agentic LLM Agent
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.agents.agentic_llm import AgenticLLMAgent, AgentConfig
from src.tools.search_tool import InternetSearchTool, SearchQueryOptimizer
from src.models.llm_models import OpenAIModel, AgentModelOrchestrator, HuggingFaceModel
from src import SearchResult, AgentResponse

class TestSearchQueryOptimizer:
    """Test cases for SearchQueryOptimizer"""
    
    def test_basic_query_optimization(self):
        optimizer = SearchQueryOptimizer()
        result = optimizer.optimize_query("test query")
        assert isinstance(result, str)
        assert "test query" in result
    
    def test_time_sensitive_optimization(self):
        optimizer = SearchQueryOptimizer()
        result = optimizer.optimize_query("latest AI developments")
        assert "2025" in result or "latest" in result

class TestInternetSearchTool:
    """Test cases for InternetSearchTool"""
    
    @patch('src.tools.search_tool.DDGS')
    @patch('src.tools.search_tool.requests.get')
    def test_search_with_mock(self, mock_get, mock_ddgs):
        # Mock DDGS response
        mock_ddgs_instance = Mock()
        mock_ddgs.return_value = mock_ddgs_instance
        mock_ddgs_instance.text.return_value = [
            {
                'title': 'Test Title',
                'href': 'https://example.com',
                'body': 'Test content'
            }
        ]
        
        # Mock requests response
        mock_response = Mock()
        mock_response.content = b'<html><body>Test content</body></html>'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test search
        search_tool = InternetSearchTool(max_results=1)
        results = search_tool.search("test query")
        
        assert len(results) == 1
        assert results[0].title == "Test Title"
        assert results[0].url == "https://example.com"
        assert isinstance(results[0], SearchResult)

class TestLLMModels:
    """Test cases for LLM models"""
    
    @patch('src.models.llm_models.OpenAI')
    def test_openai_model_initialization(self, mock_openai):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            model = OpenAIModel()
            assert model.model_name == "gpt-4"
            assert model.api_key == "test-key"
    
    @patch('src.models.llm_models.OpenAI')
    def test_openai_model_response_generation(self, mock_openai):
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            model = OpenAIModel()
            result = model.generate_response("test prompt")
            assert result == "Test response"
    
    @patch('src.models.llm_models.pipeline')
    @patch('src.models.llm_models.AutoTokenizer')
    @patch('src.models.llm_models.AutoModelForCausalLM')
    def test_huggingface_model_initialization(self, mock_auto_model, mock_tokenizer, mock_pipeline):
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_auto_model.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        # Set HUGGINGFACE_AVAILABLE to True for the test
        with patch('src.models.llm_models.HUGGINGFACE_AVAILABLE', True):
            model = HuggingFaceModel(model_name="./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
            assert model.model_name == "./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
            assert model.api_key is None
            
            # Verify the right methods were called
            mock_tokenizer.from_pretrained.assert_called_once_with("./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
            mock_auto_model.from_pretrained.assert_called_once()
            mock_pipeline.assert_called_once()

class TestAgenticLLMAgent:
    """Test cases for the main agent"""
    
    @patch('src.agents.agentic_llm.HuggingFaceModel')
    def test_agent_initialization(self, mock_hf_model):
        agent = AgenticLLMAgent()
        assert agent.model_name == "./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        assert agent.model_provider == "huggingface"
        assert agent.max_search_results == 5
        assert agent.enable_search is True
        
    @patch('src.agents.agentic_llm.OpenAIModel')
    def test_agent_initialization_openai(self, mock_openai_model):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = AgenticLLMAgent(model_provider="openai", model_name="gpt-4")
            assert agent.model_name == "gpt-4"
            assert agent.model_provider == "openai"
            assert agent.max_search_results == 5
            assert agent.enable_search is True
    
    @patch('src.agents.agentic_llm.OpenAIModel')
    @patch('src.tools.search_tool.InternetSearchTool.async_search')
    async def test_process_query_with_search(self, mock_search, mock_openai_model):
        # Mock search results
        mock_search_results = [
            SearchResult(
                title="Test Title",
                url="https://example.com",
                content="Test content",
                source="Test Source",
                timestamp=datetime.now()
            )
        ]
        mock_search.return_value = mock_search_results
        
        # Mock LLM response
        mock_model_instance = Mock()
        mock_openai_model.return_value = mock_model_instance
        
        mock_orchestrator = Mock()
        mock_orchestrator.generate_research_response.return_value = AgentResponse(
            answer="Test answer",
            sources=mock_search_results,
            query="test query",
            timestamp=datetime.now(),
            model_used="gpt-4"
        )
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            with patch('src.agents.agentic_llm.AgentModelOrchestrator', return_value=mock_orchestrator):
                agent = AgenticLLMAgent()
                response = await agent.process_query("test query")
                
                assert isinstance(response, AgentResponse)
                assert response.answer == "Test answer"
                assert len(response.sources) == 1

class TestAgentConfig:
    """Test cases for AgentConfig"""
    
    def test_default_config(self):
        config = AgentConfig()
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.max_search_results == 5
    
    def test_config_to_dict(self):
        config = AgentConfig()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "model_name" in config_dict
        assert "temperature" in config_dict

# Integration tests
class TestIntegration:
    """Integration tests (require actual API keys)"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_pipeline_huggingface(self):
        """Test the full pipeline with HuggingFace model"""
        try:
            # This will automatically use HuggingFace model
            agent = AgenticLLMAgent()
            response = await agent.process_query("What is Python programming language?")
            
            assert isinstance(response, AgentResponse)
            assert len(response.answer) > 0
            assert "TinyLlama" in response.model_used
        except ImportError:
            pytest.skip("HuggingFace dependencies not installed")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_pipeline_openai(self):
        """Test the full pipeline with real OpenAI APIs"""
        import os
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("OPENAI_API_KEY not set")
        
        agent = AgenticLLMAgent(model_name="gpt-3.5-turbo", model_provider="openai")
        response = await agent.process_query("What is Python programming language?")
        
        assert isinstance(response, AgentResponse)
        assert len(response.answer) > 0
        assert response.model_used in ["gpt-3.5-turbo", "gpt-4"]

# Fixtures
@pytest.fixture
def mock_search_results():
    """Fixture for mock search results"""
    return [
        SearchResult(
            title="Test Title 1",
            url="https://example1.com",
            content="Test content 1",
            source="Test Source",
            timestamp=datetime.now()
        ),
        SearchResult(
            title="Test Title 2",
            url="https://example2.com",
            content="Test content 2",
            source="Test Source",
            timestamp=datetime.now()
        )
    ]

@pytest.fixture
def mock_agent_response():
    """Fixture for mock agent response"""
    return AgentResponse(
        answer="Test answer with references [1] and [2]",
        sources=[],
        query="test query",
        timestamp=datetime.now(),
        model_used="gpt-4"
    )

if __name__ == "__main__":
    pytest.main([__file__])
