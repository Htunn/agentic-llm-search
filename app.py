"""
Streamlit web interface for the Agentic LLM Agent
"""

import streamlit as st
import asyncio
from datetime import datetime
import os
from dotenv import load_dotenv

from src.agents.agentic_llm import AgenticLLMAgent, AgentConfig
from src import AgentResponse

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Agentic LLM Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_agent(model_name=None, model_provider=None, max_memory=10, enable_memory=True):
    """Initialize the agent (cached)"""
    config = AgentConfig()
    
    # Override config with parameters if provided
    if model_name:
        config.model_name = model_name
    if model_provider:
        config.model_provider = model_provider
        
    return AgenticLLMAgent(
        model_name=config.model_name,
        model_provider=config.model_provider,
        max_search_results=config.max_search_results,
        enable_memory=enable_memory,
        max_memory=max_memory
    )

def display_response(response: AgentResponse):
    """Display the agent's response"""
    # Main answer
    st.markdown("### 📝 Answer")
    st.markdown(response.answer)
    
    # Sources
    if response.sources:
        st.markdown("### 📚 Sources")
        for i, source in enumerate(response.sources, 1):
            with st.expander(f"[{i}] {source.title}"):
                st.write(f"**URL:** {source.url}")
                st.write(f"**Source:** {source.source}")
                st.write(f"**Timestamp:** {source.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                if source.content:
                    st.write("**Content Preview:**")
                    st.text(source.content[:300] + "..." if len(source.content) > 300 else source.content)
    
    # Metadata
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Used", response.model_used)
    with col2:
        st.metric("Sources Found", len(response.sources))
    with col3:
        st.metric("Response Time", response.timestamp.strftime('%H:%M:%S'))

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🤖 Agentic LLM Agent")
    st.markdown("Ask me anything and I'll search the internet for the most up-to-date information!")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model provider selection
        model_provider = st.radio("Model Provider", ["huggingface", "openai", "azure-openai"], index=0)
        
        # Model selection based on provider
        if model_provider == "huggingface":
            model_options = ["./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "TheBloke/Llama-2-7B-Chat-GGUF", "microsoft/phi-2"]
        elif model_provider == "azure-openai":
            # For Azure OpenAI, we should use the model names that match Azure's naming
            # The actual deployment name will be taken from the environment variable
            model_options = ["gpt-35-turbo", "gpt-4", "gpt-4-turbo"]
        else:
            model_options = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
            
        selected_model = st.selectbox("Select Model", model_options, index=0)
        
        # Search settings
        enable_search = st.checkbox("Enable Internet Search", value=True)
        max_results = st.slider("Max Search Results", min_value=1, max_value=10, value=5)
        
        # Memory settings
        st.markdown("---")
        st.markdown("### 🧠 Conversation Memory")
        enable_memory = st.checkbox("Enable Conversation Memory", value=True)
        max_memory = st.slider("Max Memory Exchanges", min_value=1, max_value=20, value=10)
        
        # Clear memory button
        if st.button("🗑️ Clear Conversation Memory"):
            if "agent" in st.session_state and hasattr(st.session_state.agent, "clear_memory"):
                st.session_state.agent.clear_memory()
                st.success("Conversation memory cleared!")
        
        # Search type
        st.markdown("---")
        search_type = st.radio("Search Type", ["web", "news"], index=0)
        
        st.markdown("---")
        st.markdown("### 📊 Session Stats")
        if 'query_count' not in st.session_state:
            st.session_state.query_count = 0
        st.metric("Queries Processed", st.session_state.query_count)
    
    # Initialize agent
    try:
        # For Azure OpenAI, ensure we're using the deployment name from environment variables
        if model_provider == "azure-openai":
            # Display Azure OpenAI configuration info in the UI
            azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "Not set")
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "Not set")
            st.sidebar.info(f"Using Azure OpenAI\nDeployment: {azure_deployment}\nEndpoint: {azure_endpoint}")
            
            # Check if deployment name is configured
            if not azure_deployment:
                st.error("Azure OpenAI deployment name is not configured in .env file")
                st.info("Please set AZURE_OPENAI_DEPLOYMENT in your .env file")
                return
        
        agent = initialize_agent(selected_model, model_provider, max_memory, enable_memory)
        agent.set_search_enabled(enable_search)
        agent.set_memory_enabled(enable_memory)
        if hasattr(agent, 'max_search_results'):
            agent.max_search_results = max_results
        
        # Store the agent in the session state for buttons to use
        st.session_state.agent = agent
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        st.info("Please make sure you have set up your API keys in the .env file")
        return
    
    # Main interface
    query = st.text_input(
        "💬 Your Question:",
        placeholder="Ask me anything...",
        help="Type your question and press Enter"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary")
    
    # Example queries
    st.markdown("### 💡 Example Queries")
    examples = [
        "What are the latest developments in artificial intelligence?",
        "How does quantum computing work?",
        "What happened in tech news today?",
        "Explain the current state of renewable energy technology"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"📝 {example[:40]}...", key=f"example_{i}"):
                query = example
                ask_button = True
    
    # Process query
    if (ask_button or query) and query:
        st.session_state.query_count += 1
        
        with st.spinner("🔍 Searching and analyzing..."):
            try:
                # Process the query
                response = agent.process_query_sync(query, search_type)
                
                # Display results
                st.markdown("---")
                display_response(response)
                
                # Add to history
                if 'history' not in st.session_state:
                    st.session_state.history = []
                
                st.session_state.history.append({
                    'query': query,
                    'response': response,
                    'timestamp': datetime.now()
                })
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    # Query history
    if 'history' in st.session_state and st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Query History")
        
        for i, item in enumerate(reversed(st.session_state.history[-5:])):  # Show last 5
            with st.expander(f"🕐 {item['timestamp'].strftime('%H:%M:%S')} - {item['query'][:50]}..."):
                st.markdown(f"**Query:** {item['query']}")
                st.markdown(f"**Answer:** {item['response'].answer[:200]}...")
                if item['response'].sources:
                    st.markdown(f"**Sources:** {len(item['response'].sources)} found")
    
    # Show conversation memory
    if agent.enable_memory and agent.memory.messages:
        st.markdown("---")
        st.markdown("### 🧠 Conversation Context")
        
        with st.expander("Show current conversation memory", expanded=False):
            messages = agent.memory.get_conversation_history()
            for msg in messages:
                if msg.role == "user":
                    st.markdown(f"**👤 You:** {msg.content}")
                else:
                    st.markdown(f"**🤖 Agent:** {msg.content[:100]}..." if len(msg.content) > 100 else f"**🤖 Agent:** {msg.content}")
                st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit, Azure OpenAI, OpenAI, and DuckDuckGo Search | "
        "[GitHub](https://github.com/Htunn/agentic-llm-search)"
    )

if __name__ == "__main__":
    main()
