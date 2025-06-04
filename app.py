"""
Streamlit web interface for the Agentic LLM Agent
"""

import streamlit as st
import asyncio
from datetime import datetime
import os
from dotenv import load_dotenv

from src import AgentResponse

# Import the agent with Criminal IP integration
try:
    from src.agents.agentic_llm_with_criminalip import AgenticLLMAgent, AgentConfig
    CRIMINALIP_AVAILABLE = True
except ImportError:
    # Fall back to standard agent if Criminal IP integration is unavailable
    from src.agents.agentic_llm import AgenticLLMAgent, AgentConfig
    CRIMINALIP_AVAILABLE = False

# Load environment variables
load_dotenv()

# Check if Criminal IP API key is configured
CRIMINALIP_CONFIGURED = bool(os.getenv("CRIMINAL_IP_API_KEY"))

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
        
        # Security tools section
        st.header("🔒 Security Tools")
        
        # Criminal IP settings
        criminalip_section = st.expander("Criminal IP", expanded=CRIMINALIP_CONFIGURED)
        with criminalip_section:
            if CRIMINALIP_CONFIGURED:
                st.success("✅ Criminal IP API key configured")
                criminalip_enabled = st.checkbox("Enable Criminal IP search", value=True)
                
                criminalip_mode = st.radio("Query Type", 
                    ["Combined", "IP Lookup", "Domain Search"], 
                    index=0,
                    help="Combined mode will use regular search but include Criminal IP results if relevant")
                
                if criminalip_mode == "IP Lookup":
                    criminalip_ip = st.text_input("IP Address to lookup", placeholder="8.8.8.8")
                elif criminalip_mode == "Domain Search":
                    criminalip_domain = st.text_input("Domain to search for", placeholder="example.com")
            else:
                st.warning("⚠️ Criminal IP API key not configured")
                st.markdown("""
                To use Criminal IP, add your API key to the `.env` file:
                ```
                CRIMINAL_IP_API_KEY=your_api_key_here
                ```
                Get your API key from [Criminal IP](https://www.criminalip.io/developer/api)
                """)
                criminalip_enabled = False
        
        # FOFA configuration
        st.markdown("---")
        st.markdown("### 🌐 FOFA Integration")
        # Check if FOFA API credentials are available
        fofa_email = os.getenv("FOFA_EMAIL")
        fofa_api_key = os.getenv("FOFA_API_KEY")
        if fofa_email and fofa_api_key:
            st.success("FOFA API credentials are configured")
            enable_fofa = st.checkbox("Enable FOFA Search", value=True)
        else:
            st.warning("FOFA API credentials not found")
            st.info("Add FOFA_EMAIL and FOFA_API_KEY to your .env file to enable FOFA search capabilities")
            enable_fofa = st.checkbox("Enable FOFA Search", value=False, disabled=True)
        
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
        st.markdown("### 🔍 Search Options")
        
        # Build search type options based on available tools
        search_options = ["web", "news"]
        if os.getenv("FOFA_EMAIL") and os.getenv("FOFA_API_KEY"):
            search_options.append("fofa")
        if os.getenv("CRIMINAL_IP_API_KEY") and CRIMINALIP_AVAILABLE:
            search_options.append("criminalip")
            
        search_type = st.radio("Search Type", search_options, 
                              index=0,
                              help="Select specialized search engines for cybersecurity research")
                              
        # Show FOFA specific inputs if selected
        if search_type == "fofa":
            fofa_search_type = st.radio("FOFA Search Type", ["Query", "Host Lookup"], index=0)
            if fofa_search_type == "Host Lookup":
                st.info("For Host Lookup, please enter an IP address in the main search box")
        
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
    
    # Example queries - adjust based on selected search type
    st.markdown("### 💡 Example Queries")
    
    if search_type == "fofa":
        if fofa_search_type == "Host Lookup":
            examples = [
                "8.8.8.8",  # Google DNS
                "1.1.1.1",  # Cloudflare DNS
                "140.82.121.4",  # GitHub
                "13.107.42.16"   # Microsoft
            ]
        else:
            examples = [
                "domain=example.com",
                "cert=\"google.com\"",
                "title=\"admin login\"",
                "country=US && port=3389"
            ]
    elif search_type == "criminalip":
        if 'criminalip_mode' in locals() and criminalip_mode == "IP Lookup":
            examples = [
                "8.8.8.8",  # Google DNS
                "1.1.1.1",  # Cloudflare DNS
                "104.18.21.226",  # Cloudflare
                "142.250.190.78"  # Google
            ]
        elif 'criminalip_mode' in locals() and criminalip_mode == "Domain Search":
            examples = [
                "example.com",
                "google.com",
                "microsoft.com",
                "cloudflare.com"
            ]
        else:
            examples = [
                "apache 2.4",
                "nginx servers",
                "wordpress vulnerabilities",
                "exposed databases"
            ]
    else:
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
                # Process the query based on search type
                if search_type == "fofa":
                    # Handle FOFA searches
                    if not (os.getenv("FOFA_EMAIL") and os.getenv("FOFA_API_KEY")):
                        st.error("FOFA API credentials not configured. Please add FOFA_EMAIL and FOFA_API_KEY to your .env file.")
                        return
                    
                    # Determine which FOFA search type to use
                    if fofa_search_type == "Host Lookup":
                        # Check if the input looks like an IP address
                        import re
                        ip_pattern = re.compile(r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$')
                        if ip_pattern.match(query):
                            st.info(f"Looking up FOFA information for IP: {query}")
                            response = agent.process_fofa_host_sync(query)
                        else:
                            st.warning("Please enter a valid IP address for Host Lookup")
                            return
                    else:
                        # Regular FOFA search
                        st.info(f"Searching FOFA for: {query}")
                        response = agent.process_fofa_query_sync(query)
                elif search_type == "criminalip":
                    # Handle Criminal IP searches
                    if not os.getenv("CRIMINAL_IP_API_KEY"):
                        st.error("Criminal IP API key not configured. Please add CRIMINAL_IP_API_KEY to your .env file.")
                        return
                    
                    # Determine which Criminal IP search type to use
                    if criminalip_mode == "IP Lookup":
                        # Check if the input looks like an IP address
                        import re
                        ip_pattern = re.compile(r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$')
                        if ip_pattern.match(query):
                            st.info(f"Looking up Criminal IP information for IP: {query}")
                            if hasattr(agent, "process_criminalip_host_sync"):
                                response = agent.process_criminalip_host_sync(query)
                            else:
                                st.error("Criminal IP integration not available")
                                return
                        else:
                            st.warning("Please enter a valid IP address for IP Lookup")
                            return
                    else:
                        # Regular Criminal IP search or domain search
                        st.info(f"Searching Criminal IP for: {query}")
                        if hasattr(agent, "process_criminalip_query_sync"):
                            response = agent.process_criminalip_query_sync(query)
                        else:
                            st.error("Criminal IP integration not available")
                            return
                else:
                    # Regular web or news search
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
        "Built with ❤️ using Streamlit, Azure OpenAI, OpenAI, DuckDuckGo Search, and FOFA | "
        "[GitHub](https://github.com/Htunn/agentic-llm-search)"
    )

if __name__ == "__main__":
    main()
