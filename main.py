"""
Improved Streamlit interface with comprehensive error handling and better UX.
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Any

import streamlit as st
from dotenv import load_dotenv

from src.config import Config
from src.agents import AgentManager, AgentError
from src.tools import ToolError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Agentic JSON-RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stAlert > div {
        padding: 10px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .assistant-message {
        background-color: #e8f4fd;
    }
    .source-info {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state with default values."""
    defaults = {
        'config': None,
        'agent_manager': None,
        'chat_history': [],
        'is_initialized': False,
        'initialization_error': None,
        'query_count': 0,
        'start_time': datetime.now()
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def load_configuration() -> bool:
    """Load and validate configuration."""
    try:
        if st.session_state.config is None:
            st.session_state.config = Config()
            logger.info("Configuration loaded successfully")
        return True
    except Exception as e:
        st.session_state.initialization_error = str(e)
        logger.error(f"Configuration failed: {e}")
        return False

def initialize_agent_manager() -> bool:
    """Initialize the agent manager."""
    try:
        if st.session_state.agent_manager is None:
            with st.spinner("🤖 Initializing AI agent and loading knowledge bases..."):
                st.session_state.agent_manager = AgentManager(st.session_state.config)
                # Pre-create the agent to catch initialization errors early
                st.session_state.agent_manager.get_agent()
                logger.info("Agent manager initialized successfully")
        return True
    except (AgentError, ToolError) as e:
        st.session_state.initialization_error = str(e)
        logger.error(f"Agent initialization failed: {e}")
        return False
    except Exception as e:
        st.session_state.initialization_error = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected initialization error: {e}")
        return False

def display_sidebar():
    """Display sidebar with system information and controls."""
    with st.sidebar:
        st.header("📊 System Status")
        
        # System status
        if st.session_state.is_initialized:
            st.success("✅ System Ready")
            
            # Stats
            st.subheader("📈 Statistics")
            st.metric("Queries Processed", st.session_state.query_count)
            
            uptime = datetime.now() - st.session_state.start_time
            st.metric("Uptime", f"{uptime.total_seconds():.0f}s")
            
            # Configuration info
            st.subheader("⚙️ Configuration")
            config = st.session_state.config
            st.text(f"LLM: {config.model.llm_model.split('/')[-1]}")
            st.text(f"Embedding: {config.model.embedding_model.split('/')[-1]}")
            st.text(f"Chunk Size: {config.rag.chunk_size}")
            st.text(f"Retrieval K: {config.rag.retrieval_k}")
            
            # Data files
            st.subheader("📁 Data Sources")
            try:
                data_paths = config.get_data_paths()
                for path in data_paths:
                    st.text(f"✓ {path.name}")
            except Exception as e:
                st.error(f"Error loading data paths: {e}")
            
        else:
            st.error("❌ System Not Ready")
            if st.session_state.initialization_error:
                st.error(st.session_state.initialization_error)
        
        # Controls
        st.subheader("🔧 Controls")
        
        if st.button("🔄 Restart System"):
            restart_system()
        
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("💾 Clear Cache"):
            if st.session_state.agent_manager:
                st.session_state.agent_manager.clear_cache()
                st.success("Cache cleared!")

def restart_system():
    """Restart the entire system."""
    logger.info("Restarting system...")
    
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Reinitialize
    initialize_session_state()
    st.rerun()

def display_initialization_error():
    """Display initialization error with helpful suggestions."""
    st.error("🚨 System Initialization Failed")
    
    error_msg = st.session_state.initialization_error
    st.error(error_msg)
    
    # Provide helpful suggestions based on error type
    if "environment variable" in error_msg.lower():
        st.info("""
        **Environment Variable Missing:**
        1. Create a `.env` file in your project root
        2. Add the following variables:
        ```
        HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
        SERPAPI_API_KEY=your_serpapi_key
        ```
        3. Restart the application
        """)
    
    elif "data file" in error_msg.lower() or "filenotfounderror" in error_msg.lower():
        st.info("""
        **Data Files Missing:**
        1. Ensure your JSON files are in the `data/` directory
        2. Check that the file names match your configuration
        3. Verify that the JSON files are valid and not empty
        """)
    
    elif "api" in error_msg.lower():
        st.info("""
        **API Connection Issue:**
        1. Check your internet connection
        2. Verify your API keys are valid and have sufficient credits
        3. Try again in a few moments
        """)
    
    if st.button("🔄 Retry Initialization"):
        restart_system()

def format_chat_message(role: str, content: str, tools_used: List[str] = None) -> None:
    """Format and display a chat message."""
    with st.chat_message(role):
        st.markdown(content)
        
        if role == "assistant" and tools_used:
            tools_str = ", ".join(tools_used) if tools_used else "Unknown"
            st.markdown(
                f'<div class="source-info">🔧 Tools used: {tools_str}</div>',
                unsafe_allow_html=True
            )

def process_query(query: str) -> None:
    """Process user query with comprehensive error handling."""
    start_time = time.time()
    
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user", 
        "content": query,
        "timestamp": datetime.now()
    })
    
    # Display user message
    format_chat_message("user", query)
    
    # Process with assistant
    with st.chat_message("assistant"):
        try:
            # Show thinking indicator
            with st.spinner("🤔 Thinking..."):
                response, tools_used = st.session_state.agent_manager.run_with_error_handling(query)
            
            # Display response
            st.markdown(response)
            
            # Display source information
            if tools_used:
                tools_str = ", ".join(tools_used)
                st.markdown(
                    f'<div class="source-info">🔧 Tools used: {tools_str}</div>',
                    unsafe_allow_html=True
                )
            
            # Performance info
            processing_time = time.time() - start_time
            st.markdown(
                f'<div class="source-info">⏱️ Processed in {processing_time:.2f}s</div>',
                unsafe_allow_html=True
            )
            
            # Add to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "tools_used": tools_used,
                "processing_time": processing_time,
                "timestamp": datetime.now()
            })
            
            # Update query count
            st.session_state.query_count += 1
            
        except Exception as e:
            error_msg = f"❌ I encountered an unexpected error: {str(e)}"
            st.error(error_msg)
            logger.error(f"Query processing failed: {e}")
            
            # Add error to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg,
                "error": True,
                "timestamp": datetime.now()
            })

def display_chat_history():
    """Display chat history from session state."""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            format_chat_message("user", message["content"])
        else:
            tools_used = message.get("tools_used", [])
            format_chat_message("assistant", message["content"], tools_used)

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.title("🔍 Agentic JSON-RAG System")
    st.markdown("*Intelligent multi-source information retrieval with web search fallback*")
    
    # Initialize system if needed
    if not st.session_state.is_initialized:
        with st.status("🚀 Initializing system...", expanded=True) as status:
            st.write("Loading configuration...")
            if not load_configuration():
                status.update(label="❌ Initialization failed", state="error")
                display_initialization_error()
                return
            
            st.write("Setting up AI agent...")
            if not initialize_agent_manager():
                status.update(label="❌ Initialization failed", state="error")
                display_initialization_error()
                return
            
            st.session_state.is_initialized = True
            status.update(label="✅ System ready!", state="complete")
            time.sleep(1)  # Brief pause to show success
            st.rerun()
    
    # Display sidebar
    display_sidebar()
    
    # Main chat interface
    if st.session_state.is_initialized:
        # Display existing chat history
        display_chat_history()
        
        # Chat input
        query = st.chat_input("Ask me anything about your data or search the web...")
        
        if query:
            process_query(query)
            st.rerun()  # Refresh to show new messages
        
        # Help section
        with st.expander("💡 How to use this system"):
            st.markdown("""
            **This AI assistant can help you with:**
            
            1. **📚 Local Knowledge**: Search through your uploaded JSON data files
            2. **🌐 Web Search**: Find current information from the internet
            3. **🤖 Intelligent Routing**: Automatically chooses the best source for your question
            
            **Example queries:**
            - "What information do you have about [topic]?"
            - "Search for recent news about [subject]"
            - "Compare data from different sources"
            - "What's the latest information on [current event]?"
            
            **Tips:**
            - Be specific in your questions for better results
            - The system will automatically choose between local data and web search
            - Check the "Tools used" information to see which sources were consulted
            """)
    
    else:
        display_initialization_error()

if __name__ == "__main__":
    main()