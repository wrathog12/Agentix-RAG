"""
Agentic RAG System - A robust multi-source information retrieval system.

This package provides:
- Intelligent agent-based document retrieval
- Multi-format JSON data processing  
- Web search fallback capabilities
- Comprehensive error handling
- Streamlit web interface
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .config import Config
from .agents import AgentManager, create_agent
from .tools import build_json_index, create_web_search_tool

__all__ = [
    "Config",
    "AgentManager", 
    "create_agent",
    "build_json_index",
    "create_web_search_tool"
]