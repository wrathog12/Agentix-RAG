"""
Improved agents module with better error handling and agent management.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.agents.agent import AgentExecutor
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.base import BaseCallbackHandler

from .tools import JSONIndexBuilder, WebSearchTool, ToolError
from .config import Config

logger = logging.getLogger(__name__)

class AgentError(Exception):
    """Custom exception for agent-related errors."""
    pass

class SourceTrackingCallback(BaseCallbackHandler):
    """Callback to track which tools are being used by the agent."""
    
    def __init__(self):
        self.last_used_tools = []
        self.current_tools = []
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Track when a tool is used."""
        tool_name = action.tool
        self.current_tools.append(tool_name)
        logger.info(f"Agent using tool: {tool_name}")
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Update tracking when agent finishes."""
        self.last_used_tools = self.current_tools.copy()
        self.current_tools = []
    
    def get_last_used_tools(self) -> List[str]:
        """Get the tools used in the last query."""
        return self.last_used_tools

class LLMManager:
    """Manages language model instances with error handling."""
    
    _instance = None
    _llm = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_llm(self, config: Config) -> HuggingFaceEndpoint:
        """Get cached LLM instance."""
        if self._llm is None:
            try:
                api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
                if not api_token:
                    raise AgentError("HUGGINGFACEHUB_API_TOKEN not found")
                
                self._llm = HuggingFaceEndpoint(
                    model=config.model.llm_model,
                    huggingfacehub_api_token=api_token,
                    temperature=config.model.temperature,
                    max_new_tokens=config.model.max_new_tokens,
                    timeout=60,  # Add timeout
                )
                logger.info(f"Initialized LLM: {config.model.llm_model}")
                
            except Exception as e:
                raise AgentError(f"Failed to initialize LLM: {str(e)}")
                
        return self._llm

class SafeRetrievalTool:
    """Wrapper for retrieval tools with error handling."""
    
    def __init__(self, index, tool_name: str, config: Config):
        self.index = index
        self.tool_name = tool_name
        self.config = config
        
    def search(self, query: str) -> str:
        """Perform retrieval with error handling."""
        try:
            if not query or query.strip() == "":
                return f"Empty query provided to {self.tool_name}."
            
            # Get retriever with similarity threshold
            retriever = self.index.as_retriever(
                search_kwargs={
                    "k": self.config.rag.retrieval_k,
                    "score_threshold": self.config.rag.similarity_threshold
                }
            )
            
            docs = retriever.get_relevant_documents(query.strip())
            
            if not docs:
                return f"No relevant information found in {self.tool_name}."
            
            # Format results
            results = []
            for i, doc in enumerate(docs, 1):
                content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                source = doc.metadata.get("source_file", "unknown")
                results.append(f"Result {i} from {source}:\n{content}\n")
            
            return "\n".join(results)
            
        except Exception as e:
            error_msg = f"Error searching {self.tool_name}: {str(e)}"
            logger.error(error_msg)
            return f"Sorry, I couldn't search {self.tool_name} right now. {error_msg}"

class AgentManager:
    """Manages the creation and lifecycle of agents."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.llm_manager = LLMManager()
        self.index_builder = JSONIndexBuilder(self.config)
        self.web_search_tool = WebSearchTool()
        self.source_tracker = SourceTrackingCallback()
        self._agent = None
        
    def _create_retrieval_tools(self) -> List[Tool]:
        """Create retrieval tools from available data files."""
        tools = []
        descriptions = self.config.get_tool_descriptions()
        
        try:
            data_paths = self.config.get_data_paths()
            
            for i, path in enumerate(data_paths, 1):
                try:
                    # Build index for this file
                    index = self.index_builder.build_index(path)
                    
                    # Create safe retrieval wrapper
                    tool_name = f"Topic{i}Info"
                    retrieval_tool = SafeRetrievalTool(index, tool_name, self.config)
                    
                    # Create LangChain tool
                    tool = Tool(
                        name=tool_name,
                        func=retrieval_tool.search,
                        description=descriptions.get(tool_name, f"Search information from {path.name}")
                    )
                    
                    tools.append(tool)
                    logger.info(f"Created tool: {tool_name} for {path.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to create tool for {path}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to create retrieval tools: {str(e)}")
            
        return tools
    
    def _create_web_tool(self) -> Tool:
        """Create web search tool."""
        return Tool(
            name="WebSearch",
            func=self.web_search_tool.search,
            description=self.config.get_tool_descriptions()["WebSearch"]
        )
    
    def create_agent(self) -> AgentExecutor:
        """Create agent with all tools and error handling."""
        try:
            # Get LLM
            llm = self.llm_manager.get_llm(self.config)
            
            # Create tools
            retrieval_tools = self._create_retrieval_tools()
            web_tool = self._create_web_tool()
            
            all_tools = retrieval_tools + [web_tool]
            
            if not all_tools:
                raise AgentError("No tools available for agent")
            
            # Create agent
            agent = initialize_agent(
                tools=all_tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=3,  # Limit iterations to prevent infinite loops
                early_stopping_method="generate",
                callbacks=[self.source_tracker],
                handle_parsing_errors=True,  # Handle parsing errors gracefully
            )
            
            logger.info(f"Created agent with {len(all_tools)} tools")
            self._agent = agent
            return agent
            
        except Exception as e:
            raise AgentError(f"Failed to create agent: {str(e)}")
    
    def get_agent(self) -> AgentExecutor:
        """Get cached agent or create new one."""
        if self._agent is None:
            self._agent = self.create_agent()
        return self._agent
    
    def get_last_used_tools(self) -> List[str]:
        """Get tools used in the last query."""
        return self.source_tracker.get_last_used_tools()
    
    def run_with_error_handling(self, query: str) -> Tuple[str, List[str]]:
        """Run agent query with comprehensive error handling."""
        if not query or query.strip() == "":
            return "Please provide a valid question.", []
        
        try:
            agent = self.get_agent()
            
            # Run the agent
            response = agent.run(query.strip())
            used_tools = self.get_last_used_tools()
            
            logger.info(f"Query processed successfully. Tools used: {used_tools}")
            return response, used_tools
            
        except Exception as e:
            error_msg = f"I encountered an error while processing your question: {str(e)}"
            logger.error(f"Agent run failed: {str(e)}")
            return error_msg, []
    
    def clear_cache(self):
        """Clear all caches to free memory."""
        self.index_builder.clear_cache()
        self._agent = None
        logger.info("Cleared agent caches")

# Factory function for backward compatibility
def create_agent(config: Config = None) -> AgentExecutor:
    """Create agent with default configuration."""
    manager = AgentManager(config)
    return manager.create_agent()

# Convenience function for loading LLM
def load_llm(config: Config = None) -> HuggingFaceEndpoint:
    """Load LLM with default configuration."""
    if config is None:
        config = Config()
    return LLMManager().get_llm(config)