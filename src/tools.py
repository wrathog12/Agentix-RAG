"""
Improved tools module with comprehensive error handling and caching.
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from functools import lru_cache

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.utilities import SerpAPIWrapper

from .config import Config

logger = logging.getLogger(__name__)

class ToolError(Exception):
    """Custom exception for tool-related errors."""
    pass

class EmbeddingManager:
    """Manages embedding models with caching and error handling."""
    
    _instance = None
    _embeddings = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_embeddings(self, config: Config) -> HuggingFaceInferenceAPIEmbeddings:
        """Get cached embeddings instance."""
        if self._embeddings is None:
            try:
                self._embeddings = HuggingFaceInferenceAPIEmbeddings(
                    model_name=config.model.embedding_model,
                    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
                )
                logger.info(f"Initialized embeddings: {config.model.embedding_model}")
            except Exception as e:
                raise ToolError(f"Failed to initialize embeddings: {str(e)}")
                
        return self._embeddings

class JSONIndexBuilder:
    """Builds and manages FAISS indexes from JSON files."""
    
    def __init__(self, config: Config):
        self.config = config
        self.embedding_manager = EmbeddingManager()
        self._index_cache = {}
        
    def _get_cache_key(self, json_path: Path) -> str:
        """Generate cache key based on file path and modification time."""
        try:
            stat = json_path.stat()
            content = f"{json_path.absolute()}_{stat.st_mtime}_{stat.st_size}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return str(json_path.absolute())
    
    def _load_json_safely(self, json_path: Path) -> List[Dict[str, Any]]:
        """Safely load and parse JSON file."""
        try:
            if not json_path.exists():
                raise FileNotFoundError(f"JSON file not found: {json_path}")
                
            if json_path.stat().st_size == 0:
                raise ValueError(f"JSON file is empty: {json_path}")
                
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle different JSON structures
            if isinstance(data, dict):
                if "records" in data and isinstance(data["records"], list):
                    items = data["records"]
                elif "data" in data and isinstance(data["data"], list):
                    items = data["data"]
                else:
                    # Try to find any list in the dict
                    list_values = [v for v in data.values() if isinstance(v, list)]
                    if list_values:
                        items = list_values[0]
                    else:
                        items = [data]  # Treat the dict as a single item
            elif isinstance(data, list):
                items = data
            else:
                items = [data]  # Wrap single item in list
                
            if not items:
                raise ValueError(f"No data items found in {json_path}")
                
            logger.info(f"Loaded {len(items)} items from {json_path}")
            return items
            
        except json.JSONDecodeError as e:
            raise ToolError(f"Invalid JSON in {json_path}: {str(e)}")
        except Exception as e:
            raise ToolError(f"Failed to load {json_path}: {str(e)}")
    
    def _create_documents(self, items: List[Dict[str, Any]], source_file: str) -> List[Document]:
        """Convert JSON items to LangChain Documents."""
        documents = []
        
        for i, item in enumerate(items):
            try:
                # Extract content with fallback hierarchy
                content = (
                    item.get("markdown") or 
                    item.get("html") or 
                    item.get("content") or 
                    item.get("text") or 
                    str(item)
                )
                
                # Ensure content is not empty
                if not content or content.strip() == "":
                    content = f"Item {i} from {source_file}"
                    
                # Create metadata
                metadata = item.copy() if isinstance(item, dict) else {"raw_data": item}
                metadata.update({
                    "source_file": source_file,
                    "item_index": i,
                    "content_length": len(content)
                })
                
                documents.append(Document(page_content=content, metadata=metadata))
                
            except Exception as e:
                logger.warning(f"Failed to process item {i} from {source_file}: {str(e)}")
                continue
                
        logger.info(f"Created {len(documents)} documents from {source_file}")
        return documents
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.rag.chunk_size,
                chunk_overlap=self.config.rag.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = splitter.split_documents(documents)
            logger.info(f"Split into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            raise ToolError(f"Failed to split documents: {str(e)}")
    
    def build_index(self, json_path: Path) -> FAISS:
        """Build FAISS index from JSON file with caching."""
        cache_key = self._get_cache_key(json_path)
        
        # Check cache first
        if cache_key in self._index_cache:
            logger.info(f"Using cached index for {json_path.name}")
            return self._index_cache[cache_key]
        
        try:
            # Load and process data
            items = self._load_json_safely(json_path)
            documents = self._create_documents(items, json_path.name)
            
            if not documents:
                raise ToolError(f"No valid documents created from {json_path}")
                
            chunks = self._split_documents(documents)
            
            # Build FAISS index
            embeddings = self.embedding_manager.get_embeddings(self.config)
            index = FAISS.from_documents(chunks, embeddings)
            
            # Cache the index
            self._index_cache[cache_key] = index
            logger.info(f"Built and cached index for {json_path.name}")
            
            return index
            
        except Exception as e:
            raise ToolError(f"Failed to build index for {json_path}: {str(e)}")
    
    def clear_cache(self):
        """Clear the index cache to free memory."""
        self._index_cache.clear()
        logger.info("Cleared index cache")

class WebSearchTool:
    """Wrapper for web search with error handling."""
    
    def __init__(self):
        self._search_wrapper = None
        self._initialize_search()
    
    def _initialize_search(self):
        """Initialize SerpAPI wrapper with error handling."""
        try:
            api_key = os.getenv("SERPAPI_API_KEY")
            if not api_key:
                raise ToolError("SERPAPI_API_KEY not found in environment")
                
            self._search_wrapper = SerpAPIWrapper(
                params={
                    "engine": "google",
                    "api_key": api_key,
                    "num": 5,  # Limit results
                }
            )
            logger.info("Initialized web search tool")
            
        except Exception as e:
            raise ToolError(f"Failed to initialize web search: {str(e)}")
    
    def search(self, query: str) -> str:
        """Perform web search with error handling."""
        if not query or query.strip() == "":
            return "Empty search query provided."
            
        try:
            if self._search_wrapper is None:
                self._initialize_search()
                
            result = self._search_wrapper.run(query.strip())
            logger.info(f"Web search completed for query: {query[:50]}...")
            return result
            
        except Exception as e:
            error_msg = f"Web search failed: {str(e)}"
            logger.error(error_msg)
            return f"Sorry, I couldn't search the web right now. {error_msg}"

# Factory functions for backward compatibility and ease of use
def get_hf_embedding(config: Config = None) -> HuggingFaceInferenceAPIEmbeddings:
    """Get HuggingFace embeddings instance."""
    if config is None:
        config = Config()
    return EmbeddingManager().get_embeddings(config)

def build_json_index(json_path: Union[str, Path], config: Config = None) -> FAISS:
    """Build FAISS index from JSON file."""
    if config is None:
        config = Config()
    
    path = Path(json_path) if isinstance(json_path, str) else json_path
    builder = JSONIndexBuilder(config)
    return builder.build_index(path)

def create_web_search_tool() -> WebSearchTool:
    """Create web search tool instance."""
    return WebSearchTool()