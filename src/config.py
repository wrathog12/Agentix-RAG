"""
Configuration management for the Agentic RAG system.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for language models and embeddings."""
    llm_model: str = "tiiuae/falcon-7b-instruct"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    temperature: float = 0.7
    max_new_tokens: int = 512
    
@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 5
    similarity_threshold: float = 0.7
    
@dataclass
class DataConfig:
    """Configuration for data sources."""
    data_dir: str = "data"
    json_files: List[str] = None
    
    def __post_init__(self):
        if self.json_files is None:
            self.json_files = [
                "topic(1).json",
                "scrape_results_20250526_124734.json", 
                "scrape_results_20250526_140158.json"
            ]

class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.rag = RAGConfig()
        self.data = DataConfig()
        self._validate_environment()
        
    def _validate_environment(self) -> None:
        """Validate that required environment variables are set."""
        required_vars = [
            "HUGGINGFACEHUB_API_TOKEN",
            "SERPAPI_API_KEY"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
                
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}\n"
                "Please set them in your .env file or environment."
            )
            
    def get_data_paths(self) -> List[Path]:
        """Get validated data file paths."""
        data_dir = Path(self.data.data_dir)
        paths = []
        
        for filename in self.data.json_files:
            file_path = data_dir / filename
            if file_path.exists():
                paths.append(file_path)
                logger.info(f"Found data file: {file_path}")
            else:
                logger.warning(f"Data file not found: {file_path}")
                
        if not paths:
            raise FileNotFoundError(
                f"No valid data files found in {data_dir}. "
                f"Expected files: {self.data.json_files}"
            )
            
        return paths
        
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get tool descriptions based on available data files."""
        descriptions = {}
        
        for i, filename in enumerate(self.data.json_files, 1):
            file_path = Path(self.data.data_dir) / filename
            if file_path.exists():
                # Generate description based on filename
                if "topic" in filename.lower():
                    desc = f"Search through topic-specific information in {filename}"
                elif "scrape" in filename.lower():
                    desc = f"Search through scraped web content from {filename}"
                else:
                    desc = f"Search through data from {filename}"
                    
                descriptions[f"Topic{i}Info"] = desc
                
        descriptions["WebSearch"] = (
            "Search the web for current information when local knowledge "
            "sources don't contain relevant information."
        )
        
        return descriptions