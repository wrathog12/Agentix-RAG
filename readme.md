# 🔍 Agentic JSON-RAG System

A robust, intelligent multi-source information retrieval system that combines local JSON data with web search capabilities. The system uses an AI agent to automatically choose the best information source for each query.

## ✨ Features

- **🤖 Intelligent Agent**: Automatically routes queries to the most appropriate data source
- **📚 Multi-Source RAG**: Processes multiple JSON files into searchable vector databases
- **🌐 Web Search Fallback**: Uses SerpAPI for real-time web information when local data is insufficient
- **🛡️ Robust Error Handling**: Comprehensive error handling and recovery mechanisms
- **💾 Smart Caching**: Efficient caching system to improve performance
- **🖥️ Modern UI**: Clean Streamlit interface with real-time status monitoring
- **📊 Source Tracking**: Shows which data sources were used for each response

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   Agent Manager  │    │   Tool Manager  │
│                 │────│                  │────│                 │
│  - Chat Interface│    │  - Query Routing │    │  - JSON Indexing│
│  - Status Monitor│    │  - Tool Selection│    │  - Web Search   │
│  - Error Display │    │  - Response Gen. │    │  - Caching      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- HuggingFace API Token
- SerpAPI Key (for web search)

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd agentic-rag-system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Prepare your data:**
   - Create a `data/` directory
   - Add your JSON files to the directory
   - Supported formats:
     ```json
     // Array format
     [{"content": "...", "metadata": "..."}, ...]
     
     // Object format
     {"records": [{"content": "...", "metadata": "..."}, ...]}
     ```

## 🎯 Usage

### Running the Application

```bash
streamlit run main.py
```

### Using the System

1. **Ask questions** about your local data
2. **Search the web** for current information
3. **Let the agent decide** which source is best for your query

### Example Queries

- `"What information do you have about machine learning?"`
- `"Search for recent AI developments"`
- `"Compare data from different sources"`
- `"What's the latest news on climate change?"`

## 📁 Project Structure

```
agentic-rag-system/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration management
│   ├── tools.py             # Core tools and utilities
│   └── agents.py            # Agent management
├── data/                    # Your JSON data files
├── main.py                  # Streamlit application
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

## ⚙️ Configuration

The system uses a hierarchical configuration system:

### Environment Variables
```bash
HUGGINGFACEHUB_API_TOKEN=your_token
SERPAPI_API_KEY=your_key
```

### Data Configuration
- **Location**: `data/` directory (configurable)
- **Formats**: JSON arrays or objects with `records` key
- **Content**: Supports `markdown`, `html`, `content`, or `text` fields

### Model Configuration
- **LLM**: Falcon-7B-Instruct (configurable)
- **Embeddings**: all-MiniLM-L6-v2 (configurable)
- **Parameters**: Temperature, max tokens, etc.

## 🔧 Advanced Features

### Custom Data Sources

To add new data sources, modify the `DataConfig` class:

```python
@dataclass
class DataConfig:
    data_dir: str = "data"
    json_files: List[str] = field(default_factory=lambda: [
        "your_file1.json",
        "your_file2.json"
    ])
```

### Custom Models

Update the `ModelConfig` class:

```python
@dataclass 
class ModelConfig:
    llm_model: str = "your-preferred-model"
    embedding_model: str = "your-embedding-model"
```

### Performance Tuning

Adjust RAG parameters:

```python
@dataclass
class RAGConfig:
    chunk_size: int = 500          # Chunk size for text splitting
    chunk_overlap: int = 50        # Overlap between chunks
    retrieval_k: int = 5           # Number of documents to retrieve
    similarity_threshold: float = 0.7  # Minimum similarity score
```

## 🛡️ Error Handling

The system includes comprehensive error handling for:

- **API Failures**: Graceful degradation when APIs are unavailable
- **Data Issues**: Handles malformed or missing JSON files
- **Memory Management**: Automatic cleanup and caching
- **Network Problems**: Timeout handling and retries
- **User Input**: Validates and sanitizes all inputs

## 📊 Monitoring

The system provides real-time monitoring:

- **System Status**: Current operational state
- **Query Statistics**: Number of processed queries
- **Performance Metrics**: Response times and resource usage
- **Data Sources**: Available files and their status
- **Tool Usage**: Which tools were used for each query

## 🔍 Troubleshooting

### Common Issues

1. **API Token Errors**
   - Verify your tokens are correct
   - Check token permissions and quotas

2. **Data Loading Issues**
   - Ensure JSON files are valid
   - Check file permissions and paths

3. **Memory Issues**
   - Use the "Clear Cache" button
   - Reduce chunk size or retrieval parameters

4. **Slow Performance**
   - Check internet connection
   - Verify API service status

### Debug Mode

Enable verbose logging by setting:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain**: For the agent framework
- **Streamlit**: For the web interface
- **HuggingFace**: For model hosting
- **SerpAPI**: For web search capabilities
- **FAISS**: For vector similarity search

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the error logs
3. Open an issue on GitHub
4. Contact the maintainers

---

*Built with ❤️ for intelligent information retrieval*