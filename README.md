# Local AI Agents RAG Demo Project

A comprehensive demonstration of Retrieval-Augmented Generation (RAG) using local AI models, comparing LLM-only responses versus RAG-enhanced answers across different data sources (CSV, PDF, and web content).

## 🎯 What This Project Achieves

This project demonstrates the power of RAG by showing how local AI models can provide significantly better answers when augmented with relevant context from various data sources. Each demo compares:

- **Without RAG**: Direct LLM responses using only the model's training knowledge
- **With RAG**: Context-aware responses using retrieved documents from vector databases

The goal is to showcase practical implementation of local AI agents that can work with your own data while maintaining privacy and avoiding API costs.

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **LangChain**: Framework for building LLM applications and RAG pipelines
- **ChromaDB**: Vector database for efficient similarity search
- **Ollama**: Local LLM and embedding model server

### Key Libraries
- `langchain-ollama`: Ollama integration for LangChain
- `langchain-chroma`: ChromaDB integration
- `langchain-community`: Community-contributed LangChain components
- `pypdf`: PDF document processing
- `beautifulsoup4`: HTML parsing for web scraping
- `requests`: HTTP requests for web scraping
- `pytest`: Testing framework

### Data Sources
- **CSV**: Structured book dataset (`./csvs/book_dataset_500.csv`)
- **PDF**: Board game rulebooks (`./pdfs/`)
- **Web**: Scraped content from URLs (Wikipedia, etc.)

## 🏗️ Project Structure

```
learn_local_ai_agents_2025/
├── .github/
│   └── copilot-instructions.md    # AI agent development guidance
├── csvs/
│   └── book_dataset_500.csv       # Book data for CSV demo
├── pdfs/                          # PDF files for PDF demo
├── chroma_csv_db/                 # Vector DB for CSV data
├── pdf_chroma_db/                 # Vector DB for PDF data
├── web_chroma_db/                 # Vector DB for web data
├── csv_main.py                    # CSV RAG demo
├── csv_vector.py                  # CSV data processing & vector DB
├── pdf_main.py                    # PDF RAG demo
├── pdf_vector.py                  # PDF data processing & vector DB
├── web_main.py                    # Web scraping RAG demo
├── web_vector.py                  # Web data processing & vector DB
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

### Architecture Pattern

Each data source follows a consistent modular pattern:

- **`*_main.py`**: Demonstration script comparing RAG vs non-RAG approaches
- **`*_vector.py`**: Data ingestion, chunking, embedding, and vector database management

## 🤖 Models Used

### Language Model
- **Model**: `llama3.2:3b`
- **Provider**: Ollama (local)
- **Purpose**: Text generation for both RAG and non-RAG responses
- **Size**: ~2GB (efficient for local deployment)

### Embedding Model
- **Model**: `mxbai-embed-large:335m`
- **Provider**: Ollama (local)
- **Purpose**: Converting text chunks into vector embeddings
- **Dimensions**: 335 million parameters
- **Size**: ~335MB

### Configuration
```python
EMBEDDING_MODEL = "mxbai-embed-large:335m"
CHUNK_SIZE = 1000      # Characters per text chunk
CHUNK_OVERLAP = 200    # Overlap between chunks
```

## 🚀 Quick Start

### Prerequisites Checklist

- [ ] Python 3.8 or higher installed
- [ ] Ollama installed and running
- [ ] Required Ollama models downloaded
- [ ] Git repository cloned
- [ ] Dependencies installed

### 1. Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai)

**Windows (PowerShell):**
```powershell
# Download and install Ollama
winget install Ollama.Ollama
```

**Other platforms:**
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Download Required Models

```bash
# Download the language model
ollama pull llama3.2:3b

# Download the embedding model
ollama pull mxbai-embed-large:335m
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Ollama is Running

```bash
ollama list
```

You should see both models in the list:
```
NAME                    ID              SIZE    MODIFIED
llama3.2:3b            4aa6152e2ccb    2.0 GB  2 hours ago
mxbai-embed-large:335m 468836162de7    334 MB 2 hours ago
```

## 🎮 Running the Demos

### CSV Book Data Demo
Demonstrates RAG with structured book data from CSV.

```bash
python csv_main.py
```

**Example Query:** "Give popular books by author 'Timothy Wells' and 'Thomas Waters'"

### PDF Board Game Rules Demo
Demonstrates RAG with PDF documents containing board game rules.

```bash
python pdf_main.py
```

**Example Query:** "How many players can play CATANIC RIDE TO HEAVEN?"

### Web Scraping Demo
Demonstrates RAG with web-scraped content from URLs.

```bash
python web_main.py
```

**Example Query:** "Who identified dragon bones?"

## 🧪 Testing

### Available Tests
Currently, only PDF vector database tests are implemented:

```bash
# Run PDF vector tests
python test_pdf_vector.py

# Or with pytest
pytest test_pdf_vector.py -v
```

### Test Coverage
The test suite verifies:
- Vector database creation and persistence
- Document chunking and embedding
- Similarity search functionality
- Metadata preservation
- Retriever functionality

## 🔧 Development Workflows

### Building Vector Databases
Process data sources independently to create/update vector databases:

```bash
# Build CSV vector database
python csv_vector.py

# Build PDF vector database
python pdf_vector.py

# Build web vector database
python web_vector.py
```

### Understanding the RAG Pipeline

1. **Data Loading**: Load documents from CSV, PDF, or web sources
2. **Text Chunking**: Split documents into manageable chunks (1000 chars with 200 char overlap)
3. **Embedding Generation**: Convert chunks to vectors using `mxbai-embed-large:335m`
4. **Vector Storage**: Store embeddings in ChromaDB with metadata
5. **Query Processing**: Convert user questions to embeddings
6. **Similarity Search**: Find most relevant document chunks (k=10)
7. **Context Assembly**: Combine retrieved chunks with the original question
8. **Answer Generation**: Use LLM with context to generate informed response

## 📊 Performance Characteristics

### Vector Database Sizes
- **CSV Database**: ~500 book records → small, fast queries
- **PDF Database**: Variable size based on PDF content → medium performance
- **Web Database**: Variable size based on scraped pages → medium performance

### Query Performance
- **Embedding Generation**: ~1-2 seconds per query
- **Similarity Search**: <100ms for k=10 results
- **LLM Generation**: 2-5 seconds per response

### Memory Usage
- **Ollama Models**: ~2.3GB total (LLM + embeddings)
- **Vector Databases**: 100MB - 1GB depending on data size
- **Python Process**: ~500MB during operation

## 🐛 Troubleshooting

### Common Issues

**"Model not found" errors:**
```bash
# Ensure Ollama is running and models are downloaded
ollama list
ollama pull llama3.2:3b
ollama pull mxbai-embed-large:335m
```

**Web scraping failures:**
- Some websites block automated requests
- Try Wikipedia URLs which are usually scraping-friendly
- Check your internet connection

**PDF processing issues:**
- Ensure PDFs are not password-protected
- Check file permissions
- Verify PyPDF compatibility

**Vector database errors:**
- Delete the relevant `*_chroma_db/` directory and rebuild
- Check available disk space
- Ensure write permissions

### Debug Mode
Add debug prints to see what's happening:

```python
# In any *_main.py file, add before running:
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

### Adding New Data Sources
1. Create `*_vector.py` for data processing and vector DB management
2. Create `*_main.py` for RAG vs non-RAG comparison demo
3. Follow the established patterns for consistency
4. Add appropriate tests
5. Update this README

### Code Style
- Follow existing naming conventions (`*_vector.py`, `*_main.py`)
- Use consistent configuration variables
- Include comprehensive error handling
- Add docstrings and comments

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [RAG Paper](https://arxiv.org/abs/2005.11401) (original research)

## 📄 License

This project is for educational and demonstration purposes. Please check individual component licenses for production use.

---

**Happy RAG-ing! 🚀**