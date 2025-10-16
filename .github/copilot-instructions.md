# AI Agent Instructions for Local RAG Demo Project

## Project Overview
This is a Retrieval-Augmented Generation (RAG) demonstration project comparing LLM-only responses vs. RAG-enhanced answers using CSV, PDF, and web data sources. The project uses LangChain, ChromaDB, and Ollama for local AI processing.

## Architecture Patterns

### Modular Data Pipeline
Each data source follows a consistent `*_main.py` + `*_vector.py` pattern:
- **Main modules** (`csv_main.py`, `pdf_main.py`, `web_main.py`): Demonstrate RAG vs non-RAG comparison
- **Vector modules** (`csv_vector.py`, `pdf_vector.py`, `web_vector.py`): Handle ingestion, chunking, embedding, and storage

### Vector Database Management
- **Incremental updates**: Existing ChromaDB instances are loaded and only new content is added
- **Persistent storage**: Separate DB directories (`chroma_csv_db/`, `pdf_chroma_db/`, `web_chroma_db/`)
- **Auto-update logic**: Checks existing sources before processing new data

### Consistent Configuration
```python
EMBEDDING_MODEL = "mxbai-embed-large:335m"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

## Key Implementation Patterns

### Global State Management
Vector DB and retriever instances are stored as global variables for reuse across function calls:
```python
vectordb_web = web_vector.create_vector_db_from_urls(urls=[], force_rebuild=False)
retriever_web = vectordb_web.as_retriever(search_type="similarity", search_kwargs={"k": 10})
```

### Document Processing Pipeline
1. **Load**: Use appropriate LangChain loaders (`WebBaseLoader`, `PyPDFLoader`, `DirectoryLoader`)
2. **Chunk**: Apply `RecursiveCharacterTextSplitter` with consistent size/overlap parameters
3. **Embed**: Generate embeddings using Ollama's `mxbai-embed-large:335m` model
4. **Store**: Persist in ChromaDB with metadata preservation

### Error Handling
- Graceful degradation when data sources are unavailable
- User-friendly error messages with troubleshooting tips
- Validation of file paths and network connectivity

## Development Workflows

### Running Demos
```bash
# Compare RAG vs non-RAG for different data types
python csv_main.py   # Book data from CSV
python pdf_main.py   # Board game rules from PDFs
python web_main.py   # Web-scraped dinosaur information
```

### Building Vector Databases
```bash
# Process data sources independently
python csv_vector.py  # Books CSV → embeddings
python pdf_vector.py  # PDFs in ./pdfs/ → embeddings
python web_vector.py  # URLs → embeddings
```

## Integration Points

### External Dependencies
- **Ollama**: Must be running locally with `llama3.2:3b` and `mxbai-embed-large:335m` models
- **Data sources**: CSV files in `./csvs/`, PDFs in `./pdfs/`, web URLs
- **ChromaDB**: Persistent vector storage with SQLite backend

### LangChain Components
- `OllamaLLM` for text generation
- `OllamaEmbeddings` for vector creation
- `Chroma` for vector database operations
- Various document loaders for different data types

## Code Conventions

### Naming Patterns
- Data-specific prefixes: `csv_*`, `pdf_*`, `web_*`
- Consistent function signatures across vector modules
- Descriptive variable names (`vectordb_web`, `retriever_pdf`)

### Metadata Handling
- Preserve source information in document metadata
- Include page numbers for PDFs, URLs for web content
- Structured metadata for CSV data (title, author, rating, etc.)

### Prompt Engineering
- Domain-specific system prompts ("You are a book expert", "You are a knowledgeable assistant")
- Consistent RAG prompt structure with retrieved document inclusion
- Clear separation between context and question

## Common Pitfalls to Avoid

### Data Source Issues
- Web scraping may fail due to anti-bot measures (prefer Wikipedia URLs)
- PDF loading requires PyPDF2/pypdf compatibility
- CSV parsing needs flexible column name handling

### Performance Considerations
- Embedding generation is computationally expensive - reuse existing DBs when possible
- Large documents are chunked to fit context windows
- Similarity search uses `k=10` for balanced retrieval vs. latency

### State Management
- Global vector DB instances prevent redundant loading
- Force rebuild only when data schema changes
- Auto-update prevents duplicate processing