# Web RAG Demo Flow Diagram

## Complete Flow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WEB RAG DEMO INITIALIZATION                      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │  Import Dependencies     │
                    │  Load LLM Model          │
                    │  (llama3.2:3b)          │
                    └──────────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ Vector DB Exists?        │
                    └──────────────────────────┘
                          │              │
                ┌─────────┘              └─────────┐
                │                                  │
                ▼ YES                              ▼ NO
        ┌──────────────────┐         ┌────────────────────────┐
        │ Load Existing DB │         │ Create New Database    │
        │ Check for new    │         │ (web_vector.py steps)  │
        │ URLs & update    │         │ Scrape URLs            │
        └──────────────────┘         └────────────────────────┘
                │                                  │
                └─────────────────┬────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────────┐
                    │ Create Retriever             │
                    │ (similarity search, k=10)    │
                    └──────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│                         USER QUERY INPUT                             │
│            "Who identified dragon bones?"                           │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │  Choose Approach         │
                    └──────────────────────────┘
                          │              │
                ┌─────────┘              └─────────┐
                │                                  │
                ▼                                  ▼
        ┌──────────────────┐        ┌──────────────────────┐
        │   WITHOUT RAG    │        │     WITH RAG         │
        │                  │        │                      │
        │ Direct LLM Query │        │ 1. Retrieve Chunks   │
        │ (LLM only uses   │        │ 2. Build Context     │
        │  training data)  │        │ 3. LLM with Context  │
        └──────────────────┘        └──────────────────────┘
                │                                  │
                ▼                                  ▼
        ┌──────────────────────┐    ┌──────────────────────────┐
        │  LLM-Only Response   │    │   RAG-Enhanced Response  │
        │ (General knowledge   │    │   (Fact-based, accurate) │
        │  may be inaccurate)  │    │                          │
        └──────────────────────┘    └──────────────────────────┘
                │                                  │
                └─────────────────┬────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │  Display Both Answers    │
                    │  for Comparison          │
                    └──────────────────────────┘
```

---

## STEP 1: Web Vector Database Creation (web_vector.py)

### 1.1 Check if Vector DB Already Exists
```
┌──────────────────────────────────┐
│  Does web_chroma_db/ exist?      │
└──────────────────────────────────┘
         │                    │
      YES│                    │NO
         │                    │
         ▼                    ▼
    Load DB              Scrape URLs
    (from disk)          from the internet
    Check for new            │
    URLs & update        ✓ Fetches web pages
         │               ✓ Extracts text
         │               ✓ Handles HTML/JS
         └────────┬───────────┘
                  │
              ✓ Faster (reuse existing data)
              ✓ Auto-update with new URLs
```

**Code:**
```python
# web_vector.py - Lines 79-110
if os.path.exists(persist_directory) and not force_rebuild:
    print(f"Loading existing vector DB from {persist_directory}")
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    existing_count = vectordb._collection.count()
    print(f"✓ Loaded existing DB with {existing_count} chunks")
    
    if auto_update:
        existing_sources = get_existing_sources(vectordb)
        new_urls = [url for url in urls if url not in existing_sources]
        
        if new_urls:
            print(f"\n🔍 Found {len(new_urls)} new URL(s) to scrape")
```

---

### 1.2 Fetch Web Pages from URLs
```
URLs List
   │
   ├─► https://en.wikipedia.org/wiki/Dinosaur
   ├─► https://en.wikipedia.org/wiki/Paleontology
   ├─► https://en.wikipedia.org/wiki/Fossil
   │
   ▼
WebBaseLoader
   │
   ├─► Sends HTTP requests to URLs
   ├─► Fetches HTML content
   ├─► Extracts readable text
   ├─► Handles JavaScript content
   ├─► Cleans up formatting
   │
   ▼
List of Document Objects
   │
   ├─► Document 1: Wikipedia page on Dinosaur
   ├─► Document 2: Wikipedia page on Paleontology
   ├─► Document 3: Wikipedia page on Fossil
   │
   ▼
Total: N pages loaded
```

**Code:**
```python
# web_vector.py - Lines 12-28
def load_webpages(urls):
    """Load web pages using LangChain's WebBaseLoader."""
    try:
        print(f"Loading {len(urls)} URLs with WebBaseLoader...")
        loader = WebBaseLoader(urls)
        documents = loader.load()
        print(f"✓ Successfully loaded {len(documents)} pages")
        return documents
    except Exception as e:
        print(f"❌ Error loading URLs: {e}")
        return []
```

**Example Output:**
```
Loading 1 URLs with WebBaseLoader...
✓ Successfully loaded 1 pages
```

---

### 1.3 Chunk Web Content
```
Web Page Content
   │
   "Dinosaurs are extinct reptiles that lived
    millions of years ago. Fossils show...
    The first dinosaur bones were identified
    by Mary Anning in the 1800s..."
   │
   ▼
RecursiveCharacterTextSplitter
   │
   ├─ Chunk size: 1000 characters
   ├─ Overlap: 200 characters
   ├─ Split intelligently at:
   │  ├─ Double newlines (paragraphs)
   │  ├─ Single newlines (sentences)
   │  ├─ Spaces (words)
   │  └─ Characters (fallback)
   │
   ▼
Multiple Chunks
   │
   ├─► Chunk 1: "Dinosaurs are extinct..." (overlap ends here)
   ├─► Chunk 2: "...from 1800s. The fossil record..." (overlap starts from chunk 1)
   ├─► Chunk 3: "record shows dragon bones..."
   │
   ▼
Chunked Documents (overlap preserves context)
```

**Code:**
```python
# web_vector.py - Lines 30-47
def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)
```

**Example:**
- 1 Wikipedia page → ~15-30 chunks
- Each chunk: max 1000 characters
- Overlap: 200 characters between chunks

---

### 1.4 Initialize Embedding Model
```
┌────────────────────────────────┐
│ Ollama Embedding Model         │
│ mxbai-embed-large:335m         │
└────────────────────────────────┘
         │
         ▼
    Ready to convert web chunks
    to 335-dimensional vectors
```

**Code:**
```python
# web_vector.py - Lines 138-139
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
print("✓ Embedding model initialized")
```

---

### 1.5 Generate Embeddings and Store in ChromaDB
```
Chunked Web Content: "Dinosaurs are extinct..."
   │
   ▼
Embedding Model
   │
   ▼
Vector: [0.234, -0.156, 0.890, ..., 0.123]  (335 dimensions)
   │
   ├─► Store in ChromaDB
   ├─► Link to metadata (source URL)
   └─► Save to disk (web_chroma_db/)

[Repeat for all chunks]
   │
   ▼
ChromaDB fully populated with vectors
Ready for similarity search
```

**Code:**
```python
# web_vector.py - Lines 141-147
vectordb = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    persist_directory=persist_directory
)
print(f"✓ Vector DB created and saved to {persist_directory}")
print(f"  Total chunks embedded: {len(chunked_docs)}")
```

**Metadata Stored:**
```python
{
  "source": "https://en.wikipedia.org/wiki/Dinosaur",
  "title": "Dinosaur - Wikipedia"
}
```

---

## STEP 2: Query Processing - Without RAG (web_main.py)

```
User Question
  │
  "Who identified dragon bones?"
  │
  ▼
┌──────────────────────────────────┐
│ LLM Model (llama3.2:3b)          │
│ WITHOUT any web content context  │
└──────────────────────────────────┘
  │
  ├─► Uses only training data
  ├─► No access to actual web pages
  ├─► May have outdated information
  └─► Limited historical accuracy
  │
  ▼
"Dragon bones were likely identified 
by ancient Chinese scholars who 
confused them with mythical dragons. 
Modern paleontologists later..."
(Vague, possibly inaccurate)
```

**Code:**
```python
# web_main.py - Lines 15-20
def ask_without_rag(question: str) -> str:
    """Ask the LLM directly without vector database context."""
    prompt = f"You are a knowledgeable assistant. Answer the following question:\n\nQuestion: {question}\n\nAnswer:"
    return model.invoke(prompt)
```

---

## STEP 3: Query Processing - With RAG (web_main.py)

### 3.1 Retrieve Relevant Web Content Chunks
```
User Question
  │
  "Who identified dragon bones?"
  │
  ▼
Convert to Embedding
(same model: mxbai-embed-large:335m)
  │
  ▼
Vector: [0.145, -0.234, 0.567, ..., 0.890]
  │
  ▼
┌──────────────────────────────────┐
│ ChromaDB Similarity Search        │
│ Find top 10 most similar chunks  │
└──────────────────────────────────┘
  │
  ├─► Chunk 1: "Mary Anning identified fossils called dragon bones..." - Similarity: 0.98
  ├─► Chunk 2: "In the 1800s paleontology emerged..." - Similarity: 0.93
  ├─► Chunk 3: "Dragon bones were actually dinosaur bones..." - Similarity: 0.91
  ├─► ... (more chunks)
  │
  ▼
Return top 10 chunks with metadata
   (source URL, exact text)
```

**Code:**
```python
# web_main.py - Lines 35-36
retrieved_docs = retriever_web.invoke(question)
```

---

### 3.2 Build Context Prompt with Retrieved Web Chunks
```
Retrieved Chunks (10 web page sections)
  │
  ├─► Chunk 1: Historical facts from Wikipedia
  ├─► Chunk 2: Scientific discovery info
  ├─► Chunk 3: Paleontology details
  ├─► ... (more chunks with actual web content)
  │
  ▼
Format into prompt:
  │
  "You are a knowledgeable assistant.
   
   Here are the relevant web pages:
   
   === Web Page 1 ===
   Mary Anning was a pioneer of paleontology
   who discovered many fossils in the 1800s.
   Fossils that were originally called 'dragon bones'
   in ancient times were identified as...
   
   === Web Page 2 ===
   The term 'dragon bones' was used historically
   to refer to what we now know as dinosaur fossils.
   Modern paleontologists identified these...
   
   Question: Who identified dragon bones?
   
   Provide a clear answer based on the web content above:"
  │
  ▼
Full context-rich prompt ready for LLM
```

**Code:**
```python
# web_main.py - Lines 40-46
prompt = f"You are a knowledgeable assistant. Answer the following question based on the web content provided.\n\n"
prompt += f"Here are the relevant web pages:\n\n"

for i, doc in enumerate(retrieved_docs, 1):
    prompt += f"=== Web Page {i} ===\n{doc.page_content}\n\n"

prompt += f"Question: {question}\n\nProvide a clear answer based on the web content above:"
```

---

### 3.3 Generate RAG-Enhanced Answer
```
LLM Model (with web content)
  │
  ├─► Sees actual web page content
  ├─► Can reference Wikipedia facts
  ├─► Provides accurate, sourced information
  └─► Backed by current web sources
  │
  ▼
┌──────────────────────────────────────┐
│ RAG-ENHANCED RESPONSE                │
└──────────────────────────────────────┘
  │
  "Based on the web content provided:
  
  Mary Anning, a pioneering paleontologist of the 1800s,
  was instrumental in identifying fossils that were
  historically referred to as 'dragon bones'. These were
  later scientifically identified as dinosaur fossils.
  
  The term 'dragon bones' originated from ancient times
  when these fossils were misidentified as belonging to
  mythical dragons before modern paleontology emerged."
  │
  ▼
(Accurate! Sourced from Wikipedia!)
```

---

## STEP 4: Web Data Pipeline Detailed View

### Data Transformation Pipeline
```
URLs to Scrape
   │
   ├─► https://en.wikipedia.org/wiki/Dinosaur
   ├─► https://en.wikipedia.org/wiki/Paleontology
   │
   ▼
Step 1: Fetch Web Pages
   │
   ├─► Send HTTP requests
   ├─► Extract readable text from HTML
   ├─► 2-3 pages loaded
   │
   ▼
Step 2: Chunk Content
   │
   ├─► 1000 char chunks with 200 char overlap
   ├─► ~40-50 total chunks (from 2-3 pages)
   │
   ▼
Step 3: Generate Embeddings
   │
   ├─► Convert each chunk to vector (335 dimensions)
   ├─► 40-50 vectors created
   │
   ▼
Step 4: Store in ChromaDB
   │
   ├─► Save to web_chroma_db/
   ├─► Ready for queries
   │
   ▼
NEXT TIME: 
   ├─ Load from disk (< 1 second)
   ├─ Check for new URLs
   ├─ Add new URLs if needed
```

---

## Configuration

```yaml
Web Scraping:
  Loader: WebBaseLoader (LangChain)
  URLs: Wikipedia, news sites, etc.
  Note: Some sites block automated scraping
  Recommended: Wikipedia (usually scraping-friendly)
  
Chunking Strategy:
  Chunk size: 1000 characters
  Overlap: 200 characters
  Split at: Double newline, newline, space, character
  
Embedding:
  Model: mxbai-embed-large:335m
  Dimensions: 335 million parameters
  Size: ~335MB
  
LLM:
  Model: llama3.2:3b
  Size: ~2GB
  
Vector Database:
  Type: ChromaDB
  Backend: SQLite
  Location: web_chroma_db/
  Retrieval: Similarity search (cosine distance)
  
Query Parameters:
  Top K results: 10 chunks
  Search type: similarity
  
Auto-Update:
  Enabled: YES
  New URLs added automatically
  Existing data preserved
```

---

## Running the Web Demo

```bash
# Step 1: Create/update vector database from URLs
python web_vector.py

# Step 2: Run the comparison demo
python web_main.py
```

**Expected Output:**
```
Topic: DINOSAURS
URLs: 1 pages
Question: Who identified dragon bones?

WITHOUT RAG (Using LLM Knowledge Only)
================================================================================
[LLM-only response - vague about dragon bones]

================================================================================
WITH RAG (Using Web-Scraped Content)
================================================================================
[RAG-enhanced response - accurate from Wikipedia]
```

---

## Key Differences: Data Source Comparison

| Aspect | CSV Data | PDF Data | Web Data |
|--------|----------|----------|----------|
| **Source Type** | Structured records | Unstructured PDFs | Semi-structured HTML |
| **Processing** | Direct load | Extract + chunk | Scrape + extract + chunk |
| **Dynamic** | Static file | Static file | Can change over time |
| **Chunks** | One per record | Multiple per page | Multiple per page |
| **Auto-Update** | Check for files | Check for files | Check for URLs |
| **Reliability** | High (local) | High (local) | Medium (network dependent) |
| **Metadata** | Full columns | Page numbers | URL, title |
| **Chunking** | Not needed | Required | Required |

---

## Troubleshooting

### "Websites blocking requests"
```
Solution:
1. Try Wikipedia URLs (usually work)
2. Some sites block User-Agent requests
3. Use sites that allow scraping in their robots.txt
4. Add delays between requests if needed
```

### "Network connectivity issues"
```
Solution:
1. Check your internet connection
2. Verify URLs are accessible in browser
3. Try different URLs
4. Check if proxy is needed
```

### "Empty or partial content"
```
Solution:
1. Some websites use JavaScript (WebBaseLoader may not handle)
2. Try simpler, text-heavy pages
3. Wikipedia articles work well
4. Static HTML pages preferred over dynamic content
```

### "Too many or too few chunks"
```
Solution:
- Adjust CHUNK_SIZE (current: 1000)
- Adjust CHUNK_OVERLAP (current: 200)
- Larger chunks = fewer total chunks
- More overlap = better context but slower
```

---

## Running All Three Demos

```bash
# Step 1: Create all vector databases
python csv_vector.py
python pdf_vector.py
python web_vector.py

# Step 2: Run all three demos
python csv_main.py
python pdf_main.py
python web_main.py

# Compare results across different data sources!
```

This allows you to see how RAG improves answers across three different data source types:
- **Structured (CSV)**: Book records → Great accuracy
- **Unstructured (PDF)**: Board game rules → Rules-based accuracy
- **Semi-structured (Web)**: Wikipedia content → General knowledge accuracy
