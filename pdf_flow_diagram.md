# PDF RAG Demo Flow Diagram

## Complete Flow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PDF RAG DEMO INITIALIZATION                      │
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
        │ Check for new    │         │ (pdf_vector.py steps)  │
        │ PDFs & update    │         └────────────────────────┘
        └──────────────────┘                      │
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
│       "How many players can play CATANIC RIDE TO HEAVEN?"           │
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
        │ Direct LLM Query │        │ 1. Retrieve Docs     │
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

## STEP 1: PDF Vector Database Creation (pdf_vector.py)

### 1.1 Check if Vector DB Already Exists
```
┌──────────────────────────────────┐
│  Does pdf_chroma_db/ exist?      │
└──────────────────────────────────┘
         │                    │
      YES│                    │NO
         │                    │
         ▼                    ▼
    Load DB              Load PDFs from
    (from disk)          ./pdfs/ directory
         │                    │
         ├─ Check for    Extract pages
         │  new PDFs     from each PDF
         │                    │
         │              ✓ Loads all PDF files
         │              ✓ Extracts text from pages
         └────────┬───────────┘
                  │
              ✓ Faster (reuse existing data)
              ✓ Auto-update with new PDFs
```

**Code:**
```python
# pdf_vector.py - Lines 88-121
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
        current_pdfs = get_pdf_files(pdf_path)
        new_pdfs = [pdf for pdf in current_pdfs if pdf not in existing_sources]
        
        if new_pdfs:
            print(f"\n🔍 Found {len(new_pdfs)} new PDF(s) to add")
```

---

### 1.2 Load PDF Files from Directory
```
./pdfs/ Directory
   │
   ├─► file1.pdf
   ├─► file2.pdf
   ├─► file3.pdf
   │
   ▼
PyPDFLoader (for each PDF)
   │
   ├─► Extract all pages
   ├─► Convert to Document objects
   │
   ▼
List of Documents
   │
   ├─► Document 1: Page 1 of file1.pdf
   ├─► Document 2: Page 2 of file1.pdf
   ├─► Document 3: Page 1 of file2.pdf
   ├─► ... (all pages from all PDFs)
   │
   ▼
Total: N pages loaded
```

**Code:**
```python
# pdf_vector.py - Lines 18-29
def load_pdfs_from_directory(pdf_dir):
    loader = DirectoryLoader(
        pdf_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()

# Called in main
documents = load_pdfs_from_directory(pdf_path)
print(f"✓ Loaded {len(documents)} pages from PDF(s)")
```

---

### 1.3 Chunk PDF Documents
```
PDF Page Content
   │
   "Rules of Monopoly:
    1. Roll the dice...
    2. Move your piece...
    3. If you land on property..."
   │
   ▼
RecursiveCharacterTextSplitter
   │
   ├─ Split at: \n\n (double newline)
   ├─ Then at: \n (newline)
   ├─ Then at: space
   ├─ Chunk size: 1000 characters
   ├─ Overlap: 200 characters (for context)
   │
   ▼
Multiple Chunks
   │
   ├─► Chunk 1: "Rules of Monopoly: 1. Roll the dice..."
   ├─► Chunk 2: "2. Move your piece... [continues from chunk 1]"
   ├─► Chunk 3: "3. If you land on property..."
   │
   ▼
Chunked Documents (overlap ensures context)
```

**Code:**
```python
# pdf_vector.py - Lines 31-47
def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

# Main
print(f"Chunking documents (size=1000, overlap=200)...")
chunked_docs = chunk_documents(documents)
print(f"✓ Created {len(chunked_docs)} chunks from {len(documents)} pages")
```

**Example:**
- 10 PDF pages → 50 total chunks (with overlap for context)
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
    Ready to convert PDF chunks
    to 335-dimensional vectors
```

**Code:**
```python
# pdf_vector.py - Lines 141-142
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
print("✓ Embedding model initialized")
```

---

### 1.5 Generate Embeddings and Store in ChromaDB
```
Chunked Document 1: "Rules of Monopoly..."
   │
   ▼
Embedding Model
   │
   ▼
Vector: [0.234, -0.156, 0.890, ..., 0.123]  (335 dimensions)
   │
   ├─► Store in ChromaDB
   ├─► Link to metadata (source PDF, page number)
   └─► Save to disk (pdf_chroma_db/)

[Repeat for all chunks]
   │
   ▼
ChromaDB fully populated with vectors
Ready for similarity search
```

**Code:**
```python
# pdf_vector.py - Lines 144-150
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
  "source": "/path/to/pdfs/board_games.pdf",
  "page": 2,
  "title": "Board Games Rules"
}
```

---

## STEP 2: Query Processing - Without RAG (pdf_main.py)

```
User Question
  │
  "How many players can play 
   CATANIC RIDE TO HEAVEN?"
  │
  ▼
┌──────────────────────────────────┐
│ LLM Model (llama3.2:3b)          │
│ WITHOUT any PDF context          │
└──────────────────────────────────┘
  │
  ├─► Uses only training data
  ├─► No access to actual PDFs
  ├─► May be wrong/made-up
  └─► Cannot answer obscure games
  │
  ▼
"I don't have information about 
CATANIC RIDE TO HEAVEN in my 
training data. It may not be 
a real board game."
(Inaccurate!)
```

**Code:**
```python
# pdf_main.py - Lines 18-22
def ask_without_rag(question: str) -> str:
    """Ask the LLM directly without vector database context."""
    prompt = f"You are a board games expert. Answer the following question:\n\nQuestion: {question}\n\nAnswer:"
    return model.invoke(prompt)
```

---

## STEP 3: Query Processing - With RAG (pdf_main.py)

### 3.1 Retrieve Relevant Document Chunks from PDF Vector Database
```
User Question
  │
  "How many players can play 
   CATANIC RIDE TO HEAVEN?"
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
  ├─► Chunk 1: "CATANIC RIDE TO HEAVEN rules page 1..." - Similarity: 0.98
  ├─► Chunk 2: "Player count: 2-6 players per game..." - Similarity: 0.95
  ├─► Chunk 3: "Setup instructions for this game..." - Similarity: 0.89
  ├─► ... (more chunks)
  │
  ▼
Return top 10 chunks with metadata
   (source file, page number, exact text)
```

**Code:**
```python
# pdf_main.py - Lines 36-37
def ask_with_rag(question: str) -> str:
    retrieved_docs = retriever_pdf.invoke(question)
```

---

### 3.2 Build Context Prompt with Retrieved PDF Chunks
```
Retrieved Chunks (10 board game rules)
  │
  ├─► Chunk 1: Rules details
  ├─► Chunk 2: Player count info
  ├─► Chunk 3: More rules
  ├─► ... (more chunks with actual PDF content)
  │
  ▼
Format into prompt:
  │
  "You are a board games expert.
   
   Here are the relevant info:
   
   === Board Game 1 ===
   CATANIC RIDE TO HEAVEN
   Rules: Players compete to reach heaven...
   Player Count: 2-6 players
   Setup: Each player gets...
   
   === Board Game 2 ===
   [More rules from PDFs...]
   
   Question: How many players can play CATANIC RIDE TO HEAVEN?
   
   Provide a clear answer based on the board games data above:"
  │
  ▼
Full context-rich prompt ready for LLM
```

**Code:**
```python
# pdf_main.py - Lines 40-46
prompt = f"You are a board games expert. Answer the following question based on the data provided.\n\n"
prompt += f"Here are the relevant info:\n\n"

for i, doc in enumerate(retrieved_docs, 1):
    prompt += f"=== Board Game {i} ===\n{doc.page_content}\n\n"

prompt += f"Question: {question}\n\nProvide a clear answer based on the board games data above:"
```

---

### 3.3 Generate RAG-Enhanced Answer
```
LLM Model (with PDF context)
  │
  ├─► Sees actual PDF content
  ├─► Can reference real rules
  ├─► Provides accurate information
  └─► Backed by source documents
  │
  ▼
┌──────────────────────────────────────┐
│ RAG-ENHANCED RESPONSE                │
└──────────────────────────────────────┘
  │
  "Based on the board games data provided:
  
  CATANIC RIDE TO HEAVEN can be played with 
  2 to 6 players. The game supports both 
  competitive and cooperative play modes."
  │
  ▼
(Accurate! Verifiable from PDF!)
```

---

## STEP 4: PDF Processing Pipeline Detailed View

### Data Transformation Pipeline
```
PDF Files in ./pdfs/
   │
   ├─► board_game_1.pdf
   ├─► board_game_2.pdf
   ├─► board_game_3.pdf
   │
   ▼
Step 1: Load PDFs
   │
   ├─► Extract all pages
   ├─► 30 total pages
   │
   ▼
Step 2: Chunk Documents
   │
   ├─► 1000 char chunks with 200 char overlap
   ├─► 85 total chunks
   │
   ▼
Step 3: Generate Embeddings
   │
   ├─► Convert each chunk to vector (335 dimensions)
   ├─► 85 vectors created
   │
   ▼
Step 4: Store in ChromaDB
   │
   ├─► Save to pdf_chroma_db/
   ├─► Ready for queries
   │
   ▼
NEXT TIME: Load from disk (< 1 second)
```

---

## Configuration

```yaml
PDF Processing:
  Source directory: ./pdfs/
  File pattern: **/*.pdf (recursive)
  PDF Loader: PyPDFLoader
  
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
  Location: pdf_chroma_db/
  Retrieval: Similarity search (cosine distance)
  
Query Parameters:
  Top K results: 10 chunks
  Search type: similarity
  
Auto-Update:
  Enabled: YES
  New PDFs added automatically
  Existing data preserved
```

---

## Running the PDF Demo

```bash
# Step 1: Create/update vector database from PDFs
python pdf_vector.py

# Step 2: Run the comparison demo
python pdf_main.py
```

**Expected Output:**
```
WITHOUT RAG (Using LLM Knowledge Only)
================================================================================
[LLM-only response - may be generic or wrong]

================================================================================
WITH RAG (Using PDFs Content)
================================================================================
[RAG-enhanced response - accurate, from PDF rules]
```

---

## Key Differences: PDF vs CSV Data

| Aspect | CSV Data | PDF Data |
|--------|----------|----------|
| **Source** | Structured records | Unstructured text |
| **Processing** | One record per doc | Pages chunked into overlapping pieces |
| **Chunking** | Not needed | Required (chunk_size=1000) |
| **Documents** | 500 books | Hundreds/thousands of chunks |
| **Context** | Metadata preserved | Page numbers & source file |
| **Auto-update** | Check for new files | Add new PDFs automatically |
| **Storage** | Simpler | More complex (chunks with overlap) |

---

## Troubleshooting

### "No PDFs found in directory"
- Verify PDFs are in `./pdfs/` folder
- Check file permissions
- Ensure `.pdf` extension (lowercase)

### "PDF parsing errors"
- Check if PDFs are password-protected
- Try opening PDF in your system first
- Some PDFs may have encoding issues

### "Chunking creates too many/too few chunks"
- Adjust `CHUNK_SIZE` (current: 1000)
- Adjust `CHUNK_OVERLAP` (current: 200)
- Larger chunks = fewer total chunks
- More overlap = more context but slower search
