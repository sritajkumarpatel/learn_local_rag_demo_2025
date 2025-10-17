# CSV RAG Demo Flow Diagram

## Complete Flow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CSV RAG DEMO INITIALIZATION                      │
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
        │ from disk        │         │ (csv_vector.py steps)  │
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
│         "Give popular books by author 'Timothy Wells'"              │
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

## STEP 1: Vector Database Creation (csv_vector.py)

### 1.1 Check if Vector DB Already Exists
```
┌──────────────────────────────────┐
│  Does chroma_csv_db/ exist?      │
└──────────────────────────────────┘
         │                    │
      YES│                    │NO
         │                    │
         ▼                    ▼
    Load DB              Read CSV File
    (from disk)          (book_dataset_500.csv)
         │                    │
         └────────┬───────────┘
                  │
              ✓ Faster (reuse existing data)
              ✓ No re-embedding needed
```

**Code:**
```python
# csv_vector.py - Lines 29-35
if os.path.exists(persist_directory) and not force_rebuild:
    print(f"Loading existing vector DB from {persist_directory}")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
```

---

### 1.2 Load and Process CSV Data
```
CSV File
   │
   ├─► Row 1: The Great Gatsby by Timothy Wells
   ├─► Row 2: To Kill a Mockingbird by Thomas Waters
   ├─► Row 3: [500 more books...]
   │
   ▼
Total: 500 books loaded into DataFrame
```

**Code:**
```python
# csv_vector.py - Lines 38-40
print("Step 1: Loading CSV data...")
df = pd.read_csv(csv_path)
print(f"✓ Loaded {len(df)} books from CSV")
```

**Example CSV Data:**
```
title,rating,reviews_count,author,publisher,publication_date,copies_sold
The Great Gatsby,4.5,12000,Timothy Wells,Penguin,2010,150000
To Kill a Mockingbird,4.8,25000,Thomas Waters,HarperCollins,2005,500000
```

---

### 1.3 Initialize Embedding Model
```
┌────────────────────────────────┐
│ Ollama Embedding Model         │
│ mxbai-embed-large:335m         │
└────────────────────────────────┘
         │
         ▼
    Ready to convert text
    to 335-dimensional vectors
```

**Code:**
```python
# csv_vector.py - Lines 42-44
print(f"Step 2: Initializing embedding model ({EMBEDDING_MODEL})...")
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
print("✓ Embedding model initialized")
```

---

### 1.4 Create Documents from CSV Rows
```
Row 1: {"title": "The Great Gatsby", "author": "Timothy Wells", ...}
   │
   ▼
Create Document object with:
   ├─ page_content: "Title: The Great Gatsby\nRating: 4.5\n..."
   └─ metadata: {"title": "...", "author": "...", "rating": ...}
   │
   ▼
Add to documents list

[Repeat for all 500 rows]
   │
   ▼
documents = [Document(...), Document(...), ..., Document(...)]
```

**Code:**
```python
# csv_vector.py - Lines 46-75
documents = []
for index, row in df.iterrows():
    page_content = f"""Title: {title}
Rating: {rating}
Reviews Count: {reviews_count}
Author: {author}
Publisher: {publication_house}
Publication Date: {publishing_date}
Copies Sold: {copies_sold}"""
    
    metadata = {
        "title": title,
        "rating": rating,
        "author": author,
        ...
    }
    documents.append(Document(page_content=page_content, metadata=metadata))
```

**Example Document Created:**
```
Document {
  page_content: "Title: The Great Gatsby
                Rating: 4.5
                Reviews Count: 12000
                Author: Timothy Wells
                Publisher: Penguin
                Publication Date: 2010
                Copies Sold: 150000",
  metadata: {
    "title": "The Great Gatsby",
    "author": "Timothy Wells",
    "rating": 4.5,
    "reviews_count": 12000,
    ...
  }
}
```

---

### 1.5 Generate Embeddings and Store in ChromaDB
```
Document 1: "Title: The Great Gatsby..."
   │
   ▼
Embedding Model
   │
   ▼
Vector: [0.234, -0.156, 0.890, ..., 0.123]  (335 dimensions)
   │
   ├─► Store in ChromaDB
   ├─► Link to metadata
   └─► Save to disk (chroma_csv_db/)

[Repeat for all 500 documents]
   │
   ▼
ChromaDB fully populated with vectors
Ready for similarity search
```

**Code:**
```python
# csv_vector.py - Lines 77-83
print(f"Step 4: Creating vector database and generating embeddings...")
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=persist_directory
)
print(f"✓ Vector DB created and saved to {persist_directory}")
```

**What Happens Internally:**
- ✓ Each document text → embedding (335 numbers)
- ✓ All vectors stored in ChromaDB
- ✓ Metadata preserved for retrieval
- ✓ Database persisted to `chroma_csv_db/` directory
- ✓ Next time: just load existing DB (much faster!)

---

## STEP 2: Query Processing - Without RAG (csv_main.py)

```
User Question
  │
  "Give popular books by 
   author 'Timothy Wells' 
   and 'Thomas Waters'"
  │
  ▼
┌──────────────────────────────────────┐
│ LLM Model (llama3.2:3b)              │
│ WITHOUT any vector database context  │
└──────────────────────────────────────┘
  │
  ├─► Uses only training data
  ├─► No access to actual books
  ├─► May be outdated/inaccurate
  └─► Can hallucinate details
  │
  ▼
"Based on my training data,
Timothy Wells is known for...
Some popular works include..."
(May not exist or be wrong!)
```

**Code:**
```python
# csv_main.py - Lines 18-22
def ask_without_rag(question: str) -> str:
    """Ask the LLM directly without vector database context."""
    prompt = f"You are a book expert. Answer the following question:\n\nQuestion: {question}\n\nAnswer:"
    return model.invoke(prompt)

# Usage
question = "Give popular books by author 'Timothy Wells' and 'Thomas Waters'"
resultFromRegularLLM = ask_without_rag(question)
```

**Prompt Sent to LLM:**
```
You are a book expert. Answer the following question:

Question: Give popular books by author 'Timothy Wells' and 'Thomas Waters'

Answer:
```

**Example Response (Without RAG):**
```
Based on my training data, Timothy Wells is known for writing 
contemporary fiction, and Thomas Waters has written science fiction novels. 

Some of their popular works include:
- Timothy Wells: Various contemporary fiction pieces
- Thomas Waters: Science fiction novels

(Note: This may be inaccurate, outdated, or completely made up!)
```

---

## STEP 3: Query Processing - With RAG (csv_main.py)

### 3.1 Retrieve Relevant Documents from Vector Database
```
User Question
  │
  "Give popular books by 
   author 'Timothy Wells' 
   and 'Thomas Waters'"
  │
  ▼
Convert to Embedding
(same model used for training: mxbai-embed-large:335m)
  │
  ▼
Vector: [0.145, -0.234, 0.567, ..., 0.890]
  │
  ▼
┌──────────────────────────────────────┐
│ ChromaDB Similarity Search            │
│ Find top 10 most similar documents   │
└──────────────────────────────────────┘
  │
  ├─► Document 1: The Great Gatsby (Timothy Wells) - Similarity: 0.95
  ├─► Document 2: To Kill a Mockingbird (Thomas Waters) - Similarity: 0.93
  ├─► Document 3: [Similar book] - Similarity: 0.87
  ├─► ... (more documents)
  │
  ▼
Return top 10 documents with metadata
```

**Code:**
```python
# csv_main.py - Lines 25-26
def ask_with_rag(question: str) -> str:
    retrieved_docs = retriever_csvbooks.invoke(question)
```

**What `retriever.invoke()` Returns:**
```
[
  Document(
    page_content="Title: The Great Gatsby\nRating: 4.5\nReviews Count: 12000\nAuthor: Timothy Wells\n...",
    metadata={"title": "The Great Gatsby", "author": "Timothy Wells", "rating": 4.5, ...}
  ),
  Document(
    page_content="Title: To Kill a Mockingbird\nRating: 4.8\nReviews Count: 25000\nAuthor: Thomas Waters\n...",
    metadata={"title": "To Kill a Mockingbird", "author": "Thomas Waters", "rating": 4.8, ...}
  ),
  ... (8 more documents)
]
```

---

### 3.2 Build Context Prompt with Retrieved Documents
```
Retrieved Documents (10 books)
  │
  ├─► Book 1 details
  ├─► Book 2 details
  ├─► Book 3 details
  ├─► ... (more books)
  │
  ▼
Format into prompt:
  │
  "You are a book expert.
   
   Here are the relevant books:
   
   === Book 1 ===
   Title: The Great Gatsby
   Author: Timothy Wells
   Rating: 4.5
   ...
   
   === Book 2 ===
   Title: To Kill a Mockingbird
   Author: Thomas Waters
   Rating: 4.8
   ...
   
   Question: [User Question]
   
   Provide a clear answer based on the book data above:"
  │
  ▼
Full context-rich prompt ready for LLM
```

**Code:**
```python
# csv_main.py - Lines 28-37
prompt = f"You are a book expert. Answer the following question based on the book data provided.\n\n"
prompt += f"Here are the relevant books:\n\n"

for i, doc in enumerate(retrieved_docs, 1):
    prompt += f"=== Book {i} ===\n{doc.page_content}\n\n"

prompt += f"Question: {question}\n\nProvide a clear answer based on the book data above:"
```

**Final Prompt Example:**
```
You are a book expert. Answer the following question based on the book data provided.

Here are the relevant books:

=== Book 1 ===
Title: The Great Gatsby
Rating: 4.5
Reviews Count: 12000
Author: Timothy Wells
Publisher: Penguin
Publication Date: 2010
Copies Sold: 150000

=== Book 2 ===
Title: To Kill a Mockingbird
Rating: 4.8
Reviews Count: 25000
Author: Thomas Waters
Publisher: HarperCollins
Publication Date: 2005
Copies Sold: 500000

[... more books with actual data ...]

Question: Give popular books by author 'Timothy Wells' and 'Thomas Waters'

Provide a clear answer based on the book data above:
```

---

### 3.3 Generate RAG-Enhanced Answer
```
LLM Model (with context)
  │
  ├─► Sees actual book data
  ├─► Can reference real ratings/reviews
  ├─► Provides accurate information
  └─► Backed by source documents
  │
  ▼
┌──────────────────────────────────────┐
│ RAG-ENHANCED RESPONSE                │
└──────────────────────────────────────┘
  │
  "Based on the book data provided:
  
  Popular books by Timothy Wells:
  - The Great Gatsby (Rating: 4.5, Reviews: 12,000, 
    Published: 2010, Copies Sold: 150,000)
  
  Popular books by Thomas Waters:
  - To Kill a Mockingbird (Rating: 4.8, Reviews: 25,000, 
    Published: 2005, Copies Sold: 500,000)
  
  Both authors have highly rated books with strong 
  reader engagement. The Great Gatsby has sold 150,000 
  copies while To Kill a Mockingbird has sold 500,000 copies."
  │
  ▼
(Accurate! Verifiable from source data!)
```

**Code:**
```python
# csv_main.py - Lines 40-42
return model.invoke(prompt)
```

---

## STEP 4: Initialization Phase (csv_main.py)

**Load the LLM and vector database:**

```python
# csv_main.py - Lines 12-14
from langchain_ollama import OllamaLLM
import csv_vector

model = OllamaLLM(model="llama3.2:3b")
vectordb_csvbooks = csv_vector.create_vector_db_from_csv(force_rebuild=False)
retriever_csvbooks = vectordb_csvbooks.as_retriever(search_type="similarity", search_kwargs={"k": 10})
```

**Initialization Flow:**
```
csv_main.py starts
   │
   ├─► Import OllamaLLM
   ├─► Import csv_vector module
   │
   ▼
Load LLM Model (llama3.2:3b)
   │
   ▼
Call csv_vector.create_vector_db_from_csv()
   │
   ├─ Check if DB exists
   │  ├─ YES: Load from disk (fast)
   │  └─ NO: Create from CSV (one-time cost)
   │
   ▼
Create Retriever from vectordb
   (similarity search with k=10)
   │
   ▼
Ready for queries!
```

---

## STEP 5: Key Differences - RAG vs Non-RAG

| Aspect | **Without RAG** | **With RAG** |
|--------|-----------------|-------------|
| **Source** | LLM training data only | Retrieved CSV data + LLM |
| **Accuracy** | ⚠️ May be outdated/inaccurate | ✅ Current, factual data |
| **Data Freshness** | ❌ Static (training cutoff) | ✅ Always current (CSV is source) |
| **Reliability** | ⚠️ Prone to hallucination | ✅ Verifiable from source |
| **Process** | Direct question → LLM → Answer | Question → Search → Context → LLM → Answer |
| **Transparency** | ❌ Unknown source | ✅ Can cite specific books |
| **Update Frequency** | Requires retraining | Update CSV → Immediate effect |

---

## Complete Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    INITIALIZATION (One Time)                    │
└─────────────────────────────────────────────────────────────────┘

CSV File (500 books)
   │
   ├─► Extract fields: title, author, rating, publisher, etc.
   │
   ▼
Create Documents from each row
   │
   ├─► Document 1: page_content + metadata
   ├─► Document 2: page_content + metadata
   ├─► ... (500 documents)
   │
   ▼
Generate Embeddings (mxbai-embed-large:335m)
   │
   ├─► Each document → 335-dimensional vector
   │
   ▼
Store in ChromaDB
   │
   ├─► Persist to: chroma_csv_db/
   └─► Ready for queries


┌─────────────────────────────────────────────────────────────────┐
│                    QUERY TIME (Every Question)                  │
└─────────────────────────────────────────────────────────────────┘

User Question
   │
   ├─────────────────────┬─────────────────────┐
   │                     │                     │
   ▼                     ▼                     ▼
WITHOUT RAG        WITH RAG (Main Path)
   │                     │
   │              Convert Q to embedding
   │                     │
   │              Search ChromaDB
   │              (find 10 similar docs)
   │                     │
   │              Retrieve book data
   │                     │
   │              Build context prompt
   │                     │
   └──────────┬──────────┘
              │
              ▼
        LLM (llama3.2:3b)
              │
              ├─ WITHOUT RAG: Generic response
              ├─ WITH RAG: Context-aware response
              │
              ▼
        ANSWER
              │
              ├─ Display both for comparison
              │
              ▼
        END


┌─────────────────────────────────────────────────────────────────┐
│                    KEY PERFORMANCE METRICS                       │
└─────────────────────────────────────────────────────────────────┘

CSV Processing (csv_vector.py):
  ├─ Load 500 books: ~1-2 seconds
  ├─ Generate embeddings: ~30-60 seconds (one-time)
  ├─ Store in ChromaDB: ~5-10 seconds
  └─ Total first run: ~45-75 seconds
  └─ Reload from disk: <1 second ✓

Query Processing (csv_main.py):
  ├─ Without RAG: 2-3 seconds (LLM only)
  ├─ With RAG: 3-5 seconds (search + LLM)
  │  ├─ Embedding query: ~0.5 seconds
  │  ├─ Similarity search: ~0.1 seconds
  │  └─ LLM generation: ~2-3 seconds
  └─ Difference: Worth it for accuracy!
```

---

## Configuration Summary

```yaml
Embedding Model:
  Name: mxbai-embed-large:335m
  Dimensions: 335 million parameters
  Size: ~335MB
  Used for: Converting text to vectors

LLM Model:
  Name: llama3.2:3b
  Size: ~2GB
  Used for: Generating answers

Vector Database:
  Type: ChromaDB
  Backend: SQLite
  Location: chroma_csv_db/
  Retrieval: Similarity search (cosine distance)
  
Query Parameters:
  Top K results: 10 documents
  Search type: similarity
  
CSV Configuration:
  Source file: csvs/book_dataset_500.csv
  Records: 500 books
  Fields: title, author, rating, publisher, publication_date, copies_sold
```

---

## Running the Demo

```bash
# Step 1: Create vector database (one-time)
python csv_vector.py

# Step 2: Run the comparison demo
python csv_main.py
```

**Output:**
```
WITHOUT RAG (Using LLM Knowledge Only)
================================================================================
[LLM-only response - may be inaccurate]

================================================================================
WITH RAG (Using CSV Content)
================================================================================
[RAG-enhanced response - accurate, fact-based]
```