# updated_chroma_ollama_book_ingest.py
import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Use this import if your LangChain version exposes Document here.
# If that import fails in your environment, try: from langchain.schema import Document
try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document

CSV_PATH = "book_dataset_500.csv"  # <- your CSV file (change if needed)
DB_LOCATION = "./chroma_langchain_ollama_db"

# Load CSV
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at {CSV_PATH}. Put your CSV there or update CSV_PATH.")

df = pd.read_csv(CSV_PATH)
print("Loaded CSV with columns:", df.columns.tolist())
print("Total rows:", len(df))

# --- Configuration / Embeddings / DB init ---
embeddings = OllamaEmbeddings(model="mxbai-embed-large:335m")

# Only add documents when DB does not already exist
add_documents = not os.path.exists(DB_LOCATION)

vectordb = Chroma(embedding_function=embeddings, persist_directory=DB_LOCATION)

if add_documents:
    documents = []
    # A mapping from our CSV column names to the names used in the Document's page_content and metadata
    # Adjust these keys if your CSV uses different column names
    # Expected CSV columns (from the dataset generated earlier): 
    # ['title', 'rating', 'review_counts', 'publication_house', 'author', 'publishing_date', 'copies_sold']
    for index, row in df.iterrows():
        # Safely fetch values using .get-like behavior for pandas Series
        def val(key, default="N/A"):
            return row[key] if key in df.columns else default

        title = val("title")
        rating = val("rating")
        reviews_count = val("review_counts") if "review_counts" in df.columns else val("reviews_count")
        publication_house = val("publication_house") if "publication_house" in df.columns else val("publisher")
        author = val("author") if "author" in df.columns else val("authors")
        publishing_date = val("publishing_date") if "publishing_date" in df.columns else val("publication_date")
        copies_sold = val("copies_sold")

        page_content = f"""Title: {title}
Rating: {rating}
Reviews Count: {reviews_count}
Author: {author}
Publisher: {publication_house}
Publication Date: {publishing_date}
Copies Sold: {copies_sold}
"""

        metadata = {
            "title": title,
            "rating": rating,
            "reviews_count": reviews_count,
            "author": author,
            "publisher": publication_house,
            "publication_date": publishing_date,
            "copies_sold": copies_sold,
            # add any other CSV columns you want to keep as metadata
        }

        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

        # print first row example for verification
        if index == 0:
            print("Sample Document page_content:\n", page_content)
            print("Sample metadata:\n", metadata)

    # add documents in one call (or batch if memory concerns)
    print(f"Adding {len(documents)} documents to Chroma DB at {DB_LOCATION} ...")
    vectordb.add_documents(documents)
    vectordb.persist()
    print("Persisted vector DB.")
else:
    print("Vector DB already exists. Skipping document ingestion.")

# --- Create retriever and run a test query ---
# Important: ensure search_kwargs uses numeric k only (no stray shell commands or concatenations)
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Example queries (customize as required)
queries = [
    # "books with rating 4.0 and above",
    "popular books by author 'Timothy Wells'",    # example; replace author name with one in your CSV
    # "books that sold more than 50000 copies"
]

for q in queries:
    print("\n" + "="*80)
    print(f"Query: {q}")
    print("="*80)
    results = retriever.get_relevant_documents(q)  # returns list[Document]
    if not results:
        print("No results found.")
        continue

    for i, doc in enumerate(results, start=1):
        md = doc.metadata or {}
        print(f"{i}. {md.get('title','N/A')}")
        print(f"   Rating: {md.get('rating','N/A')} | Reviews: {md.get('reviews_count','N/A')} | Copies Sold: {md.get('copies_sold','N/A')}")
        print(f"   Publisher: {md.get('publisher','N/A')} | Author: {md.get('author','N/A')}")
        print("-"*80)

# If you want to reuse vectordb across sessions without reloading, simply run the block after the DB exists.
