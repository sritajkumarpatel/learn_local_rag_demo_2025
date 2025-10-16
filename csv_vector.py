"""
CSV Vector Database Module
===========================
Creates embeddings from CSV book data and stores them in ChromaDB
for semantic search and retrieval.
"""

import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document

CSV_PATH = "./csvs/book_dataset_500.csv"
EMBEDDING_MODEL = "mxbai-embed-large:335m"
DB_LOCATION = "./chroma_csv_db"


def create_vector_db_from_csv(csv_path=CSV_PATH, persist_directory=DB_LOCATION, force_rebuild=False):
    """
    Create or load a vector database from CSV book data.
    
    Args:
        csv_path (str): Path to CSV file
        persist_directory (str): Where to save the vector database
        force_rebuild (bool): If True, rebuild even if DB exists
        
    Returns:
        Chroma: Vector database instance
    """
    
    if os.path.exists(persist_directory) and not force_rebuild:
        print(f"Loading existing vector DB from {persist_directory}")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vectordb = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        print(f"✓ Loaded existing DB with {vectordb._collection.count()} documents")
        return vectordb
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    print("Step 1: Loading CSV data...")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} books from CSV")
    
    print(f"\nStep 2: Initializing embedding model ({EMBEDDING_MODEL})...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    print("✓ Embedding model initialized")
    
    print("\nStep 3: Creating documents...")
    documents = []
    
    for index, row in df.iterrows():
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
Copies Sold: {copies_sold}"""

        metadata = {
            "title": title,
            "rating": rating,
            "reviews_count": reviews_count,
            "author": author,
            "publisher": publication_house,
            "publication_date": publishing_date,
            "copies_sold": copies_sold,
        }

        documents.append(Document(page_content=page_content, metadata=metadata))
    
    print(f"✓ Created {len(documents)} documents")
    
    print(f"\nStep 4: Creating vector database and generating embeddings...")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"✓ Vector DB created and saved to {persist_directory}")
    
    return vectordb


if __name__ == "__main__":
    vectordb = create_vector_db_from_csv(force_rebuild=False)
    
    print("\n" + "="*80)
    print("Vector database created successfully!")
    print("="*80)
    
    print("\n" + "="*80)
    print("EXAMPLE QUERY")
    print("="*80)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    query = "Give popular books by author 'Timothy Wells' and 'Thomas Waters'"
    print(f"\nQuery: {query}")
    results = retriever.invoke(query)
    
    if not results:
        print("No results found.")
    else:
        for i, doc in enumerate(results, 1):
            md = doc.metadata
            print(f"\n{i}. {md.get('title','N/A')}")
            print(f"   Rating: {md.get('rating','N/A')} | Reviews: {md.get('reviews_count','N/A')}")
            print(f"   Author: {md.get('author','N/A')} | Publisher: {md.get('publisher','N/A')}")
