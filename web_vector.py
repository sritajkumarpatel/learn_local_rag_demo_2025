"""
Web Vector Database Module
===========================
This module handles web scraping, content processing, chunking them into smaller pieces,
creating embeddings, and storing them in a ChromaDB vector database for
efficient semantic search and retrieval.
"""

import os
import requests
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document

EMBEDDING_MODEL = "mxbai-embed-large:335m"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DB_LOCATION = "./web_chroma_db"


def load_webpages(urls):
    """
    Load web pages using LangChain's WebBaseLoader.
    
    Args:
        urls (list): List of URLs to load
        
    Returns:
        list: List of Document objects
    """
    try:
        print(f"Loading {len(urls)} URLs with WebBaseLoader...")
        
        # WebBaseLoader can handle multiple URLs
        loader = WebBaseLoader(urls)
        documents = loader.load()
        
        print(f"✓ Successfully loaded {len(documents)} pages")
        return documents
        
    except Exception as e:
        print(f"❌ Error loading URLs: {e}")
        return []


def chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split documents into smaller chunks for better retrieval and embedding.
    
    Args:
        documents (list): List of Document objects
        chunk_size (int): Maximum characters per chunk
        chunk_overlap (int): Characters to overlap between chunks
        
    Returns:
        list: List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


def get_existing_sources(vectordb):
    """
    Get set of source URLs already in the database.
    
    Args:
        vectordb: Chroma vector database instance
        
    Returns:
        set: Set of source URLs
    """
    try:
        results = vectordb.get()
        if results and 'metadatas' in results:
            return {meta.get('source') for meta in results['metadatas'] if meta.get('source')}
        return set()
    except:
        return set()


def create_vector_db_from_urls(urls, persist_directory=DB_LOCATION, force_rebuild=False, auto_update=True):
    """
    Create or load a vector database from web URLs with automatic incremental updates.
    
    Args:
        urls (list): List of URLs to scrape and process
        persist_directory (str): Where to save the vector database
        force_rebuild (bool): If True, rebuild entire DB from scratch
        auto_update (bool): If True, automatically add new URLs to existing DB
        
    Returns:
        Chroma: Vector database instance
    """
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
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
                for url in new_urls:
                    print(f"  + {url}")
                
                print("\nLoading new URLs...")
                new_documents = load_webpages(new_urls)
                print(f"✓ Successfully loaded {len(new_documents)} pages")
                
                if new_documents:
                    chunked_docs = chunk_documents(new_documents)
                    print(f"✓ Created {len(chunked_docs)} chunks")
                    
                    vectordb.add_documents(chunked_docs)
                    print(f"✓ Added new documents to database")
                    print(f"  Total chunks: {existing_count} → {vectordb._collection.count()}")
            else:
                print("✓ No new URLs found, database is up to date")
        
        return vectordb
    
    print("Step 1: Loading URLs...")
    documents = load_webpages(urls)
    print(f"✓ Successfully loaded {len(documents)} pages")
    
    if not documents:
        print("\n❌ No content could be loaded from the provided URLs.")
        print("This could be due to:")
        print("- Websites blocking automated requests")
        print("- Network connectivity issues")
        print("- Invalid or unreachable URLs")
        print("- Anti-bot measures on the target websites")
        print("\nTry using different URLs, such as Wikipedia pages which are usually scraping-friendly.")
        raise ValueError("No content could be loaded from the provided URLs")
    
    print(f"\nStep 2: Chunking documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunked_docs = chunk_documents(documents)
    print(f"✓ Created {len(chunked_docs)} chunks from {len(documents)} pages")
    print(f"  Average chunk size: {sum(len(doc.page_content) for doc in chunked_docs) // len(chunked_docs)} characters")
    
    print(f"\nStep 3: Initializing embedding model ({EMBEDDING_MODEL})...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    print("✓ Embedding model initialized")
    
    print(f"\nStep 4: Creating vector database and generating embeddings...")
    vectordb = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"✓ Vector DB created and saved to {persist_directory}")
    print(f"  Total chunks embedded: {len(chunked_docs)}")
    
    return vectordb


if __name__ == "__main__":
    # Example URLs to scrape - using reliable, scraping-friendly sites
    urls_to_scrape = [
        "https://en.wikipedia.org/wiki/Dinosaur"
    ]
    
    print("Web Scraping Vector Database Demo")
    print("=" * 50)
    print(f"URLs to scrape: {len(urls_to_scrape)}")
    for url in urls_to_scrape:
        print(f"  • {url}")
    print()
    
    # Create vector database from URLs
    try:
        vectordb = create_vector_db_from_urls(
            urls_to_scrape,
            force_rebuild=False,  # Set to True to rebuild from scratch
            auto_update=True      # Automatically add new URLs
        )
        
        print("\n" + "="*80)
        print("Vector database created successfully!")
        print("="*80)
        
        print("\n" + "="*80)
        print("EXAMPLE QUERY")
        print("="*80)
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        query = "Who identified dragon bones?"
        print(f"\nQuery: {query}\n")
        results = retriever.invoke(query)
        
        if not results:
            print("No results found.")
        else:
            for i, doc in enumerate(results, 1):
                print(f"{'─' * 80}")
                print(f"Result {i}/{len(results)}")
                print(f"{'─' * 80}")
                
                content = doc.page_content.strip()
                if len(content) > 500:
                    print(content[:500] + "...\n")
                else:
                    print(content + "\n")
                
                if doc.metadata:
                    md = doc.metadata
                    print(f"🌐 Source: {md.get('source', 'Unknown')}")
                    if 'title' in md:
                        print(f"📑 Title: {md.get('title')}")
                print()
                
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if the URLs are accessible in your browser")
        print("2. Some websites block automated scraping")
        print("3. Try different URLs (Wikipedia pages usually work well)")
        print("4. Check your internet connection")
        exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        exit(1)