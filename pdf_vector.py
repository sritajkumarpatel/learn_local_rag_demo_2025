"""
PDF Vector Database Module
===========================
This module handles loading PDF documents, chunking them into smaller pieces,
creating embeddings, and storing them in a ChromaDB vector database for
efficient semantic search and retrieval.
"""

import os
import glob
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "mxbai-embed-large:335m"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DB_LOCATION = "./pdf_chroma_db"
PDF_PATH = "./pdfs"

def load_pdfs_from_directory(pdf_dir):
    """
    Load all PDF files from a directory and its subdirectories.
    
    Args:
        pdf_dir (str): Path to directory containing PDF files
        
    Returns:
        list: List of Document objects, one per page
    """
    loader = DirectoryLoader(
        pdf_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


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
    Get set of source file paths already in the database.
    
    Args:
        vectordb: Chroma vector database instance
        
    Returns:
        set: Set of source file paths
    """
    try:
        results = vectordb.get()
        if results and 'metadatas' in results:
            return {meta.get('source') for meta in results['metadatas'] if meta.get('source')}
        return set()
    except:
        return set()


def get_pdf_files(pdf_path):
    """
    Get all PDF files from a path (file or directory).
    
    Args:
        pdf_path (str): Path to PDF file or directory
        
    Returns:
        list: List of PDF file paths
    """
    if os.path.isfile(pdf_path):
        return [pdf_path]
    elif os.path.isdir(pdf_path):
        return glob.glob(os.path.join(pdf_path, "**/*.pdf"), recursive=True)
    return []


def create_vector_db_from_pdfs(pdf_path=PDF_PATH, persist_directory=DB_LOCATION, force_rebuild=False, auto_update=True):
    """
    Create or load a vector database from PDF files with automatic incremental updates.
    
    Args:
        pdf_path (str): Path to PDF file or directory of PDFs
        persist_directory (str): Where to save the vector database
        force_rebuild (bool): If True, rebuild entire DB from scratch
        auto_update (bool): If True, automatically add new PDFs to existing DB
        
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
            current_pdfs = get_pdf_files(pdf_path)
            new_pdfs = [pdf for pdf in current_pdfs if pdf not in existing_sources]
            
            if new_pdfs:
                print(f"\n🔍 Found {len(new_pdfs)} new PDF(s) to add")
                for pdf in new_pdfs:
                    print(f"  + {os.path.basename(pdf)}")
                
                print("\nProcessing new PDFs...")
                new_documents = []
                for pdf_file in new_pdfs:
                    loader = PyPDFLoader(pdf_file)
                    new_documents.extend(loader.load())
                
                print(f"✓ Loaded {len(new_documents)} pages from new PDF(s)")
                
                chunked_docs = chunk_documents(new_documents)
                print(f"✓ Created {len(chunked_docs)} new chunks")
                
                vectordb.add_documents(chunked_docs)
                print(f"✓ Added new documents to database")
                print(f"  Total chunks: {existing_count} → {vectordb._collection.count()}")
            else:
                print("✓ No new PDFs found, database is up to date")
        
        return vectordb
    
    print("Step 1: Loading PDF(s)...")
    if os.path.isdir(pdf_path):
        documents = load_pdfs_from_directory(pdf_path)
    else:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    
    print(f"✓ Loaded {len(documents)} pages from PDF(s)")
    
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
    pdf_directory = "./pdfs"
    
    if not os.path.exists(pdf_directory):
        print(f"ERROR: Path does not exist: {pdf_directory}")
        print("Please update the pdf_directory variable with a valid path.")
        exit(1)
    
    vectordb = create_vector_db_from_pdfs(pdf_directory, force_rebuild=False, auto_update=True)
    
    print("\n" + "="*80)
    print("Vector database created successfully!")
    print("To run tests, execute: python test_pdf_vector.py")
    print("="*80)
    
    print("\n" + "="*80)
    print("EXAMPLE QUERY")
    print("="*80)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    query = "What are the rules of monopoly?"
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
                print(f"📄 Source: {md.get('source', 'Unknown')}")
                print(f"📖 Page: {md.get('page', 'N/A') + 1}")
                if 'title' in md:
                    print(f"📑 Title: {md.get('title')}")
            print()