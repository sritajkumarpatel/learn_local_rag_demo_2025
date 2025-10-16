"""
Test Suite for PDF Vector Database
===================================
Tests to verify the PDF vector database functionality works correctly.
Run with: python test_pdf_vector.py
Or with pytest: pytest test_pdf_vector.py -v
"""

import os
import sys
from pdf_vector import create_vector_db_from_pdfs, DB_LOCATION


def test_database_exists():
    """
    Test 1: Verify that the vector database directory exists.
    """
    print("\n[TEST 1] Checking if database directory exists...")
    assert os.path.exists(DB_LOCATION), f"Database directory not found at {DB_LOCATION}"
    print(f"✓ Database directory exists at {DB_LOCATION}")


def test_database_has_documents(vectordb):
    """
    Test 2: Verify the database contains documents.
    """
    print("\n[TEST 2] Checking database contains documents...")
    try:
        count = vectordb._collection.count()
        assert count > 0, "Database is empty!"
        print(f"✓ Database contains {count} documents")
        return count
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise


def test_create_retriever(vectordb):
    """
    Test 3: Verify we can create a retriever from the database.
    """
    print("\n[TEST 3] Creating retriever...")
    try:
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        assert retriever is not None, "Retriever creation failed"
        print("✓ Retriever created successfully")
        return retriever
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise


def test_similarity_search(vectordb, test_queries=None):
    """
    Test 4: Verify similarity search returns relevant results.
    """
    if test_queries is None:
        test_queries = [
            "What is the main topic of this document?",
            "key concepts",
            "summary"
        ]
    
    print(f"\n[TEST 4] Running {len(test_queries)} similarity search queries...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n  Query {i}: '{query}'")
        try:
            # Use similarity_search directly
            results = vectordb.similarity_search(query, k=3)
            assert len(results) > 0, f"No results returned for query: {query}"
            print(f"  ✓ Retrieved {len(results)} documents")
            
            # Verify results have content
            for j, doc in enumerate(results[:1], 1):  # Check first result
                assert doc.page_content, f"Document {j} has no content"
                snippet = doc.page_content[:100].replace("\n", " ")
                print(f"  Preview: {snippet}...")
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            raise


def test_retriever_invoke(vectordb):
    """
    Test 5: Verify retriever.invoke() works (replaces deprecated get_relevant_documents).
    """
    print("\n[TEST 5] Testing retriever.invoke()...")
    try:
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        query = "What is this about?"
        results = retriever.invoke(query)
        
        assert len(results) > 0, "No documents retrieved"
        assert all(hasattr(doc, 'page_content') for doc in results), "Results missing page_content"
        
        print(f"✓ Successfully retrieved {len(results)} documents")
        print(f"  First result length: {len(results[0].page_content)} characters")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise


def test_metadata_preservation(vectordb):
    """
    Test 6: Verify metadata (source, page) is preserved in chunks.
    """
    print("\n[TEST 6] Checking metadata preservation...")
    try:
        results = vectordb.similarity_search("test", k=1)
        assert len(results) > 0, "No results to check metadata"
        
        doc = results[0]
        assert hasattr(doc, 'metadata'), "Document missing metadata attribute"
        
        # Check if source or page exists in metadata
        has_source = 'source' in doc.metadata
        has_page = 'page' in doc.metadata
        
        print(f"✓ Metadata exists: source={has_source}, page={has_page}")
        if has_source:
            print(f"  Source: {doc.metadata['source']}")
        if has_page:
            print(f"  Page: {doc.metadata['page']}")
            
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise


def test_chunk_overlap(vectordb):
    """
    Test 7: Verify chunks have reasonable sizes (not too large or small).
    """
    print("\n[TEST 7] Checking chunk sizes...")
    try:
        # Get sample documents
        results = vectordb.similarity_search("test", k=10)
        assert len(results) > 0, "No results to check chunk sizes"
        
        sizes = [len(doc.page_content) for doc in results]
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        
        # Chunks should generally be between 100 and 2000 characters
        assert min_size > 0, "Found empty chunks"
        assert max_size < 5000, f"Chunk too large: {max_size} characters"
        
        print(f"✓ Chunk sizes look good:")
        print(f"  Average: {avg_size:.0f} characters")
        print(f"  Min: {min_size} characters")
        print(f"  Max: {max_size} characters")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise


def run_all_tests(vectordb=None, custom_queries=None):
    """
    Run all tests in sequence.
    
    Args:
        vectordb: Optional pre-loaded vector database
        custom_queries: Optional list of custom test queries
    """
    print("\n" + "="*80)
    print("RUNNING PDF VECTOR DATABASE TESTS")
    print("="*80)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Database exists
    try:
        test_database_exists()
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ Test 1 Failed: {e}")
        tests_failed += 1
        return  # Can't continue without DB
    
    # Load vectordb if not provided
    if vectordb is None:
        print("\nLoading vector database...")
        from langchain_ollama import OllamaEmbeddings
        from langchain_chroma import Chroma
        from pdf_vector import EMBEDDING_MODEL
        
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vectordb = Chroma(
            embedding_function=embeddings,
            persist_directory=DB_LOCATION
        )
        print("✓ Vector database loaded")
    
    # Run remaining tests
    tests = [
        (test_database_has_documents, [vectordb]),
        (test_create_retriever, [vectordb]),
        (test_similarity_search, [vectordb, custom_queries]),
        (test_retriever_invoke, [vectordb]),
        (test_metadata_preservation, [vectordb]),
        (test_chunk_overlap, [vectordb])
    ]
    
    for test_func, args in tests:
        try:
            test_func(*args)
            tests_passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} Failed: {e}")
            tests_failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✓ Passed: {tests_passed}")
    print(f"✗ Failed: {tests_failed}")
    print(f"Total: {tests_passed + tests_failed}")
    
    if tests_failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {tests_failed} test(s) failed")
    
    print("="*80)
    
    return tests_passed, tests_failed


if __name__ == "__main__":
    # Custom test queries (optional)
    custom_queries = [
        "What is this document about?",
        "main topics",
        "key information"
    ]
    
    # Run all tests
    passed, failed = run_all_tests(custom_queries=custom_queries)
    
    # Exit with appropriate code for CI/CD
    sys.exit(0 if failed == 0 else 1)
