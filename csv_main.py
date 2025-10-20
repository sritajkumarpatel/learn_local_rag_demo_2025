"""
RAG Comparison Demo
===================
This module demonstrates the difference between using an LLM with and without
Retrieval-Augmented Generation (RAG) by comparing answers from both approaches.
"""

from langchain_ollama import OllamaLLM
import csv_vector

model = OllamaLLM(model="llama3.2:3b")

# Load the vector database once at startup
vectordb_csvbooks = csv_vector.create_vector_db_from_csv(force_rebuild=False)
retriever_csvbooks = vectordb_csvbooks.as_retriever(search_type="similarity", search_kwargs={"k": 10})


def ask_without_rag(question: str) -> str:
    """
    Ask the LLM directly without vector database context.
    """
    prompt = f"You are a book expert. Answer the following question:\n\nQuestion: {question}\n\nAnswer:"
    return model.invoke(prompt)

def ask_with_rag(question: str) -> str:
    """
    Answer questions using Retrieval-Augmented Generation with pdf data.
    
    Args:
        question: The user's question about pdf documents
        retriever: The retriever instance to use for document retrieval
        
    Returns:
        AI-generated answer based on retrieved documents
    """
    retrieved_docs = retriever_csvbooks.invoke(question)
    
    prompt = f"You are a book expert. Answer the following question based on the book data provided.\n\n"
    prompt += f"Here are the relevant books:\n\n"
    
    for i, doc in enumerate(retrieved_docs, 1):
        prompt += f"=== Book {i} ===\n{doc.page_content}\n\n"
    
    prompt += f"Question: {question}\n\nProvide a clear answer based on the book data above:"
    
    return model.invoke(prompt)

user_question = "Give popular books by author 'Timothy Wells' and 'Thomas Waters'"

if __name__ == "__main__":
    question = user_question

    # Get the answer using LLM data only
    print("WITHOUT RAG (Using LLM Knowledge Only)")
    print("="*80)
    resultFromRegularLLM = ask_without_rag(question)
    print(resultFromRegularLLM)

    print("\n" + "="*80)
    print("WITH RAG (Using CSV Content)")
    print("="*80)

    # Get the answer using RAG with CSV content
    resultFromRAGSystem = ask_with_rag(question)
    print(resultFromRAGSystem)