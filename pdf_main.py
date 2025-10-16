"""
RAG Comparison Demo
===================
This module demonstrates the difference between using an LLM with and without
Retrieval-Augmented Generation (RAG) by comparing answers from both approaches.
"""

from langchain_ollama import OllamaLLM
import pdf_vector

model = OllamaLLM(model="llama3.2:3b")

# Load the vector database once at startup
vectordb_pdf = pdf_vector.create_vector_db_from_pdfs(force_rebuild=False)
retriever_pdf = vectordb_pdf.as_retriever(search_type="similarity", search_kwargs={"k": 10})


def ask_without_rag(question: str) -> str:
    """
    Ask the LLM directly without vector database context.
    """
    prompt = f"You are a board games expert. Answer the following question:\n\nQuestion: {question}\n\nAnswer:"
    return model.invoke(prompt)

def ask_with_rag(question: str) -> str:
    """
    Answer questions using Retrieval-Augmented Generation with book data.
    
    Args:
        question: The user's question about books
        retriever: The retriever instance to use for document retrieval
        
    Returns:
        AI-generated answer based on retrieved documents
    """
    retrieved_docs = retriever_pdf.invoke(question)
    
    prompt = f"You are a board games expert. Answer the following question based on the data provided.\n\n"
    prompt += f"Here are the relevant info:\n\n"
    
    for i, doc in enumerate(retrieved_docs, 1):
        prompt += f"=== Board Game {i} ===\n{doc.page_content}\n\n"
    
    prompt += f"Question: {question}\n\nProvide a clear answer based on the board games data above:"
    
    return model.invoke(prompt)

user_question = "How many players can play CATANIC RIDE TO HEAVEN?"

if __name__ == "__main__":
    question = user_question

    # Get the answer using LLM data only
    print("WITHOUT RAG (Using LLM Knowledge Only)")
    print("="*80)
    resultFromRegularLLM = ask_without_rag(question)
    print(resultFromRegularLLM)

    print("\n" + "="*80)
    print("WITH RAG (Using PDFs Content)")
    print("="*80)

    # Get the answer using RAG with PDFs content
    resultFromRAGSystem = ask_with_rag(question)
    print(resultFromRAGSystem)