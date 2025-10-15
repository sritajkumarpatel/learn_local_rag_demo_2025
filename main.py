from langchain_ollama import OllamaLLM
import csv_vector

# Initialize the Ollama language model with llama3.2:3b
model = OllamaLLM(model="llama3.2:3b")

# Load the vector database once at startup
vectordb = csv_vector.create_vector_db_from_csv(force_rebuild=False)
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 10})

def ask_without_rag(question: str) -> str:
    """
    Ask the LLM directly WITHOUT any context from the vector database.
    The model will answer based only on its training data.
    """
    prompt = f"You are a book expert. Answer the following question:\n\nQuestion: {question}\n\nAnswer:"
    return model.invoke(prompt)

def ask_with_rag(question: str) -> str:
    """
    Retrieval-Augmented Generation (RAG) function to answer questions using book data.
    
    Args:
        question: The user's question about books
        
    Returns:
        AI-generated answer based on retrieved relevant documents
    """
    # Retrieve relevant documents from the vector store based on the question
    retrieved_docs = retriever.invoke(question)
    
    # Build the initial prompt with system instructions
    prompt = f"You are a book expert. Answer the following question based on the book data provided.\n\n"
    prompt += f"Here are the relevant books:\n\n"
    
    # Add each retrieved document to the prompt as context
    for i, doc in enumerate(retrieved_docs, 1):
        prompt += f"=== Book {i} ===\n{doc.page_content}\n\n"
    
    # Append the user's question and instructions for answering
    prompt += f"Question: {question}\n\nProvide a clear answer based on the book data above:"
    
    # Generate and return the answer using the LLM
    return model.invoke(prompt)

user_question = "Give popular books by author 'Timothy Wells' and 'Thomas Waters'"

if __name__ == "__main__":
    # Example question to test the RAG system
    question = user_question

    # Get the answer using LLM data only
    resultFromRegularLLM = ask_without_rag(question)
    print(resultFromRegularLLM)

    print("\n" + "="*60)
    print("WITH RAG (Using Vector Database)")
    print("="*60 + "\n")

    # Get the answer using RAG
    resultFromRAGSystem = ask_with_rag(question)
    print(resultFromRAGSystem)