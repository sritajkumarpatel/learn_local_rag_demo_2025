from langchain_ollama import OllamaLLM
import vector as vec

model = OllamaLLM(model="llama3.2:3b")

def ask_with_rag(question: str) -> str:
    # Retrieve relevant documents
    retrieved_docs = vec.retriever.invoke(question)
    
    # Build prompt with context
    prompt = f"You are a book expert. Answer the following question based on the book data provided.\n\n"
    prompt += f"Here are the relevant books:\n\n"
    
    for i, doc in enumerate(retrieved_docs, 1):
        prompt += f"=== Book {i} ===\n{doc.page_content}\n\n"
    
    prompt += f"Question: {question}\n\nProvide a clear answer based on the book data above:"
    
    # Generate answer
    return model.invoke(prompt)

if __name__ == "__main__":
    question = "Which book has the highest reviews count?"
    
    result = ask_with_rag(question)
    print(result)