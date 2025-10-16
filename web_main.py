"""
Web Scraping RAG Comparison Demo
=================================
This module demonstrates the difference between using an LLM with and without
Retrieval-Augmented Generation (RAG) by comparing answers from both approaches
using web-scraped content.
"""

from langchain_ollama import OllamaLLM
import web_vector

model = OllamaLLM(model="llama3.2:3b")

# Load the vector database once at startup
vectordb_web = web_vector.create_vector_db_from_urls(
    urls=[],  # Will be set in the function
    force_rebuild=False
)
retriever_web = vectordb_web.as_retriever(search_type="similarity", search_kwargs={"k": 10})


def ask_without_rag(question: str) -> str:
    """
    Ask the LLM directly without vector database context.
    """
    prompt = f"You are a knowledgeable assistant. Answer the following question:\n\nQuestion: {question}\n\nAnswer:"
    return model.invoke(prompt)


def ask_with_rag(question: str, urls_to_scrape: list) -> str:
    """
    Answer questions using Retrieval-Augmented Generation with web-scraped content.

    Args:
        question: The user's question
        urls_to_scrape: List of URLs to scrape for context

    Returns:
        AI-generated answer based on retrieved web content
    """
    # Update vector database with new URLs if needed
    global vectordb_web, retriever_web
    if urls_to_scrape:
        vectordb_web = web_vector.create_vector_db_from_urls(
            urls_to_scrape,
            force_rebuild=False,
            auto_update=True
        )
        retriever_web = vectordb_web.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    retrieved_docs = retriever_web.invoke(question)

    prompt = f"You are a knowledgeable assistant. Answer the following question based on the web content provided.\n\n"
    prompt += f"Here are the relevant web pages:\n\n"

    for i, doc in enumerate(retrieved_docs, 1):
        prompt += f"=== Web Page {i} ===\n{doc.page_content}\n\n"

    prompt += f"Question: {question}\n\nProvide a clear answer based on the web content above:"

    return model.invoke(prompt)


# Example URLs for different topics
DINOSAUR_URLS = [
    "https://en.wikipedia.org/wiki/Dinosaur"
]

if __name__ == "__main__":
    # Choose topic and URLs
    topic = "dinosaurs"
    urls = DINOSAUR_URLS
    question = "Who identified dragon bones?"

    print(f"Topic: {topic.upper()}")
    print(f"URLs: {len(urls)} pages")
    print(f"Question: {question}")
    print("\n" + "="*80)

    # Get the answer using LLM data only
    print("WITHOUT RAG (Using LLM Knowledge Only)")
    print("="*80)
    resultFromRegularLLM = ask_without_rag(question)
    print(resultFromRegularLLM)

    print("\n" + "="*80)
    print("WITH RAG (Using Web-Scraped Content)")
    print("="*80)

    # Get the answer using RAG with web content
    resultFromRAGSystem = ask_with_rag(question, urls)
    print(resultFromRAGSystem)