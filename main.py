from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


model = OllamaLLM(model="llama3.2:3b")

template = """
You are an expert in answering questions about Books
"""

templateForRAG = """
You are an expert in answering questions about Books

Here are the book details: {details}
Please answer the question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

promptForRAG = ChatPromptTemplate.from_template(templateForRAG)
chainForRag = promptForRAG | model

result = chain.invoke({
    "question": "Which book is the highest rated?"
})

resultForRAG = chainForRag.invoke({
    "details": [],
    "question": "Which book is the highest rated?"
})


print(result)
print("\n\n\n.............................................\n\n\n")
print(resultForRAG)