from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("Amazon_popular_books.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large:335m")

db_location = "./chroma_langchain_ollama_db"
add_documents = not os.path.exists(db_location)