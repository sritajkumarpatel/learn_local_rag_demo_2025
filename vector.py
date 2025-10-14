from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load the Amazon books dataset from CSV
df = pd.read_csv("Amazon_popular_books_dataset.csv")

# Initialize Ollama embeddings model for vector representation
embeddings = OllamaEmbeddings(model="mxbai-embed-large:335m")

# Define the persistent storage location for the vector database
db_location = "./chroma_langchain_ollama_db"

# Check if database already exists to avoid duplicate data insertion
add_documents = not os.path.exists(db_location)

# Initialize or load the Chroma vector database with the embedding function
vectordb = Chroma(embedding_function=embeddings, persist_directory=db_location)

# Only add documents if this is the first time creating the database
if add_documents:
    documents = []
    
    # Process each book entry in the dataset
    for index, row in df.iterrows():
        # Format the main content with essential book information for retrieval
        # This content will be embedded and used for similarity search
        page_content = f"""Title: {row.get('title', 'N/A')}
Rating: {row.get('rating', 'N/A')}
Reviews Count: {row.get('reviews_count', 'N/A')}
Authors: {row.get('authors', 'N/A')}
Publisher: {row.get('publisher', 'N/A')}
Publication Date: {row.get('publication_date', 'N/A')}
Categories: {row.get('categories', 'N/A')}
Price: {row.get('final_price', 'N/A')}"""
        
        # Create a Document object with page content and comprehensive metadata
        # Metadata allows for filtering and additional context without affecting embeddings
        doc = Document(
            page_content=page_content,
            metadata={
                "availability": row.get("availability"),
                "brand": row.get("brand"),
                "currency": row.get("currency"),
                "date_first_available": row.get("date_first_available"),
                "delivery": row.get("delivery"),
                "department": row.get("department"),
                "discount": row.get("discount"),
                "domain": row.get("domain"),
                "features": row.get("features"),
                "final_price": row.get("final_price"),
                "format": row.get("format"),
                "image_url": row.get("image_url"),
                "images_count": row.get("images_count"),
                "initial_price": row.get("initial_price"),
                "item_weight": row.get("item_weight"),
                "manufacturer": row.get("manufacturer"),
                "model_number": row.get("model_number"),
                "plus_content": row.get("plus_content"),
                "product_dimensions": row.get("product_dimensions"),
                "rating": row.get("rating"),
                "reviews_count": row.get("reviews_count"),
                "root_bs_rank": row.get("root_bs_rank"),
                "seller_id": row.get("seller_id"),
                "seller_name": row.get("seller_name"),
                "timestamp": row.get("timestamp"),
                "title": row.get("title"),
                "upc": row.get("upc"),
                "url": row.get("url"),
                "video": row.get("video"),
                "video_count": row.get("video_count"),
                "categories": row.get("categories"),
                "best_sellers_rank": row.get("best_sellers_rank"),
                "buybox_seller": row.get("buybox_seller"),
                "image": row.get("image"),
                "number_of_sellers": row.get("number_of_sellers"),
                "colors": row.get("colors")
            }
        )
        documents.append(doc)
    
    # Add all documents to the vector database
    vectordb.add_documents(documents)
    
    # Persist the database to disk for future use
    vectordb.persist()

# Create a retriever for similarity search
# Returns top 5 most similar documents for any query
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3000})
# Perform a sample query to test retrieval
query = "Which book has the highest reviews count?"
results = retriever.invoke(query)

# Find the book with the highest reviews count from the results
max_reviews_book = None
max_reviews_count = 0

for doc in results:
    reviews_count = doc.metadata.get('reviews_count', 0)
    # Convert to int if it's a string, handle 'N/A' or None values
    try:
        reviews_count = int(float(reviews_count)) if reviews_count else 0
    except (ValueError, TypeError):
        reviews_count = 0
    
    if reviews_count > max_reviews_count:
        max_reviews_count = reviews_count
        max_reviews_book = doc

# Display the book with the highest reviews count (outside the loop)
if max_reviews_book:
    print(f"\n{'='*80}")
    print(f"Book with Highest Reviews Count:")
    print(f"{'='*80}")
    print(f"Title: {max_reviews_book.metadata.get('title', 'N/A')}")
    print(f"Reviews Count: {max_reviews_count:,}")
    print(f"Rating: {max_reviews_book.metadata.get('rating', 'N/A')}")
    print(f"Authors: {max_reviews_book.metadata.get('authors', 'N/A')}")
    print(f"Price: {max_reviews_book.metadata.get('final_price', 'N/A')}")
    print(f"{'='*80}")
else:
    print("\nNo valid book found with reviews count.")
