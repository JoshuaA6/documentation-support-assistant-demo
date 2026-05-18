"""
Step 3B: Test your embeddings pipeline & retrieve docs

This script:
1. Tests your ChromaDB connection
2. Performs a sample query
3. Shows you how retrieval works (this is the foundation for Step 4: RAG)
"""

import os
import chromadb
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "text-embedding-3-small")
VECTOR_DB_FOLDER = "vector_db"


def generate_embedding(text: str) -> list:
    """Generate embedding for a query."""
    response = client.embeddings.create(
        input=text,
        model=DEPLOYMENT_NAME,
    )
    return response.data[0].embedding


def retrieve_similar_docs(query: str, top_k: int = 3) -> list:
    """
    Retrieve top_k most similar documents for a query.
    
    This is the core of RAG (Retrieval-Augmented Generation).
    """
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_FOLDER)
    collection = chroma_client.get_or_create_collection(name="klaviyo_docs")
    
    # Generate embedding for query
    query_embedding = generate_embedding(query)
    
    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    return results


def main():
    """Test the retrieval system."""
    print("=" * 60)
    print("Step 3B: Testing Your Embeddings & Retrieval System")
    print("=" * 60)
    
    # Example queries to test
    test_queries = [
        "How do I create a flow?",
        "How do I add a conditional split?",
        "How do I test a flow?",
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 60)
        
        try:
            results = retrieve_similar_docs(query, top_k=3)
            
            if results['documents'][0]:
                for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                    print(f"\n{i}. From: {metadata['pdf_name']}")
                    print(f"   Preview: {doc[:150]}...")
                    print(f"   Confidence: {results['distances'][0][i-1]:.4f}")
            else:
                print("   ⚠️  No results found")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Retrieval test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
