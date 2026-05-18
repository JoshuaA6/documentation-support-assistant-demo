"""
Step 3: Extract Docs → Chunk → Embed → Store in Vector DB

This script supports two input sources:
1. PDFs in `pdfs/`
2. Scraped article JSON files in `scraped_docs/articles/`
"""

import os
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from pypdf import PdfReader
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
PDFS_FOLDER = "pdfs"
SCRAPED_ARTICLES_FOLDER = "scraped_docs/articles"
CHUNKS_FOLDER = "chunks"
VECTOR_DB_FOLDER = "vector_db"

# Create necessary folders
Path(CHUNKS_FOLDER).mkdir(exist_ok=True)
Path(VECTOR_DB_FOLDER).mkdir(exist_ok=True)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_FOLDER)
collection = chroma_client.get_or_create_collection(
    name="klaviyo_docs",
    metadata={"hnsw:space": "cosine"}
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for source selection."""
    parser = argparse.ArgumentParser(description="Embed Klaviyo docs into ChromaDB.")
    parser.add_argument(
        "--input-mode",
        choices=["all", "pdf", "scraped"],
        default="all",
        help="Choose which document sources to ingest.",
    )
    parser.add_argument(
        "--scraped-locale",
        default="all",
        help="Filter scraped articles by locale in their URL, e.g. 'en-us'. Use 'all' to keep every locale.",
    )
    return parser.parse_args()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text


def load_scraped_article(article_path: Path) -> Optional[Dict]:
    """Load a scraped article JSON file."""
    try:
        with article_path.open("r", encoding="utf-8") as handle:
            article = json.load(handle)
    except Exception as e:
        print(f"Error reading {article_path}: {e}")
        return None

    content = article.get("content", "").strip()
    if not content:
        print(f"  ⚠️  No content found in {article_path.name}")
        return None
    return article


def scraped_article_matches_locale(article: Dict, scraped_locale: str) -> bool:
    """Check whether a scraped article URL matches the requested locale."""
    if scraped_locale == "all":
        return True

    url = article.get("url", "")
    return f"/hc/{scraped_locale.lower()}/" in url.lower()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Full text to chunk
        chunk_size: Target size of each chunk (characters)
        overlap: Overlap between chunks (characters)
    
    Returns:
        List of text chunks
    """
    # Split by sentences first for better coherence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if len(chunk) > 50]  # Filter out very small chunks


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for text using Azure OpenAI.
    
    Args:
        text: Text to embed
    
    Returns:
        List of embedding values
    """
    try:
        response = client.embeddings.create(
            input=text,
            model=DEPLOYMENT_NAME,
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def index_chunks(
    source_name: str,
    document_name: str,
    chunks: List[str],
    docs_index: List[Dict],
    chunk_counter: int,
    source_type: str,
    category: Optional[str] = None,
    url: Optional[str] = None,
) -> int:
    """Embed and store chunks for a single source document."""
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{chunk_counter}"

        embedding = generate_embedding(chunk)
        if embedding is None:
            print(f"    ⚠️  Failed to embed chunk {i + 1}")
            continue

        metadata = {
            "source": source_name,
            "document_name": document_name,
            "chunk_index": i,
            "total_chunks_in_doc": len(chunks),
            "source_type": source_type,
            "category": category or "General",
            "url": url or "",
        }

        if source_type == "pdf":
            metadata["pdf_name"] = document_name
        else:
            metadata["article_title"] = document_name

        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[metadata],
        )

        docs_index.append(
            {
                "id": chunk_id,
                "source": source_name,
                "document_name": document_name,
                "chunk_index": i,
                "content_preview": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                "content_length": len(chunk),
                "source_type": source_type,
                "category": category or "General",
                "url": url or "",
            }
        )

        chunk_counter += 1
        if (i + 1) % 5 == 0:
            print(f"    • Processed {i + 1}/{len(chunks)} chunks")

    return chunk_counter


def process_all_sources(input_mode: str = "all", scraped_locale: str = "all"):
    """Main pipeline: Extract → Chunk → Embed → Store."""
    docs_index = []
    chunk_counter = 1

    pdf_files = sorted(Path(PDFS_FOLDER).glob("*.pdf")) if input_mode in {"all", "pdf"} else []
    scraped_article_files = sorted(Path(SCRAPED_ARTICLES_FOLDER).glob("*.json")) if input_mode in {"all", "scraped"} else []

    if not pdf_files and not scraped_article_files:
        print(f"No PDF files found in {PDFS_FOLDER}")
        print(f"No scraped articles found in {SCRAPED_ARTICLES_FOLDER}")
        return

    print(f"Found {len(pdf_files)} PDF files")
    print(f"Found {len(scraped_article_files)} scraped articles")
    print("-" * 50)

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        text = extract_text_from_pdf(str(pdf_path))
        if not text:
            print(f"  ⚠️  No text extracted from {pdf_path.name}")
            continue

        print(f"  ✓ Extracted {len(text)} characters")
        chunks = chunk_text(text)
        print(f"  ✓ Created {len(chunks)} chunks")
        chunk_counter = index_chunks(
            source_name=pdf_path.stem,
            document_name=pdf_path.name,
            chunks=chunks,
            docs_index=docs_index,
            chunk_counter=chunk_counter,
            source_type="pdf",
            category="Flows",
        )
        print(f"  ✓ Stored all chunks in ChromaDB")

    for article_path in scraped_article_files:
        article = load_scraped_article(article_path)
        if article is None:
            continue
        if not scraped_article_matches_locale(article, scraped_locale):
            continue

        print(f"\nProcessing scraped article: {article.get('title', article_path.stem)}")
        text = article["content"]
        print(f"  ✓ Extracted {len(text)} characters")
        chunks = chunk_text(text)
        print(f"  ✓ Created {len(chunks)} chunks")
        chunk_counter = index_chunks(
            source_name=article.get("title", article_path.stem),
            document_name=article.get("title", article_path.stem),
            chunks=chunks,
            docs_index=docs_index,
            chunk_counter=chunk_counter,
            source_type="scraped_article",
            category=article.get("category"),
            url=article.get("url"),
        )
        print(f"  ✓ Stored all chunks in ChromaDB")

    with open("docs_index.json", "w") as f:
        json.dump(docs_index, f, indent=2)

    print("\n" + "=" * 50)
    print(f"✅ Pipeline Complete!")
    print(f"   • Total chunks processed: {chunk_counter - 1}")
    print(f"   • ChromaDB stored at: {VECTOR_DB_FOLDER}")
    print(f"   • Index saved to: docs_index.json")
    print("=" * 50)


if __name__ == "__main__":
    args = parse_args()
    process_all_sources(input_mode=args.input_mode, scraped_locale=args.scraped_locale)
