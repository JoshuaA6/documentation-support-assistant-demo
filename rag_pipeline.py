"""
Step 4: Build the RAG Pipeline (Retrieval-Augmented Generation)

This script combines:
1. Retrieval: Find relevant docs from ChromaDB
2. Augmentation: Add them to the LLM context
3. Generation: LLM answers using retrieved docs + citations

The AI Support Agent is built on this foundation!
"""

import os
import json
import html
import re
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse
import chromadb
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

VECTOR_DB_FOLDER = "vector_db"

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_FOLDER)
collection = None  # Initialize lazily

LANGUAGE_TO_LOCALES = {
    "English": ["en-us", "en"],
    "Spanish": ["es", "es-es", "es-419"],
    "French": ["fr"],
    "German": ["de"],
    "Portuguese": ["pt", "pt-br"],
}


def get_config_value(name: str, default: str | None = None) -> str | None:
    """Load config from environment first, then Streamlit secrets when available."""
    env_value = os.getenv(name)
    if env_value:
        return env_value

    try:
        import streamlit as st

        secret_value = st.secrets.get(name)
        if secret_value:
            return str(secret_value)
    except Exception:
        pass

    return default


def create_azure_client() -> AzureOpenAI:
    """Create an Azure OpenAI client using local env vars or Streamlit secrets."""
    return AzureOpenAI(
        api_version="2024-02-01",
        azure_endpoint=get_config_value("AZURE_OPENAI_ENDPOINT"),
        api_key=get_config_value("AZURE_OPENAI_API_KEY"),
    )


def extract_locale_from_url(url: str) -> str:
    """Extract the locale slug from a Klaviyo help-center URL."""
    if not url:
        return ""

    parts = urlparse(url).path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "hc":
        return parts[1].lower()
    return ""


def extract_article_id_from_url(url: str) -> str:
    """Extract the article ID from a Klaviyo help-center URL when present."""
    if not url:
        return ""

    parts = urlparse(url).path.strip("/").split("/")
    if "articles" in parts:
        article_index = parts.index("articles")
        if article_index + 1 < len(parts):
            return parts[article_index + 1]
    return ""


def format_recent_history(chat_history: Optional[List[Dict]], max_messages: int = 4) -> str:
    """Format a small slice of recent chat history for follow-up questions."""
    if not chat_history:
        return ""

    def strip_legacy_wrapper(text: str) -> str:
        cleaned = html.unescape(text)
        cleaned = re.sub(r"^\s*<strong>.*?</strong><br\s*/?>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"^\s*Klaviyo Assistant:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*You:\s*", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    recent_messages = chat_history[-max_messages:]
    formatted = []
    for message in recent_messages:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = strip_legacy_wrapper((message.get("content") or "").strip())
        if content:
            formatted.append(f"{role}: {content}")

    if not formatted:
        return ""

    return "Recent conversation:\n" + "\n".join(formatted)


class RAGPipeline:
    """
    Retrieval-Augmented Generation Pipeline
    
    Workflow:
    1. User asks a question
    2. Convert question to embedding
    3. Search ChromaDB for similar chunks
    4. Build context from top-k chunks
    5. Send to LLM with system prompt
    6. Extract answer + citations + follow-ups
    """
    
    def __init__(self, top_k: int = 3):
        """
        Initialize RAG pipeline.
        
        Args:
            top_k: Number of relevant chunks to retrieve
        """
        self.top_k = top_k
        self.collection = None  # Initialize lazily
        self.client = create_azure_client()
        self.embedding_deployment = get_config_value(
            "AZURE_OPENAI_DEPLOYMENT_NAME", "text-embedding-3-small"
        )
        self.llm_deployment = get_config_value(
            "AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1"
        )
        self.system_prompt = """You are a helpful Klaviyo documentation assistant. Your job is to:

1. Answer questions ONLY based on the provided documentation
2. Be specific and actionable
3. Suggest related topics
4. Ask follow-up questions if clarification is needed
5. Use recent conversation only to resolve follow-up context like "that", "this", or "what about pricing?"
6. If retrieval looks limited, say that clearly before giving the best documentation-based answer you can

If the documentation doesn't contain the answer, say: "I couldn't find specific information about this in the Klaviyo documentation. I recommend checking the Klaviyo help center or contacting support."

Always cite your sources!
Do not include HTML tags like <strong>, <br>, <div>, or XML-style markup in your answer."""
    
    def _get_collection(self):
        """Lazily initialize ChromaDB collection."""
        if self.collection is None:
            self.collection = chroma_client.get_or_create_collection(name="klaviyo_docs")
        return self.collection
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for query using Azure OpenAI."""
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_deployment,
        )
        return response.data[0].embedding
    
    def _locale_priority(self, locale: str, response_language: Optional[str]) -> int:
        """Rank document locales so we can prefer the user's language without dropping good sources."""
        preferred_locales = LANGUAGE_TO_LOCALES.get(response_language or "", [])
        if locale in preferred_locales:
            return 0
        if locale.startswith("en"):
            return 1
        if not locale:
            return 2
        return 3

    def retrieve_context(
        self,
        query: str,
        response_language: Optional[str] = None,
    ) -> Tuple[str, List[Dict], Dict]:
        """
        Retrieve relevant context from ChromaDB.
        
        Returns:
            context: String of concatenated relevant chunks
            metadata: List of source metadata for citations
        """
        # Generate embedding for query
        query_embedding = self.generate_embedding(query)
        collection = self._get_collection()
        
        # Search ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max(self.top_k * 4, self.top_k),
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [[]])
        distances = results.get("distances", [[]])
        if not documents or not documents[0]:
            return "No relevant documentation was found.", [], {
                "weak_retrieval": True,
                "best_distance": None,
                "average_distance": None,
                "distinct_sources": 0,
            }

        deduped_candidates: Dict[str, Dict] = {}
        raw_candidates = zip(documents[0], metadatas[0], distances[0] if distances else [])

        for doc, meta, distance in raw_candidates:
            source_name = (
                meta.get("pdf_name")
                or meta.get("article_title")
                or meta.get("document_name")
                or meta.get("source")
                or "Klaviyo Help Center"
            )
            url = meta.get("url", "")
            locale = extract_locale_from_url(url)
            article_id = extract_article_id_from_url(url)
            source_key = article_id or source_name.lower().strip()

            candidate = {
                "doc": doc,
                "meta": meta,
                "source_name": source_name,
                "url": url,
                "locale": locale,
                "article_id": article_id,
                "source_key": source_key,
                "distance": float(distance) if distance is not None else 1.0,
                "locale_priority": self._locale_priority(locale, response_language),
            }

            existing = deduped_candidates.get(source_key)
            if existing is None:
                deduped_candidates[source_key] = candidate
                continue

            existing_rank = (existing["locale_priority"], existing["distance"])
            candidate_rank = (candidate["locale_priority"], candidate["distance"])
            if candidate_rank < existing_rank:
                deduped_candidates[source_key] = candidate

        selected_candidates = sorted(
            deduped_candidates.values(),
            key=lambda item: (item["distance"], item["locale_priority"]),
        )[: self.top_k]

        if not selected_candidates:
            return "No relevant documentation was found.", [], {
                "weak_retrieval": True,
                "best_distance": None,
                "average_distance": None,
                "distinct_sources": 0,
            }
        
        # Build context string
        context = "Here is relevant documentation:\n\n"
        metadata_list = []
        
        for i, candidate in enumerate(selected_candidates, 1):
            doc = candidate["doc"]
            meta = candidate["meta"]
            source_name = candidate["source_name"]
            context += f"[Source {i}: {source_name}]\n{doc}\n\n"
            metadata_list.append({
                "source": source_name,
                "chunk_index": meta.get('chunk_index', 0),
                "category": meta.get("category", "General"),
                "url": candidate["url"],
                "locale": candidate["locale"],
                "article_id": candidate["article_id"],
                "distance": candidate["distance"],
                "relevance_score": round(max(0.0, 1.0 - candidate["distance"]), 3),
                "content_preview": doc[:100] + "..." if len(doc) > 100 else doc
            })

        best_distance = min(candidate["distance"] for candidate in selected_candidates)
        average_distance = sum(candidate["distance"] for candidate in selected_candidates) / len(selected_candidates)
        retrieval_meta = {
            "weak_retrieval": (
                len(selected_candidates) < 2
                or best_distance > 0.45
                or average_distance > 0.55
            ),
            "best_distance": round(best_distance, 4),
            "average_distance": round(average_distance, 4),
            "distinct_sources": len(selected_candidates),
        }

        return context, metadata_list, retrieval_meta
    
    def generate_answer(
        self,
        user_query: str,
        context: str,
        response_language: Optional[str] = None,
        chat_history: Optional[List[Dict]] = None,
        weak_retrieval: bool = False,
    ) -> str:
        """
        Generate answer using LLM with retrieved context.
        
        Args:
            user_query: The user's question
            context: Retrieved documentation context
        
        Returns:
            LLM-generated answer
        """
        language_instruction = ""
        if response_language:
            language_instruction = (
                f"Write the answer in {response_language}. If the source material is in a "
                "different language, you may translate it for the user while preserving the meaning."
            )
        retrieval_instruction = ""
        if weak_retrieval:
            retrieval_instruction = (
                "The retrieved documentation may be only partially relevant. "
                "Briefly say that the answer may be limited before you answer."
            )
        history_block = format_recent_history(chat_history)

        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": (
                    f"{context}\n\n"
                    f"{history_block}\n\n"
                    f"User question: {user_query}\n\n"
                    "Provide a helpful answer based on the documentation above. "
                    f"{language_instruction} {retrieval_instruction}"
                ),
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.llm_deployment,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )
        
        return response.choices[0].message.content
    
    def generate_followups(
        self,
        user_query: str,
        answer: str,
        response_language: Optional[str] = None,
    ) -> List[str]:
        """
        Generate suggested follow-up questions.
        
        Args:
            user_query: Original user question
            answer: Generated answer
        
        Returns:
            List of 3 suggested follow-up questions
        """
        language_instruction = ""
        if response_language:
            language_instruction = (
                f"Write all follow-up questions in {response_language}. "
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Generate 3 natural follow-up questions "
                    "based on the user's original question and the answer provided. "
                    f"{language_instruction}Format as a JSON array of strings."
                ),
            },
            {
                "role": "user",
                "content": f"Original question: {user_query}\n\nAnswer provided: {answer}\n\nGenerate 3 follow-up questions users might ask next. Return ONLY a JSON array like [\"Q1\", \"Q2\", \"Q3\"]"
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.llm_deployment,
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )
        
        try:
            # Extract JSON from response
            response_text = response.choices[0].message.content
            # Find JSON array in response
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            if start >= 0 and end > start:
                followups = json.loads(response_text[start:end])
                return followups[:3]  # Return top 3
        except:
            pass
        
        return []
    
    def answer_question(
        self,
        query: str,
        verbose: bool = False,
        response_language: Optional[str] = None,
        chat_history: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Complete RAG pipeline: Retrieve → Augment → Generate
        
        Args:
            query: User's question
            verbose: Print intermediate steps if True
        
        Returns:
            Dictionary with answer, citations, and follow-ups
        """
        if verbose:
            print(f"\n🔍 Query: {query}")
            print("-" * 60)
        
        # Step 1: Retrieve relevant context
        if verbose:
            print("📚 Retrieving relevant documentation...")
        context, sources, retrieval_meta = self.retrieve_context(
            query,
            response_language=response_language,
        )
        
        if verbose:
            print(f"   Found {len(sources)} relevant sources")
        
        # Step 2: Generate answer
        if verbose:
            print("🤖 Generating answer with LLM...")
        answer = self.generate_answer(
            query,
            context,
            response_language=response_language,
            chat_history=chat_history,
            weak_retrieval=retrieval_meta["weak_retrieval"],
        )
        
        if verbose:
            print("✓ Answer generated")
        
        # Step 3: Generate follow-ups
        if verbose:
            print("💡 Generating follow-up questions...")
        followups = self.generate_followups(
            query,
            answer,
            response_language=response_language,
        )
        
        if verbose:
            print("✓ Follow-ups generated")
        
        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "followup_questions": followups,
            "retrieval": retrieval_meta,
        }


def main():
    """Demo the RAG pipeline with example queries."""
    
    print("=" * 70)
    print("Step 4: RAG Pipeline Demo")
    print("=" * 70)
    
    # Initialize RAG pipeline
    rag = RAGPipeline(top_k=3)
    
    # Example queries
    example_queries = [
        "How do I create a flow?",
        "What are the best practices for testing a flow?",
        "How do I add conditional logic to my flow?",
    ]
    
    for query in example_queries:
        result = rag.answer_question(query, verbose=True)
        
        print("\n" + "=" * 70)
        print("📋 ANSWER:")
        print("=" * 70)
        print(result['answer'])
        
        print("\n📚 SOURCES USED:")
        for i, source in enumerate(result['sources'], 1):
            print(f"   {i}. {source['source']}")
            print(f"      → {source['content_preview']}")
        
        if result['followup_questions']:
            print("\n💡 SUGGESTED FOLLOW-UPS:")
            for i, q in enumerate(result['followup_questions'], 1):
                print(f"   {i}. {q}")
        
        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()
