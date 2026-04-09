"""
generation/generator.py — LLM Response Generator.

Uses Ollama (local) with Qwen2.5:7B.
Builds a grounded prompt from retrieved context chunks.
Supports both standard and streaming responses.
"""
from typing import List, Dict, Any, Generator
from loguru import logger
import ollama
from src.config import config


SYSTEM_PROMPT = """You are a precise, helpful assistant that answers questions
based strictly on the provided context documents.

Rules:
1. Answer ONLY from the context. Do not add external knowledge.
2. If the context does not contain the answer, say: "I could not find this in the provided documents."
3. Always cite the source document(s) you used, e.g. [Source: filename.pdf, Page 3].
4. Be concise and factual. Avoid hallucination.
"""


def generate_response(
    query: str,
    context_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate a grounded response from retrieved context.

    Args:
        query: User's question.
        context_chunks: Reranked top-N chunks from retriever.

    Returns:
        Dict with 'answer', 'sources', 'context_used'.
    """
    if not context_chunks:
        return {
            "answer": "No relevant documents found to answer this question.",
            "sources": [],
            "context_used": []
        }

    # Build context string with source citations
    context_str = _build_context_string(context_chunks)

    prompt = f"""Context documents:
{context_str}

---
Question: {query}

Answer (cite sources):"""

    try:
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": 0.1,   # Low temp for factual RAG
                "num_predict": 512,
                "top_p": 0.9
            }
        )
        answer = response["message"]["content"].strip()
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        answer = f"Error generating response: {str(e)}"

    sources = _extract_sources(context_chunks)

    return {
        "answer": answer,
        "sources": sources,
        "context_used": [c["content"][:200] + "..." for c in context_chunks]
    }


def generate_stream(
    query: str,
    context_chunks: List[Dict[str, Any]]
) -> Generator[str, None, None]:
    """
    Streaming version of generate_response.
    Yields tokens one-by-one for real-time display.
    """
    if not context_chunks:
        yield "No relevant documents found to answer this question."
        return

    context_str = _build_context_string(context_chunks)
    prompt = f"""Context documents:
{context_str}

---
Question: {query}

Answer (cite sources):"""

    try:
        stream = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.1, "num_predict": 512},
            stream=True
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token
    except Exception as e:
        logger.error(f"Streaming generation failed: {e}")
        yield f"Error: {str(e)}"


def _build_context_string(chunks: List[Dict[str, Any]]) -> str:
    """Format chunks into a readable context block with citations."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        section = chunk.get("section", "")
        header = f"[Doc {i}: {source}, Page {page}"
        if section:
            header += f", Section: {section}"
        header += "]"
        parts.append(f"{header}\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


def _extract_sources(chunks: List[Dict[str, Any]]) -> List[Dict]:
    """Deduplicate and return source list."""
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk.get("source", ""), chunk.get("page", 1))
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": chunk.get("source", ""),
                "page": chunk.get("page", 1),
                "section": chunk.get("section", "")
            })
    return sources
