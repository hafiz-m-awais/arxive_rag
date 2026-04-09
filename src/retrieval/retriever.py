"""
retrieval/retriever.py — Hybrid retriever with HyDE (Hypothetical Document Embeddings).

Two techniques combined:
1. Hybrid search: dense (cosine) + sparse (BM25) → improves recall 5-15%
2. HyDE: generate hypothetical answer → embed it → use for retrieval → improves recall 10-20%

Pipeline: Query → [HyDE] → Embed → Hybrid Search → Top-K candidates
"""
from typing import List, Dict, Any, Tuple
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    NamedVector, NamedSparseVector, SparseVector,
    SearchRequest, FusionQuery, Prefetch, Fusion
)
from src.config import config
from src.ingestion.embedder import embed_query, get_embedder
from src.ingestion.vector_store import get_client, DENSE_NAME, SPARSE_NAME


def retrieve(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """
    Retrieve top-k relevant chunks for a query.
    Uses HyDE if enabled in config, then hybrid search.

    Returns list of result dicts with content, source, score.
    """
    top_k = top_k or config.RETRIEVAL_TOP_K

    # Step 1: Optionally expand query with HyDE
    search_query = query
    if config.USE_HYDE:
        search_query = _generate_hypothetical_doc(query)
        logger.debug(f"HyDE expansion generated ({len(search_query)} chars)")

    # Step 2: Embed the (possibly expanded) query
    dense_vec, sparse_vec = embed_query(search_query)

    # Step 3: Hybrid search in Qdrant
    results = _hybrid_search(dense_vec, sparse_vec, top_k)

    logger.info(f"Retrieved {len(results)} candidates for query: '{query[:60]}...'")
    return results


def _generate_hypothetical_doc(query: str) -> str:
    """
    HyDE: Ask the LLM to generate a hypothetical answer,
    then use THAT as the search string instead of the raw query.
    The hypothetical answer is closer in embedding space to real documents.
    """
    try:
        import ollama
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    f"Write a short, factual passage (2-3 sentences) that would "
                    f"directly answer this question:\n\n{query}\n\n"
                    f"Write only the passage, no preamble."
                )
            }],
            options={"temperature": 0.0, "num_predict": 150}
        )
        return response["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"HyDE generation failed, using original query: {e}")
        return query


def _hybrid_search(
    dense_vec: List[float],
    sparse_vec: Dict[int, float],
    top_k: int
) -> List[Dict[str, Any]]:
    """
    Run hybrid search using Qdrant's built-in fusion.
    Combines dense (semantic) + sparse (lexical) retrieval.
    """
    client = get_client()

    try:
        results = client.query_points(
            collection_name=config.QDRANT_COLLECTION,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using=DENSE_NAME,
                    limit=top_k
                ),
                Prefetch(
                    query=SparseVector(
                        indices=list(sparse_vec.keys()),
                        values=list(sparse_vec.values())
                    ),
                    using=SPARSE_NAME,
                    limit=top_k
                )
            ],
            query=FusionQuery(fusion=Fusion.RRF),  # Reciprocal Rank Fusion
            limit=top_k
        )

        return [
            {
                "content": hit.payload["content"],
                "source": hit.payload["source"],
                "page": hit.payload.get("page", 1),
                "section": hit.payload.get("section", ""),
                "score": hit.score,
                "id": str(hit.id)
            }
            for hit in results.points
        ]

    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        # Fallback to dense-only search
        logger.warning("Falling back to dense-only search")
        return _dense_only_search(dense_vec, top_k)


def _dense_only_search(dense_vec: List[float], top_k: int) -> List[Dict[str, Any]]:
    """Fallback dense-only search."""
    client = get_client()
    results = client.search(
        collection_name=config.QDRANT_COLLECTION,
        query_vector=NamedVector(name=DENSE_NAME, vector=dense_vec),
        limit=top_k
    )
    return [
        {
            "content": hit.payload["content"],
            "source": hit.payload["source"],
            "page": hit.payload.get("page", 1),
            "section": hit.payload.get("section", ""),
            "score": hit.score,
            "id": str(hit.id)
        }
        for hit in results
    ]
