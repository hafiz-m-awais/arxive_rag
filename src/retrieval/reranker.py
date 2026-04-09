"""
retrieval/reranker.py — BGE Cross-Encoder Reranker.

BGE-reranker-v2-m3: 278M params, Apache 2.0, multilingual, CPU-friendly.
Takes top-K candidates from retriever and re-scores each (query, chunk) pair
jointly using a cross-encoder — far more precise than bi-encoder similarity alone.

Research shows reranking improves retrieval quality by up to 48% (Databricks, 2025).
"""
from typing import List, Dict, Any
from loguru import logger
from FlagEmbedding import FlagReranker
from src.config import config

_reranker = None


def get_reranker() -> FlagReranker:
    """Lazy-load the reranker model (singleton)."""
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker model: {config.RERANKER_MODEL}")
        _reranker = FlagReranker(
            config.RERANKER_MODEL,
            use_fp16=False   # CPU mode — set True for GPU
        )
        logger.info("Reranker loaded successfully")
    return _reranker


def rerank(query: str, candidates: List[Dict[str, Any]], top_n: int = None) -> List[Dict[str, Any]]:
    """
    Re-rank candidates using cross-encoder scoring.

    Args:
        query: The user's original question.
        candidates: List of retrieved chunks (must have 'content' key).
        top_n: How many to return after reranking.

    Returns:
        Top-N candidates sorted by reranker score (descending).
    """
    top_n = top_n or config.RERANK_TOP_N

    if not candidates:
        return []

    if len(candidates) <= top_n:
        logger.debug("Fewer candidates than top_n — skipping reranker")
        return candidates

    reranker = get_reranker()

    # Build (query, passage) pairs for cross-encoder
    pairs = [[query, c["content"]] for c in candidates]

    try:
        scores = reranker.compute_score(pairs, normalize=True)
        # Handle single result (returns float, not list)
        if isinstance(scores, float):
            scores = [scores]
    except Exception as e:
        logger.error(f"Reranker failed: {e}. Returning original order.")
        return candidates[:top_n]

    # Attach scores and sort
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    top = reranked[:top_n]

    logger.info(
        f"Reranked {len(candidates)} → {len(top)} | "
        f"top score: {top[0]['rerank_score']:.3f}"
    )
    return top
