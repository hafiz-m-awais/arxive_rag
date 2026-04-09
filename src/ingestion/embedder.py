"""
ingestion/embedder.py — BGE-M3 Embedder.
Uses FlagEmbedding to generate both dense and sparse vectors from a single model.
BGE-M3: 568M params, MIT license, 100+ languages, 8192-token context.
Supports dense (cosine similarity) + sparse (BM25-style lexical) retrieval.
"""
from typing import List, Dict, Any, Tuple
from loguru import logger
from FlagEmbedding import BGEM3FlagModel
from src.config import config

_model = None


def get_embedder() -> BGEM3FlagModel:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _model = BGEM3FlagModel(
            config.EMBEDDING_MODEL,
            use_fp16=False,       # CPU mode — set True if you have GPU
            device="cpu"
        )
        logger.info("Embedding model loaded successfully")
    return _model


def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add dense and sparse embeddings to each chunk dict.
    Returns chunks enriched with 'dense_vector' and 'sparse_vector' keys.
    """
    model = get_embedder()
    texts = [c["content"] for c in chunks]

    logger.info(f"Embedding {len(texts)} chunks in batches of {config.EMBEDDING_BATCH_SIZE}...")

    all_dense = []
    all_sparse = []

    for i in range(0, len(texts), config.EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + config.EMBEDDING_BATCH_SIZE]
        output = model.encode(
            batch,
            batch_size=len(batch),
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )
        all_dense.extend(output["dense_vecs"].tolist())
        # Convert sparse dict to Qdrant-compatible format
        for sparse_weights in output["lexical_weights"]:
            all_sparse.append(_convert_sparse(sparse_weights))

        logger.debug(f"Embedded batch {i // config.EMBEDDING_BATCH_SIZE + 1}")

    for chunk, dense, sparse in zip(chunks, all_dense, all_sparse):
        chunk["dense_vector"] = dense
        chunk["sparse_vector"] = sparse

    logger.info("Embedding complete")
    return chunks


def embed_query(query: str) -> Tuple[List[float], Dict[int, float]]:
    """
    Embed a single query string.
    Returns (dense_vector, sparse_vector) tuple.
    """
    model = get_embedder()
    output = model.encode(
        [query],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False
    )
    dense = output["dense_vecs"][0].tolist()
    sparse = _convert_sparse(output["lexical_weights"][0])
    return dense, sparse


def _convert_sparse(lexical_weights: Dict) -> Dict[int, float]:
    """Convert BGE-M3 sparse weights to {token_id: weight} dict."""
    return {int(k): float(v) for k, v in lexical_weights.items()}
