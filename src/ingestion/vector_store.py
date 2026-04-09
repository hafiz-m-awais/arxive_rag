"""
ingestion/vector_store.py — Qdrant vector store manager.
Runs in local mode (no Docker required).
Supports hybrid indexing: dense (cosine) + sparse (BM25-style).
"""
import uuid
from typing import List, Dict, Any
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams,
    PointStruct, SparseVector, NamedVector,
    NamedSparseVector, SearchRequest
)
from src.config import config

_client = None
DENSE_VECTOR_SIZE = 1024   # BGE-M3 output dimension
DENSE_NAME = "dense"
SPARSE_NAME = "sparse"


def get_client() -> QdrantClient:
    """Lazy-load Qdrant client (local file-based mode)."""
    global _client
    if _client is None:
        logger.info(f"Connecting to Qdrant at: {config.QDRANT_PATH}")
        _client = QdrantClient(path=config.QDRANT_PATH)
    return _client


def ensure_collection():
    """Create collection if it doesn't exist."""
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]

    if config.QDRANT_COLLECTION not in collections:
        logger.info(f"Creating collection: {config.QDRANT_COLLECTION}")
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config={
                DENSE_NAME: VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_NAME: SparseVectorParams()
            }
        )
        logger.info("Collection created")
    else:
        logger.info(f"Collection '{config.QDRANT_COLLECTION}' already exists")


def upsert_chunks(chunks: List[Dict[str, Any]]):
    """
    Upsert embedded chunks into Qdrant.
    Each chunk must have 'dense_vector' and 'sparse_vector' keys.
    """
    client = get_client()
    ensure_collection()

    points = []
    for chunk in chunks:
        point_id = str(uuid.uuid4())
        sparse = chunk["sparse_vector"]

        points.append(PointStruct(
            id=point_id,
            vector={
                DENSE_NAME: chunk["dense_vector"],
                SPARSE_NAME: SparseVector(
                    indices=list(sparse.keys()),
                    values=list(sparse.values())
                )
            },
            payload={
                "content": chunk["content"],
                "source": chunk["source"],
                "page": chunk.get("page", 1),
                "section": chunk.get("section", ""),
                "chunk_index": chunk.get("chunk_index", 0)
            }
        ))

    # Upsert in batches
    batch_size = 64
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=config.QDRANT_COLLECTION,
            points=batch
        )
        logger.debug(f"Upserted batch {i // batch_size + 1} ({len(batch)} points)")

    logger.info(f"Upserted {len(points)} chunks into Qdrant")


def get_collection_info() -> Dict:
    """Return collection stats."""
    client = get_client()
    info = client.get_collection(config.QDRANT_COLLECTION)
    return {
        "name": config.QDRANT_COLLECTION,
        "points_count": info.points_count,
        "status": str(info.status)
    }


def delete_collection():
    """Drop and recreate the collection (for re-indexing)."""
    client = get_client()
    client.delete_collection(config.QDRANT_COLLECTION)
    logger.info(f"Deleted collection: {config.QDRANT_COLLECTION}")
