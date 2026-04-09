"""
ingestion/pipeline.py — Full ingestion pipeline orchestrator.
Runs: Load → Chunk → Embed → Store
"""
from loguru import logger
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_chunks
from src.ingestion.vector_store import upsert_chunks, ensure_collection, get_collection_info


def run_ingestion(source_dir: str = "./data/raw") -> Dict:
    """
    Full ingestion pipeline.
    Args:
        source_dir: Directory containing raw documents.
    Returns:
        Summary dict with counts and status.
    """
    logger.info(f"Starting ingestion from: {source_dir}")

    # Step 1: Load
    documents = load_documents(source_dir)
    if not documents:
        return {"status": "error", "message": "No documents found in source directory"}

    # Step 2: Chunk
    chunks = chunk_documents(documents)

    # Step 3: Embed
    chunks = embed_chunks(chunks)

    # Step 4: Store
    ensure_collection()
    upsert_chunks(chunks)

    info = get_collection_info()
    summary = {
        "status": "success",
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "total_indexed": info["points_count"]
    }
    logger.info(f"Ingestion complete: {summary}")
    return summary


# Allow running standalone
if __name__ == "__main__":
    from typing import Dict
    result = run_ingestion()
    print(result)
