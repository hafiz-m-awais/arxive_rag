"""
api/app.py — Flask REST API for the RAG system.
Endpoints: /api/ingest, /api/query, /api/query/stream, /api/health, /api/evaluate
"""
import os
import json
from loguru import logger
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from src.config import config
from src.ingestion.pipeline import run_ingestion
from src.retrieval.retriever import retrieve
from src.retrieval.reranker import rerank
from src.generation.generator import generate_response, generate_stream
from src.ingestion.vector_store import get_collection_info

app = Flask(__name__)
CORS(app)

# Setup logging
os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
logger.add(config.LOG_FILE, level=config.LOG_LEVEL, rotation="10 MB")


# ─────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    """System health check — verifies Qdrant and Ollama are reachable."""
    status = {"status": "ok", "components": {}}

    # Check Qdrant
    try:
        info = get_collection_info()
        status["components"]["qdrant"] = {"status": "ok", "points": info["points_count"]}
    except Exception as e:
        status["components"]["qdrant"] = {"status": "error", "error": str(e)}
        status["status"] = "degraded"

    # Check Ollama
    try:
        import ollama
        ollama.list()
        status["components"]["ollama"] = {"status": "ok", "model": config.OLLAMA_MODEL}
    except Exception as e:
        status["components"]["ollama"] = {"status": "error", "error": str(e)}
        status["status"] = "degraded"

    return jsonify(status), 200 if status["status"] == "ok" else 503


# ─────────────────────────────────────────
# Ingest Documents
# ─────────────────────────────────────────
@app.route("/api/ingest", methods=["POST"])
def ingest():
    """
    Ingest documents from data/raw/ directory.
    Optional body: {"source_dir": "path/to/docs"}
    """
    body = request.get_json(silent=True) or {}
    source_dir = body.get("source_dir", "./data/raw")

    logger.info(f"Ingest request from {request.remote_addr}, dir={source_dir}")

    result = run_ingestion(source_dir)
    code = 200 if result["status"] == "success" else 400
    return jsonify(result), code


# ─────────────────────────────────────────
# Query (Standard)
# ─────────────────────────────────────────
@app.route("/api/query", methods=["POST"])
def query():
    """
    Answer a question using the full RAG pipeline.
    Body: {"question": "your question", "top_k": 20, "top_n": 5}
    """
    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()

    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400

    top_k = int(body.get("top_k", config.RETRIEVAL_TOP_K))
    top_n = int(body.get("top_n", config.RERANK_TOP_N))

    logger.info(f"Query: '{question[:80]}'")

    # Retrieve → Rerank → Generate
    candidates = retrieve(question, top_k=top_k)
    reranked = rerank(question, candidates, top_n=top_n)
    result = generate_response(question, reranked)

    result["query"] = question
    result["candidates_retrieved"] = len(candidates)
    result["chunks_used"] = len(reranked)

    return jsonify(result), 200


# ─────────────────────────────────────────
# Query (Streaming)
# ─────────────────────────────────────────
@app.route("/api/query/stream", methods=["POST"])
def query_stream():
    """
    Streaming version of /api/query.
    Returns tokens as Server-Sent Events (SSE).
    """
    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()

    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400

    top_k = int(body.get("top_k", config.RETRIEVAL_TOP_K))
    top_n = int(body.get("top_n", config.RERANK_TOP_N))

    candidates = retrieve(question, top_k=top_k)
    reranked = rerank(question, candidates, top_n=top_n)

    def event_stream():
        for token in generate_stream(question, reranked):
            yield f"data: {json.dumps({'token': token})}\n\n"
        sources = [
            {"file": c.get("source", ""), "page": c.get("page", 1)}
            for c in reranked
        ]
        yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ─────────────────────────────────────────
# Collection Info
# ─────────────────────────────────────────
@app.route("/api/collection", methods=["GET"])
def collection_info():
    """Return Qdrant collection statistics."""
    info = get_collection_info()
    return jsonify(info), 200


# ─────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"Starting RAG API on {config.FLASK_HOST}:{config.FLASK_PORT}")
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )
