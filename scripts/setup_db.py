"""
scripts/setup_db.py — One-time database initialisation.
Run this before first use to create the Qdrant collection and verify Ollama.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.ingestion.vector_store import ensure_collection, get_collection_info
from src.config import config


def setup():
    logger.info("=== RAG System Setup ===")

    # 1. Create data directories
    os.makedirs("./data/raw", exist_ok=True)
    os.makedirs("./data/processed", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    logger.info("Directories created")

    # 2. Copy .env.example if .env missing
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            import shutil
            shutil.copy(".env.example", ".env")
            logger.info("Created .env from .env.example")

    # 3. Init Qdrant collection
    ensure_collection()
    info = get_collection_info()
    logger.info(f"Qdrant collection ready: {info}")

    # 4. Verify Ollama
    try:
        import ollama
        models = [m.model for m in ollama.list().models]
        if config.OLLAMA_MODEL in models or any(config.OLLAMA_MODEL in m for m in models):
            logger.info(f"Ollama model '{config.OLLAMA_MODEL}' is available")
        else:
            logger.warning(
                f"Model '{config.OLLAMA_MODEL}' not found. "
                f"Run: ollama pull {config.OLLAMA_MODEL}"
            )
            logger.info(f"Available models: {models}")
    except Exception as e:
        logger.error(f"Ollama not reachable: {e}. Is Ollama running? Run: ollama serve")

    logger.info("=== Setup complete ===")
    logger.info("Next step: drop PDF/TXT files into data/raw/ then run:")
    logger.info("  python -c \"from src.ingestion.pipeline import run_ingestion; run_ingestion()\"")
    logger.info("Or start the API:")
    logger.info("  python src/api/app.py")


if __name__ == "__main__":
    setup()
