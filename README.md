# Production RAG System

A production-ready Retrieval-Augmented Generation (RAG) pipeline built entirely offline.
Runs on CPU-only hardware (tested on Dell Latitude 7310, 16GB RAM).

## Stack
- **Embedding**: BAAI/bge-m3 (dense + sparse hybrid)
- **Vector DB**: Qdrant (local mode)
- **Reranker**: BAAI/bge-reranker-v2-m3
- **LLM**: Qwen2.5:7B via Ollama
- **API**: Flask + Gunicorn
- **Evaluation**: RAGAS

## Quickstart
```bash
git clone <your-repo-url>
cd production_rag
pip install -r requirements.txt
ollama pull qwen2.5:7b
python scripts/setup_db.py
python src/api/app.py
```

## Project Structure
```
production_rag/
├── src/
│   ├── ingestion/        # Document loading, chunking, embedding
│   ├── retrieval/        # Hybrid search, HyDE, reranking
│   ├── generation/       # LLM response generation
│   ├── api/              # Flask REST API
│   └── evaluation/       # RAGAS evaluation pipeline
├── data/
│   ├── raw/              # Drop your PDFs/TXTs here
│   └── processed/        # Chunked output (auto-generated)
├── tests/                # Unit + integration tests
├── scripts/              # Setup and utility scripts
└── docs/                 # Documentation
```

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/ingest | Ingest documents from data/raw/ |
| POST | /api/query | Ask a question |
| GET  | /api/health | Health check |
| POST | /api/evaluate | Run RAGAS evaluation |

## Environment
Copy `.env.example` to `.env` and adjust settings.
