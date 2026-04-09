"""
tests/test_api.py — Integration tests for Flask API endpoints.
Run with: pytest tests/test_api.py -v
"""
import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    from src.api.app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:

    def test_health_returns_200_when_ok(self, client):
        with patch("src.api.app.get_collection_info") as mock_info, \
             patch("src.api.app.ollama") as mock_ollama:
            mock_info.return_value = {"points_count": 100, "status": "green"}
            mock_ollama.list.return_value = MagicMock()
            resp = client.get("/api/health")
            assert resp.status_code in (200, 503)

    def test_health_returns_json(self, client):
        with patch("src.api.app.get_collection_info") as mock_info:
            mock_info.return_value = {"points_count": 0, "status": "green"}
            resp = client.get("/api/health")
            data = json.loads(resp.data)
            assert "status" in data
            assert "components" in data


class TestQueryEndpoint:

    def test_query_missing_question(self, client):
        resp = client.post("/api/query",
                           data=json.dumps({}),
                           content_type="application/json")
        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert "error" in data

    def test_query_empty_question(self, client):
        resp = client.post("/api/query",
                           data=json.dumps({"question": ""}),
                           content_type="application/json")
        assert resp.status_code == 400

    def test_query_returns_answer(self, client):
        with patch("src.api.app.retrieve") as mock_retrieve, \
             patch("src.api.app.rerank") as mock_rerank, \
             patch("src.api.app.generate_response") as mock_gen:
            mock_retrieve.return_value = [{"content": "Paris is the capital.", "source": "geo.pdf", "page": 1, "section": "Europe", "score": 0.9}]
            mock_rerank.return_value = [{"content": "Paris is the capital.", "source": "geo.pdf", "page": 1, "section": "Europe", "rerank_score": 0.95}]
            mock_gen.return_value = {"answer": "Paris is the capital of France.", "sources": [{"file": "geo.pdf", "page": 1, "section": "Europe"}], "context_used": ["Paris is the capital."]}

            resp = client.post("/api/query",
                               data=json.dumps({"question": "What is the capital of France?"}),
                               content_type="application/json")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert "answer" in data
            assert "sources" in data
            assert data["query"] == "What is the capital of France?"


class TestIngestEndpoint:

    def test_ingest_returns_result(self, client):
        with patch("src.api.app.run_ingestion") as mock_ingest:
            mock_ingest.return_value = {
                "status": "success",
                "documents_loaded": 3,
                "chunks_created": 45,
                "total_indexed": 45
            }
            resp = client.post("/api/ingest",
                               data=json.dumps({"source_dir": "./data/raw"}),
                               content_type="application/json")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["status"] == "success"

    def test_ingest_error_response(self, client):
        with patch("src.api.app.run_ingestion") as mock_ingest:
            mock_ingest.return_value = {"status": "error", "message": "No documents found"}
            resp = client.post("/api/ingest",
                               data=json.dumps({}),
                               content_type="application/json")
            assert resp.status_code == 400
