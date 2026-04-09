"""
tests/test_retrieval.py — Unit tests for retrieval components.
Run with: pytest tests/test_retrieval.py -v
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestReranker:

    def _make_candidates(self, n=5):
        return [
            {"content": f"Candidate chunk number {i} with some relevant text.", "source": f"doc{i}.pdf", "page": i, "score": 0.9 - i * 0.05}
            for i in range(n)
        ]

    def test_rerank_returns_top_n(self):
        from src.retrieval.reranker import rerank
        candidates = self._make_candidates(10)
        mock_scores = [0.9 - i * 0.05 for i in range(10)]

        with patch("src.retrieval.reranker.get_reranker") as mock_get:
            mock_reranker = MagicMock()
            mock_reranker.compute_score.return_value = mock_scores
            mock_get.return_value = mock_reranker

            results = rerank("test query", candidates, top_n=3)
            assert len(results) == 3

    def test_rerank_empty_candidates(self):
        from src.retrieval.reranker import rerank
        results = rerank("test query", [], top_n=5)
        assert results == []

    def test_rerank_fewer_than_top_n(self):
        from src.retrieval.reranker import rerank
        candidates = self._make_candidates(2)
        results = rerank("test query", candidates, top_n=5)
        assert len(results) == 2

    def test_rerank_sorted_by_score(self):
        from src.retrieval.reranker import rerank
        candidates = self._make_candidates(5)
        scores = [0.3, 0.9, 0.5, 0.8, 0.1]

        with patch("src.retrieval.reranker.get_reranker") as mock_get:
            mock_reranker = MagicMock()
            mock_reranker.compute_score.return_value = scores
            mock_get.return_value = mock_reranker

            results = rerank("test query", candidates, top_n=5)
            result_scores = [r["rerank_score"] for r in results]
            assert result_scores == sorted(result_scores, reverse=True)

    def test_rerank_fallback_on_error(self):
        from src.retrieval.reranker import rerank
        candidates = self._make_candidates(5)

        with patch("src.retrieval.reranker.get_reranker") as mock_get:
            mock_reranker = MagicMock()
            mock_reranker.compute_score.side_effect = RuntimeError("model error")
            mock_get.return_value = mock_reranker

            results = rerank("test query", candidates, top_n=3)
            assert len(results) == 3  # fallback returns candidates[:top_n]


class TestGenerator:

    def test_generate_response_no_context(self):
        from src.generation.generator import generate_response
        result = generate_response("What is AI?", [])
        assert "No relevant documents" in result["answer"]
        assert result["sources"] == []

    def test_generate_response_returns_sources(self):
        from src.generation.generator import generate_response
        chunks = [
            {"content": "AI stands for artificial intelligence.", "source": "ai.pdf", "page": 1, "section": "Intro"}
        ]
        with patch("src.generation.generator.ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "AI stands for artificial intelligence. [Source: ai.pdf, Page 1]"}}
            result = generate_response("What is AI?", chunks)
            assert len(result["sources"]) == 1
            assert result["sources"][0]["file"] == "ai.pdf"

    def test_generate_response_handles_ollama_error(self):
        from src.generation.generator import generate_response
        chunks = [{"content": "Some content.", "source": "test.pdf", "page": 1, "section": ""}]
        with patch("src.generation.generator.ollama.chat") as mock_chat:
            mock_chat.side_effect = ConnectionError("Ollama not running")
            result = generate_response("test?", chunks)
            assert "Error" in result["answer"]
