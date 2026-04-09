"""
tests/test_ingestion.py — Unit tests for ingestion pipeline.
Run with: pytest tests/test_ingestion.py -v
"""
import os
import sys
import pytest
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.chunker import chunk_documents, _split_into_sections, _split_by_words
from src.ingestion.loader import _load_text


class TestChunker:

    def test_split_into_sections_with_headers(self):
        text = "# Introduction\nThis is intro.\n\n# Methods\nThis is methods."
        sections = _split_into_sections(text)
        assert len(sections) == 2
        assert sections[0][0] == "Introduction"
        assert sections[1][0] == "Methods"

    def test_split_into_sections_no_headers(self):
        text = "Just plain text with no headers at all."
        sections = _split_into_sections(text)
        assert len(sections) == 1
        assert sections[0][0] == ""

    def test_split_by_words_small_text(self):
        text = "Hello world this is a short text."
        chunks = _split_by_words(text, chunk_size=300, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_by_words_large_text(self):
        words = ["word"] * 700
        text = " ".join(words)
        chunks = _split_by_words(text, chunk_size=300, overlap=50)
        assert len(chunks) > 1

    def test_chunk_documents_adds_header_context(self):
        docs = [{
            "content": "# My Section\nThis is section content here.",
            "source": "test.txt",
            "page": 1,
            "file_type": "txt"
        }]
        chunks = chunk_documents(docs)
        assert len(chunks) > 0
        assert "My Section" in chunks[0]["content"]

    def test_chunk_documents_preserves_source(self):
        docs = [{
            "content": "Simple document content.",
            "source": "myfile.pdf",
            "page": 3,
            "file_type": "pdf"
        }]
        chunks = chunk_documents(docs)
        assert all(c["source"] == "myfile.pdf" for c in chunks)

    def test_chunk_documents_empty_input(self):
        chunks = chunk_documents([])
        assert chunks == []


class TestLoader:

    def test_load_text_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, this is test content.\nSecond line here.")
            f.flush()
            docs = _load_text(type("Path", (), {
                "read_text": lambda self, **kw: "Hello, this is test content.\nSecond line here.",
                "name": "test.txt",
                "suffix": ".txt"
            })())
        assert len(docs) == 1
        assert "Hello" in docs[0]["content"]

    def test_load_text_empty_file(self):
        result = type("Path", (), {
            "read_text": lambda self, **kw: "   ",
            "name": "empty.txt",
            "suffix": ".txt"
        })()
        from src.ingestion.loader import _load_text
        docs = _load_text(result)
        assert docs == []
