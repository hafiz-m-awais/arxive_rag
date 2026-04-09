"""
ingestion/loader.py — Document loader.
Supports PDF (pdfplumber), TXT, and DOCX files.
Returns a list of raw Document dicts with content + metadata.
"""
import os
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
import pdfplumber
from docx import Document as DocxDocument


def load_documents(source_dir: str) -> List[Dict[str, Any]]:
    """
    Load all supported documents from source_dir.
    Returns list of: {"content": str, "source": str, "page": int}
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    documents = []
    supported = {".pdf", ".txt", ".md", ".docx"}

    for file_path in sorted(source_path.rglob("*")):
        if file_path.suffix.lower() not in supported:
            continue
        try:
            docs = _load_file(file_path)
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def _load_file(file_path: Path) -> List[Dict[str, Any]]:
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(file_path)
    elif ext in (".txt", ".md"):
        return _load_text(file_path)
    elif ext == ".docx":
        return _load_docx(file_path)
    return []


def _load_pdf(file_path: Path) -> List[Dict[str, Any]]:
    docs = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append({
                    "content": text.strip(),
                    "source": str(file_path.name),
                    "page": i + 1,
                    "file_type": "pdf"
                })
    return docs


def _load_text(file_path: Path) -> List[Dict[str, Any]]:
    text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    return [{
        "content": text,
        "source": str(file_path.name),
        "page": 1,
        "file_type": file_path.suffix.lstrip(".")
    }]


def _load_docx(file_path: Path) -> List[Dict[str, Any]]:
    doc = DocxDocument(file_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    if not text:
        return []
    return [{
        "content": text,
        "source": str(file_path.name),
        "page": 1,
        "file_type": "docx"
    }]
