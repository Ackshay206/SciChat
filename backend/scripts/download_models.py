"""
Download all runtime models at Docker build time, so a cold start doesn't fetch them.

Usage:
    cd backend && python -m scripts.download_models
"""

import fitz
from easyocr import Reader
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import config
from app.core.document_parser import _docling_converter

SentenceTransformer(config.EMBEDDING_MODEL)
CrossEncoder(config.RERANKER_MODEL)
Reader(["en"], gpu=False, verbose=False)

warmup = fitz.open()
warmup.new_page().insert_text((72, 72), "warm-up")
warmup.save("/tmp/warmup.pdf")
_docling_converter().convert("/tmp/warmup.pdf")

print("✅ Models downloaded")
