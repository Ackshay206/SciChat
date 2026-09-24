"""
LLM and embedding model service — initialization from notebook Cell 5.

Provides:
- Gemini LLM (primary generation)
- OpenAI GPT-4o-mini (evaluation judge)
- BGE-M3 embedding model
"""

import logging
from typing import Optional

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI

from app.config import config

logger = logging.getLogger(__name__)


def init_llm() -> Gemini:
    """
    Initialize the primary LLM (Gemini).
    From notebook Cell 5 — unchanged.
    """
    llm = Gemini(
        model=config.LLM_MODEL,
        api_key=config.GOOGLE_API_KEY,
    )
    logger.info(f"✅ LLM initialized: {config.LLM_MODEL}")
    return llm


def init_judge_llm() -> Optional[OpenAI]:
    """
    Initialize the judge LLM for evaluation (GPT-4o-mini).
    Returns None when OPENAI_API_KEY is not set (evaluation is then unavailable).
    """
    if not config.OPENAI_API_KEY:
        logger.info("ℹ️ OPENAI_API_KEY not set — judge LLM disabled")
        return None
    judge_llm = OpenAI(
        model=config.JUDGE_LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
    )
    logger.info(f"✅ Judge LLM initialized: {config.JUDGE_LLM_MODEL}")
    return judge_llm


def init_embed_model() -> HuggingFaceEmbedding:
    """
    Initialize the embedding model (BAAI/bge-m3).
    From notebook Cell 5 — unchanged.

    Note: First load downloads ~2GB model weights.
    """
    embed_model = HuggingFaceEmbedding(
        model_name=config.EMBEDDING_MODEL,
        trust_remote_code=True,
        device="cpu",  # Explicit CPU — avoids MPS memory fragmentation/pickle errors
    )
    logger.info(f"✅ Embedding model initialized: {config.EMBEDDING_MODEL}")
    return embed_model


def configure_settings(llm, embed_model) -> None:
    """Set global LlamaIndex settings."""
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP
    logger.info("✅ LlamaIndex global settings configured")
