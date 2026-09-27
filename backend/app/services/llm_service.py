"""
LLM and embedding model service — initialization from notebook Cell 5.

Provides:
- Gemini LLM (primary generation)
- OpenAI GPT-4o-mini (evaluation judge)
- BGE-M3 embedding model
"""

import logging
import os
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


def configure_torch_threads() -> None:
    """
    Match PyTorch's thread count to the CPUs the container is actually allowed to use.
    In containers PyTorch sizes its thread pool from the visible host CPUs and ignores
    OMP_NUM_THREADS, so on Cloud Run it would use 3 threads whatever vCPU count is allocated.
    Set TORCH_NUM_THREADS to the vCPU count on Cloud Run: during startup CPU boost the cgroup
    quota is temporarily inflated, so reading it at startup overcounts.
    """
    import torch

    if os.getenv("TORCH_NUM_THREADS"):
        torch.set_num_threads(int(os.environ["TORCH_NUM_THREADS"]))
        logger.info(f"✅ Torch threads: {torch.get_num_threads()} (TORCH_NUM_THREADS)")
        return

    quota = period = None
    try:  # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
    except FileNotFoundError:
        try:  # cgroup v1 (Cloud Run)
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
                quota = f.read().strip()
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                period = f.read().strip()
        except FileNotFoundError:
            pass
    if quota and quota not in ("max", "-1"):
        torch.set_num_threads(max(1, int(quota) // int(period)))
    logger.info(f"✅ Torch threads: {torch.get_num_threads()} (cpu quota: {quota}/{period})")


def configure_settings(llm, embed_model) -> None:
    """Set global LlamaIndex settings."""
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP
    logger.info("✅ LlamaIndex global settings configured")
