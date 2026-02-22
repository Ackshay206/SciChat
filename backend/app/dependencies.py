"""
Shared dependency singletons for FastAPI.

Manages lifecycle of:
- LLM instances
- Embedding model
- Pinecone index
- Query engines
- Cached nodes for BM25
"""

import logging
from typing import Any, Dict, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine

from app.config import config
from app.services.llm_service import (
    init_llm,
    init_embed_model,
    init_judge_llm,
    configure_settings,
)

logger = logging.getLogger(__name__)


class AppState:
    """Application state holding all shared singletons."""

    def __init__(self):
        self.llm = None
        self.judge_llm = None
        self.embed_model = None
        self.index: Optional[VectorStoreIndex] = None
        self.nodes: list = []  # Stored for BM25 retriever
        self.query_engine: Optional[RetrieverQueryEngine] = None
        self.documents_metadata: Dict[str, Any] = {}  # doc_id -> metadata
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all ML models and services."""
        if self._initialized:
            return

        logger.info("🚀 Initializing application state...")

        # Validate config
        config.validate()

        # Initialize models
        self.llm = init_llm()
        self.judge_llm = init_judge_llm()
        self.embed_model = init_embed_model()

        # Configure global LlamaIndex settings
        configure_settings(self.llm, self.embed_model)

        self._initialized = True
        logger.info("✅ Application state initialized")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def has_index(self) -> bool:
        return self.index is not None


# Singleton
app_state = AppState()
