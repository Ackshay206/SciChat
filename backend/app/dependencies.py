"""
Shared dependency singletons for FastAPI.

Manages lifecycle of:
- LLM instances
- Embedding model
- Pinecone index
- Query engines
- Document metadata persistence
"""

import json
import logging
import os
from pathlib import Path
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

# Path for persisted document metadata
_METADATA_DIR = Path("data")
_METADATA_PATH = _METADATA_DIR / "documents_metadata.json"


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
        self.active_doc_id: Optional[str] = None  # Currently loaded document
        self.engines: Dict[str, tuple] = {}  # doc_id -> (index, nodes, query_engine)
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

        # Load persisted document metadata
        self.load_metadata()

        # Preload first available document from Pinecone
        if self.documents_metadata:
            first_doc_id = next(iter(self.documents_metadata))
            try:
                self.switch_document(first_doc_id)
                logger.info(f"📄 Preloaded document: {self.documents_metadata[first_doc_id].get('title', 'Unknown')}")
            except Exception as e:
                logger.warning(f"⚠️ Could not preload document {first_doc_id}: {e}")

        self._initialized = True
        logger.info("✅ Application state initialized")

    def switch_document(self, doc_id: str) -> None:
        """
        Switch the active document by loading its Pinecone namespace.
        Rebuilds index and query engine for the selected document.
        """
        if self.active_doc_id == doc_id and self.query_engine is not None:
            logger.info(f"📄 Document {doc_id} already active, skipping switch")
            return

        if doc_id in self.engines:
            self.index, self.nodes, self.query_engine = self.engines[doc_id]
            self.active_doc_id = doc_id
            return

        meta = self.documents_metadata.get(doc_id)
        if not meta:
            raise ValueError(f"Document '{doc_id}' not found in metadata")

        namespace = meta.get("namespace", f"doc_{doc_id}")
        logger.info(f"🔄 Switching to document: {meta.get('title', 'Unknown')} (namespace: {namespace})")

        from app.core.indexing import load_existing_index, load_namespace_nodes
        from app.core.retrieval import create_query_engine

        # Load index from existing Pinecone namespace (no re-embedding)
        self.index = load_existing_index(
            embed_model=self.embed_model,
            namespace=namespace,
        )

        # Rebuild text nodes from Pinecone metadata so BM25 (hybrid) works for loaded documents
        self.nodes = load_namespace_nodes(namespace)
        self.query_engine = create_query_engine(
            index=self.index,
            nodes=self.nodes,
            llm=self.llm,
            retriever_type="hybrid" if self.nodes else "vector",
        )

        self.engines[doc_id] = (self.index, self.nodes, self.query_engine)
        self.active_doc_id = doc_id
        logger.info(f"✅ Switched to document: {meta.get('title', 'Unknown')}")

    def save_metadata(self) -> None:
        """Persist document metadata to JSON file."""
        try:
            _METADATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(_METADATA_PATH, "w") as f:
                json.dump(self.documents_metadata, f, indent=2, default=str)
            logger.info(f"💾 Metadata saved ({len(self.documents_metadata)} documents)")
        except Exception as e:
            logger.error(f"❌ Failed to save metadata: {e}")

    def load_metadata(self) -> None:
        """Load document metadata from JSON file."""
        if not _METADATA_PATH.exists():
            logger.info("📂 No persisted metadata found, starting fresh")
            return

        try:
            with open(_METADATA_PATH, "r") as f:
                self.documents_metadata = json.load(f)
            logger.info(f"📂 Loaded metadata for {len(self.documents_metadata)} documents")
        except Exception as e:
            logger.error(f"❌ Failed to load metadata: {e}")
            self.documents_metadata = {}

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def has_index(self) -> bool:
        return self.index is not None


# Singleton
app_state = AppState()
