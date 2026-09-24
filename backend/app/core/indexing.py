"""
Pinecone vector store indexing module.

Replaces the notebook's in-memory VectorStoreIndex with Pinecone-backed storage.
Uses Pinecone serverless on GCP (us-central1) with async support via SDK v6.0.0+.
"""

import logging
from typing import List, Optional

from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from llama_index.vector_stores.pinecone import PineconeVectorStore

from app.config import config

logger = logging.getLogger(__name__)


def get_pinecone_client() -> Pinecone:
    """Create and return a Pinecone client."""
    return Pinecone(api_key=config.PINECONE_API_KEY)


def get_or_create_index(pc: Optional[Pinecone] = None) -> object:
    """
    Get existing Pinecone index or create a new one.
    Uses serverless spec on GCP.

    Returns:
        Pinecone Index object
    """
    if pc is None:
        pc = get_pinecone_client()

    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if config.PINECONE_INDEX_NAME not in existing_indexes:
        logger.info(f"🔨 Creating Pinecone index '{config.PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIMENSION,  # 1024 for BAAI/bge-m3
            metric="cosine",
            spec=ServerlessSpec(
                cloud=config.PINECONE_CLOUD,    # "gcp"
                region=config.PINECONE_REGION,   # "us-central1"
            ),
        )
        import time
        logger.info(f"   ⏳ Waiting for index '{config.PINECONE_INDEX_NAME}' to be ready...")
        while not pc.describe_index(config.PINECONE_INDEX_NAME).status['ready']:
            time.sleep(2)
        logger.info(f"   ✅ Index '{config.PINECONE_INDEX_NAME}' created and ready")
    else:
        logger.info(f"   ✅ Using existing index '{config.PINECONE_INDEX_NAME}'")

    return pc.Index(config.PINECONE_INDEX_NAME)


def create_pinecone_index(
    nodes: List,
    embed_model,
    namespace: Optional[str] = None,
) -> VectorStoreIndex:
    """
    Create a VectorStoreIndex backed by Pinecone.

    Args:
        nodes: List of embedded nodes from the ingestion pipeline
        embed_model: The embedding model instance
        namespace: Optional Pinecone namespace (for multi-document isolation)

    Returns:
        VectorStoreIndex backed by Pinecone
    """
    logger.info("\n" + "=" * 60)
    logger.info("PINECONE INDEX CREATION")
    logger.info("=" * 60)

    # Sanitize node metadata — Pinecone rejects None values
    for node in nodes:
        if hasattr(node, 'metadata') and node.metadata:
            clean_meta = {}
            for k, v in node.metadata.items():
                if v is None:
                    clean_meta[k] = ""  # Replace None with empty string
                elif isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                elif isinstance(v, list):
                    # Pinecone accepts list of strings
                    clean_meta[k] = [str(item) for item in v]
                else:
                    clean_meta[k] = str(v)
            node.metadata = clean_meta

    # Get or create the Pinecone index
    pc = get_pinecone_client()
    pinecone_index = get_or_create_index(pc)

    # Create LlamaIndex PineconeVectorStore
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=namespace or "default",
    )

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build the index from nodes
    logger.info(f"📤 Upserting {len(nodes)} nodes to Pinecone...")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    logger.info(f"✅ Pinecone index ready with {len(nodes)} vectors")
    return index


def load_existing_index(
    embed_model,
    namespace: Optional[str] = None,
) -> VectorStoreIndex:
    """
    Load an existing Pinecone-backed VectorStoreIndex (no new ingestion).

    Args:
        embed_model: The embedding model instance
        namespace: Optional Pinecone namespace

    Returns:
        VectorStoreIndex from existing Pinecone data
    """
    pc = get_pinecone_client()
    pinecone_index = get_or_create_index(pc)

    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=namespace or "default",
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    logger.info(f"✅ Loaded existing Pinecone index (namespace: {namespace or 'default'})")
    return index


def wait_for_namespace(namespace: str, expected: int, timeout_s: int = 120) -> None:
    """Block until Pinecone reports all expected vectors (writes are eventually consistent)."""
    import time
    pinecone_index = get_or_create_index()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        summary = pinecone_index.describe_index_stats().namespaces.get(namespace)
        if summary and summary.vector_count >= expected:
            return
        time.sleep(2)
    logger.warning(f"⚠️ Namespace '{namespace}' not fully visible after {timeout_s}s")


def load_namespace_nodes(namespace: str) -> List:
    """Rebuild text nodes from a Pinecone namespace (node text is stored in vector metadata)."""
    pinecone_index = get_or_create_index()
    nodes = []
    for page in pinecone_index.list(namespace=namespace):
        ids = [item.id for item in page.vectors]
        if ids:
            fetched = pinecone_index.fetch(ids=ids, namespace=namespace)
            nodes.extend(metadata_dict_to_node(dict(v.metadata)) for v in fetched.vectors.values())
    logger.info(f"✅ Loaded {len(nodes)} nodes from namespace '{namespace}'")
    return nodes


def delete_namespace(namespace: str) -> None:
    """Delete all vectors in a namespace."""
    pc = get_pinecone_client()
    pinecone_index = get_or_create_index(pc)
    
    try:
        pinecone_index.delete(delete_all=True, namespace=namespace)
        logger.info(f"🗑️ Deleted namespace '{namespace}'")
    except Exception as e:
        error_msg = str(e)
        if "NotFoundException" in error_msg or "Namespace not found" in error_msg or "404" in error_msg:
            logger.info(f"🗑️ Namespace '{namespace}' was already deleted or not found")
        else:
            logger.error(f"❌ Error deleting namespace '{namespace}': {e}")
            raise
