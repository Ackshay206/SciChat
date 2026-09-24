"""
Content-type-aware chunking pipeline — extracted from notebook Cell 9.

Uses different SentenceSplitter configurations based on content_type:
- Text/Sections: 512 tokens, 128 overlap
- Tables: 1400 tokens, 50 overlap (preserve structure)
- Figures: 500 tokens, 80 overlap
- References: 700 tokens, 50 overlap
- Metadata: kept as single node
"""

import hashlib
import logging
from typing import List

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from app.config import config

logger = logging.getLogger(__name__)


def create_optimized_chunking(documents: List[Document]) -> List:
    """
    Create optimized chunking strategy based on content type.
    Uses SentenceSplitter with paragraph_separator="\\n\\n" for all types.

    From notebook Cell 9 — unchanged logic.
    """
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZED CHUNKING PIPELINE")
    logger.info("=" * 60)

    all_nodes = []

    # Track counts by type
    type_counts = {
        "text": 0,
        "section": 0,
        "metadata": 0,
        "table": 0,
        "figure": 0,
        "references": 0,
    }

    for doc in documents:
        content_type = doc.metadata.get("content_type", "text")
        doc_type = doc.metadata.get("type", "text")

        # Determine chunking strategy based on content type
        if content_type == "table" or doc_type == "table":
            # Large chunks for tables to preserve structure
            splitter = SentenceSplitter(
                chunk_size=config.TABLE_CHUNK_SIZE,
                chunk_overlap=config.TABLE_CHUNK_OVERLAP,
                paragraph_separator="\n\n"
            )
            nodes = splitter.get_nodes_from_documents([doc])
            type_counts["table"] += len(nodes)

        elif content_type == "figure" or doc_type in ["image_ocr", "image_description"]:
            # Medium chunks for figures
            splitter = SentenceSplitter(
                chunk_size=config.FIGURE_CHUNK_SIZE,
                chunk_overlap=config.FIGURE_CHUNK_OVERLAP,
                paragraph_separator="\n\n"
            )
            nodes = splitter.get_nodes_from_documents([doc])
            type_counts["figure"] += len(nodes)

        elif content_type == "references" or doc_type == "references":
            # Dedicated chunk size for references
            splitter = SentenceSplitter(
                chunk_size=config.REF_CHUNK_SIZE,
                chunk_overlap=config.REF_CHUNK_OVERLAP,
                paragraph_separator="\n\n"
            )
            nodes = splitter.get_nodes_from_documents([doc])
            type_counts["references"] += len(nodes)

        elif content_type == "metadata" or doc_type == "metadata":
            # Keep metadata as single node (usually small)
            nodes = [TextNode(
                text=doc.text,
                metadata={
                    **doc.metadata,
                    "chunk_id": 0,
                    "total_chunks": 1,
                }
            )]
            type_counts["metadata"] += len(nodes)

        elif content_type == "section" or doc_type == "section":
            splitter = SentenceSplitter(
                chunk_size=config.SEC_CHUNK_SIZE,
                chunk_overlap=config.SEC_CHUNK_OVERLAP,
                paragraph_separator="\n\n"
            )
            nodes = splitter.get_nodes_from_documents([doc])
            type_counts["section"] += len(nodes)

        else:
            splitter = SentenceSplitter(
                chunk_size=config.TEXT_CHUNK_SIZE,
                chunk_overlap=config.TEXT_CHUNK_OVERLAP,
                paragraph_separator="\n\n"
            )
            nodes = splitter.get_nodes_from_documents([doc])
            type_counts["text"] += len(nodes)

        # Ensure all nodes have content_type metadata and a content-based ID
        for node in nodes:
            if "content_type" not in node.metadata:
                node.metadata["content_type"] = content_type
            node.id_ = hashlib.md5(f"{content_type}:{node.get_content()}".encode()).hexdigest()

        all_nodes.extend(nodes)

    # Log summary
    logger.info("\n📊 CHUNKING SUMMARY:")
    logger.info(f"   📄 Text nodes: {type_counts['text']}")
    logger.info(f"   📑 Section nodes: {type_counts['section']}")
    logger.info(f"   📌 Metadata nodes: {type_counts['metadata']}")
    logger.info(f"   📊 Table nodes: {type_counts['table']}")
    logger.info(f"   🖼️ Figure nodes: {type_counts['figure']}")
    logger.info(f"   📚 Reference nodes: {type_counts['references']}")
    logger.info(f"\n✅ Total nodes created: {len(all_nodes)}")

    return all_nodes


def run_optimized_ingestion_pipeline(
    documents: List[Document],
    embed_model
) -> List:
    """
    Run the complete ingestion  pipeline with optimized chunking.

    Args:
        documents: List of documents from collect_all_documents()
        embed_model: The embedding model instance

    Returns:
        List of embedded nodes ready for indexing
    """
    logger.info("\n" + "=" * 60)
    logger.info("INGESTION PIPELINE (OPTIMIZED)")
    logger.info("=" * 60)

    # Step 1: Create optimized chunks
    nodes = create_optimized_chunking(documents)

    # Step 2: Apply embeddings with batch parallelism
    embed_model.embed_batch_size = config.EMBED_BATCH_SIZE

    # num_workers > 1 uses multiprocessing which can't pickle torch
    # tensors on MPS/GPU. Only use workers on CPU.
    num_workers = config.EMBED_NUM_WORKERS
    try:
        import torch
        if torch.backends.mps.is_available() or torch.cuda.is_available():
            num_workers = 1  # Sequential — avoids pickle error on GPU/MPS
    except Exception:
        pass

    logger.info(f"\n🔄 Generating embeddings (batch_size={config.EMBED_BATCH_SIZE}, workers={num_workers})...")

    pipeline = IngestionPipeline(transformations=[embed_model])
    nodes = pipeline.run(
        nodes=nodes,
        show_progress=True,
        num_workers=num_workers,
    )

    logger.info(f"\n✅ Created {len(nodes)} embedded nodes")

    return nodes
