"""
Hybrid retrieval module — extracted from notebook Cells 11-12, 16.

Contains:
- Vector retriever (Pinecone-backed, top_k=50)
- BM25 retriever (PyStemmer, top_k=50)
- QueryFusionRetriever with Reciprocal Rank Fusion (0.9/0.1 weights)
- SentenceTransformerRerank cross-encoder (top_n=10)
- RetrieverQueryEngine with reranker postprocessor
- Streaming query support for SSE
"""

import logging
from typing import Any, Dict, List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from app.config import config

logger = logging.getLogger(__name__)


def create_vector_retriever(index: VectorStoreIndex):
    """Create vector retriever from index."""
    return index.as_retriever(similarity_top_k=config.RETRIEVER_TOP_K)


def create_bm25_retriever(nodes: List):
    """Create BM25 retriever from nodes."""
    return BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=config.RETRIEVER_TOP_K,
    )


def create_hybrid_retriever(
    index: VectorStoreIndex,
    nodes: List,
) -> QueryFusionRetriever:
    """
    Create hybrid fusion retriever combining Vector + BM25.
    Uses Reciprocal Rank Fusion with configurable weights.

    From notebook Cell 11 — unchanged logic.
    """
    vector_retriever = create_vector_retriever(index)
    bm25_retriever = create_bm25_retriever(nodes)

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        retriever_weights=[config.VECTOR_WEIGHT, config.BM25_WEIGHT],
        similarity_top_k=config.RETRIEVER_TOP_K,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=True,
        verbose=False,
    )

    logger.info(
        f"✅ Hybrid retriever created "
        f"(Vector: {config.VECTOR_WEIGHT}, BM25: {config.BM25_WEIGHT}, "
        f"top_k: {config.RETRIEVER_TOP_K})"
    )
    return hybrid_retriever


def create_reranker() -> SentenceTransformerRerank:
    """
    Create cross-encoder reranker.
    Uses ms-marco-MiniLM-L-6-v2, returns top 10.

    From notebook Cell 12 — unchanged.
    """
    reranker = SentenceTransformerRerank(
        model=config.RERANKER_MODEL,
        top_n=config.RERANKER_TOP_N,
    )
    logger.info(f"✅ Reranker created (model: {config.RERANKER_MODEL}, top_n: {config.RERANKER_TOP_N})")
    return reranker


def create_query_engine(
    index: VectorStoreIndex,
    nodes: List,
    llm,
    retriever_type: str = "hybrid",
) -> RetrieverQueryEngine:
    """
    Create a query engine with retriever + reranker.

    Args:
        index: VectorStoreIndex (Pinecone-backed)
        nodes: List of nodes for BM25 retriever
        llm: LLM instance for generation
        retriever_type: "vector", "bm25", or "hybrid"

    Returns:
        RetrieverQueryEngine with reranker postprocessor
    """
    reranker = create_reranker()

    if retriever_type == "vector":
        retriever = create_vector_retriever(index)
    elif retriever_type == "bm25":
        retriever = create_bm25_retriever(nodes)
    else:
        retriever = create_hybrid_retriever(index, nodes)

    engine = RetrieverQueryEngine.from_args(
        retriever,
        llm=llm,
        node_postprocessors=[reranker],
    )

    logger.info(f"✅ Query engine created (retriever: {retriever_type})")
    return engine


def create_all_retrievers(
    index: VectorStoreIndex,
    nodes: List,
) -> Dict[str, Any]:
    """
    Create all three retrievers for evaluation purposes.

    Returns dict with keys: "vector", "bm25", "hybrid"
    """
    return {
        "vector": create_vector_retriever(index),
        "bm25": create_bm25_retriever(nodes),
        "hybrid": create_hybrid_retriever(index, nodes),
    }


async def aquery(engine: RetrieverQueryEngine, question: str) -> Dict[str, Any]:
    """
    Async query wrapper that returns structured response.

    Returns:
        Dict with answer, sources, and metadata
    """
    response = await engine.aquery(question)

    sources = []
    for node in response.source_nodes:
        sources.append({
            "text": node.node.text[:500],  # Truncate for API response
            "score": float(node.score) if node.score is not None else 0.0,
            "metadata": {
                k: v for k, v in node.node.metadata.items()
                if k in ["content_type", "type", "section_name", "page", "table_index"]
            },
        })

    return {
        "answer": str(response),
        "sources": sources,
    }


async def aquery_streaming(engine: RetrieverQueryEngine, question: str):
    """
    Async streaming query for SSE responses.
    Yields token chunks as they arrive.

    Used by the /query endpoint when stream=true.
    """
    streaming_response = await engine.aquery(question)

    # For SSE, we yield the full response and sources
    # LlamaIndex streaming is handled at the LLM level
    yield {
        "type": "answer",
        "content": str(streaming_response),
    }

    # Yield sources after the answer
    sources = []
    for node in streaming_response.source_nodes:
        sources.append({
            "text": node.node.text[:500],
            "score": node.score,
            "metadata": {
                k: v for k, v in node.node.metadata.items()
                if k in ["content_type", "type", "section_name", "page", "table_index"]
            },
        })

    yield {
        "type": "sources",
        "content": sources,
    }
