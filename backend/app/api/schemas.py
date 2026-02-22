"""Pydantic request/response models for the API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ============================================================
# Query
# ============================================================

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask")
    document_id: Optional[str] = Field(None, description="Specific document namespace")
    stream: bool = Field(False, description="Enable SSE streaming response")


class SourceInfo(BaseModel):
    text: str = Field(..., description="Source text snippet")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo] = []
    cached: bool = False
    latency_ms: float = 0.0


# ============================================================
# Ingest
# ============================================================

class IngestResponse(BaseModel):
    document_id: str
    title: str
    num_nodes: int
    num_documents: int
    sections: List[str] = []
    message: str = "Document ingested successfully"


# ============================================================
# Documents
# ============================================================

class DocumentInfo(BaseModel):
    document_id: str
    title: str
    num_nodes: int
    sections: List[str] = []
    source_file: str = ""


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


# ============================================================
# Evaluation
# ============================================================

class EvaluateRequest(BaseModel):
    eval_type: str = Field(
        "full",
        description="Type of evaluation: 'retriever', 'rag_quality', or 'full'"
    )


class RetrieverMetrics(BaseModel):
    retriever_name: str
    mrr: float = 0.0
    hit_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0


class RagQualityMetrics(BaseModel):
    avg_faithfulness: float = 0.0
    avg_relevancy: float = 0.0
    num_questions: int = 0


class EvaluateResponse(BaseModel):
    retriever_metrics: List[RetrieverMetrics] = []
    rag_quality_metrics: Optional[RagQualityMetrics] = None
    summary: str = ""


# ============================================================
# Health
# ============================================================

class HealthResponse(BaseModel):
    status: str = "ok"
    pinecone_connected: bool = False
    redis_connected: bool = False
    models_loaded: bool = False
    index_ready: bool = False
