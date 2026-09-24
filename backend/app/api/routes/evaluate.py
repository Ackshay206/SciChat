"""
Evaluation endpoint — developer/CI use only.

POST /api/v1/evaluate
- eval_type: "retriever" | "rag_quality" | "full"
- Runs pipeline evaluation against test datasets
- NOT for end-user use
"""

import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    RagQualityMetrics,
    RetrieverMetrics,
)
from app.config import config
from app.dependencies import app_state

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/evaluate", response_model=EvaluateResponse)
async def run_evaluation(request: EvaluateRequest):
    """
    Run pipeline evaluation.

    - retriever: MRR, Hit Rate, Precision, Recall (generated QA pairs)
    - rag_quality: Faithfulness, Relevancy (gold QA dataset)
    - full: both

    Developer/CI endpoint only.
    """
    if config.DEMO_MODE:
        raise HTTPException(status_code=403, detail="Evaluation is disabled in the hosted demo.")
    if app_state.judge_llm is None:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured.")
    if not app_state.has_index or not app_state.nodes:
        raise HTTPException(
            status_code=400,
            detail="No document indexed. Ingest a PDF first."
        )

    # Lazy import to avoid loading evaluation deps at startup
    from app.core.evaluation import evaluate_rag_quality, evaluate_retrievers

    retriever_results = []
    rag_quality_result = None

    if request.eval_type in ("retriever", "full"):
        logger.info("📊 Running retriever evaluation...")
        retriever_results = await evaluate_retrievers(
            index=app_state.index,
            nodes=app_state.nodes,
            llm=app_state.llm,
        )

    if request.eval_type in ("rag_quality", "full"):
        logger.info("📊 Running RAG quality evaluation...")
        rag_quality_result = await evaluate_rag_quality(
            query_engine=app_state.query_engine,
            judge_llm=app_state.judge_llm,
        )

    # Build summary
    summary_parts = []
    if retriever_results:
        for r in retriever_results:
            summary_parts.append(
                f"{r.retriever_name}: MRR={r.mrr:.3f}, "
                f"HitRate={r.hit_rate:.3f}, "
                f"Precision={r.precision:.3f}, "
                f"Recall={r.recall:.3f}"
            )
    if rag_quality_result:
        summary_parts.append(
            f"RAG Quality: Faithfulness={rag_quality_result.avg_faithfulness:.3f}, "
            f"Relevancy={rag_quality_result.avg_relevancy:.3f}"
        )

    return EvaluateResponse(
        retriever_metrics=retriever_results,
        rag_quality_metrics=rag_quality_result,
        summary=" | ".join(summary_parts),
    )
