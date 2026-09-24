"""
Query endpoint with streaming (SSE) and non-streaming support.

POST /api/v1/query
- stream=false: Returns JSON { answer, sources[], latency_ms }
- stream=true: Returns Server-Sent Events stream
"""

import json
import logging
import re
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.schemas import QueryRequest, QueryResponse, SourceInfo
from app.core.retrieval import aquery
from app.dependencies import app_state
from app.services.cache_service import cache_service

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_QUESTION_CHARS = 1000
INJECTION_PATTERN = re.compile(
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)"
    r"|disregard (all |the )?(previous|prior|above)"
    r"|(reveal|show|print|repeat) (your|the) (system )?prompt"
    r"|you are now",
    re.IGNORECASE,
)


def _check_question(question: str) -> Optional[str]:
    """Return a refusal message if the question fails input guardrails, else None."""
    question = question.strip()
    if not question:
        return "Please enter a question."
    if len(question) > MAX_QUESTION_CHARS:
        return f"Please keep your question under {MAX_QUESTION_CHARS} characters."
    if INJECTION_PATTERN.search(question):
        return "I can only answer questions about the selected paper."
    return None


def _error_message(e: Exception) -> str:
    """Map LLM quota/billing errors to a user-facing message."""
    text = str(e)
    if any(marker in text for marker in ("429", "RESOURCE_EXHAUSTED", "402", "quota")):
        return "The demo has reached its AI usage limit for now. Please try again later."
    return f"Something went wrong: {text}"


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG pipeline.

    If stream=true, returns SSE stream.
    If stream=false, returns JSON response with answer and sources.
    """
    refusal = _check_question(request.question)
    if refusal:
        if request.stream:
            return StreamingResponse(
                _sse_events({"answer": refusal, "sources": []}),
                media_type="text/event-stream",
            )
        return QueryResponse(answer=refusal)

    # Switch to the requested document's namespace if needed
    if request.document_id:
        try:
            app_state.switch_document(request.document_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load document: {str(e)}")

    if not app_state.has_index or app_state.query_engine is None:
        raise HTTPException(
            status_code=400,
            detail="No document indexed yet. Upload a PDF first via /api/v1/ingest"
        )

    start = time.time()
    doc_id = app_state.active_doc_id
    engine = app_state.query_engine

    embedding = None
    cached = None
    if cache_service.is_connected:
        embedding = app_state.embed_model.get_text_embedding(request.question)
        cached = await cache_service.get_cached_response(doc_id, request.question, embedding)

    if request.stream:
        return StreamingResponse(
            _stream_response(engine, request.question, doc_id, embedding, cached, start),
            media_type="text/event-stream",
        )

    if cached:
        return QueryResponse(
            answer=cached["answer"],
            sources=[SourceInfo(**s) for s in cached["sources"]],
            cached=True,
            latency_ms=(time.time() - start) * 1000,
        )

    try:
        result = await aquery(engine, request.question)
    except Exception as e:
        raise HTTPException(status_code=503, detail=_error_message(e))
    latency = (time.time() - start) * 1000

    await cache_service.cache_response(
        doc_id, request.question, embedding, result["answer"], result["sources"]
    )

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        cached=False,
        latency_ms=latency,
    )


def _sse_events(result: dict, meta: Optional[dict] = None):
    yield f"data: {json.dumps({'type': 'answer', 'content': result['answer']})}\n\n"
    yield f"data: {json.dumps({'type': 'sources', 'content': result['sources']})}\n\n"
    if meta:
        yield f"data: {json.dumps({'type': 'meta', 'content': meta})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


async def _stream_response(engine, question: str, doc_id: str, embedding, cached: Optional[dict], start: float):
    """SSE stream generator."""
    try:
        result = cached or await aquery(engine, question)

        meta = {"latency_ms": (time.time() - start) * 1000, "cached": bool(cached)}
        for event in _sse_events(result, meta):
            yield event

        if not cached:
            await cache_service.cache_response(
                doc_id, question, embedding, result["answer"], result["sources"]
            )

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': _error_message(e)})}\n\n"
