"""
Query endpoint with streaming (SSE) and non-streaming support.

POST /api/v1/query
- stream=false: Returns JSON { answer, sources[], latency_ms }
- stream=true: Returns Server-Sent Events stream
"""

import json
import logging
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.schemas import QueryRequest, QueryResponse, SourceInfo
from app.core.retrieval import aquery
from app.dependencies import app_state
from app.services.cache_service import cache_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG pipeline.

    If stream=true, returns SSE stream.
    If stream=false, returns JSON response with answer and sources.
    """
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

    # Streaming response
    if request.stream:
        return StreamingResponse(
            _stream_response(request.question),
            media_type="text/event-stream",
        )

    # Non-streaming response
    start = time.time()

    # Check cache first
    if cache_service.is_connected:
        try:
            # Get question embedding for cache lookup
            question_embedding = app_state.embed_model.get_text_embedding(request.question)
            cached = await cache_service.get_cached_response(
                request.question, question_embedding
            )
            if cached:
                latency = (time.time() - start) * 1000
                return QueryResponse(
                    answer=cached["answer"],
                    sources=[SourceInfo(**s) for s in cached["sources"]],
                    cached=True,
                    latency_ms=latency,
                )
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")

    # Query the RAG pipeline
    result = await aquery(app_state.query_engine, request.question)

    latency = (time.time() - start) * 1000

    # Cache the result
    if cache_service.is_connected:
        try:
            question_embedding = app_state.embed_model.get_text_embedding(request.question)
            await cache_service.cache_response(
                request.question,
                question_embedding,
                result["answer"],
                result["sources"],
            )
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        cached=False,
        latency_ms=latency,
    )


async def _stream_response(question: str):
    """SSE stream generator."""
    try:
        result = await aquery(app_state.query_engine, question)

        # Stream the answer
        yield f"data: {json.dumps({'type': 'answer', 'content': result['answer']})}\n\n"

        # Stream sources
        yield f"data: {json.dumps({'type': 'sources', 'content': result['sources']})}\n\n"

        # Done event
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
