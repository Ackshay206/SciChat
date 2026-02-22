"""Health check endpoint."""

import logging

from fastapi import APIRouter

from app.api.schemas import HealthResponse
from app.dependencies import app_state
from app.services.cache_service import cache_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health status."""
    pinecone_ok = False
    try:
        if app_state.has_index:
            pinecone_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if app_state.is_initialized else "starting",
        pinecone_connected=pinecone_ok,
        redis_connected=cache_service.is_connected,
        models_loaded=app_state.is_initialized,
        index_ready=app_state.has_index,
    )
