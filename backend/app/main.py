"""
FastAPI application entry point.

Includes:
- CORS configuration
- Lifespan management (model loading, Redis connection)
- Route registration
- Logging setup
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import config
from app.dependencies import app_state
from app.services.cache_service import cache_service
from app.api.routes import health, ingest, query, documents, evaluate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown."""
    # Startup
    logger.info("=" * 60)
    logger.info("🚀 Starting Scientific RAG API")
    logger.info("=" * 60)

    # Initialize ML models
    await app_state.initialize()

    # Connect to Redis
    await cache_service.connect()

    logger.info("✅ All services ready")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("🛑 Shutting down...")
    await cache_service.disconnect()
    logger.info("👋 Goodbye!")


# Create FastAPI app
app = FastAPI(
    title="Scientific RAG API",
    description="Production RAG pipeline for scientific papers with hybrid retrieval",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
app.include_router(evaluate.router, prefix="/api/v1", tags=["Evaluation"])

# Built frontend (single-container deployment); mounted last so API routes take precedence
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
