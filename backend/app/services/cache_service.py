"""
Redis semantic caching service.

On query:
1. Embed the question
2. Check Redis for cached question embeddings with cosine similarity > 0.95
3. If hit → return cached answer (sub-millisecond)
4. If miss → query RAG pipeline → cache result with TTL
"""

import json
import hashlib
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import redis.asyncio as redis

from app.config import config

logger = logging.getLogger(__name__)


class CacheService:
    """Semantic caching with Redis."""

    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self._client = redis.from_url(
                config.REDIS_URL,
                decode_responses=False,
            )
            await self._client.ping()
            self._connected = True
            logger.info(f"✅ Redis connected: {config.REDIS_URL}")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}. Caching disabled.")
            self._connected = False

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def get_cached_response(
        self,
        document_id: str,
        question: str,
        question_embedding: List[float],
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a semantically similar question has been cached for this document.

        Args:
            document_id: Document the question is asked against
            question: The query string
            question_embedding: Embedding vector of the question

        Returns:
            Cached response dict or None if no cache hit
        """
        if not self._connected:
            return None

        try:
            async for key in self._client.scan_iter(match=f"cache:{document_id}:*"):
                cached_data = await self._client.get(key)
                if not cached_data:
                    continue

                cached = json.loads(cached_data)
                cached_embedding = np.array(cached.get("embedding", []))

                if len(cached_embedding) == 0:
                    continue

                # Compute cosine similarity
                query_vec = np.array(question_embedding)
                similarity = np.dot(query_vec, cached_embedding) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(cached_embedding)
                )

                if similarity >= config.CACHE_SIMILARITY_THRESHOLD:
                    logger.info(f"🎯 Cache HIT (similarity: {similarity:.4f})")
                    return {
                        "answer": cached["answer"],
                        "sources": cached["sources"],
                        "cached": True,
                        "similarity": float(similarity),
                    }

            return None

        except Exception as e:
            logger.warning(f"Cache lookup error: {e}")
            return None

    async def cache_response(
        self,
        document_id: str,
        question: str,
        question_embedding: List[float],
        answer: str,
        sources: List[Dict],
    ) -> None:
        """
        Cache a query response.

        Args:
            document_id: Document the question was asked against
            question: The query string
            question_embedding: Embedding vector of the question
            answer: The generated answer
            sources: List of source citations
        """
        if not self._connected:
            return

        try:
            cache_key = f"cache:{document_id}:{hashlib.md5(question.encode()).hexdigest()}"

            cache_data = json.dumps({
                "question": question,
                "embedding": question_embedding,
                "answer": answer,
                "sources": sources,
            })

            await self._client.setex(
                cache_key,
                config.CACHE_TTL,
                cache_data,
            )
            logger.info(f"💾 Response cached (TTL: {config.CACHE_TTL}s)")

        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    async def clear_cache(self, document_id: str) -> int:
        """Clear cached queries for a document. Returns count of deleted keys."""
        if not self._connected:
            return 0

        keys = [key async for key in self._client.scan_iter(match=f"cache:{document_id}:*")]
        if keys:
            deleted = await self._client.delete(*keys)
            logger.info(f"🗑️ Cleared {deleted} cached entries")
            return deleted
        return 0


# Singleton instance
cache_service = CacheService()
