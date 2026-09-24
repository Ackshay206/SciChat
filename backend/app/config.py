"""
Configuration module for the Scientific RAG application.
Dataclass-based config extracted from the enhanced notebook (Cell 3).
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()  # Loads backend/.env (CWD)


@dataclass
class Config:
    """
    Application configuration.
    Original notebook fields preserved exactly, with production additions.
    """

    # ============================================================
    # API Keys
    # ============================================================
    GOOGLE_API_KEY: str = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", "").strip()
    )
    OPENAI_API_KEY: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "").strip()
    )
    PINECONE_API_KEY: str = field(
        default_factory=lambda: os.getenv("PINECONE_API_KEY", "").strip()
    )

    # ============================================================
    # Model Configuration (from notebook Cell 3)
    # ============================================================
    LLM_MODEL: str = "models/gemini-3.5-flash-lite"
    JUDGE_LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"

    # ============================================================
    # Chunking Configuration (from notebook Cell 3)
    # ============================================================
    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 128

    TEXT_CHUNK_SIZE = 512
    TEXT_CHUNK_OVERLAP = 128

    SEC_CHUNK_SIZE = 256
    SEC_CHUNK_OVERLAP = 80

    TABLE_CHUNK_SIZE = 1000
    TABLE_CHUNK_OVERLAP = 128

    # Embedding batch settings
    EMBED_BATCH_SIZE: int = 32  # Smaller batches → lower peak RAM on CPU
    EMBED_NUM_WORKERS: int = 4

    FIGURE_CHUNK_SIZE = 500
    FIGURE_CHUNK_OVERLAP = 120

    REF_CHUNK_SIZE = 700
    REF_CHUNK_OVERLAP = 120
    
    # Retrieval
    TOP_K: int = 5
    VECTOR_WEIGHT: float = 0.9
    BM25_WEIGHT: float = 0.1

    # ============================================================
    # Retrieval Configuration (from notebook Cells 11-12)
    # ============================================================
  
    RETRIEVER_TOP_K: int = 50       # Top-k for all retrievers
    RERANKER_TOP_N: int = 10        # Top-n after cross-encoder reranking

    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ============================================================
    # Pinecone Configuration
    # ============================================================
    PINECONE_INDEX_NAME: str = field(
        default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "scientific-papers").strip()
    )
    PINECONE_CLOUD: str = field(
        default_factory=lambda: os.getenv("PINECONE_CLOUD", "aws").strip()
    )
    PINECONE_REGION: str = field(
        default_factory=lambda: os.getenv("PINECONE_REGION", "us-east-1").strip()
    )
    EMBEDDING_DIMENSION: int = 768  # BAAI/bge-base-en-v1.5 dimension

    # ============================================================
    # Redis Configuration
    # ============================================================
    REDIS_URL: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379")
    )
    CACHE_TTL: int = 3600  # Cache TTL in seconds (1 hour)
    CACHE_SIMILARITY_THRESHOLD: float = 0.95  # Semantic similarity threshold for cache hit

    # ============================================================
    # Evaluation Configuration
    # ============================================================
    GOLD_QA_PATH: str = "data/eval/transformer_gold_qa_50.txt"
    EVAL_METRICS_RETRIEVER: list = field(
        default_factory=lambda: ["mrr", "hit_rate", "precision", "recall"]
    )

    # ============================================================
    # Deployment
    # ============================================================
    DEMO_MODE: bool = field(
        default_factory=lambda: os.getenv("DEMO_MODE", "false").strip().lower() == "true"
    )
    CORS_ORIGINS: list = field(
        default_factory=lambda: os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
    )
    MAX_UPLOAD_MB: int = 10

    def validate(self) -> None:
        """Validate that required API keys are present (OPENAI_API_KEY is only needed for evaluation)."""
        missing = []
        if not self.GOOGLE_API_KEY:
            missing.append("GOOGLE_API_KEY")
        if not self.PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")
        if missing:
            raise ValueError(
                f"Missing required API keys: {', '.join(missing)}. "
                f"Set them in your .env file or environment variables."
            )


# Singleton config instance
config = Config()
