"""
PDF ingestion endpoint.

POST /api/v1/ingest
- Accepts: multipart/form-data (PDF file upload)
- Runs: collect_all_documents() → run_optimized_ingestion_pipeline() → create_pinecone_index()
- Returns: { document_id, title, num_nodes, sections[], authors[], emails[], organizations[] }
"""

import hashlib
import logging
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import IngestResponse
from app.core.chunking import run_optimized_ingestion_pipeline
from app.core.document_parser import collect_all_documents, analyze_document_structure
from app.core.indexing import create_pinecone_index
from app.core.retrieval import create_query_engine
from app.dependencies import app_state

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF document.

    Pipeline: Parse → Chunk → Embed → Index (Pinecone)
    """
    if not app_state.is_initialized:
        raise HTTPException(status_code=503, detail="Models not yet loaded. Please wait.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    logger.info(f"📥 Ingesting: {file.filename}")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Generate document ID from content hash
        doc_id = hashlib.md5(content).hexdigest()[:12]

        # Step 1: Analyze structure first (for metadata)
        structure = analyze_document_structure(tmp_path)

        # Step 2: Collect all documents from PDF
        documents = collect_all_documents(tmp_path)

        if not documents:
            raise HTTPException(status_code=422, detail="No content extracted from PDF")

        # Step 3: Run ingestion pipeline (chunking + embedding)
        nodes = run_optimized_ingestion_pipeline(documents, app_state.embed_model)

        # Step 4: Create Pinecone index
        namespace = f"doc_{doc_id}"
        index = create_pinecone_index(nodes, app_state.embed_model, namespace=namespace)

        # Step 5: Create query engine
        query_engine = create_query_engine(
            index=index,
            nodes=nodes,
            llm=app_state.llm,
            retriever_type="hybrid",
        )

        # Store state
        app_state.index = index
        app_state.nodes = nodes
        app_state.query_engine = query_engine
        app_state.active_doc_id = doc_id

        # Extract title and sections for response
        title = structure.get("title", "Unknown") or "Unknown"
        authors = structure.get("authors", [])
        emails = structure.get("emails", [])
        organizations = structure.get("organizations", [])

        sections = []
        for doc in documents:
            if doc.metadata.get("type") == "section":
                section_name = doc.metadata.get("section_name", "")
                if section_name and section_name not in sections:
                    sections.append(section_name)

        # Store document metadata and persist
        app_state.documents_metadata[doc_id] = {
            "title": title,
            "filename": file.filename,
            "num_nodes": len(nodes),
            "num_documents": len(documents),
            "sections": sections,
            "namespace": namespace,
            "authors": authors,
            "emails": emails,
            "organizations": organizations,
        }
        app_state.save_metadata()

        logger.info(f"✅ Ingestion complete: {title} ({len(nodes)} nodes)")

        return IngestResponse(
            document_id=doc_id,
            title=title,
            num_nodes=len(nodes),
            num_documents=len(documents),
            sections=sections,
            authors=authors,
            emails=emails,
            organizations=organizations,
        )

    finally:
        # Cleanup temp file
        os.unlink(tmp_path)
