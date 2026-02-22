"""Document management endpoints."""

import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import DocumentInfo, DocumentListResponse
from app.core.indexing import delete_namespace
from app.dependencies import app_state

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all indexed documents."""
    docs = [
        DocumentInfo(
            document_id=doc_id,
            title=meta.get("title", "Unknown"),
            num_nodes=meta.get("num_nodes", 0),
            sections=meta.get("sections", []),
            source_file=meta.get("filename", ""),
        )
        for doc_id, meta in app_state.documents_metadata.items()
    ]
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get("/documents/{document_id}", response_model=DocumentInfo)
async def get_document(document_id: str):
    """Get document metadata by ID."""
    meta = app_state.documents_metadata.get(document_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

    return DocumentInfo(
        document_id=document_id,
        title=meta.get("title", "Unknown"),
        num_nodes=meta.get("num_nodes", 0),
        sections=meta.get("sections", []),
        source_file=meta.get("filename", ""),
    )


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its vectors from Pinecone."""
    meta = app_state.documents_metadata.get(document_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

    try:
        namespace = meta.get("namespace", f"doc_{document_id}")
        delete_namespace(namespace)
        del app_state.documents_metadata[document_id]

        # Reset index/engine if this was the active document
        if not app_state.documents_metadata:
            app_state.index = None
            app_state.nodes = []
            app_state.query_engine = None

        logger.info(f"🗑️ Deleted document: {document_id}")
        return {"message": f"Document '{document_id}' deleted", "document_id": document_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")
