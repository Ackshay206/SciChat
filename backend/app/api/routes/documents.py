"""Document management endpoints."""

import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import DocumentInfo, DocumentListResponse
from app.core.indexing import delete_namespace
from app.dependencies import app_state
from app.services.cache_service import cache_service

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
            authors=meta.get("authors", []),
            emails=meta.get("emails", []),
            organizations=meta.get("organizations", []),
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
        authors=meta.get("authors", []),
        emails=meta.get("emails", []),
        organizations=meta.get("organizations", []),
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
        await cache_service.clear_cache(document_id)
        del app_state.documents_metadata[document_id]

        # Persist metadata after deletion
        app_state.save_metadata()

        # Reset index/engine if this was the active document
        if app_state.active_doc_id == document_id:
            app_state.active_doc_id = None
            if app_state.documents_metadata:
                # Switch to the next available document
                next_id = next(iter(app_state.documents_metadata))
                try:
                    app_state.switch_document(next_id)
                except Exception:
                    app_state.index = None
                    app_state.nodes = []
                    app_state.query_engine = None
            else:
                app_state.index = None
                app_state.nodes = []
                app_state.query_engine = None

        logger.info(f"🗑️ Deleted document: {document_id}")
        return {"message": f"Document '{document_id}' deleted", "document_id": document_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")
