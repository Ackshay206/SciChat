export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL
    ?? (process.env.NODE_ENV === 'development' ? 'http://localhost:8000/api/v1' : '/api/v1');

export interface DocumentInfo {
    document_id: string;
    title: string;
    num_nodes: number;
    sections: string[];
    source_file: string;
    authors: string[];
    emails: string[];
    organizations: string[];
}

export interface DocumentsResponse {
    total: number;
    documents: DocumentInfo[];
}

export const api = {
    // Fetch all documents
    getDocuments: async (): Promise<DocumentsResponse> => {
        const res = await fetch(`${API_BASE_URL}/documents`);
        if (!res.ok) throw new Error('Failed to fetch documents');
        return res.json();
    },

    // Delete a document
    deleteDocument: async (documentId: string): Promise<void> => {
        const res = await fetch(`${API_BASE_URL}/documents/${documentId}`, {
            method: 'DELETE',
        });
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || 'Failed to delete document');
    },
};
