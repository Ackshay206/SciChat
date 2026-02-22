const API_BASE_URL = 'http://localhost:8000/api/v1';

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
        if (!res.ok) throw new Error('Failed to delete document');
    },
};
