'use client';

import { useEffect, useState } from 'react';
import { FileText, PlusCircle, BookOpen, Trash2, Loader2 } from 'lucide-react';
import { api, DocumentInfo } from '@/lib/api';

interface SidebarProps {
    selectedDocId: string | null;
    onSelectDoc: (id: string | null) => void;
    onNewUploadClick: () => void;
}

export default function Sidebar({ selectedDocId, onSelectDoc, onNewUploadClick }: SidebarProps) {
    const [documents, setDocuments] = useState<DocumentInfo[]>([]);
    const [loading, setLoading] = useState(true);

    const loadDocs = async () => {
        try {
            setLoading(true);
            const res = await api.getDocuments();
            setDocuments(res.documents);
        } catch (err) {
            console.error("Failed to load documents", err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadDocs();
        // Expose a global event listener so other components can trigger a refresh
        window.addEventListener('refresh-docs', loadDocs);
        return () => window.removeEventListener('refresh-docs', loadDocs);
    }, []);

    const handleDelete = async (e: React.MouseEvent, id: string) => {
        e.stopPropagation();
        if (!confirm('Are you sure you want to delete this document?')) return;
        try {
            await api.deleteDocument(id);
            if (selectedDocId === id) onSelectDoc(null);
            loadDocs();
        } catch (err) {
            console.error("Failed to delete", err);
        }
    };

    return (
        <div className="flex flex-col h-full bg-card/50 text-card-foreground">
            {/* Header */}
            <div className="p-4 border-b border-border flex items-center justify-between shrink-0">
                <div className="flex items-center gap-2">
                    <BookOpen className="text-primary w-6 h-6" />
                    <h1 className="font-semibold text-lg tracking-tight">SciChat</h1>
                </div>
            </div>

            {/* Documents List */}
            <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-2">
                <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                    Reference Library
                </h2>

                {loading ? (
                    <div className="flex justify-center p-4">
                        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
                    </div>
                ) : documents.length === 0 ? (
                    <div className="text-sm text-muted-foreground text-center p-4">
                        No papers uploaded yet.
                    </div>
                ) : (
                    documents.map(doc => (
                        <div
                            key={doc.document_id}
                            onClick={() => onSelectDoc(doc.document_id)}
                            className={`flex items-center justify-between p-3 rounded-md cursor-pointer transition-colors border group
                ${selectedDocId === doc.document_id
                                    ? 'bg-primary/10 border-primary/30'
                                    : 'bg-muted/30 border-transparent hover:bg-muted/60 hover:border-border'
                                }
              `}
                        >
                            <div className="flex items-center gap-3 min-w-0 pr-2">
                                <FileText className={`w-4 h-4 shrink-0 ${selectedDocId === doc.document_id ? 'text-primary' : 'text-muted-foreground group-hover:text-foreground'}`} />
                                <div className="flex-1 min-w-0">
                                    <p className="text-sm font-medium truncate" title={doc.title}>{doc.title}</p>
                                    <p className="text-xs text-muted-foreground truncate">{doc.num_nodes} nodes indexed</p>
                                </div>
                            </div>
                            <button
                                onClick={(e) => handleDelete(e, doc.document_id)}
                                className="p-1.5 text-muted-foreground hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity rounded-md hover:bg-background"
                                title="Delete Document"
                            >
                                <Trash2 className="w-3.5 h-3.5" />
                            </button>
                        </div>
                    ))
                )}
            </div>

            {/* Footer Actions */}
            <div className="p-4 border-t border-border flex flex-col gap-2 shrink-0">
                <button
                    onClick={onNewUploadClick}
                    className="flex items-center gap-2 w-full p-2 text-sm font-medium rounded-md transition-colors text-primary hover:text-primary-foreground hover:bg-primary/90 justify-center border border-primary/20 shadow-sm"
                >
                    <PlusCircle className="w-4 h-4" />
                    Add Paper
                </button>
            </div>
        </div>
    );
}
