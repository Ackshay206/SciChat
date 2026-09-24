'use client';

import { useEffect, useState } from 'react';
import { FileText, PlusCircle, BookOpen, Trash2, Loader2, Users, Mail, Building2, ChevronDown, ChevronRight } from 'lucide-react';
import { api, DocumentInfo } from '@/lib/api';
import { lastEval } from '@/lib/evalResults';

interface SidebarProps {
    selectedDocId: string | null;
    onSelectDoc: (id: string | null) => void;
    onNewUploadClick: () => void;
}

export default function Sidebar({ selectedDocId, onSelectDoc, onNewUploadClick }: SidebarProps) {
    const [documents, setDocuments] = useState<DocumentInfo[]>([]);
    const [loading, setLoading] = useState(true);
    const [expandedDocId, setExpandedDocId] = useState<string | null>(null);

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
        window.addEventListener('refresh-docs', loadDocs);
        return () => window.removeEventListener('refresh-docs', loadDocs);
    }, []);

    const handleDelete = async (e: React.MouseEvent, id: string) => {
        e.stopPropagation();
        if (!confirm('Are you sure you want to delete this document?')) return;
        try {
            await api.deleteDocument(id);
            if (selectedDocId === id) onSelectDoc(null);
            if (expandedDocId === id) setExpandedDocId(null);
            loadDocs();
        } catch (err) {
            alert(err instanceof Error ? err.message : 'Failed to delete paper');
        }
    };

    const handleSelectDoc = (id: string) => {
        onSelectDoc(id);
        setExpandedDocId(expandedDocId === id ? null : id);
    };

    const selectedDoc = documents.find(d => d.document_id === selectedDocId);

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
                <h2 className="text-sm font-medium text-muted-foreground mb-2">
                    Papers
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
                        <div key={doc.document_id} className="flex flex-col">
                            <div
                                onClick={() => handleSelectDoc(doc.document_id)}
                                className={`flex items-center justify-between p-3 rounded-md cursor-pointer transition-colors border group
                                    ${selectedDocId === doc.document_id
                                        ? 'bg-accent/10 border-accent/30'
                                        : 'bg-muted/30 border-transparent hover:bg-muted/60 hover:border-border'
                                    }
                                `}
                            >
                                <div className="flex items-center gap-3 min-w-0 pr-2">
                                    {selectedDocId === doc.document_id ? (
                                        <ChevronDown className="w-4 h-4 shrink-0 text-accent" />
                                    ) : (
                                        <ChevronRight className="w-4 h-4 shrink-0 text-muted-foreground group-hover:text-foreground" />
                                    )}
                                    <div className="flex-1 min-w-0">
                                        <p className="text-sm font-medium truncate" title={doc.title}>{doc.title}</p>
                                        <p className="text-xs text-muted-foreground truncate">{doc.num_nodes} chunks indexed</p>
                                    </div>
                                </div>
                                <button
                                    onClick={(e) => handleDelete(e, doc.document_id)}
                                    className="p-1.5 text-muted-foreground hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity rounded-md hover:bg-background"
                                    title="Delete paper"
                                >
                                    <Trash2 className="w-3.5 h-3.5" />
                                </button>
                            </div>

                            {/* Expanded metadata panel */}
                            {selectedDocId === doc.document_id && expandedDocId === doc.document_id && (
                                <div className="ml-4 mt-1 mb-2 p-3 rounded-md bg-muted/20 border border-border/50 space-y-2 text-xs">
                                    {doc.title && (
                                        <p className="text-foreground font-medium text-sm leading-snug">{doc.title}</p>
                                    )}

                                    {doc.authors && doc.authors.length > 0 && (
                                        <div className="flex items-start gap-2">
                                            <Users className="w-3.5 h-3.5 text-accent shrink-0 mt-0.5" />
                                            <p className="text-muted-foreground leading-relaxed">
                                                {doc.authors.join(', ')}
                                            </p>
                                        </div>
                                    )}

                                    {doc.emails && doc.emails.length > 0 && (
                                        <div className="flex items-start gap-2">
                                            <Mail className="w-3.5 h-3.5 text-accent shrink-0 mt-0.5" />
                                            <div className="flex flex-col gap-0.5">
                                                {doc.emails.map((email, i) => (
                                                    <a key={i} href={`mailto:${email}`}
                                                        className="text-accent/90 hover:text-accent underline-offset-2 hover:underline truncate">
                                                        {email}
                                                    </a>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {doc.organizations && doc.organizations.length > 0 && (
                                        <div className="flex items-start gap-2">
                                            <Building2 className="w-3.5 h-3.5 text-accent shrink-0 mt-0.5" />
                                            <p className="text-muted-foreground leading-relaxed">
                                                {doc.organizations.join('; ')}
                                            </p>
                                        </div>
                                    )}

                                    <div className="pt-1 border-t border-border/30 flex items-center gap-3 text-muted-foreground">
                                        <span>{doc.num_nodes} chunks</span>
                                        <span>{doc.sections?.length || 0} sections</span>
                                    </div>
                                </div>
                            )}
                        </div>
                    ))
                )}
            </div>

            {/* Last evaluation, set like a results table in a paper */}
            <figure className="hidden md:block mx-4 mb-4 text-xs shrink-0">
                <figcaption className="text-muted-foreground leading-relaxed mb-2">
                    <span className="text-foreground font-medium">Table 1.</span> Answer quality on {lastEval.questions} hand-written
                    questions about <span className="italic">{lastEval.paper}</span>, graded by {lastEval.judge}.
                </figcaption>
                <table className="w-full border-y border-foreground/40 tabular-nums">
                    <thead>
                        <tr className="border-b border-foreground/20 text-muted-foreground">
                            <th className="text-left font-normal py-1.5">Metric</th>
                            <th className="text-right font-normal py-1.5">Score</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr title="Share of answers whose claims are supported by the retrieved passages">
                            <td className="pt-1.5">Faithfulness</td>
                            <td className="pt-1.5 text-right font-medium">{lastEval.faithfulness.toFixed(2)}</td>
                        </tr>
                        <tr title="Share of answers that address the question asked">
                            <td className="pb-1.5">Relevancy</td>
                            <td className="pb-1.5 text-right font-medium">{lastEval.relevancy.toFixed(2)}</td>
                        </tr>
                    </tbody>
                </table>
                <p className="mt-2 text-muted-foreground">
                    Answers by {lastEval.generator}. Evaluated in CI on {lastEval.date}.{' '}
                    <a href={lastEval.runUrl} target="_blank" rel="noreferrer" className="text-accent hover:underline underline-offset-2">
                        View run
                    </a>
                </p>
            </figure>

            {/* Footer Actions */}
            <div className="p-4 border-t border-border flex flex-col gap-2 shrink-0">
                <button
                    onClick={onNewUploadClick}
                    className="flex items-center gap-2 w-full p-2 text-sm font-medium rounded-md transition-colors text-primary hover:text-primary-foreground hover:bg-primary/90 justify-center border border-primary/20 shadow-sm"
                >
                    <PlusCircle className="w-4 h-4" />
                    Upload a paper
                </button>
            </div>
        </div>
    );
}
