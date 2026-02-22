'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, Loader2, StopCircle, FileText, Upload } from 'lucide-react';
import { useStreamingQuery } from '@/hooks/useStreamingQuery';
import { marked } from 'marked';

interface ChatInterfaceProps {
    selectedDocId: string | null;
    onNewUploadClick: () => void;
}

export default function ChatInterface({ selectedDocId, onNewUploadClick }: ChatInterfaceProps) {
    const [input, setInput] = useState('');
    const { messages, sendMessage, isLoading, stopGeneration, setMessages } = useStreamingQuery();
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to latest message
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Reset chat when document changes
    useEffect(() => {
        setMessages([]);
    }, [selectedDocId, setMessages]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;
        sendMessage(input, selectedDocId || undefined);
        setInput('');
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    // Safe markdown parsing
    const parseMarkdown = (text: string) => {
        try {
            return { __html: marked.parse(text) as string };
        } catch (e) {
            return { __html: text };
        }
    };

    return (
        <>
            {/* Header */}
            <div className="h-16 px-6 border-b border-border/50 flex items-center justify-between shrink-0 bg-background/80 backdrop-blur-sm z-10 sticky top-0">
                <h2 className="text-lg font-medium text-foreground tracking-tight flex items-center gap-2">
                    Research Assistant <Sparkles className="w-4 h-4 text-primary" />
                </h2>
                {selectedDocId && (
                    <span className="text-xs font-medium px-2.5 py-1 bg-primary/10 text-primary rounded-full border border-primary/20 flex items-center gap-1.5">
                        <FileText className="w-3.5 h-3.5" />
                        Active Document
                    </span>
                )}
            </div>

            {/* Messages Area - Scrollable */}
            <div className="flex-1 overflow-y-auto px-6 py-6 scroll-smooth">
                <div className="max-w-3xl mx-auto space-y-8 pb-20">

                    {/* AI Welcome Message (shows if no messages) */}
                    {messages.length === 0 && (
                        <div className="flex gap-4">
                            <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center shrink-0 border border-primary/30">
                                <Sparkles className="w-4 h-4 text-primary" />
                            </div>
                            <div className="space-y-2 mt-1 flex-1">
                                <p className="text-sm font-semibold text-foreground">SciChat AI</p>
                                <div className="prose prose-invert max-w-none text-sm text-foreground leading-relaxed">
                                    <p>Hello! I'm your scientific research assistant.</p>
                                    {selectedDocId ? (
                                        <p>I am ready to answer questions about the selected document. Ask me anything!</p>
                                    ) : (
                                        <p>Select a document from the sidebar or upload a new one to begin analyzing.</p>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Render Messages */}
                    {messages.map((msg) => (
                        <div key={msg.id} className="flex gap-4">
                            {msg.role === 'assistant' ? (
                                <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center shrink-0 border border-primary/30">
                                    <Sparkles className="w-4 h-4 text-primary" />
                                </div>
                            ) : (
                                <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center shrink-0 border border-border">
                                    <div className="w-4 h-4 rounded-full bg-foreground/20" />
                                </div>
                            )}

                            <div className="space-y-2 mt-1 flex-1 min-w-0">
                                <p className="text-sm font-semibold text-foreground">
                                    {msg.role === 'assistant' ? 'SciChat AI' : 'You'}
                                </p>

                                <div className="prose prose-invert max-w-none text-sm text-foreground leading-relaxed">
                                    {msg.content ? (
                                        <div dangerouslySetInnerHTML={parseMarkdown(msg.content)} />
                                    ) : (
                                        <Loader2 className="w-4 h-4 animate-spin text-muted-foreground mt-1" />
                                    )}
                                </div>

                                {/* Sources Array (only on assistant messages that aren't streaming) */}
                                {msg.sources && msg.sources.length > 0 && (
                                    <div className="mt-4 space-y-2">
                                        <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Sources ({msg.sources.length})</p>
                                        <div className="flex flex-col gap-2">
                                            {msg.sources.map((source: any, i: number) => (
                                                <div key={i} className="text-xs border border-border bg-card/50 rounded-md p-3">
                                                    <div className="flex justify-between items-start mb-1">
                                                        <span className="font-medium text-primary">
                                                            {source.metadata?.section_name || `Page ${source.metadata?.page || 'N/A'}`}
                                                        </span>
                                                        <span className="text-muted-foreground bg-background px-1.5 py-0.5 rounded text-[10px] uppercase border border-border">
                                                            {source.metadata?.content_type || 'text'}
                                                        </span>
                                                    </div>
                                                    <p className="text-muted-foreground line-clamp-2 leading-relaxed">{source.text}</p>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                    <div ref={messagesEndRef} />
                </div>
            </div>

            {/* Input Area - Fixed at bottom */}
            <div className="p-4 bg-background border-t border-border shrink-0">
                <form onSubmit={handleSubmit} className="max-w-3xl mx-auto relative group">
                    <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder={selectedDocId ? "Ask a question about the active document..." : "Search across all documents or upload a new one..."}
                        className="w-full bg-card border border-border rounded-xl px-4 py-4 pr-32 resize-none h-16 focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary text-sm shadow-sm transition-all focus:shadow-md disabled:opacity-50"
                        disabled={isLoading && !input}
                    />
                    <div className="absolute right-2 top-2 flex items-center gap-2">
                        {!selectedDocId && messages.length === 0 && (
                            <button
                                type="button"
                                onClick={onNewUploadClick}
                                className="p-2 text-muted-foreground hover:text-foreground transition-colors hover:bg-muted rounded-lg"
                                title="Upload Document"
                            >
                                <Upload className="w-5 h-5" />
                            </button>
                        )}

                        {isLoading ? (
                            <button
                                type="button"
                                onClick={stopGeneration}
                                className="p-2 bg-red-500/20 text-red-500 rounded-lg hover:bg-red-500/30 transition-colors shadow-sm border border-red-500/30 font-medium text-xs flex items-center gap-1.5 px-3"
                            >
                                <StopCircle className="w-4 h-4" /> Stop
                            </button>
                        ) : (
                            <button
                                type="submit"
                                disabled={!input.trim()}
                                className="p-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                <Send className="w-5 h-5" />
                            </button>
                        )}
                    </div>
                </form>
                <p className="text-center text-[10px] text-muted-foreground mt-3">
                    SciChat can make mistakes. Verify important information with the original papers.
                </p>
            </div>
        </>
    );
}
