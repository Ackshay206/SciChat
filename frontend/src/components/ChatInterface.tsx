'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, StopCircle, FileText, Upload, Clock, Zap } from 'lucide-react';
import { useStreamingQuery, Message } from '@/hooks/useStreamingQuery';
import { marked } from 'marked';
import markedKatex from 'marked-katex-extension';
import 'katex/dist/katex.min.css';

// Answers quote the paper's math in LaTeX ($d_{model}$); render it instead of showing raw TeX
marked.use(markedKatex({ throwOnError: false, nonStandard: true }));

interface ChatInterfaceProps {
    selectedDocId: string | null;
    onNewUploadClick: () => void;
    messages: Message[];
    setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}

const CITATION = /\[(\d+)\]/g;

// Answers cite passages as [n]; the backend prefixes each passage with "Source n:".
interface Source {
    text: string;
    metadata?: { section_name?: string; page?: number; content_type?: string };
}

function parseSource(source: Source, index: number) {
    const match = source.text?.match(/^Source (\d+):\s*/);
    return {
        number: match ? Number(match[1]) : index + 1,
        text: (match ? source.text.slice(match[0].length) : source.text).replace(/^#+\s*/gm, ''),
        meta: source.metadata || {},
    };
}

function citedNumbers(answer: string) {
    return new Set(Array.from(answer.matchAll(CITATION), m => Number(m[1])));
}

function formatLatency(ms: number) {
    return ms < 1000 ? `${(ms / 1000).toFixed(2)} s` : `${(ms / 1000).toFixed(1)} s`;
}

export default function ChatInterface({ selectedDocId, onNewUploadClick, messages, setMessages }: ChatInterfaceProps) {
    const [input, setInput] = useState('');
    const { sendMessage, isLoading, stopGeneration } = useStreamingQuery({
        messages,
        setMessages,
        documentId: selectedDocId,
    });
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to latest message
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;
        sendMessage(input);
        setInput('');
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    // Markdown, with [n] turned into buttons that jump to passage n
    const renderAnswer = (text: string) => {
        let html: string;
        try {
            html = marked.parse(text) as string;
        } catch {
            html = text;
        }
        return {
            __html: html.replace(CITATION, (_, n) =>
                `<button type="button" class="cite" data-cite="${n}" aria-label="Show source ${n}">${n}</button>`
            ),
        };
    };

    const handleCitationClick = (e: React.MouseEvent, messageId: string) => {
        const n = (e.target as HTMLElement).closest('[data-cite]')?.getAttribute('data-cite');
        if (!n) return;
        const card = document.getElementById(`src-${messageId}-${n}`);
        if (!card) return;
        const details = card.closest('details');
        if (details) details.open = true;
        card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        card.classList.remove('source-flash');
        void card.offsetWidth;
        card.classList.add('source-flash');
    };

    const renderSourceCard = (source: ReturnType<typeof parseSource>, messageId: string, cited: boolean) => (
        <div
            key={source.number}
            id={`src-${messageId}-${source.number}`}
            className={`flex gap-3 text-sm rounded-md p-3 border ${cited ? 'border-primary/25 bg-highlight' : 'border-border bg-card/60'}`}
        >
            <span className={`shrink-0 w-6 h-6 rounded text-xs font-semibold flex items-center justify-center tabular-nums
                ${cited ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'}`}>
                {source.number}
            </span>
            <div className="min-w-0 space-y-1">
                <p className="text-xs text-muted-foreground">
                    {source.meta.section_name || (source.meta.page ? `Page ${source.meta.page}` : 'Passage')}
                    {source.meta.content_type && source.meta.content_type !== 'text' && (
                        <span className="ml-2 px-1.5 py-0.5 rounded border border-border">{source.meta.content_type}</span>
                    )}
                </p>
                <p className="font-serif text-foreground/85 leading-relaxed line-clamp-3">{source.text}</p>
            </div>
        </div>
    );

    return (
        <>
            {/* Header */}
            <div className="h-16 px-6 border-b border-border flex items-center justify-between shrink-0 bg-background/80 backdrop-blur-sm z-10 sticky top-0">
                <h2 className="text-base font-medium text-foreground">Research assistant</h2>
                {selectedDocId && (
                    <span className="text-xs font-medium px-2.5 py-1 bg-accent/10 text-accent rounded-full border border-accent/20 flex items-center gap-1.5">
                        <FileText className="w-3.5 h-3.5" />
                        Paper selected
                    </span>
                )}
            </div>

            {/* Messages Area - Scrollable */}
            <div className="flex-1 overflow-y-auto px-6 py-8 scroll-smooth">
                <div className="max-w-3xl mx-auto space-y-10 pb-20">

                    {messages.length === 0 && (
                        <div className="answer text-foreground/90 space-y-3">
                            {selectedDocId ? (
                                <>
                                    <p>Ask anything about the selected paper.</p>
                                    <p className="text-muted-foreground">
                                        Each claim in the answer is marked with a <span className="cite" aria-hidden="true">1</span> that
                                        opens the passage it came from. If the paper doesn&apos;t cover your question, you&apos;ll be told so
                                        instead of getting a guess.
                                    </p>
                                </>
                            ) : (
                                <p>Pick a paper from the sidebar, or upload one, to start asking questions.</p>
                            )}
                        </div>
                    )}

                    {messages.map((msg) => {
                        if (msg.role === 'user') {
                            return (
                                <h3 key={msg.id} className="text-lg font-medium text-foreground leading-snug border-l-2 border-accent pl-4">
                                    {msg.content}
                                </h3>
                            );
                        }

                        const sources = (msg.sources || []).map(parseSource);
                        const cited = citedNumbers(msg.content || '');
                        const citedSources = sources.filter(s => cited.has(s.number));
                        const otherSources = sources.filter(s => !cited.has(s.number));

                        return (
                            <div key={msg.id} className="space-y-4">
                                {msg.content ? (
                                    <div
                                        className="answer text-foreground"
                                        onClick={(e) => handleCitationClick(e, msg.id)}
                                        dangerouslySetInnerHTML={renderAnswer(msg.content)}
                                    />
                                ) : (
                                    <p className="flex items-center gap-2 text-sm text-muted-foreground">
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        Searching the paper and writing an answer…
                                    </p>
                                )}

                                {!msg.isStreaming && msg.latencyMs !== undefined && (
                                    <p className="flex items-center gap-1.5 text-xs text-muted-foreground tabular-nums">
                                        {msg.cached ? <Zap className="w-3.5 h-3.5 text-primary" /> : <Clock className="w-3.5 h-3.5" />}
                                        {msg.cached
                                            ? `Served from cache in ${formatLatency(msg.latencyMs)}`
                                            : `Answered in ${formatLatency(msg.latencyMs)}`}
                                        {sources.length > 0 && ` using ${citedSources.length} of ${sources.length} retrieved passages`}
                                    </p>
                                )}

                                {citedSources.length > 0 && (
                                    <div className="space-y-2">
                                        <p className="text-sm font-medium text-muted-foreground">Cited passages</p>
                                        {citedSources.map(s => renderSourceCard(s, msg.id, true))}
                                    </div>
                                )}

                                {otherSources.length > 0 && (
                                    <details className="group">
                                        <summary className="text-sm text-muted-foreground cursor-pointer hover:text-foreground select-none">
                                            Other retrieved passages ({otherSources.length})
                                        </summary>
                                        <div className="space-y-2 mt-2">
                                            {otherSources.map(s => renderSourceCard(s, msg.id, false))}
                                        </div>
                                    </details>
                                )}
                            </div>
                        );
                    })}
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
                        aria-label="Question"
                        placeholder={selectedDocId ? "Ask about this paper, e.g. what BLEU score did the big model reach?" : "Select a paper to start asking questions"}
                        className="w-full bg-card border border-border rounded-xl px-4 py-4 pr-32 resize-none h-16 focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary text-sm transition-colors disabled:opacity-50"
                        disabled={isLoading && !input}
                    />
                    <div className="absolute right-2 top-2 flex items-center gap-2">
                        {!selectedDocId && messages.length === 0 && (
                            <button
                                type="button"
                                onClick={onNewUploadClick}
                                className="p-2 text-muted-foreground hover:text-foreground transition-colors hover:bg-muted rounded-lg"
                                title="Upload a paper"
                            >
                                <Upload className="w-5 h-5" />
                            </button>
                        )}

                        {isLoading ? (
                            <button
                                type="button"
                                onClick={stopGeneration}
                                className="p-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors border border-red-500/30 font-medium text-xs flex items-center gap-1.5 px-3"
                            >
                                <StopCircle className="w-4 h-4" /> Stop
                            </button>
                        ) : (
                            <button
                                type="submit"
                                disabled={!input.trim()}
                                aria-label="Ask"
                                className="p-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                            >
                                <Send className="w-5 h-5" />
                            </button>
                        )}
                    </div>
                </form>
                <p className="text-center text-xs text-muted-foreground mt-3">
                    Answers come only from the selected paper. Check the cited passages before relying on them.
                </p>
            </div>
        </>
    );
}
