import { useState, useRef, useCallback } from 'react';
import { API_BASE_URL } from '@/lib/api';

export interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    sources?: any[];
    isStreaming?: boolean;
    latencyMs?: number;
    cached?: boolean;
}

interface UseStreamingQueryOptions {
    messages: Message[];
    setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
    documentId?: string | null;
}

export function useStreamingQuery({ messages, setMessages, documentId }: UseStreamingQueryOptions) {
    const [isLoading, setIsLoading] = useState(false);
    const abortControllerRef = useRef<AbortController | null>(null);

    const sendMessage = useCallback(async (question: string) => {
        if (!question.trim()) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: question,
        };

        const assistantMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: 'assistant',
            content: '',
            isStreaming: true,
        };

        setMessages(prev => [...prev, userMessage, assistantMessage]);
        setIsLoading(true);

        const abortController = new AbortController();
        abortControllerRef.current = abortController;

        try {
            const body: any = {
                question,
                stream: true,
            };
            if (documentId) {
                body.document_id = documentId;
            }

            const response = await fetch(`${API_BASE_URL}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
                signal: abortController.signal,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body?.getReader();
            if (!reader) throw new Error('Response body is not readable');

            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed || !trimmed.startsWith('data: ')) continue;

                    try {
                        const data = JSON.parse(trimmed.slice(6));

                        if (data.type === 'answer') {
                            setMessages(prev =>
                                prev.map(m =>
                                    m.id === assistantMessage.id
                                        ? { ...m, content: data.content }
                                        : m
                                )
                            );
                        } else if (data.type === 'sources') {
                            setMessages(prev =>
                                prev.map(m =>
                                    m.id === assistantMessage.id
                                        ? { ...m, sources: data.content }
                                        : m
                                )
                            );
                        } else if (data.type === 'meta') {
                            setMessages(prev =>
                                prev.map(m =>
                                    m.id === assistantMessage.id
                                        ? { ...m, latencyMs: data.content.latency_ms, cached: data.content.cached }
                                        : m
                                )
                            );
                        } else if (data.type === 'done') {
                            setMessages(prev =>
                                prev.map(m =>
                                    m.id === assistantMessage.id
                                        ? { ...m, isStreaming: false }
                                        : m
                                )
                            );
                        } else if (data.type === 'error') {
                            setMessages(prev =>
                                prev.map(m =>
                                    m.id === assistantMessage.id
                                        ? { ...m, content: `Error: ${data.content}`, isStreaming: false }
                                        : m
                                )
                            );
                        }
                    } catch {
                        // Skip unparseable lines
                    }
                }
            }

            // Ensure streaming is marked complete
            setMessages(prev =>
                prev.map(m =>
                    m.id === assistantMessage.id
                        ? { ...m, isStreaming: false }
                        : m
                )
            );
        } catch (err: any) {
            if (err.name !== 'AbortError') {
                setMessages(prev =>
                    prev.map(m =>
                        m.id === assistantMessage.id
                            ? { ...m, content: `Error: ${err.message}`, isStreaming: false }
                            : m
                    )
                );
            }
        } finally {
            setIsLoading(false);
            abortControllerRef.current = null;
        }
    }, [setMessages, documentId]);

    const stopGeneration = useCallback(() => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
        }
    }, []);

    return { messages, sendMessage, isLoading, stopGeneration, setMessages };
}
