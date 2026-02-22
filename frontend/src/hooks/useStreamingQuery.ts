import { useState, useRef, useCallback } from 'react';

export interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    sources?: any[];
    isStreaming?: boolean;
}

export function useStreamingQuery() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const abortControllerRef = useRef<AbortController | null>(null);

    const sendMessage = useCallback(async (question: string, documentId?: string) => {
        if (!question.trim()) return;

        // Add user message immediately
        const userMessage: Message = { id: Date.now().toString(), role: 'user', content: question };
        const assistantMessageId = (Date.now() + 1).toString();

        setMessages((prev) => [
            ...prev,
            userMessage,
            { id: assistantMessageId, role: 'assistant', content: '', isStreaming: true },
        ]);

        setIsLoading(true);

        try {
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
            abortControllerRef.current = new AbortController();

            const response = await fetch('http://localhost:8000/api/v1/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream',
                },
                body: JSON.stringify({ question, document_id: documentId, stream: true }),
                signal: abortControllerRef.current.signal,
            });

            if (!response.ok) throw new Error('Query request failed');
            const reader = response.body?.getReader();
            const decoder = new TextDecoder();

            if (!reader) throw new Error('No reader available');

            let answerContent = '';
            let sources: any[] = [];
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                // Keep the last potentially incomplete line in the buffer
                buffer = lines.pop() || '';

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed.startsWith('data: ')) continue;
                    const dataStr = trimmed.slice(6); // Remove 'data: ' prefix

                    try {
                        const data = JSON.parse(dataStr);

                        if (data.type === 'answer') {
                            // Backend sends the full answer in one event
                            answerContent = data.content;
                            setMessages(prev => prev.map(msg =>
                                msg.id === assistantMessageId
                                    ? { ...msg, content: answerContent }
                                    : msg
                            ));
                        } else if (data.type === 'sources') {
                            // Backend sends sources as a separate event
                            sources = data.content || [];
                        } else if (data.type === 'done') {
                            // Stream finished
                        } else if (data.type === 'error') {
                            answerContent = `Error: ${data.content}`;
                            setMessages(prev => prev.map(msg =>
                                msg.id === assistantMessageId
                                    ? { ...msg, content: answerContent }
                                    : msg
                            ));
                        }
                    } catch (e) {
                        // Ignore unparseable lines
                    }
                }
            }

            // Finalize the message
            setMessages(prev => prev.map(msg =>
                msg.id === assistantMessageId
                    ? { ...msg, content: answerContent, sources, isStreaming: false }
                    : msg
            ));

        } catch (err: any) {
            if (err.name !== 'AbortError') {
                console.error("Streaming error:", err);
                setMessages(prev => prev.map(msg =>
                    msg.id === assistantMessageId
                        ? { ...msg, content: 'Sorry, I encountered an error while processing your request.', isStreaming: false }
                        : msg
                ));
            }
        } finally {
            setIsLoading(false);
            abortControllerRef.current = null;
        }
    }, []);

    const stopGeneration = useCallback(() => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            setIsLoading(false);
            setMessages(prev => prev.map(msg =>
                msg.isStreaming ? { ...msg, isStreaming: false } : msg
            ));
        }
    }, []);

    return { messages, sendMessage, isLoading, stopGeneration, setMessages };
}
