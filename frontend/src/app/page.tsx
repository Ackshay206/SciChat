'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import Sidebar from "@/components/Sidebar";
import ChatInterface from "@/components/ChatInterface";
import PdfUploader from '@/components/PdfUploader';
import { Message } from '@/hooks/useStreamingQuery';
import { API_BASE_URL } from '@/lib/api';

export default function Home() {
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [showUploader, setShowUploader] = useState(false);
  const [backendReady, setBackendReady] = useState(true);

  // Free-tier backend sleeps when idle: poll health until models are loaded, then reload the paper list
  useEffect(() => {
    let cancelled = false;
    let wasAsleep = false;
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if ((await res.json()).models_loaded) {
          if (!cancelled) {
            setBackendReady(true);
            if (wasAsleep) window.dispatchEvent(new Event('refresh-docs'));
          }
          return;
        }
      } catch { }
      wasAsleep = true;
      if (!cancelled) {
        setBackendReady(false);
        setTimeout(check, 5000);
      }
    };
    check();
    return () => { cancelled = true; };
  }, []);

  // Per-paper chat history: Map<docId, Message[]>
  const chatHistoryRef = useRef<Map<string, Message[]>>(new Map());

  // Messages for the currently selected document
  const [messages, setMessages] = useState<Message[]>([]);

  // Sync messages with the chat history map
  const selectedDocIdRef = useRef<string | null>(null);

  useEffect(() => {
    // Save current messages before switching
    if (selectedDocIdRef.current) {
      chatHistoryRef.current.set(selectedDocIdRef.current, messages);
    }
  }, [messages]);

  const handleDocumentSelect = useCallback((id: string | null) => {
    // Save current doc's messages
    if (selectedDocIdRef.current) {
      chatHistoryRef.current.set(selectedDocIdRef.current, messages);
    }

    setSelectedDocId(id);
    selectedDocIdRef.current = id;
    setShowUploader(false);

    // Load the new doc's messages
    if (id) {
      const savedMessages = chatHistoryRef.current.get(id) || [];
      setMessages(savedMessages);
    } else {
      setMessages([]);
    }
  }, [messages]);

  const handleNewUploadClick = () => {
    setSelectedDocId(null);
    selectedDocIdRef.current = null;
    setShowUploader(true);
  };

  const handleUploadSuccess = () => {
    window.dispatchEvent(new Event('refresh-docs'));
    setShowUploader(false);
  };

  return (
    <main className="flex flex-col md:flex-row h-screen bg-background text-foreground font-sans antialiased selection:bg-primary/30">

      {/* Left Sidebar - Fixed width */}
      <div className="w-full md:w-72 max-h-[45vh] md:max-h-none border-b md:border-b-0 md:border-r border-border bg-card/30 flex flex-col shrink-0 z-20">
        <Sidebar
          selectedDocId={selectedDocId}
          onSelectDoc={handleDocumentSelect}
          onNewUploadClick={handleNewUploadClick}
        />
      </div>

      {/* Main Area - Flexible width */}
      <div className="flex-1 flex flex-col min-h-0 md:h-full bg-background relative overflow-hidden">

        {!backendReady && (
          <div className="px-6 py-2 text-sm text-center bg-amber-400/10 text-amber-200 border-b border-amber-400/20 z-20">
            The backend is waking up (free hosting sleeps when idle). This takes about a minute…
          </div>
        )}

        {showUploader ? (
          <div className="flex-1 flex flex-col items-center justify-center p-8 z-10">
            <div className="max-w-xl w-full text-center space-y-6 mb-8">
              <h2 className="text-2xl font-semibold tracking-tight">Upload a paper</h2>
              <p className="text-muted-foreground text-sm">
                SciChat uses content-aware chunking and Vision AI to understand layout, extract tables, and describe figures before embedding the text into Pinecone.
              </p>
            </div>
            <PdfUploader onUploadSuccess={handleUploadSuccess} />
          </div>
        ) : (
          <ChatInterface
            selectedDocId={selectedDocId}
            onNewUploadClick={handleNewUploadClick}
            messages={messages}
            setMessages={setMessages}
          />
        )}
      </div>
    </main>
  );
}
