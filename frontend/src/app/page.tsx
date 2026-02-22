'use client';

import { useState } from 'react';
import Sidebar from "@/components/Sidebar";
import ChatInterface from "@/components/ChatInterface";
import PdfUploader from '@/components/PdfUploader';

export default function Home() {
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [showUploader, setShowUploader] = useState(false);

  const handleDocumentSelect = (id: string | null) => {
    setSelectedDocId(id);
    setShowUploader(false);
  };

  const handleNewUploadClick = () => {
    setSelectedDocId(null);
    setShowUploader(true);
  };

  const handleUploadSuccess = () => {
    // Optionally auto-select the new doc here, but for now just refresh sidebar
    window.dispatchEvent(new Event('refresh-docs'));
    setShowUploader(false);
  };

  return (
    <main className="flex h-screen bg-background text-foreground font-sans antialiased selection:bg-primary/30">

      {/* Left Sidebar - Fixed width */}
      <div className="w-72 border-r border-border bg-card/30 flex flex-col shrink-0 shadow-[4px_0_24px_-12px_rgba(0,0,0,0.5)] z-20">
        <Sidebar
          selectedDocId={selectedDocId}
          onSelectDoc={handleDocumentSelect}
          onNewUploadClick={handleNewUploadClick}
        />
      </div>

      {/* Main Area - Flexible width */}
      <div className="flex-1 flex flex-col h-full bg-background relative overflow-hidden">

        {/* Subtle background glow effect */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-4/5 h-64 bg-primary/5 blur-[120px] rounded-full pointer-events-none" />

        {showUploader ? (
          <div className="flex-1 flex flex-col items-center justify-center p-8 z-10">
            <div className="max-w-xl w-full text-center space-y-6 mb-8">
              <h2 className="text-2xl font-semibold tracking-tight">Upload Scientific Paper</h2>
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
          />
        )}
      </div>
    </main>
  );
}
