'use client';

import { useState, useRef } from 'react';
import { UploadCloud, File, X, CheckCircle2, Loader2, AlertCircle } from 'lucide-react';
import { API_BASE_URL } from '@/lib/api';

export default function PdfUploader({ onUploadSuccess }: { onUploadSuccess?: (doc: any) => void }) {
    const [isDragging, setIsDragging] = useState(false);
    const [file, setFile] = useState<File | null>(null);
    const [status, setStatus] = useState<'idle' | 'uploading' | 'processing' | 'success' | 'error'>('idle');
    const [errorMsg, setErrorMsg] = useState('');
    const fileInputRef = useRef<HTMLInputElement>(null);
    const statusRef = useRef(status);
    statusRef.current = status;

    const handleFile = (selectedFile: File) => {
        if (selectedFile.type !== 'application/pdf') {
            setErrorMsg('Please upload a PDF file.');
            setStatus('error');
            return;
        }
        setFile(selectedFile);
        setStatus('idle');
        setErrorMsg('');
    };

    const onDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    };

    const handleUpload = async () => {
        if (!file) return;
        setStatus('uploading');

        try {
            const formData = new FormData();
            formData.append('file', file);

            // Simulate step change for UX
            setTimeout(() => { if (statusRef.current === 'uploading') setStatus('processing') }, 1000);

            const res = await fetch(`${API_BASE_URL}/ingest`, {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || 'Upload failed');
            const data = await res.json();

            setStatus('success');
            if (onUploadSuccess) onUploadSuccess(data);

        } catch (err: any) {
            setStatus('error');
            setErrorMsg(err.message || 'Something went wrong');
        }
    };

    return (
        <div className="w-full max-w-md mx-auto">
            <div
                className={`relative border-2 border-dashed rounded-xl p-8 transition-colors text-center
          ${isDragging ? 'border-primary bg-primary/5' : 'border-border bg-card'}
          ${status === 'success' ? 'border-green-500/50 bg-green-500/5' : ''}
          ${status === 'error' ? 'border-red-500/50 bg-red-500/5' : ''}
        `}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
                onDrop={onDrop}
            >
                <input
                    type="file"
                    accept=".pdf"
                    className="hidden"
                    ref={fileInputRef}
                    onChange={(e) => e.target.files && handleFile(e.target.files[0])}
                />

                {status === 'idle' || status === 'error' ? (
                    <div className="flex flex-col items-center gap-3">
                        <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mb-2">
                            <UploadCloud className="w-6 h-6 text-primary" />
                        </div>
                        <h3 className="font-medium text-foreground">
                            {file ? file.name : 'Upload Scientific Paper'}
                        </h3>
                        <p className="text-sm text-muted-foreground">
                            {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : 'Drag and drop your PDF here, or click to browse'}
                        </p>

                        {status === 'error' && (
                            <div className="flex items-center gap-1.5 text-xs font-medium text-red-500 mt-2 bg-red-500/10 px-3 py-1.5 rounded-md">
                                <AlertCircle className="w-3.5 h-3.5" />
                                {errorMsg}
                            </div>
                        )}

                        <div className="flex gap-2 mt-4">
                            <button
                                onClick={() => fileInputRef.current?.click()}
                                className="px-4 py-2 text-sm font-medium bg-muted text-foreground rounded-lg hover:bg-muted/80 transition-colors"
                            >
                                {file ? 'Change File' : 'Browse Files'}
                            </button>
                            {file && (
                                <button
                                    onClick={handleUpload}
                                    className="px-4 py-2 text-sm font-medium bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors shadow-sm"
                                >
                                    Upload & Analyze
                                </button>
                            )}
                        </div>
                    </div>
                ) : status === 'uploading' || status === 'processing' ? (
                    <div className="flex flex-col items-center gap-4 py-4">
                        <Loader2 className="w-8 h-8 text-primary animate-spin" />
                        <div className="space-y-1">
                            <h3 className="font-medium text-foreground">
                                {status === 'uploading' ? 'Uploading PDF...' : 'Analyzing Document...'}
                            </h3>
                            <p className="text-sm text-muted-foreground">
                                {status === 'processing' ? 'Extracting text, figures, and mapping structure' : 'Transferring file securely'}
                            </p>
                        </div>
                        {/* Progress bar simulation */}
                        <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden mt-2">
                            <div className="h-full bg-primary rounded-full transition-all duration-1000 ease-in-out" style={{ width: status === 'processing' ? '85%' : '40%' }} />
                        </div>
                    </div>
                ) : (
                    <div className="flex flex-col items-center gap-3 py-4">
                        <div className="w-12 h-12 rounded-full bg-green-500/10 flex items-center justify-center mb-2">
                            <CheckCircle2 className="w-6 h-6 text-green-500" />
                        </div>
                        <h3 className="font-medium text-foreground">Analysis Complete</h3>
                        <p className="text-sm text-muted-foreground">The document is now ready for querying.</p>
                        <button
                            onClick={() => { setFile(null); setStatus('idle'); }}
                            className="mt-4 px-4 py-2 text-sm font-medium border border-border text-foreground hover:bg-muted rounded-lg transition-colors"
                        >
                            Upload Another
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}
