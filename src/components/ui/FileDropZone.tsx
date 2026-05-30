import { useCallback, useState, useRef } from 'react';
import { useFileStore } from '@/stores/file-store';
import { useProcessStore } from '@/stores/process-store';
import { usePipeline } from '@/hooks/usePipeline';
import { getVideoMetadata } from '@/lib/video/extract';
import type { VideoMetadata } from '@/types';

export function FileDropZone() {
  const setFile = useFileStore((s) => s.setFile);
  const { startScan } = usePipeline();
  const [dragOver, setDragOver] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File | null) => {
      if (!file) return;
      setError(null);

      if (!file.type.startsWith('video/')) {
        setError('Please upload a video file.');
        return;
      }

      setFile(file);
      setScanning(true);

      try {
        const meta = await getVideoMetadata(file);
        const metadata: VideoMetadata = {
          duration: meta.duration,
          width: meta.width,
          height: meta.height,
          fps: meta.fps,
          codec: 'h264',
          fileSize: file.size,
          fileName: file.name,
        };
        useFileStore.getState().setMetadata(metadata);
        await startScan(file);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to process video');
      } finally {
        setScanning(false);
      }
    },
    [setFile, startScan]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      handleFile(file);
    },
    [handleFile]
  );

  return (
    <div className="space-y-4 animate-fade-in">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`
          relative rounded-doodle-lg border-2 border-dashed p-10 text-center
          cursor-pointer transition-all duration-200 select-none
          ${
            dragOver
              ? 'border-accent bg-accent/5 scale-[1.01]'
              : 'border-cream-border hover:border-cream-border/60'
          }
        `}
      >
        <div className="space-y-3">
          <div className="w-12 h-12 mx-auto rounded-full bg-accent/10 flex items-center justify-center">
            <svg
              className="w-6 h-6 text-accent"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
              />
            </svg>
          </div>
          <div>
            <p className="text-sm font-medium text-ink-soft">
              {scanning ? 'Scanning faces...' : 'Drop a video to blur faces'}
            </p>
            <p className="text-[11px] text-ink-muted mt-1">
              MP4, WebM, MOV — up to any size
            </p>
          </div>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
        />
      </div>

      {error && (
        <div className="rounded-md p-3 text-xs border border-danger/20 bg-danger/5 text-danger animate-slide-up">
          {error}
        </div>
      )}
    </div>
  );
}
