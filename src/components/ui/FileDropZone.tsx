import { useCallback, useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { useFileStore } from '@/stores/file-store';
import { useProcessStore } from '@/stores/process-store';
import { usePipeline } from '@/hooks/usePipeline';
import { getVideoMetadata } from '@/lib/video/extract';
import type { VideoMetadata } from '@/types';
import { springSoft, floatLoop } from '@/lib/motion';

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
    <div className="space-y-4">
      <motion.div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        animate={{
          scale: dragOver ? 1.01 : 1,
          backgroundColor: dragOver ? 'rgba(37,99,235,0.04)' : '#FFFFFF',
          borderColor: dragOver ? '#2563EB' : '#B9B9AF',
        }}
        transition={springSoft}
        className="relative rounded-[20px] border-[1.5px] border-dashed p-10 sm:p-12 text-center cursor-pointer select-none shadow-doodle"
      >
        <div className="space-y-4">
          <motion.div
            className="w-14 h-14 mx-auto rounded-[16px] flex items-center justify-center"
            style={{ background: 'rgba(37,99,235,0.08)', color: '#2563EB' }}
            animate={dragOver ? { y: [0, -6, 0] } : { y: [0, -4, 0] }}
            transition={floatLoop.transition}
          >
            <svg
              className="w-7 h-7"
              fill="none"
              stroke="currentColor"
              strokeWidth={1.5}
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
              />
            </svg>
          </motion.div>
          <div>
            <p className="text-sm font-medium text-ink">
              {scanning ? 'Scanning faces…' : 'Drop a video, or click to browse'}
            </p>
            <p className="text-[11px] text-ink-muted mt-1.5">
              MP4, WebM, MOV — processed on your machine, never uploaded
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
      </motion.div>

      {error && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-[10px] p-3 text-xs border border-danger/20 bg-danger/5 text-danger"
        >
          {error}
        </motion.div>
      )}
    </div>
  );
}
