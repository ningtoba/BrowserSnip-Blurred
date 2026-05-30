import { useMemo } from 'react';
import { useFileStore } from '@/stores/file-store';

export function VideoPreview() {
  const file = useFileStore((s) => s.file);
  const url = useMemo(
    () => (file ? URL.createObjectURL(file) : null),
    [file]
  );

  if (!file || !url) return null;

  return (
    <div className="animate-slide-up space-y-2">
      <div className="sketch-border">
        <video
          src={url}
          controls
          className="w-full max-h-[300px] object-contain bg-black"
          preload="metadata"
        />
      </div>
      <p className="text-[11px] text-ink-muted truncate">
        {file.name} ({(file.size / (1024 * 1024)).toFixed(1)} MB)
      </p>
    </div>
  );
}
