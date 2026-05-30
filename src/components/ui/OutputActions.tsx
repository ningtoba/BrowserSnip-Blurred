import { useProcessStore } from '@/stores/process-store';
import { downloadBlob } from '@/lib/utils/download';
import { useFileStore } from '@/stores/file-store';

export function OutputActions() {
  const outputBlob = useProcessStore((s) => s.outputBlob);
  const outputUrl = useProcessStore((s) => s.outputUrl);
  const file = useFileStore((s) => s.file);

  if (!outputBlob || !outputUrl) return null;

  const sizeMB = (outputBlob.size / (1024 * 1024)).toFixed(1);
  const originalName = file?.name ?? 'video.mp4';
  const outName = originalName.replace(/\.\w+$/, '_blurred.mp4');

  return (
    <div className="space-y-4 animate-doodle-pop">
      <div className="doodle-section space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-ink">Blurred Video Ready</h3>
            <p className="text-[11px] text-ink-muted">
              {sizeMB} MB — same resolution and format as original
            </p>
          </div>
        </div>

        <div className="sketch-border">
          <video
            src={outputUrl}
            controls
            className="w-full max-h-[300px] object-contain bg-black"
          />
        </div>

        <button
          onClick={() => downloadBlob(outputBlob, outName)}
          className="doodle-btn"
        >
          Download {outName}
        </button>
      </div>
    </div>
  );
}
