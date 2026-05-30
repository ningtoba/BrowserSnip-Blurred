import { useProcessStore } from '@/stores/process-store';

export function ProgressBar() {
  const progress = useProcessStore((s) => s.progress);

  return (
    <div className="doodle-section space-y-3 animate-fade-in">
      <div className="flex items-center justify-between text-xs">
        <span className="text-ink-soft font-medium">
          {progress.phaseDescription || progress.phase.replace(/-/g, ' ')}
        </span>
        <span className="text-ink-muted font-mono tabular-nums">
          {Math.round(progress.overallPercent)}%
        </span>
      </div>

      <div className="w-full h-1.5 rounded-full bg-cream overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-300 ease-out"
          style={{
            width: `${progress.overallPercent}%`,
            background: 'linear-gradient(90deg, #6366f1, #818cf8)',
            boxShadow: '0 0 10px rgba(99, 102, 241, 0.3)',
          }}
        />
      </div>

      {progress.detail && (
        <p className="text-[10px] text-ink-muted font-mono">{progress.detail}</p>
      )}
    </div>
  );
}
