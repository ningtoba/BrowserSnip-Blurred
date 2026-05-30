import { useProcessStore } from '@/stores/process-store';
import { PHASE_DESCRIPTIONS } from '@/lib/constants';

export function PhaseIndicator() {
  const phase = useProcessStore((s) => s.phase);
  const progress = useProcessStore((s) => s.progress);

  const phaseLabel = PHASE_DESCRIPTIONS[phase] || phase;

  return (
    <div className="flex items-center gap-3 px-1">
      <div className="w-2 h-2 rounded-full bg-accent animate-pulse-glow shrink-0" />
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium text-ink-soft">{phaseLabel}</p>
        {progress.detail && (
          <p className="text-[10px] text-ink-muted truncate">{progress.detail}</p>
        )}
      </div>
    </div>
  );
}
