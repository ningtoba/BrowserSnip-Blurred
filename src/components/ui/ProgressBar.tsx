import { motion } from 'framer-motion';
import { useProcessStore } from '@/stores/process-store';
import { springSoft, fadeUp } from '@/lib/motion';

export function ProgressBar() {
  const progress = useProcessStore((s) => s.progress);

  return (
    <motion.div
      variants={fadeUp}
      initial="hidden"
      animate="visible"
      className="rounded-[14px] p-4 bg-cream-light border border-cream-border shadow-doodle space-y-3"
    >
      <div className="flex items-center justify-between text-xs">
        <span className="text-ink-soft font-medium">
          {progress.phaseDescription || progress.phase.replace(/-/g, ' ')}
        </span>
        <span className="text-ink font-mono text-sm font-semibold tabular-nums">
          {Math.round(progress.overallPercent)}
          <span className="text-[10px] text-ink-muted font-normal ml-0.5">%</span>
        </span>
      </div>

      <div className="w-full h-1.5 rounded-full bg-cream-border overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ background: '#2563EB' }}
          initial={{ width: 0 }}
          animate={{ width: `${progress.overallPercent}%` }}
          transition={springSoft}
        />
      </div>

      {progress.detail && (
        <p className="text-[10px] text-ink-muted font-mono">{progress.detail}</p>
      )}
    </motion.div>
  );
}
