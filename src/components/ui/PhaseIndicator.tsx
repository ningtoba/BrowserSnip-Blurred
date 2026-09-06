import { Fragment } from 'react';
import { motion } from 'framer-motion';
import { useProcessStore } from '@/stores/process-store';
import type { PipelinePhase } from '@/types';
import { springSoft } from '@/lib/motion';

/** Ordered pipeline steps shown in the horizontal stepper (§2E). */
const STEPS: { phase: PipelinePhase; label: string }[] = [
  { phase: 'extracting-frames', label: 'Extract' },
  { phase: 'detecting-faces', label: 'Detect' },
  { phase: 'recognizing-faces', label: 'Identify' },
  { phase: 'clustering', label: 'Group' },
  { phase: 'processing-frames', label: 'Blur' },
  { phase: 'reconstructing', label: 'Encode' },
];

function stepIndexOf(phase: PipelinePhase): number {
  if (phase === 'loading-models') return 0;
  const idx = STEPS.findIndex((s) => s.phase === phase);
  return idx; // -1 when phase is waiting-selection/done/idle (handled below)
}

export function PhaseIndicator() {
  const phase = useProcessStore((s) => s.phase);
  const progress = useProcessStore((s) => s.progress);

  const raw = stepIndexOf(phase);
  // waiting-selection / done: everything reached so far is complete
  const activeIndex = raw >= 0 ? raw : phase === 'waiting-selection' || phase === 'done' ? STEPS.length : 0;

  return (
    <div className="rounded-[14px] p-4 bg-cream-light border border-cream-border shadow-doodle">
      {/* horizontal stepper with accent fill progression */}
      <div className="flex items-start">
        {STEPS.map((step, i) => {
          const done = i < activeIndex;
          const active = i === activeIndex && phase !== 'done' && phase !== 'waiting-selection';
          return (
            <Fragment key={step.phase}>
              {i > 0 && (
                <div className="flex-1 h-[2px] mt-[7px] rounded-full bg-cream-border overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ background: '#2563EB' }}
                    initial={false}
                    animate={{ width: done || active ? '100%' : '0%' }}
                    transition={springSoft}
                  />
                </div>
              )}
              <div className="flex flex-col items-center w-11 shrink-0 -mt-[5px]">
                <span
                  className={`relative w-4 h-4 rounded-full border-2 flex items-center justify-center bg-cream-light transition-colors duration-200 ${
                    done
                      ? 'border-accent'
                      : active
                        ? 'border-accent'
                        : 'border-cream-border'
                  }`}
                >
                  {done ? (
                    <svg className="w-2 h-2 text-accent" fill="none" stroke="currentColor" strokeWidth={3.5} viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  ) : active ? (
                    /* animated active dot */
                    <motion.span
                      className="w-1.5 h-1.5 rounded-full"
                      style={{ background: '#2563EB' }}
                      animate={{ scale: [1, 1.5, 1], opacity: [1, 0.55, 1] }}
                      transition={{ duration: 1.4, repeat: Infinity, ease: 'easeInOut' }}
                    />
                  ) : null}
                </span>
                <span
                  className={`mt-1.5 text-[9px] font-medium uppercase tracking-wide whitespace-nowrap ${
                    active ? 'text-accent' : done ? 'text-ink-soft' : 'text-ink-muted'
                  }`}
                >
                  {step.label}
                </span>
              </div>
            </Fragment>
          );
        })}
      </div>

      <div className="mt-2.5 flex items-center gap-2 min-w-0">
        <p className="text-xs font-medium text-ink-soft truncate">
          {progress.phaseDescription ||
            (phase === 'waiting-selection'
              ? 'Select faces to blur'
              : phase === 'done'
                ? 'Done'
                : STEPS[activeIndex]?.label)}
        </p>
        {progress.detail && (
          <p className="text-[10px] text-ink-muted truncate font-mono">{progress.detail}</p>
        )}
      </div>
    </div>
  );
}
