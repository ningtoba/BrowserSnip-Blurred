import { motion } from 'framer-motion';
import { fadeUp, staggerParent, springSoft } from '@/lib/motion';

interface Props {
  message: string;
  percent: number;
}

export function LoadingScreen({ message, percent }: Props) {
  return (
    <div className="h-screen-safe flex items-center justify-center bg-cream relative overflow-hidden">
      {/* soft drifting wash */}
      <motion.div
        aria-hidden
        className="absolute inset-0 pointer-events-none"
        animate={{ x: [0, 20, 0], y: [0, -12, 0] }}
        transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut' }}
        style={{
          background:
            'radial-gradient(40% 36% at 50% 40%, rgba(37,99,235,0.08) 0%, rgba(124,58,237,0.06) 60%, transparent 100%)',
        }}
      />
      <motion.div
        variants={staggerParent}
        initial="hidden"
        animate="visible"
        className="relative w-full max-w-md px-6 text-center space-y-6"
      >
        <motion.div variants={fadeUp} className="space-y-2">
          <div
            className="w-12 h-12 mx-auto rounded-[14px] flex items-center justify-center mb-4"
            style={{ background: 'rgba(37,99,235,0.08)', color: '#2563EB' }}
            aria-hidden
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth={1.7} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 10.5a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 1115 0z" />
            </svg>
          </div>
          <h1 className="font-display text-xl font-semibold text-ink tracking-[-0.02em]">
            Face Blur
          </h1>
          <p className="text-xs text-ink-muted">
            100% on-device AI face blurring
          </p>
        </motion.div>

        <motion.div variants={fadeUp} className="rounded-[14px] p-4 bg-cream-light border border-cream-border shadow-doodle space-y-3 text-left">
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-ink-soft font-medium truncate pr-3">{message}</span>
            <span className="text-ink-muted font-mono tabular-nums shrink-0">
              {Math.round(percent)}%
            </span>
          </div>
          <div className="w-full h-1.5 rounded-full bg-cream-border overflow-hidden">
            <motion.div
              className="h-full rounded-full"
              style={{ background: '#2563EB' }}
              initial={{ width: 0 }}
              animate={{ width: `${percent}%` }}
              transition={springSoft}
            />
          </div>
        </motion.div>

        <motion.p variants={fadeUp} className="text-[10px] text-ink-muted leading-relaxed">
          AI models are downloaded once and cached by your browser.
          <br />
          Subsequent visits will be instant.
        </motion.p>
      </motion.div>
    </div>
  );
}
