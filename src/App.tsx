import { MotionConfig, motion } from 'framer-motion';
import { useFileStore } from '@/stores/file-store';
import { useProcessStore } from '@/stores/process-store';
import { useUIStore } from '@/stores/ui-store';
import { LoadingScreen } from '@/components/ui/LoadingScreen';

import { FileDropZone } from '@/components/ui/FileDropZone';
import { VideoPreview } from '@/components/player/VideoPreview';
import { ProgressBar } from '@/components/ui/ProgressBar';
import { PhaseIndicator } from '@/components/ui/PhaseIndicator';
import { FacePicker } from '@/components/face/FacePicker';
import { BlurTypeSelector } from '@/components/face/BlurTypeSelector';
import { OutputActions } from '@/components/ui/OutputActions';
import { MemoryWarning } from '@/components/ui/MemoryWarning';
import { LogMonitor } from '@/components/ui/LogMonitor';
import { useONNX } from '@/hooks/useONNX';
import { terminateFFmpeg } from '@/lib/ffmpeg/core';
import { fadeUp, scaleIn, staggerParent } from '@/lib/motion';

export default function App() {
  return (
    <MotionConfig reducedMotion="user">
      <FaceBlurWorkspace />
    </MotionConfig>
  );
}

function FaceBlurWorkspace() {
  const file = useFileStore((s) => s.file);
  const phase = useProcessStore((s) => s.phase);
  const identities = useProcessStore((s) => s.identities);
  const outputBlob = useProcessStore((s) => s.outputBlob);
  const isProcessing = useProcessStore((s) => s.isProcessing);
  const showLogMonitor = useUIStore((s) => s.showLogMonitor);
  const toggleLogMonitor = useUIStore((s) => s.toggleLogMonitor);
  const { gpuAccelerated, modelsReady, loadingMessage, loadingPercent } = useONNX();

  const handleReset = async () => {
    if (isProcessing) return;
    // Revoke any output URLs to free memory
    const outputUrl = useProcessStore.getState().outputUrl;
    if (outputUrl) URL.revokeObjectURL(outputUrl);
    // Terminate ffmpeg to free WASM memory
    try { await terminateFFmpeg(); } catch { /* ignore */ }
    // Reset all stores
    useProcessStore.getState().reset();
    useFileStore.getState().reset();
  };

  if (!modelsReady) {
    return (
      <LoadingScreen
        message={loadingMessage}
        percent={loadingPercent}
      />
    );
  }

  const showFacePicker =
    phase === 'waiting-selection' && identities.length > 0;
  const showOutput = phase === 'done' && outputBlob;
  const showProcessing =
    isProcessing || (phase !== 'idle' && phase !== 'waiting-selection' && phase !== 'done');

  return (
    <div className="flex flex-col h-screen-safe">
      {/* ── Tool sub-header (v3 workspace chrome) ─────────────────────── */}
      <header className="h-[52px] shrink-0 flex items-center justify-between px-4 sm:px-6 border-b border-cream-border bg-glass z-20">
        <div className="flex items-center gap-3 min-w-0">
          <a
            href="https://www.browsersnip.com"
            className="group flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink transition-colors"
            title="Back to BrowserSnip"
          >
            <svg
              className="w-3.5 h-3.5 transition-transform duration-200 ease-spring group-hover:-translate-x-0.5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            <span className="hidden sm:inline">BrowserSnip</span>
          </a>
          <span className="h-4 w-px bg-cream-border shrink-0" aria-hidden />
          <span
            className="w-7 h-7 shrink-0 rounded-[10px] flex items-center justify-center"
            style={{ background: 'rgba(37,99,235,0.08)', color: '#2563EB' }}
            aria-hidden
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={1.7} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 10.5a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 1115 0z" />
            </svg>
          </span>
          <h1 className="font-display text-[15px] font-semibold tracking-tight text-ink truncate">
            Face Blur
          </h1>
          {phase !== 'idle' && phase !== 'done' && (
            <span className="text-[11px] px-2 py-0.5 rounded-full bg-cream-soft border border-cream-border text-ink-muted hidden sm:inline whitespace-nowrap">
              {phase.replace(/-/g, ' ')}
            </span>
          )}
        </div>
        <div className="flex items-center gap-4 shrink-0">
          {file && !isProcessing && (
            <button
              onClick={handleReset}
              className="text-xs text-ink-muted hover:text-accent transition-colors"
              title="Start over with a new video"
            >
              New Video
            </button>
          )}
          <button
            onClick={toggleLogMonitor}
            className="text-xs text-ink-muted hover:text-ink transition-colors"
          >
            {showLogMonitor ? 'Hide Logs' : 'Logs'}
          </button>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto">
        {!file ? (
          /* ── Landing: centered drop-zone hero ──────────────────────── */
          <div className="relative min-h-full flex items-center justify-center overflow-hidden">
            {/* slow-drifting accent wash behind hero */}
            <motion.div
              aria-hidden
              className="absolute inset-0 pointer-events-none"
              animate={{ x: [0, 24, 0], y: [0, -16, 0] }}
              transition={{ duration: 14, repeat: Infinity, ease: 'easeInOut' }}
              style={{
                background:
                  'radial-gradient(42% 38% at 50% 34%, rgba(37,99,235,0.10) 0%, rgba(124,58,237,0.08) 55%, transparent 100%)',
              }}
            />
            <motion.div
              variants={staggerParent}
              initial="hidden"
              animate="visible"
              className="relative w-full max-w-2xl mx-auto px-4 py-12 sm:py-16 text-center"
            >
              <motion.p
                variants={fadeUp}
                className="text-[11px] font-semibold uppercase tracking-[0.12em] text-accent"
              >
                Face Blur Studio
              </motion.p>
              <motion.h2
                variants={fadeUp}
                className="mt-3 font-display font-semibold text-ink"
                style={{ fontSize: 'clamp(2.25rem, 6vw, 3.75rem)', letterSpacing: '-0.03em', lineHeight: 1.05 }}
              >
                Blur faces before you share.
              </motion.h2>
              <motion.p
                variants={fadeUp}
                className="mt-4 text-sm sm:text-base text-ink-soft leading-relaxed max-w-lg mx-auto"
              >
                On-device AI detects every face and tracks each person across frames.
                Pick who to hide, choose a blur style, download the result.
              </motion.p>
              <motion.div variants={fadeUp} className="mt-8">
                <FileDropZone />
              </motion.div>
              <motion.div
                variants={fadeUp}
                className="mt-6 flex flex-wrap items-center justify-center gap-2"
              >
                {['100% in-browser', 'No uploads', 'No accounts', 'WebAssembly + WebGPU'].map((pill) => (
                  <span
                    key={pill}
                    className="inline-flex items-center gap-1.5 rounded-full border border-cream-border bg-cream-light px-3 py-1 text-[11px] text-ink-soft shadow-doodle"
                  >
                    <span className="w-1 h-1 rounded-full bg-accent" aria-hidden />
                    {pill}
                  </span>
                ))}
              </motion.div>
            </motion.div>
          </div>
        ) : (
          /* ── Workspace: preview + options rail two-pane ────────────── */
          <div className="max-w-6xl mx-auto px-4 py-6 lg:py-8">
            {!gpuAccelerated && (
              <motion.div
                variants={fadeUp}
                initial="hidden"
                animate="visible"
                className="mb-5 rounded-[14px] p-3 text-xs border border-warn/20 bg-warn/5 text-warn flex items-start gap-2"
              >
                <svg className="w-4 h-4 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
                </svg>
                <div>
                  <p className="font-medium">Running on CPU — WebGPU not available</p>
                  <p className="mt-0.5 text-ink-soft">
                    Inference will be 3-6x slower. For best performance, use Chrome 113+ or Edge 113+.
                  </p>
                </div>
              </motion.div>
            )}

            <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px] items-start">
              {/* Preview / progress column */}
              <motion.section
                variants={fadeUp}
                initial="hidden"
                animate="visible"
                className="space-y-5 min-w-0"
              >
                <MemoryWarning />
                <VideoPreview />

                {showProcessing && (
                  <div className="space-y-3">
                    <PhaseIndicator />
                    <ProgressBar />
                  </div>
                )}

                {phase === 'waiting-selection' && identities.length === 0 && (
                  <motion.div
                    variants={scaleIn}
                    initial="hidden"
                    animate="visible"
                    className="doodle-section space-y-3 text-center"
                  >
                    <p className="text-sm text-ink-soft">No faces detected in this video</p>
                    <p className="text-[11px] text-ink-muted">
                      Try a different video with clearer faces, or trim to a section where faces are visible.
                    </p>
                    <button
                      onClick={() => {
                        useProcessStore.getState().reset();
                        useFileStore.getState().reset();
                      }}
                      className="doodle-btn-secondary text-xs"
                    >
                      Try Another Video
                    </button>
                  </motion.div>
                )}

                {showOutput && <OutputActions />}
              </motion.section>

              {/* Options rail */}
              <motion.aside
                variants={staggerParent}
                initial="hidden"
                animate="visible"
                className="space-y-5 lg:sticky lg:top-6"
              >
                {showFacePicker ? (
                  <>
                    <FacePicker />
                    <BlurTypeSelector />
                  </>
                ) : (
                  <div className="doodle-section text-center space-y-1.5 py-6">
                    <p className="text-xs font-medium text-ink-soft">
                      {showProcessing ? 'Analyzing your video…' : 'Preparing workspace…'}
                    </p>
                    <p className="text-[11px] text-ink-muted">
                      Face options appear here once detection finishes.
                    </p>
                  </div>
                )}
              </motion.aside>
            </div>
          </div>
        )}
      </main>

      {showLogMonitor && <LogMonitor />}
    </div>
  );
}
