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

export default function App() {
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
      <header className="h-[44px] shrink-0 flex items-center justify-between px-4 border-b border-cream-border bg-glass z-20">
        <div className="flex items-center gap-3">
          <a
            href="https://www.browsersnip.com"
            className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink-soft transition-colors"
            title="Back to BrowserSnip"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            <span className="hidden sm:inline">BrowserSnip</span>
          </a>
          <span className="text-cream-border">|</span>
          <span className="text-sm font-semibold tracking-wide text-ink">
            Face Blur
          </span>
          {phase !== 'idle' && phase !== 'done' && (
            <span className="text-[11px] text-ink-muted hidden sm:inline">
              {phase.replace(/-/g, ' ')}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
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
            className="text-xs text-ink-muted hover:text-ink-soft transition-colors"
          >
            {showLogMonitor ? 'Hide Logs' : 'Logs'}
          </button>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto">
        <div className="max-w-2xl mx-auto px-4 py-6 space-y-6">
          {!gpuAccelerated && (
            <div className="rounded-md p-3 text-xs border border-warn/20 bg-warn/5 text-warn animate-slide-up flex items-start gap-2">
              <svg className="w-4 h-4 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
              </svg>
              <div>
                <p className="font-medium">Running on CPU — WebGPU not available</p>
                <p className="mt-0.5 text-warn/70">
                  Inference will be 3-6x slower. For best performance, use Chrome 113+ or Edge 113+.
                </p>
              </div>
            </div>
          )}
          {!file && <FileDropZone />}

          {file && (
            <>
              <MemoryWarning />
              <VideoPreview />

              {showProcessing && (
                <div className="space-y-3 animate-fade-in">
                  <PhaseIndicator />
                  <ProgressBar />
                </div>
              )}

              {showFacePicker && (
                <div className="space-y-4 animate-slide-up">
                  <FacePicker />
                  <BlurTypeSelector />
                </div>
              )}

              {phase === 'waiting-selection' && identities.length === 0 && (
                <div className="doodle-section space-y-3 text-center animate-fade-in">
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
                </div>
              )}

              {showOutput && <OutputActions />}
            </>
          )}
        </div>
      </main>

      {showLogMonitor && <LogMonitor />}
    </div>
  );
}
