import { useFileStore } from '@/stores/file-store';
import { useProcessStore } from '@/stores/process-store';
import { useUIStore } from '@/stores/ui-store';
import { LoadingScreen } from '@/components/ui/LoadingScreen';
import { WebGPUBanner } from '@/components/ui/WebGPUBanner';
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

export default function App() {
  const file = useFileStore((s) => s.file);
  const phase = useProcessStore((s) => s.phase);
  const identities = useProcessStore((s) => s.identities);
  const outputBlob = useProcessStore((s) => s.outputBlob);
  const isProcessing = useProcessStore((s) => s.isProcessing);
  const showLogMonitor = useUIStore((s) => s.showLogMonitor);
  const toggleLogMonitor = useUIStore((s) => s.toggleLogMonitor);
  const { webGPUSupported, modelsReady, loadingMessage, loadingPercent } = useONNX();

  if (!webGPUSupported) {
    return <WebGPUBanner />;
  }

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
          <span className="text-sm font-semibold tracking-wide text-ink">
            BrowserSnip Face Blur
          </span>
          {phase !== 'idle' && phase !== 'done' && (
            <span className="text-[11px] text-ink-muted hidden sm:inline">
              {phase.replace(/-/g, ' ')}
            </span>
          )}
        </div>
        <button
          onClick={toggleLogMonitor}
          className="text-xs text-ink-muted hover:text-ink-soft transition-colors"
        >
          {showLogMonitor ? 'Hide Logs' : 'Logs'}
        </button>
      </header>

      <main className="flex-1 overflow-y-auto">
        <div className="max-w-2xl mx-auto px-4 py-6 space-y-6">
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

              {showOutput && <OutputActions />}
            </>
          )}
        </div>
      </main>

      {showLogMonitor && <LogMonitor />}
    </div>
  );
}
