interface Props {
  message: string;
  percent: number;
}

export function LoadingScreen({ message, percent }: Props) {
  return (
    <div className="h-screen-safe flex items-center justify-center bg-cream">
      <div className="w-full max-w-md px-6 text-center space-y-6 animate-fade-in">
        <div className="space-y-2">
          <h1 className="text-lg font-semibold text-ink tracking-tight">
            BrowserSnip Face Blur
          </h1>
          <p className="text-xs text-ink-muted">
            100% client-side AI face blurring
          </p>
        </div>

        <div className="doodle-section space-y-3">
          <div className="w-full h-1.5 rounded-full bg-cream-border overflow-hidden">
            <div
              className="h-full rounded-full bg-accent transition-all duration-500 ease-out"
              style={{ width: `${percent}%` }}
            />
          </div>
          <p className="text-xs text-ink-soft animate-pulse">{message}</p>
          <p className="text-[11px] text-ink-muted">{Math.round(percent)}%</p>
        </div>

        <p className="text-[10px] text-ink-muted leading-relaxed">
          AI models are loaded once and cached by your browser.
          <br />
          Subsequent visits will be instant.
        </p>
      </div>
    </div>
  );
}
