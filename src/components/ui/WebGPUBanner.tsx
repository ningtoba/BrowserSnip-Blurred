export function WebGPUBanner() {
  const browserName = getBrowserName();

  return (
    <div className="h-screen-safe flex items-center justify-center bg-cream p-6">
      <div className="w-full max-w-lg text-center space-y-6 animate-fade-in">
        <div className="w-16 h-16 mx-auto rounded-full bg-warn/10 border border-warn/20 flex items-center justify-center">
          <svg
            className="w-8 h-8 text-warn"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"
            />
          </svg>
        </div>

        <div className="space-y-2">
          <h1 className="text-lg font-semibold text-ink">WebGPU Required</h1>
          <p className="text-sm text-ink-soft leading-relaxed">
            This tool uses WebGPU for on-device AI inference. All processing is
            100% client-side — your videos never leave your machine.
          </p>
        </div>

        <div className="doodle-section space-y-2 text-left">
          <p className="text-xs text-warn font-medium">
            {browserName} does not support WebGPU
          </p>
          <p className="text-xs text-ink-soft leading-relaxed">
            Please switch to a compatible browser:
          </p>
          <ul className="text-xs text-ink-soft space-y-1 list-disc list-inside">
            <li>Google Chrome 113 or newer</li>
            <li>Microsoft Edge 113 or newer</li>
            <li>Opera 99 or newer</li>
            <li>Chrome for Android 121 or newer</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

function getBrowserName(): string {
  const ua = navigator.userAgent;
  if (ua.includes('Firefox')) return 'Firefox';
  if (ua.includes('Safari') && !ua.includes('Chrome')) return 'Safari';
  if (ua.includes('Edg')) return 'Microsoft Edge';
  if (ua.includes('Opera')) return 'Opera';
  if (ua.includes('Chrome')) return 'Google Chrome';
  return 'Your browser';
}
