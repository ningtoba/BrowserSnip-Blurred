import { useRef, useEffect } from 'react';
import { useProcessStore } from '@/stores/process-store';

export function LogMonitor() {
  const logs = useProcessStore((s) => s.logs);
  const error = useProcessStore((s) => s.error);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="h-[180px] shrink-0 border-t border-cream-border bg-cream-soft overflow-hidden">
      <div className="flex items-center justify-between px-4 py-1.5 border-b border-cream-border">
        <span className="text-[10px] font-medium text-ink-muted uppercase tracking-wider">
          Processing Log
        </span>
        <span className="text-[10px] text-ink-muted">
          {logs.length} lines
        </span>
      </div>
      <div
        ref={scrollRef}
        className="h-full overflow-y-auto p-2 font-mono text-[10px] leading-relaxed"
      >
        {logs.length === 0 && !error && (
          <p className="text-ink-muted">Waiting for processing to begin...</p>
        )}
        {error && (
          <p className="text-danger mb-2">Error: {error}</p>
        )}
        {logs.map((line, i) => (
          <div key={i} className="log-line">{line}</div>
        ))}
      </div>
    </div>
  );
}
