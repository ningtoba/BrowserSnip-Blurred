import { useProcessStore } from '@/stores/process-store';
import type { BlurType } from '@/types';

const BLUR_OPTIONS: { type: BlurType; label: string; description: string }[] = [
  {
    type: 'pixelate',
    label: 'Pixelated Mosaic',
    description: 'Blocky pixelation over the entire face region',
  },
  {
    type: 'eye-bar',
    label: 'Black Bar (Eyes)',
    description: 'Solid black bar covering only the eyes',
  },
];

export function BlurTypeSelector() {
  const blurConfig = useProcessStore((s) => s.blurConfig);
  const setBlurType = useProcessStore((s) => s.setBlurType);
  const identityBlurTypes = useProcessStore((s) => s.identityBlurTypes);

  const hasPerIdentity = identityBlurTypes.size > 0;

  return (
    <div className={`doodle-section space-y-3 ${hasPerIdentity ? 'opacity-40 pointer-events-none' : ''}`}>
      <h4 className="text-xs font-semibold text-ink">
        Blur Type
        {hasPerIdentity && (
          <span className="ml-2 text-[10px] font-normal text-ink-muted">
            (per-face overrides active)
          </span>
        )}
      </h4>
      <div className="flex gap-2">
        {BLUR_OPTIONS.map((option) => (
          <button
            key={option.type}
            onClick={() => setBlurType(option.type)}
            className={`doodle-chip ${
              blurConfig.type === option.type
                ? 'doodle-chip-active'
                : 'doodle-chip-inactive'
            }`}
          >
            <div className="text-left space-y-0.5">
              <p className="text-xs font-medium">{option.label}</p>
              <p className="text-[10px] opacity-70">{option.description}</p>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
