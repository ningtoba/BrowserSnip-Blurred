import { useProcessStore } from '@/stores/process-store';
import type { BlurType } from '@/types';

const BLUR_OPTIONS: { type: BlurType; label: string; description: string }[] = [
  {
    type: 'pixelate',
    label: 'Mosaic',
    description: 'Blocky pixelation over the face',
  },
  {
    type: 'eye-bar',
    label: 'Eye Bar',
    description: 'Black bar over the eyes',
  },
  {
    type: 'black-box',
    label: 'Black Box',
    description: 'Solid black rectangle covering the face',
  },
];

export function BlurTypeSelector() {
  const blurConfig = useProcessStore((s) => s.blurConfig);
  const setBlurType = useProcessStore((s) => s.setBlurType);
  const identityBlurTypes = useProcessStore((s) => s.identityBlurTypes);

  const hasPerIdentity = identityBlurTypes.size > 0;

  const handleSetBlurType = (type: BlurType) => {
    setBlurType(type);
    if (hasPerIdentity) {
      useProcessStore.setState({ identityBlurTypes: new Map() });
    }
  };

  return (
    <div className="rounded-[14px] p-4 bg-cream-light border border-cream-border shadow-doodle space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="font-display text-xs font-semibold text-ink tracking-tight uppercase" style={{ letterSpacing: '0.02em' }}>
          Blur Style
        </h4>
        {hasPerIdentity && (
          <button
            onClick={() => useProcessStore.setState({ identityBlurTypes: new Map() })}
            className="text-[10px] text-accent hover:text-accent-hover transition-colors"
          >
            Reset to global
          </button>
        )}
      </div>
      <div className="flex gap-2">
        {BLUR_OPTIONS.map((option) => (
          <button
            key={option.type}
            onClick={() => handleSetBlurType(option.type)}
            className={`doodle-chip ${
              blurConfig.type === option.type && !hasPerIdentity
                ? 'doodle-chip-active'
                : hasPerIdentity
                  ? 'doodle-chip-inactive opacity-50'
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
      {hasPerIdentity && (
        <p className="text-[10px] text-ink-muted">
          Per-face blur types are active. Click a blur type above to apply it globally.
        </p>
      )}
    </div>
  );
}
