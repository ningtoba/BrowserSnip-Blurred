import { useProcessStore } from '@/stores/process-store';
import type { FaceIdentity, BlurType } from '@/types';

const BLUR_OPTIONS: { type: BlurType; label: string }[] = [
  { type: 'pixelate', label: 'Mosaic' },
  { type: 'eye-bar', label: 'Bar' },
  { type: 'black-box', label: 'Box' },
];

interface Props {
  identity: FaceIdentity;
  thumbnailUrl?: string;
  isSelected: boolean;
  onToggle: () => void;
}

export function FaceThumbnail({
  identity,
  thumbnailUrl,
  isSelected,
  onToggle,
}: Props) {
  const identityBlurTypes = useProcessStore((s) => s.identityBlurTypes);
  const setIdentityBlurType = useProcessStore((s) => s.setIdentityBlurType);
  const globalBlurType = useProcessStore((s) => s.blurConfig.type);

  // Show per-identity type if set, otherwise fall back to global
  const activeType = identityBlurTypes.get(identity.id) ?? globalBlurType;

  return (
    <div
      className={`
        rounded-doodle p-3 transition-all duration-200 select-none
        ${
          isSelected
            ? 'border border-accent bg-accent/5 glow-border'
            : 'tool-card'
        }
      `}
    >
      <div onClick={onToggle} className="cursor-pointer">
        <div className="aspect-square rounded overflow-hidden bg-cream border border-cream-border mb-2.5 flex items-center justify-center">
          {thumbnailUrl ? (
            <img
              src={thumbnailUrl}
              alt={`Person ${identity.id + 1}`}
              className="w-full h-full object-cover"
              draggable={false}
            />
          ) : (
            <div className="text-ink-muted text-[10px]">No preview</div>
          )}
        </div>

        <div className="space-y-1">
          <p className="text-xs font-medium text-ink-soft truncate">
            Person {identity.id + 1}
          </p>
          <p className="text-[10px] text-ink-muted">
            {identity.faces.length} frame{identity.faces.length !== 1 ? 's' : ''}
          </p>
        </div>

        <div className="mt-2 flex items-center gap-1.5">
          <div
            className={`
              w-3 h-3 rounded border transition-all duration-150 flex items-center justify-center
              ${isSelected ? 'bg-accent border-accent' : 'border-cream-border bg-cream-soft'}
            `}
          >
            {isSelected && (
              <svg className="w-2 h-2 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
              </svg>
            )}
          </div>
          <span className="text-[10px] text-ink-muted">
            {isSelected ? 'Blur' : 'Keep'}
          </span>
        </div>
      </div>

      {isSelected && (
        <div className="mt-2.5 pt-2 border-t border-cream-border flex gap-1">
          {BLUR_OPTIONS.map((opt) => (
            <button
              key={opt.type}
              onClick={(e) => { e.stopPropagation(); setIdentityBlurType(identity.id, opt.type); }}
              className={`flex-1 text-[9px] py-1 px-1 rounded transition-colors ${
                activeType === opt.type
                  ? 'bg-accent text-white'
                  : 'bg-cream-soft text-ink-muted hover:bg-cream'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
