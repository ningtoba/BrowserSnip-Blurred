import { useProcessStore } from '@/stores/process-store';
import { FaceThumbnail } from '@/components/face/FaceThumbnail';
import { usePipeline } from '@/hooks/usePipeline';

export function FacePicker() {
  const identities = useProcessStore((s) => s.identities);
  const thumbnails = useProcessStore((s) => s.identityThumbnails);
  const selectedIdentities = useProcessStore((s) => s.selectedIdentities);
  const toggleIdentity = useProcessStore((s) => s.toggleIdentity);
  const isProcessing = useProcessStore((s) => s.isProcessing);
  const { processAndExport } = usePipeline();

  const allSelected = identities.length > 0 && identities.every((id) => selectedIdentities.has(id.id));

  return (
    <div className="space-y-4 animate-slide-up">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-ink">
            {identities.length > 0
              ? `${identities.length} identit${identities.length === 1 ? 'y' : 'ies'} detected`
              : 'No faces detected'}
          </h3>
          <p className="text-[11px] text-ink-muted">
            {identities.length > 0
              ? 'Select faces to blur, then click Process'
              : 'Try a different video or adjust the clip'}
          </p>
        </div>
        {identities.length > 1 && (
          <button
            onClick={() => {
              const all = identities.every((id) => selectedIdentities.has(id.id));
              identities.forEach((id) => {
                if (all) {
                  if (selectedIdentities.has(id.id)) toggleIdentity(id.id);
                } else {
                  if (!selectedIdentities.has(id.id)) toggleIdentity(id.id);
                }
              });
            }}
            className="text-[11px] text-accent hover:text-accent-hover transition-colors"
          >
            {allSelected ? 'Deselect All' : 'Select All'}
          </button>
        )}
      </div>

      {identities.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          {identities.map((identity) => (
            <FaceThumbnail
              key={identity.id}
              identity={identity}
              thumbnailUrl={thumbnails.get(identity.id)}
              isSelected={selectedIdentities.has(identity.id)}
              onToggle={() => toggleIdentity(identity.id)}
            />
          ))}
        </div>
      )}

      {identities.length > 0 && (
        <button
          onClick={processAndExport}
          disabled={selectedIdentities.size === 0 || isProcessing}
          className="doodle-btn"
        >
          {isProcessing
            ? 'Processing...'
            : selectedIdentities.size === 0
              ? 'Select a face to blur'
              : `Blur ${selectedIdentities.size} face${selectedIdentities.size > 1 ? 's' : ''}`}
        </button>
      )}
    </div>
  );
}
