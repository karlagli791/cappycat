/** Small overlays: shortcut cheat sheet, paste-attributes chooser, autosave restore banner. */
import { useState } from 'react';
import { useEditor, type AttrPart } from '@/state/store';
import { discardAutosave, restoreAutosave, type AutosaveEntry } from '@/state/persistence';
import { SHORTCUTS } from '@/lib/shortcuts';

export function ShortcutsSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
  if (!open) return null;
  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="modal sheet" role="dialog" aria-label="Keyboard shortcuts" onMouseDown={(e) => e.stopPropagation()}>
        <h3>
          Keyboard shortcuts
          <button className="small ghost" onClick={onClose} title="Close (Esc)" aria-label="Close">
            <CloseIcon />
          </button>
        </h3>
        <div className="sheet-grid">
          {SHORTCUTS.map((g) => (
            <div key={g.group}>
              <h4>{g.group}</h4>
              {g.items.map(([k, what]) => (
                <div className="sheet-row" key={k}>
                  <span className="kbd">{k}</span>
                  <span>{what}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

const PARTS: Array<{ id: AttrPart; label: string }> = [
  { id: 'color', label: 'Color grade' },
  { id: 'speed', label: 'Speed' },
  { id: 'transform', label: 'Motion (transform, keyframes, blend)' },
  { id: 'audio', label: 'Audio (volume, fades, volume keyframes, keep pitch, mute, voice mode)' },
  { id: 'mask', label: 'Mask' },
];

export function PasteAttributesDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const cb = useEditor((s) => s.attrClipboard);
  const count = useEditor((s) => s.selection.clipIds.length);
  const [parts, setParts] = useState<Set<AttrPart>>(() => new Set<AttrPart>(['color']));
  if (!open) return null;
  const paste = () => {
    const s = useEditor.getState();
    const n = s.pasteAttributes(s.selection.clipIds, [...parts]);
    s.log('info', `Pasted ${[...parts].join(', ')} onto ${n} clip(s)`);
    onClose();
  };
  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="modal" role="dialog" aria-label="Paste attributes" onMouseDown={(e) => e.stopPropagation()}>
        <h3>Paste attributes</h3>
        {cb ? (
          <>
            <div className="hint" style={{ marginBottom: 8 }}>
              From <b style={{ color: 'var(--text)' }}>{cb.sourceLabel}</b> onto {count} selected clip{count === 1 ? '' : 's'}.
            </div>
            {PARTS.map((p) => (
              <label key={p.id} className="check-row">
                <input
                  type="checkbox"
                  checked={parts.has(p.id)}
                  onChange={(e) => {
                    const next = new Set(parts);
                    if (e.target.checked) next.add(p.id);
                    else next.delete(p.id);
                    setParts(next);
                  }}
                />
                {p.label}
              </label>
            ))}
          </>
        ) : (
          <div className="hint">Nothing copied yet: select a clip and press Ctrl+Alt+C.</div>
        )}
        <div className="actions">
          <button onClick={onClose}>Cancel</button>
          <button className="primary" disabled={!cb || !count || !parts.size} onClick={paste}>
            Paste
          </button>
        </div>
      </div>
    </div>
  );
}

export function RestoreBanner({ entry, onDone }: { entry: AutosaveEntry; onDone: () => void }) {
  const [busy, setBusy] = useState(false);
  return (
    <div className="banner" role="status">
      <span>
        Unsaved work from <b>{new Date(entry.at).toLocaleString()}</b> ("{entry.name}") was autosaved.
      </span>
      <button
        className="small primary"
        disabled={busy}
        onClick={async () => {
          setBusy(true);
          await restoreAutosave(entry);
          onDone();
        }}
      >
        Restore
      </button>
      <button
        className="small ghost"
        onClick={() => {
          discardAutosave(entry.projectId);
          onDone();
        }}
      >
        Dismiss
      </button>
    </div>
  );
}

export function CloseIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
      <path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}
