/**
 * In-app modal dialogs (confirmations, unsaved-changes prompts) with a promise API:
 *   const choice = await ask({ title, message, buttons: [...] })   // button id, or null on Escape
 */
import { useEffect, useRef } from 'react';
import { create } from 'zustand';

export interface DialogButton {
  id: string;
  label: string;
  kind?: 'primary' | 'danger' | 'default';
}

interface DialogRequest {
  title: string;
  message?: React.ReactNode;
  buttons: DialogButton[];
  resolve: (id: string | null) => void;
}

const useDialogs = create<{ queue: DialogRequest[] }>(() => ({ queue: [] }));

export function ask(req: Omit<DialogRequest, 'resolve'>): Promise<string | null> {
  return new Promise((resolve) => {
    useDialogs.setState((s) => ({ queue: [...s.queue, { ...req, resolve }] }));
  });
}

/** Yes/no confirmation; resolves true when the confirm button was pressed. */
export async function confirmAction(title: string, message: React.ReactNode, confirmLabel = 'OK', danger = false): Promise<boolean> {
  const id = await ask({
    title,
    message,
    buttons: [
      { id: 'cancel', label: 'Cancel' },
      { id: 'ok', label: confirmLabel, kind: danger ? 'danger' : 'primary' },
    ],
  });
  return id === 'ok';
}

export function isDialogOpen(): boolean {
  return useDialogs.getState().queue.length > 0;
}

export function DialogHost() {
  const current = useDialogs((s) => s.queue[0] ?? null);
  const ref = useRef<HTMLDivElement>(null);

  const close = (id: string | null) => {
    if (!current) return;
    current.resolve(id);
    useDialogs.setState((s) => ({ queue: s.queue.slice(1) }));
  };

  useEffect(() => {
    if (!current) return;
    // focus the primary button so Enter confirms
    const btn = ref.current?.querySelector<HTMLButtonElement>('button.primary, button.danger') ?? ref.current?.querySelector('button');
    btn?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopImmediatePropagation();
        close(null);
      }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current]);

  if (!current) return null;
  return (
    <div className="modal-backdrop" onMouseDown={() => close(null)}>
      <div className="modal dialog" ref={ref} role="alertdialog" aria-modal="true" aria-label={current.title} onMouseDown={(e) => e.stopPropagation()}>
        <h3>{current.title}</h3>
        {current.message ? <div className="dialog-message">{current.message}</div> : null}
        <div className="actions">
          {current.buttons.map((b) => (
            <button key={b.id} className={b.kind === 'primary' ? 'primary' : b.kind === 'danger' ? 'danger' : ''} onClick={() => close(b.id)}>
              {b.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
