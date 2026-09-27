/**
 * First-run AI setup (FEATURES_V2 §9): `setup_status`, `setup_plan`, `setup_ai` / `cancel_setup`
 * and the `setup://progress` / `setup://done` events. The wizard opens by itself only in the
 * installed app (`installed: true`) when a component is missing and the user has not skipped it;
 * it can always be opened from Help -> AI setup. Without the AI components the editor, colour and
 * export work; AI actions are disabled with a hint.
 */
import { useEffect } from 'react';
import { create } from 'zustand';
import { api, onEvent, type SetupDonePayload, type SetupProgressPayload, type SetupStatus } from '@/lib/tauri';

const SKIP_KEY = 'cappycat:ai-setup-skipped';

export const SETUP_STEPS: Array<{ id: string; label: string }> = [
  { id: 'ffmpeg', label: 'ffmpeg' },
  { id: 'uv', label: 'uv (Python manager)' },
  { id: 'python', label: 'Python 3.12 environment' },
  { id: 'torch', label: 'PyTorch' },
  { id: 'packages', label: 'Pipeline packages' },
  { id: 'models', label: 'AI models' },
  { id: 'verify', label: 'Verify (doctor)' },
];

export interface SetupState {
  status: SetupStatus | null;
  open: boolean;
  jobId: string | null;
  step: string | null;
  /** progress of the current step, 0..1 */
  pct: number;
  message: string;
  /** steps seen so far in this run, in order */
  seen: string[];
  result: { ok: boolean; error: string | null } | null;
  skipped: boolean;
}

function readSkipped(): boolean {
  try {
    return localStorage.getItem(SKIP_KEY) === '1';
  } catch {
    return false;
  }
}

export const useSetup = create<SetupState>(() => ({
  status: null,
  open: false,
  jobId: null,
  step: null,
  pct: 0,
  message: '',
  seen: [],
  result: null,
  skipped: readSkipped(),
}));

/** Components the installed app still misses. */
export function missingComponents(s: SetupStatus | null): string[] {
  if (!s) return [];
  return (['ffmpeg', 'python', 'models'] as const).filter((k) => !s[k]);
}

/**
 * Can AI features run? Dev / repo mode (installed = false) and browser mode keep today's
 * behaviour (the pipeline reports what is missing itself); the installed app needs Python + models.
 */
export function aiReadyFor(s: SetupStatus | null): boolean {
  if (!s || !s.installed) return true;
  return s.python && s.models;
}

export const AI_MISSING_HINT = 'The AI components are not installed yet: Help → AI setup…';

export function useAiReady(): { ready: boolean; hint: string } {
  const status = useSetup((s) => s.status);
  const ready = aiReadyFor(status);
  return { ready, hint: ready ? '' : AI_MISSING_HINT };
}

export async function refreshSetupStatus(): Promise<SetupStatus | null> {
  try {
    const status = await api.setupStatus();
    useSetup.setState({ status });
    return status;
  } catch {
    return null;
  }
}

export function openSetupWizard(): void {
  useSetup.setState((s) => ({ open: true, result: s.jobId ? s.result : null }));
  void refreshSetupStatus();
}

export function closeSetupWizard(skip = false): void {
  if (skip) {
    try {
      localStorage.setItem(SKIP_KEY, '1');
    } catch {
      /* ignore */
    }
  }
  useSetup.setState((s) => ({ open: false, skipped: s.skipped || skip }));
}

export async function startSetup(components: string[] | undefined): Promise<void> {
  useSetup.setState({ result: null, step: null, pct: 0, message: 'starting…', seen: [] });
  try {
    const jobId = await api.setupAi(components);
    useSetup.setState({ jobId });
  } catch (e) {
    useSetup.setState({ jobId: null, result: { ok: false, error: String(e) } });
  }
}

export async function cancelSetup(): Promise<void> {
  const id = useSetup.getState().jobId;
  if (!id) return;
  try {
    await api.cancelSetup(id);
  } catch {
    /* the done event reports it */
  }
}

/** Subscribe to setup events and check the status once (App). Opens the wizard on first run. */
export function useSetupBoot(): void {
  useEffect(() => {
    let alive = true;
    const offs: Array<() => void> = [];
    const keep = (p: Promise<() => void>) =>
      void p.then((off) => {
        if (alive) offs.push(off);
        else off();
      });
    keep(
      onEvent<SetupProgressPayload>('setup://progress', (p) => {
        const st = useSetup.getState();
        if (st.jobId && p.jobId !== st.jobId) return;
        useSetup.setState({
          jobId: p.jobId,
          step: p.step,
          pct: Math.max(0, Math.min(1, p.pct ?? 0)),
          message: p.message ?? '',
          seen: st.seen.includes(p.step) ? st.seen : [...st.seen, p.step],
        });
      }),
    );
    keep(
      onEvent<SetupDonePayload>('setup://done', (p) => {
        const st = useSetup.getState();
        if (st.jobId && p.jobId !== st.jobId) return;
        useSetup.setState({ jobId: null, result: { ok: p.ok, error: p.error }, pct: p.ok ? 1 : st.pct });
        void refreshSetupStatus();
      }),
    );
    void refreshSetupStatus().then((status) => {
      if (!alive || !status) return;
      if (status.installed && missingComponents(status).length && !useSetup.getState().skipped) useSetup.setState({ open: true });
    });
    return () => {
      alive = false;
      offs.forEach((f) => f());
    };
  }, []);
}
