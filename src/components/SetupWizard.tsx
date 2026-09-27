/**
 * AI setup wizard (FEATURES_V2 §9): what is installed, what will be downloaded from where (and how
 * big) before anything starts, then per-step progress from `setup://progress`. Start / Skip /
 * Cancel; the editor, colour and export work without the AI components.
 */
import { useEffect, useMemo, useState } from 'react';
import { api, isTauri, type SetupPlanStep } from '@/lib/tauri';
import { SETUP_STEPS, cancelSetup, closeSetupWizard, missingComponents, startSetup, useSetup } from '@/state/setup';
import { CloseIcon } from './Overlays';

const COMPONENTS: Array<{ id: 'ffmpeg' | 'python' | 'models'; label: string; hint: string }> = [
  { id: 'ffmpeg', label: 'ffmpeg', hint: 'decoding, waveforms, export' },
  { id: 'python', label: 'Python environment', hint: 'uv, Python 3.12, PyTorch and the pipeline packages' },
  { id: 'models', label: 'AI models', hint: 'shot detection, detection / tracking, Demucs, RAFT' },
];

function fmtBytes(n?: number): string {
  if (!n || n <= 0) return '';
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  return `${Math.round(n / 1e3)} kB`;
}

export default function SetupWizard() {
  const open = useSetup((s) => s.open);
  if (!open) return null;
  return <WizardBody />;
}

function WizardBody() {
  const status = useSetup((s) => s.status);
  const jobId = useSetup((s) => s.jobId);
  const step = useSetup((s) => s.step);
  const pct = useSetup((s) => s.pct);
  const message = useSetup((s) => s.message);
  const seen = useSetup((s) => s.seen);
  const result = useSetup((s) => s.result);
  const missing = useMemo(() => missingComponents(status), [status]);
  const [chosen, setChosen] = useState<Set<string>>(() => new Set(missing.length ? missing : ['ffmpeg', 'python', 'models']));
  const [plan, setPlan] = useState<SetupPlanStep[] | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
  const running = !!jobId;

  // what will run (sources, sizes) for the chosen components
  useEffect(() => {
    let alive = true;
    setPlan(null);
    setPlanError(null);
    api
      .setupPlan([...chosen])
      .then((p) => alive && setPlan(p))
      .catch((e) => alive && setPlanError(String(e)));
    return () => {
      alive = false;
    };
  }, [chosen]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !running) {
        e.preventDefault();
        e.stopImmediatePropagation();
        closeSetupWizard(false);
      }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [running]);

  const planned = plan?.map((p) => p.step) ?? [];
  const stepIndex = step ? planned.indexOf(step) : -1;
  const overall = planned.length ? Math.max(0, Math.min(1, (Math.max(0, stepIndex) + pct) / planned.length)) : 0;
  const totalBytes = (plan ?? []).reduce((n, p) => n + (p.sizeBytes ?? 0), 0);
  const labelOf = (id: string) => SETUP_STEPS.find((s) => s.id === id)?.label ?? id;
  const toggle = (id: string) => {
    const next = new Set(chosen);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setChosen(next);
  };

  return (
    <div className="modal-backdrop">
      <div className="modal setup-wizard" role="dialog" aria-label="AI setup" aria-modal="true">
        <h3>
          AI setup
          {!running ? (
            <button className="small ghost" onClick={() => closeSetupWizard(false)} title="Close (Esc)" aria-label="Close">
              <CloseIcon />
            </button>
          ) : null}
        </h3>
        <div className="hint" style={{ marginBottom: 10 }}>
          The AI features (analysis, duplicate removal, auto-zoom, voice separation, optical-flow slow motion) need ffmpeg, a Python environment and the models on this PC. Editing, colour and export work
          without them. Nothing is downloaded before you press <b>Download and install</b>.
        </div>

        <div className="setup-status">
          {COMPONENTS.map((c) => {
            const ok = !!status?.[c.id];
            return (
              <label key={c.id} className="check-row" title={c.hint}>
                <input type="checkbox" checked={chosen.has(c.id)} disabled={running} onChange={() => toggle(c.id)} />
                <span className={`status-dot ${ok ? 'ok' : 'missing'}`} />
                <span style={{ flex: 1 }}>
                  {c.label} <span className="hint">· {c.hint}</span>
                </span>
                <span className="hint">{status ? (ok ? 'installed' : 'missing') : '…'}</span>
              </label>
            );
          })}
          <div className="hint" style={{ marginTop: 4 }}>
            GPU: {status ? status.gpu ?? 'no NVIDIA GPU detected: PyTorch installs the CPU wheels (slower AI)' : '…'}
          </div>
        </div>

        <h4 className="setup-h">What will be downloaded{totalBytes ? ` · about ${fmtBytes(totalBytes)}` : ''}</h4>
        {planError ? <div className="finding" style={{ borderLeftColor: 'var(--red)' }}>{planError}</div> : null}
        {!plan && !planError ? <div className="hint">Checking…</div> : null}
        <ol className="setup-steps">
          {(plan ?? []).map((p) => {
            const done = running || result ? seen.includes(p.step) && (p.step !== step || pct >= 1 || !!result?.ok) : false;
            const active = running && p.step === step;
            return (
              <li key={p.step} className={`${done ? 'done' : ''} ${active ? 'active' : ''}`}>
                <div className="setup-step-head">
                  <span className="dot" />
                  <b>{labelOf(p.step)}</b>
                  {p.sizeBytes ? <span className="hint">{fmtBytes(p.sizeBytes)}</span> : null}
                </div>
                <div className="hint">{p.note}</div>
                {p.url ? (
                  <div className="setup-url mono" title={p.url}>
                    {p.url}
                  </div>
                ) : null}
                {active ? (
                  <div className="progress" style={{ marginTop: 4 }}>
                    <div style={{ width: `${Math.round(pct * 100)}%` }} />
                  </div>
                ) : null}
              </li>
            );
          })}
        </ol>

        {running ? (
          <div style={{ marginTop: 10 }}>
            <div className="progress">
              <div style={{ width: `${Math.round(overall * 100)}%` }} />
            </div>
            <div className="hint">
              {Math.round(overall * 100)} % · {step ? labelOf(step) : 'starting'} · {message}
            </div>
          </div>
        ) : null}
        {result ? (
          <div className="finding" style={{ marginTop: 10, borderLeftColor: result.ok ? 'var(--green)' : 'var(--red)' }}>
            {result.ok ? 'AI setup finished: every component is installed and verified.' : result.error === 'cancelled' ? 'AI setup cancelled. Run it again from Help → AI setup…' : `AI setup failed: ${result.error ?? 'unknown error'}`}
          </div>
        ) : null}
        {!isTauri() ? <div className="hint" style={{ marginTop: 6 }}>Browser mode: the steps are simulated.</div> : null}

        <div className="actions">
          {running ? (
            <button onClick={() => void cancelSetup()}>Cancel</button>
          ) : result?.ok ? (
            <button className="primary" onClick={() => closeSetupWizard(false)}>
              Done
            </button>
          ) : (
            <>
              <button onClick={() => closeSetupWizard(true)} title="Use the editor without the AI features (Help → AI setup… later)">
                Skip
              </button>
              <button className="primary" disabled={!chosen.size || !plan} onClick={() => void startSetup([...chosen])}>
                {result ? 'Retry' : 'Download and install'}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
