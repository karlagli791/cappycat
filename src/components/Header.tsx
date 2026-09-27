import { useEffect, useRef, useState } from 'react';
import { useEditor } from '@/state/store';
import { api } from '@/lib/tauri';
import { basename } from '@/lib/format';
import { newProject, openProject, saveProject } from '@/state/persistence';
import { openSetupWizard, useAiReady } from '@/state/setup';

interface Props {
  workflowState: string;
  busy: boolean;
  onRunPipeline: () => void;
  onExport: () => void;
  onImport: () => void;
  onPasteAttributes: () => void;
  onShortcuts: () => void;
  onProjectSettings: () => void;
}

export default function Header({ workflowState, busy, onRunPipeline, onExport, onImport, onPasteAttributes, onShortcuts, onProjectSettings }: Props) {
  const ai = useAiReady();
  const name = useEditor((s) => s.project.name);
  const fmt = useEditor((s) => `${s.project.width}×${s.project.height} @ ${s.project.fps}fps`);
  const undo = useEditor((s) => s.undo);
  const redo = useEditor((s) => s.redo);
  const canUndo = useEditor((s) => s.past.length > 0);
  const canRedo = useEditor((s) => s.future.length > 0);
  const hasSel = useEditor((s) => s.selection.clipIds.length > 0);
  const hasClipboard = useEditor((s) => !!s.attrClipboard);
  const setInspectorTab = useEditor((s) => s.setInspectorTab);
  const universal = useEditor((s) => s.project.universalAdjust);
  const setUniversalEnabled = useEditor((s) => s.setUniversalEnabled);
  const [menu, setMenu] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  // menus close on Escape and on a click outside
  useEffect(() => {
    if (!menu) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopImmediatePropagation();
        setMenu(null);
      }
    };
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setMenu(null);
    };
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('mousedown', onDown);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      window.removeEventListener('mousedown', onDown);
    };
  }, [menu]);

  const run = (fn: () => void) => () => {
    setMenu(null);
    fn();
  };
  const recent = menu === 'file' ? api.recentProjects() : [];
  const chipClass = busy ? 'busy' : workflowState === 'ready' ? 'ready' : workflowState === 'failed' ? 'failed' : '';

  return (
    <header className="header">
      <div className="brand">
        <div className="logo">C</div>
        Cappycat
      </div>
      <div className="menu" ref={ref}>
        <MenuButton label="File" open={menu === 'file'} onOpen={() => setMenu(menu === 'file' ? null : 'file')} onHover={() => menu && setMenu('file')}>
          <MenuItem onClick={run(() => void newProject())} kbd="Ctrl+N">New project</MenuItem>
          <MenuItem onClick={run(() => void openProject())} kbd="Ctrl+O">Open project or analysis…</MenuItem>
          {recent.length ? <div className="menu-sep">Recent</div> : null}
          {recent.map((p) => (
            <MenuItem key={p} onClick={run(() => void openProject(p))} title={p}>
              {basename(p)}
            </MenuItem>
          ))}
          <div className="menu-sep" />
          <MenuItem onClick={run(() => void saveProject())} kbd="Ctrl+S">Save</MenuItem>
          <MenuItem onClick={run(() => void saveProject(true))} kbd="Ctrl+Shift+S">Save as…</MenuItem>
          <MenuItem onClick={run(onImport)} kbd="Ctrl+I">Import media…</MenuItem>
          <MenuItem onClick={run(onProjectSettings)}>Project settings…</MenuItem>
          <MenuItem onClick={run(onExport)} kbd="Ctrl+E" disabled={busy}>Export…</MenuItem>
        </MenuButton>
        <MenuButton label="Edit" open={menu === 'edit'} onOpen={() => setMenu(menu === 'edit' ? null : 'edit')} onHover={() => menu && setMenu('edit')}>
          <MenuItem disabled={!canUndo} onClick={run(undo)} kbd="Ctrl+Z">Undo</MenuItem>
          <MenuItem disabled={!canRedo} onClick={run(redo)} kbd="Ctrl+Y">Redo</MenuItem>
          <div className="menu-sep" />
          <MenuItem disabled={!hasSel} onClick={run(() => { const s = useEditor.getState(); s.copyAttributes(s.selection.clipIds[0]); })} kbd="Ctrl+Alt+C">Copy attributes</MenuItem>
          <MenuItem disabled={!hasSel || !hasClipboard} onClick={run(onPasteAttributes)} kbd="Ctrl+Alt+V">Paste attributes…</MenuItem>
          <MenuItem onClick={run(() => useEditor.getState().selectAll())} kbd="Ctrl+A">Select all clips</MenuItem>
          <div className="menu-sep" />
          <MenuItem onClick={run(onShortcuts)} kbd="?">Keyboard shortcuts</MenuItem>
        </MenuButton>
        <MenuButton label="Help" open={menu === 'help'} onOpen={() => setMenu(menu === 'help' ? null : 'help')} onHover={() => menu && setMenu('help')}>
          <MenuItem onClick={run(openSetupWizard)} title="Install or repair ffmpeg, the Python environment and the AI models">AI setup…</MenuItem>
          <MenuItem onClick={run(onShortcuts)} kbd="?">Keyboard shortcuts</MenuItem>
          <MenuItem onClick={run(() => void openLogsFolder())}>Open logs folder</MenuItem>
        </MenuButton>
        <button className="ghost" onClick={() => { setInspectorTab('ai'); onRunPipeline(); }} disabled={busy || !ai.ready} title={ai.hint || undefined}>
          AI Pipeline
        </button>
        <button className="ghost" onClick={() => setInspectorTab('color')}>
          Color
        </button>
        <button className="ghost" onClick={onExport} disabled={busy} title="Export (Ctrl+E)">
          Export
        </button>
      </div>
      <div className="spacer" />
      <button
        className={`small ${universal?.enabled ? 'active' : ''}`}
        onClick={() => setUniversalEnabled(!universal?.enabled)}
        title="Universal adjust: the house look applied on top of every clip. Click to switch it on or off for this project (U)."
      >
        Universal adjust: {universal?.enabled ? 'ON' : 'OFF'}
      </button>
      <span className={`state-chip ${chipClass}`}>
        <span className="dot" />
        {labelFor(workflowState)}
      </span>
      <div className="project-chip" role="button" tabIndex={0} onClick={onProjectSettings} onKeyDown={(e) => (e.key === 'Enter' ? onProjectSettings() : undefined)} title="Project settings: frame rate and frame interpolation">
        Project: <b>{name}</b>
        <SaveIndicator />
        <span className="fmt">{fmt}</span>
      </div>
    </header>
  );
}

async function openLogsFolder() {
  const log = useEditor.getState().log;
  try {
    const paths = await api.appPaths();
    if (!paths.logsDir) {
      log('info', 'Browser mode: there is no logs folder (native app only).');
      return;
    }
    await api.openPath(paths.logsDir);
  } catch (e) {
    log('warn', `Could not open the logs folder: ${String(e)}`);
  }
}

/** "Unsaved" / "Saving…" / "Saved 14:32" / "Autosaved 14:35". */
function SaveIndicator() {
  const dirty = useEditor((s) => s.dirty);
  const status = useEditor((s) => s.saveStatus);
  const path = useEditor((s) => s.projectPath);
  const t = (ms: number) => new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  let text: string;
  let cls = 'save-ind';
  if (status.state === 'saving') text = 'Saving…';
  else if (status.state === 'error') {
    text = 'Save failed';
    cls += ' error';
  } else if (dirty) {
    text = status.autosavedAt ? `Unsaved · autosaved ${t(status.autosavedAt)}` : 'Unsaved';
    cls += ' dirty';
  } else text = status.savedAt ? `Saved ${t(status.savedAt)}` : path ? 'Saved' : 'Not saved yet';
  return (
    <span className={cls} title={status.message ?? (path ? `${path} (Ctrl+S saves)` : 'Ctrl+S saves the project')}>
      {dirty || status.state !== 'idle' ? <span className="dot" /> : null}
      {text}
    </span>
  );
}

function labelFor(state: string): string {
  switch (state) {
    case 'idle':
      return 'Idle';
    case 'launching':
      return 'Starting pipeline';
    case 'ingesting':
      return 'Ingesting media';
    case 'detectingShots':
      return 'Detecting cuts';
    case 'scanningDuplicates':
      return 'Scanning duplicates';
    case 'reframing':
      return 'Reframing';
    case 'normalizingAudio':
      return 'Audio & beats';
    case 'assembling':
      return 'Assembling timeline';
    case 'ready':
      return 'Pipeline ready';
    case 'failed':
      return 'Pipeline failed';
    case 'cancelled':
      return 'Cancelled';
    default:
      return state;
  }
}

function MenuButton({ label, open, onOpen, onHover, children }: { label: string; open: boolean; onOpen: () => void; onHover: () => void; children: React.ReactNode }) {
  return (
    <div style={{ position: 'relative' }} onMouseEnter={onHover}>
      <button className={`ghost ${open ? 'active' : ''}`} onClick={onOpen} aria-haspopup="menu" aria-expanded={open}>
        {label}
      </button>
      {open ? (
        <div className="menu-pop" role="menu">
          {children}
        </div>
      ) : null}
    </div>
  );
}

function MenuItem({ children, onClick, disabled, kbd, title }: { children: React.ReactNode; onClick: () => void; disabled?: boolean; kbd?: string; title?: string }) {
  return (
    <button className="ghost menu-item" role="menuitem" disabled={disabled} onClick={onClick} title={title}>
      <span className="menu-label">{children}</span>
      {kbd ? <span className="kbd">{kbd}</span> : null}
    </button>
  );
}
