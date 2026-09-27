import { useCallback, useEffect, useRef, useState } from 'react';
import { useMachine } from '@xstate/react';
import Header from './components/Header';
import LibraryPanels from './components/LibraryPanels';
import ProjectSettings from './components/ProjectSettings';
import SetupWizard from './components/SetupWizard';
import { PASTE_ATTRIBUTES_EVENT } from './components/ClipContextMenu';
import Preview from './components/Preview';
import PlayerControls from './components/PlayerControls';
import Inspector from './components/Inspector';
import Timeline, { timelineContentHeight } from './components/Timeline';
import KeyframeDrawer from './components/KeyframeDrawer';
import ExportDialog, { type ExportJob, type ExportResult } from './components/ExportDialog';
import { DialogHost, isDialogOpen } from './components/Dialogs';
import { ShortcutsSheet, PasteAttributesDialog, RestoreBanner } from './components/Overlays';
import { cutPoints, findClip, mediaPaths, projectDurationMs, storyOrder, useEditor } from './state/store';
import { playClock } from './state/clock';
import { useSeparationEvents } from './state/separation';
import { useSetup, useSetupBoot } from './state/setup';
import { guardUnsaved, newProject, openProject, pendingRestore, saveProject, setAutosaveDir, startAutosave, type AutosaveEntry } from './state/persistence';
import { defaultPipelineOptions, workflowMachine } from './state/workflowMachine';
import {
  api,
  isTauri,
  onEvent,
  type AnalysisPayload,
  type ExportDonePayload,
  type ExportProgressPayload,
  type PipelineEventPayload,
  type PipelineExitPayload,
} from './lib/tauri';
import type { PipelineOptions } from './types/project';
import { basename } from './lib/format';
import { isTextEntry } from './lib/shortcuts';

const HEADER_H = 44;
const STATUS_H = 22;
/** the preview + side panels never get less than this */
const MIN_WORKSPACE_H = 280;

export default function App() {
  const [state, send] = useMachine(workflowMachine);
  const [options, setOptions] = useState<PipelineOptions>(defaultPipelineOptions);
  const [exportJob, setExportJob] = useState<ExportJob | null>(null);
  const [exportResult, setExportResult] = useState<ExportResult | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [pasteOpen, setPasteOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [restore, setRestore] = useState<AutosaveEntry | null>(null);
  const [paths, setPaths] = useState<{ ffmpeg: string | null; python: string | null } | null>(null);
  const [winH, setWinH] = useState(() => window.innerHeight);

  const addAssets = useEditor((s) => s.addAssets);
  const loadClipsFolder = useEditor((s) => s.loadClipsFolder);
  const clipsFolder = useEditor((s) => s.clipsFolder);
  const applyAnalysis = useEditor((s) => s.applyAnalysis);
  const setMediaServerUrl = useEditor((s) => s.setMediaServerUrl);
  const log = useEditor((s) => s.log);
  const assets = useEditor((s) => s.project.assets);
  const tracks = useEditor((s) => s.project.tracks);
  const largePreview = useEditor((s) => s.largePreview);
  const drawerOpen = useEditor((s) => s.drawerOpen);
  const userTimelineH = useEditor((s) => s.timelineHeight);
  const leftWidth = useEditor((s) => s.leftWidth);
  const rightWidth = useEditor((s) => s.rightWidth);
  const setUi = useEditor((s) => s.setUi);

  const busy = !['idle', 'ready', 'failed', 'cancelled'].includes(String(state.value));

  // ---- voice separation events (separate://*) -> asset stems / status ----
  useSeparationEvents();
  // ---- AI setup: status, first-run wizard (installed app), setup://* events ----
  useSetupBoot();
  // ---- the clip context menu asks for the paste-attributes dialog ----
  useEffect(() => {
    const open = () => setPasteOpen(true);
    window.addEventListener(PASTE_ATTRIBUTES_EVENT, open);
    return () => window.removeEventListener(PASTE_ATTRIBUTES_EVENT, open);
  }, []);

  // ---- window height drives the bottom area (timeline + graph drawer) ----
  useEffect(() => {
    const onResize = () => setWinH(window.innerHeight);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);
  const budget = Math.max(200, winH - HEADER_H - STATUS_H - MIN_WORKSPACE_H);
  const drawerH = drawerOpen ? Math.max(150, Math.min(240, budget - 180)) : 0;
  const contentH = timelineContentHeight(tracks);
  const wantTimeline = userTimelineH > 0 ? userTimelineH : contentH;
  const timelineH = Math.max(120, Math.min(largePreview ? Math.min(wantTimeline, 170) : wantTimeline, budget - drawerH));
  // new tracks (Separate to tracks, stacked FX tracks): grow the timeline so they are visible
  const trackCount = tracks.length;
  const lastTrackCount = useRef(trackCount);
  useEffect(() => {
    if (trackCount > lastTrackCount.current && userTimelineH > 0 && userTimelineH < contentH) setUi({ timelineHeight: 0 });
    lastTrackCount.current = trackCount;
  }, [trackCount, userTimelineH, contentH, setUi]);

  // ---- boot: media server + tool paths + autosave ----
  useEffect(() => {
    void api.mediaServerUrl().then((u) => {
      setMediaServerUrl(u);
      if (u) log('info', 'Media server ready');
      else log('warn', 'Running in browser mode: no media server, previews are synthetic.');
    });
    void api.appPaths().then((p) => {
      setPaths({ ffmpeg: p.ffmpeg, python: p.python });
      setAutosaveDir(p.cacheDir);
      if (!p.ffmpeg && isTauri()) log('error', 'ffmpeg not found. Install it (winget install Gyan.FFmpeg) or set CAPPYCAT_FFMPEG_DIR.');
      if (!p.python && isTauri()) log('warn', 'Python pipeline not found. Create pipeline/.venv to enable the AI pipeline.');
    });
    setRestore(pendingRestore());
    return startAutosave();
  }, [setMediaServerUrl, log]);

  // ---- the media server only serves registered files: register every asset / LUT / stem ----
  useEffect(() => {
    const registered = new Set<string>();
    const register = () => {
      const fresh = mediaPaths(useEditor.getState().project).filter((p) => !registered.has(p));
      if (!fresh.length) return;
      fresh.forEach((p) => registered.add(p));
      api.registerMedia(fresh).catch((e) => {
        fresh.forEach((p) => registered.delete(p));
        log('warn', `Could not register media with the media server: ${String(e)}`);
      });
    };
    register();
    let lastAssets = useEditor.getState().project.assets;
    return useEditor.subscribe((s) => {
      if (s.project.assets === lastAssets) return;
      lastAssets = s.project.assets;
      register();
    });
  }, [log]);

  // ---- unsaved changes: close guard (native) / beforeunload (browser) ----
  useEffect(() => {
    if (!isTauri()) {
      const onBeforeUnload = (e: BeforeUnloadEvent) => {
        if (!useEditor.getState().dirty) return;
        e.preventDefault();
        e.returnValue = '';
      };
      window.addEventListener('beforeunload', onBeforeUnload);
      return () => window.removeEventListener('beforeunload', onBeforeUnload);
    }
    let alive = true;
    let off: (() => void) | null = null;
    void (async () => {
      const { invoke } = await import('@tauri-apps/api/core');
      // a JS close handler must destroy the window itself: only install it when that is permitted
      try {
        await invoke('plugin:window|destroy', { label: '__cappycat_probe__' });
      } catch (e) {
        if (/not allowed|permission/i.test(String(e))) {
          log('warn', 'Close guard disabled: add "core:window:allow-destroy" to src-tauri/capabilities (autosave still protects your work).');
          return;
        }
      }
      const { getCurrentWindow } = await import('@tauri-apps/api/window');
      const win = getCurrentWindow();
      const un = await win.onCloseRequested(async (event) => {
        if (!useEditor.getState().dirty) return;
        event.preventDefault();
        if (await guardUnsaved('close Cappycat')) await win.destroy();
      });
      if (alive) off = un;
      else un();
    })();
    return () => {
      alive = false;
      off?.();
    };
  }, [log]);

  // ---- clips folder: load on start, ordered by filename ----
  const scanFolder = useCallback(
    async (path?: string | null, initial = false) => {
      try {
        const scan = await api.scanClipsFolder(path);
        loadClipsFolder(scan.folder, scan.assets, scan.warnings, { initial });
        const vids = scan.assets.filter((a) => a.kind === 'video');
        if (vids.length) {
          log('info', `Loaded ${vids.length} clip(s) from ${scan.folder}`);
          vids
            .slice()
            .sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
            .forEach((a, i) => log('info', `  ${i + 1}. ${a.name} <- ${a.orderReason ?? 'filename order'}`));
        } else log('info', `Clips folder ${scan.folder} is empty. Drop your clips there, then press Rescan.`);
        scan.warnings.forEach((w) => log('warn', `clip order: ${w}`));
      } catch (e) {
        log('warn', `Could not scan the clips folder: ${String(e)}`);
      }
    },
    [loadClipsFolder, log],
  );

  useEffect(() => {
    if (useEditor.getState().project.assets.length === 0) void scanFolder(null, true);
  }, [scanFolder]);

  // ---- universal adjust preset: applied to every project unless switched off ----
  useEffect(() => {
    void api
      .loadUniversalAdjust()
      .then(({ path, preset }) => {
        useEditor.getState().setUniversalPreset(preset, path);
        log('info', `Universal adjust "${preset.name}" loaded (${preset.enabledByDefault ? 'on' : 'off'} by default) from ${path}`);
      })
      .catch((e) => log('warn', `Universal adjust preset not loaded: ${String(e)}`));
  }, [log]);

  const onScanFolder = useCallback(
    async (pick: boolean) => {
      if (pick) {
        const folder = await api.pickFolder(clipsFolder);
        if (folder) await scanFolder(folder);
      } else await scanFolder(clipsFolder);
    },
    [scanFolder, clipsFolder],
  );

  // ---- pipeline + export events (listeners resolve async: drop the ones that arrive after cleanup) ----
  useEffect(() => {
    let alive = true;
    const offs: Array<() => void> = [];
    const keep = (p: Promise<() => void>) =>
      void p.then((off) => {
        if (alive) offs.push(off);
        else off();
      });
    keep(
      onEvent<PipelineEventPayload>('pipeline://progress', (p) => {
        if (p.event === 'progress') send({ type: 'PROGRESS', progress: { stage: p.stage, pct: p.pct, message: p.message, clip: p.clip } });
      }),
    );
    keep(
      onEvent<PipelineEventPayload>('pipeline://log', (p) => {
        if (p.event === 'log') log(p.level, p.message);
      }),
    );
    keep(
      onEvent<AnalysisPayload>('pipeline://analysis', (p) => {
        applyAnalysis(p.analysis);
        send({ type: 'RESULT', result: p.analysis });
        log('info', `Analysis applied: ${p.analysis.clips.length} clips, ${p.analysis.timeline.tracks.reduce((n, t) => n + t.clips.length, 0)} timeline clips`);
      }),
    );
    keep(
      onEvent<PipelineEventPayload & { error?: string }>('pipeline://result', () => {
        if (!isTauri()) send({ type: 'DONE' });
      }),
    );
    keep(
      onEvent<PipelineExitPayload>('pipeline://exit', (p) => {
        if (p.cancelled) return;
        if (!p.success) {
          const msg = `Pipeline exited with code ${p.code ?? '?'}; see the log for details.`;
          send({ type: 'FAIL', error: msg });
          log('error', msg);
        } else {
          // success without an analysis payload (e.g. parse failure) still ends the run
          send({ type: 'DONE' });
        }
      }),
    );
    keep(onEvent<ExportProgressPayload>('export://progress', (p) => setExportJob({ id: p.jobId, pct: p.pct, message: p.message })));
    keep(
      onEvent<ExportDonePayload>('export://done', (p) => {
        setExportJob(null);
        setExportResult(p.ok ? { ok: true, text: `Saved ${basename(p.outPath)}`, path: p.outPath } : { ok: false, text: p.error ?? 'Export failed' });
        if (p.ok) log('info', `Export finished: ${basename(p.outPath)}`);
        else log('error', `Export failed: ${p.error ?? 'unknown error'}`);
      }),
    );
    keep(
      onEvent<{ jobId: string; level: 'info' | 'warn' | 'error'; message: string }>('export://log', (p) => {
        if (p.level !== 'info') log(p.level, `[export] ${p.message}`);
      }),
    );
    return () => {
      alive = false;
      offs.forEach((f) => f());
    };
  }, [send, log, applyAnalysis]);

  // ---- actions ----
  const importPaths = useCallback(
    async (files: string[]) => {
      if (!files.length) return;
      try {
        const imported = await api.importMedia(files);
        addAssets(imported);
        log('info', `Imported ${imported.length} asset(s)`);
      } catch (e) {
        log('error', `Import failed: ${String(e)}`);
      }
    },
    [addAssets, log],
  );

  const onImport = useCallback(async () => {
    const files = await api.pickMedia();
    await importPaths(files);
  }, [importPaths]);

  const runPipeline = useCallback(() => {
    const videoPaths = storyOrder(assets).map((a) => a.path);
    if (!videoPaths.length) {
      log('warn', 'Import some video clips first.');
      return;
    }
    log('info', `Starting pipeline on ${videoPaths.length} clip(s) in story order`);
    send({ type: 'START', paths: videoPaths, options });
  }, [assets, options, send, log]);

  const cancelPipeline = useCallback(() => send({ type: 'CANCEL' }), [send]);

  const onExport = useCallback(() => setExportOpen(true), []);

  // ---- keyboard shortcuts ----
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || isDialogOpen() || useSetup.getState().open) return;
      const target = e.target as HTMLElement | null;
      if (isTextEntry(target)) return;
      const onRange = target instanceof HTMLInputElement && target.type === 'range';
      const s = useEditor.getState();
      const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      const ctrl = e.ctrlKey || e.metaKey;
      const frame = 1000 / s.project.fps;
      const sel = s.selection.clipIds;
      const head = playClock.get();
      const handled = () => e.preventDefault();

      // ---- file / edit (Ctrl) ----
      if (ctrl && !e.altKey) {
        if (key === 's') return handled(), void saveProject(e.shiftKey);
        if (key === 'o') return handled(), void openProject();
        if (key === 'n') return handled(), void newProject();
        if (key === 'i') return handled(), void onImport();
        if (key === 'e') return handled(), setExportOpen(true);
        if (key === 'z' && !e.shiftKey) return handled(), s.undo();
        if (key === 'y' || (key === 'z' && e.shiftKey)) return handled(), s.redo();
        if (key === 'b') return handled(), void s.splitAtPlayhead();
        if (key === 'a') return handled(), s.selectAll();
        if (key === '=' || key === '+') return handled(), s.setZoom(s.zoom * 1.3);
        if (key === '-') return handled(), s.setZoom(s.zoom / 1.3);
        return;
      }
      if (ctrl && e.altKey) {
        if (key === 'c' && sel[0]) {
          handled();
          if (s.copyAttributes(sel[0])) log('info', 'Attributes copied: paste them with Ctrl+Alt+V');
          return;
        }
        if (key === 'v') {
          handled();
          if (!s.attrClipboard) log('warn', 'Copy attributes first (Ctrl+Alt+C on a clip).');
          else if (!sel.length) log('warn', 'Select the clips to paste the attributes on.');
          else setPasteOpen(true);
          return;
        }
        return;
      }
      if (e.altKey) {
        if (key === 'k') return handled(), s.setDrawer(!s.drawerOpen, 'keyframes');
        if (key === 'f' && sel[0]) return handled(), s.freezeFrameAt(sel[0], head);
        if (key === 'r' && sel[0]) return handled(), s.toggleReverse(sel[0]);
        return;
      }

      // ---- transport ----
      if (e.code === 'Space') {
        handled();
        if (!s.playing && head >= projectDurationMs(s.project) - 1) s.setPlayhead(0); // CapCut restarts at the end
        s.setPlaying(!s.playing);
        return;
      }
      if (key === 'k') return handled(), s.setShuttle(0);
      if (key === 'l') return handled(), s.setShuttle(s.playing && s.shuttle > 0 ? Math.min(4, s.shuttle * 2) : 1);
      if (key === 'j') return handled(), s.setShuttle(s.playing && s.shuttle < 0 ? Math.max(-4, s.shuttle * 2) : -1);
      if (!onRange) {
        if (key === 'ArrowLeft') return handled(), s.setPlayhead(head - (e.shiftKey ? frame * 10 : frame));
        if (key === 'ArrowRight') return handled(), s.setPlayhead(head + (e.shiftKey ? frame * 10 : frame));
        if (key === 'ArrowUp' || key === 'ArrowDown') {
          handled();
          const cuts = cutPoints(s.project);
          const next = key === 'ArrowUp' ? [...cuts].reverse().find((t) => t < head - 1) : cuts.find((t) => t > head + 1);
          if (next != null) s.setPlayhead(next);
          return;
        }
        if (key === 'Home') return handled(), s.setPlayhead(0);
        if (key === 'End') return handled(), s.setPlayhead(projectDurationMs(s.project));
      }

      // ---- edit ----
      if (key === 's' && !e.shiftKey) return handled(), void s.splitAtPlayhead();
      if ((key === 'Delete' || key === 'Backspace') && sel.length) return handled(), s.deleteClips(sel);
      if ((key === 'Delete' || key === 'Backspace') && s.selectedCut) {
        // a selected cut: remove its transition
        handled();
        return s.setTransition(s.selectedCut, null);
      }
      if (key === 'Escape') {
        handled();
        if (s.gestureBase) return s.cancelGesture();
        if (sheetOpen) return setSheetOpen(false);
        if (pasteOpen) return setPasteOpen(false);
        if (exportOpen) return setExportOpen(false);
        if (settingsOpen) return setSettingsOpen(false);
        if (s.selectedCut) return s.selectCut(null);
        if (s.drawerOpen) return s.setDrawer(false);
        if (s.largePreview) return s.toggleLargePreview();
        if (sel.length || s.selection.assetId) return s.select([]);
        return;
      }
      if (key === '?' || (key === '/' && e.shiftKey)) return handled(), setSheetOpen((v) => !v);

      // ---- toggles ----
      if (key === 'c') return handled(), s.toggleCompare();
      if (key === 'u') return handled(), s.setUniversalEnabled(!s.project.universalAdjust?.enabled);
      if (key === 'f') return handled(), s.toggleLargePreview();
      if (key === 'n') {
        handled();
        s.toggleSnapping();
        return log('info', `Snapping ${useEditor.getState().snapping ? 'on' : 'off'}`);
      }
      if (key === 'p') {
        handled();
        s.toggleRipple();
        return log('info', `Main-track magnet ${useEditor.getState().rippleEdit ? 'on' : 'off'}`);
      }
      if (key === 'z' && e.shiftKey) return handled(), s.requestFit();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onImport, log, sheetOpen, pasteOpen, exportOpen, settingsOpen]);

  const progress = state.context.progress;
  const workflowState = String(state.value);
  const selCount = useEditor((s) => s.selection.clipIds.length);
  const selLabel = useEditor((s) => (s.selection.clipIds[0] ? findClip(s.project, s.selection.clipIds[0])?.clip.label ?? null : null));

  const startPanelResize = (side: 'left' | 'right') => (e: React.PointerEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = side === 'left' ? leftWidth : rightWidth;
    const move = (ev: PointerEvent) => {
      const d = ev.clientX - startX;
      const w = Math.max(200, Math.min(520, side === 'left' ? startW + d : startW - d));
      setUi(side === 'left' ? { leftWidth: w } : { rightWidth: w });
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  return (
    <div className="app" style={{ gridTemplateRows: `${HEADER_H}px minmax(0, 1fr) auto auto ${STATUS_H}px` }}>
      <Header
        workflowState={workflowState}
        busy={busy}
        onRunPipeline={runPipeline}
        onExport={onExport}
        onImport={onImport}
        onPasteAttributes={() => setPasteOpen(true)}
        onShortcuts={() => setSheetOpen(true)}
        onProjectSettings={() => setSettingsOpen(true)}
      />
      <div className={`workspace ${largePreview ? 'large' : ''}`} style={largePreview ? undefined : { gridTemplateColumns: `${leftWidth}px minmax(0, 1fr) ${rightWidth}px` }}>
        <aside className="panel">
          <LibraryPanels onImport={onImport} onDropFiles={(p) => void importPaths(p)} onScanFolder={(pick) => void onScanFolder(pick)} />
          <div className="panel-resizer right" onPointerDown={startPanelResize('left')} title="Drag to resize" />
        </aside>
        <main>
          <Preview />
          <PlayerControls />
        </main>
        <aside className="panel">
          <div className="panel-resizer left" onPointerDown={startPanelResize('right')} title="Drag to resize" />
          <Inspector
            workflowState={workflowState}
            busy={busy}
            progress={progress}
            options={options}
            setOptions={setOptions}
            onRun={runPipeline}
            onCancel={cancelPipeline}
            error={state.context.error}
          />
        </aside>
        {restore ? <RestoreBanner entry={restore} onDone={() => setRestore(null)} /> : null}
      </div>
      <Timeline height={timelineH} />
      <KeyframeDrawer height={drawerH} />
      <ExportDialog
        open={exportOpen}
        onClose={() => setExportOpen(false)}
        job={exportJob}
        result={exportResult}
        onStarted={(id) => setExportJob({ id, pct: 0, message: 'starting…' })}
        onResult={(r) => {
          setExportResult(r);
          if (r && !r.ok) setExportJob(null);
        }}
      />
      <ShortcutsSheet open={sheetOpen} onClose={() => setSheetOpen(false)} />
      <ProjectSettings open={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <SetupWizard />
      <PasteAttributesDialog open={pasteOpen} onClose={() => setPasteOpen(false)} />
      <DialogHost />
      <footer className="status">
        <span>{isTauri() ? 'native' : 'browser mode'}</span>
        <span>ffmpeg: {paths ? (paths.ffmpeg ? 'ok' : 'missing') : '…'}</span>
        <span>pipeline: {paths ? (paths.python ? 'ok' : 'missing') : '…'}</span>
        {selCount ? <span>{selCount > 1 ? `${selCount} clips selected` : `selected: ${selLabel ?? 'clip'}`}</span> : null}
        {exportJob ? (
          <span className="status-export" onClick={() => setExportOpen(true)} title="Show the export">
            export {Math.round(exportJob.pct * 100)}% · {exportJob.message}
          </span>
        ) : null}
        <span className="spacer" />
        <span>
          <span className="kbd">Space</span> play · <span className="kbd">Ctrl+B</span> split · <span className="kbd">Shift+Z</span> fit ·{' '}
          <span className="kbd">Ctrl+S</span> save ·{' '}
          <button className="link" onClick={() => setSheetOpen(true)}>
            <span className="kbd">?</span> all shortcuts
          </button>
        </span>
      </footer>
    </div>
  );
}
