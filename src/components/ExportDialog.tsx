import { useState } from 'react';
import { clipEndMs, projectDurationMs, useEditor } from '@/state/store';
import { playClock } from '@/state/clock';
import { api, isTauri, type ExportPreset } from '@/lib/tauri';
import { basename, fmtDuration, timecode } from '@/lib/format';
import { CloseIcon } from './Overlays';
import { FrameRateFields, INTERPOLATION_NOTE } from './ProjectSettings';
import { frameInterpolationOf } from '@/engine/defaults';
import type { FrameInterpolation } from '@/types/project';

type RangeMode = 'all' | 'selection' | 'preview';

export interface ExportJob {
  id: string;
  pct: number;
  message: string;
}

export interface ExportResult {
  ok: boolean;
  text: string;
  path?: string;
}

interface Props {
  open: boolean;
  onClose: () => void;
  /** the running export (tracked by App so it survives closing the dialog) */
  job: ExportJob | null;
  result: ExportResult | null;
  onStarted: (id: string) => void;
  onResult: (r: ExportResult | null) => void;
}

/** The dialog body only mounts while open, so it never re-renders with the playhead otherwise. */
export default function ExportDialog(props: Props) {
  if (!props.open) return null;
  return <ExportDialogBody {...props} />;
}

function ExportDialogBody({ onClose, job, result, onStarted, onResult }: Props) {
  const project = useEditor((s) => s.project);
  const selection = useEditor((s) => s.selection);
  const log = useEditor((s) => s.log);
  const [playheadMs] = useState(() => playClock.get());
  const [preset, setPreset] = useState<ExportPreset>('h264_nvenc_mp4');
  const [rangeMode, setRangeMode] = useState<RangeMode>('all');
  const [outPath, setOutPath] = useState<string | null>(null);
  // frame rate / interpolation: the project's settings, overridable for this export
  const [fps, setFps] = useState(project.fps);
  const [interp, setInterp] = useState<FrameInterpolation>(frameInterpolationOf(project));
  const overridden = fps !== project.fps || interp !== frameInterpolationOf(project);

  const total = projectDurationMs(project);
  const selClip = project.tracks.flatMap((t) => t.clips).find((c) => c.id === selection.clipIds[0]);
  const range =
    rangeMode === 'selection' && selClip
      ? { startMs: selClip.startMs, endMs: clipEndMs(selClip) }
      : rangeMode === 'preview'
        ? { startMs: Math.min(playheadMs, Math.max(0, total - 1000)), endMs: Math.min(total, playheadMs + 10_000) }
        : null;
  const rangeDur = range ? range.endMs - range.startMs : total;

  const ext = preset === 'prores_mov' ? 'mov' : 'mp4';
  const chooseOut = async () => {
    const p = await api.pickSavePath(`${project.name || 'export'}${rangeMode === 'preview' ? '_preview' : ''}`, ext);
    if (p) setOutPath(p);
  };

  const start = async () => {
    let path = outPath;
    if (!path) {
      path = await api.pickSavePath(`${project.name || 'export'}${rangeMode === 'preview' ? '_preview' : ''}`, ext);
      if (!path) return;
      setOutPath(path);
    }
    onResult(null);
    try {
      const id = await api.exportProject({ ...project, fps, frameInterpolation: interp }, path, preset, range);
      onStarted(id);
      log('info', `Export started: ${basename(path)} (${preset}, ${fps} fps, ${interp}${range ? `, ${fmtDuration(rangeDur)}` : ''})`);
    } catch (e) {
      onResult({ ok: false, text: String(e) });
    }
  };

  const cancel = async () => {
    if (job) await api.cancelExport(job.id);
    onResult({ ok: false, text: 'Export cancelled' });
  };

  const reveal = async () => {
    if (!result?.path || !isTauri()) return;
    try {
      const { revealItemInDir } = await import('@tauri-apps/plugin-opener');
      await revealItemInDir(result.path);
    } catch (e) {
      log('warn', `Could not open folder: ${String(e)}`);
    }
  };

  const hasClips = project.tracks.some((t) => t.clips.length > 0);

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="modal" role="dialog" aria-label="Export" onMouseDown={(e) => e.stopPropagation()}>
        <h3>
          Export
          <button className="small ghost" onClick={onClose} title={job ? 'Keep exporting in the background (Esc)' : 'Close (Esc)'} aria-label="Close">
            <CloseIcon />
          </button>
        </h3>
        <div className="export-grid">
          <label>Format</label>
          <select value={preset} onChange={(e) => setPreset(e.target.value as ExportPreset)} disabled={!!job}>
            <option value="h264_nvenc_mp4">H.264 MP4 · NVIDIA GPU encoder (fast)</option>
            <option value="h264_mp4">H.264 MP4 · CPU x264 (compatible)</option>
            <option value="prores_mov">ProRes 422 HQ MOV · for further editing</option>
          </select>
          <label>Range</label>
          <select value={rangeMode} onChange={(e) => setRangeMode(e.target.value as RangeMode)} disabled={!!job}>
            <option value="all">Whole timeline ({fmtDuration(total)})</option>
            <option value="selection" disabled={!selClip}>
              Selected clip{selClip ? ` (${fmtDuration(clipEndMs(selClip) - selClip.startMs)})` : ''}
            </option>
            <option value="preview">10 s preview from playhead ({timecode(playheadMs, project.fps)})</option>
          </select>
          <label>Output</label>
          <div style={{ display: 'flex', gap: 6, minWidth: 0 }}>
            <span className="hint" style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={outPath ?? ''}>
              {outPath ? basename(outPath) : 'choose when exporting'}
            </span>
            <button className="small" onClick={chooseOut} disabled={!!job}>
              Browse…
            </button>
          </div>
          <FrameRateFields fps={fps} interpolation={interp} onFps={setFps} onInterpolation={setInterp} disabled={!!job} />
          <label>Video</label>
          <span className="hint">
            {project.width}×{project.height} @ {fps} fps{overridden ? ' (this export only)' : ''} · grade, keyframes, masks, speed ramps, transitions, effects, fades and loudness normalisation are rendered
          </span>
        </div>
        <div className="hint" style={{ marginTop: 8 }}>
          {INTERPOLATION_NOTE}
        </div>

        {job ? (
          <div style={{ marginTop: 12 }}>
            <div className="progress">
              <div style={{ width: `${Math.round(job.pct * 100)}%` }} />
            </div>
            <div className="hint">
              {Math.round(job.pct * 100)}% · {job.message}
            </div>
          </div>
        ) : null}
        {result ? (
          <div className="finding" style={{ borderLeftColor: result.ok ? 'var(--green)' : 'var(--red)', marginTop: 12 }}>
            {result.text}
            {result.ok && result.path ? (
              <button className="small" style={{ marginLeft: 8 }} onClick={reveal}>
                Show in folder
              </button>
            ) : null}
          </div>
        ) : null}

        <div className="actions">
          {job ? (
            <>
              <button onClick={cancel}>Cancel export</button>
              <button className="primary" onClick={onClose} title="The footer keeps showing the progress">
                Run in background
              </button>
            </>
          ) : (
            <>
              <button onClick={onClose}>Close</button>
              <button className="primary" onClick={start} disabled={!hasClips}>
                Export
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
