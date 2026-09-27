import { useState } from 'react';
import { useAiReady } from '@/state/setup';
import { useEditor } from '@/state/store';
import { WORKFLOW_STATES } from '@/state/workflowMachine';
import type { PipelineOptions } from '@/types/project';
import { Slider } from './ColorPanel';
import { fmtDuration } from '@/lib/format';
import { isWaitingForGpu } from '@/lib/tauri';

interface Props {
  workflowState: string;
  busy: boolean;
  progress: { stage: string; pct: number; message: string; clip: string | null } | null;
  options: PipelineOptions;
  setOptions: (o: PipelineOptions) => void;
  onRun: () => void;
  onCancel: () => void;
  error: string | null;
}

export default function AIPanel({ workflowState, busy, progress, options, setOptions, onRun, onCancel, error }: Props) {
  const assets = useEditor((s) => s.project.assets);
  const analysis = useEditor((s) => s.lastAnalysis);
  const logs = useEditor((s) => s.logs);
  const clearLogs = useEditor((s) => s.clearLogs);
  const setPlayhead = useEditor((s) => s.setPlayhead);
  const project = useEditor((s) => s.project);
  const select = useEditor((s) => s.select);
  const roughCut = useEditor((s) => s.roughCutInStoryOrder);
  const [promptDraft, setPromptDraft] = useState('');
  const ai = useAiReady();
  const videoCount = assets.filter((a) => a.kind === 'video').length;

  const idx = WORKFLOW_STATES.findIndex((s) => s.id === workflowState);
  const set = (patch: Partial<PipelineOptions>) => setOptions({ ...options, ...patch });

  const dupCount = analysis?.clips.reduce((n, c) => n + c.duplicates.length, 0) ?? 0;
  const shotCount = analysis?.clips.reduce((n, c) => n + c.shots.length, 0) ?? 0;

  const jumpToFinding = (path: string, timeMs: number) => {
    // find the clip on the timeline that covers this source time
    for (const t of project.tracks) {
      if (t.kind !== 'video') continue;
      for (const c of t.clips) {
        const a = project.assets.find((x) => x.id === c.assetId);
        if (a?.path === path && timeMs >= c.inMs && timeMs <= c.outMs) {
          select([c.id]);
          setPlayhead(c.startMs + (timeMs - c.inMs));
          return;
        }
      }
    }
  };

  return (
    <div>
      <div className="section">
        <h4>Open-vocabulary prompts</h4>
        {options.prompts.length === 0 ? (
          <div className="hint" style={{ marginBottom: 6 }}>
            Using the main cast from <code>characters/</code>: Suzie, Raccoon, Turtle, Felix, Bunny, Otter. Add prompts only for extra characters.
          </div>
        ) : null}
        <div className="tag-list" style={{ marginBottom: 6 }}>
          {options.prompts.map((p) => (
            <span key={p} className="tag">
              {p}
              <button onClick={() => set({ prompts: options.prompts.filter((x) => x !== p) })}>×</button>
            </span>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 4 }}>
          <input
            type="text"
            placeholder="e.g. raccoon in hoodie"
            value={promptDraft}
            onChange={(e) => setPromptDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && promptDraft.trim()) {
                set({ prompts: [...new Set([...options.prompts, promptDraft.trim()])] });
                setPromptDraft('');
              }
            }}
            style={{ flex: 1 }}
          />
          <button className="small" onClick={() => { if (promptDraft.trim()) { set({ prompts: [...new Set([...options.prompts, promptDraft.trim()])] }); setPromptDraft(''); } }}>
            Add
          </button>
        </div>
      </div>

      <div className="section">
        <h4>Perception</h4>
        <div className="row">
          <label>Detector</label>
          <select value={options.detector} onChange={(e) => set({ detector: e.target.value as PipelineOptions['detector'] })} style={{ gridColumn: '2 / 4' }}>
            <option value="hybrid">Hybrid (YOLO-World → Grounded SAM 2)</option>
            <option value="yolo_world">YOLO-World + ByteTrack</option>
            <option value="grounded_sam2">Grounded SAM 2 (high VRAM)</option>
          </select>
        </div>
        <div className="row">
          <label>Shot detector</label>
          <select value={options.shotDetector} onChange={(e) => set({ shotDetector: e.target.value as PipelineOptions['shotDetector'] })} style={{ gridColumn: '2 / 4' }}>
            <option value="auto">Auto (TransNetV2 if available)</option>
            <option value="transnetv2">TransNetV2 (ONNX)</option>
            <option value="pyscenedetect">PySceneDetect (CPU prior)</option>
          </select>
        </div>
        <Slider label="Similarity τ" value={Math.round(options.similarityThreshold * 100)} min={50} max={99} onChange={(v) => set({ similarityThreshold: v / 100 })} fmt={(v) => (v / 100).toFixed(2)} />
        <div className="row">
          <label>Smoothing</label>
          <select value={options.smoothing} onChange={(e) => set({ smoothing: e.target.value as PipelineOptions['smoothing'] })}>
            <option value="ema">EMA</option>
            <option value="savgol">Savitzky-Golay</option>
          </select>
          <input type="number" min={0.02} max={0.9} step={0.01} value={options.smoothingAlpha} onChange={(e) => set({ smoothingAlpha: Number(e.target.value) })} />
        </div>
      </div>

      <div className="section">
        <h4>Assembly & audio</h4>
        <Slider label="Target length" value={Math.round(options.targetDurationMs / 1000)} min={30} max={300} onChange={(v) => set({ targetDurationMs: v * 1000 })} fmt={(v) => fmtDuration(v * 1000)} />
        <Slider label="Target LUFS" value={options.targetLufs} min={-24} max={-8} onChange={(v) => set({ targetLufs: v })} />
        <div className="row">
          <label>Normalize</label>
          <input type="checkbox" checked={options.normalizeAudio} onChange={(e) => set({ normalizeAudio: e.target.checked })} />
          <span />
        </div>
        <div className="row">
          <label>Beat markers</label>
          <input type="checkbox" checked={options.detectBeats} onChange={(e) => set({ detectBeats: e.target.checked })} />
          <span />
        </div>
      </div>

      <div className="section">
        {busy ? (
          <button className="primary" style={{ width: '100%' }} onClick={onCancel}>
            Cancel pipeline
          </button>
        ) : (
          <button className="primary" style={{ width: '100%' }} onClick={onRun} disabled={videoCount === 0 || !ai.ready} title={ai.hint || undefined}>
            ▶ Run AI pipeline on {videoCount} clip{videoCount === 1 ? '' : 's'}
          </button>
        )}
        {!busy ? (
          <button style={{ width: '100%', marginTop: 6 }} onClick={roughCut} disabled={videoCount === 0} title="Lay every clip on the timeline in story order without AI analysis">
            Rough cut in story order (no AI)
          </button>
        ) : null}
        {!ai.ready ? <div className="hint" style={{ marginTop: 6 }}>{ai.hint}</div> : null}
        {progress && busy && isWaitingForGpu(progress.message) ? (
          <>
            <div className="progress waiting">
              <div style={{ width: '100%' }} />
            </div>
            <div className="gpu-wait">
              <span className="dot" />
              Queued: waiting for the GPU (another AI job, e.g. voice separation, is running). The analysis starts automatically.
            </div>
          </>
        ) : progress ? (
          <>
            <div className="progress">
              <div style={{ width: `${Math.round(progress.pct * 100)}%` }} />
            </div>
            <div className="hint">
              {progress.stage}: {progress.message} {progress.clip ? `· ${progress.clip.split(/[\\/]/).pop()}` : ''}
            </div>
          </>
        ) : null}
        {error ? <div className="finding" style={{ borderLeftColor: 'var(--red)' }}>{error}</div> : null}
        <div className="steps">
          {WORKFLOW_STATES.map((s, i) => (
            <div key={s.id} className={`step ${i < idx || workflowState === 'ready' ? 'done' : ''} ${s.id === workflowState ? 'active' : ''}`}>
              <span className="dot" />
              {s.label}
            </div>
          ))}
        </div>
      </div>

      {analysis ? (
        <div className="section">
          <h4>Findings</h4>
          <div className="hint" style={{ marginBottom: 6 }}>
            {shotCount} shots · {dupCount} duplicate collisions · {analysis.clips.filter((c) => c.audio).length} audio analyses
          </div>
          {analysis.clips.flatMap((c) =>
            c.duplicates.map((d, i) => (
              <div key={`${c.path}-${i}`} className="finding" onClick={() => jumpToFinding(c.path, d.timeMs)} style={{ cursor: 'pointer' }}>
                <div>
                  <b>{d.characterName ?? d.duplicate.label}</b> {d.characterName ? 'appears twice' : 'duplicated'} · sim {(d.similarity * 100).toFixed(0)}%
                </div>
                <div className="t">
                  {c.path.split(/[\\/]/).pop()} · shot {d.shotIndex + 1} · {(d.timeMs / 1000).toFixed(2)}s · reframe {c.reframe.some((r) => r.shotIndex === d.shotIndex) ? 'applied' : 'n/a'}
                </div>
              </div>
            )),
          )}
          {analysis.clips.flatMap((c) =>
            c.transitions
              .filter((t) => t.suggestion === 'dissolve')
              .map((t, i) => (
                <div key={`${c.path}-tr-${i}`} className="finding" style={{ borderLeftColor: 'var(--info)' }}>
                  Rough cut: shot {t.fromShot + 1} → {t.toShot >= c.shots.length ? 'next clip' : `shot ${t.toShot + 1}`} (smoothness {(t.smoothness * 100).toFixed(0)}%) · suggest dissolve
                </div>
              )),
          )}
          {analysis.clips.map((c) =>
            c.audio ? (
              <div key={`${c.path}-au`} className="finding" style={{ borderLeftColor: 'var(--green)' }}>
                {c.path.split(/[\\/]/).pop()}: {c.audio.integratedLufs.toFixed(1)} LUFS → gain {c.audio.recommendedGainDb >= 0 ? '+' : ''}
                {c.audio.recommendedGainDb.toFixed(1)} dB · {c.audio.beats.length} beats{c.audio.tempoBpm ? ` · ${Math.round(c.audio.tempoBpm)} BPM` : ''}
              </div>
            ) : null,
          )}
        </div>
      ) : null}

      <div className="section">
        <h4>
          Log
          <button className="small ghost" onClick={clearLogs}>
            Clear
          </button>
        </h4>
        <div className="log">
          {logs.length ? logs.slice(-80).map((l, i) => (
            <div key={i} className={l.level}>
              {new Date(l.ts).toLocaleTimeString()} {l.message}
            </div>
          )) : <span className="hint">No messages yet.</span>}
        </div>
      </div>
    </div>
  );
}
