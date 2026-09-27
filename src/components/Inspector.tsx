import { clipDurationMs, findClip, useEditor, useSelectedClip, type InspectorTab, type KeyframeProperty } from '@/state/store';
import { evaluate } from '@/engine/keyframes';
import { SPEED_PRESETS, averageSpeed, hasSlowMotion, outputDuration, presetCurve } from '@/engine/speed';
import { QUICK_SPEEDS, keyframed } from '@/engine/defaults';
import type { BlendMode, EffectType, MaskShape, PipelineOptions, SpeedPreset, VoiceMode } from '@/types/project';
import { clampedAudioFades, keepPitchOf, levelDbAt, volumeDbAt } from '@/engine/audio';
import { EFFECT_INFO, effectInfo, effectParam } from '@/engine/effects';
import { TRANSITION_MAX_MS, TRANSITION_MIN_MS, TRANSITION_TYPES, clampTransitionMs, transitionLabel } from '@/engine/transitions';
import { frameMs, prevAdjacent } from '@/state/edits';
import { canSeparateToTracks, hasStemClips, separableClipIds } from '@/state/stemTracks';
import { separateClipsToTracks } from '@/state/separation';
import { useAiReady } from '@/state/setup';
import { confirmAction } from './Dialogs';
import {
  VOICE_MODES,
  assetHasSound,
  assetsNeedingStems,
  audioBearingClips,
  mirrorClip,
  separationStatus,
  voiceMode,
  type SeparationStatus,
} from '@/engine/voice';
import { cancelSeparation, requestSeparation } from '@/state/separation';
import ColorPanel, { Slider, beginControlGesture } from './ColorPanel';
import { isWaitingForGpu } from '@/lib/tauri';
import AIPanel from './AIPanel';
import UniversalAdjustPanel from './UniversalAdjustPanel';

const TABS: Array<{ id: InspectorTab; label: string }> = [
  { id: 'ai', label: 'AI' },
  { id: 'color', label: 'Color' },
  { id: 'speed', label: 'Speed' },
  { id: 'transform', label: 'Motion' },
  { id: 'audio', label: 'Audio' },
  { id: 'mask', label: 'Mask' },
];
const EFFECT_TAB = { id: 'effect' as InspectorTab, label: 'Effect' };
const TRANSITION_TAB = { id: 'transition' as InspectorTab, label: 'Transition' };

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

export default function Inspector(props: Props) {
  const tab = useEditor((s) => s.inspectorTab);
  const setTab = useEditor((s) => s.setInspectorTab);
  const sel = useSelectedClip();
  const assets = useEditor((s) => s.project.assets);
  const selectedCut = useEditor((s) => s.selectedCut);
  const isEffect = !!sel?.clip.effect;
  const tabs = isEffect ? [TABS[0], EFFECT_TAB] : selectedCut && !sel ? [TABS[0], TRANSITION_TAB, ...TABS.slice(1)] : TABS;

  return (
    <>
      <div className="panel-title">
        <span>Inspector</span>
        <span className="hint panel-sub" title={sel?.clip.label}>
          {sel ? sel.clip.label : selectedCut ? 'cut selected' : 'no clip selected'}
        </span>
      </div>
      <div className="tabs">
        {tabs.map((t) => (
          <button key={t.id} className={tab === t.id || (isEffect && t.id === 'effect' && tab !== 'ai') ? 'active' : ''} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="panel-body">
        {tab === 'ai' ? (
          <AIPanel {...props} />
        ) : isEffect ? (
          <EffectTab />
        ) : tab === 'transition' ? (
          <TransitionTab />
        ) : tab === 'effect' ? (
          <div className="hint">Select an effect clip on an FX track, or add one from the Effects tab on the left.</div>
        ) : tab === 'color' && !sel ? (
          <>
            <UniversalAdjustPanel />
            <div className="hint">Select a clip to grade it individually on top of the universal adjust.</div>
          </>
        ) : !sel ? (
          <div className="hint">Select a clip on the timeline to edit its {tab} settings.</div>
        ) : tab === 'color' ? (
          <ColorPanel clip={sel.clip} assets={assets} />
        ) : tab === 'speed' ? (
          <SpeedTab />
        ) : tab === 'transform' ? (
          <TransformTab />
        ) : tab === 'audio' ? (
          <AudioTab />
        ) : (
          <MaskTab />
        )}
      </div>
    </>
  );
}

function SpeedTab() {
  const sel = useSelectedClip();
  const setClipSpeed = useEditor((s) => s.setClipSpeed);
  const setDrawer = useEditor((s) => s.setDrawer);
  if (!sel) return null;
  const { clip } = sel;
  const src = clip.outMs - clip.inMs;
  const avg = averageSpeed(clip.speed);
  const uniform = clip.speed.points.every((p) => Math.abs(p.speed - (clip.speed.points[0]?.speed ?? 1)) < 1e-9);
  return (
    <div>
      <div className="section">
        <h4>Speed</h4>
        <div className="speed-grid" role="group" aria-label="Quick speed">
          {QUICK_SPEEDS.map((v) => (
            <button key={v} className={`small ${uniform && Math.abs(avg - v) < 0.005 ? 'active' : ''}`} onClick={() => useEditor.getState().setConstantSpeed(useEditor.getState().selection.clipIds.length ? useEditor.getState().selection.clipIds : [clip.id], v)} title={`Constant ${v}× (later clips ripple)`}>
              {v}×
            </button>
          ))}
        </div>
      </div>
      <div className="section">
        <h4>Preset speed curves</h4>
        <div className="chips">
          {(Object.keys(SPEED_PRESETS) as SpeedPreset[]).map((p) => (
            <button key={p} className={`small ${clip.speed.preset === p ? 'active' : ''}`} onClick={() => setClipSpeed(clip.id, presetCurve(p, clip.speed.opticalFlow))}>
              {p.replace('_', ' ')}
            </button>
          ))}
          <button className={`small ${clip.speed.preset === 'custom' ? 'active' : ''}`} onClick={() => setDrawer(true, 'speed')}>
            custom…
          </button>
        </div>
      </div>
      <div className="section">
        <h4>Constant speed</h4>
        <Slider
          label="Speed"
          value={Number(averageSpeed(clip.speed).toFixed(2))}
          min={0.1}
          max={10}
          step={0.05}
          onChange={(v) => setClipSpeed(clip.id, { ...clip.speed, preset: 'custom', points: [{ t: 0, speed: v }, { t: 1, speed: v }] })}
          fmt={(v) => `${v.toFixed(2)}×`}
        />
      </div>
      <div className="section">
        <label
          style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 12 }}
          title="Optical flow: when this clip plays slower than its source frame rate (slow motion, or a project frame rate above the clip's), the export synthesises real in-between frames with RAFT instead of repeating or blending frames. Slower to export; the preview always shows the source frames."
        >
          <input type="checkbox" checked={clip.speed.opticalFlow} onChange={(e) => setClipSpeed(clip.id, { ...clip.speed, opticalFlow: e.target.checked })} />
          Optical flow (RAFT) for slow motion
        </label>
        <div className="hint" style={{ marginTop: 6 }}>
          Source {(src / 1000).toFixed(2)}s → output {(outputDuration(clip.speed, src) / 1000).toFixed(2)}s
          {hasSlowMotion(clip.speed) ? (clip.speed.opticalFlow ? ' · slow-mo will be interpolated at export' : ' · slow-mo will duplicate frames') : ''}
        </div>
        <button className="small" style={{ marginTop: 8 }} onClick={() => setDrawer(true, 'speed')}>
          Open speed graph
        </button>
      </div>
    </div>
  );
}

const TRANSFORM_ROWS: Array<{ key: KeyframeProperty; label: string; min: number; max: number; step: number; comp?: number }> = [
  { key: 'position', label: 'Position X', min: -1, max: 1, step: 0.005, comp: 0 },
  { key: 'position', label: 'Position Y', min: -1, max: 1, step: 0.005, comp: 1 },
  { key: 'scale', label: 'Scale', min: 0.1, max: 3, step: 0.01 },
  { key: 'rotation', label: 'Rotation', min: -180, max: 180, step: 1 },
  { key: 'opacity', label: 'Opacity', min: 0, max: 1, step: 0.01 },
  { key: 'blur', label: 'Blur', min: 0, max: 40, step: 0.5 },
];

function TransformTab() {
  const sel = useSelectedClip();
  const playheadMs = useEditor((s) => s.playheadMs);
  const setStatic = useEditor((s) => s.setTransformStatic);
  const setKf = useEditor((s) => s.setTransformKeyframe);
  const removeKf = useEditor((s) => s.removeTransformKeyframe);
  const updateClip = useEditor((s) => s.updateClip);
  const setDrawer = useEditor((s) => s.setDrawer);
  if (!sel) return null;
  const { clip } = sel;
  // keyframes live inside the clip: clamp the playhead to its length
  const local = Math.max(0, Math.min(clipDurationMs(clip), playheadMs - clip.startMs));

  return (
    <div>
      <div className="section">
        <h4>
          Transform
          <button className="small ghost" onClick={() => updateClip(clip.id, { transform: { position: keyframed([0, 0]), scale: keyframed(1), rotation: keyframed(0), opacity: keyframed(1), blur: keyframed(0) } })}>
            Reset
          </button>
        </h4>
        {TRANSFORM_ROWS.map((r) => {
          const kf = clip.transform[r.key];
          const cur = evaluate(kf as never, local) as number | number[];
          const value = Array.isArray(cur) ? cur[r.comp ?? 0] : cur;
          const hasKeys = kf.keyframes.length > 0;
          const atHead = kf.keyframes.some((k) => Math.abs(k.timeMs - local) < 0.5);
          const apply = (v: number) => {
            const full = Array.isArray(cur) ? cur.map((x, i) => (i === (r.comp ?? 0) ? v : x)) : v;
            if (hasKeys) setKf(clip.id, r.key, local, full as never);
            else setStatic(clip.id, r.key, full as never);
          };
          return (
            <div className="row" key={r.label}>
              <label style={{ display: 'flex', alignItems: 'center' }}>
                <button className={`kf ${atHead ? 'on' : hasKeys ? 'has' : ''}`} title="Toggle keyframe at playhead" onClick={() => (atHead ? removeKf(clip.id, r.key, local) : setKf(clip.id, r.key, local, cur as never))} />
                <span style={{ marginLeft: 6 }}>{r.label}</span>
              </label>
              <input type="range" min={r.min} max={r.max} step={r.step} value={value} onPointerDown={beginControlGesture} onChange={(e) => apply(Number(e.target.value))} aria-label={r.label} />
              <span className="value">{value.toFixed(r.step >= 1 ? 0 : 2)}</span>
            </div>
          );
        })}
        <button className="small" style={{ marginTop: 6 }} onClick={() => setDrawer(true, 'keyframes')}>
          Open keyframe graph <span className="kbd">Alt+K</span>
        </button>
      </div>
      {sel.track.kind === 'video' ? <VideoFadeSection /> : null}
      <div className="section">
        <h4>Blend mode</h4>
        <select value={clip.blendMode} onChange={(e) => updateClip(clip.id, { blendMode: e.target.value as BlendMode })} style={{ width: '100%' }}>
          {(['normal', 'multiply', 'screen', 'overlay', 'softLight', 'darken', 'lighten', 'colorDodge'] as BlendMode[]).map((b) => (
            <option key={b} value={b}>
              {b}
            </option>
          ))}
        </select>
        <div className="hint" style={{ marginTop: 4 }}>
          Blend modes composite against underlying video tracks at export.
        </div>
      </div>
      {clip.reframe ? (
        <div className="section">
          <h4>
            AI reframe
            <button className="small ghost" onClick={() => updateClip(clip.id, { reframe: null })}>
              Remove
            </button>
          </h4>
          <div className="hint">
            {clip.reframe.keyframes.length} camera keyframes · zoom ×{clip.reframe.keyframes[0]?.zoom.toFixed(2)}
            {clip.reframe.reason ? <div>{clip.reframe.reason}</div> : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function AudioTab() {
  const sel = useSelectedClip();
  const setAudio = useEditor((s) => s.setClipAudio);
  const beats = useEditor((s) => s.project.beatMarkers);
  const playheadMs = useEditor((s) => s.playheadMs);
  if (!sel) return null;
  const { clip, asset } = sel;
  const dur = clipDurationMs(clip);
  const local = Math.max(0, Math.min(dur, playheadMs - clip.startMs));
  const fades = clampedAudioFades(clip.audio, dur);
  const keys = clip.audio.volume?.keyframes ?? [];
  const atHead = keys.some((k) => Math.abs(k.timeMs - local) < 1);
  const level = levelDbAt(clip.audio, local);
  const st = useEditor.getState;
  const setVolume = (v: number) => {
    if (keys.length) st().setVolumeKeyframe(clip.id, local, v - clip.audio.gainDb);
    else setAudio(clip.id, { gainDb: v });
  };
  const dbFmt = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(1)} dB`;
  return (
    <div>
      {asset?.stemOf ? (
        <div className="section">
          <h4>Stem</h4>
          <div className="hint">
            {asset.stemOf.stem === 'vocals' ? 'Voice' : 'Background'} stem of {useEditor.getState().project.assets.find((a) => a.id === asset.stemOf!.assetId)?.name ?? 'its source'}: its own volume, fades, keyframes and mute. Moves, trims and speed follow the linked video clip.
          </div>
        </div>
      ) : (
        <>
          <VoiceSection />
          <SeparateTracksSection />
        </>
      )}
      <div className="section">
        <h4>Volume</h4>
        <div className="row">
          <label style={{ display: 'flex', alignItems: 'center' }} title="Volume keyframe at the playhead (Alt+click the volume line on the timeline)">
            <button
              className={`kf ${atHead ? 'on' : keys.length ? 'has' : ''}`}
              aria-label="Toggle volume keyframe at the playhead"
              title="Toggle a volume keyframe at the playhead"
              onClick={() => (atHead ? st().removeVolumeKeyframe(clip.id, local) : st().setVolumeKeyframe(clip.id, local, volumeDbAt(clip.audio, local)))}
            />
            <span style={{ marginLeft: 6 }} onDoubleClick={() => setVolume(0)}>
              Volume
            </span>
          </label>
          <input type="range" min={-30} max={12} step={0.5} value={Math.round(level * 2) / 2} onPointerDown={beginControlGesture} onChange={(e) => setVolume(Number(e.target.value))} aria-label="Volume (dB)" />
          <span className="value">{dbFmt(level)}</span>
        </div>
        {keys.length ? (
          <div className="hint" style={{ marginBottom: 4 }}>
            {keys.length} volume keyframe{keys.length === 1 ? '' : 's'} · the slider sets the one at the playhead ·{' '}
            <button className="link accent-link" onClick={() => setAudio(clip.id, { volume: undefined })}>
              clear
            </button>
          </div>
        ) : null}
        <Slider label="Fade in" value={Math.round(fades.fadeIn / 100) / 10} min={0} max={Math.max(0.1, Math.floor(dur / 200) / 10)} step={0.1} onChange={(v) => setAudio(clip.id, { fadeInMs: Math.round(v * 1000) })} fmt={(v) => `${v.toFixed(1)} s`} />
        <Slider label="Fade out" value={Math.round(fades.fadeOut / 100) / 10} min={0} max={Math.max(0.1, Math.floor(dur / 200) / 10)} step={0.1} onChange={(v) => setAudio(clip.id, { fadeOutMs: Math.round(v * 1000) })} fmt={(v) => `${v.toFixed(1)} s`} />
        <label className="check-row" title="On: speed changes keep the voice's pitch (time-stretch). Off: the pitch follows the speed, like a tape (CapCut: Pitch).">
          <input type="checkbox" checked={keepPitchOf(clip.audio)} onChange={(e) => setAudio(clip.id, { keepPitch: e.target.checked })} />
          Keep pitch when the speed changes
        </label>
        <label className="check-row">
          <input type="checkbox" checked={clip.audio.normalize} onChange={(e) => setAudio(clip.id, { normalize: e.target.checked })} />
          Loudness normalize with look-ahead limiter at export
        </label>
        <label className="check-row">
          <input type="checkbox" checked={clip.audio.muted} onChange={(e) => setAudio(clip.id, { muted: e.target.checked })} />
          Mute
        </label>
        <div className="chips" style={{ marginTop: 8 }}>
          <button className="small" onClick={() => setAudio(clip.id, { gainDb: 12, normalize: true })}>
            Boost +12 dB
          </button>
          <button className="small" onClick={() => setAudio(clip.id, { gainDb: 0, fadeInMs: 0, fadeOutMs: 0, volume: undefined })}>
            Reset
          </button>
        </div>
        {clip.linkId && !asset?.stemOf ? <div className="hint" style={{ marginTop: 6 }}>Linked clip: volume, fades, mute and voice mode apply to the video and its audio together (stem clips keep their own).</div> : null}
      </div>
      <div className="section">
        <h4>Rhythm</h4>
        <div className="hint">
          {beats.length} beat markers on the timeline ({beats.filter((b) => b.kind === 'beat1').length} strong). Snapping aligns clip edges to them.
        </div>
      </div>
    </div>
  );
}

const GPU_WAIT = 'Waiting for the GPU (another AI job is running)';

function statusText(st: SeparationStatus, waitingGpu: boolean): string {
  switch (st.kind) {
    case 'ready':
      return 'Stems ready';
    case 'running':
      return waitingGpu ? GPU_WAIT : `Separating ${Math.round(st.pct * 100)} %`;
    case 'queued':
      return waitingGpu ? GPU_WAIT : 'Queued for separation';
    case 'error':
      return `Separation failed: ${st.message}`;
    default:
      return 'Not separated';
  }
}

/** CapCut-style voice separation: Original · Isolate voice · Remove vocals. */
function VoiceSection() {
  const sel = useSelectedClip();
  const project = useEditor((s) => s.project);
  const jobs = useEditor((s) => s.separationJobs);
  const errors = useEditor((s) => s.separationErrors);
  const setClipVoice = useEditor((s) => s.setClipVoice);
  const setVoiceForAll = useEditor((s) => s.setVoiceForAll);
  const log = useEditor((s) => s.log);
  const aiState = useAiReady();
  if (!sel) return null;
  const { clip, asset } = sel;
  const mode = voiceMode(clip);
  const sound = assetHasSound(asset);
  const status = separationStatus(asset, jobs, errors);
  const linked = mirrorClip(project, clip);
  const running = Object.values(jobs);
  const statusJob = status.kind === 'running' || status.kind === 'queued' ? jobs[status.jobId] : undefined;
  const waitingGpu = isWaitingForGpu(statusJob?.message);

  const choose = (m: VoiceMode) => {
    setClipVoice(clip.id, m);
    if (m !== 'original' && asset && sound && !asset.stems && status.kind !== 'running' && status.kind !== 'queued' && aiState.ready) {
      void requestSeparation([asset.path]);
    }
  };
  const applyAll = () => {
    const changed = setVoiceForAll(mode);
    const p = useEditor.getState().project;
    const need = mode === 'original' ? [] : assetsNeedingStems(p, audioBearingClips(p)).map((a) => a.path);
    log('info', `Audio mode "${VOICE_MODES.find((v) => v.id === mode)?.label}" applied to ${changed} clip(s)`);
    if (need.length && aiState.ready) void requestSeparation(need);
  };

  return (
    <div className="section">
      <h4>Voice separation</h4>
      {!sound ? (
        <div className="hint">This clip has no audio.</div>
      ) : (
        <>
          <div role="radiogroup" aria-label="Audio mode" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 4 }}>
            {VOICE_MODES.map((m) => (
              <button
                key={m.id}
                role="radio"
                aria-checked={mode === m.id}
                title={m.hint}
                className={`small ${mode === m.id ? 'active' : ''}`}
                style={{ padding: '4px 2px' }}
                onClick={() => choose(m.id)}
              >
                {m.label}
              </button>
            ))}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8, fontSize: 12 }}>
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: 4,
                flex: 'none',
                background:
                  status.kind === 'ready' ? 'var(--accent)' : status.kind === 'error' ? 'var(--red)' : status.kind === 'none' ? 'var(--text-faint)' : 'var(--accent-2)',
              }}
            />
            <span style={{ color: status.kind === 'error' ? 'var(--red)' : 'var(--text-dim)', flex: 1, minWidth: 0, overflowWrap: 'anywhere' }}>
              {statusText(status, waitingGpu)}
            </span>
            {status.kind === 'running' || status.kind === 'queued' ? (
              <button className="small ghost" onClick={() => void cancelSeparation(status.jobId)}>
                Cancel
              </button>
            ) : (
              <button className="small" disabled={status.kind === 'ready' || !aiState.ready} title={aiState.hint || undefined} onClick={() => asset && void requestSeparation([asset.path])}>
                Separate
              </button>
            )}
          </div>
          {status.kind === 'running' || waitingGpu ? (
            <div className={`progress ${waitingGpu ? 'waiting' : ''}`}>
              <div style={{ width: waitingGpu || status.kind !== 'running' ? '100%' : `${Math.round(status.pct * 100)}%` }} />
            </div>
          ) : null}
          {mode !== 'original' && status.kind !== 'ready' ? (
            <div className="hint" style={{ marginTop: 4 }}>
              Plays and exports the original audio until the stems are ready.
            </div>
          ) : null}
          <button className="small" style={{ marginTop: 8 }} onClick={applyAll} title="Set this mode on every clip with sound and separate what is missing">
            Apply to all clips
          </button>
          {running.map((j) => {
            const waiting = isWaitingForGpu(j.message);
            return (
              <div key={j.jobId} style={{ marginTop: 8 }}>
                <div className="hint">
                  {waiting
                    ? `Queued · ${GPU_WAIT} · ${j.paths.length} file(s)`
                    : `Separating ${Math.min(j.paths.length, j.done.length + 1)}/${j.paths.length} · ${j.message}`}
                </div>
                <div className={`progress ${waiting ? 'waiting' : ''}`}>
                  <div style={{ width: waiting ? '100%' : `${Math.round(j.pct * 100)}%` }} />
                </div>
              </div>
            );
          })}
          <div className="hint" style={{ marginTop: 6 }}>
            {linked ? 'Also applies to the linked audio clip. ' : ''}Demucs v4 separation runs locally on the GPU; stems are cached per file.
          </div>
        </>
      )}
    </div>
  );
}

function MaskTab() {
  const sel = useSelectedClip();
  const updateClip = useEditor((s) => s.updateClip);
  if (!sel) return null;
  const { clip } = sel;
  const mask = clip.mask;
  const setMask = (patch: Partial<NonNullable<typeof mask>>) =>
    updateClip(clip.id, { mask: { ...(mask ?? { shape: 'rectangle', feather: 0.05, rect: keyframed([0.25, 0.25, 0.5, 0.5]), inverted: false }), ...patch } });
  const rect = mask ? mask.rect.static : [0.25, 0.25, 0.5, 0.5];
  return (
    <div>
      <div className="section">
        <h4>
          Vector mask
          {mask ? (
            <button className="small ghost" onClick={() => updateClip(clip.id, { mask: null })}>
              Remove
            </button>
          ) : null}
        </h4>
        <div className="chips">
          {(['rectangle', 'circle', 'split', 'filmstrip'] as MaskShape[]).map((s) => (
            <button key={s} className={`small ${mask?.shape === s ? 'active' : ''}`} onClick={() => setMask({ shape: s })}>
              {s}
            </button>
          ))}
        </div>
      </div>
      {mask ? (
        <div className="section">
          <Slider label="Feather" value={Math.round(mask.feather * 100)} min={0} max={50} onChange={(v) => setMask({ feather: v / 100 })} />
          {(['X', 'Y', 'W', 'H'] as const).map((k, i) => (
            <Slider key={k} label={k} value={Math.round(rect[i] * 100)} min={0} max={100} onChange={(v) => setMask({ rect: { ...mask.rect, static: rect.map((x, j) => (j === i ? v / 100 : x)) as [number, number, number, number] } })} />
          ))}
          <label style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 12 }}>
            <input type="checkbox" checked={mask.inverted} onChange={(e) => setMask({ inverted: e.target.checked })} />
            Invert
          </label>
        </div>
      ) : (
        <div className="hint">Pick a shape to add a feathered mask to this clip.</div>
      )}
    </div>
  );
}

/* ------------------------------------------------------ separate to tracks */

function SeparateTracksSection() {
  const sel = useSelectedClip();
  const project = useEditor((s) => s.project);
  const pending = useEditor((s) => s.pendingStemTracks);
  const ai = useAiReady();
  if (!sel || sel.track.kind === 'fx') return null;
  const { clip } = sel;
  const can = canSeparateToTracks(project, clip.id);
  if (!can) return null;
  const done = hasStemClips(project, clip.id);
  const waiting = pending.includes(clip.id);
  const all = separableClipIds(project);
  return (
    <div className="section">
      <h4>Separate to tracks</h4>
      <div className="hint" style={{ marginBottom: 6 }}>
        Voice and background as their own clips on the <b>Voice</b> / <b>Background</b> tracks, linked to this clip: each gets its own volume, fades, keyframes and mute. The original audio is muted.
      </div>
      <div className="chips">
        <button className="small primary" disabled={done || waiting || !ai.ready} title={ai.hint || undefined} onClick={() => void separateClipsToTracks([clip.id])}>
          {done ? 'Separated' : waiting ? 'Separating…' : 'Separate to tracks'}
        </button>
        <button
          className="small"
          disabled={!all.length || !ai.ready}
          title={ai.hint || 'Separate every clip with sound'}
          onClick={async () => {
            if (await confirmAction('Separate every clip to tracks?', `${all.length} clip(s) get Voice / Background stem clips (their original audio is muted). Undo with Ctrl+Z.`, 'Separate')) void separateClipsToTracks(all);
          }}
        >
          Apply to all clips
        </button>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------- video fades */

function VideoFadeSection() {
  const sel = useSelectedClip();
  const updateClip = useEditor((s) => s.updateClip);
  if (!sel || sel.clip.effect) return null;
  const { clip } = sel;
  const dur = clipDurationMs(clip);
  const max = Math.max(0.1, Math.floor(dur / 200) / 10);
  const set = (key: 'fadeInMs' | 'fadeOutMs', v: number) => updateClip(clip.id, { [key]: Math.round(Math.min(dur / 2, v * 1000)) });
  return (
    <div className="section">
      <h4>Fade (video)</h4>
      <Slider label="Fade in" value={Math.round(Math.min(dur / 2, clip.fadeInMs ?? 0) / 100) / 10} min={0} max={max} step={0.1} onChange={(v) => set('fadeInMs', v)} fmt={(v) => `${v.toFixed(1)} s`} hint="From black over the first seconds of the clip" />
      <Slider label="Fade out" value={Math.round(Math.min(dur / 2, clip.fadeOutMs ?? 0) / 100) / 10} min={0} max={max} step={0.1} onChange={(v) => set('fadeOutMs', v)} fmt={(v) => `${v.toFixed(1)} s`} hint="To black over the last seconds of the clip" />
      <div className="hint">Or drag the handles on the clip's top corners in the timeline.</div>
    </div>
  );
}

/* ------------------------------------------------------------------ effect */

function EffectTab() {
  const sel = useSelectedClip();
  const updateClip = useEditor((s) => s.updateClip);
  const deleteClips = useEditor((s) => s.deleteClips);
  if (!sel?.clip.effect) return null;
  const { clip } = sel;
  const effect = clip.effect!;
  const info = effectInfo(effect.type);
  const dur = clip.outMs - clip.inMs;
  const setEffect = (patch: Partial<typeof effect>) => updateClip(clip.id, (c) => ({ effect: { ...c.effect!, ...patch } }));
  return (
    <div>
      <div className="section">
        <h4>
          Effect
          <button className="small ghost" onClick={() => deleteClips([clip.id])} title="Delete the effect clip (Delete)">
            Remove
          </button>
        </h4>
        <select
          value={effect.type}
          aria-label="Effect type"
          style={{ width: '100%', marginBottom: 6 }}
          onChange={(e) => {
            const t = e.target.value as EffectType;
            updateClip(clip.id, (c) => ({ effect: { type: t, intensity: c.effect?.intensity ?? 1 }, label: effectInfo(t).label }));
          }}
        >
          {EFFECT_INFO.map((e) => (
            <option key={e.id} value={e.id}>
              {e.label}
            </option>
          ))}
        </select>
        <div className="hint" style={{ marginBottom: 6 }}>
          {info.hint}. {info.envelope ? 'Ramps in and out over 120 ms.' : 'Follows its own timing over the clip.'}
        </div>
        <Slider label="Intensity" value={Math.round(effect.intensity * 100)} min={0} max={100} onChange={(v) => setEffect({ intensity: v / 100 })} fmt={(v) => `${v} %`} defaultValue={100} />
        {info.params.map((p) => (
          <Slider
            key={p.key}
            label={p.label}
            value={effectParam(effect, p.key)}
            min={p.min}
            max={p.max}
            step={p.step}
            defaultValue={p.default}
            onChange={(v) => setEffect({ params: { ...(effect.params ?? {}), [p.key]: v } })}
            fmt={p.fmt}
          />
        ))}
        <Slider label="Duration" value={Math.round(dur / 100) / 10} min={0.1} max={Math.max(10, Math.ceil(dur / 1000))} step={0.1} onChange={(v) => updateClip(clip.id, { outMs: clip.inMs + Math.round(v * 1000) })} fmt={(v) => `${v.toFixed(1)} s`} />
        <div className="hint">Applies to every video track under it; stack effects on more FX tracks. Trim and move it like any clip.</div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------- transition */

function TransitionTab() {
  const cutId = useEditor((s) => s.selectedCut);
  const project = useEditor((s) => s.project);
  const setTransition = useEditor((s) => s.setTransition);
  const found = cutId ? findClip(project, cutId) : null;
  if (!found) return <div className="hint">Click the bow-tie marker on a cut in the timeline to edit its transition.</div>;
  const { clip, track } = found;
  const prev = prevAdjacent(track, clip, frameMs(project));
  if (!prev) return <div className="hint">This clip no longer follows another clip directly.</div>;
  const tr = clip.transitionIn;
  const maxMs = clampTransitionMs(TRANSITION_MAX_MS, clipDurationMs(prev), clipDurationMs(clip));
  const apply = (type: typeof TRANSITION_TYPES[number]['id'], durationMs = tr?.durationMs ?? 500) => setTransition(clip.id, { type, durationMs });
  return (
    <div>
      <div className="section">
        <h4>
          Transition
          {tr ? (
            <button className="small ghost" onClick={() => setTransition(clip.id, null)} title="Remove (Delete)">
              Remove
            </button>
          ) : null}
        </h4>
        <div className="hint" style={{ marginBottom: 6 }}>
          {prev.label ?? 'clip'} → {clip.label ?? 'clip'} · {tr ? `${transitionLabel(tr.type)}, centred on the cut` : 'no transition (hard cut)'}
        </div>
        <div className="transition-picker">
          {TRANSITION_TYPES.map((t) => (
            <button key={t.id} className={`small ${tr?.type === t.id ? 'active' : ''}`} onClick={() => apply(t.id)} title={t.hint}>
              {t.label}
            </button>
          ))}
        </div>
        {tr ? (
          <>
            <Slider
              label="Duration"
              value={Math.round(tr.durationMs / 100) / 10}
              min={TRANSITION_MIN_MS / 1000}
              max={Math.max(TRANSITION_MIN_MS, maxMs) / 1000}
              step={0.1}
              defaultValue={0.5}
              onChange={(v) => apply(tr.type, Math.round(v * 1000))}
              fmt={(v) => `${v.toFixed(1)} s`}
              hint="0.1 to 3 s, at most the shorter of the two clips"
            />
            <button
              className="small"
              style={{ marginTop: 6 }}
              onClick={() => {
                const n = useEditor.getState().applyTransitionToAllCuts({ type: tr.type, durationMs: tr.durationMs });
                useEditor.getState().log('info', `${transitionLabel(tr.type)} applied to ${n} cut(s) of the main track`);
              }}
            >
              Apply to all cuts
            </button>
          </>
        ) : null}
      </div>
    </div>
  );
}
