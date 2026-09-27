/**
 * Left panel: Media · Transitions · Effects. The transition and effect tiles are cheap CSS
 * previews (transform / opacity / clip-path animations on two tiny panes).
 */
import { useState } from 'react';
import { useEditor } from '@/state/store';
import { cutsOfTrack, frameMs, mainTrackId } from '@/state/edits';
import { playClock } from '@/state/clock';
import { TRANSITION_DEFAULT_MS, TRANSITION_MAX_MS, TRANSITION_MIN_MS, TRANSITION_TYPES, transitionLabel } from '@/engine/transitions';
import { EFFECT_INFO } from '@/engine/effects';
import type { EffectType, TransitionType } from '@/types/project';
import AssetPool from './AssetPool';
import { EFFECT_MIME, TRANSITION_MIME } from './Timeline';
import { Slider } from './ColorPanel';

type LeftTab = 'media' | 'transitions' | 'effects';

interface Props {
  onImport: () => void;
  onDropFiles: (paths: string[]) => void;
  onScanFolder: (pick: boolean) => void;
}

export default function LibraryPanels(props: Props) {
  const [tab, setTab] = useState<LeftTab>(() => {
    try {
      const v = localStorage.getItem('cappycat:left-tab');
      return v === 'transitions' || v === 'effects' ? v : 'media';
    } catch {
      return 'media';
    }
  });
  const choose = (t: LeftTab) => {
    setTab(t);
    try {
      localStorage.setItem('cappycat:left-tab', t);
    } catch {
      /* ignore */
    }
  };
  return (
    <>
      <div className="tabs left-tabs" role="tablist" aria-label="Library">
        {(
          [
            ['media', 'Media'],
            ['transitions', 'Transitions'],
            ['effects', 'Effects'],
          ] as Array<[LeftTab, string]>
        ).map(([id, label]) => (
          <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? 'active' : ''} onClick={() => choose(id)}>
            {label}
          </button>
        ))}
      </div>
      {tab === 'media' ? <AssetPool {...props} /> : tab === 'transitions' ? <TransitionsPanel /> : <EffectsPanel />}
    </>
  );
}

/* ------------------------------------------------------------ transitions */

function TransitionsPanel() {
  const selectedCut = useEditor((s) => s.selectedCut);
  const project = useEditor((s) => s.project);
  const [durationMs, setDurationMs] = useState(TRANSITION_DEFAULT_MS);
  const [allType, setAllType] = useState<TransitionType>('dissolve');
  const current = selectedCut ? project.tracks.flatMap((t) => t.clips).find((c) => c.id === selectedCut)?.transitionIn ?? null : null;

  /** The selected cut, else the main-track cut nearest the playhead. */
  const targetCut = (): string | null => {
    const st = useEditor.getState();
    if (st.selectedCut) return st.selectedCut;
    const main = st.project.tracks.find((t) => t.id === mainTrackId(st.project));
    if (!main) return null;
    const head = playClock.get();
    let best: string | null = null;
    let bestD = Infinity;
    for (const cut of cutsOfTrack(main, frameMs(st.project))) {
      const d = Math.abs(cut.atMs - head);
      if (d < bestD) {
        bestD = d;
        best = cut.next.id;
      }
    }
    return best;
  };

  const apply = (type: TransitionType) => {
    const st = useEditor.getState();
    const id = targetCut();
    if (!id) {
      st.log('warn', 'There is no cut to put a transition on: put two clips next to each other first.');
      return;
    }
    st.setTransition(id, { type, durationMs });
    st.selectCut(id);
  };

  const applyAll = (type: TransitionType) => {
    const st = useEditor.getState();
    const n = st.applyTransitionToAllCuts({ type, durationMs });
    st.log('info', `${transitionLabel(type)} applied to ${n} cut(s) of the main track`);
  };

  return (
    <>
      <div className="panel-title">
        <span>Transitions</span>
        <span className="hint panel-sub">{selectedCut ? (current ? transitionLabel(current.type) : 'cut selected') : 'nearest cut to the playhead'}</span>
      </div>
      <div className="panel-body">
        <div className="hint" style={{ marginBottom: 8 }}>
          Drag a transition onto a cut, or select a cut (its bow-tie marker) and click one. Transitions are centred on the cut; the timeline length does not change.
        </div>
        <Slider label="Duration" value={durationMs / 1000} min={TRANSITION_MIN_MS / 1000} max={TRANSITION_MAX_MS / 1000} step={0.1} defaultValue={0.5} onChange={(v) => setDurationMs(Math.round(v * 1000))} fmt={(v) => `${v.toFixed(1)} s`} />
        <div className="fx-grid">
          {TRANSITION_TYPES.map((t) => (
            <div
              key={t.id}
              className={`fx-tile ${current?.type === t.id ? 'active' : ''}`}
              draggable
              role="button"
              tabIndex={0}
              title={`${t.hint}. Click: apply to the ${selectedCut ? 'selected cut' : 'cut nearest the playhead'} · drag onto a cut`}
              onClick={() => apply(t.id)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  apply(t.id);
                }
              }}
              onDragStart={(e) => {
                e.dataTransfer.setData(TRANSITION_MIME, t.id);
                e.dataTransfer.effectAllowed = 'copy';
              }}
            >
              <div className={`tp tp-${t.id}`} aria-hidden="true">
                <i className="a" />
                <i className="b" />
                <i className="x" />
              </div>
              <span>{t.label}</span>
            </div>
          ))}
        </div>
        <div className="section" style={{ marginTop: 10 }}>
          <h4>All cuts</h4>
          <div className="hint" style={{ marginBottom: 6 }}>Put one transition on every cut of the main track.</div>
          <select aria-label="Transition for all cuts" value={allType} onChange={(e) => setAllType(e.target.value as TransitionType)} style={{ width: '100%', marginBottom: 6 }}>
            {TRANSITION_TYPES.map((t) => (
              <option key={t.id} value={t.id}>
                {t.label}
              </option>
            ))}
          </select>
          <button className="small" onClick={() => applyAll(allType)}>
            Apply to all cuts
          </button>
        </div>
      </div>
    </>
  );
}

/* ---------------------------------------------------------------- effects */

function EffectsPanel() {
  const add = (type: EffectType) => {
    const st = useEditor.getState();
    const clip = st.addEffect(type);
    if (clip) {
      st.select([clip.id]);
      st.setInspectorTab('effect');
    }
  };
  return (
    <>
      <div className="panel-title">
        <span>Effects</span>
        <span className="hint panel-sub">applied to the picture under them</span>
      </div>
      <div className="panel-body">
        <div className="hint" style={{ marginBottom: 8 }}>
          Click to add an effect at the playhead, or drag one onto the FX track. Trim and move effect clips like any clip; the Inspector sets the intensity.
        </div>
        <div className="fx-grid">
          {EFFECT_INFO.map((e) => (
            <div
              key={e.id}
              className="fx-tile"
              draggable
              role="button"
              tabIndex={0}
              title={`${e.hint} (${(e.defaultMs / 1000).toFixed(1)} s). Click: add at the playhead · drag onto the FX track`}
              onClick={() => add(e.id)}
              onKeyDown={(ev) => {
                if (ev.key === 'Enter' || ev.key === ' ') {
                  ev.preventDefault();
                  add(e.id);
                }
              }}
              onDragStart={(ev) => {
                ev.dataTransfer.setData(EFFECT_MIME, e.id);
                ev.dataTransfer.effectAllowed = 'copy';
              }}
            >
              <div className={`ep ep-${e.id}`} aria-hidden="true">
                <i className="scene" />
                <i className="x" />
              </div>
              <span>{e.label}</span>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
