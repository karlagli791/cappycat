import { projectDurationMs, useEditor } from '@/state/store';
import { playClock } from '@/state/clock';
import { timecode } from '@/lib/format';

/** Timecode + duration: the only transport parts that follow the (throttled) playhead. */
function Timecode() {
  const playheadMs = useEditor((s) => s.playheadMs);
  const fps = useEditor((s) => s.project.fps);
  const duration = useEditor((s) => projectDurationMs(s.project));
  const shuttle = useEditor((s) => (s.playing && s.shuttle !== 1 ? s.shuttle : null));
  return (
    <>
      <span className="timecode">{timecode(playheadMs, fps)}</span>
      <span className="hint mono">/ {timecode(duration, fps)}</span>
      {shuttle ? <span className="shuttle-badge">{shuttle > 0 ? `▶ ${shuttle}×` : `◀ ${-shuttle}×`}</span> : null}
    </>
  );
}

export default function PlayerControls() {
  const playing = useEditor((s) => s.playing);
  const loop = useEditor((s) => s.loop);
  const compare = useEditor((s) => s.compareMode);
  const snapping = useEditor((s) => s.snapping);
  const ripple = useEditor((s) => s.rippleEdit);
  const showGrid = useEditor((s) => s.showGrid);
  const showBoxes = useEditor((s) => s.showBoxes);
  const largePreview = useEditor((s) => s.largePreview);
  const sel = useEditor((s) => s.selection.clipIds[0] ?? null);
  const selCount = useEditor((s) => s.selection.clipIds.length);
  const drawerOpen = useEditor((s) => s.drawerOpen);
  const drawerMode = useEditor((s) => s.drawerMode);
  const s = useEditor.getState;

  const frameMs = () => 1000 / s().project.fps;
  const togglePlay = () => {
    const st = s();
    if (!st.playing && playClock.get() >= projectDurationMs(st.project) - 1) st.setPlayhead(0);
    st.setPlaying(!st.playing);
  };

  return (
    <>
      <div className="player">
        <button className="small" title="Go to start (Home)" onClick={() => s().setPlayhead(0)}>
          |◀◀
        </button>
        <button className="small" title="Previous frame (←)" onClick={() => s().setPlayhead(playClock.get() - frameMs())}>
          ◀
        </button>
        <button className="primary play" onClick={togglePlay} title="Play / pause (Space) · J/K/L shuttle">
          {playing ? '❚❚' : '▶'}
        </button>
        <button className="small" title="Next frame (→)" onClick={() => s().setPlayhead(playClock.get() + frameMs())}>
          ▶
        </button>
        <button className="small" title="Go to end (End)" onClick={() => s().setPlayhead(projectDurationMs(s().project))}>
          ▶▶|
        </button>
        <Timecode />
        <button className={`small ${loop ? 'active' : ''}`} onClick={() => s().toggleLoop()} title="Loop playback">
          ⟲
        </button>
        <span className="spacer" />
        <button className={`small ${showGrid ? 'active' : ''}`} onClick={() => s().toggleGrid()} title="Rule-of-thirds grid">
          Grid
        </button>
        <button className={`small ${showBoxes ? 'active' : ''}`} onClick={() => s().toggleBoxes()} title="Detection boxes and the auto-zoom crop in the compare view">
          Boxes
        </button>
        <button className={`small ${compare ? 'active' : ''}`} onClick={() => s().toggleCompare()} title="Raw source vs graded, with detections (C)">
          Compare
        </button>
        <button className={`small ${largePreview ? 'active' : ''}`} onClick={() => s().toggleLargePreview()} title="Large preview: hide the side panels (F, Esc exits)">
          {largePreview ? 'Exit large' : 'Large'}
        </button>
      </div>
      <div className="tools">
        <button className="small" onClick={() => void s().splitAtPlayhead()} title="Split at the playhead: the selected clips, or every clip under it (Ctrl+B / S)">
          Split
        </button>
        <button className="small" disabled={!sel} onClick={() => sel && s().freezeFrameAt(sel, playClock.get())} title="Insert a 1 s freeze frame at the playhead (Alt+F)">
          Freeze
        </button>
        <button className="small" disabled={!sel} onClick={() => sel && s().toggleReverse(sel)} title="Reverse playback (Alt+R)">
          Reverse
        </button>
        <button className="small" disabled={!selCount} onClick={() => s().deleteClips(s().selection.clipIds)} title="Delete the selection (Delete)">
          Delete
        </button>
        <span style={{ width: 6 }} />
        <button
          className={`small ${drawerOpen && drawerMode === 'speed' ? 'active' : ''}`}
          disabled={!sel}
          onClick={() => {
            s().setInspectorTab('speed');
            s().setDrawer(!(drawerOpen && drawerMode === 'speed'), 'speed');
          }}
          title="Speed curve graph"
        >
          Curve
        </button>
        <button
          className={`small ${drawerOpen && drawerMode === 'keyframes' ? 'active' : ''}`}
          disabled={!sel}
          onClick={() => {
            s().setInspectorTab('transform');
            s().setDrawer(!(drawerOpen && drawerMode === 'keyframes'), 'keyframes');
          }}
          title="Keyframe graph (Alt+K)"
        >
          Keyframe
        </button>
        <button className="small" disabled={!sel} onClick={() => s().setInspectorTab('mask')} title="Mask settings">
          Mask
        </button>
        <span className="spacer" style={{ flex: 1 }} />
        <button className={`small ${snapping ? 'active' : ''}`} onClick={() => s().toggleSnapping()} title="Snap moves and trims to cuts, the playhead and beats (N)">
          Snap
        </button>
        <button className={`small ${ripple ? 'active' : ''}`} onClick={() => s().toggleRipple()} title="Main-track magnet: no gaps on the main track, edits ripple later clips (P)">
          Magnet
        </button>
      </div>
    </>
  );
}
