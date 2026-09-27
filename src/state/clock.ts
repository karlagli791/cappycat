/**
 * The frame-accurate playhead, outside React. The preview's rAF loop advances it every frame and
 * canvases that follow the playhead (timeline overlay) subscribe directly; the store's
 * `playheadMs` is a throttled copy (~15 Hz while playing) for React UI such as the timecode.
 */
type Listener = (ms: number) => void;

let current = 0;
const listeners = new Set<Listener>();

export const playClock = {
  get(): number {
    return current;
  },
  set(ms: number): void {
    const v = Math.max(0, ms);
    if (v === current) return;
    current = v;
    listeners.forEach((f) => f(v));
  },
  subscribe(f: Listener): () => void {
    listeners.add(f);
    return () => void listeners.delete(f);
  },
};

/** Store writes of the playhead while playing are throttled to this interval. */
export const UI_PLAYHEAD_INTERVAL_MS = 1000 / 15;
