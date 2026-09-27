import { beforeEach, describe, expect, it } from 'vitest';
import type { Asset, Clip, Project } from '@/types/project';
import { emptyProject } from '@/engine/defaults';
import { outputDuration, outputDurationUncached, presetCurve } from '@/engine/speed';
import { resolveClip } from '@/engine/playback';
import { evaluate } from '@/engine/keyframes';
import { clipDurationMs, clipEndMs, findClip, inferLinks, mainTrackId, projectDurationMs, useEditor } from './store';

const asset = (i: number, durationMs = 20000): Asset => ({
  id: `ast_${i}`,
  path: `C:/clips/clip${i}.mp4`,
  name: `clip${i}.mp4`,
  kind: 'video',
  durationMs,
  width: 1920,
  height: 1080,
  fps: 24,
  hasAudio: true,
  order: i,
});

/** A project with `n` 20 s clips laid out by the rough cut (linked video + audio clips). */
function setup(n = 4, opts: { durations?: number[] } = {}): Project {
  const p = emptyProject('Test');
  p.assets = Array.from({ length: n }, (_, i) => asset(i, opts.durations?.[i] ?? 20000));
  const st = useEditor.getState();
  st.loadDocument(p);
  useEditor.setState({ rippleEdit: true, snapping: true });
  useEditor.getState().roughCutInStoryOrder();
  // start every test with a clean history
  useEditor.setState({ past: [], future: [], dirty: false });
  // shorten clips to 5 s so trims have room on both sides
  const cur = useEditor.getState().project;
  const shortened: Project = {
    ...cur,
    tracks: cur.tracks.map((t) => ({ ...t, clips: t.clips.map((c) => ({ ...c, inMs: 2000, outMs: 7000 })) })),
  };
  useEditor.getState().setProject(shortened, { undoable: false });
  useEditor.getState().trimClip(video(0).id, 'out', 0); // settle (compact) once
  useEditor.setState({ past: [], future: [], dirty: false });
  return useEditor.getState().project;
}

const S = () => useEditor.getState();
const P = () => useEditor.getState().project;
const videoTrack = () => P().tracks.find((t) => t.id === mainTrackId(P()))!;
const audioTrack = () => P().tracks.find((t) => t.kind === 'audio')!;
const byStart = (clips: Clip[]) => [...clips].sort((a, b) => a.startMs - b.startMs);
const video = (i: number) => byStart(videoTrack().clips)[i];
const partner = (c: Clip) => audioTrack().clips.find((a) => a.linkId === c.linkId)!;

function expectGapless(clips: Clip[]) {
  const sorted = byStart(clips);
  let cursor = 0;
  for (const c of sorted) {
    expect(c.startMs).toBeCloseTo(cursor, 3);
    cursor = clipEndMs(c);
  }
}

function expectLinkedInSync() {
  for (const v of videoTrack().clips) {
    const a = partner(v);
    expect(a).toBeTruthy();
    expect(a.startMs).toBeCloseTo(v.startMs, 6);
    expect(a.inMs).toBeCloseTo(v.inMs, 6);
    expect(a.outMs).toBeCloseTo(v.outMs, 6);
    expect(a.speed).toEqual(v.speed);
    expect(a.freezeFrame).toEqual(v.freezeFrame);
    expect(a.reversed).toBe(v.reversed);
  }
}

beforeEach(() => {
  setup();
});

describe('undo gestures', () => {
  it('a slider drag is one undo entry', () => {
    const id = video(0).id;
    S().beginGesture();
    for (let v = 1; v <= 30; v++) S().updateClipColor(id, { exposure: v });
    S().endGesture();
    expect(S().past).toHaveLength(1);
    expect(findClip(P(), id)!.clip.color.exposure).toBe(30);
    S().undo();
    expect(findClip(P(), id)!.clip.color.exposure).toBe(0);
    S().redo();
    expect(findClip(P(), id)!.clip.color.exposure).toBe(30);
  });

  it('commits that change nothing are no-ops (no entry, redo kept, still clean)', () => {
    const id = video(0).id;
    S().updateClipColor(id, { exposure: 5 });
    S().undo();
    expect(S().future).toHaveLength(1);
    S().updateClipColor(id, { exposure: 0 }); // same value
    expect(S().past).toHaveLength(0);
    expect(S().future).toHaveLength(1);
    // a gesture that ends where it started leaves no entry and restores the dirty flag
    useEditor.setState({ dirty: false });
    S().beginGesture();
    S().updateClipColor(id, { exposure: 12 });
    S().updateClipColor(id, { exposure: 0 });
    S().endGesture();
    expect(S().past).toHaveLength(0);
    expect(S().dirty).toBe(false);
  });

  it('keeps at most UNDO_LIMIT entries', () => {
    const id = video(0).id;
    for (let v = 1; v <= 230; v++) S().updateClipColor(id, { exposure: (v % 100) - 50 });
    expect(S().past.length).toBe(200);
  });
});

describe('moves', () => {
  it('are idempotent from the drag snapshot (neighbours are not displaced)', () => {
    const before = P();
    const c = video(1);
    S().beginGesture();
    S().moveClip(c.id, c.startMs + 3000);
    S().moveClip(c.id, c.startMs + 6000);
    S().moveClip(c.id, c.startMs);
    S().endGesture();
    expect(S().past).toHaveLength(0);
    expect(videoTrack().clips.map((x) => [x.id, x.startMs])).toEqual(before.tracks.find((t) => t.id === videoTrack().id)!.clips.map((x) => [x.id, x.startMs]));
  });

  it('reorder on the magnetic main track and keep the linked audio in sync', () => {
    const [a, b] = [video(0), video(1)];
    S().beginGesture();
    // drag clip 0 past the middle of clip 1
    S().moveClip(a.id, b.startMs + clipDurationMs(b) * 0.6);
    S().endGesture();
    expect(S().past).toHaveLength(1);
    expect(video(0).id).toBe(b.id);
    expect(video(1).id).toBe(a.id);
    expectGapless(videoTrack().clips);
    expectLinkedInSync();
  });

  it('dragging the audio member moves the linked pair', () => {
    const v = video(2);
    const a = partner(v);
    S().moveClip(a.id, 0);
    expect(video(0).id).toBe(v.id);
    expectLinkedInSync();
  });

  it('without the magnet a move leaves a gap and pushes overlaps', () => {
    useEditor.setState({ rippleEdit: false });
    const c = video(3);
    S().moveClip(c.id, c.startMs + 2000);
    expect(findClip(P(), c.id)!.clip.startMs).toBeCloseTo(c.startMs + 2000);
    expectLinkedInSync();
  });
});

describe('speed changes', () => {
  it('ripple later clips and keep keyframes/freeze on their source frames', () => {
    const c = video(0);
    const next = video(1);
    S().updateClip(c.id, {
      freezeFrame: { atMs: 3000, holdMs: 500 },
      transform: { ...c.transform, scale: { static: 1, keyframes: [ { timeMs: 2000, value: 1, easing: 'linear' }, { timeMs: 4500, value: 2, easing: 'linear' } ] } },
    });
    const oldDur = clipDurationMs(findClip(P(), c.id)!.clip); // 5000 + 500
    expect(oldDur).toBeCloseTo(5500);
    S().setClipSpeed(c.id, { preset: 'custom', points: [ { t: 0, speed: 2 }, { t: 1, speed: 2 } ], opticalFlow: false });
    const after = findClip(P(), c.id)!.clip;
    expect(clipDurationMs(after)).toBeCloseTo(2500 + 500, 0);
    expect(after.freezeFrame!.atMs).toBeCloseTo(1500, 0);
    expect(after.freezeFrame!.holdMs).toBe(500);
    // 2000 (before the hold) -> 1000; 4500 (1000 after the hold) -> 1500 + 500 + 500
    expect(after.transform.scale.keyframes.map((k) => Math.round(k.timeMs))).toEqual([1000, 2500]);
    // the next clip rippled to the new end (no gap, no overlap)
    expect(findClip(P(), next.id)!.clip.startMs).toBeCloseTo(clipEndMs(after), 3);
    expectGapless(videoTrack().clips);
    expectLinkedInSync();
  });

  it('speed-ramp presets keep the main track gapless (no overlaps)', () => {
    for (const [i, preset] of (['hero_time', 'bullet', 'montage'] as const).entries()) S().setClipSpeed(video(i).id, presetCurve(preset));
    expectGapless(videoTrack().clips);
    expectLinkedInSync();
  });
});

describe('split', () => {
  it('is speed-aware: both halves meet on the same source frame and keep the timeline length', () => {
    const c0 = video(1);
    S().setClipSpeed(c0.id, presetCurve('hero_time'));
    const c = findClip(P(), c0.id)!.clip;
    const dur = clipDurationMs(c);
    const local = dur * 0.4;
    const expectSrc = resolveClip(c, local).sourceMs;
    S().splitClipAt(c.id, c.startMs + local);
    const left = findClip(P(), c.id)!.clip;
    const right = findClip(P(), S().selection.clipIds[0])!.clip;
    expect(left.outMs).toBeCloseTo(expectSrc, 0);
    expect(right.inMs).toBeCloseTo(expectSrc, 0);
    expect(Math.abs(clipDurationMs(left) - local) / local).toBeLessThan(0.02);
    expect(Math.abs(clipDurationMs(left) + clipDurationMs(right) - dur) / dur).toBeLessThan(0.01);
    // the right half starts where the left one plays (no replay of the ramp): first frame = split frame
    expect(resolveClip(right, 0).sourceMs).toBeCloseTo(expectSrc, 0);
    expect(resolveClip(left, clipDurationMs(left) - 1).sourceMs).toBeCloseTo(expectSrc, -2);
    expectGapless(videoTrack().clips);
    expectLinkedInSync();
  });

  it('reversed clips keep the playback order', () => {
    const c0 = video(0);
    S().toggleReverse(c0.id);
    const c = findClip(P(), c0.id)!.clip;
    expect(c.reversed).toBe(true);
    const splitSrc = resolveClip(c, 1500).sourceMs; // reversed: 7000 - 1500
    expect(splitSrc).toBeCloseTo(5500);
    S().splitClipAt(c.id, c.startMs + 1500);
    const left = findClip(P(), c.id)!.clip;
    const right = findClip(P(), S().selection.clipIds[0])!.clip;
    expect([left.inMs, left.outMs]).toEqual([5500, 7000]);
    expect([right.inMs, right.outMs]).toEqual([2000, 5500]);
    expect(resolveClip(right, 0).sourceMs).toBeCloseTo(5500);
    expect(resolveClip(left, 0).sourceMs).toBeCloseTo(7000);
    expectLinkedInSync();
  });

  it('keeps the freeze frame in its half and keyframes on both sides of the cut', () => {
    const c0 = video(0);
    S().updateClip(c0.id, {
      freezeFrame: { atMs: 1000, holdMs: 500 },
      transform: { ...c0.transform, scale: { static: 1, keyframes: [ { timeMs: 0, value: 1, easing: 'linear' }, { timeMs: 4000, value: 2, easing: 'linear' } ] } },
    });
    const c = findClip(P(), c0.id)!.clip;
    S().splitClipAt(c.id, c.startMs + 3000);
    const left = findClip(P(), c.id)!.clip;
    const right = findClip(P(), S().selection.clipIds[0])!.clip;
    expect(left.freezeFrame).toEqual({ atMs: 1000, holdMs: 500 });
    expect(right.freezeFrame).toBeNull();
    const vCut = evaluate(c.transform.scale, 3000);
    expect(evaluate(left.transform.scale, 3000)).toBeCloseTo(vCut);
    expect(evaluate(right.transform.scale, 0)).toBeCloseTo(vCut);
    expect(evaluate(right.transform.scale, 1000)).toBeCloseTo(2);
    expect(left.transform.scale.keyframes.map((k) => k.timeMs)).toEqual([0, 3000]);
    // source continuity through the hold: left plays 2000..(2000+2500), right starts there
    expect(left.outMs).toBeCloseTo(4500);
    expect(right.inMs).toBeCloseTo(4500);
  });

  it('Ctrl+B splits the clip under the playhead and its linked audio when nothing is selected', () => {
    S().select([]);
    const c = video(1);
    S().setPlayhead(c.startMs + 1000);
    expect(S().splitAtPlayhead()).toBe(2);
    expect(videoTrack().clips).toHaveLength(5);
    expect(audioTrack().clips).toHaveLength(5);
    expectLinkedInSync();
  });
});

describe('linked partners', () => {
  it('trim, delete, freeze and reverse apply to the audio partner', () => {
    const c = video(1);
    S().trimClip(c.id, 'out', -1000);
    expectLinkedInSync();
    S().freezeFrameAt(c.id, c.startMs + 500, 800);
    expectLinkedInSync();
    S().toggleReverse(partner(findClip(P(), c.id)!.clip).id);
    expect(findClip(P(), c.id)!.clip.reversed).toBe(true);
    expectLinkedInSync();
    S().deleteClips([c.id]);
    expect(videoTrack().clips).toHaveLength(3);
    expect(audioTrack().clips).toHaveLength(3);
    expectLinkedInSync();
  });

  it('are inferred for older projects without linkId', () => {
    const p = P();
    const stripped: Project = { ...p, tracks: p.tracks.map((t) => ({ ...t, clips: t.clips.map(({ linkId: _l, ...c }) => c as Clip) })) };
    const linked = inferLinks(stripped);
    const v = linked.tracks.find((t) => t.kind === 'video')!.clips;
    const a = linked.tracks.find((t) => t.kind === 'audio')!.clips;
    expect(v.every((c) => !!c.linkId)).toBe(true);
    expect(v.map((c) => c.linkId).sort()).toEqual(a.map((c) => c.linkId).sort());
  });

  it('freezing twice ripples by the hold change only', () => {
    const c = video(0);
    S().freezeFrameAt(c.id, c.startMs + 1000, 1000);
    S().freezeFrameAt(c.id, c.startMs + 1000, 1000);
    expect(clipDurationMs(findClip(P(), c.id)!.clip)).toBeCloseTo(6000);
    expectGapless(videoTrack().clips);
  });
});

describe('delete', () => {
  it('multi-delete leaves no gaps on the magnetic main track or its audio', () => {
    const ids = [video(0).id, video(2).id];
    S().deleteClips(ids);
    expect(videoTrack().clips).toHaveLength(2);
    expectGapless(videoTrack().clips);
    expectGapless(audioTrack().clips);
    expectLinkedInSync();
    expect(projectDurationMs(P())).toBeCloseTo(10000);
  });

  it('multi-delete on a non-main track ripples every gap closed', () => {
    const au = audioTrack();
    // unlink the audio so it is a free track of its own
    useEditor.getState().setProject({ ...P(), tracks: P().tracks.map((t) => (t.id === au.id ? { ...t, clips: t.clips.map(({ linkId: _l, ...c }) => c as Clip) } : t)) }, { undoable: false });
    const clips = byStart(audioTrack().clips);
    S().deleteClips([clips[0].id, clips[2].id]);
    expectGapless(audioTrack().clips);
  });
});

describe('in-point trim', () => {
  it('ripples with the magnet on and moves keyframes with the content', () => {
    const c0 = video(0);
    const next = video(1);
    S().updateClip(c0.id, { transform: { ...c0.transform, opacity: { static: 1, keyframes: [ { timeMs: 3000, value: 0.5, easing: 'linear' } ] } } });
    S().beginGesture();
    S().trimClip(c0.id, 'in', 400);
    S().trimClip(c0.id, 'in', 1000); // absolute from the snapshot, not cumulative
    S().endGesture();
    const c = findClip(P(), c0.id)!.clip;
    expect(c.inMs).toBeCloseTo(3000);
    expect(c.startMs).toBe(0);
    expect(clipDurationMs(c)).toBeCloseTo(4000);
    expect(c.transform.opacity.keyframes.map((k) => k.timeMs)).toEqual([0, 2000]);
    expect(findClip(P(), next.id)!.clip.startMs).toBeCloseTo(4000);
    expectLinkedInSync();
    expect(S().past).toHaveLength(2); // updateClip + one trim gesture
  });

  it('is speed-aware and reverse-aware', () => {
    const c0 = video(1);
    S().setClipSpeed(c0.id, { preset: 'custom', points: [ { t: 0, speed: 2 }, { t: 1, speed: 2 } ], opticalFlow: false });
    S().trimClip(c0.id, 'in', 500); // 500 ms of timeline = 1000 ms of source at 2x
    expect(findClip(P(), c0.id)!.clip.inMs).toBeCloseTo(3000);
    const r = video(2);
    S().toggleReverse(r.id);
    S().trimClip(r.id, 'in', 1000); // reversed: the timeline in-point is the source out-point
    const rc = findClip(P(), r.id)!.clip;
    expect(rc.outMs).toBeCloseTo(6000);
    expect(rc.inMs).toBeCloseTo(2000);
  });

  it('without the magnet the start follows the edge and a gap is left', () => {
    useEditor.setState({ rippleEdit: false });
    const c = video(1);
    S().trimClip(c.id, 'in', 1000);
    expect(findClip(P(), c.id)!.clip.startMs).toBeCloseTo(c.startMs + 1000);
    expect(findClip(P(), video(2).id)!.clip.startMs).toBeCloseTo(video(2).startMs);
  });
});

describe('selection', () => {
  it('undo prunes the selection and a no-op delete adds no entry', () => {
    const c = video(1);
    S().splitClipAt(c.id, c.startMs + 2000);
    const right = S().selection.clipIds[0];
    expect(right).toBeTruthy();
    S().undo();
    expect(S().selection.clipIds).toEqual([]);
    const past = S().past.length;
    S().deleteClips(S().selection.clipIds);
    expect(S().past.length).toBe(past);
    expect(S().future.length).toBe(1);
  });
});

describe('paste attributes', () => {
  it('copies grade and speed to other clips with ripple', () => {
    const a = video(0);
    S().updateClipColor(a.id, { contrast: 20 });
    S().setClipSpeed(a.id, presetCurve('flash_in'));
    S().copyAttributes(a.id);
    const n = S().pasteAttributes([video(2).id, video(3).id], ['color', 'speed']);
    expect(n).toBe(2);
    expect(video(2).color.contrast).toBe(20);
    expect(video(3).speed.preset).toBe('flash_in');
    expectGapless(videoTrack().clips);
    expectLinkedInSync();
  });
});

describe('memoised speed durations', () => {
  it('equal the unmemoised value', () => {
    for (const preset of ['normal', 'montage', 'hero_time', 'bullet', 'jump_cut', 'flash_in', 'flash_out'] as const) {
      const curve = presetCurve(preset);
      for (const src of [1000, 5000, 12345]) {
        expect(outputDuration(curve, src)).toBeCloseTo(outputDurationUncached(curve, src), 6);
        expect(outputDuration(curve, src)).toBeCloseTo(outputDurationUncached(curve, src), 6); // cached path
      }
    }
    const twoX = { preset: 'custom' as const, points: [ { t: 0, speed: 2 }, { t: 1, speed: 2 } ], opticalFlow: false };
    expect(outputDuration(twoX, 4000)).toBeCloseTo(outputDurationUncached(twoX, 4000), 6);
  });
});
