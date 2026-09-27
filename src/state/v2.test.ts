/**
 * Feature set v2 edit semantics: N-member link groups, "Separate to tracks", transitions (clamping
 * and survival across split / trim / move / delete), effect clips, fades, volume keyframes, quick
 * speed buttons.
 */
import { beforeEach, describe, expect, it } from 'vitest';
import type { Asset, Clip, Project } from '@/types/project';
import { emptyProject } from '@/engine/defaults';
import { clipDurationMs, clipEndMs, findClip, mainTrackId, useEditor } from './store';
import { cutsOfTrack, frameMs, partnersOf, prevAdjacent, sanitizeProject } from './edits';
import { hasStemClips, separableClipIds, stemAssetOf } from './stemTracks';

const STEMS = (i: number) => ({ vocals: `C:/cache/stems/${i}/vocals.wav`, background: `C:/cache/stems/${i}/background.wav` });

const asset = (i: number, stems = true): Asset => ({
  id: `ast_${i}`,
  path: `C:/clips/clip${i}.mp4`,
  name: `clip${i}.mp4`,
  kind: 'video',
  durationMs: 20000,
  width: 1920,
  height: 1080,
  fps: 24,
  hasAudio: true,
  order: i,
  ...(stems ? { stems: STEMS(i) } : {}),
});

function setup(n = 4, stems = true): Project {
  const p = emptyProject('V2');
  p.assets = Array.from({ length: n }, (_, i) => asset(i, stems));
  useEditor.getState().loadDocument(p);
  useEditor.setState({ rippleEdit: true, snapping: true });
  useEditor.getState().roughCutInStoryOrder();
  const cur = useEditor.getState().project;
  // 5 s clips (source 2..7 s) so trims have room on both sides
  useEditor.getState().setProject({ ...cur, tracks: cur.tracks.map((t) => ({ ...t, clips: t.clips.map((c) => ({ ...c, inMs: 2000, outMs: 7000 })) })) }, { undoable: false });
  useEditor.getState().trimClip(video(0).id, 'out', 0);
  useEditor.setState({ past: [], future: [], dirty: false });
  return useEditor.getState().project;
}

const S = () => useEditor.getState();
const P = () => useEditor.getState().project;
const videoTrack = () => P().tracks.find((t) => t.id === mainTrackId(P()))!;
const byStart = (clips: Clip[]) => [...clips].sort((a, b) => a.startMs - b.startMs);
const video = (i: number) => byStart(videoTrack().clips)[i];
const clip = (id: string) => findClip(P(), id)!.clip;
const group = (c: Clip) => [c, ...partnersOf(P(), c)];

function expectGroupInSync(v: Clip) {
  const members = group(clip(v.id));
  const lead = clip(v.id);
  for (const m of members) {
    expect(m.startMs).toBeCloseTo(lead.startMs, 6);
    expect(m.inMs).toBeCloseTo(lead.inMs, 6);
    expect(m.outMs).toBeCloseTo(lead.outMs, 6);
    expect(m.speed).toEqual(lead.speed);
    expect(m.freezeFrame).toEqual(lead.freezeFrame);
    expect(m.reversed).toBe(lead.reversed);
  }
}

beforeEach(() => {
  setup();
});

/* ------------------------------------------------------------ separate */

describe('separate to tracks', () => {
  it('creates stem assets, Voice / Background tracks and aligned clips in the same link group, and mutes the original audio', () => {
    const v = video(1);
    const mirror = partnersOf(P(), v)[0];
    S().setClipAudio(mirror.id, { gainDb: -3, fadeInMs: 400 });
    const r = S().separateToTracks([v.id]);
    expect(r.done).toEqual([v.id]);
    expect(S().past).toHaveLength(2); // gain + separate: one undo step for the separation
    const voiceTrack = P().tracks.find((t) => t.role === 'voice')!;
    const bgTrack = P().tracks.find((t) => t.role === 'background')!;
    expect(voiceTrack.name).toBe('Voice');
    expect(bgTrack.name).toBe('Background');
    expect(voiceTrack.kind).toBe('audio');
    // Voice right after the existing audio track, Background after it
    const idx = (id: string) => P().tracks.findIndex((t) => t.id === id);
    expect(idx(bgTrack.id)).toBe(idx(voiceTrack.id) + 1);

    const va = stemAssetOf(P(), v.assetId, 'vocals')!;
    const ba = stemAssetOf(P(), v.assetId, 'background')!;
    expect(va).toMatchObject({ kind: 'audio', path: STEMS(1).vocals, durationMs: 20000, name: 'clip1 · Voice', stemOf: { assetId: v.assetId, stem: 'vocals' } });
    expect(ba).toMatchObject({ kind: 'audio', path: STEMS(1).background, name: 'clip1 · Background', stemOf: { assetId: v.assetId, stem: 'background' } });

    const members = group(clip(v.id));
    expect(members).toHaveLength(4); // video, original audio, voice, background
    const vs = voiceTrack.clips[0];
    const bs = bgTrack.clips[0];
    for (const s of [vs, bs]) {
      expect(s.linkId).toBe(v.linkId);
      expect(s.startMs).toBeCloseTo(v.startMs, 6);
      expect(s.inMs).toBe(v.inMs);
      expect(s.outMs).toBe(v.outMs);
      expect(s.speed).toEqual(v.speed);
      expect(s.audio.muted).toBe(false);
      // the stems sound like the original did: same gain and fades
      expect(s.audio.gainDb).toBe(-3);
      expect(s.audio.fadeInMs).toBe(400);
    }
    expect(vs.assetId).toBe(va.id);
    expect(bs.assetId).toBe(ba.id);
    expect(clip(mirror.id).audio.muted).toBe(true);
    // idempotent
    expect(hasStemClips(P(), v.id)).toBe(true);
    S().separateToTracks([v.id]);
    expect(voiceTrack.clips).toHaveLength(1);
    expect(P().tracks.filter((t) => t.role === 'voice')).toHaveLength(1);
    // one undo removes tracks, clips and assets again
    S().undo();
    S().undo();
    expect(P().tracks.some((t) => t.role)).toBe(false);
    expect(clip(mirror.id).audio.muted).toBe(false);
  });

  it('keeps what was heard: a clip in "Isolate voice" mode gets a muted background stem', () => {
    const v = video(0);
    S().setClipVoice(v.id, 'voice');
    S().separateToTracks([v.id]);
    const stems = group(clip(v.id)).filter((c) => P().assets.find((a) => a.id === c.assetId)?.stemOf);
    const bg = stems.find((c) => P().assets.find((a) => a.id === c.assetId)!.stemOf!.stem === 'background')!;
    const vo = stems.find((c) => P().assets.find((a) => a.id === c.assetId)!.stemOf!.stem === 'vocals')!;
    expect(bg.audio.muted).toBe(true);
    expect(vo.audio.muted).toBe(false);
  });

  it('works from the audio clip too, and "apply to all" separates every clip once', () => {
    const v = video(2);
    const a = partnersOf(P(), v)[0];
    S().separateToTracks([a.id]);
    expect(group(clip(v.id))).toHaveLength(4);
    const rest = separableClipIds(P());
    expect(rest).toHaveLength(3); // clips 0, 1, 3
    S().separateToTracks(rest);
    expect(P().tracks.find((t) => t.role === 'voice')!.clips).toHaveLength(4);
    expect(P().tracks.find((t) => t.role === 'background')!.clips).toHaveLength(4);
    expect(separableClipIds(P())).toHaveLength(0);
  });

  it('reports clips whose source has no stems yet and queues them until the stems arrive', () => {
    setup(3, false);
    const v = video(0);
    const r = S().separateToTracks([v.id]);
    expect(r.done).toHaveLength(0);
    expect(r.waiting).toEqual([v.id]);
    expect(r.missing.map((x) => x.path)).toEqual(['C:/clips/clip0.mp4']);
    S().queueStemTracks(r.waiting);
    expect(S().runPendingStemTracks()).toBe(0);
    expect(S().pendingStemTracks).toEqual([v.id]);
    S().setAssetStems('C:\\clips\\clip0.mp4', STEMS(0));
    expect(S().runPendingStemTracks()).toBe(1);
    expect(S().pendingStemTracks).toEqual([]);
    expect(group(clip(v.id))).toHaveLength(4);
    // a failed separation drops the queued request
    const w = video(1);
    S().queueStemTracks([w.id]);
    S().runPendingStemTracks(['C:/clips/clip1.mp4']);
    expect(S().pendingStemTracks).toEqual([]);
  });

  it('stem clips keep their own audio settings; the video clip still drives its mirror', () => {
    const v = video(1);
    S().separateToTracks([v.id]);
    const stem = P().tracks.find((t) => t.role === 'voice')!.clips[0];
    const mirror = partnersOf(P(), clip(v.id)).find((c) => c.assetId === v.assetId)!;
    S().setClipAudio(stem.id, { gainDb: 6 });
    expect(clip(stem.id).audio.gainDb).toBe(6);
    expect(clip(mirror.id).audio.gainDb).toBe(0);
    expect(clip(v.id).audio.gainDb).toBe(0);
    S().setClipAudio(v.id, { gainDb: -6 });
    expect(clip(mirror.id).audio.gainDb).toBe(-6);
    expect(clip(stem.id).audio.gainDb).toBe(6);
  });
});

/* ----------------------------------------------------- N-member link groups */

describe('link groups with more than two members', () => {
  beforeEach(() => {
    S().separateToTracks([video(1).id]);
    useEditor.setState({ past: [], future: [] });
  });

  it('move, trim, speed, freeze and reverse apply to every member', () => {
    const v = video(1);
    expect(group(v)).toHaveLength(4);
    S().trimClip(v.id, 'in', 800);
    expectGroupInSync(v);
    S().trimClip(v.id, 'out', -500);
    expectGroupInSync(v);
    S().setConstantSpeed([v.id], 2);
    expectGroupInSync(v);
    expect(clipDurationMs(clip(v.id))).toBeCloseTo((5000 - 800 / 1 - 500) / 2, 0);
    S().freezeFrameAt(v.id, clip(v.id).startMs + 300, 700);
    expectGroupInSync(v);
    S().toggleReverse(v.id);
    expectGroupInSync(v);
    // dragging a stem member moves the whole group (the video drives)
    const stem = P().tracks.find((t) => t.role === 'background')!.clips[0];
    S().moveClip(stem.id, 0);
    expect(video(0).id).toBe(v.id);
    expectGroupInSync(v);
  });

  it('split gives every member a right half in one new group; delete removes the whole group', () => {
    const v = video(1);
    S().splitClipAt(v.id, v.startMs + 2000);
    const right = video(2);
    expect(right.linkId).not.toBe(v.linkId);
    expect(group(right)).toHaveLength(4);
    expect(group(clip(v.id))).toHaveLength(4);
    expectGroupInSync(right);
    expectGroupInSync(v);
    const before = P().tracks.reduce((n, t) => n + t.clips.length, 0);
    S().deleteClips([P().tracks.find((t) => t.role === 'voice')!.clips[0].id]);
    expect(P().tracks.reduce((n, t) => n + t.clips.length, 0)).toBe(before - 4);
  });
});

/* ---------------------------------------------------------- transitions */

describe('transitions', () => {
  const dissolve = (durationMs = 500) => ({ type: 'dissolve' as const, durationMs });

  it('sit on the incoming clip, need a cut, and clamp 100..3000 and to the shorter clip', () => {
    const [a, b] = [video(0), video(1)];
    S().setTransition(a.id, dissolve()); // first clip: no previous clip, refused
    expect(clip(a.id).transitionIn).toBeUndefined();
    S().setTransition(b.id, dissolve(20));
    expect(clip(b.id).transitionIn).toEqual({ type: 'dissolve', durationMs: 100 });
    S().setTransition(b.id, dissolve(9000));
    expect(clip(b.id).transitionIn!.durationMs).toBe(3000);
    // shorten the outgoing clip to 1.2 s: the transition follows
    S().trimClip(a.id, 'out', -3800);
    expect(clipDurationMs(clip(a.id))).toBeCloseTo(1200, 3);
    expect(clip(b.id).transitionIn!.durationMs).toBeCloseTo(1200, 3);
    expect(prevAdjacent(videoTrack(), clip(b.id), frameMs(P()))!.id).toBe(a.id);
  });

  it('survive splitting either clip (the left half of the incoming clip keeps it)', () => {
    const [a, b] = [video(0), video(1)];
    S().setTransition(b.id, { type: 'wipeLeft', durationMs: 600 });
    S().splitClipAt(a.id, a.startMs + 2500); // split the outgoing clip: its right half is the new previous clip
    expect(clip(b.id).transitionIn).toEqual({ type: 'wipeLeft', durationMs: 600 });
    S().splitClipAt(b.id, clip(b.id).startMs + 2000); // split the incoming clip
    expect(clip(b.id).transitionIn).toEqual({ type: 'wipeLeft', durationMs: 600 });
    const rightHalf = video(3);
    expect(rightHalf.transitionIn).toBeUndefined();
    expect(cutsOfTrack(videoTrack(), frameMs(P()))).toHaveLength(5); // 6 clips after two splits
  });

  it('survive trims and speed changes that keep the cut (magnet)', () => {
    const [a, b] = [video(0), video(1)];
    S().setTransition(b.id, dissolve(800));
    S().trimClip(a.id, 'out', -1000);
    S().trimClip(b.id, 'in', 500);
    S().setConstantSpeed([a.id], 1.5);
    expect(clip(b.id).transitionIn).toEqual({ type: 'dissolve', durationMs: 800 });
  });

  it('are dropped when the previous clip moves away, the cut opens or the previous clip is deleted', () => {
    const [a, b, c] = [video(0), video(1), video(2)];
    S().setTransition(b.id, dissolve());
    S().setTransition(c.id, dissolve());
    // reorder: the clip before b changes
    S().moveClip(a.id, clipEndMs(clip(c.id)) + 10);
    expect(clip(b.id).transitionIn).toBeUndefined();
    expect(clip(c.id).transitionIn).toBeDefined(); // c still follows b
    S().undo();
    expect(clip(b.id).transitionIn).toBeDefined();
    // without the magnet, trimming the outgoing clip opens a gap
    useEditor.setState({ rippleEdit: false });
    S().trimClip(a.id, 'out', -1000);
    expect(clip(b.id).transitionIn).toBeUndefined();
    useEditor.setState({ rippleEdit: true });
    S().undo();
    S().deleteClips([b.id]);
    expect(clip(c.id).transitionIn).toBeUndefined(); // now follows a, not b
  });

  it('"apply to all cuts" sets every cut of the main track; removing clears one', () => {
    const n = S().applyTransitionToAllCuts({ type: 'flash', durationMs: 400 });
    expect(n).toBe(3);
    expect(videoTrack().clips.filter((c) => c.transitionIn?.type === 'flash')).toHaveLength(3);
    S().setTransition(video(2).id, null);
    expect(video(2).transitionIn).toBeUndefined();
  });

  it('unknown types and bad durations are cleaned on load', () => {
    const p = P();
    const bad = { ...p, tracks: p.tracks.map((t) => ({ ...t, clips: t.clips.map((c, i) => (i === 1 ? { ...c, transitionIn: { type: 'spin' as never, durationMs: 500 } } : i === 2 ? { ...c, transitionIn: { type: 'dissolve' as const, durationMs: NaN } } : c)) })) };
    const s = sanitizeProject(bad);
    const vt = s.tracks.find((t) => t.kind === 'video')!;
    expect(vt.clips[1].transitionIn).toBeUndefined();
    expect(vt.clips[2].transitionIn).toEqual({ type: 'dissolve', durationMs: 500 });
  });
});

/* ------------------------------------------------------------- effects */

describe('effect clips', () => {
  it('are added at the playhead with their default length, stack on new FX tracks, trim and move freely', () => {
    S().setPlayhead(1000);
    const e = S().addEffect('cameraSnap')!;
    expect(e.startMs).toBe(1000);
    expect(clipDurationMs(e)).toBe(1500);
    expect(e.assetId).toBe('');
    expect(findClip(P(), e.id)!.track.kind).toBe('fx');
    const e2 = S().addEffect('shake', 1200)!;
    expect(findClip(P(), e2.id)!.track.id).not.toBe(findClip(P(), e.id)!.track.id);
    expect(P().tracks.filter((t) => t.kind === 'fx')).toHaveLength(2);
    // trim the in edge left: the start moves, the length grows
    S().trimClip(e.id, 'in', -400);
    expect(clip(e.id).startMs).toBe(600);
    expect(clipDurationMs(clip(e.id))).toBe(1900);
    S().trimClip(e.id, 'out', 600);
    expect(clipDurationMs(clip(e.id))).toBe(2500);
    // speed does not apply
    S().setConstantSpeed([e.id], 2);
    expect(clipDurationMs(clip(e.id))).toBe(2500);
    S().moveClip(e.id, 5000);
    expect(clip(e.id).startMs).toBe(5000);
  });
});

/* ---------------------------------------------------- fades, volume, speed */

describe('audio fades and volume keyframes', () => {
  it('fades are clamped to half the clip and apply to the audio mirror', () => {
    const v = video(0);
    const a = partnersOf(P(), v)[0];
    S().setClipAudio(a.id, { fadeInMs: 9000, fadeOutMs: 1000 });
    expect(clip(a.id).audio.fadeInMs).toBe(2500);
    expect(clip(v.id).audio.fadeInMs).toBe(2500);
    expect(clip(a.id).audio.fadeOutMs).toBe(1000);
  });

  it('a fade drag is one undo step', () => {
    const a = partnersOf(P(), video(0))[0];
    S().beginGesture();
    for (let ms = 50; ms <= 1500; ms += 50) S().setClipAudio(a.id, { fadeOutMs: ms });
    S().endGesture();
    expect(S().past).toHaveLength(1);
    S().undo();
    expect(clip(a.id).audio.fadeOutMs).toBeUndefined();
  });

  it('volume keyframes are set on the pair, move with trims and split', () => {
    const v = video(0);
    const a = partnersOf(P(), v)[0];
    S().setVolumeKeyframe(a.id, 1000, -6);
    S().setVolumeKeyframe(a.id, 3000, 0);
    expect(clip(v.id).audio.volume!.keyframes.map((k) => [k.timeMs, k.value])).toEqual([
      [1000, -6],
      [3000, 0],
    ]);
    S().trimClip(v.id, 'in', 500);
    // like transform keyframes: the value at the new start is kept by a key at 0
    expect(clip(a.id).audio.volume!.keyframes.map((k) => [k.timeMs, k.value])).toEqual([
      [0, -6],
      [500, -6],
      [2500, 0],
    ]);
    S().removeVolumeKeyframe(a.id, 500);
    expect(clip(a.id).audio.volume!.keyframes.map((k) => k.timeMs)).toEqual([0, 2500]);
  });

  it('split: the left half keeps the fade-in, the right half the fade-out', () => {
    const v = video(0);
    S().setClipAudio(v.id, { fadeInMs: 300, fadeOutMs: 700 });
    S().updateClip(v.id, { fadeInMs: 200, fadeOutMs: 400 });
    S().splitClipAt(v.id, v.startMs + 2000);
    const left = clip(v.id);
    const right = video(1);
    expect(left.audio.fadeInMs).toBe(300);
    expect(left.audio.fadeOutMs).toBeUndefined();
    expect(right.audio.fadeOutMs).toBe(700);
    expect(right.audio.fadeInMs).toBeUndefined();
    expect(left.fadeInMs).toBe(200);
    expect(left.fadeOutMs).toBeUndefined();
    expect(right.fadeOutMs).toBe(400);
  });
});

describe('quick speed buttons', () => {
  it('set a constant speed and ripple the main track', () => {
    const [a, b] = [video(0), video(1)];
    S().setConstantSpeed([a.id], 0.5);
    expect(clipDurationMs(clip(a.id))).toBeCloseTo(10000, 3);
    expect(clip(b.id).startMs).toBeCloseTo(10000, 3);
    expect(clip(a.id).speed.points.every((p) => p.speed === 0.5)).toBe(true);
    S().setConstantSpeed([a.id], 1);
    expect(clip(a.id).speed.preset).toBe('normal');
    expect(clip(b.id).startMs).toBeCloseTo(5000, 3);
    expect(S().past).toHaveLength(2);
  });
});

describe('project settings', () => {
  it('frame rate and interpolation are one undoable change; old projects default to optical flow', () => {
    expect(P().frameInterpolation).toBeUndefined();
    S().setProjectSettings({ fps: 60, frameInterpolation: 'frameBlend' });
    expect(P().fps).toBe(60);
    expect(P().frameInterpolation).toBe('frameBlend');
    S().undo();
    expect(P().fps).toBe(24);
  });
});
