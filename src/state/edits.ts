/**
 * Pure timeline edit semantics (no store, no React). The store wraps these with undo, gestures
 * and selection; the tests exercise them directly.
 *
 * Model (CapCut conventions):
 *  - The MAIN track is the first video track. With the magnet on (`rippleEdit`), it stays gapless
 *    from 0: deletes close gaps, trims/speed/freeze ripple, moves are insert-reorders.
 *  - Other tracks: with the magnet on, edits ripple the clips after the edit point on the same
 *    track; with it off, clips keep their place and overlaps are pushed right.
 *  - Linked clips (`linkId`: a video clip, its mirrored audio clip and any stem clips) share their
 *    timing: every edit is applied to the whole group (any number of members), and after each edit
 *    the other members are re-synced to the video member.
 *  - Transitions (`transitionIn`, on the incoming clip) survive an edit while the incoming clip still
 *    follows the same previous clip (or its split right half) without a gap; otherwise they are
 *    dropped. Durations are clamped to the shorter of the two clips (see `settle`).
 *  - Effect clips (`effect`, on fx tracks) last `outMs - inMs`: speed, freeze and reverse do not apply.
 */
import type { Asset, Clip, ClipAudio, ClipMask, ClipTransform, ClipTransition, Easing, FreezeFrame, Keyframe, Keyframed, Project, SpeedCurve, Track } from '@/types/project';
import { outputDuration, outputToSourceOffset, sliceCurve, sourceToOutputMs, speedFactor } from '@/engine/speed';
import { evaluate, type Interpolable } from '@/engine/keyframes';
import { newId } from '@/engine/ids';
import { clampTransitionMs, isTransitionType, TRANSITION_DEFAULT_MS } from '@/engine/transitions';
import { isEffectType } from '@/engine/effects';

/** Shortest clip a trim may leave (timeline ms). */
export const MIN_CLIP_MS = 40;
const EPS = 0.001;

/* ------------------------------------------------------------------ basics */

export function clipDurationMs(clip: Clip): number {
  const src = Math.max(0, clip.outMs - clip.inMs);
  if (clip.effect) return src; // effect clips: speed ignored
  return outputDuration(clip.speed, src) + (clip.freezeFrame?.holdMs ?? 0);
}

/** One frame of the project (ms): the tolerance of a "cut" between two clips. */
export function frameMs(p: Pick<Project, 'fps'>): number {
  return 1000 / (p.fps > 0 ? p.fps : 24);
}

/** The clip on `track` that ends where `clip` starts (gap or overlap under one frame), closest first. */
export function prevAdjacent(track: Track, clip: Clip, frame: number): Clip | null {
  let best: Clip | null = null;
  let bestD = frame;
  for (const c of track.clips) {
    if (c.id === clip.id || c.startMs >= clip.startMs) continue;
    const d = Math.abs(clipEndMs(c) - clip.startMs);
    if (d < bestD) {
      best = c;
      bestD = d;
    }
  }
  return best;
}

/** The clip on `track` that starts where `clip` ends (gap under one frame). */
export function nextAdjacent(track: Track, clip: Clip, frame: number): Clip | null {
  const end = clipEndMs(clip);
  let best: Clip | null = null;
  let bestD = frame;
  for (const c of track.clips) {
    if (c.id === clip.id || c.startMs <= clip.startMs) continue;
    const d = Math.abs(c.startMs - end);
    if (d < bestD) {
      best = c;
      bestD = d;
    }
  }
  return best;
}

/** Cuts of a video track: every clip that directly follows another one (the incoming clips). */
export function cutsOfTrack(track: Track, frame: number): Array<{ prev: Clip; next: Clip; atMs: number }> {
  const out: Array<{ prev: Clip; next: Clip; atMs: number }> = [];
  for (const c of track.clips) {
    const prev = prevAdjacent(track, c, frame);
    if (prev) out.push({ prev, next: c, atMs: c.startMs });
  }
  return out.sort((a, b) => a.atMs - b.atMs);
}

/** The effective (clamped) transition duration of an incoming clip, or 0 when it has none / no cut. */
export function transitionDurationMs(track: Track, clip: Clip, frame: number): number {
  if (!clip.transitionIn) return 0;
  const prev = prevAdjacent(track, clip, frame);
  if (!prev) return 0;
  return clampTransitionMs(clip.transitionIn.durationMs, clipDurationMs(prev), clipDurationMs(clip));
}

export function clipEndMs(clip: Clip): number {
  return clip.startMs + clipDurationMs(clip);
}

export function mainTrackId(p: Project): string | null {
  return p.tracks.find((t) => t.kind === 'video')?.id ?? null;
}

export function findClip(p: Project, clipId: string): { clip: Clip; track: Track } | null {
  for (const track of p.tracks) {
    const clip = track.clips.find((c) => c.id === clipId);
    if (clip) return { clip, track };
  }
  return null;
}

function trackIndexOf(p: Project, clipId: string): number {
  return p.tracks.findIndex((t) => t.clips.some((c) => c.id === clipId));
}

/** Clips sharing `clip`'s linkId (excluding the clip itself). */
export function partnersOf(p: Project, clip: Clip): Clip[] {
  if (!clip.linkId) return [];
  const out: Clip[] = [];
  for (const t of p.tracks) for (const c of t.clips) if (c.linkId === clip.linkId && c.id !== clip.id) out.push(c);
  return out;
}

/** The ids plus every linked partner (skipping partners on locked tracks when `unlockedOnly`). */
export function withPartners(p: Project, ids: Iterable<string>, unlockedOnly = false): Set<string> {
  const out = new Set<string>();
  const links = new Set<string>();
  for (const id of ids) {
    const f = findClip(p, id);
    if (!f) continue;
    if (unlockedOnly && f.track.locked) continue;
    out.add(id);
    if (f.clip.linkId) links.add(f.clip.linkId);
  }
  if (links.size) {
    for (const t of p.tracks) {
      if (unlockedOnly && t.locked) continue;
      for (const c of t.clips) if (c.linkId && links.has(c.linkId)) out.add(c.id);
    }
  }
  return out;
}

function mapClips(p: Project, ids: Set<string>, fn: (c: Clip, track: Track) => Clip): Project {
  let changed = false;
  const tracks = p.tracks.map((t) => {
    if (!t.clips.some((c) => ids.has(c.id))) return t;
    changed = true;
    return { ...t, clips: t.clips.map((c) => (ids.has(c.id) ? fn(c, t) : c)) };
  });
  return changed ? { ...p, tracks } : p;
}

/** Deep structural equality with reference shortcuts (shared subtrees cost O(1)). */
export function sameDoc(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) {
    return typeof a === 'number' && typeof b === 'number' ? Number.isNaN(a) && Number.isNaN(b) : false;
  }
  if (Array.isArray(a)) {
    if (!Array.isArray(b) || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!sameDoc(a[i], b[i])) return false;
    return true;
  }
  if (Array.isArray(b)) return false;
  const ka = Object.keys(a as object).filter((k) => (a as Record<string, unknown>)[k] !== undefined);
  const kb = Object.keys(b as object).filter((k) => (b as Record<string, unknown>)[k] !== undefined);
  if (ka.length !== kb.length) return false;
  for (const k of ka) if (!sameDoc((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k])) return false;
  return true;
}

/* ------------------------------------------------------------------- links */

/**
 * Older projects / analyses without `linkId`: a video clip and an audio clip with the same asset,
 * start and source range are the same shot. Gives each such pair a shared link id.
 */
export function inferLinks(p: Project): Project {
  const audio = p.tracks.filter((t) => t.kind === 'audio').flatMap((t) => t.clips.filter((c) => !c.linkId));
  if (!audio.length) return p;
  const used = new Set<string>();
  const assign = new Map<string, string>();
  for (const t of p.tracks) {
    if (t.kind !== 'video') continue;
    for (const v of t.clips) {
      if (v.linkId) continue;
      const a = audio.find(
        (c) =>
          !used.has(c.id) &&
          c.assetId === v.assetId &&
          Math.abs(c.startMs - v.startMs) < 1 &&
          Math.abs(c.inMs - v.inMs) < 1 &&
          Math.abs(c.outMs - v.outMs) < 1,
      );
      if (!a) continue;
      used.add(a.id);
      const id = newId('lnk');
      assign.set(v.id, id);
      assign.set(a.id, id);
    }
  }
  if (!assign.size) return p;
  return mapClips(p, new Set(assign.keys()), (c) => ({ ...c, linkId: assign.get(c.id) }));
}

type Timing = Pick<Clip, 'startMs' | 'inMs' | 'outMs' | 'speed' | 'freezeFrame' | 'reversed'>;

function timingOf(c: Clip): Timing {
  return { startMs: c.startMs, inMs: c.inMs, outMs: c.outMs, speed: c.speed, freezeFrame: c.freezeFrame, reversed: c.reversed };
}

function sameTiming(c: Clip, t: Timing): boolean {
  return (
    c.startMs === t.startMs &&
    c.inMs === t.inMs &&
    c.outMs === t.outMs &&
    c.speed === t.speed &&
    c.reversed === t.reversed &&
    (c.freezeFrame === t.freezeFrame || sameDoc(c.freezeFrame, t.freezeFrame))
  );
}

/** Copy the timing of each link group's video member onto its other members. */
function syncLinks(p: Project): Project {
  const authority = new Map<string, Timing>();
  const main = mainTrackId(p);
  // main track first, then other video tracks
  const order = [...p.tracks].sort((a, b) => (a.id === main ? -1 : b.id === main ? 1 : 0));
  for (const t of order) {
    if (t.kind !== 'video') continue;
    for (const c of t.clips) if (c.linkId && !authority.has(c.linkId)) authority.set(c.linkId, timingOf(c));
  }
  if (!authority.size) return p;
  let changed = false;
  const tracks = p.tracks.map((t) => {
    if (t.kind === 'video') return t;
    let touched = false;
    const clips = t.clips.map((c) => {
      const a = c.linkId ? authority.get(c.linkId) : undefined;
      if (!a || sameTiming(c, a)) return c;
      touched = true;
      return { ...c, ...a };
    });
    if (!touched) return t;
    changed = true;
    return { ...t, clips };
  });
  return changed ? { ...p, tracks } : p;
}

/* ----------------------------------------------------------- track packing */

function byStart(a: Clip, b: Clip): number {
  return a.startMs - b.startMs;
}

/** Gapless from 0 in start order (magnetic main track). */
export function compactTrack(track: Track): Track {
  const clips = [...track.clips].sort(byStart);
  let cursor = 0;
  let changed = clips.some((c, i) => c !== track.clips[i]);
  const out = clips.map((c) => {
    const start = cursor;
    cursor += clipDurationMs(c);
    if (Math.abs(start - c.startMs) < EPS) return c;
    changed = true;
    return { ...c, startMs: start };
  });
  return changed ? { ...track, clips: out } : track;
}

/** Keep start order and push overlapping clips right; `pinned` clips never move. */
export function resolveOverlaps(track: Track, pinned: Set<string> = new Set()): Track {
  const clips = [...track.clips].sort(byStart);
  let cursor = 0;
  let changed = clips.some((c, i) => c !== track.clips[i]);
  const out = clips.map((c) => {
    if (pinned.has(c.id)) {
      cursor = Math.max(cursor, clipEndMs(c));
      return c;
    }
    const start = Math.max(c.startMs, cursor);
    cursor = start + clipDurationMs(c);
    if (Math.abs(start - c.startMs) < EPS) return c;
    changed = true;
    return { ...c, startMs: start };
  });
  return changed ? { ...track, clips: out } : track;
}

/**
 * Settle a project after an edit: pack the main track (gapless with the magnet on, overlaps
 * pushed otherwise), sync linked partners to their video member, then resolve overlaps on the
 * other tracks without moving linked clips.
 */
export function settle(p: Project, magnet: boolean, before?: Project): Project {
  return reconcileTransitions(settleTimings(p, magnet), before);
}

function settleTimings(p: Project, magnet: boolean): Project {
  const main = mainTrackId(p);
  let tracks = p.tracks.map((t) => {
    if (t.id !== main || t.locked) return t;
    return magnet ? compactTrack(t) : resolveOverlaps(t);
  });
  let next: Project = tracks.every((t, i) => t === p.tracks[i]) ? p : { ...p, tracks };
  next = syncLinks(next);
  const linkedVideo = new Set<string>();
  for (const t of next.tracks) if (t.kind === 'video') for (const c of t.clips) if (c.linkId) linkedVideo.add(c.linkId);
  tracks = next.tracks.map((t) => {
    if (t.id === main || t.locked) return t;
    const pinned = new Set(t.clips.filter((c) => c.linkId && (t.kind !== 'video' ? linkedVideo.has(c.linkId) : false)).map((c) => c.id));
    return resolveOverlaps(t, pinned);
  });
  return tracks.every((t, i) => t === next.tracks[i]) ? next : { ...next, tracks };
}

/* ------------------------------------------------------------- transitions */

function withoutTransition(c: Clip): Clip {
  const { transitionIn: _dropped, ...rest } = c;
  return rest as Clip;
}

/** Source point where a clip's playback ends (its out point, or its in point when reversed). */
function playbackSourceEnd(c: Clip): number {
  return c.reversed ? c.inMs : c.outMs;
}

/**
 * Keep each transition only while its cut exists: the incoming clip still directly follows the
 * previous clip it followed in `before` (or that clip's split right half: same asset, same
 * playback end). Durations are clamped to 100..3000 ms and to the shorter of the two clips.
 */
export function reconcileTransitions(p: Project, before?: Project): Project {
  const frame = frameMs(p);
  let changed = false;
  const tracks = p.tracks.map((t) => {
    if (!t.clips.some((c) => c.transitionIn !== undefined)) return t;
    let touched = false;
    const clips = t.clips.map((c) => {
      if (c.transitionIn === undefined) return c;
      const tr = c.transitionIn;
      const prev = t.kind === 'video' && tr ? prevAdjacent(t, c, frame) : null;
      let keep = !!prev;
      if (keep && prev && before) {
        const bf = findClip(before, c.id);
        if (bf) {
          const bprev = bf.track.kind === 'video' && bf.track.id === t.id ? prevAdjacent(bf.track, bf.clip, frameMs(before)) : null;
          if (bf.track.id !== t.id) keep = false;
          else if (bprev && bprev.id !== prev.id) {
            const splitHalf = !findClip(before, prev.id) && prev.assetId === bprev.assetId && Math.abs(playbackSourceEnd(prev) - playbackSourceEnd(bprev)) < 1;
            keep = splitHalf;
          }
        }
      }
      if (!keep || !tr || !prev) {
        touched = true;
        return withoutTransition(c);
      }
      const d = clampTransitionMs(tr.durationMs, clipDurationMs(prev), clipDurationMs(c));
      if (Math.abs(d - tr.durationMs) < 0.5 && isTransitionType(tr.type)) return c;
      touched = true;
      return { ...c, transitionIn: { type: isTransitionType(tr.type) ? tr.type : 'dissolve', durationMs: d } };
    });
    if (!touched) return t;
    changed = true;
    return { ...t, clips };
  });
  return changed ? { ...p, tracks } : p;
}

/**
 * Set (or with null remove) the transition into `clipId` from the clip before it. Refused (no
 * change) when the clip does not directly follow another clip on a video track.
 */
export function setTransition(p: Project, clipId: string, tr: ClipTransition | null): Project {
  const f = findClip(p, clipId);
  if (!f || f.track.locked || f.track.kind !== 'video') return p;
  if (!tr) {
    if (f.clip.transitionIn === undefined) return p;
    return mapClips(p, new Set([clipId]), withoutTransition);
  }
  const prev = prevAdjacent(f.track, f.clip, frameMs(p));
  if (!prev) return p;
  const durationMs = clampTransitionMs(tr.durationMs, clipDurationMs(prev), clipDurationMs(f.clip));
  return mapClips(p, new Set([clipId]), (c) => ({ ...c, transitionIn: { type: tr.type, durationMs } }));
}

/** Put the same transition on every cut of a track (default: the main track). Returns the count. */
export function setTransitionOnAllCuts(p: Project, tr: ClipTransition, trackId?: string): { project: Project; count: number } {
  const t = p.tracks.find((x) => x.id === (trackId ?? mainTrackId(p)));
  if (!t || t.locked || t.kind !== 'video') return { project: p, count: 0 };
  let q = p;
  const cuts = cutsOfTrack(t, frameMs(p));
  for (const cut of cuts) q = setTransition(q, cut.next.id, tr);
  return { project: q, count: cuts.length };
}

/* ----------------------------------------------------------- load hygiene */

/**
 * Older / hand-edited documents: drop unknown transition and effect types, give transitions a
 * duration, and remove effect-less clips from fx tracks. Returns `p` when nothing needed fixing.
 */
export function sanitizeProject(p: Project): Project {
  let changed = false;
  const tracks = p.tracks.map((t) => {
    let touched = false;
    const clips: Clip[] = [];
    for (const c of t.clips) {
      let x = c;
      if (x.transitionIn !== undefined && x.transitionIn !== null) {
        const tr = x.transitionIn;
        if (!isTransitionType(tr.type)) x = withoutTransition(x);
        else if (!Number.isFinite(tr.durationMs)) x = { ...x, transitionIn: { type: tr.type, durationMs: TRANSITION_DEFAULT_MS } };
      } else if (x.transitionIn === null) x = withoutTransition(x);
      if (x.effect && !isEffectType(x.effect.type)) {
        const { effect: _bad, ...rest } = x;
        x = rest as Clip;
      }
      if (t.kind === 'fx' && !x.assetId && !x.effect) {
        touched = true;
        continue;
      }
      if (x !== c) touched = true;
      clips.push(x);
    }
    if (!touched) return t;
    changed = true;
    return { ...t, clips };
  });
  let next: Project = changed ? { ...p, tracks } : p;
  const fi = next.frameInterpolation;
  if (fi !== undefined && fi !== 'opticalFlow' && fi !== 'frameBlend' && fi !== 'none') {
    const { frameInterpolation: _bad, ...rest } = next;
    next = rest as Project;
  }
  return next;
}

/** Shift clips of a track starting at/after `fromMs` by `deltaMs` (skipping `except`). */
function rippleTrack(track: Track, fromMs: number, deltaMs: number, except: Set<string> = new Set()): Track {
  if (Math.abs(deltaMs) < EPS) return track;
  let changed = false;
  const clips = track.clips.map((c) => {
    if (except.has(c.id) || c.startMs < fromMs - EPS) return c;
    changed = true;
    return { ...c, startMs: Math.max(0, c.startMs + deltaMs) };
  });
  return changed ? { ...track, clips } : track;
}

/**
 * Ripple after a clip changed length on a non-main track (magnet on): clips after its old end on
 * its own track shift by the length change. The main track is compacted by `settle` instead.
 */
function rippleAfter(p: Project, driver: Clip, oldEnd: number, delta: number, magnet: boolean): Project {
  if (!magnet || Math.abs(delta) < EPS) return p;
  const ti = trackIndexOf(p, driver.id);
  if (ti < 0) return p;
  const t = p.tracks[ti];
  if (t.id === mainTrackId(p)) return p;
  const tracks = [...p.tracks];
  tracks[ti] = rippleTrack(t, oldEnd, delta, new Set([driver.id]));
  return { ...p, tracks };
}

/* -------------------------------------------------------------- keyframes */

type AnyKeyframed = Keyframed<Interpolable>;

function mapKeyframed<T>(k: Keyframed<T>, fn: (t: number) => number | null): Keyframed<T> {
  if (!k.keyframes.length) return k;
  const keys: Keyframe<T>[] = [];
  for (const kf of k.keyframes) {
    const t = fn(kf.timeMs);
    if (t == null) continue;
    keys.push(t === kf.timeMs ? kf : { ...kf, timeMs: t });
  }
  keys.sort((a, b) => a.timeMs - b.timeMs);
  return { ...k, keyframes: keys };
}

const TRANSFORM_KEYS = ['position', 'scale', 'rotation', 'opacity', 'blur'] as const;

/** Re-time every keyframe of a clip (transform + mask + volume) through `fn` (null drops the key). */
function mapClipKeyframes(c: Clip, fn: (t: number) => number | null): Pick<Clip, 'transform' | 'mask' | 'audio'> {
  const transform = { ...c.transform } as ClipTransform;
  for (const key of TRANSFORM_KEYS) (transform as unknown as Record<string, AnyKeyframed>)[key] = mapKeyframed(c.transform[key] as AnyKeyframed, fn);
  const mask: ClipMask | null = c.mask ? { ...c.mask, rect: mapKeyframed(c.mask.rect, fn) } : null;
  const audio: ClipAudio = c.audio.volume?.keyframes.length ? { ...c.audio, volume: mapKeyframed(c.audio.volume, fn) } : c.audio;
  return { transform, mask, audio };
}

function segmentEasing<T>(keys: Keyframe<T>[], at: number): { easing: Easing; bezier?: [number, number, number, number] } {
  let prev: Keyframe<T> | null = null;
  for (const k of keys) if (k.timeMs <= at + EPS) prev = k;
  const src = prev ?? keys[0];
  return src?.bezier ? { easing: src.easing, bezier: src.bezier } : { easing: src?.easing ?? 'linear' };
}

/** Split a keyframed property at clip-local `at`: both halves keep the value at the cut. */
export function splitKeyframed<T extends Interpolable>(k: Keyframed<T>, at: number): [Keyframed<T>, Keyframed<T>] {
  if (!k.keyframes.length) return [k, k];
  const keys = [...k.keyframes].sort((a, b) => a.timeMs - b.timeMs);
  const value = evaluate(k, at);
  const ease = segmentEasing(keys, at);
  const exact = keys.find((x) => Math.abs(x.timeMs - at) <= 0.5);
  const left = keys.filter((x) => x.timeMs < at - 0.5);
  left.push(exact ? { ...exact, timeMs: at } : { timeMs: at, value, ...ease });
  const right = keys.filter((x) => x.timeMs > at + 0.5).map((x) => ({ ...x, timeMs: x.timeMs - at }));
  right.unshift(exact ? { ...exact, timeMs: 0 } : { timeMs: 0, value, ...ease });
  return [
    { ...k, keyframes: left },
    { ...k, keyframes: right },
  ];
}

/** Keyframes after an in-point trim of `d` ms (content moves left by d): keeps the value at the new start. */
function trimKeyframedIn<T extends Interpolable>(k: Keyframed<T>, d: number): Keyframed<T> {
  if (!k.keyframes.length || Math.abs(d) < EPS) return k;
  if (d < 0) return mapKeyframed(k, (t) => t - d);
  return splitKeyframed(k, d)[1];
}

/* ------------------------------------------------------------- local time */

/** Clip-local timeline ms -> playback offset without the freeze hold. */
function localToPlay(freeze: FreezeFrame | null, t: number): number {
  if (!freeze) return t;
  if (t <= freeze.atMs) return t;
  if (t < freeze.atMs + freeze.holdMs) return freeze.atMs;
  return t - freeze.holdMs;
}

/* ------------------------------------------------------------------- moves */

/**
 * Move clips (with their linked partners) so that `anchorId` starts at `anchorStart`, optionally
 * onto another track of the same kind. Pure function of `base`: calling it again with the same
 * arguments from the drag-start snapshot gives the same result (idempotent drags).
 */
export function moveClips(base: Project, ids: string[], anchorId: string, anchorStart: number, targetTrackId: string | undefined, magnet: boolean): Project {
  const anchorFound = findClip(base, anchorId);
  if (!anchorFound || anchorFound.track.locked) return base;
  const settleMove = (q: Project) => settle(q, magnet, base);
  let anchor = anchorFound.clip;
  let anchorTrack = anchorFound.track;
  // dragging the audio member of a link moves the pair: the video member drives
  if (anchorTrack.kind !== 'video' && anchor.linkId) {
    const v = partnersOf(base, anchor).map((c) => findClip(base, c.id)!).find((f) => f.track.kind === 'video' && !f.track.locked);
    if (v) {
      anchor = v.clip;
      anchorTrack = v.track;
      targetTrackId = undefined;
    }
  }
  const all = withPartners(base, [...ids, anchor.id], true);
  const main = mainTrackId(base);
  const target = targetTrackId ? base.tracks.find((t) => t.id === targetTrackId) : anchorTrack;
  const canRetrack = !!target && !target.locked && target.kind === anchorTrack.kind && target.id !== anchorTrack.id;
  const movingOnAnchorTrack = anchorTrack.clips.filter((c) => all.has(c.id));

  if (magnet && anchorTrack.id === main && (!canRetrack || target!.id === main)) {
    // CapCut main track: insert-reorder, then gapless
    const moving = [...movingOnAnchorTrack].sort(byStart);
    const rest = anchorTrack.clips.filter((c) => !all.has(c.id)).sort(byStart);
    // compare centres in the drag-start layout: the block goes before the first clip whose centre is after its own
    const blockStart = anchorStart - (anchor.startMs - moving[0].startMs);
    const blockDur = moving.reduce((m, c) => m + clipDurationMs(c), 0);
    const centre = blockStart + blockDur / 2;
    let idx = rest.findIndex((c) => c.startMs + clipDurationMs(c) / 2 >= centre);
    if (idx < 0) idx = rest.length;
    const order = [...rest.slice(0, idx), ...moving, ...rest.slice(idx)];
    let pos = 0;
    const packed = order.map((c) => {
      const start = pos;
      pos += clipDurationMs(c);
      return Math.abs(c.startMs - start) < EPS ? c : { ...c, startMs: start };
    });
    // other selected clips (not linked) on other tracks move by the anchor's delta
    const delta = anchorStart - anchor.startMs;
    const tracks = base.tracks.map((t) => {
      if (t.id === anchorTrack.id) return { ...t, clips: packed };
      if (t.locked || !t.clips.some((c) => all.has(c.id) && !c.linkId)) return t;
      return { ...t, clips: t.clips.map((c) => (all.has(c.id) && !c.linkId ? { ...c, startMs: Math.max(0, c.startMs + delta) } : c)) };
    });
    return settleMove({ ...base, tracks });
  }

  // free move by a common delta (never before 0)
  let delta = anchorStart - anchor.startMs;
  let minStart = Infinity;
  for (const t of base.tracks) for (const c of t.clips) if (all.has(c.id)) minStart = Math.min(minStart, c.startMs);
  if (Number.isFinite(minStart)) delta = Math.max(delta, -minStart);
  const retrack = canRetrack && movingOnAnchorTrack.length > 0 ? target! : null;
  const movedToTarget: Clip[] = [];
  let tracks = base.tracks.map((t) => {
    if (t.locked || !t.clips.some((c) => all.has(c.id))) return t;
    const clips: Clip[] = [];
    for (const c of t.clips) {
      if (!all.has(c.id)) {
        clips.push(c);
        continue;
      }
      const moved = { ...c, startMs: Math.max(0, c.startMs + delta) };
      if (retrack && t.id === anchorTrack.id) movedToTarget.push({ ...moved, trackId: retrack.id });
      else clips.push(moved);
    }
    return { ...t, clips };
  });
  if (retrack) tracks = tracks.map((t) => (t.id === retrack.id ? { ...t, clips: [...t.clips, ...movedToTarget] } : t));
  return settleMove({ ...base, tracks });
}

/* ------------------------------------------------------------------- trims */

function sourceBounds(p: Project, c: Clip): { min: number; max: number } {
  const asset: Asset | undefined = p.assets.find((a) => a.id === c.assetId);
  const max = !asset || asset.kind === 'image' || !(asset.durationMs > 0) ? Infinity : asset.durationMs;
  return { min: 0, max };
}

/** New timing of a clip whose `edge` moved by `delta` timeline ms (+ = right). */
function trimmedClip(p: Project, c: Clip, edge: 'in' | 'out', delta: number, magnet: boolean): Clip {
  if (c.effect) {
    // effect clips: free length, the in edge moves the start (no source range)
    const dur = Math.max(0, c.outMs - c.inMs);
    if (edge === 'out') return { ...c, outMs: c.inMs + Math.max(MIN_CLIP_MS, dur + delta) };
    const d = Math.min(dur - MIN_CLIP_MS, Math.max(-c.startMs, delta));
    return { ...c, startMs: c.startMs + d, outMs: c.inMs + (dur - d) };
  }
  const k = speedFactor(c.speed); // timeline ms per source ms
  const dur = clipDurationMs(c);
  const fz = c.freezeFrame;
  const { min, max } = sourceBounds(p, c);
  const minSrc = MIN_CLIP_MS / Math.max(k, 1e-6);
  const range = c.outMs - c.inMs;
  if (edge === 'out') {
    // + extends; the room is the source left after the playback end
    const room = c.reversed ? c.inMs - min : max - c.outMs;
    const d = Math.max(MIN_CLIP_MS - dur, Math.min(room * k, delta));
    let freezeFrame = fz;
    let playDelta = d;
    if (fz && d < 0) {
      const newTotal = dur + d;
      const tail = dur - fz.atMs - fz.holdMs; // playback after the hold
      if (newTotal < fz.atMs + fz.holdMs && newTotal > fz.atMs) {
        freezeFrame = { atMs: fz.atMs, holdMs: newTotal - fz.atMs };
        playDelta = -tail;
      } else if (newTotal <= fz.atMs) {
        freezeFrame = null;
        playDelta = newTotal - (dur - fz.holdMs);
      }
    }
    let src = playDelta / k;
    if (range + src < minSrc) src = Math.min(0, minSrc - range);
    const timing = c.reversed ? { inMs: Math.max(min, c.inMs - src) } : { outMs: Math.min(max, c.outMs + src) };
    return { ...c, ...timing, freezeFrame };
  }
  // in edge: + shortens from the left
  const room = c.reversed ? max - c.outMs : c.inMs - min;
  const d = Math.min(dur - MIN_CLIP_MS, Math.max(-room * k, delta));
  let freezeFrame: FreezeFrame | null = fz;
  let srcDelta: number;
  if (!fz || d <= fz.atMs) {
    srcDelta = d / k;
    if (fz) freezeFrame = { atMs: fz.atMs - d, holdMs: fz.holdMs };
  } else if (d < fz.atMs + fz.holdMs) {
    srcDelta = fz.atMs / k;
    freezeFrame = { atMs: 0, holdMs: fz.atMs + fz.holdMs - d };
  } else {
    srcDelta = (d - fz.holdMs) / k;
    freezeFrame = null;
  }
  if (range - srcDelta < minSrc) srcDelta = Math.max(0, range - minSrc);
  const timing = c.reversed ? { outMs: Math.min(max, c.outMs - srcDelta) } : { inMs: Math.max(min, c.inMs + srcDelta) };
  const kf = {
    transform: Object.fromEntries(TRANSFORM_KEYS.map((key) => [key, trimKeyframedIn(c.transform[key] as AnyKeyframed, d)])) as unknown as ClipTransform,
    mask: c.mask ? { ...c.mask, rect: trimKeyframedIn(c.mask.rect, d) } : null,
    audio: c.audio.volume?.keyframes.length ? { ...c.audio, volume: trimKeyframedIn(c.audio.volume, d) } : c.audio,
  };
  // with the magnet the clip keeps its slot and later clips ripple; without it the start follows the edge
  return { ...c, ...timing, ...kf, freezeFrame, startMs: magnet ? c.startMs : Math.max(0, c.startMs + d) };
}

/**
 * Trim a clip edge by `deltaMs` of TIMELINE time (speed-aware; reversed clips trim the matching
 * source end). Linked partners get the same trim. With the magnet on, later clips ripple.
 */
export function trimClip(base: Project, clipId: string, edge: 'in' | 'out', deltaMs: number, magnet: boolean): Project {
  const found = findClip(base, clipId);
  if (!found || found.track.locked) return base;
  const c = found.clip;
  const next = trimmedClip(base, c, edge, deltaMs, magnet);
  const ids = withPartners(base, [clipId], true);
  const oldEnd = clipEndMs(c);
  // every member re-times its own keyframes, then takes the driver's timing
  let p = mapClips(base, ids, (x) => (x.id === c.id ? next : { ...trimmedClip(base, x, edge, deltaMs, magnet), ...timingOf(next) }));
  p = rippleAfter(p, next, oldEnd, clipDurationMs(next) - clipDurationMs(c) + (next.startMs - c.startMs), magnet);
  return settle(p, magnet, base);
}

/* ------------------------------------------------------------------- split */

function stripUndefined<T extends object>(o: T): T {
  const out = { ...o } as Record<string, unknown>;
  for (const k of Object.keys(out)) if (out[k] === undefined) delete out[k];
  return out as T;
}

function splitOne(c: Clip, local: number, rightId: string, rightLink: string | undefined): [Clip, Clip] {
  if (c.effect) {
    // effect clips have no source: each half keeps the effect over its own span
    const dur = Math.max(0, c.outMs - c.inMs);
    const left: Clip = { ...c, outMs: c.inMs + local };
    const right: Clip = { ...c, id: rightId, startMs: c.startMs + local, inMs: 0, outMs: Math.max(0, dur - local), ...(rightLink ? { linkId: rightLink } : {}) };
    return [left, right];
  }
  const fz = c.freezeFrame;
  let play = local;
  let leftFreeze: FreezeFrame | null = null;
  let rightFreeze: FreezeFrame | null = null;
  if (fz) {
    if (local <= fz.atMs) {
      rightFreeze = { atMs: fz.atMs - local, holdMs: fz.holdMs };
    } else if (local < fz.atMs + fz.holdMs) {
      play = fz.atMs;
      leftFreeze = { atMs: fz.atMs, holdMs: local - fz.atMs };
      rightFreeze = { atMs: 0, holdMs: fz.atMs + fz.holdMs - local };
    } else {
      play = local - fz.holdMs;
      leftFreeze = fz;
    }
  }
  const range = c.outMs - c.inMs;
  const srcOff = Math.max(0, Math.min(range, outputToSourceOffset(c.speed, range, play)));
  const u = range > 0 ? srcOff / range : 0;
  const splitSrc = c.reversed ? c.outMs - srcOff : c.inMs + srcOff;
  const leftRange = c.reversed ? { inMs: splitSrc, outMs: c.outMs } : { inMs: c.inMs, outMs: splitSrc };
  const rightRange = c.reversed ? { inMs: c.inMs, outMs: splitSrc } : { inMs: splitSrc, outMs: c.outMs };
  const lt = { ...c.transform } as ClipTransform;
  const rt = { ...c.transform } as ClipTransform;
  for (const key of TRANSFORM_KEYS) {
    const [a, b] = splitKeyframed(c.transform[key] as AnyKeyframed, local);
    (lt as unknown as Record<string, AnyKeyframed>)[key] = a;
    (rt as unknown as Record<string, AnyKeyframed>)[key] = b;
  }
  let lm: ClipMask | null = c.mask;
  let rm: ClipMask | null = c.mask;
  if (c.mask) {
    const [a, b] = splitKeyframed(c.mask.rect, local);
    lm = { ...c.mask, rect: a };
    rm = { ...c.mask, rect: b };
  }
  // audio: volume keyframes split like transforms; the left half keeps the fade-in, the right the fade-out
  let la: ClipAudio = { ...c.audio, fadeOutMs: undefined };
  let ra: ClipAudio = { ...c.audio, fadeInMs: undefined };
  if (c.audio.volume?.keyframes.length) {
    const [a, b] = splitKeyframed(c.audio.volume, local);
    la = { ...la, volume: a };
    ra = { ...ra, volume: b };
  }
  const left: Clip = stripUndefined({ ...c, ...leftRange, speed: sliceCurve(c.speed, 0, u), freezeFrame: leftFreeze, transform: lt, mask: lm, audio: stripUndefined(la), fadeOutMs: undefined });
  const right: Clip = stripUndefined({
    ...withoutTransition(c),
    ...rightRange,
    id: rightId,
    startMs: c.startMs + local,
    speed: sliceCurve(c.speed, u, 1),
    freezeFrame: rightFreeze,
    transform: rt,
    mask: rm,
    audio: stripUndefined(ra),
    fadeInMs: undefined,
    ...(rightLink ? { linkId: rightLink } : {}),
  });
  return [left, right];
}

/**
 * Split clips (and their linked partners) at a timeline time. Speed-aware and reverse-aware: the
 * left half ends exactly on the source frame where the right half starts; each half keeps its part
 * of the speed ramp, the freeze frame and the keyframes (with the value at the cut on both sides).
 * Returns the new project and the ids of the right halves.
 */
export function splitClips(p: Project, clipIds: string[], timelineMs: number, magnet: boolean): { project: Project; rightIds: string[] } {
  const ids = withPartners(p, clipIds, true);
  const rightIds: string[] = [];
  const newLinks = new Map<string, string>();
  let changed = false;
  const tracks = p.tracks.map((t) => {
    if (t.locked || !t.clips.some((c) => ids.has(c.id))) return t;
    const clips: Clip[] = [];
    for (const c of t.clips) {
      const local = timelineMs - c.startMs;
      if (!ids.has(c.id) || local <= 20 || local >= clipDurationMs(c) - 20) {
        clips.push(c);
        continue;
      }
      let link: string | undefined;
      if (c.linkId) {
        link = newLinks.get(c.linkId) ?? newId('lnk');
        newLinks.set(c.linkId, link);
      }
      const [l, r] = splitOne(c, local, newId('clp'), link);
      clips.push(l, r);
      rightIds.push(r.id);
      changed = true;
    }
    return { ...t, clips };
  });
  if (!changed) return { project: p, rightIds };
  return { project: settle({ ...p, tracks }, magnet, p), rightIds };
}

/* ------------------------------------------------------------------ delete */

/** Delete clips and their linked partners. With the magnet on no gaps are left behind. */
export function deleteClips(p: Project, clipIds: string[], magnet: boolean): Project {
  const ids = withPartners(p, clipIds, true);
  if (!ids.size) return p;
  const main = mainTrackId(p);
  const tracks = p.tracks.map((t) => {
    if (t.locked || !t.clips.some((c) => ids.has(c.id))) return t;
    if (t.id === main || !magnet) return { ...t, clips: t.clips.filter((c) => !ids.has(c.id)) };
    // ripple each gap closed, latest first so earlier start times stay valid
    let track = t;
    const removed = t.clips.filter((c) => ids.has(c.id)).sort((a, b) => b.startMs - a.startMs);
    for (const r of removed) {
      const dur = clipDurationMs(r);
      track = { ...track, clips: track.clips.filter((c) => c.id !== r.id) };
      track = rippleTrack(track, r.startMs + dur, -dur);
    }
    return track;
  });
  return settle({ ...p, tracks }, magnet, p);
}

/* ------------------------------------------------------------------- speed */

/** Map clip-local times from one timing (speed + freeze) to another over the same source range. */
function retimer(oldSpeed: SpeedCurve, oldFz: FreezeFrame | null, newSpeed: SpeedCurve, newFz: FreezeFrame | null, range: number) {
  const toSrc = (play: number) => outputToSourceOffset(oldSpeed, range, play);
  const toPlay = (src: number) => sourceToOutputMs(newSpeed, range, src);
  return (t: number): number => {
    const play = localToPlay(oldFz, t);
    const inHold = !!oldFz && t > oldFz.atMs && t < oldFz.atMs + oldFz.holdMs;
    const np = toPlay(toSrc(play));
    if (!newFz) return np;
    if (inHold) return newFz.atMs + (t - (oldFz as FreezeFrame).atMs);
    return np <= newFz.atMs + EPS ? np : np + newFz.holdMs;
  };
}

function speedClip(c: Clip, curve: SpeedCurve): Clip {
  const range = c.outMs - c.inMs;
  const fz = c.freezeFrame;
  let freezeFrame = fz;
  if (fz) {
    const src = outputToSourceOffset(c.speed, range, fz.atMs);
    freezeFrame = { atMs: sourceToOutputMs(curve, range, src), holdMs: fz.holdMs };
  }
  const map = retimer(c.speed, fz, curve, freezeFrame, range);
  return { ...c, speed: curve, freezeFrame, ...mapClipKeyframes(c, map) };
}

/**
 * Change a clip's speed curve (and its partners'): keyframes and the freeze frame stay on the same
 * source frames, and later clips ripple by the length change.
 */
export function setClipSpeed(p: Project, clipId: string, curve: SpeedCurve, magnet: boolean): Project {
  const found = findClip(p, clipId);
  if (!found || found.track.locked || found.clip.effect) return p;
  const c = found.clip;
  const next = speedClip(c, curve);
  const ids = withPartners(p, [clipId], true);
  let q = mapClips(p, ids, (x) => (x.id === c.id ? next : speedClip(x, curve)));
  q = rippleAfter(q, next, clipEndMs(c), clipDurationMs(next) - clipDurationMs(c), magnet);
  return settle(q, magnet, p);
}

/** Quick speed buttons: a constant curve (keeps the optical-flow flag), rippling like any speed change. */
export function constantSpeedCurve(speed: number, opticalFlow: boolean): SpeedCurve {
  return {
    preset: speed === 1 ? 'normal' : 'custom',
    points: [
      { t: 0, speed },
      { t: 1, speed },
    ],
    opticalFlow,
  };
}

/* ------------------------------------------------------------------ freeze */

function freezeClip(c: Clip, local: number, holdMs: number): Clip {
  const old = c.freezeFrame;
  const atMs = Math.max(0, localToPlay(old, Math.max(0, Math.min(clipDurationMs(c), local))));
  const fz = { atMs, holdMs };
  const map = (t: number) => {
    const play = localToPlay(old, t);
    return play <= atMs + EPS ? play : play + holdMs;
  };
  return { ...c, freezeFrame: fz, ...mapClipKeyframes(c, map) };
}

/** Freeze frame at a timeline time (replaces an existing one; ripples by the hold change). */
export function freezeFrameAt(p: Project, clipId: string, timelineMs: number, holdMs: number, magnet: boolean): Project {
  const found = findClip(p, clipId);
  if (!found || found.track.locked || found.clip.effect) return p;
  const c = found.clip;
  const local = timelineMs - c.startMs;
  const next = freezeClip(c, local, holdMs);
  const ids = withPartners(p, [clipId], true);
  let q = mapClips(p, ids, (x) => (x.id === c.id ? next : freezeClip(x, local, holdMs)));
  q = rippleAfter(q, next, clipEndMs(c), clipDurationMs(next) - clipDurationMs(c), magnet);
  return settle(q, magnet, p);
}

export function toggleReverse(p: Project, clipId: string): Project {
  const found = findClip(p, clipId);
  if (!found || found.track.locked || found.clip.effect) return p;
  const reversed = !found.clip.reversed;
  return settle(mapClips(p, withPartners(p, [clipId], true), (c) => ({ ...c, reversed })), false, p);
}

/* -------------------------------------------------------------- add clips */

export function makeClip(asset: Asset, trackId: string, startMs: number, defaults: Pick<Clip, 'speed' | 'transform' | 'color' | 'audio'>): Clip {
  return {
    id: newId('clp'),
    assetId: asset.id,
    trackId,
    startMs,
    inMs: 0,
    outMs: asset.kind === 'image' ? 5000 : asset.durationMs,
    ...defaults,
    mask: null,
    blendMode: 'normal',
    reframe: null,
    label: asset.name,
    freezeFrame: null,
    reversed: false,
  };
}

/** Put clips into a (main) track in insert order at `atMs` (before the clip whose middle is after it). */
export function insertOrdered(track: Track, clip: Clip, atMs: number): Track {
  const rest = [...track.clips].sort(byStart);
  let idx = rest.findIndex((c) => c.startMs + clipDurationMs(c) / 2 > atMs);
  if (idx < 0) idx = rest.length;
  const order = [...rest.slice(0, idx), clip, ...rest.slice(idx)];
  let pos = 0;
  return {
    ...track,
    clips: order.map((c) => {
      const s = pos;
      pos += clipDurationMs(c);
      return { ...c, startMs: s };
    }),
  };
}

/* -------------------------------------------------------------- effects */

/** A new effect clip (fx track, no asset): `durationMs` long from `startMs`. */
export function makeEffectClip(effect: NonNullable<Clip['effect']>, trackId: string, startMs: number, durationMs: number, defaults: Pick<Clip, 'speed' | 'transform' | 'color' | 'audio'>, label: string): Clip {
  return {
    id: newId('clp'),
    assetId: '',
    trackId,
    startMs: Math.max(0, startMs),
    inMs: 0,
    outMs: Math.max(MIN_CLIP_MS, durationMs),
    ...defaults,
    mask: null,
    blendMode: 'normal',
    reframe: null,
    label,
    freezeFrame: null,
    reversed: false,
    effect,
  };
}

/** Is [start, end) free on a track (ignoring `except`)? */
export function spanIsFree(track: Track, start: number, end: number, except?: string): boolean {
  return !track.clips.some((c) => c.id !== except && c.startMs < end - EPS && clipEndMs(c) > start + EPS);
}

/* ----------------------------------------------------------- audio mirror */

/**
 * The clips whose AUDIO settings change together with `clip`: the clip and the members of its link
 * group that play the same asset (the video clip and its mirrored audio clip). Stem clips of the
 * group (other assets) keep their own volume, fades and mute.
 */
export function audioMirrorIds(p: Project, clip: Clip): string[] {
  return [clip.id, ...partnersOf(p, clip).filter((c) => c.assetId === clip.assetId).map((c) => c.id)];
}
