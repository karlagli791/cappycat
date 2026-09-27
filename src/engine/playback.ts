/**
 * Playback resolution helpers: which clip is under the playhead, what source
 * time it maps to (speed ramps, freeze frames, reverse), and what the reframe
 * engine's crop is at that moment.
 */
import type { BBox, Clip, ClipMask, ClipTransition, Project, Rect, ReframeTrack, Track } from '@/types/project';
import { evaluate } from './keyframes';
import { isConstantSpeed, isUniformSpeed, outputToSourceMs, speedAt, speedFactor, speedLutFor as curveLut, type SpeedLut } from './speed';
import { clipDurationMs, clipsAt } from '@/state/store';
import { frameMs, nextAdjacent, prevAdjacent, transitionDurationMs } from '@/state/edits';
import { transitionProgress } from './transitions';
import { videoFadeFactor } from './audio';

function speedLutFor(clip: Clip): SpeedLut | null {
  if (isUniformSpeed(clip.speed)) return null;
  return curveLut(clip.speed, clip.outMs - clip.inMs);
}

export interface ResolvedFrame {
  clip: Clip;
  /** time within the clip on the timeline (ms) */
  localMs: number;
  /** absolute source time in the asset (ms) */
  sourceMs: number;
  /** instantaneous playback rate (source ms per timeline ms) */
  rate: number;
  frozen: boolean;
  crop: BBox | null;
  position: [number, number];
  scale: number;
  rotation: number;
  opacity: number;
  blur: number;
  maskRect: Rect | null;
  mask: ClipMask | null;
  /** video fade (to / from black) factor at this time, 1 = none */
  fade: number;
}

/** Top visible video clip at a timeline time (hidden/muted video tracks are skipped). */
export function resolveAt(project: Project, timelineMs: number): ResolvedFrame | null {
  const candidates = clipsAt(project, timelineMs, 'video');
  if (!candidates.length) return null;
  // top-most video track wins (last track in list is drawn on top)
  const clip = candidates[candidates.length - 1];
  return resolveClip(clip, timelineMs - clip.startMs);
}

export function resolveClip(clip: Clip, localMs: number): ResolvedFrame {
  const total = clipDurationMs(clip);
  const local = Math.max(0, Math.min(total, localMs));
  let playableLocal = local;
  let frozen = false;
  if (clip.freezeFrame) {
    const { atMs, holdMs } = clip.freezeFrame;
    if (local >= atMs && local < atMs + holdMs) {
      playableLocal = atMs;
      frozen = true;
    } else if (local >= atMs + holdMs) {
      playableLocal = local - holdMs;
    }
  }
  const srcRange = clip.outMs - clip.inMs;
  const lut = speedLutFor(clip);
  let srcOffset: number;
  let rate: number;
  if (lut) {
    srcOffset = outputToSourceMs(lut, playableLocal);
    rate = speedAt(clip.speed.points, srcRange > 0 ? srcOffset / srcRange : 0);
  } else if (isConstantSpeed(clip.speed)) {
    srcOffset = playableLocal;
    rate = 1;
  } else {
    // uniform speed (e.g. a constant 2x)
    const k = speedFactor(clip.speed);
    srcOffset = playableLocal / k;
    rate = 1 / k;
  }
  if (clip.reversed) srcOffset = srcRange - srcOffset;
  const sourceMs = clip.inMs + Math.max(0, Math.min(srcRange, srcOffset));

  const position = evaluate(clip.transform.position, local);
  const scale = evaluate(clip.transform.scale, local);
  const rotation = evaluate(clip.transform.rotation, local);
  const opacity = evaluate(clip.transform.opacity, local);
  const blur = evaluate(clip.transform.blur, local);
  const maskRect = clip.mask ? evaluate(clip.mask.rect, local) : null;

  return {
    clip,
    localMs: local,
    sourceMs,
    rate: frozen ? 0 : rate,
    frozen,
    crop: clip.reframe ? reframeCropAt(clip.reframe, sourceMs) : null,
    position,
    scale,
    rotation,
    opacity,
    blur,
    maskRect,
    mask: clip.mask,
    fade: videoFadeFactor(clip.fadeInMs, clip.fadeOutMs, local, total),
  };
}

/**
 * Like resolveClip, but past the clip's edges (transition windows): the outgoing clip keeps playing
 * with the source frames after its out point, the incoming one starts early with the frames before
 * its in point, at the boundary speed (speed maps extend linearly); without such handles the first /
 * last frame is held. `assetDurationMs` bounds the source (Infinity for images).
 */
export function resolveClipExtended(clip: Clip, localMs: number, assetDurationMs: number): ResolvedFrame {
  const total = clipDurationMs(clip);
  const r = resolveClip(clip, localMs);
  if (localMs >= 0 && localMs <= total) return r;
  const maxSrc = Number.isFinite(assetDurationMs) && assetDurationMs > 0 ? assetDurationMs : Infinity;
  const after = localMs > total;
  const extra = after ? localMs - total : -localMs;
  // playback-order boundary speed (source ms per timeline ms)
  const u = after ? 1 : 0;
  const v = isUniformSpeed(clip.speed) ? 1 / speedFactor(clip.speed) : speedAt(clip.speed.points, u);
  const src = extra * v;
  // playback moves forward through the source unless reversed
  const forward = after !== clip.reversed;
  const edge = after ? (clip.reversed ? clip.inMs : clip.outMs) : clip.reversed ? clip.outMs : clip.inMs;
  const sourceMs = forward ? Math.min(maxSrc, edge + src) : Math.max(0, edge - src);
  const held = forward ? edge + src > maxSrc : edge - src < 0;
  return {
    ...r,
    localMs,
    sourceMs,
    rate: held ? 0 : v,
    frozen: held,
    crop: clip.reframe ? reframeCropAt(clip.reframe, sourceMs) : r.crop,
  };
}

/** A transition in progress: `a` (outgoing) and `b` (incoming) on the same track. */
export interface ActiveTransition {
  a: Clip;
  b: Clip;
  track: Track;
  transition: ClipTransition;
  /** effective (clamped) duration */
  durationMs: number;
  /** cut time (b.startMs) */
  cutMs: number;
  /** raw window position 0..1 and eased progress p */
  raw: number;
  p: number;
}

/**
 * The transition on the track of `clip` (the clip under the playhead) at `timelineMs`, if any:
 * either `clip`'s own transitionIn (first half: before the cut we are still in the outgoing clip,
 * so this is the second half) or the next clip's (we are in the outgoing clip's last d/2).
 */
export function transitionAt(p: Project, clip: Clip, timelineMs: number): ActiveTransition | null {
  const track = p.tracks.find((t) => t.clips.includes(clip)) ?? p.tracks.find((t) => t.id === clip.trackId);
  if (!track || track.kind !== 'video') return null;
  const frame = frameMs(p);
  const check = (a: Clip, b: Clip): ActiveTransition | null => {
    const d = transitionDurationMs(track, b, frame);
    if (d <= 0 || !b.transitionIn) return null;
    const from = b.startMs - d / 2;
    if (timelineMs < from || timelineMs >= from + d) return null;
    const raw = (timelineMs - from) / d;
    return { a, b, track, transition: b.transitionIn, durationMs: d, cutMs: b.startMs, raw, p: transitionProgress(raw) };
  };
  if (clip.transitionIn) {
    const prev = prevAdjacent(track, clip, frame);
    const t = prev ? check(prev, clip) : null;
    if (t) return t;
  }
  const next = nextAdjacent(track, clip, frame);
  return next ? check(clip, next) : null;
}

/** Interpolate the reframe crop for an absolute source time. */
export function reframeCropAt(track: ReframeTrack, sourceMs: number): BBox | null {
  const keys = track.keyframes;
  if (!keys.length) return null;
  if (sourceMs <= keys[0].timeMs) return keys[0].crop;
  const last = keys[keys.length - 1];
  if (sourceMs >= last.timeMs) return last.crop;
  let lo = 0;
  let hi = keys.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (keys[mid].timeMs <= sourceMs) lo = mid;
    else hi = mid;
  }
  const a = keys[lo];
  const b = keys[hi];
  const span = b.timeMs - a.timeMs;
  const k = span <= 0 ? 0 : (sourceMs - a.timeMs) / span;
  return [
    a.crop[0] + (b.crop[0] - a.crop[0]) * k,
    a.crop[1] + (b.crop[1] - a.crop[1]) * k,
    a.crop[2] + (b.crop[2] - a.crop[2]) * k,
    a.crop[3] + (b.crop[3] - a.crop[3]) * k,
  ];
}
