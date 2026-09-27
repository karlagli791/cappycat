/**
 * Keyframe & motion engine (TypeScript mirror of the Rust `keyframes` crate).
 * Used for real-time preview so the UI never blocks on IPC; the Rust crate is the
 * reference implementation used by the headless exporter.
 */
import type { Easing, Keyframe, Keyframed } from '@/types/project';

export type BezierHandles = [number, number, number, number];

export const EASING_PRESETS: Record<Exclude<Easing, 'bezier'>, BezierHandles | null> = {
  linear: [0, 0, 1, 1],
  easeIn: [0.42, 0, 1, 1],
  easeOut: [0, 0, 0.58, 1],
  easeInOut: [0.42, 0, 0.58, 1],
  bounce: null, // procedural
  elastic: null, // procedural
};

/** Solve a CSS-style cubic-bezier: given x in [0,1] return y. */
export function cubicBezier(x1: number, y1: number, x2: number, y2: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  if (x1 === y1 && x2 === y2) return x;
  const ax = 1 - 3 * x2 + 3 * x1;
  const bx = 3 * x2 - 6 * x1;
  const cx = 3 * x1;
  const ay = 1 - 3 * y2 + 3 * y1;
  const by = 3 * y2 - 6 * y1;
  const cy = 3 * y1;
  const sampleX = (t: number) => ((ax * t + bx) * t + cx) * t;
  const sampleY = (t: number) => ((ay * t + by) * t + cy) * t;
  const sampleDX = (t: number) => (3 * ax * t + 2 * bx) * t + cx;

  // Newton-Raphson
  let t = x;
  for (let i = 0; i < 8; i++) {
    const err = sampleX(t) - x;
    if (Math.abs(err) < 1e-6) return sampleY(t);
    const d = sampleDX(t);
    if (Math.abs(d) < 1e-6) break;
    t -= err / d;
  }
  // Bisection fallback
  let lo = 0;
  let hi = 1;
  t = x;
  while (hi - lo > 1e-6) {
    const cur = sampleX(t);
    if (Math.abs(cur - x) < 1e-6) break;
    if (cur < x) lo = t;
    else hi = t;
    t = (lo + hi) / 2;
  }
  return sampleY(t);
}

export function bounceOut(t: number): number {
  const n1 = 7.5625;
  const d1 = 2.75;
  if (t < 1 / d1) return n1 * t * t;
  if (t < 2 / d1) return n1 * (t -= 1.5 / d1) * t + 0.75;
  if (t < 2.5 / d1) return n1 * (t -= 2.25 / d1) * t + 0.9375;
  return n1 * (t -= 2.625 / d1) * t + 0.984375;
}

export function elasticOut(t: number): number {
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  const c4 = (2 * Math.PI) / 3;
  return Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
}

/** Map normalised time u in [0,1] through an easing definition. */
export function ease(u: number, easing: Easing, handles?: BezierHandles): number {
  const c = Math.min(1, Math.max(0, u));
  switch (easing) {
    case 'linear':
      return c;
    case 'bounce':
      return bounceOut(c);
    case 'elastic':
      return elasticOut(c);
    case 'bezier': {
      const h = handles ?? [0.25, 0.1, 0.25, 1];
      return cubicBezier(h[0], h[1], h[2], h[3], c);
    }
    default: {
      const h = EASING_PRESETS[easing] ?? [0, 0, 1, 1];
      return cubicBezier(h[0], h[1], h[2], h[3], c);
    }
  }
}

export type Interpolable = number | number[];

export function lerpValue<T extends Interpolable>(a: T, b: T, k: number): T {
  if (typeof a === 'number' && typeof b === 'number') {
    return (a + (b - a) * k) as T;
  }
  const av = a as number[];
  const bv = b as number[];
  return av.map((x, i) => x + ((bv[i] ?? x) - x) * k) as T;
}

/**
 * Evaluate a keyframed property at clip-relative time. The easing stored on a
 * keyframe shapes the segment that STARTS at that keyframe (outgoing easing,
 * CapCut/After Effects semantics, identical to the Rust `keyframes` crate);
 * the last keyframe's easing is therefore unused.
 */
export function evaluate<T extends Interpolable>(kf: Keyframed<T>, timeMs: number): T {
  const keys = kf.keyframes;
  if (!keys || keys.length === 0) return kf.static;
  if (keys.length === 1) return keys[0].value;
  const sorted = !isSorted(keys) ? [...keys].sort((a, b) => a.timeMs - b.timeMs) : keys;
  if (timeMs <= sorted[0].timeMs) return sorted[0].value;
  const last = sorted[sorted.length - 1];
  if (timeMs >= last.timeMs) return last.value;
  let lo = 0;
  let hi = sorted.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid].timeMs <= timeMs) lo = mid;
    else hi = mid;
  }
  const a = sorted[lo];
  const b = sorted[hi];
  const span = b.timeMs - a.timeMs;
  const u = span <= 0 ? 1 : (timeMs - a.timeMs) / span;
  const k = ease(u, a.easing, a.bezier);
  return lerpValue(a.value, b.value, k);
}

function isSorted<T>(keys: Keyframe<T>[]): boolean {
  for (let i = 1; i < keys.length; i++) if (keys[i].timeMs < keys[i - 1].timeMs) return false;
  return true;
}

/** Insert or replace a keyframe at the given time. Returns a new Keyframed. */
export function setKeyframe<T extends Interpolable>(
  kf: Keyframed<T>,
  timeMs: number,
  value: T,
  easing: Easing = 'easeInOut',
  bezier?: BezierHandles,
  toleranceMs = 0.5,
): Keyframed<T> {
  const keys = kf.keyframes.filter((k) => Math.abs(k.timeMs - timeMs) > toleranceMs);
  keys.push({ timeMs, value, easing, ...(bezier ? { bezier } : {}) });
  keys.sort((a, b) => a.timeMs - b.timeMs);
  return { ...kf, keyframes: keys };
}

export function removeKeyframe<T>(kf: Keyframed<T>, timeMs: number, toleranceMs = 0.5): Keyframed<T> {
  return { ...kf, keyframes: kf.keyframes.filter((k) => Math.abs(k.timeMs - timeMs) > toleranceMs) };
}

/** Sample an easing into N points for drawing the graph editor. */
export function sampleEasing(easing: Easing, handles: BezierHandles | undefined, n = 64): Array<[number, number]> {
  const pts: Array<[number, number]> = [];
  for (let i = 0; i <= n; i++) {
    const u = i / n;
    pts.push([u, ease(u, easing, handles)]);
  }
  return pts;
}
