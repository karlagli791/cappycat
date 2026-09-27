/**
 * Speed ramping engine: CapCut-style preset curves and custom speed ramps.
 * Speed points are (t, speed) with t normalised over the SOURCE range; output
 * (timeline) time is the integral of 1/speed over source time.
 */
import type { SpeedCurve, SpeedPoint, SpeedPreset } from '@/types/project';

export const SPEED_MIN = 0.1;
export const SPEED_MAX = 10;

export const SPEED_PRESETS: Record<Exclude<SpeedPreset, 'custom'>, SpeedPoint[]> = {
  normal: [
    { t: 0, speed: 1 },
    { t: 1, speed: 1 },
  ],
  montage: [
    { t: 0, speed: 0.6 },
    { t: 0.2, speed: 2.2 },
    { t: 0.4, speed: 0.6 },
    { t: 0.6, speed: 2.2 },
    { t: 0.8, speed: 0.6 },
    { t: 1, speed: 1.5 },
  ],
  hero_time: [
    { t: 0, speed: 1 },
    { t: 0.3, speed: 2.5 },
    { t: 0.5, speed: 0.25 },
    { t: 0.7, speed: 0.25 },
    { t: 1, speed: 1.5 },
  ],
  bullet: [
    { t: 0, speed: 3 },
    { t: 0.35, speed: 3 },
    { t: 0.5, speed: 0.15 },
    { t: 0.65, speed: 3 },
    { t: 1, speed: 3 },
  ],
  jump_cut: [
    { t: 0, speed: 1 },
    { t: 0.24, speed: 1 },
    { t: 0.26, speed: 5 },
    { t: 0.49, speed: 5 },
    { t: 0.51, speed: 1 },
    { t: 0.74, speed: 1 },
    { t: 0.76, speed: 5 },
    { t: 1, speed: 5 },
  ],
  flash_in: [
    { t: 0, speed: 4 },
    { t: 0.3, speed: 1.5 },
    { t: 1, speed: 1 },
  ],
  flash_out: [
    { t: 0, speed: 1 },
    { t: 0.7, speed: 1.5 },
    { t: 1, speed: 4 },
  ],
};

export function presetCurve(preset: SpeedPreset, opticalFlow = true): SpeedCurve {
  if (preset === 'custom') {
    return { preset, points: SPEED_PRESETS.normal.map((p) => ({ ...p })), opticalFlow };
  }
  return { preset, points: SPEED_PRESETS[preset].map((p) => ({ ...p })), opticalFlow };
}

function clampSpeed(s: number): number {
  return Math.min(SPEED_MAX, Math.max(SPEED_MIN, s));
}

function normalisePoints(points: SpeedPoint[]): SpeedPoint[] {
  const pts = points
    .map((p) => ({ t: Math.min(1, Math.max(0, p.t)), speed: clampSpeed(p.speed) }))
    .sort((a, b) => a.t - b.t);
  const out: SpeedPoint[] = [];
  for (const p of pts) {
    if (out.length && Math.abs(out[out.length - 1].t - p.t) < 1e-9) out[out.length - 1] = p;
    else out.push(p);
  }
  return out;
}

interface PreparedCurve {
  xs: number[];
  ys: number[];
  h: number[];
  m: number[];
}

/** Normalised points + monotone tangents, computed once per (immutable) points array. */
const prepared = new WeakMap<SpeedPoint[], PreparedCurve>();

function prepare(points: SpeedPoint[]): PreparedCurve {
  const hit = prepared.get(points);
  if (hit) return hit;
  const pts = normalisePoints(points);
  const n = pts.length;
  const xs = pts.map((p) => p.t);
  const ys = pts.map((p) => p.speed);
  const h: number[] = [];
  const d: number[] = [];
  for (let i = 0; i < n - 1; i++) {
    h.push(xs[i + 1] - xs[i]);
    d.push((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]));
  }
  const m: number[] = new Array(n).fill(0);
  if (n >= 2) {
    m[0] = d[0];
    m[n - 1] = d[n - 2];
    for (let i = 1; i < n - 1; i++) {
      m[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
    }
    for (let i = 0; i < n - 1; i++) {
      if (d[i] === 0) {
        m[i] = 0;
        m[i + 1] = 0;
        continue;
      }
      const a = m[i] / d[i];
      const b = m[i + 1] / d[i];
      const s = a * a + b * b;
      if (s > 9) {
        const tau = 3 / Math.sqrt(s);
        m[i] = tau * a * d[i];
        m[i + 1] = tau * b * d[i];
      }
    }
  }
  const out = { xs, ys, h, m };
  prepared.set(points, out);
  return out;
}

/** Fritsch-Carlson monotone cubic interpolation of speed over t in [0,1]. */
export function speedAt(points: SpeedPoint[], t: number): number {
  const { xs, ys, h, m } = prepare(points);
  const n = xs.length;
  if (n === 0) return 1;
  if (n === 1) return ys[0];
  if (t <= xs[0]) return ys[0];
  if (t >= xs[n - 1]) return ys[n - 1];
  // binary search for the segment
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= t) lo = mid;
    else hi = mid;
  }
  const i = lo;
  const hh = h[i];
  const u = (t - xs[i]) / hh;
  const u2 = u * u;
  const u3 = u2 * u;
  const h00 = 2 * u3 - 3 * u2 + 1;
  const h10 = u3 - 2 * u2 + u;
  const h01 = -2 * u3 + 3 * u2;
  const h11 = u3 - u2;
  return clampSpeed(h00 * ys[i] + h10 * hh * m[i] + h01 * ys[i + 1] + h11 * hh * m[i + 1]);
}

export interface SpeedLut {
  /** source time (ms, relative to inMs) for each sample of output time */
  outputToSource: Float64Array;
  outputDurationMs: number;
  sourceDurationMs: number;
  samples: number;
}

/**
 * Build a lookup table mapping output (timeline) time to source time by
 * integrating 1/speed over the source range with the trapezoid rule.
 */
export function buildSpeedLut(curve: SpeedCurve, sourceDurationMs: number, samples = 512): SpeedLut {
  const n = Math.max(2, samples);
  const cum = new Float64Array(n);
  let prevInv = 1 / speedAt(curve.points, 0);
  cum[0] = 0;
  for (let i = 1; i < n; i++) {
    const t = i / (n - 1);
    const inv = 1 / speedAt(curve.points, t);
    cum[i] = cum[i - 1] + ((prevInv + inv) / 2) * (sourceDurationMs / (n - 1));
    prevInv = inv;
  }
  const outputDurationMs = cum[n - 1];
  const outputToSource = new Float64Array(n);
  let j = 0;
  for (let i = 0; i < n; i++) {
    const target = (i / (n - 1)) * outputDurationMs;
    while (j < n - 2 && cum[j + 1] < target) j++;
    const span = cum[j + 1] - cum[j];
    const k = span <= 0 ? 0 : (target - cum[j]) / span;
    outputToSource[i] = ((j + k) / (n - 1)) * sourceDurationMs;
  }
  outputToSource[n - 1] = sourceDurationMs;
  return { outputToSource, outputDurationMs, sourceDurationMs, samples: n };
}

/** Normal (1x everywhere) speed. */
export function isConstantSpeed(curve: SpeedCurve): boolean {
  return isUniformSpeed(curve) && Math.abs((curve.points[0]?.speed ?? 1) - 1) < 1e-9;
}

/** The same speed everywhere (e.g. a constant 2x). */
export function isUniformSpeed(curve: SpeedCurve): boolean {
  const s0 = curve.points[0]?.speed ?? 1;
  return curve.points.every((p) => Math.abs(p.speed - s0) < 1e-9);
}

/** Output duration of `sourceDurationMs` of source through the curve, without memoisation. */
export function outputDurationUncached(curve: SpeedCurve, sourceDurationMs: number): number {
  if (isConstantSpeed(curve)) return sourceDurationMs;
  return buildSpeedLut(curve, sourceDurationMs, 256).outputDurationMs;
}

/**
 * Output ms per source ms of a curve. The curve is normalised over the source range, so the
 * output duration is linear in the source duration: memoise the factor per (immutable) curve.
 */
const durationFactor = new WeakMap<SpeedCurve, number>();

export function speedFactor(curve: SpeedCurve): number {
  let k = durationFactor.get(curve);
  if (k === undefined) {
    k = isUniformSpeed(curve) ? 1 / clampSpeed(curve.points[0]?.speed ?? 1) : buildSpeedLut(curve, 1, 256).outputDurationMs;
    if (curve.points.length === 0) k = 1;
    durationFactor.set(curve, k);
  }
  return k;
}

export function outputDuration(curve: SpeedCurve, sourceDurationMs: number): number {
  return sourceDurationMs * speedFactor(curve);
}

/** Output (timeline) offset reached after `sourceOffsetMs` of source (playback order). */
export function sourceToOutputMs(curve: SpeedCurve, sourceDurationMs: number, sourceOffsetMs: number): number {
  if (sourceDurationMs <= 0) return 0;
  const u = Math.min(1, Math.max(0, sourceOffsetMs / sourceDurationMs));
  if (isUniformSpeed(curve)) return sourceOffsetMs * speedFactor(curve);
  const n = Math.max(2, Math.ceil(u * 256));
  let acc = 0;
  let prev = 1 / speedAt(curve.points, 0);
  for (let i = 1; i <= n; i++) {
    const t = (i / n) * u;
    const inv = 1 / speedAt(curve.points, t);
    acc += ((prev + inv) / 2) * (u / n);
    prev = inv;
  }
  return acc * sourceDurationMs;
}

/** Source offset (playback order) reached after `outputOffsetMs` of timeline. */
export function outputToSourceOffset(curve: SpeedCurve, sourceDurationMs: number, outputOffsetMs: number): number {
  if (isUniformSpeed(curve)) return Math.min(sourceDurationMs, Math.max(0, outputOffsetMs / speedFactor(curve)));
  return outputToSourceMs(speedLutFor(curve, sourceDurationMs), outputOffsetMs);
}

const lutCache = new WeakMap<SpeedCurve, Map<number, SpeedLut>>();

/** 1024-sample output->source table, cached per curve object and source duration. */
export function speedLutFor(curve: SpeedCurve, sourceDurationMs: number): SpeedLut {
  let byDur = lutCache.get(curve);
  if (!byDur) {
    byDur = new Map();
    lutCache.set(curve, byDur);
  }
  const key = Math.round(sourceDurationMs * 1000) / 1000;
  let lut = byDur.get(key);
  if (!lut) {
    lut = buildSpeedLut(curve, sourceDurationMs, 1024);
    if (byDur.size > 8) byDur.clear();
    byDur.set(key, lut);
  }
  return lut;
}

export function outputToSourceMs(lut: SpeedLut, outputMs: number): number {
  if (lut.outputDurationMs <= 0) return 0;
  const u = Math.min(1, Math.max(0, outputMs / lut.outputDurationMs));
  const x = u * (lut.samples - 1);
  const i = Math.floor(x);
  if (i >= lut.samples - 1) return lut.outputToSource[lut.samples - 1];
  const k = x - i;
  return lut.outputToSource[i] + (lut.outputToSource[i + 1] - lut.outputToSource[i]) * k;
}

/** Harmonic-mean speed, which matches the duration behaviour of the curve. */
export function averageSpeed(curve: SpeedCurve): number {
  const n = 64;
  let s = 0;
  for (let i = 0; i <= n; i++) s += 1 / speedAt(curve.points, i / n);
  return (n + 1) / s;
}

/** Does this curve dip below 1x anywhere (optical-flow slow-mo candidate)? */
export function hasSlowMotion(curve: SpeedCurve): boolean {
  for (let i = 0; i <= 32; i++) if (speedAt(curve.points, i / 32) < 0.999) return true;
  return false;
}

/**
 * The part [u0, u1] (normalised playback position) of a curve, re-normalised to its own 0..1:
 * used when a clip is split so each half keeps its share of the ramp. Keeps the original knots
 * inside the range, adds the boundary speeds and a few samples on long spans.
 */
export function sliceCurve(curve: SpeedCurve, u0: number, u1: number): SpeedCurve {
  if (isUniformSpeed(curve)) return curve;
  const a = Math.max(0, Math.min(1, u0));
  const b = Math.max(a + 1e-6, Math.min(1, u1));
  const span = b - a;
  const knots = normalisePoints(curve.points)
    .map((p) => p.t)
    .filter((t) => t > a + 1e-6 && t < b - 1e-6);
  const anchors = [a, ...knots, b];
  const ts: number[] = [];
  const maxGap = span / 4;
  for (let i = 0; i < anchors.length - 1; i++) {
    const s = anchors[i];
    const e = anchors[i + 1];
    ts.push(s);
    const extra = Math.ceil((e - s) / maxGap) - 1;
    for (let k = 1; k <= extra; k++) ts.push(s + ((e - s) * k) / (extra + 1));
  }
  ts.push(b);
  return {
    preset: 'custom',
    points: ts.map((t) => ({ t: Math.min(1, Math.max(0, (t - a) / span)), speed: speedAt(curve.points, t) })),
    opticalFlow: curve.opticalFlow,
  };
}
