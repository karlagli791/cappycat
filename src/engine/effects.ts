/**
 * Effect catalogue and per-frame effect maths (FEATURES_V2 §7). Everything that does not vary per
 * pixel is computed here on the CPU (shared with the tests); the per-pixel part is the GLSL in
 * ./color/fxShaders.ts. The exporter mirrors both in src-tauri/src/render/fx.rs. The constants
 * the spec leaves open are listed in the header of fxShaders.ts.
 */
import type { Clip, ClipEffect, EffectType, Project } from '@/types/project';

export interface EffectParamInfo {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
  fmt?: (v: number) => string;
}

export interface EffectInfo {
  id: EffectType;
  label: string;
  hint: string;
  /** default clip length when added from the Effects tab */
  defaultMs: number;
  /** false: the type has its own timing and ignores the 120 ms ramp-in / ramp-out envelope */
  envelope: boolean;
  params: EffectParamInfo[];
}

const pct = (v: number) => `${(v * 100).toFixed(1)} %`;

export const EFFECT_INFO: EffectInfo[] = [
  {
    id: 'cameraSnap',
    label: 'Camera snap',
    hint: 'Take a photo: flash, shutter sound, the picture freezes as a polaroid',
    defaultMs: 1500,
    envelope: false,
    params: [
      { key: 'border', label: 'Border', min: 0, max: 0.1, step: 0.005, default: 0.03, fmt: pct },
      { key: 'scale', label: 'Photo size', min: 0.8, max: 1, step: 0.01, default: 0.92, fmt: pct },
    ],
  },
  { id: 'fadeFromBlack', label: 'Fade from black', hint: 'Black to picture over the effect', defaultMs: 1000, envelope: false, params: [] },
  { id: 'fadeToBlack', label: 'Fade to black', hint: 'Picture to black over the effect', defaultMs: 1000, envelope: false, params: [] },
  { id: 'fadeFromWhite', label: 'Fade from white', hint: 'White to picture over the effect', defaultMs: 1000, envelope: false, params: [] },
  { id: 'fadeToWhite', label: 'Fade to white', hint: 'Picture to white over the effect', defaultMs: 1000, envelope: false, params: [] },
  { id: 'blackAndWhite', label: 'Black & white', hint: 'Desaturate with a slight contrast boost', defaultMs: 3000, envelope: true, params: [] },
  { id: 'sepia', label: 'Sepia', hint: 'Classic sepia toning', defaultMs: 3000, envelope: true, params: [] },
  {
    id: 'letterbox',
    label: 'Letterbox',
    hint: 'Cinematic bars that slide in and out',
    defaultMs: 3000,
    envelope: true,
    params: [{ key: 'ratio', label: 'Aspect ratio', min: 1.5, max: 3, step: 0.01, default: 2.39, fmt: (v) => `${v.toFixed(2)}:1` }],
  },
  {
    id: 'shake',
    label: 'Shake',
    hint: 'Camera shake',
    defaultMs: 1000,
    envelope: true,
    params: [
      { key: 'amplitude', label: 'Amplitude', min: 0, max: 0.05, step: 0.001, default: 0.01, fmt: pct },
      { key: 'frequency', label: 'Frequency', min: 1, max: 30, step: 0.5, default: 12, fmt: (v) => `${v.toFixed(1)} Hz` },
    ],
  },
  { id: 'zoomPunch', label: 'Zoom punch', hint: 'Quick push-in and back', defaultMs: 500, envelope: false, params: [] },
  { id: 'blurIn', label: 'Blur in', hint: 'Blurred to sharp', defaultMs: 1000, envelope: false, params: [] },
  { id: 'blurOut', label: 'Blur out', hint: 'Sharp to blurred', defaultMs: 1000, envelope: false, params: [] },
  {
    id: 'rgbSplit',
    label: 'RGB split',
    hint: 'Glitchy colour-channel offset',
    defaultMs: 1000,
    envelope: true,
    params: [{ key: 'amount', label: 'Amount', min: 0, max: 0.03, step: 0.0005, default: 0.006, fmt: pct }],
  },
  { id: 'vhs', label: 'VHS', hint: 'Scanlines, chroma bleed, noise and tracking wobble', defaultMs: 3000, envelope: true, params: [] },
  { id: 'vignettePulse', label: 'Vignette pulse', hint: 'Vignette breathing at 1 Hz', defaultMs: 3000, envelope: true, params: [] },
  { id: 'flashWhite', label: 'Flash', hint: 'White flash peaking in the middle', defaultMs: 400, envelope: false, params: [] },
];

export const EFFECT_IDS = EFFECT_INFO.map((e) => e.id);

export function effectInfo(type: EffectType): EffectInfo {
  return EFFECT_INFO.find((e) => e.id === type) ?? EFFECT_INFO[0];
}

export function isEffectType(x: unknown): x is EffectType {
  return typeof x === 'string' && (EFFECT_IDS as string[]).includes(x);
}

/** Integer code of each type in the shaders (u_type). Order of EFFECT_INFO. */
export function effectCode(type: EffectType): number {
  return Math.max(0, EFFECT_IDS.indexOf(type));
}

export function effectParam(effect: ClipEffect, key: string): number {
  const v = effect.params?.[key];
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  return effectInfo(effect.type).params.find((p) => p.key === key)?.default ?? 0;
}

export function defaultEffect(type: EffectType): ClipEffect {
  return { type, intensity: 1 };
}

/** Effect clip length (timeline ms): outMs - inMs, speed ignored. */
export function effectDurationMs(clip: Clip): number {
  return Math.max(0, clip.outMs - clip.inMs);
}

/* ------------------------------------------------------------ envelope */

export const ENVELOPE_RAMP_MS = 120;

/** 120 ms linear ramp in and out (the two meet on clips shorter than 240 ms). */
export function envelope(tMs: number, durMs: number): number {
  if (durMs <= 0) return 0;
  const a = tMs / ENVELOPE_RAMP_MS;
  const b = (durMs - tMs) / ENVELOPE_RAMP_MS;
  return Math.max(0, Math.min(1, a, b));
}

/** Strength of an effect at effect-local t: intensity x envelope (types with their own timing: intensity). */
export function effectStrength(effect: ClipEffect, tMs: number, durMs: number): number {
  const k = Math.min(1, Math.max(0, Number.isFinite(effect.intensity) ? effect.intensity : 1));
  return effectInfo(effect.type).envelope ? k * envelope(tMs, durMs) : k;
}

/* ------------------------------------------------------ deterministic noise */

/** PCG hash (u32 -> u32). Identical in GLSL (fxShaders.ts) and Rust (fx.rs). */
export function pcg(v: number): number {
  const state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
  const shift = ((state >>> 28) + 4) & 31;
  const word = Math.imul(((state >>> shift) ^ state) >>> 0, 277803737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}

/** Uniform [0, 1] from an integer lattice point and a seed: pcg(i ^ pcg(seed)) / (2^32 - 1). */
export function rnd(i: number, seed: number): number {
  return pcg(((i >>> 0) ^ pcg(seed >>> 0)) >>> 0) / 4294967295;
}

/** 1-D value noise in [-1, 1] (x >= 0): smoothstep between hashed lattice values. */
export function valueNoise(x: number, seed: number): number {
  const cx = Math.max(0, x);
  const xi = Math.floor(cx);
  const f = cx - xi;
  const u = f * f * (3 - 2 * f);
  const a = rnd(xi, seed);
  const b = rnd(xi + 1, seed);
  return 2 * (a + (b - a) * u) - 1;
}

/* ------------------------------------------------------------ easings */

export function easeOutBack(x: number): number {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
}

export function easeInOutCubic(x: number): number {
  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
}

export function easeOutCubic(x: number): number {
  return 1 - Math.pow(1 - x, 3);
}

/* ------------------------------------------------------ per-frame params */

/** Camera snap: flash 1 -> 0 over 250 ms; the polaroid settles over 350 ms. */
export const SNAP_FLASH_MS = 250;
export const SNAP_SETTLE_MS = 350;
/** Zoom punch: scale 1 -> 1.15 (ease-out-back) peaking at 35 % of D, back to 1 (ease-in-out cubic). */
export const ZOOM_PUNCH_PEAK = 0.35;
export const ZOOM_PUNCH_AMOUNT = 0.15;
/** Blur in / out radius (output px, house blur kernel). */
export const BLUR_EFFECT_PX = 20;

/**
 * Everything an effect pass needs for one frame (the shader uniforms). Pixel values are OUTPUT
 * pixels (project width / height); fractions are of the output width unless noted.
 */
export interface EffectFrame {
  type: EffectType;
  /** intensity x envelope (see effectStrength) */
  s: number;
  /** effect-local time, seconds */
  t: number;
  /** effect duration, seconds */
  d: number;
  /** fade amount / flash alpha / blur radius px / vignette strength / zoom scale (see effectFrame) */
  a: number;
  /** shake: translation (fractions of the width); rgbSplit: channel offset in [0] */
  offset: [number, number];
  /** shake: rotation, degrees */
  rot: number;
  /** cameraSnap: polaroid progress k (0..1); letterbox: target aspect ratio */
  k: number;
  /** cameraSnap: photo scale */
  scale: number;
  /** cameraSnap: border thickness, fraction of the output HEIGHT */
  border: number;
}

export function effectFrame(effect: ClipEffect, tMs: number, durMs: number): EffectFrame {
  const s = effectStrength(effect, tMs, durMs);
  const u = durMs > 0 ? Math.min(1, Math.max(0, tMs / durMs)) : 1;
  const t = tMs / 1000;
  const out: EffectFrame = { type: effect.type, s, t, d: durMs / 1000, a: 0, offset: [0, 0], rot: 0, k: 0, scale: 1, border: 0 };
  switch (effect.type) {
    case 'fadeFromBlack':
    case 'fadeFromWhite':
      out.a = s * (1 - u); // amount of black / white
      break;
    case 'fadeToBlack':
    case 'fadeToWhite':
      out.a = s * u;
      break;
    case 'flashWhite':
      out.a = s * (1 - Math.abs(2 * u - 1));
      break;
    case 'blurIn':
      out.a = BLUR_EFFECT_PX * s * (1 - u);
      break;
    case 'blurOut':
      out.a = BLUR_EFFECT_PX * s * u;
      break;
    case 'zoomPunch': {
      const k = u < ZOOM_PUNCH_PEAK ? easeOutBack(u / ZOOM_PUNCH_PEAK) : 1 - easeInOutCubic((u - ZOOM_PUNCH_PEAK) / (1 - ZOOM_PUNCH_PEAK));
      out.a = 1 + ZOOM_PUNCH_AMOUNT * s * k;
      break;
    }
    case 'shake': {
      const amp = effectParam(effect, 'amplitude');
      const freq = effectParam(effect, 'frequency');
      const x = t * freq;
      out.offset = [amp * s * valueNoise(x, 1), amp * s * valueNoise(x, 2)];
      out.rot = 100 * amp * s * valueNoise(x, 3); // 1 degree at the default amplitude
      out.a = 1 + 2 * amp * s; // overscan so the frame edges stay covered
      break;
    }
    case 'rgbSplit': {
      const amount = effectParam(effect, 'amount');
      out.offset = [amount * s * (1 + 0.5 * valueNoise(t * 8, 4)), 0];
      break;
    }
    case 'vignettePulse':
      out.a = s * (0.4 - 0.2 * Math.cos(2 * Math.PI * t)); // 0.2 <-> 0.6 at 1 Hz
      break;
    case 'letterbox':
      out.k = effectParam(effect, 'ratio');
      break;
    case 'cameraSnap':
      out.a = s * Math.max(0, 1 - tMs / SNAP_FLASH_MS); // flash alpha
      out.k = s * easeOutCubic(Math.min(1, Math.max(0, tMs / SNAP_SETTLE_MS)));
      out.scale = effectParam(effect, 'scale');
      out.border = effectParam(effect, 'border');
      break;
    default:
      break;
  }
  return out;
}

/** An effect clip under the playhead. */
export interface ActiveEffect {
  clip: Clip;
  effect: ClipEffect;
  trackIndex: number;
  /** effect-local ms */
  tMs: number;
  durMs: number;
}

/** Effects active at a timeline time, in track order (earlier fx tracks first); hidden fx tracks skipped. */
export function activeEffects(p: Project, timelineMs: number): ActiveEffect[] {
  const out: ActiveEffect[] = [];
  p.tracks.forEach((t, ti) => {
    if (t.kind !== 'fx' || t.muted) return;
    for (const c of t.clips) {
      if (!c.effect) continue;
      const d = effectDurationMs(c);
      if (timelineMs >= c.startMs && timelineMs < c.startMs + d) out.push({ clip: c, effect: c.effect, trackIndex: ti, tMs: timelineMs - c.startMs, durMs: d });
    }
  });
  return out;
}

/* ------------------------------------------------------------ shutter */

export const SHUTTER_SEED = 0x43415050; // "CAPP"
export const SHUTTER_SECONDS = 0.12;
/** The shutter is mixed at -6 dB. */
export const SHUTTER_GAIN = Math.pow(10, -6 / 20);

/** mulberry32 PRNG (u32 state), [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * `procedural_shutter(sample_rate)`: a 120 ms camera shutter, deterministic (mirrors Rust):
 *  - white noise (mulberry32, seed 0x43415050, x = 2r - 1) through a one-pole high-pass at
 *    2 kHz (y = a (y' + x - x'), a = RC / (RC + 1/sr), RC = 1 / (2 pi 2000)),
 *    amplitude 0.5 x exp(-t / 18 ms);
 *  - click 1 at 0 ms: 0.9 exp(-t / 1.5 ms) sin(2 pi 3500 t);
 *  - click 2 at 60 ms: 0.7 exp(-(t - 0.06) / 2 ms) sin(2 pi 2400 (t - 0.06));
 *  - normalised to a 0.9 peak. Mono, n = round(0.12 sr) samples.
 */
export function proceduralShutter(sampleRate: number): Float32Array {
  const n = Math.max(1, Math.round(SHUTTER_SECONDS * sampleRate));
  const out = new Float32Array(n);
  const rng = mulberry32(SHUTTER_SEED);
  const rc = 1 / (2 * Math.PI * 2000);
  const alpha = rc / (rc + 1 / sampleRate);
  let px = 0;
  let py = 0;
  let peak = 0;
  const click = (t: number, t0: number, f: number, amp: number, tau: number) =>
    t >= t0 ? amp * Math.exp(-(t - t0) / tau) * Math.sin(2 * Math.PI * f * (t - t0)) : 0;
  for (let i = 0; i < n; i++) {
    const t = i / sampleRate;
    const x = rng() * 2 - 1;
    const y = alpha * (py + x - px);
    px = x;
    py = y;
    const s = 0.5 * y * Math.exp(-t / 0.018) + click(t, 0, 3500, 0.9, 0.0015) + click(t, 0.06, 2400, 0.7, 0.002);
    out[i] = s;
    peak = Math.max(peak, Math.abs(s));
  }
  if (peak > 0) for (let i = 0; i < n; i++) out[i] = (out[i] * 0.9) / peak;
  return out;
}
