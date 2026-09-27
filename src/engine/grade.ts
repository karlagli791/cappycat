/**
 * Effective grade = a clip's own grade + the project's universal adjustment (when enabled).
 * Mirrors `effective_grade` in src-tauri/src/model.rs so the preview matches the export.
 */
import type { AdjustValues, ColorGrade, HslChannel, UniversalAdjust } from '@/types/project';
import { HSL_CHANNELS } from './defaults';

type ScalarKey = Exclude<keyof AdjustValues, 'hsl'>;

// CapCut scale: adjust sliders -50..50, sharpness 0..50
const RANGES: Record<ScalarKey, [number, number]> = {
  exposure: [-50, 50],
  brilliance: [-50, 50],
  contrast: [-50, 50],
  brightness: [-50, 50],
  highlights: [-50, 50],
  shadows: [-50, 50],
  saturation: [-50, 50],
  vibrance: [-50, 50],
  sharpness: [0, 50],
  temperature: [-50, 50],
  tint: [-50, 50],
  vignette: [0, 1],
  grain: [0, 1],
};

export const ADJUST_KEYS = Object.keys(RANGES) as ScalarKey[];

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function effectiveGrade(clip: ColorGrade, universal: UniversalAdjust | null | undefined): ColorGrade {
  // older projects / pipeline output may not carry every field yet
  const base: ColorGrade = { ...clip, brilliance: clip.brilliance ?? 0 };
  if (!universal || !universal.enabled) return base;
  const v = universal.values ?? {};
  const out: ColorGrade = { ...base };
  for (const k of ADJUST_KEYS) {
    const d = v[k];
    if (d == null) continue;
    const [lo, hi] = RANGES[k];
    out[k] = clamp((base[k] ?? 0) + d, lo, hi);
  }
  if (v.hsl) {
    const hsl = { ...base.hsl };
    for (const ch of HSL_CHANNELS as HslChannel[]) {
      const a = base.hsl[ch] ?? { h: 0, s: 0, l: 0 };
      const b = v.hsl[ch];
      if (!b) continue;
      hsl[ch] = { h: clamp(a.h + b.h, -100, 100), s: clamp(a.s + b.s, -100, 100), l: clamp(a.l + b.l, -100, 100) };
    }
    out.hsl = hsl;
  }
  return out;
}

/** The user's house look (2026-09-25); also the default of presets/universal-adjust.json. */
export function defaultUniversalValues(): AdjustValues {
  return {
    sharpness: 40,
    brilliance: 6,
    highlights: 5,
    contrast: 5,
    exposure: 5,
    temperature: -10,
    tint: 10,
    saturation: 10,
    hsl: {
      red: { h: -24, s: 17, l: 0 },
      orange: { h: 16, s: 20, l: 0 },
      yellow: { h: -8, s: 17, l: 0 },
      green: { h: -33, s: 50, l: 0 },
      cyan: { h: -12, s: 26, l: 0 },
      blue: { h: 0, s: 17, l: -6 },
      purple: { h: -23, s: 33, l: -10 },
      magenta: { h: -22, s: 27, l: 0 },
    },
  };
}
