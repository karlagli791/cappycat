import { describe, expect, it } from 'vitest';
import { defaultUniversalValues, effectiveGrade } from './grade';
import { defaultColorGrade } from './defaults';
import type { UniversalAdjust } from '@/types/project';

const universal = (enabled = true): UniversalAdjust => ({ enabled, name: 'Universal adjust', values: defaultUniversalValues() });

describe('effectiveGrade (mirrors src-tauri effective_grade)', () => {
  it('adds the universal values and clamps to each slider range', () => {
    // CapCut scale: adjust sliders clamp at +-50 (sharpness 0..50), HSL at +-100
    const clip = { ...defaultColorGrade(), saturation: 45, contrast: -5, sharpness: 30 };
    clip.hsl = { ...clip.hsl, green: { h: 0, s: 60, l: 0 } };
    const g = effectiveGrade(clip, universal());
    expect(g.saturation).toBe(50);
    expect(g.contrast).toBe(0);
    expect(g.sharpness).toBe(50);
    expect(g.brilliance).toBe(6);
    expect(g.temperature).toBe(-10);
    expect(g.tint).toBe(10);
    expect(g.exposure).toBe(5);
    expect(g.highlights).toBe(5);
    expect(g.hsl.green).toEqual({ h: -33, s: 100, l: 0 });
    expect(g.hsl.purple).toEqual({ h: -23, s: 33, l: -10 });
    expect(g.hsl.blue).toEqual({ h: 0, s: 17, l: -6 });
  });

  it('is a no-op when switched off or absent', () => {
    const clip = { ...defaultColorGrade(), saturation: 12 };
    expect(effectiveGrade(clip, universal(false))).toEqual(clip);
    expect(effectiveGrade(clip, undefined)).toEqual(clip);
  });

  it('tolerates grades that predate brilliance', () => {
    const legacy = { ...defaultColorGrade() } as Partial<ReturnType<typeof defaultColorGrade>>;
    delete legacy.brilliance;
    const g = effectiveGrade(legacy as ReturnType<typeof defaultColorGrade>, universal());
    expect(g.brilliance).toBe(6);
  });
});
