import { describe, expect, it } from 'vitest';
import { averageSpeed, buildSpeedLut, hasSlowMotion, outputDuration, outputToSourceMs, presetCurve, speedAt } from './speed';

describe('speed curves', () => {
  it('normal preset keeps duration', () => {
    expect(outputDuration(presetCurve('normal'), 5000)).toBeCloseTo(5000);
  });
  it('constant 2x halves duration', () => {
    const curve = { preset: 'custom' as const, points: [{ t: 0, speed: 2 }, { t: 1, speed: 2 }], opticalFlow: false };
    expect(outputDuration(curve, 4000)).toBeCloseTo(2000, 0);
  });
  it('monotone interpolation stays within point range', () => {
    const c = presetCurve('hero_time');
    for (let i = 0; i <= 100; i++) {
      const s = speedAt(c.points, i / 100);
      expect(s).toBeGreaterThanOrEqual(0.25 - 1e-9);
      expect(s).toBeLessThanOrEqual(2.5 + 1e-9);
    }
  });
  it('output->source mapping is monotonic and hits the ends', () => {
    const lut = buildSpeedLut(presetCurve('bullet'), 3000, 128);
    let prev = -1;
    for (let i = 0; i <= 50; i++) {
      const s = outputToSourceMs(lut, (i / 50) * lut.outputDurationMs);
      expect(s).toBeGreaterThanOrEqual(prev);
      prev = s;
    }
    expect(outputToSourceMs(lut, 0)).toBeCloseTo(0);
    expect(outputToSourceMs(lut, lut.outputDurationMs)).toBeCloseTo(3000);
  });
  it('detects slow motion phases', () => {
    expect(hasSlowMotion(presetCurve('hero_time'))).toBe(true);
    expect(hasSlowMotion(presetCurve('flash_in'))).toBe(false);
  });
  it('average speed is the harmonic mean', () => {
    const curve = { preset: 'custom' as const, points: [{ t: 0, speed: 1 }, { t: 0.5, speed: 1 }, { t: 0.5001, speed: 3 }, { t: 1, speed: 3 }], opticalFlow: false };
    expect(averageSpeed(curve)).toBeCloseTo(1.5, 0);
  });
});
