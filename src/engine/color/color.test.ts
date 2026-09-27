import { describe, expect, it } from 'vitest';
import { bakeCurves, evalCurve } from './curves';
import { identityLut, parseCube, sampleLut } from './lut';

function cubeText(size: number, fn: (r: number, g: number, b: number) => [number, number, number]): string {
  const lines = ['TITLE "test"', `LUT_3D_SIZE ${size}`];
  for (let b = 0; b < size; b++)
    for (let g = 0; g < size; g++)
      for (let r = 0; r < size; r++) {
        const [x, y, z] = fn(r / (size - 1), g / (size - 1), b / (size - 1));
        lines.push(`${x.toFixed(6)} ${y.toFixed(6)} ${z.toFixed(6)}`);
      }
  return lines.join('\n');
}

describe('cube LUT', () => {
  it('parses an identity cube and samples it', () => {
    const lut = parseCube(cubeText(17, (r, g, b) => [r, g, b]));
    expect(lut.size).toBe(17);
    expect(lut.title).toBe('test');
    const s = sampleLut(lut, 0.3, 0.6, 0.9);
    expect(s[0]).toBeCloseTo(0.3, 4);
    expect(s[1]).toBeCloseTo(0.6, 4);
    expect(s[2]).toBeCloseTo(0.9, 4);
  });
  it('rejects wrong sizes', () => {
    expect(() => parseCube('LUT_3D_SIZE 3\n0 0 0')).toThrow();
    expect(() => parseCube('LUT_1D_SIZE 4')).toThrow();
  });
  it('identity LUT is identity', () => {
    const s = sampleLut(identityLut(2), 0.25, 0.5, 0.75);
    expect(s).toEqual([0.25, 0.5, 0.75]);
  });
});

describe('curves', () => {
  it('identity curve is identity', () => {
    expect(evalCurve([[0, 0], [1, 1]], 0.4)).toBeCloseTo(0.4);
  });
  it('S-curve boosts contrast', () => {
    const s: Array<[number, number]> = [[0, 0], [0.25, 0.15], [0.75, 0.85], [1, 1]];
    expect(evalCurve(s, 0.25)).toBeLessThan(0.25);
    expect(evalCurve(s, 0.75)).toBeGreaterThan(0.75);
    expect(evalCurve(s, 0.5)).toBeCloseTo(0.5, 2);
  });
  it('bakes 256x4 RGBA', () => {
    const px = bakeCurves({ master: [[0, 0], [1, 1]], r: [[0, 0], [1, 1]], g: [[0, 0], [1, 1]], b: [[0, 0], [1, 1]] });
    expect(px.length).toBe(256 * 4 * 4);
    expect(px[255 * 4]).toBe(255);
    expect(px[0]).toBe(0);
  });
});
