import { describe, expect, it } from 'vitest';
import { cubicBezier, ease, evaluate, removeKeyframe, setKeyframe } from './keyframes';
import { keyframed } from './defaults';

describe('cubicBezier', () => {
  it('is identity for linear handles', () => {
    for (let i = 0; i <= 10; i++) {
      const x = i / 10;
      expect(cubicBezier(0, 0, 1, 1, x)).toBeCloseTo(x, 6);
    }
  });
  it('matches CSS ease-in-out at midpoint', () => {
    expect(cubicBezier(0.42, 0, 0.58, 1, 0.5)).toBeCloseTo(0.5, 3);
    expect(cubicBezier(0.42, 0, 0.58, 1, 0.25)).toBeLessThan(0.25);
    expect(cubicBezier(0.42, 0, 0.58, 1, 0.75)).toBeGreaterThan(0.75);
  });
  it('clamps endpoints', () => {
    expect(ease(-1, 'easeIn')).toBe(0);
    expect(ease(2, 'bounce')).toBe(1);
    expect(ease(1, 'elastic')).toBe(1);
  });
});

describe('evaluate', () => {
  it('returns static when no keyframes', () => {
    expect(evaluate(keyframed(3), 100)).toBe(3);
  });
  it('interpolates numbers linearly', () => {
    let kf = keyframed(0);
    kf = setKeyframe(kf, 0, 0, 'linear');
    kf = setKeyframe(kf, 1000, 100, 'linear');
    expect(evaluate(kf, 500)).toBeCloseTo(50);
    expect(evaluate(kf, -10)).toBe(0);
    expect(evaluate(kf, 5000)).toBe(100);
  });
  it('interpolates vectors with the outgoing keyframe easing', () => {
    let kf = keyframed<[number, number]>([0, 0]);
    kf = setKeyframe(kf, 0, [0, 0], 'easeIn');
    kf = setKeyframe(kf, 1000, [10, -10], 'linear');
    const v = evaluate(kf, 500);
    expect(v[0]).toBeLessThan(5);
    expect(v[1]).toBeGreaterThan(-5);
    // the last keyframe's easing does not affect the segment
    let kf2 = keyframed(0);
    kf2 = setKeyframe(kf2, 0, 0, 'linear');
    kf2 = setKeyframe(kf2, 1000, 100, 'easeIn');
    expect(evaluate(kf2, 500)).toBeCloseTo(50);
  });
  it('replaces keyframes at the same time and removes them', () => {
    let kf = keyframed(0);
    kf = setKeyframe(kf, 100, 1);
    kf = setKeyframe(kf, 100, 2);
    expect(kf.keyframes).toHaveLength(1);
    expect(kf.keyframes[0].value).toBe(2);
    kf = removeKeyframe(kf, 100);
    expect(kf.keyframes).toHaveLength(0);
  });
  it('evaluates 10k times quickly', () => {
    let kf = keyframed(0);
    for (let i = 0; i < 50; i++) kf = setKeyframe(kf, i * 100, i, 'easeInOut');
    const t0 = performance.now();
    let acc = 0;
    for (let i = 0; i < 10000; i++) acc += evaluate(kf, (i * 7) % 5000);
    const dt = performance.now() - t0;
    expect(acc).toBeGreaterThan(0);
    expect(dt).toBeLessThan(200);
  });
});
