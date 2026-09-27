/** Audio envelope, transition timing and effect maths shared with the exporter (FEATURES_V2 §4, §6, §7). */
import { describe, expect, it } from 'vitest';
import type { ClipAudio } from '@/types/project';
import { audioFadeGain, clampedAudioFades, clipGainAt, dbToLin, keepPitchOf, videoFadeFactor } from './audio';
import { clampTransitionMs, transitionProgress, TRANSITION_TYPES } from './transitions';
import { EFFECT_INFO, effectFrame, effectStrength, envelope, mulberry32, pcg, proceduralShutter, rnd, SHUTTER_GAIN, valueNoise } from './effects';
import { resolveClipExtended } from './playback';
import { defaultAudio, defaultColorGrade, defaultSpeed, defaultTransform } from './defaults';
import type { Clip } from '@/types/project';

const audio = (patch: Partial<ClipAudio> = {}): ClipAudio => ({ ...defaultAudio(), ...patch });

describe('audio gain', () => {
  it('final gain = dbToLin(gainDb + volume(t)) x fadeIn(t) x fadeOut(t), 0 when muted', () => {
    const a = audio({
      gainDb: -6,
      fadeInMs: 1000,
      fadeOutMs: 2000,
      volume: { static: 0, keyframes: [ { timeMs: 0, value: 0, easing: 'linear' }, { timeMs: 4000, value: -12, easing: 'linear' } ] },
    });
    const dur = 10000;
    // t = 500: fade-in half way (sin(pi/4)), volume -1.5 dB
    expect(clipGainAt(a, 500, dur)).toBeCloseTo(dbToLin(-6 - 1.5) * Math.sin(Math.PI / 4), 9);
    // t = 5000: past the keys (-12 dB), no fade
    expect(clipGainAt(a, 5000, dur)).toBeCloseTo(dbToLin(-18), 9);
    // t = 9000: fade-out half way
    expect(clipGainAt(a, 9000, dur)).toBeCloseTo(dbToLin(-18) * Math.sin(Math.PI / 4), 9);
    expect(clipGainAt(a, 0, dur)).toBe(0);
    expect(clipGainAt(a, dur, dur)).toBeCloseTo(0, 12);
    expect(clipGainAt({ ...a, muted: true }, 5000, dur)).toBe(0);
    expect(clipGainAt(audio(), 1234, dur)).toBe(1);
  });

  it('equal-power fades: the fade-in and fade-out gains sum to unit power at the same position', () => {
    const a = audio({ fadeInMs: 1000 });
    const b = audio({ fadeOutMs: 1000 });
    for (const t of [0, 100, 250, 500, 900]) {
      const gi = audioFadeGain(a, t, 1000 * 4);
      const go = audioFadeGain(b, 4000 - 1000 + t, 4000);
      expect(gi * gi + go * go).toBeCloseTo(1, 9);
    }
  });

  it('fades are clamped to half the clip each', () => {
    expect(clampedAudioFades(audio({ fadeInMs: 5000, fadeOutMs: 600 }), 2000)).toEqual({ fadeIn: 1000, fadeOut: 600 });
    expect(clampedAudioFades(audio({ fadeInMs: -5 }), 2000)).toEqual({ fadeIn: 0, fadeOut: 0 });
    // clamped fades still reach full level in the middle
    expect(audioFadeGain(audio({ fadeInMs: 5000, fadeOutMs: 5000 }), 1000, 2000)).toBeCloseTo(1, 12);
    expect(videoFadeFactor(5000, 5000, 1000, 2000)).toBeCloseTo(1, 12);
    expect(videoFadeFactor(500, 0, 250, 2000)).toBeCloseTo(0.5, 12);
    expect(videoFadeFactor(0, 500, 1750, 2000)).toBeCloseTo(0.5, 12);
  });

  it('keep pitch defaults to true', () => {
    expect(keepPitchOf({ gainDb: 0, normalize: false, muted: false })).toBe(true);
    expect(keepPitchOf(audio({ keepPitch: false }))).toBe(false);
  });
});

describe('transitions', () => {
  it('has 16 types and eases progress with the easeInOut cubic-bezier', () => {
    expect(TRANSITION_TYPES).toHaveLength(16);
    expect(transitionProgress(0)).toBe(0);
    expect(transitionProgress(1)).toBe(1);
    expect(transitionProgress(0.5)).toBeCloseTo(0.5, 6);
    expect(transitionProgress(0.25)).toBeLessThan(0.25);
    expect(transitionProgress(0.75)).toBeGreaterThan(0.75);
  });

  it('clamps durations', () => {
    expect(clampTransitionMs(20, 5000, 5000)).toBe(100);
    expect(clampTransitionMs(5000, 9000, 9000)).toBe(3000);
    expect(clampTransitionMs(800, 600, 5000)).toBe(600);
    expect(clampTransitionMs(800, 5000, 60)).toBe(60);
  });

  it('extends clips past their edges with source handles at the boundary speed, or holds', () => {
    const base: Clip = {
      id: 'c',
      assetId: 'a',
      trackId: 't',
      startMs: 10000,
      inMs: 2000,
      outMs: 6000,
      speed: { ...defaultSpeed(), points: [ { t: 0, speed: 2 }, { t: 1, speed: 2 } ] },
      transform: defaultTransform(),
      color: defaultColorGrade(),
      audio: defaultAudio(),
      mask: null,
      blendMode: 'normal',
      reframe: null,
      freezeFrame: null,
      reversed: false,
    };
    // 2x: 4000 source ms play in 2000 ms; 300 ms past the end = 600 source ms past outMs
    expect(resolveClipExtended(base, 2300, 20000).sourceMs).toBeCloseTo(6600, 6);
    expect(resolveClipExtended(base, -300, 20000).sourceMs).toBeCloseTo(1400, 6);
    // no handles: hold the first / last frame
    expect(resolveClipExtended(base, 2300, 6100).sourceMs).toBeCloseTo(6100, 6);
    expect(resolveClipExtended({ ...base, inMs: 100 }, -300, 20000).sourceMs).toBe(0);
    // reversed: past the end goes further back in the source
    expect(resolveClipExtended({ ...base, reversed: true }, 2300, 20000).sourceMs).toBeCloseTo(1400, 6);
    // inside the clip it is the normal resolution
    expect(resolveClipExtended(base, 1000, 20000).sourceMs).toBeCloseTo(4000, 6);
  });
});

describe('effects', () => {
  it('lists 16 types with defaults', () => {
    expect(EFFECT_INFO).toHaveLength(16);
    expect(EFFECT_INFO.find((e) => e.id === 'cameraSnap')!.defaultMs).toBe(1500);
  });

  it('120 ms envelope x intensity; own-timing types ignore the envelope', () => {
    expect(envelope(0, 1000)).toBe(0);
    expect(envelope(60, 1000)).toBeCloseTo(0.5, 12);
    expect(envelope(500, 1000)).toBe(1);
    expect(envelope(940, 1000)).toBeCloseTo(0.5, 12);
    expect(envelope(100, 200)).toBeCloseTo(100 / 120, 12);
    expect(effectStrength({ type: 'sepia', intensity: 0.5 }, 60, 1000)).toBeCloseTo(0.25, 12);
    expect(effectStrength({ type: 'fadeToBlack', intensity: 0.5 }, 0, 1000)).toBe(0.5);
  });

  it('per-frame parameters follow the spec formulas', () => {
    const fx = (type: Parameters<typeof effectFrame>[0]['type'], t: number, d = 1000, params?: Record<string, number>) => effectFrame({ type, intensity: 1, params }, t, d);
    expect(fx('fadeFromBlack', 250).a).toBeCloseTo(0.75, 12);
    expect(fx('fadeToWhite', 250).a).toBeCloseTo(0.25, 12);
    expect(fx('flashWhite', 500).a).toBeCloseTo(1, 12);
    expect(fx('flashWhite', 250).a).toBeCloseTo(0.5, 12);
    expect(fx('blurIn', 0).a).toBe(20);
    expect(fx('blurOut', 500).a).toBe(10);
    expect(fx('zoomPunch', 0).a).toBeCloseTo(1, 12);
    expect(fx('zoomPunch', 350).a).toBeCloseTo(1.15, 12);
    expect(fx('zoomPunch', 999.999).a).toBeCloseTo(1, 6);
    expect(fx('vignettePulse', 0, 3000).a).toBeCloseTo(0, 12); // envelope 0 at t = 0
    expect(fx('vignettePulse', 500, 3000).a).toBeCloseTo(0.6, 12);
    expect(fx('vignettePulse', 1000, 3000).a).toBeCloseTo(0.2, 12);
    const snap = fx('cameraSnap', 125, 1500);
    expect(snap.a).toBeCloseTo(0.5, 12);
    expect(fx('cameraSnap', 400, 1500).k).toBe(1);
    expect(snap.scale).toBe(0.92);
    expect(snap.border).toBe(0.03);
    expect(fx('letterbox', 500, 3000).k).toBe(2.39);
    const sh = fx('shake', 500, 1000);
    expect(Math.abs(sh.offset[0])).toBeLessThanOrEqual(0.01);
    expect(Math.abs(sh.rot)).toBeLessThanOrEqual(1);
    const rgb = fx('rgbSplit', 500, 1000);
    expect(rgb.offset[0]).toBeGreaterThanOrEqual(0.003);
    expect(rgb.offset[0]).toBeLessThanOrEqual(0.009);
  });

  it('noise is deterministic (PCG hash, value noise) with fixed reference values', () => {
    expect(pcg(0)).toBe(pcg(0));
    expect(pcg(1)).not.toBe(pcg(2));
    // reference values the Rust port must reproduce
    expect(pcg(0)).toBe(129708002);
    expect(pcg(1)).toBe(2831084092);
    expect(rnd(3, 1)).toBeCloseTo(rnd(3, 1), 15);
    for (let x = 0; x < 20; x += 0.37) {
      const v = valueNoise(x, 2);
      expect(v).toBeGreaterThanOrEqual(-1);
      expect(v).toBeLessThanOrEqual(1);
    }
    expect(valueNoise(4, 7)).toBeCloseTo(2 * rnd(4, 7) - 1, 12);
  });

  it('the procedural shutter is deterministic, 120 ms long and peaks at 0.9', () => {
    const a = proceduralShutter(48000);
    const b = proceduralShutter(48000);
    expect(a.length).toBe(5760);
    expect(Array.from(a.slice(0, 64))).toEqual(Array.from(b.slice(0, 64)));
    let peak = 0;
    for (const v of a) peak = Math.max(peak, Math.abs(v));
    expect(peak).toBeCloseTo(0.9, 5);
    expect(proceduralShutter(44100).length).toBe(5292);
    expect(SHUTTER_GAIN).toBeCloseTo(0.501187, 5);
    const r = mulberry32(1);
    expect(r()).toBeCloseTo(0.6270739405881613, 12);
  });
});
