/**
 * Clip audio envelope (FEATURES_V2 §3-4): keep-pitch, equal-power fades and volume keyframes.
 * Pure functions shared by the preview (media element gain) and the tests; the exporter implements
 * the same formula in Rust:
 *
 *   gain(t) = muted ? 0 : dbToLin(gainDb + volume(t)) * fadeIn(t) * fadeOut(t)
 *   fadeIn(t)  = sin(pi/2 * clamp(t / fadeIn, 0, 1))            (1 when fadeIn = 0)
 *   fadeOut(t) = sin(pi/2 * clamp((dur - t) / fadeOut, 0, 1))   (1 when fadeOut = 0)
 *
 * t is clip-local TIMELINE ms clamped to [0, dur]; each fade is clamped to dur / 2.
 */
import type { ClipAudio, Keyframed } from '@/types/project';
import { evaluate } from './keyframes';

export const VOLUME_KEYFRAMED_DEFAULT: Keyframed<number> = { static: 0, keyframes: [] };

export function dbToLin(db: number): number {
  return Math.pow(10, db / 20);
}

export function linToDb(lin: number): number {
  return lin <= 1e-6 ? -120 : 20 * Math.log10(lin);
}

/** Default true (CapCut "Pitch" off). */
export function keepPitchOf(audio: ClipAudio): boolean {
  return audio.keepPitch !== false;
}

/** Fade lengths clamped to half the clip each (and to >= 0). */
export function clampedAudioFades(audio: ClipAudio, durMs: number): { fadeIn: number; fadeOut: number } {
  const half = Math.max(0, durMs / 2);
  const fi = Number.isFinite(audio.fadeInMs) ? Math.max(0, audio.fadeInMs as number) : 0;
  const fo = Number.isFinite(audio.fadeOutMs) ? Math.max(0, audio.fadeOutMs as number) : 0;
  return { fadeIn: Math.min(half, fi), fadeOut: Math.min(half, fo) };
}

/** Equal-power (sine) fade gain for a normalised position u in [0, 1]. */
export function equalPower(u: number): number {
  const c = Math.min(1, Math.max(0, u));
  return Math.sin((Math.PI / 2) * c);
}

/** fadeIn(t) * fadeOut(t) at clip-local ms. */
export function audioFadeGain(audio: ClipAudio, localMs: number, durMs: number): number {
  const { fadeIn, fadeOut } = clampedAudioFades(audio, durMs);
  const t = Math.min(durMs, Math.max(0, localMs));
  const a = fadeIn > 0 ? equalPower(t / fadeIn) : 1;
  const b = fadeOut > 0 ? equalPower((durMs - t) / fadeOut) : 1;
  return a * b;
}

/** Volume keyframe offset (dB) at clip-local ms; 0 without keyframes. */
export function volumeDbAt(audio: ClipAudio, localMs: number): number {
  const v = audio.volume;
  if (!v) return 0;
  const x = evaluate(v, localMs);
  return Number.isFinite(x) ? x : 0;
}

/** gainDb + volume(t): the level shown by the timeline's volume line. */
export function levelDbAt(audio: ClipAudio, localMs: number): number {
  return (audio.gainDb ?? 0) + volumeDbAt(audio, localMs);
}

/** Final linear gain of a clip's audio at clip-local ms (see the header). */
export function clipGainAt(audio: ClipAudio, localMs: number, durMs: number): number {
  if (audio.muted) return 0;
  return dbToLin(levelDbAt(audio, localMs)) * audioFadeGain(audio, localMs, durMs);
}

/** Video fade (black) factor at clip-local ms: linear ramps, each clamped to half the clip. */
export function videoFadeFactor(fadeInMs: number | undefined, fadeOutMs: number | undefined, localMs: number, durMs: number): number {
  const half = Math.max(0, durMs / 2);
  const fi = Math.min(half, Math.max(0, fadeInMs ?? 0));
  const fo = Math.min(half, Math.max(0, fadeOutMs ?? 0));
  if (fi <= 0 && fo <= 0) return 1;
  const t = Math.min(durMs, Math.max(0, localMs));
  const a = fi > 0 ? Math.min(1, t / fi) : 1;
  const b = fo > 0 ? Math.min(1, (durMs - t) / fo) : 1;
  return Math.max(0, a) * Math.max(0, b);
}
