/**
 * Transition catalogue and timing (FEATURES_V2 §6). The GLSL is in ./color/fx-shaders.ts; the
 * exporter mirrors it in src-tauri/src/render/transitions.rs.
 *
 * Timing: the transition sits on the INCOMING clip (`transitionIn`) and is centred on the cut:
 * window = [cut - d/2, cut + d/2), raw progress r = (t - (cut - d/2)) / d, and every type uses
 * p = easeInOut(r) with easeInOut = cubic-bezier(0.42, 0, 0.58, 1) (the keyframe easing).
 * d = clamp(durationMs, 100, 3000), then min(d, shorter of the two clips).
 */
import type { TransitionType } from '@/types/project';
import { ease } from './keyframes';

export const TRANSITION_MIN_MS = 100;
export const TRANSITION_MAX_MS = 3000;
export const TRANSITION_DEFAULT_MS = 500;

export const TRANSITION_TYPES: Array<{ id: TransitionType; label: string; hint: string }> = [
  { id: 'dissolve', label: 'Dissolve', hint: 'Crossfade' },
  { id: 'dipToBlack', label: 'Dip to black', hint: 'Fade out to black, then in' },
  { id: 'dipToWhite', label: 'Dip to white', hint: 'Fade out to white, then in' },
  { id: 'wipeLeft', label: 'Wipe left', hint: 'Soft-edged wipe towards the left' },
  { id: 'wipeRight', label: 'Wipe right', hint: 'Soft-edged wipe towards the right' },
  { id: 'wipeUp', label: 'Wipe up', hint: 'Soft-edged wipe upwards' },
  { id: 'wipeDown', label: 'Wipe down', hint: 'Soft-edged wipe downwards' },
  { id: 'slideLeft', label: 'Slide left', hint: 'The next clip slides in from the right over the current one' },
  { id: 'slideRight', label: 'Slide right', hint: 'The next clip slides in from the left over the current one' },
  { id: 'pushLeft', label: 'Push left', hint: 'The next clip pushes the current one out to the left' },
  { id: 'pushRight', label: 'Push right', hint: 'The next clip pushes the current one out to the right' },
  { id: 'zoomIn', label: 'Zoom in', hint: 'Current clip zooms in and fades, the next one grows in' },
  { id: 'zoomOut', label: 'Zoom out', hint: 'Current clip shrinks away, the next one settles from large' },
  { id: 'blurDissolve', label: 'Blur dissolve', hint: 'Crossfade through a blur' },
  { id: 'flash', label: 'Flash', hint: 'Crossfade through a white flash' },
  { id: 'circleOpen', label: 'Circle open', hint: 'The next clip opens in a growing circle' },
];

export const TRANSITION_IDS = TRANSITION_TYPES.map((t) => t.id);

/** Integer code of each type in the shaders (u_type). Order of TRANSITION_TYPES. */
export function transitionCode(type: TransitionType): number {
  return Math.max(0, TRANSITION_IDS.indexOf(type));
}

export function isTransitionType(x: unknown): x is TransitionType {
  return typeof x === 'string' && (TRANSITION_IDS as string[]).includes(x);
}

export function transitionLabel(type: TransitionType): string {
  return TRANSITION_TYPES.find((t) => t.id === type)?.label ?? type;
}

/** Clamp a transition duration: 100..3000 ms, then to the shorter of the two clips. */
export function clampTransitionMs(durationMs: number, outgoingMs: number, incomingMs: number): number {
  const d = Number.isFinite(durationMs) ? durationMs : TRANSITION_DEFAULT_MS;
  const base = Math.min(TRANSITION_MAX_MS, Math.max(TRANSITION_MIN_MS, d));
  return Math.max(0, Math.min(base, outgoingMs, incomingMs));
}

/** Eased progress p for the raw window position r (0..1). */
export function transitionProgress(raw: number): number {
  return ease(Math.min(1, Math.max(0, raw)), 'easeInOut');
}
