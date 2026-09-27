import type {
  ClipAudio,
  ClipTransform,
  ColorGrade,
  FrameInterpolation,
  HslChannel,
  HslOffset,
  Keyframed,
  Project,
  SpeedCurve,
  Track,
} from '@/types/project';
import { newId } from './ids';

/** Adjust sliders use CapCut's scale: -50..50 (sharpness 0..50). HSL sliders are -100..100. */
export const ADJUST_SCALE = 50;

export const HSL_CHANNELS: HslChannel[] = [
  'red',
  'orange',
  'yellow',
  'green',
  'cyan',
  'blue',
  'purple',
  'magenta',
];

export function keyframed<T>(value: T): Keyframed<T> {
  return { static: value, keyframes: [] };
}

export function defaultHsl(): Record<HslChannel, HslOffset> {
  const out = {} as Record<HslChannel, HslOffset>;
  for (const c of HSL_CHANNELS) out[c] = { h: 0, s: 0, l: 0 };
  return out;
}

export function defaultColorGrade(): ColorGrade {
  return {
    exposure: 0,
    brilliance: 0,
    contrast: 0,
    brightness: 0,
    highlights: 0,
    shadows: 0,
    saturation: 0,
    vibrance: 0,
    sharpness: 0,
    temperature: 0,
    tint: 0,
    lift: [0, 0, 0],
    gamma: [0, 0, 0],
    gain: [0, 0, 0],
    offset: [0, 0, 0],
    hsl: defaultHsl(),
    curves: {
      master: [
        [0, 0],
        [1, 1],
      ],
      r: [
        [0, 0],
        [1, 1],
      ],
      g: [
        [0, 0],
        [1, 1],
      ],
      b: [
        [0, 0],
        [1, 1],
      ],
    },
    lutAssetId: null,
    lutIntensity: 1,
    vignette: 0,
    grain: 0,
  };
}

export function defaultTransform(): ClipTransform {
  return {
    position: keyframed<[number, number]>([0, 0]),
    scale: keyframed(1),
    rotation: keyframed(0),
    opacity: keyframed(1),
    blur: keyframed(0),
  };
}

export function defaultSpeed(): SpeedCurve {
  return {
    preset: 'normal',
    points: [
      { t: 0, speed: 1 },
      { t: 1, speed: 1 },
    ],
    opticalFlow: true,
  };
}

export function defaultAudio(): ClipAudio {
  return { gainDb: 0, normalize: false, muted: false, voice: 'original', keepPitch: true };
}

/** Frame rates offered in Project settings and the Export dialog (the model also accepts 25 / 48). */
export const PROJECT_FPS_OPTIONS = [24, 30, 40, 50, 60] as const;

export const FRAME_INTERPOLATIONS: Array<{ id: FrameInterpolation; label: string; hint: string }> = [
  { id: 'opticalFlow', label: 'Optical flow', hint: 'RAFT synthesises true in-between frames (best, slowest)' },
  { id: 'frameBlend', label: 'Frame blend', hint: 'Crossfades the two neighbouring source frames' },
  { id: 'none', label: 'None', hint: 'Repeats the nearest source frame' },
];

export const DEFAULT_FRAME_INTERPOLATION: FrameInterpolation = 'opticalFlow';

export function frameInterpolationOf(p: Pick<Project, 'frameInterpolation'>): FrameInterpolation {
  const v = p.frameInterpolation;
  return v === 'frameBlend' || v === 'none' || v === 'opticalFlow' ? v : DEFAULT_FRAME_INTERPOLATION;
}

/** Quick speed buttons (Speed tab, clip menu): each sets a constant curve and ripples. */
export const QUICK_SPEEDS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 5] as const;

export function defaultTracks(): Track[] {
  return [
    { id: newId('trk'), kind: 'video', name: 'Video 1', locked: false, muted: false, clips: [] },
    { id: newId('trk'), kind: 'fx', name: 'FX', locked: false, muted: false, clips: [] },
    { id: newId('trk'), kind: 'audio', name: 'Audio 1', locked: false, muted: false, clips: [] },
  ];
}

export function emptyProject(name = 'Untitled'): Project {
  return {
    version: 1,
    id: newId('proj'),
    name,
    fps: 24,
    width: 1920,
    height: 1080,
    assets: [],
    tracks: defaultTracks(),
    beatMarkers: [],
  };
}

/** Well-known CapCut-style grading presets expressed as partial ColorGrade overrides. */
/** CapCut scale (-50..50). */
export const COLOR_PRESETS: Record<string, Partial<ColorGrade>> = {
  Neutral: {},
  Cinematic_Teal_Orange: {
    contrast: 6,
    saturation: 3,
    temperature: 4,
    lift: [-0.03, 0.01, 0.06],
    gain: [0.06, 0.02, -0.05],
    vignette: 0.25,
  },
  Bleach_Bypass: {
    contrast: 14,
    saturation: -22,
    highlights: 5,
    shadows: -6,
    grain: 0.25,
  },
  Warm_Film: {
    temperature: 9,
    tint: 2,
    contrast: 3,
    vibrance: 5,
    grain: 0.15,
    vignette: 0.15,
  },
  Cool_Noir: {
    temperature: -8,
    saturation: -15,
    contrast: 11,
    shadows: -10,
    vignette: 0.35,
  },
  Vibrant_Pop: {
    saturation: 11,
    vibrance: 9,
    contrast: 5,
    sharpness: 10,
  },
};
