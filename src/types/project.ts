/**
 * Cappycat timeline document model.
 * Source of truth for docs/CONTRACTS.md. Mirrored by
 * src-tauri/src/model.rs and pipeline/cappycat_pipeline/schema.py.
 *
 * Units: milliseconds for time, source pixels for geometry, bbox = [x1, y1, x2, y2].
 */

export type AssetKind = 'video' | 'audio' | 'image' | 'lut';

export interface Asset {
  id: string;
  path: string;
  name: string;
  kind: AssetKind;
  durationMs: number;
  width: number;
  height: number;
  fps: number;
  hasAudio: boolean;
  codec?: string;
  sceneTags?: string[];
  /** story position (0-based), from the filename or a manual reorder */
  order?: number;
  /** why the clip got this position, e.g. "leading number 3" */
  orderReason?: string;
  /** voice / background stems (48 kHz WAVs on the source's timeline), set once `separate_audio` ran */
  stems?: AssetStems;
  /**
   * Set on the audio assets created by "Separate to tracks": this asset is the `stem` of the
   * asset `assetId` (path = the stem file, durationMs = the source's duration).
   */
  stemOf?: AssetStemOf;
}

export type StemKind = 'vocals' | 'background';

export interface AssetStemOf {
  assetId: string;
  stem: StemKind;
}

/** Separated audio of an asset (`python -m cappycat_pipeline separate`, Demucs v4). */
export interface AssetStems {
  /** dialogue / vocals only */
  vocals: string;
  /** everything else: ambience, music, SFX */
  background: string;
}

export type TrackKind = 'video' | 'audio' | 'fx';

/** Audio tracks created by "Separate to tracks" ("Voice" / "Background"). */
export type TrackRole = 'voice' | 'background';

export interface Track {
  id: string;
  kind: TrackKind;
  name: string;
  locked: boolean;
  muted: boolean;
  clips: Clip[];
  role?: TrackRole;
}

export type Easing =
  | 'linear'
  | 'easeIn'
  | 'easeOut'
  | 'easeInOut'
  | 'bounce'
  | 'elastic'
  | 'bezier';

export interface Keyframe<T> {
  timeMs: number; // relative to clip start on the timeline
  value: T;
  easing: Easing;
  /** cubic-bezier control points (x1,y1,x2,y2) used when easing === 'bezier' */
  bezier?: [number, number, number, number];
}

export interface Keyframed<T> {
  static: T;
  keyframes: Keyframe<T>[];
}

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];
export type Rect = [number, number, number, number]; // x, y, w, h
export type BBox = [number, number, number, number]; // x1, y1, x2, y2

export interface ClipTransform {
  position: Keyframed<Vec2>; // normalised offset of the clip centre, [-1,1]
  scale: Keyframed<number>; // 1 = fit
  rotation: Keyframed<number>; // degrees
  opacity: Keyframed<number>; // 0..1
  blur: Keyframed<number>; // px
}

export type SpeedPreset =
  | 'normal'
  | 'montage'
  | 'hero_time'
  | 'bullet'
  | 'jump_cut'
  | 'flash_in'
  | 'flash_out'
  | 'custom';

export interface SpeedPoint {
  /** normalised position over the SOURCE range, 0..1 */
  t: number;
  /** playback speed multiplier, 0.1 .. 10 */
  speed: number;
}

export interface SpeedCurve {
  preset: SpeedPreset;
  points: SpeedPoint[];
  /** use RAFT optical-flow interpolation when speed < 1 */
  opticalFlow: boolean;
}

export type HslChannel =
  | 'red'
  | 'orange'
  | 'yellow'
  | 'green'
  | 'cyan'
  | 'blue'
  | 'purple'
  | 'magenta';

export interface HslOffset {
  h: number; // -100..100, CapCut scale: +-100 shifts the colour +-30 deg towards its neighbour
  s: number; // -100..100
  l: number; // -100..100
}

export type CurvePoint = [number, number]; // x,y in 0..1

export interface ColorCurves {
  master: CurvePoint[];
  r: CurvePoint[];
  g: CurvePoint[];
  b: CurvePoint[];
}

/** Adjust sliders use CapCut's scale: -50..50 (sharpness 0..50). HSL offsets use -100..100. */
export interface ColorGrade {
  exposure: number;
  /** CapCut-style brilliance: lifts shadows, recovers highlights, keeps midtone contrast */
  brilliance: number;
  contrast: number;
  brightness: number;
  highlights: number;
  shadows: number;
  saturation: number;
  vibrance: number;
  sharpness: number;
  temperature: number; // blue(-) .. yellow(+)
  tint: number; // green(-) .. magenta(+)
  lift: Vec3;
  gamma: Vec3;
  gain: Vec3;
  offset: Vec3;
  hsl: Record<HslChannel, HslOffset>;
  curves: ColorCurves;
  lutAssetId: string | null;
  lutIntensity: number; // 0..1
  vignette: number; // 0..1
  grain: number; // 0..1
}

/**
 * CapCut-style voice separation: `original` (unchanged), `voice` ("Isolate voice": the vocals stem)
 * or `background` ("Remove vocals": ambience, music and SFX). Needs `Asset.stems`.
 */
export type VoiceMode = 'original' | 'voice' | 'background';

export interface ClipAudio {
  gainDb: number;
  normalize: boolean;
  muted: boolean;
  /** default 'original' */
  voice?: VoiceMode;
  /**
   * Keep the pitch when the speed is not 1 (CapCut "Pitch" off). Default true. The export
   * time-stretches pitch-preserving; the preview sets `HTMLMediaElement.preservesPitch`.
   */
  keepPitch?: boolean;
  /** Equal-power (sine) fade-in, timeline ms from the clip start; default 0, clamped to half the clip. */
  fadeInMs?: number;
  /** Equal-power (sine) fade-out, timeline ms before the clip end; default 0, clamped to half the clip. */
  fadeOutMs?: number;
  /**
   * Volume keyframes: a dB offset added to `gainDb`, keyed in clip-local timeline ms (same
   * `Keyframed` semantics as transforms; static value 0). Final gain at clip-local t:
   * `dbToLin(gainDb + volume(t)) * fadeIn(t) * fadeOut(t)`, or 0 when muted.
   */
  volume?: Keyframed<number>;
}

export type MaskShape = 'rectangle' | 'circle' | 'split' | 'filmstrip';

export interface ClipMask {
  shape: MaskShape;
  feather: number; // 0..1
  rect: Keyframed<Rect>; // normalised 0..1 of the frame
  inverted: boolean;
}

export type BlendMode =
  | 'normal'
  | 'multiply'
  | 'screen'
  | 'overlay'
  | 'softLight'
  | 'darken'
  | 'lighten'
  | 'colorDodge';

export interface ReframeKeyframe {
  frame: number;
  timeMs: number;
  crop: BBox;
  zoom: number;
  tx: number; // normalised translation -1..1
  ty: number;
}

export interface ReframeTrack {
  sourceWidth: number;
  sourceHeight: number;
  keyframes: ReframeKeyframe[];
  reason?: string;
}

export interface FreezeFrame {
  atMs: number;
  holdMs: number;
}

/** Transitions between two clips on the same video track (FEATURES_V2 §6). */
export type TransitionType =
  | 'dissolve'
  | 'dipToBlack'
  | 'dipToWhite'
  | 'wipeLeft'
  | 'wipeRight'
  | 'wipeUp'
  | 'wipeDown'
  | 'slideLeft'
  | 'slideRight'
  | 'pushLeft'
  | 'pushRight'
  | 'zoomIn'
  | 'zoomOut'
  | 'blurDissolve'
  | 'flash'
  | 'circleOpen';

/**
 * On the INCOMING clip: the transition from the previous clip on the same track that ends where
 * this clip starts (gap under one frame). Centred on the cut (cut - d/2 .. cut + d/2); the
 * timeline length does not change. `durationMs` 100..3000 (default 500), clamped to the shorter clip.
 */
export interface ClipTransition {
  type: TransitionType;
  durationMs: number;
}

/** Effects: clips on an `fx` track (FEATURES_V2 §7). */
export type EffectType =
  | 'cameraSnap'
  | 'fadeFromBlack'
  | 'fadeToBlack'
  | 'fadeFromWhite'
  | 'fadeToWhite'
  | 'blackAndWhite'
  | 'sepia'
  | 'letterbox'
  | 'shake'
  | 'zoomPunch'
  | 'blurIn'
  | 'blurOut'
  | 'rgbSplit'
  | 'vhs'
  | 'vignettePulse'
  | 'flashWhite';

export interface ClipEffect {
  type: EffectType;
  /** 0..1, default 1 */
  intensity: number;
  /** type-specific parameters (see src/engine/effects.ts EFFECT_INFO) */
  params?: Record<string, number>;
}

export interface Clip {
  id: string;
  assetId: string;
  trackId: string;
  startMs: number;
  inMs: number;
  outMs: number;
  speed: SpeedCurve;
  transform: ClipTransform;
  color: ColorGrade;
  audio: ClipAudio;
  mask: ClipMask | null;
  blendMode: BlendMode;
  reframe: ReframeTrack | null;
  label?: string;
  freezeFrame: FreezeFrame | null;
  reversed: boolean;
  /**
   * Linked clips: the same id on a video clip, its mirrored audio clip and any stem clips
   * ("Separate to tracks"). A group can have more than two members; every timing edit
   * (move, trim, split, delete, speed, freeze, reverse) applies to the whole group.
   */
  linkId?: string;
  /** transition from the previous clip on the same track (this clip is the incoming one) */
  transitionIn?: ClipTransition | null;
  /** set on effect clips (fx tracks, `assetId: ''`); duration = outMs - inMs (speed ignored) */
  effect?: ClipEffect;
  /** video fade from black over the first ms of the clip (clip-local timeline time); default 0 */
  fadeInMs?: number;
  /** video fade to black over the last ms of the clip; default 0 */
  fadeOutMs?: number;
}

/** Slider values of a universal adjustment; each one is ADDED to the clip's own value. */
export interface AdjustValues {
  exposure?: number;
  brilliance?: number;
  contrast?: number;
  brightness?: number;
  highlights?: number;
  shadows?: number;
  saturation?: number;
  vibrance?: number;
  sharpness?: number;
  temperature?: number;
  tint?: number;
  vignette?: number;
  grain?: number;
  hsl?: Record<HslChannel, HslOffset>;
}

/** Project-wide adjustment layered on top of every clip's grade (toggleable). */
export interface UniversalAdjust {
  enabled: boolean;
  name: string;
  values: AdjustValues;
}

/** presets/universal-adjust.json */
export interface UniversalPresetFile {
  version: number;
  name: string;
  enabledByDefault: boolean;
  values: AdjustValues;
}

export interface BeatMarker {
  timeMs: number;
  strength: number; // 0..1
  kind: 'beat1' | 'beat2';
}

/**
 * How the exporter makes frames when the output frame rate is higher than a clip's effective rate
 * (source fps x speed): `none` repeats the nearest source frame, `frameBlend` crossfades the two
 * neighbours, `opticalFlow` synthesises in-between frames with RAFT. Export-only; default 'opticalFlow'.
 */
export type FrameInterpolation = 'frameBlend' | 'opticalFlow' | 'none';

export interface Project {
  version: 1;
  id: string;
  name: string;
  /** 24, 25, 30, 40, 48, 50 or 60 (23.976 / 29.97 for sources); the UI offers 24/30/40/50/60 */
  fps: number;
  width: number;
  height: number;
  assets: Asset[];
  tracks: Track[];
  beatMarkers: BeatMarker[];
  /** universal adjust preset applied on top of every clip (see presets/universal-adjust.json) */
  universalAdjust?: UniversalAdjust;
  /** default 'opticalFlow' */
  frameInterpolation?: FrameInterpolation;
}

/* ---------------- pipeline analysis result ---------------- */

export type ShotMethod = 'transnetv2' | 'pyscenedetect';

export interface Shot {
  index: number;
  startFrame: number;
  endFrame: number;
  startMs: number;
  endMs: number;
  confidence: number;
  method: ShotMethod;
  /** ids of the main-cast characters seen in this shot (characters/characters.json) */
  cast?: string[];
}

export interface DetectedInstance {
  trackId: number;
  label: string;
  bbox: BBox;
  score: number;
}

export interface DuplicateFinding {
  shotIndex: number;
  frame: number;
  timeMs: number;
  primary: DetectedInstance;
  duplicate: DetectedInstance;
  similarity: number;
  /** main-cast character id when the duplicate is a named character, e.g. "bunny" */
  character?: string;
  characterName?: string;
}

export interface ShotReframe {
  shotIndex: number;
  track: ReframeTrack;
}

export interface AudioAnalysis {
  integratedLufs: number;
  truePeakDb: number;
  recommendedGainDb: number;
  beats: number[]; // ms
  tempoBpm: number | null;
}

export interface TransitionAnalysis {
  fromShot: number;
  toShot: number;
  flowMagnitude: number;
  smoothness: number; // 0..1
  suggestion: 'cut' | 'dissolve';
}

export interface ClipAnalysis {
  path: string;
  asset: Asset;
  shots: Shot[];
  duplicates: DuplicateFinding[];
  reframe: ShotReframe[];
  audio: AudioAnalysis | null;
  transitions: TransitionAnalysis[];
}

export interface AnalysisResult {
  version: 1;
  generatedAt: string;
  clips: ClipAnalysis[];
  timeline: Project;
}

/* ---------------- pipeline progress events ---------------- */

export type PipelineStage =
  | 'ingest'
  | 'shots'
  | 'perception'
  | 'reframe'
  | 'audio'
  | 'transitions'
  | 'assemble'
  | 'export'
  | 'download'
  | 'separate';

export interface PipelineProgress {
  event: 'progress';
  stage: PipelineStage;
  clip: string | null;
  pct: number;
  message: string;
}

export interface PipelineLog {
  event: 'log';
  level: 'info' | 'warn' | 'error';
  message: string;
}

export interface PipelineResultEvent {
  event: 'result';
  path: string;
}

export type PipelineEvent = PipelineProgress | PipelineLog | PipelineResultEvent;

export interface PipelineOptions {
  /** open-vocabulary prompts, e.g. ["raccoon in hoodie", "person"] */
  prompts: string[];
  detector: 'yolo_world' | 'grounded_sam2' | 'hybrid';
  shotDetector: 'transnetv2' | 'pyscenedetect' | 'auto';
  targetDurationMs: number; // e.g. 165000 (2:45)
  normalizeAudio: boolean;
  targetLufs: number; // e.g. -14
  detectBeats: boolean;
  similarityThreshold: number; // 0..1, default 0.85
  smoothing: 'ema' | 'savgol';
  smoothingAlpha: number; // EMA inertia 0..1
  /** analyse clips in the order given (the UI's story order) instead of re-ordering by filename */
  keepOrder: boolean;
}
