/**
 * Canvas colours (timeline, graph drawer, preview overlays). Mirrors the CSS tokens in
 * styles.css: a black / grey / blue scheme. Change the palette here and in :root together.
 */
export const THEME = {
  bg0: '#08090b',
  bg1: '#0f1114',
  bg2: '#15181c',
  bg3: '#1c2026',
  line: '#262b33',
  lineStrong: '#353c47',
  text: '#e6e9ee',
  textDim: '#97a0ae',
  textFaint: '#7d8797',

  accent: '#3b82f6', // blue: selection, focus, lines
  accentFill: '#2563eb', // primary button fill (white text >= 4.5:1)
  accentBright: '#7fb0ff',
  playhead: '#3b82f6',
  accentSoft: 'rgba(59, 130, 246, 0.18)',

  videoClip: '#27466f', // steel blue
  audioClip: '#2a323d', // graphite
  fxClip: '#3a4150', // slate
  waveform: 'rgba(147, 197, 253, 0.6)',
  speedBadge: '#9cc3ff',
  voiceBadge: '#cfe0ff', // "VOICE" / "NO VOCALS" pill text
  voiceBadgeBg: 'rgba(59, 130, 246, 0.42)', // pill fill (stems ready)
  voiceBadgePending: 'rgba(151, 160, 174, 0.28)', // pill fill while the asset is not separated yet
  reframeBar: '#60a5fa',
  keyframe: '#e6e9ee',
  keyframeSelected: '#60a5fa',
  curve: '#60a5fa',
  speedCurve: '#93c5fd',
  speedPoint: '#e6e9ee',
  freezeBand: 'rgba(147, 197, 253, 0.2)',

  // RGB curves editor
  curvesBg: '#0f1114',
  curvesGrid: '#262b33',
  curvesDiagonal: '#353c47',
  curveMaster: '#e6e9ee',
  curveR: '#f87171',
  curveG: '#4ade80',
  curveB: '#60a5fa',

  beatStrong: '#cfd8e6',
  beatWeak: '#7d8797',
  snapGuide: '#93c5fd',

  // feature set v2: effects, transitions, fades, volume, stems
  effectClip: '#33407a', // indigo-blue: effect clips on FX tracks
  effectClipEdge: '#8ea2ff',
  effectLabel: '#dbe3ff',
  transitionMarker: '#7fb0ff', // bow-tie on a cut with a transition
  transitionMarkerIdle: 'rgba(230, 233, 238, 0.28)', // cut without a transition
  transitionMarkerSelected: '#e6e9ee',
  transitionBand: 'rgba(127, 176, 255, 0.16)', // the transition window over the two clips
  fadeShade: 'rgba(8, 9, 11, 0.5)', // area above a fade ramp
  fadeRamp: 'rgba(230, 233, 238, 0.75)',
  fadeHandle: '#e6e9ee',
  volumeLine: '#93c5fd',
  volumeKey: '#e6e9ee',
  waveVoice: 'rgba(147, 197, 253, 0.75)', // Voice stem waveform
  waveBackground: 'rgba(148, 163, 184, 0.65)', // Background stem waveform
  stemClip: '#24303f',

  keep: '#4ade80',
  duplicate: '#f87171',
  cropGuide: '#60a5fa',
} as const;
