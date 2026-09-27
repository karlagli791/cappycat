/**
 * Dev-only perf harness (`npm run dev`, browser console):
 *   await __cappy.buildSynthetic()        // 43 clips (16 speed-ramped, keyframes, LUT) + linked audio
 *   await __cappy.buildSynthetic({ v2: true }) // + transitions on every 3rd cut, 6 effects, audio fades
 *   await __cappy.measure(3)              // play 3 s and report commit rate, redraw time, GL uploads
 *   await __cappy.measurePaused(2)        // same while paused
 * Never imported in production builds (main.tsx guards it with import.meta.env.DEV).
 */
import { clipDurationMs, useEditor } from '@/state/store';
import { api } from '@/lib/tauri';
import { presetCurve } from '@/engine/speed';
import { defaultAudio, defaultColorGrade, defaultTransform, emptyProject } from '@/engine/defaults';
import type { Asset, Clip, EffectType, Project, SpeedPreset, TransitionType } from '@/types/project';
import { TRANSITION_IDS } from '@/engine/transitions';
import { makeEffectClip } from '@/state/edits';
import { effectInfo } from '@/engine/effects';
import { perfStats } from './perfStats';

const gl = {
  texImage3D: 0,
  texImage2D: 0,
};

function patchGl() {
  const proto = (window as unknown as { WebGL2RenderingContext?: { prototype: WebGL2RenderingContext } }).WebGL2RenderingContext?.prototype;
  if (!proto || (proto as unknown as { __patched?: boolean }).__patched) return;
  (proto as unknown as { __patched?: boolean }).__patched = true;
  const t3 = proto.texImage3D;
  const t2 = proto.texImage2D;
  proto.texImage3D = function (this: WebGL2RenderingContext, ...args: unknown[]) {
    gl.texImage3D++;
    return (t3 as (...a: unknown[]) => void).apply(this, args);
  } as typeof proto.texImage3D;
  proto.texImage2D = function (this: WebGL2RenderingContext, ...args: unknown[]) {
    gl.texImage2D++;
    return (t2 as (...a: unknown[]) => void).apply(this, args);
  } as typeof proto.texImage2D;
}

/** A 17^3 .cube text (cool shadows) so the LUT path is exercised in browser mode. */
function cubeText(size = 17): string {
  const lines = [`TITLE "Synthetic cool"`, `LUT_3D_SIZE ${size}`];
  for (let b = 0; b < size; b++)
    for (let g = 0; g < size; g++)
      for (let r = 0; r < size; r++) {
        const R = r / (size - 1);
        const G = g / (size - 1);
        const B = b / (size - 1);
        lines.push(`${(R * 0.95).toFixed(5)} ${(G * 0.98).toFixed(5)} ${Math.min(1, B * 1.04 + 0.02).toFixed(5)}`);
      }
  return lines.join('\n');
}

const RAMPS: SpeedPreset[] = ['hero_time', 'montage', 'bullet', 'jump_cut', 'flash_in', 'flash_out'];

/** v2 extras: a transition on every 3rd cut (cycling through the types), effects, fades, volume keys. */
export function withV2Extras(p: Project): Project {
  const video = p.tracks.find((t) => t.kind === 'video')!;
  const fx = p.tracks.find((t) => t.kind === 'fx')!;
  const clips = [...video.clips].sort((a, b) => a.startMs - b.startMs);
  let k = 0;
  const vclips = clips.map((c, i) => (i > 0 && i % 3 === 0 ? { ...c, transitionIn: { type: TRANSITION_IDS[k++ % TRANSITION_IDS.length] as TransitionType, durationMs: 800 }, fadeInMs: 0 } : c));
  const effects: EffectType[] = ['cameraSnap', 'shake', 'vhs', 'blurIn', 'letterbox', 'rgbSplit'];
  const fxClips: Clip[] = effects.map((type, i) => {
    const info = effectInfo(type);
    return makeEffectClip({ type, intensity: 1 }, fx.id, 4000 + i * 9000, info.defaultMs, { speed: presetCurve('normal'), transform: defaultTransform(), color: defaultColorGrade(), audio: defaultAudio() }, info.label);
  });
  return {
    ...p,
    tracks: p.tracks.map((t) =>
      t.id === video.id
        ? { ...t, clips: vclips }
        : t.id === fx.id
          ? { ...t, clips: fxClips }
          : t.kind === 'audio'
            ? { ...t, clips: t.clips.map((c) => ({ ...c, audio: { ...c.audio, fadeInMs: 200, fadeOutMs: 300, volume: { static: 0, keyframes: [ { timeMs: 0, value: 0, easing: 'linear' as const }, { timeMs: 1500, value: -6, easing: 'linear' as const } ] } } })) }
            : t,
    ),
  };
}

export function syntheticProject(): Project {
  const p = emptyProject('Perf_43');
  const assets: Asset[] = [];
  for (let i = 0; i < 12; i++) {
    assets.push({
      id: `ast_syn${i}`,
      path: `C:/mock/synthetic/clip${String(i + 1).padStart(2, '0')}.mp4`,
      name: `clip${String(i + 1).padStart(2, '0')}.mp4`,
      kind: 'video',
      durationMs: 20000 + i * 2500,
      width: 1920,
      height: 1080,
      fps: 24,
      hasAudio: true,
      order: i,
    });
  }
  assets.push({ id: 'ast_synlut', path: 'C:/mock/synthetic/cool.cube', name: 'cool.cube', kind: 'lut', durationMs: 0, width: 0, height: 0, fps: 0, hasAudio: false });
  const video = p.tracks.find((t) => t.kind === 'video')!;
  const audio = p.tracks.find((t) => t.kind === 'audio')!;
  const vclips: Clip[] = [];
  const aclips: Clip[] = [];
  let cursor = 0;
  let ramped = 0;
  for (let i = 0; i < 43; i++) {
    const a = assets[i % 12];
    const inMs = (i * 700) % 6000;
    const outMs = inMs + 2500 + ((i * 1300) % 4000);
    const ramp = i % 8 < 3 && ramped < 16;
    if (ramp) ramped++;
    const transform = defaultTransform();
    if (i % 3 === 0) {
      transform.scale = { static: 1, keyframes: [ { timeMs: 0, value: 1, easing: 'easeInOut' }, { timeMs: 1200, value: 1.25, easing: 'easeInOut' } ] };
      transform.position = { static: [0, 0], keyframes: [ { timeMs: 0, value: [0, 0], easing: 'linear' }, { timeMs: 1500, value: [0.1, 0], easing: 'linear' } ] };
    }
    const color = { ...defaultColorGrade(), contrast: 6, lutAssetId: i % 4 === 0 ? 'ast_synlut' : null };
    const base = {
      assetId: a.id,
      startMs: cursor,
      inMs,
      outMs,
      speed: ramp ? presetCurve(RAMPS[ramped % RAMPS.length]) : presetCurve('normal'),
      transform,
      color,
      audio: defaultAudio(),
      mask: null,
      blendMode: 'normal' as const,
      reframe: null,
      freezeFrame: null,
      reversed: false,
      linkId: `lnk_syn${i}`,
    };
    const v = { ...base, id: `clp_synv${i}`, trackId: video.id, label: `${i + 1}. ${a.name}` } as Clip;
    const au = { ...base, id: `clp_syna${i}`, trackId: audio.id, transform: defaultTransform(), label: `${i + 1}. ${a.name} (audio)` } as Clip;
    vclips.push(v);
    aclips.push(au);
    cursor += clipDurationMs(v);
  }
  video.clips = vclips;
  audio.clips = aclips;
  return { ...p, assets };
}

async function sleep(ms: number) {
  await new Promise((r) => setTimeout(r, ms));
}

function stats(arr: number[]) {
  if (!arr.length) return { n: 0, avg: 0, p95: 0, max: 0 };
  const s = [...arr].sort((a, b) => a - b);
  return { n: arr.length, avg: +(arr.reduce((m, x) => m + x, 0) / arr.length).toFixed(3), p95: +s[Math.floor(s.length * 0.95)].toFixed(3), max: +s[s.length - 1].toFixed(3) };
}

async function run(seconds: number, play: boolean) {
  patchGl();
  const st = useEditor.getState();
  perfStats.commits = 0;
  perfStats.timelineDraws.length = 0;
  perfStats.overlayDraws.length = 0;
  gl.texImage2D = 0;
  gl.texImage3D = 0;
  let frames = 0;
  let alive = true;
  const count = () => {
    frames++;
    if (alive) requestAnimationFrame(count);
  };
  requestAnimationFrame(count);
  if (play) {
    st.setPlayhead(0);
    st.setPlaying(true);
  }
  const t0 = performance.now();
  await sleep(seconds * 1000);
  const dt = (performance.now() - t0) / 1000;
  const commits = perfStats.commits;
  const t3 = gl.texImage3D;
  const t2 = gl.texImage2D;
  const draws = [...perfStats.timelineDraws];
  const overlays = [...perfStats.overlayDraws];
  alive = false;
  if (play) useEditor.getState().setPlaying(false);
  return {
    seconds: +dt.toFixed(2),
    rafFps: +(frames / dt).toFixed(1),
    reactCommitsPerSec: +(commits / dt).toFixed(1),
    timelineRedrawMs: stats(draws),
    timelineRedrawsPerSec: +(draws.length / dt).toFixed(1),
    playheadOverlayMs: stats(overlays),
    lutUploadsPerSec: +(t3 / dt).toFixed(1),
    frameUploadsPerSec: +(t2 / dt).toFixed(1),
    playheadMs: Math.round(useEditor.getState().playheadMs),
  };
}

export function installPerfHarness(): void {
  const w = window as unknown as Record<string, unknown>;
  // old code path: LUT text via readTextFile; new code path: api.loadLut (browser mode synthesises one)
  const anyApi = api as unknown as Record<string, unknown>;
  if ('readTextFile' in anyApi) anyApi.readTextFile = async () => cubeText();
  w.__cappy = {
    store: useEditor,
    perfStats,
    gl,
    async buildSynthetic(opts: { v2?: boolean } = {}) {
      const p = opts.v2 ? withV2Extras(syntheticProject()) : syntheticProject();
      useEditor.getState().setProject(p, { undoable: false, path: null });
      await sleep(300);
      const s = useEditor.getState();
      return { clips: s.project.tracks.reduce((n, t) => n + t.clips.length, 0) };
    },
    measure: (seconds = 3) => run(seconds, true),
    /** full timeline redraws while scrolling (document layer) */
    async measureRedraw(steps = 60) {
      perfStats.timelineDraws.length = 0;
      const s = useEditor.getState();
      for (let i = 0; i < steps; i++) {
        s.setScroll((i * 1500) % 120000);
        await new Promise((r) => setTimeout(r, 20));
      }
      s.setScroll(0);
      return stats(perfStats.timelineDraws);
    },
    measurePaused: (seconds = 2) => run(seconds, false),
  };
}
