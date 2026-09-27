/**
 * Program monitor. A single rAF loop owns playback and drawing, outside React:
 *  - the frame-accurate playhead lives in `playClock`; the store gets a ~15 Hz copy for the UI;
 *  - it only renders when something changed (playing, playhead, document, view options, a video
 *    frame arrived), so a paused preview costs nothing;
 *  - two <video> elements alternate: the next shot is preloaded (and pre-seeked) in the standby
 *    element, and the clock holds while the active element is not ready, so cuts don't stall.
 *    During a transition both play: the standby element carries the other clip (with handles past
 *    its edges), both are drawn to off-screen targets and mixed by the transition pass;
 *  - effects (fx tracks) run as a chain of full-frame passes on the composite; a camera snap
 *    freezes the composite of its first frame (captured to a texture) and plays the shutter;
 *  - audio: the video element plays the top clip's audio as heard through its mirrored audio clip
 *    (gain, fades, volume keyframes, mute, voice mode, keep-pitch); every other audio-track clip
 *    under the playhead (stems, music, voice-over) plays on a pooled <audio> element with its own
 *    gain node.
 * With nothing to mix (no transition, no effect) the layer is drawn straight to the canvas.
 */
import { useEffect, useRef, useState } from 'react';
import { assetOf, clipDurationMs, clipsAt, projectDurationMs, useEditor } from '@/state/store';
import { audioMirrorIds } from '@/state/edits';
import { playClock, UI_PLAYHEAD_INTERVAL_MS } from '@/state/clock';
import { resolveAt, resolveClip, resolveClipExtended, transitionAt, type ActiveTransition, type ResolvedFrame } from '@/engine/playback';
import { ColorRenderer, type LayerParams, type RenderTarget } from '@/engine/color/renderer';
import { FxCompositor } from '@/engine/color/compositor';
import type { Lut3D } from '@/engine/color/lut';
import { effectiveGrade } from '@/engine/grade';
import { THEME } from '@/lib/theme';
import { api } from '@/lib/tauri';
import type { Clip, DuplicateFinding, Project } from '@/types/project';
import { stemFor, stemNeedsResync, voiceMode } from '@/engine/voice';
import { clipGainAt, keepPitchOf } from '@/engine/audio';
import { activeEffects, effectCode, effectFrame, proceduralShutter, SHUTTER_GAIN, type ActiveEffect } from '@/engine/effects';
import { transitionCode } from '@/engine/transitions';

/** LUTs by file path (loaded once through `load_lut`). */
const lutCache = new Map<string, Lut3D | 'loading' | 'error'>();
/** how long playback waits for a video element before letting the clock run anyway */
const MAX_HOLD_MS = 1500;
/** pooled <audio> elements for audio-track clips (stems, music, voice-over) */
const AUDIO_POOL = 4;

type EditorSnapshot = ReturnType<typeof useEditor.getState>;

export default function Preview() {
  const compare = useEditor((s) => s.compareMode);
  const projectW = useEditor((s) => s.project.width);
  const projectH = useEditor((s) => s.project.height);
  const hasClips = useEditor((s) => s.project.tracks.some((t) => t.clips.length > 0));

  const stageRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rawRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const videoARef = useRef<HTMLVideoElement>(null);
  const videoBRef = useRef<HTMLVideoElement>(null);
  /** plays the vocals / background stem when the active clip isolates the voice or removes it */
  const stemRef = useRef<HTMLAudioElement>(null);
  const poolRefs = useRef<Array<HTMLAudioElement | null>>([]);
  const rendererRef = useRef<ColorRenderer | null>(null);
  const compositorRef = useRef<FxCompositor | null>(null);
  const invalidateRef = useRef<() => void>(() => undefined);
  const [glError, setGlError] = useState<string | null>(null);
  const [stageSize, setStageSize] = useState({ w: 960, h: 540 });

  // fit canvas to stage, keep project aspect
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      const cols = compare ? 2 : 1;
      const availW = Math.max(64, r.width / cols - 8);
      const availH = Math.max(64, r.height - 8);
      const aspect = projectW / projectH;
      let w = availW;
      let h = w / aspect;
      if (h > availH) {
        h = availH;
        w = h * aspect;
      }
      setStageSize((s) => (s.w === Math.floor(w) && s.h === Math.floor(h) ? s : { w: Math.floor(w), h: Math.floor(h) }));
    };
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
    return () => ro.disconnect();
  }, [projectW, projectH, compare]);

  // create renderer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      rendererRef.current = new ColorRenderer(canvas);
      compositorRef.current = new FxCompositor(rendererRef.current.gl);
      setGlError(null);
    } catch (e) {
      setGlError(String(e));
    }
    return () => {
      compositorRef.current?.dispose();
      compositorRef.current = null;
      rendererRef.current?.dispose();
      rendererRef.current = null;
    };
  }, []);

  // canvas backing size (only when the stage changes)
  useEffect(() => {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    for (const c of [canvasRef.current, rawRef.current, overlayRef.current]) {
      if (!c) continue;
      c.width = Math.round(stageSize.w * dpr);
      c.height = Math.round(stageSize.h * dpr);
      c.style.width = `${stageSize.w}px`;
      c.style.height = `${stageSize.h}px`;
    }
    invalidateRef.current();
  }, [stageSize, compare]);

  // ---- the playback / render loop ----
  useEffect(() => {
    const videos = [videoARef.current!, videoBRef.current!];
    const srcOf = ['', ''];
    const pool = poolRefs.current.filter((x): x is HTMLAudioElement => !!x);
    const poolSrc = pool.map(() => '');
    const poolClip: Array<string | null> = pool.map(() => null);
    let active = 0;
    let raf = 0;
    let lastTs = performance.now();
    let lastSync = 0;
    let needs = true;
    let holdSince: number | null = null;
    let waiting = false;
    let stemSrc = '';
    let activeClip = '';
    let lastHead = playClock.get();
    let lastPlaying = false;
    /** camera snap: the captured composite of its first frame */
    let snapKey = '';
    let audio: { ctx: AudioContext; gains: Map<HTMLMediaElement, GainNode>; shutter: AudioBuffer | null } | null = null;
    const invalidate = () => {
      needs = true;
    };
    invalidateRef.current = invalidate;

    // --- audio graph: created on the first play (a user gesture), resumed on every play ---
    const wire = (el: HTMLMediaElement | null) => {
      if (!el || !audio || audio.gains.has(el)) return;
      try {
        const g = audio.ctx.createGain();
        audio.ctx.createMediaElementSource(el).connect(g);
        g.connect(audio.ctx.destination);
        audio.gains.set(el, g);
      } catch {
        /* element volume fallback */
      }
    };
    const setGain = (el: HTMLMediaElement, lin: number) => {
      const g = audio?.gains.get(el);
      if (g) {
        if (Math.abs(g.gain.value - lin) > 1e-4) g.gain.value = lin;
        if (el.volume !== 1) el.volume = 1;
      } else el.volume = Math.max(0, Math.min(1, lin));
    };
    const onPlayStart = () => {
      if (!audio) {
        try {
          const ctx = new AudioContext();
          audio = { ctx, gains: new Map(), shutter: null };
        } catch {
          audio = null;
        }
      }
      if (audio) {
        [...videos, stemRef.current, ...pool].forEach((el) => wire(el));
        if (audio.ctx.state === 'suspended') void audio.ctx.resume().catch(() => undefined);
      }
    };
    const playShutter = () => {
      if (!audio) return;
      try {
        if (!audio.shutter) {
          const data = proceduralShutter(audio.ctx.sampleRate);
          const buf = audio.ctx.createBuffer(1, data.length, audio.ctx.sampleRate);
          buf.getChannelData(0).set(data);
          audio.shutter = buf;
        }
        const src = audio.ctx.createBufferSource();
        src.buffer = audio.shutter;
        const g = audio.ctx.createGain();
        g.gain.value = SHUTTER_GAIN;
        src.connect(g);
        g.connect(audio.ctx.destination);
        src.start();
      } catch {
        /* no audio output */
      }
    };

    const unsubStore = useEditor.subscribe((s, prev) => {
      if (s.playing && !prev.playing) onPlayStart();
      if (
        s.project !== prev.project ||
        s.compareMode !== prev.compareMode ||
        s.showGrid !== prev.showGrid ||
        s.showBoxes !== prev.showBoxes ||
        s.lastAnalysis !== prev.lastAnalysis ||
        s.mediaServerUrl !== prev.mediaServerUrl ||
        s.playing !== prev.playing
      )
        needs = true;
    });
    const unsubClock = playClock.subscribe(invalidate);
    const mediaEvents = ['loadeddata', 'seeked', 'canplay'];
    videos.forEach((v) => mediaEvents.forEach((ev) => v.addEventListener(ev, invalidate)));

    const pauseEl = (el: HTMLMediaElement | null) => {
      if (el && !el.paused) el.pause();
    };
    const setSrc = (i: number, url: string) => {
      srcOf[i] = url;
      videos[i].src = url;
      videos[i].load();
    };

    // --- LUT of a clip (loaded once per file, uploaded only when it changes) ---
    const lutFor = (p: Project, assetId: string | null): Lut3D | null => {
      if (!assetId) return null;
      const asset = p.assets.find((a) => a.id === assetId);
      if (!asset) return null;
      const hit = lutCache.get(asset.path);
      if (hit && typeof hit !== 'string') return hit;
      if (!hit) {
        lutCache.set(asset.path, 'loading');
        api
          .loadLut(asset.path)
          .then((lut) => {
            lutCache.set(asset.path, lut);
            invalidate();
          })
          .catch((e) => {
            lutCache.set(asset.path, 'error');
            useEditor.getState().log('warn', `LUT ${asset.name} could not be loaded: ${String(e)}`);
          });
      }
      return null;
    };

    const tick = (ts: number) => {
      raf = requestAnimationFrame(tick);
      const dt = Math.min(100, ts - lastTs);
      lastTs = ts;
      const s = useEditor.getState();
      let head = playClock.get();
      if (s.playing) {
        // hold the clock while the active video element is not ready (cut to a new file)
        const hold = waiting && s.shuttle > 0;
        if (hold && holdSince == null) holdSince = ts;
        if (!hold) holdSince = null;
        if (!hold || ts - (holdSince ?? ts) > MAX_HOLD_MS) {
          head += dt * s.shuttle;
          const duration = projectDurationMs(s.project);
          if (head >= duration) {
            if (s.loop && duration > 0 && s.shuttle > 0) head = 0;
            else {
              head = duration;
              s.setPlaying(false);
            }
          } else if (head <= 0 && s.shuttle < 0) {
            head = 0;
            s.setPlaying(false);
          }
          playClock.set(head);
        }
        if (ts - lastSync >= UI_PLAYHEAD_INTERVAL_MS) {
          lastSync = ts;
          s.syncPlayhead(head);
        }
      }
      if (!needs && !s.playing) return;
      needs = false;
      render(s, head, s.playing && useEditor.getState().playing);
      if (import.meta.env.DEV && probeRequest) answerProbe();
    };

    // dev only (perf / verification harness): sample the program monitor right after a render
    let probeRequest: ((px: number[][]) => void) | null = null;
    const answerProbe = () => {
      const r = probeRequest;
      probeRequest = null;
      const gl = rendererRef.current?.gl;
      const c = canvasRef.current;
      if (!r || !gl || !c) return;
      const out: number[][] = [];
      const px = new Uint8Array(4);
      for (let j = 0; j < 5; j++)
        for (let i = 0; i < 9; i++) {
          const x = Math.round(((i + 0.5) / 9) * (c.width - 1));
          const y = Math.round(((4 - j + 0.5) / 5) * (c.height - 1)); // row 0 = top
          gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
          out.push([px[0], px[1], px[2]]);
        }
      r(out);
    };
    if (import.meta.env.DEV) {
      (window as unknown as Record<string, unknown>).__cappyPreview = {
        /** 9 x 5 grid of RGB samples (row-major from the top-left) after the next render */
        probe: () =>
          new Promise<number[][]>((res) => {
            probeRequest = res;
            needs = true;
          }),
      };
    }

    /** One layer (clip) as the renderer wants it. */
    const layerOf = (p: Project, frame: ResolvedFrame, timeMs: number): Omit<LayerParams, 'sourceWidth' | 'sourceHeight'> => ({
      grade: effectiveGrade(frame.clip.color, p.universalAdjust),
      position: frame.position,
      scale: frame.scale,
      rotationDeg: frame.rotation,
      opacity: frame.opacity,
      blurPx: frame.blur,
      crop: frame.crop,
      mask: frame.mask,
      maskRect: frame.maskRect,
      timeMs,
      fade: frame.fade,
    });

    /** Drive an element to a resolved frame (play at its rate while playing forward, else seek + pause). */
    const drive = (el: HTMLVideoElement, frame: ResolvedFrame, s: EditorSnapshot, playing: boolean) => {
      const wantSec = frame.sourceMs / 1000;
      const canSeek = el.readyState >= 1;
      const drift = Math.abs(el.currentTime - wantSec);
      const forward = playing && s.shuttle > 0 && !frame.frozen && !frame.clip.reversed && frame.rate > 0;
      if (forward) {
        el.playbackRate = Math.max(0.0625, Math.min(16, frame.rate * s.shuttle));
        if (drift > 0.12 && canSeek && !el.seeking) el.currentTime = wantSec;
        if (el.paused) void el.play().catch(() => undefined);
      } else {
        pauseEl(el);
        if (canSeek && drift > 0.008 && !el.seeking) el.currentTime = wantSec;
      }
      return { drift, forward };
    };

    const render = (s: EditorSnapshot, head: number, playing: boolean) => {
      const renderer = rendererRef.current;
      const compositor = compositorRef.current;
      const canvas = canvasRef.current;
      const p = s.project;

      // shutter: the playhead reached a camera snap's first frame while playing forward
      const effects = activeEffects(p, head);
      const snap = effects.find((e) => e.effect.type === 'cameraSnap') ?? null;
      if (playing && s.shuttle > 0) {
        for (const t of p.tracks) {
          if (t.kind !== 'fx' || t.muted) continue;
          for (const c of t.clips) {
            if (c.effect?.type !== 'cameraSnap') continue;
            const crossed = lastPlaying ? lastHead < c.startMs : lastHead <= c.startMs;
            if (crossed && head >= c.startMs && head < c.startMs + 250) playShutter();
          }
        }
      }
      lastHead = head;
      lastPlaying = playing;

      playAudioClips(s, head, playing);
      if (!renderer || !canvas || !compositor) return;

      // --- camera snap: the composite of its first frame, captured once ---
      const key = snap ? `${snap.clip.id}:${snap.clip.startMs}:${canvas.width}x${canvas.height}:${docVersion(p)}` : '';
      if (key !== snapKey && snapKey && !snap) snapKey = '';
      const snapCaptured = !!snap && snapKey === key;
      // paused and not captured yet: show (and capture) the snap's first frame; playing: the current one
      const videoTime = snap && !snapCaptured && !playing ? snap.clip.startMs : head;

      const frame = resolveAt(p, videoTime);
      if (!frame) {
        waiting = false;
        videos.forEach(pauseEl);
        pauseEl(stemRef.current);
        if (effects.length && !snap) runEffects(p, effects, null);
        else renderer.clear();
        drawOverlay(null, s.showGrid, null, effects);
        return;
      }
      const asset = assetOf(p, frame.clip);
      if (!asset) {
        renderer.clear();
        return;
      }
      const trans = transitionAt(p, frame.clip, videoTime);
      const other = trans ? (trans.a.id === frame.clip.id ? trans.b : trans.a) : null;
      const otherAsset = other ? assetOf(p, other) : undefined;
      const otherFrame = other && otherAsset ? resolveClipExtended(other, videoTime - other.startMs, otherAsset.kind === 'image' ? Infinity : otherAsset.durationMs) : null;
      const mainFrame = trans ? resolveClipExtended(frame.clip, videoTime - frame.clip.startMs, asset.kind === 'image' ? Infinity : asset.durationMs) : frame;

      const url = api.mediaUrl(s.mediaServerUrl, asset.path);
      const W = p.width || canvas.width;
      const H = p.height || canvas.height;
      const post = !!trans || effects.length > 0;
      type Src = { src: HTMLVideoElement | HTMLCanvasElement; w: number; h: number } | null;
      let mainSrc: Src = null;
      let otherSrc: Src = null;

      if (!url) {
        // Browser / mock mode: synthesise frames so the GPU pipeline (and transitions) is exercised.
        waiting = false;
        const ph = placeholderFrame(0, asset.name, mainFrame, asset.width, asset.height);
        mainSrc = { src: ph, w: ph.width, h: ph.height };
        if (otherFrame && otherAsset) {
          const ph2 = placeholderFrame(1, otherAsset.name, otherFrame, otherAsset.width, otherAsset.height);
          otherSrc = { src: ph2, w: ph2.width, h: ph2.height };
        }
      } else {
        // --- pick the element: at a cut, the standby one when it was preloaded for this shot ---
        const wantAt = mainFrame.sourceMs / 1000;
        if (frame.clip.id !== activeClip) {
          activeClip = frame.clip.id;
          const o = 1 - active;
          if (srcOf[o] === url && videos[o].readyState >= 2 && Math.abs(videos[o].currentTime - wantAt) < 0.1) {
            pauseEl(videos[active]);
            active = o;
          }
        }
        if (srcOf[active] !== url) {
          const o = 1 - active;
          pauseEl(videos[active]);
          if (srcOf[o] === url && !other) active = o;
          else setSrc(active, url);
        }
        const video = videos[active];
        const standby = videos[1 - active];
        standby.muted = true;

        if (other && otherFrame && otherAsset) {
          // --- transition: the standby element plays the other clip ---
          const otherUrl = api.mediaUrl(s.mediaServerUrl, otherAsset.path);
          if (srcOf[1 - active] !== otherUrl) setSrc(1 - active, otherUrl);
          drive(standby, otherFrame, s, playing);
          if (standby.readyState >= 2) otherSrc = { src: standby, w: standby.videoWidth || otherAsset.width, h: standby.videoHeight || otherAsset.height };
        } else {
          pauseEl(standby);
          // --- preload the next shot in the standby element (at its transition start when it has one) ---
          const next = nextShot(p, frame);
          if (next) {
            const nextAsset = assetOf(p, next.clip);
            const nextUrl = nextAsset ? api.mediaUrl(s.mediaServerUrl, nextAsset.path) : '';
            if (nextUrl) {
              if (srcOf[1 - active] !== nextUrl) setSrc(1 - active, nextUrl);
              const want = next.sourceMs / 1000;
              if (standby.readyState >= 1 && !standby.seeking && Math.abs(standby.currentTime - want) > 0.05) standby.currentTime = want;
            }
          }
        }

        // --- audio of the top clip: its mirrored audio clip decides gain / mute / voice mode ---
        const mirror = mirrorOf(p, frame.clip);
        const mirrorTrack = mirror ? p.tracks.find((t) => t.clips.includes(mirror)) : undefined;
        const heard = mirror ?? frame.clip;
        const silent = !!mirrorTrack?.muted || playingReverse(s) || videoTime !== head;
        const stem = stemRef.current;
        const stemPath = stemFor(asset, voiceMode(heard));
        const stemUrl = stemPath ? api.mediaUrl(s.mediaServerUrl, stemPath) : '';
        const useStem = !!(stem && stemUrl);
        if (stem && useStem && stemSrc !== stemUrl) {
          stemSrc = stemUrl;
          stem.src = stemUrl;
          stem.load();
          wire(stem);
        }
        if (!useStem) pauseEl(stem);
        const gain = silent ? 0 : clipGainAt(heard.audio, head - heard.startMs, clipDurationMs(heard));
        const keepPitch = keepPitchOf(heard.audio);
        setGain(video, gain);
        if (video.preservesPitch !== keepPitch) video.preservesPitch = keepPitch;
        if (stem) {
          setGain(stem, gain);
          if (stem.preservesPitch !== keepPitch) stem.preservesPitch = keepPitch;
        }
        video.muted = useStem || !playing || gain <= 0;
        if (stem) stem.muted = !useStem || !playing || gain <= 0;

        // --- sync the element to the resolved source time ---
        const { drift, forward } = drive(video, mainFrame, s, playing);
        if (useStem && stem) {
          if (forward) {
            // stems share the source's timeline: follow the video element (rate, time, play state)
            stem.playbackRate = video.playbackRate;
            const target = video.readyState >= 1 ? video.currentTime : mainFrame.sourceMs / 1000;
            if (stem.readyState >= 1 && stemNeedsResync(stem.currentTime, target)) stem.currentTime = target;
            if (stem.paused) void stem.play().catch(() => undefined);
          } else {
            pauseEl(stem);
            if (stem.readyState >= 1 && stemNeedsResync(stem.currentTime, mainFrame.sourceMs / 1000)) stem.currentTime = mainFrame.sourceMs / 1000;
          }
        }
        waiting = playing && (video.readyState < 2 || (video.seeking && drift > 0.25) || (!!other && standby.readyState < 2));
        if (video.readyState >= 2) mainSrc = { src: video, w: video.videoWidth || asset.width, h: video.videoHeight || asset.height };
      }

      const lut = lutFor(p, frame.clip.color.lutAssetId);
      const cw = canvas.width;
      const ch = canvas.height;
      if (!post) {
        // plain path: the layer straight to the canvas
        if (mainSrc) {
          renderer.render({ ...layerOf(p, mainFrame, head), sourceWidth: mainSrc.w, sourceHeight: mainSrc.h }, lut, { source: mainSrc.src });
          if (s.compareMode && rawRef.current) drawRaw(rawRef.current, mainSrc.src, mainSrc.w, mainSrc.h, frame, asset.path, s);
        }
        drawOverlay(frame, s.showGrid, null, effects);
        return;
      }

      // --- composite (off-screen): one layer, or two mixed by the transition ---
      const x = compositor.target('x', cw, ch);
      let composite: RenderTarget | null = null;
      if (snap && snapCaptured) composite = compositor.target('snap', cw, ch);
      else if (mainSrc || !url) {
        if (trans && otherFrame && other) {
          const a = compositor.target('a', cw, ch);
          const b = compositor.target('b', cw, ch);
          const isA = trans.a.id === frame.clip.id;
          const draw = (t: RenderTarget, src: Src, fr: ResolvedFrame, clip: Clip, slot: number) => {
            if (!src) return renderer.clear(t);
            renderer.render({ ...layerOf(p, fr, head), sourceWidth: src.w, sourceHeight: src.h }, lutFor(p, clip.color.lutAssetId), { source: src.src, target: t, slot });
          };
          draw(isA ? a : b, mainSrc, mainFrame, frame.clip, 0);
          draw(isA ? b : a, otherSrc, otherFrame, other, 1);
          compositor.transition(transitionCode(trans.transition.type), trans.p, a, b, effects.length ? x : null, [W, H], cw, ch);
          if (!effects.length) {
            drawOverlay(frame, s.showGrid, trans, effects);
            return;
          }
        } else if (mainSrc) {
          renderer.render({ ...layerOf(p, mainFrame, head), sourceWidth: mainSrc.w, sourceHeight: mainSrc.h }, lut, { source: mainSrc.src, target: x });
        } else renderer.clear(x);
        composite = x;
        if (snap) {
          // capture the snap's first frame once the element shows it (mock frames are exact)
          const exact = !url || videoTime === snap.clip.startMs || playing;
          const settled = !url || (mainSrc && !videos[active].seeking);
          if (exact && settled) {
            compositor.copy(x, compositor.target('snap', cw, ch), cw, ch);
            snapKey = key;
          }
        }
      }
      if (!composite) {
        drawOverlay(frame, s.showGrid, trans, effects);
        return;
      }
      runEffects(p, effects, composite);
      if (s.compareMode && rawRef.current && mainSrc) drawRaw(rawRef.current, mainSrc.src, mainSrc.w, mainSrc.h, frame, asset.path, s);
      drawOverlay(frame, s.showGrid, trans, effects);
    };

    /** The effect chain in track order; the last pass draws to the canvas. */
    const runEffects = (p: Project, effects: ActiveEffect[], input: RenderTarget | null) => {
      const renderer = rendererRef.current!;
      const compositor = compositorRef.current!;
      const canvas = canvasRef.current!;
      const cw = canvas.width;
      const ch = canvas.height;
      const W = p.width || cw;
      const H = p.height || ch;
      let src = input;
      if (!src) {
        // effects over an empty timeline: run them on black
        src = compositor.target('x', cw, ch);
        renderer.clear(src);
      }
      effects.forEach((e, i) => {
        const last = i === effects.length - 1;
        const out = last ? null : compositor.target(src === compositor.target('x', cw, ch) ? 'y' : 'x', cw, ch);
        compositor.effect(effectCode(e.effect.type), effectFrame(e.effect, e.tMs, e.durMs), src!, out, [W, H], cw, ch);
        if (out) src = out;
      });
    };

    /** Audio-track clips other than the top clip's mirror: pooled <audio> elements (stems, music, VO). */
    const playAudioClips = (s: EditorSnapshot, head: number, playing: boolean) => {
      const p = s.project;
      const top = resolveAt(p, head);
      const skip = new Set(top ? audioMirrorIds(p, top.clip) : []);
      const want: Clip[] = [];
      for (const c of clipsAt(p, head, 'audio')) {
        if (skip.has(c.id) || c.audio.muted) continue;
        want.push(c);
        if (want.length >= pool.length) break;
      }
      const wantIds = new Set(want.map((c) => c.id));
      // free the elements of clips that ended
      poolClip.forEach((id, i) => {
        if (id && !wantIds.has(id)) {
          poolClip[i] = null;
          pauseEl(pool[i]);
        }
      });
      for (const clip of want) {
        let i = poolClip.indexOf(clip.id);
        if (i < 0) {
          i = poolClip.indexOf(null);
          if (i < 0) continue;
          poolClip[i] = clip.id;
        }
        const el = pool[i];
        const asset = assetOf(p, clip);
        const path = asset ? (asset.stemOf ? asset.path : stemFor(asset, voiceMode(clip)) ?? asset.path) : '';
        const url = path ? api.mediaUrl(s.mediaServerUrl, path) : '';
        if (!url) {
          pauseEl(el);
          continue;
        }
        if (poolSrc[i] !== url) {
          poolSrc[i] = url;
          el.src = url;
          el.load();
          wire(el);
        }
        const r = resolveClip(clip, head - clip.startMs);
        const keepPitch = keepPitchOf(clip.audio);
        if (el.preservesPitch !== keepPitch) el.preservesPitch = keepPitch;
        setGain(el, playingReverse(s) ? 0 : clipGainAt(clip.audio, head - clip.startMs, clipDurationMs(clip)));
        const want = r.sourceMs / 1000;
        if (playing && s.shuttle > 0 && !r.frozen && !clip.reversed) {
          el.playbackRate = Math.max(0.0625, Math.min(16, r.rate * s.shuttle));
          if (el.readyState >= 1 && Math.abs(el.currentTime - want) > 0.15) el.currentTime = want;
          if (el.paused) void el.play().catch(() => undefined);
        } else {
          pauseEl(el);
          if (el.readyState >= 1 && Math.abs(el.currentTime - want) > 0.05) el.currentTime = want;
        }
      }
    };

    const drawRaw = (raw: HTMLCanvasElement, src: CanvasImageSource, vw: number, vh: number, frame: ResolvedFrame, assetPath: string, s: EditorSnapshot) => {
      const rc = raw.getContext('2d');
      if (!rc) return;
      rc.fillStyle = '#000';
      rc.fillRect(0, 0, raw.width, raw.height);
      const scale = Math.min(raw.width / vw, raw.height / vh);
      const dw = vw * scale;
      const dh = vh * scale;
      const dx = (raw.width - dw) / 2;
      const dy = (raw.height - dh) / 2;
      rc.drawImage(src, dx, dy, dw, dh);
      if (s.showBoxes) drawRawAnnotations(rc, frame, assetPath, s.lastAnalysis, dx, dy, scale, Math.min(2, window.devicePixelRatio || 1));
    };

    const drawOverlay = (frame: ResolvedFrame | null, showGrid: boolean, _trans: ActiveTransition | null, _effects: ActiveEffect[]) => {
      const ov = overlayRef.current;
      if (!ov) return;
      const ctx = ov.getContext('2d');
      if (!ctx) return;
      // draw in CSS pixels, crisp on high-DPI screens
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = ov.width / dpr;
      const h = ov.height / dpr;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, ov.width, ov.height);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (showGrid) {
        ctx.strokeStyle = 'rgba(255,255,255,0.18)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 1; i < 3; i++) {
          ctx.moveTo(Math.round((w * i) / 3) + 0.5, 0);
          ctx.lineTo(Math.round((w * i) / 3) + 0.5, h);
          ctx.moveTo(0, Math.round((h * i) / 3) + 0.5);
          ctx.lineTo(w, Math.round((h * i) / 3) + 0.5);
        }
        ctx.stroke();
      }
      ctx.font = '11px ui-monospace, Consolas, monospace';
      ctx.textBaseline = 'middle';
      const pill = (text: string, x: number, y: number, color: string, alignRight = false) => {
        const tw = ctx.measureText(text).width;
        const bw = tw + 12;
        const bx = alignRight ? x - bw : x;
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(bx, y, bw, 18);
        ctx.fillStyle = color;
        ctx.fillText(text, bx + 6, y + 9.5);
      };
      if (frame?.crop && frame.clip.reframe) {
        const z = (frame.clip.reframe.sourceWidth / Math.max(1, frame.crop[2] - frame.crop[0])).toFixed(2);
        pill(`AUTO-ZOOM ×${z} · duplicate removed`, 8, h - 26, THEME.accentBright);
      }
      if (frame?.frozen) pill('FREEZE FRAME', w - 8, 8, THEME.text, true);
    };

    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      unsubStore();
      unsubClock();
      videos.forEach((v) => mediaEvents.forEach((ev) => v.removeEventListener(ev, invalidate)));
      pool.forEach(pauseEl);
      invalidateRef.current = () => undefined;
      const a = audio as { ctx: AudioContext } | null;
      if (a) void a.ctx.close().catch(() => undefined);
    };
  }, []);

  return (
    <div className="preview">
      <div className="stage" ref={stageRef}>
        <video ref={videoARef} style={{ display: 'none' }} playsInline preload="auto" crossOrigin="anonymous" />
        <video ref={videoBRef} style={{ display: 'none' }} playsInline preload="auto" crossOrigin="anonymous" muted />
        <audio ref={stemRef} style={{ display: 'none' }} preload="auto" crossOrigin="anonymous" />
        {Array.from({ length: AUDIO_POOL }, (_, i) => (
          <audio
            key={i}
            ref={(el) => {
              poolRefs.current[i] = el;
            }}
            style={{ display: 'none' }}
            preload="auto"
            crossOrigin="anonymous"
          />
        ))}
        {!hasClips ? (
          <div className="empty" style={{ position: 'absolute' }}>
            <div style={{ fontSize: 15, color: 'var(--text-dim)' }}>No clips on the timeline</div>
            Double-click or drag a clip from the media library, use <b>Rough cut in story order</b>,
            <br />
            or run the <b>AI Pipeline</b> to auto-cut, de-duplicate, reframe and normalise.
          </div>
        ) : null}
        {glError ? (
          <div className="empty" style={{ position: 'absolute', color: 'var(--red)' }}>
            {glError}
          </div>
        ) : null}
        <div className={compare ? 'compare' : ''} style={{ ...(compare ? {} : { position: 'relative' }), visibility: hasClips ? 'visible' : 'hidden' }}>
          <div style={compare ? undefined : { display: 'none' }}>
            <canvas ref={rawRef} />
            <span className="tag">Raw source · detections</span>
          </div>
          <div style={{ position: 'relative', display: 'grid', placeItems: 'center' }}>
            <canvas ref={canvasRef} />
            <canvas ref={overlayRef} className="overlay" style={{ width: stageSize.w, height: stageSize.h, left: 'auto', top: 'auto' }} />
            {compare ? <span className="tag">Graded · reframed</span> : null}
          </div>
        </div>
        {hasClips ? <PreviewInfo /> : null}
      </div>
    </div>
  );
}

const docVersions = new WeakMap<Project, number>();
let docCounter = 0;
/** A number per document object (the snap capture is redone when the document changes). */
function docVersion(p: Project): number {
  let v = docVersions.get(p);
  if (v === undefined) {
    v = ++docCounter;
    docVersions.set(p, v);
  }
  return v;
}

function playingReverse(s: { playing: boolean; shuttle: number }): boolean {
  return s.playing && s.shuttle < 0;
}

/** The audio-track clip that plays a video clip's sound (same link and asset), if any. */
function mirrorOf(p: Project, clip: Clip): Clip | undefined {
  if (!clip.linkId) return undefined;
  for (const t of p.tracks) {
    if (t.kind !== 'audio') continue;
    const m = t.clips.find((c) => c.linkId === clip.linkId && c.assetId === clip.assetId);
    if (m) return m;
  }
  return undefined;
}

/**
 * The first frame the standby element must show for the next shot (preloading): the next clip's
 * start, or the start of its transition window (a frame before its in point) when it has one.
 */
function nextShot(p: Project, frame: ResolvedFrame): ResolvedFrame | null {
  const end = frame.clip.startMs + frame.localMs;
  const clipEnd = frame.clip.startMs + clipDurationMs(frame.clip);
  // only when we are within 4 s of the cut
  if (clipEnd - end > 4000) return null;
  const next = resolveAt(p, clipEnd + 0.5);
  if (!next) return null;
  const t = transitionAt(p, next.clip, clipEnd + 0.5);
  if (t && t.b.id === next.clip.id) {
    const a = assetOf(p, next.clip);
    return resolveClipExtended(next.clip, -t.durationMs / 2, a && a.kind !== 'image' ? a.durationMs : Infinity);
  }
  return next;
}

/** Clip label · source time · rate · scale, at the throttled playhead (~15 Hz while playing). */
function PreviewInfo() {
  const text = useEditor((s) => {
    const r = resolveAt(s.project, s.playheadMs);
    return r ? `${r.clip.label ?? 'clip'} · src ${(r.sourceMs / 1000).toFixed(2)}s · ${r.rate.toFixed(2)}× · scale ${r.scale.toFixed(2)}` : '';
  });
  return text ? <div className="preview-info">{text}</div> : null;
}

const placeholderCanvases: Array<HTMLCanvasElement | null> = [null, null];

function hueOf(name: string): number {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return h % 360;
}

/** Browser / mock mode: synthesise a colourful test frame per clip (no media server). */
function placeholderFrame(slot: number, name: string, frame: ResolvedFrame, w0: number, h0: number): HTMLCanvasElement {
  const w = Math.max(320, Math.min(960, w0 / 2));
  const h = Math.round((w * h0) / Math.max(1, w0));
  if (!placeholderCanvases[slot]) placeholderCanvases[slot] = document.createElement('canvas');
  const c = placeholderCanvases[slot]!;
  if (c.width !== w || c.height !== h) {
    c.width = w;
    c.height = h;
  }
  const ctx = c.getContext('2d')!;
  const t = frame.sourceMs / 1000;
  // each source gets its own hue so cuts and transitions are visible
  const hue = hueOf(name);
  const g = ctx.createLinearGradient(0, 0, w, h);
  g.addColorStop(0, `hsl(${hue}, 55%, 38%)`);
  g.addColorStop(0.5, `hsl(${(hue + 40) % 360}, 25%, 28%)`);
  g.addColorStop(1, `hsl(${(hue + 120) % 360}, 40%, 18%)`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
  // test colour bars (one per HSL channel) so grading is visible
  const bars = ['#ff4d4d', '#ff9a3c', '#ffe14d', '#4dff70', '#4df2ff', '#4d7dff', '#a64dff', '#ff4dd2'];
  bars.forEach((col, i) => {
    ctx.fillStyle = col;
    ctx.fillRect((i * w) / bars.length, h * 0.75, w / bars.length, h * 0.25);
  });
  // a moving "character" so motion/reframe is visible
  const cx = w * (0.3 + 0.2 * Math.sin(t * 0.8));
  ctx.fillStyle = '#f2e6d0';
  ctx.beginPath();
  ctx.arc(cx, h * 0.45, h * 0.12, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = 'rgba(255,255,255,0.55)';
  ctx.font = `${Math.round(h / 14)}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.fillText(name, w / 2, h * 0.16);
  ctx.font = `${Math.round(h / 26)}px monospace`;
  ctx.fillText(`${t.toFixed(2)}s · browser mode (no media server)`, w / 2, h * 0.24);
  ctx.textAlign = 'start';
  return c;
}

function drawRawAnnotations(
  ctx: CanvasRenderingContext2D,
  frame: ResolvedFrame,
  assetPath: string,
  analysis: ReturnType<typeof useEditor.getState>['lastAnalysis'],
  dx: number,
  dy: number,
  scale: number,
  dpr: number,
) {
  if (frame.crop) {
    const [x1, y1, x2, y2] = frame.crop;
    ctx.strokeStyle = THEME.cropGuide;
    ctx.lineWidth = 2 * dpr;
    ctx.setLineDash([6 * dpr, 4 * dpr]);
    ctx.strokeRect(dx + x1 * scale, dy + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
    ctx.setLineDash([]);
  }
  if (!analysis) return;
  const clipA = analysis.clips.find((c) => c.path === assetPath);
  if (!clipA) return;
  const nearby: DuplicateFinding[] = clipA.duplicates.filter((d) => Math.abs(d.timeMs - frame.sourceMs) < 700);
  for (const d of nearby) {
    const who = d.characterName ?? d.primary.label;
    box(ctx, d.primary.bbox, THEME.keep, `KEEP ${who}`, dx, dy, scale, dpr);
    box(ctx, d.duplicate.bbox, THEME.duplicate, `DUPLICATE ${who} ${(d.similarity * 100).toFixed(0)}%`, dx, dy, scale, dpr);
  }
}

function box(ctx: CanvasRenderingContext2D, b: number[], color: string, label: string, dx: number, dy: number, s: number, dpr: number) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 * dpr;
  ctx.strokeRect(dx + b[0] * s, dy + b[1] * s, (b[2] - b[0]) * s, (b[3] - b[1]) * s);
  ctx.fillStyle = color;
  ctx.font = `${11 * dpr}px monospace`;
  ctx.fillText(label, dx + b[0] * s + 3 * dpr, dy + b[1] * s + 12 * dpr);
}

