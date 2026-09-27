/**
 * Canvas timeline. Two stacked canvases: the document layer (tracks, clips, ruler) is redrawn only
 * when the document / view changes; the overlay layer (playhead, snap guide, marquee) follows the
 * per-frame playhead clock without React. Drags run as store gestures (one undo step) and are
 * computed from the drag-start snapshot, so they are idempotent.
 *
 * Feature set v2 on the canvas: transition bow-ties on cuts (click selects the cut, Delete removes
 * the transition, drop a transition on a cut), effect clips on FX tracks, fade handles on clip
 * corners (video fades on video clips; audio fades on audio clips, and on the bottom corners of
 * video clips that carry audio without an audio-track mirror) with the ramp drawn over the clip,
 * a volume line on audio clips (drag: volume; Alt+click: add a keyframe, Alt+click a keyframe:
 * remove it; drag a keyframe up / down), and a right-click clip menu.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { assetOf, clipEndMs, findClip, partnersOf, projectDurationMs, useEditor } from '@/state/store';
import { cutsOfTrack, frameMs, transitionDurationMs } from '@/state/edits';
import { playClock } from '@/state/clock';
import { api } from '@/lib/tauri';
import { isConstantSpeed } from '@/engine/speed';
import { assetHasSound, mirrorClip, voiceBadge, voiceMode } from '@/engine/voice';
import { clampedAudioFades, clipGainAt, levelDbAt, volumeDbAt } from '@/engine/audio';
import { effectInfo, isEffectType } from '@/engine/effects';
import { TRANSITION_DEFAULT_MS, isTransitionType } from '@/engine/transitions';
import type { Clip, EffectType, Project, Track } from '@/types/project';
import { THEME } from '@/lib/theme';
import { recordPerf } from '@/dev/perfStats';
import { EyeIcon, FitIcon, LockIcon, SpeakerIcon, UnlockIcon } from './Icons';
import ClipContextMenu from './ClipContextMenu';

const RULER_H = 24;
const BEAT_H = 14;
const TRACK_H: Record<Track['kind'], number> = { video: 60, fx: 34, audio: 56 };
const HEAD_W = 150;
const EDGE_PX = 7;
const DRAG_THRESHOLD_PX = 3;
const AUTOSCROLL_EDGE_PX = 36;
/** volume line range (dB, bottom .. top of the clip body) */
const VOL_MIN_DB = -30;
const VOL_MAX_DB = 12;
const HANDLE_R = 4.5;

export const TRANSITION_MIME = 'application/x-cappycat-transition';
export const EFFECT_MIME = 'application/x-cappycat-effect';

/** Height that shows every track without scrolling (the default timeline height). */
export function timelineContentHeight(tracks: Track[]): number {
  return RULER_H + BEAT_H + tracks.reduce((h, t) => h + TRACK_H[t.kind], 0) + 16;
}

const waveCache = new Map<string, number[] | 'loading'>();

type DragKind = 'move' | 'trimIn' | 'trimOut' | 'scrub' | 'marquee' | 'fade' | 'volLine' | 'volKey';

interface Drag {
  kind: DragKind;
  clipId?: string;
  ids: string[];
  startX: number;
  startY: number;
  startMs: number;
  origStart: number;
  origEnd: number;
  started: boolean;
  /** snap candidates, computed once when the drag starts */
  targets: number[];
  additive: boolean;
  /** clicked an already-selected clip without moving: select it alone on release */
  selectOnClick: string | null;
  lastX: number;
  lastY: number;
  // fades / volume
  edge?: 'in' | 'out';
  target?: 'audio' | 'video';
  keyMs?: number;
  baseDb?: number;
  rowY?: number;
  rowH?: number;
}

interface Row {
  track: Track;
  y: number;
  h: number;
}

type SpecialHit =
  | { kind: 'transition'; clip: Clip }
  | { kind: 'fade'; clip: Clip; track: Track; edge: 'in' | 'out'; target: 'audio' | 'video' }
  | { kind: 'volKey'; clip: Clip; track: Track; keyMs: number }
  | { kind: 'volLine'; clip: Clip; track: Track };

function body(row: Row) {
  return { y: row.y + 4, h: row.h - 8 };
}

function dbToY(db: number, y: number, h: number): number {
  const pad = 6;
  const u = (Math.min(VOL_MAX_DB, Math.max(VOL_MIN_DB, db)) - VOL_MIN_DB) / (VOL_MAX_DB - VOL_MIN_DB);
  return y + pad + (1 - u) * (h - 2 * pad);
}

function yToDb(py: number, y: number, h: number): number {
  const pad = 6;
  const u = 1 - (py - y - pad) / Math.max(1, h - 2 * pad);
  return VOL_MIN_DB + Math.min(1, Math.max(0, u)) * (VOL_MAX_DB - VOL_MIN_DB);
}

/** Does this video clip carry its own sound (no audio-track mirror)? Its audio fades go on the bottom corners. */
function videoCarriesAudio(p: Project, clip: Clip): boolean {
  return assetHasSound(assetOf(p, clip)) && !mirrorClip(p, clip);
}

interface FadeHandle {
  edge: 'in' | 'out';
  target: 'audio' | 'video';
  x: number;
  y: number;
}

function fadeHandles(p: Project, track: Track, clip: Clip, x1: number, x2: number, y: number, h: number, pxPerMs: number): FadeHandle[] {
  if (track.kind === 'fx' || x2 - x1 < 28) return [];
  const dur = (x2 - x1) / pxPerMs;
  const out: FadeHandle[] = [];
  const place = (target: 'audio' | 'video', fi: number, fo: number, hy: number) => {
    const mid = (x1 + x2) / 2;
    out.push({ edge: 'in', target, x: Math.min(mid - 3, x1 + Math.max(6, fi * pxPerMs)), y: hy });
    out.push({ edge: 'out', target, x: Math.max(mid + 3, x2 - Math.max(6, fo * pxPerMs)), y: hy });
  };
  if (track.kind === 'video') {
    const half = dur / 2;
    place('video', Math.min(half, clip.fadeInMs ?? 0), Math.min(half, clip.fadeOutMs ?? 0), y + 6);
    if (videoCarriesAudio(p, clip)) {
      const f = clampedAudioFades(clip.audio, dur);
      place('audio', f.fadeIn, f.fadeOut, y + h - 6);
    }
  } else {
    const f = clampedAudioFades(clip.audio, dur);
    place('audio', f.fadeIn, f.fadeOut, y + 6);
  }
  return out;
}

export default function Timeline({ height }: { height: number }) {
  const project = useEditor((s) => s.project);
  const zoom = useEditor((s) => s.zoom);
  const scrollMs = useEditor((s) => s.scrollMs);
  const selection = useEditor((s) => s.selection);
  const selectedCut = useEditor((s) => s.selectedCut);
  const serverUrl = useEditor((s) => s.mediaServerUrl);
  const fitNonce = useEditor((s) => s.fitNonce);
  const setZoom = useEditor((s) => s.setZoom);
  const setScroll = useEditor((s) => s.setScroll);
  const toggleTrackLock = useEditor((s) => s.toggleTrackLock);
  const toggleTrackMute = useEditor((s) => s.toggleTrackMute);

  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 800, h: 200 });
  const dragRef = useRef<Drag | null>(null);
  const guideRef = useRef<number | null>(null);
  const marqueeRef = useRef<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
  const [waveTick, bump] = useState(0);
  const [menu, setMenu] = useState<{ x: number; y: number; clipId: string; atMs: number } | null>(null);

  const pxPerMs = zoom / 1000;
  const msToX = useCallback((ms: number) => (ms - scrollMs) * pxPerMs, [scrollMs, pxPerMs]);
  const xToMs = useCallback((x: number) => x / pxPerMs + scrollMs, [scrollMs, pxPerMs]);

  const rows: Row[] = useMemo(() => {
    let y = RULER_H + BEAT_H;
    return project.tracks.map((t) => {
      const h = TRACK_H[t.kind];
      const row = { track: t, y, h };
      y += h;
      return row;
    });
  }, [project.tracks]);

  // canvas backing store: resized only when the element size changes
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      const w = Math.max(50, Math.floor(r.width));
      const h = Math.max(50, Math.floor(r.height));
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      for (const c of [canvasRef.current, overlayRef.current]) {
        if (!c) continue;
        c.width = w * dpr;
        c.height = h * dpr;
      }
      setSize((s) => (s.w === w && s.h === h ? s : { w, h }));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // waveforms (source files and stem files)
  useEffect(() => {
    for (const a of project.assets) {
      if (!a.hasAudio || waveCache.has(a.path)) continue;
      waveCache.set(a.path, 'loading');
      void api
        .extractWaveform(a.path, 20)
        .then((w) => {
          waveCache.set(a.path, w);
          bump((n) => n + 1);
        })
        .catch(() => waveCache.delete(a.path));
    }
  }, [project.assets, serverUrl]);

  // ---------- document layer ----------
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = canvas.width / Math.max(1, size.w);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const t0 = performance.now();
    draw(ctx, { project, rows, size, msToX, xToMs, selection: selection.clipIds, selectedCut, zoom });
    if (import.meta.env.DEV) recordPerf('timelineDraws', performance.now() - t0);
  }, [project, rows, size, msToX, xToMs, selection, selectedCut, zoom, waveTick]);

  // ---------- overlay layer (playhead, snap guide, marquee) ----------
  const viewRef = useRef({ msToX, size, pxPerMs, scrollMs });
  viewRef.current = { msToX, size, pxPerMs, scrollMs };
  const drawOverlayNow = useCallback(() => {
    const c = overlayRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const t0 = performance.now();
    const { msToX: toX, size: sz } = viewRef.current;
    const dpr = c.width / Math.max(1, sz.w);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawOverlay(ctx, sz, toX, playClock.get(), guideRef.current, marqueeRef.current);
    if (import.meta.env.DEV) recordPerf('overlayDraws', performance.now() - t0);
  }, []);
  useEffect(() => {
    drawOverlayNow();
  }, [drawOverlayNow, msToX, size]);
  useEffect(() => {
    // follow the playhead per frame; page the view while playing (CapCut style)
    return playClock.subscribe((ms) => {
      const { size: sz, pxPerMs: ppm, scrollMs: sc } = viewRef.current;
      const s = useEditor.getState();
      const x = (ms - sc) * ppm;
      if (s.playing && (x > sz.w - 24 || x < 0)) s.setScroll(Math.max(0, ms - (sz.w * 0.1) / ppm));
      drawOverlayNow();
    });
  }, [drawOverlayNow]);

  // ---------- zoom to fit (Shift+Z, after rough cut / analysis / open) ----------
  const fit = useCallback(() => {
    const s = useEditor.getState();
    const dur = projectDurationMs(s.project);
    const w = viewRef.current.size.w;
    if (dur <= 0 || w <= 80) return;
    s.setZoom(Math.max(0.5, Math.min(400, ((w - 48) / dur) * 1000)));
    s.setScroll(0);
  }, []);
  const firstFit = useRef(true);
  useEffect(() => {
    if (firstFit.current) {
      firstFit.current = false;
      return;
    }
    fit();
  }, [fitNonce, fit]);

  // ---------- hit testing ----------
  const hitClip = (x: number, y: number): { clip: Clip; track: Track; edge: 'in' | 'out' | null } | null => {
    for (const row of rows) {
      if (y < row.y || y >= row.y + row.h) continue;
      for (const clip of row.track.clips) {
        const x1 = msToX(clip.startMs);
        const x2 = msToX(clipEndMs(clip));
        if (x >= x1 && x <= x2) {
          const edgeW = Math.min(EDGE_PX, (x2 - x1) / 3);
          const edge = x - x1 < edgeW ? 'in' : x2 - x < edgeW ? 'out' : null;
          return { clip, track: row.track, edge };
        }
      }
    }
    return null;
  };

  /** Transition markers, fade handles and the volume line take precedence over the clip body. */
  const hitSpecial = (x: number, y: number): SpecialHit | null => {
    const p = useEditor.getState().project;
    const frame = frameMs(p);
    for (const row of rows) {
      if (y < row.y || y >= row.y + row.h) continue;
      const { y: by, h: bh } = body(row);
      if (row.track.kind === 'video') {
        for (const cut of cutsOfTrack(row.track, frame)) {
          const cx = msToX(cut.atMs);
          if (Math.abs(x - cx) <= 7 && Math.abs(y - (row.y + row.h / 2)) <= 8 && markerVisible(cut.prev, cut.next, pxPerMs)) return { kind: 'transition', clip: cut.next };
        }
      }
      if (row.track.locked) return null;
      for (const clip of row.track.clips) {
        const x1 = msToX(clip.startMs);
        const x2 = msToX(clipEndMs(clip));
        if (x < x1 - HANDLE_R || x > x2 + HANDLE_R) continue;
        for (const hnd of fadeHandles(p, row.track, clip, x1, x2, by, bh, pxPerMs)) {
          if (Math.hypot(x - hnd.x, y - hnd.y) <= HANDLE_R + 2.5) return { kind: 'fade', clip, track: row.track, edge: hnd.edge, target: hnd.target };
        }
        if (row.track.kind === 'audio' && x >= x1 && x <= x2 && x2 - x1 > 12) {
          const dur = (x2 - x1) / pxPerMs;
          for (const k of clip.audio.volume?.keyframes ?? []) {
            if (k.timeMs > dur + 1) continue;
            const kx = x1 + k.timeMs * pxPerMs;
            const ky = dbToY(clip.audio.gainDb + k.value, by, bh);
            if (Math.hypot(x - kx, y - ky) <= 5.5) return { kind: 'volKey', clip, track: row.track, keyMs: k.timeMs };
          }
          const ly = dbToY(levelDbAt(clip.audio, (x - x1) / pxPerMs), by, bh);
          if (Math.abs(y - ly) <= 4.5) return { kind: 'volLine', clip, track: row.track };
        }
      }
    }
    return null;
  };

  const snapTargetsFor = (exclude: Set<string>): number[] => {
    const s = useEditor.getState();
    const t: number[] = [0, playClock.get()];
    for (const tr of s.project.tracks) for (const c of tr.clips) if (!exclude.has(c.id)) t.push(c.startMs, clipEndMs(c));
    for (const b of s.project.beatMarkers) t.push(b.timeMs);
    return t;
  };

  const snapTo = (ms: number, targets: number[]): { ms: number; snapped: boolean } => {
    if (!useEditor.getState().snapping) return { ms, snapped: false };
    const threshold = 8 / pxPerMs;
    let best = ms;
    let bestD = threshold;
    for (const t of targets) {
      const d = Math.abs(t - ms);
      if (d < bestD) {
        bestD = d;
        best = t;
      }
    }
    return { ms: best, snapped: best !== ms };
  };

  const local = (e: { clientX: number; clientY: number }) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  // ---------- pointer interactions ----------
  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    setMenu(null);
    canvasRef.current!.setPointerCapture(e.pointerId);
    const { x, y } = local(e);
    const st = useEditor.getState();
    const base: Omit<Drag, 'kind'> = {
      ids: [],
      startX: x,
      startY: y,
      startMs: xToMs(x),
      origStart: 0,
      origEnd: 0,
      started: false,
      targets: [],
      additive: e.shiftKey || e.ctrlKey || e.metaKey,
      selectOnClick: null,
      lastX: x,
      lastY: y,
    };
    if (y < RULER_H + BEAT_H) {
      dragRef.current = { ...base, kind: 'scrub', started: true, targets: snapTargetsFor(new Set()) };
      st.setPlayhead(Math.max(0, snapTo(xToMs(x), dragRef.current.targets).ms));
      return;
    }
    const special = hitSpecial(x, y);
    if (special?.kind === 'transition') {
      st.selectCut(special.clip.id);
      return;
    }
    if (special) {
      const row = rows.find((r) => r.track.id === special.track.id)!;
      const b = body(row);
      if (!st.selection.clipIds.includes(special.clip.id)) st.select([special.clip.id]);
      if (special.kind === 'volKey' && e.altKey) {
        st.removeVolumeKeyframe(special.clip.id, special.keyMs);
        return;
      }
      if (special.kind === 'volLine' && e.altKey) {
        const localMs = (x - msToX(special.clip.startMs)) / pxPerMs;
        st.setVolumeKeyframe(special.clip.id, localMs, volumeDbAt(special.clip.audio, localMs));
        return;
      }
      dragRef.current = {
        ...base,
        kind: special.kind,
        clipId: special.clip.id,
        origStart: special.clip.startMs,
        origEnd: clipEndMs(special.clip),
        rowY: b.y,
        rowH: b.h,
        ...(special.kind === 'fade' ? { edge: special.edge, target: special.target } : {}),
        ...(special.kind === 'volKey' ? { keyMs: special.keyMs, baseDb: special.clip.audio.gainDb } : {}),
        ...(special.kind === 'volLine' ? { baseDb: special.clip.audio.gainDb } : {}),
      };
      return;
    }
    const hit = hitClip(x, y);
    if (!hit) {
      dragRef.current = { ...base, kind: 'marquee' };
      return;
    }
    const sel = st.selection.clipIds;
    const inSel = sel.includes(hit.clip.id);
    let ids: string[];
    if (base.additive) {
      // Shift/Ctrl+click toggles the clip in the selection
      ids = inSel ? sel.filter((id) => id !== hit.clip.id) : [...sel, hit.clip.id];
      st.select(ids);
      if (inSel) return;
    } else if (inSel && !hit.edge) {
      ids = sel; // keep the multi-selection to move it together
    } else {
      ids = [hit.clip.id];
      st.select(ids);
    }
    if (hit.clip.effect) st.setInspectorTab('effect');
    else if (st.inspectorTab === 'effect' || st.inspectorTab === 'transition') st.setInspectorTab(hit.track.kind === 'audio' ? 'audio' : 'transform');
    if (hit.track.locked) return;
    const moving = new Set<string>();
    for (const id of hit.edge ? [hit.clip.id] : ids) {
      moving.add(id);
      const f = findClip(st.project, id);
      if (f) partnersOf(st.project, f.clip).forEach((p) => moving.add(p.id));
    }
    dragRef.current = {
      ...base,
      kind: hit.edge === 'in' ? 'trimIn' : hit.edge === 'out' ? 'trimOut' : 'move',
      clipId: hit.clip.id,
      ids: hit.edge ? [hit.clip.id] : ids,
      origStart: hit.clip.startMs,
      origEnd: clipEndMs(hit.clip),
      targets: snapTargetsFor(moving),
      selectOnClick: inSel && !hit.edge && !base.additive && sel.length > 1 ? hit.clip.id : null,
    };
  };

  /** Apply the drag for a pointer position (also called by edge auto-scroll). */
  const applyDrag = (x: number, y: number) => {
    const d = dragRef.current;
    if (!d) return;
    const st = useEditor.getState();
    const ms = xToMsLive(x);
    if (d.kind === 'scrub') {
      st.setPlayhead(Math.max(0, snapTo(ms, d.targets).ms));
      return;
    }
    if (d.kind === 'marquee') {
      marqueeRef.current = { x1: d.startX, y1: d.startY, x2: x, y2: y };
      drawOverlayNow();
      return;
    }
    if (d.started && !st.gestureBase) {
      // the gesture was cancelled (Escape)
      dragRef.current = null;
      guideRef.current = null;
      drawOverlayNow();
      return;
    }
    const delta = ms - d.startMs;
    const found = findClip(st.gestureBase ?? st.project, d.clipId!);
    if (!found) return;
    if (d.kind === 'fade') {
      const dur = d.origEnd - d.origStart;
      const v = Math.round(Math.max(0, Math.min(dur / 2, d.edge === 'in' ? ms - d.origStart : d.origEnd - ms)));
      if (d.target === 'audio') st.setClipAudio(d.clipId!, d.edge === 'in' ? { fadeInMs: v } : { fadeOutMs: v });
      else st.updateClip(d.clipId!, d.edge === 'in' ? { fadeInMs: v } : { fadeOutMs: v });
      return;
    }
    if (d.kind === 'volLine') {
      const dDb = yToDb(y, d.rowY!, d.rowH!) - yToDb(d.startY, d.rowY!, d.rowH!);
      st.setClipAudio(d.clipId!, { gainDb: Math.round(Math.max(VOL_MIN_DB, Math.min(VOL_MAX_DB, (d.baseDb ?? 0) + dDb)) * 10) / 10 });
      return;
    }
    if (d.kind === 'volKey') {
      const db = yToDb(y, d.rowY!, d.rowH!) - (d.baseDb ?? 0);
      st.setVolumeKeyframe(d.clipId!, d.keyMs!, Math.round(db * 10) / 10);
      return;
    }
    if (d.kind === 'move') {
      const dur = d.origEnd - d.origStart;
      const a = snapTo(d.origStart + delta, d.targets);
      const b = snapTo(d.origStart + delta + dur, d.targets);
      const useEnd = b.snapped && (!a.snapped || Math.abs(b.ms - (d.origStart + delta + dur)) < Math.abs(a.ms - (d.origStart + delta)));
      const start = useEnd ? b.ms - dur : a.ms;
      guideRef.current = a.snapped || b.snapped ? (useEnd ? b.ms : a.ms) : null;
      const row = rows.find((r) => y >= r.y && y < r.y + r.h && r.track.kind === found.track.kind);
      st.moveClips(d.ids, d.clipId!, Math.max(0, start), row?.track.id);
    } else if (d.kind === 'trimIn') {
      const edge = snapTo(d.origStart + delta, d.targets);
      guideRef.current = edge.snapped ? edge.ms : null;
      st.trimClip(d.clipId!, 'in', edge.ms - d.origStart);
    } else {
      const edge = snapTo(d.origEnd + delta, d.targets);
      guideRef.current = edge.snapped ? edge.ms : null;
      st.trimClip(d.clipId!, 'out', edge.ms - d.origEnd);
    }
    drawOverlayNow();
  };

  /** x -> ms with the live scroll (auto-scroll changes it during a drag). */
  const xToMsLive = (x: number) => {
    const s = useEditor.getState();
    return x / (s.zoom / 1000) + s.scrollMs;
  };

  // edge auto-scroll while dragging
  const autoRef = useRef<number | null>(null);
  const stopAuto = () => {
    if (autoRef.current != null) cancelAnimationFrame(autoRef.current);
    autoRef.current = null;
  };
  const runAuto = () => {
    const d = dragRef.current;
    if (!d || d.kind === 'marquee' || d.kind === 'volLine' || d.kind === 'volKey') return stopAuto();
    const w = viewRef.current.size.w;
    const x = d.lastX;
    const speed = x < AUTOSCROLL_EDGE_PX ? -(AUTOSCROLL_EDGE_PX - x) : x > w - AUTOSCROLL_EDGE_PX ? x - (w - AUTOSCROLL_EDGE_PX) : 0;
    if (!speed) return stopAuto();
    const s = useEditor.getState();
    s.setScroll(Math.max(0, s.scrollMs + (speed * 0.6) / (s.zoom / 1000)));
    applyDrag(d.lastX, d.lastY);
    autoRef.current = requestAnimationFrame(runAuto);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const { x, y } = local(e);
    const d = dragRef.current;
    if (!d) {
      const special = hitSpecial(x, y);
      if (special) {
        canvasRef.current!.style.cursor = special.kind === 'transition' ? 'pointer' : special.kind === 'fade' ? 'ew-resize' : e.altKey ? 'copy' : 'ns-resize';
        canvasRef.current!.title =
          special.kind === 'transition'
            ? special.clip.transitionIn
              ? 'Transition: click to edit, Delete removes it'
              : 'Cut: click to select it, then pick a transition'
            : special.kind === 'fade'
              ? `${special.target === 'audio' ? 'Audio' : 'Video'} fade ${special.edge === 'in' ? 'in' : 'out'}: drag`
              : special.kind === 'volKey'
                ? 'Volume keyframe: drag up / down · Alt+click removes it'
                : 'Volume: drag up / down · Alt+click adds a keyframe';
        return;
      }
      canvasRef.current!.title = '';
      const hit = hitClip(x, y);
      canvasRef.current!.style.cursor = hit ? (hit.track.locked ? 'not-allowed' : hit.edge ? 'ew-resize' : 'grab') : y < RULER_H + BEAT_H ? 'text' : 'default';
      return;
    }
    d.lastX = x;
    d.lastY = y;
    if (!d.started) {
      if (Math.hypot(x - d.startX, y - d.startY) < DRAG_THRESHOLD_PX) return;
      d.started = true;
      if (d.kind !== 'marquee') {
        useEditor.getState().beginGesture();
        canvasRef.current!.style.cursor = d.kind === 'move' ? 'grabbing' : d.kind === 'volLine' || d.kind === 'volKey' ? 'ns-resize' : 'ew-resize';
      }
    }
    applyDrag(x, y);
    if ((d.kind === 'move' || d.kind === 'trimIn' || d.kind === 'trimOut' || d.kind === 'fade') && autoRef.current == null && (x < AUTOSCROLL_EDGE_PX || x > viewRef.current.size.w - AUTOSCROLL_EDGE_PX)) {
      autoRef.current = requestAnimationFrame(runAuto);
    }
  };

  const onPointerUp = (e: React.PointerEvent) => {
    const d = dragRef.current;
    dragRef.current = null;
    stopAuto();
    if (canvasRef.current?.hasPointerCapture(e.pointerId)) canvasRef.current.releasePointerCapture(e.pointerId);
    const st = useEditor.getState();
    if (d?.kind === 'marquee') {
      if (!d.started) {
        // plain click on an empty area: deselect and move the playhead there
        if (!d.additive) {
          st.select([]);
          st.selectCut(null);
        }
        st.setPlayhead(Math.max(0, d.startMs));
      } else {
        const m = marqueeRef.current;
        if (m) {
          const t1 = xToMs(Math.min(m.x1, m.x2));
          const t2 = xToMs(Math.max(m.x1, m.x2));
          const y1 = Math.min(m.y1, m.y2);
          const y2 = Math.max(m.y1, m.y2);
          const hits: string[] = [];
          for (const r of rows) {
            if (r.y + r.h < y1 || r.y > y2) continue;
            for (const c of r.track.clips) if (clipEndMs(c) > t1 && c.startMs < t2) hits.push(c.id);
          }
          st.select(d.additive ? [...new Set([...st.selection.clipIds, ...hits])] : hits);
        }
      }
      marqueeRef.current = null;
    } else if (d && d.kind !== 'scrub') {
      if (d.started) st.endGesture();
      else if (d.selectOnClick) st.select([d.selectOnClick]);
    }
    guideRef.current = null;
    drawOverlayNow();
    if (canvasRef.current) canvasRef.current.style.cursor = 'default';
  };

  const onWheel = (e: React.WheelEvent) => {
    const st = useEditor.getState();
    if (e.ctrlKey || e.metaKey) {
      const { x } = local(e);
      const msAtCursor = xToMs(x);
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const next = Math.min(400, Math.max(0.5, zoom * factor));
      setZoom(next);
      setScroll(Math.max(0, msAtCursor - x / (next / 1000)));
    } else {
      const delta = (e.deltaY || e.deltaX) / pxPerMs;
      setScroll(Math.max(0, st.scrollMs + delta));
    }
  };

  const onDoubleClick = (e: React.MouseEvent) => {
    const { x, y } = local(e);
    const hit = hitClip(x, y);
    if (!hit) return;
    const st = useEditor.getState();
    st.select([hit.clip.id]);
    if (hit.clip.effect) {
      st.setInspectorTab('effect');
      return;
    }
    st.setInspectorTab(isConstantSpeed(hit.clip.speed) ? 'transform' : 'speed');
    st.setDrawer(true, isConstantSpeed(hit.clip.speed) ? 'keyframes' : 'speed');
  };

  const onContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    const { x, y } = local(e);
    const hit = hitClip(x, y);
    if (!hit) {
      setMenu(null);
      return;
    }
    const st = useEditor.getState();
    if (!st.selection.clipIds.includes(hit.clip.id)) st.select([hit.clip.id]);
    setMenu({ x: e.clientX, y: e.clientY, clipId: hit.clip.id, atMs: xToMs(x) });
  };

  /** The cut nearest to x on a video row (within 40 px), for dropped transitions. */
  const cutNear = (x: number, row: Row | undefined): Clip | null => {
    if (!row || row.track.kind !== 'video') return null;
    let best: Clip | null = null;
    let bestD = 40;
    for (const cut of cutsOfTrack(row.track, frameMs(useEditor.getState().project))) {
      const d = Math.abs(msToX(cut.atMs) - x);
      if (d < bestD) {
        bestD = d;
        best = cut.next;
      }
    }
    return best;
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const { x, y } = local(e);
    const row = rows.find((r) => y >= r.y && y < r.y + r.h);
    const st = useEditor.getState();
    const transition = e.dataTransfer.getData(TRANSITION_MIME);
    if (transition) {
      if (!isTransitionType(transition)) return;
      const cut = cutNear(x, row) ?? cutNear(x, rows.find((r) => r.track.kind === 'video'));
      if (!cut) {
        st.log('warn', 'Drop the transition on a cut between two clips of a video track.');
        return;
      }
      st.setTransition(cut.id, { type: transition, durationMs: cut.transitionIn?.durationMs ?? TRANSITION_DEFAULT_MS });
      st.selectCut(cut.id);
      return;
    }
    const effect = e.dataTransfer.getData(EFFECT_MIME);
    if (effect) {
      if (!isEffectType(effect)) return;
      const at = Math.max(0, snapTo(xToMs(x), snapTargetsFor(new Set())).ms);
      const clip = st.addEffect(effect as EffectType, at, row?.track.kind === 'fx' ? row.track.id : undefined);
      if (clip) {
        st.select([clip.id]);
        st.setInspectorTab('effect');
      }
      return;
    }
    const assetId = e.dataTransfer.getData('application/x-cappycat-asset');
    if (!assetId) return;
    const clip = st.addClipFromAsset(assetId, row?.track.id, Math.max(0, snapTo(xToMs(x), snapTargetsFor(new Set())).ms));
    if (clip) st.select([clip.id]);
  };

  // vertical resize handle (persisted)
  const onResizeStart = (e: React.PointerEvent) => {
    e.preventDefault();
    const startY = e.clientY;
    const startH = height;
    const move = (ev: PointerEvent) => useEditor.getState().setUi({ timelineHeight: Math.max(120, Math.min(700, startH - (ev.clientY - startY))) });
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  return (
    <div className="timeline" style={{ height }}>
      <div className="resize-handle" onPointerDown={onResizeStart} onDoubleClick={() => useEditor.getState().setUi({ timelineHeight: 0 })} title="Drag to resize · double-click: fit the tracks" />
      <div className="track-heads">
        <div className="track-head" style={{ height: RULER_H }}>
          <span className="ruler-head" title="Timeline zoom (Ctrl+wheel)">
            {zoom >= 10 ? Math.round(zoom) : zoom.toFixed(1)} px/s
          </span>
          <span style={{ flex: 1 }} />
          <button onClick={() => setZoom(zoom / 1.3)} title="Zoom out (Ctrl+-)" aria-label="Zoom out">
            −
          </button>
          <button onClick={() => setZoom(zoom * 1.3)} title="Zoom in (Ctrl+=)" aria-label="Zoom in">
            +
          </button>
          <button onClick={fit} title="Zoom to fit (Shift+Z)" aria-label="Zoom to fit">
            <FitIcon size={11} />
          </button>
        </div>
        <div className="track-head" style={{ height: BEAT_H, padding: '0 8px' }}>
          <span className="ruler-head">Beats · {project.beatMarkers.length}</span>
        </div>
        {rows.map(({ track, h }) => (
          <div className={`track-head ${track.muted ? 'muted' : ''}`} key={track.id} style={{ height: h }}>
            <span
              className="name"
              style={{ color: track.kind === 'video' ? 'var(--text)' : 'var(--text-dim)' }}
              title={track === rows.find((r) => r.track.kind === 'video')?.track ? `${track.name}: main track (magnetic when the Magnet is on)` : track.role ? `${track.name}: separated ${track.role} stems` : track.name}
            >
              {track === rows.find((r) => r.track.kind === 'video')?.track ? <span className="main-dot" /> : null}
              {track.role ? <span className={`role-dot ${track.role}`} /> : null}
              {track.name}
            </span>
            <button className={track.locked ? 'active' : ''} onClick={() => toggleTrackLock(track.id)} title={track.locked ? 'Unlock track' : 'Lock track'} aria-label="Lock track">
              {track.locked ? <LockIcon /> : <UnlockIcon />}
            </button>
            <button
              className={track.muted ? 'active' : ''}
              onClick={() => toggleTrackMute(track.id)}
              title={track.kind === 'audio' ? (track.muted ? 'Unmute track' : 'Mute track') : track.muted ? 'Show track' : 'Hide track'}
              aria-label="Mute or hide track"
            >
              {track.kind === 'audio' ? <SpeakerIcon off={track.muted} /> : <EyeIcon off={track.muted} />}
            </button>
          </div>
        ))}
      </div>
      <div
        className="canvas-wrap"
        ref={wrapRef}
        onDragOver={(e) => {
          e.preventDefault();
          e.dataTransfer.dropEffect = 'copy';
        }}
        onDrop={onDrop}
      >
        <canvas
          ref={canvasRef}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onWheel={onWheel}
          onDoubleClick={onDoubleClick}
          onContextMenu={onContextMenu}
        />
        <canvas ref={overlayRef} className="timeline-overlay" />
      </div>
      {menu ? <ClipContextMenu x={menu.x} y={menu.y} clipId={menu.clipId} atMs={menu.atMs} onClose={() => setMenu(null)} /> : null}
    </div>
  );
}

/* ------------------------------ drawing ------------------------------ */

interface DrawArgs {
  project: Project;
  rows: Row[];
  size: { w: number; h: number };
  msToX: (ms: number) => number;
  xToMs: (x: number) => number;
  selection: string[];
  selectedCut: string | null;
  zoom: number;
}

function niceStep(zoom: number): number {
  // choose a ruler step in ms so labels are >= 70px apart
  const candidates = [100, 250, 500, 1000, 2000, 5000, 10000, 15000, 30000, 60000, 120000, 300000, 600000];
  for (const c of candidates) if ((c / 1000) * zoom >= 70) return c;
  return 1200000;
}

function fmtRuler(ms: number): string {
  const s = ms / 1000;
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${String(m).padStart(2, '0')}:${r < 10 ? '0' : ''}${Number.isInteger(r) ? r : r.toFixed(1)}`;
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

/** Markers on cuts only when both clips are wide enough to see them. */
function markerVisible(prev: Clip, next: Clip, pxPerMs: number): boolean {
  return (clipEndMs(prev) - prev.startMs) * pxPerMs > 16 && (clipEndMs(next) - next.startMs) * pxPerMs > 16;
}

function drawBowTie(ctx: CanvasRenderingContext2D, x: number, y: number, fill: string, stroke: string | null) {
  ctx.beginPath();
  ctx.moveTo(x - 6, y - 6);
  ctx.lineTo(x, y);
  ctx.lineTo(x - 6, y + 6);
  ctx.closePath();
  ctx.moveTo(x + 6, y - 6);
  ctx.lineTo(x, y);
  ctx.lineTo(x + 6, y + 6);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

/** Fade ramp: the area above the ramp line is shaded, the ramp drawn, the handle on top. */
function drawFade(ctx: CanvasRenderingContext2D, edge: 'in' | 'out', xa: number, fadePx: number, yTop: number, yBottom: number, fromTop: boolean) {
  if (fadePx > 0.5) {
    const xb = edge === 'in' ? xa + fadePx : xa - fadePx;
    const yEdge = fromTop ? yBottom : yTop; // the ramp starts at silence / black
    const yFull = fromTop ? yTop : yBottom;
    ctx.beginPath();
    ctx.moveTo(xa, yFull);
    ctx.lineTo(xb, yFull);
    ctx.lineTo(xa, yEdge);
    ctx.closePath();
    ctx.fillStyle = THEME.fadeShade;
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(xa, yEdge);
    ctx.lineTo(xb, yFull);
    ctx.strokeStyle = THEME.fadeRamp;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

function draw(ctx: CanvasRenderingContext2D, a: DrawArgs) {
  const { project, rows, size, msToX, xToMs, selection, selectedCut, zoom } = a;
  const pxPerMs = zoom / 1000;
  ctx.fillStyle = THEME.bg1;
  ctx.fillRect(0, 0, size.w, size.h);

  const visStart = xToMs(0);
  const visEnd = xToMs(size.w);
  const selected = new Set(selection);
  const frame = frameMs(project);

  // ruler
  ctx.fillStyle = THEME.bg2;
  ctx.fillRect(0, 0, size.w, RULER_H);
  const step = niceStep(zoom);
  const minor = step / 5;
  ctx.font = '10px ui-monospace, Consolas, monospace';
  ctx.textBaseline = 'middle';
  ctx.beginPath();
  ctx.strokeStyle = THEME.line;
  for (let ms = Math.floor(visStart / minor) * minor; ms <= visEnd; ms += minor) {
    const x = Math.round(msToX(ms)) + 0.5;
    ctx.moveTo(x, RULER_H - 5);
    ctx.lineTo(x, RULER_H);
  }
  ctx.stroke();
  ctx.beginPath();
  ctx.strokeStyle = THEME.lineStrong;
  ctx.fillStyle = THEME.textDim;
  for (let ms = Math.floor(visStart / step) * step; ms <= visEnd; ms += step) {
    const x = Math.round(msToX(ms)) + 0.5;
    ctx.moveTo(x, RULER_H - 12);
    ctx.lineTo(x, RULER_H);
    ctx.fillText(fmtRuler(ms), x + 3, 8);
  }
  ctx.stroke();

  // beat track
  ctx.fillStyle = THEME.bg0;
  ctx.fillRect(0, RULER_H, size.w, BEAT_H);
  for (const b of project.beatMarkers) {
    if (b.timeMs < visStart || b.timeMs > visEnd) continue;
    const x = msToX(b.timeMs);
    ctx.fillStyle = b.kind === 'beat1' ? THEME.beatStrong : THEME.beatWeak;
    const r = b.kind === 'beat1' ? 3 : 2;
    ctx.beginPath();
    ctx.arc(x, RULER_H + BEAT_H / 2, r, 0, Math.PI * 2);
    ctx.fill();
  }

  const empty = rows.every((r) => r.track.clips.length === 0);

  // tracks
  for (const row of rows) {
    ctx.fillStyle = row.track.kind === 'video' ? THEME.bg2 : THEME.bg1;
    ctx.fillRect(0, row.y, size.w, row.h);
    ctx.strokeStyle = THEME.line;
    ctx.beginPath();
    ctx.moveTo(0, row.y + row.h - 0.5);
    ctx.lineTo(size.w, row.y + row.h - 0.5);
    ctx.stroke();

    for (const clip of row.track.clips) {
      const start = clip.startMs;
      const end = clipEndMs(clip);
      if (end < visStart || start > visEnd) continue;
      const x1 = msToX(start);
      const x2 = msToX(end);
      const w = Math.max(2, x2 - x1);
      const { y, h } = body(row);
      const isSel = selected.has(clip.id);
      const asset = assetOf(project, clip);
      const stem = asset?.stemOf?.stem;
      const base = clip.effect
        ? THEME.effectClip
        : row.track.kind === 'video'
          ? THEME.videoClip
          : row.track.kind === 'audio'
            ? stem
              ? THEME.stemClip
              : THEME.audioClip
            : THEME.fxClip;
      roundRect(ctx, x1, y, w, h, 4);
      ctx.fillStyle = base;
      ctx.fill();
      ctx.save();
      ctx.clip();
      // visible part of the clip only (long clips at high zoom)
      const vx1 = Math.max(x1, -2);
      const vx2 = Math.min(x1 + w, size.w + 2);
      const dur = end - start;

      // audio waveform (shaped by the gain, the volume keyframes and the fades)
      if (row.track.kind === 'audio') {
        const wave = asset ? waveCache.get(asset.path) : undefined;
        if (wave && wave !== 'loading' && asset) {
          ctx.fillStyle = stem === 'vocals' ? THEME.waveVoice : stem === 'background' ? THEME.waveBackground : THEME.waveform;
          const sps = 20;
          const srcSpan = clip.outMs - clip.inMs;
          const audioForWave = { ...clip.audio, muted: false };
          ctx.beginPath();
          for (let px = Math.max(0, Math.floor(vx1 - x1)); px < vx2 - x1; px += 2) {
            const frac = px / w;
            const srcMs = clip.reversed ? clip.outMs - frac * srcSpan : clip.inMs + frac * srcSpan;
            const idx = Math.min(wave.length - 1, Math.max(0, Math.floor((srcMs / 1000) * sps)));
            const v = Math.min(1, (wave[idx] ?? 0) * clipGainAt(audioForWave, frac * dur, dur));
            const bh = Math.max(1, v * (h - 6));
            ctx.rect(x1 + px, y + h / 2 - bh / 2, 1.5, bh);
          }
          ctx.fill();
        }
        if (clip.audio.muted) {
          ctx.fillStyle = 'rgba(8,9,11,0.6)';
          ctx.fillRect(x1, y, w, h);
        }
      }

      // effect clips: name + intensity bar
      if (clip.effect) {
        const info = effectInfo(clip.effect.type);
        ctx.fillStyle = THEME.effectClipEdge;
        ctx.fillRect(x1, y + h - 3, w * Math.max(0, Math.min(1, clip.effect.intensity)), 3);
        ctx.fillStyle = THEME.effectLabel;
        ctx.font = '11px system-ui, "Segoe UI", sans-serif';
        ctx.textBaseline = 'middle';
        ctx.fillText(`✦ ${clip.label && clip.label !== info.label ? clip.label : info.label}`, Math.max(x1, 0) + 6, y + h / 2);
      }

      // speed ramp strip
      if (row.track.kind === 'video' && !isConstantSpeed(clip.speed)) {
        ctx.fillStyle = 'rgba(0,0,0,0.35)';
        ctx.fillRect(x1, y + h - 13, w, 13);
        ctx.fillStyle = THEME.speedBadge;
        ctx.font = '10px ui-monospace, Consolas, monospace';
        ctx.textBaseline = 'middle';
        const uniform = clip.speed.points.every((p) => p.speed === clip.speed.points[0]?.speed);
        const label = uniform ? `${clip.speed.points[0]?.speed ?? 1}×` : clip.speed.preset === 'custom' ? 'CURVE' : clip.speed.preset.replace('_', ' ').toUpperCase();
        ctx.fillText(`${label}${clip.speed.opticalFlow ? ' · RAFT' : ''}`, vx1 + 5, y + h - 6);
      }
      if (row.track.kind === 'video' && clip.reframe) {
        ctx.fillStyle = THEME.reframeBar;
        ctx.fillRect(x1, y, w, 3);
      }
      if (clip.freezeFrame) {
        const fx = msToX(clip.startMs + clip.freezeFrame.atMs);
        const fw = clip.freezeFrame.holdMs * pxPerMs;
        ctx.fillStyle = THEME.freezeBand;
        ctx.fillRect(fx, y, fw, h);
      }

      // fades: video (top corners of video clips), audio (audio clips; bottom corners of video clips with their own sound)
      if (row.track.kind === 'video' && !clip.effect) {
        const half = dur / 2;
        drawFade(ctx, 'in', x1, Math.min(half, clip.fadeInMs ?? 0) * pxPerMs, y, y + h, true);
        drawFade(ctx, 'out', x2, Math.min(half, clip.fadeOutMs ?? 0) * pxPerMs, y, y + h, true);
        if (videoCarriesAudio(project, clip)) {
          const f = clampedAudioFades(clip.audio, dur);
          drawFade(ctx, 'in', x1, f.fadeIn * pxPerMs, y + h / 2, y + h, true);
          drawFade(ctx, 'out', x2, f.fadeOut * pxPerMs, y + h / 2, y + h, true);
        }
      } else if (row.track.kind === 'audio') {
        const f = clampedAudioFades(clip.audio, dur);
        drawFade(ctx, 'in', x1, f.fadeIn * pxPerMs, y, y + h, true);
        drawFade(ctx, 'out', x2, f.fadeOut * pxPerMs, y, y + h, true);
      }

      // label (kept in view when the clip starts off-screen)
      if (!clip.effect) {
        ctx.fillStyle = THEME.text;
        ctx.font = '11px system-ui, "Segoe UI", sans-serif';
        ctx.textBaseline = 'top';
        ctx.fillText((clip.reversed ? '◀ ' : '') + (clip.label ?? 'Clip'), Math.max(x1, 0) + 14, y + 4);
      }
      ctx.textBaseline = 'middle';

      // voice separation badge (top right): VOICE / NO VOCALS, dimmed until the stems exist; MUTED audio clips
      const badge = row.track.kind === 'audio' && clip.audio.muted ? 'MUTED' : row.track.kind !== 'fx' && !stem ? voiceBadge(voiceMode(clip)) : null;
      if (badge && w > 36) {
        ctx.font = 'bold 9px ui-monospace, Consolas, monospace';
        const tw = ctx.measureText(badge).width + 8;
        const bx = Math.max(x1 + 2, Math.min(size.w, x1 + w) - tw - 12);
        const ready = badge !== 'MUTED' && !!asset?.stems;
        roundRect(ctx, bx, y + 3, tw, 13, 3);
        ctx.fillStyle = ready ? THEME.voiceBadgeBg : THEME.voiceBadgePending;
        ctx.fill();
        ctx.fillStyle = ready ? THEME.voiceBadge : THEME.textDim;
        ctx.fillText(badge, bx + 4, y + 10);
      }

      // keyframe diamonds
      const kfs = new Set<number>();
      for (const key of ['position', 'scale', 'rotation', 'opacity', 'blur'] as const) {
        for (const k of clip.transform[key].keyframes) kfs.add(k.timeMs);
      }
      if (kfs.size && row.track.kind !== 'audio') {
        ctx.fillStyle = THEME.keyframe;
        ctx.beginPath();
        for (const t of kfs) {
          const kx = msToX(clip.startMs + t);
          ctx.moveTo(kx, y + h / 2 - 4);
          ctx.lineTo(kx + 4, y + h / 2);
          ctx.lineTo(kx, y + h / 2 + 4);
          ctx.lineTo(kx - 4, y + h / 2);
          ctx.closePath();
        }
        ctx.fill();
      }

      // volume line + keyframes (audio clips)
      if (row.track.kind === 'audio' && w > 12) {
        ctx.strokeStyle = THEME.volumeLine;
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        const keys = clip.audio.volume?.keyframes ?? [];
        if (!keys.length) {
          const ly = dbToY(clip.audio.gainDb, y, h);
          ctx.moveTo(vx1, ly);
          ctx.lineTo(vx2, ly);
        } else {
          for (let px = Math.max(0, Math.floor(vx1 - x1)); px <= Math.min(w, vx2 - x1); px += 3) {
            const ly = dbToY(levelDbAt(clip.audio, px / pxPerMs), y, h);
            if (px === Math.max(0, Math.floor(vx1 - x1))) ctx.moveTo(x1 + px, ly);
            else ctx.lineTo(x1 + px, ly);
          }
        }
        ctx.stroke();
        ctx.lineWidth = 1;
        for (const k of keys) {
          if (k.timeMs > dur + 1) continue;
          const kx = x1 + k.timeMs * pxPerMs;
          const ky = dbToY(clip.audio.gainDb + k.value, y, h);
          ctx.beginPath();
          ctx.arc(kx, ky, 3.2, 0, Math.PI * 2);
          ctx.fillStyle = THEME.volumeKey;
          ctx.fill();
          ctx.strokeStyle = THEME.bg0;
          ctx.stroke();
        }
      }

      // fade handles
      if (isSel || w > 60) {
        for (const hnd of fadeHandles(project, row.track, clip, x1, x2, y, h, pxPerMs)) {
          ctx.beginPath();
          ctx.arc(hnd.x, hnd.y, isSel ? HANDLE_R : 3, 0, Math.PI * 2);
          ctx.fillStyle = THEME.fadeHandle;
          ctx.globalAlpha = isSel ? 1 : 0.55;
          ctx.fill();
          ctx.globalAlpha = 1;
          ctx.strokeStyle = THEME.bg0;
          ctx.stroke();
        }
      }
      ctx.restore();

      roundRect(ctx, x1 + 0.5, y + 0.5, w - 1, h - 1, 4);
      ctx.strokeStyle = isSel ? THEME.accentBright : clip.effect ? THEME.effectClipEdge : clip.linkId ? 'rgba(255,255,255,0.16)' : 'rgba(255,255,255,0.1)';
      ctx.lineWidth = isSel ? 2 : 1;
      ctx.stroke();
      ctx.lineWidth = 1;
    }

    // transitions: the window band and a bow-tie on every cut
    if (row.track.kind === 'video') {
      const midY = row.y + row.h / 2;
      for (const cut of cutsOfTrack(row.track, frame)) {
        if (cut.atMs < visStart - 4000 || cut.atMs > visEnd + 4000) continue;
        if (!markerVisible(cut.prev, cut.next, pxPerMs)) continue;
        const cx = msToX(cut.atMs);
        const isSelCut = selectedCut === cut.next.id;
        if (cut.next.transitionIn) {
          const d = transitionDurationMs(row.track, cut.next, frame);
          const { y, h } = body(row);
          ctx.fillStyle = THEME.transitionBand;
          ctx.fillRect(msToX(cut.atMs - d / 2), y, d * pxPerMs, h);
          drawBowTie(ctx, cx, midY, isSelCut ? THEME.transitionMarkerSelected : THEME.transitionMarker, THEME.bg0);
        } else {
          drawBowTie(ctx, cx, midY, isSelCut ? THEME.transitionMarkerSelected : THEME.transitionMarkerIdle, null);
        }
        if (isSelCut) {
          ctx.strokeStyle = THEME.transitionMarkerSelected;
          ctx.strokeRect(cx - 9.5, midY - 9.5, 19, 19);
        }
      }
    }

    if (row.track.locked) {
      ctx.fillStyle = 'rgba(0,0,0,0.3)';
      ctx.fillRect(0, row.y, size.w, row.h);
    }
    if (row.track.muted) {
      ctx.fillStyle = 'rgba(8,9,11,0.45)';
      ctx.fillRect(0, row.y, size.w, row.h);
    }
  }

  if (empty) {
    const main = rows.find((r) => r.track.kind === 'video');
    if (main) {
      ctx.fillStyle = THEME.textFaint;
      ctx.font = '12px system-ui, "Segoe UI", sans-serif';
      ctx.textBaseline = 'middle';
      ctx.fillText('Drag clips here from the media library, double-click one there, or use “Rough cut in story order” in the AI tab.', 16, main.y + main.h / 2);
    }
  }

  // end-of-sequence marker
  const endX = Math.round(msToX(projectDurationMs(project))) + 0.5;
  ctx.strokeStyle = THEME.lineStrong;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(endX, RULER_H);
  ctx.lineTo(endX, size.h);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawOverlay(
  ctx: CanvasRenderingContext2D,
  size: { w: number; h: number },
  msToX: (ms: number) => number,
  playheadMs: number,
  guide: number | null,
  marquee: { x1: number; y1: number; x2: number; y2: number } | null,
) {
  if (guide != null) {
    const gx = Math.round(msToX(guide)) + 0.5;
    ctx.strokeStyle = THEME.snapGuide;
    ctx.beginPath();
    ctx.moveTo(gx, RULER_H);
    ctx.lineTo(gx, size.h);
    ctx.stroke();
  }
  if (marquee) {
    const x = Math.min(marquee.x1, marquee.x2);
    const y = Math.min(marquee.y1, marquee.y2);
    const w = Math.abs(marquee.x2 - marquee.x1);
    const h = Math.abs(marquee.y2 - marquee.y1);
    ctx.fillStyle = THEME.accentSoft;
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = THEME.accent;
    ctx.strokeRect(x + 0.5, y + 0.5, w, h);
  }
  const px = Math.round(msToX(playheadMs)) + 0.5;
  if (px < -8 || px > size.w + 8) return;
  ctx.strokeStyle = THEME.playhead;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(px, 0);
  ctx.lineTo(px, size.h);
  ctx.stroke();
  ctx.fillStyle = THEME.playhead;
  ctx.beginPath();
  ctx.moveTo(px - 6, 0);
  ctx.lineTo(px + 6, 0);
  ctx.lineTo(px, 8);
  ctx.closePath();
  ctx.fill();
  ctx.lineWidth = 1;
}

export { HEAD_W };
