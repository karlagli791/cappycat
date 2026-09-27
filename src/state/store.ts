/**
 * Editor document store (Zustand). Holds the non-destructive Project document
 * plus transient UI state, with a bounded undo/redo edit stack.
 *
 * Undo model: every document mutation goes through `commit`. A commit that changes nothing is a
 * no-op. Continuous interactions (drags, sliders) run inside a *gesture*: `beginGesture()` snapshots
 * the project, commits during the gesture replace the document without touching the history, and
 * `endGesture()` pushes the snapshot once if the document changed — one undo step per gesture.
 * Timeline edit semantics (magnet, ripple, linked clips) live in ./edits.ts.
 */
import { useMemo } from 'react';
import { create } from 'zustand';
import type {
  AnalysisResult,
  Asset,
  AssetStems,
  BeatMarker,
  Clip,
  ClipAudio,
  ClipMask,
  ClipTransform,
  ColorGrade,
  ClipTransition,
  Easing,
  EffectType,
  FrameInterpolation,
  Keyframed,
  Project,
  SpeedCurve,
  Track,
  AdjustValues,
  UniversalPresetFile,
  VoiceMode,
} from '@/types/project';
import { defaultAudio, defaultColorGrade, defaultSpeed, defaultTransform, emptyProject } from '@/engine/defaults';
import { defaultEffect, effectInfo } from '@/engine/effects';
import { clampedAudioFades, VOLUME_KEYFRAMED_DEFAULT } from '@/engine/audio';
import { separateToTracks as separateToTracksEdit, type SeparateResult } from './stemTracks';
import { newId } from '@/engine/ids';
import { removeKeyframe, setKeyframe, type Interpolable } from '@/engine/keyframes';
import { audioBearingClips, linkedClipIds, normPath as voiceKey, withStems, withVoiceMode, type SeparationJob } from '@/engine/voice';
import * as E from './edits';
import { clipDurationMs, clipEndMs, findClip, sameDoc } from './edits';
import { playClock } from './clock';

export { clipDurationMs, clipEndMs, findClip, mainTrackId, partnersOf, inferLinks } from './edits';

export type InspectorTab = 'ai' | 'color' | 'speed' | 'transform' | 'audio' | 'mask' | 'effect' | 'transition';
export type KeyframeProperty = keyof ClipTransform;

export interface Selection {
  clipIds: string[];
  assetId: string | null;
}

export interface LogLine {
  ts: number;
  level: 'info' | 'warn' | 'error';
  message: string;
}

/** Parts of a clip that "Paste attributes" (Ctrl+Alt+V) can apply. */
export type AttrPart = 'color' | 'speed' | 'transform' | 'audio' | 'mask';
export interface AttrClipboard {
  sourceLabel: string;
  color: ColorGrade;
  speed: SpeedCurve;
  transform: ClipTransform;
  audio: ClipAudio;
  mask: ClipMask | null;
  blendMode: Clip['blendMode'];
}

export interface SaveStatus {
  state: 'idle' | 'saving' | 'saved' | 'error';
  /** last explicit save (ms since epoch) */
  savedAt: number | null;
  /** last autosave (ms since epoch) */
  autosavedAt: number | null;
  message?: string;
}

/** UI layout/preferences persisted in localStorage (not part of the document). */
export interface UiPrefs {
  zoom: number;
  timelineHeight: number;
  leftWidth: number;
  rightWidth: number;
  inspectorTab: InspectorTab;
  showGrid: boolean;
  showBoxes: boolean;
  snapping: boolean;
  rippleEdit: boolean;
}

export interface EditorState extends UiPrefs {
  project: Project;
  projectPath: string | null;
  dirty: boolean;
  past: Project[];
  future: Project[];
  /** drag/slider in progress: the document as it was when the gesture started */
  gestureBase: Project | null;

  /** throttled copy of the playhead (see ./clock.ts for the per-frame value) */
  playheadMs: number;
  playing: boolean;
  /** J/K/L shuttle: playback rate (1 normal, 2/4 fast forward, -1/-2/-4 reverse) */
  shuttle: number;
  loop: boolean;
  scrollMs: number;
  compareMode: boolean;
  /** large preview: side panels hidden, compact timeline */
  largePreview: boolean;
  selection: Selection;
  drawerOpen: boolean;
  drawerMode: 'keyframes' | 'speed';
  drawerProperty: KeyframeProperty;
  mediaServerUrl: string;
  logs: LogLine[];
  lastAnalysis: AnalysisResult | null;
  clipsFolder: string | null;
  clipsWarnings: string[];
  /** the saved universal preset (presets/universal-adjust.json) */
  universalPreset: UniversalPresetFile | null;
  universalPresetPath: string | null;
  /** running voice-separation jobs (`separate_audio`) by job id */
  separationJobs: Record<string, SeparationJob>;
  /** last separation error per normalised source path */
  separationErrors: Record<string, string>;
  attrClipboard: AttrClipboard | null;
  /** selected cut: the id of its INCOMING clip (transition marker / Transition section) */
  selectedCut: string | null;
  /** "Separate to tracks" requests waiting for their stems (clip ids) */
  pendingStemTracks: string[];
  /** bumped to ask the timeline to zoom-to-fit (after rough cut / analysis / open) */
  fitNonce: number;
  saveStatus: SaveStatus;

  // ---- document mutations (undoable) ----
  setProject(p: Project, opts?: { undoable?: boolean; path?: string | null }): void;
  /** Replace the document (open / restore): no history, clean, links inferred, UI reset. */
  loadDocument(p: Project, opts?: { path?: string | null; analysis?: AnalysisResult | null; dirty?: boolean }): void;
  newProject(name?: string): void;
  addAssets(assets: Asset[]): void;
  /** Merge a scanned clips folder (not undoable): new files are added, known files get the folder's order. */
  loadClipsFolder(folder: string, assets: Asset[], warnings: string[], opts?: { initial?: boolean }): void;
  /** Move a video asset one step earlier (-1) or later (+1) in the story order. */
  reorderAsset(assetId: string, dir: -1 | 1): void;
  /** Replace the video/audio tracks with every clip in story order (rough cut, no AI). */
  roughCutInStoryOrder(): void;
  removeAsset(assetId: string): void;
  addClipFromAsset(assetId: string, trackId?: string, atMs?: number): Clip | null;
  updateClip(clipId: string, patch: Partial<Clip> | ((c: Clip) => Partial<Clip>)): void;
  updateClipColor(clipId: string, patch: Partial<ColorGrade>): void;
  /** Set the whole grade on several clips (Apply to all / same source). */
  setClipsColor(clipIds: string[], grade: ColorGrade): void;
  /** Audio settings of a clip and its linked partner (the preview/export hear the audio-track mirror). */
  setClipAudio(clipId: string, patch: Partial<ClipAudio>): void;
  setClipSpeed(clipId: string, curve: SpeedCurve): void;
  /** Quick speed buttons: a constant speed on each clip (ripples; keeps each clip's optical-flow flag). */
  setConstantSpeed(clipIds: string[], speed: number): void;
  /** Volume keyframe (dB offset) at clip-local ms, on the clip and its audio mirror. */
  setVolumeKeyframe(clipId: string, localMs: number, db: number): void;
  removeVolumeKeyframe(clipId: string, localMs: number): void;
  /** Transition into `clipId` from the clip before it (null removes it). */
  setTransition(clipId: string, tr: ClipTransition | null): void;
  /** The same transition on every cut of the main track; returns the number of cuts. */
  applyTransitionToAllCuts(tr: ClipTransition): number;
  selectCut(clipId: string | null): void;
  /** Add an effect clip at `atMs` (default: the playhead) on an FX track with room (created if needed). */
  addEffect(type: EffectType, atMs?: number, trackId?: string): Clip | null;
  /** "Separate to tracks" for clips whose stems exist; the others are queued (see state/separation.ts). */
  separateToTracks(clipIds: string[]): SeparateResult;
  queueStemTracks(clipIds: string[]): void;
  /** Run the queued "Separate to tracks" requests whose stems arrived (drops failed ones). */
  runPendingStemTracks(failedPaths?: string[]): number;
  setProjectSettings(patch: { fps?: number; frameInterpolation?: FrameInterpolation }): void;
  setTransformKeyframe(clipId: string, prop: KeyframeProperty, timeMs: number, value: Interpolable, easing?: Easing, bezier?: [number, number, number, number]): void;
  removeTransformKeyframe(clipId: string, prop: KeyframeProperty, timeMs: number): void;
  setTransformStatic(clipId: string, prop: KeyframeProperty, value: Interpolable): void;
  moveClip(clipId: string, startMs: number, trackId?: string): void;
  /** Move clips together so `anchorId` starts at `anchorStart` (from the gesture snapshot while dragging). */
  moveClips(clipIds: string[], anchorId: string, anchorStart: number, trackId?: string): void;
  /** Trim an edge by `deltaMs` of timeline time (absolute from the gesture snapshot while dragging). */
  trimClip(clipId: string, edge: 'in' | 'out', deltaMs: number): void;
  splitClipAt(clipId: string, timelineMs: number): void;
  /** Ctrl+B: split the selected clips under the playhead, or every clip under it when none is selected. */
  splitAtPlayhead(): number;
  deleteClips(clipIds: string[]): void;
  freezeFrameAt(clipId: string, timelineMs: number, holdMs?: number): void;
  toggleReverse(clipId: string): void;
  toggleTrackLock(trackId: string): void;
  toggleTrackMute(trackId: string): void;
  setBeatMarkers(markers: BeatMarker[]): void;
  /** Voice separation mode of a clip and its linked (mirrored) clip; undoable. */
  setClipVoice(clipId: string, mode: VoiceMode): void;
  /** Set the mode on every clip with sound (video + audio tracks); returns how many changed. */
  setVoiceForAll(mode: VoiceMode): number;
  /** Stems arrived for a source file: attached to every asset with that path (kept across undo). */
  setAssetStems(path: string, stems: AssetStems): void;
  separationStarted(jobId: string, paths: string[]): void;
  separationProgress(jobId: string, pct: number, clipPct: number | null, clip: string | null, message: string): void;
  separationDone(jobId: string, ok: boolean, error: string | null): void;
  applyAnalysis(result: AnalysisResult): void;
  copyAttributes(clipId: string): boolean;
  pasteAttributes(clipIds: string[], parts: AttrPart[]): number;
  /** Copy a clip's grade to every video clip (or only to clips of the same source). */
  applyGradeToAll(clipId: string, sameSourceOnly?: boolean): number;
  undo(): void;
  redo(): void;
  beginGesture(): void;
  endGesture(): void;
  /** Abort a gesture and restore the snapshot (Escape while dragging). */
  cancelGesture(): void;

  // ---- transient ----
  setPlayhead(ms: number): void;
  /** Store copy of the playhead written by the playback loop (does not move the clock). */
  syncPlayhead(ms: number): void;
  setPlaying(playing: boolean): void;
  setShuttle(rate: number): void;
  setZoom(zoom: number): void;
  setScroll(ms: number): void;
  toggleSnapping(): void;
  toggleRipple(): void;
  toggleCompare(): void;
  toggleLargePreview(): void;
  toggleGrid(): void;
  toggleBoxes(): void;
  setUi(patch: Partial<Pick<UiPrefs, 'timelineHeight' | 'leftWidth' | 'rightWidth'>>): void;
  requestFit(): void;
  /** Universal adjust: loaded preset, per-project toggle, live edits, save as the new universal preset */
  setUniversalPreset(preset: UniversalPresetFile, path: string | null): void;
  setUniversalEnabled(enabled: boolean): void;
  setUniversalValues(values: AdjustValues): void;
  resetUniversalToPreset(): void;
  toggleLoop(): void;
  select(clipIds: string[], assetId?: string | null): void;
  selectAll(): void;
  setInspectorTab(tab: InspectorTab): void;
  setDrawer(open: boolean, mode?: 'keyframes' | 'speed', property?: KeyframeProperty): void;
  setMediaServerUrl(url: string): void;
  log(level: LogLine['level'], message: string): void;
  clearLogs(): void;
  setProjectPath(path: string | null): void;
  markSaved(): void;
  setSaveStatus(patch: Partial<SaveStatus>): void;
}

export const UNDO_LIMIT = 200;

const durationCache = new WeakMap<Project, number>();

export function projectDurationMs(p: Project): number {
  const hit = durationCache.get(p);
  if (hit !== undefined) return hit;
  let end = 0;
  for (const t of p.tracks) for (const c of t.clips) end = Math.max(end, clipEndMs(c));
  durationCache.set(p, end);
  return end;
}

export function assetOf(p: Project, clip: Clip): Asset | undefined {
  return p.assets.find((a) => a.id === clip.assetId);
}

/** Clips under a timeline time on tracks of `kind`; hidden (muted) tracks are skipped. */
export function clipsAt(p: Project, timelineMs: number, kind: Track['kind'] = 'video', includeMuted = false): Clip[] {
  const out: Clip[] = [];
  for (const t of p.tracks) {
    if (t.kind !== kind || (t.muted && !includeMuted)) continue;
    for (const c of t.clips) if (timelineMs >= c.startMs && timelineMs < clipEndMs(c)) out.push(c);
  }
  return out;
}

/** Sorted, de-duplicated cut points (clip starts/ends) of all tracks. */
export function cutPoints(p: Project): number[] {
  const set = new Set<number>([0]);
  for (const t of p.tracks) for (const c of t.clips) {
    set.add(Math.round(c.startMs));
    set.add(Math.round(clipEndMs(c)));
  }
  return [...set].sort((a, b) => a - b);
}

function clipDefaults() {
  return { speed: defaultSpeed(), transform: defaultTransform(), color: defaultColorGrade(), audio: defaultAudio() };
}

function trackEnd(track: Track): number {
  return track.clips.reduce((m, c) => Math.max(m, clipEndMs(c)), 0);
}

function mapClip(p: Project, clipId: string, fn: (c: Clip) => Clip): Project {
  return {
    ...p,
    tracks: p.tracks.map((t) => (t.clips.some((c) => c.id === clipId) ? { ...t, clips: t.clips.map((c) => (c.id === clipId ? fn(c) : c)) } : t)),
  };
}

function mapClipIds(p: Project, ids: Set<string>, fn: (c: Clip) => Clip): Project {
  return {
    ...p,
    tracks: p.tracks.map((t) => (t.clips.some((c) => ids.has(c.id)) ? { ...t, clips: t.clips.map((c) => (ids.has(c.id) ? fn(c) : c)) } : t)),
  };
}

function allClipIds(p: Project): Set<string> {
  const s = new Set<string>();
  for (const t of p.tracks) for (const c of t.clips) s.add(c.id);
  return s;
}

function pruneSelection(sel: Selection, p: Project): Selection {
  const ids = allClipIds(p);
  const clipIds = sel.clipIds.filter((id) => ids.has(id));
  const assetId = sel.assetId && p.assets.some((a) => a.id === sel.assetId) ? sel.assetId : null;
  return clipIds.length === sel.clipIds.length && assetId === sel.assetId ? sel : { clipIds, assetId };
}

/* ---------------------------------------------------------------- UI prefs */

const UI_KEY = 'cappycat:ui';
const DEFAULT_UI: UiPrefs = {
  zoom: 40,
  timelineHeight: 196,
  leftWidth: 250,
  rightWidth: 290,
  inspectorTab: 'ai',
  showGrid: true,
  showBoxes: true,
  snapping: true,
  rippleEdit: true,
};

function loadUi(): UiPrefs {
  try {
    const raw = typeof localStorage !== 'undefined' ? localStorage.getItem(UI_KEY) : null;
    if (!raw) return DEFAULT_UI;
    const v = JSON.parse(raw) as Partial<UiPrefs>;
    return { ...DEFAULT_UI, ...v };
  } catch {
    return DEFAULT_UI;
  }
}

export const useEditor = create<EditorState>((set, get) => {
  /** Apply a new document. No-op when nothing changed; inside a gesture the history is untouched. */
  const commit = (next: Project, undoable = true) => {
    const { project, past, gestureBase } = get();
    if (next === project) return;
    if (gestureBase) {
      set({ project: next, dirty: true });
      return;
    }
    if (sameDoc(project, next)) return;
    set({
      project: next,
      dirty: true,
      past: undoable ? [...past.slice(-UNDO_LIMIT + 1), project] : past,
      future: undoable ? [] : get().future,
    });
  };
  let dirtyAtGestureStart = false;
  const magnet = () => get().rippleEdit;
  /** Drags compute from the drag-start snapshot so they are idempotent. */
  const base = () => get().gestureBase ?? get().project;
  const finishGesture = () => {
    if (get().gestureBase) get().endGesture();
  };

  return {
    ...loadUi(),
    project: emptyProject('Ep_01'),
    projectPath: null,
    dirty: false,
    past: [],
    future: [],
    gestureBase: null,

    playheadMs: 0,
    playing: false,
    shuttle: 1,
    loop: false,
    scrollMs: 0,
    compareMode: false,
    largePreview: false,
    selection: { clipIds: [], assetId: null },
    drawerOpen: false,
    drawerMode: 'keyframes',
    drawerProperty: 'scale',
    mediaServerUrl: '',
    logs: [],
    lastAnalysis: null,
    clipsFolder: null,
    clipsWarnings: [],
    universalPreset: null,
    universalPresetPath: null,
    separationJobs: {},
    separationErrors: {},
    attrClipboard: null,
    selectedCut: null,
    pendingStemTracks: [],
    fitNonce: 0,
    saveStatus: { state: 'idle', savedAt: null, autosavedAt: null },

    setProject(p, opts) {
      if (opts?.undoable === false) {
        finishGesture();
        set((s) => ({ project: p, dirty: false, past: [], future: [], projectPath: opts.path ?? s.projectPath, selection: pruneSelection(s.selection, p) }));
      } else commit(p);
    },
    loadDocument(p, opts) {
      finishGesture();
      const preset = get().universalPreset;
      let doc = E.inferLinks(E.sanitizeProject(p));
      // older files without the setting adopt the saved universal preset, like new projects
      if (!doc.universalAdjust && preset) doc = { ...doc, universalAdjust: { enabled: preset.enabledByDefault, name: preset.name, values: preset.values } };
      set((s) => ({
        project: doc,
        projectPath: opts?.path ?? null,
        dirty: !!opts?.dirty,
        past: [],
        future: [],
        selection: { clipIds: [], assetId: null },
        selectedCut: null,
        pendingStemTracks: [],
        lastAnalysis: opts?.analysis ?? null,
        playing: false,
        shuttle: 1,
        scrollMs: 0,
        fitNonce: s.fitNonce + 1,
      }));
      get().setPlayhead(0);
    },
    newProject(name = 'Untitled') {
      finishGesture();
      const preset = get().universalPreset;
      const fresh = emptyProject(name);
      if (preset) fresh.universalAdjust = { enabled: preset.enabledByDefault, name: preset.name, values: preset.values };
      set({ project: fresh, projectPath: null, dirty: false, past: [], future: [], playing: false, selection: { clipIds: [], assetId: null }, lastAnalysis: null, scrollMs: 0 });
      get().setPlayhead(0);
    },
    addAssets(assets) {
      const p = get().project;
      const existing = new Set(p.assets.map((a) => a.path));
      const fresh = assets.filter((a) => !existing.has(a.path)).map((a, i) => ({ ...a, order: a.order ?? p.assets.length + i }));
      if (!fresh.length) return;
      commit({ ...p, assets: [...p.assets, ...fresh] });
    },
    loadClipsFolder(folder, assets, warnings, opts) {
      const p = get().project;
      const byPath = new Map(p.assets.map((a) => [normPath(a.path), a]));
      const merged: Asset[] = [...p.assets];
      for (const a of assets) {
        const existing = byPath.get(normPath(a.path));
        if (existing) {
          const i = merged.indexOf(existing);
          merged[i] = { ...existing, order: a.order, orderReason: a.orderReason };
        } else merged.push(a);
      }
      const assetsOrdered = renumberVideos(merged);
      const empty = p.tracks.every((t) => t.clips.length === 0);
      const first = storyOrder(assetsOrdered)[0];
      const format = empty && first && first.width > 0 ? { width: first.width, height: first.height, fps: snapFps(first.fps || p.fps) } : {};
      const next = { ...p, ...format, assets: assetsOrdered };
      // a folder scan describes the disk, not an edit: never undoable, and the boot scan keeps the project clean
      const changed = !sameDoc(p, next);
      const patch = (proj: Project): Project => ({ ...proj, assets: mergeAssets(proj.assets, assetsOrdered) });
      set((s) => ({
        project: next,
        past: changed ? s.past.map(patch) : s.past,
        future: changed ? s.future.map(patch) : s.future,
        dirty: s.dirty || (changed && !opts?.initial),
        clipsFolder: folder,
        clipsWarnings: warnings,
      }));
    },
    reorderAsset(assetId, dir) {
      const p = get().project;
      const videos = storyOrder(p.assets);
      const i = videos.findIndex((a) => a.id === assetId);
      const j = i + dir;
      if (i < 0 || j < 0 || j >= videos.length) return;
      [videos[i], videos[j]] = [videos[j], videos[i]];
      const newOrder = new Map(videos.map((a, k) => [a.id, k]));
      const moved = new Set([videos[i].id, videos[j].id]);
      commit({
        ...p,
        assets: p.assets.map((a) =>
          newOrder.has(a.id)
            ? { ...a, order: newOrder.get(a.id), orderReason: moved.has(a.id) ? 'moved manually' : a.orderReason }
            : a,
        ),
      });
    },
    roughCutInStoryOrder() {
      const p = get().project;
      const videoTrack = p.tracks.find((t) => t.kind === 'video');
      const audioTrack = p.tracks.find((t) => t.kind === 'audio');
      if (!videoTrack) return;
      const vclips: Clip[] = [];
      const aclips: Clip[] = [];
      let cursor = 0;
      storyOrder(p.assets).forEach((asset, i) => {
        const withAudio = !!audioTrack && asset.hasAudio;
        const linkId = withAudio ? newId('lnk') : undefined;
        const clip: Clip = { ...E.makeClip(asset, videoTrack.id, cursor, clipDefaults()), label: `${i + 1}. ${asset.name}`, ...(linkId ? { linkId } : {}) };
        vclips.push(clip);
        if (withAudio) aclips.push({ ...E.makeClip(asset, audioTrack!.id, cursor, clipDefaults()), label: `${i + 1}. ${asset.name} (audio)`, linkId });
        cursor += clipDurationMs(clip);
      });
      commit({
        ...p,
        tracks: p.tracks.map((t) =>
          t.id === videoTrack.id ? { ...t, clips: vclips } : audioTrack && t.id === audioTrack.id ? { ...t, clips: aclips } : t,
        ),
      });
      set((s) => ({ selection: { clipIds: [], assetId: null }, scrollMs: 0, fitNonce: s.fitNonce + 1 }));
      get().setPlayhead(0);
    },
    removeAsset(assetId) {
      const p = get().project;
      const ids = p.tracks.flatMap((t) => t.clips.filter((c) => c.assetId === assetId).map((c) => c.id));
      const cleared = ids.length ? E.deleteClips(p, ids, magnet()) : p;
      commit({ ...cleared, assets: cleared.assets.filter((a) => a.id !== assetId) });
      set((s) => ({ selection: pruneSelection(s.selection, get().project) }));
    },
    addClipFromAsset(assetId, trackId, atMs) {
      const p = get().project;
      const asset = p.assets.find((a) => a.id === assetId);
      if (!asset || asset.kind === 'lut') return null;
      const wantKind: Track['kind'] = asset.kind === 'audio' ? 'audio' : 'video';
      // a video dropped on the audio/FX row goes to the first video track (and vice versa)
      const given = trackId ? p.tracks.find((t) => t.id === trackId) : undefined;
      const track = given && given.kind === wantKind && !given.locked ? given : p.tracks.find((t) => t.kind === wantKind && !t.locked);
      if (!track) return null;
      const audioTrack = wantKind === 'video' && asset.hasAudio ? p.tracks.find((t) => t.kind === 'audio' && !t.locked) : undefined;
      const linkId = audioTrack ? newId('lnk') : undefined;
      const start = atMs ?? trackEnd(track);
      let clip: Clip = { ...E.makeClip(asset, track.id, start, clipDefaults()), ...(linkId ? { linkId } : {}) };
      const isMain = track.id === E.mainTrackId(p);
      let next: Project = {
        ...p,
        tracks: p.tracks.map((t) => {
          if (t.id !== track.id) return t;
          if (isMain && magnet() && atMs != null) return E.insertOrdered(t, clip, atMs);
          return { ...t, clips: [...t.clips, clip] };
        }),
      };
      clip = findClip(next, clip.id)!.clip;
      if (audioTrack) {
        const aclip: Clip = { ...E.makeClip(asset, audioTrack.id, clip.startMs, clipDefaults()), label: `${asset.name} (audio)`, linkId };
        next = { ...next, tracks: next.tracks.map((t) => (t.id === audioTrack.id ? { ...t, clips: [...t.clips, aclip] } : t)) };
      }
      commit(E.settle(next, magnet(), p));
      return findClip(get().project, clip.id)?.clip ?? clip;
    },
    updateClip(clipId, patch) {
      commit(mapClip(get().project, clipId, (c) => ({ ...c, ...(typeof patch === 'function' ? patch(c) : patch) })));
    },
    updateClipColor(clipId, patch) {
      commit(mapClip(get().project, clipId, (c) => ({ ...c, color: { ...c.color, ...patch } })));
    },
    setClipsColor(clipIds, grade) {
      commit(mapClipIds(get().project, new Set(clipIds), (c) => ({ ...c, color: grade })));
    },
    setClipAudio(clipId, patch) {
      const p = get().project;
      const found = findClip(p, clipId);
      if (!found) return;
      // the clip and its audio mirror (same asset); stem clips of the group keep their own settings
      const ids = new Set(E.audioMirrorIds(p, found.clip));
      commit(
        mapClipIds(p, ids, (c) => {
          const audio = { ...c.audio, ...patch };
          // fades are clamped to half the clip each
          if (patch.fadeInMs !== undefined || patch.fadeOutMs !== undefined) {
            const f = clampedAudioFades(audio, clipDurationMs(c));
            if (patch.fadeInMs !== undefined) audio.fadeInMs = Math.round(f.fadeIn);
            if (patch.fadeOutMs !== undefined) audio.fadeOutMs = Math.round(f.fadeOut);
          }
          return { ...c, audio };
        }),
      );
    },
    setClipSpeed(clipId, curve) {
      commit(E.setClipSpeed(get().project, clipId, curve, magnet()));
    },
    setConstantSpeed(clipIds, speed) {
      let p = get().project;
      const done = new Set<string>();
      for (const id of clipIds) {
        const f = findClip(p, id);
        if (!f || done.has(id) || f.clip.effect) continue;
        // a linked group changes once (through any member)
        E.withPartners(p, [id]).forEach((x) => done.add(x));
        p = E.setClipSpeed(p, id, E.constantSpeedCurve(speed, f.clip.speed.opticalFlow), magnet());
      }
      commit(p);
    },
    setVolumeKeyframe(clipId, localMs, db) {
      const p = get().project;
      const f = findClip(p, clipId);
      if (!f) return;
      const t = Math.max(0, Math.min(clipDurationMs(f.clip), localMs));
      commit(
        mapClipIds(p, new Set(E.audioMirrorIds(p, f.clip)), (c) => ({
          ...c,
          audio: { ...c.audio, volume: setKeyframe(c.audio.volume ?? VOLUME_KEYFRAMED_DEFAULT, t, Math.max(-60, Math.min(24, db)), 'linear') },
        })),
      );
    },
    removeVolumeKeyframe(clipId, localMs) {
      const p = get().project;
      const f = findClip(p, clipId);
      if (!f || !f.clip.audio.volume) return;
      commit(
        mapClipIds(p, new Set(E.audioMirrorIds(p, f.clip)), (c) => {
          if (!c.audio.volume) return c;
          const volume = removeKeyframe(c.audio.volume, localMs, 1);
          return { ...c, audio: { ...c.audio, volume } };
        }),
      );
    },
    setTransition(clipId, tr) {
      commit(E.setTransition(get().project, clipId, tr));
      if (!tr && get().selectedCut === clipId) set({ selectedCut: null });
    },
    applyTransitionToAllCuts(tr) {
      const { project, count } = E.setTransitionOnAllCuts(get().project, tr);
      commit(project);
      return count;
    },
    selectCut(clipId) {
      set({ selectedCut: clipId, ...(clipId ? { selection: { clipIds: [], assetId: null }, inspectorTab: 'transition' as InspectorTab } : {}) });
    },
    addEffect(type, atMs, trackId) {
      let p = get().project;
      const info = effectInfo(type);
      const start = Math.max(0, atMs ?? playClock.get());
      const end = start + info.defaultMs;
      const fxTracks = p.tracks.filter((t) => t.kind === 'fx' && !t.locked);
      const given = trackId ? fxTracks.find((t) => t.id === trackId) : undefined;
      let track = given && E.spanIsFree(given, start, end) ? given : fxTracks.find((t) => E.spanIsFree(t, start, end));
      if (!track) {
        // every FX track is taken there: a new one after the last FX track (CapCut stacks effects)
        const n = p.tracks.filter((t) => t.kind === 'fx').length;
        track = { id: newId('trk'), kind: 'fx', name: n ? `FX ${n + 1}` : 'FX', locked: false, muted: false, clips: [] };
        let at = -1;
        p.tracks.forEach((t, i) => {
          if (t.kind === 'fx' || t.kind === 'video') at = i;
        });
        const tracks = [...p.tracks];
        tracks.splice(at + 1, 0, track);
        p = { ...p, tracks };
      }
      const clip = E.makeEffectClip(defaultEffect(type), track.id, start, info.defaultMs, clipDefaults(), info.label);
      const tid = track.id;
      commit({ ...p, tracks: p.tracks.map((t) => (t.id === tid ? { ...t, clips: [...t.clips, clip] } : t)) });
      return findClip(get().project, clip.id)?.clip ?? null;
    },
    separateToTracks(clipIds) {
      const r = separateToTracksEdit(get().project, clipIds, magnet());
      commit(r.project);
      return r;
    },
    queueStemTracks(clipIds) {
      if (!clipIds.length) return;
      set((s) => ({ pendingStemTracks: [...new Set([...s.pendingStemTracks, ...clipIds])] }));
    },
    runPendingStemTracks(failedPaths) {
      const pending = get().pendingStemTracks;
      if (!pending.length) return 0;
      const p = get().project;
      const failed = new Set((failedPaths ?? []).map(voiceKey));
      const ready: string[] = [];
      const keep: string[] = [];
      for (const id of pending) {
        const f = findClip(p, id);
        if (!f) continue;
        const partnersAndSelf = [f.clip, ...E.partnersOf(p, f.clip)];
        const asset = partnersAndSelf.map((c) => assetOf(p, c)).find((a) => a && !a.stemOf);
        if (!asset) continue;
        if (asset.stems) ready.push(id);
        else if (!failed.has(voiceKey(asset.path))) keep.push(id);
      }
      set({ pendingStemTracks: keep });
      if (!ready.length) return 0;
      return get().separateToTracks(ready).done.length;
    },
    setProjectSettings(patch) {
      const p = get().project;
      const next: Project = { ...p };
      if (patch.fps !== undefined && patch.fps > 0) next.fps = patch.fps;
      if (patch.frameInterpolation !== undefined) next.frameInterpolation = patch.frameInterpolation;
      // the frame tolerance of cuts depends on the frame rate: re-validate the transitions
      commit(E.reconcileTransitions(next));
    },
    setTransformKeyframe(clipId, prop, timeMs, value, easing = 'easeInOut', bezier) {
      commit(
        mapClip(get().project, clipId, (c) => ({
          ...c,
          transform: {
            ...c.transform,
            // keyframes live inside the clip: clamp to its length
            [prop]: setKeyframe(c.transform[prop] as Keyframed<Interpolable>, Math.max(0, Math.min(clipDurationMs(c), timeMs)), value, easing, bezier),
          },
        })),
      );
    },
    removeTransformKeyframe(clipId, prop, timeMs) {
      commit(mapClip(get().project, clipId, (c) => ({ ...c, transform: { ...c.transform, [prop]: removeKeyframe(c.transform[prop] as Keyframed<Interpolable>, timeMs) } })));
    },
    setTransformStatic(clipId, prop, value) {
      commit(mapClip(get().project, clipId, (c) => ({ ...c, transform: { ...c.transform, [prop]: { ...c.transform[prop], static: value } } })));
    },
    moveClip(clipId, startMs, trackId) {
      commit(E.moveClips(base(), [clipId], clipId, startMs, trackId, magnet()));
    },
    moveClips(clipIds, anchorId, anchorStart, trackId) {
      commit(E.moveClips(base(), clipIds, anchorId, anchorStart, trackId, magnet()));
    },
    trimClip(clipId, edge, deltaMs) {
      commit(E.trimClip(base(), clipId, edge, deltaMs, magnet()));
    },
    splitClipAt(clipId, timelineMs) {
      const { project, rightIds } = E.splitClips(get().project, [clipId], timelineMs, magnet());
      if (!rightIds.length) return;
      commit(project);
      const video = rightIds.filter((id) => findClip(project, id)?.track.kind === 'video');
      set({ selection: { clipIds: video.length ? video : rightIds.slice(0, 1), assetId: null } });
    },
    splitAtPlayhead() {
      const s = get();
      const t = playClock.get();
      const under = (id: string) => {
        const f = findClip(s.project, id);
        return !!f && t > f.clip.startMs && t < clipEndMs(f.clip);
      };
      let ids = s.selection.clipIds.filter(under);
      if (!ids.length) {
        // nothing selected under the playhead: every visible clip there (partners follow)
        ids = s.project.tracks.filter((tr) => !tr.locked).flatMap((tr) => tr.clips.filter((c) => under(c.id)).map((c) => c.id));
      }
      if (!ids.length) return 0;
      const { project, rightIds } = E.splitClips(s.project, ids, t, magnet());
      if (!rightIds.length) return 0;
      commit(project);
      if (s.selection.clipIds.length) set({ selection: { clipIds: rightIds.filter((id) => findClip(project, id)?.track.kind === 'video'), assetId: null } });
      return rightIds.length;
    },
    deleteClips(clipIds) {
      if (!clipIds.length) return;
      commit(E.deleteClips(get().project, clipIds, magnet()));
      set((s) => ({
        selection: pruneSelection({ clipIds: [], assetId: s.selection.assetId }, get().project),
        selectedCut: s.selectedCut && findClip(get().project, s.selectedCut) ? s.selectedCut : null,
      }));
    },
    freezeFrameAt(clipId, timelineMs, holdMs = 1000) {
      commit(E.freezeFrameAt(get().project, clipId, timelineMs, holdMs, magnet()));
    },
    toggleReverse(clipId) {
      commit(E.toggleReverse(get().project, clipId));
    },
    toggleTrackLock(trackId) {
      const p = get().project;
      commit({ ...p, tracks: p.tracks.map((t) => (t.id === trackId ? { ...t, locked: !t.locked } : t)) });
    },
    toggleTrackMute(trackId) {
      const p = get().project;
      commit({ ...p, tracks: p.tracks.map((t) => (t.id === trackId ? { ...t, muted: !t.muted } : t)) });
    },
    setBeatMarkers(markers) {
      commit({ ...get().project, beatMarkers: markers });
    },
    setClipVoice(clipId, mode) {
      const p = get().project;
      const found = findClip(p, clipId);
      if (!found) return;
      commit(withVoiceMode(p, linkedClipIds(p, found.clip), mode));
    },
    setVoiceForAll(mode) {
      const p = get().project;
      const ids = audioBearingClips(p)
        .filter((c) => (c.audio.voice ?? 'original') !== mode)
        .map((c) => c.id);
      if (ids.length) commit(withVoiceMode(p, ids, mode));
      return ids.length;
    },
    setAssetStems(path, stems) {
      const patch = (proj: Project): Project => {
        const assets = withStems(proj.assets, path, stems);
        return assets === proj.assets ? proj : { ...proj, assets };
      };
      const key = voiceKey(path);
      const { [key]: _cleared, ...errors } = get().separationErrors;
      const jobs = Object.fromEntries(
        Object.entries(get().separationJobs).map(([id, j]) => [id, j.paths.some((x) => voiceKey(x) === key) && !j.done.includes(key) ? { ...j, done: [...j.done, key] } : j]),
      );
      // stems describe the file, not an edit: patch the undo history (and a running gesture) too
      set((st) => ({
        project: patch(st.project),
        past: st.past.map(patch),
        future: st.future.map(patch),
        gestureBase: st.gestureBase ? patch(st.gestureBase) : null,
        dirty: true,
        separationErrors: errors,
        separationJobs: jobs,
      }));
    },
    separationStarted(jobId, paths) {
      const errors = { ...get().separationErrors };
      for (const x of paths) delete errors[voiceKey(x)];
      set((st) => ({
        separationErrors: errors,
        separationJobs: { ...st.separationJobs, [jobId]: { jobId, paths, pct: 0, message: 'starting', clip: null, clipPct: 0, done: [] } },
      }));
    },
    separationProgress(jobId, pct, clipPct, clip, message) {
      const job = get().separationJobs[jobId];
      if (!job) return;
      set((st) => ({
        separationJobs: { ...st.separationJobs, [jobId]: { ...job, pct, clipPct: clipPct ?? job.clipPct, clip: clip ?? job.clip, message } },
      }));
    },
    separationDone(jobId, ok, error) {
      const job = get().separationJobs[jobId];
      if (!job) return;
      const { [jobId]: _finished, ...jobs } = get().separationJobs;
      const errors = { ...get().separationErrors };
      if (!ok && error !== 'cancelled') {
        for (const x of job.paths) if (!job.done.includes(voiceKey(x))) errors[voiceKey(x)] = error ?? 'separation failed';
      }
      set({ separationJobs: jobs, separationErrors: errors });
    },
    applyAnalysis(result) {
      finishGesture();
      const cur = get().project;
      const tl = E.sanitizeProject(result.timeline);
      // keep the current project identity but take assets/tracks/markers from the assembled timeline
      const next: Project = E.inferLinks({
        ...cur, // keeps the project's universalAdjust setting
        fps: tl.fps || cur.fps,
        width: tl.width || cur.width,
        height: tl.height || cur.height,
        assets: mergeAssets(cur.assets, tl.assets),
        tracks: tl.tracks.length ? tl.tracks : cur.tracks,
        beatMarkers: tl.beatMarkers ?? [],
      });
      commit(next);
      set((s) => ({ lastAnalysis: result, selection: { clipIds: [], assetId: null }, scrollMs: 0, fitNonce: s.fitNonce + 1 }));
      get().setPlayhead(0);
    },
    copyAttributes(clipId) {
      const f = findClip(get().project, clipId);
      if (!f) return false;
      const c = f.clip;
      set({ attrClipboard: { sourceLabel: c.label ?? 'clip', color: c.color, speed: c.speed, transform: c.transform, audio: c.audio, mask: c.mask, blendMode: c.blendMode } });
      return true;
    },
    pasteAttributes(clipIds, parts) {
      const cb = get().attrClipboard;
      if (!cb || !clipIds.length || !parts.length) return 0;
      let p = get().project;
      const want = new Set(parts);
      let n = 0;
      for (const id of clipIds) {
        const f = findClip(p, id);
        if (!f || f.track.locked) continue;
        n++;
        if (want.has('speed')) p = E.setClipSpeed(p, id, cb.speed, magnet());
        p = mapClip(p, id, (c) => ({
          ...c,
          ...(want.has('color') ? { color: cb.color } : {}),
          ...(want.has('transform') ? { transform: cb.transform, blendMode: cb.blendMode } : {}),
          ...(want.has('mask') ? { mask: cb.mask } : {}),
        }));
        if (want.has('audio')) {
          const ids = new Set(E.audioMirrorIds(p, findClip(p, id)!.clip));
          p = mapClipIds(p, ids, (c) => ({ ...c, audio: { ...cb.audio } }));
        }
      }
      commit(p);
      return n;
    },
    applyGradeToAll(clipId, sameSourceOnly = false) {
      const p = get().project;
      const f = findClip(p, clipId);
      if (!f) return 0;
      const ids = p.tracks
        .filter((t) => t.kind === 'video' && !t.locked)
        .flatMap((t) => t.clips)
        .filter((c) => c.id !== clipId && (!sameSourceOnly || c.assetId === f.clip.assetId))
        .map((c) => c.id);
      if (ids.length) commit(mapClipIds(p, new Set(ids), (c) => ({ ...c, color: f.clip.color })));
      return ids.length;
    },
    undo() {
      finishGesture();
      const { past, project, future } = get();
      if (!past.length) return;
      const prev = past[past.length - 1];
      set((s) => ({ project: prev, past: past.slice(0, -1), future: [project, ...future].slice(0, UNDO_LIMIT), dirty: true, selection: pruneSelection(s.selection, prev) }));
    },
    redo() {
      finishGesture();
      const { past, project, future } = get();
      if (!future.length) return;
      const next = future[0];
      set((s) => ({ project: next, past: [...past, project].slice(-UNDO_LIMIT), future: future.slice(1), dirty: true, selection: pruneSelection(s.selection, next) }));
    },
    beginGesture() {
      if (get().gestureBase) return;
      dirtyAtGestureStart = get().dirty;
      set({ gestureBase: get().project });
    },
    endGesture() {
      const { gestureBase, project, past } = get();
      if (!gestureBase) return;
      if (gestureBase === project || sameDoc(gestureBase, project)) {
        set({ gestureBase: null, project: gestureBase, dirty: dirtyAtGestureStart });
        return;
      }
      set({ gestureBase: null, past: [...past.slice(-UNDO_LIMIT + 1), gestureBase], future: [], dirty: true });
    },
    cancelGesture() {
      const { gestureBase } = get();
      if (!gestureBase) return;
      set((s) => ({ gestureBase: null, project: gestureBase, dirty: dirtyAtGestureStart, selection: pruneSelection(s.selection, gestureBase) }));
    },

    setPlayhead(ms) {
      const v = Math.max(0, ms);
      playClock.set(v);
      if (get().playheadMs !== v) set({ playheadMs: v });
    },
    syncPlayhead(ms) {
      const v = Math.max(0, ms);
      if (get().playheadMs !== v) set({ playheadMs: v });
    },
    setPlaying(playing) {
      if (playing === get().playing && (playing || get().shuttle === 1)) return;
      set({ playing, ...(playing ? {} : { shuttle: 1 }) });
      if (!playing) get().syncPlayhead(playClock.get());
    },
    setShuttle(rate) {
      if (rate === 0) {
        set({ playing: false, shuttle: 1 });
        get().syncPlayhead(playClock.get());
      } else set({ playing: true, shuttle: rate });
    },
    setZoom(zoom) {
      set({ zoom: Math.min(400, Math.max(0.5, zoom)) });
    },
    setScroll(ms) {
      set({ scrollMs: Math.max(0, ms) });
    },
    toggleSnapping() {
      set((s) => ({ snapping: !s.snapping }));
    },
    toggleRipple() {
      set((s) => ({ rippleEdit: !s.rippleEdit }));
    },
    toggleCompare() {
      set((s) => ({ compareMode: !s.compareMode }));
    },
    toggleLargePreview() {
      set((s) => ({ largePreview: !s.largePreview }));
    },
    toggleGrid() {
      set((s) => ({ showGrid: !s.showGrid }));
    },
    toggleBoxes() {
      set((s) => ({ showBoxes: !s.showBoxes }));
    },
    setUi(patch) {
      set(patch);
    },
    requestFit() {
      set((s) => ({ fitNonce: s.fitNonce + 1 }));
    },
    setUniversalPreset(preset, path) {
      set({ universalPreset: preset, universalPresetPath: path });
      const p = get().project;
      // every project gets the universal preset unless it already has its own setting
      if (!p.universalAdjust) {
        set({ project: { ...p, universalAdjust: { enabled: preset.enabledByDefault, name: preset.name, values: preset.values } } });
      }
    },
    setUniversalEnabled(enabled) {
      const p = get().project;
      const preset = get().universalPreset;
      const cur = p.universalAdjust ?? { enabled, name: preset?.name ?? 'Universal adjust', values: preset?.values ?? {} };
      commit({ ...p, universalAdjust: { ...cur, enabled } });
    },
    setUniversalValues(values) {
      const p = get().project;
      const cur = p.universalAdjust ?? { enabled: true, name: 'Universal adjust', values: {} };
      commit({ ...p, universalAdjust: { ...cur, values } });
    },
    resetUniversalToPreset() {
      const p = get().project;
      const preset = get().universalPreset;
      if (!preset) return;
      commit({ ...p, universalAdjust: { enabled: p.universalAdjust?.enabled ?? preset.enabledByDefault, name: preset.name, values: preset.values } });
    },
    toggleLoop() {
      set((s) => ({ loop: !s.loop }));
    },
    select(clipIds, assetId = null) {
      set((s) => ({ selection: { clipIds, assetId }, selectedCut: clipIds.length || assetId ? null : s.selectedCut }));
    },
    selectAll() {
      set((s) => ({ selection: { clipIds: s.project.tracks.filter((t) => !t.locked).flatMap((t) => t.clips.map((c) => c.id)), assetId: null } }));
    },
    setInspectorTab(tab) {
      set({ inspectorTab: tab });
    },
    setDrawer(open, mode, property) {
      set((s) => ({ drawerOpen: open, drawerMode: mode ?? s.drawerMode, drawerProperty: property ?? s.drawerProperty }));
    },
    setMediaServerUrl(url) {
      set({ mediaServerUrl: url });
    },
    log(level, message) {
      set((s) => ({ logs: [...s.logs.slice(-499), { ts: Date.now(), level, message }] }));
    },
    clearLogs() {
      set({ logs: [] });
    },
    setProjectPath(path) {
      set({ projectPath: path });
    },
    markSaved() {
      set((s) => ({ dirty: false, saveStatus: { ...s.saveStatus, state: 'saved', savedAt: Date.now(), message: undefined } }));
    },
    setSaveStatus(patch) {
      set((s) => ({ saveStatus: { ...s.saveStatus, ...patch } }));
    },
  };
});

// persist layout / preferences (debounced)
if (typeof window !== 'undefined') {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let last = '';
  useEditor.subscribe((s) => {
    const prefs: UiPrefs = {
      zoom: s.zoom,
      timelineHeight: s.timelineHeight,
      leftWidth: s.leftWidth,
      rightWidth: s.rightWidth,
      inspectorTab: s.inspectorTab,
      showGrid: s.showGrid,
      showBoxes: s.showBoxes,
      snapping: s.snapping,
      rippleEdit: s.rippleEdit,
    };
    const json = JSON.stringify(prefs);
    if (json === last) return;
    last = json;
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      try {
        localStorage.setItem(UI_KEY, json);
      } catch {
        /* storage unavailable */
      }
    }, 400);
  });
}

const STANDARD_FPS = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60];

/** Snap to the nearest standard frame rate when within 0.5 % (24.04 -> 24). */
export function snapFps(fps: number): number {
  let best: number | null = null;
  for (const s of STANDARD_FPS) {
    const d = Math.abs(fps - s) / s;
    if (d < 0.005 && (best === null || d < Math.abs(fps - best) / best)) best = s;
  }
  return best ?? Math.round(fps * 1000) / 1000;
}

function normPath(p: string): string {
  return p.split('\\').join('/').toLowerCase();
}

/** Video assets in story order (order, then natural name). */
export function storyOrder(assets: Asset[]): Asset[] {
  return assets
    .filter((a) => a.kind === 'video')
    .sort((a, b) => (a.order ?? 1e9) - (b.order ?? 1e9) || a.name.localeCompare(b.name, undefined, { numeric: true }));
}

/** Give video assets consecutive order numbers following their current story order. */
function renumberVideos(assets: Asset[]): Asset[] {
  const ids = new Map(storyOrder(assets).map((a, i) => [a.id, i]));
  return assets.map((a) => (ids.has(a.id) ? { ...a, order: ids.get(a.id) } : a));
}

/** Merge by normalised path (the pipeline writes '/' separators, the native scanner '\\').
 *  Incoming fields win (timeline clips reference the incoming ids); an existing manual
 *  order and reason are kept when the incoming asset has none. */
export function mergeAssets(current: Asset[], incoming: Asset[]): Asset[] {
  const byPath = new Map(current.map((a) => [normPath(a.path), a]));
  for (const a of incoming) {
    const key = normPath(a.path);
    const prev = byPath.get(key);
    byPath.set(key, {
      ...prev,
      ...a,
      order: a.order ?? prev?.order,
      orderReason: a.orderReason ?? prev?.orderReason,
      sceneTags: a.sceneTags ?? prev?.sceneTags,
      stems: a.stems ?? prev?.stems,
    });
  }
  return [...byPath.values()];
}

/** Selected clip helper for components (memoised so the snapshot stays referentially stable). */
export function useSelectedClip(): { clip: Clip; track: Track; asset: Asset | undefined } | null {
  const project = useEditor((s) => s.project);
  const id = useEditor((s) => s.selection.clipIds[0] ?? null);
  return useMemo(() => {
    if (!id) return null;
    const found = findClip(project, id);
    if (!found) return null;
    return { ...found, asset: assetOf(project, found.clip) };
  }, [project, id]);
}

/** Every path the media server must serve for a project (assets, LUTs, stems). */
export function mediaPaths(p: Project): string[] {
  const out = new Set<string>();
  for (const a of p.assets) {
    if (a.path) out.add(a.path);
    if (a.stems?.vocals) out.add(a.stems.vocals);
    if (a.stems?.background) out.add(a.stems.background);
  }
  return [...out];
}
