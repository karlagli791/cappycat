/**
 * Voice separation helpers (CapCut "Isolate voice" / "Remove vocals"): which clips a mode applies
 * to, which stem file a clip plays, which assets still need `separate_audio`, and the per-asset
 * separation status shown in the Inspector. Pure functions (no store / React).
 */
import type { Asset, AssetStems, Clip, Project, Track, VoiceMode } from '@/types/project';

export const VOICE_MODES: Array<{ id: VoiceMode; label: string; hint: string }> = [
  { id: 'original', label: 'Original', hint: 'Unchanged audio' },
  { id: 'voice', label: 'Isolate voice', hint: 'Dialogue only: background, music and SFX removed' },
  { id: 'background', label: 'Remove vocals', hint: 'Ambience, music and SFX without the voices' },
];

/** Timeline badge text for a non-original mode. */
export function voiceBadge(mode: VoiceMode): string | null {
  return mode === 'voice' ? 'VOICE' : mode === 'background' ? 'NO VOCALS' : null;
}

export function voiceMode(clip: Clip): VoiceMode {
  const v = clip.audio.voice;
  return v === 'voice' || v === 'background' ? v : 'original';
}

export function normPath(p: string): string {
  return p.split('\\').join('/').toLowerCase();
}

export function assetHasSound(asset: Asset | undefined): boolean {
  if (!asset) return false;
  return asset.kind === 'audio' || (asset.kind === 'video' && asset.hasAudio);
}

function trackOf(p: Project, clip: Clip): Track | undefined {
  return p.tracks.find((t) => t.id === clip.trackId) ?? p.tracks.find((t) => t.clips.some((c) => c.id === clip.id));
}

/**
 * The linked partner of a clip: for a video-track clip, the audio-track clip with the same `linkId` and asset
 * (older projects without links: the same asset and start, what the exporter treats as its
 * mirrored audio); for an audio-track clip, that video clip.
 */
export function mirrorClip(p: Project, clip: Clip): Clip | null {
  const own = trackOf(p, clip);
  if (!own || own.kind === 'fx') return null;
  const want = own.kind === 'video' ? 'audio' : 'video';
  for (const t of p.tracks) {
    if (t.kind !== want) continue;
    // the mirror plays the same asset: stem clips of the group (other assets) are not mirrors
    const m = clip.linkId
      ? t.clips.find((c) => c.id !== clip.id && c.linkId === clip.linkId && c.assetId === clip.assetId)
      : t.clips.find((c) => c.id !== clip.id && !c.linkId && c.assetId === clip.assetId && Math.abs(c.startMs - clip.startMs) < 1);
    if (m) return m;
  }
  return null;
}

/** The mode that is heard for a video clip: its mirrored audio clip's mode when there is one
 *  (the exporter mixes the mirror, not the video clip), otherwise its own. */
export function effectiveVoiceMode(p: Project, clip: Clip): VoiceMode {
  const own = trackOf(p, clip);
  if (own?.kind === 'video') {
    const m = mirrorClip(p, clip);
    if (m) return voiceMode(m);
  }
  return voiceMode(clip);
}

/** Stem file for a mode, or null for `original` / not separated yet. */
export function stemFor(asset: Asset | undefined, mode: VoiceMode): string | null {
  if (!asset?.stems || mode === 'original') return null;
  return (mode === 'voice' ? asset.stems.vocals : asset.stems.background) || null;
}

/** Ids of the clip and its mirror (the pair a mode change applies to). */
export function linkedClipIds(p: Project, clip: Clip): string[] {
  const m = mirrorClip(p, clip);
  return m ? [clip.id, m.id] : [clip.id];
}

/** Clips on video / audio tracks whose asset carries sound. */
export function audioBearingClips(p: Project): Clip[] {
  const out: Clip[] = [];
  for (const t of p.tracks) {
    if (t.kind === 'fx') continue;
    for (const c of t.clips) if (assetHasSound(p.assets.find((a) => a.id === c.assetId))) out.push(c);
  }
  return out;
}

/** Set `mode` on the given clips (returns a new project; untouched tracks keep their identity). */
export function withVoiceMode(p: Project, clipIds: Iterable<string>, mode: VoiceMode): Project {
  const ids = new Set(clipIds);
  return {
    ...p,
    tracks: p.tracks.map((t) =>
      t.clips.some((c) => ids.has(c.id))
        ? { ...t, clips: t.clips.map((c) => (ids.has(c.id) ? { ...c, audio: { ...c.audio, voice: mode } } : c)) }
        : t,
    ),
  };
}

/** Assets (unique, with sound) used by `clips` that have no stems yet. */
export function assetsNeedingStems(p: Project, clips: Clip[]): Asset[] {
  const seen = new Set<string>();
  const out: Asset[] = [];
  for (const c of clips) {
    const a = p.assets.find((x) => x.id === c.assetId);
    if (!a || a.stems || !assetHasSound(a) || seen.has(normPath(a.path))) continue;
    seen.add(normPath(a.path));
    out.push(a);
  }
  return out;
}

/** Attach stems to every asset with this source path (ids may differ after an analysis merge). */
export function withStems(assets: Asset[], path: string, stems: AssetStems): Asset[] {
  const key = normPath(path);
  let changed = false;
  const next = assets.map((a) => {
    if (normPath(a.path) !== key) return a;
    changed = true;
    return { ...a, stems: { vocals: stems.vocals, background: stems.background } };
  });
  return changed ? next : assets;
}

/* ------------------------------------------------------------- separation jobs */

export interface SeparationJob {
  jobId: string;
  /** source paths in the order the pipeline processes them */
  paths: string[];
  /** overall progress 0..1 */
  pct: number;
  message: string;
  /** file being separated now and its own progress */
  clip: string | null;
  clipPct: number;
  /** normalised paths whose result arrived */
  done: string[];
}

export type SeparationStatus =
  | { kind: 'none' }
  | { kind: 'ready' }
  | { kind: 'queued'; jobId: string }
  | { kind: 'running'; jobId: string; pct: number }
  | { kind: 'error'; message: string };

function baseName(p: string): string {
  return p.split(/[\\/]/).pop() ?? p;
}

/** Status of one asset given the running jobs and the last errors (keyed by normalised path). */
export function separationStatus(
  asset: Asset | undefined,
  jobs: Record<string, SeparationJob>,
  errors: Record<string, string>,
): SeparationStatus {
  if (!asset) return { kind: 'none' };
  if (asset.stems) return { kind: 'ready' };
  const key = normPath(asset.path);
  for (const job of Object.values(jobs)) {
    const i = job.paths.findIndex((p) => normPath(p) === key);
    if (i < 0 || job.done.includes(key)) continue;
    const current = job.paths.findIndex((p) => !job.done.includes(normPath(p)) && baseName(p) === job.clip);
    if (current === i) return { kind: 'running', jobId: job.jobId, pct: job.clipPct };
    return { kind: 'queued', jobId: job.jobId };
  }
  const err = errors[key];
  return err ? { kind: 'error', message: err } : { kind: 'none' };
}

/** Paths among `paths` that are not already being separated by a running job. */
export function pathsNotRunning(paths: string[], jobs: Record<string, SeparationJob>): string[] {
  const running = new Set<string>();
  for (const j of Object.values(jobs)) for (const p of j.paths) if (!j.done.includes(normPath(p))) running.add(normPath(p));
  const seen = new Set<string>();
  return paths.filter((p) => {
    const k = normPath(p);
    if (running.has(k) || seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

/** Seconds of drift after which the preview re-seeks the stem element to the video's time. */
export const STEM_RESYNC_SEC = 0.08;

export function stemNeedsResync(stemTime: number, wantTime: number, threshold = STEM_RESYNC_SEC): boolean {
  return !Number.isFinite(stemTime) || Math.abs(stemTime - wantTime) > threshold;
}
