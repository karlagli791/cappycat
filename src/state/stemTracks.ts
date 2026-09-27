/**
 * "Separate to tracks" (FEATURES_V2 §5), as a pure edit: for each clip whose source has stems,
 *  1. register one audio asset per stem (`stemOf`, name "<source> · Voice" / "<source> · Background",
 *     path = the stem file, durationMs = the source's duration) — reused when it already exists;
 *  2. create the audio tracks "Voice" / "Background" (`role`) when missing (another track with the
 *     same role when the span is taken, e.g. by stems of an overlapping upper video track);
 *  3. add one clip per stem aligned with the source clip (startMs, inMs, outMs, speed, reversed,
 *     freezeFrame) in the SAME link group as the video clip; the stems inherit the original audio's
 *     gain, normalize, keep-pitch, fades and volume keyframes, and a stem the old voice mode did not
 *     play starts muted (so the result sounds exactly as before);
 *  4. mute the original linked audio clip (or the video clip itself when it has no audio mirror).
 * Clips whose source has no stems yet are reported in `missing` (the caller runs `separate_audio`).
 */
import type { Asset, Clip, ClipAudio, Project, StemKind, Track, TrackRole } from '@/types/project';
import { defaultColorGrade, defaultTransform } from '@/engine/defaults';
import { newId } from '@/engine/ids';
import { assetHasSound, normPath, voiceMode } from '@/engine/voice';
import { clipEndMs, findClip, partnersOf, settle, spanIsFree } from './edits';

export const STEM_KINDS: StemKind[] = ['vocals', 'background'];
const ROLE_OF: Record<StemKind, TrackRole> = { vocals: 'voice', background: 'background' };
const LABEL_OF: Record<StemKind, string> = { vocals: 'Voice', background: 'Background' };

function baseName(name: string): string {
  return name.replace(/\.[^.\\/]+$/, '');
}

export function stemAssetName(source: Asset, stem: StemKind): string {
  return `${baseName(source.name)} · ${LABEL_OF[stem]}`;
}

/** The stem asset registered for `source` (by `stemOf`), if any. */
export function stemAssetOf(p: Project, sourceId: string, stem: StemKind): Asset | undefined {
  return p.assets.find((a) => a.stemOf?.assetId === sourceId && a.stemOf.stem === stem);
}

function ensureStemAsset(p: Project, source: Asset, stem: StemKind): { project: Project; asset: Asset } {
  const path = stem === 'vocals' ? source.stems!.vocals : source.stems!.background;
  const existing = stemAssetOf(p, source.id, stem);
  if (existing) {
    if (normPath(existing.path) === normPath(path)) return { project: p, asset: existing };
    // re-separated with another model / cache dir: point the asset at the new file
    const asset = { ...existing, path };
    return { project: { ...p, assets: p.assets.map((a) => (a.id === existing.id ? asset : a)) }, asset };
  }
  const asset: Asset = {
    id: newId('ast'),
    path,
    name: stemAssetName(source, stem),
    kind: 'audio',
    durationMs: source.durationMs,
    width: 0,
    height: 0,
    fps: 0,
    hasAudio: true,
    stemOf: { assetId: source.id, stem },
  };
  return { project: { ...p, assets: [...p.assets, asset] }, asset };
}

/** A track with `role` where [start, end) is free; created (after the last audio track) when needed. */
function ensureRoleTrack(p: Project, role: TrackRole, start: number, end: number): { project: Project; track: Track } {
  const candidates = p.tracks.filter((t) => t.kind === 'audio' && t.role === role && !t.locked);
  const free = candidates.find((t) => spanIsFree(t, start, end));
  if (free) return { project: p, track: free };
  const n = p.tracks.filter((t) => t.role === role).length;
  const track: Track = { id: newId('trk'), kind: 'audio', name: n ? `${LABEL_OF[role === 'voice' ? 'vocals' : 'background']} ${n + 1}` : LABEL_OF[role === 'voice' ? 'vocals' : 'background'], locked: false, muted: false, clips: [], role };
  // after the last audio track (Voice before Background)
  let at = -1;
  p.tracks.forEach((t, i) => {
    if (t.kind === 'audio' && (role === 'background' || t.role !== 'background')) at = i;
  });
  const tracks = [...p.tracks];
  tracks.splice(at + 1, 0, track);
  return { project: { ...p, tracks }, track };
}

export interface SeparateResult {
  project: Project;
  /** clip ids (as given) that now have stem tracks */
  done: string[];
  /** clip ids skipped because their source has no stems yet */
  waiting: string[];
  /** unique source assets that need `separate_audio` */
  missing: Asset[];
}

/** The member of a link group that anchors the stems: the video clip, else the (audio) clip itself. */
function anchorOf(p: Project, clip: Clip, track: Track): Clip {
  if (track.kind === 'video') return clip;
  const v = partnersOf(p, clip).find((c) => findClip(p, c.id)?.track.kind === 'video');
  return v ?? clip;
}

/** Can this clip be separated to tracks (a video / audio clip with sound that is not itself a stem)? */
export function canSeparateToTracks(p: Project, clipId: string): boolean {
  const f = findClip(p, clipId);
  if (!f || f.track.kind === 'fx' || f.clip.effect) return false;
  const anchor = anchorOf(p, f.clip, f.track);
  const asset = p.assets.find((a) => a.id === anchor.assetId);
  return !!asset && !asset.stemOf && assetHasSound(asset);
}

/** Does the link group of this clip already carry stem clips of its source? */
export function hasStemClips(p: Project, clipId: string): boolean {
  const f = findClip(p, clipId);
  if (!f) return false;
  const anchor = anchorOf(p, f.clip, f.track);
  return [anchor, ...partnersOf(p, anchor)].some((c) => p.assets.find((a) => a.id === c.assetId)?.stemOf?.assetId === anchor.assetId);
}

export function separateToTracks(base: Project, clipIds: string[], magnet: boolean): SeparateResult {
  let p = base;
  const done: string[] = [];
  const waiting: string[] = [];
  const missing = new Map<string, Asset>();
  const handledAnchors = new Set<string>();
  for (const id of clipIds) {
    const f = findClip(p, id);
    if (!f || f.track.kind === 'fx' || f.clip.effect) continue;
    const anchor = anchorOf(p, f.clip, f.track);
    if (handledAnchors.has(anchor.id)) {
      done.push(id);
      continue;
    }
    const source = p.assets.find((a) => a.id === anchor.assetId);
    if (!source || source.stemOf || !assetHasSound(source)) continue;
    if (!source.stems) {
      waiting.push(id);
      missing.set(normPath(source.path), source);
      continue;
    }
    handledAnchors.add(anchor.id);
    const group = [anchor, ...partnersOf(p, anchor)];
    const existingStems = new Set(group.map((c) => p.assets.find((a) => a.id === c.assetId)?.stemOf).filter((s) => s?.assetId === source.id).map((s) => s!.stem));
    // the original audio: the mirrored audio clip (same asset), or the anchor itself
    const mirror = group.find((c) => c.id !== anchor.id && c.assetId === anchor.assetId && findClip(p, c.id)?.track.kind === 'audio');
    const orig = mirror ?? anchor;
    const linkId = anchor.linkId ?? newId('lnk');
    const mode = voiceMode(orig);
    const start = anchor.startMs;
    const end = clipEndMs(anchor);
    const added: Clip[] = [];
    for (const stem of STEM_KINDS) {
      if (existingStems.has(stem)) continue;
      const a = ensureStemAsset(p, source, stem);
      p = a.project;
      const t = ensureRoleTrack(p, ROLE_OF[stem], start, end);
      p = t.project;
      const silent = orig.audio.muted || (mode === 'voice' && stem === 'background') || (mode === 'background' && stem === 'vocals');
      const audio: ClipAudio = {
        gainDb: orig.audio.gainDb,
        normalize: orig.audio.normalize,
        muted: silent,
        voice: 'original',
        keepPitch: orig.audio.keepPitch !== false,
        ...(orig.audio.fadeInMs ? { fadeInMs: orig.audio.fadeInMs } : {}),
        ...(orig.audio.fadeOutMs ? { fadeOutMs: orig.audio.fadeOutMs } : {}),
        ...(orig.audio.volume ? { volume: orig.audio.volume } : {}),
      };
      const clip: Clip = {
        id: newId('clp'),
        assetId: a.asset.id,
        trackId: t.track.id,
        startMs: start,
        inMs: anchor.inMs,
        outMs: anchor.outMs,
        speed: anchor.speed,
        transform: defaultTransform(),
        color: defaultColorGrade(),
        audio,
        mask: null,
        blendMode: 'normal',
        reframe: null,
        label: `${anchor.label ?? baseName(source.name)} · ${LABEL_OF[stem]}`,
        freezeFrame: anchor.freezeFrame,
        reversed: anchor.reversed,
        linkId,
      };
      const trackId = t.track.id;
      p = { ...p, tracks: p.tracks.map((tr) => (tr.id === trackId ? { ...tr, clips: [...tr.clips, clip] } : tr)) };
      added.push(clip);
    }
    // link the anchor (and its mirror) and mute the original audio
    p = {
      ...p,
      tracks: p.tracks.map((tr) =>
        tr.clips.some((c) => c.id === anchor.id || c.id === orig.id)
          ? {
              ...tr,
              clips: tr.clips.map((c) => {
                if (c.id !== anchor.id && c.id !== orig.id) return c;
                let x = c.linkId === linkId ? c : { ...c, linkId };
                if (c.id === orig.id && added.length && !x.audio.muted) x = { ...x, audio: { ...x.audio, muted: true } };
                return x;
              }),
            }
          : tr,
      ),
    };
    done.push(id);
  }
  return { project: done.length ? settle(p, magnet, base) : base, done, waiting, missing: [...missing.values()] };
}

/** Clips "Apply to all clips" separates: video clips with sound, and audio clips with no video partner. */
export function separableClipIds(p: Project): string[] {
  const out: string[] = [];
  for (const t of p.tracks) {
    if (t.kind === 'fx' || t.locked) continue;
    for (const c of t.clips) {
      if (t.kind === 'audio' && partnersOf(p, c).some((x) => findClip(p, x.id)?.track.kind === 'video')) continue;
      if (canSeparateToTracks(p, c.id) && !hasStemClips(p, c.id)) out.push(c.id);
    }
  }
  return out;
}
