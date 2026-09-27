import { beforeEach, describe, expect, it } from 'vitest';
import type { Asset, Clip, Project, Track } from '@/types/project';
import { defaultAudio, defaultColorGrade, defaultSpeed, defaultTransform } from './defaults';
import {
  assetsNeedingStems,
  audioBearingClips,
  effectiveVoiceMode,
  linkedClipIds,
  mirrorClip,
  pathsNotRunning,
  separationStatus,
  stemFor,
  stemNeedsResync,
  voiceBadge,
  voiceMode,
  withStems,
  withVoiceMode,
  type SeparationJob,
} from './voice';
import { useEditor } from '@/state/store';

const asset = (id: string, path: string, extra: Partial<Asset> = {}): Asset => ({
  id,
  path,
  name: path.split(/[\\/]/).pop()!,
  kind: 'video',
  durationMs: 15000,
  width: 1280,
  height: 720,
  fps: 24,
  hasAudio: true,
  ...extra,
});

const clip = (id: string, assetId: string, trackId: string, startMs: number, voice?: Clip['audio']['voice']): Clip => ({
  id,
  assetId,
  trackId,
  startMs,
  inMs: 0,
  outMs: 5000,
  speed: defaultSpeed(),
  transform: defaultTransform(),
  color: defaultColorGrade(),
  audio: { ...defaultAudio(), ...(voice ? { voice } : {}) },
  mask: null,
  blendMode: 'normal',
  reframe: null,
  freezeFrame: null,
  reversed: false,
});

const STEMS = { vocals: 'C:/cache/stems/a/vocals.wav', background: 'C:/cache/stems/a/background.wav' };

function project(): Project {
  const tracks: Track[] = [
    { id: 'v', kind: 'video', name: 'Video 1', locked: false, muted: false, clips: [clip('v1', 'a', 'v', 0), clip('v2', 'b', 'v', 5000), clip('v3', 'img', 'v', 10000)] },
    { id: 'fx', kind: 'fx', name: 'FX', locked: false, muted: false, clips: [] },
    { id: 'au', kind: 'audio', name: 'Audio 1', locked: false, muted: false, clips: [clip('a1', 'a', 'au', 0)] },
  ];
  return {
    version: 1,
    id: 'p',
    name: 'p',
    fps: 24,
    width: 1280,
    height: 720,
    assets: [asset('a', 'C:\\clips\\clip1.mp4'), asset('b', 'C:/clips/clip2.mp4'), asset('img', 'C:/clips/still.png', { kind: 'image', hasAudio: false })],
    tracks,
    beatMarkers: [],
  };
}

describe('voice mode helpers', () => {
  it('defaults to original and labels non-original modes', () => {
    const p = project();
    expect(voiceMode(p.tracks[0].clips[0])).toBe('original');
    expect(voiceMode({ ...p.tracks[0].clips[0], audio: { ...defaultAudio(), voice: undefined } })).toBe('original');
    expect(voiceBadge('voice')).toBe('VOICE');
    expect(voiceBadge('background')).toBe('NO VOCALS');
    expect(voiceBadge('original')).toBeNull();
  });

  it('finds the mirrored clip in both directions and applies modes to the pair', () => {
    const p = project();
    const [v1, v2] = p.tracks[0].clips;
    const a1 = p.tracks[2].clips[0];
    expect(mirrorClip(p, v1)?.id).toBe('a1');
    expect(mirrorClip(p, a1)?.id).toBe('v1');
    expect(mirrorClip(p, v2)).toBeNull();
    expect(linkedClipIds(p, v1)).toEqual(['v1', 'a1']);
    const next = withVoiceMode(p, linkedClipIds(p, v1), 'voice');
    expect(voiceMode(next.tracks[0].clips[0])).toBe('voice');
    expect(voiceMode(next.tracks[2].clips[0])).toBe('voice');
    expect(voiceMode(next.tracks[0].clips[1])).toBe('original');
    expect(next.tracks[1]).toBe(p.tracks[1]); // untouched tracks keep identity
  });

  it('the heard mode of a video clip is its mirror mode (like the exporter)', () => {
    const p = withVoiceMode(withVoiceMode(project(), ['v1'], 'background'), ['a1'], 'voice');
    expect(effectiveVoiceMode(p, p.tracks[0].clips[0])).toBe('voice');
    expect(effectiveVoiceMode(p, p.tracks[2].clips[0])).toBe('voice');
    const q = withVoiceMode(project(), ['v2'], 'background');
    expect(effectiveVoiceMode(q, q.tracks[0].clips[1])).toBe('background');
  });

  it('picks the stem for a mode only when the asset is separated', () => {
    const a = asset('a', 'C:/x.mp4');
    expect(stemFor(a, 'voice')).toBeNull();
    const s = { ...a, stems: STEMS };
    expect(stemFor(s, 'voice')).toBe(STEMS.vocals);
    expect(stemFor(s, 'background')).toBe(STEMS.background);
    expect(stemFor(s, 'original')).toBeNull();
    expect(stemFor(undefined, 'voice')).toBeNull();
  });

  it('lists audio-bearing clips and the assets that still need stems', () => {
    const p = project();
    expect(audioBearingClips(p).map((c) => c.id)).toEqual(['v1', 'v2', 'a1']);
    expect(assetsNeedingStems(p, audioBearingClips(p)).map((a) => a.id)).toEqual(['a', 'b']);
    const withA = { ...p, assets: withStems(p.assets, 'c:/clips/CLIP1.mp4', STEMS) };
    expect(withA.assets[0].stems).toEqual(STEMS); // matched by normalised path
    expect(assetsNeedingStems(withA, audioBearingClips(withA)).map((a) => a.id)).toEqual(['b']);
    expect(withStems(p.assets, 'C:/elsewhere.mp4', STEMS)).toBe(p.assets);
  });
});

describe('separation status', () => {
  const job = (extra: Partial<SeparationJob> = {}): SeparationJob => ({
    jobId: 'j1',
    paths: ['C:\\clips\\clip1.mp4', 'C:/clips/clip2.mp4'],
    pct: 0.25,
    message: '',
    clip: 'clip1.mp4',
    clipPct: 0.5,
    done: [],
    ...extra,
  });

  it('reports running / queued / ready / error / none', () => {
    const [a, b] = project().assets;
    expect(separationStatus(a, { j1: job() }, {})).toEqual({ kind: 'running', jobId: 'j1', pct: 0.5 });
    expect(separationStatus(b, { j1: job() }, {})).toEqual({ kind: 'queued', jobId: 'j1' });
    expect(separationStatus(b, { j1: job({ clip: 'clip2.mp4', clipPct: 0.43, done: ['c:/clips/clip1.mp4'] }) }, {})).toEqual({
      kind: 'running',
      jobId: 'j1',
      pct: 0.43,
    });
    expect(separationStatus({ ...a, stems: STEMS }, { j1: job() }, {})).toEqual({ kind: 'ready' });
    expect(separationStatus(a, {}, { 'c:/clips/clip1.mp4': 'boom' })).toEqual({ kind: 'error', message: 'boom' });
    expect(separationStatus(a, {}, {})).toEqual({ kind: 'none' });
  });

  it('does not start duplicate work', () => {
    expect(pathsNotRunning(['C:/clips/clip1.mp4', 'C:/clips/clip3.mp4', 'c:\\clips\\clip3.mp4'], { j1: job() })).toEqual(['C:/clips/clip3.mp4']);
    expect(pathsNotRunning(['C:/clips/clip1.mp4'], { j1: job({ done: ['c:/clips/clip1.mp4'] }) })).toEqual(['C:/clips/clip1.mp4']);
  });

  it('re-seeks the stem only beyond 80 ms of drift', () => {
    expect(stemNeedsResync(1.0, 1.05)).toBe(false);
    expect(stemNeedsResync(1.0, 1.09)).toBe(true);
    expect(stemNeedsResync(NaN, 1)).toBe(true);
  });
});

describe('store voice actions', () => {
  beforeEach(() => {
    useEditor.setState({ project: project(), past: [], future: [], separationJobs: {}, separationErrors: {} });
  });

  it('setClipVoice updates the clip and its mirror in one undo step', () => {
    const st = useEditor.getState();
    st.setClipVoice('v1', 'background');
    const p = useEditor.getState().project;
    expect(voiceMode(p.tracks[0].clips[0])).toBe('background');
    expect(voiceMode(p.tracks[2].clips[0])).toBe('background');
    useEditor.getState().undo();
    expect(voiceMode(useEditor.getState().project.tracks[2].clips[0])).toBe('original');
  });

  it('setVoiceForAll touches every clip with sound', () => {
    expect(useEditor.getState().setVoiceForAll('voice')).toBe(3);
    const p = useEditor.getState().project;
    expect(p.tracks.flatMap((t) => t.clips).filter((c) => voiceMode(c) === 'voice').map((c) => c.id)).toEqual(['v1', 'v2', 'a1']);
    expect(useEditor.getState().setVoiceForAll('voice')).toBe(0);
  });

  it('tracks a separation job and keeps stems across undo', () => {
    const st = useEditor.getState();
    st.setClipVoice('v1', 'voice'); // something to undo
    st.separationStarted('j1', ['C:\\clips\\clip1.mp4', 'C:/clips/clip2.mp4']);
    st.separationProgress('j1', 0.2, 0.4, 'clip1.mp4', 'clip1.mp4: segment 2/5');
    let s = useEditor.getState();
    expect(separationStatus(s.project.assets[0], s.separationJobs, s.separationErrors)).toEqual({ kind: 'running', jobId: 'j1', pct: 0.4 });
    s.setAssetStems('C:/clips/clip1.mp4', STEMS);
    s = useEditor.getState();
    expect(s.project.assets[0].stems).toEqual(STEMS);
    expect(s.separationJobs.j1.done).toEqual(['c:/clips/clip1.mp4']);
    s.undo();
    expect(useEditor.getState().project.assets[0].stems).toEqual(STEMS);
    useEditor.getState().separationDone('j1', false, 'clip2.mp4: separation failed');
    s = useEditor.getState();
    expect(s.separationJobs).toEqual({});
    expect(separationStatus(s.project.assets[1], s.separationJobs, s.separationErrors)).toEqual({ kind: 'error', message: 'clip2.mp4: separation failed' });
    expect(separationStatus(s.project.assets[0], s.separationJobs, s.separationErrors)).toEqual({ kind: 'ready' });
  });
});
