/**
 * Thin bridge over Tauri `invoke`/`listen`. When the app runs in a plain
 * browser (vite dev without Tauri) every command falls back to a mock so the
 * UI stays usable for layout work.
 */
import type {
  AnalysisResult,
  Asset,
  AssetStems,
  Keyframed,
  PipelineEvent,
  PipelineOptions,
  Project,
  SpeedCurve,
  UniversalPresetFile,
} from '@/types/project';
import { defaultUniversalValues } from '@/engine/grade';
import type { Lut3D } from '@/engine/color/lut';

export const isTauri = (): boolean =>
  typeof window !== 'undefined' && ('__TAURI_INTERNALS__' in window || '__TAURI__' in window);

type Unlisten = () => void;

async function tauriInvoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke } = await import('@tauri-apps/api/core');
  return invoke<T>(cmd, args);
}

async function tauriListen<T>(event: string, handler: (payload: T) => void): Promise<Unlisten> {
  const { listen } = await import('@tauri-apps/api/event');
  return listen<T>(event, (e) => handler(e.payload));
}

/* ------------------------------- mocks ------------------------------- */

const mockListeners = new Map<string, Set<(p: unknown) => void>>();
/** browser mode: one fake "GPU" shared by the mock analysis and separation (GPU queue demo) */
let mockGpuBusyUntil = 0;
const GPU_WAIT_MESSAGE = 'waiting for the GPU (another AI job is running)';
function mockEmit(event: string, payload: unknown) {
  mockListeners.get(event)?.forEach((h) => h(payload));
}

function mockAsset(path: string, i: number): Asset {
  const name = path.split(/[\\/]/).pop() ?? path;
  return {
    id: `ast_mock${i}_${Math.random().toString(36).slice(2, 6)}`,
    path,
    name,
    kind: /\.(cube)$/i.test(name) ? 'lut' : /\.(mp3|wav|aac|flac|m4a)$/i.test(name) ? 'audio' : 'video',
    durationMs: 15000 + ((i * 7919) % 35000),
    width: 1920,
    height: 1080,
    fps: 24,
    hasAudio: true,
    codec: 'h264',
    order: i,
  };
}

/* ------------------------------ commands ----------------------------- */

export interface AppPaths {
  cacheDir: string;
  thumbsDir?: string;
  analysisDir?: string;
  ffmpeg: string | null;
  ffprobe: string | null;
  python: string | null;
  pipelineDir: string | null;
  clipsDir?: string | null;
  mediaServerUrl?: string;
  /** app logs (installed: %LOCALAPPDATA%/Cappycat/logs) */
  logsDir?: string | null;
}

/** `setup_status`: what the AI setup has installed (FEATURES_V2 §9). */
export interface SetupStatus {
  ffmpeg: boolean;
  python: boolean;
  models: boolean;
  /** NVIDIA GPU name, null when none was detected (CPU wheels) */
  gpu: string | null;
  /** running as the installed app (not from the repo) */
  installed: boolean;
}

/** `setup_plan`: what `setup_ai` will do, shown (with sources and sizes) before it starts. */
export interface SetupPlanStep {
  step: string;
  url?: string;
  sizeBytes?: number;
  note: string;
}

export interface SetupProgressPayload {
  jobId: string;
  step: string;
  /** progress of `step`, 0..1 */
  pct: number;
  message: string;
}

export interface SetupDonePayload {
  jobId: string;
  ok: boolean;
  error: string | null;
}

/** Browser mode: `?installed` makes the mock behave like a fresh install (wizard on first run). */
function mockInstalled(): boolean {
  try {
    return new URLSearchParams(location.search).has('installed');
  } catch {
    return false;
  }
}

let mockSetupDone = false;
let mockSetupTimers: Array<ReturnType<typeof setTimeout>> = [];

const MOCK_PLAN: Record<string, SetupPlanStep[]> = {
  ffmpeg: [
    {
      step: 'ffmpeg',
      url: 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip',
      sizeBytes: 196_000_000,
      note: 'No ffmpeg on PATH or in WinGet: the official static build is extracted to %LOCALAPPDATA%\\Cappycat\\ffmpeg',
    },
  ],
  python: [
    { step: 'uv', url: 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip', sizeBytes: 19_000_000, note: 'uv.exe into %LOCALAPPDATA%\\Cappycat\\bin' },
    { step: 'python', url: 'https://github.com/astral-sh/python-build-standalone/releases', sizeBytes: 32_000_000, note: 'uv venv --python 3.12 (managed CPython)' },
    { step: 'torch', url: 'https://download.pytorch.org/whl/cu130', sizeBytes: 2_900_000_000, note: 'torch + torchvision, CUDA 13.0 wheels (NVIDIA GPU detected)' },
    { step: 'packages', url: 'https://pypi.org/simple', sizeBytes: 450_000_000, note: 'pipeline/requirements-installer.txt, then demucs etc. with --no-deps' },
  ],
  models: [{ step: 'models', url: 'https://huggingface.co', sizeBytes: 1_300_000_000, note: 'python -m cappycat_pipeline download-models' }],
};



export interface ClipsFolderScan {
  folder: string;
  assets: Asset[];
  warnings: string[];
}

export type ExportPreset = 'h264_mp4' | 'h264_nvenc_mp4' | 'prores_mov';

export interface SpeedLutResult {
  sourceDurationMs: number;
  outputDurationMs: number;
  effectiveSpeed: number;
  sourceMs: number[];
  speed: number[];
}

export const api = {
  async appPaths(): Promise<AppPaths> {
    if (!isTauri()) return { cacheDir: '(browser)', ffmpeg: null, ffprobe: null, python: null, pipelineDir: null, logsDir: null };
    return tauriInvoke<AppPaths>('app_paths');
  },

  /** What the AI setup has installed. Browser mode: nothing (installed only with `?installed`). */
  async setupStatus(): Promise<SetupStatus> {
    if (!isTauri()) {
      const done = mockSetupDone;
      return { ffmpeg: done, python: done, models: done, gpu: 'NVIDIA GeForce RTX 4070 (mock)', installed: mockInstalled() };
    }
    return tauriInvoke<SetupStatus>('setup_status');
  },

  /** The steps `setup_ai` would run for `components` (default: everything missing), with sources and sizes. */
  async setupPlan(components?: string[]): Promise<SetupPlanStep[]> {
    if (!isTauri()) {
      const want = components?.length ? components : ['ffmpeg', 'python', 'models'];
      return [...want.flatMap((c) => MOCK_PLAN[c] ?? []), { step: 'verify', note: 'python -m cappycat_pipeline doctor' }];
    }
    return tauriInvoke<SetupPlanStep[]>('setup_plan', { components: components ?? null });
  },

  /** Start the AI setup; progress on `setup://progress`, the end on `setup://done`. */
  async setupAi(components?: string[]): Promise<string> {
    if (!isTauri()) {
      const jobId = `setup_mock_${Date.now()}`;
      const plan = await api.setupPlan(components);
      mockSetupTimers.forEach(clearTimeout);
      mockSetupTimers = [];
      let at = 150;
      for (const st of plan) {
        for (let k = 0; k <= 5; k++) {
          mockSetupTimers.push(
            setTimeout(() => mockEmit('setup://progress', { jobId, step: st.step, pct: k / 5, message: `${st.step}: ${k === 5 ? 'done' : `${Math.round((k / 5) * 100)} % (mock)`}` }), at),
          );
          at += 110;
        }
      }
      mockSetupTimers.push(
        setTimeout(() => {
          mockSetupDone = true;
          mockEmit('setup://done', { jobId, ok: true, error: null });
        }, at + 100),
      );
      return jobId;
    }
    return tauriInvoke<string>('setup_ai', { components: components ?? null });
  },

  async cancelSetup(jobId: string): Promise<void> {
    if (!isTauri()) {
      mockSetupTimers.forEach(clearTimeout);
      mockSetupTimers = [];
      setTimeout(() => mockEmit('setup://done', { jobId, ok: false, error: 'cancelled' }), 50);
      return;
    }
    return tauriInvoke<void>('cancel_setup', { jobId });
  },

  /** Open a folder (logs) in the file manager. */
  async openPath(path: string): Promise<void> {
    if (!isTauri()) return;
    const { openPath } = await import('@tauri-apps/plugin-opener');
    await openPath(path);
  },

  async mediaServerUrl(): Promise<string> {
    if (!isTauri()) return '';
    return tauriInvoke<string>('media_server_url');
  },

  async probeMedia(path: string): Promise<Asset> {
    if (!isTauri()) return mockAsset(path, 0);
    return tauriInvoke<Asset>('probe_media', { path });
  },

  async importMedia(paths: string[]): Promise<Asset[]> {
    if (!isTauri()) return paths.map(mockAsset);
    return tauriInvoke<Asset[]>('import_media', { paths });
  },

  async extractThumbnails(path: string, count: number, width: number): Promise<string[]> {
    if (!isTauri()) return [];
    return tauriInvoke<string[]>('extract_thumbnails', { path, count, width });
  },

  async extractWaveform(path: string, samplesPerSecond: number): Promise<number[]> {
    if (!isTauri()) {
      // distinct per file (stems look different from their source)
      let h = 0;
      for (let i = 0; i < path.length; i++) h = (h * 31 + path.charCodeAt(i)) >>> 0;
      const a = 2 + (h % 5);
      const b = 11 + (h % 13);
      const base = /vocals/i.test(path) ? 0.1 : /background/i.test(path) ? 0.25 : 0.3;
      const n = Math.round(samplesPerSecond * 60);
      return Array.from({ length: n }, (_, i) => Math.min(1, base + 0.6 * Math.abs(Math.sin(i / a) * Math.cos(i / b)) * (/vocals/i.test(path) && Math.sin(i / 9) < -0.3 ? 0.15 : 1)));
    }
    return tauriInvoke<number[]>('extract_waveform', { path, samplesPerSecond });
  },

  async runPipeline(paths: string[], options: PipelineOptions): Promise<string> {
    if (!isTauri()) {
      const jobId = `job_mock_${Date.now()}`;
      const stages = ['ingest', 'shots', 'perception', 'reframe', 'audio', 'transitions', 'assemble'] as const;
      const wait = Math.max(0, mockGpuBusyUntil - Date.now());
      if (wait > 0) setTimeout(() => mockEmit('pipeline://progress', { jobId, event: 'progress', stage: 'ingest', clip: null, pct: 0, message: GPU_WAIT_MESSAGE }), 30);
      mockGpuBusyUntil = Math.max(Date.now(), mockGpuBusyUntil) + stages.length * 520 + 200;
      let i = 0;
      const tick = () => {
        if (i >= stages.length) {
          mockEmit('pipeline://result', { jobId, event: 'result', path: '(mock)' });
          return;
        }
        const stage = stages[i++];
        for (let k = 1; k <= 4; k++) {
          setTimeout(
            () =>
              mockEmit('pipeline://progress', {
                jobId,
                event: 'progress',
                stage,
                clip: paths[0] ?? null,
                pct: k / 4,
                message: `${stage} ${Math.round((k / 4) * 100)}%`,
              }),
            k * 120,
          );
        }
        setTimeout(tick, 520);
      };
      setTimeout(tick, 100 + wait);
      return jobId;
    }
    return tauriInvoke<string>('run_pipeline', { paths, options });
  },

  /**
   * Split the audio of `paths` into vocals / background stems (Demucs v4, cached per file).
   * Events: `separate://progress|result|log|done`. Browser mode fakes progress and fake stems.
   */
  async separateAudio(paths: string[]): Promise<string> {
    if (!isTauri()) {
      const jobId = `separate_mock_${Date.now()}`;
      const steps = 5;
      const tick = 140;
      const wait = Math.max(0, mockGpuBusyUntil - Date.now());
      if (wait > 0) setTimeout(() => mockEmit('separate://progress', { jobId, pct: 0, clipPct: 0, clip: null, message: GPU_WAIT_MESSAGE }), 30);
      mockGpuBusyUntil = Math.max(Date.now(), mockGpuBusyUntil) + paths.length * steps * tick + 100;
      const at = (ms: number) => ms + wait;
      paths.forEach((path, i) => {
        const clip = path.split(/[\\/]/).pop() ?? path;
        for (let k = 1; k <= steps; k++) {
          const clipPct = k / steps;
          setTimeout(
            () =>
              mockEmit('separate://progress', {
                jobId,
                pct: (i + clipPct) / paths.length,
                clipPct,
                clip,
                message: `${clip}: segment ${k}/${steps} (mock)`,
              }),
            at((i * steps + k) * tick),
          );
        }
        const stem = (n: string) => `C:/mock/stems/${clip.replace(/\.[^.]+$/, '')}/${n}.wav`;
        setTimeout(
          () => mockEmit('separate://result', { jobId, path, stems: { vocals: stem('vocals'), background: stem('background') } }),
          at((i + 1) * steps * tick + 10),
        );
      });
      setTimeout(() => mockEmit('separate://done', { jobId, ok: true, error: null }), at(paths.length * steps * tick + 60));
      return jobId;
    }
    return tauriInvoke<string>('separate_audio', { paths });
  },

  async cancelSeparate(jobId: string): Promise<void> {
    if (!isTauri()) return;
    return tauriInvoke<void>('cancel_separate', { jobId });
  },

  async cancelPipeline(jobId: string): Promise<void> {
    if (!isTauri()) return;
    return tauriInvoke<void>('cancel_pipeline', { jobId });
  },

  async exportProject(
    project: Project,
    outPath: string,
    preset: ExportPreset,
    range?: { startMs: number; endMs: number } | null,
  ): Promise<string> {
    if (!isTauri()) {
      const jobId = `job_mock_export_${Date.now()}`;
      for (let k = 1; k <= 10; k++) {
        setTimeout(() => mockEmit('export://progress', { jobId, pct: k / 10, message: `frame ${k * 24}/240` }), k * 150);
      }
      setTimeout(() => mockEmit('export://done', { jobId, ok: true, outPath, error: null }), 1700);
      return jobId;
    }
    return tauriInvoke<string>('export_project', { project, outPath, preset, range: range ?? null });
  },

  /** The universal adjust preset (presets/universal-adjust.json; created with the house look). */
  async loadUniversalAdjust(): Promise<{ path: string; preset: UniversalPresetFile }> {
    if (!isTauri()) {
      try {
        const raw = localStorage.getItem('cappycat:universal-adjust');
        if (raw) return { path: '(browser storage)', preset: JSON.parse(raw) as UniversalPresetFile };
      } catch {
        /* ignore */
      }
      return {
        path: '(browser storage)',
        preset: { version: 1, name: 'Universal adjust', enabledByDefault: true, values: defaultUniversalValues() },
      };
    }
    return tauriInvoke('load_universal_adjust');
  },

  async saveUniversalAdjust(preset: UniversalPresetFile): Promise<{ path: string; preset: UniversalPresetFile }> {
    if (!isTauri()) {
      try {
        localStorage.setItem('cappycat:universal-adjust', JSON.stringify(preset));
      } catch {
        /* ignore */
      }
      return { path: '(browser storage)', preset };
    }
    return tauriInvoke('save_universal_adjust', { preset });
  },

  /** Scan the project's clips folder (default <repo>/clips) and order it by filename. */
  async scanClipsFolder(path?: string | null): Promise<ClipsFolderScan> {
    if (!isTauri()) {
      const names = ['01_intro.mp4', '02_kitchen.mp4', '03_chase.mp4', '10_finale.mp4'];
      return {
        folder: 'C:/mock/clips',
        assets: names.map((n, i) => ({ ...mockAsset(`C:/mock/clips/${n}`, i), order: i, orderReason: `leading number ${parseInt(n, 10)}` })),
        warnings: ['numbering skips 4, 5, 6, 7, 8, 9 - is a clip missing?'],
      };
    }
    return tauriInvoke<ClipsFolderScan>('scan_clips_folder', { path: path ?? null });
  },

  async pickFolder(defaultPath?: string | null): Promise<string | null> {
    if (!isTauri()) return 'C:/mock/clips';
    const { open } = await import('@tauri-apps/plugin-dialog');
    const res = await open({ directory: true, multiple: false, defaultPath: defaultPath ?? undefined });
    return typeof res === 'string' ? res : null;
  },

  async saveProject(path: string, project: Project): Promise<void> {
    if (!isTauri()) {
      localStorage.setItem(`cappycat:project:${path}`, JSON.stringify(project));
      return;
    }
    await tauriInvoke<Project>('save_project', { path, project });
  },

  async loadProject(path: string): Promise<Project> {
    if (!isTauri()) {
      const raw = localStorage.getItem(`cappycat:project:${path}`);
      if (!raw) throw new Error('not found');
      return JSON.parse(raw) as Project;
    }
    return tauriInvoke<Project>('load_project', { path });
  },

  async evaluateKeyframes(keyframed: Keyframed<unknown>, timeMs: number): Promise<unknown> {
    if (!isTauri()) return keyframed.static;
    return tauriInvoke<unknown>('evaluate_keyframes', { keyframed, timeMs });
  },

  async speedCurveLut(curve: SpeedCurve, durationMs: number, samples: number): Promise<SpeedLutResult | null> {
    if (!isTauri()) return null;
    return tauriInvoke<SpeedLutResult>('speed_curve_lut', { curve, durationMs, samples });
  },

  async cancelExport(jobId: string): Promise<void> {
    if (!isTauri()) return;
    return tauriInvoke<void>('cancel_export', { jobId });
  },

  /** Open the native file picker for media. Returns absolute paths. */
  async pickMedia(): Promise<string[]> {
    if (!isTauri()) {
      return [
        'C:/mock/Clip_01.mp4',
        'C:/mock/Clip_02.mp4',
        'C:/mock/Clip_03.mp4',
        'C:/mock/Clip_04.mp4',
        'C:/mock/LUT_Cinematic.cube',
      ];
    }
    const { open } = await import('@tauri-apps/plugin-dialog');
    const res = await open({
      multiple: true,
      directory: false,
      filters: [
        { name: 'Media', extensions: ['mp4', 'mov', 'mkv', 'webm', 'avi', 'm4v', 'mp3', 'wav', 'aac', 'flac', 'png', 'jpg', 'jpeg', 'cube'] },
      ],
    });
    if (!res) return [];
    return Array.isArray(res) ? res : [res];
  },

  async pickSavePath(defaultName: string, ext: string): Promise<string | null> {
    if (!isTauri()) return `C:/mock/${defaultName}.${ext}`;
    const { save } = await import('@tauri-apps/plugin-dialog');
    return save({ defaultPath: `${defaultName}.${ext}`, filters: [{ name: ext.toUpperCase(), extensions: [ext] }] });
  },

  async pickOpenPath(ext: string): Promise<string | null> {
    if (!isTauri()) {
      // browser mode: "open" the most recently saved project from local storage
      return api.recentProjects()[0] ?? null;
    }
    const { open } = await import('@tauri-apps/plugin-dialog');
    const res = await open({ multiple: false, filters: [{ name: ext.toUpperCase(), extensions: [ext] }] });
    return typeof res === 'string' ? res : null;
  },

  /**
   * A .cube LUT parsed by the Rust core (`load_lut`). Browser mode synthesises a mild cool look so
   * the GPU LUT path is still exercised.
   */
  async loadLut(path: string): Promise<Lut3D> {
    if (!isTauri()) return syntheticLut();
    const r = await tauriInvoke<LoadLutResult>('load_lut', { path });
    return {
      size: r.size,
      data: Float32Array.from(r.data),
      domainMin: r.domainMin,
      domainMax: r.domainMax,
      title: r.title ?? undefined,
    };
  },

  /** File > Open: a saved project or a pipeline analysis, read and parsed by the Rust core. */
  async openDocument(path: string): Promise<OpenedDocument> {
    if (!isTauri()) {
      const raw = localStorage.getItem(`cappycat:project:${path}`) ?? localStorage.getItem(path);
      if (!raw) throw new Error(`${path}: not found in browser storage`);
      return { kind: 'project', project: JSON.parse(raw) as Project };
    }
    return tauriInvoke<OpenedDocument>('open_document', { path });
  },

  /** Allow the media server to serve these files (assets, LUTs, stems). No-op in browser mode. */
  async registerMedia(paths: string[]): Promise<void> {
    if (!isTauri() || !paths.length) return;
    await tauriInvoke<void>('register_media', { paths });
  },

  /** Recent project files (most recent first), kept in local storage. */
  recentProjects(): string[] {
    try {
      const v = JSON.parse(localStorage.getItem('cappycat:recent') ?? '[]');
      return Array.isArray(v) ? v.filter((x) => typeof x === 'string').slice(0, 8) : [];
    } catch {
      return [];
    }
  },

  addRecentProject(path: string): void {
    try {
      const list = [path, ...api.recentProjects().filter((p) => p !== path)].slice(0, 8);
      localStorage.setItem('cappycat:recent', JSON.stringify(list));
    } catch {
      /* storage unavailable */
    }
  },

  /** URL the <video> element can stream from. */
  mediaUrl(serverUrl: string, path: string): string {
    if (!serverUrl) return '';
    return `${serverUrl}/media?path=${encodeURIComponent(path)}`;
  },

  frameUrl(serverUrl: string, path: string, ms: number, w = 320): string {
    if (!serverUrl) return '';
    return `${serverUrl}/frame?path=${encodeURIComponent(path)}&ms=${Math.round(ms)}&w=${w}`;
  },
};

interface LoadLutResult {
  size: number;
  /** RGB floats, red fastest */
  data: number[];
  domainMin: [number, number, number];
  domainMax: [number, number, number];
  title?: string | null;
}

export interface OpenedDocument {
  kind: 'project' | 'analysis';
  project: Project;
  analysis?: AnalysisResult;
}

/** Browser mode stand-in for `load_lut`: 17^3, slightly cooler shadows. */
function syntheticLut(size = 17): Lut3D {
  const data = new Float32Array(size * size * size * 3);
  let i = 0;
  for (let b = 0; b < size; b++)
    for (let g = 0; g < size; g++)
      for (let r = 0; r < size; r++) {
        const R = r / (size - 1);
        const G = g / (size - 1);
        const B = b / (size - 1);
        data[i++] = R * 0.95;
        data[i++] = G * 0.98;
        data[i++] = Math.min(1, B * 1.04 + 0.02);
      }
  return { size, data, domainMin: [0, 0, 0], domainMax: [1, 1, 1], title: 'Synthetic cool (browser mode)' };
}

/** GPU queue: analysis / separation report this while another AI job holds the GPU. */
export function isWaitingForGpu(message: string | null | undefined): boolean {
  return !!message && /waiting for the gpu/i.test(message);
}

export type PipelineEventPayload = PipelineEvent & { jobId: string };
export interface AnalysisPayload {
  jobId: string;
  analysis: AnalysisResult;
}
export interface ExportProgressPayload {
  jobId: string;
  pct: number;
  message: string;
}
export interface ExportDonePayload {
  jobId: string;
  ok: boolean;
  outPath: string;
  error: string | null;
}
export interface SeparateProgressPayload {
  jobId: string;
  /** overall 0..1 over every file of the job */
  pct: number;
  /** progress of the file named by `clip` */
  clipPct: number | null;
  clip: string | null;
  message: string;
}
export interface SeparateResultPayload {
  jobId: string;
  /** source path as passed to separate_audio */
  path: string;
  stems: AssetStems;
}
export interface SeparateDonePayload {
  jobId: string;
  ok: boolean;
  error: string | null;
}
export interface PipelineExitPayload {
  jobId: string;
  code: number | null;
  cancelled: boolean;
  success: boolean;
}

export async function onEvent<T>(event: string, handler: (payload: T) => void): Promise<Unlisten> {
  if (!isTauri()) {
    const set = mockListeners.get(event) ?? new Set();
    const h = handler as (p: unknown) => void;
    set.add(h);
    mockListeners.set(event, set);
    return () => set.delete(h);
  }
  return tauriListen<T>(event, handler);
}
