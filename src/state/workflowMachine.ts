/**
 * Operational workflow state machine (XState v5), mirroring the design spec:
 * idle -> ingesting -> detectingShots -> scanningDuplicates -> reframing
 *      -> normalizingAudio -> assembling -> ready  (with failure/cancel paths)
 */
import { assign, fromPromise, setup } from 'xstate';
import type { AnalysisResult, PipelineOptions, PipelineStage } from '@/types/project';
import { api } from '@/lib/tauri';

export interface StageProgress {
  stage: PipelineStage;
  pct: number;
  message: string;
  clip: string | null;
}

export interface WorkflowContext {
  paths: string[];
  options: PipelineOptions;
  jobId: string | null;
  progress: StageProgress | null;
  stageHistory: PipelineStage[];
  error: string | null;
  result: AnalysisResult | null;
}

export type WorkflowEvent =
  | { type: 'START'; paths: string[]; options: PipelineOptions }
  | { type: 'PROGRESS'; progress: StageProgress }
  | { type: 'RESULT'; result: AnalysisResult }
  | { type: 'DONE' }
  | { type: 'FAIL'; error: string }
  | { type: 'CANCEL' }
  | { type: 'RESET' };

export const defaultPipelineOptions = (): PipelineOptions => ({
  // empty = the pipeline detects the main cast from characters/characters.json plus generic prompts
  prompts: [],
  detector: 'hybrid',
  shotDetector: 'auto',
  targetDurationMs: 165_000,
  normalizeAudio: true,
  targetLufs: -14,
  detectBeats: true,
  similarityThreshold: 0.85,
  smoothing: 'ema',
  smoothingAlpha: 0.15,
  keepOrder: true,
});

const STAGE_TO_STATE: Record<PipelineStage, string> = {
  ingest: 'ingesting',
  shots: 'detectingShots',
  perception: 'scanningDuplicates',
  reframe: 'reframing',
  audio: 'normalizingAudio',
  transitions: 'normalizingAudio',
  assemble: 'assembling',
  export: 'assembling',
  download: 'ingesting',
  separate: 'normalizingAudio',
};

export function stateForStage(stage: PipelineStage): string {
  return STAGE_TO_STATE[stage];
}

const launch = fromPromise<string, { paths: string[]; options: PipelineOptions }>(async ({ input }) =>
  api.runPipeline(input.paths, input.options),
);

export const workflowMachine = setup({
  types: {
    context: {} as WorkflowContext,
    events: {} as WorkflowEvent,
  },
  actors: { launch },
  actions: {
    recordProgress: assign({
      progress: ({ event }) => (event.type === 'PROGRESS' ? event.progress : null),
      stageHistory: ({ context, event }) => {
        if (event.type !== 'PROGRESS') return context.stageHistory;
        const s = event.progress.stage;
        return context.stageHistory.includes(s) ? context.stageHistory : [...context.stageHistory, s];
      },
    }),
    cancelJob: ({ context }) => {
      if (context.jobId) void api.cancelPipeline(context.jobId);
    },
  },
}).createMachine({
  id: 'workflow',
  initial: 'idle',
  context: {
    paths: [],
    options: defaultPipelineOptions(),
    jobId: null,
    progress: null,
    stageHistory: [],
    error: null,
    result: null,
  },
  states: {
    idle: {
      on: {
        START: {
          target: 'launching',
          actions: assign({
            paths: ({ event }) => event.paths,
            options: ({ event }) => event.options,
            error: null,
            result: null,
            progress: null,
            stageHistory: [],
          }),
        },
      },
    },
    launching: {
      invoke: {
        src: 'launch',
        input: ({ context }) => ({ paths: context.paths, options: context.options }),
        onDone: { target: 'ingesting', actions: assign({ jobId: ({ event }) => event.output }) },
        onError: { target: 'failed', actions: assign({ error: ({ event }) => String(event.error) }) },
      },
    },
    ingesting: { on: { PROGRESS: [{ guard: ({ event }) => event.progress.stage === 'shots', target: 'detectingShots', actions: 'recordProgress' }, { guard: ({ event }) => event.progress.stage === 'perception', target: 'scanningDuplicates', actions: 'recordProgress' }, { actions: 'recordProgress' }] } },
    detectingShots: { on: { PROGRESS: [{ guard: ({ event }) => event.progress.stage === 'perception', target: 'scanningDuplicates', actions: 'recordProgress' }, { guard: ({ event }) => event.progress.stage === 'reframe', target: 'reframing', actions: 'recordProgress' }, { guard: ({ event }) => event.progress.stage === 'audio', target: 'normalizingAudio', actions: 'recordProgress' }, { actions: 'recordProgress' }] } },
    scanningDuplicates: { on: { PROGRESS: [{ guard: ({ event }) => event.progress.stage === 'reframe', target: 'reframing', actions: 'recordProgress' }, { guard: ({ event }) => event.progress.stage === 'audio', target: 'normalizingAudio', actions: 'recordProgress' }, { actions: 'recordProgress' }] } },
    reframing: { on: { PROGRESS: [{ guard: ({ event }) => event.progress.stage === 'audio' || event.progress.stage === 'transitions', target: 'normalizingAudio', actions: 'recordProgress' }, { guard: ({ event }) => event.progress.stage === 'assemble', target: 'assembling', actions: 'recordProgress' }, { actions: 'recordProgress' }] } },
    normalizingAudio: { on: { PROGRESS: [{ guard: ({ event }) => event.progress.stage === 'assemble', target: 'assembling', actions: 'recordProgress' }, { guard: ({ event }) => event.progress.stage === 'shots', target: 'detectingShots', actions: 'recordProgress' }, { actions: 'recordProgress' }] } },
    assembling: { on: { PROGRESS: { actions: 'recordProgress' } } },
    ready: {
      on: { RESET: 'idle', START: { target: 'launching', actions: assign({ paths: ({ event }) => event.paths, options: ({ event }) => event.options, error: null, result: null, progress: null, stageHistory: [] }) } },
    },
    failed: {
      on: { RESET: 'idle', START: { target: 'launching', actions: assign({ paths: ({ event }) => event.paths, options: ({ event }) => event.options, error: null, result: null, progress: null, stageHistory: [] }) } },
    },
    cancelled: { on: { RESET: 'idle', START: { target: 'launching', actions: assign({ paths: ({ event }) => event.paths, options: ({ event }) => event.options, error: null, result: null, progress: null, stageHistory: [] }) } } },
  },
  on: {
    RESULT: { target: '.ready', actions: assign({ result: ({ event }) => event.result }) },
    DONE: { target: '.ready' },
    FAIL: { target: '.failed', actions: assign({ error: ({ event }) => event.error }) },
    CANCEL: { target: '.cancelled', actions: 'cancelJob' },
  },
});

export const WORKFLOW_STATES: Array<{ id: string; label: string }> = [
  { id: 'idle', label: 'Idle' },
  { id: 'ingesting', label: 'Ingesting & sorting media' },
  { id: 'detectingShots', label: 'TransNetV2 cut detection' },
  { id: 'scanningDuplicates', label: 'Duplicate scanning (Grounded SAM 2 / YOLO-World + ByteTrack)' },
  { id: 'reframing', label: 'Camera zoom & crop matrix' },
  { id: 'normalizingAudio', label: 'Audio normalization & beat sync' },
  { id: 'assembling', label: 'Timeline assembly' },
  { id: 'ready', label: 'Ready for preview' },
];
