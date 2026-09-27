/**
 * Voice separation controller: starts `separate_audio` jobs for assets that need stems and
 * feeds `separate://*` events into the store (asset.stems, per-asset status, overall progress).
 */
import { useEffect } from 'react';
import { api, onEvent, type SeparateDonePayload, type SeparateProgressPayload, type SeparateResultPayload } from '@/lib/tauri';
import { normPath, pathsNotRunning } from '@/engine/voice';
import { useEditor } from './store';

/** Separate `paths` (skipping ones a running job already covers). Returns the job id, or null
 *  when there was nothing to do or the command failed (logged). */
export async function requestSeparation(paths: string[]): Promise<string | null> {
  const st = useEditor.getState();
  const todo = pathsNotRunning(paths, st.separationJobs);
  if (!todo.length) return null;
  try {
    const jobId = await api.separateAudio(todo);
    useEditor.getState().separationStarted(jobId, todo);
    st.log('info', `Separating voice / background of ${todo.length} file(s)…`);
    return jobId;
  } catch (e) {
    st.log('error', `Voice separation could not start: ${String(e)}`);
    return null;
  }
}

/**
 * "Separate to tracks" for clips: the ones whose stems exist are done at once (one undo step);
 * the others are queued and completed when `separate_audio` delivers their stems.
 */
export async function separateClipsToTracks(clipIds: string[]): Promise<{ done: number; waiting: number }> {
  const st = useEditor.getState();
  const r = st.separateToTracks(clipIds);
  if (r.done.length) st.log('info', `Separated ${r.done.length} clip(s) to the Voice / Background tracks`);
  if (r.waiting.length) {
    st.queueStemTracks(r.waiting);
    const paths = r.missing.map((a) => a.path);
    await requestSeparation(paths);
    // nothing running for a path (the command failed): drop its queued requests
    const running = new Set(Object.values(useEditor.getState().separationJobs).flatMap((j) => j.paths.map(normPath)));
    const failed = paths.filter((x) => !running.has(normPath(x)));
    if (failed.length) useEditor.getState().runPendingStemTracks(failed);
    else st.log('info', `${r.waiting.length} clip(s) go to the stem tracks when their voice separation finishes`);
  }
  return { done: r.done.length, waiting: r.waiting.length };
}

export async function cancelSeparation(jobId: string): Promise<void> {
  try {
    await api.cancelSeparate(jobId);
  } finally {
    useEditor.getState().separationDone(jobId, false, 'cancelled');
  }
}

/** Subscribe to the separation events once (App). */
export function useSeparationEvents(): void {
  useEffect(() => {
    // listeners resolve asynchronously: one that arrives after cleanup (StrictMode) is dropped at once
    let alive = true;
    const offs: Array<() => void> = [];
    const keep = (p: Promise<() => void>) =>
      void p.then((off) => {
        if (alive) offs.push(off);
        else off();
      });
    const s = () => useEditor.getState();
    keep(
      onEvent<SeparateProgressPayload>('separate://progress', (p) => {
        s().separationProgress(p.jobId, p.pct ?? 0, p.clipPct ?? null, p.clip ?? null, p.message ?? '');
      }),
    );
    keep(
      onEvent<SeparateResultPayload>('separate://result', (p) => {
        s().setAssetStems(p.path, p.stems);
        s().log('info', `Stems ready: ${p.path.split(/[\\/]/).pop() ?? p.path}`);
        const n = s().runPendingStemTracks();
        if (n) s().log('info', `Separated ${n} clip(s) to the Voice / Background tracks`);
      }),
    );
    keep(
      onEvent<{ jobId: string; level: 'info' | 'warn' | 'error'; message: string }>('separate://log', (p) => {
        if (p.level === 'error') s().log('error', `[separate] ${p.message}`);
      }),
    );
    keep(
      onEvent<SeparateDonePayload>('separate://done', (p) => {
        const job = s().separationJobs[p.jobId];
        s().separationDone(p.jobId, p.ok, p.error);
        // queued "Separate to tracks" requests whose file failed (or was cancelled) are dropped
        if (job) s().runPendingStemTracks(job.paths.filter((x) => !job.done.includes(normPath(x))));
        if (!p.ok && p.error !== 'cancelled') s().log('error', `Voice separation failed: ${p.error ?? 'unknown error'}`);
      }),
    );
    return () => {
      alive = false;
      offs.forEach((f) => f());
    };
  }, []);
}
