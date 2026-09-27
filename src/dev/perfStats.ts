/**
 * Dev-only performance counters (React commits, timeline redraw time, ...). Written by
 * instrumented code under `import.meta.env.DEV`, read by the `window.__cappy` harness.
 */
export const perfStats = {
  commits: 0,
  timelineDraws: [] as number[],
  overlayDraws: [] as number[],
};

export function recordPerf(kind: 'timelineDraws' | 'overlayDraws', ms: number): void {
  const arr = perfStats[kind];
  arr.push(ms);
  if (arr.length > 5000) arr.splice(0, arr.length - 5000);
}
