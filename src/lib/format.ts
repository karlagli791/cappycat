export function pad(n: number, w = 2): string {
  return String(Math.floor(n)).padStart(w, '0');
}

/** SMPTE-style timecode HH:MM:SS:FF */
export function timecode(ms: number, fps: number): string {
  const totalFrames = Math.floor((Math.max(0, ms) / 1000) * fps + 1e-6);
  const f = totalFrames % Math.round(fps);
  const s = Math.floor(totalFrames / fps);
  return `${pad(s / 3600)}:${pad((s / 60) % 60)}:${pad(s % 60)}:${pad(f)}`;
}

export function shortTime(ms: number): string {
  const s = Math.max(0, ms) / 1000;
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${pad(m)}:${pad(r)}${m === 0 && r < 10 ? '' : ''}`;
}

export function fmtDuration(ms: number): string {
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

export function basename(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

export function naturalCompare(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
}
