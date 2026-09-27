import type { ColorCurves, CurvePoint } from '@/types/project';

/** Monotone cubic spline through control points, evaluated on 0..1, clamped. */
export function evalCurve(points: CurvePoint[], x: number): number {
  const pts = [...points].sort((a, b) => a[0] - b[0]);
  const n = pts.length;
  if (n === 0) return x;
  if (n === 1) return pts[0][1];
  if (x <= pts[0][0]) return pts[0][1];
  if (x >= pts[n - 1][0]) return pts[n - 1][1];
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const d: number[] = [];
  const h: number[] = [];
  for (let i = 0; i < n - 1; i++) {
    h.push(xs[i + 1] - xs[i]);
    d.push(h[i] === 0 ? 0 : (ys[i + 1] - ys[i]) / h[i]);
  }
  const m: number[] = new Array(n).fill(0);
  m[0] = d[0];
  m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) m[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
  for (let i = 0; i < n - 1; i++) {
    if (d[i] === 0) {
      m[i] = 0;
      m[i + 1] = 0;
      continue;
    }
    const a = m[i] / d[i];
    const b = m[i + 1] / d[i];
    const s = a * a + b * b;
    if (s > 9) {
      const tau = 3 / Math.sqrt(s);
      m[i] = tau * a * d[i];
      m[i + 1] = tau * b * d[i];
    }
  }
  let i = 0;
  while (i < n - 2 && x > xs[i + 1]) i++;
  const u = (x - xs[i]) / h[i];
  const u2 = u * u;
  const u3 = u2 * u;
  const y =
    (2 * u3 - 3 * u2 + 1) * ys[i] +
    (u3 - 2 * u2 + u) * h[i] * m[i] +
    (-2 * u3 + 3 * u2) * ys[i + 1] +
    (u3 - u2) * h[i] * m[i + 1];
  return Math.min(1, Math.max(0, y));
}

/** Bake curves into a width x 4 RGBA8 texture (rows: master, r, g, b). */
export function bakeCurves(curves: ColorCurves, width = 256): Uint8Array {
  const rows: CurvePoint[][] = [curves.master, curves.r, curves.g, curves.b];
  const out = new Uint8Array(width * 4 * 4);
  for (let row = 0; row < 4; row++) {
    for (let x = 0; x < width; x++) {
      const v = evalCurve(rows[row], x / (width - 1));
      const i = (row * width + x) * 4;
      const b = Math.round(v * 255);
      out[i] = b;
      out[i + 1] = b;
      out[i + 2] = b;
      out[i + 3] = 255;
    }
  }
  return out;
}

export function isIdentityCurve(points: CurvePoint[]): boolean {
  return points.every(([x, y]) => Math.abs(x - y) < 1e-6);
}
