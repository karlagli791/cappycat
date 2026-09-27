/** Parser for Adobe/Resolve `.cube` 3D LUT files (17^3 to 64^3). */
export interface Lut3D {
  size: number;
  /** RGB float triplets, red fastest as per the .cube spec, length size^3 * 3 */
  data: Float32Array;
  domainMin: [number, number, number];
  domainMax: [number, number, number];
  title?: string;
}

export function parseCube(text: string): Lut3D {
  let size = 0;
  let title: string | undefined;
  const domainMin: [number, number, number] = [0, 0, 0];
  const domainMax: [number, number, number] = [1, 1, 1];
  const values: number[] = [];
  const lines = text.split(/\r?\n/);
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    if (line.startsWith('TITLE')) {
      title = line.slice(5).trim().replace(/^"|"$/g, '');
      continue;
    }
    if (line.startsWith('LUT_3D_SIZE')) {
      size = parseInt(line.split(/\s+/)[1], 10);
      continue;
    }
    if (line.startsWith('LUT_1D_SIZE')) {
      throw new Error('1D LUTs are not supported; expected LUT_3D_SIZE');
    }
    if (line.startsWith('DOMAIN_MIN')) {
      const p = line.split(/\s+/).slice(1).map(Number);
      domainMin[0] = p[0];
      domainMin[1] = p[1];
      domainMin[2] = p[2];
      continue;
    }
    if (line.startsWith('DOMAIN_MAX')) {
      const p = line.split(/\s+/).slice(1).map(Number);
      domainMax[0] = p[0];
      domainMax[1] = p[1];
      domainMax[2] = p[2];
      continue;
    }
    const parts = line.split(/\s+/);
    if (parts.length >= 3) {
      const r = Number(parts[0]);
      const g = Number(parts[1]);
      const b = Number(parts[2]);
      if (Number.isFinite(r) && Number.isFinite(g) && Number.isFinite(b)) values.push(r, g, b);
    }
  }
  if (!size) throw new Error('Missing LUT_3D_SIZE');
  if (size < 2 || size > 128) throw new Error(`Unsupported LUT size ${size}`);
  const expected = size * size * size * 3;
  if (values.length !== expected) {
    throw new Error(`Expected ${expected} values for a ${size}^3 LUT, got ${values.length}`);
  }
  return { size, data: Float32Array.from(values), domainMin, domainMax, title };
}

/** Identity LUT used when no LUT is assigned. */
export function identityLut(size = 2): Lut3D {
  const data = new Float32Array(size * size * size * 3);
  let i = 0;
  for (let b = 0; b < size; b++)
    for (let g = 0; g < size; g++)
      for (let r = 0; r < size; r++) {
        data[i++] = r / (size - 1);
        data[i++] = g / (size - 1);
        data[i++] = b / (size - 1);
      }
  return { size, data, domainMin: [0, 0, 0], domainMax: [1, 1, 1] };
}

/** CPU trilinear sample, mainly for tests. */
export function sampleLut(lut: Lut3D, r: number, g: number, b: number): [number, number, number] {
  const s = lut.size;
  const n = s - 1;
  const fx = Math.min(n, Math.max(0, r * n));
  const fy = Math.min(n, Math.max(0, g * n));
  const fz = Math.min(n, Math.max(0, b * n));
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const z0 = Math.floor(fz);
  const x1 = Math.min(n, x0 + 1);
  const y1 = Math.min(n, y0 + 1);
  const z1 = Math.min(n, z0 + 1);
  const tx = fx - x0;
  const ty = fy - y0;
  const tz = fz - z0;
  const at = (x: number, y: number, z: number, c: number) => lut.data[((z * s + y) * s + x) * 3 + c];
  const out: [number, number, number] = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const c00 = at(x0, y0, z0, c) * (1 - tx) + at(x1, y0, z0, c) * tx;
    const c10 = at(x0, y1, z0, c) * (1 - tx) + at(x1, y1, z0, c) * tx;
    const c01 = at(x0, y0, z1, c) * (1 - tx) + at(x1, y0, z1, c) * tx;
    const c11 = at(x0, y1, z1, c) * (1 - tx) + at(x1, y1, z1, c) * tx;
    const c0 = c00 * (1 - ty) + c10 * ty;
    const c1 = c01 * (1 - ty) + c11 * ty;
    out[c] = c0 * (1 - tz) + c1 * tz;
  }
  return out;
}
