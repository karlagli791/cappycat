//! Baked colour grade: the per-pixel, position-independent part of a clip's
//! grade ([`GradeParams::apply_pre`] and [`GradeParams::apply_tone`]: exposure,
//! brilliance, contrast, highlights / shadows, white balance, wheels, gamma,
//! saturation / vibrance and the 8-channel HSL) sampled at plan time into 3D
//! lattices, so the compositor does two lattice lookups per pixel instead of
//! the full stage chain. Exact per pixel: everything that depends on the
//! pixel's position — the sharpening term (added between the two lattices),
//! blur, mask, vignette, grain — and the two table stages after HSL (RGB
//! curves and the 3D LUT, [`GradeParams::apply_looks`]), which are cheap
//! lookups whose 8-bit texels / lattice kinks a coarse lattice cannot follow
//! within 1/255.
//!
//! * `pre` over the source cube `[0,1]³` (skipped when those stages are
//!   neutral); then `+ 2·sharpness·detail`; then `tone` over the box `pre` can
//!   reach (plus a margin for the detail term when sharpening). Colours
//!   outside that box are graded exactly.
//!
//! **Accuracy.** Lookups are tetrahedral (6 tetrahedra per cell). The grade
//! has kinks: `max(c, 0)` before gamma, the hue sectors of the HSL stage (the
//! planes where two channels are equal, after white balance), clamps. The
//! `tone` lattice uses one step on all axes and node offsets chosen so that
//! the `max(c, 0)` planes are cell faces and the hue-sector planes are the
//! tetrahedra's diagonal faces (exact when the wheels / gamma are neutral —
//! the universal preset), so interpolation is exact across them. Every cell is
//! then checked against the exact grade at 64 interior points when it is
//! built; cells whose error exceeds the cell tolerance (other kinks: the
//! saturation clamps) are flagged and graded exactly at render time. The
//! lattice size (33³ or 65³) is chosen by the measured result, the end-to-end
//! error is measured on a probe set (`probe_max_err`, `probe_mean_err`), and a
//! bake is retried with a tighter tolerance or discarded (the clip is graded
//! exactly) rather than exceed 1/255. `baked_grade_matches_exact_grade`
//! proves the bound over a dense grid.

use super::color::GradeParams;
use rayon::prelude::*;

/// Largest error (in 0..1 units) tolerated at a cell's interior probe points
/// before the cell is graded exactly.
pub const CELL_TOLERANCE: f32 = 0.6 / 255.0;
/// Upper bound for the end-to-end error of an accepted bake.
pub const MAX_BAKE_ERROR: f32 = 1.0 / 255.0;
/// Extra room around `pre`'s range for the sharpening term.
const DETAIL_MARGIN: f32 = 0.12;
/// Probe error an accepted bake must stay under (margin below [`MAX_BAKE_ERROR`]: the
/// probe set is finite).
const PROBE_TARGET: f32 = 0.8 / 255.0;
/// The `tone` lattice is dropped (tone graded exactly, only `pre` baked) when more than
/// this share of its cells needs the exact path: a lookup that mostly falls through to the
/// exact grade is slower than the exact grade alone. (On real footage the exact cells —
/// dark and near-grey colours under an HSL grade — hold far more pixels than their volume
/// share: ~45 % of a 720p frame for the universal preset's 16 %; see `diag_real_frame`.)
const MAX_TONE_EXACT: f64 = 0.25;
/// A 33³ lattice is used when at most this fraction of its cells needs the exact path.
const MAX_EXACT_FRACTION_33: f64 = 0.04;

type ColorFn<'a> = dyn Fn([f32; 3]) -> [f32; 3] + Sync + 'a;

#[inline]
fn clamp01(c: [f32; 3]) -> [f32; 3] {
    [c[0].clamp(0.0, 1.0), c[1].clamp(0.0, 1.0), c[2].clamp(0.0, 1.0)]
}

#[inline]
fn max_abs_diff(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).abs().max((a[1] - b[1]).abs()).max((a[2] - b[2]).abs())
}

/// Tetrahedral interpolation in a cell (corners `p[x + 2y + 4z]`).
#[inline]
fn tetrahedral(p: &[[f32; 3]; 8], tx: f32, ty: f32, tz: f32) -> [f32; 3] {
    let (a, b, c, d, w1, w2, w3) = if tx >= ty {
        if ty >= tz {
            (0, 1, 3, 7, tx, ty, tz) // tx >= ty >= tz
        } else if tx >= tz {
            (0, 1, 5, 7, tx, tz, ty) // tx >= tz > ty
        } else {
            (0, 4, 5, 7, tz, tx, ty) // tz > tx >= ty
        }
    } else if tx >= tz {
        (0, 2, 3, 7, ty, tx, tz) // ty > tx >= tz
    } else if ty >= tz {
        (0, 2, 6, 7, ty, tz, tx) // ty >= tz > tx
    } else {
        (0, 4, 6, 7, tz, ty, tx) // tz > ty > tx
    };
    let mut out = [0.0; 3];
    for i in 0..3 {
        out[i] = p[a][i] + (p[b][i] - p[a][i]) * w1 + (p[c][i] - p[b][i]) * w2 + (p[d][i] - p[c][i]) * w3;
    }
    out
}

/// A sampled colour function on a regular grid with per-cell exact flags.
#[derive(Debug, Clone)]
pub struct Lattice {
    /// nodes per axis
    n: [usize; 3],
    /// coordinates of node 0 per axis
    origin: [f32; 3],
    inv_step: [f32; 3],
    /// the box inside which lookups are answered (⊆ the node span)
    lo: [f32; 3],
    hi: [f32; 3],
    data: Vec<[f32; 3]>,
    /// the requested nodes per axis (33 / 65)
    nominal: usize,
    /// bitset over the cells: grade this cell exactly
    exact: Vec<u64>,
    exact_cells: usize,
}

impl Lattice {
    /// Sample `f` on the grid (`n` nodes per axis from `origin` in `step`s) and flag
    /// cells whose interior differs from `f` by more than `tol` (compared after
    /// clamping to 0..1 when `clamped`). Lookups are answered inside `[lo, hi]`.
    #[allow(clippy::too_many_arguments)]
    fn build(n: [usize; 3], origin: [f32; 3], step: [f32; 3], lo: [f32; 3], hi: [f32; 3], f: &ColorFn, tol: f32, clamped: bool) -> Self {
        let inv_step: [f32; 3] = std::array::from_fn(|i| 1.0 / step[i]);
        let idx = |r: usize, g: usize, b: usize| (b * n[1] + g) * n[0] + r;
        let data: Vec<[f32; 3]> = (0..n[0] * n[1] * n[2])
            .into_par_iter()
            .map(|i| {
                let (r, g, b) = (i % n[0], (i / n[0]) % n[1], i / (n[0] * n[1]));
                f([origin[0] + r as f32 * step[0], origin[1] + g as f32 * step[1], origin[2] + b as f32 * step[2]])
            })
            .collect();
        let cells = [n[0] - 1, n[1] - 1, n[2] - 1];
        let cmp = |x: [f32; 3]| if clamped { clamp01(x) } else { x };
        const FR: [f32; 4] = [0.125, 0.375, 0.625, 0.875];
        let flags: Vec<bool> = (0..cells[0] * cells[1] * cells[2])
            .into_par_iter()
            .map(|ci| {
                let (x0, y0, z0) = (ci % cells[0], (ci / cells[0]) % cells[1], ci / (cells[0] * cells[1]));
                let corners: [[f32; 3]; 8] = std::array::from_fn(|k| data[idx(x0 + (k & 1), y0 + ((k >> 1) & 1), z0 + ((k >> 2) & 1))]);
                for tz in FR {
                    for ty in FR {
                        for tx in FR {
                            let p = [
                                origin[0] + (x0 as f32 + tx) * step[0],
                                origin[1] + (y0 as f32 + ty) * step[1],
                                origin[2] + (z0 as f32 + tz) * step[2],
                            ];
                            if max_abs_diff(cmp(tetrahedral(&corners, tx, ty, tz)), cmp(f(p))) > tol {
                                return true;
                            }
                        }
                    }
                }
                false
            })
            .collect();
        let mut exact = vec![0u64; flags.len().div_ceil(64)];
        let mut exact_cells = 0;
        for (i, f) in flags.iter().enumerate() {
            if *f {
                exact[i / 64] |= 1 << (i % 64);
                exact_cells += 1;
            }
        }
        Self { n, origin, inv_step, lo, hi, data, nominal: n[0].min(n[1]).min(n[2]), exact, exact_cells }
    }

    /// A lattice over exactly `[lo, hi]` with `n` nodes per axis.
    fn build_box(n: usize, lo: [f32; 3], hi: [f32; 3], f: &ColorFn, tol: f32, clamped: bool) -> Self {
        let step: [f32; 3] = std::array::from_fn(|i| (hi[i] - lo[i]) / (n - 1) as f32);
        let mut l = Self::build([n; 3], lo, step, lo, hi, f, tol, clamped);
        l.nominal = n;
        l
    }

    /// A lattice covering `[lo, hi]` with one step `(max extent) / (n - 1)` on every axis
    /// and node 0 of axis `c` at `-shift[c] + k·step` (so the planes `x_c = -shift[c]` are
    /// cell faces and the planes `x_a - x_b = shift[b] - shift[a]` are tetrahedron faces).
    fn build_aligned(n: usize, lo: [f32; 3], hi: [f32; 3], shift: [f32; 3], f: &ColorFn, tol: f32, clamped: bool) -> Self {
        let extent = (0..3).map(|i| hi[i] - lo[i]).fold(0.0f32, f32::max).max(1e-3);
        let h = extent / (n - 1) as f32;
        let mut origin = [0.0f32; 3];
        let mut counts = [0usize; 3];
        for i in 0..3 {
            let k = ((lo[i] + shift[i]) / h).floor();
            origin[i] = -shift[i] + k * h;
            counts[i] = (((hi[i] - origin[i]) / h).ceil() as usize + 1).max(2);
        }
        let mut l = Self::build(counts, origin, [h; 3], lo, hi, f, tol, clamped);
        l.nominal = n;
        l
    }

    /// Nominal nodes per axis (33 or 65).
    pub fn size(&self) -> usize {
        self.nominal
    }

    /// Fraction of cells graded exactly.
    pub fn exact_fraction(&self) -> f64 {
        let cells = (self.n[0] - 1) * (self.n[1] - 1) * (self.n[2] - 1);
        self.exact_cells as f64 / cells.max(1) as f64
    }

    /// Interpolated value, or `None` when `c` is outside the box or in an exact cell.
    #[inline]
    pub fn lookup(&self, c: [f32; 3]) -> Option<[f32; 3]> {
        let mut idx = [0usize; 3];
        let mut t = [0f32; 3];
        for i in 0..3 {
            if !(c[i] >= self.lo[i] && c[i] <= self.hi[i]) {
                return None; // outside (or NaN)
            }
            let f = (c[i] - self.origin[i]) * self.inv_step[i];
            let k = (f.max(0.0) as usize).min(self.n[i] - 2);
            idx[i] = k;
            t[i] = f - k as f32;
        }
        let cells = [self.n[0] - 1, self.n[1] - 1];
        let ci = (idx[2] * cells[1] + idx[1]) * cells[0] + idx[0];
        if self.exact[ci / 64] & (1 << (ci % 64)) != 0 {
            return None;
        }
        let (nx, nxy) = (self.n[0], self.n[0] * self.n[1]);
        let base = idx[2] * nxy + idx[1] * nx + idx[0];
        let d = &self.data;
        let corners = [d[base], d[base + 1], d[base + nx], d[base + nx + 1], d[base + nxy], d[base + nxy + 1], d[base + nxy + nx], d[base + nxy + nx + 1]];
        Some(tetrahedral(&corners, t[0], t[1], t[2]))
    }

    pub fn domain(&self) -> ([f32; 3], [f32; 3]) {
        (self.lo, self.hi)
    }

    fn values(&self) -> &[[f32; 3]] {
        &self.data
    }
}

/// A clip's grade baked into lattices (see the module docs).
#[derive(Debug, Clone)]
pub struct BakedGrade {
    /// `pre` stages over `[0,1]³` (`None` when they are neutral)
    pre: Option<Lattice>,
    /// `tone` over the range `pre` (+ the sharpening term) can reach; `None` when too many
    /// of its cells need the exact path for the lookup to pay off
    main: Option<Lattice>,
    /// share of the `tone` lattice's cells that need the exact path
    tone_exact: f64,
    /// `2 · sharpness` (0 = no sharpening)
    detail_k: f32,
    /// measured on the probe set, final (clamped) colour
    pub probe_max_err: f32,
    pub probe_mean_err: f32,
}

impl BakedGrade {
    /// Lattice size (33 or 65).
    pub fn size(&self) -> usize {
        self.main.as_ref().or(self.pre.as_ref()).map(Lattice::size).unwrap_or(0)
    }

    /// Fraction of the `tone` lattice's cells that need the exact path.
    pub fn exact_fraction(&self) -> f64 {
        self.tone_exact
    }

    /// Is the `tone` part baked (else only `pre` is, and `tone` is graded exactly)?
    pub fn tone_baked(&self) -> bool {
        self.main.is_some()
    }

    /// Is baking worth it for this grade? Only when stages that are expensive
    /// per pixel are active (HSL, brilliance, the exposure roll-off, gamma):
    /// two lattice lookups cost more than a handful of multiply-adds.
    pub fn worthwhile(g: &GradeParams) -> bool {
        !g.is_identity() && (g.hsl_active_count() > 0 || g.brilliance != 0.0 || g.exposure_mul > 1.0 || g.gamma_exp != [1.0; 3])
    }

    /// Bake `g` (33³ or 65³ by measured error). A bake whose probe error comes
    /// close to [`MAX_BAKE_ERROR`] is rebuilt with a tighter cell tolerance (more
    /// exact cells); `None` when even that does not meet the bound.
    pub fn bake(g: &GradeParams) -> Option<Self> {
        let first = Self::bake_with(g, 33, CELL_TOLERANCE);
        let small_ok = first.tone_exact <= MAX_EXACT_FRACTION_33
            && first.pre.as_ref().map(|p| p.exact_fraction() <= MAX_EXACT_FRACTION_33).unwrap_or(true)
            && first.probe_max_err <= PROBE_TARGET;
        if small_ok {
            return Some(first);
        }
        let mut tol = CELL_TOLERANCE;
        for _ in 0..3 {
            let b = Self::bake_with(g, 65, tol);
            if b.pre.is_none() && b.main.is_none() {
                return None; // nothing worth a lookup
            }
            if b.probe_max_err <= PROBE_TARGET {
                return Some(b);
            }
            tol *= 0.5;
        }
        None
    }

    /// Bake with a fixed lattice size and cell tolerance (tests / diagnostics).
    pub fn bake_with(g: &GradeParams, n: usize, tol: f32) -> Self {
        let k = g.detail_weight();
        let pre = g.has_pre().then(|| Lattice::build_box(n, [0.0; 3], [1.0; 3], &|c| g.apply_pre(c), tol * 0.4, false));
        // the range `pre` can produce (+ room for the detail term)
        let (mut lo, mut hi) = ([0.0f32; 3], [1.0f32; 3]);
        if let Some(p) = &pre {
            lo = [f32::INFINITY; 3];
            hi = [f32::NEG_INFINITY; 3];
            for v in p.values() {
                for i in 0..3 {
                    lo[i] = lo[i].min(v[i]);
                    hi[i] = hi[i].max(v[i]);
                }
            }
        }
        let margin = if k > 0.0 { DETAIL_MARGIN } else { 1e-3 };
        for i in 0..3 {
            lo[i] -= margin;
            hi[i] += margin;
        }
        let main = Lattice::build_aligned(n, lo, hi, g.wb_offsets(), &|c| g.apply_tone(c), tol, true);
        let tone_exact = main.exact_fraction();
        let main = (tone_exact <= MAX_TONE_EXACT).then_some(main);
        let mut baked = Self { pre, main, tone_exact, detail_k: k, probe_max_err: 0.0, probe_mean_err: 0.0 };
        let (max, mean) = baked.measure(g);
        baked.probe_max_err = max;
        baked.probe_mean_err = mean;
        baked
    }

    /// End-to-end error against the exact grade on a jittered 29³ grid plus
    /// 16384 pseudo-random colours (with a small random detail term when sharpening).
    fn measure(&self, g: &GradeParams) -> (f32, f32) {
        let m = 29;
        let mut pts: Vec<([f32; 3], [f32; 3])> = Vec::with_capacity(m * m * m + 16384);
        for b in 0..m {
            for gg in 0..m {
                for r in 0..m {
                    let j = |i: usize| ((i as f32 + 0.37) / m as f32).min(1.0);
                    pts.push(([j(r), j(gg), j(b)], [0.0; 3]));
                }
            }
        }
        let mut seed = 0x9E37_79B9u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32) / (u32::MAX as f32)
        };
        for _ in 0..16384 {
            let c = [rnd(), rnd(), rnd()];
            let d = if self.detail_k > 0.0 { [(rnd() - 0.5) * 0.1, (rnd() - 0.5) * 0.1, (rnd() - 0.5) * 0.1] } else { [0.0; 3] };
            pts.push((c, d));
        }
        let errs: Vec<f32> = pts
            .par_iter()
            .map(|(c, d)| max_abs_diff(self.apply(g, *c, *d, [0.5, 0.5], 0.0), g.apply(*c, *d, [0.5, 0.5], 0.0)))
            .collect();
        let max = errs.iter().cloned().fold(0.0, f32::max);
        let mean = errs.iter().sum::<f32>() / errs.len() as f32;
        (max, mean)
    }

    /// The grade of one pixel, like [`GradeParams::apply`].
    #[inline]
    pub fn apply(&self, g: &GradeParams, c: [f32; 3], detail: [f32; 3], canvas_uv: [f32; 2], time_ms: f32) -> [f32; 3] {
        let mut x = match &self.pre {
            Some(p) => p.lookup(c).unwrap_or_else(|| g.apply_pre(c)),
            None => c,
        };
        if self.detail_k > 0.0 {
            for (v, d) in x.iter_mut().zip(detail) {
                *v += d * self.detail_k;
            }
        }
        let m = match &self.main {
            Some(l) => l.lookup(x).unwrap_or_else(|| g.apply_tone(x)),
            None => g.apply_tone(x),
        };
        g.apply_post(g.apply_looks(m), canvas_uv, time_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{effective_grade, ColorGrade, HslOffset};
    use crate::presets::UniversalPresetFile;
    use crate::render::lut::Lut3D;
    use std::sync::Arc;

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.f()
        }
        /// a slider value, zero half of the time
        fn slider(&mut self, lo: f64, hi: f64) -> f64 {
            if self.f() < 0.5 {
                0.0
            } else {
                self.range(lo, hi).round()
            }
        }
    }

    fn random_grade(seed: u64) -> (ColorGrade, Option<Arc<Lut3D>>) {
        let mut r = Rng(seed);
        let mut g = ColorGrade {
            exposure: r.slider(-50.0, 50.0),
            brilliance: r.slider(-50.0, 50.0),
            contrast: r.slider(-50.0, 50.0),
            brightness: r.slider(-30.0, 30.0),
            highlights: r.slider(-50.0, 50.0),
            shadows: r.slider(-50.0, 50.0),
            saturation: r.slider(-50.0, 50.0),
            vibrance: r.slider(-50.0, 50.0),
            sharpness: r.slider(0.0, 50.0),
            temperature: r.slider(-50.0, 50.0),
            tint: r.slider(-50.0, 50.0),
            vignette: if r.f() < 0.5 { r.range(0.0, 1.0) } else { 0.0 },
            grain: if r.f() < 0.3 { r.range(0.0, 0.5) } else { 0.0 },
            ..Default::default()
        };
        if r.f() < 0.5 {
            g.lift = [r.range(-0.1, 0.1), r.range(-0.1, 0.1), r.range(-0.1, 0.1)];
            g.gain = [r.range(-0.2, 0.2), r.range(-0.2, 0.2), r.range(-0.2, 0.2)];
        }
        if r.f() < 0.5 {
            g.gamma = [r.range(-0.3, 0.3), r.range(-0.3, 0.3), r.range(-0.3, 0.3)];
        }
        for ch in [&mut g.hsl.red, &mut g.hsl.green, &mut g.hsl.blue, &mut g.hsl.orange, &mut g.hsl.purple] {
            if r.f() < 0.6 {
                *ch = HslOffset { h: r.range(-100.0, 100.0).round(), s: r.range(-100.0, 100.0).round(), l: r.range(-100.0, 100.0).round() };
            }
        }
        if r.f() < 0.5 {
            g.curves.master = vec![[0.0, 0.0], [0.25, r.range(0.15, 0.35)], [0.75, r.range(0.65, 0.85)], [1.0, 1.0]];
            g.curves.g = vec![[0.0, r.range(0.0, 0.1)], [0.5, r.range(0.4, 0.6)], [1.0, r.range(0.9, 1.0)]];
        }
        let lut = (r.f() < 0.5).then(|| {
            let mut l = Lut3D::identity(17);
            for v in l.data.iter_mut() {
                *v = [(v[0] * 1.08 + 0.02).min(1.0), v[1].powf(0.9), v[2] * 0.9];
            }
            Arc::new(l)
        });
        if lut.is_some() {
            g.lut_asset_id = Some("lut".into());
            g.lut_intensity = r.range(0.3, 1.0);
        }
        (g, lut)
    }

    /// Final colours of the baked vs the exact grade over a dense 49³ grid of
    /// source colours (0..1 inclusive), with and without a sharpening detail term.
    fn compare_dense(g: &GradeParams, baked: &BakedGrade) -> (f32, f32) {
        let m = 49;
        let errs: Vec<f32> = (0..m * m * m)
            .into_par_iter()
            .map(|i| {
                let c = [(i % m) as f32 / (m - 1) as f32, ((i / m) % m) as f32 / (m - 1) as f32, (i / (m * m)) as f32 / (m - 1) as f32];
                // deterministic small detail (edges) for every other colour
                let d = if i % 2 == 0 { [0.0; 3] } else { [((i * 7) % 11) as f32 / 110.0 - 0.05, ((i * 5) % 13) as f32 / 130.0 - 0.05, ((i * 3) % 7) as f32 / 70.0 - 0.05] };
                let uv = [((i * 31) % 97) as f32 / 97.0, ((i * 17) % 89) as f32 / 89.0];
                max_abs_diff(baked.apply(g, c, d, uv, 1234.0), g.apply(c, d, uv, 1234.0))
            })
            .collect();
        (errs.iter().cloned().fold(0.0, f32::max), errs.iter().sum::<f32>() / errs.len() as f32)
    }

    #[test]
    fn baked_grade_matches_exact_grade() {
        let universal = UniversalPresetFile::default().to_adjust();
        let mut cases: Vec<(String, ColorGrade, Option<Arc<Lut3D>>)> =
            vec![("universal preset on a neutral clip".into(), effective_grade(&ColorGrade::default(), Some(&universal)), None)];
        let mut no_sharp = cases[0].1.clone();
        no_sharp.sharpness = 0.0;
        cases.push(("universal preset without sharpening".into(), no_sharp, None));
        for seed in 1..=6u64 {
            let (g, lut) = random_grade(seed * 7919);
            cases.push((format!("random grade #{seed}"), g, lut));
        }
        for (name, grade, lut) in cases {
            let p = GradeParams::new(&grade, lut);
            let baked = BakedGrade::bake(&p).unwrap_or_else(|| panic!("{name}: bake rejected"));
            let (max, mean) = compare_dense(&p, &baked);
            eprintln!(
                "BAKE {name}: {}³, exact cells {:.1}%, probe max {:.3}/255 — dense grid max {:.3}/255, mean {:.4}/255",
                baked.size(),
                baked.exact_fraction() * 100.0,
                baked.probe_max_err * 255.0,
                max * 255.0,
                mean * 255.0
            );
            assert!(max <= MAX_BAKE_ERROR + 1e-6, "{name}: max error {:.3}/255", max * 255.0);
        }
    }

    /// How a real 720p frame of the user's footage (`<clips folder>/clip1.mp4`) goes through
    /// the baked universal preset: share of pixels on the exact path, per-pixel and
    /// composite cost, exact vs baked. Run in release.
    #[test]
    #[ignore = "diagnostic on the user's clips (release)"]
    fn diag_real_frame() {
        use crate::render::compositor::{composite_layer, Canvas, Layer};
        use crate::render::sample::{Frame, Placement, SourceImage, UvMatrix};
        let Some(repo) = crate::clips::repo_dir() else { return };
        let clip = crate::clips::discover_clips_dir(&repo).join("clip1.mp4");
        let (Ok(bins), true) = (crate::ffmpeg::find_binaries(), clip.is_file()) else {
            eprintln!("SKIP: no clip1.mp4");
            return;
        };
        let out = crate::procs::output_timeout(
            crate::ffmpeg::command(&bins.ffmpeg).args(["-v", "error", "-ss", "5", "-i"]).arg(&clip).args(["-frames:v", "1", "-vf", "scale=1280:720", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]),
            std::time::Duration::from_secs(60),
        )
        .unwrap();
        let f = Frame { width: 1280, height: 720, data: out.stdout };
        let det = f.sharpen_detail_into(1.0, 1.0, Vec::new());
        let universal = UniversalPresetFile::default().to_adjust();
        let p = GradeParams::new(&effective_grade(&ColorGrade::default(), Some(&universal)), None);
        let k = p.detail_weight();
        let px: Vec<([f32; 3], [f32; 3])> = (0..1280 * 720)
            .map(|i| {
                let c: [f32; 3] = std::array::from_fn(|j| f.data[i * 3 + j] as f32 / 255.0);
                (c, std::array::from_fn(|j| det.data[i * 3 + j]))
            })
            .collect();
        {
            let full = effective_grade(&ColorGrade::default(), Some(&universal));
            let mut no_hsl = full.clone();
            no_hsl.hsl = Default::default();
            let mut hsl_only = ColorGrade { sharpness: full.sharpness, ..Default::default() };
            hsl_only.hsl = full.hsl;
            let src = SourceImage::Bytes(Arc::new(f.clone()));
            let pl = Placement { canvas_w: 1280.0, canvas_h: 720.0, source_w: 1280.0, source_h: 720.0, crop: None, scale: 1.0, position: [0.0, 0.0], rotation_deg: 0.0 };
            let mut canvas = Canvas::new(1280, 720);
            for (name, g) in [("identity", ColorGrade::default()), ("sharpen only", ColorGrade { sharpness: 40.0, ..Default::default() }), ("universal w/o hsl", no_hsl), ("sharpen + hsl", hsl_only), ("universal", full)] {
                let gp = GradeParams::new(&g, None);
                let t = std::time::Instant::now();
                for _ in 0..20 {
                    canvas.clear();
                    let layer = Layer { source: &src, detail: gp.needs_detail().then_some(&det), uv: UvMatrix::new(&pl), grade: &gp, baked: None, opacity: 1.0, mask: None, blend: crate::model::BlendMode::Normal, time_ms: 0.0, fade: 0.0 };
                    composite_layer(&mut canvas, &layer);
                }
                eprintln!("DIAG composite 720p exact {name}: {:.2} ms/frame", t.elapsed().as_secs_f64() * 1000.0 / 20.0);
            }
        }
        for b in [BakedGrade::bake(&p).unwrap(), BakedGrade::bake_with(&p, 33, CELL_TOLERANCE)] {
        eprintln!("DIAG lattice {}³ (probe max {:.2}/255)", b.size(), b.probe_max_err * 255.0);
        let exact_path = px
            .iter()
            .filter(|(c, d)| {
                let mut x = b.pre.as_ref().and_then(|l| l.lookup(*c)).unwrap_or_else(|| p.apply_pre(*c));
                for j in 0..3 {
                    x[j] += d[j] * k;
                }
                b.main.as_ref().map(|m| m.lookup(x).is_none()).unwrap_or(true)
            })
            .count();
        eprintln!("DIAG real frame: {:.1}% of pixels take the exact tone path ({:.1}% of cells)", exact_path as f64 * 100.0 / px.len() as f64, b.exact_fraction() * 100.0);
        for mode in 0..2 {
            let t = std::time::Instant::now();
            let mut acc = 0.0;
            for (c, d) in &px {
                let o = if mode == 0 { p.apply(*c, *d, [0.5, 0.5], 0.0) } else { b.apply(&p, *c, *d, [0.5, 0.5], 0.0) };
                acc += o[0];
            }
            eprintln!("DIAG {}: {:.1} ns/px ({acc:.0})", if mode == 0 { "exact" } else { "baked" }, t.elapsed().as_secs_f64() * 1e9 / px.len() as f64);
        }
        let src = SourceImage::Bytes(Arc::new(f.clone()));
        let pl = Placement { canvas_w: 1280.0, canvas_h: 720.0, source_w: 1280.0, source_h: 720.0, crop: None, scale: 1.0, position: [0.0, 0.0], rotation_deg: 0.0 };
        let mut canvas = Canvas::new(1280, 720);
        for (name, baked) in [("exact", None), ("baked", Some(&b))] {
            let t = std::time::Instant::now();
            for _ in 0..20 {
                canvas.clear();
                let layer = Layer { source: &src, detail: Some(&det), uv: UvMatrix::new(&pl), grade: &p, baked, opacity: 1.0, mask: None, blend: crate::model::BlendMode::Normal, time_ms: 0.0, fade: 0.0 };
                composite_layer(&mut canvas, &layer);
            }
            eprintln!("DIAG composite 720p {name}: {:.2} ms/frame", t.elapsed().as_secs_f64() * 1000.0 / 20.0);
        }
        }
    }

    #[test]
    fn identity_and_cheap_grades_are_not_baked() {
        assert!(!BakedGrade::worthwhile(&GradeParams::new(&ColorGrade::default(), None)));
        assert!(!BakedGrade::worthwhile(&GradeParams::new(&ColorGrade { contrast: 10.0, ..Default::default() }, None)));
        let cheap = ColorGrade { contrast: 20.0, temperature: 25.0, saturation: -10.0, lift: [0.03, 0.01, 0.0], ..Default::default() };
        assert!(!BakedGrade::worthwhile(&GradeParams::new(&cheap, None)), "cheap stages are faster exact");
        let mut h = ColorGrade::default();
        h.hsl.green.s = 20.0;
        assert!(BakedGrade::worthwhile(&GradeParams::new(&h, None)));
    }

    #[test]
    #[ignore = "micro-benchmark (release)"]
    fn bench_baked_vs_exact() {
        let universal = UniversalPresetFile::default().to_adjust();
        let p = GradeParams::new(&effective_grade(&ColorGrade::default(), Some(&universal)), None);
        let t0 = std::time::Instant::now();
        let baked = BakedGrade::bake(&p).unwrap();
        eprintln!("bake: {:.1} ms ({}³)", t0.elapsed().as_secs_f64() * 1000.0, baked.size());
        let px: Vec<[f32; 3]> = (0..2_000_000u32).map(|i| [((i * 13) % 255) as f32 / 255.0, ((i * 7) % 255) as f32 / 255.0, ((i * 3) % 255) as f32 / 255.0]).collect();
        for (name, f) in [("exact", 0), ("baked", 1)] {
            let t0 = std::time::Instant::now();
            let mut acc = 0.0;
            for c in &px {
                let o = if f == 0 { p.apply(*c, [0.01; 3], [0.5, 0.5], 0.0) } else { baked.apply(&p, *c, [0.01; 3], [0.5, 0.5], 0.0) };
                acc += o[0];
            }
            eprintln!("{name}: {:.1} ns/px ({acc:.0})", t0.elapsed().as_secs_f64() * 1e9 / px.len() as f64);
        }
    }
}
