//! CPU port of the preview's GLSL colour pipeline (`src/engine/color/shaders.ts`).
//!
//! Same stage order and the same maths in f32, working directly on the
//! (display-referred, not linearised) 0..1 RGB values like the shader does:
//!
//! exposure → brightness → contrast → highlights/shadows → sharpness →
//! temperature/tint → lift/gamma/gain/offset → saturation/vibrance →
//! 8-channel HSL → RGB curves → 3D LUT → vignette → grain → clamp.
//!
//! [`GradeParams`] pre-normalises the slider values once per clip (the same
//! `/100` scaling `renderer.ts` applies to the uniforms), bakes the curves to
//! 256-entry tables and records which stages are neutral so the per-pixel
//! function can skip them. Skipped stages are exact no-ops except the RGB
//! curves: the preview samples an 8-bit 256-texel texture even for identity
//! curves (an error of < 1/255), the exporter skips identity curves entirely.

use super::lut::Lut3D;
use crate::model::{ColorCurves, ColorGrade, CurvePoint};
use std::sync::Arc;

pub const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];
/// HSL hue slider scale (CapCut): +-100 on a channel rotates its hue by +-30 degrees.
pub const HSL_HUE_TURNS: f32 = 30.0 / 360.0;
/// Adjust sliders use CapCut's scale: -50..50 (sharpness 0..50) map to -1..1.
pub const ADJUST_SCALE: f64 = 50.0;
const HSL_CENTERS: [f32; 8] = [0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0];

#[inline]
pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn fract(x: f32) -> f32 {
    x - x.floor()
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// GLSL `rgb2hsv` from the shader (the `mix`/`step` selects unrolled).
#[inline]
pub fn rgb2hsv(c: [f32; 3]) -> [f32; 3] {
    let (r, g, b) = (c[0], c[1], c[2]);
    // K = (0, -1/3, 2/3, -1)
    let p = if g >= b { [g, b, 0.0, -1.0 / 3.0] } else { [b, g, -1.0, 2.0 / 3.0] };
    let q = if r >= p[0] { [r, p[1], p[2], p[0]] } else { [p[0], p[1], p[3], r] };
    let d = q[0] - q[3].min(q[1]);
    let e = 1.0e-10;
    [(q[2] + (q[3] - q[1]) / (6.0 * d + e)).abs(), d / (q[0] + e), q[0]]
}

/// GLSL `hsv2rgb` from the shader.
#[inline]
pub fn hsv2rgb(c: [f32; 3]) -> [f32; 3] {
    let k = [1.0f32, 2.0 / 3.0, 1.0 / 3.0];
    let mut out = [0.0; 3];
    for i in 0..3 {
        let p = (fract(c[0] + k[i]) * 6.0 - 3.0).abs();
        let v = (p - 1.0).clamp(0.0, 1.0);
        out[i] = c[2] * (1.0 + (v - 1.0) * c[1]);
    }
    out
}

/// Highlight roll-off applied after a positive exposure (mirrors the GLSL): values above 0.8
/// ease towards 1.0 with a tanh shoulder (slope 1 at the knee) instead of clipping.
#[inline]
pub fn exposure_rolloff(v: f32) -> f32 {
    const KNEE: f32 = 0.8;
    if v <= KNEE {
        v
    } else {
        KNEE + (1.0 - KNEE) * ((v - KNEE) / (1.0 - KNEE)).tanh()
    }
}

/// Brilliance (mirrors the GLSL in `src/engine/color/shaders.ts`): lift shadows and compress
/// highlights along the luma axis, scaling the colour to keep its chroma, then a slight
/// saturation lift so the result doesn't look washed out. `b` in -1..1.
#[inline]
pub fn brilliance(c: [f32; 3], b: f32) -> [f32; 3] {
    let l = dot3(c, LUMA).clamp(0.0, 1.0);
    let lift = 0.6 * l * (1.0 - l) * (1.0 - l);
    let comp = 0.4 * l * l * (1.0 - l);
    let nl = l + b * (lift - comp);
    let ratio = if l > 1e-4 { nl / l } else { 1.0 };
    let mut o = [c[0] * ratio, c[1] * ratio, c[2] * ratio];
    if l <= 1e-4 {
        o = [c[0] + nl, c[1] + nl, c[2] + nl];
    }
    let l2 = dot3(o, LUMA);
    let k = 1.0 + 0.15 * b;
    [l2 + (o[0] - l2) * k, l2 + (o[1] - l2) * k, l2 + (o[2] - l2) * k]
}

/// GLSL `hash` used for film grain.
#[inline]
pub fn hash(p: [f32; 2]) -> f32 {
    fract((p[0] * 12.9898 + p[1] * 78.233).sin() * 43758.547)
}

/* ------------------------------------------------------------------ curves */

/// Monotone cubic through the control points, evaluated on 0..1 and clamped
/// (port of `evalCurve` in `src/engine/color/curves.ts`).
pub fn eval_curve(points: &[CurvePoint], x: f64) -> f64 {
    let mut pts: Vec<CurvePoint> = points.to_vec();
    pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
    let n = pts.len();
    if n == 0 {
        return x;
    }
    if n == 1 {
        return pts[0][1];
    }
    if x <= pts[0][0] {
        return pts[0][1];
    }
    if x >= pts[n - 1][0] {
        return pts[n - 1][1];
    }
    let xs: Vec<f64> = pts.iter().map(|p| p[0]).collect();
    let ys: Vec<f64> = pts.iter().map(|p| p[1]).collect();
    let mut h = Vec::with_capacity(n - 1);
    let mut d = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        h.push(xs[i + 1] - xs[i]);
        d.push(if h[i] == 0.0 { 0.0 } else { (ys[i + 1] - ys[i]) / h[i] });
    }
    let mut m = vec![0.0; n];
    m[0] = d[0];
    m[n - 1] = d[n - 2];
    for i in 1..n - 1 {
        m[i] = if d[i - 1] * d[i] <= 0.0 { 0.0 } else { (d[i - 1] + d[i]) / 2.0 };
    }
    for i in 0..n - 1 {
        if d[i] == 0.0 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
            continue;
        }
        let a = m[i] / d[i];
        let b = m[i + 1] / d[i];
        let s = a * a + b * b;
        if s > 9.0 {
            let tau = 3.0 / s.sqrt();
            m[i] = tau * a * d[i];
            m[i + 1] = tau * b * d[i];
        }
    }
    let mut i = 0;
    while i < n - 2 && x > xs[i + 1] {
        i += 1;
    }
    let u = (x - xs[i]) / h[i];
    let (u2, u3) = (u * u, u * u * u);
    let y = (2.0 * u3 - 3.0 * u2 + 1.0) * ys[i]
        + (u3 - 2.0 * u2 + u) * h[i] * m[i]
        + (-2.0 * u3 + 3.0 * u2) * ys[i + 1]
        + (u3 - u2) * h[i] * m[i + 1];
    y.clamp(0.0, 1.0)
}

pub fn is_identity_curve(points: &[CurvePoint]) -> bool {
    points.iter().all(|p| (p[0] - p[1]).abs() < 1e-6)
}

/// One baked curve row: 256 texels quantised to 8 bits like the preview's
/// RGBA8 texture, sampled with LINEAR filtering + CLAMP_TO_EDGE.
#[derive(Debug, Clone)]
pub struct CurveRow(pub [f32; 256]);

impl CurveRow {
    pub fn bake(points: &[CurvePoint]) -> Self {
        let mut t = [0.0f32; 256];
        for (x, v) in t.iter_mut().enumerate() {
            *v = (eval_curve(points, x as f64 / 255.0) * 255.0).round() as f32 / 255.0;
        }
        Self(t)
    }

    /// `texture(u_curves, vec2(c, row)).r`
    #[inline]
    pub fn lookup(&self, c: f32) -> f32 {
        let pos = (c * 256.0 - 0.5).clamp(0.0, 255.0);
        let i = pos.floor() as usize;
        let j = (i + 1).min(255);
        let f = pos - i as f32;
        self.0[i] + (self.0[j] - self.0[i]) * f
    }
}

/// Resolution of the fused per-channel curve tables.
const FUSED: usize = 4096;

/// The two texture lookups per channel (`rgb` row, then `master` row) fused
/// into one 4096-entry table per channel, read with linear interpolation.
/// The fused table is sampled from the exact two-lookup composition, so the
/// only difference is the re-sampling (< 1e-4 on 0..1).
#[derive(Debug, Clone)]
pub struct CurveTables {
    pub master: CurveRow,
    pub rgb: [CurveRow; 3],
    fused: Vec<[f32; 3]>,
}

impl CurveTables {
    /// `None` when every curve is the identity.
    pub fn bake(c: &ColorCurves) -> Option<Self> {
        if [&c.master, &c.r, &c.g, &c.b].iter().all(|p| is_identity_curve(p)) {
            return None;
        }
        let master = CurveRow::bake(&c.master);
        let rgb = [CurveRow::bake(&c.r), CurveRow::bake(&c.g), CurveRow::bake(&c.b)];
        let fused = (0..FUSED)
            .map(|k| {
                let x = k as f32 / (FUSED - 1) as f32;
                std::array::from_fn(|i| master.lookup(rgb[i].lookup(x)))
            })
            .collect();
        Some(Self { master, rgb, fused })
    }

    /// Exact two-lookup evaluation (reference for the fused table).
    pub fn apply_exact(&self, c: [f32; 3]) -> [f32; 3] {
        std::array::from_fn(|i| self.master.lookup(self.rgb[i].lookup(c[i].clamp(0.0, 1.0))))
    }

    #[inline]
    pub fn apply(&self, c: [f32; 3]) -> [f32; 3] {
        let n = (FUSED - 1) as f32;
        let mut out = [0.0; 3];
        for i in 0..3 {
            let p = c[i].clamp(0.0, 1.0) * n;
            let k = (p as usize).min(FUSED - 2);
            let f = p - k as f32;
            let (a, b) = (self.fused[k][i], self.fused[k + 1][i]);
            out[i] = a + (b - a) * f;
        }
        out
    }
}

/* ------------------------------------------------------------------ params */

/// Per-clip, pre-normalised grade (uniform values as set by `renderer.ts`).
#[derive(Debug, Clone)]
pub struct GradeParams {
    pub exposure_mul: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
    /// -1..1
    pub brilliance: f32,
    pub sharpness: f32,
    pub temperature: f32,
    pub tint: f32,
    pub lift: [f32; 3],
    pub gain: [f32; 3],
    pub offset: [f32; 3],
    /// `1 / max(1 + gamma, 0.05)`
    pub gamma_exp: [f32; 3],
    pub saturation: f32,
    pub vibrance: f32,
    /// h, s, l per channel (-1..1)
    pub hsl: [[f32; 3]; 8],
    pub curves: Option<CurveTables>,
    pub lut: Option<Arc<Lut3D>>,
    pub lut_intensity: f32,
    pub vignette: f32,
    pub grain: f32,
    // stage switches
    use_primary: bool,
    use_hs: bool,
    use_wb: bool,
    use_wheels: bool,
    use_gamma: bool,
    use_sat: bool,
    use_hsl: bool,
    /// (centre in degrees, h, s, l) of the HSL channels that are not neutral
    hsl_active: Vec<(f32, [f32; 3])>,
    /// `Σ offset · (1 - smoothstep(0, 35°, hue distance))` per hue, `HUE_STEPS` entries over
    /// one turn (the per-channel loop of the shader, tabulated: < 1e-5 apart)
    hsl_table: Vec<[f32; 3]>,
}

/// Resolution of the HSL hue table (≈ 0.09° per entry).
const HUE_STEPS: usize = 4096;

impl GradeParams {
    pub fn new(g: &ColorGrade, lut: Option<Arc<Lut3D>>) -> Self {
        // adjust sliders use CapCut's scale (-50..50, sharpness 0..50); HSL uses -100..100
        let a = |v: f64| (v / ADJUST_SCALE) as f32;
        let n = |v: f64| (v / 100.0) as f32;
        let v3 = |v: [f64; 3]| [v[0] as f32, v[1] as f32, v[2] as f32];
        let h = &g.hsl;
        let hsl = [h.red, h.orange, h.yellow, h.green, h.cyan, h.blue, h.purple, h.magenta].map(|o| [n(o.h), n(o.s), n(o.l)]);
        let gamma = v3(g.gamma);
        let gamma_exp = gamma.map(|x| 1.0 / (1.0 + x).max(0.05));
        let mut p = Self {
            exposure_mul: (a(g.exposure) * 3.0).exp2(),
            brightness: a(g.brightness),
            contrast: a(g.contrast),
            highlights: a(g.highlights),
            shadows: a(g.shadows),
            brilliance: a(g.brilliance),
            sharpness: a(g.sharpness).max(0.0),
            temperature: a(g.temperature),
            tint: a(g.tint),
            lift: v3(g.lift),
            gain: v3(g.gain),
            offset: v3(g.offset),
            gamma_exp,
            saturation: a(g.saturation),
            vibrance: a(g.vibrance),
            hsl,
            curves: CurveTables::bake(&g.curves),
            lut_intensity: g.lut_intensity as f32,
            lut: None,
            vignette: g.vignette as f32,
            grain: g.grain as f32,
            use_primary: false,
            use_hs: false,
            use_wb: false,
            use_wheels: false,
            use_gamma: false,
            use_sat: false,
            use_hsl: false,
            hsl_active: Vec::new(),
            hsl_table: Vec::new(),
        };
        // `u_useLut` is only set when the clip names a LUT that loaded.
        p.lut = if g.lut_asset_id.is_some() { lut } else { None };
        p.use_primary = p.exposure_mul != 1.0 || p.brightness != 0.0 || p.contrast != 0.0;
        p.use_hs = p.highlights != 0.0 || p.shadows != 0.0 || p.brilliance != 0.0;
        p.use_wb = p.temperature != 0.0 || p.tint != 0.0;
        p.use_wheels = p.lift != [0.0; 3] || p.gain != [0.0; 3] || p.offset != [0.0; 3];
        p.use_gamma = p.gamma_exp != [1.0; 3];
        p.use_sat = p.saturation != 0.0 || p.vibrance != 0.0;
        p.use_hsl = p.hsl.iter().any(|c| *c != [0.0; 3]);
        // Neutral channels contribute exactly 0 to dh/ds/dl, so only the
        // active ones are evaluated per pixel.
        p.hsl_active = HSL_CENTERS.iter().zip(p.hsl.iter()).filter(|(_, v)| **v != [0.0; 3]).map(|(c, v)| (*c, *v)).collect();
        if p.use_hsl {
            p.hsl_table = (0..=HUE_STEPS).map(|k| p.hsl_weights(k as f32 / HUE_STEPS as f32 * 360.0)).collect();
        }
        p
    }

    /// Per-channel offsets the temperature / tint stage adds (0 when neutral).
    pub fn wb_offsets(&self) -> [f32; 3] {
        if !self.use_wb {
            return [0.0; 3];
        }
        [self.temperature * 0.12 + self.tint * 0.05, self.tint * -0.1, -self.temperature * 0.12 + self.tint * 0.05]
    }

    /// `(dh, ds, dl)` before the grey factor at `hue_deg` (the shader's channel loop).
    pub fn hsl_weights(&self, hue_deg: f32) -> [f32; 3] {
        let mut acc = [0.0f32; 3];
        for (center, v) in &self.hsl_active {
            let mut d = (hue_deg - center).abs();
            d = d.min(360.0 - d);
            if d >= 35.0 {
                continue; // outside the channel's smoothstep window: weight 0
            }
            let w = 1.0 - smoothstep(0.0, 35.0, d);
            for i in 0..3 {
                acc[i] += v[i] * w;
            }
        }
        acc
    }

    /// [`GradeParams::hsl_weights`] from the hue table (hue in turns, 0..1).
    #[inline]
    fn hsl_lookup(&self, hue: f32) -> [f32; 3] {
        let p = hue.clamp(0.0, 1.0) * HUE_STEPS as f32;
        let k = (p as usize).min(HUE_STEPS - 1);
        let f = p - k as f32;
        let (a, b) = (self.hsl_table[k], self.hsl_table[k + 1]);
        [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
    }

    /// Number of HSL channels that are not neutral.
    pub fn hsl_active_count(&self) -> usize {
        self.hsl_active.len()
    }

    /// Needs the sharpening detail layer (`tex(uv) - avg4(tex)`).
    pub fn needs_detail(&self) -> bool {
        self.sharpness > 0.0
    }

    /// True when the grade is an exact no-op on in-range colours.
    pub fn is_identity(&self) -> bool {
        !(self.use_primary
            || self.use_hs
            || self.use_wb
            || self.use_wheels
            || self.use_gamma
            || self.use_sat
            || self.use_hsl
            || self.sharpness > 0.0
            || self.curves.is_some()
            || self.lut.is_some()
            || self.vignette > 0.0
            || self.grain > 0.0)
    }

    /// Grade one pixel.
    ///
    /// * `c` — source colour (0..1, possibly blurred),
    /// * `detail` — sharpening detail at the same uv (ignored unless sharpness > 0),
    /// * `canvas_uv` — output-frame uv in the shader's bottom-up convention,
    /// * `time_ms` — the preview's `u_time` (timeline ms) for grain.
    ///
    /// Returns the clamped 0..1 colour (the shader's `clamp(c, 0.0, 1.0)`).
    ///
    /// The stages split into three parts: [`GradeParams::apply_pre`] (per-pixel colour
    /// function before the sharpening term), the sharpening term itself (needs the
    /// spatial detail layer), [`GradeParams::apply_mid`] (per-pixel colour function
    /// from white balance through the 3D LUT) and [`GradeParams::apply_post`]
    /// (vignette and grain, which depend on the output position).
    #[inline]
    pub fn apply(&self, c: [f32; 3], detail: [f32; 3], canvas_uv: [f32; 2], time_ms: f32) -> [f32; 3] {
        let mut c = self.apply_pre(c);
        if self.sharpness > 0.0 {
            let k = self.sharpness * 2.0;
            for (v, d) in c.iter_mut().zip(detail) {
                *v += d * k;
            }
        }
        self.apply_post(self.apply_mid(c), canvas_uv, time_ms)
    }

    /// `sharpness * 2`: the weight of the detail layer (0 when not sharpening).
    #[inline]
    pub fn detail_weight(&self) -> f32 {
        if self.sharpness > 0.0 {
            self.sharpness * 2.0
        } else {
            0.0
        }
    }

    /// Are any of the stages of [`GradeParams::apply_pre`] active?
    pub fn has_pre(&self) -> bool {
        self.use_primary || self.use_hs
    }

    /// Are any of the stages of [`GradeParams::apply_post`] active?
    pub fn has_post(&self) -> bool {
        self.vignette > 0.0 || self.grain > 0.0
    }

    /// Exposure, brightness, contrast, highlights / shadows, brilliance (no clamp).
    #[inline]
    pub fn apply_pre(&self, mut c: [f32; 3]) -> [f32; 3] {
        // ---- primary adjustments
        if self.use_primary {
            let k = 1.0 + self.contrast;
            let roll = self.exposure_mul > 1.0;
            for v in c.iter_mut() {
                let mut e = *v * self.exposure_mul;
                if roll {
                    e = exposure_rolloff(e);
                }
                *v = ((e + self.brightness * 0.5) - 0.5) * k + 0.5;
            }
        }
        if self.use_hs {
            let luma = dot3(c, LUMA);
            let add = self.highlights * 0.5 * smoothstep(0.5, 1.0, luma) + self.shadows * 0.5 * (1.0 - smoothstep(0.0, 0.5, luma));
            for v in c.iter_mut() {
                *v += add;
            }
            if self.brilliance != 0.0 {
                c = brilliance(c, self.brilliance);
            }
        }
        c
    }

    /// Temperature / tint through the 3D LUT: [`GradeParams::apply_tone`] then
    /// [`GradeParams::apply_looks`] (unclamped result).
    #[inline]
    pub fn apply_mid(&self, c: [f32; 3]) -> [f32; 3] {
        self.apply_looks(self.apply_tone(c))
    }

    /// Temperature / tint, lift / gamma / gain / offset, saturation / vibrance, clamp and
    /// HSL (the result is in 0..1).
    #[inline]
    pub fn apply_tone(&self, mut c: [f32; 3]) -> [f32; 3] {
        // ---- temperature & tint
        if self.use_wb {
            c[0] += self.temperature * 0.12 + self.tint * 0.05;
            c[1] += self.tint * -0.1;
            c[2] += -self.temperature * 0.12 + self.tint * 0.05;
        }
        // ---- lift / gamma / gain / offset (the max(c, 0) always runs in the shader)
        for (i, cv) in c.iter_mut().enumerate() {
            let mut v = *cv;
            if self.use_wheels {
                v = v * (1.0 + self.gain[i]) + self.lift[i] * (1.0 - v) + self.offset[i];
            }
            v = v.max(0.0);
            if self.use_gamma && self.gamma_exp[i] != 1.0 {
                v = v.powf(self.gamma_exp[i]);
            }
            *cv = v;
        }
        // ---- saturation & vibrance
        if self.use_sat {
            let luma = dot3(c, LUMA);
            let d = [c[0] - luma, c[1] - luma, c[2] - luma];
            let sat = dot3(d, d).sqrt();
            let vib = self.vibrance * (1.0 - (sat * 1.5).clamp(0.0, 1.0));
            let k = 1.0 + self.saturation + vib;
            for i in 0..3 {
                c[i] = luma + d[i] * k;
            }
        }
        // ---- HSL 8-colour tuning (rgb→hsv→rgb of a clamped colour is a clamp)
        for v in c.iter_mut() {
            *v = v.clamp(0.0, 1.0);
        }
        if self.use_hsl {
            let mut hsv = rgb2hsv(c);
            let grey = smoothstep(0.0, 0.25, hsv[1]);
            let w = self.hsl_lookup(hsv[0]);
            let (dh, ds, dl) = (w[0] * grey, w[1] * grey, w[2] * grey);
            hsv[0] = fract(hsv[0] + dh * HSL_HUE_TURNS);
            hsv[1] = (hsv[1] * (1.0 + ds)).clamp(0.0, 1.0);
            hsv[2] = (hsv[2] * (1.0 + dl * 0.5)).clamp(0.0, 1.0);
            c = hsv2rgb(hsv);
        }
        c
    }

    /// RGB curves and the 3D LUT (table lookups: cheap, and exact only when evaluated
    /// per pixel — the 8-bit curve texels and the LUT lattice are full of small kinks).
    #[inline]
    pub fn apply_looks(&self, mut c: [f32; 3]) -> [f32; 3] {
        // ---- RGB curves (per-channel first, then master)
        if let Some(t) = &self.curves {
            c = t.apply(c);
        }
        // ---- 3D LUT
        if let Some(lut) = &self.lut {
            let g = lut.sample(c);
            for i in 0..3 {
                c[i] += (g[i] - c[i]) * self.lut_intensity;
            }
        }
        c
    }

    /// Vignette and grain (output-frame space) and the final clamp.
    #[inline]
    pub fn apply_post(&self, mut c: [f32; 3], canvas_uv: [f32; 2], time_ms: f32) -> [f32; 3] {
        // ---- vignette & grain (output-frame space)
        if self.vignette > 0.0 {
            let (dx, dy) = (canvas_uv[0] - 0.5, canvas_uv[1] - 0.5);
            let v = 1.0 - smoothstep(0.35, 1.1, (dx * dx + dy * dy).sqrt() * 1.35);
            let k = 1.0 + (v - 1.0) * self.vignette;
            for x in c.iter_mut() {
                *x *= k;
            }
        }
        if self.grain > 0.0 {
            let t = fract(time_ms * 0.001) * 100.0;
            let n = hash([canvas_uv[0] * 1024.0 + t, canvas_uv[1] * 1024.0 + t]) - 0.5;
            for x in c.iter_mut() {
                *x += n * self.grain * 0.25;
            }
        }
        [c[0].clamp(0.0, 1.0), c[1].clamp(0.0, 1.0), c[2].clamp(0.0, 1.0)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ColorGrade;

    fn close3(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() <= eps)
    }

    const SAMPLES: [[f32; 3]; 6] = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.2, 0.4, 0.6], [0.9, 0.1, 0.3], [0.5, 0.5, 0.5], [0.33, 0.8, 0.05]];

    #[test]
    fn identity_grade_is_identity() {
        let p = GradeParams::new(&ColorGrade::default(), None);
        assert!(p.is_identity());
        for c in SAMPLES {
            assert_eq!(p.apply(c, [0.0; 3], [0.3, 0.7], 1234.0), c);
        }
    }

    #[test]
    fn exposure_plus_one_stop_doubles() {
        // exposure slider is -100..100 → ±3 stops, so 100/3 = +1 stop.
        // exposure slider is -50..50 (CapCut scale) -> +-3 stops, so 50/3 = +1 stop
        let g = ColorGrade { exposure: 50.0 / 3.0, ..Default::default() };
        let p = GradeParams::new(&g, None);
        let out = p.apply([0.1, 0.2, 0.3], [0.0; 3], [0.5, 0.5], 0.0);
        assert!(close3(out, [0.2, 0.4, 0.6], 1e-5), "{out:?}");
        // brightening rolls highlights off smoothly towards 1 instead of clipping
        let hi = p.apply([0.8, 0.8, 0.8], [0.0; 3], [0.5, 0.5], 0.0);
        assert!(hi[0] < 1.0 && hi[0] > 0.99, "{hi:?}");
        let a = p.apply([0.45; 3], [0.0; 3], [0.5, 0.5], 0.0)[0];
        let b = p.apply([0.5; 3], [0.0; 3], [0.5, 0.5], 0.0)[0];
        assert!(b > a, "roll-off stays monotonic");
    }

    #[test]
    fn contrast_temperature_and_wheels_match_the_shader() {
        // CapCut scale: 25 of 50 = +0.5 contrast
        let g = ColorGrade { contrast: 25.0, ..Default::default() };
        let p = GradeParams::new(&g, None);
        // (0.25 - 0.5) * 1.5 + 0.5 = 0.125
        assert!(close3(p.apply([0.25; 3], [0.0; 3], [0.5, 0.5], 0.0), [0.125; 3], 1e-6));

        let g = ColorGrade { temperature: 25.0, tint: 10.0, ..Default::default() }; // 0.5 / 0.2 normalised
        let p = GradeParams::new(&g, None);
        // r += 0.5*0.12 + 0.2*0.05 ; g -= 0.2*0.1 ; b += -0.06 + 0.01
        let out = p.apply([0.5; 3], [0.0; 3], [0.5, 0.5], 0.0);
        assert!(close3(out, [0.57, 0.48, 0.45], 1e-6), "{out:?}");

        let g = ColorGrade { lift: [0.1, 0.0, 0.0], gain: [0.0, 0.2, 0.0], offset: [0.0, 0.0, -0.1], gamma: [0.0, 0.0, 1.0], ..Default::default() };
        let p = GradeParams::new(&g, None);
        let out = p.apply([0.5; 3], [0.0; 3], [0.5, 0.5], 0.0);
        // r = 0.5 + 0.1*0.5 = 0.55 ; g = 0.5*1.2 = 0.6 ; b = (0.5-0.1)^(1/2)
        assert!(close3(out, [0.55, 0.6, 0.4f32.sqrt()], 1e-6), "{out:?}");
    }

    #[test]
    fn saturation_minimum_is_greyscale() {
        let g = ColorGrade { saturation: -50.0, ..Default::default() }; // CapCut minimum
        let p = GradeParams::new(&g, None);
        let c = [0.9, 0.1, 0.3];
        let l = dot3(c, LUMA);
        assert!(close3(p.apply(c, [0.0; 3], [0.5, 0.5], 0.0), [l; 3], 1e-6));
    }

    #[test]
    fn hsv_round_trip_and_hsl_targets_only_its_hue() {
        for c in SAMPLES {
            assert!(close3(hsv2rgb(rgb2hsv(c)), c, 1e-5), "{c:?}");
        }
        let mut g = ColorGrade::default();
        g.hsl.red.s = -100.0; // desaturate reds fully
        let p = GradeParams::new(&g, None);
        let red = p.apply([0.8, 0.1, 0.1], [0.0; 3], [0.5, 0.5], 0.0);
        assert!(close3(red, [0.8; 3], 1e-5), "red → grey, got {red:?}");
        let blue = [0.1, 0.1, 0.8];
        assert!(close3(p.apply(blue, [0.0; 3], [0.5, 0.5], 0.0), blue, 1e-5));
        let grey = [0.4; 3];
        assert!(close3(p.apply(grey, [0.0; 3], [0.5, 0.5], 0.0), grey, 1e-5));
    }

    #[test]
    fn hsl_hue_table_matches_the_channel_loop() {
        let mut g = ColorGrade::default();
        g.hsl.red = crate::model::HslOffset { h: -24.0, s: 17.0, l: 5.0 };
        g.hsl.orange = crate::model::HslOffset { h: 16.0, s: 20.0, l: -40.0 };
        g.hsl.magenta = crate::model::HslOffset { h: -100.0, s: 100.0, l: 100.0 };
        let p = GradeParams::new(&g, None);
        let mut max = 0.0f32;
        for k in 0..=36_000 {
            let hue = k as f32 / 36_000.0;
            let (a, b) = (p.hsl_lookup(hue), p.hsl_weights(hue * 360.0));
            max = max.max((0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0, f32::max));
        }
        assert!(max < 2e-5, "table vs loop: {max}");
    }

    #[test]
    fn curves_match_monotone_cubic_and_bake() {
        let pts = vec![[0.0, 0.0], [0.5, 0.7], [1.0, 1.0]];
        assert!((eval_curve(&pts, 0.5) - 0.7).abs() < 1e-12);
        assert_eq!(eval_curve(&pts, 0.0), 0.0);
        assert_eq!(eval_curve(&pts, 1.0), 1.0);
        let row = CurveRow::bake(&pts);
        // at x = 0.5 the texture lookup sits exactly between texels 127 and 128
        let q = |x: f64| (eval_curve(&pts, x / 255.0) * 255.0).round() / 255.0;
        let expect = 0.5 * q(127.0) + 0.5 * q(128.0);
        assert!((row.lookup(0.5) as f64 - expect).abs() < 1e-6);
        let mut g = ColorGrade::default();
        g.curves.master = pts;
        let p = GradeParams::new(&g, None);
        let out = p.apply([0.5; 3], [0.0; 3], [0.5, 0.5], 0.0);
        assert!((out[0] - 0.7).abs() < 0.01, "{out:?}");
        assert!(CurveTables::bake(&ColorGrade::default().curves).is_none());
        let mut cc = ColorGrade::default().curves;
        cc.r = vec![[0.0, 0.1], [0.4, 0.6], [1.0, 0.9]];
        cc.master = vec![[0.0, 0.0], [0.3, 0.2], [1.0, 1.0]];
        let t = CurveTables::bake(&cc).unwrap();
        for k in 0..=1000 {
            let x = k as f32 / 1000.0;
            let (a, b) = (t.apply([x; 3]), t.apply_exact([x; 3]));
            assert!(close3(a, b, 2e-4), "{x}: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn lut_intensity_mixes() {
        let mut lut = Lut3D::identity(2);
        for v in lut.data.iter_mut() {
            *v = [1.0 - v[0], 1.0 - v[1], 1.0 - v[2]]; // invert
        }
        let g = ColorGrade { lut_asset_id: Some("lut".into()), lut_intensity: 0.5, ..Default::default() };
        let p = GradeParams::new(&g, Some(Arc::new(lut.clone())));
        assert!(close3(p.apply([0.2, 0.4, 1.0], [0.0; 3], [0.5, 0.5], 0.0), [0.5; 3], 1e-6));
        // no lutAssetId → LUT ignored (u_useLut = 0)
        let p = GradeParams::new(&ColorGrade::default(), Some(Arc::new(lut)));
        assert!(p.is_identity());
    }

    #[test]
    fn vignette_darkens_corners_only() {
        let g = ColorGrade { vignette: 1.0, ..Default::default() };
        let p = GradeParams::new(&g, None);
        assert_eq!(p.apply([0.5; 3], [0.0; 3], [0.5, 0.5], 0.0), [0.5; 3]);
        let corner = p.apply([0.5; 3], [0.0; 3], [0.0, 0.0], 0.0);
        assert!(corner[0] < 0.25, "{corner:?}");
    }

    #[test]
    fn grain_is_deterministic_and_bounded() {
        let g = ColorGrade { grain: 1.0, ..Default::default() };
        let p = GradeParams::new(&g, None);
        let a = p.apply([0.5; 3], [0.0; 3], [0.31, 0.62], 400.0);
        let b = p.apply([0.5; 3], [0.0; 3], [0.31, 0.62], 400.0);
        assert_eq!(a, b);
        assert!((a[0] - 0.5).abs() <= 0.125 + 1e-6);
    }

    #[test]
    fn brilliance_lifts_shadows_and_recovers_highlights() {
        let dark = [0.2f32, 0.2, 0.2];
        let bright = [0.85f32, 0.85, 0.85];
        let mid = [0.5f32, 0.5, 0.5];
        let up = brilliance(dark, 0.5);
        let down = brilliance(bright, 0.5);
        assert!(up[0] > dark[0], "shadows lifted: {:?}", up);
        assert!(down[0] < bright[0], "highlights recovered: {:?}", down);
        assert!((brilliance(mid, 0.5)[0] - 0.5).abs() < 0.03, "midtones barely move");
        assert_eq!(brilliance(dark, 0.0), dark);
        // grey stays grey (chroma preserved)
        let g = brilliance(dark, 0.3);
        assert!((g[0] - g[1]).abs() < 1e-6 && (g[1] - g[2]).abs() < 1e-6);
        // the grade uses it when the slider is set
        let p = GradeParams::new(&ColorGrade { brilliance: 25.0, ..Default::default() }, None);
        assert!(!p.is_identity());
        let out = p.apply(dark, [0.0; 3], [0.5, 0.5], 0.0);
        assert!(out[0] > dark[0]);
    }

    #[test]
    fn sharpness_adds_scaled_detail() {
        let g = ColorGrade { sharpness: 25.0, ..Default::default() }; // half of the 0..50 range
        let p = GradeParams::new(&g, None);
        assert!(p.needs_detail());
        let out = p.apply([0.5; 3], [0.1, -0.1, 0.0], [0.5, 0.5], 0.0);
        assert!(close3(out, [0.6, 0.4, 0.5], 1e-6), "{out:?}");
    }
}
