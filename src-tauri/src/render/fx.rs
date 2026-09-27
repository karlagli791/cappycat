//! Effects of FX-track clips (`Clip.effect`, `docs/FEATURES_V2.md` §7) and video clip fades
//! (§8), applied to the composited frame. A 1:1 port of the preview's GLSL
//! (`src/engine/color/fxShaders.ts`, `EFFECT_FRAGMENT_SHADER`) and of its per-frame CPU part
//! (`src/engine/effects.ts`, `effectFrame`); the constants the spec leaves open were chosen there
//! and are repeated here.
//!
//! # Conventions (identical to the preview)
//!
//! * `uv = ((x + ½)/W, (y + ½)/H)`, `x` left → right, `yT` **top → bottom**; the GLSL uses
//!   bottom-up uv and converts `yT = 1 − uv.y`. `W`, `H` = output (project) size; "px" constants
//!   are **output pixels**.
//! * Colour: every mix is on the display-referred (sRGB-encoded) 0..1 values — no linearisation.
//!   Output is clamped to 0..1.
//! * Sampling outside `[0, 1]²` (shake, zoomPunch, rgbSplit, vhs, the polaroid photo) is
//!   **black**; blur taps clamp to the edge.
//! * Blur "R px" = the house clip-blur kernel: 7 taps at `i·R/3` px, `i = −3..3`, weights
//!   `exp(−i²/6)`, normalised (separable = the shader's 7×7 product kernel); none when `R < 0.5`.
//! * `smoothstep` = GLSL Hermite; luma = Rec.709 `(0.2126, 0.7152, 0.0722)`.
//! * Noise ([`pcg`], [`rnd`], [`value_noise`]): PCG-RXS-M-XS
//!   `s = v·747796405 + 2891336453; w = ((s >> ((s >> 28) + 4)) ^ s)·277803737; pcg = (w >> 22) ^ w`
//!   (wrapping u32); `rnd(i, seed) = pcg(i ^ pcg(seed)) / 4294967295`;
//!   `valueNoise(x, seed) = 2·mix(rnd(⌊x⌋), rnd(⌊x⌋ + 1), f²(3 − 2f)) − 1` (x clamped ≥ 0).
//!   `pcg(0) = 129708002`, `pcg(1) = 2831084092`.
//!
//! # Strength
//!
//! `s = intensity × envelope` for blackAndWhite, sepia, letterbox, shake, rgbSplit, vhs,
//! vignettePulse (envelope = 120 ms linear ramps: `max(0, min(1, t/120, (D − t)/120))`); every other
//! type uses `s = intensity` (its own timing). `t` = ms since the effect started, `D` = its duration
//! (`outMs − inMs`), `u = t/D`.
//!
//! # Per type
//!
//! | type | output |
//! |---|---|
//! | `cameraSnap` | flash `a = s·max(0, 1 − t/250 ms)`; `k = s·easeOutCubic(min(1, t/350 ms))`; `sc = mix(1, scale, k)`; border `b = border·k·H` px (fraction of the HEIGHT); `bg = blur(F, 20k px)·(1 − 0.15k)`; shadow = the card rect (photo + b on every side) offset `0.012·H` px down, alpha `0.5k(1 − smoothstep(0, 0.03H, dist))`, `dist` = distance outside it; `c = bg(1 − shadow)`; inside the card white; inside the photo (`sc·W × sc·H`, centred) `F(P/sc)`; then `mix(c, 1, a)`. The input `F` is the video composite at the snap's start (the exporter freezes the video layers); effects earlier in track order apply to it at `t`. |
//! | `fadeFromBlack` / `fadeToBlack` | `mix(c, 0, a)`, `a = s(1 − u)` / `s·u` |
//! | `fadeFromWhite` / `fadeToWhite` | `mix(c, 1, a)`, same `a` |
//! | `flashWhite` | `mix(c, 1, s(1 − |2u − 1|))` |
//! | `blackAndWhite` | `g = clamp((luma − ½)·1.1 + ½)`; `mix(c, g, s)` |
//! | `sepia` | `R' = .393R + .769G + .189B`, `G' = .349R + .686G + .168B`, `B' = .272R + .534G + .131B`, clamped; `mix(c, sepia, s)` |
//! | `letterbox` | ratio `k` (default 2.39), `a = W/H`; `k > a`: rows with `yT < bar` or `yT > 1 − bar` black, `bar = s(1 − a/k)/2`; `k < a`: columns with `x < bar` or `x > 1 − bar` black, `bar = s(1 − k/a)/2`. Hard edges. |
//! | `shake` | `n_i = valueNoise(t_s·frequency, i)`, i = 1, 2, 3; `T = (n1, n2)·amplitude·s·W` px; `θ = 100·amplitude·s·n3` degrees (clockwise on screen); `z = 1 + 2·amplitude·s`; `P` = pixel − centre (y down), `d = P − T`, `r = (cos θ·dx + sin θ·dy, −sin θ·dx + cos θ·dy)`, output `src(r / z)` |
//! | `zoomPunch` | `m = u < .35 ? easeOutBack(u/.35) : 1 − easeInOutCubic((u − .35)/.65)` (`c1 = 1.70158`); `src(P / (1 + 0.15·s·m))` |
//! | `blurIn` / `blurOut` | blur `R = 20·s(1 − u)` / `20·s·u` px |
//! | `rgbSplit` | `o = amount·s·(1 + 0.5·valueNoise(8t_s, 4))` (fraction of W, amount default 0.006); `(src(x − o).r, c.g, src(x + o).b)` — the red image moves right |
//! | `vhs` | `row = ⌊yT·H⌋`, `k = ⌊30 t_s⌋`, `bandY = fract(0.25 t_s)`, `band = exp(−((yT − bandY)/0.035)²)`, `jit = rnd(row, k) − ½`, `dx = s(0.0015 sin(2π(2yT + 1.3 t_s)) + 0.02·band·jit)` (fraction of W), `bleed = 0.002 s`: `(src(x + dx + bleed).r, src(x + dx).g, src(x + dx − bleed).b)`; scanlines `× 1 − 0.2 s(½ + ½ cos(2π·yT·H/3))`; noise `+ 0.08 s (rnd((px·73856093) ^ (py·19349663), k) − ½)`, `(px, py) = ⌊(x·W, yT·H)⌋` |
//! | `vignettePulse` | `v = s(0.4 − 0.2 cos 2πt_s)`; `f = 1 − smoothstep(.35, 1.1, 1.35·|uv − ½|)`; `c·mix(1, f, v)` |
//! | clip fades | `Clip.fadeInMs/fadeOutMs`: the layer colour × `min(1, t/fadeIn)·min(1, (L − t)/fadeOut)` (each clamped to `L/2`, `t` clamped to `[0, L]`) before its opacity ([`clip_fade_amount`] returns `1 −` that) |
//!
//! # Shutter sound ([`procedural_shutter`], = `proceduralShutter` in effects.ts)
//!
//! `n = round(0.12·sr)` mono samples: white noise `x = 2r − 1` from mulberry32 (seed `0x43415050`)
//! through a one-pole high-pass at 2 kHz (`y = a(y' + x − x')`, `a = RC/(RC + 1/sr)`,
//! `RC = 1/(2π·2000)`) times `0.5·exp(−t/18 ms)`; plus a click `0.9·exp(−t/1.5 ms)·sin(2π·3500 t)` at
//! 0 ms and `0.7·exp(−(t − 0.06)/2 ms)·sin(2π·2400(t − 0.06))` at 60 ms; normalised to a 0.9 peak.
//! The exporter mixes it at −6 dB × intensity at the snap's start.

use super::compositor::Canvas;
use super::transitions::{bilinear, blur_canvas, smoothstep};
use crate::model::{ClipEffect, EffectType};
use rayon::prelude::*;

/// Envelope ramp (ms) at both ends of an effect.
pub const ENVELOPE_MS: f64 = 120.0;
/// `cameraSnap`: flash length and polaroid settle time (ms).
pub const SNAP_FLASH_MS: f64 = 250.0;
pub const SNAP_SETTLE_MS: f64 = 350.0;
/// `cameraSnap` default duration (ms), `border` (fraction of the height) and `scale`.
pub const SNAP_DEFAULT_MS: f64 = 1500.0;
pub const SNAP_BORDER: f64 = 0.03;
pub const SNAP_SCALE: f64 = 0.92;
/// `cameraSnap` background blur (px) and darkening; shadow offset / softness (× H) and opacity.
pub const SNAP_BG_BLUR_PX: f32 = 20.0;
pub const SNAP_BG_DARKEN: f32 = 0.15;
pub const SNAP_SHADOW_OFFSET: f32 = 0.012;
pub const SNAP_SHADOW_SOFT: f32 = 0.03;
pub const SNAP_SHADOW_ALPHA: f32 = 0.5;
/// `zoomPunch`: peak position (fraction of D) and amount.
pub const ZOOM_PUNCH_PEAK: f64 = 0.35;
pub const ZOOM_PUNCH_AMOUNT: f64 = 0.15;
/// `blurIn` / `blurOut` radius (px).
pub const BLUR_EFFECT_PX: f64 = 20.0;
/// Shutter sound: length (ms), PRNG seed, mix level (dB).
pub const SHUTTER_MS: f64 = 120.0;
pub const SHUTTER_SEED: u32 = 0x4341_5050;
pub const SHUTTER_DB: f64 = -6.0;

/* ------------------------------------------------------------------ helpers */

/// PCG-RXS-M-XS hash (wrapping u32).
#[inline]
pub fn pcg(v: u32) -> u32 {
    let s = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let w = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277_803_737);
    (w >> 22) ^ w
}

/// Uniform 0..1 from an integer lattice point and a seed.
#[inline]
pub fn rnd(i: u32, seed: u32) -> f32 {
    (pcg(i ^ pcg(seed)) as f64 / 4_294_967_295.0) as f32
}

/// 1-D value noise in −1..1 (x clamped ≥ 0).
#[inline]
pub fn value_noise(x: f64, seed: u32) -> f64 {
    let cx = x.max(0.0);
    let xi = cx.floor();
    let f = cx - xi;
    let u = f * f * (3.0 - 2.0 * f);
    let a = rnd(xi as u32, seed) as f64;
    let b = rnd((xi as u32).wrapping_add(1), seed) as f64;
    2.0 * (a + (b - a) * u) - 1.0
}

/// 120 ms linear ramps in and out.
pub fn envelope(t_ms: f64, dur_ms: f64) -> f64 {
    if dur_ms <= 0.0 {
        return 0.0;
    }
    (t_ms / ENVELOPE_MS).min((dur_ms - t_ms) / ENVELOPE_MS).clamp(0.0, 1.0)
}

/// Does the type use the envelope (otherwise it has its own timing)?
pub fn uses_envelope(kind: &EffectType) -> bool {
    matches!(
        kind,
        EffectType::BlackAndWhite | EffectType::Sepia | EffectType::Letterbox | EffectType::Shake | EffectType::RgbSplit | EffectType::Vhs | EffectType::VignettePulse
    )
}

/// `s`: intensity (clamped 0..1) × envelope for the types that use it.
pub fn strength(fx: &ClipEffect, t_ms: f64, dur_ms: f64) -> f64 {
    let k = if fx.intensity.is_finite() { fx.intensity.clamp(0.0, 1.0) } else { 1.0 };
    if uses_envelope(&fx.kind) {
        k * envelope(t_ms, dur_ms)
    } else {
        k
    }
}

pub fn ease_out_back(x: f64) -> f64 {
    let (c1, c3) = (1.70158, 2.70158);
    1.0 + c3 * (x - 1.0).powi(3) + c1 * (x - 1.0).powi(2)
}

pub fn ease_in_out_cubic(x: f64) -> f64 {
    if x < 0.5 {
        4.0 * x * x * x
    } else {
        1.0 - (-2.0 * x + 2.0).powi(3) / 2.0
    }
}

pub fn ease_out_cubic(x: f64) -> f64 {
    1.0 - (1.0 - x).powi(3)
}

/// Everything an effect pass needs for one frame (the shader uniforms of `effectFrame`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EffectFrame {
    /// intensity × envelope
    pub s: f64,
    /// effect-local time, seconds
    pub t: f64,
    /// fade amount / flash alpha / blur radius px / vignette strength / zoom scale
    pub a: f64,
    /// shake: translation (fractions of the width); rgbSplit: channel offset in `[0]`
    pub offset: [f64; 2],
    /// shake: rotation, degrees
    pub rot: f64,
    /// cameraSnap: polaroid progress; letterbox: target aspect ratio
    pub k: f64,
    /// cameraSnap: photo scale
    pub scale: f64,
    /// cameraSnap: border, fraction of the output height
    pub border: f64,
}

/// A numeric param or the type's default (`effectParam` in effects.ts).
fn param(fx: &ClipEffect, key: &str) -> f64 {
    let default = match key {
        "border" => SNAP_BORDER,
        "scale" => SNAP_SCALE,
        "ratio" => 2.39,
        "amplitude" => 0.01,
        "frequency" => 12.0,
        "amount" => 0.006,
        _ => 0.0,
    };
    fx.param(key, default)
}

/// `effectFrame(effect, tMs, durMs)`.
pub fn effect_frame(fx: &ClipEffect, t_ms: f64, dur_ms: f64) -> EffectFrame {
    let s = strength(fx, t_ms, dur_ms);
    let u = if dur_ms > 0.0 { (t_ms / dur_ms).clamp(0.0, 1.0) } else { 1.0 };
    let t = t_ms / 1000.0;
    let mut f = EffectFrame { s, t, a: 0.0, offset: [0.0; 2], rot: 0.0, k: 0.0, scale: 1.0, border: 0.0 };
    match fx.kind {
        EffectType::FadeFromBlack | EffectType::FadeFromWhite => f.a = s * (1.0 - u),
        EffectType::FadeToBlack | EffectType::FadeToWhite => f.a = s * u,
        EffectType::FlashWhite => f.a = s * (1.0 - (2.0 * u - 1.0).abs()),
        EffectType::BlurIn => f.a = BLUR_EFFECT_PX * s * (1.0 - u),
        EffectType::BlurOut => f.a = BLUR_EFFECT_PX * s * u,
        EffectType::ZoomPunch => {
            let m = if u < ZOOM_PUNCH_PEAK { ease_out_back(u / ZOOM_PUNCH_PEAK) } else { 1.0 - ease_in_out_cubic((u - ZOOM_PUNCH_PEAK) / (1.0 - ZOOM_PUNCH_PEAK)) };
            f.a = 1.0 + ZOOM_PUNCH_AMOUNT * s * m;
        }
        EffectType::Shake => {
            let amp = param(fx, "amplitude");
            let x = t * param(fx, "frequency");
            f.offset = [amp * s * value_noise(x, 1), amp * s * value_noise(x, 2)];
            f.rot = 100.0 * amp * s * value_noise(x, 3);
            f.a = 1.0 + 2.0 * amp * s;
        }
        EffectType::RgbSplit => f.offset = [param(fx, "amount") * s * (1.0 + 0.5 * value_noise(t * 8.0, 4)), 0.0],
        EffectType::VignettePulse => f.a = s * (0.4 - 0.2 * (2.0 * std::f64::consts::PI * t).cos()),
        EffectType::Letterbox => f.k = param(fx, "ratio"),
        EffectType::CameraSnap => {
            f.a = s * (1.0 - t_ms / SNAP_FLASH_MS).max(0.0);
            f.k = s * ease_out_cubic((t_ms / SNAP_SETTLE_MS).clamp(0.0, 1.0));
            f.scale = param(fx, "scale");
            f.border = param(fx, "border");
        }
        _ => {}
    }
    f
}

/// `tex()`: bilinear, black outside `[0, 1]²` (top-down uv).
#[inline]
fn tex(data: &[f32], w: usize, h: usize, u: f32, v: f32) -> [f32; 3] {
    if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
        return [0.0; 3];
    }
    bilinear(data, w, h, u * w as f32 - 0.5, v * h as f32 - 0.5)
}

/// `atPixel()`: the source at a pixel offset `P` from the centre (y down, output px).
#[inline]
fn at_pixel(data: &[f32], w: usize, h: usize, px: f32, py: f32) -> [f32; 3] {
    tex(data, w, h, px / w as f32 + 0.5, py / h as f32 + 0.5)
}

#[inline]
fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Apply one effect at `t_ms` since its start (`dur_ms` = its duration) to the canvas. Unknown
/// types do nothing. For `cameraSnap` the canvas must already hold the frozen composite.
pub fn apply(fx: &ClipEffect, t_ms: f64, dur_ms: f64, canvas: &mut Canvas) {
    if dur_ms <= 0.0 || !(0.0..=dur_ms).contains(&t_ms) || !fx.kind.is_known() {
        return;
    }
    let f = effect_frame(fx, t_ms, dur_ms);
    let (w, h) = (canvas.width, canvas.height);
    let (fw, fh) = (w as f32, h as f32);
    let s = f.s as f32;
    let a = f.a as f32;
    // the input frame, for the types that sample it elsewhere than at the pixel itself
    let src: Vec<f32> = match fx.kind {
        EffectType::CameraSnap | EffectType::Shake | EffectType::ZoomPunch | EffectType::RgbSplit | EffectType::Vhs => canvas.data.clone(),
        _ => Vec::new(),
    };
    // blurred inputs (house kernel, output px)
    let blurred: Option<Canvas> = match fx.kind {
        EffectType::BlurIn | EffectType::BlurOut => Some(blur_canvas(canvas, a)),
        EffectType::CameraSnap => Some(blur_canvas(canvas, SNAP_BG_BLUR_PX * f.k as f32)),
        _ => None,
    };
    let kind = fx.kind.clone();
    let t = f.t;
    let two_pi = std::f32::consts::TAU;
    canvas.data.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let yt = (y as f32 + 0.5) / fh;
        for xi in 0..w {
            let x = (xi as f32 + 0.5) / fw;
            let i = xi * 3;
            let c = [row[i], row[i + 1], row[i + 2]];
            // pixel offset from the centre, y down
            let (px, py) = ((x - 0.5) * fw, (yt - 0.5) * fh);
            let out: [f32; 3] = match kind {
                EffectType::CameraSnap => {
                    let k = f.k as f32;
                    let sc = 1.0 + (f.scale as f32 - 1.0) * k;
                    let b = f.border as f32 * k * fh;
                    let bgb = blurred.as_ref().unwrap();
                    let bi = (y * w + xi) * 3;
                    let dark = 1.0 - SNAP_BG_DARKEN * k;
                    let bg = [bgb.data[bi] * dark, bgb.data[bi + 1] * dark, bgb.data[bi + 2] * dark];
                    let (hix, hiy) = (0.5 * sc * fw, 0.5 * sc * fh);
                    let (hcx, hcy) = (hix + b, hiy + b);
                    let qx = px.abs() - hcx;
                    let qy = (py - SNAP_SHADOW_OFFSET * fh).abs() - hcy;
                    let dist = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
                    let shadow = SNAP_SHADOW_ALPHA * k * (1.0 - smoothstep(0.0, SNAP_SHADOW_SOFT * fh, dist));
                    let mut o = bg.map(|v| v * (1.0 - shadow));
                    if px.abs() <= hcx && py.abs() <= hcy {
                        o = [1.0; 3];
                    }
                    if px.abs() <= hix && py.abs() <= hiy {
                        o = at_pixel(&src, w, h, px / sc, py / sc);
                    }
                    mix3(o, [1.0; 3], a)
                }
                EffectType::FadeFromBlack | EffectType::FadeToBlack => mix3(c, [0.0; 3], a),
                EffectType::FadeFromWhite | EffectType::FadeToWhite | EffectType::FlashWhite => mix3(c, [1.0; 3], a),
                EffectType::BlackAndWhite => {
                    let l = c[0] * LUMA[0] + c[1] * LUMA[1] + c[2] * LUMA[2];
                    let g = ((l - 0.5) * 1.1 + 0.5).clamp(0.0, 1.0);
                    mix3(c, [g; 3], s)
                }
                EffectType::Sepia => {
                    let sp = [
                        (0.393 * c[0] + 0.769 * c[1] + 0.189 * c[2]).clamp(0.0, 1.0),
                        (0.349 * c[0] + 0.686 * c[1] + 0.168 * c[2]).clamp(0.0, 1.0),
                        (0.272 * c[0] + 0.534 * c[1] + 0.131 * c[2]).clamp(0.0, 1.0),
                    ];
                    mix3(c, sp, s)
                }
                EffectType::Letterbox => {
                    let asp = fw / fh;
                    let k = f.k as f32;
                    if k > asp {
                        let bar = s * (1.0 - asp / k) * 0.5;
                        if yt < bar || yt > 1.0 - bar {
                            [0.0; 3]
                        } else {
                            c
                        }
                    } else if k < asp {
                        let bar = s * (1.0 - k / asp) * 0.5;
                        if x < bar || x > 1.0 - bar {
                            [0.0; 3]
                        } else {
                            c
                        }
                    } else {
                        c
                    }
                }
                EffectType::Shake => {
                    let th = (f.rot as f32).to_radians();
                    let (dx, dy) = (px - f.offset[0] as f32 * fw, py - f.offset[1] as f32 * fw);
                    let (sn, cs) = th.sin_cos();
                    let (rx, ry) = (cs * dx + sn * dy, -sn * dx + cs * dy);
                    at_pixel(&src, w, h, rx / a, ry / a)
                }
                EffectType::ZoomPunch => at_pixel(&src, w, h, px / a, py / a),
                EffectType::BlurIn | EffectType::BlurOut => {
                    let b = blurred.as_ref().unwrap();
                    let bi = (y * w + xi) * 3;
                    [b.data[bi], b.data[bi + 1], b.data[bi + 2]]
                }
                EffectType::RgbSplit => {
                    let o = f.offset[0] as f32;
                    [tex(&src, w, h, x - o, yt)[0], c[1], tex(&src, w, h, x + o, yt)[2]]
                }
                EffectType::Vhs => {
                    let rowi = (yt * fh).floor().max(0.0) as u32;
                    let k = (t * 30.0).floor() as u32;
                    let band_y = (0.25 * t).fract() as f32;
                    let z = (yt - band_y) / 0.035;
                    let band = (-z * z).exp();
                    let jit = rnd(rowi, k) - 0.5;
                    let dx = s * (0.0015 * (two_pi * (2.0 * yt + 1.3 * t as f32)).sin() + 0.02 * band * jit);
                    let bleed = 0.002 * s;
                    let mut o = [tex(&src, w, h, x + dx + bleed, yt)[0], tex(&src, w, h, x + dx, yt)[1], tex(&src, w, h, x + dx - bleed, yt)[2]];
                    let scan = 1.0 - 0.2 * s * (0.5 + 0.5 * (two_pi * yt * fh / 3.0).cos());
                    let (pxu, pyu) = ((x * fw).floor().max(0.0) as u32, (yt * fh).floor().max(0.0) as u32);
                    let n = 0.08 * s * (rnd(pxu.wrapping_mul(73_856_093) ^ pyu.wrapping_mul(19_349_663), k) - 0.5);
                    for v in o.iter_mut() {
                        *v = *v * scan + n;
                    }
                    o
                }
                EffectType::VignettePulse => {
                    let (du, dv) = (x - 0.5, yt - 0.5);
                    let vig = 1.0 - smoothstep(0.35, 1.1, (du * du + dv * dv).sqrt() * 1.35);
                    let m = 1.0 + (vig - 1.0) * a;
                    c.map(|v| v * m)
                }
                EffectType::Other(_) => c,
            };
            row[i] = out[0].clamp(0.0, 1.0);
            row[i + 1] = out[1].clamp(0.0, 1.0);
            row[i + 2] = out[2].clamp(0.0, 1.0);
        }
    });
}

/// mulberry32 PRNG (u32 state) in `[0, 1)`, as in effects.ts.
pub struct Mulberry32(u32);

impl Mulberry32 {
    pub fn new(seed: u32) -> Self {
        Self(seed)
    }

    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x6d2b_79f5);
        let mut t = self.0;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        (t ^ (t >> 14)) as f64 / 4_294_967_296.0
    }
}

/// The `cameraSnap` shutter: `round(0.12·sr)` mono samples, 0.9 peak (see the module docs).
pub fn procedural_shutter(sample_rate: u32) -> Vec<f32> {
    let sr = sample_rate.max(1) as f64;
    let n = ((SHUTTER_MS / 1000.0 * sr).round() as usize).max(1);
    let mut rng = Mulberry32::new(SHUTTER_SEED);
    let rc = 1.0 / (2.0 * std::f64::consts::PI * 2000.0);
    let alpha = rc / (rc + 1.0 / sr);
    let (mut px, mut py) = (0.0f64, 0.0f64);
    let click = |t: f64, t0: f64, f: f64, amp: f64, tau: f64| if t >= t0 { amp * (-(t - t0) / tau).exp() * (2.0 * std::f64::consts::PI * f * (t - t0)).sin() } else { 0.0 };
    let mut out = Vec::with_capacity(n);
    let mut peak = 0.0f64;
    for i in 0..n {
        let t = i as f64 / sr;
        let x = rng.next_f64() * 2.0 - 1.0;
        let y = alpha * (py + x - px);
        px = x;
        py = y;
        let s = 0.5 * y * (-t / 0.018).exp() + click(t, 0.0, 3500.0, 0.9, 0.0015) + click(t, 0.06, 2400.0, 0.7, 0.002);
        peak = peak.max(s.abs());
        out.push(s);
    }
    if peak > 0.0 {
        for v in out.iter_mut() {
            *v = *v * 0.9 / peak;
        }
    }
    out.into_iter().map(|v| v as f32).collect()
}

/// Video clip fade: amount of black `1 − min(1, t/fadeIn)·min(1, (L − t)/fadeOut)` at clip-local
/// time `local_ms` (clamped to `[0, L]`) of a clip `len_ms` long, `fadeInMs` / `fadeOutMs` each
/// clamped to `L/2`. The compositor multiplies the layer colour by `1 − amount` before opacity.
pub fn clip_fade_amount(local_ms: f64, len_ms: f64, fade_in_ms: f64, fade_out_ms: f64) -> f32 {
    let half = (len_ms / 2.0).max(0.0);
    let fi = fade_in_ms.clamp(0.0, half);
    let fo = fade_out_ms.clamp(0.0, half);
    let t = local_ms.clamp(0.0, len_ms.max(0.0));
    let mut vis = 1.0f64;
    if fi > 0.0 {
        vis *= (t / fi).min(1.0);
    }
    if fo > 0.0 {
        vis *= ((len_ms - t) / fo).min(1.0);
    }
    (1.0 - vis.clamp(0.0, 1.0)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: usize, h: usize, c: [f32; 3]) -> Canvas {
        let mut k = Canvas::new(w, h);
        for p in k.data.chunks_mut(3) {
            p.copy_from_slice(&c);
        }
        k
    }

    fn px(c: &Canvas, x: usize, y: usize) -> [f32; 3] {
        let i = (y * c.width + x) * 3;
        [c.data[i], c.data[i + 1], c.data[i + 2]]
    }

    fn fx(kind: EffectType) -> ClipEffect {
        ClipEffect::new(kind)
    }

    fn close(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() <= tol)
    }

    #[test]
    fn noise_matches_the_preview() {
        // values documented in fxShaders.ts / effects.ts (TS `pcg`)
        assert_eq!(pcg(0), 129_708_002);
        assert_eq!(pcg(1), 2_831_084_092);
        let a: Vec<f64> = (0..1000).map(|i| value_noise(i as f64 * 0.037, 3)).collect();
        assert!(a.iter().all(|v| (-1.0..=1.0).contains(v)));
        let max_step = a.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f64::max);
        assert!(max_step < 0.2, "continuous: {max_step}");
        assert_eq!(value_noise(7.0, 9), 2.0 * rnd(7, 9) as f64 - 1.0, "lattice values at integers");
        assert_eq!(value_noise(-3.0, 9), value_noise(0.0, 9), "x clamped at 0");
    }

    #[test]
    fn envelope_only_for_the_listed_types() {
        assert_eq!(envelope(0.0, 1000.0), 0.0);
        assert!((envelope(60.0, 1000.0) - 0.5).abs() < 1e-12);
        assert_eq!(envelope(500.0, 1000.0), 1.0);
        assert!((envelope(50.0, 100.0) - 50.0 / 120.0).abs() < 1e-12);
        let mut e = fx(EffectType::Sepia);
        e.intensity = 0.8;
        assert!((strength(&e, 60.0, 1000.0) - 0.4).abs() < 1e-12);
        e.kind = EffectType::FlashWhite;
        assert_eq!(strength(&e, 0.0, 1000.0), 0.8, "own timing: intensity only");
        let enveloped: Vec<&str> = EffectType::ALL.iter().filter(|k| uses_envelope(k)).map(|k| k.as_str()).collect();
        assert_eq!(enveloped, ["blackAndWhite", "sepia", "letterbox", "shake", "rgbSplit", "vhs", "vignettePulse"]);
    }

    #[test]
    fn fades_mix_display_values() {
        let grey = solid(8, 8, [0.5; 3]);
        let run = |kind: EffectType, t: f64| {
            let mut c = grey.clone();
            apply(&fx(kind), t, 1000.0, &mut c);
            px(&c, 3, 3)
        };
        assert_eq!(run(EffectType::FadeToBlack, 0.0), [0.5; 3]);
        assert_eq!(run(EffectType::FadeToBlack, 1000.0), [0.0; 3], "fadeToBlack ends black");
        assert!(close(run(EffectType::FadeToBlack, 500.0), [0.25; 3], 1e-6), "linear ramp of the display values");
        assert!(close(run(EffectType::FadeFromBlack, 250.0), [0.125; 3], 1e-6));
        assert!(close(run(EffectType::FadeFromWhite, 0.0), [1.0; 3], 1e-6));
        assert!(close(run(EffectType::FadeToWhite, 500.0), [0.75; 3], 1e-6));
        assert!(close(run(EffectType::FlashWhite, 500.0), [1.0; 3], 1e-6));
        assert!(close(run(EffectType::FlashWhite, 250.0), [0.75; 3], 1e-6));
    }

    #[test]
    fn letterbox_bars_for_2_39() {
        let mut c = solid(1920, 1080, [1.0; 3]);
        apply(&fx(EffectType::Letterbox), 500.0, 1000.0, &mut c);
        // bar = (1 − (16/9)/2.39)/2 of the height = 138.33 px: rows 0..=137 and 942..=1079
        for y in [0, 100, 137, 942, 1079] {
            assert_eq!(px(&c, 960, y), [0.0; 3], "row {y} is bar");
        }
        for y in [138, 540, 941] {
            assert_eq!(px(&c, 960, y), [1.0; 3], "row {y} is picture");
        }
        let black_rows = (0..1080).filter(|y| px(&c, 5, *y) == [0.0; 3]).count();
        assert_eq!(black_rows, 276, "138 rows each");
        // half-way in (t = 60 ms): half the bar
        let mut c = solid(1920, 1080, [1.0; 3]);
        apply(&fx(EffectType::Letterbox), 60.0, 1000.0, &mut c);
        assert_eq!((0..1080).filter(|y| px(&c, 5, *y) == [0.0; 3]).count(), 138);
        // narrower ratio than the frame: pillar bars left and right
        let mut e = fx(EffectType::Letterbox);
        e.params = Some([("ratio".to_string(), 1.5)].into_iter().collect());
        let mut c = solid(1920, 1080, [1.0; 3]);
        apply(&e, 500.0, 1000.0, &mut c);
        assert_eq!(px(&c, 10, 540), [0.0; 3]);
        assert_eq!(px(&c, 960, 540), [1.0; 3]);
    }

    #[test]
    fn black_and_white_and_sepia() {
        let mut c = solid(4, 4, [0.8, 0.2, 0.1]);
        apply(&fx(EffectType::BlackAndWhite), 500.0, 1000.0, &mut c);
        let p = px(&c, 1, 1);
        let l = 0.2126 * 0.8 + 0.7152 * 0.2 + 0.0722 * 0.1;
        assert!(close(p, [((l - 0.5) * 1.1 + 0.5) as f32; 3], 1e-6), "{p:?}");
        let mut c = solid(4, 4, [0.5; 3]);
        apply(&fx(EffectType::Sepia), 500.0, 1000.0, &mut c);
        assert!(close(px(&c, 1, 1), [0.6755, 0.6015, 0.4685], 1e-4));
    }

    #[test]
    fn camera_snap_polaroid_and_flash() {
        // picture: left half red, right half blue
        let (w, h) = (320, 180);
        let mut base = solid(w, h, [0.0, 0.0, 1.0]);
        for y in 0..h {
            for x in 0..w / 2 {
                base.data[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&[1.0, 0.0, 0.0]);
            }
        }
        let snap = fx(EffectType::CameraSnap);
        let mut c = base.clone();
        apply(&snap, 0.0, 1500.0, &mut c);
        assert!(c.data.iter().all(|v| (*v - 1.0).abs() < 1e-6), "white flash at t=0");
        // settled after 350 ms: identical frames
        let mut c1 = base.clone();
        apply(&snap, 600.0, 1500.0, &mut c1);
        let mut c2 = base.clone();
        apply(&snap, 1200.0, 1500.0, &mut c2);
        assert_eq!(c1.data, c2.data, "the snapshot holds identical frames");
        // photo 0.92 × 320 = 294.4 wide (x 12.8..307.2); border 0.03 × 180 = 5.4 px (x 7.4..12.8)
        assert_eq!(px(&c1, 10, 90), [1.0; 3], "left border white");
        assert_eq!(px(&c1, 309, 90), [1.0; 3], "right border white");
        assert_eq!(px(&c1, 160, 5), [1.0; 3], "top border white");
        assert_eq!(px(&c1, 60, 90), [1.0, 0.0, 0.0]);
        assert_eq!(px(&c1, 260, 90), [0.0, 0.0, 1.0]);
        // outside the card: the blurred frame darkened by 15 %, shadowed below
        let bg = px(&c1, 2, 90);
        assert!(bg[0] > 0.5 && bg[0] <= 0.851, "darkened background: {bg:?}");
        let below = px(&c1, 160, 179);
        assert!(below[2] < 0.85 * 0.8, "shadow under the card: {below:?}");
        let mut c = base.clone();
        apply(&snap, 125.0, 1500.0, &mut c);
        assert!(px(&c, 60, 90)[1] > 0.4, "half flash lifts green");
    }

    #[test]
    fn geometric_effects_follow_the_formulas() {
        let mut grad = Canvas::new(64, 36);
        for y in 0..36 {
            for x in 0..64 {
                let i = (y * 64 + x) * 3;
                grad.data[i] = x as f32 / 63.0;
                grad.data[i + 1] = y as f32 / 35.0;
            }
        }
        for kind in [EffectType::Shake, EffectType::RgbSplit, EffectType::Vhs, EffectType::ZoomPunch, EffectType::VignettePulse, EffectType::BlurIn] {
            let mut a = grad.clone();
            let mut b = grad.clone();
            apply(&fx(kind.clone()), 400.0, 1000.0, &mut a);
            apply(&fx(kind.clone()), 400.0, 1000.0, &mut b);
            assert_eq!(a.data, b.data, "{kind:?} reproducible");
            assert_ne!(a.data, grad.data, "{kind:?} changes the frame");
        }
        let mut a = grad.clone();
        apply(&fx(EffectType::Other("hologram".into())), 400.0, 1000.0, &mut a);
        assert_eq!(a.data, grad.data, "unknown effects do nothing");
        // zoomPunch: 1.15 at 35 % of D (after the overshoot), 1 at both ends
        let z = |t: f64| effect_frame(&fx(EffectType::ZoomPunch), t, 1000.0).a;
        assert!((z(350.0) - 1.15).abs() < 1e-9);
        assert!((z(0.0) - 1.0).abs() < 1e-12 && (z(1000.0) - 1.0).abs() < 1e-12);
        assert!(z(250.0) > 1.15, "ease-out-back overshoot");
        // shake uniforms: 1 degree of rotation per unit noise at the default amplitude
        let f = effect_frame(&fx(EffectType::Shake), 500.0, 1000.0);
        assert!((f.rot - value_noise(6.0, 3)).abs() < 1e-12);
        assert!((f.a - 1.02).abs() < 1e-12);
        // zoomPunch sampling: at scale a the pixel at offset P shows the source at P/a
        let mut c = grad.clone();
        apply(&fx(EffectType::ZoomPunch), 350.0, 1000.0, &mut c);
        let expect = bilinear(&grad.data, 64, 36, 32.0 + (40.5 - 32.0) / 1.15 - 0.5, 18.0 + (10.5 - 18.0) / 1.15 - 0.5);
        assert!(close(px(&c, 40, 10), expect, 1e-5));
    }

    #[test]
    fn rgb_split_moves_the_red_image_right() {
        let mut c = Canvas::new(200, 4);
        for y in 0..4 {
            let i = (y * 200 + 100) * 3;
            c.data[i..i + 3].copy_from_slice(&[1.0; 3]);
        }
        apply(&fx(EffectType::RgbSplit), 500.0, 1000.0, &mut c);
        let red_x = (0..200).max_by(|a, b| px(&c, *a, 1)[0].total_cmp(&px(&c, *b, 1)[0])).unwrap();
        let blue_x = (0..200).max_by(|a, b| px(&c, *a, 1)[2].total_cmp(&px(&c, *b, 1)[2])).unwrap();
        assert!(red_x > 100 && blue_x < 100, "red right, blue left: {red_x} {blue_x}");
        assert_eq!(px(&c, 100, 1)[1], 1.0, "green unchanged");
    }

    #[test]
    fn shutter_matches_the_preview_recipe() {
        let s = procedural_shutter(48_000);
        assert_eq!(s.len(), 5760, "round(0.12 × 48000)");
        let peak = s.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!((peak - 0.9).abs() < 1e-6, "0.9 peak");
        assert_eq!(s, procedural_shutter(48_000), "deterministic");
        let rms = |a: usize, b: usize| (s[a..b].iter().map(|v| v * v).sum::<f32>() / (b - a) as f32).sqrt();
        assert!(rms(0, 480) > 0.1 && rms(2880, 3360) > 0.05 && rms(5000, 5760) < rms(2880, 3360));
        assert_eq!(procedural_shutter(44_100).len(), 5292);
    }

    /// Parity with the preview: two-colour inputs against values computed by hand from
    /// `EFFECT_FRAGMENT_SHADER` / `effectFrame`.
    #[test]
    fn parity_with_the_preview_formulas() {
        const A: [f32; 3] = [0.8, 0.2, 0.1];
        const B: [f32; 3] = [0.1, 0.3, 0.9];
        let (w, h) = (64usize, 36usize);
        // left half A, right half B
        let mut base = solid(w, h, B);
        for y in 0..h {
            for x in 0..w / 2 {
                base.data[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&A);
            }
        }
        let mix = |x: [f32; 3], y: [f32; 3], t: f32| [x[0] + (y[0] - x[0]) * t, x[1] + (y[1] - x[1]) * t, x[2] + (y[2] - x[2]) * t];
        let run = |e: &ClipEffect, t: f64, d: f64| {
            let mut c = base.clone();
            apply(e, t, d, &mut c);
            c
        };
        let tol = 1e-5;
        // fadeToBlack u = .3: mix(c, 0, .3); fadeFromWhite u = .25: mix(c, 1, .75)
        assert!(close(px(&run(&fx(EffectType::FadeToBlack), 300.0, 1000.0), 5, 5), mix(A, [0.0; 3], 0.3), tol));
        assert!(close(px(&run(&fx(EffectType::FadeFromWhite), 250.0, 1000.0), 50, 5), mix(B, [1.0; 3], 0.75), tol));
        // intensity scales the own-timing types: flashWhite at u = .5 with intensity .5
        let mut e = fx(EffectType::FlashWhite);
        e.intensity = 0.5;
        assert!(close(px(&run(&e, 500.0, 1000.0), 5, 5), mix(A, [1.0; 3], 0.5), tol));
        // blackAndWhite at t = 60 ms (envelope .5)
        let g = (((A[0] * 0.2126 + A[1] * 0.7152 + A[2] * 0.0722) - 0.5) * 1.1 + 0.5).clamp(0.0, 1.0);
        assert!(close(px(&run(&fx(EffectType::BlackAndWhite), 60.0, 1000.0), 5, 5), mix(A, [g; 3], 0.5), tol));
        // sepia at full strength
        let sp = [
            (0.393 * B[0] + 0.769 * B[1] + 0.189 * B[2]).min(1.0),
            (0.349 * B[0] + 0.686 * B[1] + 0.168 * B[2]).min(1.0),
            (0.272 * B[0] + 0.534 * B[1] + 0.131 * B[2]).min(1.0),
        ];
        assert!(close(px(&run(&fx(EffectType::Sepia), 500.0, 1000.0), 50, 5), sp, tol));
        // vignettePulse at t = 500 ms: v = .4 − .2 cos(π) = .6; corner pixel
        let (du, dv) = (0.5f32 / 64.0 - 0.5, 0.5f32 / 36.0 - 0.5);
        let d = (du * du + dv * dv).sqrt() * 1.35;
        let f = 1.0 - {
            let t = ((d - 0.35) / (1.1 - 0.35)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        let k = 1.0 + (f - 1.0) * 0.6;
        assert!(close(px(&run(&fx(EffectType::VignettePulse), 500.0, 1000.0), 0, 0), A.map(|v| v * k), tol));
        // rgbSplit: red moves right by o·W px, blue left; o = amount·(1 + .5·noise(8t, 4))
        let t = 0.5;
        let o = (0.006 * (1.0 + 0.5 * value_noise(8.0 * t, 4))) as f32;
        let out = run(&fx(EffectType::RgbSplit), 500.0, 1000.0);
        let xi = 32usize; // first B column: its red comes from o·W px to the left (A's red)
        let src_x = (xi as f32 + 0.5) / 64.0 - o;
        let expect_r = bilinear(&base.data, w, h, src_x * 64.0 - 0.5, 4.5 + 0.0)[0];
        assert!((px(&out, xi, 5)[0] - expect_r).abs() < tol, "rgbSplit red");
        // cameraSnap settled: centre-left pixel shows the photo (scaled about the centre)
        let out = run(&fx(EffectType::CameraSnap), 700.0, 1500.0);
        assert!(close(px(&out, 20, 18), A, tol) && close(px(&out, 44, 18), B, tol));
        // zoomPunch at its peak (u = .35): scale 1.15
        let out = run(&fx(EffectType::ZoomPunch), 350.0, 1000.0);
        let sx = 32.0 + (10.5 - 32.0) / 1.15;
        assert!(close(px(&out, 10, 18), bilinear(&base.data, w, h, sx - 0.5, 18.0 + (18.5 - 18.0) / 1.15 - 0.5), tol));
    }

    #[test]
    fn clip_fades() {
        assert_eq!(clip_fade_amount(0.0, 2000.0, 500.0, 0.0), 1.0);
        assert_eq!(clip_fade_amount(250.0, 2000.0, 500.0, 0.0), 0.5);
        assert_eq!(clip_fade_amount(1000.0, 2000.0, 500.0, 500.0), 0.0);
        assert_eq!(clip_fade_amount(2000.0, 2000.0, 0.0, 500.0), 1.0);
        assert_eq!(clip_fade_amount(500.0, 1000.0, 5000.0, 0.0), 0.0, "clamped to half the clip");
        assert_eq!(clip_fade_amount(250.0, 1000.0, 5000.0, 0.0), 0.5);
        assert_eq!(clip_fade_amount(-100.0, 1000.0, 500.0, 0.0), 1.0, "t clamped to the clip");
    }
}
