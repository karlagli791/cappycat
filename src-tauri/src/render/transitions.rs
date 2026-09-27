//! Transitions between two clips on the same video track (`Clip.transitionIn`,
//! `docs/FEATURES_V2.md` §6). A 1:1 port of the preview's `TRANSITION_FRAGMENT_SHADER`
//! (`src/engine/color/fxShaders.ts`); the constants the spec leaves open were chosen there.
//!
//! # Conventions (identical to the preview)
//!
//! * `A` = outgoing clip, `B` = incoming clip, each drawn as a full frame (over whatever the lower
//!   tracks show, with its own grade / transform / fades) into its own canvas; the transition
//!   mixes the two canvases.
//! * Timing: window `[cut − d/2, cut + d/2)`, `raw = (t − (cut − d/2)) / d`,
//!   `p = easeInOut(raw)` with `easeInOut = cubic-bezier(0.42, 0, 0.58, 1)` ([`progress`]);
//!   `d = clamp(durationMs, 100, 3000)`, then at most the shorter clip.
//! * `uv = ((x + ½)/W, (y + ½)/H)`; `x` left → right, `yT` top → bottom (the GLSL converts
//!   `yT = 1 − uv.y`). `W`, `H` = output size; px constants are output pixels.
//! * `A(uv)` = bilinear sample of A at the pixel itself; shifted / scaled samples (slides, pushes,
//!   zooms) are **black outside `[0, 1]²`**; blur taps clamp to the edge.
//! * Colours: display-referred 0..1 values, every mix is `mix()` on them; the output is clamped.
//!
//! # Formulas
//!
//! | type | output |
//! |---|---|
//! | `dissolve` | `mix(A, B, p)` |
//! | `dipToBlack` | `p < ½`: `mix(A, 0, 2p)`; else `mix(0, B, 2p − 1)` |
//! | `dipToWhite` | the same through 1 |
//! | `wipe*` | `e = 0.02` ([`SOFT_EDGE`], along the wipe axis), `q = p(1 + e) − e/2`, `wB = 1 − smoothstep(q − e/2, q + e/2, c)` with `c = x` (wipeRight), `1 − x` (wipeLeft), `yT` (wipeDown), `1 − yT` (wipeUp); `mix(A, B, wB)` |
//! | `slideLeft` | `x ≥ 1 − p`: `B(x − (1 − p))`, else `A(x)` |
//! | `slideRight` | `x < p`: `B(x + 1 − p)`, else `A(x)` |
//! | `pushLeft` | `x < 1 − p`: `A(x + p)`, else `B(x − (1 − p))` |
//! | `pushRight` | `x < p`: `B(x + 1 − p)`, else `A(x − p)` |
//! | `zoomIn` | `sA = 1 + 0.3p`, `sB = 0.8 + 0.2p`, each sampled at `(uv − ½)/s + ½`; `mix(A', B', p)` |
//! | `zoomOut` | `sA = 1 − 0.2p`, `sB = 1.3 − 0.3p`; `mix(A', B', p)` |
//! | `blurDissolve` | `R = 12(1 − |2p − 1|)` px ([`BLUR_DISSOLVE_PX`]); `mix(blur(A, R), blur(B, R), p)`, blur = the house kernel (7 taps at `i·R/3` px, weights `exp(−i²/6)`; none when `R < 0.5`) |
//! | `flash` | `w = (1 − |2p − 1|)²`; `mix(mix(A, B, p), 1, w)` |
//! | `circleOpen` | `d = |(uv − ½)·(W, H)| / (½·|(W, H)|)` (1 at the corners), `e = 0.02`, `R = p(1 + e)`, `wB = 1 − smoothstep(R − e, R, d)`; `mix(A, B, wB)` |
//! | unknown | `dissolve` |

use super::compositor::Canvas;
use super::sample::FloatImage;
use crate::model::TransitionType;
use keyframes::Easing;
use rayon::prelude::*;

/// Soft edge of the wipes (along the wipe axis) and of `circleOpen` (in half-diagonals).
pub const SOFT_EDGE: f32 = 0.02;
/// `zoomIn`: A grows to this scale; `zoomOut`: B starts at it.
pub const ZOOM_BIG: f32 = 1.3;
/// `zoomIn`: B starts at this scale; `zoomOut`: A shrinks to it.
pub const ZOOM_SMALL: f32 = 0.8;
/// Peak blur radius of `blurDissolve` (output px).
pub const BLUR_DISSOLVE_PX: f32 = 12.0;

/// Eased progress 0..1 of a transition of length `dur_ms` centred on `cut_ms` at time `t_ms`.
pub fn progress(t_ms: f64, cut_ms: f64, dur_ms: f64) -> f32 {
    let raw = if dur_ms <= 0.0 { if t_ms >= cut_ms { 1.0 } else { 0.0 } } else { ((t_ms - (cut_ms - dur_ms / 2.0)) / dur_ms).clamp(0.0, 1.0) };
    Easing::EaseInOut.apply(raw, None).clamp(0.0, 1.0) as f32
}

/// GLSL `smoothstep`.
#[inline]
pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let k = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    k * k * (3.0 - 2.0 * k)
}

/// Bilinear sample of an interleaved RGB f32 image at pixel coordinates (`x = 0` is the centre of
/// the first column), clamped to the edge.
#[inline]
pub fn bilinear(data: &[f32], w: usize, h: usize, x: f32, y: f32) -> [f32; 3] {
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let i = |xx: usize, yy: usize| (yy * w + xx) * 3;
    let (a, b, c, d) = (i(x0, y0), i(x1, y0), i(x0, y1), i(x1, y1));
    let mut out = [0.0; 3];
    for k in 0..3 {
        let top = data[a + k] + (data[b + k] - data[a + k]) * fx;
        let bot = data[c + k] + (data[d + k] - data[c + k]) * fx;
        out[k] = top + (bot - top) * fy;
    }
    out
}

/// `texture(img, uv)` for a top-down uv (clamp-to-edge).
#[inline]
pub fn sample_uv(img: &Canvas, u: f32, v: f32) -> [f32; 3] {
    bilinear(&img.data, img.width, img.height, u * img.width as f32 - 0.5, v * img.height as f32 - 0.5)
}

/// Like [`sample_uv`] but black outside `[0, 1]²`.
#[inline]
fn sample_uv_black(img: &Canvas, u: f32, v: f32) -> [f32; 3] {
    if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
        return [0.0; 3];
    }
    sample_uv(img, u, v)
}

#[inline]
fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

/// Blur a canvas with the preview's 7-tap separable gaussian, radius `r` px (taps at `i·r/3`).
pub fn blur_canvas(src: &Canvas, radius_px: f32) -> Canvas {
    if radius_px < 0.5 {
        return src.clone();
    }
    let img = FloatImage { width: src.width, height: src.height, data: src.data.clone() };
    let b = img.gaussian_blur(radius_px / 3.0, radius_px / 3.0);
    Canvas { width: b.width, height: b.height, data: b.data }
}

/// Render the transition `kind` between `a` (outgoing) and `b` (incoming) at eased progress `p`
/// into `out` (all three the same size). See the module docs for the formulas.
pub fn render(kind: &TransitionType, a: &Canvas, b: &Canvas, p: f32, out: &mut Canvas) {
    let (w, h) = (out.width, out.height);
    debug_assert!(a.width == w && a.height == h && b.width == w && b.height == h);
    let p = p.clamp(0.0, 1.0);
    let (fw, fh) = (w as f32, h as f32);
    let e = SOFT_EDGE;
    // blurred inputs for blurDissolve
    let blurred;
    let (a, b) = if *kind == TransitionType::BlurDissolve {
        let r = BLUR_DISSOLVE_PX * (1.0 - (2.0 * p - 1.0).abs());
        blurred = (blur_canvas(a, r), blur_canvas(b, r));
        (&blurred.0, &blurred.1)
    } else {
        (a, b)
    };
    let half_diag = 0.5 * (fw * fw + fh * fh).sqrt();
    out.data.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let v = (y as f32 + 0.5) / fh;
        for x in 0..w {
            let u = (x as f32 + 0.5) / fw;
            let i = (y * w + x) * 3;
            let pa = [a.data[i], a.data[i + 1], a.data[i + 2]];
            let pb = [b.data[i], b.data[i + 1], b.data[i + 2]];
            let wipe = |c: f32| {
                let q = p * (1.0 + e) - e * 0.5;
                let m = 1.0 - smoothstep(q - e * 0.5, q + e * 0.5, c);
                mix3(pa, pb, m)
            };
            let c = match kind {
                TransitionType::DipToBlack => {
                    if p < 0.5 {
                        mix3(pa, [0.0; 3], 2.0 * p)
                    } else {
                        mix3([0.0; 3], pb, 2.0 * p - 1.0)
                    }
                }
                TransitionType::DipToWhite => {
                    if p < 0.5 {
                        mix3(pa, [1.0; 3], 2.0 * p)
                    } else {
                        mix3([1.0; 3], pb, 2.0 * p - 1.0)
                    }
                }
                TransitionType::WipeLeft => wipe(1.0 - u),
                TransitionType::WipeRight => wipe(u),
                TransitionType::WipeUp => wipe(1.0 - v),
                TransitionType::WipeDown => wipe(v),
                TransitionType::SlideLeft => {
                    if u >= 1.0 - p {
                        sample_uv_black(b, u - (1.0 - p), v)
                    } else {
                        pa
                    }
                }
                TransitionType::SlideRight => {
                    if u < p {
                        sample_uv_black(b, u + 1.0 - p, v)
                    } else {
                        pa
                    }
                }
                TransitionType::PushLeft => {
                    if u < 1.0 - p {
                        sample_uv_black(a, u + p, v)
                    } else {
                        sample_uv_black(b, u - (1.0 - p), v)
                    }
                }
                TransitionType::PushRight => {
                    if u < p {
                        sample_uv_black(b, u + 1.0 - p, v)
                    } else {
                        sample_uv_black(a, u - p, v)
                    }
                }
                TransitionType::ZoomIn | TransitionType::ZoomOut => {
                    let (sa, sb) = if *kind == TransitionType::ZoomIn {
                        (1.0 + (ZOOM_BIG - 1.0) * p, ZOOM_SMALL + (1.0 - ZOOM_SMALL) * p)
                    } else {
                        (1.0 - (1.0 - ZOOM_SMALL) * p, ZOOM_BIG - (ZOOM_BIG - 1.0) * p)
                    };
                    let ca = sample_uv_black(a, 0.5 + (u - 0.5) / sa, 0.5 + (v - 0.5) / sa);
                    let cb = sample_uv_black(b, 0.5 + (u - 0.5) / sb, 0.5 + (v - 0.5) / sb);
                    mix3(ca, cb, p)
                }
                TransitionType::Flash => {
                    let c = mix3(pa, pb, p);
                    let t = 1.0 - (2.0 * p - 1.0).abs();
                    mix3(c, [1.0; 3], t * t)
                }
                TransitionType::CircleOpen => {
                    let (dx, dy) = ((u - 0.5) * fw, (v - 0.5) * fh);
                    let d = (dx * dx + dy * dy).sqrt() / half_diag;
                    let r = p * (1.0 + e);
                    let m = 1.0 - smoothstep(r - e, r, d);
                    mix3(pa, pb, m)
                }
                // dissolve, blurDissolve (inputs already blurred) and unknown types
                _ => mix3(pa, pb, p),
            };
            row[x * 3] = c[0].clamp(0.0, 1.0);
            row[x * 3 + 1] = c[1].clamp(0.0, 1.0);
            row[x * 3 + 2] = c[2].clamp(0.0, 1.0);
        }
    });
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

    fn close(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() <= tol)
    }

    const RED: [f32; 3] = [1.0, 0.0, 0.0];
    const BLUE: [f32; 3] = [0.0, 0.0, 1.0];

    fn run(kind: TransitionType, p: f32) -> Canvas {
        let (a, b) = (solid(64, 36, RED), solid(64, 36, BLUE));
        let mut out = Canvas::new(64, 36);
        render(&kind, &a, &b, p, &mut out);
        out
    }

    #[test]
    fn progress_is_eased_and_centred_on_the_cut() {
        assert_eq!(progress(750.0, 1000.0, 500.0), 0.0);
        assert_eq!(progress(1250.0, 1000.0, 500.0), 1.0);
        assert!((progress(1000.0, 1000.0, 500.0) - 0.5).abs() < 1e-4, "midpoint at the cut");
        // ease-in-out: slow start
        assert!(progress(800.0, 1000.0, 500.0) < 0.1);
        let mut last = 0.0;
        for i in 0..=100 {
            let p = progress(750.0 + i as f64 * 5.0, 1000.0, 500.0);
            assert!(p >= last - 1e-6);
            last = p;
        }
    }

    #[test]
    fn every_type_starts_at_a_and_ends_at_b() {
        for kind in TransitionType::ALL.iter().chain([TransitionType::Other("starWipe".into())].iter()) {
            let start = run(kind.clone(), 0.0);
            let end = run(kind.clone(), 1.0);
            for (x, y) in [(0, 0), (32, 18), (63, 35), (10, 30)] {
                assert!(close(px(&start, x, y), RED, 1e-4), "{kind:?} at p=0 ({x},{y}) = {:?}", px(&start, x, y));
                assert!(close(px(&end, x, y), BLUE, 1e-4), "{kind:?} at p=1 ({x},{y}) = {:?}", px(&end, x, y));
            }
        }
    }

    #[test]
    fn dissolve_midpoint_is_the_average() {
        let out = run(TransitionType::Dissolve, 0.5);
        assert!(close(px(&out, 10, 10), [0.5, 0.0, 0.5], 1e-6));
        let out = run(TransitionType::BlurDissolve, 0.5);
        assert!(close(px(&out, 32, 18), [0.5, 0.0, 0.5], 1e-4), "blurring a flat colour changes nothing");
    }

    #[test]
    fn dips_pass_through_black_and_white() {
        assert!(close(px(&run(TransitionType::DipToBlack, 0.5), 5, 5), [0.0; 3], 1e-6));
        assert!(close(px(&run(TransitionType::DipToBlack, 0.25), 5, 5), [0.5, 0.0, 0.0], 1e-6));
        assert!(close(px(&run(TransitionType::DipToBlack, 0.75), 5, 5), [0.0, 0.0, 0.5], 1e-6));
        assert!(close(px(&run(TransitionType::DipToWhite, 0.5), 5, 5), [1.0; 3], 1e-6));
        assert!(close(px(&run(TransitionType::Flash, 0.5), 5, 5), [1.0; 3], 1e-6));
    }

    #[test]
    fn wipe_edges_are_where_the_formula_puts_them() {
        // 200 px wide: at p = 0.5 the 50 % point of wipeRight is at u = 0.5
        let (a, b) = (solid(200, 10, RED), solid(200, 10, BLUE));
        let mut out = Canvas::new(200, 10);
        render(&TransitionType::WipeRight, &a, &b, 0.5, &mut out);
        // left of the edge: B; right: A; the soft edge is 2 % = 4 px wide
        assert!(close(px(&out, 90, 5), BLUE, 1e-6));
        assert!(close(px(&out, 110, 5), RED, 1e-6));
        let m = |x: usize| px(&out, x, 5)[2];
        assert!(m(98) > 0.5 && m(101) < 0.5, "50 % crossing between px 98 and 101: {} {}", m(98), m(101));
        // the soft edge spans 4 px: fully B before u = E - e = 0.49, fully A after u = E = 0.51
        assert!(m(97) > 0.99 && m(102) < 0.01);
        // wipeLeft mirrors it
        render(&TransitionType::WipeLeft, &a, &b, 0.25, &mut out);
        // q = 1 − u; E = 0.255 → B for u > 1 − 0.235 = 0.765, A for u < 0.745
        assert!(close(px(&out, 160, 5), BLUE, 1e-6) && close(px(&out, 140, 5), RED, 1e-6));
        // wipeDown: B above the edge
        let (a, b) = (solid(10, 200, RED), solid(10, 200, BLUE));
        let mut out = Canvas::new(10, 200);
        render(&TransitionType::WipeDown, &a, &b, 0.5, &mut out);
        assert!(close(px(&out, 5, 90), BLUE, 1e-6) && close(px(&out, 5, 110), RED, 1e-6));
        render(&TransitionType::WipeUp, &a, &b, 0.5, &mut out);
        assert!(close(px(&out, 5, 90), RED, 1e-6) && close(px(&out, 5, 110), BLUE, 1e-6));
    }

    #[test]
    fn slides_and_pushes_move_the_right_image() {
        // A: left half green, right half red; B: blue
        let mut a = solid(100, 4, RED);
        for y in 0..4 {
            for x in 0..50 {
                a.data[(y * 100 + x) * 3..(y * 100 + x) * 3 + 3].copy_from_slice(&[0.0, 1.0, 0.0]);
            }
        }
        let b = solid(100, 4, BLUE);
        let mut out = Canvas::new(100, 4);
        render(&TransitionType::SlideLeft, &a, &b, 0.3, &mut out);
        assert!(close(px(&out, 10, 2), [0.0, 1.0, 0.0], 1e-6), "A static");
        assert!(close(px(&out, 69, 2), RED, 1e-6));
        assert!(close(px(&out, 71, 2), BLUE, 1e-6), "B covers u >= 0.7");
        render(&TransitionType::PushLeft, &a, &b, 0.3, &mut out);
        // A moved left by 30 px: pixel 25 shows A's pixel 55 (red); B from u >= 0.7
        assert!(close(px(&out, 25, 2), RED, 1e-6));
        assert!(close(px(&out, 15, 2), [0.0, 1.0, 0.0], 1e-6));
        assert!(close(px(&out, 75, 2), BLUE, 1e-6));
        render(&TransitionType::PushRight, &a, &b, 0.3, &mut out);
        assert!(close(px(&out, 20, 2), BLUE, 1e-6), "B enters from the left");
        assert!(close(px(&out, 50, 2), [0.0, 1.0, 0.0], 1e-6), "A moved right by 30 px: pixel 50 shows A's 20");
        render(&TransitionType::SlideRight, &a, &b, 0.3, &mut out);
        assert!(close(px(&out, 20, 2), BLUE, 1e-6) && close(px(&out, 40, 2), [0.0, 1.0, 0.0], 1e-6));
    }

    #[test]
    fn circle_opens_from_the_centre() {
        let (a, b) = (solid(160, 90, RED), solid(160, 90, BLUE));
        let mut out = Canvas::new(160, 90);
        render(&TransitionType::CircleOpen, &a, &b, 0.3, &mut out);
        assert!(close(px(&out, 80, 45), BLUE, 1e-6), "centre revealed");
        assert!(close(px(&out, 0, 0), RED, 1e-6), "corner not yet");
    }

    /// Parity with the preview: two-colour inputs through the Rust path against values computed
    /// by hand from the GLSL formulas in `src/engine/color/fxShaders.ts`.
    #[test]
    fn parity_with_the_preview_formulas() {
        const A: [f32; 3] = [0.8, 0.2, 0.1];
        const B: [f32; 3] = [0.1, 0.3, 0.9];
        let (w, h) = (64usize, 36usize);
        let (a, b) = (solid(w, h, A), solid(w, h, B));
        let render_at = |kind: TransitionType, p: f32| {
            let mut out = Canvas::new(w, h);
            render(&kind, &a, &b, p, &mut out);
            out
        };
        let mix = |x: [f32; 3], y: [f32; 3], t: f32| [x[0] + (y[0] - x[0]) * t, x[1] + (y[1] - x[1]) * t, x[2] + (y[2] - x[2]) * t];
        let ss = |e0: f32, e1: f32, x: f32| {
            let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        let tol = 1e-5;
        // dissolve p = .3
        assert!(close(px(&render_at(TransitionType::Dissolve, 0.3), 7, 7), mix(A, B, 0.3), tol));
        // dipToBlack p = .3 → mix(A, 0, .6); dipToWhite p = .7 → mix(1, B, .4)
        assert!(close(px(&render_at(TransitionType::DipToBlack, 0.3), 7, 7), mix(A, [0.0; 3], 0.6), tol));
        assert!(close(px(&render_at(TransitionType::DipToWhite, 0.7), 7, 7), mix([1.0; 3], B, 0.4), tol));
        // wipeRight p = .3: q = .3·1.02 − .01, wB = 1 − smoothstep(q − .01, q + .01, x)
        let out = render_at(TransitionType::WipeRight, 0.3);
        for xi in [10usize, 18, 19, 20, 30] {
            let x = (xi as f32 + 0.5) / w as f32;
            let q = 0.3 * 1.02 - 0.01;
            let wb = 1.0 - ss(q - 0.01, q + 0.01, x);
            assert!(close(px(&out, xi, 5), mix(A, B, wb), tol), "wipeRight x={xi}");
        }
        // wipeUp p = .6 on rows: c = 1 − yT
        let out = render_at(TransitionType::WipeUp, 0.6);
        for yi in [5usize, 14, 15, 30] {
            let c = 1.0 - (yi as f32 + 0.5) / h as f32;
            let q = 0.6 * 1.02 - 0.01;
            assert!(close(px(&out, 3, yi), mix(A, B, 1.0 - ss(q - 0.01, q + 0.01, c)), tol), "wipeUp y={yi}");
        }
        // flash p = .3: w = (1 − .4)² = .36
        assert!(close(px(&render_at(TransitionType::Flash, 0.3), 7, 7), mix(mix(A, B, 0.3), [1.0; 3], 0.36), tol));
        // circleOpen p = .5: d normalised by the half-diagonal, R = .51, e = .02
        let out = render_at(TransitionType::CircleOpen, 0.5);
        let hd = 0.5 * ((w * w + h * h) as f32).sqrt();
        for (xi, yi) in [(32usize, 18usize), (50, 30), (60, 34), (0, 0)] {
            let (dx, dy) = ((xi as f32 + 0.5) - 32.0, (yi as f32 + 0.5) - 18.0);
            let d = (dx * dx + dy * dy).sqrt() / hd;
            let wb = 1.0 - ss(0.51 - 0.02, 0.51, d);
            assert!(close(px(&out, xi, yi), mix(A, B, wb), tol), "circleOpen ({xi},{yi})");
        }
        // zoomIn p = .5: B at scale .9 is black outside → corner = mix(A, 0, .5)
        assert!(close(px(&render_at(TransitionType::ZoomIn, 0.5), 0, 0), mix(A, [0.0; 3], 0.5), tol));
        // pushLeft p = .25: A shifted left, B enters at x ≥ .75
        let out = render_at(TransitionType::PushLeft, 0.25);
        assert!(close(px(&out, 10, 5), A, tol) && close(px(&out, 60, 5), B, tol));
        // slideRight p = .25: B from the left over x < .25
        let out = render_at(TransitionType::SlideRight, 0.25);
        assert!(close(px(&out, 10, 5), B, tol) && close(px(&out, 20, 5), A, tol));
        // blurDissolve of flat colours = dissolve
        assert!(close(px(&render_at(TransitionType::BlurDissolve, 0.3), 30, 20), mix(A, B, 0.3), tol));
    }

    #[test]
    fn zoom_scales_and_crossfades() {
        let out = run(TransitionType::ZoomIn, 0.5);
        // B at scale 0.9 leaves a black border: the corner is A/2 + black/2
        assert!(close(px(&out, 0, 0), [0.5, 0.0, 0.0], 1e-4));
        assert!(close(px(&out, 32, 18), [0.5, 0.0, 0.5], 1e-4));
        let out = run(TransitionType::ZoomOut, 0.5);
        assert!(close(px(&out, 0, 0), [0.0, 0.0, 0.5], 1e-4), "A at 0.9 leaves black; B fills");
    }
}
