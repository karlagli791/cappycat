//! Layer blend modes (W3C Compositing Level 1 separable formulas).
//!
//! The canvas is always opaque (the preview clears to black), so compositing
//! a source colour `cs` with coverage `a` (opacity × mask) over a backdrop `cb`
//! reduces to `a·B(cb, cs) + (1 − a)·cb`. For `normal` that is exactly the
//! preview's premultiplied `blendFunc(ONE, ONE_MINUS_SRC_ALPHA)`.

use crate::model::BlendMode;

#[inline]
fn soft_light_d(cb: f32) -> f32 {
    if cb <= 0.25 {
        ((16.0 * cb - 12.0) * cb + 4.0) * cb
    } else {
        cb.sqrt()
    }
}

/// Blend function `B(cb, cs)` for one channel.
#[inline]
pub fn blend_channel(mode: BlendMode, cb: f32, cs: f32) -> f32 {
    match mode {
        BlendMode::Normal => cs,
        BlendMode::Multiply => cb * cs,
        BlendMode::Screen => cb + cs - cb * cs,
        // overlay(cb, cs) = hardlight(cs, cb)
        BlendMode::Overlay => {
            if cb <= 0.5 {
                2.0 * cb * cs
            } else {
                let t = 2.0 * cb - 1.0;
                t + cs - t * cs
            }
        }
        BlendMode::SoftLight => {
            if cs <= 0.5 {
                cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb)
            } else {
                cb + (2.0 * cs - 1.0) * (soft_light_d(cb) - cb)
            }
        }
        BlendMode::Darken => cb.min(cs),
        BlendMode::Lighten => cb.max(cs),
        BlendMode::ColorDodge => {
            if cb <= 0.0 {
                0.0
            } else if cs >= 1.0 {
                1.0
            } else {
                (cb / (1.0 - cs)).min(1.0)
            }
        }
    }
}

/// Composite `cs` with coverage `alpha` onto the opaque backdrop `cb`.
#[inline]
pub fn composite(mode: BlendMode, cb: [f32; 3], cs: [f32; 3], alpha: f32) -> [f32; 3] {
    let mut out = [0.0; 3];
    for i in 0..3 {
        let b = blend_channel(mode, cb[i], cs[i]);
        out[i] = cb[i] + (b - cb[i]) * alpha;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use BlendMode::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn formulas() {
        assert!(close(blend_channel(Normal, 0.2, 0.7), 0.7));
        assert!(close(blend_channel(Multiply, 0.5, 0.4), 0.2));
        assert!(close(blend_channel(Screen, 0.5, 0.4), 0.7));
        assert!(close(blend_channel(Overlay, 0.25, 0.5), 0.25)); // 2*0.25*0.5
        assert!(close(blend_channel(Overlay, 0.75, 0.5), 0.75)); // screen(0.5, 0.5)
        assert!(close(blend_channel(SoftLight, 0.5, 0.5), 0.5));
        assert!(close(blend_channel(SoftLight, 0.5, 0.0), 0.25)); // 0.5 - 1*0.5*0.5
        assert!(close(blend_channel(SoftLight, 0.25, 1.0), 0.5)); // D(0.25) = 0.25*... = 0.5
        assert!(close(blend_channel(Darken, 0.3, 0.6), 0.3));
        assert!(close(blend_channel(Lighten, 0.3, 0.6), 0.6));
        assert!(close(blend_channel(ColorDodge, 0.25, 0.5), 0.5));
        assert!(close(blend_channel(ColorDodge, 0.0, 0.9), 0.0));
        assert!(close(blend_channel(ColorDodge, 0.5, 1.0), 1.0));
        assert!(close(blend_channel(ColorDodge, 0.6, 0.5), 1.0));
    }

    #[test]
    fn coverage_matches_premultiplied_source_over() {
        let cb = [0.2, 0.4, 0.6];
        let cs = [1.0, 0.5, 0.0];
        let a = 0.25;
        let out = composite(Normal, cb, cs, a);
        for i in 0..3 {
            // premultiplied: cs*a + cb*(1-a)
            assert!(close(out[i], cs[i] * a + cb[i] * (1.0 - a)));
        }
        assert_eq!(composite(Screen, cb, cs, 0.0), cb);
        // screen always brightens
        let s = composite(Screen, cb, [0.5; 3], 0.5);
        assert!((0..3).all(|i| s[i] >= cb[i]));
    }
}
