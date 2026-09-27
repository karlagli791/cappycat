//! Clip masks — port of the shader's `maskWeight`.
//!
//! The shader works in bottom-up canvas uv with the rect converted to
//! `(x, 1 - y - h, w, h)`. Every shape is symmetric about the rect centre (or,
//! for `split`, only uses x), so evaluating in top-down uv with the rect as
//! stored (`[x, y, w, h]`, top-left normalised) gives identical weights.

use super::color::smoothstep;
use crate::model::MaskShape;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskParams {
    pub shape: MaskShape,
    /// `[x, y, w, h]`, normalised, top-left origin
    pub rect: [f32; 4],
    pub feather: f32,
    pub inverted: bool,
}

impl MaskParams {
    /// Weight at canvas uv `(u, v_top_down)`.
    #[inline]
    pub fn weight(&self, u: f32, v: f32) -> f32 {
        let f = self.feather.max(0.0005);
        let [rx, ry, rw, rh] = self.rect;
        let (cx, cy) = (rx + rw * 0.5, ry + rh * 0.5);
        let w = match self.shape {
            MaskShape::Rectangle => {
                let dx = (u - cx).abs() - rw * 0.5;
                let dy = (v - cy).abs() - rh * 0.5;
                1.0 - smoothstep(0.0, f, dx.max(dy))
            }
            MaskShape::Circle => {
                let dx = (u - cx) / (rw * 0.5).max(1e-4);
                let dy = (v - cy) / (rh * 0.5).max(1e-4);
                let m = (dx * dx + dy * dy).sqrt() - 1.0;
                1.0 - smoothstep(0.0, f * 2.0, m)
            }
            MaskShape::Split => 1.0 - smoothstep(0.0, f, u - rx),
            MaskShape::Filmstrip => {
                let dy = (v - cy).abs() - rh * 0.5;
                1.0 - smoothstep(0.0, f, dy)
            }
        };
        if self.inverted {
            1.0 - w
        } else {
            w
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(shape: MaskShape, rect: [f32; 4], feather: f32, inverted: bool) -> MaskParams {
        MaskParams { shape, rect, feather, inverted }
    }

    #[test]
    fn rectangle_edges_and_feather() {
        let m = mask(MaskShape::Rectangle, [0.25, 0.25, 0.5, 0.5], 0.1, false);
        assert_eq!(m.weight(0.5, 0.5), 1.0);
        assert_eq!(m.weight(0.75, 0.5), 1.0); // exactly on the edge
        assert!((m.weight(0.8, 0.5) - 0.5).abs() < 1e-6); // half-way through the feather
        assert_eq!(m.weight(0.9, 0.5), 0.0);
        assert_eq!(m.weight(0.0, 0.0), 0.0);
        let inv = mask(MaskShape::Rectangle, [0.25, 0.25, 0.5, 0.5], 0.1, true);
        assert_eq!(inv.weight(0.5, 0.5), 0.0);
        assert_eq!(inv.weight(0.0, 0.0), 1.0);
    }

    #[test]
    fn circle_split_filmstrip() {
        let c = mask(MaskShape::Circle, [0.1, 0.1, 0.8, 0.8], 0.05, false);
        assert_eq!(c.weight(0.5, 0.5), 1.0);
        assert_eq!(c.weight(0.9, 0.5), 1.0); // on the ellipse
        assert!((c.weight(0.92, 0.5) - 0.5).abs() < 1e-5); // 0.42/0.4 - 1 = 0.05 = half the 2f band
        assert_eq!(c.weight(0.0, 0.0), 0.0); // corner outside
        // corner at 45°: distance sqrt(2) > 1 + 0.1
        assert_eq!(c.weight(0.1, 0.1), 0.0);

        let s = mask(MaskShape::Split, [0.5, 0.0, 0.5, 1.0], 0.0, false);
        assert_eq!(s.weight(0.4, 0.3), 1.0);
        assert_eq!(s.weight(0.6, 0.3), 0.0);

        let f = mask(MaskShape::Filmstrip, [0.0, 0.2, 1.0, 0.6], 0.0, false);
        assert_eq!(f.weight(0.1, 0.5), 1.0);
        assert_eq!(f.weight(0.1, 0.1), 0.0);
        assert_eq!(f.weight(0.1, 0.9), 0.0);
    }

    #[test]
    fn top_down_equals_shader_bottom_up() {
        // Shader: rect' = (x, 1-y-h, w, h), uv' = (u, 1-v). Check an asymmetric rect.
        let rect = [0.1f32, 0.05, 0.3, 0.2];
        let m = mask(MaskShape::Rectangle, rect, 0.05, false);
        let shader = |u: f32, v_bu: f32| {
            let r = [rect[0], 1.0 - rect[1] - rect[3], rect[2], rect[3]];
            let (cx, cy) = (r[0] + r[2] * 0.5, r[1] + r[3] * 0.5);
            let d = ((u - cx).abs() - r[2] * 0.5).max((v_bu - cy).abs() - r[3] * 0.5);
            1.0 - smoothstep(0.0, 0.05, d)
        };
        for (u, v) in [(0.2, 0.1), (0.2, 0.9), (0.42, 0.27), (0.05, 0.02)] {
            assert!((m.weight(u, v) - shader(u, 1.0 - v)).abs() < 1e-6, "({u},{v})");
        }
    }
}
