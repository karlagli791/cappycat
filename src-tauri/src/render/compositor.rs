//! Frame compositor: draws graded, transformed, masked layers onto an opaque
//! canvas, row-parallel with rayon.
//!
//! Per output pixel (centre `(x+0.5, y+0.5)`), exactly like the preview's
//! fragment shader: canvas uv → source uv through [`UvMatrix`]; outside
//! `[0,1]²` the layer is transparent; otherwise bilinear-sample the (blurred)
//! source, run [`GradeParams::apply`], multiply opacity by the mask weight and
//! blend onto the canvas with the clip's blend mode.

use super::bake::BakedGrade;
use super::blend::composite;
use super::color::GradeParams;
use super::mask::MaskParams;
use super::sample::{FloatImage, SourceImage, Texture, UvMatrix};
use crate::model::BlendMode;
use rayon::prelude::*;

/// Opaque RGB f32 canvas (rows top-down).
#[derive(Debug, Clone)]
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl Canvas {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![0.0; width * height * 3] }
    }

    /// Reset to opaque black (`gl.clearColor(0, 0, 0, 1)`).
    pub fn clear(&mut self) {
        self.data.par_chunks_mut(64 * 1024).for_each(|c| c.fill(0.0));
    }

    /// Quantise to `rgb24` (round to nearest).
    pub fn write_rgb24(&self, out: &mut [u8]) {
        let row = self.width * 3;
        out.par_chunks_mut(row).zip(self.data.par_chunks(row)).for_each(|(dst, src)| {
            for (d, s) in dst.iter_mut().zip(src) {
                *d = (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        });
    }

    pub fn to_rgb24(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.width * self.height * 3];
        self.write_rgb24(&mut out);
        out
    }
}

/// One clip's contribution to one output frame.
pub struct Layer<'a> {
    /// source image, already neighbour-blended and blurred (or the raw 8-bit frame)
    pub source: &'a SourceImage,
    /// sharpening detail layer (same size as `source`), when sharpness > 0
    pub detail: Option<&'a FloatImage>,
    pub uv: UvMatrix,
    pub grade: &'a GradeParams,
    /// the grade baked into lattices (see `render::bake`); `None` = exact per-pixel grade
    pub baked: Option<&'a BakedGrade>,
    pub opacity: f32,
    pub mask: Option<MaskParams>,
    pub blend: BlendMode,
    /// `u_time` for grain (timeline ms)
    pub time_ms: f32,
    /// clip video fade: amount of black 0..1; the graded colour is multiplied by `1 − fade`
    /// before opacity / blending (`render::fx::clip_fade_amount`); 0 = no fade
    pub fade: f32,
}

/// Composite one layer onto the canvas.
pub fn composite_layer(canvas: &mut Canvas, layer: &Layer) {
    if layer.opacity <= 0.0 {
        return;
    }
    match layer.source {
        SourceImage::Float(img) => composite_with(canvas, layer, img),
        SourceImage::Bytes(frame) => composite_with(canvas, layer, frame.as_ref()),
    }
}

fn composite_with<T: Texture>(canvas: &mut Canvas, layer: &Layer, src: &T) {
    let (w, h) = (canvas.width, canvas.height);
    let (fw, fh) = (w as f64, h as f64);
    let identity = layer.grade.is_identity();
    let m = layer.uv.m;
    let (sw, sh) = (src.width() as f32, src.height() as f32);
    canvas.data.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let v_td = (y as f64 + 0.5) / fh;
        let v = 1.0 - v_td; // shader canvas uv is bottom-up
        // su/sv are affine in u: start + x * step
        let u0 = 0.5 / fw;
        let du = 1.0 / fw;
        let su0 = m[0] * u0 + m[1] * v + m[2];
        let sv0 = m[3] * u0 + m[4] * v + m[5];
        let (dsu, dsv) = (m[0] * du, m[3] * du);
        for x in 0..w {
            let su = su0 + dsu * x as f64;
            let sv = sv0 + dsv * x as f64;
            if !(0.0..=1.0).contains(&su) || !(0.0..=1.0).contains(&sv) {
                continue;
            }
            let u = (x as f64 + 0.5) / fw;
            let mut alpha = layer.opacity;
            if let Some(mask) = &layer.mask {
                alpha *= mask.weight(u as f32, v_td as f32);
                if alpha <= 0.0 {
                    continue;
                }
            }
            let (px, py) = (su as f32 * sw - 0.5, (1.0 - sv as f32) * sh - 0.5);
            let graded = if identity {
                let c = src.sample(px, py);
                [c[0].clamp(0.0, 1.0), c[1].clamp(0.0, 1.0), c[2].clamp(0.0, 1.0)]
            } else {
                let (c, d) = match layer.detail {
                    Some(d) => src.sample_with(d, px, py),
                    None => (src.sample(px, py), [0.0; 3]),
                };
                match layer.baked {
                    Some(b) => b.apply(layer.grade, c, d, [u as f32, v as f32], layer.time_ms),
                    None => layer.grade.apply(c, d, [u as f32, v as f32], layer.time_ms),
                }
            };
            let graded = if layer.fade > 0.0 { graded.map(|c| c * (1.0 - layer.fade)) } else { graded };
            let i = x * 3;
            let cb = [row[i], row[i + 1], row[i + 2]];
            let out = if layer.blend == BlendMode::Normal && alpha >= 1.0 {
                graded
            } else {
                composite(layer.blend, cb, graded, alpha.min(1.0))
            };
            row[i] = out[0];
            row[i + 1] = out[1];
            row[i + 2] = out[2];
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ColorGrade, MaskShape};
    use crate::render::sample::{Frame, Placement};

    fn solid(w: usize, h: usize, rgb: [u8; 3]) -> SourceImage {
        let mut f = Frame::black(w, h);
        for p in f.data.chunks_mut(3) {
            p.copy_from_slice(&rgb);
        }
        // alternate the two representations: they must composite identically
        if rgb[0] == 255 {
            SourceImage::Bytes(std::sync::Arc::new(f))
        } else {
            SourceImage::Float(FloatImage::from_frames(&f, None))
        }
    }

    fn placement(cw: usize, ch: usize, sw: usize, sh: usize) -> Placement {
        Placement {
            canvas_w: cw as f64,
            canvas_h: ch as f64,
            source_w: sw as f64,
            source_h: sh as f64,
            crop: None,
            scale: 1.0,
            position: [0.0, 0.0],
            rotation_deg: 0.0,
        }
    }

    fn px(c: &Canvas, x: usize, y: usize) -> [f32; 3] {
        let i = (y * c.width + x) * 3;
        [c.data[i], c.data[i + 1], c.data[i + 2]]
    }

    #[test]
    fn letterbox_opacity_mask_and_blend() {
        let grade = GradeParams::new(&ColorGrade::default(), None);
        // 1:1 source on a 2:1 canvas → pillarbox: columns 0..16 and 48..64 stay black
        let src = solid(32, 32, [255, 0, 0]);
        let mut canvas = Canvas::new(64, 32);
        let layer = Layer {
            source: &src,
            detail: None,
            uv: UvMatrix::new(&placement(64, 32, 32, 32)),
            grade: &grade,
            baked: None,
            opacity: 1.0,
            mask: None,
            blend: BlendMode::Normal,
            time_ms: 0.0,
            fade: 0.0,
        };
        composite_layer(&mut canvas, &layer);
        assert_eq!(px(&canvas, 2, 16), [0.0; 3]);
        assert_eq!(px(&canvas, 32, 16), [1.0, 0.0, 0.0]);
        assert_eq!(px(&canvas, 60, 16), [0.0; 3]);

        // 50% white screen on top of red → (1, 0.5, 0.5)
        let white = solid(32, 32, [255, 255, 255]);
        let layer2 = Layer { source: &white, opacity: 0.5, blend: BlendMode::Screen, ..layer };
        composite_layer(&mut canvas, &layer2);
        let p = px(&canvas, 32, 16);
        assert!((p[0] - 1.0).abs() < 1e-6 && (p[1] - 0.5).abs() < 1e-6, "{p:?}");

        // circle mask: corners untouched, centre drawn
        let mut canvas = Canvas::new(32, 32);
        let green = solid(32, 32, [0, 255, 0]);
        let layer3 = Layer {
            source: &green,
            detail: None,
            uv: UvMatrix::new(&placement(32, 32, 32, 32)),
            grade: &grade,
            baked: None,
            opacity: 1.0,
            mask: Some(MaskParams { shape: MaskShape::Circle, rect: [0.1, 0.1, 0.8, 0.8], feather: 0.05, inverted: false }),
            blend: BlendMode::Normal,
            time_ms: 0.0,
            fade: 0.0,
        };
        composite_layer(&mut canvas, &layer3);
        assert_eq!(px(&canvas, 0, 0), [0.0; 3]);
        assert_eq!(px(&canvas, 16, 16), [0.0, 1.0, 0.0]);
        let bytes = canvas.to_rgb24();
        assert_eq!(&bytes[(16 * 32 + 16) * 3..(16 * 32 + 16) * 3 + 3], &[0, 255, 0]);
    }

    #[test]
    fn crop_selects_the_right_half() {
        // left half black, right half white; crop the right half onto a 1:2 canvas
        let mut f = Frame::black(64, 32);
        for y in 0..32 {
            for x in 32..64 {
                for c in 0..3 {
                    f.data[(y * 64 + x) * 3 + c] = 255;
                }
            }
        }
        let src = SourceImage::Float(FloatImage::from_frames(&f, None));
        let grade = GradeParams::new(&ColorGrade::default(), None);
        let mut p = placement(16, 16, 64, 32);
        p.crop = Some([32.0, 0.0, 64.0, 32.0]);
        let mut canvas = Canvas::new(16, 16);
        composite_layer(
            &mut canvas,
            &Layer { source: &src, detail: None, uv: UvMatrix::new(&p), grade: &grade, baked: None, opacity: 1.0, mask: None, blend: BlendMode::Normal, time_ms: 0.0, fade: 0.0 },
        );
        assert!(canvas.data.iter().all(|v| (*v - 1.0).abs() < 1e-6), "every pixel shows the white crop");
    }
}
