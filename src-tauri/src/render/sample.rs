//! Source images, bilinear sampling and the preview's uv mapping.
//!
//! Coordinate conventions follow the WebGL preview (`renderer.ts`):
//! * canvas uv is bottom-up (`v = 0` is the bottom row of the output frame),
//! * source uv is bottom-up too (the video texture is uploaded with
//!   `UNPACK_FLIP_Y_WEBGL`), with `(0,0)` the bottom-left corner of the image,
//! * sampling is bilinear with CLAMP_TO_EDGE, texel centres at `(i + 0.5) / n`.

use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// A small pool of reusable buffers (per export layer), so the per-frame
/// images (tens of MB at 1080p) are not allocated — and page-faulted in —
/// again for every frame.
#[derive(Debug, Default)]
pub struct BufPool<T> {
    bufs: Mutex<Vec<Vec<T>>>,
}

const POOL_KEEP: usize = 8;

impl<T: Copy + Default> BufPool<T> {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { bufs: Mutex::new(Vec::new()) })
    }

    /// A buffer of exactly `len` elements. Contents are unspecified: callers overwrite it.
    pub fn take(&self, len: usize) -> Vec<T> {
        let reused = {
            let mut b = self.bufs.lock().unwrap();
            let i = b.iter().position(|v| v.len() == len).or_else(|| b.iter().position(|v| v.capacity() >= len));
            i.map(|i| b.swap_remove(i))
        };
        let mut v = reused.unwrap_or_default();
        if v.len() != len {
            v.clear();
            v.resize(len, T::default());
        }
        v
    }

    pub fn put(&self, v: Vec<T>) {
        if v.capacity() == 0 {
            return;
        }
        let mut b = self.bufs.lock().unwrap();
        if b.len() < POOL_KEEP {
            b.push(v);
        }
    }
}

/// Bilinear read over an interleaved RGB buffer of any texel type.
#[inline]
fn bilinear_raw<T: Copy>(data: &[T], w: usize, h: usize, x: f32, y: f32, f: impl Fn(T) -> f32) -> [f32; 3] {
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let x0 = x as usize; // non-negative: floor
    let y0 = y as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let (a, b, c, d) = ((y0 * w + x0) * 3, (y0 * w + x1) * 3, (y1 * w + x0) * 3, (y1 * w + x1) * 3);
    let mut out = [0.0; 3];
    for i in 0..3 {
        let top = f(data[a + i]) + (f(data[b + i]) - f(data[a + i])) * fx;
        let bot = f(data[c + i]) + (f(data[d + i]) - f(data[c + i])) * fx;
        out[i] = top + (bot - top) * fy;
    }
    out
}

/// Something the compositor can sample: a float image, or an 8-bit frame read
/// directly (no blend / blur needed, so no float copy is made).
pub trait Texture: Sync {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    /// Bilinear sample at texel coordinates (`x = 0` is the centre of the first column).
    fn sample(&self, x: f32, y: f32) -> [f32; 3];
    /// Sample this texture and a same-sized float detail layer at once.
    fn sample_with(&self, detail: &FloatImage, x: f32, y: f32) -> ([f32; 3], [f32; 3]) {
        (self.sample(x, y), detail.bilinear(x, y))
    }
}

impl Texture for Frame {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    #[inline]
    fn sample(&self, x: f32, y: f32) -> [f32; 3] {
        bilinear_raw(&self.data, self.width, self.height, x, y, |v: u8| v as f32 * INV255)
    }
}

impl Texture for FloatImage {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    #[inline]
    fn sample(&self, x: f32, y: f32) -> [f32; 3] {
        self.bilinear(x, y)
    }
    #[inline]
    fn sample_with(&self, detail: &FloatImage, x: f32, y: f32) -> ([f32; 3], [f32; 3]) {
        if detail.width == self.width && detail.height == self.height {
            self.bilinear2(detail, x, y)
        } else {
            (self.bilinear(x, y), detail.bilinear(x, y))
        }
    }
}

/// A layer's source pixels for one output frame.
#[derive(Debug, Clone)]
pub enum SourceImage {
    Float(FloatImage),
    Bytes(Arc<Frame>),
}

impl SourceImage {
    pub fn width(&self) -> usize {
        match self {
            Self::Float(f) => f.width,
            Self::Bytes(b) => b.width,
        }
    }
    pub fn height(&self) -> usize {
        match self {
            Self::Float(f) => f.height,
            Self::Bytes(b) => b.height,
        }
    }
    /// Bilinear sample (convenience; the compositor dispatches once per layer).
    pub fn sample(&self, x: f32, y: f32) -> [f32; 3] {
        match self {
            Self::Float(f) => f.bilinear(x, y),
            Self::Bytes(b) => b.sample(x, y),
        }
    }
}

/// Sharpening detail (`tex(p) - avg4(tex(p ± step))`) of an interleaved RGB buffer into `out`.
fn detail_into<T: Copy + Sync>(data: &[T], w: usize, h: usize, step_x: f32, step_y: f32, out: &mut [f32], f: impl Fn(T) -> f32 + Sync + Copy) {
    let row = w * 3;
    let unit = (step_x - 1.0).abs() < 1e-6 && (step_y - 1.0).abs() < 1e-6;
    out.par_chunks_mut(row).enumerate().for_each(|(y, dst)| {
        if unit {
            // one-texel offsets land on texel centres: plain neighbour reads
            let up = &data[y.saturating_sub(1) * row..][..row];
            let mid = &data[y * row..][..row];
            let dn = &data[(y + 1).min(h - 1) * row..][..row];
            for x in 0..w {
                let (xl, xr) = (x.saturating_sub(1), (x + 1).min(w - 1));
                for c in 0..3 {
                    let i = x * 3 + c;
                    dst[i] = f(mid[i]) - (f(mid[xl * 3 + c]) + f(mid[xr * 3 + c]) + f(up[i]) + f(dn[i])) * 0.25;
                }
            }
            return;
        }
        let yf = y as f32;
        for x in 0..w {
            let xf = x as f32;
            let i = (y * w + x) * 3;
            let c = [f(data[i]), f(data[i + 1]), f(data[i + 2])];
            let l = bilinear_raw(data, w, h, xf - step_x, yf, f);
            let r = bilinear_raw(data, w, h, xf + step_x, yf, f);
            let u = bilinear_raw(data, w, h, xf, yf - step_y, f);
            let d = bilinear_raw(data, w, h, xf, yf + step_y, f);
            for k in 0..3 {
                dst[x * 3 + k] = c[k] - (l[k] + r[k] + u[k] + d[k]) * 0.25;
            }
        }
    });
}

impl Frame {
    /// Sharpening detail layer of the 8-bit frame, written into a pooled buffer.
    pub fn sharpen_detail_into(&self, step_x: f32, step_y: f32, buf: Vec<f32>) -> FloatImage {
        let mut out = FloatImage { width: self.width, height: self.height, data: buf };
        out.data.resize(self.width * self.height * 3, 0.0);
        detail_into(&self.data, self.width, self.height, step_x, step_y, &mut out.data, |v: u8| v as f32 * INV255);
        out
    }
}

/// A decoded 8-bit RGB frame (`rgb24`, rows top-down).
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl Frame {
    pub fn black(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![0; width * height * 3] }
    }
}

/// Interleaved f32 RGB image, rows top-down, values nominally 0..1.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

const INV255: f32 = 1.0 / 255.0;

impl FloatImage {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![0.0; width * height * 3] }
    }

    /// Convert a frame to f32, optionally cross-fading towards a second frame
    /// (`b` with weight `w`) — the neighbour blend used for slow motion.
    pub fn from_frames(a: &Frame, b: Option<(&Frame, f32)>) -> Self {
        Self::from_frames_into(a, b, Vec::new())
    }

    /// [`FloatImage::from_frames`] into a reused buffer (every element is written).
    pub fn from_frames_into(a: &Frame, b: Option<(&Frame, f32)>, buf: Vec<f32>) -> Self {
        let (w, h) = (a.width, a.height);
        let mut out = Self { width: w, height: h, data: buf };
        out.data.resize(w * h * 3, 0.0);
        let row = w * 3;
        match b {
            Some((b, t)) if t > 0.0 && b.width == w && b.height == h => {
                let s = 1.0 - t;
                out.data.par_chunks_mut(row).enumerate().for_each(|(y, dst)| {
                    let ra = &a.data[y * row..(y + 1) * row];
                    let rb = &b.data[y * row..(y + 1) * row];
                    for i in 0..row {
                        dst[i] = (ra[i] as f32 * s + rb[i] as f32 * t) * INV255;
                    }
                });
            }
            _ => {
                out.data.par_chunks_mut(row).enumerate().for_each(|(y, dst)| {
                    let ra = &a.data[y * row..(y + 1) * row];
                    for i in 0..row {
                        dst[i] = ra[i] as f32 * INV255;
                    }
                });
            }
        }
        out
    }

    #[inline]
    fn px(&self, x: usize, y: usize) -> [f32; 3] {
        let i = (y * self.width + x) * 3;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }

    /// Bilinear sample at texel coordinates (`x = 0` is the centre of the
    /// first column), clamped to the edge.
    #[inline]
    pub fn bilinear(&self, x: f32, y: f32) -> [f32; 3] {
        let xm = (self.width - 1) as f32;
        let ym = (self.height - 1) as f32;
        let x = x.clamp(0.0, xm);
        let y = y.clamp(0.0, ym);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let a = self.px(x0, y0);
        let b = self.px(x1, y0);
        let c = self.px(x0, y1);
        let d = self.px(x1, y1);
        let mut out = [0.0; 3];
        for i in 0..3 {
            let top = a[i] + (b[i] - a[i]) * fx;
            let bot = c[i] + (d[i] - c[i]) * fx;
            out[i] = top + (bot - top) * fy;
        }
        out
    }

    /// `texture(tex, uv)` for a bottom-up uv.
    #[inline]
    pub fn sample_uv(&self, u: f32, v: f32) -> [f32; 3] {
        self.bilinear(u * self.width as f32 - 0.5, (1.0 - v) * self.height as f32 - 0.5)
    }

    /// Separable version of the shader's 7×7 gaussian (`sampleFrame`): taps
    /// at `i * step` texels with weights `exp(-i²/6)`, `i ∈ -3..=3`. The
    /// shader's kernel is the outer product of this 1D kernel, and bilinear
    /// resampling commutes with it, so blurring once and sampling the result
    /// equals blurring at every sample position (up to edge clamping).
    pub fn gaussian_blur(&self, step_x: f32, step_y: f32) -> FloatImage {
        self.gaussian_blur_pooled(step_x, step_y, &BufPool::default())
    }

    /// [`FloatImage::gaussian_blur`] with the intermediate and output buffers from `pool`.
    pub fn gaussian_blur_pooled(&self, step_x: f32, step_y: f32, pool: &BufPool<f32>) -> FloatImage {
        let weights: [f32; 7] = std::array::from_fn(|k| {
            let i = k as f32 - 3.0;
            (-(i * i) / 6.0).exp()
        });
        let total: f32 = weights.iter().sum();
        let (w, h) = (self.width, self.height);
        let row = w * 3;
        // horizontal (every element written)
        let mut tmp = FloatImage { width: w, height: h, data: pool.take(w * h * 3) };
        tmp.data.par_chunks_mut(row).enumerate().for_each(|(y, dst)| {
            let src = &self.data[y * row..(y + 1) * row];
            let xm = (w - 1) as f32;
            for x in 0..w {
                let mut acc = [0.0f32; 3];
                for (k, wt) in weights.iter().enumerate() {
                    let sx = (x as f32 + (k as f32 - 3.0) * step_x).clamp(0.0, xm);
                    let x0 = sx.floor() as usize;
                    let x1 = (x0 + 1).min(w - 1);
                    let f = sx - x0 as f32;
                    for c in 0..3 {
                        let a = src[x0 * 3 + c];
                        let b = src[x1 * 3 + c];
                        acc[c] += (a + (b - a) * f) * wt;
                    }
                }
                for c in 0..3 {
                    dst[x * 3 + c] = acc[c] / total;
                }
            }
        });
        // vertical (each row is zeroed, then accumulated)
        let mut out = FloatImage { width: w, height: h, data: pool.take(w * h * 3) };
        let ym = (h - 1) as f32;
        out.data.par_chunks_mut(row).enumerate().for_each(|(y, dst)| {
            for v in dst.iter_mut() {
                *v = 0.0;
            }
            for (k, wt) in weights.iter().enumerate() {
                let sy = (y as f32 + (k as f32 - 3.0) * step_y).clamp(0.0, ym);
                let y0 = sy.floor() as usize;
                let y1 = (y0 + 1).min(h - 1);
                let f = sy - y0 as f32;
                let r0 = &tmp.data[y0 * row..(y0 + 1) * row];
                let r1 = &tmp.data[y1 * row..(y1 + 1) * row];
                let k0 = wt * (1.0 - f) / total;
                let k1 = wt * f / total;
                for i in 0..row {
                    dst[i] += r0[i] * k0 + r1[i] * k1;
                }
            }
        });
        pool.put(tmp.data);
        out
    }

    /// Sharpening detail layer: `tex(p) - avg(tex(p ± step_x), tex(p ± step_y))`
    /// evaluated at texel centres (the shader's unsharp term, before `* 2 * sharpness`).
    pub fn sharpen_detail(&self, step_x: f32, step_y: f32) -> FloatImage {
        self.sharpen_detail_into(step_x, step_y, Vec::new())
    }

    /// [`FloatImage::sharpen_detail`] into a reused buffer.
    pub fn sharpen_detail_into(&self, step_x: f32, step_y: f32, buf: Vec<f32>) -> FloatImage {
        let mut out = FloatImage { width: self.width, height: self.height, data: buf };
        out.data.resize(self.width * self.height * 3, 0.0);
        detail_into(&self.data, self.width, self.height, step_x, step_y, &mut out.data, |v: f32| v);
        out
    }

    /// Bilinear sample of two same-sized images at once (source + detail).
    #[inline]
    pub fn bilinear2(&self, other: &FloatImage, x: f32, y: f32) -> ([f32; 3], [f32; 3]) {
        let xm = (self.width - 1) as f32;
        let ym = (self.height - 1) as f32;
        let x = x.clamp(0.0, xm);
        let y = y.clamp(0.0, ym);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let (i00, i10, i01, i11) = ((y0 * self.width + x0) * 3, (y0 * self.width + x1) * 3, (y1 * self.width + x0) * 3, (y1 * self.width + x1) * 3);
        let mut a = [0.0; 3];
        let mut b = [0.0; 3];
        for c in 0..3 {
            let s = &self.data;
            let top = s[i00 + c] + (s[i10 + c] - s[i00 + c]) * fx;
            let bot = s[i01 + c] + (s[i11 + c] - s[i01 + c]) * fx;
            a[c] = top + (bot - top) * fy;
            let o = &other.data;
            let top = o[i00 + c] + (o[i10 + c] - o[i00 + c]) * fx;
            let bot = o[i01 + c] + (o[i11 + c] - o[i01 + c]) * fx;
            b[c] = top + (bot - top) * fy;
        }
        (a, b)
    }
}

/// Affine map canvas uv → source uv (both bottom-up), i.e. the preview's
/// `u_uvTransform` (`ColorRenderer.uvMatrix`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UvMatrix {
    /// `[su, sv] = [m[0]*u + m[1]*v + m[2], m[3]*u + m[4]*v + m[5]]`
    pub m: [f64; 6],
}

/// Spatial parameters of one layer at one instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    pub canvas_w: f64,
    pub canvas_h: f64,
    /// source dimensions the crop is expressed in
    pub source_w: f64,
    pub source_h: f64,
    /// reframe crop `[x1, y1, x2, y2]` in source pixels (top-left origin)
    pub crop: Option<[f64; 4]>,
    pub scale: f64,
    /// normalised centre offset (preview convention: +y moves the layer down)
    pub position: [f64; 2],
    pub rotation_deg: f64,
}

impl UvMatrix {
    pub fn new(p: &Placement) -> Self {
        let cw = p.canvas_w.max(1.0);
        let ch = p.canvas_h.max(1.0);
        let w = p.source_w.max(1.0);
        let h = p.source_h.max(1.0);
        let crop = p.crop.unwrap_or([0.0, 0.0, w, h]);
        let crop_w = (crop[2] - crop[0]).max(1.0);
        let crop_h = (crop[3] - crop[1]).max(1.0);
        let src_aspect = crop_w / crop_h;
        let ax = cw / ch;
        let (mut fit_w, mut fit_h) = (1.0, 1.0);
        if src_aspect > ax {
            fit_h = ax / src_aspect;
        } else {
            fit_w = src_aspect / ax;
        }
        fit_w *= p.scale.max(1e-4);
        fit_h *= p.scale.max(1e-4);
        let cx = 0.5 + p.position[0] * 0.5;
        let cy = 0.5 - p.position[1] * 0.5;
        let rad = p.rotation_deg.to_radians();
        let (sin, cos) = rad.sin_cos();
        let a00 = cos / fit_w;
        let a01 = sin / (fit_w * ax);
        let a10 = (-sin * ax) / fit_h;
        let a11 = cos / fit_h;
        let t0 = 0.5 - a00 * cx - a01 * cy;
        let t1 = 0.5 - a10 * cx - a11 * cy;
        let sx = crop_w / w;
        let sy = crop_h / h;
        let ox = crop[0] / w;
        let oy = 1.0 - crop[3] / h;
        Self { m: [sx * a00, sx * a01, sx * t0 + ox, sy * a10, sy * a11, sy * t1 + oy] }
    }

    #[inline]
    pub fn apply(&self, u: f64, v: f64) -> (f64, f64) {
        let m = &self.m;
        (m[0] * u + m[1] * v + m[2], m[3] * u + m[4] * v + m[5])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn placement(cw: f64, ch: f64, sw: f64, sh: f64) -> Placement {
        Placement { canvas_w: cw, canvas_h: ch, source_w: sw, source_h: sh, crop: None, scale: 1.0, position: [0.0, 0.0], rotation_deg: 0.0 }
    }

    #[test]
    fn same_aspect_is_identity_mapping() {
        let m = UvMatrix::new(&placement(1280.0, 720.0, 1920.0, 1080.0));
        for (u, v) in [(0.0, 0.0), (1.0, 1.0), (0.25, 0.8)] {
            let (su, sv) = m.apply(u, v);
            assert!(close(su, u) && close(sv, v), "({u},{v}) -> ({su},{sv})");
        }
    }

    #[test]
    fn crop_maps_canvas_onto_the_crop_window() {
        // 1920x1080 source, crop the centre 540x1080 column into a 9:16 canvas.
        let mut p = placement(1080.0, 1920.0, 1920.0, 1080.0);
        p.crop = Some([690.0, 0.0, 1230.0, 1080.0]);
        let m = UvMatrix::new(&p);
        // canvas aspect 0.5625 == crop aspect 0.5 ? no: 540/1080 = 0.5 < 0.5625 → pillarbox
        let (su, sv) = m.apply(0.5, 0.5);
        assert!(close(su, 960.0 / 1920.0) && close(sv, 0.5));
        // top-left of the crop window (bottom-up v = 1): x = 690 px
        let fit_w = 0.5 / 0.5625; // crop aspect / canvas aspect
        let u_left = 0.5 - fit_w / 2.0;
        let (su, sv) = m.apply(u_left, 1.0);
        assert!(close(su, 690.0 / 1920.0), "su={su}");
        assert!(close(sv, 1.0));
        // left of the fitted window → outside the crop (su < crop x1)
        let (su, _) = m.apply(0.0, 0.5);
        assert!(su < 690.0 / 1920.0);
    }

    #[test]
    fn letterbox_scale_position_rotation() {
        // 4:3 source into 16:9 canvas → pillarbox, source spans u ∈ [0.125, 0.875]
        let m = UvMatrix::new(&placement(1600.0, 900.0, 1200.0, 900.0));
        assert!(close(m.apply(0.125, 0.0).0, 0.0));
        assert!(close(m.apply(0.875, 0.0).0, 1.0));
        // scale 2 about the centre: canvas u=0.75 shows source u=0.625
        let mut p = placement(1000.0, 1000.0, 1000.0, 1000.0);
        p.scale = 2.0;
        assert!(close(UvMatrix::new(&p).apply(0.75, 0.5).0, 0.625));
        // position (+0.5, +0.5) moves the centre to canvas u = 0.75 and, as in
        // the preview (`cy = 0.5 - y * 0.5` in bottom-up uv), downwards to v = 0.25
        let mut p = placement(1000.0, 1000.0, 1000.0, 1000.0);
        p.position = [0.5, 0.5];
        let (su, sv) = UvMatrix::new(&p).apply(0.75, 0.25);
        assert!(close(su, 0.5) && close(sv, 0.5));
        // rotation 90° on a square: canvas (0.5, 1.0) ↔ source (1.0, 0.5)
        let mut p = placement(1000.0, 1000.0, 1000.0, 1000.0);
        p.rotation_deg = 90.0;
        let (su, sv) = UvMatrix::new(&p).apply(0.5, 1.0);
        assert!(close(su, 1.0) && close(sv, 0.5), "({su},{sv})");
    }

    #[test]
    fn bilinear_and_uv_sampling() {
        let f = Frame { width: 2, height: 2, data: vec![0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255] };
        let img = FloatImage::from_frames(&f, None);
        assert_eq!(img.bilinear(0.0, 0.0), [0.0; 3]);
        assert_eq!(img.bilinear(0.5, 0.0), [0.5; 3]);
        // bottom-up uv: v near 1 is the top row
        assert_eq!(img.sample_uv(0.75, 0.75), [1.0; 3]);
        assert_eq!(img.sample_uv(0.25, 0.25), [1.0, 0.0, 0.0]);
        // clamp to edge
        assert_eq!(img.bilinear(-5.0, 10.0), [1.0, 0.0, 0.0]);
        let mix = FloatImage::from_frames(&f, Some((&Frame { width: 2, height: 2, data: vec![255; 12] }, 0.5)));
        assert!((mix.data[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn blur_preserves_flat_fields_and_softens_edges() {
        let mut f = Frame::black(16, 4);
        for y in 0..4 {
            for x in 8..16 {
                for c in 0..3 {
                    f.data[(y * 16 + x) * 3 + c] = 255;
                }
            }
        }
        let img = FloatImage::from_frames(&f, None);
        let b = img.gaussian_blur(1.0, 1.0);
        assert!((b.px(0, 0)[0] - 0.0).abs() < 1e-6 && (b.px(15, 0)[0] - 1.0).abs() < 1e-6);
        assert!(b.px(7, 1)[0] > 0.1 && b.px(8, 1)[0] < 0.9);
        let flat = FloatImage::from_frames(&Frame { width: 3, height: 3, data: vec![128; 27] }, None);
        let d = flat.sharpen_detail(1.0, 1.0);
        assert!(d.data.iter().all(|v| v.abs() < 1e-6));
        assert!(flat.gaussian_blur(2.0, 2.0).data.iter().all(|v| (v - 128.0 / 255.0).abs() < 1e-5));
    }

    /// Sampling / sharpening straight from the 8-bit frame equals the float path,
    /// and pooled buffers give the same results as fresh ones.
    #[test]
    fn byte_frames_sample_like_float_images_and_pools_reuse_buffers() {
        let mut f = Frame::black(37, 23);
        for (i, v) in f.data.iter_mut().enumerate() {
            *v = ((i * 7919) % 251) as u8;
        }
        let img = FloatImage::from_frames(&f, None);
        for (x, y) in [(0.0, 0.0), (3.3, 7.8), (36.0, 22.0), (-4.0, 30.0), (12.5, 0.25)] {
            let (a, b) = (img.bilinear(x, y), f.sample(x, y));
            assert!((0..3).all(|i| (a[i] - b[i]).abs() < 1e-6), "{a:?} vs {b:?}");
        }
        for step in [1.0f32, 0.6] {
            let (d1, d2) = (img.sharpen_detail(step, step), f.sharpen_detail_into(step, step, Vec::new()));
            assert!(d1.data.iter().zip(&d2.data).all(|(a, b)| (a - b).abs() < 1e-6));
        }
        let pool = BufPool::<f32>::new();
        let fresh = img.gaussian_blur(1.5, 1.5);
        let first = img.gaussian_blur_pooled(1.5, 1.5, &pool);
        let ptr = first.data.as_ptr();
        assert_eq!(first, fresh);
        pool.put(first.data);
        let again = img.gaussian_blur_pooled(1.5, 1.5, &pool);
        assert_eq!(again, fresh, "a reused (dirty) buffer gives the same result");
        assert!(again.data.as_ptr() == ptr || pool.take(1).len() == 1, "buffers are recycled");
        let reused = FloatImage::from_frames_into(&f, None, vec![9.0; 37 * 23 * 3]);
        assert_eq!(reused, img);
    }
}
