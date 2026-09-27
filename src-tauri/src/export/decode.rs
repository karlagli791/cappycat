//! Source decoding for the compositor: one ffmpeg child per clip streaming
//! `rawvideo rgb24` frames on stdout, with a tiny ring for forward playback
//! and chunked in-memory decoding for reversed clips.
//!
//! Frame indices live on a uniform grid of `decode_fps` ticks over the file's
//! timeline: the decode chain always contains `fps=<grid>:start_time=0`, so
//! output frame `k` is the source frame nearest (by timestamp) to `k / grid`.
//! This is what makes variable-frame-rate sources (phones: `r_frame_rate=60`,
//! ~24 fps average, irregular pts) map correctly — the index never depends on
//! the container's nominal rate. The grid is the project fps (finer for
//! slow-motion clips from high-fps sources, see `export::slow_grid_factor`).
//!
//! A decoder for index `i` seeks with `-ss` *before* `-i` to `i / grid − ε`
//! (ε = 1 ms): ffmpeg's accurate seek drops everything before that point and
//! restarts timestamps there, so grid tick 0 of the new stream is tick `i`.

use crate::jobs::TaskCtl;
use crate::render::sample::{BufPool, Frame};
use std::collections::VecDeque;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, ChildStdout, Stdio};
use std::sync::{Arc, Mutex};

/// Where a layer's pixels come from (the asset itself or an optical-flow
/// intermediate) and how source time maps to frame indices.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoSource {
    pub path: PathBuf,
    pub is_image: bool,
    /// constant frame rate the file is decoded at (and seeks are computed with)
    pub decode_fps: f64,
    /// source time (ms, in the asset's timebase) of file frame 0
    pub t0_ms: f64,
    /// frames per source millisecond
    pub index_rate: f64,
    /// number of decodable frames (upper bound; EOF is handled gracefully)
    pub frame_count: i64,
    /// decoded size (≤ the file size; downscaled when the project needs less)
    pub dec_w: usize,
    pub dec_h: usize,
    /// file size (a `scale` filter is added when it differs from dec_w/dec_h)
    pub file_w: usize,
    pub file_h: usize,
    /// cross-fade the two neighbouring decoded frames when the clip plays them sparser than this
    /// rate — i.e. when `decode_fps × speed < blend_below_fps` (0 = never blend). The pre-v2
    /// slow-motion rule "blend below 1×" is `decode_fps × 0.999`; frame-rate conversion
    /// (`frameBlend`) uses `project fps × 0.999`.
    pub blend_below_fps: f64,
    /// swscale `in_color_matrix` / `in_range` for YUV sources (`None` for RGB sources:
    /// see [`crate::ffmpeg::input_color_matrix`])
    pub in_matrix: Option<&'static str>,
    pub in_range: Option<&'static str>,
}

/// Which source frames an output frame needs: `a` blended towards `b` by `w`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameRequest {
    pub a: i64,
    pub b: i64,
    pub w: f32,
}

impl VideoSource {
    /// Frame(s) for a source time. Nearest (the frame whose display interval
    /// contains the time, like a `<video>` element) unless the clip plays its
    /// decoded frames sparser than `blend_below_fps` (see there).
    pub fn request(&self, source_ms: f64, rate: f64, frozen: bool) -> FrameRequest {
        if self.is_image {
            return FrameRequest { a: 0, b: 0, w: 0.0 };
        }
        let last = (self.frame_count - 1).max(0);
        let f = ((source_ms - self.t0_ms) * self.index_rate).max(0.0);
        if !frozen && self.decode_fps * rate.abs() < self.blend_below_fps {
            let a = f.floor();
            let w = (f - a) as f32;
            let a = (a as i64).min(last);
            if w < 1e-3 || a >= last {
                FrameRequest { a, b: a, w: 0.0 }
            } else if w > 0.999 {
                FrameRequest { a: a + 1, b: a + 1, w: 0.0 }
            } else {
                FrameRequest { a, b: a + 1, w }
            }
        } else {
            let a = ((f + 1e-3).floor() as i64).min(last);
            FrameRequest { a, b: a, w: 0.0 }
        }
    }

    pub fn frame_bytes(&self) -> usize {
        self.dec_w * self.dec_h * 3
    }

    /// ffmpeg arguments decoding `count` frames starting at `first`.
    pub fn decoder_args(&self, first: i64, count: i64) -> Vec<String> {
        let mut args: Vec<String> = vec!["-hide_banner".into(), "-v".into(), "error".into(), "-nostdin".into()];
        if self.is_image {
            args.extend(["-i".into(), self.path.to_string_lossy().into_owned(), "-frames:v".into(), "1".into()]);
        } else {
            let ss = first as f64 / self.decode_fps - (0.1 / self.decode_fps).min(0.001);
            if first > 0 && ss > 0.0 {
                args.extend(["-ss".into(), format!("{ss:.6}")]);
            }
            args.extend(["-i".into(), self.path.to_string_lossy().into_owned()]);
            args.extend(["-frames:v".into(), count.max(1).to_string()]);
        }
        args.extend(["-an".into(), "-sn".into(), "-dn".into()]);
        let mut vf: Vec<String> = Vec::new();
        if !self.is_image {
            vf.push(format!("fps={}:start_time=0", fmt_rate(self.decode_fps)));
        }
        // YUV → RGB with the source's matrix (untagged HD = BT.709, like the preview's
        // browser decoder and the BT.709-tagged output), not swscale's BT.601 default.
        let colour = match (self.in_matrix, self.in_range) {
            (Some(m), r) => format!(":in_color_matrix={m}:in_range={}", r.unwrap_or("tv")),
            _ => String::new(),
        };
        // Downscaling: `full_chroma_int` keeps swscale's area path exact (without it the
        // result is up to 2 levels off). Same size: the default conversion is within 1 level
        // and ~30 % cheaper than the exact path (the decoder shares the CPU with the compositor).
        if self.dec_w != self.file_w || self.dec_h != self.file_h {
            vf.push(format!("scale={}:{}:flags=area+full_chroma_int{colour}", self.dec_w, self.dec_h));
        } else if !colour.is_empty() {
            vf.push(format!("scale={}", colour.trim_start_matches(':')));
        }
        vf.push("format=rgb24".into());
        args.extend(["-vf".into(), vf.join(",")]);
        args.extend(["-f".into(), "rawvideo".into(), "-pix_fmt".into(), "rgb24".into(), "pipe:1".into()]);
        args
    }
}

/// A frame rate as ffmpeg accepts it: NTSC rates exactly (`23.976` → `24000/1001`),
/// anything else as a short decimal.
pub fn fmt_rate(v: f64) -> String {
    for base in [24u32, 30, 48, 60, 72, 96, 120, 144, 180, 240] {
        let ntsc = base as f64 * 1000.0 / 1001.0;
        if (v - ntsc).abs() < 0.005 {
            return format!("{}/1001", base * 1000);
        }
    }
    let s = format!("{v:.6}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// A running decoder.
struct Stream {
    child: Arc<Mutex<Child>>,
    stdout: ChildStdout,
    /// index of the next frame to be read
    next: i64,
    eof: bool,
    stderr_tail: Arc<Mutex<String>>,
}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Ok(mut c) = self.child.lock() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// Random-access-ish frame source over one [`VideoSource`].
pub struct FrameProvider {
    src: Arc<VideoSource>,
    ffmpeg: PathBuf,
    ctl: Arc<TaskCtl>,
    reverse: bool,
    /// needed index range (inclusive)
    lo: i64,
    hi: i64,
    chunk_frames: i64,
    stream: Option<Stream>,
    ring: VecDeque<(i64, Arc<Frame>)>,
    chunk: Vec<(i64, Arc<Frame>)>,
    image: Option<Arc<Frame>>,
    last_good: Option<Arc<Frame>>,
    pub warnings: Vec<String>,
    /// recycled frame buffers (frames evicted from the ring, released by the compositor)
    pub bytes: Arc<BufPool<u8>>,
}

/// Memory budget for one reversed clip's in-memory chunk.
const REVERSE_BUDGET_BYTES: usize = 1_200_000_000;

impl FrameProvider {
    pub fn new(src: Arc<VideoSource>, ffmpeg: PathBuf, ctl: Arc<TaskCtl>, reverse: bool, lo: i64, hi: i64) -> Self {
        let per = src.frame_bytes().max(1);
        let chunk_frames = (REVERSE_BUDGET_BYTES / per).clamp(8, 600) as i64;
        Self {
            src,
            ffmpeg,
            ctl,
            reverse,
            lo: lo.max(0),
            hi: hi.max(lo),
            chunk_frames,
            stream: None,
            ring: VecDeque::new(),
            chunk: Vec::new(),
            image: None,
            last_good: None,
            warnings: Vec::new(),
            bytes: BufPool::new(),
        }
    }

    /// Give a frame's buffer back to the pool if nobody else holds it.
    pub fn recycle(pool: &BufPool<u8>, f: Arc<Frame>) {
        if let Ok(f) = Arc::try_unwrap(f) {
            pool.put(f.data);
        }
    }

    fn spawn(&self, first: i64, count: i64) -> Result<Stream, String> {
        if self.ctl.is_cancelled() {
            return Err("cancelled".into());
        }
        let mut cmd = crate::ffmpeg::command(&self.ffmpeg);
        cmd.args(self.src.decoder_args(first, count)).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = crate::procs::spawn(&mut cmd).map_err(|e| format!("cannot start ffmpeg decoder: {e}"))?;
        let stdout = child.stdout.take().ok_or("decoder has no stdout")?;
        let stderr = child.stderr.take();
        let tail = Arc::new(Mutex::new(String::new()));
        if let Some(mut err) = stderr {
            let t = tail.clone();
            std::thread::spawn(move || {
                let mut s = String::new();
                let _ = err.read_to_string(&mut s);
                let keep: String = s.chars().rev().take(2000).collect::<Vec<_>>().into_iter().rev().collect();
                *t.lock().unwrap() = keep;
            });
        }
        let child = self.ctl.register(child);
        Ok(Stream { child, stdout, next: first, eof: false, stderr_tail: tail })
    }

    /// Read one frame from the stream (None at EOF) into a pooled buffer.
    fn read_frame(src: &VideoSource, stream: &mut Stream, pool: &BufPool<u8>) -> Option<Frame> {
        if stream.eof {
            return None;
        }
        let mut data = pool.take(src.frame_bytes());
        match stream.stdout.read_exact(&mut data) {
            Ok(()) => {
                stream.next += 1;
                Some(Frame { width: src.dec_w, height: src.dec_h, data })
            }
            Err(_) => {
                stream.eof = true;
                pool.put(data);
                None
            }
        }
    }

    fn fallback(&mut self, idx: i64, why: &str) -> Arc<Frame> {
        let msg = format!("{}: frame {idx} unavailable ({why}); holding the previous frame", self.src.path.display());
        if self.warnings.len() < 5 {
            self.warnings.push(msg);
        }
        match &self.last_good {
            Some(f) => f.clone(),
            None => Arc::new(Frame::black(self.src.dec_w, self.src.dec_h)),
        }
    }

    /// Fetch frame `idx` (clamped to the needed range).
    pub fn get(&mut self, idx: i64) -> Result<Arc<Frame>, String> {
        if self.ctl.is_cancelled() {
            return Err("cancelled".into());
        }
        if self.src.is_image {
            if self.image.is_none() {
                let mut s = self.spawn(0, 1)?;
                let f = Self::read_frame(&self.src, &mut s, &self.bytes);
                let tail = s.stderr_tail.clone();
                drop(s);
                match f {
                    Some(f) => self.image = Some(Arc::new(f)),
                    None => {
                        let t = tail.lock().unwrap().clone();
                        return Err(format!("cannot decode image {}: {}", self.src.path.display(), t.trim()));
                    }
                }
            }
            return Ok(self.image.clone().unwrap());
        }
        let idx = idx.clamp(self.lo, self.hi);
        let frame = if self.reverse { self.get_reverse(idx)? } else { self.get_forward(idx)? };
        self.last_good = Some(frame.clone());
        Ok(frame)
    }

    fn get_forward(&mut self, idx: i64) -> Result<Arc<Frame>, String> {
        if let Some((_, f)) = self.ring.iter().find(|(i, _)| *i == idx) {
            return Ok(f.clone());
        }
        let gap_limit = (self.src.decode_fps * 3.0).max(24.0) as i64;
        let restart = match &self.stream {
            None => true,
            Some(s) => idx < s.next || idx - s.next > gap_limit,
        };
        if restart {
            self.stream = None;
            self.stream = Some(self.spawn(idx, self.hi - idx + 2)?);
        }
        let stream = self.stream.as_mut().unwrap();
        while stream.next <= idx {
            let i = stream.next;
            match Self::read_frame(&self.src, stream, &self.bytes) {
                Some(f) => {
                    self.ring.push_back((i, Arc::new(f)));
                    while self.ring.len() > 4 {
                        if let Some((_, old)) = self.ring.pop_front() {
                            Self::recycle(&self.bytes, old);
                        }
                    }
                }
                None => break,
            }
            if self.ctl.is_cancelled() {
                return Err("cancelled".into());
            }
        }
        if let Some((_, f)) = self.ring.iter().find(|(i, _)| *i == idx) {
            return Ok(f.clone());
        }
        // EOF before idx: the newest frame at or below idx.
        if let Some((_, f)) = self.ring.iter().rev().find(|(i, _)| *i <= idx) {
            return Ok(f.clone());
        }
        let why = self.stream.as_ref().map(|s| s.stderr_tail.lock().unwrap().trim().to_string()).unwrap_or_default();
        Ok(self.fallback(idx, if why.is_empty() { "end of stream" } else { &why }))
    }

    fn get_reverse(&mut self, idx: i64) -> Result<Arc<Frame>, String> {
        if let Some(first) = self.chunk.first().map(|c| c.0) {
            let k = idx - first;
            if k >= 0 && (k as usize) < self.chunk.len() {
                return Ok(self.chunk[k as usize].1.clone());
            }
        }
        // Decode a chunk ending just above idx (so idx+1 is there for blending).
        let hi = (idx + 1).min(self.hi);
        let lo = (hi - self.chunk_frames + 1).max(self.lo).min(idx);
        for (_, f) in self.chunk.drain(..) {
            Self::recycle(&self.bytes, f);
        }
        let mut s = self.spawn(lo, hi - lo + 1)?;
        let mut i = lo;
        while i <= hi {
            match Self::read_frame(&self.src, &mut s, &self.bytes) {
                Some(f) => self.chunk.push((i, Arc::new(f))),
                None => break,
            }
            i += 1;
            if self.ctl.is_cancelled() {
                return Err("cancelled".into());
            }
        }
        let tail = s.stderr_tail.clone();
        drop(s);
        if let Some(first) = self.chunk.first().map(|c| c.0) {
            let k = (idx - first).clamp(0, self.chunk.len() as i64 - 1);
            return Ok(self.chunk[k as usize].1.clone());
        }
        let why = tail.lock().unwrap().trim().to_string();
        Ok(self.fallback(idx, if why.is_empty() { "end of stream" } else { &why }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src(fps: f64, blend: bool) -> VideoSource {
        VideoSource {
            path: PathBuf::from("C:/x.mp4"),
            is_image: false,
            decode_fps: fps,
            t0_ms: 0.0,
            index_rate: fps / 1000.0,
            frame_count: 100,
            dec_w: 64,
            dec_h: 36,
            file_w: 128,
            file_h: 72,
            blend_below_fps: if blend { fps * 0.999 } else { 0.0 },
            in_matrix: None,
            in_range: None,
        }
    }

    #[test]
    fn requests_nearest_or_blend() {
        let s = src(24.0, true);
        assert_eq!(s.request(1000.0, 1.0, false), FrameRequest { a: 24, b: 24, w: 0.0 });
        // 1000/24 ms * 2.5 → frame 2.5
        let r = s.request(2.5 * 1000.0 / 24.0, 0.25, false);
        assert_eq!((r.a, r.b), (2, 3));
        assert!((r.w - 0.5).abs() < 1e-4);
        // frozen / fast → nearest (floor)
        assert_eq!(s.request(2.5 * 1000.0 / 24.0, 0.0, true).w, 0.0);
        assert_eq!(s.request(2.5 * 1000.0 / 24.0, 2.0, false).a, 2);
        // clamped to the last frame
        assert_eq!(s.request(1e9, 1.0, false).a, 99);
        let no_blend = src(24.0, false);
        assert_eq!(no_blend.request(2.5 * 1000.0 / 24.0, 0.25, false).w, 0.0);
    }

    #[test]
    fn decoder_args_seek_scale_and_format() {
        let s = src(24.0, true);
        let a = s.decoder_args(48, 10).join(" ");
        assert!(a.contains("-ss 1.999000 -i C:/x.mp4 -frames:v 10"), "{a}");
        assert!(a.contains("-vf fps=24:start_time=0,scale=64:36:flags=area+full_chroma_int,format=rgb24"), "{a}");
        assert!(a.ends_with("-f rawvideo -pix_fmt rgb24 pipe:1"));
        assert!(!s.decoder_args(0, 5).join(" ").contains("-ss"));
        let mut full = src(29.97, false);
        full.dec_w = 128;
        full.dec_h = 72;
        assert!(full.decoder_args(0, 1).join(" ").contains("-vf fps=30000/1001:start_time=0,format=rgb24"));
        // YUV sources convert with their matrix, with or without a resize
        full.in_matrix = Some("bt709");
        full.in_range = Some("tv");
        assert!(full.decoder_args(0, 1).join(" ").contains("-vf fps=30000/1001:start_time=0,scale=in_color_matrix=bt709:in_range=tv,format=rgb24"));
        let mut small = src(24.0, true);
        small.in_matrix = Some("bt601");
        assert!(small.decoder_args(0, 1).join(" ").contains("scale=64:36:flags=area+full_chroma_int:in_color_matrix=bt601:in_range=tv,format=rgb24"));
    }

    #[test]
    fn ntsc_rates_are_exact() {
        assert_eq!(fmt_rate(24.0), "24");
        assert_eq!(fmt_rate(25.0), "25");
        assert_eq!(fmt_rate(24000.0 / 1001.0), "24000/1001");
        assert_eq!(fmt_rate(23.976), "24000/1001");
        assert_eq!(fmt_rate(29.97), "30000/1001");
        assert_eq!(fmt_rate(59.94), "60000/1001");
        assert_eq!(fmt_rate(2.0 * 24000.0 / 1001.0), "48000/1001");
        assert_eq!(fmt_rate(12.5), "12.5");
    }

    #[test]
    fn decodes_frames_forward_and_reverse_from_a_real_clip() {
        let Some(clip) = crate::ffmpeg::tests::synth_clip() else {
            eprintln!("SKIP: ffmpeg not available");
            return;
        };
        let bins = crate::ffmpeg::find_binaries().unwrap();
        let s = Arc::new(VideoSource {
            path: clip,
            is_image: false,
            decode_fps: 24.0,
            t0_ms: 0.0,
            index_rate: 0.024,
            frame_count: 48,
            dec_w: 160,
            dec_h: 120,
            file_w: 320,
            file_h: 240,
            blend_below_fps: 24.0 * 0.999,
            in_matrix: None,
            in_range: None,
        });
        let ctl = Arc::new(TaskCtl::default());
        let mut fwd = FrameProvider::new(s.clone(), bins.ffmpeg.clone(), ctl.clone(), false, 0, 47);
        let f10 = fwd.get(10).unwrap();
        let f11 = fwd.get(11).unwrap();
        let f30 = fwd.get(30).unwrap();
        assert_eq!(f10.data.len(), 160 * 120 * 3);
        assert_ne!(f10.data, f11.data, "testsrc changes every frame");
        let mut rev = FrameProvider::new(s, bins.ffmpeg.clone(), ctl, true, 0, 47);
        let r30 = rev.get(30).unwrap();
        let r10 = rev.get(10).unwrap();
        // same frames regardless of access pattern (seek accuracy)
        assert_eq!(r30.data, f30.data);
        assert_eq!(r10.data, f10.data);
        // past EOF → last decodable frame, no error
        assert!(rev.get(1000).is_ok());
    }
}
