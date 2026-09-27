//! Per-clip audio rendering for the mix: plays a clip's decoded PCM along its timeline → source
//! time map and applies its gain envelope.
//!
//! Three modes ([`PlayMode`]):
//!
//! * **Direct** — the clip plays at 1× everywhere (no speed curve): linear interpolation at the
//!   mapped position, exactly as before v2.
//! * **Stretch** — `audio.keepPitch` (default) and a speed ≠ 1 somewhere: pitch-preserving
//!   **WSOLA** (waveform-similarity overlap-add) that follows the speed map sample by sample, so
//!   ramps are handled natively. Output frames of [`WSOLA_FRAME`] samples (25 ms at 48 kHz, periodic
//!   Hann window, 50 % overlap → the windows sum to exactly 1) are centred every [`WSOLA_HOP`]
//!   output samples; frame `k` reads the source around the mapped position of its centre, shifted
//!   by up to ±[`WSOLA_TOLERANCE`] samples (10 ms) to the offset whose first half best matches
//!   (normalised cross-correlation of the L+R sum; coarse search every 4 samples on every 4th
//!   sample, then ±3 at full resolution) the natural continuation of the previous frame. Tempo
//!   0.1×–10×. Freeze holds are silent and reset the similarity chain; reversed clips run WSOLA
//!   over the time-reversed PCM (they play backwards, as before).
//! * **Varispeed** — `keepPitch: false`: pitch follows speed (as before v2); when the clip plays
//!   faster than 1× every output sample is a Blackman-windowed sinc interpolation with its cutoff
//!   at `0.92 × 0.5 / speed` of the source rate (anti-alias low-pass, 8 lobes each side at the
//!   scaled rate); ≤ 1× uses linear interpolation.
//!
//! Gain at clip-local timeline time `t` (ms), `L` = the clip's timeline length:
//! `dbToLin(gainDb + volume(t)) × fadeIn(t) × fadeOut(t)` (0 when muted — muted clips are not
//! mixed at all), with `fadeIn(t) = sin(π/2 · clamp(t / fi, 0, 1))`,
//! `fadeOut(t) = sin(π/2 · clamp((L − t) / fo, 0, 1))` (equal-power), `fi`, `fo` = `fadeInMs`,
//! `fadeOutMs` clamped to `L/2`, and `volume` the keyframed dB offset (outgoing-keyframe easing,
//! keyframes crate), evaluated every [`GAIN_STEP`] samples and interpolated linearly in between.

use crate::model::Clip;
use crate::render::timemap::ClipTimeMap;

pub const CHANNELS: usize = 2;
/// WSOLA frame length (samples at 48 kHz: 25 ms).
pub const WSOLA_FRAME: usize = 1200;
/// Output hop between WSOLA frames (half a frame).
pub const WSOLA_HOP: usize = WSOLA_FRAME / 2;
/// Maximum similarity shift (samples, 10 ms).
pub const WSOLA_TOLERANCE: usize = 480;
/// The volume envelope is evaluated every `GAIN_STEP` output samples (≈ 0.33 ms).
pub const GAIN_STEP: usize = 16;

/// How a clip's PCM is played.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayMode {
    Direct,
    Stretch,
    Varispeed,
}

/// The mode for a clip (see the module docs).
pub fn play_mode(clip: &Clip) -> PlayMode {
    let constant_1x = clip.speed.points.iter().all(|p| (p.speed - 1.0).abs() < 1e-9);
    if constant_1x {
        PlayMode::Direct
    } else if clip.audio.keep_pitch() {
        PlayMode::Stretch
    } else {
        PlayMode::Varispeed
    }
}

/// Equal-power fade factor and dB gain of a clip at clip-local time `t` (ms).
#[derive(Debug, Clone)]
pub struct GainEnvelope {
    gain_db: f64,
    volume: Option<keyframes::Keyframed<f64>>,
    len_ms: f64,
    fade_in: f64,
    fade_out: f64,
}

impl GainEnvelope {
    pub fn new(clip: &Clip, len_ms: f64) -> Self {
        let half = (len_ms / 2.0).max(0.0);
        let a = &clip.audio;
        let clamp = |v: Option<f64>| v.filter(|x| x.is_finite()).unwrap_or(0.0).clamp(0.0, half);
        Self {
            gain_db: a.gain_db,
            volume: a.volume.clone().filter(|v| v.is_animated() || v.static_value != 0.0),
            len_ms,
            fade_in: clamp(a.fade_in_ms),
            fade_out: clamp(a.fade_out_ms),
        }
    }

    /// Is the gain the same everywhere (no volume keyframes, no fades)?
    pub fn is_constant(&self) -> bool {
        self.volume.as_ref().map(|v| !v.is_animated()).unwrap_or(true) && self.fade_in <= 0.0 && self.fade_out <= 0.0
    }

    /// Linear gain at clip-local time `t_ms`.
    pub fn at(&self, t_ms: f64) -> f32 {
        let db = self.gain_db + self.volume.as_ref().map(|v| v.evaluate(t_ms)).unwrap_or(0.0);
        let mut g = 10f64.powf(db / 20.0);
        let half_pi = std::f64::consts::FRAC_PI_2;
        if self.fade_in > 0.0 {
            g *= (half_pi * (t_ms / self.fade_in).clamp(0.0, 1.0)).sin();
        }
        if self.fade_out > 0.0 {
            g *= (half_pi * ((self.len_ms - t_ms) / self.fade_out).clamp(0.0, 1.0)).sin();
        }
        g as f32
    }
}

/* -------------------------------------------------------------------- WSOLA */

/// Streaming WSOLA generator over one clip. Output sample `j` is clip-local (`j / sr` ms after
/// the clip's start on the timeline).
struct Wsola {
    /// source PCM (interleaved stereo; time-reversed for reversed clips)
    pcm: Vec<f32>,
    /// L+R sum of `pcm` (similarity search)
    mono: Vec<f32>,
    window: Vec<f32>,
    /// pending overlap-add output (interleaved), `acc[0]` = clip-local sample `acc_base`
    acc: Vec<f32>,
    acc_base: i64,
    /// next frame index to add
    next_k: i64,
    /// source start of the previous frame (None at the start and after a freeze hold)
    prev_src: Option<i64>,
}

impl Wsola {
    fn new(pcm: Vec<f32>, first_sample: i64) -> Self {
        let mono: Vec<f32> = pcm.chunks_exact(CHANNELS).map(|f| f[0] + f[1]).collect();
        let n = WSOLA_FRAME;
        let window = (0..n).map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos()) as f32).collect();
        // the first frame whose second half covers `first_sample`
        let k0 = first_sample.div_euclid(WSOLA_HOP as i64) - 1;
        let base = k0 * WSOLA_HOP as i64 - (n / 2) as i64;
        Self { pcm, mono, window, acc: Vec::new(), acc_base: base, next_k: k0, prev_src: None }
    }

    fn frames(&self) -> i64 {
        self.mono.len() as i64
    }

    /// Normalised similarity of the candidate at `cand` against the reference at `refp` over
    /// `len` samples, reading every `step`-th sample.
    fn similarity(&self, refp: i64, cand: i64, len: usize, step: usize) -> f32 {
        let (mut xy, mut yy) = (0.0f32, 1e-9f32);
        let n = self.frames();
        let mut i = 0usize;
        while i < len {
            let (a, b) = (refp + i as i64, cand + i as i64);
            if a >= 0 && a < n && b >= 0 && b < n {
                let (x, y) = (self.mono[a as usize], self.mono[b as usize]);
                xy += x * y;
                yy += y * y;
            }
            i += step;
        }
        xy / yy.sqrt()
    }

    /// Best source start for a frame nominally at `nominal`, continuing from `prev`.
    fn best_start(&self, nominal: i64, prev: i64) -> i64 {
        let natural = prev + WSOLA_HOP as i64;
        let tol = WSOLA_TOLERANCE as i64;
        let len = WSOLA_HOP;
        let (mut best, mut best_s) = (nominal, f32::NEG_INFINITY);
        let mut d = -tol;
        while d <= tol {
            let s = self.similarity(natural, nominal + d, len, 4);
            if s > best_s {
                best_s = s;
                best = nominal + d;
            }
            d += 4;
        }
        let coarse = best;
        best_s = f32::NEG_INFINITY;
        for d in -3..=3i64 {
            let c = coarse + d;
            if (c - nominal).abs() > tol {
                continue;
            }
            let s = self.similarity(natural, c, len, 1);
            if s > best_s {
                best_s = s;
                best = c;
            }
        }
        best
    }

    /// Add frame `k` (centred on clip-local output sample `k·hop`) reading around `pos`.
    fn add_frame(&mut self, k: i64, pos: Option<f64>) {
        let n = WSOLA_FRAME;
        let out0 = k * WSOLA_HOP as i64 - (n / 2) as i64;
        let need = ((out0 - self.acc_base) as usize + n) * CHANNELS;
        if self.acc.len() < need {
            self.acc.resize(need, 0.0);
        }
        let Some(pos) = pos else {
            self.prev_src = None;
            return;
        };
        let nominal = pos.round() as i64 - (n / 2) as i64;
        let start = match self.prev_src {
            Some(p) => self.best_start(nominal, p),
            None => nominal,
        };
        self.prev_src = Some(start);
        let frames = self.frames();
        let o = ((out0 - self.acc_base) as usize) * CHANNELS;
        for j in 0..n {
            let s = start + j as i64;
            if s < 0 || s >= frames {
                continue;
            }
            let w = self.window[j];
            let si = s as usize * CHANNELS;
            for ch in 0..CHANNELS {
                self.acc[o + j * CHANNELS + ch] += self.pcm[si + ch] * w;
            }
        }
    }

    /// Produce clip-local samples `[j0, j0 + out.len()/2)` (must be requested in increasing,
    /// contiguous order). `pos(j)` maps a clip-local output sample to a source position in `pcm`
    /// samples (`None` while frozen).
    fn render(&mut self, j0: i64, out: &mut [f32], pos: &dyn Fn(i64) -> Option<f64>) {
        let count = (out.len() / CHANNELS) as i64;
        let end = j0 + count;
        // every frame starting before `end` contributes; later ones cannot touch [j0, end)
        while self.next_k * WSOLA_HOP as i64 - (WSOLA_FRAME / 2) as i64 <= end {
            let k = self.next_k;
            self.add_frame(k, pos(k * WSOLA_HOP as i64));
            self.next_k += 1;
        }
        for (i, dst) in out.chunks_exact_mut(CHANNELS).enumerate() {
            let j = j0 + i as i64;
            let a = ((j - self.acc_base) * CHANNELS as i64) as usize;
            for (ch, d) in dst.iter_mut().enumerate() {
                *d = self.acc.get(a + ch).copied().unwrap_or(0.0);
            }
        }
        // samples before `end` are final: drop them
        let drop = ((end - self.acc_base).max(0) as usize * CHANNELS).min(self.acc.len());
        self.acc.drain(..drop);
        self.acc_base += (drop / CHANNELS) as i64;
    }
}

/* ---------------------------------------------------------------- varispeed */

/// Blackman-windowed sinc interpolation of `pcm` at fractional frame `x` with cutoff `fc`
/// (cycles per source sample, ≤ 0.5) and half-width `hw` samples.
fn sinc_sample(pcm: &[f32], frames: usize, x: f64, fc: f64, hw: f64) -> [f32; 2] {
    let lo = (x - hw).ceil().max(0.0) as usize;
    let hi = ((x + hw).floor() as i64).min(frames as i64 - 1);
    let (mut l, mut r, mut wsum) = (0.0f64, 0.0f64, 0.0f64);
    if hi < lo as i64 {
        return [0.0; 2];
    }
    for i in lo..=hi as usize {
        let d = i as f64 - x;
        let arg = 2.0 * fc * d;
        let sinc = if arg.abs() < 1e-9 { 1.0 } else { (std::f64::consts::PI * arg).sin() / (std::f64::consts::PI * arg) };
        let z = (d / hw).clamp(-1.0, 1.0); // −1..1 → Blackman over the kernel
        let t = std::f64::consts::PI * (z + 1.0);
        let win = 0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos();
        let k = sinc * win;
        wsum += k;
        l += pcm[i * CHANNELS] as f64 * k;
        r += pcm[i * CHANNELS + 1] as f64 * k;
    }
    if wsum.abs() < 1e-9 {
        return [0.0; 2];
    }
    // normalise the DC gain (the kernel's taps sum to ~2·fc·… ; unity at DC after this)
    [(l / wsum) as f32, (r / wsum) as f32]
}

/// Linear interpolation at fractional frame `x` (0 outside the buffer).
fn linear_sample(pcm: &[f32], frames: usize, x: f64) -> Option<[f32; 2]> {
    if x < 0.0 {
        return None;
    }
    let k = x.floor() as usize;
    if k >= frames {
        return None;
    }
    let f = (x - k as f64) as f32;
    let k1 = (k + 1).min(frames - 1);
    let mut o = [0.0; 2];
    for (ch, v) in o.iter_mut().enumerate() {
        let a = pcm[k * CHANNELS + ch];
        let b = pcm[k1 * CHANNELS + ch];
        *v = a + (b - a) * f;
    }
    Some(o)
}

/* ------------------------------------------------------------------- voice */

/// One clip's player: decoded PCM + time map + gain, producing its samples block by block.
pub struct ClipVoice {
    mode: PlayMode,
    map: ClipTimeMap,
    clip_start_ms: f64,
    reversed: bool,
    /// source time (ms) of the first PCM frame
    win_start_ms: f64,
    frames: usize,
    pcm: Vec<f32>,
    gain: GainEnvelope,
    wsola: Option<Wsola>,
    /// clip-local output sample the next stretch block starts at
    next_j: Option<i64>,
    sample_rate: f64,
}

impl ClipVoice {
    /// `pcm`: interleaved stereo at `sample_rate` covering source time `win_start_ms…`.
    pub fn new(clip: &Clip, map: ClipTimeMap, pcm: Vec<f32>, win_start_ms: f64, sample_rate: u32) -> Self {
        let frames = pcm.len() / CHANNELS;
        let gain = GainEnvelope::new(clip, map.total_ms());
        Self {
            mode: play_mode(clip),
            map,
            clip_start_ms: clip.start_ms,
            reversed: clip.reversed,
            win_start_ms,
            frames,
            pcm,
            gain,
            wsola: None,
            next_j: None,
            sample_rate: sample_rate as f64,
        }
    }

    pub fn mode(&self) -> PlayMode {
        self.mode
    }

    /// Add this clip's samples for timeline times `t0_ms + i / sr` (`i` in `0..out.len()/2`) to
    /// `out` (interleaved stereo). Calls must cover increasing, contiguous time ranges.
    pub fn mix_into(&mut self, t0_ms: f64, out: &mut [f32]) {
        let spm = self.sample_rate / 1000.0;
        let n = out.len() / CHANNELS;
        if n == 0 || self.frames == 0 {
            return;
        }
        let mut buf = vec![0.0f32; n * CHANNELS];
        match self.mode {
            PlayMode::Stretch => self.render_stretch(t0_ms, &mut buf),
            PlayMode::Direct | PlayMode::Varispeed => {
                for (i, dst) in buf.chunks_exact_mut(CHANNELS).enumerate() {
                    let local = t0_ms + i as f64 / spm - self.clip_start_ms;
                    let (_, source_ms, rate, frozen) = self.map.source_at(local);
                    if frozen {
                        continue;
                    }
                    let x = (source_ms - self.win_start_ms) * spm;
                    let s = if self.mode == PlayMode::Varispeed && rate > 1.0 + 1e-6 {
                        if x < 0.0 || x >= self.frames as f64 {
                            continue;
                        }
                        let fc = 0.5 * 0.92 / rate;
                        let hw = 8.0 * rate;
                        sinc_sample(&self.pcm, self.frames, x, fc, hw)
                    } else {
                        match linear_sample(&self.pcm, self.frames, x) {
                            Some(s) => s,
                            None => continue,
                        }
                    };
                    dst.copy_from_slice(&s);
                }
            }
        }
        // gain envelope
        let local0 = t0_ms - self.clip_start_ms;
        if self.gain.is_constant() {
            let g = self.gain.at(local0);
            for (o, b) in out.iter_mut().zip(&buf) {
                *o += b * g;
            }
        } else {
            let mut i = 0usize;
            while i < n {
                let e = (i + GAIN_STEP).min(n);
                let (ga, gb) = (self.gain.at(local0 + i as f64 / spm), self.gain.at(local0 + e as f64 / spm));
                for j in i..e {
                    let g = ga + (gb - ga) * ((j - i) as f32 / (e - i) as f32);
                    for ch in 0..CHANNELS {
                        out[j * CHANNELS + ch] += buf[j * CHANNELS + ch] * g;
                    }
                }
                i = e;
            }
        }
    }

    fn render_stretch(&mut self, t0_ms: f64, buf: &mut [f32]) {
        let spm = self.sample_rate / 1000.0;
        // clip-local output sample index of t0 (sub-sample offsets of the clip start ignored);
        // later blocks continue exactly where the previous one ended
        let j0 = self.next_j.unwrap_or_else(|| ((t0_ms - self.clip_start_ms) * spm).round() as i64);
        self.next_j = Some(j0 + (buf.len() / CHANNELS) as i64);
        if self.wsola.is_none() {
            let pcm = if self.reversed {
                let mut rev = Vec::with_capacity(self.pcm.len());
                for f in self.pcm.chunks_exact(CHANNELS).rev() {
                    rev.extend_from_slice(f);
                }
                rev
            } else {
                self.pcm.clone()
            };
            self.wsola = Some(Wsola::new(pcm, j0));
        }
        let (map, win, frames, rev, total) = (&self.map, self.win_start_ms, self.frames as f64, self.reversed, self.map.total_ms());
        let pos = move |j: i64| -> Option<f64> {
            let local = j as f64 / spm;
            if local < 0.0 || local > total {
                return None;
            }
            let (_, source_ms, _, frozen) = map.source_at(local);
            if frozen {
                return None;
            }
            let x = (source_ms - win) * spm;
            Some(if rev { frames - 1.0 - x } else { x })
        };
        let w = self.wsola.as_mut().unwrap();
        w.render(j0, buf, &pos);
        // freeze holds are silent sample-exactly (as before v2)
        if self.map_has_freeze() {
            for (i, dst) in buf.chunks_exact_mut(CHANNELS).enumerate() {
                let local = t0_ms + i as f64 / spm - self.clip_start_ms;
                if self.map.source_at(local).3 {
                    dst.fill(0.0);
                }
            }
        }
    }

    fn map_has_freeze(&self) -> bool {
        self.map.has_freeze()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::*;

    const SR: u32 = 48_000;

    fn tone(freq: f64, secs: f64, amp: f32) -> Vec<f32> {
        let n = (secs * SR as f64) as usize;
        let mut v = Vec::with_capacity(n * 2);
        for i in 0..n {
            let s = amp * (std::f64::consts::TAU * freq * i as f64 / SR as f64).sin() as f32;
            v.push(s);
            v.push(s);
        }
        v
    }

    /// Render a clip over its whole timeline length in blocks of `block` samples.
    fn render(clip: &Clip, pcm: Vec<f32>, block: usize) -> Vec<f32> {
        let map = ClipTimeMap::new(clip);
        let total = map.total_ms();
        let n = (total * SR as f64 / 1000.0).round() as usize;
        let mut v = ClipVoice::new(clip, map, pcm, clip.in_ms, SR);
        let mut out = vec![0.0f32; n * 2];
        let mut i = 0;
        while i < n {
            let e = (i + block).min(n);
            v.mix_into(clip.start_ms + i as f64 * 1000.0 / SR as f64, &mut out[i * 2..e * 2]);
            i = e;
        }
        out
    }

    fn left(v: &[f32]) -> Vec<f32> {
        v.chunks_exact(2).map(|f| f[0]).collect()
    }

    /// Radix-2 FFT magnitude peak frequency of `x` (zero-padded to a power of two, Hann window).
    fn fft_peak_hz(x: &[f32]) -> f64 {
        let n = x.len().next_power_of_two();
        let mut re: Vec<f64> = vec![0.0; n];
        let mut im: Vec<f64> = vec![0.0; n];
        for (i, v) in x.iter().enumerate() {
            let w = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / x.len() as f64).cos();
            re[i] = *v as f64 * w;
        }
        // bit reversal
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let ang = -std::f64::consts::TAU / len as f64;
            for s in (0..n).step_by(len) {
                for k in 0..len / 2 {
                    let (wr, wi) = ((ang * k as f64).cos(), (ang * k as f64).sin());
                    let (a, b) = (s + k, s + k + len / 2);
                    let (xr, xi) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                }
            }
            len <<= 1;
        }
        let (mut best, mut bi) = (0.0, 0);
        for i in 1..n / 2 {
            let m = re[i] * re[i] + im[i] * im[i];
            if m > best {
                best = m;
                bi = i;
            }
        }
        // parabolic interpolation on the log magnitude
        let mag = |i: usize| (re[i] * re[i] + im[i] * im[i]).sqrt().max(1e-12).ln();
        let (a, b, c) = (mag(bi - 1), mag(bi), mag(bi + 1));
        let off = 0.5 * (a - c) / (a - 2.0 * b + c);
        (bi as f64 + off) * SR as f64 / n as f64
    }

    fn rms(x: &[f32]) -> f64 {
        (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64).sqrt()
    }

    fn clip_with_speed(speed: SpeedCurve, src_ms: f64) -> Clip {
        Clip { id: "c".into(), start_ms: 1000.0, in_ms: 0.0, out_ms: src_ms, speed, ..Default::default() }
    }

    #[test]
    fn keep_pitch_at_2x_and_half_speed() {
        for (speed, want_ms) in [(2.0, 1000.0), (0.5, 4000.0)] {
            let clip = clip_with_speed(SpeedCurve::constant(speed), 2000.0);
            assert_eq!(play_mode(&clip), PlayMode::Stretch);
            let out = left(&render(&clip, tone(440.0, 2.0, 0.5), 4801));
            let got_ms = out.len() as f64 * 1000.0 / SR as f64;
            assert!((got_ms - want_ms).abs() < 1.0, "{speed}x lasts {got_ms} ms");
            // steady part (skip 30 ms at both ends)
            let m = (0.03 * SR as f64) as usize;
            let body = &out[m..out.len() - m];
            let peak = fft_peak_hz(body);
            assert!((peak - 440.0).abs() < 2.0, "{speed}x: peak at {peak:.2} Hz");
            let r = rms(body);
            assert!((r - 0.5 / 2f64.sqrt()).abs() < 0.05, "{speed}x: level {r}");
            // sound until the very end of the clip
            assert!(rms(&out[out.len() - m..out.len() - m / 2]) > 0.2);
        }
    }

    #[test]
    fn varispeed_follows_speed_and_filters_aliases() {
        let mut clip = clip_with_speed(SpeedCurve::constant(2.0), 2000.0);
        clip.audio.keep_pitch = Some(false);
        assert_eq!(play_mode(&clip), PlayMode::Varispeed);
        let out = left(&render(&clip, tone(440.0, 2.0, 0.5), 10_000));
        let peak = fft_peak_hz(&out[2000..out.len() - 2000]);
        assert!((peak - 880.0).abs() < 2.0, "pitch doubles: {peak}");
        // a 15 kHz tone at 2x would alias to 18 kHz; the low-pass removes it
        let out = left(&render(&clip, tone(15_000.0, 2.0, 0.5), 10_000));
        let r = rms(&out[2000..out.len() - 2000]);
        assert!(r < 0.01, "anti-alias: residual {r}");
        // 5 kHz at 2x = 10 kHz: passes
        let out = left(&render(&clip, tone(5_000.0, 2.0, 0.5), 10_000));
        assert!(rms(&out[2000..out.len() - 2000]) > 0.3);
    }

    #[test]
    fn ramps_are_continuous_and_blocks_do_not_matter() {
        let clip = clip_with_speed(SpeedCurve::custom(vec![SpeedPoint::new(0.0, 0.5), SpeedPoint::new(0.5, 2.0), SpeedPoint::new(1.0, 0.7)]), 3000.0);
        let a = render(&clip, tone(440.0, 3.0, 0.5), 480);
        let b = render(&clip, tone(440.0, 3.0, 0.5), 7777);
        assert_eq!(a, b, "block size never changes the result");
        let l = left(&a);
        // a 440 Hz sine of amplitude 0.5 moves at most 0.029 per sample; WSOLA splices stay close
        let max_jump = l.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        assert!(max_jump < 0.06, "no clicks: max step {max_jump}");
        let m = 2000;
        let peak = fft_peak_hz(&l[m..l.len() - m]);
        assert!((peak - 440.0).abs() < 3.0, "ramp keeps the pitch: {peak}");
        // no dropouts: 10 ms RMS windows all carry the tone
        let min_rms = l[m..l.len() - m].chunks(480).map(rms).fold(f64::INFINITY, f64::min);
        assert!(min_rms > 0.25, "no dropouts: {min_rms}");
    }

    #[test]
    fn speech_like_signal_keeps_pitch_and_level() {
        // glottal-pulse-like harmonic series (f0 = 140 Hz, 1/k amplitudes) under a 4 Hz
        // syllable envelope with pauses
        let secs = 3.0;
        let n = (secs * SR as f64) as usize;
        let mut pcm = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = i as f64 / SR as f64;
            let f0 = 140.0 * (1.0 + 0.05 * (std::f64::consts::TAU * 0.7 * t).sin());
            let phase = std::f64::consts::TAU * 140.0 * t + 0.05 * 140.0 / 0.7 * (1.0 - (std::f64::consts::TAU * 0.7 * t).cos());
            let _ = f0;
            let mut s = 0.0;
            for k in 1..=12 {
                let formant = if (500.0..900.0).contains(&(140.0 * k as f64)) { 2.0 } else { 1.0 };
                s += formant * (k as f64 * phase).sin() / k as f64;
            }
            let env = ((std::f64::consts::TAU * 4.0 * t).sin().max(0.0)).powf(0.5);
            let v = (0.15 * s * env) as f32;
            pcm.push(v);
            pcm.push(v);
        }
        let src_rms = rms(&left(&pcm));
        for speed in [1.5, 0.75] {
            let clip = clip_with_speed(SpeedCurve::constant(speed), secs * 1000.0);
            let out = left(&render(&clip, pcm.clone(), 4800));
            assert!(out.iter().all(|v| v.is_finite()));
            let r = rms(&out);
            assert!((r / src_rms - 1.0).abs() < 0.2, "{speed}x level {r} vs {src_rms}");
            // fundamental by autocorrelation over 60..400 Hz on a voiced stretch
            let seg = &out[out.len() / 3..out.len() / 3 + 9600];
            let best = (120..=800)
                .max_by(|a, b| {
                    let c = |lag: usize| seg.iter().zip(&seg[lag..]).map(|(x, y)| x * y).sum::<f32>();
                    c(*a).total_cmp(&c(*b))
                })
                .unwrap();
            let f0 = SR as f64 / best as f64;
            assert!((f0 - 140.0).abs() < 12.0, "{speed}x: f0 {f0:.1} Hz");
        }
    }

    #[test]
    fn reverse_and_freeze_with_stretch() {
        // a rising chirp played reversed at 2x falls
        let n = 2 * SR as usize;
        let mut pcm = Vec::with_capacity(n * 2);
        let mut phase = 0.0f64;
        for i in 0..n {
            let f = 300.0 + 400.0 * i as f64 / n as f64;
            phase += std::f64::consts::TAU * f / SR as f64;
            let v = 0.5 * phase.sin() as f32;
            pcm.push(v);
            pcm.push(v);
        }
        let mut clip = clip_with_speed(SpeedCurve::constant(2.0), 2000.0);
        clip.reversed = true;
        let out = left(&render(&clip, pcm.clone(), 4800));
        let early = fft_peak_hz(&out[1000..9000]);
        let late = fft_peak_hz(&out[out.len() - 9000..out.len() - 1000]);
        assert!(early > late + 150.0, "reversed: {early:.0} Hz then {late:.0} Hz");
        // freeze hold: silence during the hold
        let mut clip = clip_with_speed(SpeedCurve::constant(2.0), 2000.0);
        clip.freeze_frame = Some(FreezeFrame { at_ms: 400.0, hold_ms: 500.0 });
        let out = left(&render(&clip, tone(440.0, 2.0, 0.5), 4800));
        let at = |ms: f64| (ms * SR as f64 / 1000.0) as usize;
        assert!(out[at(450.0)..at(850.0)].iter().all(|v| *v == 0.0), "silent while frozen");
        assert!(rms(&out[at(950.0)..at(1300.0)]) > 0.25, "sound resumes");
    }

    #[test]
    fn gain_envelope_fades_and_volume_keyframes() {
        let mut clip = Clip { out_ms: 4000.0, ..Default::default() };
        clip.audio.gain_db = -6.0;
        clip.audio.fade_in_ms = Some(1000.0);
        clip.audio.fade_out_ms = Some(10_000.0); // clamped to half the clip
        let g = GainEnvelope::new(&clip, 4000.0);
        let lin = |db: f64| 10f64.powf(db / 20.0) as f32;
        assert_eq!(g.at(0.0), 0.0);
        assert!((g.at(500.0) - lin(-6.0) * (std::f32::consts::FRAC_PI_4).sin() * (std::f32::consts::FRAC_PI_2 * 3500.0 / 2000.0).min(std::f32::consts::FRAC_PI_2).sin()).abs() < 1e-6);
        assert!((g.at(2000.0) - lin(-6.0)).abs() < 1e-6, "full level in the middle");
        assert!(g.at(4000.0).abs() < 1e-7, "silent at the end");
        // equal power: fade-in² + fade-out² = 1 at the same relative position
        let fi = (std::f64::consts::FRAC_PI_2 * 0.3).sin();
        let fo = (std::f64::consts::FRAC_PI_2 * 0.7).sin();
        assert!((fi * fi + fo * fo - 1.0).abs() < 1e-12);
        // volume keyframes in dB with outgoing easing
        let mut clip = Clip { out_ms: 4000.0, ..Default::default() };
        clip.audio.volume = Some(Keyframed::with_keyframes(
            0.0,
            vec![Keyframe::new(1000.0, 0.0).with_easing(Easing::EaseInOut), Keyframe::new(2000.0, -20.0)],
        ));
        let g = GainEnvelope::new(&clip, 4000.0);
        assert!(!g.is_constant());
        assert!((g.at(500.0) - 1.0).abs() < 1e-6, "held before the first keyframe");
        assert!((g.at(1500.0) - lin(-10.0)).abs() < 1e-4, "ease-in-out midpoint = half the dB");
        assert!((g.at(1250.0) - lin(-20.0 * Easing::EaseInOut.apply(0.25, None))).abs() < 1e-5);
        assert!((g.at(3000.0) - lin(-20.0)).abs() < 1e-6);
        // a rendered fade: the first samples are silent, the level rises smoothly
        let mut clip = clip_with_speed(SpeedCurve::default(), 2000.0);
        clip.speed.points.clear();
        clip.audio.fade_in_ms = Some(500.0);
        let out = left(&render(&clip, tone(440.0, 2.0, 0.5), 1000));
        assert!(rms(&out[..240]) < 0.02);
        assert!(rms(&out[SR as usize..SR as usize + 4800]) > 0.34);
    }
}
