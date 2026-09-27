//! Speed-ramp engine.
//!
//! A [`SpeedCurve`] is a list of `(t, speed)` control points where `t ∈ [0,1]`
//! is the normalised position over the **source** range of a clip and `speed`
//! is the playback multiplier (`0.1 ..= 10`). Between points the speed is
//! interpolated with a monotone cubic (Fritsch–Carlson), so a ramp never
//! overshoots its control values.
//!
//! Playback (output) time is the integral of `1 / speed` over source time.
//! [`SpeedLut`] pre-integrates that on a uniform grid so both directions of the
//! mapping (`source → output`, `output → source`) are O(log n) lookups.

use serde::{Deserialize, Serialize};

/// Minimum speed multiplier accepted by the engine (CapCut: 0.1x).
pub const MIN_SPEED: f64 = 0.1;
/// Maximum speed multiplier accepted by the engine (CapCut: 10x).
pub const MAX_SPEED: f64 = 10.0;
/// Default LUT resolution (samples over the source range).
pub const DEFAULT_LUT_SAMPLES: usize = 1024;

/// CapCut-style preset identifiers, serialised in snake_case (`"hero_time"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeedPreset {
    #[default]
    Normal,
    Montage,
    HeroTime,
    Bullet,
    JumpCut,
    FlashIn,
    FlashOut,
    Custom,
}

impl SpeedPreset {
    /// All named presets (excluding `Custom`), in UI order.
    pub const ALL: [SpeedPreset; 7] = [
        SpeedPreset::Normal,
        SpeedPreset::Montage,
        SpeedPreset::HeroTime,
        SpeedPreset::Bullet,
        SpeedPreset::JumpCut,
        SpeedPreset::FlashIn,
        SpeedPreset::FlashOut,
    ];

    /// Parse the snake_case JSON form.
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "normal" => SpeedPreset::Normal,
            "montage" => SpeedPreset::Montage,
            "hero_time" => SpeedPreset::HeroTime,
            "bullet" => SpeedPreset::Bullet,
            "jump_cut" => SpeedPreset::JumpCut,
            "flash_in" => SpeedPreset::FlashIn,
            "flash_out" => SpeedPreset::FlashOut,
            "custom" => SpeedPreset::Custom,
            _ => return None,
        })
    }

    /// The control points that define this preset. `Custom` yields the
    /// `Normal` points (a custom curve is expected to carry its own points).
    pub fn points(self) -> Vec<SpeedPoint> {
        let pts: &[(f64, f64)] = match self {
            SpeedPreset::Normal | SpeedPreset::Custom => &[(0.0, 1.0), (1.0, 1.0)],
            // Rhythmic push/pull between half and double speed.
            SpeedPreset::Montage => &[
                (0.0, 1.0),
                (0.15, 2.0),
                (0.3, 0.5),
                (0.5, 2.0),
                (0.7, 0.5),
                (0.85, 2.0),
                (1.0, 1.0),
            ],
            // Normal → dramatic slow-motion in the middle → normal.
            SpeedPreset::HeroTime => &[(0.0, 1.0), (0.3, 1.0), (0.5, 0.3), (0.7, 1.0), (1.0, 1.0)],
            // Fast approach, "bullet time" freeze, fast exit.
            SpeedPreset::Bullet => &[
                (0.0, 3.0),
                (0.3, 3.0),
                (0.45, 0.2),
                (0.55, 0.2),
                (0.7, 3.0),
                (1.0, 3.0),
            ],
            // Alternating normal / 4x bursts.
            SpeedPreset::JumpCut => &[
                (0.0, 1.0),
                (0.2, 1.0),
                (0.25, 4.0),
                (0.4, 4.0),
                (0.45, 1.0),
                (0.6, 1.0),
                (0.65, 4.0),
                (0.8, 4.0),
                (0.85, 1.0),
                (1.0, 1.0),
            ],
            // Fast start settling into normal speed.
            SpeedPreset::FlashIn => &[(0.0, 4.0), (0.3, 1.0), (1.0, 1.0)],
            // Normal speed accelerating out.
            SpeedPreset::FlashOut => &[(0.0, 1.0), (0.7, 1.0), (1.0, 4.0)],
        };
        pts.iter().map(|&(t, speed)| SpeedPoint { t, speed }).collect()
    }
}

/// One speed control point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpeedPoint {
    /// normalised position over the SOURCE range, 0..1
    pub t: f64,
    /// playback speed multiplier, 0.1 .. 10
    pub speed: f64,
}

impl SpeedPoint {
    pub const fn new(t: f64, speed: f64) -> Self {
        Self { t, speed }
    }
}

fn default_true() -> bool {
    true
}

/// A speed ramp as stored on a clip (`Clip.speed` in the JSON contract).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeedCurve {
    #[serde(default)]
    pub preset: SpeedPreset,
    #[serde(default)]
    pub points: Vec<SpeedPoint>,
    /// use RAFT optical-flow interpolation when speed < 1
    #[serde(default = "default_true")]
    pub optical_flow: bool,
}

impl Default for SpeedCurve {
    fn default() -> Self {
        SpeedCurve::preset(SpeedPreset::Normal)
    }
}

impl SpeedCurve {
    /// Build a curve from a preset (points are materialised).
    pub fn preset(preset: SpeedPreset) -> Self {
        Self { preset, points: preset.points(), optical_flow: true }
    }

    /// A constant-speed curve.
    pub fn constant(speed: f64) -> Self {
        let speed = speed.clamp(MIN_SPEED, MAX_SPEED);
        Self {
            preset: if (speed - 1.0).abs() < f64::EPSILON { SpeedPreset::Normal } else { SpeedPreset::Custom },
            points: vec![SpeedPoint::new(0.0, speed), SpeedPoint::new(1.0, speed)],
            optical_flow: true,
        }
    }

    /// A custom curve from raw points.
    pub fn custom(points: Vec<SpeedPoint>) -> Self {
        Self { preset: SpeedPreset::Custom, points, optical_flow: true }
    }

    /// Effective control points: the stored points if present, else the
    /// preset's points. Sorted by `t`, clamped into range, de-duplicated, and
    /// guaranteed to span `[0, 1]` (end points are extended flat).
    pub fn effective_points(&self) -> Vec<SpeedPoint> {
        let mut pts: Vec<SpeedPoint> = if self.points.is_empty() {
            self.preset.points()
        } else {
            self.points.clone()
        };
        pts.retain(|p| p.t.is_finite() && p.speed.is_finite());
        if pts.is_empty() {
            pts = SpeedPreset::Normal.points();
        }
        for p in pts.iter_mut() {
            p.t = p.t.clamp(0.0, 1.0);
            p.speed = p.speed.clamp(MIN_SPEED, MAX_SPEED);
        }
        pts.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
        pts.dedup_by(|b, a| (b.t - a.t).abs() < 1e-9);
        if pts[0].t > 0.0 {
            pts.insert(0, SpeedPoint::new(0.0, pts[0].speed));
        }
        if pts[pts.len() - 1].t < 1.0 {
            let s = pts[pts.len() - 1].speed;
            pts.push(SpeedPoint::new(1.0, s));
        }
        pts
    }

    /// True when every point has the same speed (export can use plain `setpts`).
    pub fn is_constant(&self) -> bool {
        let pts = self.effective_points();
        pts.iter().all(|p| (p.speed - pts[0].speed).abs() < 1e-9)
    }

    /// Speed multiplier at normalised source position `t ∈ [0,1]`.
    pub fn speed_at(&self, t: f64) -> f64 {
        MonotoneCubic::new(&self.effective_points()).eval(t)
    }

    /// Build a lookup table for a clip whose source range is `source_duration_ms` long.
    pub fn lut(&self, source_duration_ms: f64) -> SpeedLut {
        SpeedLut::new(self, source_duration_ms, DEFAULT_LUT_SAMPLES)
    }
}

/// Fritsch–Carlson monotone cubic interpolant over sorted `(t, speed)` points.
#[derive(Debug, Clone)]
pub struct MonotoneCubic {
    xs: Vec<f64>,
    ys: Vec<f64>,
    tangents: Vec<f64>,
}

impl MonotoneCubic {
    pub fn new(points: &[SpeedPoint]) -> Self {
        let n = points.len();
        let xs: Vec<f64> = points.iter().map(|p| p.t).collect();
        let ys: Vec<f64> = points.iter().map(|p| p.speed).collect();
        if n < 2 {
            return Self { xs, ys, tangents: vec![0.0; n] };
        }
        // Secant slopes.
        let mut deltas = Vec::with_capacity(n - 1);
        for k in 0..n - 1 {
            let h = xs[k + 1] - xs[k];
            deltas.push(if h > 0.0 { (ys[k + 1] - ys[k]) / h } else { 0.0 });
        }
        // Initial tangents: average of neighbouring secants, zero at sign changes.
        let mut m = vec![0.0; n];
        m[0] = deltas[0];
        m[n - 1] = deltas[n - 2];
        for k in 1..n - 1 {
            if deltas[k - 1] * deltas[k] <= 0.0 {
                m[k] = 0.0;
            } else {
                m[k] = 0.5 * (deltas[k - 1] + deltas[k]);
            }
        }
        // Fritsch–Carlson limiter keeps the interpolant monotone on each interval.
        for k in 0..n - 1 {
            if deltas[k].abs() < f64::EPSILON {
                m[k] = 0.0;
                m[k + 1] = 0.0;
                continue;
            }
            let alpha = m[k] / deltas[k];
            let beta = m[k + 1] / deltas[k];
            let s = alpha * alpha + beta * beta;
            if s > 9.0 {
                let tau = 3.0 / s.sqrt();
                m[k] = tau * alpha * deltas[k];
                m[k + 1] = tau * beta * deltas[k];
            }
        }
        Self { xs, ys, tangents: m }
    }

    /// Evaluate the interpolant at `x` (clamped to the point range).
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        if n == 0 {
            return 1.0;
        }
        if n == 1 || x <= self.xs[0] {
            return self.ys[0];
        }
        if x >= self.xs[n - 1] {
            return self.ys[n - 1];
        }
        let i = self.xs.partition_point(|&xi| xi <= x) - 1;
        let i = i.min(n - 2);
        let h = self.xs[i + 1] - self.xs[i];
        if h <= 0.0 {
            return self.ys[i + 1];
        }
        let s = (x - self.xs[i]) / h;
        let s2 = s * s;
        let s3 = s2 * s;
        let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
        let h10 = s3 - 2.0 * s2 + s;
        let h01 = -2.0 * s3 + 3.0 * s2;
        let h11 = s3 - s2;
        let y = h00 * self.ys[i] + h10 * h * self.tangents[i] + h01 * self.ys[i + 1] + h11 * h * self.tangents[i + 1];
        y.clamp(MIN_SPEED, MAX_SPEED)
    }
}

/// Pre-integrated speed curve for a specific clip.
///
/// `cum[i]` is the output time (ms) reached after consuming source time
/// `i / samples * source_duration_ms`. It is strictly increasing because speed
/// is always positive.
#[derive(Debug, Clone)]
pub struct SpeedLut {
    source_duration_ms: f64,
    cum: Vec<f64>,
    curve: MonotoneCubic,
}

impl SpeedLut {
    pub fn new(curve: &SpeedCurve, source_duration_ms: f64, samples: usize) -> Self {
        let samples = samples.max(2);
        let source_duration_ms = source_duration_ms.max(0.0);
        let mc = MonotoneCubic::new(&curve.effective_points());
        let dt = 1.0 / samples as f64;
        let mut cum = Vec::with_capacity(samples + 1);
        cum.push(0.0);
        let mut acc = 0.0;
        let mut prev_inv = 1.0 / mc.eval(0.0);
        for i in 1..=samples {
            let t = i as f64 * dt;
            // Simpson's rule per cell for better accuracy than the trapezoid.
            let mid_inv = 1.0 / mc.eval(t - 0.5 * dt);
            let inv = 1.0 / mc.eval(t);
            acc += dt * (prev_inv + 4.0 * mid_inv + inv) / 6.0;
            prev_inv = inv;
            cum.push(acc * source_duration_ms);
        }
        Self { source_duration_ms, cum, curve: mc }
    }

    pub fn source_duration_ms(&self) -> f64 {
        self.source_duration_ms
    }

    /// Duration of the clip after speed remapping.
    pub fn output_duration_ms(&self) -> f64 {
        *self.cum.last().unwrap_or(&0.0)
    }

    /// Constant speed that would yield the same output duration
    /// (`source / output`) — what the exporter uses for `setpts` / `atempo`.
    pub fn effective_speed(&self) -> f64 {
        let out = self.output_duration_ms();
        if out <= 0.0 || self.source_duration_ms <= 0.0 {
            1.0
        } else {
            (self.source_duration_ms / out).clamp(MIN_SPEED, MAX_SPEED)
        }
    }

    /// Speed at a normalised source position.
    pub fn speed_at_source_t(&self, t: f64) -> f64 {
        self.curve.eval(t.clamp(0.0, 1.0))
    }

    /// Speed at an absolute source offset (ms into the clip's source range).
    pub fn speed_at_source(&self, source_ms: f64) -> f64 {
        if self.source_duration_ms <= 0.0 {
            return self.curve.eval(0.0);
        }
        self.speed_at_source_t(source_ms / self.source_duration_ms)
    }

    /// Speed at an output offset (ms into the remapped clip).
    pub fn speed_at_output(&self, output_ms: f64) -> f64 {
        self.speed_at_source(self.output_to_source_time(output_ms))
    }

    /// Map a source offset (ms) to the output offset (ms).
    pub fn source_to_output_time(&self, source_ms: f64) -> f64 {
        if self.source_duration_ms <= 0.0 {
            return 0.0;
        }
        let n = self.cum.len() - 1;
        let x = (source_ms / self.source_duration_ms).clamp(0.0, 1.0) * n as f64;
        let i = (x.floor() as usize).min(n - 1);
        let f = x - i as f64;
        self.cum[i] + (self.cum[i + 1] - self.cum[i]) * f
    }

    /// Map an output offset (ms) back to the source offset (ms).
    pub fn output_to_source_time(&self, output_ms: f64) -> f64 {
        if self.source_duration_ms <= 0.0 {
            return 0.0;
        }
        let n = self.cum.len() - 1;
        let total = self.output_duration_ms();
        if output_ms <= 0.0 {
            return 0.0;
        }
        if output_ms >= total {
            return self.source_duration_ms;
        }
        // cum is strictly increasing → binary search for the cell.
        let idx = self.cum.partition_point(|&c| c <= output_ms);
        let i = idx.clamp(1, n) - 1;
        let span = self.cum[i + 1] - self.cum[i];
        let f = if span > 0.0 { (output_ms - self.cum[i]) / span } else { 0.0 };
        (i as f64 + f) / n as f64 * self.source_duration_ms
    }

    /// Uniform table of `samples + 1` entries mapping output time → source time,
    /// for the frontend preview scrubber.
    pub fn output_to_source_table(&self, samples: usize) -> Vec<f64> {
        let samples = samples.max(1);
        let total = self.output_duration_ms();
        (0..=samples)
            .map(|i| self.output_to_source_time(total * i as f64 / samples as f64))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn normal_curve_is_identity() {
        let lut = SpeedCurve::preset(SpeedPreset::Normal).lut(10_000.0);
        assert!(close(lut.output_duration_ms(), 10_000.0, 1e-6));
        for ms in [0.0, 1234.5, 5000.0, 9999.0, 10_000.0] {
            assert!(close(lut.source_to_output_time(ms), ms, 1e-6));
            assert!(close(lut.output_to_source_time(ms), ms, 1e-6));
        }
        assert!(close(lut.effective_speed(), 1.0, 1e-9));
    }

    #[test]
    fn constant_double_speed_halves_duration() {
        let lut = SpeedCurve::constant(2.0).lut(10_000.0);
        assert!(close(lut.output_duration_ms(), 5_000.0, 1e-6));
        assert!(close(lut.source_to_output_time(4_000.0), 2_000.0, 1e-6));
        assert!(close(lut.output_to_source_time(2_000.0), 4_000.0, 1e-6));
        assert!(close(lut.effective_speed(), 2.0, 1e-9));
        assert!(SpeedCurve::constant(2.0).is_constant());
    }

    #[test]
    fn hero_time_slows_the_middle() {
        let curve = SpeedCurve::preset(SpeedPreset::HeroTime);
        let lut = curve.lut(10_000.0);
        assert!(lut.output_duration_ms() > 10_000.0, "slow-mo must lengthen the clip");
        assert!(close(curve.speed_at(0.5), 0.3, 1e-9));
        assert!(close(curve.speed_at(0.0), 1.0, 1e-9));
        assert!(close(curve.speed_at(1.0), 1.0, 1e-9));
        // Monotone cubic must not undershoot the minimum control value.
        for i in 0..=200 {
            let s = curve.speed_at(i as f64 / 200.0);
            assert!((0.3 - 1e-9..=1.0 + 1e-9).contains(&s), "speed {s} out of range");
        }
        assert!(!curve.is_constant());
    }

    #[test]
    fn mapping_round_trips_and_is_monotone() {
        for preset in SpeedPreset::ALL {
            let lut = SpeedCurve::preset(preset).lut(8_000.0);
            let total = lut.output_duration_ms();
            let mut prev = -1.0;
            for i in 0..=100 {
                let out = total * i as f64 / 100.0;
                let src = lut.output_to_source_time(out);
                assert!(src >= prev - 1e-9, "{preset:?}: source time must be monotone");
                prev = src;
                let back = lut.source_to_output_time(src);
                assert!(close(back, out, 1e-3), "{preset:?}: out={out} src={src} back={back}");
            }
            assert!(close(lut.output_to_source_time(total), 8_000.0, 1e-9));
            assert!(close(lut.output_to_source_time(-5.0), 0.0, 1e-9));
        }
    }

    #[test]
    fn presets_are_well_formed() {
        for preset in SpeedPreset::ALL {
            let pts = preset.points();
            assert!(pts.len() >= 2);
            assert_eq!(pts[0].t, 0.0);
            assert_eq!(pts[pts.len() - 1].t, 1.0);
            for w in pts.windows(2) {
                assert!(w[0].t < w[1].t, "{preset:?} points must be strictly increasing");
            }
            for p in &pts {
                assert!(p.speed >= MIN_SPEED && p.speed <= MAX_SPEED);
            }
        }
        assert!(close(SpeedPreset::FlashIn.points()[0].speed, 4.0, 0.0));
        assert!(close(SpeedPreset::FlashOut.points().last().unwrap().speed, 4.0, 0.0));
    }

    #[test]
    fn effective_points_sanitise_input() {
        let curve = SpeedCurve::custom(vec![
            SpeedPoint::new(0.5, 50.0), // over max → clamped
            SpeedPoint::new(0.2, 0.01), // under min → clamped
        ]);
        let pts = curve.effective_points();
        assert_eq!(pts[0].t, 0.0);
        assert_eq!(pts.last().unwrap().t, 1.0);
        assert!(pts.iter().all(|p| p.speed >= MIN_SPEED && p.speed <= MAX_SPEED));
        // Empty custom points fall back to the preset.
        let hero = SpeedCurve { preset: SpeedPreset::HeroTime, points: vec![], optical_flow: true };
        assert_eq!(hero.effective_points().len(), 5);
    }

    #[test]
    fn output_table_has_expected_shape() {
        let lut = SpeedCurve::preset(SpeedPreset::Bullet).lut(4_000.0);
        let table = lut.output_to_source_table(50);
        assert_eq!(table.len(), 51);
        assert!(close(table[0], 0.0, 1e-9));
        assert!(close(table[50], 4_000.0, 1e-9));
        assert!(table.windows(2).all(|w| w[1] >= w[0]));
    }

    #[test]
    fn serde_round_trip_matches_contract() {
        let json = r#"{"preset":"hero_time","points":[{"t":0,"speed":1},{"t":0.5,"speed":0.3},{"t":1,"speed":1}],"opticalFlow":false}"#;
        let c: SpeedCurve = serde_json::from_str(json).unwrap();
        assert_eq!(c.preset, SpeedPreset::HeroTime);
        assert!(!c.optical_flow);
        assert_eq!(c.points.len(), 3);
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["preset"], "hero_time");
        assert_eq!(v["opticalFlow"], false);
        // Partial input.
        let d: SpeedCurve = serde_json::from_str(r#"{"preset":"jump_cut"}"#).unwrap();
        assert!(d.optical_flow);
        assert_eq!(d.effective_points().len(), 10);
        assert_eq!(SpeedPreset::parse("flash_in"), Some(SpeedPreset::FlashIn));
    }

    #[test]
    fn lut_build_and_lookups_are_fast() {
        let start = std::time::Instant::now();
        let lut = SpeedCurve::preset(SpeedPreset::Montage).lut(60_000.0);
        let mut acc = 0.0;
        for i in 0..10_000 {
            acc += lut.output_to_source_time(i as f64 * 7.0);
        }
        assert!(acc.is_finite());
        let budget_ms = if cfg!(debug_assertions) { 500 } else { 50 };
        assert!(start.elapsed().as_millis() < budget_ms);
    }
}
