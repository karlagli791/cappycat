//! Timeline → source time mapping, port of `resolveClip` / `reframeCropAt`
//! in `src/engine/playback.ts`.

use crate::model::{BBox, Clip, ClipMask, Rect, ReframeTrack};
use keyframes::SpeedLut;

/// Everything the renderer needs about a clip at one timeline instant.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    /// time within the clip on the timeline (ms), clamped to the clip
    pub local_ms: f64,
    /// absolute source time in the asset (ms)
    pub source_ms: f64,
    /// instantaneous playback rate (0 while frozen)
    pub rate: f64,
    pub frozen: bool,
    pub crop: Option<BBox>,
    pub position: [f64; 2],
    pub scale: f64,
    pub rotation: f64,
    pub opacity: f64,
    pub blur: f64,
    pub mask_rect: Option<Rect>,
}

/// Pre-built per-clip time map (speed LUT + durations).
#[derive(Debug, Clone)]
pub struct ClipTimeMap {
    lut: Option<SpeedLut>,
    src_range: f64,
    in_ms: f64,
    total_ms: f64,
    freeze: Option<(f64, f64)>,
    reversed: bool,
}

impl ClipTimeMap {
    pub fn new(clip: &Clip) -> Self {
        let src_range = clip.out_ms - clip.in_ms;
        // `isConstantSpeed` in speed.ts: the LUT is skipped only for 1× everywhere
        // — including a curve with no points (the preview's `every` on an empty
        // list is true, and `speedAt` returns 1), whatever its preset says.
        let constant_1x = clip.speed.points.iter().all(|p| (p.speed - 1.0).abs() < 1e-9);
        let lut = if constant_1x { None } else { Some(clip.speed.lut(src_range.max(0.0))) };
        let hold = clip.freeze_frame.map(|f| f.hold_ms.max(0.0)).unwrap_or(0.0);
        let base = match &lut {
            Some(l) => l.output_duration_ms(),
            None => src_range.max(0.0),
        };
        Self {
            lut,
            src_range,
            in_ms: clip.in_ms,
            total_ms: base + hold,
            freeze: clip.freeze_frame.map(|f| (f.at_ms, f.hold_ms.max(0.0))),
            reversed: clip.reversed,
        }
    }

    /// `clipDurationMs(clip)` (speed-remapped length + freeze hold)
    pub fn total_ms(&self) -> f64 {
        self.total_ms
    }

    /// Does the clip have a freeze-frame hold?
    pub fn has_freeze(&self) -> bool {
        self.freeze.map(|(_, h)| h > 0.0).unwrap_or(false)
    }

    /// `(local_ms clamped, source_ms, rate, frozen)` for a clip-local time.
    pub fn source_at(&self, local_ms: f64) -> (f64, f64, f64, bool) {
        let local = local_ms.clamp(0.0, self.total_ms.max(0.0));
        let mut playable = local;
        let mut frozen = false;
        if let Some((at, hold)) = self.freeze {
            if local >= at && local < at + hold {
                playable = at;
                frozen = true;
            } else if local >= at + hold {
                playable = local - hold;
            }
        }
        let (mut src_offset, rate) = match &self.lut {
            Some(lut) => {
                let s = lut.output_to_source_time(playable);
                let t = if self.src_range > 0.0 { s / self.src_range } else { 0.0 };
                (s, lut.speed_at_source_t(t))
            }
            None => (playable, 1.0),
        };
        if self.reversed {
            src_offset = self.src_range - src_offset;
        }
        let source_ms = self.in_ms + src_offset.clamp(0.0, self.src_range.max(0.0));
        (local, source_ms, if frozen { 0.0 } else { rate }, frozen)
    }

    /// `source_at` extended past the clip's ends for transition handles: before 0 / after
    /// `total_ms` the source time continues linearly at the boundary speed (backwards for reversed
    /// clips; a clip that ends frozen keeps holding), clamped to `[0, source_len_ms]` — past the
    /// file's first / last frame the picture holds. Inside the clip it equals `source_at`.
    pub fn source_at_extended(&self, local_ms: f64, source_len_ms: f64) -> (f64, f64, f64, bool) {
        let total = self.total_ms.max(0.0);
        if (0.0..=total).contains(&local_ms) {
            return self.source_at(local_ms);
        }
        let edge = if local_ms < 0.0 { 0.0 } else { total };
        // the boundary speed: sampled just inside the clip (a freeze at the very end holds)
        let inside = if local_ms < 0.0 { edge.min(total) } else { (edge - 1e-6).max(0.0) };
        let (_, _, rate, frozen) = self.source_at(inside);
        let (_, src_at_edge, _, _) = self.source_at(edge);
        let dir = if self.reversed { -1.0 } else { 1.0 };
        let rate = if frozen { 0.0 } else { rate };
        let hi = if source_len_ms > 0.0 { source_len_ms } else { self.in_ms + self.src_range.max(0.0) };
        let src = (src_at_edge + dir * (local_ms - edge) * rate).clamp(0.0, hi.max(0.0));
        (edge, src, rate, frozen || rate == 0.0)
    }

    /// [`ClipTimeMap::resolve`] with the time map extended past the clip's ends
    /// ([`ClipTimeMap::source_at_extended`]); transforms / masks hold their boundary values.
    pub fn resolve_extended(&self, clip: &Clip, local_ms: f64, source_len_ms: f64) -> Resolved {
        let mut r = self.resolve(clip, local_ms);
        if !(0.0..=self.total_ms.max(0.0)).contains(&local_ms) {
            let (_, source_ms, rate, frozen) = self.source_at_extended(local_ms, source_len_ms);
            r.source_ms = source_ms;
            r.rate = rate;
            r.frozen = frozen;
            r.crop = clip.reframe.as_ref().and_then(|t| reframe_crop_at(t, source_ms));
        }
        r
    }

    /// Full resolution including transform keyframes, reframe crop and mask rect.
    pub fn resolve(&self, clip: &Clip, local_ms: f64) -> Resolved {
        let (local, source_ms, rate, frozen) = self.source_at(local_ms);
        let t = &clip.transform;
        Resolved {
            local_ms: local,
            source_ms,
            rate,
            frozen,
            crop: clip.reframe.as_ref().and_then(|r| reframe_crop_at(r, source_ms)),
            position: t.position.evaluate(local),
            scale: t.scale.evaluate(local),
            rotation: t.rotation.evaluate(local),
            opacity: t.opacity.evaluate(local),
            blur: t.blur.evaluate(local),
            mask_rect: clip.mask.as_ref().map(|m: &ClipMask| m.rect.evaluate(local)),
        }
    }
}

/// Linear interpolation of the reframe crop at an absolute source time.
pub fn reframe_crop_at(track: &ReframeTrack, source_ms: f64) -> Option<BBox> {
    let keys = &track.keyframes;
    let first = keys.first()?;
    let last = keys.last()?;
    if source_ms <= first.time_ms {
        return Some(first.crop);
    }
    if source_ms >= last.time_ms {
        return Some(last.crop);
    }
    let (mut lo, mut hi) = (0usize, keys.len() - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if keys[mid].time_ms <= source_ms {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (a, b) = (&keys[lo], &keys[hi]);
    let span = b.time_ms - a.time_ms;
    let k = if span <= 0.0 { 0.0 } else { (source_ms - a.time_ms) / span };
    Some(std::array::from_fn(|i| a.crop[i] + (b.crop[i] - a.crop[i]) * k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::*;

    fn clip(in_ms: f64, out_ms: f64) -> Clip {
        Clip { id: "c".into(), in_ms, out_ms, ..Default::default() }
    }

    #[test]
    fn normal_speed_is_identity_offset() {
        let c = clip(1000.0, 3000.0);
        let m = ClipTimeMap::new(&c);
        assert_eq!(m.total_ms(), 2000.0);
        assert_eq!(m.source_at(500.0), (500.0, 1500.0, 1.0, false));
        // clamped past the end
        assert_eq!(m.source_at(5000.0).1, 3000.0);
    }

    #[test]
    fn freeze_and_reverse() {
        let mut c = clip(0.0, 2000.0);
        c.freeze_frame = Some(FreezeFrame { at_ms: 500.0, hold_ms: 1000.0 });
        let m = ClipTimeMap::new(&c);
        assert_eq!(m.total_ms(), 3000.0);
        assert_eq!(m.source_at(499.0).1, 499.0);
        assert_eq!(m.source_at(700.0), (700.0, 500.0, 0.0, true));
        assert_eq!(m.source_at(1600.0).1, 600.0);
        c.reversed = true;
        let m = ClipTimeMap::new(&c);
        assert_eq!(m.source_at(0.0).1, 2000.0);
        assert_eq!(m.source_at(700.0).1, 1500.0);
        assert_eq!(m.source_at(3000.0).1, 0.0);
    }

    #[test]
    fn speed_curves_use_the_keyframes_lut() {
        let mut c = clip(0.0, 4000.0);
        c.speed = SpeedCurve::constant(2.0);
        let m = ClipTimeMap::new(&c);
        assert!((m.total_ms() - 2000.0).abs() < 1e-6);
        let (_, s, rate, _) = m.source_at(1000.0);
        assert!((s - 2000.0).abs() < 1e-6 && (rate - 2.0).abs() < 1e-9);
        c.speed = SpeedCurve::preset(SpeedPreset::HeroTime);
        let m = ClipTimeMap::new(&c);
        assert!((m.total_ms() - c.output_duration_ms()).abs() < 1e-9);
        let lut = c.speed.lut(4000.0);
        assert!((m.total_ms() - lut.output_duration_ms()).abs() < 1e-9);
        let (_, s, _, _) = m.source_at(1234.0);
        assert!((s - lut.output_to_source_time(1234.0)).abs() < 1e-9);
        // no points → 1× like the preview, regardless of the preset name
        c.speed.points.clear();
        let m = ClipTimeMap::new(&c);
        assert_eq!(m.total_ms(), 4000.0);
        assert_eq!(m.source_at(1234.0).1, 1234.0);
    }

    #[test]
    fn extended_map_uses_handles_at_the_boundary_speed() {
        let mut c = clip(1000.0, 3000.0);
        let m = ClipTimeMap::new(&c);
        // inside: unchanged
        assert_eq!(m.source_at_extended(500.0, 10_000.0), m.source_at(500.0));
        // 200 ms past the end: 200 ms of handle after out
        assert_eq!(m.source_at_extended(2200.0, 10_000.0).1, 3200.0);
        // 300 ms before the start: 300 ms before in
        assert_eq!(m.source_at_extended(-300.0, 10_000.0).1, 700.0);
        // no handle in the file: holds the first / last frame
        assert_eq!(m.source_at_extended(-2000.0, 10_000.0).1, 0.0);
        assert_eq!(m.source_at_extended(9000.0, 3100.0).1, 3100.0);
        // speed maps extend linearly at the boundary speed
        c.speed = SpeedCurve::custom(vec![SpeedPoint::new(0.0, 0.5), SpeedPoint::new(1.0, 2.0)]);
        let m = ClipTimeMap::new(&c);
        let (_, s, rate, _) = m.source_at_extended(m.total_ms() + 100.0, 10_000.0);
        assert!((rate - 2.0).abs() < 1e-6 && (s - 3200.0).abs() < 1e-3, "{s} {rate}");
        let (_, s, rate, _) = m.source_at_extended(-100.0, 10_000.0);
        assert!((rate - 0.5).abs() < 1e-6 && (s - 950.0).abs() < 1e-3, "{s} {rate}");
        // reversed clips continue backwards past their end (below in)
        c.reversed = true;
        c.speed = SpeedCurve::constant(1.0);
        let m = ClipTimeMap::new(&c);
        assert_eq!(m.source_at_extended(2100.0, 10_000.0).1, 900.0);
        assert_eq!(m.source_at_extended(-100.0, 10_000.0).1, 3100.0);
        // resolve_extended keeps transforms at their boundary values
        let r = m.resolve_extended(&c, 2100.0, 10_000.0);
        assert_eq!((r.local_ms, r.source_ms), (2000.0, 900.0));
    }

    #[test]
    fn transforms_and_reframe_interpolate() {
        let mut c = clip(0.0, 2000.0);
        c.transform.scale = Keyframed::with_keyframes(1.0, vec![Keyframe::new(0.0, 1.0), Keyframe::new(1000.0, 2.0)]);
        c.reframe = Some(ReframeTrack {
            source_width: 100,
            source_height: 100,
            keyframes: vec![
                ReframeKeyframe { time_ms: 0.0, crop: [0.0, 0.0, 50.0, 100.0], ..Default::default() },
                ReframeKeyframe { time_ms: 1000.0, crop: [50.0, 0.0, 100.0, 100.0], ..Default::default() },
            ],
            reason: None,
        });
        let m = ClipTimeMap::new(&c);
        let r = m.resolve(&c, 500.0);
        assert!((r.scale - 1.5).abs() < 1e-9);
        assert_eq!(r.crop, Some([25.0, 0.0, 75.0, 100.0]));
        assert_eq!(m.resolve(&c, 1500.0).crop, Some([50.0, 0.0, 100.0, 100.0]));
        assert!(r.mask_rect.is_none());
    }
}
