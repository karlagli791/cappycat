//! Keyframe containers and evaluation.
//!
//! Semantics (identical to the frontend preview):
//! * `keyframes` empty → `static` value.
//! * before the first keyframe → first value; after the last → last value.
//! * between keyframes `a` and `b`, progress `u = (t - a.t) / (b.t - a.t)` is
//!   eased with **`a`'s** easing (the outgoing keyframe owns the segment, like
//!   CapCut / After Effects) and the values are lerped.

use crate::easing::Easing;
use serde::{Deserialize, Serialize};

/// A value that can be linearly interpolated.
pub trait Interpolate: Clone {
    fn lerp(a: &Self, b: &Self, u: f64) -> Self;
}

impl Interpolate for f64 {
    #[inline]
    fn lerp(a: &Self, b: &Self, u: f64) -> Self {
        a + (b - a) * u
    }
}

impl Interpolate for f32 {
    #[inline]
    fn lerp(a: &Self, b: &Self, u: f64) -> Self {
        (*a as f64 + (*b as f64 - *a as f64) * u) as f32
    }
}

macro_rules! impl_interpolate_array {
    ($n:expr) => {
        impl Interpolate for [f64; $n] {
            #[inline]
            fn lerp(a: &Self, b: &Self, u: f64) -> Self {
                let mut out = [0.0; $n];
                for i in 0..$n {
                    out[i] = a[i] + (b[i] - a[i]) * u;
                }
                out
            }
        }
    };
}
impl_interpolate_array!(2);
impl_interpolate_array!(3);
impl_interpolate_array!(4);

/// One keyframe. `time_ms` is relative to the clip's start on the timeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Keyframe<T> {
    pub time_ms: f64,
    pub value: T,
    #[serde(default)]
    pub easing: Easing,
    /// cubic-bezier control points (x1, y1, x2, y2) used when `easing == Bezier`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bezier: Option<[f64; 4]>,
}

impl<T> Keyframe<T> {
    pub fn new(time_ms: f64, value: T) -> Self {
        Self { time_ms, value, easing: Easing::Linear, bezier: None }
    }
    pub fn with_easing(mut self, easing: Easing) -> Self {
        self.easing = easing;
        self
    }
    pub fn with_bezier(mut self, p: [f64; 4]) -> Self {
        self.easing = Easing::Bezier;
        self.bezier = Some(p);
        self
    }
}

/// A property that is either static or driven by keyframes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Keyframed<T> {
    #[serde(rename = "static")]
    pub static_value: T,
    #[serde(default = "Vec::new")]
    pub keyframes: Vec<Keyframe<T>>,
}

impl<T: Default> Default for Keyframed<T> {
    fn default() -> Self {
        Self { static_value: T::default(), keyframes: Vec::new() }
    }
}

impl<T> Keyframed<T> {
    pub fn constant(value: T) -> Self {
        Self { static_value: value, keyframes: Vec::new() }
    }
    pub fn with_keyframes(value: T, keyframes: Vec<Keyframe<T>>) -> Self {
        Self { static_value: value, keyframes }
    }
    pub fn is_animated(&self) -> bool {
        !self.keyframes.is_empty()
    }
}

impl<T: Interpolate> Keyframed<T> {
    /// Evaluate the property at `time_ms` (relative to the clip start).
    pub fn evaluate(&self, time_ms: f64) -> T {
        evaluate(self, time_ms)
    }
}

/// Evaluate a keyframed property at `time_ms`.
///
/// Keyframes are expected to be sorted by `time_ms`; unsorted input is handled
/// by sorting a copy (slower path), so the hot path stays allocation-free.
pub fn evaluate<T: Interpolate>(k: &Keyframed<T>, time_ms: f64) -> T {
    let kfs = &k.keyframes;
    match kfs.len() {
        0 => return k.static_value.clone(),
        1 => return kfs[0].value.clone(),
        _ => {}
    }
    if is_sorted(kfs) {
        let refs: SortedView<'_, T> = SortedView::Slice(kfs);
        refs.evaluate(time_ms)
    } else {
        let mut sorted: Vec<&Keyframe<T>> = kfs.iter().collect();
        sorted.sort_by(|a, b| {
            a.time_ms
                .partial_cmp(&b.time_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        SortedView::Refs(&sorted).evaluate(time_ms)
    }
}

fn is_sorted<T>(kfs: &[Keyframe<T>]) -> bool {
    kfs.windows(2).all(|w| w[0].time_ms <= w[1].time_ms)
}

/// Uniform access over an owned slice or a sorted slice of references.
enum SortedView<'a, T> {
    Slice(&'a [Keyframe<T>]),
    Refs(&'a [&'a Keyframe<T>]),
}

impl<'a, T: Interpolate> SortedView<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        match self {
            SortedView::Slice(s) => s.len(),
            SortedView::Refs(r) => r.len(),
        }
    }
    #[inline]
    fn get(&self, i: usize) -> &'a Keyframe<T> {
        match self {
            SortedView::Slice(s) => &s[i],
            SortedView::Refs(r) => r[i],
        }
    }
    #[inline]
    fn partition_point(&self, time_ms: f64) -> usize {
        match self {
            SortedView::Slice(s) => s.partition_point(|kf| kf.time_ms <= time_ms),
            SortedView::Refs(r) => r.partition_point(|kf| kf.time_ms <= time_ms),
        }
    }

    fn evaluate(&self, time_ms: f64) -> T {
        let first = self.get(0);
        let last = self.get(self.len() - 1);
        if time_ms <= first.time_ms {
            return first.value.clone();
        }
        if time_ms >= last.time_ms {
            return last.value.clone();
        }
        // Binary search for the segment containing time_ms.
        let idx = self.partition_point(time_ms);
        segment(self.get(idx - 1), self.get(idx), time_ms)
    }
}

#[inline]
fn segment<T: Interpolate>(a: &Keyframe<T>, b: &Keyframe<T>, time_ms: f64) -> T {
    let span = b.time_ms - a.time_ms;
    if span <= 0.0 {
        return b.value.clone();
    }
    let u = ((time_ms - a.time_ms) / span).clamp(0.0, 1.0);
    let eased = a.easing.apply(u, a.bezier);
    T::lerp(&a.value, &b.value, eased)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn static_when_no_keyframes() {
        let k = Keyframed::constant(3.5);
        assert!(close(k.evaluate(0.0), 3.5));
        assert!(close(k.evaluate(1e9), 3.5));
    }

    #[test]
    fn single_keyframe_is_constant() {
        let k = Keyframed::with_keyframes(0.0, vec![Keyframe::new(500.0, 2.0)]);
        assert!(close(k.evaluate(0.0), 2.0));
        assert!(close(k.evaluate(900.0), 2.0));
    }

    #[test]
    fn linear_interpolation_and_clamping() {
        let k = Keyframed::with_keyframes(
            0.0,
            vec![Keyframe::new(0.0, 0.0), Keyframe::new(1000.0, 10.0)],
        );
        assert!(close(k.evaluate(-50.0), 0.0));
        assert!(close(k.evaluate(250.0), 2.5));
        assert!(close(k.evaluate(500.0), 5.0));
        assert!(close(k.evaluate(1000.0), 10.0));
        assert!(close(k.evaluate(5000.0), 10.0));
    }

    #[test]
    fn vector_interpolation() {
        let k = Keyframed::with_keyframes(
            [0.0, 0.0],
            vec![Keyframe::new(0.0, [0.0, 10.0]), Keyframe::new(100.0, [10.0, 0.0])],
        );
        let v = k.evaluate(50.0);
        assert!(close(v[0], 5.0) && close(v[1], 5.0));

        let k3 = Keyframed::with_keyframes(
            [0.0; 3],
            vec![Keyframe::new(0.0, [0.0, 0.0, 0.0]), Keyframe::new(10.0, [1.0, 2.0, 3.0])],
        );
        let v3 = k3.evaluate(5.0);
        assert!(close(v3[0], 0.5) && close(v3[1], 1.0) && close(v3[2], 1.5));

        let k4 = Keyframed::with_keyframes(
            [0.0; 4],
            vec![
                Keyframe::new(0.0, [0.0, 0.0, 1.0, 1.0]),
                Keyframe::new(10.0, [0.5, 0.5, 0.5, 0.5]),
            ],
        );
        let v4 = k4.evaluate(10.0);
        assert_eq!(v4, [0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn outgoing_keyframe_easing_owns_the_segment() {
        let k = Keyframed::with_keyframes(
            0.0,
            vec![
                Keyframe::new(0.0, 0.0).with_easing(Easing::EaseIn),
                Keyframe::new(1000.0, 1.0).with_easing(Easing::EaseOut),
            ],
        );
        // ease-in: slow start → value below linear at 30 %.
        assert!(k.evaluate(300.0) < 0.3);
        let kb = Keyframed::with_keyframes(
            0.0,
            vec![
                Keyframe::new(0.0, 0.0).with_bezier([0.0, 1.0, 0.0, 1.0]),
                Keyframe::new(1000.0, 1.0),
            ],
        );
        // an "ease-out"-like custom bezier is above the diagonal
        assert!(kb.evaluate(300.0) > 0.3);
    }

    #[test]
    fn unsorted_keyframes_are_handled() {
        let k = Keyframed::with_keyframes(
            0.0,
            vec![Keyframe::new(1000.0, 10.0), Keyframe::new(0.0, 0.0)],
        );
        assert!(close(k.evaluate(500.0), 5.0));
    }

    #[test]
    fn multi_segment_binary_search() {
        let kfs: Vec<Keyframe<f64>> =
            (0..10).map(|i| Keyframe::new(i as f64 * 100.0, i as f64)).collect();
        let k = Keyframed::with_keyframes(0.0, kfs);
        assert!(close(k.evaluate(450.0), 4.5));
        assert!(close(k.evaluate(899.99), 8.9999));
        assert!(close(k.evaluate(900.0), 9.0));
    }

    #[test]
    fn serde_round_trip_matches_contract() {
        let json = r#"{"static":1,"keyframes":[{"timeMs":0,"value":1,"easing":"linear"},{"timeMs":500,"value":2,"easing":"bezier","bezier":[0.2,0,0.8,1]}]}"#;
        let k: Keyframed<f64> = serde_json::from_str(json).unwrap();
        assert_eq!(k.keyframes.len(), 2);
        assert_eq!(k.keyframes[1].bezier, Some([0.2, 0.0, 0.8, 1.0]));
        let out = serde_json::to_value(&k).unwrap();
        assert_eq!(out["static"], 1.0);
        assert_eq!(out["keyframes"][1]["easing"], "bezier");
        // partial JSON: no easing / no keyframes.
        let k2: Keyframed<[f64; 2]> = serde_json::from_str(r#"{"static":[1,2]}"#).unwrap();
        assert_eq!(k2.static_value, [1.0, 2.0]);
        assert!(k2.keyframes.is_empty());
    }

    /// Lenient timing test: 10k evaluations of a 20-keyframe bezier-eased
    /// property must finish well under the 50 ms budget (spec: sub-1ms per
    /// evaluation). Debug builds are ~10x slower than release, so we allow a
    /// generous ceiling there and assert the real budget only in release.
    #[test]
    fn ten_thousand_evaluations_are_fast() {
        let kfs: Vec<Keyframe<[f64; 2]>> = (0..20)
            .map(|i| {
                Keyframe::new(i as f64 * 50.0, [i as f64, (i * 2) as f64])
                    .with_bezier([0.25, 0.1, 0.25, 1.0])
            })
            .collect();
        let k = Keyframed::with_keyframes([0.0, 0.0], kfs);
        let start = Instant::now();
        let mut acc = 0.0;
        for i in 0..10_000 {
            let t = (i as f64 * 0.0973) % 1000.0;
            let v = k.evaluate(t);
            acc += v[0] + v[1];
        }
        let elapsed = start.elapsed();
        assert!(acc.is_finite());
        let budget_ms = if cfg!(debug_assertions) { 500 } else { 50 };
        assert!(
            elapsed.as_millis() < budget_ms,
            "10k evaluations took {elapsed:?}, budget {budget_ms} ms"
        );
    }
}
