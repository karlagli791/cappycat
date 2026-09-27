//! Easing functions.
//!
//! [`CubicBezier`] implements the same semantics as CSS `cubic-bezier(x1, y1, x2, y2)`:
//! the curve runs from (0,0) to (1,1); for a given progress `x` we solve the
//! parametric `t` (Newton–Raphson, falling back to bisection when the slope is
//! too flat) and return the curve's `y(t)`.
//!
//! [`Easing`] is the JSON-facing enum (`"linear" | "easeIn" | ... | "bezier"`).

use serde::{Deserialize, Serialize};

/// A cubic Bézier easing curve anchored at (0,0) and (1,1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CubicBezier {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

const NEWTON_ITERATIONS: usize = 8;
const NEWTON_MIN_SLOPE: f64 = 1e-3;
const SOLVER_EPSILON: f64 = 1e-7;
const BISECTION_MAX_ITERATIONS: usize = 64;

impl CubicBezier {
    pub const LINEAR: CubicBezier = CubicBezier::new(0.0, 0.0, 1.0, 1.0);
    /// CSS `ease-in`.
    pub const EASE_IN: CubicBezier = CubicBezier::new(0.42, 0.0, 1.0, 1.0);
    /// CSS `ease-out`.
    pub const EASE_OUT: CubicBezier = CubicBezier::new(0.0, 0.0, 0.58, 1.0);
    /// CSS `ease-in-out`.
    pub const EASE_IN_OUT: CubicBezier = CubicBezier::new(0.42, 0.0, 0.58, 1.0);

    pub const fn new(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// Build from a `[x1, y1, x2, y2]` array. X control points are clamped to
    /// `[0, 1]` (as CSS requires) so the curve stays a function of x.
    pub fn from_array(p: [f64; 4]) -> Self {
        Self::new(p[0].clamp(0.0, 1.0), p[1], p[2].clamp(0.0, 1.0), p[3])
    }

    #[inline]
    fn coef_a(a1: f64, a2: f64) -> f64 {
        1.0 - 3.0 * a2 + 3.0 * a1
    }
    #[inline]
    fn coef_b(a1: f64, a2: f64) -> f64 {
        3.0 * a2 - 6.0 * a1
    }
    #[inline]
    fn coef_c(a1: f64) -> f64 {
        3.0 * a1
    }

    /// Evaluate one axis of the curve at parameter `t`.
    #[inline]
    fn sample(t: f64, a1: f64, a2: f64) -> f64 {
        ((Self::coef_a(a1, a2) * t + Self::coef_b(a1, a2)) * t + Self::coef_c(a1)) * t
    }

    /// Derivative of one axis with respect to `t`.
    #[inline]
    fn slope(t: f64, a1: f64, a2: f64) -> f64 {
        3.0 * Self::coef_a(a1, a2) * t * t + 2.0 * Self::coef_b(a1, a2) * t + Self::coef_c(a1)
    }

    /// Solve the parametric `t` for a given `x` on the curve.
    pub fn solve_t_for_x(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        // Newton–Raphson from an initial guess of t = x.
        let mut t = x;
        for _ in 0..NEWTON_ITERATIONS {
            let err = Self::sample(t, self.x1, self.x2) - x;
            if err.abs() < SOLVER_EPSILON {
                return t;
            }
            let slope = Self::slope(t, self.x1, self.x2);
            if slope.abs() < NEWTON_MIN_SLOPE {
                break;
            }
            t -= err / slope;
            if !(0.0..=1.0).contains(&t) {
                break;
            }
        }
        // Verify the Newton result; fall back to bisection when it did not converge.
        if (0.0..=1.0).contains(&t)
            && (Self::sample(t, self.x1, self.x2) - x).abs() < SOLVER_EPSILON * 10.0
        {
            return t;
        }
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        let mut mid = x;
        for _ in 0..BISECTION_MAX_ITERATIONS {
            mid = 0.5 * (lo + hi);
            let err = Self::sample(mid, self.x1, self.x2) - x;
            if err.abs() < SOLVER_EPSILON {
                break;
            }
            if err > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        mid
    }

    /// Map progress `x ∈ [0,1]` to eased progress `y`.
    pub fn ease(&self, x: f64) -> f64 {
        if self.x1 == self.y1 && self.x2 == self.y2 {
            // Identity curve (any control points on the diagonal) — skip the solver.
            return x.clamp(0.0, 1.0);
        }
        let t = self.solve_t_for_x(x.clamp(0.0, 1.0));
        Self::sample(t, self.y1, self.y2)
    }
}

impl Default for CubicBezier {
    fn default() -> Self {
        Self::LINEAR
    }
}

/// JSON-facing easing selector, matching the `Easing` union in `project.ts`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Easing {
    #[default]
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Bounce,
    Elastic,
    /// Custom curve; control points are supplied alongside (see `Keyframe::bezier`).
    Bezier,
}

impl Easing {
    /// Apply the easing to progress `x ∈ [0,1]`.
    ///
    /// `bezier` is only consulted when `self == Easing::Bezier`; if it is
    /// `None` in that case the curve degrades to linear.
    pub fn apply(self, x: f64, bezier: Option<[f64; 4]>) -> f64 {
        let x = x.clamp(0.0, 1.0);
        match self {
            Easing::Linear => x,
            Easing::EaseIn => CubicBezier::EASE_IN.ease(x),
            Easing::EaseOut => CubicBezier::EASE_OUT.ease(x),
            Easing::EaseInOut => CubicBezier::EASE_IN_OUT.ease(x),
            Easing::Bounce => bounce_out(x),
            Easing::Elastic => elastic_out(x),
            Easing::Bezier => match bezier {
                Some(p) => CubicBezier::from_array(p).ease(x),
                None => x,
            },
        }
    }

    /// Parse the JSON string form (`"easeInOut"` etc.).
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "linear" => Some(Easing::Linear),
            "easeIn" => Some(Easing::EaseIn),
            "easeOut" => Some(Easing::EaseOut),
            "easeInOut" => Some(Easing::EaseInOut),
            "bounce" => Some(Easing::Bounce),
            "elastic" => Some(Easing::Elastic),
            "bezier" => Some(Easing::Bezier),
            _ => None,
        }
    }
}

/// Robert Penner's `easeOutBounce`: settles into the target with decaying bounces.
pub fn bounce_out(x: f64) -> f64 {
    const N1: f64 = 7.5625;
    const D1: f64 = 2.75;
    let x = x.clamp(0.0, 1.0);
    if x < 1.0 / D1 {
        N1 * x * x
    } else if x < 2.0 / D1 {
        let x = x - 1.5 / D1;
        N1 * x * x + 0.75
    } else if x < 2.5 / D1 {
        let x = x - 2.25 / D1;
        N1 * x * x + 0.9375
    } else {
        let x = x - 2.625 / D1;
        N1 * x * x + 0.984375
    }
}

/// Robert Penner's `easeOutElastic`: overshoots and springs back to the target.
pub fn elastic_out(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    if x == 0.0 {
        0.0
    } else if x == 1.0 {
        1.0
    } else {
        let c4 = (2.0 * std::f64::consts::PI) / 3.0;
        (2.0f64).powf(-10.0 * x) * ((x * 10.0 - 0.75) * c4).sin() + 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn linear_is_identity() {
        for i in 0..=20 {
            let x = i as f64 / 20.0;
            assert!(close(CubicBezier::LINEAR.ease(x), x, 1e-9));
            assert!(close(Easing::Linear.apply(x, None), x, 1e-12));
        }
    }

    #[test]
    fn endpoints_are_fixed_for_every_easing() {
        let all = [
            Easing::Linear,
            Easing::EaseIn,
            Easing::EaseOut,
            Easing::EaseInOut,
            Easing::Bounce,
            Easing::Elastic,
            Easing::Bezier,
        ];
        for e in all {
            assert!(close(e.apply(0.0, Some([0.3, -0.5, 0.7, 1.5])), 0.0, 1e-9), "{e:?} at 0");
            assert!(close(e.apply(1.0, Some([0.3, -0.5, 0.7, 1.5])), 1.0, 1e-9), "{e:?} at 1");
        }
    }

    #[test]
    fn ease_in_out_matches_css_reference_values() {
        // Reference values from the CSS cubic-bezier(0.42, 0, 0.58, 1) curve.
        let c = CubicBezier::EASE_IN_OUT;
        assert!(close(c.ease(0.25), 0.129, 2e-3));
        assert!(close(c.ease(0.5), 0.5, 1e-6));
        assert!(close(c.ease(0.75), 0.871, 2e-3));
        // ease-in is below the diagonal, ease-out above it.
        assert!(CubicBezier::EASE_IN.ease(0.3) < 0.3);
        assert!(CubicBezier::EASE_OUT.ease(0.3) > 0.3);
    }

    #[test]
    fn bezier_solver_is_monotone_and_invertible() {
        let c = CubicBezier::new(0.1, 0.9, 0.9, 0.1);
        let mut prev = -1.0;
        for i in 0..=100 {
            let x = i as f64 / 100.0;
            let t = c.solve_t_for_x(x);
            assert!(t >= prev - 1e-9, "t must be monotone in x");
            prev = t;
            let back = CubicBezier::sample(t, c.x1, c.x2);
            assert!(close(back, x, 1e-5), "x={x} t={t} back={back}");
        }
    }

    #[test]
    fn flat_start_curve_uses_bisection_fallback() {
        // Zero slope at t = 0 defeats Newton; bisection must still converge.
        let c = CubicBezier::new(0.0, 0.0, 0.0, 1.0);
        let y = c.ease(0.01);
        assert!(y.is_finite() && (0.0..=1.0).contains(&y));
        let back = CubicBezier::sample(c.solve_t_for_x(0.01), c.x1, c.x2);
        assert!(close(back, 0.01, 1e-5));
    }

    #[test]
    fn bounce_and_elastic_are_bounded() {
        for i in 0..=200 {
            let x = i as f64 / 200.0;
            let b = bounce_out(x);
            assert!((0.0..=1.0 + 1e-9).contains(&b));
            let e = elastic_out(x);
            assert!(e.is_finite() && e > -0.5 && e < 1.6);
        }
    }

    #[test]
    fn easing_serde_round_trip() {
        let json = serde_json::to_string(&Easing::EaseInOut).unwrap();
        assert_eq!(json, "\"easeInOut\"");
        let e: Easing = serde_json::from_str("\"bounce\"").unwrap();
        assert_eq!(e, Easing::Bounce);
        assert_eq!(Easing::parse("elastic"), Some(Easing::Elastic));
        assert_eq!(Easing::parse("nope"), None);
    }
}
