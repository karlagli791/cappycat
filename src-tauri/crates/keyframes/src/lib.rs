//! # Cappycat Keyframe & Motion Engine
//!
//! Pure-Rust, dependency-light library that powers every animated value in
//! Cappycat: transform keyframes (position / scale / rotation / opacity / blur),
//! mask rectangles, and speed-ramp time remapping.
//!
//! * [`easing`] — CSS-style cubic-bezier solver (Newton–Raphson with a bisection
//!   fallback on the x axis) plus the preset curves `linear`, `easeIn`,
//!   `easeOut`, `easeInOut`, `bounce`, `elastic`.
//! * [`interpolate`] — [`Keyframed<T>`] / [`Keyframe<T>`] containers mirroring
//!   `src/types/project.ts`, the [`Interpolate`] trait for `f64`, `[f64; 2]`,
//!   `[f64; 3]`, `[f64; 4]`, and [`evaluate`].
//! * [`speed`] — [`SpeedCurve`] with monotone-cubic (Fritsch–Carlson)
//!   interpolation, the CapCut presets, and [`SpeedLut`] for
//!   source⇄output time mapping via numeric integration of `1/speed`.
//!
//! All types (de)serialise with serde using the camelCase JSON contract from
//! `docs/CONTRACTS.md`, so the Tauri layer can pass them straight through.

pub mod easing;
pub mod interpolate;
pub mod speed;

pub use easing::{CubicBezier, Easing};
pub use interpolate::{evaluate, Interpolate, Keyframe, Keyframed};
pub use speed::{SpeedCurve, SpeedLut, SpeedPoint, SpeedPreset};
