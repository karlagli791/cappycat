//! CPU renderer used by the exporter. Every per-pixel operation mirrors the
//! WebGL preview (`src/engine/color/shaders.ts`, `renderer.ts`) and the time
//! mapping mirrors `src/engine/playback.ts`, so the export looks like the
//! preview.
//!
//! * [`lut`]        — `.cube` parser + trilinear sampling
//! * [`color`]      — the grading pipeline (`GradeParams::apply`)
//! * [`bake`]       — per-clip baked grade lattices (exact cells where the grade has kinks)
//! * [`sample`]     — frames, bilinear sampling, blur / sharpen layers, uv matrix
//! * [`mask`]       — rectangle / circle / split / filmstrip masks
//! * [`blend`]      — W3C blend modes over an opaque canvas
//! * [`timemap`]    — timeline → source time (speed LUT, freeze, reverse, reframe)
//! * [`compositor`] — draws one layer onto the canvas (rayon, row-parallel)

pub mod bake;
pub mod blend;
pub mod color;
pub mod compositor;
pub mod fx;
pub mod lut;
pub mod mask;
pub mod sample;
pub mod timemap;
pub mod transitions;
