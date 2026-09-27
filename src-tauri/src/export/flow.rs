//! Optical-flow frame interpolation: pre-interpolate the part of a clip's source range an export
//! needs with the Python pipeline so the compositor can pick real in-between frames instead of
//! repeating or cross-fading source frames. Used for slow motion and for frame-rate conversion
//! (`Project.frameInterpolation = 'opticalFlow'`, the default) whenever the output frame rate is
//! higher than the clip's effective rate (`source avg fps × speed`).
//!
//! `python -m cappycat_pipeline interpolate <src> --in-ms A --out-ms B --target-fps F --out <mp4>`
//! writes a video at exactly F fps whose frame `k` sits at source time `A + k/F`. Pipelines that
//! predate `--target-fps` are driven with `--factor N` instead (N = ⌈F / source fps⌉, 2..8).
//!
//! Results are cached under `<cache>\flow\<sha1>.mp4` keyed by
//! `sha1(path|mtime|inMs|outMs|fps:F)` (or `…|N` for a factor run).
//!
//! The caller holds the GPU queue (`JobManager::gpu`) while this runs; the python child watches
//! its stdin (`CAPPYCAT_WATCH_STDIN=1`) and is stopped with its process tree when the export is
//! cancelled.

use crate::jobs::{find_python, python_command, TaskCtl};
use crate::pipeline::find_pipeline_dir;
use serde_json::Value;
use sha1::{Digest, Sha1};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::Stdio;

/// Upper bound of an interpolation target (fps) and of its ratio to the source rate.
pub const MAX_TARGET_FPS: f64 = 480.0;
pub const MAX_TARGET_RATIO: f64 = 8.0;
/// Interpolate only when the output needs at least this many times the frames the source has.
pub const NEED_RATIO: f64 = 1.1;

/// What the pipeline is asked to produce.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlowSpec {
    /// `--target-fps F`
    TargetFps(f64),
    /// `--factor N` (older pipelines)
    Factor(u32),
}

/// Does a clip need more frames than its source has? `project_fps > NEED_RATIO × source fps ×
/// slowest speed` (speeds of the frames the export renders).
pub fn needs_interpolation(project_fps: f64, source_fps: f64, min_rate: f64) -> bool {
    source_fps > 0.0 && min_rate > 0.0 && project_fps > NEED_RATIO * source_fps * min_rate
}

/// Source-time frame rate the intermediate must have so every output frame lands on (or next
/// to) an interpolated frame: `⌈project fps / slowest speed⌉`, capped at `8 × source fps` and
/// 480 fps. `None` when no interpolation is needed.
pub fn target_fps(project_fps: f64, source_fps: f64, min_rate: f64) -> Option<f64> {
    if !needs_interpolation(project_fps, source_fps, min_rate) {
        return None;
    }
    let f = (project_fps / min_rate - 1e-6).ceil().min(MAX_TARGET_RATIO * source_fps).min(MAX_TARGET_FPS);
    (f > source_fps * 1.05).then_some(f)
}

/// `--factor` equivalent of a target rate for pipelines without `--target-fps`.
pub fn factor_for(target_fps: f64, source_fps: f64) -> u32 {
    ((target_fps / source_fps.max(1.0)).ceil() as u32).clamp(2, 8)
}

pub fn flow_cache_dir() -> PathBuf {
    crate::paths::cache_dir().join("flow")
}

fn mtime_secs(path: &Path) -> u64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn spec_key(spec: FlowSpec) -> String {
    match spec {
        FlowSpec::TargetFps(f) => format!("fps:{}", crate::export::decode::fmt_rate(f)),
        FlowSpec::Factor(n) => n.to_string(),
    }
}

/// `sha1(path|mtime|A|B|spec)` as hex.
pub fn cache_key(path: &Path, in_ms: f64, out_ms: f64, spec: FlowSpec) -> String {
    let mut h = Sha1::new();
    h.update(format!("{}|{}|{}|{}|{}", path.to_string_lossy(), mtime_secs(path), in_ms.round() as i64, out_ms.round() as i64, spec_key(spec)).as_bytes());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

pub fn cli_args(src: &Path, in_ms: f64, out_ms: f64, spec: FlowSpec, out: &Path) -> Vec<String> {
    let mut v = vec![
        "-m".into(),
        "cappycat_pipeline".into(),
        "interpolate".into(),
        src.to_string_lossy().into_owned(),
        "--in-ms".into(),
        format!("{}", in_ms.round() as i64),
        "--out-ms".into(),
        format!("{}", out_ms.round() as i64),
    ];
    match spec {
        FlowSpec::TargetFps(f) => v.extend(["--target-fps".into(), crate::export::decode::fmt_rate(f)]),
        FlowSpec::Factor(n) => v.extend(["--factor".into(), n.to_string()]),
    }
    v.extend(["--out".into(), out.to_string_lossy().into_owned()]);
    v
}

/// Did the pipeline reject `--target-fps` (an older pipeline)?
pub fn is_unsupported_target_fps(error: &str) -> bool {
    let e = error.to_ascii_lowercase();
    (e.contains("unrecognized arguments") && e.contains("--target-fps")) || e.contains("the following arguments are required: --factor")
}

/// Produce (or reuse) the interpolated intermediate. `progress(pct, message)` receives the
/// pipeline's progress events.
pub fn interpolate(
    src: &Path,
    in_ms: f64,
    out_ms: f64,
    spec: FlowSpec,
    ctl: &TaskCtl,
    progress: &mut dyn FnMut(f64, String),
) -> Result<PathBuf, String> {
    let dir = flow_cache_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let out = dir.join(format!("{}.mp4", cache_key(src, in_ms, out_ms, spec)));
    if out.is_file() && std::fs::metadata(&out).map(|m| m.len() > 0).unwrap_or(false) {
        return Ok(out);
    }
    let pipeline_dir = find_pipeline_dir().ok_or("pipeline directory not found")?;
    let python = find_python(Some(&pipeline_dir)).ok_or("python interpreter not found")?;
    let tmp = out.with_extension("partial.mp4");
    let _ = std::fs::remove_file(&tmp);
    let mut cmd = python_command(&python, &pipeline_dir);
    cmd.args(cli_args(src, in_ms, out_ms, spec, &tmp)).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = crate::procs::spawn(&mut cmd).map_err(|e| format!("cannot start {}: {e}", python.display()))?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let mut stderr = child.stderr.take().ok_or("no stderr")?;
    let stdin = child.stdin.take();
    let err_thread = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = stderr.read_to_string(&mut s);
        s
    });
    let managed = ctl.register_python(child, stdin);
    let mut result: Option<PathBuf> = None;
    let mut last_error: Option<String> = None;
    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
        let Ok(v) = serde_json::from_str::<Value>(line.trim()) else { continue };
        match v.get("event").and_then(|e| e.as_str()) {
            Some("progress") => {
                let pct = v.get("pct").and_then(|p| p.as_f64()).unwrap_or(0.0);
                let msg = v.get("message").and_then(|m| m.as_str()).unwrap_or("").to_string();
                progress(pct, msg);
            }
            Some("result") => result = v.get("path").and_then(|p| p.as_str()).map(PathBuf::from),
            Some("log") if v.get("level").and_then(|l| l.as_str()) == Some("error") => {
                last_error = v.get("message").and_then(|m| m.as_str()).map(String::from);
            }
            _ => {}
        }
    }
    let status = crate::procs::wait_child(&managed.child).map_err(|e| e.to_string())?;
    if let Some(t) = &managed.tree {
        t.kill(); // no leftover grandchildren
    }
    let err = err_thread.join().unwrap_or_default();
    if ctl.is_cancelled() {
        return Err("cancelled".into());
    }
    if !status.success() {
        let detail = last_error.unwrap_or_else(|| err.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("").trim().to_string());
        return Err(format!("interpolate exited with {:?}: {detail}", status.code()));
    }
    let produced = result.filter(|p| p.is_file()).unwrap_or(tmp.clone());
    if !produced.is_file() {
        return Err("interpolate reported success but wrote no file".into());
    }
    std::fs::rename(&produced, &out).or_else(|_| std::fs::copy(&produced, &out).map(|_| ())).map_err(|e| e.to_string())?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_follows_output_rate_and_slowest_speed() {
        // 24 fps footage into a 24 fps project at 1x: nothing to make
        assert_eq!(target_fps(24.0, 24.0, 1.0), None);
        // VFR phone clips averaging a little under the project rate: no interpolation
        assert_eq!(target_fps(24.0, 23.6, 1.0), None);
        assert_eq!(target_fps(25.0, 24.0, 1.0), None, "25 from 24 is within 10 %");
        // frame-rate conversion 24 → 60
        assert_eq!(target_fps(60.0, 24.0, 1.0), Some(60.0));
        // 1.5x at 60 fps: the source needs 40 frames per source second
        assert_eq!(target_fps(60.0, 24.0, 1.5), Some(40.0));
        // 1.5x at 30 fps from 24: 20 < 24 → the source has enough frames
        assert_eq!(target_fps(30.0, 24.0, 1.5), None);
        // slow motion 0.25x at 24 fps from 24: 96
        assert_eq!(target_fps(24.0, 24.0, 0.25), Some(96.0));
        // capped at 8x the source
        assert_eq!(target_fps(60.0, 24.0, 0.1), Some(192.0));
        assert_eq!(factor_for(96.0, 24.0), 4);
        assert_eq!(factor_for(60.0, 24.0), 3);
        assert_eq!(factor_for(25.0, 24.0), 2);
        assert!(needs_interpolation(60.0, 30.0, 1.0));
        assert!(!needs_interpolation(60.0, 60.0, 1.0));
    }

    #[test]
    fn cache_key_and_cli() {
        let a = cache_key(Path::new("C:/nope.mp4"), 0.0, 1000.0, FlowSpec::TargetFps(60.0));
        assert_eq!(a.len(), 40);
        assert_ne!(a, cache_key(Path::new("C:/nope.mp4"), 0.0, 1000.0, FlowSpec::TargetFps(48.0)), "target fps is part of the key");
        assert_ne!(a, cache_key(Path::new("C:/nope.mp4"), 0.0, 1000.0, FlowSpec::Factor(4)));
        let args = cli_args(Path::new("C:/a.mp4"), 250.4, 3000.0, FlowSpec::TargetFps(60.0), Path::new("C:/c/x.mp4")).join(" ");
        assert_eq!(args, "-m cappycat_pipeline interpolate C:/a.mp4 --in-ms 250 --out-ms 3000 --target-fps 60 --out C:/c/x.mp4");
        let args = cli_args(Path::new("C:/a.mp4"), 0.0, 3000.0, FlowSpec::Factor(4), Path::new("C:/c/x.mp4")).join(" ");
        assert_eq!(args, "-m cappycat_pipeline interpolate C:/a.mp4 --in-ms 0 --out-ms 3000 --factor 4 --out C:/c/x.mp4");
        assert!(is_unsupported_target_fps("interpolate exited with Some(2): cappycat_pipeline: error: unrecognized arguments: --target-fps 60"));
        assert!(is_unsupported_target_fps("error: the following arguments are required: --factor"));
        assert!(!is_unsupported_target_fps("CUDA out of memory"));
    }
}
