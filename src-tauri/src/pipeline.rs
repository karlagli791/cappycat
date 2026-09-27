//! Python pipeline runner.
//!
//! Spawns `python -m cappycat_pipeline analyze <clips...> --out <json> [flags]`
//! with cwd = `<repo>/pipeline`, parses one JSON event per stdout line
//! (`docs/CONTRACTS.md` → "Pipeline progress events") and re-emits them as
//! Tauri events:
//!
//! | event                 | payload                                              |
//! |-----------------------|------------------------------------------------------|
//! | `pipeline://progress` | `{ jobId, event:"progress", stage, clip, pct, message }` |
//! | `pipeline://log`      | `{ jobId, event:"log", level, message }`             |
//! | `pipeline://result`   | `{ jobId, event:"result", path }`                    |
//! | `pipeline://analysis` | `{ jobId, path, analysis: AnalysisResult }` (file parsed by Rust) |
//! | `pipeline://exit`     | `{ jobId, code, cancelled, success }` (extra, not in the contract) |
//!
//! stderr lines and non-JSON stdout lines become `pipeline://log` entries
//! (`warn` and `info` respectively).
//!
//! The run goes through the GPU queue (one AI job at a time): while it waits it
//! emits `pipeline://progress` with `pct: 0` and the message
//! [`crate::jobs::GPU_WAIT_MESSAGE`]. The python child watches its stdin
//! (`CAPPYCAT_WATCH_STDIN=1`) and is stopped with its process tree on cancel.

use crate::jobs::{find_python, python_command, EventSink, JobIo, JobManager, SpawnOpts, GPU_WAIT_MESSAGE};
use crate::media_server::MediaRegistry;
use crate::model::{AnalysisResult, PipelineEvent, PipelineOptions};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub use crate::paths::is_pipeline_dir;

/// Locate the pipeline folder (one that contains `cappycat_pipeline/__init__.py`):
/// `CAPPYCAT_PIPELINE_DIR` → `<repo>/pipeline` (a checkout found from the cwd or the exe) → the
/// bundled `<resources>/pipeline` of an installed app. See [`crate::paths`].
pub fn find_pipeline_dir() -> Option<PathBuf> {
    crate::paths::pipeline_dir()
}

/// Turn `\\?\C:\x` (from `canonicalize`) back into `C:\x` so child processes and
/// the UI get a normal path; verbatim UNC paths become `\\server\share\…`, and
/// paths that cannot be expressed without the prefix are kept (`dunce`).
pub fn strip_verbatim(p: PathBuf) -> PathBuf {
    dunce::simplified(&p).to_path_buf()
}

/// Where analysis JSON files are written: `<cache>/analysis/<job>.json`.
pub fn analysis_out_path(job_id: &str) -> PathBuf {
    crate::ffmpeg::cache_dir().join("analysis").join(format!("{job_id}.json"))
}

fn fmt_num(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// Translate [`PipelineOptions`] into CLI arguments (after the clip paths).
pub fn cli_args(paths: &[String], out: &Path, options: &PipelineOptions) -> Vec<String> {
    let mut args: Vec<String> = vec!["-m".into(), "cappycat_pipeline".into(), "analyze".into()];
    args.extend(paths.iter().cloned());
    args.push("--out".into());
    args.push(out.to_string_lossy().into_owned());
    args.push("--detector".into());
    args.push(options.detector.as_cli().into());
    args.push("--shot-detector".into());
    args.push(options.shot_detector.as_cli().into());
    args.push("--target-duration-ms".into());
    args.push(fmt_num(options.target_duration_ms));
    args.push("--target-lufs".into());
    args.push(fmt_num(options.target_lufs));
    args.push("--similarity-threshold".into());
    args.push(fmt_num(options.similarity_threshold));
    args.push("--smoothing".into());
    args.push(options.smoothing.as_cli().into());
    args.push("--smoothing-alpha".into());
    args.push(fmt_num(options.smoothing_alpha));
    if !options.normalize_audio {
        args.push("--no-normalize-audio".into());
    }
    if !options.detect_beats {
        args.push("--no-beats".into());
    }
    if options.keep_order {
        args.push("--keep-order".into());
    }
    // `--prompts` takes a list; keep it last so it cannot swallow other flags. argparse
    // would read a prompt starting with '-' (e.g. "-dog") as an option: such prompts get a
    // leading space, which argparse treats as a value and the detector ignores.
    let prompts: Vec<String> = options
        .prompts
        .iter()
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| if p.starts_with('-') { format!(" {p}") } else { p.to_string() })
        .collect();
    if !prompts.is_empty() {
        args.push("--prompts".into());
        args.extend(prompts);
    }
    args
}

/// Merge `jobId` into an event payload object.
fn with_job(job_id: &str, mut v: Value) -> Value {
    if let Value::Object(ref mut m) = v {
        m.insert("jobId".into(), Value::String(job_id.to_string()));
        v
    } else {
        json!({ "jobId": job_id, "value": v })
    }
}

/// Handle one stdout line from the pipeline process.
pub fn handle_stdout_line(sink: &dyn EventSink, job_id: &str, line: &str) {
    handle_stdout_line_with(sink, job_id, line, None)
}

/// [`handle_stdout_line`] that registers the media of a loaded analysis with the media server.
pub fn handle_stdout_line_with(sink: &dyn EventSink, job_id: &str, line: &str, registry: Option<&MediaRegistry>) {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return;
    }
    match serde_json::from_str::<PipelineEvent>(trimmed) {
        Ok(event) => {
            let name = match &event {
                PipelineEvent::Progress { .. } => "pipeline://progress",
                PipelineEvent::Log { .. } => "pipeline://log",
                PipelineEvent::Result { .. } => "pipeline://result",
            };
            let payload = serde_json::to_value(&event).unwrap_or(Value::Null);
            sink.emit(name, with_job(job_id, payload));
            if let PipelineEvent::Result { path } = event {
                emit_analysis(sink, job_id, &path, registry);
            }
        }
        Err(_) => sink.emit(
            "pipeline://log",
            json!({ "jobId": job_id, "event": "log", "level": "info", "message": trimmed }),
        ),
    }
}

fn emit_analysis(sink: &dyn EventSink, job_id: &str, path: &str, registry: Option<&MediaRegistry>) {
    match std::fs::read(path) {
        Ok(bytes) => match serde_json::from_str::<AnalysisResult>(String::from_utf8_lossy(&bytes).trim_start_matches('\u{feff}')) {
            Ok(analysis) => {
                if let Some(r) = registry {
                    r.register_analysis(&analysis);
                }
                sink.emit("pipeline://analysis", json!({ "jobId": job_id, "path": path, "analysis": analysis }))
            }
            Err(e) => sink.emit(
                "pipeline://log",
                json!({ "jobId": job_id, "event": "log", "level": "error",
                        "message": format!("analysis JSON at {path} does not match the contract: {e}") }),
            ),
        },
        Err(e) => sink.emit(
            "pipeline://log",
            json!({ "jobId": job_id, "event": "log", "level": "error",
                    "message": format!("cannot read analysis JSON {path}: {e}") }),
        ),
    }
}

/// Start the analysis pipeline; returns the job id immediately.
pub fn run_pipeline(jobs: Arc<JobManager>, sink: Arc<dyn EventSink>, paths: Vec<String>, options: PipelineOptions) -> Result<String, String> {
    run_pipeline_with(jobs, sink, paths, options, None)
}

/// [`run_pipeline`] registering the analysis' media with the media server.
pub fn run_pipeline_with(
    jobs: Arc<JobManager>,
    sink: Arc<dyn EventSink>,
    paths: Vec<String>,
    options: PipelineOptions,
    registry: Option<MediaRegistry>,
) -> Result<String, String> {
    if paths.is_empty() {
        return Err("no clip paths given".into());
    }
    for p in &paths {
        // `analyze` accepts clip files as well as folders of clips.
        if !Path::new(p).exists() {
            return Err(format!("clip not found: {p}"));
        }
    }
    let pipeline_dir = find_pipeline_dir().ok_or("pipeline directory not found (set CAPPYCAT_PIPELINE_DIR)")?;
    let python = find_python(Some(&pipeline_dir)).ok_or("python interpreter not found")?;
    let job_id = JobManager::new_job_id("job");
    let out = analysis_out_path(&job_id);
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let mut cmd = python_command(&python, &pipeline_dir);
    cmd.args(cli_args(&paths, &out, &options));

    sink.emit(
        "pipeline://log",
        json!({ "jobId": job_id, "event": "log", "level": "info",
                "message": format!("starting {} in {}", python.display(), pipeline_dir.display()) }),
    );

    let (s_out, s_err, s_exit, s_wait) = (sink.clone(), sink.clone(), sink.clone(), sink.clone());
    let (id_out, id_err, id_exit, id_wait) = (job_id.clone(), job_id.clone(), job_id.clone(), job_id.clone());
    let jobs_exit = jobs.clone();
    let gpu_wait: Box<dyn Fn() + Send + Sync> = Box::new(move || {
        s_wait.emit(
            "pipeline://progress",
            json!({ "jobId": id_wait, "event": "progress", "stage": "ingest", "clip": null, "pct": 0.0, "message": GPU_WAIT_MESSAGE }),
        )
    });
    jobs.spawn(
        &job_id,
        cmd,
        JobIo {
            on_stdout: Box::new(move |line| handle_stdout_line_with(s_out.as_ref(), &id_out, line, registry.as_ref())),
            on_stderr: Box::new(move |line| {
                if !line.trim().is_empty() {
                    s_err.emit(
                        "pipeline://log",
                        json!({ "jobId": id_err, "event": "log", "level": "warn", "message": line }),
                    );
                }
            }),
            on_exit: Box::new(move |status, cancelled| {
                let code = status.and_then(|s| s.code());
                let success = status.map(|s| s.success()).unwrap_or(false) && !cancelled;
                s_exit.emit(
                    "pipeline://exit",
                    json!({ "jobId": id_exit, "code": code, "cancelled": cancelled, "success": success }),
                );
                jobs_exit.finish(&id_exit);
            }),
        },
        SpawnOpts { python: true, gpu_wait: Some(gpu_wait) },
    )?;
    Ok(job_id)
}

pub fn cancel_pipeline(jobs: &JobManager, job_id: &str) -> Result<(), String> {
    jobs.cancel(job_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::CollectingSink;
    use crate::model::{Detector, ShotDetector, Smoothing};

    #[test]
    fn cli_args_follow_the_contract() {
        let opts = PipelineOptions {
            prompts: vec!["raccoon in hoodie".into(), "person".into()],
            detector: Detector::GroundedSam2,
            shot_detector: ShotDetector::Transnetv2,
            target_duration_ms: 165_000.0,
            normalize_audio: false,
            target_lufs: -14.0,
            detect_beats: false,
            similarity_threshold: 0.85,
            smoothing: Smoothing::Savgol,
            smoothing_alpha: 0.15,
            keep_order: true,
        };
        let args = cli_args(&["C:/a.mp4".into(), "C:/b.mp4".into()], Path::new("C:/out/analysis.json"), &opts);
        let joined = args.join(" ");
        assert!(joined.starts_with("-m cappycat_pipeline analyze C:/a.mp4 C:/b.mp4 --out C:/out/analysis.json"));
        assert!(joined.contains("--detector grounded_sam2"));
        assert!(joined.contains("--shot-detector transnetv2"));
        assert!(joined.contains("--target-duration-ms 165000"));
        assert!(joined.contains("--target-lufs -14"));
        assert!(joined.contains("--similarity-threshold 0.85"));
        assert!(joined.contains("--smoothing savgol"));
        assert!(joined.contains("--smoothing-alpha 0.15"));
        assert!(joined.contains("--no-normalize-audio"));
        assert!(joined.contains("--no-beats"));
        assert!(joined.contains("--keep-order"));
        assert!(joined.ends_with("--prompts raccoon in hoodie person"));
        // a prompt that looks like an option is passed as a value
        let dashed = PipelineOptions { prompts: vec!["-dog".into(), "  ".into(), "cat".into()], ..Default::default() };
        let a = cli_args(&["x".into()], Path::new("o.json"), &dashed);
        assert_eq!(&a[a.len() - 3..], &["--prompts".to_string(), " -dog".to_string(), "cat".to_string()]);
        let defaults = cli_args(&["x".into()], Path::new("o.json"), &PipelineOptions::default());
        assert!(!defaults.iter().any(|a| a == "--no-beats" || a == "--no-normalize-audio" || a == "--prompts" || a == "--keep-order"));
    }

    #[test]
    fn stdout_lines_are_routed_to_events() {
        let sink = CollectingSink::new();
        handle_stdout_line(sink.as_ref(), "job_1", r#"{"event":"progress","stage":"shots","clip":"a.mp4","pct":0.5,"message":"half"}"#);
        handle_stdout_line(sink.as_ref(), "job_1", r#"{"event":"log","level":"warn","message":"careful"}"#);
        handle_stdout_line(sink.as_ref(), "job_1", "plain text line");
        handle_stdout_line(sink.as_ref(), "job_1", "");
        let ev = sink.snapshot();
        assert_eq!(ev.len(), 3);
        assert_eq!(ev[0].0, "pipeline://progress");
        assert_eq!(ev[0].1["jobId"], "job_1");
        assert_eq!(ev[0].1["stage"], "shots");
        assert_eq!(ev[0].1["pct"], 0.5);
        assert_eq!(ev[1].0, "pipeline://log");
        assert_eq!(ev[1].1["level"], "warn");
        assert_eq!(ev[2].0, "pipeline://log");
        assert_eq!(ev[2].1["level"], "info");
        assert_eq!(ev[2].1["message"], "plain text line");
    }

    #[test]
    fn result_event_loads_analysis_file() {
        let dir = crate::ffmpeg::cache_dir().join("test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("analysis_test.json");
        std::fs::write(
            &path,
            r#"{"version":1,"generatedAt":"2026-09-24T20:00:00Z","clips":[],"timeline":{"version":1,"id":"p","name":"auto","fps":24,"width":1920,"height":1080,"assets":[],"tracks":[],"beatMarkers":[]}}"#,
        )
        .unwrap();
        let sink = CollectingSink::new();
        let line = json!({ "event": "result", "path": path.to_string_lossy() }).to_string();
        handle_stdout_line(sink.as_ref(), "job_2", &line);
        let ev = sink.snapshot();
        assert_eq!(ev[0].0, "pipeline://result");
        assert_eq!(ev[1].0, "pipeline://analysis");
        assert_eq!(ev[1].1["analysis"]["timeline"]["name"], "auto");
        assert_eq!(ev[1].1["jobId"], "job_2");

        // Missing file → error log, not a panic.
        let sink = CollectingSink::new();
        handle_stdout_line(sink.as_ref(), "job_3", r#"{"event":"result","path":"C:/definitely/missing.json"}"#);
        let ev = sink.snapshot();
        assert_eq!(ev[1].0, "pipeline://log");
        assert_eq!(ev[1].1["level"], "error");
    }

    #[test]
    fn run_pipeline_validates_inputs() {
        let jobs = Arc::new(JobManager::new());
        let sink = CollectingSink::new();
        assert!(run_pipeline(jobs.clone(), sink.clone(), vec![], PipelineOptions::default()).is_err());
        let err = run_pipeline(jobs, sink, vec!["C:/nope/missing.mp4".into()], PipelineOptions::default()).unwrap_err();
        assert!(err.contains("clip not found"));
    }

    #[test]
    fn strips_verbatim_prefix() {
        if cfg!(windows) {
            assert_eq!(strip_verbatim(PathBuf::from(r"\\?\C:\x\y")), PathBuf::from(r"C:\x\y"));
            // a verbatim UNC path is never mangled into "UNC\server\..." (the old prefix strip did that)
            let unc = strip_verbatim(PathBuf::from(r"\\?\UNC\server\share\x"));
            assert!(unc.as_path() == Path::new(r"\\server\share\x") || unc.as_path() == Path::new(r"\\?\UNC\server\share\x"), "{}", unc.display());
        }
        assert_eq!(strip_verbatim(PathBuf::from(r"C:\x")), PathBuf::from(r"C:\x"));
    }

    #[test]
    fn pipeline_dir_needs_the_package() {
        let root = crate::ffmpeg::cache_dir().join("test").join("pipeline_dir_check");
        let _ = std::fs::remove_dir_all(&root);
        let fake = root.join("pipeline");
        std::fs::create_dir_all(fake.join("cappycat_pipeline")).unwrap();
        assert!(!is_pipeline_dir(&fake), "an empty package folder is not enough");
        std::fs::write(fake.join("cappycat_pipeline").join("__init__.py"), "").unwrap();
        assert!(is_pipeline_dir(&fake));
        assert!(!is_pipeline_dir(&root));
        if let Some(found) = find_pipeline_dir() {
            assert!(is_pipeline_dir(&found), "{}", found.display());
            assert!(!found.to_string_lossy().starts_with(r"\\?\"));
        }
    }
}
