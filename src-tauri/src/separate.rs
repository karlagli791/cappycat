//! Voice separation runner (CapCut "Isolate voice" / "Remove vocals").
//!
//! Spawns `python -m cappycat_pipeline separate <paths...> --json` (cwd = `<repo>/pipeline`, venv
//! python preferred, same job manager as the analysis pipeline) and re-emits its JSON lines:
//!
//! | event                | payload                                                              |
//! |----------------------|----------------------------------------------------------------------|
//! | `separate://progress`| `{ jobId, pct, clipPct, message, clip }` (`pct` over all files, 0..1) |
//! | `separate://result`  | `{ jobId, path, stems: { vocals, background } }` (one per file)      |
//! | `separate://log`     | `{ jobId, level, message }` (stderr → `warn`, non-JSON stdout → `info`) |
//! | `separate://done`    | `{ jobId, ok, error }` (`error: "cancelled"` after `cancel_separate`) |
//!
//! Stems are cached by the pipeline (`%LOCALAPPDATA%\cappycat\cache\stems\<sha1>`), so asking again
//! for an already separated file returns its `result` immediately.
//!
//! The run goes through the GPU queue (one AI job at a time): while it waits it emits
//! `separate://progress` with `pct: 0` and [`crate::jobs::GPU_WAIT_MESSAGE`]. Stems of every
//! `separate://result` are registered with the media server.

use crate::jobs::{find_python, python_command, EventSink, JobIo, JobManager, SpawnOpts, GPU_WAIT_MESSAGE};
use crate::media_server::MediaRegistry;
use crate::pipeline::find_pipeline_dir;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// CLI arguments for `separate` (after the interpreter).
pub fn cli_args(paths: &[String], model: Option<&str>) -> Vec<String> {
    let mut args: Vec<String> = vec!["-m".into(), "cappycat_pipeline".into(), "separate".into()];
    args.extend(paths.iter().cloned());
    if let Some(m) = model {
        args.push("--model".into());
        args.push(m.into());
    }
    args.push("--json".into());
    args
}

fn num(v: &Value, key: &str) -> Value {
    v.get(key).and_then(Value::as_f64).map(Value::from).unwrap_or(Value::Null)
}

/// Handle one stdout line of the `separate` process. Returns the message of an `error` log
/// event (remembered for `separate://done`).
pub fn handle_stdout_line(sink: &dyn EventSink, job_id: &str, line: &str) -> Option<String> {
    handle_stdout_line_with(sink, job_id, line, None)
}

/// [`handle_stdout_line`] that registers the stems with the media server.
pub fn handle_stdout_line_with(sink: &dyn EventSink, job_id: &str, line: &str, registry: Option<&MediaRegistry>) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    let Ok(v) = serde_json::from_str::<Value>(trimmed) else {
        sink.emit("separate://log", json!({ "jobId": job_id, "level": "info", "message": trimmed }));
        return None;
    };
    match v.get("event").and_then(Value::as_str) {
        Some("progress") => {
            sink.emit(
                "separate://progress",
                json!({ "jobId": job_id, "pct": num(&v, "pct"), "clipPct": num(&v, "clipPct"),
                        "message": v.get("message").cloned().unwrap_or(Value::Null),
                        "clip": v.get("clip").cloned().unwrap_or(Value::Null) }),
            );
            None
        }
        Some("result") => {
            let stems = v.get("stems").cloned().unwrap_or(Value::Null);
            let complete = stems.get("vocals").and_then(Value::as_str).is_some()
                && stems.get("background").and_then(Value::as_str).is_some();
            if complete {
                if let Some(r) = registry {
                    for k in ["vocals", "background"] {
                        if let Some(p) = stems.get(k).and_then(Value::as_str) {
                            r.register(p);
                        }
                    }
                }
                sink.emit("separate://result", json!({ "jobId": job_id, "path": v.get("path").cloned().unwrap_or(Value::Null), "stems": stems }));
                None
            } else {
                let msg = format!("separation result without stems: {trimmed}");
                sink.emit("separate://log", json!({ "jobId": job_id, "level": "error", "message": msg }));
                Some(msg)
            }
        }
        Some("log") => {
            let level = v.get("level").and_then(Value::as_str).unwrap_or("info").to_string();
            let message = v.get("message").and_then(Value::as_str).unwrap_or("").to_string();
            sink.emit("separate://log", json!({ "jobId": job_id, "level": level, "message": message }));
            (level == "error").then_some(message)
        }
        _ => {
            sink.emit("separate://log", json!({ "jobId": job_id, "level": "info", "message": trimmed }));
            None
        }
    }
}

/// Start separating `paths` into vocals / background stems; returns the job id immediately.
pub fn run_separate(jobs: Arc<JobManager>, sink: Arc<dyn EventSink>, paths: Vec<String>) -> Result<String, String> {
    spawn_separate(jobs, sink, paths, None, &[], None)
}

/// [`run_separate`] registering the stems with the media server.
pub fn run_separate_with(jobs: Arc<JobManager>, sink: Arc<dyn EventSink>, paths: Vec<String>, registry: Option<MediaRegistry>) -> Result<String, String> {
    spawn_separate(jobs, sink, paths, None, &[], registry)
}

/// [`run_separate`] with an optional model and extra environment (tests point
/// `CAPPYCAT_STEMS_DIR` at a fixture cache).
pub fn spawn_separate(
    jobs: Arc<JobManager>,
    sink: Arc<dyn EventSink>,
    paths: Vec<String>,
    model: Option<&str>,
    env: &[(&str, &str)],
    registry: Option<MediaRegistry>,
) -> Result<String, String> {
    let mut unique: Vec<String> = Vec::new();
    for p in paths {
        let p = p.trim().to_string();
        if p.is_empty() || unique.iter().any(|u| u.eq_ignore_ascii_case(&p)) {
            continue;
        }
        if !Path::new(&p).is_file() {
            return Err(format!("media not found: {p}"));
        }
        unique.push(p);
    }
    if unique.is_empty() {
        return Err("no media paths given".into());
    }
    let pipeline_dir = find_pipeline_dir().ok_or("pipeline directory not found (set CAPPYCAT_PIPELINE_DIR)")?;
    let python = find_python(Some(&pipeline_dir)).ok_or("python interpreter not found")?;
    let job_id = JobManager::new_job_id("separate");

    let mut cmd = python_command(&python, &pipeline_dir);
    cmd.args(cli_args(&unique, model));
    for (k, v) in env {
        cmd.env(k, v);
    }
    sink.emit(
        "separate://log",
        json!({ "jobId": job_id, "level": "info",
                "message": format!("separating {} file(s) with {}", unique.len(), python.display()) }),
    );

    let last_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let last_stderr: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let (s_out, s_err, s_exit, s_wait) = (sink.clone(), sink.clone(), sink.clone(), sink.clone());
    let (id_out, id_err, id_exit, id_wait) = (job_id.clone(), job_id.clone(), job_id.clone(), job_id.clone());
    let gpu_wait: Box<dyn Fn() + Send + Sync> = Box::new(move || {
        s_wait.emit("separate://progress", json!({ "jobId": id_wait, "pct": 0.0, "clipPct": 0.0, "message": GPU_WAIT_MESSAGE, "clip": null }))
    });
    let (err_out, err_exit) = (last_error.clone(), last_error.clone());
    let (stderr_in, stderr_exit) = (last_stderr.clone(), last_stderr.clone());
    let jobs_exit = jobs.clone();
    jobs.spawn(
        &job_id,
        cmd,
        JobIo {
            on_stdout: Box::new(move |line| {
                if let Some(e) = handle_stdout_line_with(s_out.as_ref(), &id_out, line, registry.as_ref()) {
                    *err_out.lock().unwrap() = Some(e);
                }
            }),
            on_stderr: Box::new(move |line| {
                let line = line.trim();
                if line.is_empty() {
                    return;
                }
                *stderr_in.lock().unwrap() = Some(line.to_string());
                s_err.emit("separate://log", json!({ "jobId": id_err, "level": "warn", "message": line }));
            }),
            on_exit: Box::new(move |status, cancelled| {
                let success = status.map(|s| s.success()).unwrap_or(false) && !cancelled;
                let error: Option<String> = if cancelled {
                    Some("cancelled".into())
                } else if success {
                    None
                } else {
                    let code = status.and_then(|s| s.code());
                    err_exit.lock().unwrap().clone().or_else(|| {
                        let tail = stderr_exit.lock().unwrap().clone().unwrap_or_default();
                        Some(format!("separation failed (exit code {code:?}) {tail}").trim().to_string())
                    })
                };
                s_exit.emit("separate://done", json!({ "jobId": id_exit, "ok": success, "error": error }));
                jobs_exit.finish(&id_exit);
            }),
        },
        SpawnOpts { python: true, gpu_wait: Some(gpu_wait) },
    )?;
    Ok(job_id)
}

pub fn cancel_separate(jobs: &JobManager, job_id: &str) -> Result<(), String> {
    jobs.cancel(job_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::CollectingSink;
    use std::time::Duration;

    #[test]
    fn cli_args_follow_the_contract() {
        let a = cli_args(&["C:/a b/clip1.mp4".into(), "C:/c.wav".into()], None);
        assert_eq!(a, vec!["-m", "cappycat_pipeline", "separate", "C:/a b/clip1.mp4", "C:/c.wav", "--json"]);
        let a = cli_args(&["x.mp4".into()], Some("htdemucs"));
        assert_eq!(a.join(" "), "-m cappycat_pipeline separate x.mp4 --model htdemucs --json");
    }

    #[test]
    fn stdout_lines_are_routed_to_events() {
        let sink = CollectingSink::new();
        let s = sink.as_ref();
        assert!(handle_stdout_line(s, "sep_1", r#"{"event":"progress","stage":"separate","clip":"clip1.mp4","pct":0.43,"clipPct":0.86,"message":"clip1.mp4: segment 3/12"}"#).is_none());
        assert!(handle_stdout_line(s, "sep_1", r#"{"event":"result","path":"C:/m/clip1.mp4","stems":{"vocals":"C:/s/v.wav","background":"C:/s/b.wav"}}"#).is_none());
        assert_eq!(handle_stdout_line(s, "sep_1", r#"{"event":"log","level":"error","message":"clip2.mp4: separation failed"}"#).as_deref(), Some("clip2.mp4: separation failed"));
        assert!(handle_stdout_line(s, "sep_1", r#"{"event":"result","path":"C:/m/x.mp4"}"#).is_some());
        assert!(handle_stdout_line(s, "sep_1", "plain text").is_none());
        assert!(handle_stdout_line(s, "sep_1", "  ").is_none());
        let p = &sink.named("separate://progress")[0];
        assert_eq!(p["jobId"], "sep_1");
        assert_eq!(p["pct"], 0.43);
        assert_eq!(p["clipPct"], 0.86);
        assert_eq!(p["clip"], "clip1.mp4");
        let r = sink.named("separate://result");
        assert_eq!(r.len(), 1, "a result without stems is not forwarded");
        assert_eq!(r[0], json!({ "jobId": "sep_1", "path": "C:/m/clip1.mp4", "stems": { "vocals": "C:/s/v.wav", "background": "C:/s/b.wav" } }));
        let logs = sink.named("separate://log");
        assert_eq!(logs.len(), 3);
        assert_eq!(logs[0]["level"], "error");
        assert_eq!(logs[2]["message"], "plain text");
    }

    #[test]
    fn run_separate_validates_inputs() {
        let jobs = Arc::new(JobManager::new());
        let sink = CollectingSink::new();
        assert!(run_separate(jobs.clone(), sink.clone(), vec![]).is_err());
        assert!(run_separate(jobs.clone(), sink.clone(), vec!["  ".into()]).is_err());
        let err = run_separate(jobs, sink, vec!["C:/nope/missing.mp4".into()]).unwrap_err();
        assert!(err.contains("media not found"));
    }

    /// End to end through the real CLI against a fixture cache (no model needed): the cached stems
    /// come back as `separate://result`, a missing file fails the job with its error message.
    #[test]
    fn separate_job_reports_cached_stems() {
        let Some(dir) = find_pipeline_dir() else {
            eprintln!("SKIP: pipeline dir not found");
            return;
        };
        if find_python(Some(&dir)).is_none() {
            eprintln!("SKIP: python not found");
            return;
        }
        let root = crate::ffmpeg::cache_dir().join("test").join("separate_job");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let src = root.join("src.wav");
        crate::export::audio::write_wav_f32(&src, &[0.0; 960], 48_000, 2).unwrap();
        // the cache layout of cappycat_pipeline.separate: <root>/<sha1(path|size|mtime|model)>/..
        let key_py = "import sys; from cappycat_pipeline import separate as s; print(s.cache_key(sys.argv[1]))";
        let out = std::process::Command::new(find_python(Some(&dir)).unwrap())
            .args(["-c", key_py, &src.to_string_lossy()])
            .current_dir(&dir)
            .output()
            .unwrap();
        let key = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if key.len() != 40 {
            eprintln!("SKIP: pipeline not importable: {}", String::from_utf8_lossy(&out.stderr));
            return;
        }
        let stems_root = root.join("stems");
        let entry = stems_root.join(&key);
        std::fs::create_dir_all(&entry).unwrap();
        for n in ["vocals", "background"] {
            crate::export::audio::write_wav_f32(&entry.join(format!("{n}.wav")), &[0.0; 960], 48_000, 2).unwrap();
        }
        std::fs::write(entry.join("meta.json"), "{}").unwrap();

        let jobs = Arc::new(JobManager::new());
        let sink = CollectingSink::new();
        let root_s = stems_root.to_string_lossy().into_owned();
        let env = [("CAPPYCAT_STEMS_DIR", root_s.as_str())];
        let registry = MediaRegistry::new();
        let job = spawn_separate(jobs.clone(), sink.clone(), vec![src.to_string_lossy().into_owned()], None, &env, Some(registry.clone())).unwrap();
        assert!(job.starts_with("separate_"));
        assert!(jobs.wait(&job, Duration::from_secs(120)));
        let results = sink.named("separate://result");
        assert_eq!(results.len(), 1, "{:?}", sink.snapshot());
        assert_eq!(results[0]["path"], src.to_string_lossy().as_ref());
        assert!(results[0]["stems"]["vocals"].as_str().unwrap().ends_with(&format!("{key}/vocals.wav")));
        let done = sink.named("separate://done");
        assert_eq!(done, vec![json!({ "jobId": job, "ok": true, "error": null })]);
        assert!(sink.named("separate://progress").iter().any(|p| p["pct"] == 1.0));
        assert!(registry.contains(Path::new(results[0]["stems"]["vocals"].as_str().unwrap())), "stems are registered with the media server");

        // a file that cannot be separated -> ok:false with the pipeline's error message
        let silent = root.join("empty.txt");
        std::fs::write(&silent, "not media").unwrap();
        let sink = CollectingSink::new();
        let job = spawn_separate(jobs.clone(), sink.clone(), vec![silent.to_string_lossy().into_owned()], None, &env, None).unwrap();
        assert!(jobs.wait(&job, Duration::from_secs(120)));
        let done = &sink.named("separate://done")[0];
        assert_eq!(done["ok"], false);
        assert!(done["error"].as_str().unwrap().contains("empty.txt"), "{done}");
    }
}
