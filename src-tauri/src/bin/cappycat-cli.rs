//! Headless command-line renderer (design spec: "VRAM Budgeting and Headless CLI Rendering").
//!
//! ```text
//! cappycat-cli export <project.json | analysis.json> <out.mp4|out.mov>
//!              [--preset h264_nvenc_mp4|h264_mp4|prores_mov] [--range <startMs> <endMs>]
//!              [--no-universal] [-v]
//! ```
//! `-v` prints the export's info log lines (stages, timings) on the console; every
//! run also logs to `%LOCALAPPDATA%\Cappycat\logs\cappycat-cli.<date>.log`.
//! A project that doesn't carry its own universal-adjust setting gets the universal preset from
//! `<repo>/presets/universal-adjust.json` (on when the preset says `enabledByDefault`);
//! `--no-universal` renders without it.
//! Accepts a saved project or a pipeline `AnalysisResult` (its `timeline` is exported).
//! Runs the same compositor as the desktop app's Export dialog, without a window.

use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cappycat_lib::export::{self, ExportRange};
use cappycat_lib::jobs::{EventSink, JobManager};
use cappycat_lib::model::Project;
use serde_json::Value;

struct ConsoleSink {
    verbose: bool,
    last_print: Mutex<Instant>,
    result: Mutex<Option<(bool, String)>>,
}

impl EventSink for ConsoleSink {
    fn emit(&self, event: &str, payload: Value) {
        match event {
            "export://progress" => {
                let mut last = self.last_print.lock().unwrap();
                if last.elapsed() >= Duration::from_millis(1000) {
                    *last = Instant::now();
                    let pct = payload["pct"].as_f64().unwrap_or(0.0) * 100.0;
                    let msg = payload["message"].as_str().unwrap_or("");
                    println!("[{pct:5.1}%] {msg}");
                    let _ = std::io::stdout().flush();
                }
            }
            "export://log" => {
                let level = payload["level"].as_str().unwrap_or("info");
                let msg = payload["message"].as_str().unwrap_or("");
                if level != "info" {
                    eprintln!("{level}: {msg}");
                } else if self.verbose {
                    println!("info: {msg}");
                }
            }
            "export://done" => {
                let ok = payload["ok"].as_bool().unwrap_or(false);
                let text = if ok {
                    payload["outPath"].as_str().unwrap_or("").to_string()
                } else {
                    payload["error"].as_str().unwrap_or("export failed").to_string()
                };
                *self.result.lock().unwrap() = Some((ok, text));
            }
            _ => {}
        }
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: cappycat-cli export <project.json | analysis.json> <out.mp4|out.mov> \
         [--preset h264_nvenc_mp4|h264_mp4|prores_mov] [--range <startMs> <endMs>] [--no-universal] [-v]"
    );
    std::process::exit(2);
}

fn load_project(path: &str) -> Result<Project, String> {
    // a saved project or a pipeline AnalysisResult (its timeline), BOM tolerated
    let p = std::path::absolute(path).map_err(|e| format!("{path}: {e}"))?;
    cappycat_lib::commands::open_document_at(&cappycat_lib::media_server::MediaRegistry::new(), &p).map(|d| d.project)
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let verbose = args.iter().any(|a| a == "-v" || a == "--verbose");
    args.retain(|a| a != "-v" && a != "--verbose");
    // the console shows warnings from the core; `-v` adds the export's info lines (printed by
    // the sink below); everything goes to the log file
    cappycat_lib::logging::init("cappycat-cli", "warn");
    if args.first().map(String::as_str) != Some("export") || args.len() < 3 {
        usage();
    }
    let input = &args[1];
    let out = &args[2];
    let mut preset = if out.to_lowercase().ends_with(".mov") { "prores_mov".to_string() } else { "h264_nvenc_mp4".to_string() };
    let mut range = None;
    let mut universal = true;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--preset" if i + 1 < args.len() => {
                preset = args[i + 1].clone();
                i += 2;
            }
            "--no-universal" => {
                universal = false;
                i += 1;
            }
            "--range" if i + 2 < args.len() => {
                let start: f64 = args[i + 1].parse().unwrap_or_else(|_| usage());
                let end: f64 = args[i + 2].parse().unwrap_or_else(|_| usage());
                range = Some(ExportRange { start_ms: start, end_ms: end });
                i += 3;
            }
            _ => usage(),
        }
    }

    let mut project = match load_project(input) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    if !universal {
        project.universal_adjust = None;
        println!("universal adjust: off (--no-universal)");
    } else if project.universal_adjust.is_none() {
        match cappycat_lib::presets::load() {
            Ok(info) => {
                project.universal_adjust = Some(info.preset.to_adjust());
                let state = if info.preset.enabled_by_default { "on" } else { "off (disabled by default)" };
                println!("universal adjust: {state} from {}", info.path);
            }
            Err(e) => eprintln!("warn: universal preset not applied: {e}"),
        }
    } else if let Some(u) = &project.universal_adjust {
        println!("universal adjust: {} (from the project)", if u.enabled { "on" } else { "off" });
    }
    let duration = export::project_duration_ms(&project);
    println!(
        "exporting '{}' ({:.1} s, {}x{} @ {} fps) -> {out} [{preset}]",
        project.name, duration / 1000.0, project.width, project.height, project.fps
    );

    let jobs = Arc::new(JobManager::new());
    let sink = Arc::new(ConsoleSink { verbose, last_print: Mutex::new(Instant::now()), result: Mutex::new(None) });
    let started = Instant::now();
    let job_id = match export::export_project(jobs.clone(), sink.clone(), project, out.clone(), &preset, range) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    while !jobs.wait(&job_id, Duration::from_secs(3600)) {}
    // the done event is emitted just before the task is marked finished; give it a moment
    for _ in 0..50 {
        if sink.result.lock().unwrap().is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let outcome = sink.result.lock().unwrap().clone();
    match outcome {
        Some((true, path)) => {
            println!("done in {:.1} s: {path}", started.elapsed().as_secs_f64());
        }
        Some((false, err)) => {
            eprintln!("export failed: {err}");
            std::process::exit(1);
        }
        None => {
            eprintln!("export ended without a result");
            std::process::exit(1);
        }
    }
}
