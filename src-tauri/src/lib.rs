#![recursion_limit = "256"]
//! Cappycat native core (Tauri v2).
//!
//! Modules:
//! * [`model`]        — serde mirror of `src/types/project.ts`
//! * [`ffmpeg`]       — ffmpeg/ffprobe discovery, probing, frames, thumbnails, waveforms
//! * [`media_server`] — local HTTP server for `<video>` streaming / frames / thumbs (token + registry)
//! * [`jobs`]         — job manager (cancel, GPU queue) + event sink abstraction
//! * [`procs`]        — child-process hygiene (job object, timeouts, python stdin watchdog)
//! * [`pipeline`]     — Python analysis pipeline runner
//! * [`separate`]     — Python voice / background stem separation runner
//! * [`export`]       — frame-accurate compositor exporter
//! * [`render`]       — CPU port of the preview renderer (grade, masks, blends)
//! * [`clips`]        — clips-folder scan (`scan_clips_folder`)
//! * [`logging`]      — rolling log files in `%LOCALAPPDATA%\cappycat\logs`
//! * [`commands`]     — Tauri `invoke` commands
//!
//! The keyframe / speed-curve maths lives in the `keyframes` workspace crate.

pub mod clips;
pub mod commands;
pub mod export;
pub mod ffmpeg;
pub mod jobs;
pub mod logging;
pub mod media_server;
pub mod model;
pub mod paths;
pub mod pipeline;
pub mod presets;
pub mod procs;
pub mod render;
pub mod separate;
pub mod setup;

use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::Manager;

/// Process-wide managed state.
pub struct AppState {
    /// Port of the local media server.
    pub media_port: u16,
    /// Per-launch secret path prefix of the media server (`/t/<token>/…`).
    pub media_token: String,
    /// Files the media server may serve.
    pub registry: media_server::MediaRegistry,
    /// Running pipeline / separation / export jobs.
    pub jobs: Arc<jobs::JobManager>,
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    if let Some(dir) = logging::init("cappycat", "info") {
        tracing::info!("Cappycat {} starting; logs in {}", env!("CARGO_PKG_VERSION"), dir.display());
    }
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // installed layout: Documents\Cappycat\{Clips,Projects,Exports} + seeded Characters / Presets
            match paths::seed_documents() {
                Ok(copied) if !copied.is_empty() => tracing::info!("seeded {} default file(s) into {}", copied.len(), paths::resolver().documents_home().display()),
                Ok(_) => {}
                Err(e) => tracing::warn!("{e}"),
            }
            tracing::info!("layout: {:?}; app data in {}", paths::layout(), paths::home().display());
            std::fs::create_dir_all(ffmpeg::thumbs_dir())?;
            let registry = media_server::MediaRegistry::new();
            let state = media_server::MediaState::new(ffmpeg::thumbs_dir(), registry.clone());
            let token = state.token.to_string();
            let port = media_server::start(state)?;
            app.manage(AppState { media_port: port, media_token: token, registry, jobs: Arc::new(jobs::JobManager::new()) });
            match ffmpeg::find_binaries() {
                Ok(b) => tracing::info!("ffmpeg: {}", b.ffmpeg.display()),
                Err(e) => tracing::warn!("{e}"),
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::probe_media,
            commands::import_media,
            commands::register_media,
            commands::media_server_url,
            commands::extract_thumbnails,
            commands::extract_waveform,
            commands::load_lut,
            commands::run_pipeline,
            commands::cancel_pipeline,
            commands::separate_audio,
            commands::cancel_separate,
            commands::export_project,
            commands::cancel_export,
            commands::save_project,
            commands::load_project,
            commands::open_document,
            commands::evaluate_keyframes,
            commands::speed_curve_lut,
            commands::app_paths,
            commands::scan_clips_folder,
            commands::load_universal_adjust,
            commands::save_universal_adjust,
            commands::setup_ai,
            commands::cancel_setup,
            commands::setup_status,
            commands::setup_plan,
        ])
        .build(tauri::generate_context!())
        .expect("error while building Cappycat");
    app.run(|handle, event| {
        if let tauri::RunEvent::ExitRequested { .. } = event {
            // Stop every job: ffmpeg is killed, python gets its stdin closed, and exports
            // delete their partial files. (Anything still alive when the process ends is
            // killed with the job object.)
            if let Some(state) = handle.try_state::<AppState>() {
                let ids = state.jobs.running_ids();
                if !ids.is_empty() {
                    tracing::info!("exit requested: cancelling {} job(s)", ids.len());
                    state.jobs.cancel_all();
                    let t = Instant::now();
                    while !state.jobs.running_ids().is_empty() && t.elapsed() < Duration::from_secs(3) {
                        std::thread::sleep(Duration::from_millis(50));
                    }
                }
            }
        }
    });
}
