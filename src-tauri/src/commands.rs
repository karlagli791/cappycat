//! Tauri `invoke` commands (see the table in `docs/CONTRACTS.md`).
//!
//! Argument names are camelCase on the JS side (`invoke("extract_waveform",
//! { path, samplesPerSecond })`); Tauri maps them onto the snake_case Rust
//! parameters. All errors are returned as `String`.
//!
//! Every command is `async` (Tauri runs synchronous commands on the main
//! thread, where a slow file-system call or a lock would freeze the UI); work
//! that touches the disk or spawns processes runs on the blocking pool.

use crate::clips;
use crate::export::{self, ExportRange};
use crate::ffmpeg;
use crate::jobs::EventSink;
use crate::media_server::MediaRegistry;
use crate::model::{AnalysisResult, Asset, Keyframed, PipelineOptions, Project, SpeedCurve};
use crate::pipeline;
use crate::render::lut::{load_cube, LutPayload};
use crate::AppState;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tauri::{AppHandle, State};

/// `http://127.0.0.1:<port>/t/<token>`: the base the frontend appends `/media?path=…` to.
fn media_url(state: &AppState) -> String {
    crate::media_server::base_url(state.media_port, &state.media_token)
}

fn abs_path(path: &str) -> Result<PathBuf, String> {
    let p = PathBuf::from(path.trim());
    if !p.is_absolute() {
        return Err(format!("path must be absolute: {path}"));
    }
    Ok(p)
}

async fn blocking<T, F>(f: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, String> + Send + 'static,
{
    tauri::async_runtime::spawn_blocking(f).await.map_err(|e| e.to_string())?
}

/* --------------------------------------------------------------- media */

#[tauri::command]
pub async fn probe_media(state: State<'_, AppState>, path: String) -> Result<Asset, String> {
    let p = abs_path(&path)?;
    let registry = state.registry.clone();
    blocking(move || {
        let a = ffmpeg::probe(&p).map_err(String::from)?;
        registry.register_asset(&a);
        Ok(a)
    })
    .await
}

/// Probe several files (in parallel). Files that fail to probe are skipped
/// (logged); the command only errors when *every* path failed.
#[tauri::command]
pub async fn import_media(state: State<'_, AppState>, paths: Vec<String>) -> Result<Vec<Asset>, String> {
    let registry = state.registry.clone();
    blocking(move || import_paths(&registry, &paths)).await
}

pub fn import_paths(registry: &MediaRegistry, paths: &[String]) -> Result<Vec<Asset>, String> {
    let mut assets = Vec::new();
    let mut errors = Vec::new();
    let mut valid: Vec<(usize, PathBuf)> = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        match abs_path(path) {
            Ok(p) => valid.push((i, p)),
            Err(e) => errors.push(e),
        }
    }
    let probe_paths: Vec<PathBuf> = valid.iter().map(|(_, p)| p.clone()).collect();
    for ((i, p), r) in valid.into_iter().zip(ffmpeg::probe_many(&probe_paths)) {
        match r {
            Ok(mut a) => {
                a.order = Some(i as u32);
                registry.register_asset(&a);
                assets.push(a);
            }
            Err(e) => {
                tracing::warn!("import_media: {}: {e}", p.display());
                errors.push(format!("{}: {e}", p.display()));
            }
        }
    }
    if assets.is_empty() && !errors.is_empty() {
        return Err(errors.join("\n"));
    }
    Ok(assets)
}

/// Let the media server serve these files: anything the UI plays that did not come
/// through `probe_media` / `import_media` / `scan_clips_folder` / `load_project` /
/// `open_document` / an analysis / separation / export result.
#[tauri::command]
pub async fn register_media(state: State<'_, AppState>, paths: Vec<String>) -> Result<(), String> {
    let registry = state.registry.clone();
    blocking(move || {
        for p in &paths {
            let p = abs_path(p)?;
            registry.register(&p);
        }
        Ok(())
    })
    .await
}

/// `http://127.0.0.1:<port>/t/<token>`
#[tauri::command]
pub async fn media_server_url(state: State<'_, AppState>) -> Result<String, String> {
    Ok(media_url(&state))
}

/// Returns `http://127.0.0.1:<port>/t/<token>/thumb/<sha1>/<n>.jpg` URLs.
#[tauri::command]
pub async fn extract_thumbnails(state: State<'_, AppState>, path: String, count: u32, width: u32) -> Result<Vec<String>, String> {
    let base = media_url(&state);
    let p = abs_path(&path)?;
    blocking(move || {
        let files = ffmpeg::extract_thumbnails(&p, count, width).map_err(String::from)?;
        Ok(files
            .iter()
            .filter_map(|f| {
                let name = f.file_name()?.to_string_lossy().into_owned();
                let sha = f.parent()?.file_name()?.to_string_lossy().into_owned();
                Some(format!("{base}/thumb/{sha}/{name}"))
            })
            .collect())
    })
    .await
}

#[tauri::command]
pub async fn extract_waveform(path: String, samples_per_second: u32) -> Result<Vec<f32>, String> {
    let p = abs_path(&path)?;
    blocking(move || ffmpeg::extract_waveform(&p, samples_per_second).map_err(String::from)).await
}

/// Parse a `.cube` LUT for the preview (BOM-tolerant, lossy UTF-8):
/// `{ size, data: number[] (size³·3, red fastest), domainMin, domainMax, title? }`.
#[tauri::command]
pub async fn load_lut(path: String) -> Result<LutPayload, String> {
    let p = abs_path(&path)?;
    blocking(move || load_cube(&p).map(|l| LutPayload::from(&l))).await
}

/* ------------------------------------------------------------ pipeline */

#[tauri::command]
pub async fn run_pipeline(
    app: AppHandle,
    state: State<'_, AppState>,
    paths: Vec<String>,
    options: Option<PipelineOptions>,
) -> Result<String, String> {
    let sink: Arc<dyn EventSink> = Arc::new(app);
    let (jobs, registry) = (state.jobs.clone(), state.registry.clone());
    blocking(move || pipeline::run_pipeline_with(jobs, sink, paths, options.unwrap_or_default(), Some(registry))).await
}

#[tauri::command]
pub async fn cancel_pipeline(state: State<'_, AppState>, job_id: String) -> Result<(), String> {
    pipeline::cancel_pipeline(&state.jobs, &job_id)
}

/* ---------------------------------------------------- voice separation */

/// Split the audio of `paths` into vocals / background stems (Demucs, cached); returns the job id.
/// Events: `separate://progress|result|log|done`.
#[tauri::command]
pub async fn separate_audio(app: AppHandle, state: State<'_, AppState>, paths: Vec<String>) -> Result<String, String> {
    let sink: Arc<dyn EventSink> = Arc::new(app);
    let (jobs, registry) = (state.jobs.clone(), state.registry.clone());
    blocking(move || crate::separate::run_separate_with(jobs, sink, paths, Some(registry))).await
}

#[tauri::command]
pub async fn cancel_separate(state: State<'_, AppState>, job_id: String) -> Result<(), String> {
    crate::separate::cancel_separate(&state.jobs, &job_id)
}

/* -------------------------------------------------------------- export */

#[tauri::command]
pub async fn export_project(
    app: AppHandle,
    state: State<'_, AppState>,
    project: Project,
    out_path: String,
    preset: Option<String>,
    range: Option<ExportRange>,
) -> Result<String, String> {
    let sink: Arc<dyn EventSink> = Arc::new(app);
    let (jobs, registry) = (state.jobs.clone(), state.registry.clone());
    // Planning probes files / loads LUTs / bakes grades: keep it off the IPC thread.
    blocking(move || export::export_project_with(jobs, sink, project, out_path, preset.as_deref().unwrap_or("h264_mp4"), range, Some(registry))).await
}

#[tauri::command]
pub async fn cancel_export(state: State<'_, AppState>, job_id: String) -> Result<(), String> {
    export::cancel_export(&state.jobs, &job_id)
}

/* --------------------------------------------------------- clips folder */

/// Scan a clips folder (default `<repo>/clips`, created if missing) and return
/// its media as probed assets in edit order (`order` + `orderReason` set).
#[tauri::command]
pub async fn scan_clips_folder(state: State<'_, AppState>, path: Option<String>) -> Result<clips::ClipsFolderScan, String> {
    let registry = state.registry.clone();
    blocking(move || {
        let scan = clips::scan_clips_folder(path.as_deref())?;
        for a in &scan.assets {
            registry.register_asset(a);
        }
        Ok(scan)
    })
    .await
}

/* ------------------------------------------------------------- project */

#[tauri::command]
pub async fn save_project(path: String, project: Project) -> Result<Project, String> {
    let p = abs_path(&path)?;
    blocking(move || {
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let text = serde_json::to_string_pretty(&project).map_err(|e| e.to_string())?;
        // Write to a sibling temp file then rename so a crash never truncates the project.
        let tmp = p.with_extension("json.tmp");
        std::fs::write(&tmp, text).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &p)
            .or_else(|_| std::fs::copy(&tmp, &p).map(|_| ()).and_then(|_| std::fs::remove_file(&tmp)))
            .map_err(|e| e.to_string())?;
        Ok(project)
    })
    .await
}

/// Read a JSON document (lossy UTF-8, BOM tolerated).
fn read_json(p: &Path) -> Result<Value, String> {
    let bytes = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let text = String::from_utf8_lossy(&bytes);
    serde_json::from_str(text.trim_start_matches('\u{feff}')).map_err(|e| format!("{} is not valid JSON: {e}", p.display()))
}

#[tauri::command]
pub async fn load_project(state: State<'_, AppState>, path: String) -> Result<Project, String> {
    let p = abs_path(&path)?;
    let registry = state.registry.clone();
    blocking(move || {
        let project = serde_json::from_value::<Project>(read_json(&p)?).map_err(|e| format!("invalid project file: {e}"))?;
        registry.register_project(&project);
        Ok(project)
    })
    .await
}

/// `open_document` result: a saved project, or a pipeline analysis (whose timeline is the project).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenedDocument {
    /// `"project"` | `"analysis"`
    pub kind: String,
    pub project: Project,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub analysis: Option<AnalysisResult>,
}

/// Parse a saved project or a pipeline `AnalysisResult` (detected by its `timeline` +
/// `clips` keys) and register every media path in it with the media server.
pub fn open_document_at(registry: &MediaRegistry, p: &Path) -> Result<OpenedDocument, String> {
    let value = read_json(p)?;
    if value.get("timeline").is_some() && value.get("clips").is_some() {
        let analysis: AnalysisResult = serde_json::from_value(value).map_err(|e| format!("{} is not a valid analysis: {e}", p.display()))?;
        registry.register_analysis(&analysis);
        Ok(OpenedDocument { kind: "analysis".into(), project: analysis.timeline.clone(), analysis: Some(analysis) })
    } else {
        let project: Project = serde_json::from_value(value).map_err(|e| format!("{} is not a Cappycat project: {e}", p.display()))?;
        registry.register_project(&project);
        Ok(OpenedDocument { kind: "project".into(), project, analysis: None })
    }
}

#[tauri::command]
pub async fn open_document(state: State<'_, AppState>, path: String) -> Result<OpenedDocument, String> {
    let p = abs_path(&path)?;
    let registry = state.registry.clone();
    blocking(move || open_document_at(&registry, &p)).await
}

/* ----------------------------------------------------------- keyframes */

/// Interpolate a `Keyframed<number | [n,n] | [n,n,n] | [n,n,n,n]>` at `time_ms`.
pub fn evaluate_keyframes_value(keyframed: &Keyframed<Value>, time_ms: f64) -> Result<Value, String> {
    let shape = |v: &Value| -> Option<usize> {
        match v {
            Value::Number(_) => Some(0),
            Value::Array(a) if (2..=4).contains(&a.len()) && a.iter().all(|x| x.is_number()) => Some(a.len()),
            _ => None,
        }
    };
    let n = shape(&keyframed.static_value)
        .or_else(|| keyframed.keyframes.first().and_then(|k| shape(&k.value)))
        .ok_or("keyframed value must be a number or an array of 2..4 numbers")?;
    if keyframed.keyframes.iter().any(|k| shape(&k.value) != Some(n)) {
        return Err("all keyframe values must have the same shape as `static`".into());
    }
    let json = serde_json::to_value(keyframed).map_err(|e| e.to_string())?;
    macro_rules! eval_as {
        ($t:ty) => {{
            let k: Keyframed<$t> = serde_json::from_value(json).map_err(|e| e.to_string())?;
            serde_json::to_value(k.evaluate(time_ms)).map_err(|e| e.to_string())
        }};
    }
    match n {
        0 => eval_as!(f64),
        2 => eval_as!([f64; 2]),
        3 => eval_as!([f64; 3]),
        4 => eval_as!([f64; 4]),
        _ => unreachable!(),
    }
}

#[tauri::command]
pub async fn evaluate_keyframes(keyframed: Keyframed<Value>, time_ms: f64) -> Result<Value, String> {
    evaluate_keyframes_value(&keyframed, time_ms)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeedLutResponse {
    pub source_duration_ms: f64,
    pub output_duration_ms: f64,
    /// Constant speed with the same output duration (what the exporter uses).
    pub effective_speed: f64,
    /// `samples + 1` entries: source offset (ms) at output offset `i / samples * outputDurationMs`.
    pub source_ms: Vec<f64>,
    /// `samples + 1` entries: instantaneous speed at the same output offsets.
    pub speed: Vec<f64>,
}

pub fn speed_lut_response(curve: &SpeedCurve, duration_ms: f64, samples: usize) -> Result<SpeedLutResponse, String> {
    if !(duration_ms.is_finite() && duration_ms >= 0.0) {
        return Err("durationMs must be a non-negative number".into());
    }
    let samples = samples.clamp(1, 100_000);
    let lut = curve.lut(duration_ms);
    let source_ms = lut.output_to_source_table(samples);
    let speed = source_ms.iter().map(|s| lut.speed_at_source(*s)).collect();
    Ok(SpeedLutResponse {
        source_duration_ms: duration_ms,
        output_duration_ms: lut.output_duration_ms(),
        effective_speed: lut.effective_speed(),
        source_ms,
        speed,
    })
}

#[tauri::command]
pub async fn speed_curve_lut(curve: SpeedCurve, duration_ms: f64, samples: Option<usize>) -> Result<SpeedLutResponse, String> {
    speed_lut_response(&curve, duration_ms, samples.unwrap_or(256))
}

/* ---------------------------------------------------------------- info */

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppPaths {
    pub cache_dir: String,
    pub thumbs_dir: String,
    pub analysis_dir: String,
    pub ffmpeg: Option<String>,
    pub ffprobe: Option<String>,
    pub python: Option<String>,
    pub pipeline_dir: Option<String>,
    /// `<repo>/clips` — default folder for `scan_clips_folder`
    pub clips_dir: Option<String>,
    /// `http://127.0.0.1:<port>/t/<token>`
    pub media_server_url: String,
    /// `%LOCALAPPDATA%\Cappycat\logs`
    pub logs_dir: String,
    /// `"repo"` (running from a checkout) or `"installed"`
    pub layout: String,
    /// app data home: `%LOCALAPPDATA%\Cappycat` (python env, models, ffmpeg, cache, logs, autosave)
    pub home_dir: String,
    /// `Documents\Cappycat`
    pub documents_dir: String,
    pub models_dir: String,
    pub projects_dir: String,
    pub exports_dir: String,
    pub characters_dir: String,
    pub presets_dir: String,
    pub autosave_dir: String,
}

pub fn collect_app_paths(media_server_url: String) -> AppPaths {
    let bins = ffmpeg::find_binaries().ok();
    let pipeline_dir = pipeline::find_pipeline_dir();
    let python = crate::jobs::find_python(pipeline_dir.as_deref());
    let s = |p: &Path| p.to_string_lossy().into_owned();
    let r = crate::paths::resolver();
    AppPaths {
        layout: if r.is_installed() { "installed".into() } else { "repo".into() },
        home_dir: s(&r.home()),
        documents_dir: s(&r.documents_home()),
        models_dir: s(&r.models_dir()),
        projects_dir: s(&r.projects_dir()),
        exports_dir: s(&r.exports_dir()),
        characters_dir: s(&r.characters_dir()),
        presets_dir: s(&r.presets_dir()),
        autosave_dir: s(&r.autosave_dir()),
        cache_dir: s(&ffmpeg::cache_dir()),
        thumbs_dir: s(&ffmpeg::thumbs_dir()),
        analysis_dir: s(&ffmpeg::cache_dir().join("analysis")),
        ffmpeg: bins.map(|b| s(&b.ffmpeg)),
        ffprobe: bins.map(|b| s(&b.ffprobe)),
        python: python.map(|p| s(&p)),
        pipeline_dir: pipeline_dir.map(|p| s(&p)),
        clips_dir: clips::default_clips_dir().map(|p| s(&p)),
        media_server_url,
        logs_dir: s(&crate::logging::logs_dir()),
    }
}

#[tauri::command]
pub async fn app_paths(state: State<'_, AppState>) -> Result<AppPaths, String> {
    let url = media_url(&state);
    blocking(move || Ok(collect_app_paths(url))).await
}

/* ------------------------------------------------------------ AI setup */

/// Install the AI tooling (ffmpeg, uv + Python env + torch + packages, models, verify); returns the
/// job id. `components`: `ffmpeg` | `python` | `models` (or single steps); omitted = what is
/// missing. Events: `setup://progress|log|done`.
#[tauri::command]
pub async fn setup_ai(app: AppHandle, state: State<'_, AppState>, components: Option<Vec<String>>) -> Result<String, String> {
    let sink: Arc<dyn EventSink> = Arc::new(app);
    let jobs = state.jobs.clone();
    blocking(move || crate::setup::setup_ai(jobs, sink, components)).await
}

#[tauri::command]
pub async fn cancel_setup(state: State<'_, AppState>, job_id: String) -> Result<(), String> {
    crate::setup::cancel_setup(&state.jobs, &job_id)
}

/// `{ ffmpeg, python, models, gpu: string | null, installed }`
#[tauri::command]
pub async fn setup_status() -> Result<crate::setup::SetupStatus, String> {
    blocking(|| Ok(crate::setup::setup_status())).await
}

/// `[{ step, url?, sizeBytes?, note }]`: what `setup_ai` would download (for the wizard to show).
#[tauri::command]
pub async fn setup_plan(components: Option<Vec<String>>) -> Result<Vec<crate::setup::PlanItem>, String> {
    blocking(move || crate::setup::setup_plan(components)).await
}

/// The universal adjust preset (`<repo>/presets/universal-adjust.json` in a checkout,
/// `Documents\Cappycat\Presets\universal-adjust.json` when installed), created with the default
/// house values on first use.
#[tauri::command]
pub async fn load_universal_adjust() -> Result<crate::presets::UniversalPresetInfo, String> {
    tauri::async_runtime::spawn_blocking(crate::presets::load).await.map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn save_universal_adjust(preset: crate::presets::UniversalPresetFile) -> Result<crate::presets::UniversalPresetInfo, String> {
    tauri::async_runtime::spawn_blocking(move || crate::presets::save(&preset)).await.map_err(|e| e.to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn evaluates_scalar_and_vector_keyframes() {
        let k: Keyframed<Value> = serde_json::from_value(json!({
            "static": 0,
            "keyframes": [
                { "timeMs": 0, "value": 0, "easing": "linear" },
                { "timeMs": 1000, "value": 10, "easing": "linear" }
            ]
        }))
        .unwrap();
        assert_eq!(evaluate_keyframes_value(&k, 500.0).unwrap(), json!(5.0));

        let k2: Keyframed<Value> = serde_json::from_value(json!({
            "static": [0, 0],
            "keyframes": [
                { "timeMs": 0, "value": [0, 0], "easing": "linear" },
                { "timeMs": 100, "value": [10, -10], "easing": "linear" }
            ]
        }))
        .unwrap();
        assert_eq!(evaluate_keyframes_value(&k2, 50.0).unwrap(), json!([5.0, -5.0]));

        let k4: Keyframed<Value> = serde_json::from_value(json!({ "static": [0.1, 0.2, 0.3, 0.4], "keyframes": [] })).unwrap();
        assert_eq!(evaluate_keyframes_value(&k4, 5.0).unwrap(), json!([0.1, 0.2, 0.3, 0.4]));

        let bad: Keyframed<Value> = serde_json::from_value(json!({ "static": "nope", "keyframes": [] })).unwrap();
        assert!(evaluate_keyframes_value(&bad, 0.0).is_err());
        let mixed: Keyframed<Value> = serde_json::from_value(json!({ "static": 1, "keyframes": [ { "timeMs": 0, "value": [1, 2] } ] })).unwrap();
        assert!(evaluate_keyframes_value(&mixed, 0.0).is_err());
    }

    #[test]
    fn speed_lut_response_shape() {
        let curve: SpeedCurve = serde_json::from_value(json!({ "preset": "hero_time", "points": [], "opticalFlow": true })).unwrap();
        let r = speed_lut_response(&curve, 4000.0, 10).unwrap();
        assert_eq!(r.source_ms.len(), 11);
        assert_eq!(r.speed.len(), 11);
        assert_eq!(r.source_ms[0], 0.0);
        assert!((r.source_ms[10] - 4000.0).abs() < 1e-9);
        assert!(r.output_duration_ms > 4000.0);
        assert!(r.effective_speed < 1.0);
        let v = serde_json::to_value(&r).unwrap();
        assert!(v.get("outputDurationMs").is_some() && v.get("sourceMs").is_some());
        assert!(speed_lut_response(&curve, -1.0, 10).is_err());
    }

    #[test]
    fn open_document_detects_projects_and_analyses_and_registers_media() {
        let dir = crate::ffmpeg::cache_dir().join("test").join("open_document");
        std::fs::create_dir_all(&dir).unwrap();
        let project = json!({ "version": 1, "id": "p", "name": "n", "fps": 24, "width": 1280, "height": 720,
            "assets": [ { "id": "a", "path": "C:/media/one.mp4", "name": "one.mp4", "kind": "video",
                          "stems": { "vocals": "C:/stems/v.wav", "background": "C:/stems/b.wav" } },
                        { "id": "l", "path": "C:/looks/warm.cube", "name": "warm.cube", "kind": "lut" } ],
            "tracks": [], "beatMarkers": [] });
        let pp = dir.join("doc.cappycat.json");
        // with a UTF-8 BOM, like some editors write
        std::fs::write(&pp, format!("\u{feff}{project}")).unwrap();
        let reg = MediaRegistry::new();
        let d = open_document_at(&reg, &pp).unwrap();
        assert_eq!(d.kind, "project");
        assert!(d.analysis.is_none());
        for p in ["C:/media/one.mp4", "C:/stems/v.wav", "C:/stems/b.wav", "C:/looks/warm.cube"] {
            assert!(reg.contains(Path::new(p)), "{p} registered");
        }
        let v = serde_json::to_value(&d).unwrap();
        assert!(v.get("analysis").is_none() && v["project"]["name"] == "n");

        let analysis = json!({ "version": 1, "generatedAt": "x",
            "clips": [ { "path": "C:/media/two.mp4", "asset": { "id": "b", "path": "C:/media/two.mp4", "name": "two.mp4", "kind": "video" } } ],
            "timeline": project });
        let ap = dir.join("run.analysis.json");
        std::fs::write(&ap, analysis.to_string()).unwrap();
        let reg = MediaRegistry::new();
        let d = open_document_at(&reg, &ap).unwrap();
        assert_eq!(d.kind, "analysis");
        assert_eq!(d.project.name, "n", "project = the analysis timeline");
        assert_eq!(d.analysis.as_ref().unwrap().clips.len(), 1);
        assert!(reg.contains(Path::new("C:/media/two.mp4")) && reg.contains(Path::new("C:/media/one.mp4")));
        std::fs::write(&ap, "not json").unwrap();
        assert!(open_document_at(&reg, &ap).unwrap_err().contains("not valid JSON"));
    }

    #[test]
    fn app_paths_report_tooling() {
        let p = collect_app_paths("http://127.0.0.1:4321/t/abc".into());
        assert_eq!(p.media_server_url, "http://127.0.0.1:4321/t/abc");
        assert!(p.cache_dir.to_lowercase().contains("cappycat"));
        assert!(p.logs_dir.to_lowercase().contains("logs"));
        if p.ffmpeg.is_none() {
            eprintln!("note: ffmpeg not found on this machine");
        }
    }
}
