//! First-run AI setup (`docs/FEATURES_V2.md` §9): `setup_ai { components? }` → job id, with
//! `setup://progress { jobId, step, pct, message, index, total }`, `setup://log { jobId, level,
//! message }` and `setup://done { jobId, ok, error, report? }`; `cancel_setup { jobId }`;
//! `setup_status` → `{ ffmpeg, python, models, gpu, installed }`; `setup_plan { components? }` →
//! `[{ step, url?, sizeBytes?, note }]` (what would be downloaded, for the wizard to confirm).
//!
//! Steps, in order (each one idempotent; one that finds its result already present is skipped
//! with a message):
//!
//! | step | what |
//! |---|---|
//! | `ffmpeg` | ffmpeg on PATH / WinGet / `<home>\ffmpeg\bin` is used; otherwise the official static build zip (`CAPPYCAT_FFMPEG_URL`, default BtbN `ffmpeg-master-latest-win64-gpl.zip`) is downloaded (URL, size and SHA-256 logged) and `ffmpeg.exe` / `ffprobe.exe` are extracted to `<home>\ffmpeg\bin` |
//! | `uv` | `uv` on PATH or `<home>\bin\uv.exe`; otherwise `uv-x86_64-pc-windows-msvc.zip` from astral-sh/uv's latest release (`CAPPYCAT_UV_URL`) → `<home>\bin` |
//! | `python` | `uv venv --python 3.12 <home>\python` (uv's managed CPython and cache live in `<home>\uv`) |
//! | `torch` | `uv pip install torch==2.14.0 torchvision==0.29.0` from the cu130 index, or the CPU index when `nvidia-smi` finds no NVIDIA GPU (`CAPPYCAT_TORCH_INDEX` overrides) |
//! | `packages` | `uv pip install -r <pipeline>\requirements-installer.txt`, then `uv pip install --no-deps -r requirements-installer-nodeps.txt` |
//! | `models` | `python -m cappycat_pipeline download-models` (progress relayed; models go to `CAPPYCAT_MODELS_DIR`) |
//! | `verify` | `python -m cappycat_pipeline doctor` (its JSON report is `setup://done.report`) |
//!
//! `components` may name steps or the wizard's groups: `ffmpeg`, `python` (= uv + python + torch +
//! packages), `models`; `verify` runs after any Python-related step. Omitted: every missing
//! component. Every child process is in the app's kill-on-close job object and is stopped (with
//! its process tree) by `cancel_setup`; a cancelled download deletes its partial file.

use crate::jobs::{EventSink, JobManager, TaskCtl};
use crate::paths::Resolver;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

pub const DEFAULT_FFMPEG_URL: &str = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip";
pub const DEFAULT_UV_URL: &str = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip";
pub const TORCH_CUDA_INDEX: &str = "https://download.pytorch.org/whl/cu130";
pub const TORCH_CPU_INDEX: &str = "https://download.pytorch.org/whl/cpu";
pub const TORCH_PACKAGES: &[&str] = &["torch==2.14.0", "torchvision==0.29.0"];
pub const PYTHON_VERSION: &str = "3.12";
/// Written into the models folder after a successful `download-models`.
pub const MODELS_MARKER: &str = ".cappycat-models-ok";

/// One setup step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Step {
    Ffmpeg,
    Uv,
    Python,
    Torch,
    Packages,
    Models,
    Verify,
}

impl Step {
    pub const ALL: [Step; 7] = [Step::Ffmpeg, Step::Uv, Step::Python, Step::Torch, Step::Packages, Step::Models, Step::Verify];

    pub fn name(self) -> &'static str {
        match self {
            Step::Ffmpeg => "ffmpeg",
            Step::Uv => "uv",
            Step::Python => "python",
            Step::Torch => "torch",
            Step::Packages => "packages",
            Step::Models => "models",
            Step::Verify => "verify",
        }
    }
}

/// Where setup installs things and where it downloads from (env overrides for tests).
#[derive(Debug, Clone)]
pub struct SetupConfig {
    pub resolver: Resolver,
    pub ffmpeg_url: String,
    pub uv_url: String,
    pub torch_index: Option<String>,
    /// accept an ffmpeg / uv found elsewhere (PATH, WinGet); tests switch it off
    pub detect_existing: bool,
}

impl SetupConfig {
    pub fn from_env() -> Self {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        Self {
            resolver: Resolver::current(),
            ffmpeg_url: var("CAPPYCAT_FFMPEG_URL").unwrap_or_else(|| DEFAULT_FFMPEG_URL.into()),
            uv_url: var("CAPPYCAT_UV_URL").unwrap_or_else(|| DEFAULT_UV_URL.into()),
            torch_index: var("CAPPYCAT_TORCH_INDEX"),
            detect_existing: true,
        }
    }

    fn home(&self) -> PathBuf {
        self.resolver.home()
    }

    fn own_ffmpeg_bin(&self) -> PathBuf {
        self.resolver.ffmpeg_dir().join("bin")
    }

    fn own_uv(&self) -> PathBuf {
        self.resolver.bin_dir().join(exe("uv"))
    }
}

fn exe(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(exe(name))).find(|p| p.is_file())
}

/* ------------------------------------------------------------------ status */

/// `setup_status` payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetupStatus {
    pub ffmpeg: bool,
    pub python: bool,
    pub models: bool,
    /// the NVIDIA GPU's name (`nvidia-smi`), `null` without one
    pub gpu: Option<String>,
    /// running from the installed layout (not a checkout)
    pub installed: bool,
}

/// The first NVIDIA GPU's name (`nvidia-smi --query-gpu=name`), cached for the process.
pub fn nvidia_gpu() -> Option<String> {
    static GPU: OnceLock<Option<String>> = OnceLock::new();
    GPU.get_or_init(|| {
        let mut cmd = crate::ffmpeg::command(Path::new("nvidia-smi"));
        cmd.args(["--query-gpu=name", "--format=csv,noheader"]);
        let out = crate::procs::output_timeout(&mut cmd, Duration::from_secs(8)).ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8_lossy(&out.stdout).lines().map(str::trim).find(|l| !l.is_empty()).map(String::from)
    })
    .clone()
}

/// Is torch installed in the environment of `python` (a venv's or a system interpreter)?
pub fn has_torch(python: &Path) -> bool {
    let Some(dir) = python.parent() else { return false };
    // <venv>\Scripts\python.exe → <venv>\Lib\site-packages; C:\Python312\python.exe → C:\Python312\Lib\site-packages
    let roots = [dir.parent().map(Path::to_path_buf), Some(dir.to_path_buf())];
    roots.into_iter().flatten().any(|r| {
        r.join("Lib").join("site-packages").join("torch").join("__init__.py").is_file()
            || std::fs::read_dir(r.join("lib")).map(|rd| rd.flatten().any(|e| e.path().join("site-packages").join("torch").join("__init__.py").is_file())).unwrap_or(false)
    })
}

/// Are the pipeline's models present in `dir` (the marker, or the key files)?
pub fn models_present(dir: &Path) -> bool {
    if dir.join(MODELS_MARKER).is_file() {
        return true;
    }
    let transnet = dir.join("transnetv2.onnx").is_file() || dir.join("transnetv2.pt").is_file();
    let yolo = dir.join("yolov8s-worldv2.pt").is_file();
    let raft = std::fs::read_dir(dir.join("torch").join("hub").join("checkpoints"))
        .map(|rd| rd.flatten().any(|e| e.file_name().to_string_lossy().starts_with("raft_small")))
        .unwrap_or(false);
    transnet && yolo && raft
}

/// The Python the pipeline runs with (managed env or dev venv / PATH).
fn pipeline_python(r: &Resolver) -> Option<PathBuf> {
    let managed = r.managed_python();
    if managed.is_file() {
        return Some(managed);
    }
    crate::jobs::find_python(r.pipeline_dir().as_deref())
}

pub fn status_with(r: &Resolver) -> SetupStatus {
    SetupStatus {
        ffmpeg: crate::ffmpeg::find_binaries().is_ok() || (r.ffmpeg_dir().join("bin").join(exe("ffmpeg")).is_file() && r.ffmpeg_dir().join("bin").join(exe("ffprobe")).is_file()),
        python: pipeline_python(r).map(|p| has_torch(&p)).unwrap_or(false),
        models: models_present(&r.models_dir()),
        gpu: nvidia_gpu(),
        installed: r.is_installed(),
    }
}

pub fn setup_status() -> SetupStatus {
    status_with(&Resolver::current())
}

/* ------------------------------------------------------------------ steps */

/// Expand `components` (step names or the groups `ffmpeg` / `python` / `models`) into steps in
/// execution order; `None` = every missing component (per `status`).
pub fn expand_components(components: Option<&[String]>, status: &SetupStatus) -> Result<Vec<Step>, String> {
    let mut steps: Vec<Step> = Vec::new();
    match components {
        None => {
            if !status.ffmpeg {
                steps.push(Step::Ffmpeg);
            }
            if !status.python {
                steps.extend([Step::Uv, Step::Python, Step::Torch, Step::Packages]);
            }
            if !status.models {
                steps.push(Step::Models);
            }
        }
        Some(list) => {
            for c in list {
                match c.trim().to_ascii_lowercase().as_str() {
                    "ffmpeg" => steps.push(Step::Ffmpeg),
                    // the wizard's "python" = the whole environment
                    "python" => steps.extend([Step::Uv, Step::Python, Step::Torch, Step::Packages]),
                    "uv" => steps.push(Step::Uv),
                    "venv" => steps.push(Step::Python),
                    "torch" => steps.push(Step::Torch),
                    "packages" => steps.push(Step::Packages),
                    "models" => steps.push(Step::Models),
                    "verify" => steps.push(Step::Verify),
                    "" => {}
                    other => return Err(format!("unknown setup component '{other}' (ffmpeg, python, models, uv, torch, packages, verify)")),
                }
            }
        }
    }
    if steps.iter().any(|s| matches!(s, Step::Python | Step::Torch | Step::Packages | Step::Models)) {
        steps.push(Step::Verify);
    }
    steps.sort();
    steps.dedup();
    Ok(steps)
}

/// `setup_plan` entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanItem {
    pub step: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    pub note: String,
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(20)).timeout_read(Duration::from_secs(60)).redirects(8).build()
}

/// Size of a download (HEAD, following redirects), when the server says.
pub fn remote_size(url: &str) -> Option<u64> {
    let r = agent().head(url).timeout(Duration::from_secs(15)).call().ok()?;
    r.header("content-length").and_then(|v| v.parse().ok())
}

fn torch_index(cfg: &SetupConfig) -> (String, bool) {
    match &cfg.torch_index {
        Some(i) => (i.clone(), !i.contains("/cpu")),
        None => match nvidia_gpu() {
            Some(_) => (TORCH_CUDA_INDEX.into(), true),
            None => (TORCH_CPU_INDEX.into(), false),
        },
    }
}

/// What `setup_ai` would do for `components` (sizes from HEAD requests).
pub fn plan_with(cfg: &SetupConfig, components: Option<&[String]>) -> Result<Vec<PlanItem>, String> {
    let status = status_with(&cfg.resolver);
    let steps = expand_components(components, &status)?;
    let mut out = Vec::new();
    for s in steps {
        let item = |url: Option<String>, size: Option<u64>, note: String| PlanItem { step: s.name().into(), url, size_bytes: size, note };
        out.push(match s {
            Step::Ffmpeg => match existing_ffmpeg(cfg) {
                Some(p) => item(None, None, format!("already available: {}", p.display())),
                None => item(
                    Some(cfg.ffmpeg_url.clone()),
                    remote_size(&cfg.ffmpeg_url),
                    format!("ffmpeg static build (GPL, BtbN) extracted to {}", cfg.own_ffmpeg_bin().display()),
                ),
            },
            Step::Uv => match existing_uv(cfg) {
                Some(p) => item(None, None, format!("already available: {}", p.display())),
                None => item(Some(cfg.uv_url.clone()), remote_size(&cfg.uv_url), format!("uv (Astral) extracted to {}", cfg.resolver.bin_dir().display())),
            },
            Step::Python => {
                let py = cfg.resolver.managed_python();
                if py.is_file() {
                    item(None, None, format!("already available: {}", py.display()))
                } else {
                    item(None, None, format!("uv downloads CPython {PYTHON_VERSION} (~30 MB) and creates {}", cfg.resolver.python_env_dir().display()))
                }
            }
            Step::Torch => {
                let (index, cuda) = torch_index(cfg);
                let note = if cuda {
                    format!("{} with CUDA 13.0 (~2.9 GB) for {}", TORCH_PACKAGES.join(" "), nvidia_gpu().unwrap_or_else(|| "the NVIDIA GPU".into()))
                } else {
                    format!("{} CPU build (~300 MB; no NVIDIA GPU found)", TORCH_PACKAGES.join(" "))
                };
                item(Some(index), None, note)
            }
            Step::Packages => item(
                Some("https://pypi.org/simple".into()),
                None,
                "the pipeline's pinned packages (requirements-installer.txt + requirements-installer-nodeps.txt, ~1 GB)".into(),
            ),
            Step::Models => item(None, None, format!("the AI models (~1.5 GB from Hugging Face / GitHub) into {}", cfg.resolver.models_dir().display())),
            Step::Verify => item(None, None, "python -m cappycat_pipeline doctor".into()),
        });
    }
    Ok(out)
}

pub fn setup_plan(components: Option<Vec<String>>) -> Result<Vec<PlanItem>, String> {
    plan_with(&SetupConfig::from_env(), components.as_deref())
}

fn existing_ffmpeg(cfg: &SetupConfig) -> Option<PathBuf> {
    let own = cfg.own_ffmpeg_bin();
    if own.join(exe("ffmpeg")).is_file() && own.join(exe("ffprobe")).is_file() {
        return Some(own.join(exe("ffmpeg")));
    }
    if cfg.detect_existing {
        return crate::ffmpeg::find_binaries().ok().map(|b| b.ffmpeg.clone());
    }
    None
}

fn existing_uv(cfg: &SetupConfig) -> Option<PathBuf> {
    let own = cfg.own_uv();
    if own.is_file() {
        return Some(own);
    }
    if cfg.detect_existing {
        return find_on_path("uv");
    }
    None
}

/* ------------------------------------------------------------ downloads */

/// Download `url` to `dest` (via `<dest>.partial`), reporting `(bytes, total)`. Returns the size
/// and the SHA-256 (hex). Cancelling deletes the partial file.
pub fn download(url: &str, dest: &Path, ctl: &TaskCtl, progress: &mut dyn FnMut(u64, Option<u64>)) -> Result<(u64, String), String> {
    if let Some(d) = dest.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("cannot create {}: {e}", d.display()))?;
    }
    let resp = agent().get(url).call().map_err(|e| match e {
        ureq::Error::Status(code, r) => format!("download of {url} failed: HTTP {code} {}", r.status_text()),
        other => format!("download of {url} failed: {other}"),
    })?;
    let total: Option<u64> = resp.header("content-length").and_then(|v| v.parse().ok());
    let partial = dest.with_extension("partial");
    let result = (|| -> Result<(u64, String), String> {
        let mut file = std::io::BufWriter::new(std::fs::File::create(&partial).map_err(|e| format!("cannot write {}: {e}", partial.display()))?);
        let mut reader = resp.into_reader();
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 256 * 1024];
        let mut done = 0u64;
        let mut last = Instant::now();
        loop {
            if ctl.is_cancelled() {
                return Err("cancelled".into());
            }
            let n = match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(format!("download of {url} interrupted: {e}")),
            };
            file.write_all(&buf[..n]).map_err(|e| format!("cannot write {}: {e}", partial.display()))?;
            hasher.update(&buf[..n]);
            done += n as u64;
            if last.elapsed() >= Duration::from_millis(200) {
                last = Instant::now();
                progress(done, total);
            }
        }
        file.flush().map_err(|e| e.to_string())?;
        if let Some(t) = total {
            if done != t {
                return Err(format!("download of {url} ended after {done} of {t} bytes"));
            }
        }
        progress(done, total);
        Ok((done, hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()))
    })();
    match result {
        Ok(r) => {
            std::fs::rename(&partial, dest).map_err(|e| format!("cannot move {} into place: {e}", dest.display()))?;
            Ok(r)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&partial);
            Err(e)
        }
    }
}

/// Extract the entries of `zip` whose file name (any folder) is one of `names`
/// (case-insensitive) into `dest`. Returns the extracted paths.
pub fn extract_named(zip: &Path, dest: &Path, names: &[&str]) -> Result<Vec<PathBuf>, String> {
    let f = std::fs::File::open(zip).map_err(|e| format!("cannot open {}: {e}", zip.display()))?;
    let mut a = zip::ZipArchive::new(f).map_err(|e| format!("{} is not a valid zip: {e}", zip.display()))?;
    std::fs::create_dir_all(dest).map_err(|e| format!("cannot create {}: {e}", dest.display()))?;
    let mut out = Vec::new();
    for i in 0..a.len() {
        let mut entry = a.by_index(i).map_err(|e| e.to_string())?;
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().replace('\\', "/");
        let file = name.rsplit('/').next().unwrap_or("").to_string();
        if !names.iter().any(|n| n.eq_ignore_ascii_case(&file)) {
            continue;
        }
        let target = dest.join(&file);
        let tmp = target.with_extension("extracting");
        {
            let mut w = std::fs::File::create(&tmp).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
            std::io::copy(&mut entry, &mut w).map_err(|e| format!("cannot extract {name}: {e}"))?;
        }
        let _ = std::fs::remove_file(&target);
        std::fs::rename(&tmp, &target).map_err(|e| e.to_string())?;
        out.push(target);
    }
    Ok(out)
}

/* ------------------------------------------------------------------- runner */

/// Event plumbing of one setup job.
pub struct SetupCtx<'a> {
    pub job_id: &'a str,
    pub sink: &'a dyn EventSink,
    pub ctl: &'a TaskCtl,
    step: Step,
    index: usize,
    total: usize,
}

impl SetupCtx<'_> {
    fn progress(&self, pct: f64, message: impl Into<String>) {
        self.sink.emit(
            "setup://progress",
            json!({ "jobId": self.job_id, "step": self.step.name(), "pct": pct.clamp(0.0, 1.0), "message": message.into(), "index": self.index, "total": self.total }),
        );
    }

    fn log(&self, level: &str, message: impl Into<String>) {
        let message = message.into();
        tracing::info!("setup {}: [{level}] {message}", self.job_id);
        self.sink.emit("setup://log", json!({ "jobId": self.job_id, "level": level, "step": self.step.name(), "message": message }));
    }
}

fn mb(b: u64) -> String {
    format!("{:.1} MB", b as f64 / 1_048_576.0)
}

fn download_and_extract(ctx: &SetupCtx, url: &str, zip_name: &str, dest: &Path, names: &[&str], required: &[&str], home: &Path) -> Result<(), String> {
    let size = remote_size(url);
    ctx.log("info", format!("downloading {url}{}", size.map(|s| format!(" ({})", mb(s))).unwrap_or_default()));
    let zip = home.join("downloads").join(zip_name);
    let t0 = Instant::now();
    let (bytes, sha) = download(url, &zip, ctx.ctl, &mut |done, total| {
        let pct = total.map(|t| done as f64 / t.max(1) as f64 * 0.9).unwrap_or(0.0);
        ctx.progress(pct, format!("{} of {}", mb(done), total.map(mb).unwrap_or_else(|| "?".into())));
    })?;
    ctx.log("info", format!("downloaded {} in {:.1}s, sha256 {sha}", mb(bytes), t0.elapsed().as_secs_f64()));
    ctx.progress(0.92, "extracting");
    let got = extract_named(&zip, dest, names);
    let _ = std::fs::remove_file(&zip);
    let got = got?;
    for r in required {
        if !got.iter().any(|p| p.file_name().map(|f| f.to_string_lossy().eq_ignore_ascii_case(r)).unwrap_or(false)) {
            return Err(format!("{url} does not contain {r}"));
        }
    }
    ctx.log("info", format!("extracted {} to {}", got.iter().filter_map(|p| p.file_name().map(|f| f.to_string_lossy().into_owned())).collect::<Vec<_>>().join(", "), dest.display()));
    Ok(())
}

/// Run a child to completion, relaying its stderr / stdout lines as log messages and an
/// estimated progress; `on_json` sees stdout lines that are JSON (pipeline events).
fn run_child(ctx: &SetupCtx, mut cmd: Command, what: &str, python: bool, on_json: &mut dyn FnMut(&Value)) -> Result<String, String> {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    cmd.stdin(if python { Stdio::piped() } else { Stdio::null() });
    crate::procs::hide_window(&mut cmd);
    let mut child = crate::procs::spawn(&mut cmd).map_err(|e| format!("cannot start {what}: {e}"))?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let stderr = child.stderr.take().ok_or("no stderr")?;
    let stdin = child.stdin.take();
    let managed = ctx.ctl.register_python(child, stdin);
    let (tx, rx) = std::sync::mpsc::channel::<(bool, String)>();
    let tx2 = tx.clone();
    let t_err = std::thread::spawn(move || {
        for l in BufReader::new(stderr).lines().map_while(Result::ok) {
            let _ = tx2.send((false, l));
        }
    });
    let t_out = std::thread::spawn(move || {
        for l in BufReader::new(stdout).lines().map_while(Result::ok) {
            let _ = tx.send((true, l));
        }
    });
    let mut out_text = String::new();
    let mut lines = 0usize;
    let mut tail: Vec<String> = Vec::new();
    for (is_out, line) in rx {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if is_out {
            if let Ok(v) = serde_json::from_str::<Value>(t) {
                on_json(&v);
                out_text.push_str(t);
                out_text.push('\n');
                continue;
            }
            out_text.push_str(t);
            out_text.push('\n');
        }
        lines += 1;
        ctx.log("info", t.to_string());
        ctx.progress(1.0 - (-(lines as f64) / 25.0).exp() * 1.0 - 0.05, t.chars().take(160).collect::<String>());
        tail.push(t.to_string());
        if tail.len() > 6 {
            tail.remove(0);
        }
    }
    let _ = t_err.join();
    let _ = t_out.join();
    let status = crate::procs::wait_child(&managed.child).map_err(|e| e.to_string())?;
    if let Some(tree) = &managed.tree {
        tree.kill();
    }
    if ctx.ctl.is_cancelled() {
        return Err("cancelled".into());
    }
    if !status.success() {
        return Err(format!("{what} failed ({:?}): {}", status.code(), tail.join(" | ")));
    }
    Ok(out_text)
}

fn uv_command(uv: &Path, cfg: &SetupConfig) -> Command {
    let mut c = Command::new(uv);
    let home = cfg.home();
    c.env("UV_PYTHON_INSTALL_DIR", home.join("uv").join("python"))
        .env("UV_CACHE_DIR", home.join("uv").join("cache"))
        .env("UV_NO_PROGRESS", "1")
        .env("NO_COLOR", "1");
    c
}

fn run_step(cfg: &SetupConfig, ctx: &SetupCtx, report: &mut Value) -> Result<(), String> {
    let r = &cfg.resolver;
    let home = cfg.home();
    match ctx.step {
        Step::Ffmpeg => {
            if let Some(p) = existing_ffmpeg(cfg) {
                ctx.progress(1.0, format!("ffmpeg already available: {}", p.display()));
                return Ok(());
            }
            let names = [exe("ffmpeg"), exe("ffprobe")];
            let names: Vec<&str> = names.iter().map(String::as_str).collect();
            download_and_extract(ctx, &cfg.ffmpeg_url, "ffmpeg.zip", &cfg.own_ffmpeg_bin(), &names, &names, &home)?;
        }
        Step::Uv => {
            if let Some(p) = existing_uv(cfg) {
                ctx.progress(1.0, format!("uv already available: {}", p.display()));
                return Ok(());
            }
            let (uv, uvx) = (exe("uv"), exe("uvx"));
            download_and_extract(ctx, &cfg.uv_url, "uv.zip", &r.bin_dir(), &[&uv, &uvx], &[&uv], &home)?;
        }
        Step::Python => {
            let py = r.managed_python();
            if py.is_file() {
                ctx.progress(1.0, format!("Python environment already exists: {}", py.display()));
                return Ok(());
            }
            let uv = existing_uv(cfg).ok_or("uv is not installed (run the uv step first)")?;
            let mut c = uv_command(&uv, cfg);
            c.args(["venv", "--python", PYTHON_VERSION]).arg(r.python_env_dir());
            ctx.progress(0.05, format!("uv venv --python {PYTHON_VERSION} {}", r.python_env_dir().display()));
            run_child(ctx, c, "uv venv", true, &mut |_| {})?;
            if !py.is_file() {
                return Err(format!("uv venv did not create {}", py.display()));
            }
        }
        Step::Torch => {
            let py = r.managed_python();
            if !py.is_file() {
                return Err("the Python environment does not exist (run the python step first)".into());
            }
            if has_torch(&py) {
                ctx.progress(1.0, "torch is already installed");
                return Ok(());
            }
            let uv = existing_uv(cfg).ok_or("uv is not installed (run the uv step first)")?;
            let (index, cuda) = torch_index(cfg);
            ctx.log("info", format!("installing {} from {index} ({})", TORCH_PACKAGES.join(" "), if cuda { "CUDA 13.0" } else { "CPU" }));
            let mut c = uv_command(&uv, cfg);
            c.args(["pip", "install", "--python"]).arg(&py).args(TORCH_PACKAGES).args(["--index-url", &index]);
            run_child(ctx, c, "installing torch", true, &mut |_| {})?;
        }
        Step::Packages => {
            let py = r.managed_python();
            if !py.is_file() {
                return Err("the Python environment does not exist (run the python step first)".into());
            }
            let uv = existing_uv(cfg).ok_or("uv is not installed (run the uv step first)")?;
            let pipe = r.pipeline_dir().ok_or("the pipeline folder was not found")?;
            let req = pipe.join("requirements-installer.txt");
            if !req.is_file() {
                return Err(format!("{} is missing", req.display()));
            }
            let mut c = uv_command(&uv, cfg);
            c.args(["pip", "install", "--python"]).arg(&py).arg("-r").arg(&req);
            ctx.log("info", format!("uv pip install -r {}", req.display()));
            run_child(ctx, c, "installing the pipeline packages", true, &mut |_| {})?;
            let nodeps = pipe.join("requirements-installer-nodeps.txt");
            if nodeps.is_file() {
                let mut c = uv_command(&uv, cfg);
                c.args(["pip", "install", "--no-deps", "--python"]).arg(&py).arg("-r").arg(&nodeps);
                ctx.log("info", format!("uv pip install --no-deps -r {}", nodeps.display()));
                run_child(ctx, c, "installing the --no-deps packages", true, &mut |_| {})?;
            }
        }
        Step::Models | Step::Verify => {
            let py = pipeline_python(r).ok_or("no Python environment (run the python step first)")?;
            let pipe = r.pipeline_dir().ok_or("the pipeline folder was not found")?;
            let mut c = crate::jobs::python_command(&py, &pipe);
            if ctx.step == Step::Models {
                c.args(["-m", "cappycat_pipeline", "download-models"]);
                let models = r.models_dir();
                c.env("CAPPYCAT_MODELS_DIR", &models);
                ctx.log("info", format!("downloading the models into {}", models.display()));
                run_child(ctx, c, "download-models", true, &mut |v| {
                    if v.get("event").and_then(|e| e.as_str()) == Some("progress") {
                        // overall bytes when the pipeline reports them, else its pct
                        let (done, total) = (v.get("overallBytes").and_then(|b| b.as_f64()), v.get("overallTotalBytes").and_then(|b| b.as_f64()));
                        let pct = match (done, total) {
                            (Some(d), Some(t)) if t > 0.0 => d / t,
                            _ => v.get("pct").and_then(|p| p.as_f64()).unwrap_or(0.0),
                        };
                        let mut msg = v.get("message").and_then(|m| m.as_str()).unwrap_or("").to_string();
                        if let (Some(d), Some(t)) = (done, total) {
                            msg = format!("{msg} ({} of {})", mb(d as u64), mb(t as u64));
                        }
                        ctx.progress(pct, msg);
                    } else if v.get("event").and_then(|e| e.as_str()) == Some("log") {
                        let level = v.get("level").and_then(|l| l.as_str()).unwrap_or("info").to_string();
                        ctx.log(&level, v.get("message").and_then(|m| m.as_str()).unwrap_or("").to_string());
                    }
                })?;
                let _ = std::fs::write(models.join(MODELS_MARKER), b"download-models succeeded\n");
            } else {
                c.args(["-m", "cappycat_pipeline", "doctor", "--json"]);
                let out = run_child(ctx, c, "doctor", true, &mut |_| {})?;
                let v: Value = out.lines().rev().find_map(|l| serde_json::from_str::<Value>(l).ok()).unwrap_or(Value::Null);
                let summary = format!(
                    "ready {}, python {}, torch {} (CUDA {}), models ready: {}",
                    v["ready"].as_bool().map(|b| b.to_string()).unwrap_or_else(|| "?".into()),
                    v["python"].as_str().unwrap_or("?"),
                    v["torch"].as_str().unwrap_or("missing"),
                    v["torchCuda"].as_bool().unwrap_or(false),
                    v["modelsReady"].as_bool().unwrap_or(false)
                );
                ctx.log("info", format!("doctor: {summary}"));
                if v["ready"].as_bool() == Some(false) {
                    let missing: Vec<String> = v["missing"].as_array().map(|a| a.iter().map(|m| m.as_str().map(String::from).unwrap_or_else(|| m.to_string())).collect()).unwrap_or_default();
                    ctx.log("warn", format!("doctor: not ready, missing: {}", missing.join(", ")));
                }
                *report = v;
            }
        }
    }
    ctx.progress(1.0, "done");
    Ok(())
}

/// Run `steps` in order (the job body; tests call it directly). Returns the doctor report
/// (`null` when `verify` did not run).
pub fn run_steps(cfg: &SetupConfig, steps: &[Step], job_id: &str, sink: &dyn EventSink, ctl: &TaskCtl) -> Result<Value, String> {
    std::fs::create_dir_all(cfg.home()).map_err(|e| format!("cannot create {}: {e}", cfg.home().display()))?;
    let mut report = Value::Null;
    for (i, &step) in steps.iter().enumerate() {
        if ctl.is_cancelled() {
            return Err("cancelled".into());
        }
        let ctx = SetupCtx { job_id, sink, ctl, step, index: i, total: steps.len() };
        ctx.progress(0.0, format!("starting {}", step.name()));
        let t0 = Instant::now();
        match run_step(cfg, &ctx, &mut report) {
            Ok(()) => ctx.log("info", format!("{} finished in {:.1}s", step.name(), t0.elapsed().as_secs_f64())),
            Err(e) if e == "cancelled" || ctl.is_cancelled() => return Err("cancelled".into()),
            Err(e) => {
                ctx.log("error", e.clone());
                return Err(format!("{}: {e}", step.name()));
            }
        }
    }
    Ok(report)
}

static SETUP_RUNNING: AtomicBool = AtomicBool::new(false);

/// Start the setup job; returns its id. Events: `setup://progress|log|done`.
pub fn setup_ai_with(jobs: Arc<JobManager>, sink: Arc<dyn EventSink>, cfg: SetupConfig, components: Option<Vec<String>>) -> Result<String, String> {
    let status = status_with(&cfg.resolver);
    let steps = expand_components(components.as_deref(), &status)?;
    if SETUP_RUNNING.swap(true, Ordering::SeqCst) {
        return Err("AI setup is already running".into());
    }
    let job_id = JobManager::new_job_id("setup");
    let ctl = jobs.register_task(&job_id);
    let id = job_id.clone();
    let spawned = std::thread::Builder::new().name(format!("{job_id}-main")).spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_steps(&cfg, &steps, &id, sink.as_ref(), &ctl)))
            .unwrap_or_else(|_| Err("internal error in the setup job".into()));
        let payload = match &result {
            Ok(report) => json!({ "jobId": id, "ok": true, "error": null, "report": report }),
            Err(e) => json!({ "jobId": id, "ok": false, "error": e }),
        };
        sink.emit("setup://done", payload);
        ctl.mark_done();
        jobs.finish(&id);
        SETUP_RUNNING.store(false, Ordering::SeqCst);
    });
    if let Err(e) = spawned {
        SETUP_RUNNING.store(false, Ordering::SeqCst);
        return Err(e.to_string());
    }
    Ok(job_id)
}

pub fn setup_ai(jobs: Arc<JobManager>, sink: Arc<dyn EventSink>, components: Option<Vec<String>>) -> Result<String, String> {
    setup_ai_with(jobs, sink, SetupConfig::from_env(), components)
}

pub fn cancel_setup(jobs: &JobManager, job_id: &str) -> Result<(), String> {
    jobs.cancel(job_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::CollectingSink;

    fn temp_home(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("cappycat-setup-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn cfg(home: &Path, ffmpeg_url: String, uv_url: String) -> SetupConfig {
        let mut r = Resolver::current();
        r.env.insert("CAPPYCAT_HOME".into(), home.to_string_lossy().into_owned());
        SetupConfig { resolver: r, ffmpeg_url, uv_url, torch_index: None, detect_existing: false }
    }

    /// A zip (stored) with the given (name, bytes) entries.
    fn make_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut cur = std::io::Cursor::new(Vec::new());
        {
            let mut z = zip::ZipWriter::new(&mut cur);
            let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
            for (name, data) in entries {
                z.start_file(*name, opts).unwrap();
                z.write_all(data).unwrap();
            }
            z.finish().unwrap();
        }
        cur.into_inner()
    }

    /// A local HTTP server: `/ffmpeg.zip`, `/uv.zip`, `/slow.zip` (dribbles), anything else 404.
    fn serve(ffmpeg: Vec<u8>, uv: Vec<u8>) -> (String, std::sync::mpsc::Sender<()>) {
        use axum::{body::Body, routing::get, Router};
        let (tx_addr, rx_addr) = std::sync::mpsc::channel();
        let (tx_stop, rx_stop) = std::sync::mpsc::channel::<()>();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
            rt.block_on(async move {
                let (ff, uvb) = (ffmpeg.clone(), uv.clone());
                let app = Router::new()
                    .route("/ffmpeg.zip", get(move || { let b = ff.clone(); async move { b } }))
                    .route("/uv.zip", get(move || { let b = uvb.clone(); async move { b } }))
                    .route(
                        "/slow.zip",
                        get(|| async {
                            let stream = futures_util::stream::unfold(0u32, |i| async move {
                                if i >= 200 {
                                    return None;
                                }
                                tokio::time::sleep(Duration::from_millis(50)).await;
                                Some((Ok::<_, std::io::Error>(vec![0u8; 64 * 1024]), i + 1))
                            });
                            axum::response::Response::builder().header("content-length", (200 * 64 * 1024).to_string()).body(Body::from_stream(stream)).unwrap()
                        }),
                    );
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                tx_addr.send(listener.local_addr().unwrap()).unwrap();
                let server = std::future::IntoFuture::into_future(axum::serve(listener, app));
                tokio::select! {
                    _ = server => {}
                    _ = tokio::task::spawn_blocking(move || { let _ = rx_stop.recv(); }) => {}
                }
            });
        });
        let addr = rx_addr.recv().unwrap();
        (format!("http://{addr}"), tx_stop)
    }

    fn sha256_hex(b: &[u8]) -> String {
        Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn components_expand_to_steps() {
        let missing = SetupStatus { ffmpeg: false, python: false, models: false, gpu: None, installed: true };
        let all = expand_components(None, &missing).unwrap();
        assert_eq!(all, Step::ALL.to_vec());
        let present = SetupStatus { ffmpeg: true, python: true, models: true, gpu: None, installed: true };
        assert!(expand_components(None, &present).unwrap().is_empty(), "nothing missing: nothing to do");
        let only_models = SetupStatus { models: false, ..present.clone() };
        assert_eq!(expand_components(None, &only_models).unwrap(), vec![Step::Models, Step::Verify]);
        let c = |v: &[&str]| expand_components(Some(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>()), &present).unwrap();
        assert_eq!(c(&["ffmpeg"]), vec![Step::Ffmpeg]);
        assert_eq!(c(&["python"]), vec![Step::Uv, Step::Python, Step::Torch, Step::Packages, Step::Verify]);
        assert_eq!(c(&["models", "ffmpeg"]), vec![Step::Ffmpeg, Step::Models, Step::Verify], "run in the fixed order");
        assert_eq!(c(&["uv"]), vec![Step::Uv]);
        assert!(expand_components(Some(&["gpu-driver".to_string()]), &present).unwrap_err().contains("unknown setup component"));
        let names: Vec<&str> = Step::ALL.iter().map(|s| s.name()).collect();
        assert_eq!(names, ["ffmpeg", "uv", "python", "torch", "packages", "models", "verify"]);
    }

    #[test]
    fn status_json_shape() {
        let s = setup_status();
        let v = serde_json::to_value(&s).unwrap();
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort();
        assert_eq!(keys, ["ffmpeg", "gpu", "installed", "models", "python"]);
        assert!(v["gpu"].is_null() || v["gpu"].is_string());
        assert!(!s.installed, "the tests run from the checkout");
        let p = PlanItem { step: "ffmpeg".into(), url: None, size_bytes: Some(3), note: "n".into() };
        assert_eq!(serde_json::to_value(&p).unwrap(), json!({ "step": "ffmpeg", "sizeBytes": 3, "note": "n" }));
    }

    #[test]
    fn downloads_and_extracts_from_a_local_server_idempotently() {
        let ff_zip = make_zip(&[("ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe", b"fake ffmpeg"), ("ffmpeg-master-latest-win64-gpl/bin/ffprobe.exe", b"fake ffprobe"), ("ffmpeg-master-latest-win64-gpl/doc/readme.txt", b"docs")]);
        let uv_zip = make_zip(&[("uv.exe", b"fake uv"), ("uvx.exe", b"fake uvx")]);
        let (base, stop) = serve(ff_zip.clone(), uv_zip.clone());
        let home = temp_home("local");
        let cfg = cfg(&home, format!("{base}/ffmpeg.zip"), format!("{base}/uv.zip"));
        // the plan shows the URLs and sizes
        let plan = plan_with(&cfg, Some(&["ffmpeg".to_string(), "uv".to_string()])).unwrap();
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].url.as_deref(), Some(format!("{base}/ffmpeg.zip").as_str()));
        assert_eq!(plan[0].size_bytes, Some(ff_zip.len() as u64));
        assert_eq!(plan[1].size_bytes, Some(uv_zip.len() as u64));

        let sink = CollectingSink::new();
        let ctl = TaskCtl::default();
        run_steps(&cfg, &[Step::Ffmpeg, Step::Uv], "setup_t", sink.as_ref(), &ctl).unwrap();
        let bin = home.join("ffmpeg").join("bin");
        assert_eq!(std::fs::read(bin.join(exe("ffmpeg"))).unwrap(), b"fake ffmpeg");
        assert_eq!(std::fs::read(bin.join(exe("ffprobe"))).unwrap(), b"fake ffprobe");
        assert!(!bin.join("readme.txt").exists(), "only the binaries");
        assert_eq!(std::fs::read(home.join("bin").join(exe("uv"))).unwrap(), b"fake uv");
        assert!(!home.join("downloads").join("ffmpeg.zip").exists(), "the zip is removed");
        let logs: Vec<String> = sink.named("setup://log").iter().filter_map(|e| e["message"].as_str().map(String::from)).collect();
        assert!(logs.iter().any(|m| m.contains(&sha256_hex(&ff_zip))), "sha256 logged: {logs:?}");
        assert!(logs.iter().any(|m| m.contains(&format!("{base}/ffmpeg.zip"))), "URL logged");
        let prog = sink.named("setup://progress");
        assert!(prog.iter().all(|p| p["jobId"] == "setup_t" && p["pct"].as_f64().unwrap() <= 1.0));
        assert!(prog.iter().any(|p| p["step"] == "ffmpeg" && p["pct"] == 1.0));
        assert!(prog.iter().any(|p| p["step"] == "uv" && p["pct"] == 1.0));

        // idempotent: a second run skips both steps
        let sink = CollectingSink::new();
        run_steps(&cfg, &[Step::Ffmpeg, Step::Uv], "setup_t2", sink.as_ref(), &ctl).unwrap();
        let msgs: Vec<String> = sink.named("setup://progress").iter().filter_map(|e| e["message"].as_str().map(String::from)).collect();
        assert!(msgs.iter().any(|m| m.contains("ffmpeg already available")) && msgs.iter().any(|m| m.contains("uv already available")), "{msgs:?}");

        // HTTP errors name the URL and the status
        let bad = cfg.clone();
        let e = download(&format!("{base}/missing.zip"), &home.join("x.zip"), &TaskCtl::default(), &mut |_, _| {}).unwrap_err();
        assert!(e.contains("404") && e.contains("missing.zip"), "{e}");
        // a zip without the binaries fails the step
        let mut wrong = bad;
        wrong.uv_url = format!("{base}/ffmpeg.zip");
        let _ = std::fs::remove_dir_all(home.join("bin"));
        let e = run_steps(&wrong, &[Step::Uv], "setup_t3", CollectingSink::new().as_ref(), &TaskCtl::default()).unwrap_err();
        assert!(e.contains("does not contain"), "{e}");
        let _ = stop.send(());
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn a_setup_job_can_be_cancelled_mid_download() {
        let (base, stop) = serve(Vec::new(), Vec::new());
        let home = temp_home("cancel");
        let cfg = cfg(&home, format!("{base}/slow.zip"), format!("{base}/uv.zip"));
        let jobs = Arc::new(JobManager::new());
        let sink = CollectingSink::new();
        let id = setup_ai_with(jobs.clone(), sink.clone(), cfg, Some(vec!["ffmpeg".into()])).unwrap();
        assert!(id.starts_with("setup_"));
        // wait for the first byte progress, then cancel
        let t0 = Instant::now();
        while !sink.named("setup://progress").iter().any(|p| p["message"].as_str().unwrap_or("").contains(" of ")) {
            assert!(t0.elapsed() < Duration::from_secs(20), "no download progress");
            std::thread::sleep(Duration::from_millis(20));
        }
        cancel_setup(&jobs, &id).unwrap();
        assert!(jobs.wait(&id, Duration::from_secs(10)), "cancel is prompt");
        let done = sink.named("setup://done");
        assert_eq!(done.len(), 1);
        assert_eq!(done[0]["ok"], false);
        assert_eq!(done[0]["error"], "cancelled");
        assert!(!home.join("downloads").join("ffmpeg.partial").exists(), "partial file removed");
        assert!(!home.join("ffmpeg").join("bin").join(exe("ffmpeg")).exists());
        let _ = stop.send(());
        let _ = std::fs::remove_dir_all(&home);
    }

    /// Real downloads of the ffmpeg and uv zips into a temporary home (network; ~200 MB).
    #[test]
    #[ignore = "downloads the real ffmpeg + uv archives (network)"]
    fn real_ffmpeg_and_uv_download_dry_run() {
        let home = temp_home("real");
        let cfg = cfg(&home, DEFAULT_FFMPEG_URL.into(), DEFAULT_UV_URL.into());
        for p in plan_with(&cfg, Some(&["ffmpeg".to_string(), "uv".to_string()])).unwrap() {
            eprintln!("plan: {} {:?} {:?} bytes — {}", p.step, p.url, p.size_bytes, p.note);
        }
        let sink = CollectingSink::new();
        let t0 = Instant::now();
        run_steps(&cfg, &[Step::Ffmpeg, Step::Uv], "setup_real", sink.as_ref(), &TaskCtl::default()).unwrap();
        for l in sink.named("setup://log") {
            eprintln!("{}", l["message"].as_str().unwrap_or(""));
        }
        eprintln!("total {:.1}s", t0.elapsed().as_secs_f64());
        let ff = home.join("ffmpeg").join("bin").join(exe("ffmpeg"));
        let out = Command::new(&ff).arg("-version").output().unwrap();
        let v = String::from_utf8_lossy(&out.stdout);
        eprintln!("{}", v.lines().next().unwrap_or(""));
        assert!(v.starts_with("ffmpeg version"));
        let uv = home.join("bin").join(exe("uv"));
        let out = Command::new(&uv).arg("--version").output().unwrap();
        eprintln!("{}", String::from_utf8_lossy(&out.stdout).trim());
        assert!(out.status.success());
        let _ = std::fs::remove_dir_all(&home);
        assert!(!home.exists());
    }
}
