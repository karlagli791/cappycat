//! ffmpeg / ffprobe integration.
//!
//! DEVIATION FROM THE DESIGN SPEC: the spec lists `ffmpeg-next` (libav bindings)
//! for the video engine core. This machine has no libav development libraries
//! (only the Gyan static `ffmpeg.exe` build from WinGet), so phase 1 shells out
//! to the `ffmpeg` / `ffprobe` binaries instead of linking libav. Everything is
//! funnelled through this module so a later swap to in-process decoding only
//! touches one file.
//!
//! Binary discovery order:
//! 1. `CAPPYCAT_FFMPEG_DIR` environment variable (directory containing both exes)
//! 2. `PATH` lookup
//! 3. `%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg-*\bin\`
//! 4. `%LOCALAPPDATA%\Microsoft\WinGet\Links\`
//! 5. the copy `setup_ai` installs: `<app data home>\ffmpeg\bin\` (`%LOCALAPPDATA%\Cappycat\ffmpeg\bin`,
//!    see [`crate::paths`])

use crate::model::{Asset, AssetKind};
use serde::Deserialize;
use sha1::{Digest, Sha1};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::Duration;

/// Upper bound for one ffprobe run.
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(30);
/// Upper bound for short ffmpeg jobs (a frame, a thumbnail strip).
pub const FFMPEG_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, thiserror::Error)]
pub enum FfmpegError {
    #[error("ffmpeg/ffprobe not found: {0}")]
    NotFound(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{program} exited with {code:?}: {stderr}")]
    Failed { program: String, code: Option<i32>, stderr: String },
    #[error("could not parse ffprobe output: {0}")]
    Parse(String),
    #[error("{0}")]
    Other(String),
}

impl From<FfmpegError> for String {
    fn from(e: FfmpegError) -> Self {
        e.to_string()
    }
}

pub type Result<T> = std::result::Result<T, FfmpegError>;

/// Resolved binary locations.
#[derive(Debug, Clone)]
pub struct FfmpegPaths {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
}

static PATHS: OnceLock<FfmpegPaths> = OnceLock::new();

#[cfg(windows)]
const EXE: &str = ".exe";
#[cfg(not(windows))]
const EXE: &str = "";

fn pair_in_dir(dir: &Path) -> Option<FfmpegPaths> {
    let ffmpeg = dir.join(format!("ffmpeg{EXE}"));
    let ffprobe = dir.join(format!("ffprobe{EXE}"));
    if ffmpeg.is_file() && ffprobe.is_file() {
        Some(FfmpegPaths { ffmpeg, ffprobe })
    } else {
        None
    }
}

fn candidate_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(d) = std::env::var("CAPPYCAT_FFMPEG_DIR") {
        if !d.trim().is_empty() {
            dirs.push(PathBuf::from(d));
        }
    }
    if let Some(path) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&path));
    }
    if let Some(local) = std::env::var_os("LOCALAPPDATA").map(PathBuf::from) {
        let pattern = local
            .join("Microsoft")
            .join("WinGet")
            .join("Packages")
            .join("Gyan.FFmpeg*")
            .join("ffmpeg-*")
            .join("bin");
        if let Some(pat) = pattern.to_str() {
            if let Ok(matches) = glob::glob(pat) {
                let mut found: Vec<PathBuf> = matches.flatten().collect();
                // Prefer the highest version (lexicographically last).
                found.sort();
                found.reverse();
                dirs.extend(found);
            }
        }
        dirs.push(local.join("Microsoft").join("WinGet").join("Links"));
    }
    // the copy `setup_ai` downloads: <home>/ffmpeg/bin (or any <home>/ffmpeg/*/bin)
    let own = crate::paths::resolver().ffmpeg_dir();
    dirs.push(own.join("bin"));
    dirs.push(own.clone());
    if let Some(pat) = own.join("*").join("bin").to_str() {
        if let Ok(matches) = glob::glob(pat) {
            dirs.extend(matches.flatten());
        }
    }
    dirs
}

/// Locate ffmpeg + ffprobe (cached after the first success).
pub fn find_binaries() -> Result<&'static FfmpegPaths> {
    if let Some(p) = PATHS.get() {
        return Ok(p);
    }
    let dirs = candidate_dirs();
    for dir in &dirs {
        if let Some(p) = pair_in_dir(dir) {
            return Ok(PATHS.get_or_init(|| p));
        }
    }
    Err(FfmpegError::NotFound(format!(
        "searched CAPPYCAT_FFMPEG_DIR, PATH and the WinGet package folders ({} dirs)",
        dirs.len()
    )))
}

/// Build a `Command` for the given binary with the console window suppressed on Windows.
/// Start it with [`crate::procs::spawn`] / [`crate::procs::output_timeout`] so it joins the
/// kill-on-close job object.
pub fn command(program: &Path) -> Command {
    let mut cmd = Command::new(program);
    crate::procs::hide_window(&mut cmd);
    cmd
}

fn run_capture(program: &Path, args: &[String], timeout: Duration) -> Result<Vec<u8>> {
    let out = crate::procs::output_timeout(command(program).args(args), timeout)?;
    if !out.status.success() {
        return Err(FfmpegError::Failed {
            program: program.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
            code: out.status.code(),
            stderr: String::from_utf8_lossy(&out.stderr).chars().take(4000).collect(),
        });
    }
    Ok(out.stdout)
}

/* ------------------------------------------------------------- cache dirs */

/// `%LOCALAPPDATA%\Cappycat\cache` (see [`crate::paths`]).
pub fn cache_dir() -> PathBuf {
    crate::paths::cache_dir()
}

/// `%LOCALAPPDATA%\cappycat\cache\thumbs`
pub fn thumbs_dir() -> PathBuf {
    cache_dir().join("thumbs")
}

fn sha1_hex(data: &str) -> String {
    let mut h = Sha1::new();
    h.update(data.as_bytes());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Cache key for a thumbnail strip: hashes the path, file size, mtime, count and width.
pub fn thumb_key(path: &Path, count: u32, width: u32) -> String {
    let meta = std::fs::metadata(path).ok();
    let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
    let mtime = meta
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    sha1_hex(&format!("{}|{size}|{mtime}|{count}|{width}", path.to_string_lossy()))
}

/* ------------------------------------------------------------------ probe */

#[derive(Debug, Deserialize)]
struct ProbeOutput {
    #[serde(default)]
    streams: Vec<ProbeStream>,
    #[serde(default)]
    format: Option<ProbeFormat>,
}

#[derive(Debug, Deserialize)]
struct ProbeStream {
    #[serde(default)]
    codec_type: Option<String>,
    #[serde(default)]
    tags: Option<serde_json::Value>,
    #[serde(default)]
    side_data_list: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    color_space: Option<String>,
    #[serde(default)]
    color_range: Option<String>,
    #[serde(default)]
    pix_fmt: Option<String>,
    #[serde(default)]
    codec_name: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    r_frame_rate: Option<String>,
    #[serde(default)]
    avg_frame_rate: Option<String>,
    #[serde(default)]
    duration: Option<String>,
    #[serde(default)]
    nb_frames: Option<String>,
    #[serde(default)]
    disposition: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ProbeFormat {
    #[serde(default)]
    duration: Option<String>,
    #[serde(default)]
    format_name: Option<String>,
}

/// Parse an ffprobe rational such as `"24000/1001"` or `"25"` into f64.
pub fn parse_fraction(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Some((n, d)) = s.split_once('/') {
        let n: f64 = n.trim().parse().ok()?;
        let d: f64 = d.trim().parse().ok()?;
        if d == 0.0 || !n.is_finite() || !d.is_finite() {
            return None;
        }
        Some(n / d)
    } else {
        s.parse().ok()
    }
}

pub const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff", "heic", "avif"];
pub const AUDIO_EXTS: &[&str] = &["mp3", "wav", "flac", "aac", "m4a", "ogg", "opus", "wma", "aiff", "aif"];
pub const VIDEO_EXTS: &[&str] = &[
    "mp4", "mov", "mkv", "avi", "webm", "m4v", "mts", "m2ts", "mxf", "wmv", "flv", "mpg", "mpeg", "3gp", "ts",
];

/// Video, audio, image or `.cube` file (by extension)?
pub fn is_media_ext(ext: &str) -> bool {
    let ext = ext.to_ascii_lowercase();
    ext == "cube" || VIDEO_EXTS.contains(&ext.as_str()) || AUDIO_EXTS.contains(&ext.as_str()) || IMAGE_EXTS.contains(&ext.as_str())
}

pub fn extension_lower(path: &Path) -> String {
    path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default()
}

pub fn new_asset_id() -> String {
    format!("ast_{}", uuid::Uuid::new_v4().simple())
}

/// Probe a media file and build an [`Asset`] for it.
pub fn probe(path: &Path) -> Result<Asset> {
    if !path.is_file() {
        return Err(FfmpegError::Other(format!("file not found: {}", path.display())));
    }
    let name = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let path_str = path.to_string_lossy().replace('\\', "/");
    let ext = extension_lower(path);

    // LUTs are not media: no probing.
    if ext == "cube" {
        return Ok(Asset {
            id: new_asset_id(),
            path: path_str,
            name,
            kind: AssetKind::Lut,
            ..Asset::default()
        });
    }

    let bins = find_binaries()?;
    let args: Vec<String> = [
        "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams",
    ]
    .iter()
    .map(|s| s.to_string())
    .chain(std::iter::once(path.to_string_lossy().into_owned()))
    .collect();
    let out = run_capture(&bins.ffprobe, &args, PROBE_TIMEOUT)?;
    let probe: ProbeOutput = serde_json::from_slice(&out).map_err(|e| FfmpegError::Parse(e.to_string()))?;
    Ok(asset_from_probe(path, &probe, name, path_str, &ext))
}

/// Probe several files, up to 8 at a time; results in input order.
pub fn probe_many(paths: &[PathBuf]) -> Vec<Result<Asset>> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Result<Asset>>>> = Mutex::new((0..paths.len()).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..paths.len().clamp(1, 8) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let Some(p) = paths.get(i) else { break };
                let r = probe(p);
                results.lock().unwrap()[i] = Some(r);
            });
        }
    });
    results.into_inner().unwrap().into_iter().map(|r| r.unwrap_or_else(|| Err(FfmpegError::Other("not probed".into())))).collect()
}

fn main_video_stream(probe: &ProbeOutput) -> Option<&ProbeStream> {
    probe.streams.iter().find(|s| {
        s.codec_type.as_deref() == Some("video")
            && !s
                .disposition
                .as_ref()
                .and_then(|d| d.get("attached_pic"))
                .and_then(|v| v.as_i64())
                .map(|v| v == 1)
                .unwrap_or(false)
    })
}

/// Display rotation of a video stream in degrees (0, 90, 180, 270): the display
/// matrix side data (`rotation`, e.g. -90 for a portrait phone clip) or the
/// legacy `rotate` tag. ffmpeg auto-rotates decoded frames by it.
fn stream_rotation(s: &ProbeStream) -> i32 {
    let num = |r: &serde_json::Value| r.as_f64().or_else(|| r.as_str().and_then(|x| x.trim().parse::<f64>().ok()));
    let from_side = s.side_data_list.as_ref().and_then(|l| l.iter().find_map(|d| d.get("rotation").and_then(num)));
    let from_tag = s.tags.as_ref().and_then(|t| t.get("rotate")).and_then(num);
    let r = from_side.or(from_tag).unwrap_or(0.0).round() as i32;
    ((r % 360) + 360) % 360
}

/// Width/height as displayed (swapped for 90 / 270 degree rotations).
fn display_size(s: &ProbeStream) -> (u32, u32) {
    let (w, h) = (s.width.unwrap_or(0), s.height.unwrap_or(0));
    match stream_rotation(s) {
        90 | 270 => (h, w),
        _ => (w, h),
    }
}

fn asset_from_probe(path: &Path, probe: &ProbeOutput, name: String, path_str: String, ext: &str) -> Asset {
    let video = main_video_stream(probe);
    let audio = probe.streams.iter().find(|s| s.codec_type.as_deref() == Some("audio"));

    let format_duration = probe.format.as_ref().and_then(|f| f.duration.as_deref()).and_then(|d| d.parse::<f64>().ok());
    let stream_duration = video
        .and_then(|v| v.duration.as_deref())
        .or_else(|| audio.and_then(|a| a.duration.as_deref()))
        .and_then(|d| d.parse::<f64>().ok());
    let duration_s = format_duration.or(stream_duration).unwrap_or(0.0);

    // Prefer avg_frame_rate: AI generators often write variable-frame-rate files (24 fps content on a
    // 60 fps timebase) where r_frame_rate is only the timebase. Fall back to r_frame_rate when the
    // average is missing or implausible (e.g. "0/0" for still images).
    let plausible = |f: &f64| *f >= 1.0 && *f <= 240.0 && f.is_finite();
    let fps = video
        .and_then(|v| v.avg_frame_rate.as_deref().and_then(parse_fraction).filter(plausible))
        .or_else(|| video.and_then(|v| v.r_frame_rate.as_deref().and_then(parse_fraction).filter(plausible)))
        .unwrap_or(0.0);

    let is_image_ext = IMAGE_EXTS.contains(&ext);
    let single_frame = video
        .and_then(|v| v.nb_frames.as_deref())
        .and_then(|n| n.parse::<u64>().ok())
        .map(|n| n <= 1)
        .unwrap_or(false);
    let format_is_image = probe
        .format
        .as_ref()
        .and_then(|f| f.format_name.as_deref())
        .map(|f| f.contains("image2") || f.contains("_pipe") || f == "png_pipe")
        .unwrap_or(false);

    let kind = if is_image_ext || (video.is_some() && (single_frame || format_is_image) && audio.is_none()) {
        AssetKind::Image
    } else if video.is_some() {
        AssetKind::Video
    } else if audio.is_some() || AUDIO_EXTS.contains(&ext) {
        AssetKind::Audio
    } else {
        AssetKind::Video
    };

    let codec = match kind {
        AssetKind::Audio => audio.and_then(|a| a.codec_name.clone()),
        _ => video.and_then(|v| v.codec_name.clone()).or_else(|| audio.and_then(|a| a.codec_name.clone())),
    };

    let _ = path;
    let (width, height) = video.map(display_size).unwrap_or((0, 0));
    Asset {
        id: new_asset_id(),
        path: path_str,
        name,
        kind,
        duration_ms: if kind == AssetKind::Image { 0.0 } else { (duration_s * 1000.0).round() },
        width,
        height,
        fps: if kind == AssetKind::Image { 0.0 } else { (fps * 1000.0).round() / 1000.0 },
        has_audio: audio.is_some(),
        codec,
        scene_tags: None,
        order: None,
        order_reason: None,
        stems: None,
        stem_of: None,
    }
}

/// What the exporter needs to know about a video / image source beyond the
/// [`Asset`] fields (which may come from an older project file).
#[derive(Debug, Clone, PartialEq)]
pub struct VideoInfo {
    /// displayed size (rotation applied, i.e. the size of ffmpeg's auto-rotated frames)
    pub width: u32,
    pub height: u32,
    /// display rotation in degrees (0 / 90 / 180 / 270)
    pub rotation: i32,
    /// `avg_frame_rate` (the real cadence of VFR files), falling back to `r_frame_rate`
    pub avg_fps: Option<f64>,
    /// `color_space` tag (`bt709`, `smpte170m`, …), `None` when untagged
    pub color_space: Option<String>,
    /// `color_range` tag (`tv` / `pc`), `None` when untagged
    pub color_range: Option<String>,
    pub pix_fmt: Option<String>,
    pub duration_ms: f64,
}

/// One ffprobe of the main video stream (for export planning).
pub fn probe_video_info(path: &Path) -> Result<VideoInfo> {
    let bins = find_binaries()?;
    let args: Vec<String> = ["-v", "quiet", "-print_format", "json", "-show_format", "-show_streams"]
        .iter()
        .map(|s| s.to_string())
        .chain(std::iter::once(path.to_string_lossy().into_owned()))
        .collect();
    let out = run_capture(&bins.ffprobe, &args, PROBE_TIMEOUT)?;
    let probe: ProbeOutput = serde_json::from_slice(&out).map_err(|e| FfmpegError::Parse(e.to_string()))?;
    let v = main_video_stream(&probe).ok_or_else(|| FfmpegError::Other(format!("no video stream in {}", path.display())))?;
    let (width, height) = display_size(v);
    let fps = |s: &Option<String>| s.as_deref().and_then(parse_fraction).filter(|f| *f >= 1.0 && *f <= 1000.0 && f.is_finite());
    let known = |s: &Option<String>| s.clone().filter(|x| !x.is_empty() && x != "unknown" && x != "unspecified");
    let duration_ms = probe
        .format
        .as_ref()
        .and_then(|f| f.duration.as_deref())
        .or(v.duration.as_deref())
        .and_then(|d| d.parse::<f64>().ok())
        .map(|d| d * 1000.0)
        .unwrap_or(0.0);
    Ok(VideoInfo {
        width,
        height,
        rotation: stream_rotation(v),
        avg_fps: fps(&v.avg_frame_rate).or_else(|| fps(&v.r_frame_rate)),
        color_space: known(&v.color_space),
        color_range: known(&v.color_range),
        pix_fmt: v.pix_fmt.clone(),
        duration_ms,
    })
}

/// swscale `in_color_matrix` for a stream: its `color_space` tag when present,
/// else the usual convention for untagged video — BT.709 for HD (height ≥ 720),
/// BT.601 below. `None` for RGB / palette / grey sources (no matrix involved).
pub fn input_color_matrix(color_space: Option<&str>, pix_fmt: Option<&str>, height: u32) -> Option<&'static str> {
    if let Some(pf) = pix_fmt {
        let pf = pf.to_ascii_lowercase();
        let rgbish = ["rgb", "bgr", "gbr", "argb", "abgr", "0rgb", "0bgr", "pal8", "gray", "ya8", "ya16", "monow", "monob"];
        if rgbish.iter().any(|p| pf.starts_with(p)) {
            return None;
        }
    }
    let tagged = match color_space.map(|c| c.to_ascii_lowercase()).as_deref() {
        Some("bt709") => Some("bt709"),
        Some("smpte170m") | Some("bt470bg") | Some("bt601") => Some("bt601"),
        Some(c) if c.starts_with("bt2020") => Some("bt2020"),
        Some("smpte240m") => Some("smpte240m"),
        Some("fcc") => Some("fcc"),
        _ => None,
    };
    Some(tagged.unwrap_or(if height >= 720 { "bt709" } else { "bt601" }))
}

/// swscale `in_range` for a YUV stream: `pc` (full) when tagged full range or a `yuvj*`
/// format, else `tv` (limited, also the convention for untagged video).
pub fn input_range(color_range: Option<&str>, pix_fmt: Option<&str>) -> &'static str {
    let full = matches!(color_range.map(|c| c.to_ascii_lowercase()).as_deref(), Some("pc") | Some("jpeg") | Some("full"))
        || pix_fmt.map(|p| p.starts_with("yuvj")).unwrap_or(false);
    if full {
        "pc"
    } else {
        "tv"
    }
}

/* ----------------------------------------------------------------- frames */

fn seconds(ms: f64) -> String {
    format!("{:.3}", (ms.max(0.0)) / 1000.0)
}

/// ffmpeg arguments for one JPEG frame at `ms`, `width` px wide, on stdout.
pub fn frame_args(path: &Path, ms: f64, width: u32) -> Vec<String> {
    let width = width.clamp(16, 4096);
    vec![
        "-v".into(), "error".into(),
        "-ss".into(), seconds(ms),
        "-i".into(), path.to_string_lossy().into_owned(),
        "-frames:v".into(), "1".into(),
        "-vf".into(), format!("scale={width}:-2:flags=bicubic"),
        "-f".into(), "image2pipe".into(),
        "-c:v".into(), "mjpeg".into(),
        "-q:v".into(), "4".into(),
        "pipe:1".into(),
    ]
}

/// Decode a single frame at `ms`, scaled to `width` px wide, as JPEG bytes.
pub fn extract_frame(path: &Path, ms: f64, width: u32) -> Result<Vec<u8>> {
    let bins = find_binaries()?;
    let width = width.clamp(16, 4096);
    let bytes = run_capture(&bins.ffmpeg, &frame_args(path, ms, width), FFMPEG_TIMEOUT)?;
    if bytes.is_empty() {
        // Seeking past the end yields nothing; retry from the last second.
        let asset = probe(path)?;
        if asset.duration_ms > 0.0 && ms > asset.duration_ms - 100.0 {
            let retry_ms = (asset.duration_ms - 200.0).max(0.0);
            if retry_ms < ms {
                return extract_frame(path, retry_ms, width);
            }
        }
        return Err(FfmpegError::Other("ffmpeg produced no frame".into()));
    }
    Ok(bytes)
}

/// At most this many thumbnail strips are extracted at the same time.
const THUMB_CONCURRENCY: usize = 2;

struct CountingSemaphore {
    used: Mutex<usize>,
    cv: Condvar,
}

struct Permit(&'static CountingSemaphore);

impl Drop for Permit {
    fn drop(&mut self) {
        *self.0.used.lock().unwrap() -= 1;
        self.0.cv.notify_one();
    }
}

fn thumb_permit() -> Permit {
    static SEM: OnceLock<CountingSemaphore> = OnceLock::new();
    let sem = SEM.get_or_init(|| CountingSemaphore { used: Mutex::new(0), cv: Condvar::new() });
    let mut used = sem.used.lock().unwrap();
    while *used >= THUMB_CONCURRENCY {
        used = sem.cv.wait(used).unwrap();
    }
    *used += 1;
    Permit(sem)
}

/// Duration (ms) and whether the file is a still image, from a light ffprobe.
fn probe_duration(path: &Path) -> Result<(f64, bool)> {
    let bins = find_binaries()?;
    let args: Vec<String> = ["-v", "quiet", "-print_format", "json", "-show_entries", "format=duration,format_name"]
        .iter()
        .map(|s| s.to_string())
        .chain(std::iter::once(path.to_string_lossy().into_owned()))
        .collect();
    let out = run_capture(&bins.ffprobe, &args, PROBE_TIMEOUT)?;
    let v: serde_json::Value = serde_json::from_slice(&out).map_err(|e| FfmpegError::Parse(e.to_string()))?;
    let dur = v["format"]["duration"].as_str().and_then(|d| d.parse::<f64>().ok()).unwrap_or(0.0) * 1000.0;
    let fmt = v["format"]["format_name"].as_str().unwrap_or("");
    let still = IMAGE_EXTS.contains(&extension_lower(path).as_str()) || fmt.contains("image2") || fmt.ends_with("_pipe");
    Ok((dur, still))
}

/// Extract `count` evenly spaced thumbnails into the cache and return their paths
/// (`<thumbs_dir>/<sha1>/<n>.jpg`, n zero-based). Cached results are reused.
///
/// Only key frames are decoded (`-skip_frame nokey`: each thumbnail shows the
/// key frame at or before its bucket — several times faster than decoding every
/// frame), at most [`THUMB_CONCURRENCY`] strips are extracted at once, and a strip
/// is written to a temporary folder that is renamed into place only when complete,
/// so a killed ffmpeg never leaves truncated JPEGs in the cache.
pub fn extract_thumbnails(path: &Path, count: u32, width: u32) -> Result<Vec<PathBuf>> {
    extract_thumbnails_into(path, count, width, true, &thumbs_dir())
}

pub(crate) fn extract_thumbnails_into(path: &Path, count: u32, width: u32, keyframes_only: bool, root: &Path) -> Result<Vec<PathBuf>> {
    let count = count.clamp(1, 500);
    let width = width.clamp(16, 1024);
    let key = thumb_key(path, count, width);
    let dir = root.join(&key);
    let names: Vec<String> = (0..count).map(|i| format!("{i}.jpg")).collect();
    let expected: Vec<PathBuf> = names.iter().map(|n| dir.join(n)).collect();
    if expected.iter().all(|p| p.is_file()) {
        return Ok(expected);
    }
    let _permit = thumb_permit();
    if expected.iter().all(|p| p.is_file()) {
        return Ok(expected); // another request produced it meanwhile
    }
    let tmp = root.join(format!("{key}.tmp-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&tmp)?;
    let result = (|| -> Result<()> {
        let (duration_ms, still) = probe_duration(path)?;
        let bins = find_binaries()?;
        let duration_s = duration_ms / 1000.0;
        if still || duration_s <= 0.0 || count == 1 {
            // Single frame; replicate to satisfy the requested count.
            let ms = if duration_s > 0.0 { duration_ms * 0.5 } else { 0.0 };
            let bytes = extract_frame(path, ms, width)?;
            for n in &names {
                std::fs::write(tmp.join(n), &bytes)?;
            }
            return Ok(());
        }
        // One pass: sample `count` frames uniformly, centred in each bucket.
        let rate = count as f64 / duration_s;
        let mut args: Vec<String> = vec!["-v".into(), "error".into()];
        if keyframes_only {
            args.extend(["-skip_frame".into(), "nokey".into()]);
        }
        args.extend([
            "-i".into(),
            path.to_string_lossy().into_owned(),
            "-vf".into(),
            // eof_action=pass: the last bucket gets its frame without an extra seek
            format!("fps={rate:.6}:round=down:eof_action=pass,scale={width}:-2:flags=bicubic"),
            "-frames:v".into(),
            count.to_string(),
            "-q:v".into(),
            "5".into(),
            "-start_number".into(),
            "0".into(),
            tmp.join("%d.jpg").to_string_lossy().into_owned(),
        ]);
        run_capture(&bins.ffmpeg, &args, FFMPEG_TIMEOUT)?;
        // Fill any missing trailing frames (fps rounding) by seeking explicitly.
        for (i, n) in names.iter().enumerate() {
            let p = tmp.join(n);
            if !p.is_file() {
                let ms = duration_ms * (i as f64 + 0.5) / count as f64;
                let bytes = extract_frame(path, ms, width).or_else(|_| extract_frame(path, 0.0, width))?;
                std::fs::write(p, bytes)?;
            }
        }
        Ok(())
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(e);
    }
    // publish atomically (a stale partial folder from an older build is replaced)
    let _ = std::fs::remove_dir_all(&dir);
    if std::fs::rename(&tmp, &dir).is_err() {
        let _ = std::fs::remove_dir_all(&tmp); // a concurrent writer won the race
    }
    if expected.iter().all(|p| p.is_file()) {
        Ok(expected)
    } else {
        Err(FfmpegError::Other("thumbnail extraction produced no files".into()))
    }
}

/* --------------------------------------------------------------- waveform */

const WAVEFORM_RATE: u32 = 8000;

/// Decode the audio to mono 8 kHz PCM and return peak amplitude per bucket
/// (`samples_per_second` buckets per second), normalised to 0..1.
pub fn extract_waveform(path: &Path, samples_per_second: u32) -> Result<Vec<f32>> {
    let bins = find_binaries()?;
    let sps = samples_per_second.clamp(1, WAVEFORM_RATE);
    let bucket = (WAVEFORM_RATE / sps).max(1) as usize;
    let mut cmd = command(&bins.ffmpeg);
    cmd.args([
            "-v", "error",
            "-i", &path.to_string_lossy(),
            "-vn",
            "-f", "s16le",
            "-ac", "1",
            "-ar", &WAVEFORM_RATE.to_string(),
            "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    waveform_from_command(cmd, bucket)
}

/// Run a prepared waveform decoder (`s16le` mono on stdout) and bucket its peaks.
pub(crate) fn waveform_from_command(mut cmd: Command, bucket: usize) -> Result<Vec<f32>> {
    let mut child = crate::procs::spawn(&mut cmd)?;
    let mut stdout = child.stdout.take().ok_or_else(|| FfmpegError::Other("no stdout".into()))?;
    // Drain stderr concurrently: a corrupt file can print more than a pipe buffer of
    // errors, which would block ffmpeg (and this reader) forever if read only at the end.
    let stderr_thread = child.stderr.take().map(|mut e| {
        std::thread::spawn(move || {
            let mut s = String::new();
            let _ = e.read_to_string(&mut s);
            s
        })
    });
    let mut peaks: Vec<f32> = Vec::new();
    let mut buf = vec![0u8; 64 * 1024];
    let mut carry: Vec<u8> = Vec::new();
    let mut current_peak: i32 = 0;
    let mut in_bucket = 0usize;
    loop {
        let n = stdout.read(&mut buf)?;
        if n == 0 {
            break;
        }
        carry.extend_from_slice(&buf[..n]);
        let usable = carry.len() / 2 * 2;
        for s in carry[..usable].chunks_exact(2) {
            let v = i16::from_le_bytes([s[0], s[1]]) as i32;
            current_peak = current_peak.max(v.abs());
            in_bucket += 1;
            if in_bucket >= bucket {
                peaks.push(current_peak as f32 / 32768.0);
                current_peak = 0;
                in_bucket = 0;
            }
        }
        carry.drain(..usable);
    }
    if in_bucket > 0 {
        peaks.push(current_peak as f32 / 32768.0);
    }
    let status = child.wait()?;
    let err = stderr_thread.and_then(|t| t.join().ok()).unwrap_or_default();
    if !status.success() && peaks.is_empty() {
        let err: String = err.chars().rev().take(4000).collect::<Vec<_>>().into_iter().rev().collect();
        return Err(FfmpegError::Failed { program: "ffmpeg".into(), code: status.code(), stderr: err });
    }
    // Normalise so the loudest bucket hits 1.0 (keeps quiet clips readable).
    let max = peaks.iter().cloned().fold(0.0f32, f32::max);
    if max > 0.0 {
        for p in peaks.iter_mut() {
            *p = (*p / max).clamp(0.0, 1.0);
        }
    }
    Ok(peaks)
}

/* ------------------------------------------------------------------ tests */

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Generate a 2-second synthetic clip (testsrc + sine) in the cache dir.
    pub(crate) fn synth_clip() -> Option<PathBuf> {
        let bins = find_binaries().ok()?;
        let dir = cache_dir().join("test");
        std::fs::create_dir_all(&dir).ok()?;
        let out = dir.join("synth_2s.mp4");
        if out.is_file() {
            return Some(out);
        }
        // written under a unique name and renamed, so tests running in parallel never read a half-written file
        let tmp = dir.join(format!("synth_2s.{}.{:?}.tmp.mp4", std::process::id(), std::thread::current().id()).replace(['(', ')'], ""));
        let status = command(&bins.ffmpeg)
            .args([
                "-v", "error", "-y",
                "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=24",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest",
            ])
            .arg(&tmp)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .ok()?;
        if !status.success() {
            return None;
        }
        if std::fs::rename(&tmp, &out).is_err() {
            let _ = std::fs::remove_file(&tmp); // another test published it first
        }
        out.is_file().then_some(out)
    }

    /// Synthesise a fixture with ffmpeg (cached by name; unique temp + rename).
    pub(crate) fn synth_with(name: &str, args: &[&str]) -> Option<PathBuf> {
        let bins = find_binaries().ok()?;
        let dir = cache_dir().join("test");
        std::fs::create_dir_all(&dir).ok()?;
        let out = dir.join(name);
        if out.is_file() {
            return Some(out);
        }
        let ext = out.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_else(|| "mp4".into());
        let tmp = dir.join(format!("{name}.{}.{:?}.tmp.{ext}", std::process::id(), std::thread::current().id()).replace(['(', ')'], ""));
        let ok = crate::procs::output_timeout(command(&bins.ffmpeg).args(["-v", "error", "-y"]).args(args).arg(&tmp), Duration::from_secs(300))
            .ok()?
            .status
            .success();
        if !ok {
            let _ = std::fs::remove_file(&tmp);
            return None;
        }
        if std::fs::rename(&tmp, &out).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        out.is_file().then_some(out)
    }

    /// A 320x240 clip carrying a 90° display matrix (like a portrait phone clip).
    pub(crate) fn synth_rotated() -> Option<PathBuf> {
        let plain = synth_with(
            "rot_src_320x240.mp4",
            &["-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=2", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"],
        )?;
        let p = plain.to_string_lossy().into_owned();
        synth_with("rot90_320x240.mp4", &["-display_rotation", "90", "-i", &p, "-c", "copy"])
    }

    #[test]
    fn rotated_video_probes_with_display_size() {
        let Some(rot) = synth_rotated() else {
            eprintln!("SKIP: ffmpeg not available");
            return;
        };
        let a = probe(&rot).unwrap();
        assert_eq!((a.width, a.height), (240, 320), "a 90° clip is displayed portrait");
        let info = probe_video_info(&rot).unwrap();
        assert_eq!((info.width, info.height, info.rotation), (240, 320, 90));
        // ffmpeg auto-rotates decoded frames to the same size
        let bins = find_binaries().unwrap();
        let out = crate::procs::output_timeout(
            command(&bins.ffmpeg).args(["-v", "info", "-i"]).arg(&rot).args(["-frames:v", "1", "-vf", "showinfo", "-f", "null", "-"]),
            Duration::from_secs(60),
        )
        .unwrap();
        let log = String::from_utf8_lossy(&out.stderr);
        assert!(log.contains("s:240x320"), "decoded frames are 240x320: {log}");
    }

    #[test]
    fn colour_matrix_selection() {
        // untagged: BT.709 for HD, BT.601 for SD
        assert_eq!(input_color_matrix(None, Some("yuv420p"), 720), Some("bt709"));
        assert_eq!(input_color_matrix(None, Some("yuv420p"), 1080), Some("bt709"));
        assert_eq!(input_color_matrix(None, Some("yuv420p"), 480), Some("bt601"));
        // tags win
        assert_eq!(input_color_matrix(Some("smpte170m"), Some("yuv420p"), 1080), Some("bt601"));
        assert_eq!(input_color_matrix(Some("bt470bg"), None, 1080), Some("bt601"));
        assert_eq!(input_color_matrix(Some("bt709"), Some("yuv420p"), 480), Some("bt709"));
        assert_eq!(input_color_matrix(Some("bt2020nc"), Some("yuv420p10le"), 2160), Some("bt2020"));
        // RGB sources need no matrix
        assert_eq!(input_color_matrix(None, Some("rgb24"), 1080), None);
        assert_eq!(input_color_matrix(None, Some("gbrp"), 1080), None);
        assert_eq!(input_range(None, Some("yuv420p")), "tv");
        assert_eq!(input_range(Some("pc"), Some("yuv420p")), "pc");
        assert_eq!(input_range(None, Some("yuvj420p")), "pc");
    }

    /// An untagged 720p clip decodes with BT.709 (what the export is tagged with), not
    /// swscale's BT.601 default: a pure 709-encoded colour comes back unshifted.
    #[test]
    fn untagged_hd_video_decodes_with_bt709() {
        // Y'CbCr of sRGB (200, 60, 40) under BT.709, limited range, written without colour tags
        let Some(clip) = synth_with(
            "untagged_709_1280x720.mp4",
            &[
                "-f", "lavfi", "-i", "color=c=0xC83C28:size=1280x720:rate=24:duration=1",
                "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv444p,setparams=colorspace=unknown:range=unknown:color_primaries=unknown:color_trc=unknown",
                "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast",
            ],
        ) else {
            eprintln!("SKIP: ffmpeg not available");
            return;
        };
        let info = probe_video_info(&clip).unwrap();
        assert_eq!(info.color_space, None, "fixture must be untagged");
        let m = input_color_matrix(info.color_space.as_deref(), info.pix_fmt.as_deref(), info.height).unwrap();
        assert_eq!(m, "bt709");
        let bins = find_binaries().unwrap();
        let decode = |vf: &str| {
            let out = crate::procs::output_timeout(
                command(&bins.ffmpeg).args(["-v", "error", "-i"]).arg(&clip).args(["-frames:v", "1", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]),
                Duration::from_secs(60),
            )
            .unwrap();
            let px = &out.stdout[(360 * 1280 + 640) * 3..][..3];
            [px[0] as i32, px[1] as i32, px[2] as i32]
        };
        let ours = decode(&format!("scale=in_color_matrix={m}:in_range=tv,format=rgb24"));
        let naive = decode("format=rgb24");
        let err = |p: [i32; 3]| (p[0] - 200).abs().max((p[1] - 60).abs()).max((p[2] - 40).abs());
        eprintln!("bt709 decode {ours:?} (err {}), default decode {naive:?} (err {})", err(ours), err(naive));
        assert!(err(ours) <= 2, "{ours:?}");
        assert!(err(naive) > err(ours), "the default (BT.601) decode is off: {naive:?}");
    }

    /// A decoder flooding stderr must not deadlock the waveform reader.
    #[test]
    fn waveform_drains_stderr() {
        let Some(clip) = synth_clip() else {
            eprintln!("SKIP: ffmpeg not available");
            return;
        };
        let bins = find_binaries().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let clip2 = clip.clone();
        let ffmpeg = bins.ffmpeg.clone();
        std::thread::spawn(move || {
            let mut cmd = command(&ffmpeg);
            // `-v trace` prints far more than a pipe buffer (64 KB) to stderr
            cmd.args(["-v", "trace", "-i"]).arg(&clip2).args(["-vn", "-f", "s16le", "-ac", "1", "-ar", "8000", "pipe:1"]);
            cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
            let _ = tx.send(waveform_from_command(cmd, 400));
        });
        let peaks = rx.recv_timeout(Duration::from_secs(60)).expect("waveform extraction hung (stderr not drained)").unwrap();
        assert!(peaks.len() >= 35 && peaks.iter().any(|p| *p > 0.5), "{}", peaks.len());
    }

    /// Key-frame-only thumbnails are much cheaper than decoding every frame.
    #[test]
    #[ignore = "timing benchmark: flaky when the whole suite runs in parallel; run with --ignored"]
    fn thumbnails_decode_key_frames_only() {
        // 1080p30 with B-frames and a key frame every 2 s (like camera / phone footage)
        let Some(clip) = synth_with(
            "thumbs_1080p_40s.mp4",
            &["-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30:duration=40", "-c:v", "libx264", "-preset", "veryfast", "-g", "60", "-pix_fmt", "yuv420p"],
        ) else {
            eprintln!("SKIP: ffmpeg not available");
            return;
        };
        let root = cache_dir().join("test").join("thumb_speed");
        // best of three runs each (the test binary runs other tests in parallel)
        let time = |key_only: bool| {
            (0..3)
                .map(|_| {
                    let r = root.join(if key_only { "key" } else { "full" });
                    let _ = std::fs::remove_dir_all(&r);
                    std::fs::create_dir_all(&r).unwrap();
                    let t = std::time::Instant::now();
                    let files = extract_thumbnails_into(&clip, 12, 160, key_only, &r).unwrap();
                    let secs = t.elapsed().as_secs_f64();
                    assert_eq!(files.len(), 12);
                    assert!(files.iter().all(|f| std::fs::metadata(f).map(|m| m.len() > 500).unwrap_or(false)));
                    assert!(std::fs::read_dir(&r).unwrap().flatten().all(|e| !e.file_name().to_string_lossy().contains(".tmp-")), "no temp folders left");
                    secs
                })
                .fold(f64::INFINITY, f64::min)
        };
        let full = time(false);
        let key = time(true);
        eprintln!("thumbnail strip of a 40 s 1080p clip: all frames {full:.2}s, key frames only {key:.2}s ({:.1}x)", full / key.max(1e-6));
        assert!(key < full * 0.6, "key-frame decode must be clearly faster: {key:.2}s vs {full:.2}s");
    }

    #[test]
    fn parses_fractions() {
        assert!((parse_fraction("24000/1001").unwrap() - 23.976).abs() < 1e-3);
        assert_eq!(parse_fraction("25"), Some(25.0));
        assert_eq!(parse_fraction("0/0"), None);
        assert_eq!(parse_fraction(""), None);
    }

    #[test]
    fn thumb_key_is_stable_and_hex() {
        let a = thumb_key(Path::new("C:/nonexistent.mp4"), 10, 160);
        let b = thumb_key(Path::new("C:/nonexistent.mp4"), 10, 160);
        assert_eq!(a, b);
        assert_eq!(a.len(), 40);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, thumb_key(Path::new("C:/nonexistent.mp4"), 11, 160));
    }

    #[test]
    fn finds_ffmpeg_binaries() {
        match find_binaries() {
            Ok(p) => {
                assert!(p.ffmpeg.is_file());
                assert!(p.ffprobe.is_file());
            }
            Err(e) => eprintln!("SKIP: {e}"),
        }
    }

    #[test]
    fn lut_asset_needs_no_probe() {
        let dir = cache_dir().join("test");
        std::fs::create_dir_all(&dir).unwrap();
        let lut = dir.join("look.cube");
        std::fs::write(&lut, "TITLE \"x\"\nLUT_3D_SIZE 2\n").unwrap();
        let a = probe(&lut).unwrap();
        assert_eq!(a.kind, AssetKind::Lut);
        assert_eq!(a.name, "look.cube");
    }

    #[test]
    fn probe_frame_thumbs_waveform_on_synthetic_clip() {
        let Some(clip) = synth_clip() else {
            eprintln!("SKIP: ffmpeg not available");
            return;
        };
        let asset = probe(&clip).expect("probe");
        assert_eq!(asset.kind, AssetKind::Video);
        assert_eq!((asset.width, asset.height), (320, 240));
        assert!((asset.fps - 24.0).abs() < 0.01, "fps={}", asset.fps);
        assert!(asset.duration_ms > 1800.0 && asset.duration_ms < 2300.0, "duration={}", asset.duration_ms);
        assert!(asset.has_audio);
        assert_eq!(asset.codec.as_deref(), Some("h264"));
        assert!(asset.id.starts_with("ast_"));

        let jpeg = extract_frame(&clip, 1000.0, 160).expect("frame");
        assert!(jpeg.len() > 100 && jpeg[0] == 0xFF && jpeg[1] == 0xD8, "must be JPEG");

        let thumbs = extract_thumbnails(&clip, 4, 96).expect("thumbs");
        assert_eq!(thumbs.len(), 4);
        assert!(thumbs.iter().all(|p| p.is_file()));
        // second call hits the cache
        let again = extract_thumbnails(&clip, 4, 96).expect("thumbs cached");
        assert_eq!(again, thumbs);

        let wave = extract_waveform(&clip, 20).expect("waveform");
        assert!(wave.len() >= 35 && wave.len() <= 45, "len={}", wave.len());
        assert!(wave.iter().all(|v| (0.0..=1.0).contains(v)));
        assert!(wave.iter().any(|v| *v > 0.5));
    }
}
