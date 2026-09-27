//! `scan_clips_folder`: list the media in a clips folder in edit order.
//!
//! The order comes from the Python pipeline (`python -m cappycat_pipeline
//! order <folder> --json`, which prints one JSON object
//! `{ folder, files: [{ path, name, order, reason, key }], warnings }`). When
//! Python is unavailable the files are sorted naturally by name in Rust.
//! Every file is then probed with [`crate::ffmpeg::probe`]; files that fail to
//! probe are reported in `warnings` rather than failing the command.

use crate::ffmpeg::{self, extension_lower, is_media_ext};
use crate::jobs::{find_python, python_command};
use crate::model::Asset;
use crate::pipeline::find_pipeline_dir;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Upper bound for `python -m cappycat_pipeline order`.
pub const ORDER_TIMEOUT: Duration = Duration::from_secs(60);

pub const FALLBACK_REASON: &str = "natural filename sort (pipeline unavailable)";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipsFolderScan {
    pub folder: String,
    pub assets: Vec<Asset>,
    pub warnings: Vec<String>,
}

/// The repository root when running from a checkout (see [`crate::paths`]).
pub fn repo_dir() -> Option<PathBuf> {
    crate::paths::repo_root()
}

/// Default clips folder: [`discover_clips_dir`] in a checkout, `Documents\Cappycat\Clips` when
/// installed (see [`crate::paths`]). Does not create anything.
pub fn default_clips_dir() -> Option<PathBuf> {
    Some(crate::paths::resolver().clips_dir())
}

const SKIP_DIRS: &[&str] = &["node_modules", "target", "dist", "pipeline", "src", "src-tauri", ".git"];

fn has_video_files(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten().any(|e| {
                let p = e.path();
                p.is_file() && ffmpeg::VIDEO_EXTS.contains(&extension_lower(&p).as_str())
            })
        })
        .unwrap_or(false)
}

/// Pick the clips folder inside `repo`:
/// 1. `<repo>/clips` if it holds at least one video file;
/// 2. else the most recently modified direct child whose name contains
///    "clip" (case-insensitive, e.g. `cappycat- clips`), skipping
///    `node_modules`, `target`, `dist`, `pipeline`, `src`, `src-tauri`, `.git`,
///    that holds video files;
/// 3. else `<repo>/clips` (created by [`scan_clips_folder`]).
pub fn discover_clips_dir(repo: &Path) -> PathBuf {
    let plain = repo.join("clips");
    if has_video_files(&plain) {
        return plain;
    }
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    if let Ok(rd) = std::fs::read_dir(repo) {
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().to_lowercase();
            if !p.is_dir() || !name.contains("clip") || SKIP_DIRS.contains(&name.as_str()) || !has_video_files(&p) {
                continue;
            }
            // stat the folder itself: on Windows the DirEntry copy of a directory's mtime is updated lazily
            let mtime = std::fs::metadata(&p).and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
            if best.as_ref().map(|(t, _)| mtime > *t).unwrap_or(true) {
                best = Some((mtime, p));
            }
        }
    }
    best.map(|(_, p)| p).unwrap_or(plain)
}

#[derive(Debug, Deserialize)]
struct OrderOutput {
    #[serde(default)]
    files: Vec<OrderFile>,
    #[serde(default)]
    warnings: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct OrderFile {
    path: String,
    #[serde(default)]
    order: Option<i64>,
    #[serde(default)]
    reason: Option<String>,
}

/// Compare two names "naturally": digit runs by numeric value, text case-insensitively.
pub fn natural_cmp(a: &str, b: &str) -> Ordering {
    fn chunks(s: &str) -> Vec<(bool, String)> {
        let mut out: Vec<(bool, String)> = Vec::new();
        for ch in s.chars() {
            let d = ch.is_ascii_digit();
            match out.last_mut() {
                Some((is_d, buf)) if *is_d == d => buf.push(ch),
                _ => out.push((d, ch.to_string())),
            }
        }
        out
    }
    let (ca, cb) = (chunks(a), chunks(b));
    for (x, y) in ca.iter().zip(cb.iter()) {
        let o = match (x.0, y.0) {
            (true, true) => {
                let xs = x.1.trim_start_matches('0');
                let ys = y.1.trim_start_matches('0');
                xs.len().cmp(&ys.len()).then_with(|| xs.cmp(ys)).then_with(|| x.1.len().cmp(&y.1.len()))
            }
            _ => x.1.to_lowercase().cmp(&y.1.to_lowercase()),
        };
        if o != Ordering::Equal {
            return o;
        }
    }
    ca.len().cmp(&cb.len()).then_with(|| a.cmp(b))
}

fn natural_listing(folder: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(folder)
        .map_err(|e| format!("cannot read {}: {e}", folder.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && is_media_ext(&extension_lower(p)))
        .collect();
    files.sort_by(|a, b| {
        let na = a.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let nb = b.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        natural_cmp(&na, &nb)
    });
    Ok(files)
}

/// Extract the single JSON object from the command's stdout (tolerating log lines around it).
fn parse_order_stdout(stdout: &str) -> Result<OrderOutput, String> {
    if let Ok(v) = serde_json::from_str::<OrderOutput>(stdout.trim()) {
        return Ok(v);
    }
    for line in stdout.lines().rev() {
        let l = line.trim();
        if l.starts_with('{') {
            if let Ok(v) = serde_json::from_str::<OrderOutput>(l) {
                return Ok(v);
            }
        }
    }
    let start = stdout.find('{').ok_or("no JSON object in the output")?;
    let end = stdout.rfind('}').ok_or("no JSON object in the output")?;
    serde_json::from_str(&stdout[start..=end]).map_err(|e| format!("invalid JSON from `order`: {e}"))
}

fn run_pipeline_order(folder: &Path) -> Result<OrderOutput, String> {
    let pipeline_dir = find_pipeline_dir().ok_or("pipeline directory not found")?;
    let python = find_python(Some(&pipeline_dir)).ok_or("python interpreter not found")?;
    let mut cmd = python_command(&python, &pipeline_dir);
    cmd.args(["-m", "cappycat_pipeline", "order"]).arg(folder).arg("--json");
    let out = crate::procs::output_python(&mut cmd, ORDER_TIMEOUT).map_err(|e| match e.kind() {
        std::io::ErrorKind::TimedOut => format!("`order` did not finish within {} s", ORDER_TIMEOUT.as_secs()),
        _ => format!("cannot start {}: {e}", python.display()),
    })?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!("`order` exited with {:?}: {}", out.status.code(), err.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("").trim()));
    }
    parse_order_stdout(&String::from_utf8_lossy(&out.stdout))
}

/// Scan `path` (default `<repo>/clips`, created if missing).
pub fn scan_clips_folder(path: Option<&str>) -> Result<ClipsFolderScan, String> {
    let folder = match path.map(str::trim).filter(|p| !p.is_empty()) {
        Some(p) => {
            let p = PathBuf::from(p);
            if !p.is_absolute() {
                return Err(format!("path must be absolute: {}", p.display()));
            }
            if !p.is_dir() {
                return Err(format!("not a folder: {}", p.display()));
            }
            p
        }
        None => {
            let d = crate::paths::resolver().clips_dir();
            std::fs::create_dir_all(&d).map_err(|e| format!("cannot create {}: {e}", d.display()))?;
            d
        }
    };
    scan_folder_with(&folder, |f| run_pipeline_order(f).map(OrderOutputData::from))
}

/// Implementation with an injectable ordering step (tests).
pub fn scan_folder_with(folder: &Path, order: impl FnOnce(&Path) -> Result<OrderOutputData, String>) -> Result<ClipsFolderScan, String> {
    let mut warnings = Vec::new();
    let ordered: Vec<(PathBuf, String)> = match order(folder) {
        Ok(o) => {
            warnings.extend(o.warnings);
            let mut files = o.files;
            files.sort_by_key(|f| f.order.unwrap_or(i64::MAX));
            files
                .into_iter()
                .map(|f| (PathBuf::from(&f.path), f.reason.unwrap_or_else(|| "pipeline order".into())))
                .map(|(p, r)| (if p.is_absolute() { p } else { folder.join(p) }, r))
                .collect()
        }
        Err(e) => {
            warnings.push(format!("pipeline unavailable - using natural filename sort ({e})"));
            natural_listing(folder)?.into_iter().map(|p| (p, FALLBACK_REASON.to_string())).collect()
        }
    };
    let ordered: Vec<(PathBuf, String)> = ordered.into_iter().filter(|(p, _)| is_media_ext(&extension_lower(p))).collect();
    let paths: Vec<PathBuf> = ordered.iter().map(|(p, _)| p.clone()).collect();
    let mut assets = Vec::new();
    // probed in parallel, kept in edit order
    for ((path, reason), probed) in ordered.into_iter().zip(ffmpeg::probe_many(&paths)) {
        match probed {
            Ok(mut a) => {
                a.order = Some(assets.len() as u32);
                a.order_reason = Some(reason);
                assets.push(a);
            }
            Err(e) => warnings.push(format!("{}: {e}", path.display())),
        }
    }
    Ok(ClipsFolderScan { folder: pipeline_path_string(folder), assets, warnings })
}

fn pipeline_path_string(p: &Path) -> String {
    crate::pipeline::strip_verbatim(p.to_path_buf()).to_string_lossy().into_owned()
}

/// Ordering result handed to [`scan_folder_with`].
#[derive(Debug, Default)]
pub struct OrderOutputData {
    pub files: Vec<OrderFileData>,
    pub warnings: Vec<String>,
}

#[derive(Debug)]
pub struct OrderFileData {
    pub path: String,
    pub order: Option<i64>,
    pub reason: Option<String>,
}

impl From<OrderOutput> for OrderOutputData {
    fn from(o: OrderOutput) -> Self {
        Self {
            files: o.files.into_iter().map(|f| OrderFileData { path: f.path, order: f.order, reason: f.reason }).collect(),
            warnings: o.warnings,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_the_clips_folder() {
        let repo = ffmpeg::cache_dir().join("test").join("clips_discovery repo");
        let _ = std::fs::remove_dir_all(&repo);
        for d in ["clips", "characters", "node_modules", "old_clips", "cappycat- clips", "Clips notes"] {
            std::fs::create_dir_all(repo.join(d)).unwrap();
        }
        // nothing with video yet → <repo>/clips (to be created)
        assert_eq!(discover_clips_dir(&repo), repo.join("clips"));
        std::fs::write(repo.join("characters").join("ref.mp4"), b"x").unwrap(); // no "clip" in the name
        std::fs::write(repo.join("node_modules").join("clip.mp4"), b"x").unwrap();
        std::fs::write(repo.join("Clips notes").join("readme.txt"), b"x").unwrap(); // no video
        std::fs::write(repo.join("old_clips").join("a.mov"), b"x").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(60));
        std::fs::write(repo.join("cappycat- clips").join("clip1.mp4"), b"x").unwrap();
        assert_eq!(discover_clips_dir(&repo), repo.join("cappycat- clips"), "newest folder named *clip* with videos");
        // <repo>/clips wins as soon as it holds a video
        std::fs::write(repo.join("clips").join("x.mkv"), b"x").unwrap();
        assert_eq!(discover_clips_dir(&repo), repo.join("clips"));
        // folders with spaces scan fine (fallback path, .cube needs no ffmpeg)
        let spaced = repo.join("cappycat- clips");
        std::fs::write(spaced.join("my look 1.cube"), "LUT_3D_SIZE 2\n").unwrap();
        let scan = scan_folder_with(&spaced, |_| Err("offline".into())).unwrap();
        assert!(scan.assets.iter().any(|a| a.name == "my look 1.cube"));
        assert!(scan.folder.ends_with("cappycat- clips"));
    }

    #[test]
    fn natural_sort() {
        let mut v = vec!["clip10.mp4", "Clip2.mp4", "clip1.mp4", "clip02b.mp4", "a.mp4"];
        v.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(v, vec!["a.mp4", "clip1.mp4", "Clip2.mp4", "clip02b.mp4", "clip10.mp4"]);
    }

    #[test]
    fn parses_order_json_with_noise() {
        let out = "loading...\n{\"folder\": \"C:/c\", \"files\": [{\"path\": \"C:/c/b.mp4\", \"name\": \"b.mp4\", \"order\": 0, \"reason\": \"leading number 01\", \"key\": [1]}], \"warnings\": [\"dup\"]}\n";
        let o = parse_order_stdout(out).unwrap();
        assert_eq!(o.files.len(), 1);
        assert_eq!(o.files[0].reason.as_deref(), Some("leading number 01"));
        assert_eq!(o.warnings, vec!["dup"]);
    }

    #[test]
    fn scan_uses_pipeline_order_and_falls_back() {
        let dir = ffmpeg::cache_dir().join("test").join("clips_scan");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // .cube files need no ffmpeg; a bogus .mp4 fails to probe → warning.
        for n in ["look10.cube", "look2.cube", "notes.txt", "broken.mp4"] {
            std::fs::write(dir.join(n), "LUT_3D_SIZE 2\n").unwrap();
        }
        let fallback = scan_folder_with(&dir, |_| Err("python missing".into())).unwrap();
        let names: Vec<&str> = fallback.assets.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, vec!["look2.cube", "look10.cube"]);
        assert_eq!(fallback.assets[1].order, Some(1));
        assert_eq!(fallback.assets[0].order_reason.as_deref(), Some(FALLBACK_REASON));
        assert!(fallback.warnings[0].starts_with("pipeline unavailable - using natural filename sort"));
        assert!(fallback.warnings.iter().any(|w| w.contains("broken.mp4")), "{:?}", fallback.warnings);

        let d2 = dir.clone();
        let piped = scan_folder_with(&dir, move |_| {
            Ok(OrderOutputData {
                files: vec![
                    OrderFileData { path: d2.join("look2.cube").to_string_lossy().into(), order: Some(1), reason: Some("b".into()) },
                    OrderFileData { path: "look10.cube".into(), order: Some(0), reason: Some("a".into()) },
                    OrderFileData { path: d2.join("notes.txt").to_string_lossy().into(), order: Some(2), reason: None },
                ],
                warnings: vec!["w".into()],
            })
        })
        .unwrap();
        let names: Vec<&str> = piped.assets.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, vec!["look10.cube", "look2.cube"]);
        assert_eq!(piped.assets[0].order_reason.as_deref(), Some("a"));
        assert_eq!(piped.warnings, vec!["w"]);
        let v = serde_json::to_value(&piped).unwrap();
        assert_eq!(v["assets"][0]["orderReason"], "a");
    }
}
