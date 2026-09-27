//! Local HTTP media server.
//!
//! Bound to `127.0.0.1:<random port>` at startup. It exists so the webview can
//! play media through ordinary `<video>` / `<audio>` / `<img>` elements (with
//! seeking) without shovelling bytes through Tauri IPC.
//!
//! ## Security model
//!
//! Any web page the user opens could otherwise talk to this port, so:
//! * every route lives under a random per-launch secret prefix
//!   `/t/<token>/` (32 hex chars); anything else is `404`. The base URL
//!   (`http://127.0.0.1:<port>/t/<token>`) is only handed to the app's own
//!   webview (`media_server_url`, `app_paths().mediaServerUrl`);
//! * the `Host` header must be `127.0.0.1:<port>` (DNS-rebinding guard) → `403`;
//! * CORS is granted only to the app's origins (`tauri://localhost`,
//!   `http(s)://tauri.localhost`, the dev server `http://localhost:1420`);
//! * only **registered** files are served ([`MediaRegistry`], filled by
//!   `probe_media`, `import_media`, `scan_clips_folder`, `load_project` /
//!   `open_document`, `pipeline://analysis`, `separate://result`, successful
//!   exports and `register_media`) → `403` otherwise. Paths are compared
//!   canonicalised and (on Windows) case-insensitively;
//! * UNC / device paths (`\\host\share`, `\\?\…`, `//…`) and `..` are
//!   rejected (`400`) before touching the file system.
//!
//! Routes (below the prefix):
//! * `GET /media?path=<absolute path>` — byte-range streaming (`206`,
//!   `Accept-Ranges`, `Content-Range`, `Content-Type` by extension, `ETag` +
//!   `Last-Modified`, `Cache-Control: no-cache`, `304` on a matching
//!   `If-None-Match`). A malformed `Range` is ignored (`200`).
//! * `GET /frame?path=<abs>&ms=<n>&w=<n>` — single JPEG frame via ffmpeg (at
//!   most 3 at a time, small LRU cache, ffmpeg killed when the client goes away).
//! * `GET /thumb/<sha1>/<n>.jpg` — cached thumbnails from
//!   `%LOCALAPPDATA%\cappycat\cache\thumbs\<sha1>\<n>.jpg`.
//! * `GET /health` — `"ok"`.

use crate::ffmpeg;
use crate::model::{AnalysisResult, Project};
use axum::{
    body::Body,
    extract::{Path as AxPath, Query, Request, State},
    http::{header, HeaderMap, HeaderValue, Method, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use serde::Deserialize;
use std::collections::{HashSet, VecDeque};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tokio_util::io::ReaderStream;
use tower_http::cors::{AllowOrigin, CorsLayer};

/// Origins allowed to read media cross-origin (WebGL textures need CORS).
pub const APP_ORIGINS: &[&str] = &["tauri://localhost", "http://tauri.localhost", "https://tauri.localhost", "http://localhost:1420"];

/* --------------------------------------------------------------- registry */

/// Is `raw` a UNC / device / verbatim path (`\\host\share`, `\\?\C:\`, `\\.\pipe`, `//host`)?
pub fn is_unc_like(raw: &str) -> bool {
    let r = raw.trim();
    r.starts_with("\\\\") || r.starts_with("//") || r.starts_with("\\/") || r.starts_with("/\\")
}

/// Comparison key of a path: canonicalised when it exists (lexically cleaned
/// otherwise), backslashes, lower-case on Windows.
pub fn path_key(p: &Path) -> String {
    let abs = dunce::canonicalize(p).unwrap_or_else(|_| {
        let mut out = PathBuf::new();
        for c in p.components() {
            match c {
                Component::CurDir => {}
                Component::ParentDir => {
                    out.pop();
                }
                other => out.push(other.as_os_str()),
            }
        }
        out
    });
    let s = abs.to_string_lossy().into_owned();
    if cfg!(windows) {
        s.replace('/', "\\").to_lowercase()
    } else {
        s
    }
}

/// The set of files the media server may serve.
#[derive(Clone, Default)]
pub struct MediaRegistry {
    keys: Arc<RwLock<HashSet<String>>>,
}

impl MediaRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one path (empty strings are ignored).
    pub fn register(&self, path: impl AsRef<Path>) {
        let p = path.as_ref();
        if p.as_os_str().is_empty() {
            return;
        }
        let key = path_key(p);
        self.keys.write().unwrap().insert(key);
    }

    pub fn register_all<I, P>(&self, paths: I)
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        for p in paths {
            self.register(p);
        }
    }

    /// Every asset (incl. LUTs) and stem of a project.
    pub fn register_project(&self, project: &Project) {
        for a in &project.assets {
            self.register_asset(a);
        }
    }

    pub fn register_asset(&self, a: &crate::model::Asset) {
        self.register(&a.path);
        if let Some(s) = &a.stems {
            self.register(&s.vocals);
            self.register(&s.background);
        }
    }

    /// Every clip, clip asset (+ stems) and the timeline of an analysis.
    pub fn register_analysis(&self, analysis: &AnalysisResult) {
        for c in &analysis.clips {
            self.register(&c.path);
            self.register_asset(&c.asset);
        }
        self.register_project(&analysis.timeline);
    }

    pub fn contains(&self, path: &Path) -> bool {
        self.contains_key(&path_key(path))
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.keys.read().unwrap().contains(key)
    }

    pub fn len(&self) -> usize {
        self.keys.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/* ------------------------------------------------------------------ state */

#[derive(Clone)]
pub struct MediaState {
    pub thumbs_root: Arc<PathBuf>,
    /// per-launch secret path prefix (32 hex chars)
    pub token: Arc<String>,
    /// bound port (0 until [`start`] bound the socket); the `Host` check uses it
    port: Arc<AtomicU16>,
    pub registry: MediaRegistry,
    frame_permits: Arc<Semaphore>,
    frame_cache: Arc<Mutex<FrameCache>>,
}

/// Recently served `/frame` JPEGs (key, bytes), oldest first.
type FrameCache = VecDeque<(String, Arc<Vec<u8>>)>;
/// A request rejected before any work: status + message.
type Reject = (StatusCode, String);

const FRAME_CACHE_ENTRIES: usize = 64;
const FRAME_CONCURRENCY: usize = 3;

impl MediaState {
    pub fn new(thumbs_root: PathBuf, registry: MediaRegistry) -> Self {
        Self {
            thumbs_root: Arc::new(thumbs_root),
            token: Arc::new(uuid::Uuid::new_v4().simple().to_string()),
            port: Arc::new(AtomicU16::new(0)),
            registry,
            frame_permits: Arc::new(Semaphore::new(FRAME_CONCURRENCY)),
            frame_cache: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    pub fn port(&self) -> u16 {
        self.port.load(Ordering::SeqCst)
    }

    pub fn set_port(&self, port: u16) {
        self.port.store(port, Ordering::SeqCst);
    }

    /// `http://127.0.0.1:<port>/t/<token>` — the base the frontend appends `/media?path=…` to.
    pub fn base_url(&self) -> String {
        base_url(self.port(), &self.token)
    }
}

pub fn base_url(port: u16, token: &str) -> String {
    format!("http://127.0.0.1:{port}/t/{token}")
}

/// Build the router (separate from `start` so tests can drive it in-process).
pub fn router(state: MediaState) -> Router {
    let origins: Vec<HeaderValue> = APP_ORIGINS.iter().map(|o| HeaderValue::from_static(o)).collect();
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods([Method::GET, Method::HEAD, Method::OPTIONS])
        .allow_headers([header::RANGE, header::IF_NONE_MATCH, header::IF_MODIFIED_SINCE])
        .expose_headers([header::CONTENT_RANGE, header::CONTENT_LENGTH, header::ACCEPT_RANGES, header::ETAG, header::LAST_MODIFIED]);
    let inner = Router::new()
        .route("/media", get(media))
        .route("/frame", get(frame))
        .route("/thumb/{sha}/{file}", get(thumb))
        .route("/health", get(|| async { "ok" }));
    Router::new()
        .nest(&format!("/t/{}", state.token), inner)
        .fallback(|| async { (StatusCode::NOT_FOUND, "not found") })
        .layer(middleware::from_fn_with_state(state.clone(), host_guard))
        .layer(cors)
        .with_state(state)
}

/// Reject requests whose `Host` is not exactly `127.0.0.1:<port>` (DNS rebinding).
async fn host_guard(State(state): State<MediaState>, req: Request, next: Next) -> Response {
    let want = format!("127.0.0.1:{}", state.port());
    let host = req.headers().get(header::HOST).and_then(|h| h.to_str().ok()).map(|h| h.trim().to_ascii_lowercase());
    if host.as_deref() != Some(want.as_str()) {
        return (StatusCode::FORBIDDEN, "forbidden host").into_response();
    }
    next.run(req).await
}

/// Bind `127.0.0.1:0` on a dedicated runtime thread and return the port.
pub fn start(state: MediaState) -> std::io::Result<u16> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .max_blocking_threads(16)
        .thread_name("cappycat-media")
        .enable_all()
        .build()?;
    let listener = rt.block_on(TcpListener::bind(("127.0.0.1", 0)))?;
    let port = listener.local_addr()?.port();
    state.set_port(port);
    let app = router(state);
    std::thread::Builder::new()
        .name("cappycat-media-server".into())
        .spawn(move || {
            rt.block_on(async move {
                if let Err(e) = axum::serve(listener, app).await {
                    tracing::error!("media server stopped: {e}");
                }
            });
        })?;
    tracing::info!("media server listening on http://127.0.0.1:{port}/t/<token>");
    Ok(port)
}

/* ------------------------------------------------------------- validation */

/// Accept only absolute local paths without `..` components (no UNC / device paths).
pub fn validate_path(raw: &str) -> Result<PathBuf, Reject> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(bad_request("missing path"));
    }
    if is_unc_like(raw) {
        return Err(bad_request("UNC and device paths are not served"));
    }
    let p = PathBuf::from(raw);
    if !p.is_absolute() {
        return Err(bad_request("path must be absolute"));
    }
    if p.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(bad_request("path must not contain '..'"));
    }
    Ok(p)
}

/// Validate, canonicalise (off the async workers) and check the registry.
async fn authorised_path(state: &MediaState, raw: &str) -> Result<(PathBuf, std::fs::Metadata), Reject> {
    let p = validate_path(raw)?;
    let registry = state.registry.clone();
    let checked = tokio::task::spawn_blocking(move || {
        let meta = std::fs::metadata(&p).ok().filter(|m| m.is_file());
        let key = path_key(&p);
        (p, meta, registry.contains_key(&key))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    match checked {
        (_, _, false) => Err((StatusCode::FORBIDDEN, "path is not registered with the media server".into())),
        (_, None, true) => Err((StatusCode::NOT_FOUND, "file not found".into())),
        (p, Some(m), true) => Ok((p, m)),
    }
}

fn bad_request(msg: &str) -> Reject {
    (StatusCode::BAD_REQUEST, msg.to_string())
}

fn content_type_for(path: &Path) -> HeaderValue {
    let mime = mime_guess::from_path(path).first_or_octet_stream();
    // Some MOV/MKV variants are not in the mime db; give the browser a hint.
    let s = match ffmpeg::extension_lower(path).as_str() {
        "mkv" => "video/x-matroska".to_string(),
        "mov" => "video/quicktime".to_string(),
        "m4v" => "video/mp4".to_string(),
        _ => mime.essence_str().to_string(),
    };
    HeaderValue::from_str(&s).unwrap_or_else(|_| HeaderValue::from_static("application/octet-stream"))
}

/// RFC 7231 IMF-fixdate, e.g. `Tue, 15 Nov 1994 08:12:31 GMT`.
pub fn http_date(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // civil-from-days (H. Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    const WD: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];
    const MON: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    format!(
        "{}, {:02} {} {} {:02}:{:02}:{:02} GMT",
        WD[days.rem_euclid(7) as usize],
        d,
        MON[(m - 1) as usize],
        y,
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

fn etag_for(meta: &std::fs::Metadata) -> String {
    let mtime = meta.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_nanos()).unwrap_or(0);
    format!("\"{:x}-{:x}\"", meta.len(), mtime)
}

/* ------------------------------------------------------------------ range */

/// A well-formed `Range` that does not overlap the resource (→ `416`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unsatisfiable;

/// Parsed byte range (inclusive end).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    pub start: u64,
    pub end: u64,
}

/// Parse a `Range: bytes=…` header against a resource of `len` bytes.
/// Returns `Ok(None)` when no usable header is present — including malformed
/// ones, which RFC 7233 says to ignore — and `Err(())` when a well-formed
/// range is unsatisfiable. Only the first range of a multi-range request is honoured.
pub fn parse_range(header: Option<&str>, len: u64) -> Result<Option<ByteRange>, Unsatisfiable> {
    let Some(h) = header else { return Ok(None) };
    let Some(spec) = h.trim().strip_prefix("bytes=") else { return Ok(None) };
    let first = spec.split(',').next().unwrap_or("").trim();
    let Some((a, b)) = first.split_once('-') else { return Ok(None) };
    let (a, b) = (a.trim(), b.trim());
    let num = |s: &str| -> Option<u64> { (!s.is_empty() && s.bytes().all(|c| c.is_ascii_digit())).then(|| s.parse().ok()).flatten() };
    if a.is_empty() {
        // suffix: last N bytes
        let Some(n) = num(b) else { return Ok(None) };
        if n == 0 || len == 0 {
            return Err(Unsatisfiable);
        }
        let n = n.min(len);
        return Ok(Some(ByteRange { start: len - n, end: len - 1 }));
    }
    let Some(start) = num(a) else { return Ok(None) };
    let end = if b.is_empty() {
        None
    } else {
        match num(b) {
            Some(e) if e >= start => Some(e),
            _ => return Ok(None), // syntactically invalid (last < first) → ignore
        }
    };
    if start >= len {
        return Err(Unsatisfiable);
    }
    Ok(Some(ByteRange { start, end: end.unwrap_or(len - 1).min(len - 1) }))
}

#[derive(Deserialize)]
pub struct MediaQuery {
    pub path: String,
}

async fn media(State(state): State<MediaState>, method: Method, headers: HeaderMap, Query(q): Query<MediaQuery>) -> Response {
    let (path, meta) = match authorised_path(&state, &q.path).await {
        Ok(v) => v,
        Err(r) => return r.into_response(),
    };
    let len = meta.len();
    let etag = etag_for(&meta);
    let last_modified = meta.modified().ok().map(http_date);
    let validators = |mut b: axum::http::response::Builder| {
        b = b.header(header::ETAG, &etag).header(header::CACHE_CONTROL, "no-cache").header(header::ACCEPT_RANGES, "bytes");
        if let Some(lm) = &last_modified {
            b = b.header(header::LAST_MODIFIED, lm);
        }
        b
    };
    if let Some(inm) = headers.get(header::IF_NONE_MATCH).and_then(|v| v.to_str().ok()) {
        if inm.split(',').any(|t| t.trim().trim_start_matches("W/") == etag || t.trim() == "*") {
            return validators(Response::builder().status(StatusCode::NOT_MODIFIED)).body(Body::empty()).unwrap();
        }
    }
    let range_header = headers.get(header::RANGE).and_then(|v| v.to_str().ok());
    // If-Range with a stale validator → serve the full, current file
    let if_range_ok = headers.get(header::IF_RANGE).and_then(|v| v.to_str().ok()).map(|v| v.trim() == etag || Some(v.trim()) == last_modified.as_deref()).unwrap_or(true);
    let range = match parse_range(range_header.filter(|_| if_range_ok), len) {
        Ok(r) => r,
        Err(Unsatisfiable) => {
            return validators(Response::builder().status(StatusCode::RANGE_NOT_SATISFIABLE))
                .header(header::CONTENT_RANGE, format!("bytes */{len}"))
                .body(Body::empty())
                .unwrap();
        }
    };
    let (status, start, count) = match range {
        Some(r) => (StatusCode::PARTIAL_CONTENT, r.start, r.end - r.start + 1),
        None => (StatusCode::OK, 0, len),
    };
    let mut builder = validators(Response::builder().status(status))
        .header(header::CONTENT_TYPE, content_type_for(&path))
        .header(header::CONTENT_LENGTH, count);
    if let Some(r) = range {
        builder = builder.header(header::CONTENT_RANGE, format!("bytes {}-{}/{len}", r.start, r.end));
    }
    if method == Method::HEAD {
        return builder.body(Body::empty()).unwrap();
    }
    let mut file = match tokio::fs::File::open(&path).await {
        Ok(f) => f,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };
    if start > 0 {
        if let Err(e) = file.seek(std::io::SeekFrom::Start(start)).await {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    }
    let stream = ReaderStream::with_capacity(file.take(count), 256 * 1024);
    builder.body(Body::from_stream(stream)).unwrap()
}

#[derive(Deserialize)]
pub struct FrameQuery {
    pub path: String,
    #[serde(default)]
    pub ms: f64,
    #[serde(default = "default_frame_width")]
    pub w: u32,
}

fn default_frame_width() -> u32 {
    640
}

/// Run ffmpeg for one JPEG; the child is killed if this future is dropped
/// (client went away) or after 30 s.
async fn ffmpeg_frame(path: &Path, ms: f64, w: u32) -> Result<Vec<u8>, String> {
    let bins = ffmpeg::find_binaries().map_err(String::from)?;
    let mut cmd = tokio::process::Command::new(&bins.ffmpeg);
    cmd.args(ffmpeg::frame_args(path, ms, w))
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    #[cfg(windows)]
    cmd.creation_flags(crate::procs::CREATE_NO_WINDOW);
    let child = cmd.spawn().map_err(|e| format!("cannot start ffmpeg: {e}"))?;
    #[cfg(windows)]
    if let Some(h) = child.raw_handle() {
        crate::procs::adopt_raw(h);
    }
    let out = tokio::time::timeout(Duration::from_secs(30), child.wait_with_output())
        .await
        .map_err(|_| "ffmpeg timed out".to_string())?
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!("ffmpeg failed: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(out.stdout)
}

async fn frame(State(state): State<MediaState>, Query(q): Query<FrameQuery>) -> Response {
    let (path, meta) = match authorised_path(&state, &q.path).await {
        Ok(v) => v,
        Err(r) => return r.into_response(),
    };
    let (ms, w) = (q.ms.max(0.0), q.w.clamp(16, 4096));
    let mtime = meta.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_nanos()).unwrap_or(0);
    let key = format!("{}|{}|{mtime}|{}|{w}", path_key(&path), meta.len(), ms.round() as i64);
    let cached = state.frame_cache.lock().unwrap().iter().find(|(k, _)| *k == key).map(|(_, v)| v.clone());
    let bytes = match cached {
        Some(b) => b,
        None => {
            let Ok(_permit) = state.frame_permits.clone().acquire_owned().await else {
                return (StatusCode::SERVICE_UNAVAILABLE, "shutting down").into_response();
            };
            let mut result = ffmpeg_frame(&path, ms, w).await;
            if matches!(&result, Ok(b) if b.is_empty()) {
                // Seeking past the end yields nothing: retry near the end (needs a probe).
                let p = path.clone();
                result = tokio::task::spawn_blocking(move || ffmpeg::extract_frame(&p, ms, w).map_err(String::from))
                    .await
                    .unwrap_or_else(|e| Err(e.to_string()));
            }
            match result {
                Ok(b) => {
                    let b = Arc::new(b);
                    let mut c = state.frame_cache.lock().unwrap();
                    c.push_back((key, b.clone()));
                    while c.len() > FRAME_CACHE_ENTRIES {
                        c.pop_front();
                    }
                    b
                }
                Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
            }
        }
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "image/jpeg")
        .header(header::CACHE_CONTROL, "private, max-age=600")
        .body(Body::from(bytes.as_ref().clone()))
        .unwrap()
}

fn is_sha_hex(s: &str) -> bool {
    s.len() == 40 && s.chars().all(|c| c.is_ascii_hexdigit())
}

fn is_thumb_file(s: &str) -> bool {
    s.strip_suffix(".jpg").map(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit())).unwrap_or(false)
}

async fn thumb(State(state): State<MediaState>, AxPath((sha, file)): AxPath<(String, String)>) -> Response {
    if !is_sha_hex(&sha) || !is_thumb_file(&file) {
        return bad_request("invalid thumbnail id").into_response();
    }
    let path = state.thumbs_root.join(&sha).join(&file);
    match tokio::fs::read(&path).await {
        Ok(bytes) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "image/jpeg")
            .header(header::CACHE_CONTROL, "private, max-age=86400, immutable")
            .body(Body::from(bytes))
            .unwrap(),
        Err(_) => (StatusCode::NOT_FOUND, "thumbnail not found").into_response(),
    }
}

/* ------------------------------------------------------------------ tests */

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::Request;
    use tower::ServiceExt;

    fn temp_root(name: &str) -> PathBuf {
        let d = ffmpeg::cache_dir().join("test").join(name);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// In-process router with a known port for the Host check.
    fn test_state(root: PathBuf) -> MediaState {
        let s = MediaState::new(root, MediaRegistry::new());
        s.set_port(4321);
        s
    }

    fn get(state: &MediaState, path_and_query: &str) -> axum::http::request::Builder {
        Request::get(format!("/t/{}{path_and_query}", state.token)).header(header::HOST, "127.0.0.1:4321")
    }

    #[test]
    fn range_parsing() {
        assert_eq!(parse_range(None, 100), Ok(None));
        assert_eq!(parse_range(Some("bytes=0-9"), 100), Ok(Some(ByteRange { start: 0, end: 9 })));
        assert_eq!(parse_range(Some("bytes=90-"), 100), Ok(Some(ByteRange { start: 90, end: 99 })));
        assert_eq!(parse_range(Some("bytes=-10"), 100), Ok(Some(ByteRange { start: 90, end: 99 })));
        assert_eq!(parse_range(Some("bytes=50-500"), 100), Ok(Some(ByteRange { start: 50, end: 99 })));
        assert_eq!(parse_range(Some("bytes=100-"), 100), Err(Unsatisfiable));
        assert_eq!(parse_range(Some("bytes=-0"), 100), Err(Unsatisfiable));
        // malformed → ignored (full response), not 416
        assert_eq!(parse_range(Some("bytes=5-2"), 100), Ok(None));
        assert_eq!(parse_range(Some("bytes=abc-"), 100), Ok(None));
        assert_eq!(parse_range(Some("bytes=-x"), 100), Ok(None));
        assert_eq!(parse_range(Some("bytes=+1-2"), 100), Ok(None));
        assert_eq!(parse_range(Some("items=1-2"), 100), Ok(None));
    }

    #[test]
    fn path_validation_and_keys() {
        assert!(validate_path("relative/file.mp4").is_err());
        assert!(validate_path("C:/a/../b.mp4").is_err());
        assert!(validate_path("").is_err());
        assert!(validate_path(r"\\attacker\share\x.mp4").is_err());
        assert!(validate_path(r"\\?\C:\x.mp4").is_err());
        assert!(validate_path("//attacker/share/x.mp4").is_err());
        let abs = std::env::current_dir().unwrap().join("x.mp4");
        assert!(validate_path(&abs.to_string_lossy()).is_ok());
        let root = temp_root("registry");
        let f = root.join("Some File.mp4");
        std::fs::write(&f, b"x").unwrap();
        let reg = MediaRegistry::new();
        reg.register(&f);
        assert!(reg.contains(&f));
        if cfg!(windows) {
            let upper = PathBuf::from(f.to_string_lossy().to_uppercase().replace('\\', "/"));
            assert!(reg.contains(&upper), "case-insensitive, slash-agnostic on Windows");
        }
        assert!(!reg.contains(&root.join("other.mp4")));
    }

    #[test]
    fn http_dates() {
        assert_eq!(http_date(UNIX_EPOCH), "Thu, 01 Jan 1970 00:00:00 GMT");
        assert_eq!(http_date(UNIX_EPOCH + Duration::from_secs(784_887_151)), "Tue, 15 Nov 1994 08:12:31 GMT");
        assert_eq!(http_date(UNIX_EPOCH + Duration::from_secs(1_709_164_800)), "Thu, 29 Feb 2024 00:00:00 GMT");
    }

    #[tokio::test]
    async fn media_route_serves_ranges() {
        let root = temp_root("media");
        let file = root.join("blob.mp4");
        let data: Vec<u8> = (0..=255u8).collect();
        std::fs::write(&file, &data).unwrap();
        let state = test_state(root.clone());
        state.registry.register(&file);
        let app = router(state.clone());
        let uri = format!("/media?path={}", urlencode(&file.to_string_lossy()));
        let req = |b: axum::http::request::Builder| b.body(Body::empty()).unwrap();

        // full
        let res = app.clone().oneshot(req(get(&state, &uri))).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers()[header::ACCEPT_RANGES], "bytes");
        assert_eq!(res.headers()[header::CONTENT_TYPE], "video/mp4");
        assert_eq!(res.headers()[header::CONTENT_LENGTH], "256");
        assert_eq!(res.headers()[header::CACHE_CONTROL], "no-cache");
        assert!(res.headers().contains_key(header::LAST_MODIFIED));
        let etag = res.headers()[header::ETAG].to_str().unwrap().to_string();
        let body = to_bytes(res.into_body(), usize::MAX).await.unwrap();
        assert_eq!(body.as_ref(), &data[..]);

        // conditional: unchanged → 304; rewritten file → new ETag
        let res = app.clone().oneshot(req(get(&state, &uri).header(header::IF_NONE_MATCH, &etag))).await.unwrap();
        assert_eq!(res.status(), StatusCode::NOT_MODIFIED);
        std::thread::sleep(Duration::from_millis(20));
        std::fs::write(&file, &data[..200]).unwrap();
        let res = app.clone().oneshot(req(get(&state, &uri).header(header::IF_NONE_MATCH, &etag))).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_ne!(res.headers()[header::ETAG].to_str().unwrap(), etag);
        std::fs::write(&file, &data).unwrap();

        // partial
        let res = app.clone().oneshot(req(get(&state, &uri).header(header::RANGE, "bytes=10-19"))).await.unwrap();
        assert_eq!(res.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(res.headers()[header::CONTENT_RANGE], "bytes 10-19/256");
        assert_eq!(res.headers()[header::CONTENT_LENGTH], "10");
        let body = to_bytes(res.into_body(), usize::MAX).await.unwrap();
        assert_eq!(body.as_ref(), &data[10..20]);

        // open-ended
        let res = app.clone().oneshot(req(get(&state, &uri).header(header::RANGE, "bytes=250-"))).await.unwrap();
        assert_eq!(res.status(), StatusCode::PARTIAL_CONTENT);
        let body = to_bytes(res.into_body(), usize::MAX).await.unwrap();
        assert_eq!(body.as_ref(), &data[250..]);

        // unsatisfiable vs malformed
        let res = app.clone().oneshot(req(get(&state, &uri).header(header::RANGE, "bytes=999-"))).await.unwrap();
        assert_eq!(res.status(), StatusCode::RANGE_NOT_SATISFIABLE);
        assert_eq!(res.headers()[header::CONTENT_RANGE], "bytes */256");
        let res = app.clone().oneshot(req(get(&state, &uri).header(header::RANGE, "bytes=9-3"))).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK, "a malformed Range is ignored");

        // rejected paths
        let res = app.clone().oneshot(req(get(&state, "/media?path=rel.mp4"))).await.unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let res = app.clone().oneshot(req(get(&state, "/media?path=C%3A%2Fa%2F..%2Fb.mp4"))).await.unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let res = app.clone().oneshot(req(get(&state, "/media?path=%5C%5Chost%5Cshare%5Cx.mp4"))).await.unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST, "UNC");
        let missing = root.join("nope.mp4");
        state.registry.register(&missing);
        let res = app.oneshot(req(get(&state, &format!("/media?path={}", urlencode(&missing.to_string_lossy()))))).await.unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn thumb_route_serves_cache_and_rejects_bad_ids() {
        let root = temp_root("thumbs");
        let sha = "0123456789abcdef0123456789abcdef01234567";
        std::fs::create_dir_all(root.join(sha)).unwrap();
        std::fs::write(root.join(sha).join("3.jpg"), b"\xFF\xD8jpegdata").unwrap();
        let state = test_state(root);
        let app = router(state.clone());
        let req = |p: String| get(&state, &p).body(Body::empty()).unwrap();
        let res = app.clone().oneshot(req(format!("/thumb/{sha}/3.jpg"))).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers()[header::CONTENT_TYPE], "image/jpeg");
        let res = app.clone().oneshot(req(format!("/thumb/{sha}/9.jpg"))).await.unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        let res = app.clone().oneshot(req("/thumb/notasha/3.jpg".into())).await.unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let res = app.oneshot(req(format!("/thumb/{sha}/..%2Fx.jpg"))).await.unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn frame_route_returns_jpeg_when_ffmpeg_present() {
        let Some(clip) = ffmpeg::tests::synth_clip() else {
            eprintln!("SKIP: ffmpeg not available");
            return;
        };
        let state = test_state(temp_root("frames"));
        let app = router(state.clone());
        let uri = format!("/frame?path={}&ms=500&w=120", urlencode(&clip.to_string_lossy()));
        let res = app.clone().oneshot(get(&state, &uri).body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(res.status(), StatusCode::FORBIDDEN, "unregistered");
        state.registry.register(&clip);
        for _ in 0..2 {
            // second round comes from the cache
            let res = app.clone().oneshot(get(&state, &uri).body(Body::empty()).unwrap()).await.unwrap();
            assert_eq!(res.status(), StatusCode::OK);
            assert_eq!(res.headers()[header::CONTENT_TYPE], "image/jpeg");
            let body = to_bytes(res.into_body(), usize::MAX).await.unwrap();
            assert!(body.len() > 100 && body[0] == 0xFF && body[1] == 0xD8);
        }
    }

    /// Raw HTTP/1.1 GET against the real server.
    fn http_get(port: u16, target: &str, extra: &[(&str, &str)], host: Option<&str>) -> (u16, String, Vec<u8>) {
        use std::io::{Read, Write};
        let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
        let host = host.map(str::to_string).unwrap_or_else(|| format!("127.0.0.1:{port}"));
        let mut req = format!("GET {target} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n");
        for (k, v) in extra {
            req.push_str(&format!("{k}: {v}\r\n"));
        }
        req.push_str("\r\n");
        s.write_all(req.as_bytes()).unwrap();
        let mut out = Vec::new();
        s.read_to_end(&mut out).unwrap();
        let split = out.windows(4).position(|w| w == b"\r\n\r\n").unwrap_or(out.len());
        let head = String::from_utf8_lossy(&out[..split]).to_string();
        let body = out.get(split + 4..).unwrap_or(&[]).to_vec();
        let status = head.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
        (status, head.to_lowercase(), body)
    }

    /// The security model against the real server on 127.0.0.1:0: token prefix,
    /// Host check, CORS allow-list, registry, UNC rejection.
    #[test]
    fn server_enforces_token_host_cors_and_registry() {
        let root = temp_root("lockdown");
        let allowed = root.join("allowed clip.mp4");
        let secret = root.join("secret.txt");
        std::fs::write(&allowed, b"0123456789").unwrap();
        std::fs::write(&secret, b"top secret").unwrap();
        let registry = MediaRegistry::new();
        registry.register(&allowed);
        let state = MediaState::new(root.clone(), registry);
        let token = state.token.to_string();
        assert_eq!(token.len(), 32);
        assert!(token.chars().all(|c| c.is_ascii_hexdigit()));
        let port = start(state.clone()).expect("start");
        assert_eq!(state.base_url(), format!("http://127.0.0.1:{port}/t/{token}"));
        let t = |p: &str| format!("/t/{token}{p}");

        let (st, _, body) = http_get(port, &t("/health"), &[], None);
        assert_eq!((st, body.as_slice()), (200, &b"ok"[..]));
        // no token / wrong token / old unprefixed routes → 404
        assert_eq!(http_get(port, "/health", &[], None).0, 404);
        assert_eq!(http_get(port, "/t/00000000000000000000000000000000/health", &[], None).0, 404);
        let q = |p: &Path| format!("/media?path={}", urlencode(&p.to_string_lossy()));
        assert_eq!(http_get(port, &q(&allowed), &[], None).0, 404);
        // DNS rebinding: wrong Host
        assert_eq!(http_get(port, &t("/health"), &[], Some("evil.example:80")).0, 403);
        assert_eq!(http_get(port, &t("/health"), &[], Some(&format!("localhost:{port}"))).0, 403);
        // registered file: served; unregistered (even existing) file: 403
        let (st, head, body) = http_get(port, &t(&q(&allowed)), &[], None);
        assert_eq!((st, body.as_slice()), (200, &b"0123456789"[..]), "{head}");
        let (st, _, body) = http_get(port, &t(&q(&secret)), &[], None);
        assert_eq!(st, 403);
        assert!(!String::from_utf8_lossy(&body).contains("secret"));
        // case / slash variants of a registered path are the same file on Windows
        if cfg!(windows) {
            let variant = PathBuf::from(allowed.to_string_lossy().to_uppercase().replace('\\', "/"));
            assert_eq!(http_get(port, &t(&q(&variant)), &[], None).0, 200);
        }
        // UNC / device paths are rejected before any file-system access
        for p in [r"\\attacker.example\share\x.mp4", r"\\?\C:\Windows\win.ini", "//attacker/share/x.mp4"] {
            assert_eq!(http_get(port, &t(&format!("/media?path={}", urlencode(p))), &[], None).0, 400, "{p}");
            assert_eq!(http_get(port, &t(&format!("/frame?path={}&ms=0", urlencode(p))), &[], None).0, 400, "{p}");
        }
        // CORS: app origins only
        for origin in APP_ORIGINS {
            let (_, head, _) = http_get(port, &t(&q(&allowed)), &[("Origin", origin)], None);
            assert!(head.contains(&format!("access-control-allow-origin: {origin}")), "{origin}: {head}");
        }
        let (_, head, _) = http_get(port, &t(&q(&allowed)), &[("Origin", "https://evil.example")], None);
        assert!(!head.contains("access-control-allow-origin"), "{head}");
        let (_, head, _) = http_get(port, &t(&q(&allowed)), &[("Origin", "http://localhost:3000")], None);
        assert!(!head.contains("access-control-allow-origin"), "{head}");
    }

    fn urlencode(s: &str) -> String {
        let mut out = String::new();
        for b in s.bytes() {
            match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
                _ => out.push_str(&format!("%{b:02X}")),
            }
        }
        out
    }
}
