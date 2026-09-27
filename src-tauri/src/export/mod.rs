//! Exporter v2 — a frame-accurate compositor whose output matches the preview.
//!
//! Stages (one export job = one background thread + helpers):
//!
//! 1. **Plan** ([`build_timeline`], synchronous so obvious errors reach the
//!    caller): the project is validated ([`Project::validate`]), every visible
//!    clip on non-muted video tracks becomes a [`LayerPlan`] with its
//!    [`Resolved`] state for each output frame (speed LUT, freeze frame,
//!    reverse, transform / mask keyframes, reframe crop — `render::timemap`, a
//!    port of `playback.ts`). Every source is probed once (in parallel) for its
//!    displayed size (rotation), average frame rate and colour tags; missing
//!    video / image / audio sources and unreadable LUTs fail here. Each clip's
//!    effective grade is baked into lattices (`render::bake`).
//! 2. **Optical flow**: clips with `speed.opticalFlow` that dip below 1× are
//!    pre-interpolated by the Python pipeline ([`flow`]) — through the GPU
//!    queue, one AI job at a time; on failure the clip falls back to
//!    neighbour-frame blending on a refined decode grid with a warning.
//! 3. **Audio** ([`audio`]): the mix is streamed to a 48 kHz f32 WAV first.
//! 4. **Video**: per layer a producer thread decodes source frames with its
//!    own ffmpeg child ([`decode`]), blends / blurs them and hands them to the
//!    compositor through a bounded channel (buffers are recycled; frames that
//!    need no blend / blur are sampled as 8-bit directly); the compositor
//!    (`render::compositor`, rayon row-parallel) grades, masks and blends all
//!    layers onto the canvas and a writer thread streams `rgb24` into the
//!    encoder ([`encode`]). `h264_nvenc_mp4` falls back to libx264 when NVENC
//!    itself fails.
//!
//! The file is written to `<out>.partial-<jobId>.<ext>` next to the target and
//! renamed over it only when the export succeeded; a failed or cancelled
//! export deletes only its partial file. An output path that is one of the
//! project's sources (asset, LUT or stem) is refused.
//!
//! Events: `export://progress { jobId, pct, message }` (pct by frames
//! written, message with fps + ETA), `export://log { jobId, level, message }`,
//! `export://done { jobId, ok, outPath, error }`.

pub mod audio;
pub mod decode;
pub mod encode;
pub mod flow;
pub mod stretch;

use crate::ffmpeg::{self, FfmpegPaths, VideoInfo};
use crate::jobs::{EventSink, JobManager, TaskCtl, GPU_WAIT_MESSAGE};
use crate::media_server::{path_key, MediaRegistry};
use crate::model::{
    Asset, AssetKind, Clip, ClipEffect, EffectType, FrameInterpolation, Project, TrackKind, TransitionType, TRANSITION_MAX_MS, TRANSITION_MIN_MS,
};
use crate::render::{fx, transitions};
use crate::render::bake::BakedGrade;
use crate::render::color::GradeParams;
use crate::render::compositor::{composite_layer, Canvas, Layer};
use crate::render::lut::{load_cube, Lut3D};
use crate::render::mask::MaskParams;
use crate::render::sample::{BufPool, FloatImage, Placement, SourceImage, UvMatrix};
use crate::render::timemap::{ClipTimeMap, Resolved};
use decode::{FrameProvider, FrameRequest, VideoSource};
use encode::{EncodeSpec, Encoder};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::Arc;
use std::time::{Duration, Instant};

/* ------------------------------------------------------------------ presets */

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportPreset {
    H264Mp4,
    H264NvencMp4,
    ProresMov,
}

impl ExportPreset {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim() {
            "h264_mp4" => Ok(Self::H264Mp4),
            "h264_nvenc_mp4" => Ok(Self::H264NvencMp4),
            "prores_mov" => Ok(Self::ProresMov),
            other if other.ends_with("_legacy") => Err(format!(
                "the phase-1 exporter ('{other}') has been removed; use h264_mp4, h264_nvenc_mp4 or prores_mov"
            )),
            other => Err(format!("unknown export preset '{other}' (expected h264_mp4 | h264_nvenc_mp4 | prores_mov)")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::H264Mp4 => "h264_mp4",
            Self::H264NvencMp4 => "h264_nvenc_mp4",
            Self::ProresMov => "prores_mov",
        }
    }

    pub fn pix_fmt(self) -> &'static str {
        match self {
            Self::ProresMov => "yuv422p10le",
            _ => "yuv420p",
        }
    }

    /// Video codec arguments. `nvenc` selects the hardware encoder for the
    /// NVENC preset; the fallback passes `false`.
    pub fn video_args(self, nvenc: bool) -> Vec<String> {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        match self {
            Self::H264NvencMp4 if nvenc => s(&["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0"]),
            Self::H264Mp4 | Self::H264NvencMp4 => s(&["-c:v", "libx264", "-preset", "medium", "-crf", "18"]),
            Self::ProresMov => s(&["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0"]),
        }
    }

    pub fn audio_args(self) -> Vec<String> {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        match self {
            Self::ProresMov => s(&["-c:a", "pcm_s16le"]),
            _ => s(&["-c:a", "aac", "-b:a", "192k"]),
        }
    }

    pub fn container_args(self) -> Vec<String> {
        match self {
            Self::ProresMov => Vec::new(),
            _ => vec!["-movflags".into(), "+faststart".into()],
        }
    }
}

/// Optional section of the timeline to export (`export_project { range }`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportRange {
    pub start_ms: f64,
    pub end_ms: f64,
}

/* --------------------------------------------------------------------- plan */

/// One clip on a video track, resolved for every output frame it covers.
#[derive(Debug)]
pub struct LayerPlan {
    pub clip: Clip,
    /// the asset with its displayed size / duration / fps from the plan-time probe
    pub asset: Asset,
    pub track_index: usize,
    /// first output frame (inclusive) and end (exclusive)
    pub first_frame: i64,
    pub end_frame: i64,
    pub resolved: Arc<Vec<Resolved>>,
    pub grade: Arc<GradeParams>,
    /// the grade baked into lattices (`None`: graded exactly per pixel)
    pub baked: Option<Arc<BakedGrade>>,
    pub source: Arc<VideoSource>,
    pub requests: Arc<Vec<FrameRequest>>,
    /// the source's average frame rate (probe), for the decode grid and frame interpolation
    pub avg_fps: Option<f64>,
    /// slowest playback rate over the frames this export renders (frozen frames excluded; 1
    /// when every frame is frozen) — the clip's effective frame rate is `avg_fps × min_rate`
    pub min_rate: f64,
    /// video fade (amount of black, `Clip.fadeInMs / fadeOutMs`) per covered frame; `None` = none
    pub fades: Option<Arc<Vec<f32>>>,
}

/// A transition between two layers of the same track, centred on the cut.
#[derive(Debug, Clone)]
pub struct TransitionPlan {
    pub kind: TransitionType,
    /// layer indices of the outgoing (A) and incoming (B) clip
    pub a: usize,
    pub b: usize,
    /// timeline time of the cut (B's start) and the (clamped) length
    pub cut_ms: f64,
    pub dur_ms: f64,
    /// output frames `[first_frame, end_frame)` inside the window (by video time)
    pub first_frame: i64,
    pub end_frame: i64,
}

/// An FX-track effect applied to the composited frame.
#[derive(Debug, Clone)]
pub struct EffectPlan {
    pub clip_id: String,
    pub effect: ClipEffect,
    pub track_index: usize,
    pub start_ms: f64,
    /// `outMs − inMs` (speed ignored)
    pub dur_ms: f64,
    pub first_frame: i64,
    pub end_frame: i64,
}

#[derive(Debug)]
pub struct TimelinePlan {
    pub fps: f64,
    pub width: usize,
    pub height: usize,
    pub start_ms: f64,
    pub duration_ms: f64,
    pub frame_count: i64,
    /// in draw order (bottom first)
    pub layers: Vec<LayerPlan>,
    pub transitions: Vec<TransitionPlan>,
    /// in application order (FX tracks in track order, clips by start)
    pub effects: Vec<EffectPlan>,
    /// timeline time the video layers show at each output frame: the output time, except
    /// during a `cameraSnap`, which holds its first frame
    pub video_time: Arc<Vec<f64>>,
    pub interpolation: FrameInterpolation,
    pub warnings: Vec<String>,
    /// informational notes (logged at `info`)
    pub notes: Vec<String>,
}

impl TimelinePlan {
    pub fn frame_time_ms(&self, n: i64) -> f64 {
        self.start_ms + n as f64 * 1000.0 / self.fps
    }
}

/// Timeline length of a clip: an FX clip lasts `outMs − inMs` (speed ignored), any other clip
/// its speed-remapped length + freeze hold.
pub fn clip_timeline_ms(clip: &Clip, track_kind: TrackKind) -> f64 {
    if track_kind == TrackKind::Fx {
        clip.source_duration_ms()
    } else {
        ClipTimeMap::new(clip).total_ms()
    }
}

/// `projectDurationMs` from the store: the end of the last clip on any track.
pub fn project_duration_ms(project: &Project) -> f64 {
    project
        .tracks
        .iter()
        .flat_map(|t| t.clips.iter().map(move |c| c.start_ms + clip_timeline_ms(c, t.kind)))
        .fold(0.0, f64::max)
}

/// First index whose value is ≥ `t − eps` in a non-decreasing list.
fn lower_bound(v: &[f64], t: f64, eps: f64) -> i64 {
    v.partition_point(|x| *x < t - eps) as i64
}

/// Video time per output frame: `cameraSnap` effects on non-muted FX tracks hold the picture on
/// their first frame for their whole duration (the earliest-starting snap wins where they
/// overlap; the result is kept non-decreasing).
pub fn video_times(project: &Project, frame_times: impl Iterator<Item = f64>) -> Vec<f64> {
    let snaps: Vec<(f64, f64)> = project
        .tracks
        .iter()
        .filter(|t| t.kind == TrackKind::Fx && !t.muted)
        .flat_map(|t| t.clips.iter())
        .filter(|c| c.effect.as_ref().map(|e| e.kind == EffectType::CameraSnap).unwrap_or(false) && c.source_duration_ms() > 0.0)
        .map(|c| (c.start_ms, c.start_ms + c.source_duration_ms()))
        .collect();
    let mut last = f64::NEG_INFINITY;
    frame_times
        .map(|t| {
            let v = snaps.iter().filter(|(s, e)| t >= *s && t < *e).map(|(s, _)| *s).fold(t, f64::min);
            last = last.max(v);
            last
        })
        .collect()
}

/// A transition found on the timeline: (track, outgoing clip index, incoming clip index, type,
/// cut, clamped length).
#[derive(Debug, Clone, PartialEq)]
pub struct CutTransition {
    pub track: usize,
    pub a: usize,
    pub b: usize,
    pub kind: TransitionType,
    pub cut_ms: f64,
    pub dur_ms: f64,
}

fn shows_picture(project: &Project, clip: &Clip) -> bool {
    clip.source_duration_ms() > 0.0 && project.asset(&clip.asset_id).map(|a| matches!(a.kind, AssetKind::Video | AssetKind::Image)).unwrap_or(false)
}

/// Every `transitionIn` that has an outgoing clip: the previous clip on the same (non-muted)
/// video track ends where the clip starts (gap under one frame). The length is clamped to
/// 100–3000 ms and to the shorter clip. Returns the transitions and warnings for the ignored ones.
pub fn find_transitions(project: &Project, fps: f64) -> (Vec<CutTransition>, Vec<String>) {
    let mut out = Vec::new();
    let mut warnings = Vec::new();
    let frame = 1000.0 / fps;
    for (ti, track) in project.tracks.iter().enumerate() {
        if track.kind != TrackKind::Video || track.muted {
            continue;
        }
        for (bi, b) in track.clips.iter().enumerate() {
            let Some(tr) = &b.transition_in else { continue };
            if !shows_picture(project, b) {
                continue;
            }
            let prev = track
                .clips
                .iter()
                .enumerate()
                .filter(|(ai, a)| *ai != bi && a.start_ms < b.start_ms && shows_picture(project, a))
                .map(|(ai, a)| (ai, a.start_ms + ClipTimeMap::new(a).total_ms()))
                .filter(|(_, end)| (end - b.start_ms).abs() < frame)
                .min_by(|x, y| (x.1 - b.start_ms).abs().total_cmp(&(y.1 - b.start_ms).abs()));
            let Some((ai, _)) = prev else {
                warnings.push(format!("clip {}: transition ignored (no clip ends where it starts on its track)", b.id));
                continue;
            };
            let la = ClipTimeMap::new(&track.clips[ai]).total_ms();
            let lb = ClipTimeMap::new(b).total_ms();
            let d = tr.duration_ms.clamp(TRANSITION_MIN_MS, TRANSITION_MAX_MS).min(la).min(lb);
            if !tr.kind.is_known() {
                warnings.push(format!("clip {}: unknown transition '{}' rendered as a dissolve", b.id, tr.kind.as_str()));
            }
            out.push(CutTransition { track: ti, a: ai, b: bi, kind: tr.kind.clone(), cut_ms: b.start_ms, dur_ms: d });
        }
    }
    (out, warnings)
}

/// Decode resolution: the source is downscaled (keeping its aspect) when the
/// project never shows it at more than 1 canvas pixel per source pixel —
/// accounting for the reframe crop and the largest keyframed scale. (Clips that
/// are sharpened are always decoded at full size; see [`build_timeline`].)
pub fn decode_size(clip: &Clip, asset: &Asset, canvas_w: usize, canvas_h: usize) -> (usize, usize) {
    let (w, h) = (asset.width.max(1) as f64, asset.height.max(1) as f64);
    let ax = canvas_w as f64 / canvas_h as f64;
    let t = &clip.transform.scale;
    let max_scale = t.keyframes.iter().map(|k| k.value).fold(t.static_value, f64::max).max(1e-3);
    let mut crops: Vec<(f64, f64)> = clip
        .reframe
        .as_ref()
        .map(|r| r.keyframes.iter().map(|k| ((k.crop[2] - k.crop[0]).max(1.0), (k.crop[3] - k.crop[1]).max(1.0))).collect())
        .unwrap_or_default();
    if crops.is_empty() {
        crops.push((w, h));
    }
    let density = crops
        .iter()
        .map(|(cw, ch)| {
            let ca = cw / ch;
            let disp_w = if ca > ax { canvas_w as f64 } else { ca / ax * canvas_w as f64 };
            disp_w * max_scale / cw
        })
        .fold(0.0, f64::max);
    let f = density.min(1.0);
    if f >= 0.999 {
        return (asset.width.max(1) as usize, asset.height.max(1) as usize);
    }
    let even = |v: f64| ((v.ceil() as usize).div_ceil(2) * 2).max(2);
    let dw = even(w * f);
    let dh = even(dw as f64 * h / w);
    (dw.min(asset.width as usize).max(2), dh.min(asset.height as usize).max(2))
}

/// swscale colour options for a source (see [`crate::ffmpeg::input_color_matrix`]).
fn colour_options(info: Option<&VideoInfo>, is_image: bool) -> (Option<&'static str>, Option<&'static str>) {
    let Some(i) = info else { return (None, None) };
    // stills (JPEG) are BT.601 by convention whatever their size
    let h = if is_image { 0 } else { i.height };
    match ffmpeg::input_color_matrix(i.color_space.as_deref(), i.pix_fmt.as_deref(), h) {
        Some(m) => (Some(m), Some(ffmpeg::input_range(i.color_range.as_deref(), i.pix_fmt.as_deref()))),
        None => (None, None),
    }
}

/// The asset itself as a frame source, decoded on a `grid_fps` grid.
fn asset_source(asset: &Asset, dec: (usize, usize), grid_fps: f64, colour: (Option<&'static str>, Option<&'static str>)) -> VideoSource {
    let is_image = asset.kind == AssetKind::Image;
    let fps = grid_fps;
    VideoSource {
        path: PathBuf::from(&asset.path),
        is_image,
        decode_fps: fps,
        t0_ms: 0.0,
        index_rate: fps / 1000.0,
        frame_count: if is_image { 1 } else { ((asset.duration_ms * fps / 1000.0).floor() as i64).max(1) },
        dec_w: dec.0,
        dec_h: dec.1,
        file_w: asset.width.max(1) as usize,
        file_h: asset.height.max(1) as usize,
        blend_below_fps: fps * 0.999,
        in_matrix: colour.0,
        in_range: colour.1,
    }
}

fn requests_for(source: &VideoSource, resolved: &[Resolved]) -> Vec<FrameRequest> {
    resolved.iter().map(|r| source.request(r.source_ms, r.rate, r.frozen)).collect()
}

/// Probe every path once, several at a time (`None` when a probe fails).
fn probe_all(paths: Vec<String>) -> HashMap<String, Option<VideoInfo>> {
    let queue = std::sync::Mutex::new(paths);
    let out = std::sync::Mutex::new(HashMap::new());
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| loop {
                let Some(p) = queue.lock().unwrap().pop() else { break };
                let info = ffmpeg::probe_video_info(Path::new(&p)).ok();
                out.lock().unwrap().insert(p, info);
            });
        }
    });
    out.into_inner().unwrap()
}

/// Resolve the whole timeline (or `range`) into layers.
pub fn build_timeline(project: &Project, range: Option<ExportRange>) -> Result<TimelinePlan, String> {
    project.validate()?;
    let fps = project.fps;
    let width = ((project.width / 2) * 2) as usize;
    let height = ((project.height / 2) * 2) as usize;
    let total = project_duration_ms(project);
    if total <= 0.0 {
        return Err("the project is empty (no clips with a duration)".into());
    }
    let (start_ms, end_ms) = match range {
        Some(r) => {
            if !(r.start_ms.is_finite() && r.end_ms.is_finite()) {
                return Err("range must be finite".into());
            }
            (r.start_ms.clamp(0.0, total), r.end_ms.clamp(0.0, total))
        }
        None => (0.0, total),
    };
    let duration_ms = end_ms - start_ms;
    let frame_count = (duration_ms * fps / 1000.0 - 1e-6).ceil() as i64;
    if frame_count < 1 {
        return Err(format!("export range {start_ms:.0}–{end_ms:.0} ms is empty"));
    }

    // ---- video time per output frame (cameraSnap holds), transitions (handles around cuts)
    let video_time: Vec<f64> = video_times(project, (0..frame_count).map(|n| start_ms + n as f64 * 1000.0 / fps));
    let eps = 1e-6 * 1000.0 / fps;
    let (cuts, mut warnings) = find_transitions(project, fps);
    // extra timeline before / after a clip for its transitions: (track, clip) → ms
    let mut pre_roll: HashMap<(usize, usize), f64> = HashMap::new();
    let mut post_roll: HashMap<(usize, usize), f64> = HashMap::new();
    for c in &cuts {
        let e = pre_roll.entry((c.track, c.b)).or_insert(0.0);
        *e = e.max(c.dur_ms / 2.0);
        let e = post_roll.entry((c.track, c.a)).or_insert(0.0);
        *e = e.max(c.dur_ms / 2.0);
    }

    // ---- visible clips
    struct Visible<'a> {
        clip: &'a Clip,
        asset: &'a Asset,
        ti: usize,
        ci: usize,
        first: i64,
        end: i64,
        pre: f64,
        post: f64,
    }
    let mut visible: Vec<Visible> = Vec::new();
    for (ti, track) in project.tracks.iter().enumerate() {
        if track.kind != TrackKind::Video || track.muted {
            continue;
        }
        for (ci, clip) in track.clips.iter().enumerate() {
            if clip.source_duration_ms() <= 0.0 {
                continue;
            }
            let asset = project
                .asset(&clip.asset_id)
                .ok_or_else(|| format!("clip {} references unknown asset {}", clip.id, clip.asset_id))?;
            if matches!(asset.kind, AssetKind::Audio | AssetKind::Lut) {
                continue;
            }
            let pre = pre_roll.get(&(ti, ci)).copied().unwrap_or(0.0);
            let post = post_roll.get(&(ti, ci)).copied().unwrap_or(0.0);
            let s = clip.start_ms;
            let e = s + ClipTimeMap::new(clip).total_ms();
            let first = lower_bound(&video_time, s - pre, eps);
            let end = lower_bound(&video_time, e + post, eps).min(frame_count);
            if first >= end {
                continue;
            }
            if !Path::new(&asset.path).is_file() {
                return Err(format!("source file of clip {} is missing: {}", clip.id, asset.path));
            }
            visible.push(Visible { clip, asset, ti, ci, first, end, pre, post });
        }
    }
    // ---- every audible clip in the range must have its file (silence would go unnoticed)
    for ac in audio::audible_clips(project) {
        let c = ac.clip;
        let e = c.start_ms + ClipTimeMap::new(c).total_ms();
        if e > start_ms && c.start_ms < end_ms && !Path::new(&ac.asset.path).is_file() {
            return Err(format!("audio source of clip {} is missing: {}", c.id, ac.asset.path));
        }
    }

    // ---- one probe per source (displayed size, avg fps, colour tags), in parallel
    let mut paths: Vec<String> = visible.iter().map(|v| v.asset.path.clone()).collect();
    paths.sort();
    paths.dedup();
    let infos = probe_all(paths);

    let mut notes = Vec::new();
    let mut layer_of: HashMap<(usize, usize), usize> = HashMap::new();
    let mut luts: HashMap<String, Option<Arc<Lut3D>>> = HashMap::new();
    let mut bakes: HashMap<String, Option<Arc<BakedGrade>>> = HashMap::new();
    let mut layers = Vec::new();
    for v in visible {
        let clip = v.clip;
        let info = infos.get(&v.asset.path).and_then(|i| i.as_ref());
        let mut asset = v.asset.clone();
        if let Some(i) = info {
            if i.width > 0 && i.height > 0 {
                if (asset.width, asset.height) != (i.width, i.height) && asset.width != 0 {
                    notes.push(format!(
                        "clip {}: {} is displayed {}x{} (rotation {}°), not {}x{} as stored in the project",
                        clip.id, asset.name, i.width, i.height, i.rotation, asset.width, asset.height
                    ));
                }
                asset.width = i.width;
                asset.height = i.height;
            }
            if asset.kind == AssetKind::Video && asset.duration_ms <= 0.0 {
                asset.duration_ms = i.duration_ms;
            }
            if asset.kind == AssetKind::Video && asset.fps <= 0.0 {
                asset.fps = i.avg_fps.unwrap_or(0.0);
            }
        } else if asset.width == 0 || asset.height == 0 {
            let probed = ffmpeg::probe(Path::new(&asset.path)).map_err(String::from)?;
            asset.width = probed.width;
            asset.height = probed.height;
            asset.fps = probed.fps;
            asset.duration_ms = probed.duration_ms;
        }
        if asset.width == 0 || asset.height == 0 {
            return Err(format!("clip {}: {} has no picture", clip.id, asset.path));
        }
        let map = ClipTimeMap::new(clip);
        let s = clip.start_ms;
        let extended = v.pre > 0.0 || v.post > 0.0;
        let resolved: Vec<Resolved> = (v.first..v.end)
            .map(|n| {
                let local = video_time[n as usize] - s;
                if extended {
                    map.resolve_extended(clip, local, asset.duration_ms)
                } else {
                    map.resolve(clip, local)
                }
            })
            .collect();
        let rates = resolved.iter().filter(|r| !r.frozen && r.rate > 0.0).map(|r| r.rate);
        let min_rate = rates.fold(f64::INFINITY, f64::min);
        let min_rate = if min_rate.is_finite() { min_rate } else { 1.0 };
        let (fi, fo) = (clip.fade_in_ms.unwrap_or(0.0), clip.fade_out_ms.unwrap_or(0.0));
        let fades = (fi > 0.0 || fo > 0.0).then(|| {
            let total = map.total_ms();
            Arc::new(resolved.iter().map(|r| fx::clip_fade_amount(r.local_ms, total, fi, fo)).collect::<Vec<f32>>())
        });

        let lut = match &clip.color.lut_asset_id {
            Some(id) => match luts.get(id) {
                Some(l) => l.clone(),
                None => {
                    let l = match project.asset(id) {
                        Some(a) => Some(Arc::new(load_cube(Path::new(&a.path)).map_err(|e| format!("clip {}: {e}", clip.id))?)),
                        None => {
                            warnings.push(format!("LUT asset {id} not found in the project; ignored"));
                            None
                        }
                    };
                    luts.insert(id.clone(), l.clone());
                    l
                }
            },
            None => None,
        };
        // the clip's grade plus the project's universal adjustment (when switched on)
        let color = crate::model::effective_grade(&clip.color, project.universal_adjust.as_ref());
        let grade = Arc::new(GradeParams::new(&color, lut.clone()));
        let bake_key = format!("{}|{}", serde_json::to_string(&color).unwrap_or_default(), lut.as_ref().map(|l| format!("{:p}", Arc::as_ptr(l))).unwrap_or_default());
        let baked = match bakes.get(&bake_key) {
            Some(b) => b.clone(),
            None => {
                // CAPPYCAT_NO_BAKE=1 grades every pixel exactly (diagnostics / A-B timing)
                let b = if BakedGrade::worthwhile(&grade) && std::env::var_os("CAPPYCAT_NO_BAKE").is_none() {
                    let t = Instant::now();
                    match BakedGrade::bake(&grade) {
                        Some(b) => {
                            notes.push(format!(
                                "grade of clip {} baked into a {}³ lattice in {:.0} ms ({:.1}% of cells graded exactly; probe error max {:.2}/255, mean {:.3}/255)",
                                clip.id,
                                b.size(),
                                t.elapsed().as_secs_f64() * 1000.0,
                                b.exact_fraction() * 100.0,
                                b.probe_max_err * 255.0,
                                b.probe_mean_err * 255.0
                            ));
                            Some(Arc::new(b))
                        }
                        None => {
                            notes.push(format!("grade of clip {} is graded exactly per pixel (a baked lattice would exceed 1/255)", clip.id));
                            None
                        }
                    }
                } else {
                    None
                };
                bakes.insert(bake_key, b.clone());
                b
            }
        };
        // Sharpening reads the source's own texels (the preview's u_texel): decode at full size.
        let dec = if grade.needs_detail() { (asset.width as usize, asset.height as usize) } else { decode_size(clip, &asset, width, height) };
        let source = asset_source(&asset, dec, fps, colour_options(info, asset.kind == AssetKind::Image));
        let requests = requests_for(&source, &resolved);
        layer_of.insert((v.ti, v.ci), layers.len());
        layers.push(LayerPlan {
            avg_fps: info.and_then(|i| i.avg_fps).or((asset.fps > 0.0).then_some(asset.fps)),
            min_rate,
            fades,
            clip: clip.clone(),
            asset,
            track_index: v.ti,
            first_frame: v.first,
            end_frame: v.end,
            resolved: Arc::new(resolved),
            grade,
            baked,
            source: Arc::new(source),
            requests: Arc::new(requests),
        });
    }
    // ---- transitions between layers of this export
    let mut transitions = Vec::new();
    for c in cuts {
        let (Some(&a), Some(&b)) = (layer_of.get(&(c.track, c.a)), layer_of.get(&(c.track, c.b))) else { continue };
        let first_frame = lower_bound(&video_time, c.cut_ms - c.dur_ms / 2.0, eps);
        let end_frame = lower_bound(&video_time, c.cut_ms + c.dur_ms / 2.0, eps).min(frame_count);
        if first_frame >= end_frame {
            continue;
        }
        notes.push(format!(
            "transition {} ({:.0} ms) between clips {} and {} at {:.0} ms",
            c.kind.as_str(),
            c.dur_ms,
            layers[a].clip.id,
            layers[b].clip.id,
            c.cut_ms
        ));
        transitions.push(TransitionPlan { kind: c.kind, a, b, cut_ms: c.cut_ms, dur_ms: c.dur_ms, first_frame, end_frame });
    }

    // ---- effects on FX tracks (output time; stacked in track order)
    let mut effects = Vec::new();
    for (ti, track) in project.tracks.iter().enumerate() {
        if track.kind != TrackKind::Fx || track.muted {
            continue;
        }
        let mut clips: Vec<&Clip> = track.clips.iter().filter(|c| c.effect.is_some() && c.source_duration_ms() > 0.0).collect();
        clips.sort_by(|a, b| a.start_ms.total_cmp(&b.start_ms));
        for c in clips {
            let e = c.effect.clone().unwrap();
            let dur = c.source_duration_ms();
            let frame_of = |t: f64| ((((t - start_ms) * fps / 1000.0) - 1e-6).ceil().max(0.0) as i64).min(frame_count);
            let (first_frame, end_frame) = (frame_of(c.start_ms), frame_of(c.start_ms + dur));
            if first_frame >= end_frame {
                continue;
            }
            if !e.kind.is_known() {
                warnings.push(format!("effect clip {}: unknown effect '{}' skipped", c.id, e.kind.as_str()));
                continue;
            }
            effects.push(EffectPlan { clip_id: c.id.clone(), effect: e, track_index: ti, start_ms: c.start_ms, dur_ms: dur, first_frame, end_frame });
        }
    }
    if let Some(s) = effects.iter().find(|e| e.effect.kind == EffectType::CameraSnap) {
        notes.push(format!("cameraSnap {}: the picture holds its {:.0} ms frame for {:.0} ms", s.clip_id, s.start_ms, s.dur_ms));
    }

    Ok(TimelinePlan {
        fps,
        width,
        height,
        start_ms,
        duration_ms: frame_count as f64 * 1000.0 / fps,
        frame_count,
        layers,
        transitions,
        effects,
        video_time: Arc::new(video_time),
        interpolation: project.frame_interpolation(),
        warnings,
        notes,
    })
}

/* ----------------------------------------------------------------- producer */

/// A source frame ready for compositing. Its buffers go back to the layer's
/// pools when it is dropped.
pub struct Prepared {
    pub image: SourceImage,
    pub detail: Option<FloatImage>,
    floats: Option<Arc<BufPool<f32>>>,
    bytes: Option<Arc<BufPool<u8>>>,
}

impl Prepared {
    pub fn new(image: SourceImage, detail: Option<FloatImage>) -> Self {
        Self { image, detail, floats: None, bytes: None }
    }
}

impl Drop for Prepared {
    fn drop(&mut self) {
        let empty = SourceImage::Float(FloatImage { width: 0, height: 0, data: Vec::new() });
        match std::mem::replace(&mut self.image, empty) {
            SourceImage::Float(f) => {
                if let Some(p) = &self.floats {
                    p.put(f.data);
                }
            }
            SourceImage::Bytes(b) => {
                if let Some(p) = &self.bytes {
                    FrameProvider::recycle(p, b);
                }
            }
        }
        if let (Some(d), Some(p)) = (self.detail.take(), &self.floats) {
            p.put(d.data);
        }
    }
}

struct Producer {
    rx: Receiver<Result<Arc<Prepared>, String>>,
}

type LogFn = Arc<dyn Fn(&str, String) + Send + Sync>;
/// (frame a, frame b, blend weight bits, blur bits): identical keys reuse the prepared frame
type PreparedKey = (i64, i64, u32, u32);

fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "unknown panic".into())
}

fn spawn_producer(layer: &LayerPlan, ffmpeg: PathBuf, ctl: Arc<TaskCtl>, log: LogFn) -> Result<Producer, String> {
    let (tx, rx) = sync_channel::<Result<Arc<Prepared>, String>>(2);
    let src = layer.source.clone();
    let requests = layer.requests.clone();
    let resolved = layer.resolved.clone();
    let needs_detail = layer.grade.needs_detail();
    let reverse = layer.clip.reversed;
    let (orig_w, orig_h) = (layer.asset.width.max(1) as f32, layer.asset.height.max(1) as f32);
    let clip_id = layer.clip.id.clone();
    let lo = requests.iter().map(|r| r.a.min(r.b)).min().unwrap_or(0);
    let hi = requests.iter().map(|r| r.a.max(r.b)).max().unwrap_or(0);
    let tx_panic = tx.clone();
    std::thread::Builder::new()
        .name(format!("export-decode-{clip_id}"))
        .spawn(move || {
            let body = std::panic::AssertUnwindSafe(|| {
                let mut provider = FrameProvider::new(src.clone(), ffmpeg, ctl.clone(), reverse, lo, hi);
                let floats = BufPool::<f32>::new();
                let bytes = provider.bytes.clone();
                let len = src.frame_bytes();
                let (kx, ky) = (src.dec_w as f32 / orig_w, src.dec_h as f32 / orig_h);
                let mut last: Option<(PreparedKey, Arc<Prepared>)> = None;
                for (req, r) in requests.iter().zip(resolved.iter()) {
                    if ctl.is_cancelled() {
                        return;
                    }
                    let blur = if r.blur > 0.5 { r.blur.min(12.0) as f32 } else { 0.0 };
                    let key = (req.a, req.b, req.w.to_bits(), blur.to_bits());
                    let item = match &last {
                        Some((k, p)) if *k == key => Ok(p.clone()),
                        _ => (|| -> Result<Arc<Prepared>, String> {
                            let fa = provider.get(req.a)?;
                            let fb = if req.w > 0.0 { Some(provider.get(req.b)?) } else { None };
                            // u_texel is one *original* source texel; convert to decoded pixels.
                            // The frame is converted to f32 here even when it needs no blend / blur:
                            // sampling the 8-bit frame in the compositor (SourceImage::Bytes) moves
                            // that conversion into the bottleneck stage (measured: 62 vs 67 fps on
                            // the 720p universal-adjust export), while here it overlaps with it.
                            let (image, detail) = {
                                let img = FloatImage::from_frames_into(&fa, fb.as_deref().map(|f| (f, req.w)), floats.take(len));
                                let detail = needs_detail.then(|| img.sharpen_detail_into(kx, ky, floats.take(len)));
                                let image = if blur > 0.0 {
                                    let b = img.gaussian_blur_pooled(blur / 3.0 * kx, blur / 3.0 * ky, &floats);
                                    floats.put(img.data);
                                    b
                                } else {
                                    img
                                };
                                (SourceImage::Float(image), detail)
                            };
                            Ok(Arc::new(Prepared { image, detail, floats: Some(floats.clone()), bytes: Some(bytes.clone()) }))
                        })(),
                    };
                    if let Ok(p) = &item {
                        last = Some((key, p.clone()));
                    }
                    let failed = item.is_err();
                    if tx.send(item).is_err() || failed {
                        break;
                    }
                }
                for w in provider.warnings.drain(..) {
                    log("warn", format!("clip {clip_id}: {w}"));
                }
            });
            if let Err(p) = std::panic::catch_unwind(body) {
                let _ = tx_panic.send(Err(format!("internal error in the decoder: {}", panic_message(p.as_ref()))));
            }
        })
        .map_err(|e| format!("cannot start a decoder thread: {e}"))?;
    Ok(Producer { rx })
}

/* ---------------------------------------------------------------------- job */

struct ExportJob {
    jobs: Arc<JobManager>,
    sink: Arc<dyn EventSink>,
    job_id: String,
    ctl: Arc<TaskCtl>,
    /// the requested output file
    out_path: PathBuf,
    /// what the encoder writes (`<out>.partial-<jobId>.<ext>`), renamed over `out_path` on success
    partial_path: PathBuf,
    preset: ExportPreset,
    project: Project,
    bins: FfmpegPaths,
    tmp_dir: PathBuf,
    registry: Option<MediaRegistry>,
}

impl ExportJob {
    fn log(&self, level: &str, message: impl Into<String>) {
        let message = message.into();
        tracing::info!("export {}: [{level}] {message}", self.job_id);
        self.sink.emit("export://log", json!({ "jobId": self.job_id, "level": level, "message": message }));
    }

    fn progress(&self, pct: f64, message: impl Into<String>) {
        self.sink.emit("export://progress", json!({ "jobId": self.job_id, "pct": pct, "message": message.into() }));
    }

    fn log_fn(&self) -> LogFn {
        let (sink, id) = (self.sink.clone(), self.job_id.clone());
        Arc::new(move |level: &str, message: String| {
            tracing::info!("export {id}: [{level}] {message}");
            sink.emit("export://log", json!({ "jobId": id, "level": level, "message": message }));
        })
    }
}

/// Fraction of the progress bar reserved for flow + audio preparation.
const PREP_PCT: f64 = 0.03;

/// NVENC failed in this process (no NVIDIA GPU / driver): later exports go straight to libx264.
static NVENC_UNAVAILABLE: AtomicBool = AtomicBool::new(false);

/// Does an encoder error come from NVENC itself (so a libx264 retry can help)?
pub fn is_nvenc_failure(error: &str) -> bool {
    let e = error.to_ascii_lowercase();
    ["nvenc", "cuda", "no capable devices", "openencodesession", "nvencodeapi"].iter().any(|k| e.contains(k))
}

/// `<dir>/<stem>.partial-<jobId>.<ext>`: same folder (so the final rename is atomic) and
/// the same extension (so ffmpeg picks the same muxer).
pub fn partial_path(out: &Path, job_id: &str) -> PathBuf {
    let stem = out.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "export".into());
    let ext = out.extension().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "mp4".into());
    out.with_file_name(format!("{stem}.partial-{job_id}.{ext}"))
}

/// Refuse output paths that would destroy a source of the project.
fn check_output_path(project: &Project, out: &Path) -> Result<(), String> {
    if out.file_name().is_none() {
        return Err(format!("output path {} has no file name", out.display()));
    }
    if out.is_dir() {
        return Err(format!("output path {} is a folder", out.display()));
    }
    let key = path_key(out);
    for a in &project.assets {
        let mut sources: Vec<(&str, &str)> = vec![(a.path.as_str(), if a.kind == AssetKind::Lut { "LUT" } else { "source" })];
        if let Some(s) = &a.stems {
            sources.push((s.vocals.as_str(), "voice stem"));
            sources.push((s.background.as_str(), "background stem"));
        }
        for (p, what) in sources {
            if !p.trim().is_empty() && path_key(Path::new(p)) == key {
                return Err(format!(
                    "refusing to export over {}: it is the {what} file of asset '{}' in this project; choose another output file",
                    out.display(),
                    a.name
                ));
            }
        }
    }
    Ok(())
}

/// Start an export job; returns its id immediately.
pub fn export_project(
    jobs: Arc<JobManager>,
    sink: Arc<dyn EventSink>,
    project: Project,
    out_path: String,
    preset: &str,
    range: Option<ExportRange>,
) -> Result<String, String> {
    export_project_with(jobs, sink, project, out_path, preset, range, None)
}

/// [`export_project`] that registers the finished file with the media server.
pub fn export_project_with(
    jobs: Arc<JobManager>,
    sink: Arc<dyn EventSink>,
    project: Project,
    out_path: String,
    preset: &str,
    range: Option<ExportRange>,
    registry: Option<MediaRegistry>,
) -> Result<String, String> {
    let preset = ExportPreset::parse(preset)?;
    let out_path = out_path.trim();
    if out_path.is_empty() {
        return Err("no output path given".into());
    }
    let out_path = std::path::absolute(out_path).map_err(|e| format!("invalid output path {out_path}: {e}"))?;
    check_output_path(&project, &out_path)?;
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| format!("cannot create output directory: {e}"))?;
        }
    }
    let bins = ffmpeg::find_binaries()?.clone();
    // Validate up front so obvious errors are returned synchronously.
    let plan = build_timeline(&project, range)?;
    let job_id = JobManager::new_job_id("export");
    let ctl = jobs.register_task(&job_id);
    let tmp_dir = ffmpeg::cache_dir().join("export").join(&job_id);
    let partial_path = partial_path(&out_path, &job_id);
    let job = ExportJob { jobs, sink, job_id: job_id.clone(), ctl, out_path, partial_path, preset, project, bins, tmp_dir, registry };
    std::thread::Builder::new()
        .name(format!("{job_id}-main"))
        .spawn(move || run_job(job, plan))
        .map_err(|e| e.to_string())?;
    Ok(job_id)
}

pub fn cancel_export(jobs: &JobManager, job_id: &str) -> Result<(), String> {
    jobs.cancel(job_id)
}

/// Move the finished partial file over the target (retrying briefly: a player may
/// still have the old file open).
fn publish(partial: &Path, out: &Path) -> Result<(), String> {
    let mut last = None;
    for i in 0..20 {
        match std::fs::rename(partial, out) {
            Ok(()) => return Ok(()),
            Err(e) => {
                last = Some(e);
                std::thread::sleep(Duration::from_millis(100 + 50 * i));
            }
        }
    }
    Err(format!(
        "could not replace {} ({}); the finished export was kept as {}",
        out.display(),
        last.map(|e| e.to_string()).unwrap_or_default(),
        partial.display()
    ))
}

fn run_job(job: ExportJob, plan: TimelinePlan) {
    let started = Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        std::fs::create_dir_all(&job.tmp_dir)
            .map_err(|e| format!("cannot create {}: {e}", job.tmp_dir.display()))
            .and_then(|_| run_stages(&job, plan))
    }))
    .unwrap_or_else(|p| Err(format!("internal error: {}", panic_message(p.as_ref()))));
    let _ = std::fs::remove_dir_all(&job.tmp_dir);
    let cancelled = job.ctl.is_cancelled();
    // success only counts once the file is in place
    let result = match result {
        Ok(stats) if !cancelled => publish(&job.partial_path, &job.out_path).map(|_| stats),
        Ok(_) => Err("cancelled".into()),
        Err(e) => Err(e),
    };
    let (ok, error) = match result {
        Ok(stats) => {
            if let Some(r) = &job.registry {
                r.register(&job.out_path);
            }
            job.log(
                "info",
                format!(
                    "exported {} frames in {:.1}s ({:.1} fps, video stage {:.1} fps) → {}",
                    stats.frames,
                    started.elapsed().as_secs_f64(),
                    stats.frames as f64 / started.elapsed().as_secs_f64().max(1e-6),
                    stats.video_fps,
                    job.out_path.display()
                ),
            );
            job.progress(1.0, format!("done — {} frames, {:.1} fps", stats.frames, stats.video_fps));
            (true, None)
        }
        Err(e) => {
            // Only the partial file goes; an earlier export at the target stays intact.
            if job.partial_path.is_file() && !e.contains("was kept as") {
                let _ = std::fs::remove_file(&job.partial_path);
            }
            if cancelled {
                (false, Some("cancelled".to_string()))
            } else {
                job.log("error", e.clone());
                (false, Some(e))
            }
        }
    };
    job.sink.emit(
        "export://done",
        json!({ "jobId": job.job_id, "ok": ok, "outPath": job.out_path.to_string_lossy(), "error": error }),
    );
    job.ctl.mark_done();
    job.jobs.finish(&job.job_id);
}

struct Stats {
    frames: i64,
    video_fps: f64,
}

fn run_stages(job: &ExportJob, mut plan: TimelinePlan) -> Result<Stats, String> {
    for w in std::mem::take(&mut plan.warnings) {
        job.log("warn", w);
    }
    for n in std::mem::take(&mut plan.notes) {
        job.log("info", n);
    }
    job.progress(0.0, format!("{} frames at {}x{} @ {} fps, {} layer(s)", plan.frame_count, plan.width, plan.height, plan.fps, plan.layers.len()));

    // ---- frame supply per layer: decode grid, neighbour blending and optical flow
    // (`Project.frameInterpolation`; a clip needs interpolation when the output rate is above its
    // effective rate `source avg fps × speed`)
    let p_fps = plan.fps;
    let interp = plan.interpolation.clone();
    let mut flow_layers: Vec<(usize, f64)> = Vec::new();
    for (i, layer) in plan.layers.iter_mut().enumerate() {
        if layer.source.is_image {
            continue;
        }
        let Some(src_fps) = layer.avg_fps.filter(|f| *f > 0.0) else { continue };
        let supply = frame_supply(&interp, layer.clip.speed.optical_flow, src_fps, layer.min_rate, p_fps, slow_grid_factor(&layer.clip, src_fps, p_fps));
        if let Some(f) = supply.flow_target {
            flow_layers.push((i, f));
        }
        let mut src = (*layer.source).clone();
        if (src.decode_fps - supply.grid_fps).abs() > 1e-9 || (src.blend_below_fps - supply.blend_below_fps).abs() > 1e-9 {
            src.decode_fps = supply.grid_fps;
            src.index_rate = src.decode_fps / 1000.0;
            src.frame_count = ((layer.asset.duration_ms * src.decode_fps / 1000.0).floor() as i64).max(1);
            src.blend_below_fps = supply.blend_below_fps;
            layer.requests = Arc::new(requests_for(&src, &layer.resolved));
            layer.source = Arc::new(src);
        }
        if let Some(msg) = supply.describe(&layer.clip.id, src_fps, layer.min_rate, p_fps, &interp) {
            job.log("info", msg);
        }
    }

    // ---- optical flow (one GPU job at a time: wait for analysis / separation)
    let _gpu = if flow_layers.is_empty() {
        None
    } else {
        let g = job.jobs.gpu.acquire(&job.job_id, &job.ctl, || job.progress(0.0, GPU_WAIT_MESSAGE));
        if g.is_none() {
            return Err("cancelled".into());
        }
        g
    };
    let mut flow_disabled: Option<String> = None;
    let mut target_fps_unsupported = false;
    for (k, &(i, target)) in flow_layers.iter().enumerate() {
        let layer = &plan.layers[i];
        if let Some(why) = &flow_disabled {
            job.log("warn", format!("clip {}: optical flow unavailable ({why}); blending neighbour frames instead", layer.clip.id));
            continue;
        }
        let src_fps = layer.avg_fps.unwrap_or(layer.asset.fps).max(1.0);
        let (a, b) = flow_range(layer, src_fps);
        let base = PREP_PCT * 0.5 * k as f64 / flow_layers.len() as f64;
        let t0 = Instant::now();
        job.progress(base, format!("optical flow to {} fps for clip {}", decode::fmt_rate(target), layer.clip.id));
        let mut on_progress = |pct: f64, msg: String| {
            job.progress(base + PREP_PCT * 0.5 * pct.clamp(0.0, 1.0) / flow_layers.len() as f64, format!("optical flow: {msg}"));
        };
        let path = Path::new(&layer.asset.path);
        let mut spec = if target_fps_unsupported { flow::FlowSpec::Factor(flow::factor_for(target, src_fps)) } else { flow::FlowSpec::TargetFps(target) };
        let mut result = flow::interpolate(path, a, b, spec, &job.ctl, &mut on_progress);
        if let Err(e) = &result {
            if flow::is_unsupported_target_fps(e) {
                target_fps_unsupported = true;
                spec = flow::FlowSpec::Factor(flow::factor_for(target, src_fps));
                job.log("warn", format!("the pipeline has no `interpolate --target-fps`; using --factor {} instead", flow::factor_for(target, src_fps)));
                result = flow::interpolate(path, a, b, spec, &job.ctl, &mut on_progress);
            }
        }
        match result {
            Ok(fpath) => {
                let target = match spec {
                    flow::FlowSpec::TargetFps(f) => Some(f),
                    flow::FlowSpec::Factor(_) => None,
                };
                match flow_source(&fpath, layer, a, b, target, p_fps) {
                    Ok(src) => {
                        job.log(
                            "info",
                            format!(
                                "clip {}: optical-flow intermediate {} ({} frames at {} fps covering {a:.0}–{b:.0} ms, {:.1}s)",
                                layer.clip.id,
                                fpath.display(),
                                src.frame_count,
                                decode::fmt_rate(src.decode_fps),
                                t0.elapsed().as_secs_f64()
                            ),
                        );
                        let requests = requests_for(&src, &layer.resolved);
                        let layer = &mut plan.layers[i];
                        layer.source = Arc::new(src);
                        layer.requests = Arc::new(requests);
                    }
                    Err(e) => job.log("warn", format!("clip {}: unusable optical-flow intermediate ({e}); blending neighbour frames instead", layer.clip.id)),
                }
            }
            Err(e) if e == "cancelled" || job.ctl.is_cancelled() => return Err("cancelled".into()),
            Err(e) => {
                job.log("warn", format!("clip {}: optical flow unavailable ({e}); blending neighbour frames instead", layer.clip.id));
                // A missing interpreter / subcommand fails the same way for every clip: stop trying.
                if e.contains("invalid choice") || e.contains("not found") || e.contains("cannot start") || flow::is_unsupported_target_fps(&e) {
                    flow_disabled = Some(e);
                }
            }
        }
    }
    drop(_gpu);

    // ---- audio
    job.progress(PREP_PCT * 0.5, "rendering audio");
    let log = job.log_fn();
    let mut log_audio = |level: &str, msg: String| log(level, msg);
    let wav = audio::render_audio(&job.project, plan.start_ms, plan.duration_ms, &job.tmp_dir, &job.bins.ffmpeg, &job.ctl, &mut log_audio)?;
    if job.ctl.is_cancelled() {
        return Err("cancelled".into());
    }

    // ---- video
    let wants_nvenc = job.preset == ExportPreset::H264NvencMp4;
    let nvenc = wants_nvenc && !NVENC_UNAVAILABLE.load(Ordering::SeqCst);
    if wants_nvenc && !nvenc {
        job.log("info", "NVENC failed earlier in this session; encoding with libx264");
    }
    match render_video(job, &plan, &wav, nvenc) {
        Err(e) if nvenc && !job.ctl.is_cancelled() && is_nvenc_failure(&e) => {
            NVENC_UNAVAILABLE.store(true, Ordering::SeqCst);
            job.log("warn", format!("h264_nvenc failed ({}); retrying with libx264", e.lines().last().unwrap_or("")));
            render_video(job, &plan, &wav, false)
        }
        other => other,
    }
}

/// Decode-grid multiplier for a slow-motion clip: the project-fps grid is
/// refined so real in-between source frames of high-frame-rate footage are
/// available for slow motion — `min(ceil(1 / minSpeed), round(sourceFps / projectFps))`.
pub fn slow_grid_factor(clip: &Clip, source_avg_fps: f64, project_fps: f64) -> u32 {
    if clip.speed.points.is_empty() || clip.source_duration_ms() <= 0.0 || source_avg_fps <= 0.0 {
        return 1;
    }
    let min = (0..=64).map(|i| clip.speed.speed_at(i as f64 / 64.0)).fold(f64::INFINITY, f64::min);
    if min >= 0.999 {
        return 1;
    }
    let by_speed = (1.0 / min).ceil();
    let by_source = (source_avg_fps / project_fps).round();
    by_speed.min(by_source).clamp(1.0, 8.0) as u32
}

/// How a layer gets the frames the output needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameSupply {
    /// decode grid (frames per source second)
    pub grid_fps: f64,
    /// see [`VideoSource::blend_below_fps`]
    pub blend_below_fps: f64,
    /// optical-flow target rate when the clip is interpolated (neighbour blending otherwise /
    /// when it fails)
    pub flow_target: Option<f64>,
    /// the clip needs more frames than its source has
    pub needs_frames: bool,
    /// the slow-motion refinement of the project-fps grid
    pub slow_grid: u32,
}

/// Frame supply of a clip (`frameInterpolation`, see the module docs of [`flow`]):
/// * the output needs no more frames than the source has → the pre-v2 path: the project-fps
///   grid (refined by `slow_grid` for slow motion from high-fps footage), neighbour blending
///   below 1× (none with `frameInterpolation: 'none'`);
/// * it needs more and the mode is `none` → the nearest source frame is repeated;
/// * `frameBlend` → the source is decoded on its own frame grid (when slower than the output)
///   and the two neighbouring source frames are cross-faded;
/// * `opticalFlow` (the default) → the pipeline interpolates at [`flow::target_fps`], with
///   `frameBlend` as the fallback. A clip whose `speed.opticalFlow` is `false` (the Speed tab's
///   toggle; `true` by default) opts out and gets `frameBlend`.
pub fn frame_supply(interp: &FrameInterpolation, clip_flow: bool, src_fps: f64, min_rate: f64, project_fps: f64, slow_grid: u32) -> FrameSupply {
    let refined = project_fps * slow_grid.max(1) as f64;
    let needs = flow::needs_interpolation(project_fps, src_fps, min_rate);
    let none = *interp == FrameInterpolation::None;
    if !needs {
        let blend = if none { 0.0 } else { refined * 0.999 };
        return FrameSupply { grid_fps: refined, blend_below_fps: blend, flow_target: None, needs_frames: false, slow_grid };
    }
    let want_flow = *interp == FrameInterpolation::OpticalFlow && clip_flow;
    if none {
        return FrameSupply { grid_fps: refined, blend_below_fps: 0.0, flow_target: None, needs_frames: true, slow_grid };
    }
    let grid = if src_fps < project_fps { src_fps } else { refined };
    FrameSupply {
        grid_fps: grid,
        blend_below_fps: project_fps * 0.999,
        flow_target: if want_flow { flow::target_fps(project_fps, src_fps, min_rate) } else { None },
        needs_frames: true,
        slow_grid,
    }
}

impl FrameSupply {
    /// A log line for clips that are not played frame for frame.
    pub fn describe(&self, clip_id: &str, src_fps: f64, min_rate: f64, project_fps: f64, interp: &FrameInterpolation) -> Option<String> {
        if !self.needs_frames {
            return (self.slow_grid > 1).then(|| format!("clip {clip_id}: slow motion from {src_fps:.2} fps footage, decoding on a {}× grid", self.slow_grid));
        }
        let how = match self.flow_target {
            Some(f) => format!("optical flow at {} fps (frame blending if it fails)", decode::fmt_rate(f)),
            None if self.blend_below_fps > 0.0 => "blending neighbour source frames".into(),
            None => "repeating the nearest source frame".into(),
        };
        Some(format!(
            "clip {clip_id}: {src_fps:.2} fps source at {min_rate:.2}× = {:.1} fps effective, {} fps output ({}): {how}",
            src_fps * min_rate,
            decode::fmt_rate(project_fps),
            interp.as_str()
        ))
    }
}

/// Source range (ms, whole) an optical-flow run must cover for a layer: every source time the
/// export shows, ± 2 source frames, within the file.
pub fn flow_range(layer: &LayerPlan, src_fps: f64) -> (f64, f64) {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for r in layer.resolved.iter() {
        lo = lo.min(r.source_ms);
        hi = hi.max(r.source_ms);
    }
    if !lo.is_finite() {
        return (layer.clip.in_ms.floor(), layer.clip.out_ms.ceil());
    }
    let margin = 2000.0 / src_fps.max(1.0);
    let top = if layer.asset.duration_ms > 0.0 { layer.asset.duration_ms } else { f64::MAX };
    ((lo - margin).max(0.0).floor(), (hi + margin).min(top).ceil().max((lo - margin).max(0.0).floor() + 1.0))
}

/// Frame source for an optical-flow intermediate covering source `[a, b]`: frame `k` at source
/// time `a + k / target` for a `--target-fps` run, or spread evenly over `[a, b]` for a
/// `--factor` run.
fn flow_source(path: &Path, layer: &LayerPlan, a: f64, b: f64, target: Option<f64>, project_fps: f64) -> Result<VideoSource, String> {
    let probed = ffmpeg::probe_video_info(path).map_err(String::from)?;
    let fps = probed.avg_fps.unwrap_or(0.0);
    if fps <= 0.0 || probed.duration_ms <= 0.0 || probed.width == 0 {
        return Err("probe found no video".into());
    }
    let (decode_fps, t0, index_rate, frames) = match target {
        Some(f) => {
            if (fps - f).abs() > f * 0.02 {
                return Err(format!("the intermediate runs at {fps:.3} fps, not the requested {f} fps"));
            }
            (f, a.round(), f / 1000.0, (probed.duration_ms * f / 1000.0).round() as i64)
        }
        None => {
            let frames = (probed.duration_ms * fps / 1000.0).floor() as i64;
            let span = (b - a).max(1.0);
            let index_rate = frames as f64 / span;
            if index_rate <= layer.asset.fps / 1000.0 * 1.2 {
                return Err(format!("{frames} frames for {span:.0} ms is not denser than the source"));
            }
            (fps, a, index_rate, frames)
        }
    };
    let f = layer.source.dec_w as f64 / layer.source.file_w.max(1) as f64;
    let dw = (((probed.width as f64 * f).ceil() as usize).div_ceil(2) * 2).clamp(2, probed.width as usize);
    let dh = (((probed.height as f64 * f).ceil() as usize).div_ceil(2) * 2).clamp(2, probed.height as usize);
    let (in_matrix, in_range) = colour_options(Some(&probed), false);
    Ok(VideoSource {
        path: path.to_path_buf(),
        is_image: false,
        decode_fps,
        t0_ms: t0,
        index_rate,
        frame_count: frames.max(1),
        dec_w: dw,
        dec_h: dh,
        file_w: probed.width as usize,
        file_h: probed.height as usize,
        blend_below_fps: project_fps * 0.999,
        in_matrix,
        in_range,
    })
}

/// Composite one layer's frame `n` onto `canvas` (placement, mask, opacity, clip fade, blend).
fn draw_layer(plan: &TimelinePlan, l: &LayerPlan, prepared: &Prepared, n: i64, canvas: &mut Canvas) {
    let k = (n - l.first_frame) as usize;
    let r = &l.resolved[k];
    let placement = Placement {
        canvas_w: plan.width as f64,
        canvas_h: plan.height as f64,
        source_w: l.asset.width.max(1) as f64,
        source_h: l.asset.height.max(1) as f64,
        crop: r.crop,
        scale: r.scale,
        position: r.position,
        rotation_deg: r.rotation,
    };
    let mask = match (&l.clip.mask, r.mask_rect) {
        (Some(m), Some(rect)) => Some(MaskParams {
            shape: m.shape,
            rect: [rect[0] as f32, rect[1] as f32, rect[2] as f32, rect[3] as f32],
            feather: m.feather as f32,
            inverted: m.inverted,
        }),
        _ => None,
    };
    let layer = Layer {
        source: &prepared.image,
        detail: prepared.detail.as_ref(),
        uv: UvMatrix::new(&placement),
        grade: &l.grade,
        baked: l.baked.as_deref(),
        opacity: r.opacity.clamp(0.0, 1.0) as f32,
        mask,
        blend: l.clip.blend_mode,
        time_ms: plan.frame_time_ms(n) as f32,
        fade: l.fades.as_ref().map(|f| f[k]).unwrap_or(0.0),
    };
    composite_layer(canvas, &layer);
}

fn render_video(job: &ExportJob, plan: &TimelinePlan, wav: &Path, nvenc: bool) -> Result<Stats, String> {
    let spec = EncodeSpec {
        width: plan.width,
        height: plan.height,
        fps: plan.fps,
        audio: wav.to_path_buf(),
        out: job.partial_path.clone(),
        preset: job.preset,
        nvenc,
        total_frames: plan.frame_count as u64,
    };
    let mut enc = Encoder::start(&job.bins.ffmpeg, &spec, &job.ctl, job.sink.clone(), &job.job_id, PREP_PCT)?;
    job.progress(PREP_PCT, format!("rendering with {}{}", job.preset.name(), if nvenc { " (nvenc)" } else { "" }));
    let lookahead = (plan.fps.ceil() as i64).max(8);
    let mut producers: Vec<Option<Producer>> = plan.layers.iter().map(|_| None).collect();
    let mut started_layer = vec![false; plan.layers.len()];
    let mut canvas = Canvas::new(plan.width, plan.height);
    // scratch canvases for the two sides of a transition
    let (mut tmp_a, mut tmp_b) = if plan.transitions.is_empty() {
        (Canvas::new(0, 0), Canvas::new(0, 0))
    } else {
        (Canvas::new(plan.width, plan.height), Canvas::new(plan.width, plan.height))
    };
    let mut layer_transitions: Vec<Vec<usize>> = vec![Vec::new(); plan.layers.len()];
    for (t, tp) in plan.transitions.iter().enumerate() {
        layer_transitions[tp.a].push(t);
        layer_transitions[tp.b].push(t);
    }
    let frame_len = plan.width * plan.height * 3;
    let log = job.log_fn();
    let started = Instant::now();
    let mut tm = [0.0f64; 5];
    for n in 0..plan.frame_count {
        if job.ctl.is_cancelled() {
            return Err("cancelled".into());
        }
        // Start decoders a little ahead of their first frame so they are warm.
        for (i, l) in plan.layers.iter().enumerate() {
            if !started_layer[i] && n >= l.first_frame - lookahead && n < l.end_frame {
                started_layer[i] = true;
                producers[i] = Some(spawn_producer(l, job.bins.ffmpeg.clone(), job.ctl.clone(), log.clone())?);
            }
        }
        let t_clear = Instant::now();
        canvas.clear();
        tm[0] += t_clear.elapsed().as_secs_f64();
        // this frame's source image of every active layer (in draw order)
        let t_wait = Instant::now();
        let mut prepared: Vec<Option<Arc<Prepared>>> = Vec::with_capacity(plan.layers.len());
        for (i, l) in plan.layers.iter().enumerate() {
            if n < l.first_frame || n >= l.end_frame {
                prepared.push(None);
                continue;
            }
            let p = producers[i].as_ref().ok_or_else(|| format!("internal: no decoder for clip {}", l.clip.id))?;
            match p.rx.recv() {
                Ok(item) => prepared.push(Some(item?)),
                Err(_) if job.ctl.is_cancelled() => return Err("cancelled".into()),
                Err(_) => return Err(format!("decoder for clip {} stopped early", l.clip.id)),
            }
        }
        tm[1] += t_wait.elapsed().as_secs_f64();
        let t_comp = Instant::now();
        let vt = plan.video_time[n as usize];
        let mut done_transition = vec![false; plan.transitions.len()];
        for (i, l) in plan.layers.iter().enumerate() {
            let Some(pi) = &prepared[i] else { continue };
            // a transition this layer is part of, active now (both sides present)
            let tr = layer_transitions[i]
                .iter()
                .copied()
                .find(|&t| {
                    let t_ = &plan.transitions[t];
                    n >= t_.first_frame && n < t_.end_frame && prepared[t_.a].is_some() && prepared[t_.b].is_some()
                });
            match tr {
                Some(t) if done_transition[t] => continue,
                Some(t) => {
                    done_transition[t] = true;
                    let tp = &plan.transitions[t];
                    // A and B each over what the lower tracks show, then mixed
                    tmp_a.data.copy_from_slice(&canvas.data);
                    tmp_b.data.copy_from_slice(&canvas.data);
                    draw_layer(plan, &plan.layers[tp.a], prepared[tp.a].as_ref().unwrap(), n, &mut tmp_a);
                    draw_layer(plan, &plan.layers[tp.b], prepared[tp.b].as_ref().unwrap(), n, &mut tmp_b);
                    let p = transitions::progress(vt, tp.cut_ms, tp.dur_ms);
                    transitions::render(&tp.kind, &tmp_a, &tmp_b, p, &mut canvas);
                }
                None => draw_layer(plan, l, pi, n, &mut canvas),
            }
        }
        drop(prepared);
        for (i, l) in plan.layers.iter().enumerate() {
            if n + 1 >= l.end_frame && n >= l.first_frame {
                producers[i] = None;
            }
        }
        // effects on the composited frame, FX tracks in order (output time)
        let t = plan.frame_time_ms(n);
        for e in plan.effects.iter().filter(|e| n >= e.first_frame && n < e.end_frame) {
            fx::apply(&e.effect, t - e.start_ms, e.dur_ms, &mut canvas);
        }
        tm[2] += t_comp.elapsed().as_secs_f64();
        let t_out = Instant::now();
        let mut bytes = enc.buffer(frame_len);
        canvas.write_rgb24(&mut bytes);
        tm[3] += t_out.elapsed().as_secs_f64();
        let t_push = Instant::now();
        enc.push(bytes)?;
        tm[4] += t_push.elapsed().as_secs_f64();
    }
    let t_fin = Instant::now();
    enc.finish()?;
    let secs = started.elapsed().as_secs_f64();
    let per = |x: f64| x * 1000.0 / plan.frame_count.max(1) as f64;
    job.log(
        "info",
        format!(
            "timing per frame: clear {:.1} ms, waiting for decode {:.1} ms, composite {:.1} ms, rgb24 {:.1} ms, waiting for encoder {:.1} ms (+{:.0} ms flush)",
            per(tm[0]), per(tm[1]), per(tm[2]), per(tm[3]), per(tm[4]), t_fin.elapsed().as_secs_f64() * 1000.0
        ),
    );
    Ok(Stats { frames: plan.frame_count, video_fps: plan.frame_count as f64 / secs.max(1e-6) })
}

#[cfg(test)]
mod tests;
