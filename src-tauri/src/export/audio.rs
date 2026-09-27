//! Audio mix for the compositor exporter: 48 kHz stereo f32.
//!
//! The mix is rendered in 10 s blocks streamed into a float WAV (RF64 past
//! 4 GB). Every audible clip decodes only the part of its source range the
//! export needs, once, when the first block it overlaps is mixed, and is
//! dropped after its last block. A clip whose file cannot be decoded fails the
//! export (missing files are already rejected while planning). The PCM is then played
//! through the same timeline → source map as the video by a [`ClipVoice`]
//! (`export::stretch`): pitch-preserving WSOLA time-stretch when the clip changes
//! speed and `audio.keepPitch` (default), anti-aliased varispeed when `keepPitch`
//! is false, plain 1× playback otherwise; silence during freeze holds; reversed
//! clips play backwards. Each clip's gain is `dbToLin(gainDb + volume(t)) ×
//! fadeIn(t) × fadeOut(t)` (equal-power fades, keyframed dB volume). `cameraSnap`
//! effects add the procedural shutter at −6 dB. When any clip asks for
//! `audio.normalize` the mix goes through a two-pass `loudnorm=I=-14:TP=-1.5:LRA=11`
//! (linear mode).
//!
//! Clips on video tracks contribute their audio only when no audio-track clip
//! mirrors them (same `assetId` and `startMs`, muted or not), which is how the pipeline and
//! the UI represent linked audio — this avoids doubled sound. Audio-track clips are never
//! dropped by that rule, so the stem clips of "Separate to tracks" (audio assets with
//! `stemOf`) always play.
//!
//! Voice separation: a clip whose `audio.voice` is `voice` ("Isolate voice") or
//! `background` ("Remove vocals") decodes the asset's matching stem file
//! (`asset.stems`, 48 kHz WAVs on the source's timeline) instead of the source,
//! through the same in/out range, speed map, reverse, freeze and gain. Without
//! stems it logs a warning and uses the original audio. A mirrored audio-track
//! clip follows its own `audio.voice` (it is the one that is heard).

use super::stretch::{self, ClipVoice, PlayMode};
use crate::jobs::TaskCtl;
use crate::model::{Asset, AssetKind, Clip, EffectType, Project, TrackKind, VoiceMode};
use crate::render::fx;
use crate::render::timemap::ClipTimeMap;
use serde_json::Value;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

pub const SAMPLE_RATE: u32 = 48_000;
const CHANNELS: usize = 2;
const LOUDNORM: &str = "loudnorm=I=-14:TP=-1.5:LRA=11";

/// One clip that contributes sound.
#[derive(Debug, Clone)]
pub struct AudioClip<'a> {
    pub clip: &'a Clip,
    pub asset: &'a Asset,
}

/// Clips whose audio is part of the mix (see module docs for the rules).
pub fn audible_clips(project: &Project) -> Vec<AudioClip<'_>> {
    let mirrors: Vec<(&str, f64)> = project
        .tracks
        .iter()
        .filter(|t| t.kind == TrackKind::Audio)
        .flat_map(|t| t.clips.iter().map(|c| (c.asset_id.as_str(), c.start_ms)))
        .collect();
    let mut out = Vec::new();
    for track in &project.tracks {
        if track.muted || track.kind == TrackKind::Fx {
            continue;
        }
        for clip in &track.clips {
            if clip.audio.muted || clip.source_duration_ms() <= 0.0 {
                continue;
            }
            let Some(asset) = project.asset(&clip.asset_id) else { continue };
            let has_sound = match asset.kind {
                AssetKind::Audio => true,
                AssetKind::Video => asset.has_audio,
                AssetKind::Image | AssetKind::Lut => false,
            };
            if !has_sound {
                continue;
            }
            if track.kind == TrackKind::Video
                && mirrors.iter().any(|(id, start)| *id == clip.asset_id && (start - clip.start_ms).abs() < 1.0)
            {
                continue;
            }
            out.push(AudioClip { clip, asset });
        }
    }
    out
}

/// The file a clip's sound is decoded from (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct SoundSource<'a> {
    pub path: &'a str,
    /// `Some("vocals" | "background")` when a stem replaces the source audio
    pub stem: Option<&'static str>,
    /// why the requested stem could not be used (the original audio is used instead)
    pub warning: Option<String>,
}

/// Pick the source or the stem for `clip` according to `clip.audio.voice`.
pub fn sound_source<'a>(clip: &Clip, asset: &'a Asset) -> SoundSource<'a> {
    // a stem clip ("Separate to tracks") plays its own file; its voice mode is ignored
    if asset.stem_of.is_some() {
        return SoundSource { path: &asset.path, stem: None, warning: None };
    }
    let (stem, label) = match clip.audio.voice {
        VoiceMode::Original => return SoundSource { path: &asset.path, stem: None, warning: None },
        VoiceMode::Voice => ("vocals", "isolate voice"),
        VoiceMode::Background => ("background", "remove vocals"),
    };
    let fallback = |why: String| SoundSource {
        path: &asset.path,
        stem: None,
        warning: Some(format!("clip {} ({label}): {why}; exporting its original audio", clip.id)),
    };
    let Some(stems) = asset.stems.as_ref() else {
        return fallback(format!("{} has not been separated yet (no stems)", asset.name));
    };
    let path = if stem == "vocals" { &stems.vocals } else { &stems.background };
    if path.trim().is_empty() || !Path::new(path).is_file() {
        return fallback(format!("{stem} stem of {} is missing ({path})", asset.name));
    }
    SoundSource { path, stem: Some(stem), warning: None }
}

/// Decode `[in_ms, in_ms + dur_ms)` of a file to interleaved stereo f32 at 48 kHz.
/// The PCM is converted while it streams in (no second full-size byte buffer).
pub fn decode_pcm(ffmpeg: &Path, path: &Path, in_ms: f64, dur_ms: f64, ctl: &TaskCtl) -> Result<Vec<f32>, String> {
    let mut cmd = crate::ffmpeg::command(ffmpeg);
    cmd.args(["-hide_banner", "-v", "error", "-nostdin"]);
    if in_ms > 0.0 {
        cmd.args(["-ss", &format!("{:.6}", in_ms / 1000.0)]);
    }
    cmd.arg("-i").arg(path);
    cmd.args(["-t", &format!("{:.6}", dur_ms.max(0.0) / 1000.0), "-vn", "-sn", "-dn", "-ac", "2", "-ar", &SAMPLE_RATE.to_string(), "-f", "f32le", "pipe:1"]);
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = crate::procs::spawn(&mut cmd).map_err(|e| format!("cannot start ffmpeg: {e}"))?;
    let mut stdout = child.stdout.take().ok_or("no stdout")?;
    let mut stderr = child.stderr.take().ok_or("no stderr")?;
    let err_thread = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = stderr.read_to_string(&mut s);
        s
    });
    let child = ctl.register(child);
    let expected = (dur_ms.max(0.0) / 1000.0 * SAMPLE_RATE as f64) as usize * CHANNELS;
    let mut samples: Vec<f32> = Vec::with_capacity(expected.min(1 << 28) + 1024);
    let mut buf = vec![0u8; 256 * 1024];
    let mut carry = 0usize; // bytes of an incomplete sample at the start of buf
    loop {
        let n = match stdout.read(&mut buf[carry..]) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        };
        let have = carry + n;
        let whole = have / 4 * 4;
        samples.extend(buf[..whole].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
        buf.copy_within(whole..have, 0);
        carry = have - whole;
    }
    let status = crate::procs::wait_child(&child).map_err(|e| e.to_string())?;
    let err = err_thread.join().unwrap_or_default();
    if ctl.is_cancelled() {
        return Err("cancelled".into());
    }
    if !status.success() && samples.is_empty() {
        return Err(format!("audio decode of {} failed: {}", path.display(), err.trim()));
    }
    Ok(samples)
}

/// Mix statistics.
pub struct MixInfo {
    /// any mixed clip asked for loudness normalisation
    pub normalize: bool,
    pub clip_count: usize,
}

/// Result of [`render_mix`].
pub struct Mix {
    /// interleaved stereo
    pub samples: Vec<f32>,
    pub normalize: bool,
    pub clip_count: usize,
}

/// Mix every audible clip for `[start_ms, start_ms + len_samples / 48k)` into memory
/// (tests; the exporter streams with [`mix_stream`]).
pub fn render_mix(
    project: &Project,
    start_ms: f64,
    len_samples: usize,
    ffmpeg: &Path,
    ctl: &TaskCtl,
    log: &mut dyn FnMut(&str, String),
) -> Result<Mix, String> {
    let mut samples = Vec::with_capacity(len_samples * CHANNELS);
    let info = mix_stream(project, start_ms, len_samples, ffmpeg, ctl, log, &mut |block| {
        samples.extend_from_slice(block);
        Ok(())
    })?;
    Ok(Mix { samples, normalize: info.normalize, clip_count: info.clip_count })
}

/// One clip scheduled for the mix.
struct Scheduled<'a> {
    ac: AudioClip<'a>,
    map: ClipTimeMap,
    /// output sample range `[i0, i1)` it covers
    i0: usize,
    i1: usize,
}

/// A clip being mixed: its decoded source window played by a [`ClipVoice`].
struct Active<'a> {
    s: Scheduled<'a>,
    voice: ClipVoice,
}

/// Output samples per mixing block (10 s): memory stays bounded by the block plus
/// the PCM of the clips that overlap it, whatever the export length.
const MIX_BLOCK: usize = SAMPLE_RATE as usize * 10;

/// The part of a clip's source range the output samples `[i0, i1)` read (ms, with
/// a small margin for interpolation), from the monotone time map.
fn source_window(clip: &Clip, map: &ClipTimeMap, start_ms: f64, i0: usize, i1: usize) -> (f64, f64) {
    let spm = SAMPLE_RATE as f64 / 1000.0;
    let (t0, t1) = (start_ms + i0 as f64 / spm - clip.start_ms, start_ms + i1 as f64 / spm - clip.start_ms);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for k in 0..=64 {
        let s = map.source_at(t0 + (t1 - t0) * k as f64 / 64.0).1;
        lo = lo.min(s);
        hi = hi.max(s);
    }
    // WSOLA frames read up to half a frame + the similarity tolerance around the mapped position
    let margin = match stretch::play_mode(clip) {
        PlayMode::Stretch => 20.0 + (stretch::WSOLA_FRAME / 2 + stretch::WSOLA_TOLERANCE) as f64 / spm,
        _ => 20.0,
    };
    let lo = (lo - margin).max(clip.in_ms);
    let hi = (hi + margin).min(clip.out_ms);
    (lo, hi.max(lo))
}

/// `cameraSnap` effects that sound in the range: (output sample of the effect's start, linear
/// gain = −6 dB × intensity). Effects on muted FX tracks are silent.
pub fn shutter_events(project: &Project, start_ms: f64, len_samples: usize) -> Vec<(i64, f32)> {
    let spm = SAMPLE_RATE as f64 / 1000.0;
    let end_ms = start_ms + len_samples as f64 / spm;
    let mut out = Vec::new();
    for t in project.tracks.iter().filter(|t| t.kind == TrackKind::Fx && !t.muted) {
        for c in &t.clips {
            let Some(e) = &c.effect else { continue };
            if e.kind != EffectType::CameraSnap || c.source_duration_ms() <= 0.0 {
                continue;
            }
            if c.start_ms + fx::SHUTTER_MS <= start_ms || c.start_ms >= end_ms {
                continue;
            }
            let gain = 10f64.powf(fx::SHUTTER_DB / 20.0) * e.intensity.clamp(0.0, 1.0);
            out.push((((c.start_ms - start_ms) * spm).round() as i64, gain as f32));
        }
    }
    out
}

/// Decode a clip's window (stem or source per `audio.voice`; a stem that cannot
/// be decoded falls back to the source with a warning). A source that cannot be
/// decoded fails the export — silence would be a silent surprise.
fn decode_clip(
    s: &Scheduled,
    start_ms: f64,
    ffmpeg: &Path,
    ctl: &TaskCtl,
    log: &mut dyn FnMut(&str, String),
) -> Result<(Vec<f32>, f64), String> {
    let clip = s.ac.clip;
    let (w0, w1) = source_window(clip, &s.map, start_ms, s.i0, s.i1);
    let src = sound_source(clip, s.ac.asset);
    if let Some(w) = &src.warning {
        log("warn", w.clone());
    }
    let decoded = decode_pcm(ffmpeg, Path::new(src.path), w0, w1 - w0, ctl);
    let decoded = match (decoded, src.stem) {
        (Err(e), Some(stem)) if e != "cancelled" => {
            log("warn", format!("clip {}: the {stem} stem could not be decoded ({e}); exporting its original audio", clip.id));
            decode_pcm(ffmpeg, Path::new(&s.ac.asset.path), w0, w1 - w0, ctl)
        }
        (other, Some(stem)) => {
            log("info", format!("audio: clip {} plays the {stem} stem of {}", clip.id, s.ac.asset.name));
            other
        }
        (other, None) => other,
    };
    match decoded {
        Ok(p) => Ok((p, w0)),
        Err(e) if e == "cancelled" => Err(e),
        Err(e) => Err(format!("clip {}: {e}", clip.id)),
    }
}

/// Mix every audible clip for `[start_ms, start_ms + len_samples / 48k)` in blocks of
/// [`MIX_BLOCK`] samples, handing each interleaved stereo block to `sink`. Each clip
/// decodes only the part of its source the range needs, once.
pub fn mix_stream(
    project: &Project,
    start_ms: f64,
    len_samples: usize,
    ffmpeg: &Path,
    ctl: &TaskCtl,
    log: &mut dyn FnMut(&str, String),
    sink: &mut dyn FnMut(&[f32]) -> Result<(), String>,
) -> Result<MixInfo, String> {
    let spm = SAMPLE_RATE as f64 / 1000.0; // samples per ms
    let end_ms = start_ms + len_samples as f64 / spm;
    let mut scheduled: Vec<Scheduled> = audible_clips(project)
        .into_iter()
        .filter_map(|ac| {
            let map = ClipTimeMap::new(ac.clip);
            let c0 = ac.clip.start_ms.max(start_ms);
            let c1 = (ac.clip.start_ms + map.total_ms()).min(end_ms);
            if c1 <= c0 {
                return None;
            }
            let i0 = ((c0 - start_ms) * spm).ceil().max(0.0) as usize;
            let i1 = (((c1 - start_ms) * spm).ceil() as usize).min(len_samples);
            (i1 > i0).then_some(Scheduled { ac, map, i0, i1 })
        })
        .collect();
    scheduled.sort_by_key(|s| s.i0);
    let shutters = shutter_events(project, start_ms, len_samples);
    let shutter = if shutters.is_empty() { Vec::new() } else { crate::render::fx::procedural_shutter(SAMPLE_RATE) };
    let mut pending = scheduled.into_iter().peekable();
    let mut active: Vec<Active> = Vec::new();
    let mut info = MixInfo { normalize: false, clip_count: 0 };
    let mut block = vec![0.0f32; MIX_BLOCK * CHANNELS];
    let mut b0 = 0usize;
    while b0 < len_samples {
        if ctl.is_cancelled() {
            return Err("cancelled".into());
        }
        let b1 = (b0 + MIX_BLOCK).min(len_samples);
        while pending.peek().map(|s| s.i0 < b1).unwrap_or(false) {
            let s = pending.next().unwrap();
            let (pcm, win_start_ms) = decode_clip(&s, start_ms, ffmpeg, ctl, log)?;
            info.clip_count += 1;
            info.normalize |= s.ac.clip.audio.normalize;
            let voice = ClipVoice::new(s.ac.clip, s.map.clone(), pcm, win_start_ms, SAMPLE_RATE);
            if voice.mode() != PlayMode::Direct {
                let how = if voice.mode() == PlayMode::Stretch { "time-stretched (WSOLA, pitch kept)" } else { "varispeed (pitch follows speed, anti-aliased)" };
                log("info", format!("audio: clip {} {how}", s.ac.clip.id));
            }
            active.push(Active { s, voice });
        }
        let out = &mut block[..(b1 - b0) * CHANNELS];
        out.fill(0.0);
        for a in active.iter_mut() {
            let (c0, c1) = (a.s.i0.max(b0), a.s.i1.min(b1));
            if c1 > c0 {
                a.voice.mix_into(start_ms + c0 as f64 / spm, &mut out[(c0 - b0) * CHANNELS..(c1 - b0) * CHANNELS]);
            }
        }
        for &(at, gain) in &shutters {
            // shutter sample k plays at output sample `at + k`
            let (k0, k1) = ((b0 as i64 - at).max(0), (b1 as i64 - at).min(shutter.len() as i64));
            for k in k0..k1 {
                let o = (at + k) as usize - b0;
                let v = shutter[k as usize] * gain;
                out[o * CHANNELS] += v;
                out[o * CHANNELS + 1] += v;
            }
        }
        active.retain(|a| a.s.i1 > b1);
        sink(out)?;
        b0 = b1;
    }
    Ok(info)
}

/// Write interleaved f32 samples as a WAVE_FORMAT_IEEE_FLOAT file (small files: tests,
/// fixtures). Long mixes use [`WavWriter`].
pub fn write_wav_f32(path: &Path, samples: &[f32], rate: u32, channels: u16) -> std::io::Result<()> {
    let data_len = (samples.len() * 4) as u32;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&3u16.to_le_bytes())?; // IEEE float
    f.write_all(&channels.to_le_bytes())?;
    f.write_all(&rate.to_le_bytes())?;
    f.write_all(&(rate * channels as u32 * 4).to_le_bytes())?;
    f.write_all(&(channels * 4).to_le_bytes())?;
    f.write_all(&32u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    for s in samples {
        f.write_all(&s.to_le_bytes())?;
    }
    f.flush()
}

/// Size of the header [`wav_header`] writes.
pub const WAV_HEADER_LEN: usize = 80;

/// Header of a float WAV with `data_bytes` of samples: a plain RIFF file with a
/// 28-byte `JUNK` chunk, or — past the 32-bit size limit (~3.1 h of 48 kHz stereo
/// float) — RF64, where that chunk becomes `ds64` with the 64-bit sizes (EBU Tech 3306).
pub fn wav_header(data_bytes: u64, rate: u32, channels: u16) -> Vec<u8> {
    let mut h = Vec::with_capacity(WAV_HEADER_LEN);
    let riff_size = data_bytes + (WAV_HEADER_LEN as u64 - 8);
    let rf64 = riff_size > u32::MAX as u64;
    h.extend_from_slice(if rf64 { b"RF64" } else { b"RIFF" });
    h.extend_from_slice(&(if rf64 { u32::MAX } else { riff_size as u32 }).to_le_bytes());
    h.extend_from_slice(b"WAVE");
    h.extend_from_slice(if rf64 { b"ds64" } else { b"JUNK" });
    h.extend_from_slice(&28u32.to_le_bytes());
    if rf64 {
        let frames = data_bytes / (channels as u64 * 4);
        h.extend_from_slice(&riff_size.to_le_bytes());
        h.extend_from_slice(&data_bytes.to_le_bytes());
        h.extend_from_slice(&frames.to_le_bytes());
        h.extend_from_slice(&0u32.to_le_bytes()); // table length
    } else {
        h.extend_from_slice(&[0u8; 28]);
    }
    h.extend_from_slice(b"fmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    h.extend_from_slice(&channels.to_le_bytes());
    h.extend_from_slice(&rate.to_le_bytes());
    h.extend_from_slice(&(rate * channels as u32 * 4).to_le_bytes());
    h.extend_from_slice(&(channels * 4).to_le_bytes());
    h.extend_from_slice(&32u16.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&(if rf64 { u32::MAX } else { data_bytes as u32 }).to_le_bytes());
    debug_assert_eq!(h.len(), WAV_HEADER_LEN);
    h
}

/// Streaming float WAV writer (RF64 when the data outgrows 4 GB).
pub struct WavWriter {
    f: std::io::BufWriter<std::fs::File>,
    data_bytes: u64,
    rate: u32,
    channels: u16,
    scratch: Vec<u8>,
}

impl WavWriter {
    pub fn create(path: &Path, rate: u32, channels: u16) -> std::io::Result<Self> {
        let mut f = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
        f.write_all(&wav_header(0, rate, channels))?;
        Ok(Self { f, data_bytes: 0, rate, channels, scratch: Vec::new() })
    }

    pub fn write(&mut self, samples: &[f32]) -> std::io::Result<()> {
        self.scratch.clear();
        self.scratch.extend(samples.iter().flat_map(|s| s.to_le_bytes()));
        self.f.write_all(&self.scratch)?;
        self.data_bytes += self.scratch.len() as u64;
        Ok(())
    }

    pub fn finish(mut self) -> std::io::Result<u64> {
        use std::io::{Seek, SeekFrom};
        self.f.flush()?;
        let mut file = self.f.into_inner().map_err(|e| e.into_error())?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&wav_header(self.data_bytes, self.rate, self.channels))?;
        file.flush()?;
        Ok(self.data_bytes)
    }
}

/// Parse the JSON block `loudnorm=...:print_format=json` prints on stderr.
pub fn parse_loudnorm_json(stderr: &str) -> Option<Value> {
    let end = stderr.rfind('}')?;
    let start = stderr[..end].rfind('{')?;
    serde_json::from_str(&stderr[start..=end]).ok()
}

fn measured(v: &Value, key: &str) -> Option<f64> {
    v.get(key)?.as_str()?.trim().parse::<f64>().ok().filter(|x| x.is_finite())
}

/// Two-pass loudness normalisation of `input` into `output` (48 kHz f32 WAV).
/// Returns `Ok(false)` when the input is silent (nothing to normalise).
pub fn loudnorm_two_pass(ffmpeg: &Path, input: &Path, output: &Path, ctl: &TaskCtl) -> Result<bool, String> {
    let run = |args: Vec<String>| -> Result<(bool, String), String> {
        let mut cmd = crate::ffmpeg::command(ffmpeg);
        cmd.args(&args).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped());
        let mut child = crate::procs::spawn(&mut cmd).map_err(|e| format!("cannot start ffmpeg: {e}"))?;
        let mut stderr = child.stderr.take().ok_or("no stderr")?;
        let child = ctl.register(child);
        let mut s = String::new();
        let _ = stderr.read_to_string(&mut s);
        let status = crate::procs::wait_child(&child).map_err(|e| e.to_string())?;
        if ctl.is_cancelled() {
            return Err("cancelled".into());
        }
        Ok((status.success(), s))
    };
    let inp = input.to_string_lossy().into_owned();
    let (ok, err) = run(vec![
        "-hide_banner".into(), "-nostats".into(), "-nostdin".into(), "-v".into(), "info".into(),
        "-i".into(), inp.clone(),
        "-af".into(), format!("{LOUDNORM}:print_format=json"),
        "-f".into(), "null".into(), "-".into(),
    ])?;
    if !ok {
        return Err(format!("loudnorm analysis failed: {}", err.lines().last().unwrap_or("").trim()));
    }
    let json = parse_loudnorm_json(&err).ok_or("loudnorm analysis printed no measurements")?;
    let (Some(i), Some(tp), Some(lra), Some(thresh), Some(offset)) = (
        measured(&json, "input_i"),
        measured(&json, "input_tp"),
        measured(&json, "input_lra"),
        measured(&json, "input_thresh"),
        measured(&json, "target_offset"),
    ) else {
        return Ok(false); // -inf loudness: silence
    };
    let filter = format!(
        "{LOUDNORM}:measured_I={i}:measured_TP={tp}:measured_LRA={lra}:measured_thresh={thresh}:offset={offset}:linear=true:print_format=none"
    );
    let (ok, err) = run(vec![
        "-hide_banner".into(), "-nostats".into(), "-nostdin".into(), "-v".into(), "error".into(), "-y".into(),
        "-i".into(), inp,
        "-af".into(), filter,
        "-ar".into(), SAMPLE_RATE.to_string(), "-ac".into(), "2".into(), "-c:a".into(), "pcm_f32le".into(),
        "-rf64".into(), "auto".into(),
        output.to_string_lossy().into_owned(),
    ])?;
    if !ok {
        return Err(format!("loudnorm failed: {}", err.trim()));
    }
    Ok(true)
}

/// Render the whole soundtrack to `<dir>/mix.wav` (normalised when requested).
pub fn render_audio(
    project: &Project,
    start_ms: f64,
    duration_ms: f64,
    dir: &Path,
    ffmpeg: &Path,
    ctl: &Arc<TaskCtl>,
    log: &mut dyn FnMut(&str, String),
) -> Result<PathBuf, String> {
    let len = (duration_ms.max(0.0) * SAMPLE_RATE as f64 / 1000.0).round() as usize;
    let raw = dir.join("mix.wav");
    let werr = |e: std::io::Error| format!("cannot write {}: {e}", raw.display());
    let mut wav = WavWriter::create(&raw, SAMPLE_RATE, CHANNELS as u16).map_err(werr)?;
    let info = mix_stream(project, start_ms, len, ffmpeg, ctl, log, &mut |block| wav.write(block).map_err(|e| e.to_string()))?;
    wav.finish().map_err(werr)?;
    log("info", format!("audio: mixed {} clip(s), {:.2}s", info.clip_count, len as f64 / SAMPLE_RATE as f64));
    if !info.normalize {
        return Ok(raw);
    }
    let norm = dir.join("mix_loudnorm.wav");
    match loudnorm_two_pass(ffmpeg, &raw, &norm, ctl) {
        Ok(true) => {
            log("info", "audio: loudness normalised to -14 LUFS / -1.5 dBTP (two-pass loudnorm)".into());
            Ok(norm)
        }
        Ok(false) => Ok(raw),
        Err(e) if e == "cancelled" => Err(e),
        Err(e) => {
            log("warn", format!("{e}; exporting the un-normalised mix"));
            Ok(raw)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::*;

    fn asset(id: &str, kind: AssetKind, has_audio: bool) -> Asset {
        Asset { id: id.into(), path: format!("C:/{id}.mp4"), kind, has_audio, duration_ms: 5000.0, ..Default::default() }
    }

    fn clip(id: &str, asset: &str, start: f64) -> Clip {
        Clip { id: id.into(), asset_id: asset.into(), start_ms: start, in_ms: 0.0, out_ms: 1000.0, ..Default::default() }
    }

    #[test]
    fn mirrored_audio_is_not_doubled() {
        let mut p = Project {
            assets: vec![asset("a", AssetKind::Video, true), asset("b", AssetKind::Video, true), asset("i", AssetKind::Image, false)],
            ..Default::default()
        };
        p.tracks = vec![
            Track { id: "v".into(), kind: TrackKind::Video, clips: vec![clip("v1", "a", 0.0), clip("v2", "b", 1000.0), clip("v3", "i", 2000.0)], ..Default::default() },
            Track { id: "au".into(), kind: TrackKind::Audio, clips: vec![clip("a1", "a", 0.0)], ..Default::default() },
        ];
        let ids: Vec<&str> = audible_clips(&p).iter().map(|c| c.clip.id.as_str()).collect();
        assert_eq!(ids, vec!["v2", "a1"]);
        p.tracks[1].muted = true; // a muted mirror still suppresses the video clip's audio
        let ids: Vec<&str> = audible_clips(&p).iter().map(|c| c.clip.id.as_str()).collect();
        assert_eq!(ids, vec!["v2"]);
        p.tracks[0].clips[1].audio.muted = true;
        assert!(audible_clips(&p).is_empty());
    }

    #[test]
    fn stem_clips_are_never_dropped_by_the_mirror_rule() {
        // "Separate to tracks": video clip + its muted mirrored audio clip + two stem clips of
        // audio assets (stemOf set) on the Voice / Background tracks, all at the same start
        let mut vocals = asset("sv", AssetKind::Audio, true);
        vocals.stem_of = Some(StemOf { asset_id: "a".into(), stem: StemKind::Vocals });
        let mut bg = asset("sb", AssetKind::Audio, true);
        bg.stem_of = Some(StemOf { asset_id: "a".into(), stem: StemKind::Background });
        let mut p = Project { assets: vec![asset("a", AssetKind::Video, true), vocals, bg], ..Default::default() };
        let mut mirror = clip("m", "a", 0.0);
        mirror.audio.muted = true;
        let mut s1 = clip("s1", "sv", 0.0);
        s1.link_id = Some("lnk".into());
        let mut s2 = clip("s2", "sb", 0.0);
        s2.link_id = Some("lnk".into());
        p.tracks = vec![
            Track { id: "v".into(), kind: TrackKind::Video, clips: vec![clip("v1", "a", 0.0)], ..Default::default() },
            Track { id: "au".into(), kind: TrackKind::Audio, clips: vec![mirror], ..Default::default() },
            Track { id: "voice".into(), kind: TrackKind::Audio, role: Some(TrackRole::Voice), clips: vec![s1], ..Default::default() },
            Track { id: "bg".into(), kind: TrackKind::Audio, role: Some(TrackRole::Background), clips: vec![s2], ..Default::default() },
        ];
        let ids: Vec<&str> = audible_clips(&p).iter().map(|c| c.clip.id.as_str()).collect();
        assert_eq!(ids, vec!["s1", "s2"], "same sound as before: only the stems, never doubled");
        // muting one stem leaves the other
        p.tracks[2].clips[0].audio.muted = true;
        let ids: Vec<&str> = audible_clips(&p).iter().map(|c| c.clip.id.as_str()).collect();
        assert_eq!(ids, vec!["s2"]);
        // stem clips play their own file whatever their voice mode says
        p.tracks[2].clips[0].audio.voice = VoiceMode::Background;
        let s = sound_source(&p.tracks[2].clips[0], &p.assets[1]);
        assert_eq!((s.path, s.stem, s.warning), ("C:/sv.mp4", None, None));
        // a stem clip whose start matches a video clip of *another* asset is kept too
        p.tracks[3].clips[0].asset_id = "sb".into();
        p.tracks[0].clips[0].asset_id = "sb".into();
        assert!(audible_clips(&p).iter().any(|c| c.clip.id == "s2"));
    }

    #[test]
    fn shutter_events_follow_camera_snaps() {
        let mut p = Project::default();
        let mut snap = Clip { id: "e".into(), start_ms: 1000.0, in_ms: 0.0, out_ms: 1500.0, ..Default::default() };
        snap.effect = Some(ClipEffect { kind: EffectType::CameraSnap, intensity: 0.5, params: None });
        let mut sepia = snap.clone();
        sepia.effect = Some(ClipEffect::new(EffectType::Sepia));
        p.tracks = vec![Track { kind: TrackKind::Fx, clips: vec![snap, sepia], ..Default::default() }];
        let ev = shutter_events(&p, 0.0, 48_000 * 10);
        assert_eq!(ev.len(), 1);
        assert_eq!(ev[0].0, 48_000);
        assert!((ev[0].1 - 0.5 * 10f32.powf(-6.0 / 20.0)).abs() < 1e-6, "−6 dB × intensity");
        // a range starting inside the shutter still hears its tail (negative offset)
        assert_eq!(shutter_events(&p, 1050.0, 48_000)[0].0, -2400);
        assert!(shutter_events(&p, 1200.0, 48_000).is_empty());
        p.tracks[0].muted = true;
        assert!(shutter_events(&p, 0.0, 48_000 * 10).is_empty());
    }

    #[test]
    fn sound_source_follows_voice_mode_and_stems() {
        let dir = crate::ffmpeg::cache_dir().join("test").join("sound_source");
        std::fs::create_dir_all(&dir).unwrap();
        let v = dir.join("vocals.wav");
        write_wav_f32(&v, &[0.0; 4], 48000, 2).unwrap();
        let mut a = asset("a", AssetKind::Video, true);
        let mut c = clip("c", "a", 0.0);
        assert_eq!(sound_source(&c, &a), SoundSource { path: "C:/a.mp4", stem: None, warning: None });
        c.audio.voice = VoiceMode::Voice;
        let s = sound_source(&c, &a);
        assert_eq!((s.path, s.stem), ("C:/a.mp4", None));
        assert!(s.warning.unwrap().contains("not been separated"));
        a.stems = Some(Stems { vocals: v.to_string_lossy().into_owned(), background: dir.join("gone.wav").to_string_lossy().into_owned() });
        let s = sound_source(&c, &a);
        assert_eq!(s.stem, Some("vocals"));
        assert!(s.path.ends_with("vocals.wav") && s.warning.is_none());
        c.audio.voice = VoiceMode::Background;
        let s = sound_source(&c, &a);
        assert_eq!((s.path, s.stem), ("C:/a.mp4", None), "missing stem file falls back to the original");
        assert!(s.warning.unwrap().contains("background stem"));
    }

    #[test]
    fn long_mixes_switch_to_rf64() {
        let small = wav_header(1000, 48_000, 2);
        assert_eq!(small.len(), WAV_HEADER_LEN);
        assert_eq!(&small[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(small[4..8].try_into().unwrap()), 1000 + 72);
        assert_eq!(&small[12..16], b"JUNK");
        assert_eq!(&small[76..80], &1000u32.to_le_bytes());
        // 4 h of 48 kHz stereo float = 5.5 GB: sizes go to the ds64 chunk
        let big_bytes: u64 = 4 * 3600 * 48_000 * 2 * 4;
        let big = wav_header(big_bytes, 48_000, 2);
        assert_eq!(big.len(), WAV_HEADER_LEN);
        assert_eq!(&big[0..4], b"RF64");
        assert_eq!(&big[4..8], &u32::MAX.to_le_bytes());
        assert_eq!(&big[12..16], b"ds64");
        assert_eq!(u64::from_le_bytes(big[20..28].try_into().unwrap()), big_bytes + 72);
        assert_eq!(u64::from_le_bytes(big[28..36].try_into().unwrap()), big_bytes);
        assert_eq!(u64::from_le_bytes(big[36..44].try_into().unwrap()), big_bytes / 8);
        assert_eq!(&big[72..76], b"data");
        // a streamed file reads back with ffmpeg
        let Some(bins) = crate::ffmpeg::find_binaries().ok() else { return };
        let dir = crate::ffmpeg::cache_dir().join("test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("streamed.wav");
        let mut w = WavWriter::create(&p, 48_000, 2).unwrap();
        for _ in 0..3 {
            w.write(&vec![0.25f32; 48_000]).unwrap();
        }
        assert_eq!(w.finish().unwrap(), 3 * 48_000 * 4);
        let ctl = TaskCtl::default();
        let pcm = decode_pcm(&bins.ffmpeg, &p, 0.0, 10_000.0, &ctl).unwrap();
        assert_eq!(pcm.len(), 3 * 48_000);
        assert!(pcm.iter().all(|v| (*v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn wav_header_and_loudnorm_json() {
        let dir = crate::ffmpeg::cache_dir().join("test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("wav_header.wav");
        write_wav_f32(&p, &[0.0, 0.5, -0.5, 1.0], 48000, 2).unwrap();
        let b = std::fs::read(&p).unwrap();
        assert_eq!(&b[0..4], b"RIFF");
        assert_eq!(u16::from_le_bytes([b[20], b[21]]), 3);
        assert_eq!(b.len(), 44 + 16);
        let err = "[Parsed_loudnorm_0 @ 0x1]\n{\n\t\"input_i\" : \"-23.10\",\n\t\"input_tp\" : \"-5.0\",\n\t\"input_lra\" : \"1.2\",\n\t\"input_thresh\" : \"-33.4\",\n\t\"target_offset\" : \"0.1\"\n}\n";
        let v = parse_loudnorm_json(err).unwrap();
        assert_eq!(measured(&v, "input_i"), Some(-23.1));
        let silent = parse_loudnorm_json("{ \"input_i\" : \"-inf\" }").unwrap();
        assert_eq!(measured(&silent, "input_i"), None);
    }
}
