//! Encoder stage: one ffmpeg child reading `rawvideo rgb24` on stdin plus the
//! rendered WAV, fed by a writer thread through a bounded channel.

use super::ExportPreset;
use crate::jobs::{EventSink, TaskCtl};
use serde_json::json;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Everything the encoder needs to know about the output.
#[derive(Debug, Clone)]
pub struct EncodeSpec {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub audio: PathBuf,
    pub out: PathBuf,
    pub preset: ExportPreset,
    pub nvenc: bool,
    pub total_frames: u64,
}

/// ffmpeg arguments for [`EncodeSpec`].
pub fn encoder_args(spec: &EncodeSpec) -> Vec<String> {
    let mut a: Vec<String> = [
        "-hide_banner", "-v", "error", "-nostats", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    a.extend(["-s".into(), format!("{}x{}", spec.width, spec.height)]);
    a.extend(["-framerate".into(), super::decode::fmt_rate(spec.fps), "-i".into(), "pipe:0".into()]);
    a.extend(["-i".into(), spec.audio.to_string_lossy().into_owned()]);
    a.extend(["-map".into(), "0:v:0".into(), "-map".into(), "1:a:0".into()]);
    // RGB → Y'CbCr with the BT.709 matrix, tagged, limited range.
    a.extend([
        "-vf".into(),
        format!("scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd,format={}", spec.preset.pix_fmt()),
    ]);
    a.extend(
        ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv"]
            .iter()
            .map(|s| s.to_string()),
    );
    a.extend(spec.preset.video_args(spec.nvenc));
    a.extend(spec.preset.audio_args());
    a.extend(spec.preset.container_args());
    a.push(spec.out.to_string_lossy().into_owned());
    a
}

pub struct Encoder {
    tx: Option<SyncSender<Vec<u8>>>,
    /// frame buffers the writer has finished with (reused by [`Encoder::buffer`])
    recycled: Receiver<Vec<u8>>,
    writer: Option<JoinHandle<Result<(), String>>>,
    err_thread: Option<JoinHandle<()>>,
    child: Arc<Mutex<Child>>,
    tail: Arc<Mutex<Vec<String>>>,
    pub written: Arc<AtomicU64>,
}

fn fmt_eta(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    if s >= 60 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}

impl Encoder {
    /// Spawn ffmpeg and the writer thread. Progress is reported as
    /// `pct = pct_base + (1 - pct_base) * written / total`.
    pub fn start(
        ffmpeg: &Path,
        spec: &EncodeSpec,
        ctl: &Arc<TaskCtl>,
        sink: Arc<dyn EventSink>,
        job_id: &str,
        pct_base: f64,
    ) -> Result<Self, String> {
        let args = encoder_args(spec);
        tracing::info!("export {job_id}: ffmpeg {}", args.join(" "));
        let mut cmd = crate::ffmpeg::command(ffmpeg);
        cmd.args(&args).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped());
        let mut child = crate::procs::spawn(&mut cmd).map_err(|e| format!("cannot start the encoder: {e}"))?;
        let mut stdin = child.stdin.take().ok_or("encoder has no stdin")?;
        let stderr = child.stderr.take().ok_or("encoder has no stderr")?;
        let child = ctl.register(child);

        let tail: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let (t2, s2, id2) = (tail.clone(), sink.clone(), job_id.to_string());
        let err_thread = std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if line.trim().is_empty() {
                    continue;
                }
                s2.emit("export://log", json!({ "jobId": id2, "level": "warn", "message": line }));
                let mut t = t2.lock().unwrap();
                t.push(line);
                if t.len() > 40 {
                    t.remove(0);
                }
            }
        });

        let (tx, rx) = sync_channel::<Vec<u8>>(4);
        let (back_tx, recycled) = channel::<Vec<u8>>();
        let written = Arc::new(AtomicU64::new(0));
        let (w2, id3) = (written.clone(), job_id.to_string());
        let total = spec.total_frames.max(1);
        let writer = std::thread::spawn(move || -> Result<(), String> {
            let started = Instant::now();
            let mut last_emit = Instant::now() - Duration::from_secs(1);
            for frame in rx {
                stdin.write_all(&frame).map_err(|e| format!("encoder input closed: {e}"))?;
                let _ = back_tx.send(frame);
                let n = w2.fetch_add(1, Ordering::SeqCst) + 1;
                if last_emit.elapsed() >= Duration::from_millis(250) || n == total {
                    last_emit = Instant::now();
                    let el = started.elapsed().as_secs_f64();
                    let fps = n as f64 / el.max(1e-6);
                    let eta = (total - n.min(total)) as f64 / fps.max(1e-6);
                    let pct = pct_base + (1.0 - pct_base) * (n as f64 / total as f64);
                    sink.emit(
                        "export://progress",
                        json!({ "jobId": id3, "pct": pct.min(0.999),
                                "message": format!("frame {n}/{total} · {fps:.1} fps · ETA {}", fmt_eta(eta)) }),
                    );
                }
            }
            stdin.flush().map_err(|e| e.to_string())?;
            drop(stdin);
            Ok(())
        });

        Ok(Self { tx: Some(tx), recycled, writer: Some(writer), err_thread: Some(err_thread), child, tail, written })
    }

    /// A frame buffer of `len` bytes, recycled from frames already written when possible
    /// (contents unspecified: the caller overwrites it).
    pub fn buffer(&self, len: usize) -> Vec<u8> {
        while let Ok(b) = self.recycled.try_recv() {
            if b.len() == len {
                return b;
            }
        }
        vec![0u8; len]
    }

    fn tail_text(&self) -> String {
        self.tail.lock().unwrap().join("\n")
    }

    /// Queue one `rgb24` frame (blocks when the encoder is behind).
    pub fn push(&mut self, frame: Vec<u8>) -> Result<(), String> {
        let tx = self.tx.as_ref().ok_or("encoder already finished")?;
        if tx.send(frame).is_err() {
            // The writer stopped: collect its error and ffmpeg's output.
            let werr = self.writer.take().and_then(|h| h.join().ok()).and_then(|r| r.err()).unwrap_or_default();
            let _ = crate::procs::wait_child(&self.child);
            if let Some(h) = self.err_thread.take() {
                let _ = h.join();
            }
            return Err(format!("encoder failed ({werr}): {}", self.tail_text()));
        }
        Ok(())
    }

    /// Close stdin and wait for ffmpeg to finish writing the file.
    pub fn finish(mut self) -> Result<(), String> {
        drop(self.tx.take());
        let werr = match self.writer.take().map(|h| h.join()) {
            Some(Ok(Err(e))) => Some(e),
            Some(Err(_)) => Some("encoder writer panicked".into()),
            _ => None,
        };
        // poll without holding the child's lock, so a cancel can kill a slow flush
        let status = crate::procs::wait_child(&self.child).map_err(|e| e.to_string())?;
        if let Some(h) = self.err_thread.take() {
            let _ = h.join();
        }
        if !status.success() || werr.is_some() {
            return Err(format!(
                "encoder exited with {:?}{}: {}",
                status.code(),
                werr.map(|e| format!(" ({e})")).unwrap_or_default(),
                self.tail_text()
            ));
        }
        Ok(())
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        // Aborted render: make sure ffmpeg does not linger.
        if self.tx.is_some() {
            drop(self.tx.take());
            if let Ok(mut c) = self.child.lock() {
                let _ = c.kill();
                let _ = c.wait();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_for_each_preset() {
        let mut spec = EncodeSpec {
            width: 1280,
            height: 720,
            fps: 24.0,
            audio: PathBuf::from("C:/t/mix.wav"),
            out: PathBuf::from("C:/o/final.mp4"),
            preset: ExportPreset::H264Mp4,
            nvenc: false,
            total_frames: 10,
        };
        let a = encoder_args(&spec).join(" ");
        assert!(a.contains("-f rawvideo -pix_fmt rgb24 -s 1280x720 -framerate 24 -i pipe:0 -i C:/t/mix.wav -map 0:v:0 -map 1:a:0"), "{a}");
        assert!(a.contains("format=yuv420p"));
        assert!(a.contains("-c:v libx264 -preset medium -crf 18"), "{a}");
        assert!(a.contains("-c:a aac"));
        assert!(a.ends_with("-movflags +faststart C:/o/final.mp4"), "{a}");
        spec.preset = ExportPreset::H264NvencMp4;
        spec.nvenc = true;
        let a = encoder_args(&spec).join(" ");
        assert!(a.contains("-c:v h264_nvenc -preset p5 -rc vbr -cq 19"), "{a}");
        spec.nvenc = false;
        assert!(encoder_args(&spec).join(" ").contains("-c:v libx264"));
        spec.preset = ExportPreset::ProresMov;
        let a = encoder_args(&spec).join(" ");
        assert!(a.contains("format=yuv422p10le") && a.contains("-c:v prores_ks -profile:v 3") && a.contains("-c:a pcm_s16le"), "{a}");
    }
}
