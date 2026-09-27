use super::*;
use crate::jobs::CollectingSink;
use crate::model::*;
use std::process::Stdio;
use std::time::Duration;

/* ------------------------------------------------------------------ helpers */

fn test_dir() -> PathBuf {
    let d = ffmpeg::cache_dir().join("test");
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Synthesise a clip with lavfi (cached by file name).
fn synth(name: &str, video: &str, audio: Option<&str>) -> Option<PathBuf> {
    let bins = ffmpeg::find_binaries().ok()?;
    let out = test_dir().join(name);
    if out.is_file() {
        return Some(out);
    }
    // unique temp name + rename: parallel tests never see a half-written fixture
    let tmp = test_dir().join(format!("{}.{}.{:?}.tmp.mp4", name, std::process::id(), std::thread::current().id()).replace(['(', ')'], ""));
    let mut cmd = ffmpeg::command(&bins.ffmpeg);
    cmd.args(["-v", "error", "-y", "-f", "lavfi", "-i", video]);
    if let Some(a) = audio {
        cmd.args(["-f", "lavfi", "-i", a, "-c:a", "aac", "-shortest"]);
    }
    cmd.args(["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p"]);
    let ok = cmd.arg(&tmp).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status().ok()?.success();
    if !ok {
        return None;
    }
    if std::fs::rename(&tmp, &out).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    out.is_file().then_some(out)
}

fn asset_for(path: &Path, id: &str) -> Asset {
    let mut a = ffmpeg::probe(path).expect("probe");
    a.id = id.into();
    a
}

/// Run an export to completion; returns (done payload, all events).
fn run_export(project: Project, out: &Path, preset: &str, range: Option<ExportRange>) -> (serde_json::Value, Arc<CollectingSink>) {
    let _ = std::fs::remove_file(out);
    let jobs = Arc::new(JobManager::new());
    let sink = CollectingSink::new();
    let job = export_project(jobs.clone(), sink.clone(), project, out.to_string_lossy().into_owned(), preset, range).expect("start export");
    assert!(job.starts_with("export_"));
    assert!(jobs.wait(&job, Duration::from_secs(900)), "export timed out");
    let done = sink.named("export://done");
    assert_eq!(done.len(), 1, "exactly one done event");
    (done[0].clone(), sink)
}

/// `(nb_frames, duration_ms)` of the first video stream.
fn video_stream_info(path: &Path) -> (i64, f64) {
    let bins = ffmpeg::find_binaries().unwrap();
    let out = ffmpeg::command(&bins.ffprobe)
        .args(["-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames,duration", "-of", "json"])
        .arg(path)
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let s = &v["streams"][0];
    let frames = s["nb_read_frames"].as_str().and_then(|x| x.parse().ok()).unwrap_or(-1);
    let dur = s["duration"].as_str().and_then(|x| x.parse::<f64>().ok()).unwrap_or(-1.0) * 1000.0;
    (frames, dur)
}

/// Decode output frame `n` as rgb24.
fn frame_rgb(path: &Path, n: i64) -> Vec<u8> {
    let bins = ffmpeg::find_binaries().unwrap();
    let out = ffmpeg::command(&bins.ffmpeg)
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-vf", &format!("select=eq(n\\,{n})"), "-fps_mode", "passthrough", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(!out.stdout.is_empty(), "no frame {n}: {}", String::from_utf8_lossy(&out.stderr));
    out.stdout
}

/// Mean RGB over `[x0, x1) × [y0, y1)`.
fn region_mean(f: &[u8], width: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> [f64; 3] {
    let mut acc = [0.0f64; 3];
    for y in y0..y1 {
        for x in x0..x1 {
            let i = (y * width + x) * 3;
            for c in 0..3 {
                acc[c] += f[i + c] as f64;
            }
        }
    }
    let n = ((x1 - x0) * (y1 - y0)) as f64;
    acc.map(|v| v / n)
}

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).sum::<f64>() / a.len() as f64
}

fn logged_video_fps(sink: &CollectingSink) -> Option<f64> {
    sink.named("export://log").iter().filter_map(|e| e["message"].as_str().map(String::from)).find_map(|m| {
        let i = m.find("video stage ")?;
        m[i + 12..].split_whitespace().next()?.parse().ok()
    })
}

/* --------------------------------------------------------------- unit tests */

#[test]
fn preset_parsing_and_engines() {
    assert_eq!(ExportPreset::parse("h264_mp4").unwrap(), ExportPreset::H264Mp4);
    assert_eq!(ExportPreset::parse(" prores_mov ").unwrap(), ExportPreset::ProresMov);
    assert!(ExportPreset::parse("prores_mov_legacy").unwrap_err().contains("removed"), "the phase-1 exporter is gone");
    assert!(ExportPreset::parse("gif").is_err());
    assert!(ExportPreset::H264NvencMp4.video_args(true).contains(&"h264_nvenc".to_string()));
    assert!(ExportPreset::H264NvencMp4.video_args(false).contains(&"libx264".to_string()));
    let r: ExportRange = serde_json::from_value(serde_json::json!({ "startMs": 100, "endMs": 900 })).unwrap();
    assert_eq!(r, ExportRange { start_ms: 100.0, end_ms: 900.0 });
}

fn tiny_project(path: &str) -> Project {
    let asset = Asset {
        id: "a".into(),
        path: path.into(),
        name: "a.mp4".into(),
        kind: AssetKind::Video,
        duration_ms: 2000.0,
        width: 320,
        height: 240,
        fps: 24.0,
        has_audio: true,
        ..Default::default()
    };
    let c1 = Clip { id: "c1".into(), asset_id: "a".into(), start_ms: 0.0, in_ms: 0.0, out_ms: 1000.0, ..Default::default() };
    let c2 = Clip {
        id: "c2".into(),
        asset_id: "a".into(),
        start_ms: 1000.0,
        in_ms: 1000.0,
        out_ms: 2000.0,
        speed: SpeedCurve::constant(2.0),
        reversed: true,
        ..Default::default()
    };
    Project {
        fps: 24.0,
        width: 160,
        height: 120,
        assets: vec![asset],
        tracks: vec![Track { id: "v".into(), kind: TrackKind::Video, clips: vec![c1, c2], ..Default::default() }],
        ..Default::default()
    }
}

#[test]
fn timeline_plan_frames_ranges_and_errors() {
    let dir = test_dir();
    let fake = dir.join("plan_fake.mp4");
    std::fs::write(&fake, b"not really a video").unwrap();
    let p = tiny_project(&fake.to_string_lossy());
    let plan = build_timeline(&p, None).unwrap();
    assert_eq!(plan.frame_count, 36, "1000 ms + 1000 ms @2x = 1.5 s at 24 fps");
    assert_eq!(plan.layers.len(), 2);
    assert_eq!((plan.layers[0].first_frame, plan.layers[0].end_frame), (0, 24));
    assert_eq!((plan.layers[1].first_frame, plan.layers[1].end_frame), (24, 36));
    // reversed 2x clip: first output frame shows the end of its source range
    let r = &plan.layers[1].requests;
    assert_eq!(r[0].a, 47);
    assert!(r.windows(2).all(|w| w[1].a < w[0].a), "reverse playback walks backwards");
    assert_eq!(plan.layers[0].requests[5].a, 5);
    // decode size: 320x240 shown at 160x120 → half resolution
    assert_eq!((plan.layers[0].source.dec_w, plan.layers[0].source.dec_h), (160, 120));

    let part = build_timeline(&p, Some(ExportRange { start_ms: 500.0, end_ms: 1250.0 })).unwrap();
    assert_eq!(part.frame_count, 18);
    assert_eq!((part.layers[0].first_frame, part.layers[0].end_frame), (0, 12));
    assert_eq!(part.layers[0].requests[0].a, 12, "range starts half a second into clip 1");
    assert!(build_timeline(&p, Some(ExportRange { start_ms: 900.0, end_ms: 900.0 })).is_err());

    let mut muted = p.clone();
    muted.tracks[0].muted = true;
    assert_eq!(build_timeline(&muted, None).unwrap().layers.len(), 0, "muted track = hidden");
    let mut bad = p.clone();
    bad.tracks[0].clips[0].asset_id = "missing".into();
    assert!(build_timeline(&bad, None).unwrap_err().contains("unknown asset"));
    let mut gone = p.clone();
    gone.assets[0].path = dir.join("definitely_missing.mp4").to_string_lossy().into_owned();
    assert!(build_timeline(&gone, None).unwrap_err().contains("missing"));
    assert!(build_timeline(&Project::default(), None).unwrap_err().contains("empty"));
}

#[test]
fn slow_motion_grid_factor() {
    let mut c = Clip { in_ms: 0.0, out_ms: 3000.0, ..Default::default() };
    assert_eq!(slow_grid_factor(&c, 60.0, 24.0), 1, "1x never needs a finer grid");
    c.speed = SpeedCurve::constant(0.25);
    assert_eq!(slow_grid_factor(&c, 24.04, 24.0), 1, "24 fps footage has no in-between frames");
    assert_eq!(slow_grid_factor(&c, 60.0, 24.0), 3, "60 fps footage: round(2.5) = 3");
    assert_eq!(slow_grid_factor(&c, 240.0, 24.0), 4, "limited by 1 / 0.25");
    c.speed = SpeedCurve::constant(0.5);
    assert_eq!(slow_grid_factor(&c, 120.0, 24.0), 2);
}

#[test]
fn decode_size_accounts_for_crop_and_scale() {
    let asset = Asset { width: 3840, height: 2160, ..Default::default() };
    let mut clip = Clip::default();
    assert_eq!(decode_size(&clip, &asset, 1920, 1080), (1920, 1080));
    clip.transform.scale = Keyframed::with_keyframes(1.0, vec![Keyframe::new(0.0, 1.0), Keyframe::new(1000.0, 2.0)]);
    assert_eq!(decode_size(&clip, &asset, 1920, 1080), (3840, 2160), "zooming to 2x needs full resolution");
    let clip = Clip {
        reframe: Some(ReframeTrack { keyframes: vec![ReframeKeyframe { crop: [1000.0, 0.0, 2215.0, 2160.0], ..Default::default() }], ..Default::default() }),
        ..Default::default()
    };
    // a 1215x2160 crop shown full-height on a 1080x1920 canvas needs 1920/2160 of the source
    let (w, h) = decode_size(&clip, &asset, 1080, 1920);
    assert!((3410..=3420).contains(&w) && (1918..=1924).contains(&h), "{w}x{h}");
}

/* ------------------------------------------------------------ export runs */

#[test]
fn compositor_export_small_project() {
    let Some(clip) = ffmpeg::tests::synth_clip() else {
        eprintln!("SKIP: ffmpeg not available");
        return;
    };
    let project = tiny_project(&clip.to_string_lossy());
    let out = test_dir().join("export_v2_small.mp4");
    let (done, sink) = run_export(project, &out, "h264_mp4", None);
    assert_eq!(done["ok"], true, "export failed: {:?} / {:?}", done["error"], sink.named("export://log"));
    let probed = ffmpeg::probe(&out).unwrap();
    assert_eq!((probed.width, probed.height), (160, 120));
    assert!(probed.has_audio);
    let (frames, dur) = video_stream_info(&out);
    assert_eq!(frames, 36);
    assert!((dur - 1500.0).abs() <= 1000.0 / 24.0 + 1.0, "video duration {dur}");
    let progress = sink.named("export://progress");
    assert_eq!(progress.last().unwrap()["pct"], 1.0);
    assert!(progress.iter().any(|p| p["message"].as_str().unwrap_or("").contains("fps")));

    // range export + prores preset
    let out_mov = test_dir().join("export_v2_small_range.mov");
    let (done, _) = run_export(tiny_project(&clip.to_string_lossy()), &out_mov, "prores_mov", Some(ExportRange { start_ms: 250.0, end_ms: 1000.0 }));
    assert_eq!(done["ok"], true, "{:?}", done["error"]);
    assert_eq!(video_stream_info(&out_mov).0, 18);
}

#[test]
fn compositor_export_can_be_cancelled() {
    let Some(clip) = ffmpeg::tests::synth_clip() else {
        eprintln!("SKIP: ffmpeg not available");
        return;
    };
    let mut project = tiny_project(&clip.to_string_lossy());
    project.width = 1280;
    project.height = 960;
    let out = test_dir().join("export_v2_cancel.mp4");
    let jobs = Arc::new(JobManager::new());
    let sink = CollectingSink::new();
    let job = export_project(jobs.clone(), sink.clone(), project, out.to_string_lossy().into_owned(), "h264_mp4", None).unwrap();
    std::thread::sleep(Duration::from_millis(300));
    cancel_export(&jobs, &job).unwrap();
    assert!(jobs.wait(&job, Duration::from_secs(60)));
    let done = sink.named("export://done");
    assert_eq!(done.len(), 1);
    // it may have finished before the cancel landed on a fast machine
    if done[0]["ok"] == false {
        assert_eq!(done[0]["error"], "cancelled");
    }
}

/// Exports render to `<out>.partial-<jobId>.<ext>` and only replace the target on
/// success: a cancelled export leaves an existing file untouched, a finished one
/// replaces it, no partial file is left behind, and the result is registered with
/// the media server. An output path that is one of the project's sources is refused.
#[test]
fn exports_go_through_a_partial_file_and_never_overwrite_sources() {
    let Some(clip) = ffmpeg::tests::synth_clip() else {
        eprintln!("SKIP: ffmpeg not available");
        return;
    };
    let dir = test_dir().join("partial_export");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let target = dir.join("final cut.mp4");
    let partials = || -> Vec<String> {
        std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.contains(".partial-")).collect()
    };
    assert_eq!(partial_path(&target, "export_x"), dir.join("final cut.partial-export_x.mp4"));

    // cancelled mid-way: the previous export at the target survives
    std::fs::write(&target, b"previous export").unwrap();
    let mut big = tiny_project(&clip.to_string_lossy());
    big.width = 1280;
    big.height = 960;
    let jobs = Arc::new(JobManager::new());
    let sink = CollectingSink::new();
    let job = export_project(jobs.clone(), sink.clone(), big, target.to_string_lossy().into_owned(), "h264_mp4", None).unwrap();
    // wait until the encoder has created its partial file, then cancel
    let t0 = Instant::now();
    while partials().is_empty() && jobs.is_running(&job) && t0.elapsed() < Duration::from_secs(60) {
        std::thread::sleep(Duration::from_millis(10));
    }
    let saw_partial = !partials().is_empty();
    cancel_export(&jobs, &job).unwrap();
    assert!(jobs.wait(&job, Duration::from_secs(60)));
    let done = &sink.named("export://done")[0];
    if done["ok"] == false {
        assert_eq!(done["error"], "cancelled");
        assert_eq!(std::fs::read(&target).unwrap(), b"previous export", "a cancelled export must not touch the target");
    }
    assert!(saw_partial || done["ok"] == true, "the encoder writes a .partial- file next to the target");
    assert!(partials().is_empty(), "partial file removed: {:?}", partials());

    // a finished export replaces the target atomically and is registered for playback
    let registry = MediaRegistry::new();
    let jobs = Arc::new(JobManager::new());
    let sink = CollectingSink::new();
    let job = export_project_with(jobs.clone(), sink.clone(), tiny_project(&clip.to_string_lossy()), target.to_string_lossy().into_owned(), "h264_mp4", None, Some(registry.clone())).unwrap();
    assert!(jobs.wait(&job, Duration::from_secs(300)));
    let done = &sink.named("export://done")[0];
    assert_eq!(done["ok"], true, "{:?}", done["error"]);
    assert_eq!(ffmpeg::probe(&target).unwrap().width, 160);
    assert!(partials().is_empty());
    assert!(registry.contains(&target), "the finished export is registered with the media server");

    // refusing to overwrite a source (asset, LUT, stem), whatever the spelling
    let mut p = tiny_project(&clip.to_string_lossy());
    let lut = dir.join("look.cube");
    std::fs::write(&lut, Lut3D::identity(2).to_cube_text()).unwrap();
    let stem = dir.join("vocals.wav");
    audio::write_wav_f32(&stem, &[0.0; 16], 48_000, 2).unwrap();
    p.assets.push(Asset { id: "lut".into(), path: lut.to_string_lossy().into_owned(), name: "look.cube".into(), kind: AssetKind::Lut, ..Default::default() });
    p.assets[0].stems = Some(Stems { vocals: stem.to_string_lossy().into_owned(), background: String::new() });
    let before = std::fs::read(&clip).unwrap();
    let mut victims = vec![clip.clone(), lut.clone(), stem.clone()];
    if cfg!(windows) {
        victims.push(PathBuf::from(clip.to_string_lossy().to_uppercase().replace('\\', "/")));
    }
    for victim in victims {
        let jobs = Arc::new(JobManager::new());
        let err = export_project(jobs, CollectingSink::new(), p.clone(), victim.to_string_lossy().into_owned(), "h264_mp4", None).unwrap_err();
        assert!(err.contains("refusing to export over"), "{victim:?}: {err}");
    }
    assert_eq!(std::fs::read(&clip).unwrap(), before, "the source is untouched");
}

/// A phone clip with a 90° display matrix: probed portrait, decoded upright
/// (ffmpeg auto-rotates) and placed with the right aspect — also when the
/// project still carries the coded (landscape) size from an older probe.
#[test]
fn rotated_clip_exports_upright() {
    let Some(rot) = ffmpeg::tests::synth_rotated() else {
        eprintln!("SKIP: ffmpeg not available");
        return;
    };
    let bins = ffmpeg::find_binaries().unwrap();
    // what a player shows: the auto-rotated first frames
    let reference = ffmpeg::command(&bins.ffmpeg)
        .args(["-v", "error", "-i"])
        .arg(&rot)
        .args(["-vf", "select=eq(n\\,12)", "-fps_mode", "passthrough", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        .output()
        .unwrap()
        .stdout;
    assert_eq!(reference.len(), 240 * 320 * 3);
    for stale in [false, true] {
        let mut asset = asset_for(&rot, "rot");
        assert_eq!((asset.width, asset.height), (240, 320));
        if stale {
            (asset.width, asset.height) = (320, 240); // a project probed before the fix
        }
        let c = Clip { id: "r".into(), asset_id: "rot".into(), start_ms: 0.0, in_ms: 0.0, out_ms: 1000.0, ..Default::default() };
        let project = Project {
            fps: 24.0,
            width: 240,
            height: 320,
            assets: vec![asset],
            tracks: vec![Track { id: "v".into(), kind: TrackKind::Video, clips: vec![c], ..Default::default() }],
            ..Default::default()
        };
        let out = test_dir().join(format!("export_rotated_{stale}.mp4"));
        let (done, sink) = run_export(project, &out, "h264_mp4", None);
        assert_eq!(done["ok"], true, "{:?}", done["error"]);
        if stale {
            assert!(sink.named("export://log").iter().any(|l| l["message"].as_str().unwrap_or("").contains("rotation 90")), "the plan notes the corrected size");
        }
        let ours = frame_rgb(&out, 12);
        let d = mean_abs_diff(&ours, &reference);
        eprintln!("rotated export (stale size: {stale}): mean |diff| vs the auto-rotated source {d:.2}");
        assert!(d < 8.0, "the rotated clip must fill the portrait frame upright (diff {d:.1})");
    }
}

#[test]
fn plan_rejects_invalid_projects_and_missing_audio() {
    let dir = test_dir();
    let fake = dir.join("plan_fake.mp4");
    std::fs::write(&fake, b"not really a video").unwrap();
    let mut p = tiny_project(&fake.to_string_lossy());
    p.width = 100_000;
    assert!(build_timeline(&p, None).unwrap_err().contains("8192"));
    let mut p = tiny_project(&fake.to_string_lossy());
    p.fps = 0.0;
    assert!(build_timeline(&p, None).unwrap_err().contains("frame rate"));
    // an audio-only clip whose file is gone fails the plan instead of exporting silence
    let mut p = tiny_project(&fake.to_string_lossy());
    p.assets.push(Asset { id: "music".into(), path: dir.join("gone_music.mp3").to_string_lossy().into_owned(), kind: AssetKind::Audio, has_audio: true, duration_ms: 5000.0, ..Default::default() });
    p.tracks.push(Track {
        id: "a".into(),
        kind: TrackKind::Audio,
        clips: vec![Clip { id: "m".into(), asset_id: "music".into(), start_ms: 0.0, in_ms: 0.0, out_ms: 1000.0, ..Default::default() }],
        ..Default::default()
    });
    let err = build_timeline(&p, None).unwrap_err();
    assert!(err.contains("audio source of clip m is missing"), "{err}");
    // … unless it is outside the exported range
    p.tracks[1].clips[0].start_ms = 1400.0;
    p.tracks[1].clips[0].out_ms = 50.0;
    assert!(build_timeline(&p, Some(ExportRange { start_ms: 0.0, end_ms: 1000.0 })).is_ok());
    // an unreadable LUT is an error, not a silently dropped look
    let mut p = tiny_project(&fake.to_string_lossy());
    let bad_lut = dir.join("broken_look.cube");
    std::fs::write(&bad_lut, "LUT_3D_SIZE 3\n0 0 0\n").unwrap();
    p.assets.push(Asset { id: "lut".into(), path: bad_lut.to_string_lossy().into_owned(), kind: AssetKind::Lut, ..Default::default() });
    p.tracks[0].clips[0].color.lut_asset_id = Some("lut".into());
    let err = build_timeline(&p, None).unwrap_err();
    assert!(err.contains("invalid LUT") && err.contains("broken_look.cube"), "{err}");
}

#[test]
fn nvenc_retry_only_for_nvenc_failures() {
    assert!(is_nvenc_failure("encoder exited with Some(1): [h264_nvenc @ 0x1] OpenEncodeSessionEx failed: unsupported device (2): (no details)"));
    assert!(is_nvenc_failure("Cannot load nvcuda.dll"));
    assert!(is_nvenc_failure("[h264_nvenc @ 0x2] No capable devices found"));
    assert!(!is_nvenc_failure("encoder exited with Some(-28): Error writing trailer: No space left on device"));
    assert!(!is_nvenc_failure("decoder for clip c1 stopped early"));
    assert!(!is_nvenc_failure("source file of clip c1 is missing: C:/x.mp4"));
}

/// The acceptance test from the exporter brief: grade + keyframes + mask on
/// clip A, hero_time + reverse + freeze + gain + normalize + LUT on clip B,
/// a `screen` overlay at 50 % on a second video track.
#[test]
#[ignore = "slow: renders ~9 s of 1280x720 twice (run with --ignored)"]
fn export_v2_acceptance() {
    let Some(clip_a) = synth("v2_a_720p.mp4", "testsrc2=size=1280x720:rate=24:duration=3", Some("sine=frequency=440:duration=3")) else {
        eprintln!("SKIP: ffmpeg not available");
        return;
    };
    let clip_b = synth("v2_b_720p.mp4", "testsrc=size=1280x720:rate=24:duration=3", Some("sine=frequency=660:duration=3:sample_rate=48000")).unwrap();
    let grey = synth("v2_grey_720p.mp4", "color=c=0x808080:size=1280x720:rate=24:duration=3", None).unwrap();

    // 17^3 warm LUT
    let mut lut = Lut3D::identity(17);
    lut.title = Some("warm".into());
    for v in lut.data.iter_mut() {
        *v = [(v[0] * 1.1 + 0.05).min(1.0), v[1], v[2] * 0.85];
    }
    let lut_path = test_dir().join("v2_warm17.cube");
    std::fs::write(&lut_path, lut.to_cube_text()).unwrap();

    let mut a = Clip { id: "A".into(), asset_id: "ast_a".into(), track_id: "v1".into(), start_ms: 0.0, in_ms: 0.0, out_ms: 3000.0, ..Default::default() };
    a.color.contrast = 20.0;
    a.color.temperature = 30.0;
    a.color.lift = [0.05, 0.02, 0.0];
    a.color.gain = [0.1, 0.05, 0.0];
    a.transform.scale = Keyframed::with_keyframes(1.0, vec![Keyframe::new(0.0, 1.0).with_easing(Easing::EaseInOut), Keyframe::new(3000.0, 1.3)]);
    a.transform.position = Keyframed::with_keyframes([0.0, 0.0], vec![Keyframe::new(0.0, [0.0, 0.0]), Keyframe::new(3000.0, [0.1, -0.05])]);
    a.mask = Some(ClipMask { shape: MaskShape::Circle, feather: 0.05, rect: Keyframed::constant([0.2, 0.1, 0.6, 0.8]), inverted: false });

    let mut b = Clip {
        id: "B".into(),
        asset_id: "ast_b".into(),
        track_id: "v1".into(),
        start_ms: 3000.0,
        in_ms: 0.0,
        out_ms: 3000.0,
        // hero_time exactly as the UI stores it (SPEED_PRESETS in src/engine/speed.ts)
        speed: SpeedCurve {
            preset: SpeedPreset::HeroTime,
            points: vec![
                SpeedPoint::new(0.0, 1.0),
                SpeedPoint::new(0.3, 2.5),
                SpeedPoint::new(0.5, 0.25),
                SpeedPoint::new(0.7, 0.25),
                SpeedPoint::new(1.0, 1.5),
            ],
            optical_flow: true,
        },
        reversed: true,
        freeze_frame: Some(FreezeFrame { at_ms: 500.0, hold_ms: 500.0 }),
        ..Default::default()
    };
    b.audio.gain_db = 6.0;
    b.audio.normalize = true;
    b.color.lut_asset_id = Some("ast_lut".into());
    b.color.lut_intensity = 1.0;

    let overlay = Clip {
        id: "C".into(),
        asset_id: "ast_grey".into(),
        track_id: "v2".into(),
        start_ms: 500.0,
        in_ms: 0.0,
        out_ms: 1000.0,
        blend_mode: BlendMode::Screen,
        transform: ClipTransform { opacity: Keyframed::constant(0.5), ..Default::default() },
        ..Default::default()
    };

    let mut lut_asset = asset_for(&lut_path, "ast_lut");
    lut_asset.kind = AssetKind::Lut;
    let project = Project {
        id: "p".into(),
        name: "acceptance".into(),
        fps: 24.0,
        width: 1280,
        height: 720,
        assets: vec![asset_for(&clip_a, "ast_a"), asset_for(&clip_b, "ast_b"), asset_for(&grey, "ast_grey"), lut_asset],
        tracks: vec![
            Track { id: "v1".into(), kind: TrackKind::Video, name: "Video 1".into(), clips: vec![a, b.clone()], ..Default::default() },
            Track { id: "v2".into(), kind: TrackKind::Video, name: "Video 2".into(), clips: vec![overlay], ..Default::default() },
        ],
        ..Default::default()
    };

    let timeline_ms = project_duration_ms(&project);
    let expected_frames = (timeline_ms * 24.0 / 1000.0 - 1e-6).ceil() as i64;
    let b_out = ClipTimeMap::new(&b).total_ms();
    eprintln!("timeline {timeline_ms:.1} ms (clip B {b_out:.1} ms incl. 500 ms freeze) → {expected_frames} frames");

    let out = test_dir().join("export_v2_acceptance.mp4");
    let t0 = Instant::now();
    let (done, sink) = run_export(project.clone(), &out, "h264_mp4", None);
    let wall = t0.elapsed().as_secs_f64();
    for e in sink.named("export://log") {
        eprintln!("  log[{}] {}", e["level"], e["message"]);
    }
    assert_eq!(done["ok"], true, "export failed: {:?}", done["error"]);
    let fps = logged_video_fps(&sink).unwrap_or(0.0);
    eprintln!("RESULT acceptance: {expected_frames} frames 1280x720, wall {wall:.1}s, video stage {fps:.1} fps");

    // ---- container checks
    let probed = ffmpeg::probe(&out).unwrap();
    assert_eq!((probed.width, probed.height), (1280, 720));
    assert!(probed.has_audio, "must have an audio stream");
    let (frames, vdur) = video_stream_info(&out);
    assert_eq!(frames, expected_frames);
    assert!((vdur - timeline_ms).abs() <= 1000.0 / 24.0 + 1.0, "video {vdur:.1} ms vs timeline {timeline_ms:.1} ms");
    assert!((probed.duration_ms - timeline_ms).abs() <= 1000.0 / 24.0 + 30.0, "container {} ms", probed.duration_ms);

    // ---- pixels
    let (w, h) = (1280usize, 720usize);
    let f_a = frame_rgb(&out, 4); // clip A, before the overlay (starts at frame 12)
    let corner = region_mean(&f_a, w, 0, 0, 64, 64);
    let centre = region_mean(&f_a, w, 560, 280, 720, 440);
    eprintln!("clip A frame 4: corner {corner:?} centre {centre:?}");
    assert!(corner.iter().all(|c| *c < 6.0), "circle mask leaves the corner black: {corner:?}");
    assert!(centre.iter().sum::<f64>() > 90.0, "centre shows the clip: {centre:?}");

    let f_ov = frame_rgb(&out, 24); // t = 1.0 s, overlay active
    let corner_ov = region_mean(&f_ov, w, 0, 0, 64, 64);
    eprintln!("overlay frame 24: corner {corner_ov:?} (screen 50% of 0.5 grey on black ≈ 64)");
    assert!(corner_ov.iter().all(|c| (50.0..80.0).contains(c)), "screen overlay brightens the black corner: {corner_ov:?}");
    let f_after = frame_rgb(&out, 40); // t = 1.67 s, overlay over
    assert!(region_mean(&f_after, w, 0, 0, 64, 64).iter().all(|c| *c < 6.0));

    // freeze hold of clip B: timeline 3500..4000 ms = frames 84..95 show one source frame
    let (fz1, fz2) = (frame_rgb(&out, 85), frame_rgb(&out, 94));
    let d_freeze = mean_abs_diff(&fz1, &fz2);
    let d_moving = mean_abs_diff(&frame_rgb(&out, 100), &frame_rgb(&out, 110));
    eprintln!("freeze diff {d_freeze:.2}, moving diff {d_moving:.2}");
    assert!(d_freeze < 1.5 && d_moving > d_freeze * 3.0, "freeze {d_freeze} vs moving {d_moving}");

    // LUT warms clip B: compare against an ungraded render of the same section (range export)
    let mut plain = project.clone();
    plain.tracks[0].clips[1].color.lut_asset_id = None;
    let out_plain = test_dir().join("export_v2_acceptance_nolut.mp4");
    let range = ExportRange { start_ms: 4000.0, end_ms: 4800.0 };
    let (done2, _) = run_export(plain, &out_plain, "h264_mp4", Some(range));
    assert_eq!(done2["ok"], true, "{:?}", done2["error"]);
    let (frames2, _) = video_stream_info(&out_plain);
    assert_eq!(frames2, 20, "800 ms range at 24 fps");
    let graded = region_mean(&frame_rgb(&out, 108), w, 320, 180, 960, 540); // t = 4.5 s
    let ungraded = region_mean(&frame_rgb(&out_plain, 12), w, 320, 180, 960, 540); // 4.0 s + 12 frames
    eprintln!("clip B t=4.5s: LUT {graded:?} vs ungraded {ungraded:?}");
    assert!(graded[0] - graded[2] > (ungraded[0] - ungraded[2]) + 15.0, "LUT warms: R-B must grow");
    assert!(graded[0] > ungraded[0] && graded[2] < ungraded[2]);
    let _ = h;
}

/// 1080p24 throughput with the full grade (curves, HSL, LUT, wheels, mask,
/// vignette, grain, sharpening). Run in release:
/// `cargo test --release --lib export_v2_perf -- --ignored --nocapture`.
#[test]
#[ignore = "benchmark: renders 1080p (run with --ignored, preferably --release)"]
fn export_v2_perf_1080p() {
    let Some(src) = synth("v2_perf_1080p.mp4", "testsrc2=size=1920x1080:rate=24:duration=6", Some("sine=frequency=330:duration=6")) else {
        eprintln!("SKIP: ffmpeg not available");
        return;
    };
    let mut lut = Lut3D::identity(33);
    for v in lut.data.iter_mut() {
        *v = [v[0].powf(0.9), v[1], (v[2] * 0.9).min(1.0)];
    }
    let lut_path = test_dir().join("v2_perf33.cube");
    std::fs::write(&lut_path, lut.to_cube_text()).unwrap();
    let mut c = Clip { id: "P".into(), asset_id: "src".into(), start_ms: 0.0, in_ms: 0.0, out_ms: 6000.0, ..Default::default() };
    let g = &mut c.color;
    g.exposure = 5.0;
    g.contrast = 15.0;
    g.highlights = -10.0;
    g.shadows = 10.0;
    g.saturation = 10.0;
    g.vibrance = 20.0;
    g.sharpness = 20.0;
    g.temperature = 15.0;
    g.tint = 5.0;
    g.lift = [0.02, 0.0, -0.02];
    g.gamma = [0.05, 0.0, 0.0];
    g.gain = [0.05, 0.02, 0.0];
    g.hsl.orange.s = 15.0;
    g.hsl.blue.h = -10.0;
    g.curves.master = vec![[0.0, 0.0], [0.25, 0.2], [0.75, 0.8], [1.0, 1.0]];
    g.lut_asset_id = Some("lut".into());
    g.lut_intensity = 0.8;
    g.vignette = 0.3;
    g.grain = 0.1;
    c.transform.scale = Keyframed::with_keyframes(1.0, vec![Keyframe::new(0.0, 1.0), Keyframe::new(6000.0, 1.2)]);
    c.mask = Some(ClipMask { shape: MaskShape::Rectangle, feather: 0.05, rect: Keyframed::constant([0.05, 0.05, 0.9, 0.9]), inverted: false });
    let mut lut_asset = asset_for(&lut_path, "lut");
    lut_asset.kind = AssetKind::Lut;
    let project = Project {
        fps: 24.0,
        width: 1920,
        height: 1080,
        assets: vec![asset_for(&src, "src"), lut_asset],
        tracks: vec![Track { id: "v".into(), kind: TrackKind::Video, clips: vec![c], ..Default::default() }],
        ..Default::default()
    };
    for preset in ["h264_nvenc_mp4", "h264_mp4"] {
        let out = test_dir().join(format!("export_v2_perf_{preset}.mp4"));
        let t0 = Instant::now();
        let (done, sink) = run_export(project.clone(), &out, preset, None);
        assert_eq!(done["ok"], true, "{:?}", done["error"]);
        let wall = t0.elapsed().as_secs_f64();
        let fps = logged_video_fps(&sink).unwrap_or(0.0);
        for e in sink.named("export://log") { if e["message"].as_str().unwrap_or("").starts_with("timing") { eprintln!("  {}", e["message"]); } }
        let fell_back = sink.named("export://log").iter().any(|e| e["message"].as_str().unwrap_or("").contains("retrying with libx264"));
        eprintln!(
            "RESULT perf {preset}{}: 144 frames 1080p24 full grade — wall {wall:.1}s, video stage {fps:.1} fps ({:.2}x realtime)",
            if fell_back { " (fell back to libx264)" } else { "" },
            fps / 24.0
        );
        assert_eq!(video_stream_info(&out).0, 144);
    }
}

#[test]
#[ignore = "micro-benchmark of the per-frame CPU stages (release)"]
fn bench_stages_1080p() {
    use crate::render::sample::Frame;
    let (w, h) = (1920usize, 1080usize);
    let mut f = Frame::black(w, h);
    for (i, v) in f.data.iter_mut().enumerate() {
        *v = ((i * 7919) % 251) as u8;
    }
    let mut g = ColorGrade {
        exposure: 5.0,
        contrast: 15.0,
        highlights: -10.0,
        shadows: 10.0,
        saturation: 10.0,
        vibrance: 20.0,
        sharpness: 20.0,
        temperature: 15.0,
        lift: [0.02, 0.0, -0.02],
        gamma: [0.05, 0.0, 0.0],
        gain: [0.05, 0.02, 0.0],
        ..Default::default()
    };
    g.hsl.orange.s = 15.0;
    g.curves.master = vec![[0.0, 0.0], [0.25, 0.2], [0.75, 0.8], [1.0, 1.0]];
    g.lut_asset_id = Some("x".into());
    g.vignette = 0.3;
    g.grain = 0.1;
    let grade = GradeParams::new(&g, Some(Arc::new(Lut3D::identity(33))));
    let t = |name: &str, n: u32, f: &mut dyn FnMut()| {
        let t0 = Instant::now();
        for _ in 0..n {
            f();
        }
        eprintln!("BENCH {name}: {:.2} ms", t0.elapsed().as_secs_f64() * 1000.0 / n as f64);
    };
    let mut img = FloatImage::from_frames(&f, None);
    t("from_frames", 20, &mut || img = FloatImage::from_frames(&f, None));
    let pool = BufPool::<f32>::new();
    t("from_frames (pooled)", 20, &mut || {
        let i = FloatImage::from_frames_into(&f, None, pool.take(w * h * 3));
        pool.put(i.data);
    });
    let mut det = img.sharpen_detail(1.0, 1.0);
    t("sharpen_detail", 20, &mut || det = img.sharpen_detail(1.0, 1.0));
    t("gaussian_blur r=6", 10, &mut || {
        let _ = img.gaussian_blur(2.0, 2.0);
    });
    let src = SourceImage::Float(img.clone());
    let mut canvas = Canvas::new(w, h);
    let p = Placement { canvas_w: w as f64, canvas_h: h as f64, source_w: w as f64, source_h: h as f64, crop: None, scale: 1.1, position: [0.0, 0.0], rotation_deg: 0.0 };
    let mask = Some(MaskParams { shape: crate::model::MaskShape::Rectangle, rect: [0.05, 0.05, 0.9, 0.9], feather: 0.05, inverted: false });
    t("composite full grade", 20, &mut || {
        canvas.clear();
        composite_layer(&mut canvas, &Layer { source: &src, detail: Some(&det), uv: UvMatrix::new(&p), grade: &grade, baked: None, opacity: 1.0, mask, blend: BlendMode::Normal, time_ms: 0.0, fade: 0.0 });
    });
    let baked = crate::render::bake::BakedGrade::bake(&grade);
    t("composite full grade (baked)", 20, &mut || {
        canvas.clear();
        composite_layer(&mut canvas, &Layer { source: &src, detail: Some(&det), uv: UvMatrix::new(&p), grade: &grade, baked: baked.as_ref(), opacity: 1.0, mask, blend: BlendMode::Normal, time_ms: 0.0, fade: 0.0 });
    });
    let plain = GradeParams::new(&ColorGrade::default(), None);
    t("composite identity", 20, &mut || {
        canvas.clear();
        composite_layer(&mut canvas, &Layer { source: &src, detail: None, uv: UvMatrix::new(&p), grade: &plain, baked: None, opacity: 1.0, mask: None, blend: BlendMode::Normal, time_ms: 0.0, fade: 0.0 });
    });
    let bytes_src = SourceImage::Bytes(Arc::new(f.clone()));
    t("composite identity (8-bit source)", 20, &mut || {
        canvas.clear();
        composite_layer(&mut canvas, &Layer { source: &bytes_src, detail: None, uv: UvMatrix::new(&p), grade: &plain, baked: None, opacity: 1.0, mask: None, blend: BlendMode::Normal, time_ms: 0.0, fade: 0.0 });
    });
    t("to_rgb24", 20, &mut || {
        let _ = canvas.to_rgb24();
    });
}

#[test]
#[ignore = "micro-benchmark of individual grade stages (release)"]
fn bench_grade_stages() {
    let px: Vec<[f32; 3]> = (0..2_000_000u32).map(|i| [((i * 13) % 255) as f32 / 255.0, ((i * 7) % 255) as f32 / 255.0, ((i * 3) % 255) as f32 / 255.0]).collect();
    let run = |name: &str, g: ColorGrade, lut: Option<Arc<Lut3D>>| {
        let p = GradeParams::new(&g, lut);
        let t0 = Instant::now();
        let mut acc = 0.0f32;
        for (i, c) in px.iter().enumerate() {
            let o = p.apply(*c, [0.01; 3], [(i % 1920) as f32 / 1920.0, 0.5], 100.0);
            acc += o[0];
        }
        eprintln!("STAGE {name}: {:.1} ns/px ({acc:.0})", t0.elapsed().as_secs_f64() * 1e9 / px.len() as f64);
    };
    run("identity", ColorGrade::default(), None);
    run("contrast", ColorGrade { contrast: 15.0, exposure: 5.0, ..Default::default() }, None);
    run("hs", ColorGrade { highlights: 10.0, shadows: 10.0, ..Default::default() }, None);
    run("wheels", ColorGrade { lift: [0.02, 0.0, 0.0], gain: [0.05, 0.0, 0.0], ..Default::default() }, None);
    run("gamma1", ColorGrade { gamma: [0.05, 0.0, 0.0], ..Default::default() }, None);
    run("sat", ColorGrade { saturation: 10.0, vibrance: 20.0, ..Default::default() }, None);
    let mut h = ColorGrade::default();
    h.hsl.orange.s = 15.0;
    run("hsl", h, None);
    let mut c = ColorGrade::default();
    c.curves.master = vec![[0.0, 0.0], [0.25, 0.2], [0.75, 0.8], [1.0, 1.0]];
    run("curves", c, None);
    run("lut33", ColorGrade { lut_asset_id: Some("x".into()), ..Default::default() }, Some(Arc::new(Lut3D::identity(33))));
    run("vignette", ColorGrade { vignette: 0.3, ..Default::default() }, None);
    run("grain", ColorGrade { grain: 0.1, ..Default::default() }, None);
    run("sharp", ColorGrade { sharpness: 20.0, ..Default::default() }, None);
}

/// Variable-frame-rate source (24 fps content on a 1/60 time base:
/// `r_frame_rate=60`, irregular pts) — every exported frame must show the
/// source frame for its timestamp, through plain playback, a seek (range
/// export) and a 2× speed clip.
#[test]
fn vfr_source_maps_by_timestamp() {
    let Some(bins) = ffmpeg::find_binaries().ok() else {
        eprintln!("SKIP: ffmpeg not available");
        return;
    };
    let src = test_dir().join("vfr_24_on_60.mp4");
    if !src.is_file() {
        let ok = ffmpeg::command(&bins.ffmpeg)
            .args(["-v", "error", "-y", "-f", "lavfi", "-i", "nullsrc=s=128x72:r=24:d=3,geq=lum='16+3*N':cb=128:cr=128"])
            .args(["-c:v", "libx264", "-preset", "veryfast", "-qp", "0", "-pix_fmt", "yuv420p", "-enc_time_base", "1/60", "-video_track_timescale", "60"])
            .arg(&src)
            .status()
            .unwrap()
            .success();
        assert!(ok);
    }
    let asset = asset_for(&src, "vfr");
    // the probe reports the real cadence (avg_frame_rate), not the 1/60 time base
    assert!((asset.fps - 24.0).abs() < 0.1, "probe should report the average rate: {}", asset.fps);
    assert!((ffmpeg::probe_video_info(&src).unwrap().avg_fps.unwrap() - 24.0).abs() < 0.1);
    let clip = |id: &str, start: f64, in_ms: f64, out_ms: f64, speed: f64| Clip {
        id: id.into(),
        asset_id: "vfr".into(),
        start_ms: start,
        in_ms,
        out_ms,
        speed: SpeedCurve::constant(speed),
        ..Default::default()
    };
    let project = Project {
        fps: 24.0,
        width: 128,
        height: 72,
        assets: vec![asset],
        tracks: vec![Track { id: "v".into(), kind: TrackKind::Video, clips: vec![clip("a", 0.0, 0.0, 2000.0, 1.0), clip("b", 2000.0, 500.0, 2500.0, 2.0)], ..Default::default() }],
        ..Default::default()
    };
    // source frame N has R = 3N·255/219
    let expect = |n: f64| 3.0 * n * 255.0 / 219.0;
    let grey = |path: &Path, frame: i64| region_mean(&frame_rgb(path, frame), 128, 32, 16, 96, 56)[0];

    let out = test_dir().join("export_v2_vfr.mp4");
    let (done, _) = run_export(project.clone(), &out, "h264_mp4", None);
    assert_eq!(done["ok"], true, "{:?}", done["error"]);
    assert_eq!(video_stream_info(&out).0, 72, "2 s + 1 s at 2x");
    for n in [0i64, 1, 7, 13, 30, 47] {
        let g = grey(&out, n);
        assert!((g - expect(n as f64)).abs() < 3.0, "frame {n}: {g:.1}, expected source frame {n} ({:.1})", expect(n as f64));
    }
    for n in [48i64, 55, 71] {
        let src_n = 12.0 + 2.0 * (n - 48) as f64; // clip b: in 500 ms, 2x
        let g = grey(&out, n);
        assert!((g - expect(src_n)).abs() < 3.0, "frame {n}: {g:.1}, expected source frame {src_n}");
    }
    // seek path: a range starting at 1.0 s begins on source frame 24
    let out_r = test_dir().join("export_v2_vfr_range.mp4");
    let (done, _) = run_export(project, &out_r, "h264_mp4", Some(ExportRange { start_ms: 1000.0, end_ms: 1500.0 }));
    assert_eq!(done["ok"], true, "{:?}", done["error"]);
    for n in [0i64, 5, 11] {
        let g = grey(&out_r, n);
        assert!((g - expect(24.0 + n as f64)).abs() < 3.0, "range frame {n}: {g:.1}");
    }
}

/// The user's real footage (`<repo>/<clips folder>/clip1.mp4`, `clip2.mp4`:
/// 1280x720 VFR phones clips): a 10 s range across the clip1 → clip2 cut,
/// first ungraded (pixel check against a direct decode), then with a
/// cinematic grade + LUT for throughput.
#[test]
#[ignore = "needs the user's clips folder (run with --ignored, preferably --release)"]
fn export_real_clips_range() {
    let Some(repo) = crate::clips::repo_dir() else {
        eprintln!("SKIP: repo not found");
        return;
    };
    let dir = crate::clips::discover_clips_dir(&repo);
    let (p1, p2) = (dir.join("clip1.mp4"), dir.join("clip2.mp4"));
    if !p1.is_file() || !p2.is_file() {
        eprintln!("SKIP: no clip1/clip2 in {}", dir.display());
        return;
    }
    let (a1, a2) = (asset_for(&p1, "c1"), asset_for(&p2, "c2"));
    eprintln!("clips folder {} — clip1 {}x{} {:.0} ms nominal {} fps", dir.display(), a1.width, a1.height, a1.duration_ms, a1.fps);
    let d1 = a1.duration_ms.min(15000.0);
    let mk = |graded: bool| {
        let mut c1 = Clip { id: "k1".into(), asset_id: "c1".into(), start_ms: 0.0, in_ms: 0.0, out_ms: d1, ..Default::default() };
        let mut c2 = Clip { id: "k2".into(), asset_id: "c2".into(), start_ms: d1, in_ms: 0.0, out_ms: a2.duration_ms.min(15000.0), ..Default::default() };
        if graded {
            for c in [&mut c1, &mut c2] {
                c.color.contrast = 20.0;
                c.color.temperature = 25.0;
                c.color.saturation = -10.0;
                c.color.lift = [0.03, 0.01, 0.0];
                c.color.gain = [0.08, 0.04, 0.0];
                c.color.vignette = 0.3;
                c.color.lut_asset_id = Some("lut".into());
            }
        }
        let mut lut = Lut3D::identity(33);
        for v in lut.data.iter_mut() {
            *v = [(v[0] * 1.05 + 0.02).min(1.0), v[1], v[2] * 0.92];
        }
        let lut_path = test_dir().join("real_warm33.cube");
        std::fs::write(&lut_path, lut.to_cube_text()).unwrap();
        let mut la = asset_for(&lut_path, "lut");
        la.kind = AssetKind::Lut;
        Project {
            fps: 24.0,
            width: 1280,
            height: 720,
            assets: vec![a1.clone(), a2.clone(), la],
            tracks: vec![Track { id: "v".into(), kind: TrackKind::Video, clips: vec![c1, c2], ..Default::default() }],
            ..Default::default()
        }
    };
    let range = ExportRange { start_ms: d1 - 5000.0, end_ms: d1 + 5000.0 };

    // ---- ungraded: frames must match a direct decode of the source moment
    let out = test_dir().join("export_v2_real_plain.mp4");
    let (done, sink) = run_export(mk(false), &out, "h264_mp4", Some(range));
    assert_eq!(done["ok"], true, "{:?}", done["error"]);
    let (frames, dur) = video_stream_info(&out);
    eprintln!("RESULT real plain: {frames} frames, {dur:.0} ms, video stage {:.1} fps", logged_video_fps(&sink).unwrap_or(0.0));
    assert_eq!(frames, 240);
    let bins = ffmpeg::find_binaries().unwrap();
    let direct = |p: &Path, ms: f64| -> Vec<u8> {
        ffmpeg::command(&bins.ffmpeg)
            .args(["-v", "error", "-ss", &format!("{:.4}", ms / 1000.0 - 0.001), "-i"])
            .arg(p)
            .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            .output()
            .unwrap()
            .stdout
    };
    for (n, path, src_ms) in [(0i64, &p1, d1 - 5000.0), (60, &p1, d1 - 2500.0), (120, &p2, 0.0), (200, &p2, 80.0 / 24.0 * 1000.0)] {
        let ours = frame_rgb(&out, n);
        let theirs = direct(path, src_ms);
        let d = mean_abs_diff(&ours, &theirs);
        eprintln!("  frame {n}: mean |diff| vs direct decode of {} @ {src_ms:.0} ms = {d:.2}", path.file_name().unwrap().to_string_lossy());
        assert!(d < 12.0, "frame {n} does not show the expected source moment (diff {d:.1})");
    }

    // ---- graded throughput
    for preset in ["h264_nvenc_mp4", "h264_mp4"] {
        let out = test_dir().join(format!("export_v2_real_graded_{preset}.mp4"));
        let (done, sink) = run_export(mk(true), &out, preset, Some(range));
        assert_eq!(done["ok"], true, "{:?}", done["error"]);
        for e in sink.named("export://log") {
            let m = e["message"].as_str().unwrap_or("");
            if m.starts_with("timing") || m.contains("fell back") || m.contains("retrying") {
                eprintln!("  {m}");
            }
        }
        eprintln!("RESULT real graded {preset}: 240 frames 1280x720, video stage {:.1} fps", logged_video_fps(&sink).unwrap_or(0.0));
    }
}

/* ------------------------------------------------------ voice separation */

/// Amplitude of the `freq` component of an interleaved stereo buffer (mono average, single-bin DFT).
fn tone_amplitude(samples: &[f32], rate: f64, freq: f64) -> f64 {
    let mono: Vec<f64> = samples.chunks_exact(2).map(|c| (c[0] as f64 + c[1] as f64) * 0.5).collect();
    let n = mono.len().max(1) as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for (i, x) in mono.iter().enumerate() {
        let ph = 2.0 * std::f64::consts::PI * freq * i as f64 / rate;
        re += x * ph.cos();
        im -= x * ph.sin();
    }
    2.0 * (re * re + im * im).sqrt() / n
}

/// A 3 s clip whose audio is 440 Hz + 1000 Hz (0.3 each), plus synthetic stems: vocals = the
/// 440 Hz tone, background = the 1000 Hz tone (48 kHz WAVs like `separate` writes).
fn voice_fixture() -> Option<(PathBuf, Stems)> {
    static FIXTURE: std::sync::OnceLock<Option<(PathBuf, Stems)>> = std::sync::OnceLock::new();
    FIXTURE.get_or_init(make_voice_fixture).clone()
}

fn make_voice_fixture() -> Option<(PathBuf, Stems)> {
    let src = synth(
        "voice_mix.mp4",
        "color=c=gray:size=64x48:rate=24:duration=3",
        Some("aevalsrc=0.3*sin(2*PI*440*t)+0.3*sin(2*PI*1000*t):s=48000:d=3"),
    )?;
    let dir = test_dir().join("voice_stems");
    std::fs::create_dir_all(&dir).unwrap();
    let tone = |f: f64| -> Vec<f32> {
        (0..3 * 48_000).flat_map(|i| {
            let v = (0.3 * (2.0 * std::f64::consts::PI * f * i as f64 / 48_000.0).sin()) as f32;
            [v, v]
        }).collect()
    };
    let (v, b) = (dir.join("vocals.wav"), dir.join("background.wav"));
    audio::write_wav_f32(&v, &tone(440.0), 48_000, 2).unwrap();
    audio::write_wav_f32(&b, &tone(1000.0), 48_000, 2).unwrap();
    Some((src, Stems { vocals: v.to_string_lossy().into_owned(), background: b.to_string_lossy().into_owned() }))
}

fn voice_project(src: &Path, stems: Option<Stems>, video_mode: VoiceMode, mirror_mode: Option<VoiceMode>) -> Project {
    let mut asset = asset_for(src, "a");
    asset.stems = stems;
    let mut vc = Clip { id: "v1".into(), asset_id: "a".into(), start_ms: 0.0, in_ms: 500.0, out_ms: 2500.0, ..Default::default() };
    vc.audio.voice = video_mode;
    vc.audio.normalize = false;
    let mut tracks = vec![Track { id: "v".into(), kind: TrackKind::Video, clips: vec![vc.clone()], ..Default::default() }];
    if let Some(m) = mirror_mode {
        let mut ac = Clip { id: "a1".into(), ..vc };
        ac.audio.voice = m;
        tracks.push(Track { id: "au".into(), kind: TrackKind::Audio, clips: vec![ac], ..Default::default() });
    }
    Project { fps: 24.0, width: 64, height: 48, assets: vec![asset], tracks, ..Default::default() }
}

/// `(440 Hz amplitude, 1000 Hz amplitude, 880 Hz amplitude, warnings)` of a 1 s mix from 0.5 s.
fn mix_tones(p: &Project) -> (f64, f64, f64, Vec<String>) {
    let bins = ffmpeg::find_binaries().unwrap();
    let ctl = TaskCtl::default();
    let mut warns = Vec::new();
    let mut log = |level: &str, m: String| {
        if level == "warn" {
            warns.push(m);
        }
    };
    let mix = audio::render_mix(p, 500.0, 48_000, &bins.ffmpeg, &ctl, &mut log).unwrap();
    let s = &mix.samples;
    (tone_amplitude(s, 48_000.0, 440.0), tone_amplitude(s, 48_000.0, 1000.0), tone_amplitude(s, 48_000.0, 880.0), warns)
}

#[test]
fn stems_replace_the_source_audio_in_the_mix() {
    let Some((src, stems)) = voice_fixture() else {
        eprintln!("SKIP: ffmpeg not available");
        return;
    };
    // original: both tones (mono AAC source; ffmpeg's mono -> stereo upmix is -3 dB: 0.3 -> 0.21)
    let (a440, a1k, _, w) = mix_tones(&voice_project(&src, Some(stems.clone()), VoiceMode::Original, None));
    assert!(a440 > 0.18 && a1k > 0.18 && w.is_empty(), "original {a440} {a1k} {w:?}");
    // isolate voice: only the vocals tone
    let (a440, a1k, _, w) = mix_tones(&voice_project(&src, Some(stems.clone()), VoiceMode::Voice, None));
    assert!(a440 > 0.28 && a1k < 0.003 && w.is_empty(), "voice {a440} {a1k} {w:?}");
    // remove vocals: only the background tone
    let (a440, a1k, _, _) = mix_tones(&voice_project(&src, Some(stems.clone()), VoiceMode::Background, None));
    assert!(a1k > 0.28 && a440 < 0.003, "background {a440} {a1k}");
    // no stems: warning + original audio
    let (a440, a1k, _, w) = mix_tones(&voice_project(&src, None, VoiceMode::Voice, None));
    assert!(a440 > 0.18 && a1k > 0.18, "fallback {a440} {a1k}");
    assert!(w.len() == 1 && w[0].contains("not been separated"), "{w:?}");
    // the mirrored audio-track clip follows its own mode (the video clip is not heard)
    let (a440, a1k, _, _) = mix_tones(&voice_project(&src, Some(stems.clone()), VoiceMode::Background, Some(VoiceMode::Voice)));
    assert!(a440 > 0.28 && a1k < 0.003, "mirror {a440} {a1k}");
    // stems go through the same speed map and gain: 2x keeps the vocals tone at 440 Hz (pitch
    // kept by default), -6 dB; with keepPitch off it plays at 880 Hz (varispeed)
    let mut p = voice_project(&src, Some(stems.clone()), VoiceMode::Voice, None);
    p.tracks[0].clips[0].speed = SpeedCurve::constant(2.0);
    p.tracks[0].clips[0].audio.gain_db = -6.0206;
    p.tracks[0].clips[0].in_ms = 0.0;
    p.tracks[0].clips[0].out_ms = 3000.0;
    let (a440, a1k, a880, _) = mix_tones(&p);
    assert!((a440 - 0.15).abs() < 0.01 && a880 < 0.003 && a1k < 0.003, "2x voice, pitch kept {a440} {a1k} {a880}");
    p.tracks[0].clips[0].audio.keep_pitch = Some(false);
    let (a440, a1k, a880, _) = mix_tones(&p);
    assert!((a880 - 0.15).abs() < 0.01 && a440 < 0.003 && a1k < 0.003, "2x voice, varispeed {a440} {a1k} {a880}");
    // reversed: still only the vocals tone
    let mut p = voice_project(&src, Some(stems), VoiceMode::Voice, None);
    p.tracks[0].clips[0].reversed = true;
    let (a440, a1k, _, _) = mix_tones(&p);
    assert!(a440 > 0.28 && a1k < 0.003, "reversed {a440} {a1k}");
}

#[test]
fn export_isolate_voice_and_remove_vocals() {
    let Some((src, stems)) = voice_fixture() else {
        eprintln!("SKIP: ffmpeg not available");
        return;
    };
    let bins = ffmpeg::find_binaries().unwrap();
    for (mode, want, reject) in [(VoiceMode::Voice, 440.0, 1000.0), (VoiceMode::Background, 1000.0, 440.0)] {
        let out = test_dir().join(format!("export_voice_{mode:?}.mp4"));
        let (done, sink) = run_export(voice_project(&src, Some(stems.clone()), mode, None), &out, "h264_mp4", None);
        assert_eq!(done["ok"], true, "{:?}", done["error"]);
        assert!(sink.named("export://log").iter().any(|l| l["message"].as_str().unwrap_or("").contains("stem of")));
        let pcm = ffmpeg::command(&bins.ffmpeg)
            .args(["-v", "error", "-ss", "0.25", "-t", "1.5", "-i"])
            .arg(&out)
            .args(["-vn", "-ac", "2", "-ar", "48000", "-f", "f32le", "pipe:1"])
            .output()
            .unwrap()
            .stdout;
        let s: Vec<f32> = pcm.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let (w, r) = (tone_amplitude(&s, 48_000.0, want), tone_amplitude(&s, 48_000.0, reject));
        eprintln!("export {mode:?}: {want} Hz amplitude {w:.4}, {reject} Hz amplitude {r:.5}");
        assert!(w > 0.2 && r < 0.01, "{mode:?}: wanted {w}, rejected {r}");
    }
    // missing stems: the export still succeeds with the original audio and warns on export://log
    let out = test_dir().join("export_voice_missing.mp4");
    let (done, sink) = run_export(voice_project(&src, None, VoiceMode::Voice, None), &out, "h264_mp4", None);
    assert_eq!(done["ok"], true);
    assert!(sink.named("export://log").iter().any(|l| l["level"] == "warn" && l["message"].as_str().unwrap_or("").contains("original audio")));
}

/* ------------------------------------------------------------ feature set v2 */

fn solid_clip_project(path: &str) -> Project {
    let mut p = tiny_project(path);
    // c2 is not reversed / not sped up here: plain handles
    p.tracks[0].clips[1].speed = SpeedCurve::default();
    p.tracks[0].clips[1].speed.points.clear();
    p.tracks[0].clips[1].reversed = false;
    p.tracks[0].clips[1].in_ms = 1000.0;
    p.tracks[0].clips[1].out_ms = 2000.0;
    p
}

#[test]
fn plan_extends_clips_around_transitions_and_freezes_for_snaps() {
    let dir = test_dir();
    let fake = dir.join("plan_fake_v2.mp4");
    std::fs::write(&fake, b"not really a video").unwrap();
    let mut p = solid_clip_project(&fake.to_string_lossy());
    p.tracks[0].clips[0].in_ms = 500.0;
    p.tracks[0].clips[0].out_ms = 1500.0;
    p.tracks[0].clips[1].transition_in = Some(TransitionIn { kind: TransitionType::Dissolve, duration_ms: 500.0 });
    let plan = build_timeline(&p, None).unwrap();
    assert_eq!(plan.frame_count, 48);
    assert_eq!(plan.transitions.len(), 1);
    let t = &plan.transitions[0];
    assert_eq!((t.a, t.b, t.cut_ms, t.dur_ms), (0, 1, 1000.0, 500.0));
    assert_eq!((t.first_frame, t.end_frame), (18, 30), "750..1250 ms");
    // A plays on past its out point (handles), B starts early
    assert_eq!((plan.layers[0].first_frame, plan.layers[0].end_frame), (0, 30));
    assert_eq!((plan.layers[1].first_frame, plan.layers[1].end_frame), (18, 48));
    let a_last = plan.layers[0].resolved.last().unwrap();
    assert!((a_last.source_ms - (1500.0 + 1208.333 - 1000.0)).abs() < 0.01, "A's handle: {}", a_last.source_ms);
    assert!((plan.layers[1].resolved[0].source_ms - 750.0).abs() < 1e-6, "B's pre-roll: 250 ms before its in point");
    // no handle before the file start: the first frame holds
    p.tracks[0].clips[1].in_ms = 100.0;
    p.tracks[0].clips[1].out_ms = 1100.0;
    let plan = build_timeline(&p, None).unwrap();
    assert_eq!(plan.layers[1].resolved[0].source_ms, 0.0);

    // durations are clamped (3000 max, and to the shorter clip); unknown types warn + dissolve
    p.tracks[0].clips[1].transition_in = Some(TransitionIn { kind: TransitionType::Other("starWipe".into()), duration_ms: 9000.0 });
    p.tracks[0].clips[1].out_ms = 400.0; // 300 ms clip
    let plan = build_timeline(&p, None).unwrap();
    assert_eq!(plan.transitions[0].dur_ms, 300.0);
    assert!(plan.warnings.iter().any(|w| w.contains("starWipe")));
    // a transition without an adjacent clip is ignored with a warning
    p.tracks[0].clips[1].start_ms = 1500.0;
    let plan = build_timeline(&p, None).unwrap();
    assert!(plan.transitions.is_empty());
    assert!(plan.warnings.iter().any(|w| w.contains("transition ignored")));

    // cameraSnap on an FX track holds the video on its first frame; FX clips ignore speed
    let mut p = solid_clip_project(&fake.to_string_lossy());
    let mut snap = Clip { id: "snap".into(), asset_id: String::new(), start_ms: 250.0, in_ms: 0.0, out_ms: 500.0, speed: SpeedCurve::constant(4.0), ..Default::default() };
    snap.effect = Some(ClipEffect::new(EffectType::CameraSnap));
    let mut fade = Clip { id: "fade".into(), start_ms: 1500.0, in_ms: 0.0, out_ms: 1000.0, ..Default::default() };
    fade.effect = Some(ClipEffect::new(EffectType::FadeToBlack));
    p.tracks.push(Track { id: "fx".into(), kind: TrackKind::Fx, clips: vec![snap, fade], ..Default::default() });
    assert_eq!(project_duration_ms(&p), 2500.0, "the fade clip lasts outMs − inMs");
    let plan = build_timeline(&p, None).unwrap();
    assert_eq!(plan.frame_count, 60);
    assert_eq!(plan.effects.len(), 2);
    assert_eq!((plan.effects[0].first_frame, plan.effects[0].end_frame), (6, 18));
    for n in 6..18 {
        assert_eq!(plan.video_time[n], 250.0, "frame {n} frozen");
        assert_eq!(plan.layers[0].requests[n].a, plan.layers[0].requests[6].a);
    }
    assert!((plan.video_time[18] - 750.0).abs() < 1e-9, "then it jumps to the real time");
    // a muted FX track does nothing
    p.tracks[1].muted = true;
    let plan = build_timeline(&p, None).unwrap();
    assert!(plan.effects.is_empty() && plan.video_time[10] > 400.0);
}

#[test]
fn frame_supply_follows_the_interpolation_mode() {
    use crate::model::FrameInterpolation as F;
    // 24 fps footage into a 60 fps export
    let s = frame_supply(&F::OpticalFlow, true, 24.0, 1.0, 60.0, 1);
    assert_eq!((s.grid_fps, s.flow_target), (24.0, Some(60.0)));
    assert!((s.blend_below_fps - 59.94).abs() < 1e-9, "blending is the fallback");
    let s = frame_supply(&F::FrameBlend, false, 24.0, 1.5, 60.0, 1);
    assert_eq!((s.grid_fps, s.flow_target), (24.0, None));
    assert!(s.blend_below_fps > 0.0);
    let s = frame_supply(&F::None, false, 24.0, 1.0, 60.0, 1);
    assert_eq!((s.grid_fps, s.blend_below_fps, s.flow_target), (60.0, 0.0, None), "nearest frame, repeated");
    // the clip's optical-flow toggle (on by default) opts a clip out of optical flow: frame blending
    let s = frame_supply(&F::OpticalFlow, false, 24.0, 1.0, 60.0, 1);
    assert_eq!((s.grid_fps, s.flow_target), (24.0, None));
    assert!(s.blend_below_fps > 0.0);
    assert_eq!(frame_supply(&F::None, true, 24.0, 1.0, 60.0, 1).flow_target, None, "'none' never interpolates");
    // 1x at the source rate: the pre-v2 path
    let s = frame_supply(&F::OpticalFlow, true, 24.0, 1.0, 24.0, 1);
    assert_eq!((s.grid_fps, s.flow_target, s.needs_frames), (24.0, None, false));
    // slow motion from 60 fps footage at 0.5x into 24: enough real frames, refined grid
    let s = frame_supply(&F::OpticalFlow, true, 60.0, 0.5, 24.0, 2);
    assert_eq!((s.grid_fps, s.flow_target), (48.0, None));
    // 24 fps footage at 0.25x: optical flow at 96 fps
    assert_eq!(frame_supply(&F::OpticalFlow, true, 24.0, 0.25, 24.0, 1).flow_target, Some(96.0));
}

/// A solid-colour 24 fps clip (cached).
fn colour_clip(name: &str, colour: &str) -> Option<PathBuf> {
    synth(name, &format!("color=c={colour}:s=160x90:r=24:d=3"), None)
}

#[test]
fn transitions_fades_and_effects_export() {
    // unsaturated colours: the readback (swscale, untagged BT.601) must not clip, so averages stay averages
    let (Some(red), Some(blue), Some(moving)) = (
        colour_clip("v2_warm.mp4", "0xC04030"),
        colour_clip("v2_cool.mp4", "0x3050C0"),
        synth("v2_testsrc.mp4", "testsrc=s=160x90:r=24:d=3", Some("sine=frequency=440:sample_rate=48000:duration=3")),
    ) else {
        eprintln!("SKIP: ffmpeg not available");
        return;
    };
    let assets = vec![asset_for(&red, "r"), asset_for(&blue, "b"), asset_for(&moving, "m")];
    let clip = |id: &str, asset: &str, start: f64, in_ms: f64, out_ms: f64| Clip { id: id.into(), asset_id: asset.into(), start_ms: start, in_ms, out_ms, ..Default::default() };
    let mut c1 = clip("c1", "r", 0.0, 0.0, 1000.0);
    c1.fade_in_ms = Some(500.0);
    let mut c2 = clip("c2", "b", 1000.0, 500.0, 1500.0);
    c2.transition_in = Some(TransitionIn { kind: TransitionType::Dissolve, duration_ms: 500.0 });
    let mut c3 = clip("c3", "m", 2000.0, 0.0, 2000.0);
    c3.transition_in = Some(TransitionIn { kind: TransitionType::DipToBlack, duration_ms: 400.0 });
    let mut snap = clip("snap", "", 2800.0, 0.0, 800.0);
    snap.effect = Some(ClipEffect::new(EffectType::CameraSnap));
    let mut fade = clip("fade", "", 3500.0, 0.0, 460.0);
    fade.effect = Some(ClipEffect::new(EffectType::FadeToBlack));
    let project = Project {
        fps: 24.0,
        width: 160,
        height: 90,
        assets,
        tracks: vec![
            Track { id: "v".into(), kind: TrackKind::Video, clips: vec![c1, c2, c3], ..Default::default() },
            Track { id: "fx".into(), kind: TrackKind::Fx, clips: vec![snap, fade], ..Default::default() },
        ],
        frame_interpolation: Some(crate::model::FrameInterpolation::FrameBlend),
        ..Default::default()
    };
    let out = test_dir().join("export_v2_transitions.mp4");
    let (done, sink) = run_export(project.clone(), &out, "h264_mp4", None);
    assert_eq!(done["ok"], true, "{:?} / {:?}", done["error"], sink.named("export://log"));
    let (frames, _) = video_stream_info(&out);
    assert_eq!(frames, 96, "4 s at 24 fps");
    let mean = |n: i64| region_mean(&frame_rgb(&out, n), 160, 40, 20, 120, 70);
    let (red_c, blue_c) = (mean(14), mean(36)); // outside the fade-in and the 750..1250 ms dissolve
    assert!(red_c[0] > 150.0 && red_c[2] < 90.0 && blue_c[2] > 150.0 && blue_c[0] < 90.0, "{red_c:?} {blue_c:?}");
    // dissolve midpoint (frame 24 = the cut, p = 0.5) = the average of the two clips
    let mid = mean(24);
    for c in 0..3 {
        assert!((mid[c] - (red_c[c] + blue_c[c]) / 2.0).abs() < 3.0, "dissolve midpoint {mid:?} vs {red_c:?} / {blue_c:?}");
    }
    // dipToBlack midpoint (frame 48 = 2000 ms) is black
    let dip = mean(48);
    assert!(dip.iter().all(|v| *v < 6.0), "dip to black {dip:?}");
    // clip fade-in: black at the start, half the picture at 250 ms (a linear ramp of the values)
    assert!(mean(0).iter().all(|v| *v < 8.0), "{:?}", mean(0));
    let half = mean(6)[0];
    assert!((half - red_c[0] * 0.5).abs() < 4.0, "fade-in at 250 ms: {half} vs {}", red_c[0] * 0.5);
    // cameraSnap: flash at the start, then a frozen polaroid with a white border
    assert!(mean(68).iter().all(|v| *v > 200.0), "flash {:?}", mean(68));
    let (f1, f2) = (frame_rgb(&out, 75), frame_rgb(&out, 82));
    assert!(mean_abs_diff(&f1, &f2) < 1.5, "the snapshot holds: {}", mean_abs_diff(&f1, &f2));
    let border = region_mean(&f1, 160, 4, 30, 6, 60);
    assert!(border.iter().all(|v| *v > 225.0), "white border {border:?}");
    // the picture moves again after the snap (frame 87 vs 89)
    assert!(mean_abs_diff(&frame_rgb(&out, 87), &frame_rgb(&out, 89)) > 1.0);
    // fadeToBlack: the last frame is (almost) black
    let last = mean(95);
    assert!(last.iter().all(|v| *v < 16.0), "fade to black end {last:?}");
    // the shutter is in the mix: 2800..2920 ms loud, silence before (the tone clip is muted here)
    let bins = ffmpeg::find_binaries().unwrap();
    let mut quiet = project.clone();
    quiet.tracks[0].clips[2].audio.muted = true;
    let mix = audio::render_mix(&quiet, 2700.0, 48_000 / 2, &bins.ffmpeg, &TaskCtl::default(), &mut |_, _| {}).unwrap();
    let rms = |a: usize, b: usize| (mix.samples[a * 2..b * 2].iter().map(|v| v * v).sum::<f32>() / ((b - a) * 2) as f32).sqrt();
    assert!(rms(0, 4800) < 1e-6, "silent before the snap");
    assert!(rms(4800, 4800 + 2400) > 0.02, "shutter at 2800 ms: {}", rms(4800, 7200));
    let peak = mix.samples[4800 * 2..7200 * 2].iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!((peak - 0.9 * 10f32.powf(-6.0 / 20.0)).abs() < 0.01, "0.9-peak shutter at -6 dB: {peak}");
    let logs: Vec<String> = sink.named("export://log").iter().filter_map(|e| e["message"].as_str().map(String::from)).collect();
    assert!(logs.iter().any(|m| m.contains("transition dissolve")), "{logs:?}");
    assert!(logs.iter().any(|m| m.contains("cameraSnap")), "{logs:?}");
}

#[test]
fn letterbox_and_wipe_in_an_export() {
    let (Some(red), Some(blue)) = (colour_clip("v2_red.mp4", "red"), colour_clip("v2_blue.mp4", "blue")) else {
        eprintln!("SKIP: ffmpeg not available");
        return;
    };
    let assets = vec![asset_for(&red, "r"), asset_for(&blue, "b")];
    let c1 = Clip { id: "c1".into(), asset_id: "r".into(), start_ms: 0.0, in_ms: 0.0, out_ms: 1000.0, ..Default::default() };
    let mut c2 = Clip { id: "c2".into(), asset_id: "b".into(), start_ms: 1000.0, in_ms: 0.0, out_ms: 1000.0, ..Default::default() };
    c2.transition_in = Some(TransitionIn { kind: TransitionType::WipeRight, duration_ms: 1000.0 });
    let mut lb = Clip { id: "lb".into(), start_ms: 1500.0, in_ms: 0.0, out_ms: 500.0, ..Default::default() };
    lb.effect = Some(ClipEffect { kind: EffectType::Letterbox, intensity: 1.0, params: Some([("ratio".to_string(), 2.39)].into_iter().collect()) });
    let project = Project {
        fps: 24.0,
        width: 320,
        height: 180,
        assets,
        tracks: vec![
            Track { id: "v".into(), kind: TrackKind::Video, clips: vec![c1, c2], ..Default::default() },
            Track { id: "fx".into(), kind: TrackKind::Fx, clips: vec![lb], ..Default::default() },
        ],
        ..Default::default()
    };
    let out = test_dir().join("export_v2_wipe_letterbox.mp4");
    let (done, sink) = run_export(project, &out, "h264_mp4", None);
    assert_eq!(done["ok"], true, "{:?} / {:?}", done["error"], sink.named("export://log"));
    // wipeRight at the cut (p = 0.5): B (blue) on the left half, A (red) on the right half
    let f = frame_rgb(&out, 24);
    let left = region_mean(&f, 320, 40, 60, 140, 120);
    let right = region_mean(&f, 320, 180, 60, 280, 120);
    assert!(left[2] > 200.0 && right[0] > 200.0, "wipe edge in the middle: {left:?} {right:?}");
    // letterbox 2.39 at 320x180: bars of (180 − 320/2.39)/2 = 23.05 px (full strength at 1750 ms)
    let f = frame_rgb(&out, 42);
    let bar = region_mean(&f, 320, 0, 2, 320, 20);
    let pic = region_mean(&f, 320, 0, 30, 320, 150);
    let bottom = region_mean(&f, 320, 0, 160, 320, 178);
    assert!(bar.iter().all(|v| *v < 12.0) && bottom.iter().all(|v| *v < 12.0), "bars {bar:?} {bottom:?}");
    assert!(pic[2] > 200.0, "picture {pic:?}");
}

#[test]
fn frame_rate_conversion_blend_versus_nearest() {
    // every source frame 8 luma levels brighter than the previous one
    let Some(src) = synth("v2_ramp24.mp4", "nullsrc=s=160x90:r=24:d=1.5,geq=lum='16+8*N':cb=128:cr=128", None) else {
        eprintln!("SKIP: ffmpeg not available");
        return;
    };
    let run = |mode: crate::model::FrameInterpolation, name: &str| {
        let c = Clip { id: "c".into(), asset_id: "m".into(), start_ms: 0.0, in_ms: 0.0, out_ms: 1000.0, ..Default::default() };
        let p = Project {
            fps: 60.0,
            width: 160,
            height: 90,
            assets: vec![asset_for(&src, "m")],
            tracks: vec![Track { id: "v".into(), kind: TrackKind::Video, clips: vec![c], ..Default::default() }],
            frame_interpolation: Some(mode),
            ..Default::default()
        };
        let out = test_dir().join(name);
        let (done, sink) = run_export(p, &out, "h264_mp4", None);
        assert_eq!(done["ok"], true, "{:?}", done["error"]);
        let (frames, _) = video_stream_info(&out);
        assert_eq!(frames, 60, "1 s at 60 fps");
        let fr: Vec<f64> = (0..24).map(|n| region_mean(&frame_rgb(&out, n), 160, 20, 20, 140, 70)[0]).collect();
        assert!(fr.windows(2).all(|w| w[1] >= w[0] - 0.6), "brightness never goes back: {fr:?}");
        let dups = fr.windows(2).filter(|w| (w[1] - w[0]).abs() < 1.0).count();
        let logs: Vec<String> = sink.named("export://log").iter().filter_map(|e| e["message"].as_str().map(String::from)).collect();
        (dups, logs)
    };
    let (dups_none, logs) = run(crate::model::FrameInterpolation::None, "export_v2_60_none.mp4");
    assert!(dups_none >= 10, "'none' repeats source frames: {dups_none} duplicates in 23 steps");
    assert!(logs.iter().any(|m| m.contains("repeating the nearest source frame")), "{logs:?}");
    let (dups_blend, logs) = run(crate::model::FrameInterpolation::FrameBlend, "export_v2_60_blend.mp4");
    assert!(dups_blend <= 2, "'frameBlend' makes in-between frames: {dups_blend} duplicates");
    assert!(logs.iter().any(|m| m.contains("blending neighbour source frames")), "{logs:?}");
}
