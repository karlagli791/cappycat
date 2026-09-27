"""VFR handling + the ``interpolate`` subcommand on the CPU (DIS / Farneback flow, no torch)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cappycat_pipeline import ffmpeg_util
from cappycat_pipeline.interpolate import segment_frame_range, source_rate
from cappycat_pipeline.shots import detect_shots

PIPELINE_DIR = Path(__file__).resolve().parent.parent


def _ffprobe_stream(path: Path) -> dict:
    cmd = [ffmpeg_util.find_ffprobe(), "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
           "stream=r_frame_rate,avg_frame_rate,nb_read_frames,duration,pix_fmt,codec_name", "-show_entries",
           "format=nb_streams", "-of", "json", str(path)]
    d = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    s = d["streams"][0]
    s["nb_streams"] = int(d["format"]["nb_streams"])
    return s


@pytest.fixture(scope="session")
def vfr_video(tmp_path_factory, has_ffmpeg) -> Path:
    """testsrc -> red -> blue, 2 s each, 24 fps content stored on a 1/60 timebase (pts 0 3 5 8 ...):
    r_frame_rate 60/1, avg_frame_rate ~23.93 - the layout of the user's AI-generated clips."""
    if not has_ffmpeg:
        pytest.skip("ffmpeg/ffprobe not found")
    out = tmp_path_factory.mktemp("vfr") / "vfr.mp4"
    cmd = [ffmpeg_util.find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-f", "lavfi", "-i", "testsrc=size=320x180:rate=24:duration=2",
           "-f", "lavfi", "-i", "color=c=red:size=320x180:rate=24:duration=2",
           "-f", "lavfi", "-i", "color=c=blue:size=320x180:rate=24:duration=2",
           "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0,settb=1/60,setpts='round(PTS)'[v]", "-map", "[v]",
           "-fps_mode", "passthrough", "-video_track_timescale", "60", "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return out


def test_vfr_fixture_is_really_vfr(vfr_video):
    s = _ffprobe_stream(vfr_video)
    assert s["r_frame_rate"] == "60/1" and s["avg_frame_rate"] != "60/1"
    rate, vfr = source_rate(str(vfr_video))
    assert vfr and 23.5 < float(rate) < 24.5


def test_vfr_frame_count_and_shot_times(vfr_video):
    a = ffmpeg_util.probe(vfr_video)
    # every decoded frame exactly once (no CFR-60 duplication)
    n = sum(1 for _ in ffmpeg_util.iter_frames(vfr_video, src_size=(a.width, a.height), width=160))
    assert n == 144
    shots = detect_shots(vfr_video, method="pyscenedetect", asset=a)
    assert len(shots) == 3
    assert all(s.endMs <= a.durationMs + 1 for s in shots)
    assert abs(shots[1].startMs - 2000) < 120 and abs(shots[2].startMs - 4000) < 120
    assert shots[-1].endFrame == 143


def test_segment_frame_range():
    assert segment_frame_range(0, 2000, 24.0) == (0, 48)
    assert segment_frame_range(500, 1500, 24.0) == (12, 36)
    # millisecond rounding of frame times (1000 * 25 / 24 = 1041.67 -> 1042) still lands on frame 25
    assert segment_frame_range(1042, 2000, 24.0) == (25, 48)


def _run_interpolate(src, a, b, factor, out, extra_env=None):
    env = dict(os.environ, CAPPYCAT_FLOW_BACKEND="dis", CAPPYCAT_DEVICE="cpu", **(extra_env or {}))
    proc = subprocess.run([sys.executable, "-m", "cappycat_pipeline", "interpolate", str(src), "--in-ms", str(a),
                           "--out-ms", str(b), "--factor", str(factor), "--out", str(out)],
                          cwd=str(PIPELINE_DIR), capture_output=True, text=True, encoding="utf-8", timeout=300, env=env)
    events = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]  # stdout must be JSON only
    return proc, events


def test_interpolate_cli_cpu_dis(synthetic_video, tmp_path):
    out = tmp_path / "slow.mp4"
    proc, events = _run_interpolate(synthetic_video, 500, 1500, 3, out)
    assert proc.returncode == 0, proc.stderr
    assert events[-1] == {"event": "result", "path": ffmpeg_util.norm_path(out)}
    prog = [e for e in events if e["event"] == "progress"]
    assert prog and all(set(e) == {"event", "stage", "clip", "pct", "message"} for e in prog)
    assert all(e["stage"] == "export" and e["clip"] == synthetic_video.name and 0 <= e["pct"] <= 1 for e in prog)
    assert prog[-1]["pct"] == 1.0 and "dis" in prog[-1]["message"]
    s = _ffprobe_stream(out)
    assert int(s["nb_read_frames"]) == 24 * 3  # K source frames x factor
    assert s["r_frame_rate"] == "72/1" and s["codec_name"] == "h264" and s["pix_fmt"] == "yuv420p"
    assert s["nb_streams"] == 1  # no audio
    # in-betweens are real intermediate frames: consecutive output frames change smoothly
    frames = list(ffmpeg_util.iter_frames(out, src_size=(320, 180)))
    d = [np.abs(frames[i].astype(int) - frames[i + 1].astype(int)).mean() for i in range(len(frames) - 1)]
    assert max(d) < 40


def test_interpolate_holds_across_hard_cut(synthetic_video, tmp_path):
    """1.5-2.5 s straddles the testsrc -> red cut at 2 s: no morphed (half testsrc, half red) frame."""
    out = tmp_path / "cut.mp4"
    proc, events = _run_interpolate(synthetic_video, 1500, 2500, 4, out)
    assert proc.returncode == 0, proc.stderr
    assert "cut(s) held" in [e for e in events if e["event"] == "progress"][-1]["message"]
    frames = list(ffmpeg_util.iter_frames(out, src_size=(320, 180)))
    assert len(frames) == 24 * 4
    red = [f for f in frames if f[..., 2].mean() > 200 and f[..., :2].mean() < 40]
    assert len(red) >= 12 * 4 - 2  # the red half stays pure red (held, not blended)


def test_interpolate_vfr_source_resampled(vfr_video, tmp_path):
    out = tmp_path / "vfr_slow.mp4"
    proc, events = _run_interpolate(vfr_video, 500, 1500, 2, out)
    assert proc.returncode == 0, proc.stderr
    assert "VFR source" in [e for e in events if e["event"] == "progress"][0]["message"]
    rate, _ = source_rate(str(vfr_video))
    i0, i1 = segment_frame_range(500, 1500, float(rate))
    s = _ffprobe_stream(out)
    assert int(s["nb_read_frames"]) == (i1 - i0) * 2
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    assert abs(num / den - 2 * float(rate)) < 0.01
    assert abs(float(s["duration"]) - 1.0) < 0.05


def test_interpolate_rejects_bad_args(synthetic_video, tmp_path):
    proc, events = _run_interpolate(synthetic_video, 1500, 500, 2, tmp_path / "x.mp4")
    assert proc.returncode == 1
    assert any(e["event"] == "log" and e["level"] == "error" for e in events)


def _mean_rgb(path, at_ms: float, a) -> np.ndarray:
    """Mean BGR of the frame on screen at ``at_ms`` (read_frame alone returns the first frame at or
    after the time)."""
    clock = ffmpeg_util.frame_clock(path, a.fps)
    f = ffmpeg_util.read_frame(path, clock.seek_ms(clock.index_at(at_ms)), width=160, src_size=(a.width, a.height))
    assert f is not None
    return f.reshape(-1, 3).mean(axis=0)


def test_frame_clock_reads_real_presentation_times(vfr_video):
    a = ffmpeg_util.probe(vfr_video)
    clock = ffmpeg_util.frame_clock(vfr_video, a.fps)
    assert len(clock) == 144
    # 24 fps content on a 1/60 timebase: pts round(k * 2.5) ticks -> gaps alternate 33 / 50 ms
    for k in (0, 1, 2, 47, 48, 49, 97):
        assert clock.ms(k) == pytest.approx(round(k * 2.5 + 1e-9) * 1000.0 / 60.0, abs=0.01)
    assert clock.index_at(clock.ms(48)) == 48 and clock.index_at(clock.ms(48) - 0.5) == 47
    assert clock.nearest(clock.ms(48) + 5) == 48
    assert clock.ms(-1) < 0 and clock.ms(200) > clock.ms(143)
    # cached per file
    assert ffmpeg_util.frame_clock(vfr_video, a.fps) is clock


def test_vfr_shot_times_are_real_cut_times(vfr_video):
    """Shots start at the real presentation time of their first frame, not index / avg fps (which
    drifts tens of ms on these sources and put a zoomed shot's first frame on the previous angle)."""
    a = ffmpeg_util.probe(vfr_video)
    clock = ffmpeg_util.frame_clock(vfr_video, a.fps)
    shots = detect_shots(vfr_video, method="pyscenedetect", asset=a)
    assert [s.startFrame for s in shots] == [0, 48, 96]
    for s in shots:
        assert s.startMs == pytest.approx(clock.ms(s.startFrame), abs=0.001)
        assert s.endMs == pytest.approx(clock.ms(s.endFrame), abs=0.001)


def _grid_frame(clock, tick: int, grid: float, lead_ms: float, bias: float = 0.0) -> int:
    """The frame ffmpeg's ``fps=grid`` filter shows on ``tick``: the latest frame whose timestamp,
    shifted by the exporter's seek lead and rounded half-up to the grid, is <= tick (``bias``
    resolves exact ties either way)."""
    ticks = np.floor((clock.pts + lead_ms) * grid / 1000.0 + 0.5 + bias)
    return max(0, int(np.searchsorted(ticks, tick, side="right")) - 1)


def test_assembled_cuts_show_only_their_own_shot(vfr_video):
    """Each clip's in/out points sit on the right side of the cut both for the preview (frame whose
    display interval contains the time) and for the exporter's grid (nearest frame to floor(t*grid))."""
    from cappycat_pipeline.assemble import EXPORT_GRIDS, SEEK_LEAD_MS, shot_ranges

    a = ffmpeg_util.probe(vfr_video)
    clock = ffmpeg_util.frame_clock(vfr_video, a.fps)
    shots = detect_shots(vfr_video, method="pyscenedetect", asset=a)
    ranges = shot_ranges(shots, a.fps, a.durationMs)
    for shot, (i, o) in zip(shots, ranges):
        assert i < o
        # preview
        assert clock.index_at(i) >= shot.startFrame
        assert clock.index_at(o - 0.01) <= shot.endFrame
        # exporter grids (multiples of a safe grid are safe too), with and without the seek lead
        for g in EXPORT_GRIDS:
            for lead in (0.0, SEEK_LEAD_MS):
                first = int(np.floor(i * g / 1000.0 + 1e-3))
                last = int(np.floor((o - 0.01) * g / 1000.0 + 1e-3))
                assert _grid_frame(clock, first, g, lead, 1e-6) >= shot.startFrame, (g, lead, shot.index)
                assert _grid_frame(clock, last, g, lead, -1e-6) <= shot.endFrame, (g, lead, shot.index)
        # never more than about half a frame of the shot's head is skipped
        assert i - clock.ms(shot.startFrame) < 25.0 and clock.ms(shot.endFrame + 1) - o < 25.0
    # and the pixels agree: red just before the second cut, blue right at it
    red = _mean_rgb(vfr_video, ranges[1][1] - 1.0, a)
    blue = _mean_rgb(vfr_video, ranges[2][0], a)
    assert red[2] > 150 and red[0] < 80  # BGR
    assert blue[0] > 150 and blue[2] < 80


def test_sampled_shot_frames_never_include_the_previous_shot(vfr_video):
    from cappycat_pipeline.perception import sample_shot_frames

    a = ffmpeg_util.probe(vfr_video)
    shots = detect_shots(vfr_video, method="pyscenedetect", asset=a)
    frames = list(sample_shot_frames(str(vfr_video), shots[2], a.fps, (a.width, a.height), sample_fps=24.0, width=64))
    assert frames and frames[0][0] == shots[2].startFrame
    for _, f in frames:
        m = f.reshape(-1, 3).mean(axis=0)
        assert m[0] > 150 and m[2] < 80  # all blue
