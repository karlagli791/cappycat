"""``interpolate --target-fps F``: exact output rate, frame k at source time ``in + k / F``, exact
copies where a source frame coincides, flow-warped in-betweens at arbitrary positions, hard cuts
held, real timestamps for VFR sources. CPU (DIS) tests run everywhere; the RAFT / CUDA ones are
marked ``gpu``."""
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from cappycat_pipeline import ffmpeg_util, interpolate, models, transitions
from cappycat_pipeline.interpolate import iter_target_fps_frames, parse_fps, target_frame_count

PIPELINE_DIR = Path(__file__).resolve().parent.parent

SQ_W, SQ_H, SQ_FPS, SQ_FRAMES = 640, 360, 24, 48
SQ_SIZE, SQ_Y = 64, 148


def sq_x(t: float) -> float:
    """Left edge of the moving square at source time t (10 px per 24 fps frame)."""
    return 40.0 + 240.0 * t


@pytest.fixture(scope="session")
def moving_square(tmp_path_factory, has_ffmpeg) -> Path:
    """2 s, 24 fps, 640x360: a textured green 64 px square moving right at 240 px/s over a static
    textured grey background (texture everywhere so optical flow is well defined)."""
    if not has_ffmpeg:
        pytest.skip("ffmpeg/ffprobe not found")
    import cv2

    out = tmp_path_factory.mktemp("square") / "square24.mp4"
    rng = np.random.default_rng(1)
    bg = cv2.GaussianBlur(rng.integers(0, 256, (SQ_H, SQ_W)).astype(np.float32), (0, 0), 2.0)
    bg = ((bg - bg.min()) / (bg.max() - bg.min()) * 130 + 60).astype(np.uint8)
    bg = np.dstack([bg, bg, bg])
    tex = cv2.GaussianBlur(rng.integers(0, 256, (SQ_SIZE, SQ_SIZE)).astype(np.float32), (0, 0), 1.5)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    sq = np.dstack([tex * 60, 180 + tex * 75, tex * 60]).astype(np.uint8)  # BGR, green
    cmd = [ffmpeg_util.find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-s", f"{SQ_W}x{SQ_H}", "-r", str(SQ_FPS), "-i", "-", "-c:v", "libx264", "-crf", "10",
           "-pix_fmt", "yuv420p", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for i in range(SQ_FRAMES):
        f = bg.copy()
        x = int(round(sq_x(i / SQ_FPS)))
        f[SQ_Y:SQ_Y + SQ_SIZE, x:x + SQ_SIZE] = sq
        p.stdin.write(f.tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return out


@pytest.fixture(scope="session")
def square_frames(moving_square):
    return list(ffmpeg_util.iter_frames(moving_square, src_size=(SQ_W, SQ_H)))


def square_center_x(frame: np.ndarray) -> float:
    f = frame.astype(np.int32)
    mask = (f[..., 1] - np.maximum(f[..., 0], f[..., 2])) > 70
    xs = np.nonzero(mask)[1]
    assert len(xs) > 0.8 * SQ_SIZE * SQ_SIZE
    return float(xs.mean())


@pytest.fixture()
def cpu_flow(monkeypatch):
    monkeypatch.setenv("CAPPYCAT_FLOW_BACKEND", "dis")
    monkeypatch.setenv("CAPPYCAT_DEVICE", "cpu")


def _frames(src, a, b, fps, stats=None, size=(SQ_W, SQ_H)):
    scale, batch = transitions.plan_flow(size[1], size[0])
    return list(iter_target_fps_frames(str(src), a, b, parse_fps(fps), size=size, scale=scale, batch=batch,
                                       stats=stats))


def _check_square_clip(res, src_frames, fps, tol_px=3.0):
    """Every copy is byte-identical to the decoded source frame at that time; every in-between has
    the square where linear motion puts it at that time."""
    errs = []
    for tf in res:
        t = float(tf.time)
        if tf.kind == "copy":
            i = round(t * SQ_FPS)
            assert abs(t - i / SQ_FPS) < 0.0005
            assert np.array_equal(tf.frame, src_frames[i]), f"frame {tf.index} is not the source frame {i}"
        elif tf.kind == "interp":
            assert 0.0 < tf.alpha < 1.0
            errs.append(square_center_x(tf.frame) - (sq_x(t) + SQ_SIZE / 2 - 0.5))
    assert errs
    assert max(abs(e) for e in errs) < tol_px, errs
    return errs


# --------------------------------------------------------------------------- helpers / arithmetic


def test_parse_fps_and_frame_count():
    assert parse_fps("60") == 60 and parse_fps(40) == 40 and parse_fps("50") == 50
    assert parse_fps("29.97") == Fraction(30000, 1001) and parse_fps("30000/1001") == Fraction(30000, 1001)
    assert parse_fps("23.976") == Fraction(24000, 1001) and parse_fps("59.94") == Fraction(60000, 1001)
    assert parse_fps("12.5") == Fraction(25, 2)
    for bad in ("0", "-5", "abc", "1/0"):
        with pytest.raises(ValueError):
            parse_fps(bad)
    assert target_frame_count(0, 5000, Fraction(60)) == 300
    assert target_frame_count(500, 1500, Fraction(24)) == 24
    assert target_frame_count(0, 2000, Fraction(30000, 1001)) == 60  # round(59.94)
    assert target_frame_count(0, 1010, Fraction(60)) == 61  # round(60.6)
    assert interpolate.fps_label(Fraction(30000, 1001)) == "30000/1001" and interpolate.fps_label(Fraction(60)) == "60"


# --------------------------------------------------------------------------- CPU (DIS) path


def test_target_fps_24_to_60_cpu(cpu_flow, moving_square, square_frames):
    stats = {}
    res = _frames(moving_square, 0, 2000, "60", stats)
    assert len(res) == 120 and [tf.index for tf in res] == list(range(120))
    assert [float(tf.time) for tf in res[:3]] == pytest.approx([0.0, 1 / 60, 2 / 60])
    assert stats["backend"] == "dis"
    # source frames at i/24 coincide with output k/60 for even i: 24 exact copies
    assert stats["copied"] == 24 and sum(tf.kind == "copy" for tf in res) == 24
    # the last two output times (1.967, 1.983 s) lie after the last source frame (1.958 s): held
    assert [tf.kind for tf in res[-2:]] == ["hold", "hold"]
    assert np.array_equal(res[-1].frame, square_frames[-1])
    _check_square_clip(res, square_frames, 60)
    alphas = sorted({round(tf.alpha, 3) for tf in res if tf.kind == "interp"})
    assert alphas == [0.2, 0.4, 0.6, 0.8]


def test_target_fps_24_to_30_non_integer_ratio_cpu(cpu_flow, moving_square, square_frames):
    stats = {}
    res = _frames(moving_square, 500, 1500, "30", stats)
    assert len(res) == 30
    # output k at 0.5 + k/30 = (12 + 0.8 k)/24: a source frame coincides every 5 output frames
    assert stats["copied"] == 6 and stats["interpolated"] == 24
    assert sorted({round(tf.alpha, 3) for tf in res if tf.kind == "interp"}) == [0.2, 0.4, 0.6, 0.8]
    _check_square_clip(res, square_frames, 30)


def _ffprobe_stream(path: Path) -> dict:
    cmd = [ffmpeg_util.find_ffprobe(), "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
           "stream=r_frame_rate,avg_frame_rate,nb_read_frames,duration,pix_fmt,codec_name", "-of", "json", str(path)]
    return json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)["streams"][0]


def _run_cli(args, cpu=True, timeout=600):
    env = dict(os.environ)
    if cpu:
        env.update(CAPPYCAT_FLOW_BACKEND="dis", CAPPYCAT_DEVICE="cpu")
    proc = subprocess.run([sys.executable, "-m", "cappycat_pipeline", *map(str, args)], cwd=str(PIPELINE_DIR),
                          capture_output=True, text=True, encoding="utf-8", timeout=timeout, env=env)
    events = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]  # stdout must be JSON only
    return proc, events


@pytest.mark.parametrize("fps,rate,frames", [("60", "60/1", 120), ("29.97", "30000/1001", 60)])
def test_target_fps_cli_cpu(moving_square, tmp_path, fps, rate, frames):
    out = tmp_path / f"sq_{fps}.mp4"
    proc, events = _run_cli(["interpolate", moving_square, "--in-ms", 0, "--out-ms", 2000, "--target-fps", fps,
                             "--out", out])
    assert proc.returncode == 0, proc.stderr
    assert events[-1] == {"event": "result", "path": ffmpeg_util.norm_path(out)}
    prog = [e for e in events if e["event"] == "progress"]
    assert prog and all(set(e) == {"event", "stage", "clip", "pct", "message"} for e in prog)
    assert all(e["stage"] == "export" and e["clip"] == moving_square.name for e in prog)
    pcts = [e["pct"] for e in prog]
    assert pcts == sorted(pcts) and pcts[-1] == 1.0 and "dis" in prog[-1]["message"]
    s = _ffprobe_stream(out)
    assert s["r_frame_rate"] == rate and s["codec_name"] == "h264" and s["pix_fmt"] == "yuv420p"
    assert int(s["nb_read_frames"]) == frames
    assert not list(tmp_path.glob("*.part*"))  # atomic output: no leftovers


def test_target_fps_cli_rejects_bad_args(moving_square, tmp_path):
    proc, events = _run_cli(["interpolate", moving_square, "--in-ms", 0, "--out-ms", 1000, "--target-fps", "0",
                             "--out", tmp_path / "x.mp4"])
    assert proc.returncode == 1 and any(e["event"] == "log" and e["level"] == "error" for e in events)
    proc, _ = _run_cli(["interpolate", moving_square, "--in-ms", 0, "--out-ms", 1000, "--target-fps", "60",
                        "--factor", "2", "--out", tmp_path / "y.mp4"])
    assert proc.returncode == 2  # argparse: mutually exclusive
    assert not (tmp_path / "x.mp4").exists() and not (tmp_path / "y.mp4").exists()


def test_target_fps_holds_hard_cut(cpu_flow, synthetic_video):
    """1.5-2.5 s at 60 fps straddles the testsrc -> red cut at 2 s (source frames 47 @ 1.958 s and
    48 @ 2.0 s): the output frames between them are those source frames, never a morph."""
    src = list(ffmpeg_util.iter_frames(synthetic_video, src_size=(320, 180)))
    stats = {}
    res = _frames(synthetic_video, 1500, 2500, "60", stats, size=(320, 180))
    assert len(res) == 60
    between = [tf for tf in res if 47 / 24 < float(tf.time) < 2.0]
    assert between and all(tf.kind == "cut" for tf in between) and stats["cutsHeld"] == len(between)
    for tf in between:
        assert np.array_equal(tf.frame, src[47]) or np.array_equal(tf.frame, src[48])
    after = [tf for tf in res if float(tf.time) >= 2.0]
    assert all(tf.frame[..., 2].mean() > 200 and tf.frame[..., :2].mean() < 40 for tf in after)  # pure red


@pytest.fixture(scope="session")
def vfr_square(tmp_path_factory, moving_square) -> Path:
    """The moving square re-muxed on a 1/60 timebase (pts 0 3 5 8 10 ... / 60): 24 fps content with
    uneven frame durations, the layout of AI-generated clips."""
    out = tmp_path_factory.mktemp("vfrsq") / "vfr_square.mp4"
    cmd = [ffmpeg_util.find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(moving_square),
           "-vf", "settb=1/60,setpts='round(PTS)'", "-fps_mode", "passthrough", "-video_track_timescale", "60",
           "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return out


def test_target_fps_vfr_uses_real_timestamps(cpu_flow, vfr_square):
    rate, vfr = interpolate.source_rate(str(vfr_square))
    assert vfr
    timed = list(ffmpeg_util.iter_frames_timed(vfr_square, 0, src_size=(SQ_W, SQ_H)))
    assert len(timed) == SQ_FRAMES
    ts = [t for t, _ in timed]
    assert all(t.denominator in (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60) for t in ts)  # on the 1/60 grid
    assert len({b - a for a, b in zip(ts, ts[1:])}) == 2  # 2/60 and 3/60: really uneven
    stats = {}
    res = _frames(vfr_square, 500, 1500, "60", stats)
    assert len(res) == 60
    # every source frame in [0.5, 1.5) sits exactly on the 60 fps grid: all of them are copied, at
    # their real time (a resample onto the 23.9 fps average grid would have moved them)
    in_range = [(t, f) for t, f in timed if Fraction(1, 2) <= t < Fraction(3, 2)]
    copies = [tf for tf in res if tf.kind == "copy"]
    assert len(copies) == len(in_range) == stats["copied"]
    for tf, (t, f) in zip(copies, in_range):
        assert tf.time == t and np.array_equal(tf.frame, f)
    for tf in res:
        if tf.kind == "interp":
            assert abs(square_center_x(tf.frame) - (sq_x(float(tf.time)) + SQ_SIZE / 2 - 0.5)) < 3.0


# --------------------------------------------------------------------------- GPU (RAFT) path


@pytest.fixture()
def raft_cuda(monkeypatch):
    monkeypatch.delenv("CAPPYCAT_FLOW_BACKEND", raising=False)
    monkeypatch.delenv("CAPPYCAT_DEVICE", raising=False)
    if not models.cuda_ok():
        pytest.skip("torch with CUDA not available")
    if models.raft_small_path() is None:
        pytest.skip("RAFT weights missing (download-models)")
    yield
    transitions.release_raft()


@pytest.mark.gpu
def test_target_fps_24_to_60_gpu(raft_cuda, moving_square, square_frames):
    import torch

    torch.cuda.reset_peak_memory_stats()
    stats = {}
    res = _frames(moving_square, 0, 2000, "60", stats)
    assert stats["backend"] == "raft-cuda"
    assert len(res) == 120 and stats["copied"] == 24 and stats["interpolated"] == 94
    _check_square_clip(res, square_frames, 60)
    assert torch.cuda.max_memory_allocated() / 2**20 < 2300  # ~2 GB budget


@pytest.mark.gpu
def test_target_fps_24_to_30_gpu(raft_cuda, moving_square, square_frames):
    stats = {}
    res = _frames(moving_square, 500, 1500, "30", stats)
    assert stats["backend"] == "raft-cuda" and len(res) == 30
    _check_square_clip(res, square_frames, 30)


@pytest.mark.gpu
def test_target_fps_cli_gpu(raft_cuda, moving_square, tmp_path):
    out = tmp_path / "sq60.mp4"
    proc, events = _run_cli(["interpolate", moving_square, "--in-ms", 250, "--out-ms", 1750, "--target-fps", "60",
                             "--out", out], cpu=False)
    assert proc.returncode == 0, proc.stderr
    assert "raft-cuda" in [e for e in events if e["event"] == "progress"][-1]["message"]
    s = _ffprobe_stream(out)
    assert s["r_frame_rate"] == "60/1" and int(s["nb_read_frames"]) == 90
    frames = list(ffmpeg_util.iter_frames(out, src_size=(SQ_W, SQ_H)))
    xs = [square_center_x(f) for f in frames]
    expect = [sq_x(0.25 + k / 60) + SQ_SIZE / 2 - 0.5 for k in range(90)]
    assert max(abs(a - b) for a, b in zip(xs, expect)) < 3.0  # after the lossy encode too


def test_target_fps_cancelled_mid_way_leaves_no_output(cpu_flow, moving_square, tmp_path):
    from cappycat_pipeline import procs

    out = tmp_path / "sq.mp4"
    calls = []

    def progress(pct, msg):
        calls.append(pct)
        if len(calls) == 3:
            raise KeyboardInterrupt("cancelled")

    with pytest.raises(KeyboardInterrupt):
        interpolate.interpolate_to_fps(str(moving_square), 0, 2000, "60", str(out), progress=progress)
    assert not out.exists() and list(tmp_path.iterdir()) == []
    assert procs.live_children() == []  # decoder and encoder were killed
