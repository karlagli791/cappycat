"""Shared fixtures: a synthetic 6 s / 24 fps / 320x180 video with two hard cuts
(``testsrc`` -> red -> blue) and a 440 Hz sine audio track."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from cappycat_pipeline import ffmpeg_util

# every CLI run of the test-suite (also in subprocesses) uses a private per-clip result cache, so
# tests neither read the user's cache nor fill it
os.environ["CAPPYCAT_ANALYSIS_CACHE_DIR"] = tempfile.mkdtemp(prefix="cappycat-test-cache-")

SYNTH_FPS = 24
SYNTH_SIZE = "320x180"
SYNTH_SEG_SECONDS = 2


def make_synthetic_video(out_path: Path, seg_seconds: int = SYNTH_SEG_SECONDS, with_audio: bool = True) -> Path:
    ffmpeg = ffmpeg_util.find_ffmpeg()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "lavfi", "-i", f"testsrc=size={SYNTH_SIZE}:rate={SYNTH_FPS}:duration={seg_seconds}",
        "-f", "lavfi", "-i", f"color=c=red:size={SYNTH_SIZE}:rate={SYNTH_FPS}:duration={seg_seconds}",
        "-f", "lavfi", "-i", f"color=c=blue:size={SYNTH_SIZE}:rate={SYNTH_FPS}:duration={seg_seconds}",
    ]
    total = 3 * seg_seconds
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={total}"]
    cmd += ["-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]"]
    if with_audio:
        cmd += ["-map", "3:a", "-c:a", "aac", "-b:a", "128k"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-g", "12",
            "-r", str(SYNTH_FPS), "-shortest", str(out_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return out_path


@pytest.fixture(scope="session")
def has_ffmpeg() -> bool:
    return ffmpeg_util.ffmpeg_available()


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory, has_ffmpeg: bool) -> Path:
    if not has_ffmpeg:
        pytest.skip("ffmpeg/ffprobe not found (set CAPPYCAT_FFMPEG_DIR)")
    out = tmp_path_factory.mktemp("media") / "synthetic_cuts.mp4"
    return make_synthetic_video(out)
