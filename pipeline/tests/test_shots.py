import numpy as np
import pytest

from cappycat_pipeline import ffmpeg_util
from cappycat_pipeline.shots import detect_shots, frames_to_shots, predictions_to_scenes, transnet_available


def test_frames_to_shots_ranges():
    shots = frames_to_shots([48, 96], 144, 24.0, "pyscenedetect")
    assert [(s.startFrame, s.endFrame) for s in shots] == [(0, 47), (48, 95), (96, 143)]
    assert shots[0].confidence == 1.0 and shots[0].index == 0 and shots[2].index == 2
    assert shots[1].startMs == pytest.approx(2000.0) and shots[1].endMs == pytest.approx(95 / 24 * 1000, abs=0.01)
    assert frames_to_shots([], 10, 25, "transnetv2")[0].endFrame == 9
    # out-of-range / duplicate cuts are ignored
    assert len(frames_to_shots([0, 5, 5, 999], 10, 25, "transnetv2")) == 2


def test_predictions_to_scenes_reference_semantics():
    p = np.zeros(20)
    p[7] = 0.9
    p[14] = 0.6
    scenes = predictions_to_scenes(p, 0.5)
    assert scenes.tolist() == [[0, 7], [8, 14], [15, 19]]
    assert predictions_to_scenes(np.ones(5), 0.5).tolist() == [[0, 4]]
    assert predictions_to_scenes(np.zeros(5), 0.5).tolist() == [[0, 4]]


def test_probe_and_frames(synthetic_video):
    a = ffmpeg_util.probe(synthetic_video)
    assert a.kind == "video" and a.width == 320 and a.height == 180 and a.fps == 24.0 and a.hasAudio
    assert abs(a.durationMs - 6000) < 100
    assert "\\" not in a.path
    n = sum(1 for _ in ffmpeg_util.iter_frames(synthetic_video, src_size=(320, 180), width=160))
    assert n == 144
    f = ffmpeg_util.read_frame(synthetic_video, (48 - 0.1) / 24 * 1000, src_size=(320, 180))
    assert f is not None and f.shape == (180, 320, 3)
    b, g, r = f.mean(axis=(0, 1))
    assert r > 200 and b < 30 and g < 30  # first frame of the red segment
    small = next(ffmpeg_util.iter_frames(synthetic_video, src_size=(320, 180), size=(48, 27)))
    assert small.shape == (27, 48, 3)


def test_pyscenedetect_finds_three_shots(synthetic_video):
    shots = detect_shots(synthetic_video, method="pyscenedetect")
    assert len(shots) == 3
    assert all(s.method == "pyscenedetect" for s in shots)
    assert shots[0].startFrame == 0 and shots[-1].endFrame == 143
    assert abs(shots[1].startFrame - 48) <= 1 and abs(shots[2].startFrame - 96) <= 1
    for prev, nxt in zip(shots, shots[1:]):
        assert nxt.startFrame == prev.endFrame + 1
        assert nxt.startMs == pytest.approx(nxt.startFrame / 24 * 1000, abs=0.01)


def test_auto_falls_back_without_model(synthetic_video):
    if transnet_available():
        pytest.skip("TransNetV2 model present; auto would use it")
    shots = detect_shots(synthetic_video, method="auto")
    assert len(shots) == 3 and shots[0].method == "pyscenedetect"
