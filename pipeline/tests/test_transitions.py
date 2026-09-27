import numpy as np

from cappycat_pipeline.transitions import DISSOLVE_THRESHOLD, interpolate_frames, score_transition, suggest


def _textured(seed, w=320, h=180):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(40):
        x, y = rng.integers(0, w - 30), rng.integers(0, h - 30)
        img[y:y + 30, x:x + 30] = rng.integers(0, 255, 3)
    return img


def test_identical_frames_are_smooth_cut():
    a = _textured(0)
    mag, smooth = score_transition(a, a.copy(), source_width=1920)
    assert mag < 0.5 and smooth > 0.9
    assert suggest(smooth) == "cut"


def test_shifted_frame_has_flow_scaled_to_source():
    a = _textured(1)
    b = np.roll(a, 12, axis=1)
    mag, smooth = score_transition(a, b, source_width=320)
    mag_src, _ = score_transition(a, b, source_width=1920)
    assert 5 < mag < 20
    assert abs(mag_src - mag * 6) < 1e-3


def test_unrelated_frames_are_less_smooth_and_suggest_dissolve():
    a = _textured(2)
    b = _textured(3)
    _, smooth_same = score_transition(a, a)
    _, smooth_diff = score_transition(a, b)
    assert smooth_diff < smooth_same
    assert suggest(DISSOLVE_THRESHOLD - 0.01) == "dissolve" and suggest(DISSOLVE_THRESHOLD) == "cut"


def test_interpolate_frames_shape_and_blend():
    a = _textured(4)
    b = np.roll(a, 6, axis=1)
    mids = interpolate_frames(a, b, 3)
    assert len(mids) == 3 and all(m.shape == a.shape and m.dtype == np.uint8 for m in mids)
    assert interpolate_frames(a, b, 0) == []
    # the midpoint should be closer to both endpoints than they are to each other
    d_ab = np.abs(a.astype(int) - b.astype(int)).mean()
    d_am = np.abs(a.astype(int) - mids[1].astype(int)).mean()
    assert d_am < d_ab
