import numpy as np
import pytest

from cappycat_pipeline.reframe import build_reframe_track, smooth_crops, smooth_track, solve_crop, static_track
from cappycat_pipeline.schema import DetectedInstance, DuplicateFinding, ReframeKeyframe, Shot

W, H = 1920, 1080
ASPECT = W / H


def _overlaps(crop, box):
    return crop[0] < box[2] and crop[2] > box[0] and crop[1] < box[3] and crop[3] > box[1]


def _contains(crop, box, tol=1e-3):
    return crop[0] <= box[0] + tol and crop[1] <= box[1] + tol and crop[2] >= box[2] - tol and crop[3] >= box[3] - tol


def _aspect_ok(crop, aspect=ASPECT, tol=2e-3):
    w, h = crop[2] - crop[0], crop[3] - crop[1]
    return abs(w / h - aspect) < tol


def _in_frame(crop, w=W, h=H, tol=1e-6):
    return crop[0] >= -tol and crop[1] >= -tol and crop[2] <= w + tol and crop[3] <= h + tol


def test_excludes_duplicate_and_contains_primary():
    primary = (700, 300, 1000, 900)
    dup = (1500, 200, 1800, 900)
    crop = solve_crop(W, H, primary, [dup])
    assert not _overlaps(crop, dup)
    assert _contains(crop, primary)
    assert _aspect_ok(crop)
    assert _in_frame(crop)
    # largest feasible width is bounded by the duplicate's left edge
    assert crop[2] - crop[0] == pytest.approx(1500, abs=2)


def test_no_duplicates_returns_full_frame():
    crop = solve_crop(W, H, (100, 100, 400, 500), [])
    assert crop == (0.0, 0.0, float(W), float(H))


def test_multiple_duplicates_both_sides():
    primary = (800, 300, 1100, 800)
    dups = [(0, 0, 200, 1080), (1500, 0, 1920, 1080), (900, 0, 1000, 100)]
    crop = solve_crop(W, H, primary, dups)
    for d in dups:
        assert not _overlaps(crop, d)
    assert _contains(crop, primary)
    assert _aspect_ok(crop)


def test_max_area_beats_alignment():
    """Area is maximised before alignment: the only 1300-wide crop sits at x1 = 200."""
    dups = [(0, 0, 200, 1080), (1500, 0, 1920, 1080)]
    crop = solve_crop(W, H, (650, 400, 750, 600), dups)
    assert crop[0] == pytest.approx(200, abs=1e-3) and crop[2] == pytest.approx(1500, abs=1e-3)


def test_thirds_alignment_picks_nearer_line():
    # a top strip caps the height -> w <= 880 * 16/9 = 1564.4, leaving x1 free in [0, 355.6]
    dups = [(0, 0, 1920, 200)]
    w_max = 880 * ASPECT
    # primary centre at 700: only the 1/3 line is reachable (x1 = 700 - 521.5 = 178.5)
    crop = solve_crop(W, H, (650, 400, 750, 600), dups)
    w = crop[2] - crop[0]
    assert w == pytest.approx(w_max, abs=0.5) and crop[1] == pytest.approx(200, abs=1e-3)
    assert abs(700 - (crop[0] + w / 3)) < 0.5
    # primary centre at 1300: only the 2/3 line is reachable (x1 = 1300 - 1043 = 257)
    crop = solve_crop(W, H, (1250, 400, 1350, 600), dups)
    w = crop[2] - crop[0]
    assert w == pytest.approx(w_max, abs=0.5)
    assert abs(1300 - (crop[0] + 2 * w / 3)) < 0.5
    assert abs(1300 - (crop[0] + 2 * w / 3)) < abs(1300 - (crop[0] + w / 3))


def test_relaxes_head_first_when_full_containment_impossible():
    primary = (100, 50, 1700, 1050)  # nearly the whole frame
    dup = (1650, 300, 1900, 700)  # overlaps the primary's right edge
    crop = solve_crop(W, H, primary, [dup])
    assert not _overlaps(crop, dup)
    assert _aspect_ok(crop)
    assert _in_frame(crop)
    # the upper body (top 55 %, central 70 %) is kept, with the crop top at/above the head
    upper = (100 + 0.15 * 1600, 50, 1700 - 0.15 * 1600, 50 + 0.55 * 1000)
    assert _contains(crop, upper, tol=1.0)
    assert crop[1] <= 50 + 1.0


def test_tall_character_keeps_head_not_legs():
    # clip7 case: a full-height character next to its duplicate; no 16:9 crop that excludes the
    # duplicate can be tall enough, so the head (with headroom) must stay in frame and the legs go.
    W7, H7 = 1280, 720
    primary = (116, 54, 506, 720)
    dup = (720, 106, 1142, 720)
    crop = solve_crop(W7, H7, primary, [dup])
    assert not _overlaps(crop, dup)
    assert abs((crop[2] - crop[0]) / (crop[3] - crop[1]) - W7 / H7) < 1e-3
    assert crop[1] <= 54 + 1.0, f"head cut off: crop top {crop[1]:.1f} below head top 54"
    assert crop[3] < 720  # the legs are what gets cropped


def test_best_effort_when_everything_blocked():
    primary = (900, 400, 1100, 700)
    dup = (0, 0, 1920, 1080)  # the duplicate covers the whole frame -> nothing is feasible
    crop = solve_crop(W, H, primary, [dup])
    assert _aspect_ok(crop)
    assert _in_frame(crop)
    assert crop[2] > crop[0] and crop[3] > crop[1]


def test_custom_aspect_and_vertical_video():
    crop = solve_crop(1080, 1920, (300, 800, 700, 1500), [(0, 0, 1080, 300)])
    assert _aspect_ok(crop, 1080 / 1920)
    assert _in_frame(crop, 1080, 1920)
    assert not _overlaps(crop, (0, 0, 1080, 300))
    crop = solve_crop(W, H, (700, 300, 1000, 900), [(1500, 200, 1800, 900)], aspect=1.0)
    assert _aspect_ok(crop, 1.0)


def test_degenerate_inputs():
    # zero-area duplicate is ignored; primary outside frame is clamped
    crop = solve_crop(W, H, (-50, -50, 300, 300), [(500, 500, 500, 500)])
    assert _in_frame(crop) and _aspect_ok(crop)
    crop = solve_crop(W, H, (0, 0, W, H), [(1800, 0, 1920, 1080)])
    assert _in_frame(crop) and _aspect_ok(crop) and not _overlaps(crop, (1800, 0, 1920, 1080))


@pytest.mark.parametrize("method", ["ema", "savgol"])
def test_smoothing_stays_in_bounds_and_reduces_jitter(method):
    rng = np.random.default_rng(1)
    n = 120
    base_w = 1200 + 300 * np.sin(np.linspace(0, 3, n))
    x = np.clip(200 + 100 * np.sin(np.linspace(0, 6, n)) + rng.normal(0, 40, n), 0, None)
    crops = np.stack([x, np.full(n, 30.0), x + base_w, 30.0 + base_w / ASPECT], axis=1)
    crops[:, 2] = np.minimum(crops[:, 2], W)
    sm = smooth_crops(crops, W, H, method=method, alpha=0.15, window=15)
    assert sm.shape == crops.shape
    assert (sm[:, 0] >= 0).all() and (sm[:, 1] >= 0).all() and (sm[:, 2] <= W + 1e-6).all() and (sm[:, 3] <= H + 1e-6).all()
    aspects = (sm[:, 2] - sm[:, 0]) / (sm[:, 3] - sm[:, 1])
    assert np.allclose(aspects, ASPECT, atol=1e-6)
    jitter_in = np.abs(np.diff(crops[:, 0])).mean()
    jitter_out = np.abs(np.diff(sm[:, 0])).mean()
    assert jitter_out < 0.5 * jitter_in


def test_smooth_track_keyframes():
    kfs = [ReframeKeyframe(frame=i, timeMs=i * 41.67, crop=[100 + (i % 2) * 50, 0, 1700 + (i % 2) * 50, 900], zoom=1.129, tx=0, ty=0)
           for i in range(40)]
    out = smooth_track(kfs, "ema", 0.2, 15, frame_w=W, frame_h=H, fps=24)
    assert len(out) == 40
    for k in out:
        assert 0 <= k.crop[0] and k.crop[2] <= W and 0 <= k.crop[1] and k.crop[3] <= H
        assert -1 <= k.tx <= 1 and -1 <= k.ty <= 1
        assert k.zoom == pytest.approx(W / (k.crop[2] - k.crop[0]), rel=1e-3)
    assert smooth_track([], "ema") == []


def _finding(shot_idx, frame, fps, primary, dup):
    return DuplicateFinding(shotIndex=shot_idx, frame=frame, timeMs=frame / fps * 1000,
                            primary=DetectedInstance(1, "raccoon", list(primary), 0.9),
                            duplicate=DetectedInstance(2, "raccoon", list(dup), 0.8), similarity=0.93)


def test_build_reframe_track_per_frame_keyframes():
    fps = 24.0
    shot = Shot(index=0, startFrame=0, endFrame=95, startMs=0, endMs=95 / fps * 1000, confidence=1, method="pyscenedetect")
    findings = [_finding(0, f, fps, (700, 300, 1000, 900), (1500 + (f % 12) * 5, 200, 1800, 900)) for f in range(0, 96, 6)]
    track = build_reframe_track(shot, findings, W, H, fps, method="ema", alpha=0.15)
    assert track is not None
    assert track.sourceWidth == W and track.sourceHeight == H
    assert len(track.keyframes) == 96
    assert track.keyframes[0].frame == 0 and track.keyframes[-1].frame == 95
    assert "raccoon" in (track.reason or "")
    for k in track.keyframes:
        assert k.timeMs == pytest.approx(k.frame / fps * 1000, abs=0.01)
        assert k.zoom == pytest.approx(W / (k.crop[2] - k.crop[0]), rel=1e-3)
        assert -1 <= k.tx <= 1 and -1 <= k.ty <= 1
        assert k.crop[2] <= 1555 + 1  # never past the duplicate's right-most left edge (1500 + 11*5)
    # thinning to every 2nd frame when the shot is long
    long_shot = Shot(index=1, startFrame=0, endFrame=1499, startMs=0, endMs=1499 / fps * 1000, confidence=1, method="pyscenedetect")
    findings = [_finding(1, f, fps, (700, 300, 1000, 900), (1500, 200, 1800, 900)) for f in range(0, 1500, 6)]
    track = build_reframe_track(long_shot, findings, W, H, fps)
    assert track is not None and 750 <= len(track.keyframes) <= 751
    assert build_reframe_track(shot, [], W, H, fps) is None


def test_static_track():
    shot = Shot(0, 0, 47, 0, 1958.3, 1, "pyscenedetect")
    t = static_track(W, H, [120, 0, 1800, 1080], shot, 24.0, "director")
    assert len(t.keyframes) == 2 and t.keyframes[1].frame == 47
    c = t.keyframes[0].crop
    assert _aspect_ok(c) and _in_frame(c)
