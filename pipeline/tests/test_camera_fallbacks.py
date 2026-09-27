"""Camera-path fallbacks and reframe / tracking changes of the review fixes: side-flip segment
splitting, soft exclusion, the QP frame cap, secondary primaries, relaxing the sampled-frame
reframe at the ends and holding an unseen duplicate back to the shot start."""
import time

import numpy as np
import pytest

pytest.importorskip("clarabel")

from cappycat_pipeline import camerapath
from cappycat_pipeline.camerapath import CameraStyle, path_metrics, plan_camera_path, split_by_side
from cappycat_pipeline.dupetrack import track_pair
from cappycat_pipeline.reframe import build_reframe_track, solve_crop
from cappycat_pipeline.schema import DetectedInstance, DuplicateFinding, Shot
from cappycat_pipeline.tracking import Detection

W, H = 1280.0, 720.0


def _hits(c, d):
    return c[0] < d[2] - 0.5 and c[2] > d[0] + 0.5 and c[1] < d[3] - 0.5 and c[3] > d[1] + 0.5


def _crossing(n=120):
    """The duplicate walks from the right of the (static) primary to its left while staying visible:
    no single side works for the whole visibility segment."""
    prim = (560.0, 60.0, 720.0, 700.0)
    head = (592.0, 60.0, 688.0, 252.0)
    dups, targets = [], []
    for i in range(n):
        x1 = 1000.0 - (900.0 * i / (n - 1))
        d = (x1, 80.0, x1 + 180.0, 720.0)
        dups.append(d)
        targets.append(np.array(solve_crop(W, H, prim, [d])))
    return np.array(targets), [dups], [head] * n, [prim] * n


def test_side_flip_splits_the_segment_when_one_side_is_infeasible():
    targets, dups, heads, prim = _crossing()
    style = CameraStyle()
    # the approved one-side plan is infeasible here ...
    assert camerapath._plan(targets, dups, heads, W, H, None, style, prim) is None
    info = {}
    crops = plan_camera_path(targets, dups, heads, W, H, primary=prim, info=info)
    assert crops is not None and info["mode"] == "split-sides"
    for i, c in enumerate(crops):
        assert not _hits(c, dups[0][i]), f"frame {i}"
        assert c[0] >= -0.5 and c[2] <= W + 0.5
    # ... the crop is left of the duplicate first, right of it at the end
    assert crops[0][2] <= dups[0][0][0] + 0.5 and crops[-1][0] >= dups[0][-1][2] - 0.5


def test_split_by_side_hysteresis_ignores_short_flips():
    L, R = camerapath.LEFT, camerapath.RIGHT
    sides = [L] * 10 + [R, R] + [L] * 10 + [R] * 12
    assert split_by_side(sides, 0, len(sides), 6) == [(0, 22, L), (22, 34, R)]
    assert split_by_side([L] * 5, 0, 5, 6) == [(0, 5, L)]


def test_feasible_shots_keep_the_approved_plan():
    n = 50
    targets = np.tile(np.array([0.0, 0.0, 700.0, 393.75]), (n, 1))
    dups = [[(760.0, 50.0, 1000.0, 700.0)] * n]
    heads = [(300.0, 40.0, 500.0, 240.0)] * n
    info = {}
    crops = plan_camera_path(targets, dups, heads, W, H, info=info)
    plain = camerapath._plan(targets, dups, heads, W, H, None, CameraStyle(), None)
    assert info["mode"] == "sides" and np.allclose(crops, plain)


def test_soft_exclusion_is_the_last_resort():
    n = 40
    targets = np.tile(np.array([0.0, 0.0, W, H]), (n, 1))
    wall = (0.0, 0.0, W, H)  # nothing can avoid it
    dups = [[None] * 10 + [wall] * 20 + [None] * 10]
    info = {}
    crops = plan_camera_path(targets, dups, [None] * n, W, H, info=info)
    assert crops is not None and info["mode"] == "soft-exclusion"
    assert np.isfinite(crops).all()


def test_long_shots_are_planned_on_a_decimated_grid():
    n = 3000  # ~2 min at 24 fps
    t = np.arange(n)
    x1 = 900.0 + 150.0 * np.sin(t / 400.0)
    dups = [[(float(a), 100.0, float(a) + 200.0, 720.0) for a in x1]]
    targets = np.array([[0.0, 0.0, a - 20.0, (a - 20.0) * H / W] for a in x1])
    heads = [(150.0, 60.0, 400.0, 240.0)] * n
    info = {}
    t0 = time.perf_counter()
    crops = plan_camera_path(targets, dups, heads, W, H, info=info)
    took = time.perf_counter() - t0
    assert crops is not None and crops.shape == (n, 4) and info["step"] == 3
    assert took < 60
    for i in range(0, n, 7):
        assert not _hits(crops[i], dups[0][i])
    m = path_metrics(crops, W)
    assert m["maxZoomStep"] < 0.05 and m["maxPanAccelPx"] < 3.0


def test_other_primaries_are_kept_in_frame_softly():
    n = 60
    targets = np.tile(np.array([0.0, 0.0, 640.0, 360.0]), (n, 1))  # tight on the main primary
    head = (100.0, 50.0, 300.0, 250.0)
    other = [(700.0, 80.0, 820.0, 200.0)] * n  # second pair's primary, just right of the target crop
    without = plan_camera_path(targets, [], [head] * n, W, H)
    with_extra = plan_camera_path(targets, [], [head] * n, W, H, extra_heads=[other])
    assert without[:, 2].max() < 700.0
    assert with_extra[:, 2].min() >= 820.0 - 2.0


# --------------------------------------------------------------------------- reframe / tracking


def _finding(shot_idx, frame, fps, primary, dup):
    return DuplicateFinding(shotIndex=shot_idx, frame=frame, timeMs=frame / fps * 1000,
                            primary=DetectedInstance(1, "bunny", list(primary), 0.9),
                            duplicate=DetectedInstance(2, "bunny", list(dup), 0.8), similarity=0.93,
                            character="bunny", characterName="Bunny")


def test_sampled_frame_reframe_relaxes_before_and_after_the_findings():
    fps = 24.0
    shot = Shot(0, 0, 239, 0.0, 239 / fps * 1000, 1.0, "transnetv2")
    findings = [_finding(0, f, fps, (300, 200, 600, 1000), (1300, 200, 1600, 1000)) for f in range(96, 145, 6)]
    track = build_reframe_track(shot, findings, 1920, 1080, fps)
    zoom = {k.frame: k.zoom for k in track.keyframes}
    assert zoom[0] < 1.02 and zoom[239] < 1.02       # full frame at both ends
    assert zoom[120] > 1.15                          # cropped while the duplicate is there
    assert all(k.crop[2] <= 1300 + 1 for k in track.keyframes if 96 <= k.frame <= 144)


def _anchor(frame, prim, dup):
    return DuplicateFinding(0, frame, frame / 24.0 * 1000, DetectedInstance(1, "person", list(prim), 0.6),
                            DetectedInstance(2, "person", list(dup), 0.5), 0.95, "bunny", "Bunny")


def _run_backwards(dup_box):
    n = 16
    frames = [(k * 3, np.zeros((360, 640, 3), np.uint8)) for k in range(n)]
    prim = (200, 150, 400, 650)
    # the duplicate is only detected from sample 10 on (e.g. hidden / missed before)
    boxes = [[prim] + ([dup_box] if k >= 10 else []) for k in range(n)]
    it = iter(range(n))
    anchor = _anchor(36, prim, dup_box)  # sample 12
    emb = lambda img, bs: np.tile(np.array([1.0, 0, 0, 0], np.float32), (len(bs), 1))
    return track_pair(frames, [anchor], lambda img: [Detection(tuple(v / 2 for v in b), "person", 0.3)
                                                      for b in boxes[next(it)]], emb, 0.5, 8.0, hold_s=0.25)


def test_unseen_duplicate_is_held_back_to_the_shot_start():
    pt = _run_backwards((800, 150, 1000, 650))  # mid-frame
    assert all(d is not None for d in pt.duplicate), pt.duplicate
    assert pt.dup_held[0] and pt.stats["heldToStart"] > 0


def test_duplicate_that_entered_through_an_edge_is_dropped_before():
    pt = _run_backwards((1080, 150, 1280, 650))  # touching the right edge
    assert pt.duplicate[0] is None and pt.duplicate[11] is not None
