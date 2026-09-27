import numpy as np
import pytest

pytest.importorskip("clarabel")

from cappycat_pipeline.camerapath import CameraStyle, path_metrics, plan_camera_path, zoom_overshoot

W, H = 1280.0, 720.0


def _intersects(c, d):
    return c[0] < d[2] - 0.5 and c[2] > d[0] + 0.5 and c[1] < d[3] - 0.5 and c[3] > d[1] + 0.5


def scenario(n=120, enter=40):
    """A primary on the left; its duplicate walks in from the right edge at frame ``enter`` and
    drifts left. Targets: full frame before, a tight crop after (like solve_crop gives)."""
    full = np.array([0, 0, W, H])
    targets, dups, heads = [], [], []
    for i in range(n):
        heads.append((200.0, 60.0, 420.0, 260.0))
        if i < enter:
            targets.append(full)
            dups.append(None)
        else:
            x1 = 1000.0 - 3.0 * (i - enter)  # moves left over time
            dups.append((x1, 100.0, x1 + 250.0, 720.0))
            w = x1 - 10
            targets.append(np.array([0, 0, w, w * H / W]))
    return np.array(targets, dtype=float), [dups], heads


def test_never_touches_duplicate_and_keeps_head():
    targets, dups, heads = scenario()
    crops = plan_camera_path(targets, dups, heads, W, H)
    assert crops is not None
    for i, c in enumerate(crops):
        assert c[0] >= -0.5 and c[1] >= -0.5 and c[2] <= W + 0.5 and c[3] <= H + 0.5
        assert abs((c[2] - c[0]) / (c[3] - c[1]) - W / H) < 1e-3
        if dups[0][i] is not None:
            assert not _intersects(c, dups[0][i]), f"frame {i} crop {c} touches duplicate"
        h = heads[i]
        assert c[0] <= h[0] + 1 and c[1] <= h[1] + 1 and c[2] >= h[2] - 1 and c[3] >= h[3] - 1


def test_smooth_and_anticipates_the_duplicate():
    targets, dups, heads = scenario(enter=40)
    crops = plan_camera_path(targets, dups, heads, W, H)
    m = path_metrics(crops, W)
    # a hard switch from full frame to ~1.4x at frame 40 must become an eased move
    assert m["maxZoomStep"] < 0.05, m
    assert m["maxPanAccelPx"] < 3.0, m
    z = W / (crops[:, 2] - crops[:, 0])
    assert z[30] > z[0] + 0.02, "camera should start easing in before the duplicate enters"


def test_holds_still_when_nothing_changes():
    n = 60
    targets = np.tile(np.array([100.0, 50.0, 100.0 + 960, 50.0 + 540]), (n, 1))
    crops = plan_camera_path(targets, [], [None] * n, W, H)
    assert np.abs(np.diff(crops, axis=0)).max() < 0.5
    assert np.abs(crops[0] - targets[0]).max() < 40  # close to the requested framing


def test_consistent_side_per_segment():
    # duplicate sits right of the primary for the whole shot; the camera must never jump across it
    n = 50
    targets = np.tile(np.array([0.0, 0.0, 700.0, 393.75]), (n, 1))
    dups = [[(760.0, 50.0, 1000.0, 700.0)] * n]
    crops = plan_camera_path(targets, dups, [(300.0, 40.0, 500.0, 240.0)] * n, W, H)
    assert np.all(crops[:, 2] <= 760.0 + 0.5)


def test_custom_style_is_used():
    targets, dups, heads = scenario()
    stiff = plan_camera_path(targets, dups, heads, W, H, style=CameraStyle(accel=10.0, jerk=50.0, follow=2.0))
    calm = plan_camera_path(targets, dups, heads, W, H)
    assert path_metrics(calm, W)["maxZoomStep"] <= path_metrics(stiff, W)["maxZoomStep"]


def test_push_in_overshoots_then_settles():
    """Approved camera feel: a push-in slightly overshoots the zoom it will hold, then settles back
    gently (no rush, no hard stop). Guards the default CameraStyle against retuning that removes it."""
    n, enter = 150, 30
    full = np.array([0.0, 0.0, W, H])
    hold_w = 700.0
    targets = [full if i < enter else np.array([0.0, 0.0, hold_w, hold_w * H / W]) for i in range(n)]
    dups = [[None if i < enter else (hold_w + 20.0, 80.0, hold_w + 300.0, 720.0) for i in range(n)]]
    heads = [(150.0, 60.0, 400.0, 240.0)] * n
    crops = plan_camera_path(np.array(targets), dups, heads, W, H)
    z = W / (crops[:, 2] - crops[:, 0])
    peak_i = int(np.argmax(z))
    settled = float(np.median(z[-20:]))
    assert z[peak_i] > settled + 0.01, "expected a gentle overshoot before settling"
    # this synthetic duplicate appears abruptly at full size, the harshest case (12.5 % unplanned);
    # the adaptive planner brings it inside the approved band (real clip7 push-in: ~4 %)
    assert z[peak_i] < settled * 1.07, "overshoot should stay subtle"
    assert np.abs(np.diff(z[peak_i:])).max() < 0.01, "settling must be gentle"
    assert np.abs(np.diff(z)).max() < 0.06, "no rushed zoom"


def _push_in(hold_w, n=150, enter=30):
    full = np.array([0.0, 0.0, W, H])
    targets = np.array([full if i < enter else np.array([0.0, 0.0, hold_w, hold_w * H / W]) for i in range(n)])
    dups = [[None if i < enter else (hold_w + 20.0, 80.0, hold_w + 300.0, 720.0) for i in range(n)]]
    return targets, dups, [(150.0, 60.0, 400.0, 240.0)] * n


def test_adaptive_overshoot_only_tames_large_overshoots():
    style = CameraStyle()
    # abrupt, large push-in: the fixed planner overshoots well past the band, the adaptive one
    # brings it inside while staying just as smooth
    t, d, h = _push_in(700.0)
    fixed = plan_camera_path(t, d, h, W, H, style=CameraStyle(adaptive=False))
    adaptive = plan_camera_path(t, d, h, W, H, style=style)
    assert zoom_overshoot(fixed, W) > style.max_overshoot
    assert 0.01 < zoom_overshoot(adaptive, W) <= style.max_overshoot + 1e-3
    assert path_metrics(adaptive, W)["maxZoomStep"] <= path_metrics(fixed, W)["maxZoomStep"] * 1.25 + 1e-3
    for i in range(len(adaptive)):
        if d[0][i] is not None:
            assert not _intersects(adaptive[i], d[0][i])
    # gentle push-in already inside the band: left exactly as planned
    t, d, h = _push_in(900.0)
    fixed = plan_camera_path(t, d, h, W, H, style=CameraStyle(adaptive=False))
    adaptive = plan_camera_path(t, d, h, W, H, style=style)
    assert zoom_overshoot(fixed, W) <= style.max_overshoot
    assert np.allclose(fixed, adaptive)
