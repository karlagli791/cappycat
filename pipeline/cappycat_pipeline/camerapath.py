"""Cinematic camera-path planning for the reframer.

Instead of smoothing per-frame crops and then pushing each frame off the duplicate (which is
what made the old camera jump and jitter), the whole shot's virtual camera is solved in one
convex quadratic program, in the spirit of the L1-optimal camera paths used by professional
auto-reframing / stabilisation tools:

Variables per frame ``i`` (normalised to the frame): crop centre ``x_i, y_i`` and width ``s_i``
(the height follows the aspect ratio).

Hard constraints (never violated):
    * the crop stays inside the frame, zoom stays within ``[1, max_zoom]``;
    * the crop never touches any visible duplicate: per duplicate the crop stays on one side of
      it (left / right / above / below). The side is chosen once per continuous visibility
      segment by majority vote of the per-frame target crops, so the camera never flips sides.

Soft terms (weighted):
    * keep the primary character's head region in frame (large penalty on any overflow);
    * keep other duplicate pairs' primaries (``extra_heads``) in frame (a lighter version);
    * follow the per-frame target crop (``solve_crop``: rule of thirds, largest feasible crop,
      or the full frame when no duplicate is visible);
    * L1 velocity -> the camera prefers to hold perfectly still (piecewise-static segments);
    * L2 acceleration and L2 jerk -> moves ease in and out, no jitter.

Because the problem is solved over the whole shot at once, the camera anticipates: it starts
easing in before the duplicate walks into frame instead of snapping when it appears.
Solved with Clarabel (interior point), falling back to OSQP.

Infeasible side choice: when a duplicate crosses the primary while staying visible, one side per
visibility segment cannot be satisfied. The plan is then retried with the segments **split where
the geometric side flips** (with :data:`SIDE_HYSTERESIS_S` of hysteresis), and as a last resort with
the exclusion as a heavily weighted soft constraint (the final per-frame clamp still enforces it).
Shots that are feasible as they are get exactly the first plan.

Long shots: the QP grows with the number of frames and is re-solved by the adaptive overshoot
pass, so shots longer than :data:`MAX_QP_FRAMES` frames are planned on an evenly decimated grid with
the objective rescaled to per-frame units, then interpolated back to every frame.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

Box = Sequence[float]

# side codes: the crop stays to the LEFT of the duplicate, RIGHT of it, ABOVE it, BELOW it
LEFT, RIGHT, ABOVE, BELOW = 0, 1, 2, 3
MAX_QP_FRAMES = 1440      # longer shots are planned on a decimated grid (60 s at 24 fps per QP)
SIDE_HYSTERESIS_S = 0.25  # a side flip must persist this long before a visibility segment is split


@dataclass
class CameraStyle:
    """Weights of the camera-path objective (normalised units, per frame at the shot's fps).

    The defaults give a calm, operated feel: holds where possible, moves eased over roughly
    half a second to a second, zoom changes slower than pans."""

    # tuned on the user's clips (clip7 / clip10 duplicate shots): max zoom step 0.37x -> 0.037x per
    # frame, max pan step 108 px -> 14 px, pan acceleration 108 -> 1 px/frame^2, no duplicate
    # exposure, no head cut; the camera still returns to the full frame where no duplicate is visible.
    # APPROVED LOOK (user, 2026-09-25): on a push-in the zoom slightly overshoots and then settles
    # back (clip7: ~1.93x -> ~1.85x over about a second). It reads as a camera that eases in rather than
    # rushing, then holds its composition and eases out. Keep this character when retuning; it is
    # locked by tests/test_camerapath.py::test_push_in_overshoots_then_settles. The user prefers it
    # ADAPTIVE: overshoots inside ``max_overshoot`` are kept as planned, larger ones are tamed per
    # shot (plan_camera_path re-plans with an overshoot penalty).
    follow: float = 0.2            # pull towards the per-frame target crop
    follow_zoom: float = 0.6       # relative weight of the zoom (width) part of ``follow``
    hold: float = 0.05             # L1 velocity: prefer a perfectly still camera
    accel: float = 1200.0          # L2 acceleration: ease in / ease out
    jerk: float = 30000.0          # L2 jerk: remove jitter
    zoom_smooth: float = 2.0       # extra multiplier on the zoom channel's accel / jerk terms
    head: float = 400.0            # quadratic penalty on the head region leaving the crop
    head_linear: float = 4.0       # linear penalty on the same (makes it bind like a hard rule)
    margin: float = 0.004          # extra clearance from a duplicate, fraction of frame width
    max_zoom: float = 3.0
    # Penalty on zooming tighter than the ideal framing (the overshoot on a push-in). 0 = the
    # natural, approved overshoot; the adaptive planner raises it only for shots whose overshoot
    # would exceed ``max_overshoot``.
    overshoot: float = 0.0
    max_overshoot: float = 0.06    # adaptive band: a push-in may overshoot its settled zoom by <= 6 %
    adaptive: bool = True
    extra_head: float = 0.25       # other duplicate pairs' primaries: this fraction of ``head`` / ``head_linear``
    soft_exclusion: float = 5000.0  # last-resort plan: quadratic cost of a duplicate-exclusion violation


def _diff(n: int, order: int):
    import scipy.sparse as sp

    d = sp.identity(n, format="csr")
    for _ in range(order):
        m = d.shape[0]
        d = (sp.eye(m - 1, m, k=1) - sp.eye(m - 1, m)) @ d
    return d.tocsr()


def _box_side(crop: Box, dup: Box) -> Optional[int]:
    """Which side of ``dup`` the crop is on (None if they intersect)."""
    slack = {
        LEFT: dup[0] - crop[2],
        RIGHT: crop[0] - dup[2],
        ABOVE: dup[1] - crop[3],
        BELOW: crop[1] - dup[3],
    }
    side, val = max(slack.items(), key=lambda kv: kv[1])
    return side if val >= -1.0 else None


def _segments(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    segs, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            segs.append((start, i))
            start = None
    return segs


def _geometric_side(target: Box, dup: Box, prim: Optional[Box]) -> int:
    """The side the frame's target crop is on, or (target intersecting the duplicate) the side of
    the primary relative to the duplicate."""
    s = _box_side(target, dup)
    if s is not None:
        return s
    ref = prim if prim is not None else target
    pcx, dcx = (ref[0] + ref[2]) / 2, (dup[0] + dup[2]) / 2
    return LEFT if pcx < dcx else RIGHT


def split_by_side(sides: Sequence[int], a: int, b: int, hold: int) -> List[Tuple[int, int, int]]:
    """Runs ``(start, end, side)`` covering ``sides[a:b]``; a flip only starts a new run when the
    new side persists for ``hold`` frames (hysteresis against jitter)."""
    hold = max(1, int(hold))
    head = list(sides[a:min(b, a + hold)])
    cur = max(set(head), key=head.count)
    runs: List[Tuple[int, int, int]] = []
    start = a
    for i in range(a, b):
        if sides[i] != cur and b - i >= hold and all(v == sides[i] for v in sides[i:i + hold]):
            if i > start:
                runs.append((start, i, cur))
            cur, start = sides[i], i
    runs.append((start, b, cur))
    return runs


def choose_sides(targets: np.ndarray, dup_tracks: Sequence[Sequence[Optional[Box]]],
                 primary: Sequence[Optional[Box]], split: bool = False, hold: int = 6) -> List[List[Optional[int]]]:
    """One side per duplicate per continuous visibility segment (majority vote of the targets;
    ties / intersecting targets fall back to the side of the duplicate relative to the primary).
    ``split``: a segment is split where the per-frame geometric side flips for at least ``hold``
    frames (the duplicate crossed the primary); each part keeps its own side."""
    n = len(targets)
    out: List[List[Optional[int]]] = []
    for track in dup_tracks:
        sides: List[Optional[int]] = [None] * n
        for a, b in _segments([d is not None for d in track]):
            if split:
                geo = [LEFT] * n
                for i in range(a, b):
                    geo[i] = _geometric_side(targets[i], track[i], primary[i] if i < len(primary) else None)
                for s0, s1, side in split_by_side(geo, a, b, hold):
                    for i in range(s0, s1):
                        sides[i] = side
                continue
            votes = np.zeros(4)
            for i in range(a, b):
                s = _box_side(targets[i], track[i])
                if s is not None:
                    votes[s] += 1
            if votes.sum() == 0:
                # use the primary's position relative to the duplicate
                for i in range(a, b):
                    p = primary[i]
                    if p is None:
                        continue
                    d = track[i]
                    pcx, dcx = (p[0] + p[2]) / 2, (d[0] + d[2]) / 2
                    votes[LEFT if pcx < dcx else RIGHT] += 1
            side = int(np.argmax(votes)) if votes.sum() > 0 else LEFT
            for i in range(a, b):
                sides[i] = side
        out.append(sides)
    return out


def zoom_overshoot(crops: np.ndarray, frame_w: float, fps: float = 24.0) -> float:
    """Largest push-in overshoot of a path: for every zoom peak reached by a rise of >= 0.08x,
    (peak - settled) / settled, where settled is the lowest zoom in the ~1.5 s after the peak."""
    crops = np.asarray(crops, dtype=np.float64)
    if len(crops) < 3:
        return 0.0
    z = frame_w / (crops[:, 2] - crops[:, 0])
    win = max(3, int(round(1.5 * fps)))
    worst = 0.0
    for i in range(1, len(z) - 1):
        if not (z[i] >= z[i - 1] and z[i] > z[i + 1]):
            continue
        before = z[max(0, i - win):i]
        if len(before) == 0 or z[i] - before.min() < 0.08:
            continue
        after = z[i + 1:i + 1 + win]
        settled = float(after.min())
        worst = max(worst, (z[i] - settled) / settled)
    return worst


def _union(boxes: Sequence[Optional[Box]]) -> Optional[Box]:
    bs = [b for b in boxes if b is not None]
    if not bs:
        return None
    a = np.asarray(bs, dtype=np.float64)
    return (float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 2].max()), float(a[:, 3].max()))


def decimated_style(style: CameraStyle, k: int) -> CameraStyle:
    """The same objective on a grid of every ``k``-th frame: per-frame sums scale by ``k``, second /
    third differences over ``k`` frames by ``k**2`` / ``k**3`` (so their squared sums by ``k**3`` /
    ``k**5`` after the ``1/k`` fewer terms); the L1 path length is unchanged."""
    return dataclasses.replace(style, follow=style.follow * k, accel=style.accel / k ** 3, jerk=style.jerk / k ** 5,
                               head=style.head * k, head_linear=style.head_linear * k, overshoot=style.overshoot * k,
                               soft_exclusion=style.soft_exclusion * k)


def plan_camera_path(targets: np.ndarray, dup_tracks: Sequence[Sequence[Optional[Box]]],
                     heads: Sequence[Optional[Box]], frame_w: float, frame_h: float,
                     aspect: Optional[float] = None, style: Optional[CameraStyle] = None,
                     primary: Optional[Sequence[Optional[Box]]] = None, fps: float = 24.0,
                     extra_heads: Optional[Sequence[Sequence[Optional[Box]]]] = None,
                     max_frames: int = MAX_QP_FRAMES, info: Optional[dict] = None) -> Optional[np.ndarray]:
    """Solve the camera path. ``targets`` (N, 4) per-frame target crops in source px;
    ``dup_tracks`` one list of N optional duplicate boxes per duplicate; ``heads`` N optional head
    regions of the primary; ``extra_heads``: lists of N optional head regions of other primaries
    (a lighter head term). Returns (N, 4) crops in source px, or None if the solver failed.
    ``info`` (optional dict) receives ``mode`` ("sides" / "split-sides" / "soft-exclusion") and
    ``step`` (decimation of the QP grid, 1 = every frame).

    Adaptive overshoot: a push-in that overshoots its settled zoom by at most
    ``style.max_overshoot`` is kept exactly as planned (the approved "arrive close, settle back"
    feel). Only when a shot's overshoot is larger (e.g. a duplicate that appears abruptly at full
    size) is it re-planned with a growing penalty on zooming tighter than the ideal framing, which
    makes the pan lead earlier instead, until the overshoot is back inside the band."""
    style = style or CameraStyle()
    targets = np.asarray(targets, dtype=np.float64)
    n = len(targets)
    info = info if info is not None else {}
    prim = list(primary) if primary is not None else [None] * n
    extra = [list(e) for e in (extra_heads or []) if any(b is not None for b in e)]
    step = max(1, int(math.ceil(n / float(max_frames)))) if max_frames and n > max_frames else 1
    info["step"] = step
    if step == 1:
        return _plan_modes(targets, dup_tracks, heads, frame_w, frame_h, aspect, style, prim, fps, extra, info)
    # plan every ``step``-th frame; each sample excludes the duplicates of both adjacent intervals,
    # so the (linearly interpolated) crops in between stay clear of them too
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    span = [(idx[max(0, j - 1)], idx[min(len(idx) - 1, j + 1)]) for j in range(len(idx))]
    d_tracks = [[_union(list(tr[a:b + 1])) for a, b in span] for tr in dup_tracks]
    sub = _plan_modes(targets[idx], d_tracks, [heads[i] for i in idx], frame_w, frame_h, aspect,
                      decimated_style(style, step), [prim[i] for i in idx], fps / step,
                      [[e[i] for i in idx] for e in extra], info)
    if sub is None:
        return None
    frames = np.arange(n)
    cx = np.interp(frames, idx, (sub[:, 0] + sub[:, 2]) / 2)
    cy = np.interp(frames, idx, (sub[:, 1] + sub[:, 3]) / 2)
    w = np.interp(frames, idx, sub[:, 2] - sub[:, 0])
    a = float(aspect) if aspect else float(frame_w) / float(frame_h)
    h = w / a
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def _plan_modes(targets, dup_tracks, heads, frame_w, frame_h, aspect, style: CameraStyle, primary, fps, extra,
                info: dict) -> Optional[np.ndarray]:
    """First feasible of: one side per visibility segment (the approved plan) -> segments split where
    the side flips -> soft exclusion; then the adaptive overshoot pass in that mode."""
    hold = max(1, int(round(SIDE_HYSTERESIS_S * fps)))
    modes = (("sides", False, 0.0), ("split-sides", True, 0.0), ("soft-exclusion", True, style.soft_exclusion))
    crops, split, soft = None, False, 0.0
    for name, split, soft in modes:
        crops = _plan(targets, dup_tracks, heads, frame_w, frame_h, aspect, style, primary, extra, split, hold, soft)
        if crops is not None:
            info["mode"] = name
            if name != "sides":
                log.info("camera path: one side per visibility segment is infeasible; planned with %s", name)
            break
        if not dup_tracks:
            break
    if crops is None or not style.adaptive:
        return crops
    over = zoom_overshoot(crops, frame_w, fps)
    if over <= style.max_overshoot:
        return crops
    # re-plan with a growing overshoot penalty; accept a candidate only if it is still smooth
    # (no zoom step larger than the unpenalised path's), and keep the best one found
    base_step = float(np.abs(np.diff(frame_w / (crops[:, 2] - crops[:, 0]))).max()) if len(crops) > 1 else 0.0
    best, best_over = crops, over
    weight = style.overshoot or 0.5
    for _ in range(8):
        weight *= 2.0
        retry = _plan(targets, dup_tracks, heads, frame_w, frame_h, aspect,
                      dataclasses.replace(style, overshoot=weight), primary, extra, split, hold, soft)
        if retry is None:
            break
        step = float(np.abs(np.diff(frame_w / (retry[:, 2] - retry[:, 0]))).max()) if len(retry) > 1 else 0.0
        if step > base_step * 1.25 + 1e-3:
            break  # the penalty is starting to fight smoothness: stop here
        new_over = zoom_overshoot(retry, frame_w, fps)
        log.info("camera path: overshoot %.1f%% -> %.1f%% (penalty %.1f)", over * 100, new_over * 100, weight)
        if new_over < best_over:
            best, best_over = retry, new_over
        if new_over <= style.max_overshoot:
            break
    return best


def _plan(targets: np.ndarray, dup_tracks: Sequence[Sequence[Optional[Box]]],
          heads: Sequence[Optional[Box]], frame_w: float, frame_h: float,
          aspect: Optional[float], style: CameraStyle,
          primary: Optional[Sequence[Optional[Box]]], extra: Optional[Sequence[Sequence[Optional[Box]]]] = None,
          split: bool = False, hold: int = 6, soft: float = 0.0) -> Optional[np.ndarray]:
    """One QP solve for a fixed :class:`CameraStyle`. ``split``: sides per split segment (see
    :func:`choose_sides`); ``soft`` > 0: exclusion violations cost ``soft * slack**2`` instead of
    being infeasible. Without extras / split / soft this is exactly the approved problem."""
    import scipy.sparse as sp

    style = style or CameraStyle()
    targets = np.asarray(targets, dtype=np.float64)
    n = len(targets)
    if n == 0:
        return targets
    W, H = float(frame_w), float(frame_h)
    aspect = float(aspect) if aspect else W / H
    r = W / (aspect * H)  # crop height / frame height per unit of crop width / frame width
    s_max = min(1.0, 1.0 / r)
    s_min = max(0.05, s_max / style.max_zoom)

    # duplicate exclusion constraints, one consistent side per (split) visibility segment
    prim = primary if primary is not None else [None] * n
    sides = choose_sides(targets, dup_tracks, prim, split=split, hold=hold)
    m = style.margin
    excl: List[Tuple[int, int, Box]] = []  # (frame, side, duplicate box)
    for track, tsides in zip(dup_tracks, sides):
        for i in range(n):
            d = track[i]
            side = tsides[i]
            if d is None or side is None:
                continue
            excl.append((i, side, d))
    extra = [e for e in (extra or []) if any(b is not None for b in e)]

    # --- variables: [x (n), y (n), s (n), head slack (4n), |dx| (n-1), |dy| (n-1), |ds| (n-1),
    #                 overshoot slack (n): how much tighter than the target zoom the crop is,
    #                 extra-head slack (4n per extra head), exclusion slack (soft mode only)]
    nx = 3 * n
    nh = 4 * n
    nv = max(0, n - 1)
    ne = 4 * n * len(extra)
    nsl = len(excl) if soft > 0 else 0
    nvar = nx + nh + 3 * nv + n + ne + nsl
    X, Y, S = 0, n, 2 * n
    HS = nx
    V = nx + nh
    OV = nx + nh + 3 * nv
    EH = OV + n
    SL = EH + ne

    tx = (targets[:, 0] + targets[:, 2]) / 2 / W
    ty = (targets[:, 1] + targets[:, 3]) / 2 / H
    ts = (targets[:, 2] - targets[:, 0]) / W

    D1, D2, D3 = _diff(n, 1), _diff(n, 2), _diff(n, 3)
    smooth = style.accel * (D2.T @ D2) + style.jerk * (D3.T @ D3) if n >= 4 else style.accel * (D2.T @ D2) if n >= 3 else sp.csr_matrix((n, n))
    I = sp.identity(n, format="csr")
    Pxy = style.follow * I + smooth
    Ps = style.follow * style.follow_zoom * I + style.zoom_smooth * smooth
    blocks = [Pxy, Pxy, Ps, style.head * sp.identity(nh), sp.csr_matrix((3 * nv, 3 * nv)),
              style.overshoot * sp.identity(n)]
    if ne:
        blocks.append(style.extra_head * style.head * sp.identity(ne))
    if nsl:
        blocks.append(soft * sp.identity(nsl))
    P = sp.block_diag(blocks, format="csc") * 2.0
    q = np.concatenate([
        -2 * style.follow * tx,
        -2 * style.follow * ty,
        -2 * style.follow * style.follow_zoom * ts,
        np.full(nh, style.head_linear),
        np.full(3 * nv, style.hold),
        np.full(n, 0.1 * style.overshoot),
        np.full(ne, style.extra_head * style.head_linear),
        np.full(nsl, 0.1 * soft),
    ])

    rows, lo, hi = [], [], []

    def add(coefs: Sequence[Tuple[int, float]], lower: float, upper: float) -> None:
        rows.append(coefs)
        lo.append(lower)
        hi.append(upper)

    inf = np.inf
    for i in range(n):
        # frame bounds and zoom range
        add([(X + i, 1.0), (S + i, -0.5)], 0.0, inf)          # left edge >= 0
        add([(X + i, 1.0), (S + i, 0.5)], -inf, 1.0)          # right edge <= 1
        add([(Y + i, 1.0), (S + i, -0.5 * r)], 0.0, inf)      # top edge >= 0
        add([(Y + i, 1.0), (S + i, 0.5 * r)], -inf, 1.0)      # bottom edge <= 1
        add([(S + i, 1.0)], s_min, s_max)
        # head region (soft): crop.x1 <= hx1 + slack, crop.x2 >= hx2 - slack, ...
        h = heads[i]
        for k in range(4):
            add([(HS + 4 * i + k, 1.0)], 0.0, inf)
        if h is not None:
            hx1, hy1, hx2, hy2 = h[0] / W, h[1] / H, h[2] / W, h[3] / H
            add([(X + i, 1.0), (S + i, -0.5), (HS + 4 * i, -1.0)], -inf, hx1)
            add([(X + i, 1.0), (S + i, 0.5), (HS + 4 * i + 1, 1.0)], hx2, inf)
            add([(Y + i, 1.0), (S + i, -0.5 * r), (HS + 4 * i + 2, -1.0)], -inf, hy1)
            add([(Y + i, 1.0), (S + i, 0.5 * r), (HS + 4 * i + 3, 1.0)], hy2, inf)
    # overshoot slack: o_i >= ts_i - s_i (>= 0), only while a duplicate is visible (the settled,
    # constrained part of a push-in). Before the duplicate enters, easing in early is exactly what
    # we want, so zooming tighter than the (full-frame) target there stays free.
    visible = [any(track[i] is not None for track in dup_tracks) for i in range(n)]
    for i in range(n):
        add([(OV + i, 1.0)], 0.0, inf)
        if visible[i]:
            add([(S + i, 1.0), (OV + i, 1.0)], float(ts[i]), inf)
    # L1 velocity auxiliaries: |d| <= t
    for base, a in ((X, 0), (Y, 1), (S, 2)):
        for i in range(nv):
            t = V + a * nv + i
            add([(base + i + 1, 1.0), (base + i, -1.0), (t, -1.0)], -inf, 0.0)
            add([(base + i + 1, -1.0), (base + i, 1.0), (t, -1.0)], -inf, 0.0)

    # duplicate exclusion: hard, or (soft mode) relaxed by a heavily weighted slack per constraint
    for k, (i, side, d) in enumerate(excl):
        dx1, dy1, dx2, dy2 = d[0] / W, d[1] / H, d[2] / W, d[3] / H
        up = [(SL + k, -1.0)] if nsl else []   # for "<= bound" rows
        dn = [(SL + k, 1.0)] if nsl else []    # for ">= bound" rows
        if side == LEFT:     # crop right edge <= dup left
            add([(X + i, 1.0), (S + i, 0.5)] + up, -inf, dx1 - m)
        elif side == RIGHT:  # crop left edge >= dup right
            add([(X + i, 1.0), (S + i, -0.5)] + dn, dx2 + m, inf)
        elif side == ABOVE:  # crop bottom <= dup top
            add([(Y + i, 1.0), (S + i, 0.5 * r)] + up, -inf, dy1 - m)
        else:                # crop top >= dup bottom
            add([(Y + i, 1.0), (S + i, -0.5 * r)] + dn, dy2 + m, inf)
    for k in range(nsl):
        add([(SL + k, 1.0)], 0.0, inf)
    # other duplicate pairs' primaries (soft, lighter): the head rows with their own slack variables
    for e_i, ehs in enumerate(extra):
        base = EH + 4 * n * e_i
        for i in range(n):
            for k in range(4):
                add([(base + 4 * i + k, 1.0)], 0.0, inf)
            h = ehs[i]
            if h is None:
                continue
            hx1, hy1, hx2, hy2 = h[0] / W, h[1] / H, h[2] / W, h[3] / H
            add([(X + i, 1.0), (S + i, -0.5), (base + 4 * i, -1.0)], -inf, hx1)
            add([(X + i, 1.0), (S + i, 0.5), (base + 4 * i + 1, 1.0)], hx2, inf)
            add([(Y + i, 1.0), (S + i, -0.5 * r), (base + 4 * i + 2, -1.0)], -inf, hy1)
            add([(Y + i, 1.0), (S + i, 0.5 * r), (base + 4 * i + 3, 1.0)], hy2, inf)

    data, ri, ci = [], [], []
    for k, coefs in enumerate(rows):
        for c, v in coefs:
            ri.append(k)
            ci.append(c)
            data.append(v)
    A = sp.csc_matrix((data, (ri, ci)), shape=(len(rows), nvar))
    sol = _solve_qp(P, q, A, np.array(lo), np.array(hi))
    if sol is None:
        return None
    x, y, s = sol[X:X + n], sol[Y:Y + n], sol[S:S + n]
    w = s * W
    h = w / aspect
    cx, cy = x * W, y * H
    crops = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    # numerical tolerance: keep inside the frame exactly
    shift_x = np.clip(-crops[:, 0], 0, None) - np.clip(crops[:, 2] - W, 0, None)
    shift_y = np.clip(-crops[:, 1], 0, None) - np.clip(crops[:, 3] - H, 0, None)
    crops[:, [0, 2]] += shift_x[:, None]
    crops[:, [1, 3]] += shift_y[:, None]
    return crops


def _solve_qp(P, q, A, lo: np.ndarray, hi: np.ndarray) -> Optional[np.ndarray]:
    """min 1/2 x'Px + q'x  s.t.  lo <= Ax <= hi.

    Clarabel (interior point) first: the jerk / acceleration terms make the problem badly
    conditioned, which first-order ADMM solvers such as OSQP struggle to converge on. OSQP is the
    fallback when Clarabel is not installed."""
    import scipy.sparse as sp

    try:
        import clarabel

        up = np.isfinite(hi)
        dn = np.isfinite(lo)
        A_ineq = sp.vstack([A[up], -A[dn]], format="csc")
        b_ineq = np.concatenate([hi[up], -lo[dn]])
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = 400
        solver = clarabel.DefaultSolver(sp.triu(P, format="csc"), q, A_ineq, b_ineq,
                                        [clarabel.NonnegativeConeT(A_ineq.shape[0])], settings)
        res = solver.solve()
        status = str(res.status)
        if "Solved" in status and "Almost" not in status or "AlmostSolved" in status:
            return np.asarray(res.x)
        if "Infeasible" in status:
            log.info("camera path: clarabel status %s", status)
            return None  # OSQP cannot do better on an infeasible problem (the caller relaxes it)
        log.warning("camera path: clarabel status %s", status)
    except ImportError:
        pass
    try:
        import osqp

        solver = osqp.OSQP()
        solver.setup(P=P.tocsc(), q=q, A=A.tocsc(), l=lo, u=hi, verbose=False, eps_abs=1e-5, eps_rel=1e-5,
                     max_iter=200000, polish=True, adaptive_rho=True)
        res = solver.solve()
        status = str(getattr(res.info, "status", "")).lower()
        if res.x is not None and "solved" in status and not np.any(np.isnan(res.x)):
            return np.asarray(res.x)
        log.warning("camera path: osqp status %s", status)
    except ImportError:
        log.warning("camera path: no QP solver installed (pip install clarabel)")
    return None


def path_metrics(crops: np.ndarray, frame_w: float) -> dict:
    """Smoothness diagnostics: per-frame zoom / pan steps and direction reversals."""
    crops = np.asarray(crops, dtype=np.float64)
    w = crops[:, 2] - crops[:, 0]
    z = frame_w / w
    cx = (crops[:, 0] + crops[:, 2]) / 2
    cy = (crops[:, 1] + crops[:, 3]) / 2

    def reversals(v: np.ndarray, eps: float) -> int:
        d = np.diff(v)
        d = d[np.abs(d) > eps]
        return int(np.sum(np.sign(d[1:]) != np.sign(d[:-1]))) if len(d) > 1 else 0

    return {
        "zoom": (round(float(z.min()), 3), round(float(z.max()), 3)),
        "maxZoomStep": round(float(np.abs(np.diff(z)).max()) if len(z) > 1 else 0.0, 4),
        "maxPanStepPx": round(float(np.abs(np.diff(cx)).max()) if len(cx) > 1 else 0.0, 2),
        "maxTiltStepPx": round(float(np.abs(np.diff(cy)).max()) if len(cy) > 1 else 0.0, 2),
        "maxPanAccelPx": round(float(np.abs(np.diff(cx, 2)).max()) if len(cx) > 2 else 0.0, 2),
        "zoomReversals": reversals(z, 2e-3),
        "panReversals": reversals(cx, 0.5),
    }
