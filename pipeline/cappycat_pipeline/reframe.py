"""Rule-of-thirds constrained crop solver + camera-path smoothing (pure numpy / scipy).

``solve_crop`` finds the largest crop with the source aspect ratio that

    (a) fully excludes every duplicate bbox,
    (b) fully contains the primary bbox,
    (c) puts the primary's centre as close as possible to one of the rule-of-thirds
        vertical lines (x = 1/3 or 2/3 of the crop width),

by scanning candidate crop widths coarse-to-fine together with x / y offsets, ranking by
feasibility, then area, then thirds alignment. If nothing contains the whole primary the
containment constraint is relaxed head-first, because a character's face matters more than
their legs: first the upper body (top 55 % of the box), then the head (top 30 %), each placed
with a little headroom above the character rather than centred on the body. Only then does it
fall back to the primary's centre point, and finally to a best-effort crop that only excludes
the duplicates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

import logging

from .schema import DuplicateFinding, ReframeKeyframe, ReframeTrack, Shot

BBoxT = Tuple[float, float, float, float]
log = logging.getLogger(__name__)


def _as_box(b: Sequence[float]) -> np.ndarray:
    a = np.asarray(b, dtype=np.float64).reshape(4)
    return np.array([min(a[0], a[2]), min(a[1], a[3]), max(a[0], a[2]), max(a[1], a[3])])


def _shrink(box: np.ndarray, factor: float) -> np.ndarray:
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    hw, hh = (box[2] - box[0]) / 2.0 * factor, (box[3] - box[1]) / 2.0 * factor
    return np.array([cx - hw, cy - hh, cx + hw, cy + hh])


_EPS = 1e-6


def _feasible_x_intervals(w: float, h: float, y: float, frame_w: float, contain: Optional[np.ndarray],
                          dups: np.ndarray) -> List[Tuple[float, float]]:
    """Closed x-intervals where a (w x h) crop at row ``y`` contains ``contain`` (if given) and
    overlaps no duplicate. Exact 1-D computation."""
    lo, hi = 0.0, frame_w - w
    if contain is not None:
        if y > contain[1] + _EPS or y + h < contain[3] - _EPS:
            return []
        lo, hi = max(lo, contain[2] - w), min(hi, contain[0])
    if hi < lo - _EPS:
        return []
    intervals = [(lo, max(lo, hi))]
    for d in dups:
        if not (y < d[3] - _EPS and y + h > d[1] + _EPS):
            continue  # no vertical overlap -> no horizontal constraint
        fa, fb = d[0] - w, d[2]  # forbidden open interval (crop would overlap the duplicate)
        nxt: List[Tuple[float, float]] = []
        for a, b in intervals:
            if fa >= a - _EPS:
                nxt.append((a, min(b, fa)))
            if fb <= b + _EPS:
                nxt.append((max(a, fb), b))
        intervals = [(a, b) for a, b in nxt if b >= a - _EPS]
        if not intervals:
            return []
    return intervals


HEADROOM = 0.06  # fraction of the crop height kept above the character's head when anchoring to it


def _critical_ys(h: float, frame_h: float, contain: Optional[np.ndarray], dups: np.ndarray, pcy: float,
                 y_pref: Optional[float] = None) -> List[float]:
    ys = {0.0, frame_h - h, pcy - h / 2.0}
    if y_pref is not None:
        ys.add(y_pref)
    if contain is not None:
        ys |= {float(contain[1]), float(contain[3]) - h}
    for d in dups:
        ys |= {float(d[3]), float(d[1]) - h}
    top = max(frame_h - h, 0.0)
    return sorted({min(max(v, 0.0), top) for v in ys})


def _best_at_width(w: float, aspect: float, frame_w: float, frame_h: float, contain: Optional[np.ndarray],
                   dups: np.ndarray, pcx: float, pcy: float, centre_mode: bool,
                   head_top: Optional[float] = None) -> Optional[Tuple[float, float, float]]:
    """Best (x, y, score) for a crop of width ``w`` or None when infeasible. Feasible regions are
    unions of axis-aligned rectangles, so it suffices to test the critical y values and, per
    row, the thirds targets clamped into each feasible x interval."""
    h = w / aspect
    if w > frame_w + _EPS or h > frame_h + _EPS:
        return None
    targets = [pcx - w / 2.0] if centre_mode else [pcx - w / 3.0, pcx - 2.0 * w / 3.0]
    best: Optional[Tuple[float, float, float]] = None
    # head-anchored stages prefer the crop top just above the character's head
    y_pref = None if head_top is None else head_top - HEADROOM * h
    for y in _critical_ys(h, frame_h, contain, dups, pcy, y_pref):
        if y_pref is not None:
            vertical = 10.0 * abs(y - min(max(y_pref, 0.0), max(frame_h - h, 0.0))) / h
        else:
            vertical = abs(pcy - (y + h / 2.0)) / h
        for a, b in _feasible_x_intervals(w, h, y, frame_w, contain, dups):
            for t in targets:
                x = min(max(t, a), b)
                if centre_mode:
                    score = abs(pcx - (x + w / 2.0)) / w + vertical
                else:
                    score = min(abs(pcx - (x + w / 3.0)), abs(pcx - (x + 2.0 * w / 3.0))) / w + 0.05 * vertical
                if best is None or score < best[2]:
                    best = (float(x), float(y), float(score))
    return best


def _search_stage(frame_w: float, frame_h: float, aspect: float, contain: Optional[np.ndarray], dups: np.ndarray,
                  pcx: float, pcy: float, w_min: float, w_max: float, centre_mode: bool = False,
                  head_top: Optional[float] = None) -> Optional[Tuple[float, float, float]]:
    """Largest feasible width (feasibility is monotone in width, so binary search is exact),
    then the best-aligned placement at that width. Returns (x, y, w)."""
    if w_max < w_min - _EPS:
        return None

    def at(w: float):
        return _best_at_width(w, aspect, frame_w, frame_h, contain, dups, pcx, pcy, centre_mode, head_top)

    top = at(w_max)
    if top is not None:
        return top[0], top[1], float(w_max)
    if at(w_min) is None:
        return None
    lo, hi = float(w_min), float(w_max)  # lo feasible, hi infeasible
    for _ in range(48):
        mid = (lo + hi) / 2.0
        if at(mid) is not None:
            lo = mid
        else:
            hi = mid
    res = at(lo)
    assert res is not None
    return res[0], res[1], lo


def solve_crop(frame_w: float, frame_h: float, primary_bbox: Sequence[float], duplicate_bboxes: Iterable[Sequence[float]],
               aspect: Optional[float] = None, min_zoom_width_frac: float = 0.25) -> BBoxT:
    """Return the crop ``(x1, y1, x2, y2)`` in source pixels (see module docstring)."""
    frame_w, frame_h = float(frame_w), float(frame_h)
    aspect = float(aspect) if aspect else frame_w / frame_h
    primary = _as_box(primary_bbox)
    primary = np.array([max(0.0, primary[0]), max(0.0, primary[1]), min(frame_w, primary[2]), min(frame_h, primary[3])])
    dups = np.array([_as_box(d) for d in duplicate_bboxes], dtype=np.float64).reshape(-1, 4)
    dups = dups[(dups[:, 2] > dups[:, 0]) & (dups[:, 3] > dups[:, 1])] if len(dups) else dups

    w_max = min(frame_w, frame_h * aspect)
    if len(dups) == 0 and (primary[2] - primary[0]) <= w_max and (primary[3] - primary[1]) <= w_max / aspect:
        # no duplicates: full frame is the largest crop containing the primary
        h = w_max / aspect
        return (0.0, 0.0, w_max, h) if w_max >= frame_w else ((frame_w - w_max) / 2, 0.0, (frame_w + w_max) / 2, h)

    w_floor = max(8.0, frame_w * min_zoom_width_frac)
    pcx, pcy = (primary[0] + primary[2]) / 2.0, (primary[1] + primary[3]) / 2.0
    pw, ph = primary[2] - primary[0], primary[3] - primary[1]
    upper = np.array([primary[0] + 0.15 * pw, primary[1], primary[2] - 0.15 * pw, primary[1] + 0.55 * ph])
    head = np.array([primary[0] + 0.2 * pw, primary[1], primary[2] - 0.2 * pw, primary[1] + 0.30 * ph])
    centre = np.array([pcx, pcy, pcx, pcy])

    def _wmin(box: Optional[np.ndarray]) -> float:
        if box is None:
            return w_floor
        return max(w_floor, box[2] - box[0], (box[3] - box[1]) * aspect)

    # (containment box, centre-mode, head anchor) stages: whole primary -> upper body with headroom ->
    # head with headroom -> centre point -> best effort (only exclude duplicates, stay central)
    top = float(primary[1])
    stages: List[Tuple[Optional[np.ndarray], bool, Optional[float]]] = [
        (primary, False, None), (upper, False, top), (head, False, top), (centre, False, None), (None, True, None)]
    for box, centre_mode, head_top in stages:
        res = _search_stage(frame_w, frame_h, aspect, box, dups, pcx, pcy, min(_wmin(box), w_max), w_max, centre_mode,
                            head_top)
        if res is not None:
            x, y, w = res
            return _finalize(x, y, w, aspect, frame_w, frame_h)
    # everything is blocked: centre the smallest crop on the primary
    w = w_floor
    h = w / aspect
    return _finalize(pcx - w / 2.0, pcy - h / 2.0, w, aspect, frame_w, frame_h)


def _finalize(x: float, y: float, w: float, aspect: float, frame_w: float, frame_h: float) -> BBoxT:
    w = min(max(w, 1.0), min(frame_w, frame_h * aspect))
    h = w / aspect
    x = min(max(x, 0.0), frame_w - w)
    y = min(max(y, 0.0), frame_h - h)
    return (round(x, 3), round(y, 3), round(x + w, 3), round(y + h, 3))


# --------------------------------------------------------------------------- smoothing


def _ema_zero_phase(x: np.ndarray, alpha: float) -> np.ndarray:
    """Forward + backward EMA (no lag). ``alpha`` is the weight of the new sample:
    smaller = more inertia."""
    alpha = float(min(max(alpha, 1e-3), 1.0))

    def _pass(v: np.ndarray) -> np.ndarray:
        out = np.empty_like(v)
        out[0] = v[0]
        for i in range(1, len(v)):
            out[i] = alpha * v[i] + (1.0 - alpha) * out[i - 1]
        return out

    if len(x) < 2:
        return x.copy()
    return _pass(_pass(x)[::-1])[::-1]


def _savgol(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    if n < 5:
        return x.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    window = max(5, min(window, n if n % 2 == 1 else n - 1))
    try:
        from scipy.signal import savgol_filter

        return savgol_filter(x, window_length=window, polyorder=2, mode="interp")
    except Exception:  # scipy missing -> moving average
        k = np.ones(window) / window
        pad = window // 2
        return np.convolve(np.pad(x, pad, mode="edge"), k, mode="valid")


def smooth_crops(crops: np.ndarray, frame_w: float, frame_h: float, method: str = "ema", alpha: float = 0.15,
                 window: int = 15, aspect: Optional[float] = None) -> np.ndarray:
    """Smooth an (N, 4) array of crops. Smooths centre + width (height follows the aspect
    ratio) so the aspect is preserved exactly, then clamps into the frame."""
    crops = np.asarray(crops, dtype=np.float64).reshape(-1, 4)
    if len(crops) == 0:
        return crops
    aspect = float(aspect) if aspect else float(frame_w) / float(frame_h)
    cx = (crops[:, 0] + crops[:, 2]) / 2.0
    cy = (crops[:, 1] + crops[:, 3]) / 2.0
    w = crops[:, 2] - crops[:, 0]
    f = (lambda v: _savgol(v, window)) if method == "savgol" else (lambda v: _ema_zero_phase(v, alpha))
    cx, cy, w = f(cx), f(cy), f(w)
    w_max = min(float(frame_w), float(frame_h) * aspect)
    w = np.clip(w, 1.0, w_max)
    h = w / aspect
    x1 = np.clip(cx - w / 2.0, 0.0, frame_w - w)
    y1 = np.clip(cy - h / 2.0, 0.0, frame_h - h)
    return np.stack([x1, y1, x1 + w, y1 + h], axis=1)


def _kf_from_crop(frame: int, fps: float, crop: Sequence[float], frame_w: float, frame_h: float) -> ReframeKeyframe:
    x1, y1, x2, y2 = (float(v) for v in crop)
    w = max(x2 - x1, 1e-6)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return ReframeKeyframe(
        frame=int(frame),
        timeMs=round(frame / fps * 1000.0, 3),
        crop=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        zoom=round(frame_w / w, 4),
        tx=round((cx - frame_w / 2.0) / (frame_w / 2.0), 4),
        ty=round((cy - frame_h / 2.0) / (frame_h / 2.0), 4),
    )


def smooth_track(keyframes: Sequence[ReframeKeyframe], method: str = "ema", alpha: float = 0.15, window: int = 15,
                 frame_w: Optional[float] = None, frame_h: Optional[float] = None, fps: float = 25.0) -> List[ReframeKeyframe]:
    """Smooth ``x1, y1, x2, y2`` of a keyframe list, re-enforcing aspect and frame bounds."""
    if not keyframes:
        return []
    if frame_w is None or frame_h is None:
        # derive the frame size from zoom/tx of the first keyframe
        k0 = keyframes[0]
        frame_w = frame_w or (k0.crop[2] - k0.crop[0]) * k0.zoom
        frame_h = frame_h or frame_w * ((keyframes[0].crop[3] - keyframes[0].crop[1]) / max(keyframes[0].crop[2] - keyframes[0].crop[0], 1e-6))
    crops = np.array([k.crop for k in keyframes], dtype=np.float64)
    sm = smooth_crops(crops, frame_w, frame_h, method, alpha, window)
    return [_kf_from_crop(k.frame, fps, sm[i], frame_w, frame_h) for i, k in enumerate(keyframes)]


# --------------------------------------------------------------------------- track building


RELAX_HOLD_S = 0.25  # fallback reframe: the crop is held this long beyond the first / last finding ...
RELAX_RAMP_S = 0.5   # ... then eases back to the full frame over this long


def build_reframe_track(shot: Shot, findings: Sequence[DuplicateFinding], frame_w: int, frame_h: int, fps: float,
                        method: str = "ema", alpha: float = 0.15, window: int = 15, aspect: Optional[float] = None,
                        max_keyframes: int = 600, dup_pad_frac: float = 0.02,
                        relax_hold_s: float = RELAX_HOLD_S, relax_ramp_s: float = RELAX_RAMP_S) -> Optional[ReframeTrack]:
    """Solve a crop per finding frame, interpolate across the shot, smooth and emit
    per-frame keyframes (thinned to every 2nd frame when more than ``max_keyframes``).

    This is the fallback when the duplicate could not be tracked densely: the crop is only known at
    the sampled findings, so it is held ``relax_hold_s`` (one sample interval) beyond the first and
    the last finding and then eases back to the full frame over ``relax_ramp_s`` instead of zooming
    for the whole shot."""
    fps = float(fps) if fps and fps > 0 else 25.0
    rel = [f for f in findings if f.shotIndex == shot.index]
    if not rel:
        return None
    by_frame: dict[int, dict] = {}
    for f in rel:
        e = by_frame.setdefault(int(f.frame), {"primary": None, "dups": [], "labels": set()})
        if e["primary"] is None or _area(f.primary.bbox) > _area(e["primary"]):
            e["primary"] = list(f.primary.bbox)
        # safety margin: duplicates are sampled at a few fps and move in between; interpolation and
        # smoothing of the crop must not graze them
        pad = dup_pad_frac * frame_w
        b = f.duplicate.bbox
        e["dups"].append([max(0.0, b[0] - pad), max(0.0, b[1] - pad), min(float(frame_w), b[2] + pad),
                          min(float(frame_h), b[3] + pad)])
        e["labels"].add(getattr(f, "characterName", None) or f.duplicate.label)

    key_frames = sorted(by_frame)
    key_crops = np.array([solve_crop(frame_w, frame_h, by_frame[k]["primary"], by_frame[k]["dups"], aspect) for k in key_frames])

    frames = np.arange(shot.startFrame, shot.endFrame + 1)
    asp = float(aspect) if aspect else frame_w / float(frame_h)
    w_max = min(float(frame_w), frame_h * asp)
    full = np.array([(frame_w - w_max) / 2, 0.0, (frame_w + w_max) / 2, w_max / asp])
    hold, ramp = relax_hold_s * fps, max(1.0, relax_ramp_s * fps)
    xs: List[float] = [float(k) for k in key_frames]
    ys: List[np.ndarray] = list(key_crops)
    if key_frames[0] - hold > shot.startFrame:  # relax before the first finding
        xs = [key_frames[0] - hold - ramp, key_frames[0] - hold] + xs
        ys = [full, key_crops[0]] + ys
    if key_frames[-1] + hold < shot.endFrame:  # ... and after the last one
        xs = xs + [key_frames[-1] + hold, key_frames[-1] + hold + ramp]
        ys = ys + [key_crops[-1], full]
    ys_a = np.array(ys)
    if len(xs) == 1:
        crops = np.repeat(ys_a, len(frames), axis=0)
    else:
        crops = np.stack([np.interp(frames, xs, ys_a[:, i]) for i in range(4)], axis=1)
    crops = smooth_crops(crops, frame_w, frame_h, method, alpha, window, aspect)
    # smoothing blends the relaxed full frame into the first / last findings: inside the window where
    # the duplicate is known, move every frame off the bracketing findings' duplicate boxes again
    kf = np.asarray(key_frames)
    for idx, f in enumerate(frames):
        if f < key_frames[0] - hold or f > key_frames[-1] + hold:
            continue
        j = int(np.searchsorted(kf, f))
        near = {int(kf[max(0, j - 1)]), int(kf[min(len(kf) - 1, j)])}
        ds = [d for k in near for d in by_frame[k]["dups"]]
        k0 = int(kf[min(range(len(kf)), key=lambda q: abs(kf[q] - f))])
        crops[idx] = clamp_crop(crops[idx], ds, head_region(by_frame[k0]["primary"]), frame_w, frame_h, asp)

    step = 2 if len(frames) > max_keyframes else 1
    idx = list(range(0, len(frames), step))
    if idx[-1] != len(frames) - 1:
        idx.append(len(frames) - 1)
    kfs = [_kf_from_crop(int(frames[i]), fps, crops[i], frame_w, frame_h) for i in idx]
    labels = sorted({lab for e in by_frame.values() for lab in e["labels"]})
    n_dups = sum(len(e["dups"]) for e in by_frame.values())
    reason = f"excluded {n_dups} duplicate {' / '.join(labels) or 'instance'} detection(s) across {len(key_frames)} sampled frame(s)"
    return ReframeTrack(sourceWidth=int(frame_w), sourceHeight=int(frame_h), keyframes=kfs, reason=reason)


def _area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


# --------------------------------------------------------------------------- tracked duplicates


def head_region(b: Sequence[float]) -> Tuple[float, float, float, float]:
    """The head region ``solve_crop`` protects last: top 30 % of the box, central 60 % of its width."""
    x1, y1, x2, y2 = (float(v) for v in b)
    w, h = x2 - x1, y2 - y1
    return (x1 + 0.2 * w, y1, x2 - 0.2 * w, y1 + 0.30 * h)


def _intersects(a: Sequence[float], b: Sequence[float], tol: float = 0.0) -> bool:
    return a[0] < b[2] - tol and a[2] > b[0] + tol and a[1] < b[3] - tol and a[3] > b[1] + tol


def clamp_crop(crop: Sequence[float], dups: Sequence[Sequence[float]], head: Optional[Sequence[float]],
               frame_w: float, frame_h: float, aspect: float) -> Tuple[float, float, float, float]:
    """Smallest change to ``crop`` (shift first, then shrink at the fixed aspect) that moves it
    off every duplicate box, on the side of the duplicate where the head is, keeping the head
    inside the crop when geometrically possible."""
    x1, y1, x2, y2 = (float(v) for v in crop)
    for _ in range(3):
        hit = [d for d in dups if _intersects((x1, y1, x2, y2), d)]
        if not hit:
            break
        for d in hit:
            if not _intersects((x1, y1, x2, y2), d):
                continue
            hb = head if head is not None else ((x1 + x2) / 2, (y1 + y2) / 2, (x1 + x2) / 2, (y1 + y2) / 2)
            options = []
            # side: (constraint name, feasible for the head?)
            if hb[2] <= d[0]:
                options.append("left")
            if hb[0] >= d[2]:
                options.append("right")
            if hb[3] <= d[1]:
                options.append("above")
            if hb[1] >= d[3]:
                options.append("below")
            if not options:  # head overlaps the duplicate: pick the side with more room
                hcx = (hb[0] + hb[2]) / 2
                options.append("left" if hcx < (d[0] + d[2]) / 2 else "right")
            best = None
            for side in options:
                c = _move_off(x1, y1, x2, y2, d, side, hb, frame_w, frame_h, aspect)
                loss = (x2 - x1) - (c[2] - c[0]) + 0.25 * (abs(c[0] - x1) + abs(c[1] - y1))
                if best is None or loss < best[0]:
                    best = (loss, c)
            x1, y1, x2, y2 = best[1]
    if head is not None:
        x1, y1, x2, y2 = _contain_head((x1, y1, x2, y2), head, dups, frame_w, frame_h)
    return (x1, y1, x2, y2)


def _contain_head(crop, head, dups, frame_w, frame_h):
    """Shift the crop (no resize) so it contains the head region, when that keeps it inside the
    frame and off every duplicate; else leave it."""
    x1, y1, x2, y2 = crop
    w, h = x2 - x1, y2 - y1
    if head[2] - head[0] > w or head[3] - head[1] > h:
        return crop
    dx = min(0.0, head[0] - x1) + max(0.0, head[2] - x2)
    dy = min(0.0, head[1] - y1) + max(0.0, head[3] - y2)
    if dx == 0.0 and dy == 0.0:
        return crop
    nx1 = min(max(x1 + dx, 0.0), frame_w - w)
    ny1 = min(max(y1 + dy, 0.0), frame_h - h)
    cand = (nx1, ny1, nx1 + w, ny1 + h)
    if any(_intersects(cand, d) for d in dups):
        return crop
    return cand


def _move_off(x1, y1, x2, y2, d, side, hb, frame_w, frame_h, aspect):
    w, h = x2 - x1, y2 - y1
    if side in ("left", "right"):
        if side == "left":
            limit = d[0]
            x2n = min(x2, limit)
            x1n = x2n - w
            if x1n < 0:  # no room to shift: shrink, right edge on the duplicate
                x1n, w = 0.0, max(8.0, x2n)
        else:
            limit = d[2]
            x1n = max(x1, limit)
            if x1n + w > frame_w:
                w = max(8.0, frame_w - x1n)
        hn = w / aspect
        cy = (y1 + y2) / 2
        y1n = cy - hn / 2
        y1n = min(y1n, hb[1] - 0.06 * hn) if hb[3] - hb[1] < hn else y1n  # headroom above the head
        y1n = max(y1n, hb[3] - hn)
        y1n = min(max(y1n, 0.0), frame_h - hn)
        return (x1n, y1n, x1n + w, y1n + hn)
    if side == "above":
        y2n = min(y2, d[1])
        y1n = y2n - h
        if y1n < 0:
            y1n, h = 0.0, max(8.0, y2n)
    else:
        y1n = max(y1, d[3])
        if y1n + h > frame_h:
            h = max(8.0, frame_h - y1n)
    wn = h * aspect
    cx = (x1 + x2) / 2
    x1n = min(max(cx - wn / 2, hb[2] - wn), hb[0])
    x1n = min(max(x1n, 0.0), frame_w - wn)
    return (x1n, y1n, x1n + wn, y1n + h)


def _interp_box(frames: np.ndarray, samples: Sequence[int], boxes: Sequence[Optional[Sequence[float]]],
                conservative: bool) -> List[Optional[np.ndarray]]:
    """Per-frame boxes from sparse samples. ``conservative``: the union of the two bracketing
    boxes (a duplicate is somewhere between them), a single known neighbour when only one side
    has a box, None when neither has; otherwise linear interpolation / nearest."""
    s = np.asarray(samples)
    out: List[Optional[np.ndarray]] = []
    for f in frames:
        j = int(np.searchsorted(s, f))
        lo = j - 1 if j > 0 else None
        hi = j if j < len(s) else None
        if hi is not None and s[hi] == f:
            b = boxes[hi]
            out.append(None if b is None else np.asarray(b, dtype=np.float64))
            continue
        bl = boxes[lo] if lo is not None else None
        bh = boxes[hi] if hi is not None else None
        if bl is None and bh is None:
            out.append(None)
        elif bl is None or bh is None:
            out.append(np.asarray(bl if bl is not None else bh, dtype=np.float64))
        elif conservative:
            a, b = np.asarray(bl), np.asarray(bh)
            out.append(np.array([min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]))
        else:
            t = (f - s[lo]) / float(s[hi] - s[lo])
            out.append((1 - t) * np.asarray(bl, dtype=np.float64) + t * np.asarray(bh, dtype=np.float64))
    return out


@dataclass
class TrackedFrames:
    """Per-frame geometry of a tracked reframe (source px), for checks / tests."""

    frames: List[int]
    crops: np.ndarray
    duplicates: List[List[np.ndarray]]
    heads: List[Optional[np.ndarray]]


def build_tracked_reframe(shot: Shot, pairs: Sequence, frame_w: int, frame_h: int, fps: float, method: str = "ema",
                          alpha: float = 0.15, window: int = 15, aspect: Optional[float] = None,
                          max_keyframes: int = 600, dup_pad_frac: float = 0.02, held_margin: float = 0.05,
                          return_frames: bool = False, camera: str = "planned"):
    """Reframe track from duplicate pairs followed through the shot (:class:`dupetrack.PairTrack`).

    Per dense sample: ``solve_crop(primary, tracked duplicates)`` where a duplicate is visible,
    the full frame where none is (the crop relaxes back smoothly). Crops are interpolated to every
    frame and smoothed (zero-phase EMA / Savitzky-Golay), then every frame is clamped so it never
    intersects that frame's duplicate box (union of the bracketing samples, padded by
    ``dup_pad_frac`` of the frame width, held boxes by a further ``held_margin`` of their width)
    while keeping the primary's head region.

    ``camera="planned"`` (default) solves the whole shot's camera path at once
    (:mod:`camerapath`: hard duplicate exclusion, L1 holds, eased acceleration / jerk, anticipation);
    ``camera="smooth"`` is the older smooth + clamp (twice) path, also used as the fallback."""
    fps = float(fps) if fps and fps > 0 else 25.0
    aspect = float(aspect) if aspect else frame_w / float(frame_h)
    pairs = [p for p in pairs if p is not None and p.frames]
    if not pairs:
        return None
    main = max(pairs, key=lambda p: np.mean([_area(b) for b in p.primary if b is not None] or [0.0]))
    samples = main.frames
    pad = dup_pad_frac * frame_w

    def padded(b, is_held):
        m = pad + (held_margin * (b[2] - b[0]) if is_held else 0.0)
        return [max(0.0, b[0] - m), max(0.0, b[1] - m), min(float(frame_w), b[2] + m), min(float(frame_h), b[3] + m)]

    dup_samples: List[List[Optional[list]]] = []  # per pair, per sample
    for p in pairs:
        dup_samples.append([None if b is None else padded(b, h) for b, h in zip(p.duplicate, p.dup_held)])

    # primary: None where it is not visible (tracker gave up); then the crop only has to exclude the
    # duplicates and stays as central as possible
    prim = list(main.primary)
    if not any(b is not None for b in prim):
        return None
    w_max = min(float(frame_w), frame_h * aspect)
    full = np.array([(frame_w - w_max) / 2, 0.0, (frame_w + w_max) / 2, w_max / aspect])
    centre_pt = (frame_w / 2.0 - 1, frame_h / 2.0 - 1, frame_w / 2.0 + 1, frame_h / 2.0 + 1)
    key_crops = []
    for i in range(len(samples)):
        dups = [ds[i] for ds, p in zip(dup_samples, pairs) if i < len(ds) and ds[i] is not None and p.frames == samples]
        if not dups:
            key_crops.append(full)
        else:
            key_crops.append(np.array(solve_crop(frame_w, frame_h, prim[i] if prim[i] is not None else centre_pt, dups,
                                                 aspect)))
    key_crops = np.array(key_crops)

    frames = np.arange(shot.startFrame, shot.endFrame + 1)
    targets = np.stack([np.interp(frames, samples, key_crops[:, i]) for i in range(4)], axis=1)
    per_frame_dups: List[List[np.ndarray]] = [[] for _ in frames]
    dup_tracks: List[List[Optional[np.ndarray]]] = []
    for p, ds in zip(pairs, dup_samples):
        track = _interp_box(frames, p.frames, ds, conservative=True)
        dup_tracks.append(track)
        for fi, b in enumerate(track):
            if b is not None:
                per_frame_dups[fi].append(b)
    heads_src = [None if b is None else head_region(b) for b in prim]
    heads = _interp_box(frames, samples, heads_src, conservative=False)
    prim_frames = _interp_box(frames, samples, prim, conservative=False)
    # several duplicates in one shot: the other pairs' primaries are kept in frame too (a lighter,
    # soft head term; the main primary and every duplicate exclusion take precedence)
    extra_heads = []
    for p in pairs:
        if p is main:
            continue
        hs = [None if b is None else head_region(b) for b in p.primary]
        extra_heads.append([None if h is None else np.asarray(h) for h in _interp_box(frames, p.frames, hs, conservative=False)])

    def clamp_all(c: np.ndarray) -> np.ndarray:
        return np.array([clamp_crop(c[i], per_frame_dups[i], heads[i], frame_w, frame_h, aspect) for i in range(len(c))])

    # Plan the whole shot's camera move at once (holds, eased moves, anticipation; the duplicate
    # is a hard constraint). Fall back to smooth + clamp if the solver cannot find a path.
    crops = None
    plan_info: dict = {}
    if camera == "planned":
        try:
            from . import camerapath

            crops = camerapath.plan_camera_path(targets, dup_tracks, heads, frame_w, frame_h, aspect,
                                                primary=prim_frames, fps=fps, extra_heads=extra_heads or None,
                                                info=plan_info)
        except Exception as exc:  # pragma: no cover - solver import / numerical failure
            log.warning("camera path planning failed (%s); using smooth + clamp", exc)
            crops = None
    if crops is not None:
        # the planner's constraints already exclude the duplicates; the clamp only absorbs solver
        # tolerance (sub-pixel) and would only move a frame if a constraint had been violated
        clamped = clamp_all(crops)
        moved = float(np.abs(clamped - crops).max()) if len(crops) else 0.0
        if moved > 1.0:
            log.warning("camera path needed a %.1f px correction after planning", moved)
        crops = clamped
    else:
        crops = clamp_all(smooth_crops(targets, frame_w, frame_h, method, alpha, window, aspect))
        crops = clamp_all(smooth_crops(crops, frame_w, frame_h, method, min(1.0, alpha * 2), window, aspect))

    step = 2 if len(frames) > max_keyframes else 1
    idx = list(range(0, len(frames), step))
    if idx[-1] != len(frames) - 1:
        idx.append(len(frames) - 1)
    kfs = [_kf_from_crop(int(frames[i]), fps, crops[i], frame_w, frame_h) for i in idx]
    names = sorted({p.label or "instance" for p in pairs})
    vis = sum(1 for d in per_frame_dups if d)
    st = main.stats
    reason = (f"tracked duplicate {' / '.join(names)} through the shot: visible in {vis}/{len(frames)} frame(s) "
              f"({st.get('observedDuplicate', 0)} observed / {st.get('held', 0)} held of {st.get('samples', 0)} samples); "
              f"crop excludes it in every frame")
    if plan_info.get("mode") not in (None, "sides"):
        reason += f" (camera planned with {plan_info['mode']})"
    track = ReframeTrack(sourceWidth=int(frame_w), sourceHeight=int(frame_h), keyframes=kfs, reason=reason)
    if return_frames:
        return track, TrackedFrames([int(f) for f in frames], crops, per_frame_dups, heads)
    return track


def static_track(frame_w: int, frame_h: int, crop: Sequence[float], shot: Shot, fps: float, reason: str) -> ReframeTrack:
    """A two-keyframe constant crop (used for AI-director suggestions)."""
    c = _finalize(crop[0], crop[1], crop[2] - crop[0], frame_w / frame_h, frame_w, frame_h)
    kfs = [_kf_from_crop(shot.startFrame, fps, c, frame_w, frame_h), _kf_from_crop(shot.endFrame, fps, c, frame_w, frame_h)]
    return ReframeTrack(sourceWidth=int(frame_w), sourceHeight=int(frame_h), keyframes=kfs, reason=reason)
