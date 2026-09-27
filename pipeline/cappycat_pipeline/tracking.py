"""Self-contained ByteTrack (Zhang et al. 2022) in numpy.

* Constant-velocity Kalman filter on ``[cx, cy, a, h, vx, vy, va, vh]`` (aspect ``a = w/h``),
  same noise model as the reference implementation.
* Two-stage association: high-score detections are matched to predicted tracks by IoU
  (score-fused), then the remaining *tracked* tracks are matched against low-score
  detections so identities survive occlusion / motion blur.
* Linear assignment via ``scipy.optimize.linear_sum_assignment`` when available, with a
  greedy fallback. No ``supervision`` / ``lap`` dependency.

Boxes are ``[x1, y1, x2, y2]`` in pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    label: str
    score: float
    embedding: Optional[np.ndarray] = None
    # optional boolean (H, W) instance mask at the detection frame's resolution (Grounded SAM 2);
    # when present, ``bbox`` is already the tight box around the mask
    mask: Optional[np.ndarray] = None


# --------------------------------------------------------------------------- geometry


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (n,4) and (m,4) xyxy boxes -> (n, m)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def xyxy_to_xyah(b: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = b
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w / h, h], dtype=np.float64)


def xyah_to_xyxy(m: Sequence[float]) -> np.ndarray:
    cx, cy, a, h = m[0], m[1], m[2], m[3]
    h = max(h, 1e-6)
    w = max(a * h, 1e-6)
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float64)


# --------------------------------------------------------------------------- linear assignment


def linear_assignment(cost: np.ndarray, thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Return (matches, unmatched_rows, unmatched_cols) for costs <= thresh."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    matches: List[Tuple[int, int]] = []
    try:
        from scipy.optimize import linear_sum_assignment

        c = np.where(cost > thresh, thresh + 1e3, cost)
        rows, cols = linear_sum_assignment(c)
        for r, c_ in zip(rows, cols):
            if cost[r, c_] <= thresh:
                matches.append((int(r), int(c_)))
    except Exception:  # greedy fallback
        order = np.argsort(cost, axis=None)
        used_r, used_c = set(), set()
        for flat in order:
            r, c_ = divmod(int(flat), m)
            if cost[r, c_] > thresh:
                break
            if r in used_r or c_ in used_c:
                continue
            used_r.add(r)
            used_c.add(c_)
            matches.append((r, c_))
    mr = {r for r, _ in matches}
    mc = {c_ for _, c_ in matches}
    return matches, [i for i in range(n) if i not in mr], [j for j in range(m) if j not in mc]


# --------------------------------------------------------------------------- Kalman filter


class KalmanFilter:
    """Constant-velocity Kalman filter for xyah boxes (as in DeepSORT / ByteTrack)."""

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.concatenate([measurement, np.zeros_like(measurement)])
        h = measurement[3]
        std = [2 * self._std_weight_position * h, 2 * self._std_weight_position * h, 1e-2, 2 * self._std_weight_position * h,
               10 * self._std_weight_velocity * h, 10 * self._std_weight_velocity * h, 1e-5, 10 * self._std_weight_velocity * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std_pos = [self._std_weight_position * h, self._std_weight_position * h, 1e-2, self._std_weight_position * h]
        std_vel = [self._std_weight_velocity * h, self._std_weight_velocity * h, 1e-5, self._std_weight_velocity * h]
        motion_cov = np.diag(np.square(np.array(std_pos + std_vel)))
        mean = self._motion_mat @ mean
        cov = self._motion_mat @ cov @ self._motion_mat.T + motion_cov
        return mean, cov

    def project(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [self._std_weight_position * h, self._std_weight_position * h, 1e-1, self._std_weight_position * h]
        innovation_cov = np.diag(np.square(np.array(std)))
        return self._update_mat @ mean, self._update_mat @ cov @ self._update_mat.T + innovation_cov

    def update(self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        proj_mean, proj_cov = self.project(mean, cov)
        # K = P H^T S^-1  (solve instead of explicit inverse)
        kalman_gain = np.linalg.solve(proj_cov, (cov @ self._update_mat.T).T).T
        innovation = measurement - proj_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_cov = cov - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean, new_cov


# --------------------------------------------------------------------------- tracks


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


@dataclass
class Track:
    track_id: int
    label: str
    score: float
    mean: np.ndarray
    cov: np.ndarray
    frame_id: int
    start_frame: int
    state: int = TrackState.New
    is_activated: bool = False
    tracklet_len: int = 0
    embedding: Optional[np.ndarray] = None
    history: Dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def tlbr(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean[:4])

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        b = self.tlbr
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

    @property
    def area(self) -> float:
        b = self.tlbr
        return float(max(b[2] - b[0], 0) * max(b[3] - b[1], 0))


class ByteTracker:
    """ByteTrack multi-object tracker.

    Parameters mirror the paper defaults: ``track_thresh=0.5`` splits high / low score
    detections, ``match_thresh=0.8`` is the IoU-cost gate of the first association,
    ``track_buffer=30`` frames keeps lost tracks alive (scaled by ``frame_rate/30``).
    """

    def __init__(self, track_thresh: float = 0.5, match_thresh: float = 0.8, track_buffer: int = 30,
                 frame_rate: float = 30.0, low_thresh: float = 0.1, fuse_score: bool = True,
                 min_box_area: float = 1.0, same_label_only: bool = True) -> None:
        self.track_thresh = track_thresh
        self.det_thresh = track_thresh + 0.1
        self.match_thresh = match_thresh
        self.low_thresh = low_thresh
        self.fuse_score = fuse_score
        self.min_box_area = min_box_area
        self.same_label_only = same_label_only
        self.max_time_lost = max(1, int(frame_rate / 30.0 * track_buffer))
        self.kf = KalmanFilter()
        self.frame_id = 0
        self._next_id = 1
        self.tracked: List[Track] = []
        self.lost: List[Track] = []
        self.removed: List[Track] = []

    # ---- helpers
    def reset(self) -> None:
        self.__init__(self.track_thresh, self.match_thresh, self.max_time_lost, 30.0, self.low_thresh,
                      self.fuse_score, self.min_box_area, self.same_label_only)

    def _cost(self, tracks: List[Track], dets: List[Detection], fuse: bool) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        tb = np.array([t.tlbr for t in tracks])
        db = np.array([d.bbox for d in dets], dtype=np.float64)
        iou = iou_matrix(tb, db)
        if fuse and self.fuse_score:
            iou = iou * np.array([d.score for d in dets], dtype=np.float64)[None, :]
        cost = 1.0 - iou
        if self.same_label_only:
            tl = np.array([t.label for t in tracks])
            dl = np.array([d.label for d in dets])
            cost = np.where(tl[:, None] == dl[None, :], cost, 2.0)
        return cost

    def _new_track(self, det: Detection) -> Track:
        mean, cov = self.kf.initiate(xyxy_to_xyah(det.bbox))
        t = Track(track_id=self._next_id, label=det.label, score=det.score, mean=mean, cov=cov,
                  frame_id=self.frame_id, start_frame=self.frame_id, state=TrackState.Tracked,
                  is_activated=(self.frame_id == 1), tracklet_len=0, embedding=det.embedding)
        self._next_id += 1
        t.history[self.frame_id] = t.tlbr
        return t

    def _update_track(self, t: Track, det: Detection, reactivate: bool = False) -> None:
        t.mean, t.cov = self.kf.update(t.mean, t.cov, xyxy_to_xyah(det.bbox))
        t.state = TrackState.Tracked
        t.is_activated = True
        t.frame_id = self.frame_id
        t.tracklet_len = 0 if reactivate else t.tracklet_len + 1
        t.score = det.score
        t.label = det.label
        if det.embedding is not None:
            t.embedding = det.embedding
        t.history[self.frame_id] = t.tlbr

    # ---- main step
    def update(self, detections: Sequence[Detection]) -> List[Track]:
        """Advance one frame. Returns the activated tracks visible in this frame."""
        self.frame_id += 1
        dets = [d for d in detections if (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) >= self.min_box_area]
        high = [d for d in dets if d.score >= self.track_thresh]
        low = [d for d in dets if self.low_thresh <= d.score < self.track_thresh]

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked = [t for t in self.tracked if t.is_activated]

        # predict every track in the pool (tracked + lost)
        pool = tracked + [t for t in self.lost if t.track_id not in {x.track_id for x in tracked}]
        for t in pool:
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)

        # --- stage 1: high-score detections vs pool
        cost = self._cost(pool, high, fuse=True)
        matches, u_track, u_det = linear_assignment(cost, self.match_thresh)
        activated: List[Track] = []
        refound: List[Track] = []
        for ti, di in matches:
            t = pool[ti]
            was_lost = t.state == TrackState.Lost
            self._update_track(t, high[di], reactivate=was_lost)
            (refound if was_lost else activated).append(t)

        # --- stage 2: remaining *tracked* tracks vs low-score detections
        r_tracked = [pool[i] for i in u_track if pool[i].state == TrackState.Tracked]
        cost = self._cost(r_tracked, low, fuse=False)
        matches, u_track2, _ = linear_assignment(cost, 0.5)
        for ti, di in matches:
            t = r_tracked[ti]
            was_lost = t.state == TrackState.Lost
            self._update_track(t, low[di], reactivate=was_lost)
            (refound if was_lost else activated).append(t)
        lost_now: List[Track] = []
        for i in u_track2:
            t = r_tracked[i]
            if t.state != TrackState.Lost:
                t.state = TrackState.Lost
                lost_now.append(t)

        # --- unconfirmed tracks (seen once) vs leftover high detections
        leftover = [high[i] for i in u_det]
        cost = self._cost(unconfirmed, leftover, fuse=True)
        matches, u_unconf, u_det2 = linear_assignment(cost, 0.7)
        for ti, di in matches:
            self._update_track(unconfirmed[ti], leftover[di])
            activated.append(unconfirmed[ti])
        removed_now: List[Track] = []
        for i in u_unconf:
            t = unconfirmed[i]
            t.state = TrackState.Removed
            removed_now.append(t)

        # --- new tracks
        for i in u_det2:
            d = leftover[i]
            if d.score < self.det_thresh:
                continue
            activated.append(self._new_track(d))

        # --- expire old lost tracks
        for t in self.lost:
            if self.frame_id - t.frame_id > self.max_time_lost:
                t.state = TrackState.Removed
                removed_now.append(t)

        # --- bookkeeping
        self.tracked = [t for t in self.tracked if t.state == TrackState.Tracked]
        self.tracked = _join(self.tracked, activated)
        self.tracked = _join(self.tracked, refound)
        self.lost = _sub(self.lost, self.tracked)
        self.lost = _join(self.lost, lost_now)
        self.lost = _sub(self.lost, removed_now)
        self.tracked, self.lost = _remove_duplicate_tracks(self.tracked, self.lost)
        self.removed.extend(removed_now)
        if len(self.removed) > 1000:
            self.removed = self.removed[-500:]
        return [t for t in self.tracked if t.is_activated]


def _join(a: List[Track], b: List[Track]) -> List[Track]:
    seen = {t.track_id for t in a}
    out = list(a)
    for t in b:
        if t.track_id not in seen:
            seen.add(t.track_id)
            out.append(t)
    return out


def _sub(a: List[Track], b: List[Track]) -> List[Track]:
    ids = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids]


def _remove_duplicate_tracks(tracked: List[Track], lost: List[Track], iou_thresh: float = 0.85) -> Tuple[List[Track], List[Track]]:
    if not tracked or not lost:
        return tracked, lost
    iou = iou_matrix(np.array([t.tlbr for t in tracked]), np.array([t.tlbr for t in lost]))
    dup_a, dup_b = set(), set()
    for i, j in zip(*np.where(iou > iou_thresh)):
        if tracked[i].label != lost[j].label:
            continue
        age_a = tracked[i].frame_id - tracked[i].start_frame
        age_b = lost[j].frame_id - lost[j].start_frame
        if age_a > age_b:
            dup_b.add(int(j))
        else:
            dup_a.add(int(i))
    return [t for i, t in enumerate(tracked) if i not in dup_a], [t for j, t in enumerate(lost) if j not in dup_b]
