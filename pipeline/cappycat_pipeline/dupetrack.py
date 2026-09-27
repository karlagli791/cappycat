"""Follow a confirmed duplicate pair (primary + duplicate) through its whole shot.

Duplicate *confirmation* happens at a few sampled frames (identity / similarity rules in
:mod:`.perception`). The camera and the characters keep moving afterwards, so the reframer needs
both instances in **every** frame. :func:`track_shot` re-scans the shot densely (default 8 fps)
and, starting from the confirmed frames ("anchors"), follows both instances forwards and
backwards by

* box continuity: IoU / centre distance to a constant-velocity prediction (the gate widens while
  an instance is not re-found), plus a scale-ratio gate;
* appearance: cosine similarity to that instance's OpenCLIP embedding at the confirmed frame it
  started from / last passed (each confirmed frame re-snaps box and appearance), so a character
  that turns during a long shot is compared with its nearest confirmed look.

The identity score is *not* required after the anchor (the pair only needs to be confirmed once);
the two targets are assigned jointly (Hungarian), so near-identical copies cannot swap by
appearance alone. A detection confidently identified as a *different* cast member (full box and,
for tall boxes, head crop - like the perception pass) is never taken. When an instance is briefly
missed its last box is held (with a margin); after ``hold_s`` without a match the duplicate is
considered gone - going **forwards** (it left). Going **backwards** from the first confirmation the
duplicate keeps being excluded at its earliest known position all the way to the shot start, unless
that position touches a frame edge (it walked in through the edge): a copy that is on screen at the
cut but not re-detected must not reappear in the crop. The primary is always held.

The shot is decoded once and streamed: per dense sample only the detections, their embeddings and
identities are kept (shared by every duplicate group of the shot), plus the few frames nearest to
the anchors; detections the perception pass already made on the same source frame come from the
:class:`perception.DetectionCache`.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .schema import DuplicateFinding, Shot
from .tracking import iou_matrix

log = logging.getLogger("cappycat.dupetrack")

Box = Tuple[float, float, float, float]
EDGE_FRAC = 0.015  # a box within 1.5 % of the frame edge "touches" it
# how fast a target's reference appearance follows its unambiguous matches. 0 (default): only the
# confirmed frames set it - adapting (0.1-0.2) followed a close-up's partial boxes on clip10 and
# forced a 2.5x crop, and it removed the approved push-in on clip7
APPEARANCE_RATE = 0.0


@dataclass
class PairTrack:
    frames: List[int]                    # source frame index of every dense sample
    primary: List[Optional[Box]]         # source px, held boxes included
    duplicate: List[Optional[Box]]       # source px; None = duplicate not visible
    dup_held: List[bool]                 # duplicate box is a held (unobserved) position
    anchors: List[int]                   # dense sample indices of the confirmed frames
    label: str = ""
    character: Optional[str] = None
    stats: dict = field(default_factory=dict)


@dataclass
class ShotObservations:
    """Per dense sample of a shot: detections (analysis px), embeddings and confident identities."""

    frames: List[int]                                  # source frame index per sample
    boxes: List[np.ndarray]                            # (n_i, 4)
    embs: List[np.ndarray]                             # (n_i, d)
    chars: List[List[Optional[str]]]                   # confident cast identity per detection
    images: Dict[int, np.ndarray]                      # sample index -> frame (anchor neighbourhoods only)
    size: Tuple[int, int] = (0, 0)                     # analysis frame (w, h)
    detections: int = 0
    cache_hits: int = 0


@dataclass
class _Target:
    box: np.ndarray                      # analysis px
    emb: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    lost: int = 0

    @property
    def centre(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0])

    def predicted(self) -> np.ndarray:
        d = self.vel * (self.lost + 1)
        return self.box + np.array([d[0], d[1], d[0], d[1]])


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def touches_edge(b: Sequence[float], w: float, h: float, frac: float = EDGE_FRAC) -> bool:
    mx, my = frac * w, frac * h
    return b[0] <= mx or b[1] <= my or b[2] >= w - mx or b[3] >= h - my


def _associate(targets: List[_Target], boxes: np.ndarray, embs: np.ndarray, app_min: float,
               max_lost_gate: float, vetoed: Optional[Sequence[bool]] = None) -> List[Optional[int]]:
    """Joint assignment of detections to targets; returns the detection index per target."""
    n_t, n_d = len(targets), len(boxes)
    if n_d == 0:
        return [None] * n_t
    score = np.full((n_t, n_d), -1e9)
    for ti, t in enumerate(targets):
        p = t.predicted()
        pw = max(p[2] - p[0], 1.0)
        iou = iou_matrix(p[None], boxes)[0]
        gate = min(0.6 + 0.35 * t.lost, max_lost_gate)
        for j in range(n_d):
            if vetoed is not None and vetoed[j]:
                continue
            b = boxes[j]
            ratio = _area(b) / max(_area(p), 1.0)
            if not (0.4 <= ratio <= 2.5):
                continue
            cd = math.hypot((b[0] + b[2]) / 2 - (p[0] + p[2]) / 2, (b[1] + b[3]) / 2 - (p[1] + p[3]) / 2) / pw
            app = _cos(embs[j], t.emb)
            if app < app_min or (iou[j] < 0.2 and cd > gate):
                continue
            score[ti, j] = iou[j] + 0.5 * max(0.0, 1.0 - cd) + 2.0 * (app - app_min)
    out: List[Optional[int]] = [None] * n_t
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-score)
        for r, c in zip(rows, cols):
            if score[r, c] > -1e8:
                out[r] = int(c)
    except Exception:  # greedy
        used = set()
        for flat in np.argsort(-score, axis=None):
            r, c = divmod(int(flat), n_d)
            if score[r, c] <= -1e8:
                break
            if out[r] is None and c not in used:
                out[r], _ = int(c), used.add(c)
    return out


def observe(frames: Iterable[Tuple[int, np.ndarray]], detect: Callable, embed_many: Callable,
            identify_frame: Optional[Callable] = None, keep_near: Sequence[int] = (),
            keep_all: bool = False, nms_iou: float = 0.6,
            on_frame: Optional[Callable[[int], None]] = None) -> ShotObservations:
    """Detect + embed (+ identify) every dense sample once. ``detect(src_idx, frame) -> ([Detection],
    from_cache)``; ``embed_many(frame, boxes) -> (n, d)``; ``identify_frame(frame, dets, embs) ->
    [character or None]``. Frames nearest to the source indices in ``keep_near`` are kept (all
    frames with ``keep_all``)."""
    from .perception import nms

    src_idx: List[int] = []
    boxes: List[np.ndarray] = []
    embs: List[np.ndarray] = []
    chars: List[List[Optional[str]]] = []
    best_near: Dict[int, Tuple[int, int]] = {}  # anchor frame -> (distance, sample index)
    images: Dict[int, np.ndarray] = {}
    size = (0, 0)
    n_det = hits = 0
    for k, (fi, img) in enumerate(frames):
        size = (int(img.shape[1]), int(img.shape[0]))
        raw, cached = detect(fi, img)
        hits += int(bool(cached))
        dets = nms(list(raw), nms_iou, class_agnostic=True)
        b = np.array([d.bbox for d in dets], dtype=np.float64).reshape(-1, 4)
        e = np.asarray(embed_many(img, [tuple(x) for x in b]) if len(b) else np.zeros((0, 1)))
        c: List[Optional[str]] = [None] * len(b)
        if identify_frame is not None and len(b):
            c = list(identify_frame(img, dets, e))
        src_idx.append(int(fi))
        boxes.append(b)
        embs.append(e)
        chars.append(c)
        n_det += len(b)
        if keep_all:
            images[k] = img
        else:
            for a in keep_near:
                d = abs(int(fi) - int(a))
                prev = best_near.get(a)
                if prev is None or d < prev[0]:
                    best_near[a] = (d, k)
                    images[k] = img
                    if prev is not None and all(v[1] != prev[1] for v in best_near.values()):
                        images.pop(prev[1], None)  # no anchor needs that frame any more
        if on_frame:
            on_frame(k + 1)
    return ShotObservations(src_idx, boxes, embs, chars, images, size, n_det, hits)


def track_pair_obs(obs: ShotObservations, anchors: Sequence[DuplicateFinding], embed_many: Callable, scale: float,
                   sample_fps: float, hold_s: float = 1.0, app_min: float = 0.7,
                   hold_back_to_start: bool = True) -> Optional[PairTrack]:
    """Follow one duplicate group (see :func:`track_pair`) on pre-computed observations."""
    if not obs.frames or not anchors:
        return None
    src_idx = obs.frames
    inv = 1.0 / scale
    who = anchors[0].character
    det_veto = [[bool(c) and c != who for c in cs] if who else [False] * len(cs) for cs in obs.chars]

    # anchors -> nearest dense sample (analysis px)
    anchor_at = {}
    for a in anchors:
        k = int(np.argmin([abs(s - a.frame) for s in src_idx]))
        anchor_at[k] = (np.array(a.primary.bbox, dtype=np.float64) * scale, np.array(a.duplicate.bbox, dtype=np.float64) * scale)
    ks = sorted(anchor_at)
    # appearance per confirmed frame (the characters turn / the camera zooms during a shot: a target
    # starts and re-snaps with the appearance of the anchor it passes, then adapts slowly)
    anchor_emb: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for k in ks:
        img = obs.images.get(k)
        if img is None and obs.images:  # (defensive) the nearest kept frame
            img = obs.images[min(obs.images, key=lambda q: abs(q - k))]
        if img is None:
            continue
        pk, dk = anchor_at[k]
        e = np.asarray(embed_many(img, [tuple(pk), tuple(dk)]))
        anchor_emb[k] = (e[0], e[1])
    if not anchor_emb:
        return None
    for k in ks:  # anchors without a kept frame borrow the nearest anchor's appearance
        if k not in anchor_emb:
            anchor_emb[k] = anchor_emb[min(anchor_emb, key=lambda q: abs(q - k))]

    n = len(src_idx)
    fw, fh = obs.size
    prim: List[Optional[np.ndarray]] = [None] * n
    dup: List[Optional[np.ndarray]] = [None] * n
    held = [False] * n
    observed = {"primary": 0, "duplicate": 0}
    hold_n = max(1, int(round(hold_s * sample_fps)))
    max_gate = 3.0
    back_held = [0]

    def run(order: Iterable[int], start: int, backwards: bool) -> None:
        tp = _Target(anchor_at[start][0].copy(), anchor_emb[start][0].copy())
        td = _Target(anchor_at[start][1].copy(), anchor_emb[start][1].copy())
        for k in order:
            if k in anchor_at:  # confirmed frame: snap (box and appearance)
                pb, db = anchor_at[k]
                for t, b, e_ in ((tp, pb, anchor_emb[k][0]), (td, db, anchor_emb[k][1])):
                    c_old = t.centre
                    t.box = b.copy()
                    t.vel = 0.5 * t.vel + 0.5 * (t.centre - c_old) / max(1, t.lost + 1)
                    t.lost = 0
                    t.emb = e_.copy()
                prim[k], dup[k], held[k] = tp.box.copy(), td.box.copy(), False
                continue
            m = _associate([tp, td], obs.boxes[k], obs.embs[k], app_min, max_gate, det_veto[k])
            for t, j, name in ((tp, m[0], "primary"), (td, m[1], "duplicate")):
                if j is None:
                    t.lost += 1
                    continue
                nb = obs.boxes[k][j]
                c_old = t.centre
                if APPEARANCE_RATE > 0 and iou_matrix(t.predicted()[None], nb[None])[0, 0] >= 0.5:  # adapt slowly
                    e_new = APPEARANCE_RATE * np.asarray(obs.embs[k][j], dtype=np.float64) + (1 - APPEARANCE_RATE) * t.emb
                    t.emb = e_new / (np.linalg.norm(e_new) + 1e-9)
                t.box = nb.copy()
                t.vel = 0.6 * t.vel + 0.4 * (t.centre - c_old) / (t.lost + 1)
                t.lost = 0
                observed[name] += 1
            # the primary is held while briefly missed; after that it is "not visible" (None)
            prim[k] = tp.box.copy() if tp.lost <= hold_n else None
            if td.lost == 0:
                dup[k], held[k] = td.box.copy(), False
            elif td.lost <= hold_n:  # last known position; the reframer adds the held-box margin
                dup[k], held[k] = td.box.copy(), True
            elif backwards and hold_back_to_start and fw > 0 and not touches_edge(td.box, fw, fh):
                # before the first confirmation: still on screen unless it came in through an edge
                dup[k], held[k] = td.box.copy(), True
                back_held[0] += 1
            else:
                dup[k], held[k] = None, False

    first = ks[0]
    prim[first], dup[first] = anchor_at[first][0].copy(), anchor_at[first][1].copy()
    run(range(first + 1, n), first, False)
    run(range(first - 1, -1, -1), first, True)

    def to_src(b: Optional[np.ndarray]) -> Optional[Box]:
        return None if b is None else tuple(float(v * inv) for v in b)

    a0 = anchors[0]
    return PairTrack(
        frames=list(src_idx), primary=[to_src(b) for b in prim], duplicate=[to_src(b) for b in dup], dup_held=held,
        anchors=ks, label=a0.characterName or a0.duplicate.label, character=a0.character,
        stats={"samples": n, "detections": obs.detections, "observedPrimary": observed["primary"],
               "observedDuplicate": observed["duplicate"], "held": int(sum(held)), "heldToStart": back_held[0],
               "primaryVisible": int(sum(b is not None for b in prim)),
               "duplicateVisible": int(sum(d is not None for d in dup)), "cachedFrames": obs.cache_hits},
    )


def track_pair(frames: Sequence[Tuple[int, np.ndarray]], anchors: Sequence[DuplicateFinding], detect: Callable,
               embed_many: Callable, scale: float, sample_fps: float, hold_s: float = 1.0,
               app_min: float = 0.7, nms_iou: float = 0.6, identify: Optional[Callable] = None,
               hold_back_to_start: bool = True) -> Optional[PairTrack]:
    """``frames``: dense ``(source_frame, frame_bgr)`` samples of one shot (analysis resolution);
    ``anchors``: the confirmed findings of one duplicate (same character / track pair), boxes in
    source px; ``detect(frame) -> [Detection]``; ``embed_many(frame, boxes) -> (n, d)``;
    ``scale`` = analysis width / source width. ``identify(embs) -> [Identity]`` (the main-cast bank)
    vetoes detections confidently identified as a *different* cast member for a named pair."""
    if not frames or not anchors:
        return None
    ident = None
    if identify is not None:
        ident = lambda img, dets, embs: [getattr(i, "character", None) for i in identify(embs)]
    obs = observe(frames, lambda fi, img: (detect(img), False), embed_many, ident, keep_all=True, nms_iou=nms_iou)
    return track_pair_obs(obs, anchors, embed_many, scale, sample_fps, hold_s, app_min, hold_back_to_start)


def group_findings(findings: Sequence[DuplicateFinding]) -> List[List[DuplicateFinding]]:
    """Findings of one shot grouped per duplicate (named character, or track-id pair)."""
    groups: dict = {}
    for f in findings:
        key = ("char", f.character) if f.character else \
            ("pair", min(f.primary.trackId, f.duplicate.trackId), max(f.primary.trackId, f.duplicate.trackId))
        groups.setdefault(key, []).append(f)
    return [sorted(g, key=lambda f: f.frame) for g in groups.values()]


def dense_frames(path: str, shot: Shot, src_fps: float, src_size: Tuple[int, int], sample_fps: float,
                 width: int) -> List[Tuple[int, np.ndarray]]:
    """All dense samples of a shot in memory (tests / tools; :func:`track_shot` streams instead)."""
    from .perception import sample_shot_frames

    return list(sample_shot_frames(path, shot, src_fps, src_size, sample_fps, width))


def track_shot(path: str, shot: Shot, findings: Sequence[DuplicateFinding], detector, embedder, prompts: Sequence[str],
               src_fps: float, src_size: Tuple[int, int], shot_path: str = "", bank=None, track_fps: float = 8.0,
               analysis_width: int = 640, cache=None,
               progress: Optional[Callable[[float, str], None]] = None) -> List[PairTrack]:
    """Every confirmed duplicate of ``shot`` followed through the shot (see :func:`track_pair`).
    ``detector`` is the analysis detector (the hybrid's Grounded SAM 2 fallback is used when
    ``shot_path == "grounded_sam2"``), run with a lowered score floor and without SAM masks.
    ``cache`` (:class:`perception.DetectionCache`) supplies detections of frames the perception
    pass already saw; ``progress(frac, msg)`` is called per decoded frame."""
    from . import perception

    groups = group_findings([f for f in findings if f.shotIndex == shot.index])
    if not groups:
        return []
    det = perception.tracking_detector(detector, shot_path)
    n_expected = perception.expected_samples(shot, src_fps, track_fps)
    frames = perception.sample_shot_frames(path, shot, src_fps, src_size, track_fps, analysis_width)
    det_name = getattr(det, "name", "?")

    ident = None
    if bank is not None:
        def ident(img, dets, embs):
            return [i.character for i in perception.identify_detections(bank, embedder, img, dets, list(embs))]

    def embed(img, boxes):
        return perception._embed_all(embedder, img, boxes) if boxes else []

    pairs: List[PairTrack] = []
    with perception.low_confidence(det):
        thr = perception.detector_threshold(det)

        def detect(fi: int, img: np.ndarray):
            if cache is not None and thr is not None:
                hit = cache.get(det_name, fi, img, thr)
                if hit is not None:
                    return hit, True
            return det.detect(img, list(prompts)), False

        tick = (lambda k: progress(min(1.0, k / max(1, n_expected)), f"tracking frame {k}/{n_expected}")) if progress else None
        obs = observe(frames, detect, embed, ident, keep_near=[f.frame for g in groups for f in g], on_frame=tick)
        if not obs.frames:
            return []
        scale = obs.size[0] / float(src_size[0])
        for group in groups:
            pt = track_pair_obs(obs, group, embed, scale, track_fps)
            if pt is not None:
                pt.stats["detector"] = det_name
                pairs.append(pt)
    return pairs
