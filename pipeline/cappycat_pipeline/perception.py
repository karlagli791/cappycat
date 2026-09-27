"""Open-vocabulary detection, tracking glue, appearance embeddings and duplicate finding.

Detectors
    * ``YoloWorldDetector`` – ultralytics YOLO-World v2 small (``models/yolov8s-worldv2.pt``),
      prompts via ``set_classes`` (CLIP text features computed once on the CPU and cached in
      ``models/cache/yolo_world_text/``, so later runs never load the CLIP text encoder), fp16 on CUDA.
    * ``GroundedSam2Detector`` – Grounding DINO tiny boxes (text prompts, box / text thresholds)
      refined by SAM 2.1 hiera-small masks, both through Hugging Face ``transformers`` (no CUDA
      extensions to build). One caption per prompt, all captions **batched into one forward**
      (1.85x faster than one forward per prompt); masks are optional (``with_masks``) and give
      tighter boxes.
    * ``HybridDetector`` – the spec's default: YOLO-World + ByteTrack per shot, escalating the
      whole shot to Grounded SAM 2 when YOLO finds nothing for the prompts, when two instances
      that could be the same character overlap (IoU > 0.3; pairs confidently identified as two
      *different* cast members and nested head-inside-body pairs don't count), or when a
      same-label pair has a borderline appearance similarity (``threshold - 0.1 <= sim < threshold``).
      The escalated pass samples at :data:`ESCALATION_FPS`. A CUDA OOM (or any error) during the
      escalation keeps the YOLO-World result and disables escalation for the rest of the run.
      See :func:`analyze_shot`.
    * ``NullDetector`` – returns nothing (lite mode).
Embedders
    * ``OpenClipEmbedder`` – ViT-B-32 ``laion2b_s34b_b79k`` image embeddings, fp16 on CUDA, batched.
    * ``HistogramEmbedder`` – HSV colour histogram + gradient-orientation histogram, L2-normalised,
      so duplicate detection still works without torch.

All heavy imports are lazy and raise :class:`MissingDependency` with an install hint.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import logging
import math
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from . import ffmpeg_util, models
from .models import models_dir  # noqa: F401  (re-exported for backwards compatibility)
from .schema import DetectedInstance, DuplicateFinding, Shot
from .tracking import ByteTracker, Detection, Track, iou_matrix

log = logging.getLogger("cappycat.perception")

DEFAULT_PROMPTS = ["person", "animal character", "cartoon character"]
ML_INSTALL_HINT = (
    "Install the ML extras: `pip install -e \".[ml]\"` (or `pip install -r requirements-ml.txt`) and run "
    "`python -m cappycat_pipeline download-models`. Torch is a multi-GB download; pick a CUDA build from "
    "https://pytorch.org/get-started/locally/ for GPU inference."
)
GSAM_INSTALL_HINT = (
    "Grounded SAM 2 needs `transformers` (>= 4.56, for Sam2Model) and the Grounding DINO / SAM 2.1 weights: "
    "`pip install transformers accelerate` then `python -m cappycat_pipeline download-models`."
)
OVERLAP_IOU = 0.3           # hybrid: overlap of two possibly-same instances that triggers escalation
BORDERLINE_MARGIN = 0.1     # hybrid: similarity in [threshold - margin, threshold) triggers escalation
TRACK_THRESH = 0.35         # ByteTrack high/low split for open-vocabulary scores (new tracks need +0.1)
ESCALATION_FPS = 2.0        # Grounded SAM 2 sampling rate of an escalated shot (dense tracking fills in)
MIN_DUP_FRAMES = 2          # a duplicate must be seen in this many sampled frames before it is reported
NESTED_MAX = 0.6            # one box this much inside the other = head-inside-body, never a pair
# "group" boxes (one box around several characters) never get an identity / cast tag
GROUP_BOX_AREA = 0.55       # fraction of the frame area
GROUP_BOX_WIDTH, GROUP_BOX_HEIGHT = 0.7, 0.9
# hi-res YOLO-World re-run of sparse shots (small background figures are missed at 640)
HIRES_IMGSZ = 1280
SPARSE_DETS_PER_FRAME = 1.5
# raw score floors kept for the detection cache (= the tracking floors of :class:`low_confidence`)
YOLO_TRACK_CONF = 0.1
GDINO_TRACK_BOX = 0.25
GDINO_MAX_BATCH = 4         # captions per Grounding DINO forward


class MissingDependency(RuntimeError):
    pass


# --------------------------------------------------------------------------- protocols


class Detector(Protocol):
    name: str

    def detect(self, frame_bgr: np.ndarray, prompts: Sequence[str]) -> List[Detection]: ...


class Embedder(Protocol):
    name: str

    def embed(self, frame_bgr: np.ndarray, bbox: Sequence[float]) -> np.ndarray: ...


Progress = Callable[[float, str], None]


# --------------------------------------------------------------------------- helpers


def _snap_label(phrase: str, prompts: Sequence[str]) -> Optional[str]:
    """Map a Grounding DINO phrase back to the prompt it came from."""
    ph = (phrase or "").strip().lower()
    if not ph:
        return None
    for p in prompts:
        if ph == p.lower().strip():
            return p
    for p in prompts:
        pl = p.lower().strip()
        if pl in ph or ph in pl:
            return p
    words = set(ph.split())
    best, best_n = None, 0
    for p in prompts:
        n = len(words & set(p.lower().split()))
        if n > best_n:
            best, best_n = p, n
    return best


def nms(dets: List[Detection], iou_thresh: float = 0.7, class_agnostic: bool = False) -> List[Detection]:
    """Greedy NMS (class-aware by default). Class-agnostic mode keeps the highest-scoring box of
    a cluster, which is what open-vocabulary prompts need: the same character is often returned
    once per matching prompt ("cartoon rabbit" 0.55 and "animal character" 0.34 on one box)."""
    out: List[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if any((class_agnostic or o.label == d.label)
               and iou_matrix(np.array([o.bbox]), np.array([d.bbox]))[0, 0] > iou_thresh for o in out):
            continue
        out.append(d)
    return out


def max_same_label_iou(dets: Sequence[Detection]) -> float:
    best = 0.0
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            if dets[i].label == dets[j].label:
                best = max(best, float(iou_matrix(np.array([dets[i].bbox]), np.array([dets[j].bbox]))[0, 0]))
    return best


def is_oom(exc: BaseException) -> bool:
    """CUDA out-of-memory (``torch.cuda.OutOfMemoryError`` or a RuntimeError saying so)."""
    try:
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return "out of memory" in str(exc).lower()


class DetectionCache:
    """Raw detections (at the tracking score floor) per ``(detector, source frame)``, so frames the
    perception pass already detected are not detected again when a confirmed duplicate is followed
    through its shot. Entries are verified with a checksum of the frame pixels (different decode
    rates of one shot land on the same source frame, but only identical pixels are reused)."""

    def __init__(self, max_entries: int = 4000):
        self._d: "collections.OrderedDict[Tuple[str, int], Tuple[int, float, List[Detection]]]" = collections.OrderedDict()
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0

    @staticmethod
    def checksum(frame: np.ndarray) -> int:
        f = np.ascontiguousarray(frame[::5, ::5])
        return zlib.crc32(f.tobytes()) ^ (frame.shape[0] << 20) ^ frame.shape[1]

    def put(self, det_name: str, frame_idx: int, frame: np.ndarray, floor: float, dets: Sequence[Detection]) -> None:
        key = (det_name, int(frame_idx))
        self._d[key] = (self.checksum(frame), float(floor), [Detection(tuple(d.bbox), d.label, float(d.score)) for d in dets])
        self._d.move_to_end(key)
        while len(self._d) > self.max_entries:
            self._d.popitem(last=False)

    def get(self, det_name: str, frame_idx: int, frame: np.ndarray, threshold: float) -> Optional[List[Detection]]:
        e = self._d.get((det_name, int(frame_idx)))
        if e is None or e[1] > threshold + 1e-9 or e[0] != self.checksum(frame):
            self.misses += 1
            return None
        self.hits += 1
        return [Detection(tuple(d.bbox), d.label, d.score) for d in e[2] if d.score >= threshold]

    def __len__(self) -> int:
        return len(self._d)


def detector_threshold(det) -> Optional[float]:
    """The detector's current score floor (``conf`` / ``box_threshold``)."""
    for k in ("conf", "box_threshold"):
        if hasattr(det, k):
            return float(getattr(det, k))
    return None


# --------------------------------------------------------------------------- detectors


class NullDetector:
    name = "none"

    def detect(self, frame_bgr: np.ndarray, prompts: Sequence[str]) -> List[Detection]:
        return []

    def close(self) -> None:
        pass


def _yolo_text_cache_path(prompts: Sequence[str], weights: str) -> "os.PathLike":
    import os
    from pathlib import Path

    try:
        import ultralytics  # type: ignore

        uv = getattr(ultralytics, "__version__", "?")
    except Exception:
        uv = "?"
    h = hashlib.sha1()
    for part in [*prompts, str(os.path.getsize(weights)) if os.path.isfile(weights) else weights, uv,
                 str(models.clip_text_path().stat().st_size) if models.clip_text_path().is_file() else "-"]:
        h.update(part.encode("utf-8") + b"\0")
    return Path(models.models_dir()) / "cache" / "yolo_world_text" / f"{h.hexdigest()}.npy"


class YoloWorldDetector:
    """ultralytics YOLO-World v2 open-vocabulary detector.

    ``raw_floor`` (default :data:`YOLO_TRACK_CONF`): predictions are made at ``min(conf, raw_floor)``
    and filtered to ``conf``; the unfiltered set is left in ``last_raw`` for the
    :class:`DetectionCache` (NMS is greedy by score, so the filtered set is exactly what a ``conf``
    prediction returns)."""

    name = "yolo_world"

    def __init__(self, weights: Optional[str] = None, conf: float = 0.2, imgsz: int = 640, device: Optional[str] = None,
                 half: Optional[bool] = None, prompts: Optional[Sequence[str]] = None,
                 raw_floor: Optional[float] = YOLO_TRACK_CONF):
        try:
            models.import_ultralytics()
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # ImportError or a broken torch install
            raise MissingDependency(f"ultralytics/torch not available ({exc}). {ML_INSTALL_HINT}") from exc
        if weights is None:
            weights = str(models.yolo_world_path())  # ultralytics downloads it there when missing
        self.weights = str(weights)
        self.device = device or models.device()
        self.half = self.device.startswith("cuda") if half is None else bool(half)
        try:
            self.model = YOLO(weights)
        except Exception as exc:
            raise MissingDependency(f"YOLO-World weights unavailable ({exc}). {ML_INSTALL_HINT}") from exc
        self.conf = conf
        self.imgsz = imgsz
        self.raw_floor = raw_floor
        self.last_raw: Optional[List[Detection]] = None
        self.last_floor: float = conf
        self._classes: Optional[Tuple[str, ...]] = None
        if prompts:
            self._set_classes(tuple(prompts))

    def _set_classes(self, prompts: Tuple[str, ...]) -> None:
        """Set the prompt classes. The CLIP text features are cached on disk per prompt set, so the
        ~350 MB CLIP text encoder (a TorchScript model) is only loaded the first time a prompt set is
        seen; after encoding it is dropped and only the text features are kept."""
        inner = getattr(self.model, "model", None)
        cache = _yolo_text_cache_path(prompts, self.weights)
        feats = None
        try:
            if cache.is_file():
                feats = np.load(cache, allow_pickle=False)
        except Exception:
            feats = None
        if feats is not None and inner is not None and hasattr(inner, "txt_feats") and feats.shape[1] == len(prompts):
            import torch  # type: ignore

            inner.txt_feats = torch.from_numpy(feats.astype(np.float32))
            inner.model[-1].nc = len(prompts)
            names = list(prompts)
            inner.names = names
            try:
                self.model.model.names = names
            except Exception:
                pass
        else:
            self.model.set_classes(list(prompts))
            if inner is not None and getattr(inner, "txt_feats", None) is not None:
                try:
                    from . import fsutil

                    cache.parent.mkdir(parents=True, exist_ok=True)
                    with fsutil.atomic_path(cache, ".npy") as tmp:
                        np.save(tmp, inner.txt_feats.detach().float().cpu().numpy())
                except Exception as exc:  # the cache is an optimisation only
                    log.debug("could not cache YOLO-World text features: %s", exc)
        if inner is not None and getattr(inner, "clip_model", None) is not None:
            inner.clip_model = None
        self.model.predictor = None  # rebuild the predictor so names / dtype follow the new classes
        self._classes = prompts
        models.release()

    def detect(self, frame_bgr: np.ndarray, prompts: Sequence[str], imgsz: Optional[int] = None) -> List[Detection]:
        prompts = tuple(prompts) or tuple(DEFAULT_PROMPTS)
        if prompts != self._classes:
            self._set_classes(prompts)
        floor = min(self.conf, self.raw_floor) if self.raw_floor is not None else self.conf
        res = self.model.predict(frame_bgr, conf=floor, imgsz=int(imgsz or self.imgsz), device=self.device,
                                 half=self.half, verbose=False)[0]
        raw: List[Detection] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.float().cpu().numpy()
            conf = res.boxes.conf.float().cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            for b, s, c in zip(xyxy, conf, cls):
                label = prompts[int(c)] if 0 <= int(c) < len(prompts) else str(c)
                raw.append(Detection((float(b[0]), float(b[1]), float(b[2]), float(b[3])), str(label), float(s)))
        self.last_raw, self.last_floor = raw, floor
        return [Detection(d.bbox, d.label, d.score) for d in raw if d.score >= self.conf]

    def close(self) -> None:
        self.model = None
        models.release()


def gsam_available() -> Tuple[bool, str]:
    """(ok, reason): torch + transformers are installed. Uses ``importlib.util.find_spec`` so it
    does not pay the several-second transformers import at start-up; a broken install is caught
    when the fallback is first built (which then disables escalation for the run)."""
    for mod in ("torch", "transformers"):
        try:
            if importlib.util.find_spec(mod) is None:
                return False, f"{mod} not installed"
        except Exception as exc:  # pragma: no cover
            return False, f"{mod} not importable ({exc})"
    return True, "ok"


class GroundedSam2Detector:
    """Grounding DINO (tiny) boxes refined with SAM 2.1 (hiera-small) masks via ``transformers``.

    fp16 Grounding DINO on CUDA (identical boxes to fp32 on our checks, half the VRAM), SAM 2 in
    fp16 on CUDA as well when ``sam_fp16`` (masks only tighten boxes). Frames are fed at their own
    resolution (short side clamped to 480..800) instead of the processor's 800 px default to bound
    activation memory. One caption per prompt (long multi-phrase captions lower recall a lot), all
    captions of a frame in one batched forward (``batch_prompts``).

    ``raw_floor``: boxes are post-processed at ``min(box_threshold, raw_floor)``; the unfiltered
    (pre-SAM) set is left in ``last_raw`` for the :class:`DetectionCache`."""

    name = "grounded_sam2"

    def __init__(self, box_threshold: float = 0.35, text_threshold: float = 0.25, device: Optional[str] = None,
                 with_masks: bool = True, fp16: Optional[bool] = None, nms_iou: float = 0.7, per_prompt: bool = True,
                 batch_prompts: bool = True, sam_fp16: Optional[bool] = None,
                 raw_floor: Optional[float] = GDINO_TRACK_BOX):
        try:
            import torch  # type: ignore
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # type: ignore
        except Exception as exc:
            raise MissingDependency(f"transformers/torch not available ({exc}). {GSAM_INSTALL_HINT}") from exc

        self.torch = torch
        self.device = device or models.device()
        cuda = self.device.startswith("cuda")
        self.dtype = torch.float16 if (cuda if fp16 is None else fp16) else torch.float32
        self.sam_dtype = torch.float16 if (cuda if sam_fp16 is None else sam_fp16) else torch.float32
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou = nms_iou
        self.per_prompt = per_prompt  # one caption per prompt: long multi-phrase captions lower recall a lot
        self.batch_prompts = batch_prompts
        self.with_masks = with_masks
        self.raw_floor = raw_floor
        self.last_raw: Optional[List[Detection]] = None
        self.last_floor: float = box_threshold
        try:
            self.processor = _hf_load(AutoProcessor, models.GDINO_HF_REPO)
            self.model = _hf_load(AutoModelForZeroShotObjectDetection, models.GDINO_HF_REPO, dtype=self.dtype)
        except Exception as exc:
            raise MissingDependency(f"Grounding DINO weights unavailable ({exc}). {GSAM_INSTALL_HINT}") from exc
        self.model = self.model.to(self.device).eval()
        self.sam = None
        self.sam_processor = None
        self.last_masks: List[np.ndarray] = []

    def _ensure_sam(self) -> bool:
        if self.sam is not None:
            return True
        if not self.with_masks:
            return False
        try:
            from transformers import Sam2Model, Sam2Processor  # type: ignore

            self.sam_processor = _hf_load(Sam2Processor, models.SAM2_HF_REPO)
            self.sam = _hf_load(Sam2Model, models.SAM2_HF_REPO, dtype=self.sam_dtype).to(self.device).eval()
            return True
        except Exception as exc:
            log.warning("SAM 2 unavailable (%s); running Grounding DINO boxes only", exc)
            self.with_masks = False
            return False

    def _forward(self, pil, captions: List[str], size: dict, threshold: float, h: int, w: int) -> List[dict]:
        """Post-processed results for ``captions`` on one image (one forward per chunk)."""
        torch = self.torch
        out: List[dict] = []
        step = GDINO_MAX_BATCH if self.batch_prompts else 1
        for k in range(0, len(captions), step):
            chunk = captions[k:k + step]
            if len(chunk) == 1:
                inputs = self.processor(images=pil, text=chunk[0], return_tensors="pt", size=size).to(self.device)
            else:
                inputs = self.processor(images=[pil] * len(chunk), text=chunk, return_tensors="pt", size=size,
                                        padding=True).to(self.device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
            with torch.inference_mode():
                outputs = self.model(**inputs)
            res = self.processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], threshold=threshold, text_threshold=self.text_threshold,
                target_sizes=[(h, w)] * len(chunk))
            for r in res:
                out.append({"labels": r["text_labels"] if "text_labels" in r else r.get("labels"),
                            "boxes": r["boxes"].float().cpu().numpy(), "scores": r["scores"].float().cpu().numpy()})
            del outputs, inputs
        return out

    def detect(self, frame_bgr: np.ndarray, prompts: Sequence[str]) -> List[Detection]:
        import cv2
        from PIL import Image  # type: ignore

        torch = self.torch
        prompts = [p for p in (list(prompts) or list(DEFAULT_PROMPTS)) if p.strip()]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        pil = Image.fromarray(rgb)
        short = int(min(800, max(480, min(h, w))))
        size = {"shortest_edge": short, "longest_edge": int(round(short * 1333 / 800))}
        groups = [[p] for p in prompts] if self.per_prompt else [prompts]
        captions = [". ".join(p.strip().lower().rstrip(".") for p in g) + "." for g in groups]
        floor = min(self.box_threshold, self.raw_floor) if self.raw_floor is not None else self.box_threshold
        raw: List[Detection] = []
        for group, res in zip(groups, self._forward(pil, captions, size, floor, h, w)):
            for b, s, ph in zip(res["boxes"], res["scores"], res["labels"]):
                lab = group[0] if len(group) == 1 else _snap_label(str(ph), group)
                if lab is None:
                    continue
                x1, y1, x2, y2 = (float(max(0.0, b[0])), float(max(0.0, b[1])), float(min(w, b[2])), float(min(h, b[3])))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                raw.append(Detection((x1, y1, x2, y2), lab, float(s)))
        raw = nms(raw, self.nms_iou)
        self.last_raw, self.last_floor = raw, floor
        dets = [Detection(d.bbox, d.label, d.score) for d in raw if d.score >= self.box_threshold]
        self.last_masks = []
        if dets and self._ensure_sam():
            sam_in = self.sam_processor(images=pil, input_boxes=[[list(d.bbox) for d in dets]], return_tensors="pt").to(self.device)
            if "pixel_values" in sam_in:
                sam_in["pixel_values"] = sam_in["pixel_values"].to(self.sam_dtype)
            with torch.inference_mode():
                sam_out = self.sam(**sam_in, multimask_output=False)
            masks = self.sam_processor.post_process_masks(sam_out.pred_masks.float().cpu(), sam_in["original_sizes"].cpu())[0]
            del sam_out, sam_in
            for d, m in zip(dets, masks):
                mask = np.asarray(m.squeeze(0).numpy() if hasattr(m, "numpy") else m).astype(bool).reshape(h, w)
                d.mask = mask
                d.bbox = _tight_bbox(mask, d.bbox)
                self.last_masks.append(mask)
        return dets

    def close(self) -> None:
        self.model = self.sam = None
        models.release()


def _hf_load(cls, repo: str, **kw):
    """``from_pretrained`` from the local cache first (no network round-trip); online only when the
    local copy is incomplete and ``HF_HUB_OFFLINE`` is not set."""
    import os

    cache = str(models.hf_hub_cache())
    try:
        return cls.from_pretrained(repo, cache_dir=cache, local_files_only=True, **kw)
    except Exception:
        if os.environ.get("HF_HUB_OFFLINE", "").strip() in ("1", "true", "True", "YES", "yes"):
            raise
        return cls.from_pretrained(repo, cache_dir=cache, **kw)


def _tight_bbox(mask: np.ndarray, box: Tuple[float, float, float, float], min_fill: float = 0.15,
                slack: float = 0.05) -> Tuple[float, float, float, float]:
    """Box around the mask pixels inside the (slightly expanded) detection box; the original box
    when the mask is empty / implausibly small."""
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    ex1, ey1 = int(max(0, math.floor(x1 - slack * bw))), int(max(0, math.floor(y1 - slack * bh)))
    ex2, ey2 = int(min(w, math.ceil(x2 + slack * bw))), int(min(h, math.ceil(y2 + slack * bh)))
    sub = mask[ey1:ey2, ex1:ex2]
    if sub.size == 0 or sub.sum() < min_fill * max(bw * bh, 1.0):
        return box
    ys, xs = np.nonzero(sub)
    return (float(ex1 + xs.min()), float(ey1 + ys.min()), float(ex1 + xs.max() + 1), float(ey1 + ys.max() + 1))


class HybridDetector:
    """YOLO-World by default; shots are escalated to Grounded SAM 2 by :func:`analyze_shot`.

    ``detect`` (single frame, e.g. for the AI director) always uses the primary. The fallback is
    built lazily on the first escalation (``fallback_factory``) so its VRAM is only spent when a
    shot needs it; a failing factory (or an OOM / error during an escalated pass, see
    :meth:`disable_fallback`) disables escalation for the rest of the run."""

    name = "hybrid"

    def __init__(self, primary: Detector, fallback: Optional[Detector] = None,
                 fallback_factory: Optional[Callable[[], Detector]] = None, overlap_iou: float = OVERLAP_IOU,
                 borderline_margin: float = BORDERLINE_MARGIN):
        self.primary = primary
        self._fallback = fallback
        self._factory = fallback_factory
        self.fallback_error: Optional[str] = None
        self.overlap_iou = overlap_iou
        self.borderline_margin = borderline_margin

    @property
    def fallback_possible(self) -> bool:
        if self.fallback_error is not None:
            return False
        return self._fallback is not None or self._factory is not None

    def get_fallback(self) -> Optional[Detector]:
        if self.fallback_error is not None:
            return None
        if self._fallback is None and self._factory is not None:
            try:
                self._fallback = self._factory()
            except Exception as exc:
                self.fallback_error = str(exc)
                log.warning("Grounded SAM 2 fallback unavailable: %s", exc)
                models.release()
        return self._fallback

    def disable_fallback(self, reason: str) -> None:
        """Stop escalating for the rest of the run and free the fallback's VRAM."""
        self.fallback_error = reason
        fb, self._fallback = self._fallback, None
        if fb is not None and hasattr(fb, "close"):
            try:
                fb.close()
            except Exception:
                pass
        del fb
        models.release()

    def escalation_reason(self, stats: "ShotStats", threshold: float) -> Optional[str]:
        if stats.frames > 0 and stats.detections == 0:
            return "YOLO-World found nothing for the prompts"
        if stats.max_ambiguous_iou > self.overlap_iou:
            return (f"possibly-same instances overlap (IoU {stats.max_ambiguous_iou:.2f} > {self.overlap_iou}: "
                    f"{stats.ambiguous_pair or 'unidentified'})")
        if stats.borderline > 0:
            return (f"borderline similarity {stats.max_borderline_sim:.3f} in "
                    f"[{threshold - self.borderline_margin:.2f}, {threshold:.2f})")
        return None

    def detect(self, frame_bgr: np.ndarray, prompts: Sequence[str]) -> List[Detection]:
        return self.primary.detect(frame_bgr, prompts)

    def close(self) -> None:
        for d in (self.primary, self._fallback):
            if d is not None and hasattr(d, "close"):
                d.close()
        self._fallback = None


def build_detector(kind: str, prompts: Sequence[str]) -> Tuple[Detector, List[str]]:
    """Instantiate the requested detector, degrading ``grounded_sam2 -> yolo_world -> none`` and
    ``hybrid -> yolo_world -> none``. Returns ``(detector, warnings)``."""
    kind = (kind or "hybrid").lower()
    warnings: List[str] = []
    if kind == "none":
        return NullDetector(), warnings
    if kind == "grounded_sam2":
        try:
            return GroundedSam2Detector(), warnings
        except MissingDependency as exc:
            warnings.append(f"grounded_sam2 unavailable: {exc}")
            kind = "yolo_world"
    if kind in ("hybrid", "yolo_world"):
        try:
            yolo = YoloWorldDetector(prompts=list(prompts) or None)
        except MissingDependency as exc:
            warnings.append(f"yolo_world unavailable: {exc}; running in lite mode (no duplicate detection)")
            return NullDetector(), warnings
        if kind == "yolo_world":
            return yolo, warnings
        ok, why = gsam_available()
        if not ok:
            warnings.append(f"hybrid: Grounded SAM 2 fallback unavailable ({why}); using YOLO-World only")
            return HybridDetector(yolo), warnings
        if not models.hf_repo_present(models.GDINO_HF_REPO, "model.safetensors"):
            warnings.append("hybrid: Grounding DINO weights not in models/hf (run download-models); "
                            "they will be fetched on the first escalation")
        return HybridDetector(yolo, fallback_factory=GroundedSam2Detector), warnings
    warnings.append(f"unknown detector '{kind}'; running in lite mode")
    return NullDetector(), warnings


def tracking_detector(detector: Detector, shot_path: str) -> Detector:
    """The detector to follow a confirmed duplicate with: the one that confirmed it (the hybrid's
    Grounded SAM 2 fallback when the shot was escalated, else its YOLO-World primary)."""
    if isinstance(detector, HybridDetector):
        if shot_path == "grounded_sam2":
            fb = detector.get_fallback()
            if fb is not None:
                return fb
        return detector.primary
    return detector


class low_confidence:
    """Context manager: temporarily lower a detector's score floor for tracking (a character
    that was confirmed once may score low for a while, e.g. while the camera pulls back) and turn
    SAM masks off (they only tighten boxes; the tracker does not need them)."""

    def __init__(self, det: Detector, yolo_conf: float = YOLO_TRACK_CONF, gdino_box: float = GDINO_TRACK_BOX,
                 masks: bool = False):
        self.det, self.saved = det, {}
        self.want = {"conf": yolo_conf, "box_threshold": gdino_box}
        self.masks = masks

    def __enter__(self):
        for k, v in self.want.items():
            if hasattr(self.det, k):
                self.saved[k] = getattr(self.det, k)
                setattr(self.det, k, min(v, getattr(self.det, k)))
        if hasattr(self.det, "with_masks"):
            self.saved["with_masks"] = getattr(self.det, "with_masks")
            if not self.masks:
                setattr(self.det, "with_masks", False)
        return self.det

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(self.det, k, v)
        return False


def close_detector(det: Optional[Detector]) -> None:
    if det is not None and hasattr(det, "close"):
        try:
            det.close()  # type: ignore[attr-defined]
        except Exception:
            pass
    models.release()


# --------------------------------------------------------------------------- embedders


def _crop(frame: np.ndarray, bbox: Sequence[float], pad: float = 0.05) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    x1, x2 = int(max(0, math.floor(x1 - pad * bw))), int(min(w, math.ceil(x2 + pad * bw)))
    y1, y2 = int(max(0, math.floor(y1 - pad * bh))), int(min(h, math.ceil(y2 + pad * bh)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((8, 8, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2]


class HistogramEmbedder:
    """HSV colour histogram (8x8x4) + 2x2-cell gradient orientation histogram (9 bins), L2-normalised."""

    name = "histogram"

    def __init__(self, size: int = 64):
        self.size = size

    def embed(self, frame_bgr: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
        import cv2

        patch = cv2.resize(_crop(frame_bgr, bbox), (self.size, self.size), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-9)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)  # 0..1
        hog = []
        half = self.size // 2
        for cy in (0, half):
            for cx in (0, half):
                a = ang[cy:cy + half, cx:cx + half].ravel()
                m = mag[cy:cy + half, cx:cx + half].ravel()
                hh, _ = np.histogram(a, bins=9, range=(0.0, 1.0), weights=m)
                hog.append(hh / (hh.sum() + 1e-9))
        vec = np.concatenate([hist * 1.0, np.concatenate(hog) * 0.5]).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)

    def embed_many(self, frame_bgr: np.ndarray, bboxes: Sequence[Sequence[float]]) -> np.ndarray:
        return np.stack([self.embed(frame_bgr, b) for b in bboxes]) if len(bboxes) else np.zeros((0, 1), np.float32)

    def close(self) -> None:
        pass


class OpenClipEmbedder:
    """OpenCLIP ViT-B-32 (laion2b_s34b_b79k) image tower, fp16 on CUDA, one batch per frame."""

    name = "open_clip"

    def __init__(self, model_name: str = models.OPENCLIP_MODEL, pretrained: str = models.OPENCLIP_PRETRAINED,
                 device: Optional[str] = None, fp16: Optional[bool] = None, keep_text: bool = False):
        try:
            import open_clip  # type: ignore
            import torch  # type: ignore
        except Exception as exc:
            raise MissingDependency(f"open_clip/torch not available ({exc}). {ML_INSTALL_HINT}") from exc
        self.torch = torch
        self.device = device or models.device()
        use_fp16 = self.device.startswith("cuda") if fp16 is None else bool(fp16)
        if pretrained == models.OPENCLIP_PRETRAINED and model_name == models.OPENCLIP_MODEL:
            snaps = models.hf_hub_cache() / ("models--" + models.OPENCLIP_HF_REPO.replace("/", "--")) / "snapshots"
            local = sorted(snaps.glob("*/open_clip_model.safetensors")) if snaps.is_dir() else []
            if local:  # cached: load the file directly (no hub round-trip on every run)
                pretrained = str(local[0])
        try:
            model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device, precision="fp16" if use_fp16 else "fp32",
                cache_dir=str(models.hf_hub_cache()))
        except Exception as exc:
            raise MissingDependency(f"OpenCLIP weights unavailable ({exc}). {ML_INSTALL_HINT}") from exc
        self.tokenizer = open_clip.get_tokenizer(model_name) if keep_text else None
        if not keep_text:
            # only the image tower is needed: drop the text transformer (~40% of the weights)
            for attr in ("transformer", "token_embedding", "ln_final"):
                if hasattr(model, attr):
                    setattr(model, attr, None)
        self.model = model.eval()
        self.dtype = torch.float16 if use_fp16 else torch.float32
        models.release()

    def embed_images(self, images_bgr: Sequence[np.ndarray]) -> np.ndarray:
        """L2-normalised embeddings of whole images (already-cropped patches)."""
        import cv2
        from PIL import Image  # type: ignore

        if len(images_bgr) == 0:
            return np.zeros((0, 512), dtype=np.float32)
        x = self.torch.stack([self.preprocess(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
                              for im in images_bgr]).to(self.device, self.dtype)
        with self.torch.inference_mode():
            f = self.model.encode_image(x).float()
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy()

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """L2-normalised text embeddings (needs ``keep_text=True``)."""
        if self.tokenizer is None:
            raise RuntimeError("OpenClipEmbedder was built without its text tower (keep_text=False)")
        if not len(texts):
            return np.zeros((0, 512), dtype=np.float32)
        with self.torch.inference_mode():
            t = self.model.encode_text(self.tokenizer(list(texts)).to(self.device)).float()
            t = t / t.norm(dim=-1, keepdim=True)
        return t.cpu().numpy()

    def zero_shot(self, images_bgr: Sequence[np.ndarray], texts: Sequence[str]) -> np.ndarray:
        """Softmax probabilities (n_images, n_texts); needs ``keep_text=True``."""
        if self.tokenizer is None:
            raise RuntimeError("OpenClipEmbedder was built without its text tower (keep_text=False)")
        img = self.embed_images(images_bgr)
        t = self.embed_texts(texts)
        logits = 100.0 * img @ t.T
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def embed_many(self, frame_bgr: np.ndarray, bboxes: Sequence[Sequence[float]]) -> np.ndarray:
        return self.embed_images([_crop(frame_bgr, b) for b in bboxes])

    def embed(self, frame_bgr: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
        return self.embed_many(frame_bgr, [bbox])[0]

    def close(self) -> None:
        self.model = None
        models.release()


def build_embedder(prefer_clip: bool = True) -> Tuple[Embedder, Optional[str]]:
    if prefer_clip:
        try:
            return OpenClipEmbedder(), None
        except MissingDependency as exc:
            return HistogramEmbedder(), f"open_clip unavailable ({exc.args[0].split('.')[0]}); using HSV/gradient histogram embeddings"
    return HistogramEmbedder(), None


def _embed_all(embedder: Embedder, frame: np.ndarray, bboxes: Sequence[Sequence[float]]) -> List[np.ndarray]:
    if hasattr(embedder, "embed_many"):
        return list(embedder.embed_many(frame, bboxes))  # type: ignore[attr-defined]
    return [embedder.embed(frame, b) for b in bboxes]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def is_group_box(bbox: Sequence[float], w: float, h: float) -> bool:
    """One box around several characters (or the whole scene): > 55 % of the frame area, or
    (nearly) full height and > 70 % of the width. Such boxes get no identity / cast tag (on the
    real clips a "person" box over a deer, a turtle and an elephant identified as Suzie)."""
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return bw * bh > GROUP_BOX_AREA * w * h or (bh >= GROUP_BOX_HEIGHT * h and bw > GROUP_BOX_WIDTH * w)


def identify_detections(bank, embedder: Embedder, frame: np.ndarray, dets: Sequence[Detection],
                        embs: Sequence[np.ndarray]) -> List:
    """Identity per detection: full box + head crop for tall boxes (per-character max), none for
    group boxes (:func:`is_group_box`)."""
    from .characters import Identity, head_box, is_tall

    h, w = frame.shape[:2]
    group = [is_group_box(d.bbox, w, h) for d in dets]
    tall = [i for i, d in enumerate(dets) if is_tall(d.bbox) and not group[i]]
    heads: List[Optional[np.ndarray]] = [None] * len(dets)
    for i, e in zip(tall, _embed_all(embedder, frame, [head_box(dets[i].bbox) for i in tall]) if tall else []):
        heads[i] = e
    out = []
    for i, (e, hd) in enumerate(zip(embs, heads)):
        out.append(Identity(None, 0.0, 0.0, None, "group box") if group[i] else bank.identify_one(e, head_emb=hd))
    return out


# --------------------------------------------------------------------------- frame sampling


def sample_shot_frames(path: str, shot: Shot, src_fps: float, src_size: Tuple[int, int], sample_fps: float = 4.0,
                       width: int = 640) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(absolute_source_frame_index, frame_bgr)`` at ~``sample_fps`` for one shot."""
    src_fps = float(src_fps) if src_fps > 0 else 25.0
    # real frame times: in variable-frame-rate sources index / fps is tens of ms off, which would
    # start the sampling on the previous shot's last frame
    clock = ffmpeg_util.frame_clock(path, src_fps)
    start_ms = clock.seek_ms(shot.startFrame)
    end_ms = clock.seek_ms(shot.endFrame + 1)
    for k, frame in enumerate(ffmpeg_util.iter_frames(path, start_ms, end_ms, fps=sample_fps, width=width, src_size=src_size)):
        idx = int(min(shot.endFrame, max(shot.startFrame, clock.nearest(start_ms + k * 1000.0 / sample_fps))))
        yield idx, frame


def expected_samples(shot: Shot, src_fps: float, sample_fps: float) -> int:
    """How many frames :func:`sample_shot_frames` yields for ``shot`` (for progress reporting)."""
    src_fps = float(src_fps) if src_fps > 0 else 25.0
    dur = (shot.endFrame + 1 - shot.startFrame) / src_fps
    return max(1, int(math.ceil(dur * sample_fps - 1e-6)))


# --------------------------------------------------------------------------- duplicate finding


@dataclass
class _Candidate:
    frame: int
    a: DetectedInstance
    b: DetectedInstance
    similarity: float
    character: Optional[str] = None


@dataclass
class ShotStats:
    """What a detector saw in one shot (drives the hybrid escalation and ``Shot.cast``)."""

    frames: int = 0
    detections: int = 0
    max_same_label_iou: float = 0.0  # max IoU between label-compatible instances (after NMS), for logs
    max_ambiguous_iou: float = 0.0   # the same, only pairs that may be one character (drives escalation)
    ambiguous_pair: str = ""         # which pair that was (for the log)
    pairs: int = 0                 # compatible, similar-scale unnamed pairs compared
    borderline: int = 0            # pairs with threshold - margin <= sim < threshold
    max_borderline_sim: float = 0.0
    max_sim: float = 0.0
    masks: int = 0                 # detections that carried a segmentation mask
    cast_frames: Dict[str, int] = field(default_factory=dict)  # character id -> sampled frames seen in
    track_cast: Dict[int, Dict[str, int]] = field(default_factory=dict)  # track id -> {character: loose hits}
    named_pairs: int = 0           # same-character pairs found (named duplicate rule)
    dropped_single_frame: int = 0  # duplicate candidates seen in fewer than MIN_DUP_FRAMES frames
    rejected: int = 0              # detections refused a cast identity (open-set / group box)
    cache_puts: int = 0

    def loose_cast(self, min_hits: int = 0) -> List[str]:
        """Characters carried along a ByteTrack track by the looser cast rule (>= ``min_hits`` frames
        of the track, and the majority of the track's loose hits)."""
        from .characters import CAST_MIN_TRACK_FRAMES

        need = min_hits or CAST_MIN_TRACK_FRAMES
        out = set()
        for per in self.track_cast.values():
            if not per:
                continue
            c, n = max(per.items(), key=lambda kv: kv[1])
            if n >= need and n * 2 > sum(per.values()):
                out.add(c)
        return sorted(out)


CHARACTER_LIKE_MARGIN = 0.12  # generic rule with a bank: best cast score >= id_threshold - this
PSEUDO_ID_BASE = 100000  # ids for detections without a confirmed track: PSEUDO_ID_BASE + 8x8 grid cell


def _assign_ids(dets: Sequence[Detection], tracks: Sequence[Track], w: int, h: int) -> List[int]:
    """Stable id per detection: its ByteTrack track (IoU >= 0.5, greedy), else a position-derived
    pseudo id so findings of an unconfirmed instance still de-duplicate across frames."""
    ids = [-1] * len(dets)
    if tracks and dets:
        iou = iou_matrix(np.array([d.bbox for d in dets]), np.array([t.bbox for t in tracks]))
        used = set()
        for flat in np.argsort(-iou, axis=None):
            i, j = divmod(int(flat), len(tracks))
            if iou[i, j] < 0.5:
                break
            if ids[i] >= 0 or j in used:
                continue
            ids[i] = tracks[j].track_id
            used.add(j)
    for i, d in enumerate(dets):
        if ids[i] < 0:
            cx = (d.bbox[0] + d.bbox[2]) / 2.0 / max(w, 1)
            cy = (d.bbox[1] + d.bbox[3]) / 2.0 / max(h, 1)
            ids[i] = PSEUDO_ID_BASE + int(min(7, max(0, cy * 8))) * 8 + int(min(7, max(0, cx * 8)))
    return ids


def _containment(a: Sequence[float], b: Sequence[float]) -> float:
    """Fraction of ``a``'s area inside ``b``."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


def _head_in_body(a: Sequence[float], b: Sequence[float]) -> bool:
    """One box (at most half the other's area) almost entirely inside the other: a head / bust box
    inside a body box of the same character, not two overlapping characters."""
    aa = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    small, big = (a, b) if aa <= ab else (b, a)
    return min(aa, ab) <= 0.5 * max(aa, ab) and _containment(small, big) >= 0.8


def _confidently_different(ia, ib) -> bool:
    """Two different cast members: different best guesses and both at least cast-like (the looser
    cast-tag threshold), or two different confident identities."""
    from .characters import CAST_THRESHOLD

    if ia is None or ib is None:
        return False
    if ia.character and ib.character:
        return ia.character != ib.character
    return (bool(ia.best) and bool(ib.best) and ia.best != ib.best and not ia.rejected and not ib.rejected
            and min(ia.score, ib.score) >= CAST_THRESHOLD)


def _describe(d: Detection, ident, h: int) -> str:
    who = (ident.character or (f"{ident.best}?" if ident.best else "?")) if ident is not None else d.label
    sc = f" {ident.score:.2f}" if ident is not None and ident.best else ""
    return f"{who}{sc} ({(d.bbox[3] - d.bbox[1]) / max(h, 1):.0%} tall)"


def find_duplicates(frames_iter: Iterable[Tuple[int, np.ndarray]], shot: Shot, detector: Detector, tracker: ByteTracker,
                    embedder: Embedder, threshold: float = 0.85, prompts: Sequence[str] = (), src_fps: float = 25.0,
                    scale: float = 1.0, area_ratio: Tuple[float, float] = (0.5, 2.0),
                    embedding_momentum: float = 0.0, keep_all: bool = False, stats: Optional[ShotStats] = None,
                    borderline_margin: float = BORDERLINE_MARGIN, bank=None, nms_iou: float = 0.6,
                    pair_iou_max: float = OVERLAP_IOU, nested_max: float = NESTED_MAX,
                    named_min_sim: Optional[float] = None, min_height_frac: float = 0.15,
                    character_like_gate: bool = True, min_frames: int = MIN_DUP_FRAMES,
                    cache: Optional[DetectionCache] = None, on_frame: Optional[Callable[[int], None]] = None,
                    detect_kw: Optional[dict] = None) -> List[DuplicateFinding]:
    """Per sampled frame: detect, class-agnostic NMS (one box per instance even when several
    prompts fire), embed every detection, and test every pair of instances *in that frame*:

    * **named rule** (``bank`` = :class:`characters.CharacterBank`): two detections identified as
      the same unique main-cast character whose mutual similarity is >= ``named_min_sim``
      (default ``threshold``) -> finding (``character`` set); no label / scale gate; primary =
      higher identity score, tie -> nearer the frame centre;
    * **generic rule** (instances not identified as cast members, or no bank): label-compatible
      (same prompt, or either is a generic prompt like "animal character"), both at least
      ``min_height_frac`` of the frame tall, similar scale (area ratio in ``area_ratio``), cosine
      similarity >= ``threshold`` and - with a bank - both "character-like" (best cast score >=
      ``id_threshold - CHARACTER_LIKE_MARGIN``); primary = nearer the centre.

    A duplicate (per character / track pair) is only reported when it was seen in at least
    ``min_frames`` sampled frames (one misidentified pair must not trigger dense tracking).

    With a bank, tall (full-body) detections are also embedded by their top 45 % (head) and
    identified by the per-character max of the two scores; group boxes (:func:`is_group_box`) get
    no identity. Cast bookkeeping: ``stats.cast_frames`` (confident identities per frame) and
    ``stats.track_cast`` (the looser cast rule per ByteTrack track).

    Pairs that overlap (IoU > ``pair_iou_max``) or are nested (one box mostly inside the other,
    e.g. head inside body) are never duplicates. Overlapping pairs that may be one character
    (not confidently two different cast members, not nested) set ``stats.max_ambiguous_iou``,
    which drives the hybrid escalation. The tracker only provides stable ids; detections without a
    confirmed track get a position-derived id. ``frames_iter`` yields ``(source_frame_index,
    frame_bgr)``; ``scale`` is ``frame_width / source_width``. ``cache`` receives the detector's raw
    low-threshold detections per frame; ``on_frame(n_done)`` is called after every frame.
    """
    from .characters import labels_compatible

    src_fps = float(src_fps) if src_fps > 0 else 25.0
    inv = 1.0 / float(scale) if scale else 1.0
    stats = stats if stats is not None else ShotStats()
    running: Dict[int, np.ndarray] = {}
    candidates: Dict[Tuple, List[_Candidate]] = {}
    char_like_floor = (bank.id_threshold - CHARACTER_LIKE_MARGIN) if bank is not None else None
    gate_floor = char_like_floor if character_like_gate else None
    unique_ids = {c.id for c in bank.manifest.characters if c.unique} if bank is not None else set()
    named_min_sim = threshold if named_min_sim is None else named_min_sim
    det_name = getattr(detector, "name", "?")
    n_done = 0

    def inst(tid: int, d: Detection) -> DetectedInstance:
        return DetectedInstance(tid, d.label, [round(v * inv, 2) for v in d.bbox], round(d.score, 4))

    for frame_idx, frame in frames_iter:
        h, w = frame.shape[:2]
        raw = detector.detect(frame, prompts, **(detect_kw or {}))
        if cache is not None and getattr(detector, "last_raw", None) is not None:
            cache.put(det_name, frame_idx, frame, getattr(detector, "last_floor", 0.0), detector.last_raw)
            stats.cache_puts += 1
        stats.frames += 1
        stats.masks += sum(1 for d in raw if d.mask is not None)
        for d in raw:
            d.mask = None  # masks already tightened the bbox; do not keep them alive
        dets = nms(raw, nms_iou, class_agnostic=True)
        stats.detections += len(dets)
        tracks: List[Track] = tracker.update(dets)
        n_done += 1
        if not dets:
            if on_frame:
                on_frame(n_done)
            continue
        ids = _assign_ids(dets, tracks, w, h)

        def smooth(store: Dict[int, np.ndarray], tid: int, e: np.ndarray) -> np.ndarray:
            # off by default: adjacent, moving characters swap ByteTrack ids often enough that a
            # running embedding blends two characters (seen on the real clips: a turtle next to the
            # raccoon identified as "Raccoon"); per-frame decisions + per-shot dedupe are sturdier
            if embedding_momentum <= 0.0:
                return e
            if tid < PSEUDO_ID_BASE:  # smooth along a real track only
                prev = store.get(tid)
                if prev is not None and prev.shape == e.shape:
                    e = embedding_momentum * prev + (1.0 - embedding_momentum) * e
                    e = e / (np.linalg.norm(e) + 1e-9)
                store[tid] = e
            return e

        embs = [smooth(running, t, e) for t, e in zip(ids, _embed_all(embedder, frame, [d.bbox for d in dets]))]
        idents: List = [None] * len(dets)
        best_cast = [1.0] * len(dets)  # "looks like a cast-style character" score (1.0 without a bank)
        if bank is not None:
            idents = identify_detections(bank, embedder, frame, dets, embs)
            best_cast = [ident.score for ident in idents]
            stats.rejected += sum(1 for ident in idents if ident.rejected)
            for tid, ident in zip(ids, idents):
                lc = ident.loose()
                if tid < PSEUDO_ID_BASE and lc:
                    per = stats.track_cast.setdefault(tid, {})
                    per[lc] = per.get(lc, 0) + 1
        seen_chars = {i.character for i in idents if i is not None and i.character}
        for c in seen_chars:
            stats.cast_frames[c] = stats.cast_frames.get(c, 0) + 1
        if len(dets) < 2:
            if on_frame:
                on_frame(n_done)
            continue
        boxes = np.array([d.bbox for d in dets])
        iou = iou_matrix(boxes, boxes)
        cx0, cy0 = w / 2.0, h / 2.0

        def centre_dist(d: Detection) -> float:
            return math.hypot((d.bbox[0] + d.bbox[2]) / 2 - cx0, (d.bbox[1] + d.bbox[3]) / 2 - cy0)

        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                da, db = dets[i], dets[j]
                ia, ib = idents[i], idents[j]
                nested = max(_containment(da.bbox, db.bbox), _containment(db.bbox, da.bbox)) > nested_max
                if labels_compatible(da.label, db.label):
                    stats.max_same_label_iou = max(stats.max_same_label_iou, float(iou[i, j]))
                    # may be one character twice: not a head inside its body, not two different cast
                    # members, and both big enough to matter (tiny background figures never escalate)
                    if (float(iou[i, j]) > stats.max_ambiguous_iou and not _head_in_body(da.bbox, db.bbox)
                            and not _confidently_different(ia, ib)
                            and min(da.bbox[3] - da.bbox[1], db.bbox[3] - db.bbox[1]) >= min_height_frac * h):
                        stats.max_ambiguous_iou = float(iou[i, j])
                        stats.ambiguous_pair = f"{_describe(da, ia, h)} / {_describe(db, ib, h)}"
                if iou[i, j] > pair_iou_max or nested:
                    continue
                sim = cosine(embs[i], embs[j])
                if (ia is not None and ia.character) or (ib is not None and ib.character):
                    # named rule; a named instance never pairs with an unnamed extra or another name
                    # duplicate artifacts are copies: the pair must also look alike (two same-species
                    # characters, e.g. the young and the elderly turtle, identify alike but differ)
                    if ia.character == ib.character and ia.character in unique_ids and sim >= named_min_sim:
                        stats.named_pairs += 1
                        if abs(ia.score - ib.score) > 1e-6:
                            first = ia.score > ib.score
                        else:
                            first = centre_dist(da) <= centre_dist(db)
                        (pi, pd), (di, dd) = ((ids[i], da), (ids[j], db)) if first else ((ids[j], db), (ids[i], da))
                        candidates.setdefault(("char", ia.character), []).append(
                            _Candidate(frame_idx, inst(pi, pd), inst(di, dd), round(sim, 4), ia.character))
                    continue
                if ia is not None and ib is not None and ia.best and ib.best and not ia.rejected and not ib.rejected:
                    both_like = char_like_floor is not None and min(ia.score, ib.score) >= char_like_floor
                    if both_like and ia.best != ib.best:
                        continue  # most likely two different cast members, just under the id threshold
                    if (both_like and ia.best == ib.best and ia.best in unique_ids and sim >= named_min_sim
                            and min(da.bbox[3] - da.bbox[1], db.bbox[3] - db.bbox[1]) >= min_height_frac * h):
                        # same best guess on both sides + near-identical appearance: name it (the pair is
                        # its own evidence even when each instance alone is below the id threshold)
                        stats.named_pairs += 1
                        first = ia.score > ib.score if abs(ia.score - ib.score) > 1e-6 else centre_dist(da) <= centre_dist(db)
                        (pi, pd), (di, dd) = ((ids[i], da), (ids[j], db)) if first else ((ids[j], db), (ids[i], da))
                        candidates.setdefault(("char", ia.best), []).append(
                            _Candidate(frame_idx, inst(pi, pd), inst(di, dd), round(sim, 4), ia.best))
                        continue
                if not labels_compatible(da.label, db.label):
                    continue
                # generic rule = main subjects only: not tiny background figures, and (with a cast
                # bank) instances that look like cast-style characters, not birds / paws / props
                if min(da.bbox[3] - da.bbox[1], db.bbox[3] - db.bbox[1]) < min_height_frac * h:
                    continue
                if gate_floor is not None and min(best_cast[i], best_cast[j]) < gate_floor:
                    continue
                aa = (da.bbox[2] - da.bbox[0]) * (da.bbox[3] - da.bbox[1])
                ab = (db.bbox[2] - db.bbox[0]) * (db.bbox[3] - db.bbox[1])
                if aa <= 0 or ab <= 0 or not (area_ratio[0] <= aa / ab <= area_ratio[1]):
                    continue
                stats.pairs += 1
                stats.max_sim = max(stats.max_sim, sim)
                if sim < threshold:
                    if sim >= threshold - borderline_margin:
                        stats.borderline += 1
                        stats.max_borderline_sim = max(stats.max_borderline_sim, sim)
                    continue
                first = centre_dist(da) < centre_dist(db) or (abs(centre_dist(da) - centre_dist(db)) < 1e-6 and aa >= ab)
                (pi, pd), (di, dd) = ((ids[i], da), (ids[j], db)) if first else ((ids[j], db), (ids[i], da))
                key = ("pair", min(ids[i], ids[j]), max(ids[i], ids[j]))
                candidates.setdefault(key, []).append(_Candidate(frame_idx, inst(pi, pd), inst(di, dd), round(sim, 4)))
        if on_frame:
            on_frame(n_done)

    names = {c.id: c.name for c in bank.manifest.characters} if bank is not None else {}
    findings: List[DuplicateFinding] = []
    for key, cands in candidates.items():
        cands.sort(key=lambda c: c.frame)
        if len({c.frame for c in cands}) < max(1, min_frames):
            stats.dropped_single_frame += 1
            continue
        chosen = cands if keep_all else [cands[len(cands) // 2]]
        for c in chosen:
            findings.append(DuplicateFinding(
                shotIndex=shot.index, frame=int(c.frame), timeMs=round(c.frame / src_fps * 1000.0, 3),
                primary=c.a, duplicate=c.b, similarity=c.similarity, character=c.character,
                characterName=names.get(c.character) if c.character else None,
            ))
    findings.sort(key=lambda f: (f.frame, f.primary.trackId))
    return harmonize_roles(findings)


def harmonize_roles(findings: List[DuplicateFinding]) -> List[DuplicateFinding]:
    """Keep the primary on the same side for every frame of one duplicate (per shot + character,
    or per shot + track pair): per-frame identity scores fluctuate, and a primary that flips
    between the left and the right copy would make the reframe solver alternate between two
    opposite crops. The majority side wins (ties: the side whose primary is larger on average)."""
    groups: Dict[Tuple, List[DuplicateFinding]] = {}
    for f in findings:
        key = (f.shotIndex, "char", f.character) if f.character else \
            (f.shotIndex, "pair", min(f.primary.trackId, f.duplicate.trackId), max(f.primary.trackId, f.duplicate.trackId))
        groups.setdefault(key, []).append(f)

    def cx(b: Sequence[float]) -> float:
        return (b[0] + b[2]) / 2.0

    def area(b: Sequence[float]) -> float:
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    for g in groups.values():
        left = [f for f in g if cx(f.primary.bbox) < cx(f.duplicate.bbox)]
        right = [f for f in g if cx(f.primary.bbox) >= cx(f.duplicate.bbox)]
        if not left or not right:
            continue
        if len(left) != len(right):
            keep_left = len(left) > len(right)
        else:
            keep_left = np.mean([area(f.primary.bbox) for f in left]) >= np.mean([area(f.primary.bbox) for f in right])
        for f in (right if keep_left else left):
            f.primary, f.duplicate = f.duplicate, f.primary
    return findings


def shot_cast(stats: ShotStats, min_frames: int = 2) -> List[str]:
    """Character ids seen confidently in at least ``min_frames`` sampled frames, plus the ones
    carried along a track by the looser cast rule (:meth:`ShotStats.loose_cast`)."""
    confident = {c for c, n in stats.cast_frames.items() if n >= min_frames}
    return sorted(confident | set(stats.loose_cast()))


@dataclass
class ShotAnalysis:
    findings: List[DuplicateFinding]
    path: str                        # detector that produced ``findings``
    reason: str                      # why that path was used
    stats: ShotStats = field(default_factory=ShotStats)
    primary_stats: Optional[ShotStats] = None
    warning: Optional[str] = None    # e.g. escalation failed (OOM) and was disabled
    hires: bool = False              # the YOLO-World pass was re-run at HIRES_IMGSZ

    @property
    def cast(self) -> List[str]:
        return shot_cast(self.stats)


def tracker_for(detector: Detector, sample_fps: float) -> ByteTracker:
    """Class-agnostic ByteTrack whose thresholds follow the detector's own confidence floor
    (open-vocabulary scores for stylised characters are often 0.2-0.45)."""
    conf = float(getattr(detector, "conf", getattr(detector, "box_threshold", 0.3)))
    t = ByteTracker(frame_rate=sample_fps, track_thresh=conf, low_thresh=conf * 0.5, same_label_only=False)
    t.det_thresh = conf  # new tracks from any detection the detector itself reports
    return t


def _merge_cast(dst: ShotStats, src: ShotStats) -> None:
    for c, n in src.cast_frames.items():
        dst.cast_frames[c] = max(dst.cast_frames.get(c, 0), n)
    base = max(dst.track_cast, default=0) + 1_000_000
    for tid, per in src.track_cast.items():
        dst.track_cast[base + tid] = dict(per)


def analyze_shot(frames_factory: Callable[[], Iterable[Tuple[int, np.ndarray]]], shot: Shot, detector: Detector,
                 embedder: Embedder, threshold: float = 0.85, prompts: Sequence[str] = (), src_fps: float = 25.0,
                 scale: float = 1.0, sample_fps: float = 4.0, keep_all: bool = True, bank=None,
                 character_like_gate: bool = True, escalation_frames_factory: Optional[Callable[[], Iterable]] = None,
                 escalation_fps: Optional[float] = None, hires_frames_factory: Optional[Callable[[], Iterable]] = None,
                 hires_scale: float = 1.0, cache: Optional[DetectionCache] = None,
                 progress: Optional[Progress] = None) -> ShotAnalysis:
    """Duplicate finding for one shot, with the hybrid YOLO-World -> Grounded SAM 2 escalation.

    ``frames_factory`` returns a fresh ``(frame_index, frame)`` iterator (the shot is decoded again
    when it is escalated, instead of holding every frame in memory). ``escalation_frames_factory``
    (sampling at ``escalation_fps``, default :data:`ESCALATION_FPS`) is used for the escalated pass.
    ``hires_frames_factory`` (full-resolution frames, ``hires_scale`` = their width / source width)
    enables the YOLO-World re-run at :data:`HIRES_IMGSZ` for sparse shots (fewer than
    :data:`SPARSE_DETS_PER_FRAME` detections per frame). ``progress(frac, msg)`` reports per frame."""
    esc_fps = float(escalation_fps or (ESCALATION_FPS if escalation_frames_factory is not None else sample_fps))
    n_primary = expected_samples(shot, src_fps, sample_fps)
    n_esc = expected_samples(shot, src_fps, esc_fps)
    say = progress or (lambda f, m: None)

    def run(det: Detector, factory, fps: float, sc: float, lo: float, hi: float, n_expected: int, label: str,
            detect_kw: Optional[dict] = None, use_cache: bool = True) -> Tuple[List[DuplicateFinding], ShotStats]:
        st = ShotStats()

        def tick(k: int) -> None:
            say(lo + (hi - lo) * min(1.0, k / max(1, n_expected)), f"{label} frame {k}/{n_expected}")

        f = find_duplicates(factory(), shot, det, tracker_for(det, fps), embedder, threshold, prompts,
                            src_fps, sc, keep_all=keep_all, stats=st, bank=bank,
                            character_like_gate=character_like_gate, cache=cache if use_cache else None,
                            on_frame=tick, detect_kw=detect_kw)
        return f, st

    if not isinstance(detector, HybridDetector):
        f, st = run(detector, frames_factory, sample_fps, scale, 0.0, 1.0, n_primary, getattr(detector, "name", "?"))
        return ShotAnalysis(f, getattr(detector, "name", "?"), f"--detector {getattr(detector, 'name', '?')}", st)

    f, st = run(detector.primary, frames_factory, sample_fps, scale, 0.0, 0.25, n_primary, "YOLO-World")
    hires = False
    if (hires_frames_factory is not None and st.frames > 0 and st.detections < SPARSE_DETS_PER_FRAME * st.frames
            and isinstance(detector.primary, YoloWorldDetector)):
        # small background figures are missed at 640: re-run the sparse shot at full resolution
        f_hi, st_hi = run(detector.primary, hires_frames_factory, sample_fps, hires_scale, 0.25, 0.35, n_primary,
                          f"YOLO-World @{HIRES_IMGSZ}", {"imgsz": HIRES_IMGSZ}, use_cache=False)
        if st_hi.detections > st.detections:
            _merge_cast(st_hi, st)
            f, st, hires = f_hi, st_hi, True
        else:
            _merge_cast(st, st_hi)
    reason = detector.escalation_reason(st, threshold)
    if reason is None:
        say(1.0, "YOLO-World: no escalation")
        return ShotAnalysis(f, detector.primary.name, "YOLO-World: clean separation, no escalation needed", st,
                            hires=hires)
    fb = detector.get_fallback() if detector.fallback_possible else None
    if fb is None:
        why = detector.fallback_error or "Grounded SAM 2 not installed"
        return ShotAnalysis(f, detector.primary.name, f"{reason}; escalation skipped ({why})", st, hires=hires)
    try:
        f2, st2 = run(fb, escalation_frames_factory or frames_factory, esc_fps, scale, 0.35, 1.0, n_esc,
                      "Grounded SAM 2")
    except Exception as exc:  # CUDA OOM or anything else: keep YOLO-World's result, stop escalating
        kind = "CUDA out of memory" if is_oom(exc) else f"{type(exc).__name__}: {exc}"
        detector.disable_fallback(f"disabled after an error during escalation ({kind})")
        warn = (f"shot {shot.index}: Grounded SAM 2 escalation failed ({kind}); kept the YOLO-World result and "
                f"disabled escalation for the rest of the run")
        log.warning(warn)
        return ShotAnalysis(f, detector.primary.name, f"{reason}; escalation failed ({kind})", st, warning=warn,
                            hires=hires)
    # keep what YOLO-World already established: cast seen by either pass, and its named
    # findings when the escalated pass found none for that character
    _merge_cast(st2, st)
    have = {x.character for x in f2 if x.character}
    f2 = f2 + [x for x in f if x.character and x.character not in have]
    f2.sort(key=lambda x: (x.frame, x.primary.trackId))
    harmonize_roles(f2)
    return ShotAnalysis(f2, fb.name, f"escalated: {reason}", st2, primary_stats=st, hires=hires)


def dedupe_findings(findings: Sequence[DuplicateFinding]) -> List[DuplicateFinding]:
    """Keep one finding (the median frame) per (shot, character) for named duplicates and per
    (shot, id pair) for generic ones."""
    groups: Dict[Tuple, List[DuplicateFinding]] = {}
    for f in findings:
        if f.character:
            key: Tuple = (f.shotIndex, "char", f.character)
        else:
            key = (f.shotIndex, "pair", min(f.primary.trackId, f.duplicate.trackId), max(f.primary.trackId, f.duplicate.trackId))
        groups.setdefault(key, []).append(f)
    out = []
    for g in groups.values():
        g.sort(key=lambda f: f.frame)
        out.append(g[len(g) // 2])
    out.sort(key=lambda f: (f.shotIndex, f.frame, f.primary.trackId))
    return out
