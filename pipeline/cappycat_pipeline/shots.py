"""Shot-boundary detection.

* ``transnetv2`` – TransNetV2 with the official weights, fed exactly like the reference
  implementation (27x48 RGB frames, windows of 100 with 25 frames of padding on each side,
  stride 50, single-frame prediction output). Backends, best first:

  1. ``torch-cuda``  – ``models/transnetv2.pt`` + the vendored network (:mod:`.transnetv2_model`)
  2. ``onnxruntime`` – ``models/transnetv2.onnx`` (CUDA -> DirectML -> CPU providers)
  3. ``torch-cpu``   – ``models/transnetv2.pt`` on the CPU when onnxruntime is unavailable
* ``pyscenedetect`` – ``AdaptiveDetector`` (rolling-average content delta, more
  robust to AI latent flicker than ``ContentDetector``) driven with frames from our
  own ffmpeg pipe instead of ``cv2.VideoCapture``.
* ``auto`` – TransNetV2 when any backend is usable, else PySceneDetect.

Always returns at least one shot spanning the whole clip.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import ffmpeg_util, models
from .models import models_dir  # noqa: F401  (re-exported for backwards compatibility)
from .schema import Asset, Shot

log = logging.getLogger("cappycat.shots")

TRANSNET_INPUT_HW = (27, 48)  # height, width
TRANSNET_WINDOW = 100
TRANSNET_STRIDE = 50
TRANSNET_PAD = 25
TRANSNET_BATCH = 8  # windows per forward pass (~150 MB of activations on CUDA)


# --------------------------------------------------------------------------- model discovery


def transnet_model_path() -> Path:
    """The ONNX export (kept for backwards compatibility; see :func:`transnet_backend`)."""
    return models.transnet_onnx_path()


def _ort_ok() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


def transnet_backend() -> Optional[str]:
    """``"torch-cuda"`` > ``"onnxruntime"`` > ``"torch-cpu"`` > None, by what is installed."""
    pt, onnx = models.transnet_pt_path().is_file(), models.transnet_onnx_path().is_file()
    if pt and models.cuda_ok():
        return "torch-cuda"
    if onnx and _ort_ok():
        return "onnxruntime"
    if pt and models.torch_or_none() is not None:
        return "torch-cpu"
    return None


def transnet_available() -> bool:
    return transnet_backend() is not None


def preferred_providers() -> List[str]:
    """CUDA -> DirectML -> CPU, filtered to what this onnxruntime build offers."""
    try:
        import onnxruntime as ort
    except Exception:
        return []
    available = ort.get_available_providers()
    order = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    return [p for p in order if p in available] or list(available)


# --------------------------------------------------------------------------- helpers


def frames_to_shots(cuts: Sequence[int], n_frames: int, fps: float, method: str,
                    confidences: Optional[Sequence[float]] = None,
                    clock: Optional[ffmpeg_util.FrameClock] = None) -> List[Shot]:
    """``cuts`` are the *first frame indices* of each new shot (excluding 0). Times come from
    ``clock`` (the frames' real presentation times) so cuts in variable-frame-rate sources land on
    the actual first frame of the new shot; without one they are ``index / fps``."""
    n_frames = max(1, int(n_frames))
    fps = float(fps) if fps and fps > 0 else 25.0
    clock = clock or ffmpeg_util.FrameClock.uniform(fps)
    starts = [0] + sorted({int(c) for c in cuts if 0 < int(c) < n_frames})
    shots: List[Shot] = []
    for i, s in enumerate(starts):
        e = (starts[i + 1] - 1) if i + 1 < len(starts) else n_frames - 1
        conf = 1.0 if i == 0 else (float(confidences[i - 1]) if confidences and i - 1 < len(confidences) else 0.75)
        shots.append(Shot(
            index=i, startFrame=s, endFrame=e,
            startMs=round(clock.ms(s), 3), endMs=round(clock.ms(e), 3),
            confidence=round(min(1.0, max(0.0, conf)), 4), method=method,
        ))
    return shots


def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Port of ``TransNetV2.predictions_to_scenes`` -> (n, 2) inclusive [start, end] ranges."""
    predictions = (np.asarray(predictions) > threshold).astype(np.uint8)
    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)
    return np.array(scenes, dtype=np.int32)


def transnet_windows(frames: np.ndarray) -> np.ndarray:
    """Reference windowing: (n, 27, 48, 3) -> (k, 100, 27, 48, 3) windows (stride 50, the
    first / last frame repeated as padding)."""
    n = len(frames)
    no_padded_start = TRANSNET_PAD
    no_padded_end = TRANSNET_PAD + TRANSNET_STRIDE - (n % TRANSNET_STRIDE if n % TRANSNET_STRIDE != 0 else TRANSNET_STRIDE)
    padded = np.concatenate([frames[:1]] * no_padded_start + [frames] + [frames[-1:]] * no_padded_end, 0)
    starts = range(0, len(padded) - TRANSNET_WINDOW + 1, TRANSNET_STRIDE)
    return np.stack([padded[p:p + TRANSNET_WINDOW] for p in starts])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- TransNetV2


class _TransNetBase:
    backend = "?"

    def _predict_windows(self, windows: np.ndarray) -> np.ndarray:
        """(b, 100, 27, 48, 3) uint8 -> (b, 100) probabilities."""
        raise NotImplementedError

    def predict_frames(self, frames: np.ndarray) -> np.ndarray:
        """frames: (n, 27, 48, 3) RGB uint8 -> per-frame transition probability (n,)."""
        n = len(frames)
        if n == 0:
            return np.zeros((0,), dtype=np.float32)
        windows = transnet_windows(np.asarray(frames, dtype=np.uint8))
        preds = []
        for i in range(0, len(windows), TRANSNET_BATCH):
            p = self._predict_windows(windows[i:i + TRANSNET_BATCH])
            preds.extend(p[:, TRANSNET_PAD:TRANSNET_PAD + TRANSNET_STRIDE])
        return np.concatenate(preds)[:n].astype(np.float32)

    def close(self) -> None:
        pass


class TransNetV2Torch(_TransNetBase):
    """Vendored PyTorch TransNetV2 (fp32; the network is 30 MB and fp16 moves the logits)."""

    def __init__(self, weights: Optional[Path] = None, device: Optional[str] = None):
        import torch  # type: ignore

        from .transnetv2_model import TransNetV2SingleFrame, load_transnetv2

        self.path = Path(weights or models.transnet_pt_path())
        if not self.path.is_file():
            raise FileNotFoundError(f"TransNetV2 weights not found at {self.path} (run `python -m cappycat_pipeline download-models`)")
        self.device = device or models.device()
        self.backend = "torch-cuda" if self.device.startswith("cuda") else "torch-cpu"
        self.torch = torch
        self.net = TransNetV2SingleFrame(load_transnetv2(self.path, self.device)).eval()

    def _predict_windows(self, windows: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(windows)).to(self.device)
            logits = self.net(x)[..., 0]
            return torch.sigmoid(logits).float().cpu().numpy()

    def close(self) -> None:
        self.net = None
        models.release()


class TransNetV2Onnx(_TransNetBase):
    backend = "onnxruntime"

    def __init__(self, model_path: Optional[Path] = None, providers: Optional[List[str]] = None):
        import onnxruntime as ort

        self.path = Path(model_path or transnet_model_path())
        if not self.path.is_file():
            raise FileNotFoundError(f"TransNetV2 ONNX model not found at {self.path}")
        providers = providers or preferred_providers()
        if "CUDAExecutionProvider" in providers:
            models.torch_or_none()  # importing torch puts its CUDA 13 / cuDNN 9 DLLs on the search path for onnxruntime-gpu
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(self.path), sess_options=opts, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_type = inp.type  # e.g. "tensor(uint8)" or "tensor(float)"
        dim0 = inp.shape[0] if inp.shape else 1
        self.fixed_batch = dim0 if isinstance(dim0, int) else None
        self.output_names = [o.name for o in self.session.get_outputs()]

    def _predict_raw(self, window: np.ndarray) -> np.ndarray:
        """window: (b, 100, 27, 48, 3) uint8 -> single-frame prediction (b, 100) in [0, 1]."""
        x = window.astype(np.float32) if "float" in self.input_type else window
        outs = self.session.run(None, {self.input_name: x})
        single = np.asarray(outs[0], dtype=np.float32)
        single = single.reshape(single.shape[0], TRANSNET_WINDOW, -1)[:, :, 0]
        if single.min() < 0.0 or single.max() > 1.0:  # PyTorch export returns logits
            single = _sigmoid(single)
        return single

    def _predict_windows(self, windows: np.ndarray) -> np.ndarray:
        if self.fixed_batch == 1:
            return np.concatenate([self._predict_raw(w[None]) for w in windows])
        return self._predict_raw(windows)

    def close(self) -> None:
        self.session = None


def load_transnet(backend: Optional[str] = None) -> _TransNetBase:
    backend = backend or transnet_backend()
    if backend == "torch-cuda":
        return TransNetV2Torch(device="cuda")
    if backend == "torch-cpu":
        return TransNetV2Torch(device="cpu")
    if backend == "onnxruntime":
        return TransNetV2Onnx()
    raise FileNotFoundError(
        f"TransNetV2 not available: need {models.transnet_pt_path()} (+ torch) or {models.transnet_onnx_path()} "
        f"(+ onnxruntime). Run `python -m cappycat_pipeline download-models`.")


def _load_transnet_frames(path: str, src_size: Tuple[int, int]) -> np.ndarray:
    import cv2

    frames = []
    h, w = TRANSNET_INPUT_HW
    for bgr in ffmpeg_util.iter_frames(path, src_size=src_size, size=(w, h)):
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not frames:
        return np.zeros((0, h, w, 3), dtype=np.uint8)
    return np.stack(frames).astype(np.uint8)


def detect_shots_transnet(path: str, fps: float, src_size: Tuple[int, int], threshold: float = 0.5,
                          model: Optional[_TransNetBase] = None,
                          clock: Optional[ffmpeg_util.FrameClock] = None) -> List[Shot]:
    own = model is None
    model = model or load_transnet()
    try:
        frames = _load_transnet_frames(path, src_size)
        n = len(frames)
        if n == 0:
            return frames_to_shots([], 1, fps, "transnetv2", clock=clock)
        probs = model.predict_frames(frames)
    finally:
        if own:
            model.close()
    scenes = predictions_to_scenes(probs, threshold)
    cuts = [int(s[0]) for s in scenes[1:]]
    # confidence = peak probability on the transition frames just before the new shot
    confs = [float(probs[max(0, c - 3):c].max()) if c > 0 else float(probs[0]) for c in cuts]
    return frames_to_shots(cuts, n, fps, "transnetv2", confs, clock=clock)


# --------------------------------------------------------------------------- PySceneDetect


def _tc_frame(tc) -> int:
    for attr in ("frame_num",):
        v = getattr(tc, attr, None)
        if v is not None:
            return int(v)
    if hasattr(tc, "get_frames"):
        return int(tc.get_frames())
    return int(tc)


def detect_shots_pyscenedetect(path: str, fps: float, src_size: Tuple[int, int], threshold: float = 3.0,
                               min_scene_len: int = 12, analysis_width: int = 320,
                               clock: Optional[ffmpeg_util.FrameClock] = None) -> List[Shot]:
    from scenedetect import AdaptiveDetector, FrameTimecode

    fps = float(fps) if fps and fps > 0 else 25.0
    detector = AdaptiveDetector(adaptive_threshold=float(threshold), min_scene_len=int(min_scene_len))
    cuts: List[int] = []
    n = 0
    last_tc = None
    for i, frame in enumerate(ffmpeg_util.iter_frames(path, src_size=src_size, width=analysis_width)):
        last_tc = FrameTimecode(i, fps)
        for tc in detector.process_frame(last_tc, frame):
            cuts.append(_tc_frame(tc))
        n = i + 1
    if last_tc is not None:
        for tc in detector.post_process(last_tc):
            cuts.append(_tc_frame(tc))
    return frames_to_shots(cuts, max(n, 1), fps, "pyscenedetect", clock=clock)


# --------------------------------------------------------------------------- entry point


def detect_shots(path: str | os.PathLike, method: str = "auto", threshold: float = 0.5,
                 min_scene_len: int = 12, asset: Optional[Asset] = None) -> List[Shot]:
    """Detect shots. ``threshold`` is the TransNetV2 probability threshold; PySceneDetect
    uses its own adaptive threshold (3.0)."""
    path = str(path)
    asset = asset or ffmpeg_util.probe(path)
    fps = asset.fps if asset.fps > 0 else 25.0
    src_size = (asset.width, asset.height)
    if asset.kind == "image" or asset.width <= 0:
        return [Shot(0, 0, 0, 0.0, 0.0, 1.0, "pyscenedetect")]

    clock = ffmpeg_util.frame_clock(path, fps)
    method = (method or "auto").lower()
    if method == "auto":
        method = "transnetv2" if transnet_available() else "pyscenedetect"
    if method == "transnetv2":
        try:
            return detect_shots_transnet(path, fps, src_size, threshold, clock=clock)
        except FileNotFoundError:
            raise
        except Exception as exc:  # model/provider failure -> degrade gracefully
            log.warning("TransNetV2 failed (%s); falling back to PySceneDetect", exc)
    return detect_shots_pyscenedetect(path, fps, src_size, min_scene_len=min_scene_len, clock=clock)
