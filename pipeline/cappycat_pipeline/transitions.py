"""Transition scoring and optical-flow frame interpolation.

Flow backend (``CAPPYCAT_FLOW_BACKEND=auto|raft|dis|farneback``, default ``auto``): torchvision
RAFT-small (``Raft_Small_Weights.DEFAULT``, cached under ``models/torch``) on CUDA when torch is
importable, else OpenCV DIS (``PRESET_MEDIUM``), else Farneback. RAFT runs in fp32 (its
correlation lookups are not fp16-safe); its memory is dominated by the all-pairs correlation
volume, ``((H/8)(W/8))^2 * 4 bytes`` per pair, so :func:`plan_flow` picks the flow resolution
(<= 720p) and batch size from the VRAM budget (``models.vram_budget_mb``).

``score_transition(a, b)`` returns ``(flow_magnitude, smoothness)``. ``flow_magnitude`` is
the mean flow vector length in *source* pixels between the last frame of one shot and the
first of the next; ``smoothness`` in [0, 1] blends ``exp(-flow / k)`` with an HSV histogram
similarity. Hard cuts between shots are normal (on the real AI clips intentional camera cuts
scored 0.07-0.45, median ~0.29), so :func:`suggest` only proposes a dissolve for genuinely
jarring cuts: smoothness < ``DISSOLVE_THRESHOLD`` (0.12), or < ``RELATIVE_CEILING`` (0.2) and
below half of the clip's own median cut smoothness.

``interpolate_frames(a, b, n)`` synthesises ``n`` in-betweens: forward and backward flow
(estimated at <= 720p and upsampled), intermediate flows by the linear-motion approximation
of Super SloMo (Jiang et al. 2018), backward warping of *both* neighbours, and a blend whose
weights drop a neighbour where the forward/backward consistency check says the pixel is
occluded in it.
"""
from __future__ import annotations

import logging
import math
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import models

log = logging.getLogger("cappycat.transitions")

ANALYSIS_WIDTH = 320
DISSOLVE_THRESHOLD = 0.12   # absolute: below this a cut is jarring whatever the clip does
RELATIVE_CEILING = 0.20     # relative rule only applies below this
RELATIVE_FRACTION = 0.5     # ... and below this fraction of the clip's median cut smoothness
FLOW_K_PERCENT = 4.0  # smoothness_flow = exp(-(mean flow as % of frame width) / K)
MAX_FLOW_HEIGHT = 720  # flow is estimated at <= 720p and upsampled
RAFT_UPDATES = 12

_RAFT = None
_RAFT_TRIED = False


def flow_backend_pref() -> str:
    v = os.environ.get("CAPPYCAT_FLOW_BACKEND", "auto").strip().lower()
    return v if v in ("auto", "raft", "dis", "farneback") else "auto"


class RaftFlow:
    """torchvision RAFT-small, batched."""

    def __init__(self, device: Optional[str] = None):
        models.configure_env()
        import torch  # type: ignore
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small  # type: ignore

        self.torch = torch
        self.device = device or models.device()
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).eval().to(self.device)

    def _tensor(self, frames: Sequence[np.ndarray]):
        import cv2

        x = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])
        t = self.torch.from_numpy(x).to(self.device).permute(0, 3, 1, 2).float()
        return t.div_(127.5).sub_(1.0)

    def flows(self, a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Flow a[i] -> b[i] for same-size frames; any size (padded to a multiple of 8)."""
        torch = self.torch
        h, w = a[0].shape[:2]
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        ta, tb = self._tensor(a), self._tensor(b)
        if ph or pw:
            ta = torch.nn.functional.pad(ta, (0, pw, 0, ph), mode="replicate")
            tb = torch.nn.functional.pad(tb, (0, pw, 0, ph), mode="replicate")
        with torch.inference_mode():
            fl = self.model(ta, tb, num_flow_updates=RAFT_UPDATES)[-1]
        out = fl[:, :, :h, :w].permute(0, 2, 3, 1).float().cpu().numpy()
        del ta, tb, fl
        return [o for o in out]

    def close(self) -> None:
        self.model = None
        models.release()


def _raft() -> Optional[RaftFlow]:
    """Lazily load RAFT-small once per process; None when unavailable / disabled."""
    global _RAFT, _RAFT_TRIED
    if flow_backend_pref() in ("dis", "farneback"):
        return None
    if _RAFT_TRIED:
        return _RAFT
    _RAFT_TRIED = True
    try:
        _RAFT = RaftFlow()
    except Exception as exc:
        log.debug("RAFT unavailable: %s", exc)
        _RAFT = None
    return _RAFT


def release_raft() -> None:
    global _RAFT, _RAFT_TRIED
    if _RAFT is not None:
        _RAFT.close()
    _RAFT, _RAFT_TRIED = None, False
    models.release()


def raft_mem_mb(h: int, w: int, batch: int = 1) -> float:
    """Approximate peak VRAM of one RAFT-small call (correlation pyramid dominates)."""
    n = (math.ceil(h / 8) * math.ceil(w / 8))
    # 4-level pyramid (1 + 1/4 + 1/16 + 1/64) plus the full volume's matmul temporary; measured
    # peak on an RTX 4070 at 1152x648: 1766 MB for one flow (this formula: ~1.8 GB)
    corr = n * n * 4 * 2.7
    feats = h * w * 4 * 64  # encoders / GRU activations, generous
    return batch * (corr + feats) / 2**20 + 64.0


def plan_flow(h: int, w: int, budget_mb: Optional[float] = None, max_batch: int = 8) -> Tuple[float, int]:
    """(scale, batch) so that flow runs at <= 720p and within ~60 % of the VRAM budget."""
    budget = (budget_mb if budget_mb is not None else models.vram_budget_mb()) * 0.6
    s = min(1.0, MAX_FLOW_HEIGHT / float(min(h, w)))
    while s > 0.1 and raft_mem_mb(int(h * s), int(w * s)) > budget:
        s *= 0.9
    one = raft_mem_mb(int(h * s), int(w * s))
    batch = int(max(1, min(max_batch, (budget - 64.0) // max(one - 64.0, 1.0))))
    return s, batch


def _cv_flow(a_bgr: np.ndarray, b_bgr: np.ndarray, backend: str) -> Tuple[np.ndarray, str]:
    import cv2

    ga = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
    if backend != "farneback" and hasattr(cv2, "DISOpticalFlow_create"):
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        return dis.calc(ga, gb, None).astype(np.float32), "dis"
    flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow.astype(np.float32), "farneback"


def estimate_flows(a: Sequence[np.ndarray], b: Sequence[np.ndarray], batch: int = 4) -> Tuple[List[np.ndarray], str]:
    """Flows a[i] -> b[i] at the given resolution. Returns (flows, backend)."""
    pref = flow_backend_pref()
    r = _raft()
    if r is not None:
        try:
            out: List[np.ndarray] = []
            for i in range(0, len(a), max(1, batch)):
                out.extend(r.flows(a[i:i + batch], b[i:i + batch]))
            return out, "raft"
        except Exception as exc:  # e.g. CUDA OOM -> fall back for this call
            log.warning("RAFT failed (%s); using OpenCV flow", exc)
            models.release()
    backend = "farneback" if pref == "farneback" else "dis"
    res = [_cv_flow(x, y, backend) for x, y in zip(a, b)]
    return [f for f, _ in res], (res[0][1] if res else backend)


def compute_flow(a_bgr: np.ndarray, b_bgr: np.ndarray) -> Tuple[np.ndarray, str]:
    """Dense flow a->b at the given resolution, shape (H, W, 2). Returns (flow, backend)."""
    flows, backend = estimate_flows([a_bgr], [b_bgr], batch=1)
    return flows[0], backend


def histogram_similarity(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    import cv2

    ha = cv2.calcHist([cv2.cvtColor(a_bgr, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 16], [0, 180, 0, 256])
    hb = cv2.calcHist([cv2.cvtColor(b_bgr, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 16], [0, 180, 0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    corr = float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
    return min(1.0, max(0.0, corr))


def _prep(frame: np.ndarray, width: int) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    if w != width:
        nh = max(8, int(round(h * width / float(w))))
        frame = cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    return frame[: h - h % 8, : w - w % 8]


def score_transition(last_frame_of_shot_a: np.ndarray, first_frame_of_shot_b: np.ndarray,
                     source_width: Optional[int] = None) -> Tuple[float, float]:
    """Return ``(flow_magnitude_px, smoothness)``. ``flow_magnitude_px`` is scaled to
    ``source_width`` pixels when given (else to the input frame width)."""
    src_w = float(source_width or last_frame_of_shot_a.shape[1])
    a = _prep(last_frame_of_shot_a, ANALYSIS_WIDTH)
    b = _prep(first_frame_of_shot_b, ANALYSIS_WIDTH)
    if a.shape != b.shape:
        import cv2

        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    flow, _ = compute_flow(a, b)
    mag = float(np.hypot(flow[..., 0], flow[..., 1]).mean())
    pct = mag / a.shape[1] * 100.0
    flow_px = mag * (src_w / a.shape[1])
    s_flow = math.exp(-pct / FLOW_K_PERCENT)
    s_hist = histogram_similarity(a, b)
    smoothness = 0.6 * s_flow + 0.4 * s_hist
    return round(flow_px, 4), round(min(1.0, max(0.0, smoothness)), 4)


def suggest(smoothness: float, clip_smoothness: Optional[Sequence[float]] = None) -> str:
    """``"dissolve"`` for a jarring cut, else ``"cut"``. ``clip_smoothness`` = every cut of the
    same clip (the relative rule needs at least 3)."""
    if smoothness < DISSOLVE_THRESHOLD:
        return "dissolve"
    if clip_smoothness is not None and len(clip_smoothness) >= 3:
        med = float(np.median(np.asarray(clip_smoothness, dtype=np.float64)))
        if smoothness < RELATIVE_CEILING and smoothness < RELATIVE_FRACTION * med:
            return "dissolve"
    return "cut"


# --------------------------------------------------------------------------- interpolation


def _grid(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return gx, gy


def _sample(img: np.ndarray, gx: np.ndarray, gy: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Backward warp: out(x) = img(x + flow(x))."""
    import cv2

    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def resize_flow(flow: np.ndarray, h: int, w: int) -> np.ndarray:
    import cv2

    fh, fw = flow.shape[:2]
    if (fh, fw) == (h, w):
        return flow
    up = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    up[..., 0] *= w / float(fw)
    up[..., 1] *= h / float(fh)
    return up


def occlusion_masks(f01: np.ndarray, f10: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Forward-backward consistency (Sundaram et al. 2010): ``occ0`` marks frame-0 pixels with no
    valid correspondence in frame 1 (they get covered), ``occ1`` frame-1 pixels that were hidden in
    frame 0 (they get uncovered). Float masks in [0, 1], slightly dilated / blurred."""
    import cv2

    h, w = f01.shape[:2]
    gx, gy = _grid(h, w)

    def occ(fa: np.ndarray, fb: np.ndarray) -> np.ndarray:
        fb_at = _sample(fb, gx, gy, fa)
        s = fa + fb_at
        err = s[..., 0] ** 2 + s[..., 1] ** 2
        mag = (fa ** 2).sum(-1) + (fb_at ** 2).sum(-1)
        o = (err > 0.01 * mag + 0.5).astype(np.float32)
        o = cv2.dilate(o, np.ones((3, 3), np.uint8))
        return cv2.GaussianBlur(o, (5, 5), 0)

    return occ(f01, f10), occ(f10, f01)


def _warp_on_gpu() -> bool:
    return flow_backend_pref() in ("auto", "raft") and models.cuda_ok()


def _torch_ops(h: int, w: int, dev: str):
    """``(sample, occ)`` closures on a cached pixel grid: ``sample(img, flow)`` is the backward warp
    ``out(x) = img(x + flow(x))`` (bilinear, border replicate) and ``occ(fa, fb)`` the
    forward/backward-consistency occlusion mask of ``fa`` (dilated 3x3, box-blurred 5x5)."""
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                            torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    sx, sy = 2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)

    def sample(img, flow):  # img N x C x H x W, flow N x 2 x H x W
        gx = (xs + flow[:, 0]) * sx - 1.0
        gy = (ys + flow[:, 1]) * sy - 1.0
        return F.grid_sample(img, torch.stack([gx, gy], dim=-1), mode="bilinear", padding_mode="border",
                             align_corners=True)

    def occ(fa, fb):
        fb_at = sample(fb, fa)
        s = fa + fb_at
        err = (s * s).sum(1, keepdim=True)
        mag = (fa * fa).sum(1, keepdim=True) + (fb_at * fb_at).sum(1, keepdim=True)
        o = (err > 0.01 * mag + 0.5).float()
        o = F.max_pool2d(o, 3, stride=1, padding=1)
        return F.avg_pool2d(o, 5, stride=1, padding=2, count_include_pad=False)

    return sample, occ


def _torch_blend(A, B, F01, F10, times: Sequence[float], ops, chunk: int = 4):
    """In-betweens of ``A`` -> ``B`` (1 x 3 x H x W float, 0..255) at ``times`` given full-resolution
    flows ``F01`` / ``F10`` (1 x 2 x H x W): intermediate flows by the linear-motion approximation,
    both neighbours backward-warped, visibility-weighted blend. Yields ``uint8`` H x W x 3 tensors
    (on the device), ``chunk`` times per batched warp."""
    import torch  # type: ignore

    sample, occ = ops
    O0, O1 = occ(F01, F10), occ(F10, F01)
    ts = [float(t) for t in times]
    for c in range(0, len(ts), max(1, chunk)):
        t = torch.tensor(ts[c:c + chunk], device=A.device, dtype=torch.float32).view(-1, 1, 1, 1)
        n = t.shape[0]
        ft0 = -(1.0 - t) * t * F01 + t * t * F10
        ft1 = (1.0 - t) * (1.0 - t) * F01 - t * (1.0 - t) * F10
        g0, g1 = sample(A.expand(n, -1, -1, -1), ft0), sample(B.expand(n, -1, -1, -1), ft1)
        o0, o1 = sample(O0.expand(n, -1, -1, -1), ft0), sample(O1.expand(n, -1, -1, -1), ft1)
        del ft0, ft1
        w0 = (1.0 - t) * (1.0 - o1) + 1e-4 * (1.0 - t)
        w1 = t * (1.0 - o0) + 1e-4 * t
        mid = (w0 * g0 + w1 * g1) / (w0 + w1)
        del g0, g1, o0, o1, w0, w1
        out = mid.permute(0, 2, 3, 1).clamp_(0, 255).round_().to(torch.uint8)
        del mid
        for i in range(n):
            yield out[i]


def _interpolate_torch(a: np.ndarray, b: np.ndarray, f01: np.ndarray, f10: np.ndarray,
                       times: Sequence[float]) -> List[np.ndarray]:
    """Same maths as the numpy path (backward warps with ``grid_sample``, forward/backward
    consistency occlusion, dilate + blur, visibility-weighted blend), on the GPU."""
    import torch  # type: ignore

    dev = "cuda"
    h, w = a.shape[:2]
    with torch.inference_mode():
        A = torch.from_numpy(np.ascontiguousarray(a)).to(dev).permute(2, 0, 1)[None].float()
        B = torch.from_numpy(np.ascontiguousarray(b)).to(dev).permute(2, 0, 1)[None].float()
        F01 = torch.from_numpy(np.ascontiguousarray(f01)).to(dev).permute(2, 0, 1)[None].float()
        F10 = torch.from_numpy(np.ascontiguousarray(f10)).to(dev).permute(2, 0, 1)[None].float()
        out = [m.cpu().numpy() for m in _torch_blend(A, B, F01, F10, times, _torch_ops(h, w, dev), chunk=1)]
        del A, B, F01, F10
    return out


def interpolate_with_flows(a: np.ndarray, b: np.ndarray, f01: np.ndarray, f10: np.ndarray,
                           times: Sequence[float]) -> List[np.ndarray]:
    """Occlusion-aware in-betweens at ``times`` (each in (0, 1)) given full-resolution flows.
    Runs on the GPU (torch ``grid_sample``) when CUDA is usable, else numpy / OpenCV."""
    if _warp_on_gpu():
        try:
            return _interpolate_torch(a, b, f01, f10, times)
        except Exception as exc:  # e.g. OOM on a huge frame -> CPU path
            log.warning("GPU warping failed (%s); using the CPU path", exc)
            models.release()
    h, w = a.shape[:2]
    gx, gy = _grid(h, w)
    occ0, occ1 = occlusion_masks(f01, f10)
    af, bf = a.astype(np.float32), b.astype(np.float32)
    out: List[np.ndarray] = []
    for t in times:
        # linear-motion approximation of the flows from time t to the two neighbours
        ft0 = -(1.0 - t) * t * f01 + t * t * f10
        ft1 = (1.0 - t) * (1.0 - t) * f01 - t * (1.0 - t) * f10
        g0 = _sample(af, gx, gy, ft0)
        g1 = _sample(bf, gx, gy, ft1)
        # a pixel that frame 0 loses (occ0 at its source) is only valid in frame 0; one that frame 1
        # uncovers (occ1 at its source) only in frame 1
        o0 = _sample(occ0, gx, gy, ft0)
        o1 = _sample(occ1, gx, gy, ft1)
        w0 = (1.0 - t) * (1.0 - o1) + 1e-4 * (1.0 - t)
        w1 = t * (1.0 - o0) + 1e-4 * t
        mid = (w0[..., None] * g0 + w1[..., None] * g1) / (w0 + w1)[..., None]
        out.append(mid.clip(0, 255).astype(np.uint8))
    return out


def pair_flows(a_list: Sequence[np.ndarray], b_list: Sequence[np.ndarray], scale: float,
               batch: int = 4) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], str]:
    """Forward + backward flow for each (a, b) pair, estimated at ``scale`` and returned at full
    resolution."""
    import cv2

    h, w = a_list[0].shape[:2]
    if scale < 0.999:
        sw, sh = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
        small_a = [cv2.resize(x, (sw, sh), interpolation=cv2.INTER_AREA) for x in a_list]
        small_b = [cv2.resize(x, (sw, sh), interpolation=cv2.INTER_AREA) for x in b_list]
    else:
        small_a, small_b = list(a_list), list(b_list)
    flows, backend = estimate_flows(small_a + small_b, small_b + small_a, batch=batch)
    n = len(a_list)
    return [(resize_flow(flows[i], h, w), resize_flow(flows[n + i], h, w)) for i in range(n)], backend


def interpolate_frames(a: np.ndarray, b: np.ndarray, n: int) -> List[np.ndarray]:
    """Return ``n`` intermediate frames between ``a`` and ``b`` (same size), RAFT when available,
    else DIS."""
    import cv2

    if n <= 0:
        return []
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    scale, batch = plan_flow(a.shape[0], a.shape[1])
    (f01, f10), = pair_flows([a], [b], scale, batch)[0]
    return interpolate_with_flows(a, b, f01, f10, [i / (n + 1.0) for i in range(1, n + 1)])


# --------------------------------------------------------------------------- batched jobs (interpolate --target-fps)

WARP_BYTES_PER_PIXEL_T = 100.0   # ~25 float32 planes per in-between being warped at once
WARP_BYTES_PER_PIXEL_BASE = 110.0  # A, B, flows, occlusion masks and their temporaries


class TorchFlowInterpolator:
    """GPU-resident version of :func:`pair_flows` + :func:`interpolate_with_flows` for many pairs.

    The frames of a window are uploaded once as ``uint8``; RAFT runs on area-downscaled copies in
    batches of ``batch`` flows (forward and backward of every pair); the flows are upsampled on the
    GPU; each pair is then warped at all of its positions (in chunks sized from the budget) and only
    the finished ``uint8`` frames come back to the host. Same maths as the CPU path."""

    def __init__(self, raft: RaftFlow, scale: float, batch: int, budget_mb: Optional[float] = None):
        self.raft = raft
        self.torch = raft.torch
        self.dev = raft.device
        self.scale = float(scale)
        self.batch = max(1, int(batch))
        self.budget_mb = float(budget_mb if budget_mb is not None else models.vram_budget_mb())
        self._ops = None
        self._ops_size: Optional[Tuple[int, int]] = None

    def _chunk(self, h: int, w: int) -> int:
        free = self.budget_mb * 2**20 * 0.6 - h * w * WARP_BYTES_PER_PIXEL_BASE
        return int(max(1, min(8, free // max(h * w * WARP_BYTES_PER_PIXEL_T, 1.0))))

    def run(self, frames: Sequence[np.ndarray], jobs: Sequence[Tuple[int, Sequence[float]]]) -> List[List[np.ndarray]]:
        """``jobs[k] = (i, times)``: in-betweens of ``frames[i]`` -> ``frames[i + 1]`` at ``times``
        (each in (0, 1)). Returns one list of frames per job."""
        torch = self.torch
        F = torch.nn.functional
        if not jobs:
            return []
        h, w = frames[0].shape[:2]
        dev = self.dev
        if self._ops_size != (h, w):
            self._ops, self._ops_size = _torch_ops(h, w, dev), (h, w)
        sh, sw = max(8, int(round(h * self.scale))), max(8, int(round(w * self.scale)))
        ph, pw = (8 - sh % 8) % 8, (8 - sw % 8) % 8
        needed = sorted({i for i, _ in jobs} | {i + 1 for i, _ in jobs})
        results: List[List[np.ndarray]] = []
        with torch.inference_mode():
            gpu = {j: torch.from_numpy(np.ascontiguousarray(frames[j])).to(dev) for j in needed}

            def raft_in(j: int):
                x = gpu[j].permute(2, 0, 1)[None].flip(1).float()  # BGR -> RGB, 1 x 3 x H x W
                if (sh, sw) != (h, w):
                    x = F.interpolate(x, size=(sh, sw), mode="area")
                x = x.div_(127.5).sub_(1.0)
                if ph or pw:
                    x = F.pad(x, (0, pw, 0, ph), mode="replicate")
                return x

            small = {j: raft_in(j) for j in needed}
            src = [small[i] for i, _ in jobs] + [small[i + 1] for i, _ in jobs]
            dst = [small[i + 1] for i, _ in jobs] + [small[i] for i, _ in jobs]
            flows = []
            for c in range(0, len(src), self.batch):
                fl = self.raft.model(torch.cat(src[c:c + self.batch]), torch.cat(dst[c:c + self.batch]),
                                     num_flow_updates=RAFT_UPDATES)[-1]
                flows.append(fl[:, :, :sh, :sw].clone())
                del fl
            del small, src, dst
            flows_t = torch.cat(flows)
            del flows
            n = len(jobs)
            fscale = torch.tensor([w / float(sw), h / float(sh)], device=dev, dtype=torch.float32).view(1, 2, 1, 1)

            def up(f):
                if (sh, sw) == (h, w):
                    return f
                return F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False) * fscale

            chunk = self._chunk(h, w)
            for p, (i, times) in enumerate(jobs):
                if not times:
                    results.append([])
                    continue
                F01, F10 = up(flows_t[p:p + 1]), up(flows_t[n + p:n + p + 1])
                A = gpu[i].permute(2, 0, 1)[None].float()
                B = gpu[i + 1].permute(2, 0, 1)[None].float()
                results.append([m.cpu().numpy() for m in _torch_blend(A, B, F01, F10, times, self._ops, chunk)])
                del F01, F10, A, B
            gpu.clear()
            del flows_t
        return results


def interpolate_jobs(frames: Sequence[np.ndarray], jobs: Sequence[Tuple[int, Sequence[float]]], scale: float,
                     batch: int, budget_mb: Optional[float] = None,
                     state: Optional[dict] = None) -> Tuple[List[List[np.ndarray]], str]:
    """In-betweens for several pairs of one frame window: ``jobs[k] = (i, times)`` interpolates
    ``frames[i]`` -> ``frames[i + 1]`` at every position in ``times``. Uses
    :class:`TorchFlowInterpolator` when RAFT runs on CUDA (backend ``"raft-cuda"``), else
    :func:`pair_flows` + :func:`interpolate_with_flows` (RAFT on the CPU, DIS or Farneback).
    ``state`` (a dict kept by the caller across calls) caches the GPU interpolator and remembers
    a GPU failure (e.g. CUDA OOM), after which the CPU path is used for the rest of the run."""
    if not jobs:
        return [], "none"
    state = state if state is not None else {}
    if not state.get("gpu_failed"):
        r = _raft()
        if r is not None and str(r.device).startswith("cuda"):
            gi = state.get("gpu")
            if gi is None or gi.raft is not r:
                gi = state["gpu"] = TorchFlowInterpolator(r, scale, batch, budget_mb)
            try:
                return gi.run(frames, jobs), "raft-cuda"
            except Exception as exc:  # CUDA OOM etc. -> CPU flow / warping from now on
                log.warning("GPU interpolation failed (%s); using the CPU path", exc)
                state["gpu_failed"] = True
                state.pop("gpu", None)
                models.release()
    flows, backend = pair_flows([frames[i] for i, _ in jobs], [frames[i + 1] for i, _ in jobs], scale, batch)
    out = [interpolate_with_flows(frames[i], frames[i + 1], f01, f10, list(times)) if times else []
           for (i, times), (f01, f10) in zip(jobs, flows)]
    return out, backend
