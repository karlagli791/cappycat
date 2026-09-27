"""Model registry, cache locations, device selection and VRAM helpers.

Everything the ML path downloads lives under ``pipeline/models/`` (or ``CAPPYCAT_MODELS_DIR``):

=============================  ===================================================================
``transnetv2.pt``              TransNetV2 state dict (official weights, PyTorch port) - torch path
``transnetv2.onnx``            same network exported with ``torch.onnx.export`` - onnxruntime path
``yolov8s-worldv2.pt``         YOLO-World v2 small (ultralytics)
``clip/ViT-B-32.pt``           OpenAI CLIP text encoder used by YOLO-World ``set_classes``
``hf/hub/...``                 Hugging Face cache (``HF_HOME``): OpenCLIP ViT-B-32 laion2b,
                               Grounding DINO tiny, SAM 2.1 hiera-small
``torch/hub/checkpoints/...``  torchvision RAFT-small (``TORCH_HOME``)
``hf/hub/models--adefossez--`` Demucs v4 ``htdemucs_ft`` voice / background separation (``separate.py``)
``HTDemucs-ft``
``ultralytics/``               private ultralytics settings dir (``YOLO_CONFIG_DIR``), telemetry off
=============================  ===================================================================

:func:`configure_env` points ``HF_HOME`` / ``TORCH_HOME`` / ``YOLO_CONFIG_DIR`` there (without
overriding values the user already set); it runs on import so every loader sees the same cache.

VRAM policy (8 GB laptop GPUs shared with the editor and other apps): models are loaded
lazily, in fp16 where it is numerically safe (YOLO-World, OpenCLIP), and released with
:func:`release` (``del`` + ``torch.cuda.empty_cache()``) between stages. ``CAPPYCAT_VRAM_BUDGET_MB``
(default 2048) bounds the batch / resolution choices of the optical-flow stage.
``CAPPYCAT_DEVICE=cpu`` forces CPU inference.
"""
from __future__ import annotations

import gc
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("cappycat.models")

# --------------------------------------------------------------------------- identifiers

TRANSNET_HF_REPOS = ("Sn4kehead/TransNetV2", "magnusdtd/TransNetV2")  # bit-identical official weights
TRANSNET_HF_FILE = "transnetv2-pytorch-weights.pth"
YOLO_WORLD_FILE = "yolov8s-worldv2.pt"
OPENCLIP_MODEL = "ViT-B-32"
OPENCLIP_PRETRAINED = "laion2b_s34b_b79k"
OPENCLIP_HF_REPO = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
GDINO_HF_REPO = "IDEA-Research/grounding-dino-tiny"
SAM2_HF_REPO = "facebook/sam2.1-hiera-small"
HF_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model"]


# --------------------------------------------------------------------------- locations


def models_dir() -> Path:
    env = os.environ.get("CAPPYCAT_MODELS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "models"


def hf_home() -> Path:
    return models_dir() / "hf"


def hf_hub_cache() -> Path:
    return hf_home() / "hub"


def torch_home() -> Path:
    return models_dir() / "torch"


def transnet_pt_path() -> Path:
    return models_dir() / "transnetv2.pt"


def transnet_onnx_path() -> Path:
    return models_dir() / "transnetv2.onnx"


def yolo_world_path() -> Path:
    return models_dir() / YOLO_WORLD_FILE


def clip_text_path() -> Path:
    return models_dir() / "clip" / "ViT-B-32.pt"


def raft_small_path() -> Optional[Path]:
    d = torch_home() / "hub" / "checkpoints"
    hits = sorted(d.glob("raft_small*.pth")) if d.is_dir() else []
    return hits[0] if hits else None


_CONFIGURED: Optional[str] = None


def configure_env() -> None:
    """Point the HF / torch / ultralytics caches into ``models_dir()`` (idempotent, never
    overrides variables the user set explicitly). Must run before transformers /
    huggingface_hub / ultralytics are imported, which is why it runs on import of this module."""
    global _CONFIGURED
    md = str(models_dir())
    if _CONFIGURED == md:
        return
    _CONFIGURED = md
    os.environ.setdefault("HF_HOME", str(hf_home()))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TORCH_HOME", str(torch_home()))
    if "YOLO_CONFIG_DIR" not in os.environ:
        ycd = models_dir() / "ultralytics"
        try:  # ultralytics only honours the dir when its parent exists and is writable
            ycd.mkdir(parents=True, exist_ok=True)
            os.environ["YOLO_CONFIG_DIR"] = str(ycd)
        except OSError:
            pass
    os.environ.setdefault("YOLO_VERBOSE", "False")
    os.environ.setdefault("YOLO_OFFLINE", "True")  # no ultralytics update checks / telemetry
    # never let ultralytics pip-install things at runtime (it could swap the CUDA torch build)
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")


configure_env()


def import_ultralytics():
    """Import ultralytics with its settings / weights dir redirected into ``models_dir()``."""
    configure_env()
    import ultralytics  # type: ignore
    from ultralytics import utils as u  # type: ignore

    md = models_dir()
    try:
        want = {"weights_dir": str(md), "sync": False}
        if any(u.SETTINGS.get(k) != v for k, v in want.items()):
            u.SETTINGS.update(want)
    except Exception as exc:  # pragma: no cover
        log.debug("could not update ultralytics settings: %s", exc)
    u.WEIGHTS_DIR = md
    try:
        from ultralytics.nn import text_model  # type: ignore

        text_model.WEIGHTS_DIR = md
    except Exception:  # pragma: no cover
        pass
    return ultralytics


# --------------------------------------------------------------------------- device / VRAM


def torch_or_none():
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def cuda_ok() -> bool:
    if os.environ.get("CAPPYCAT_DEVICE", "").lower() == "cpu":
        return False
    torch = torch_or_none()
    try:
        return bool(torch is not None and torch.cuda.is_available())
    except Exception:
        return False


def device() -> str:
    return "cuda" if cuda_ok() else "cpu"


def vram_info() -> Optional[Dict[str, Any]]:
    """``{"device", "freeMb", "totalMb"}`` for GPU 0, or None."""
    if not cuda_ok():
        return None
    import torch  # type: ignore

    try:
        free, total = torch.cuda.mem_get_info(0)
        return {"device": torch.cuda.get_device_name(0), "freeMb": round(free / 2**20), "totalMb": round(total / 2**20)}
    except Exception:
        return None


def vram_budget_mb() -> float:
    """Working budget for one stage: ``CAPPYCAT_VRAM_BUDGET_MB`` (default 2048) capped by
    what is actually free right now (minus a 256 MB safety margin)."""
    try:
        budget = float(os.environ.get("CAPPYCAT_VRAM_BUDGET_MB", "2048"))
    except ValueError:
        budget = 2048.0
    info = vram_info()
    if info:
        budget = min(budget, max(256.0, info["freeMb"] - 256.0))
    return budget


def release(*objs: Any) -> None:
    """Collect garbage and return cached CUDA blocks to the driver. Callers ``del`` their own
    references first (``del model; models.release()``)."""
    del objs
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def peak_vram_mb(reset: bool = False) -> Optional[float]:
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        v = torch.cuda.max_memory_allocated() / 2**20
        if reset:
            torch.cuda.reset_peak_memory_stats()
        return round(v, 1)
    except Exception:
        return None


# --------------------------------------------------------------------------- presence (doctor)


def hf_repo_present(repo: str, filename: str = "config.json") -> bool:
    """True when ``filename`` of ``repo`` is in the local HF cache (no network)."""
    d = hf_hub_cache() / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not d.is_dir():
        return False
    return any((snap / filename).is_file() for snap in d.iterdir())


def _size_mb(p: Optional[Path]) -> Optional[float]:
    try:
        return round(p.stat().st_size / 2**20, 1) if p and p.is_file() else None
    except OSError:
        return None


def _dir_mb(d: Path) -> Optional[float]:
    if not d.is_dir():
        return None
    total = 0
    for p in d.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return round(total / 2**20, 1)


@dataclass
class ModelStatus:
    name: str
    present: bool
    location: str
    sizeMb: Optional[float]
    usedBy: str

    def to_json(self) -> Dict[str, Any]:
        return {"present": self.present, "location": self.location, "sizeMb": self.sizeMb, "usedBy": self.usedBy}


# (repo, file proving the download is complete, used by)
HF_MODELS = (
    (OPENCLIP_HF_REPO, ("open_clip_model.safetensors", "open_clip_pytorch_model.bin"), "duplicate embeddings (OpenCLIP)"),
    (GDINO_HF_REPO, ("model.safetensors",), "Grounded SAM 2 boxes (Grounding DINO)"),
    (SAM2_HF_REPO, ("model.safetensors",), "Grounded SAM 2 masks (SAM 2.1)"),
)


def model_status() -> List[ModelStatus]:
    raft = raft_small_path()
    out = [
        ModelStatus("transnetv2.pt", transnet_pt_path().is_file(), str(transnet_pt_path()), _size_mb(transnet_pt_path()),
                    "shots (torch)"),
        ModelStatus("transnetv2.onnx", transnet_onnx_path().is_file(), str(transnet_onnx_path()),
                    _size_mb(transnet_onnx_path()), "shots (onnxruntime)"),
        ModelStatus(YOLO_WORLD_FILE, yolo_world_path().is_file(), str(yolo_world_path()), _size_mb(yolo_world_path()),
                    "perception (YOLO-World)"),
        ModelStatus("clip/ViT-B-32.pt", clip_text_path().is_file(), str(clip_text_path()), _size_mb(clip_text_path()),
                    "YOLO-World prompt encoder"),
        ModelStatus("raft_small", raft is not None, str(raft or (torch_home() / "hub" / "checkpoints")), _size_mb(raft),
                    "transitions / interpolate (RAFT)"),
    ]
    for repo, key_files, used in HF_MODELS:
        d = hf_hub_cache() / ("models--" + repo.replace("/", "--"))
        out.append(ModelStatus(repo, any(hf_repo_present(repo, f) for f in key_files), str(d), _dir_mb(d), used))
    from . import separate  # lazy: separate imports this module

    m = separate.DEFAULT_MODEL
    d = hf_hub_cache() / ("models--" + separate.hf_repo(m).replace("/", "--"))
    out.append(ModelStatus(f"demucs/{m}", separate.model_present(m), str(d), separate.model_size_mb(m),
                           "voice / background separation (Demucs v4)"))
    return out
