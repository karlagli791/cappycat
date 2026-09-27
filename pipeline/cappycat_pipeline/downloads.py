"""``download-models``: fetch every model of the ML path into ``models_dir()`` (idempotent).

Each step checks what is already on disk first, so re-running is cheap and a partially
set-up machine only downloads what is missing. Nothing here runs at analysis time.
"""
from __future__ import annotations

import inspect
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import models

log = logging.getLogger("cappycat.downloads")

# (model name, overall pct 0..1, message[, extra: {bytes, totalBytes, overallBytes, overallTotalBytes}])
Progress = Callable[..., None]


# --------------------------------------------------------------------------- TransNetV2


def ensure_transnet_pt(force: bool = False) -> Path:
    dst = models.transnet_pt_path()
    if dst.is_file() and not force:
        return dst
    from huggingface_hub import hf_hub_download

    last: Optional[Exception] = None
    for repo in models.TRANSNET_HF_REPOS:
        try:
            src = hf_hub_download(repo, models.TRANSNET_HF_FILE, cache_dir=str(models.hf_hub_cache()))
            break
        except Exception as exc:  # try the mirror
            last = exc
    else:
        raise RuntimeError(f"could not download TransNetV2 weights: {last}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    shutil.copyfile(src, tmp)
    # validate before publishing: must load into the vendored network
    from .transnetv2_model import load_transnetv2

    load_transnetv2(tmp, "cpu")
    tmp.replace(dst)
    return dst


def export_transnet_onnx(pt_path: Optional[Path] = None, onnx_path: Optional[Path] = None, verify: bool = True) -> Path:
    """Export the torch network to ONNX (uint8 ``[batch, 100, 27, 48, 3]`` -> logits
    ``[batch, 100, 1]``) and check onnxruntime reproduces torch on random input."""
    import numpy as np
    import torch  # type: ignore

    from .transnetv2_model import TransNetV2SingleFrame, load_transnetv2

    pt_path = Path(pt_path or models.transnet_pt_path())
    onnx_path = Path(onnx_path or models.transnet_onnx_path())
    net = TransNetV2SingleFrame(load_transnetv2(pt_path, "cpu")).eval()
    dummy = torch.zeros(1, 100, 27, 48, 3, dtype=torch.uint8)
    tmp = onnx_path.with_suffix(".part.onnx")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:  # TorchScript exporter: single self-contained file, dynamic batch
            torch.onnx.export(net, (dummy,), str(tmp), opset_version=17, input_names=["frames"],
                              output_names=["single_frame_logits"],
                              dynamic_axes={"frames": {0: "batch"}, "single_frame_logits": {0: "batch"}}, dynamo=False)
        except Exception as exc:  # exporter removed in a future torch -> dynamo exporter
            log.info("legacy ONNX export failed (%s); trying the dynamo exporter", exc)
            torch.onnx.export(net, (dummy,), str(tmp), input_names=["frames"], output_names=["single_frame_logits"],
                              dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=64)},), dynamo=True,
                              external_data=False)
    if verify:
        import onnxruntime as ort

        x = np.random.default_rng(0).integers(0, 255, (2, 100, 27, 48, 3)).astype(np.uint8)
        sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
        y = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        with torch.inference_mode():
            ref = net(torch.from_numpy(x)).numpy()
        diff = float(np.abs(y - ref).max())
        del sess
        if diff > 1e-3:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ONNX export does not match torch (max |diff| = {diff:.2e})")
    tmp.replace(onnx_path)
    return onnx_path


# --------------------------------------------------------------------------- others


def ensure_yolo_world() -> Path:
    dst = models.yolo_world_path()
    if not dst.is_file():
        models.import_ultralytics()
        from ultralytics.utils.downloads import attempt_download_asset  # type: ignore

        got = Path(attempt_download_asset(str(dst)))
        if got.resolve() != dst.resolve() and got.is_file():
            shutil.move(str(got), dst)
    if not dst.is_file():
        raise RuntimeError(f"YOLO-World weights were not downloaded to {dst}")
    return dst


def ensure_clip_text() -> Path:
    """OpenAI CLIP ViT-B/32 (the text tower encodes YOLO-World prompts)."""
    dst = models.clip_text_path()
    if not dst.is_file():
        models.import_ultralytics()
        import clip  # type: ignore  (ultralytics' CLIP fork)

        dst.parent.mkdir(parents=True, exist_ok=True)
        clip.load("ViT-B/32", device="cpu", download_root=str(dst.parent))
    if not dst.is_file():
        raise RuntimeError(f"CLIP text encoder was not downloaded to {dst}")
    return dst


def ensure_openclip() -> None:
    if models.hf_repo_present(models.OPENCLIP_HF_REPO, "open_clip_model.safetensors") or \
            models.hf_repo_present(models.OPENCLIP_HF_REPO, "open_clip_pytorch_model.bin"):
        return
    import open_clip  # type: ignore

    m = open_clip.create_model_and_transforms(models.OPENCLIP_MODEL, pretrained=models.OPENCLIP_PRETRAINED, device="cpu",
                                              cache_dir=str(models.hf_hub_cache()))
    del m


def ensure_hf_snapshot(repo: str, key_file: str = "model.safetensors") -> None:
    if models.hf_repo_present(repo, key_file) and models.hf_repo_present(repo, "config.json"):
        return
    from huggingface_hub import snapshot_download

    snapshot_download(repo, cache_dir=str(models.hf_hub_cache()), allow_patterns=models.HF_ALLOW_PATTERNS)


def ensure_raft() -> None:
    if models.raft_small_path() is not None:
        return
    models.configure_env()
    from torchvision.models.optical_flow import Raft_Small_Weights  # type: ignore

    Raft_Small_Weights.DEFAULT.get_state_dict(progress=False)
    if models.raft_small_path() is None:
        raise RuntimeError("RAFT-small weights were not stored under models/torch (is TORCH_HOME set elsewhere?)")


def ensure_demucs() -> None:
    """Demucs v4 ``htdemucs_ft`` bag (yaml + 4 safetensors, ~320 MB) into ``models/hf``."""
    from . import separate

    separate.ensure_model(separate.DEFAULT_MODEL)


def steps() -> List[Tuple[str, Callable[[], object]]]:
    return [
        ("transnetv2.pt", ensure_transnet_pt),
        ("transnetv2.onnx", lambda: models.transnet_onnx_path() if models.transnet_onnx_path().is_file() else export_transnet_onnx()),
        (models.YOLO_WORLD_FILE, ensure_yolo_world),
        ("clip/ViT-B-32.pt", ensure_clip_text),
        (models.OPENCLIP_HF_REPO, ensure_openclip),
        (models.GDINO_HF_REPO, lambda: ensure_hf_snapshot(models.GDINO_HF_REPO)),
        (models.SAM2_HF_REPO, lambda: ensure_hf_snapshot(models.SAM2_HF_REPO)),
        ("raft_small", ensure_raft),
        ("demucs/htdemucs_ft", ensure_demucs),
    ]


# Download size of each step in bytes (what it adds to ``models_dir()``, measured on a complete
# install); reported as ``totalBytes`` while the step runs. 0 = nothing is downloaded (local export).
EXPECTED_BYTES: Dict[str, int] = {
    "transnetv2.pt": 30_508_183,
    "transnetv2.onnx": 0,
    models.YOLO_WORLD_FILE: 25_923_032,
    "clip/ViT-B-32.pt": 353_976_522,
    models.OPENCLIP_HF_REPO: 605_143_356,
    models.GDINO_HF_REPO: 690_307_209,
    models.SAM2_HF_REPO: 184_313_930,
    "raft_small": 4_006_189,
    "demucs/htdemucs_ft": 336_101_949,
}

# Where each step's bytes land while it downloads (HF: ``blobs/*.incomplete`` inside the repo folder;
# torch.hub / ultralytics / CLIP: a temporary or final file next to the target).
def watch_paths(name: str) -> List[Tuple[Path, str]]:
    """``[(folder, glob)]`` whose growth is the step's downloaded bytes."""
    hub = models.hf_hub_cache()

    def repo(r: str) -> Tuple[Path, str]:
        return hub / ("models--" + r.replace("/", "--")), "**/*"

    if name == "transnetv2.pt":
        return [repo(r) for r in models.TRANSNET_HF_REPOS]
    if name == models.YOLO_WORLD_FILE:
        return [(models.models_dir(), Path(models.YOLO_WORLD_FILE).stem + "*")]
    if name == "clip/ViT-B-32.pt":
        return [(models.clip_text_path().parent, "*")]
    if name == "raft_small":
        return [(models.torch_home() / "hub" / "checkpoints", "*")]
    if name == "demucs/htdemucs_ft":
        from . import separate

        return [repo(separate.hf_repo(separate.DEFAULT_MODEL))]
    if name in (models.OPENCLIP_HF_REPO, models.GDINO_HF_REPO, models.SAM2_HF_REPO):
        return [repo(name)]
    return []


def _bytes_in(paths: List[Tuple[Path, str]]) -> int:
    total = 0
    for folder, pattern in paths:
        if not folder.is_dir():
            continue
        for p in folder.glob(pattern):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                pass
    return total


WATCH_INTERVAL_S = 0.5


class _ByteWatcher:
    """Polls the size of a step's download folders on a thread and reports the growth."""

    def __init__(self, paths: List[Tuple[Path, str]], on_bytes: Callable[[int], None], interval: Optional[float] = None):
        self.paths, self.on_bytes = paths, on_bytes
        self.interval = WATCH_INTERVAL_S if interval is None else interval
        self.base = _bytes_in(paths)
        self.bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="download-bytes", daemon=True)

    def _run(self) -> None:
        last = 0
        while not self._stop.wait(self.interval):
            try:
                self.bytes = max(0, _bytes_in(self.paths) - self.base)
            except Exception:
                continue
            if self.bytes != last:
                last = self.bytes
                self.on_bytes(self.bytes)

    def __enter__(self) -> "_ByteWatcher":
        if self.paths:
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self.bytes = max(self.bytes, _bytes_in(self.paths) - self.base) if self.paths else 0


def _mb(n: float) -> str:
    return f"{n / 2**20:.1f} MB"


def _call(progress: Callable[..., None], name: str, pct: float, msg: str, extra: Dict[str, object]) -> None:
    """Progress callbacks take ``(name, pct, msg)`` and optionally a 4th ``extra`` dict (byte counts)."""
    try:
        params = inspect.signature(progress).parameters.values()
        wants = any(p.kind is p.VAR_POSITIONAL for p in params) or \
            len([p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]) >= 4
    except (TypeError, ValueError):
        wants = False
    if wants:
        progress(name, pct, msg, extra)
    else:
        progress(name, pct, msg)


def download_all(progress: Progress, only: Optional[List[str]] = None,
                 names: Optional[List[str]] = None) -> List[Tuple[str, Optional[str]]]:
    """Run every step (``only``: substring filter; ``names``: exact step names); returns
    ``[(name, error_or_None)]``. A failing step does not stop the others. Idempotent: present
    models are reported as such and not fetched again.

    ``progress(name, pct, msg[, extra])``: ``pct`` is the overall fraction, weighted by the expected
    download size of the missing steps; ``extra`` (when the callback takes a 4th argument) carries
    ``bytes`` / ``totalBytes`` of the current step and ``overallBytes`` / ``overallTotalBytes``,
    updated about twice a second while a step downloads (``totalBytes`` is the expected size and is
    raised to ``bytes`` if a download turns out larger)."""
    models.models_dir().mkdir(parents=True, exist_ok=True)
    todo = [(n, f) for n, f in steps() if (not only or any(o.lower() in n.lower() for o in only))
            and (names is None or n in names)]
    missing = [n for n, _ in todo if not _present(n)]
    totals = {n: EXPECTED_BYTES.get(n, 0) for n in missing}
    weight = {n: float(max(totals[n], 2**20)) for n in missing}  # an export still moves the bar a little
    grand_w = sum(weight.values())
    grand_b = sum(totals.values())
    done_w = 0.0
    done_b = 0
    lock = threading.Lock()
    results: List[Tuple[str, Optional[str]]] = []

    def report(i: int, name: str, frac: float, msg: str, got: int = 0, final: bool = False) -> None:
        tot = max(totals.get(name, 0), got)
        if grand_w > 0:
            pct = (done_w + weight.get(name, 0.0) * min(max(frac, 0.0), 1.0)) / grand_w
        else:
            pct = (i + (1.0 if final else 0.0)) / max(1, len(todo))
        extra = {"bytes": int(got), "totalBytes": int(tot), "overallBytes": int(done_b + got),
                 "overallTotalBytes": int(max(grand_b, done_b + got))}
        with lock:
            _call(progress, name, min(pct, 1.0), msg, extra)

    for i, (name, fn) in enumerate(todo):
        if name not in totals:
            report(i, name, 1.0, f"{name}: already present", final=True)
            results.append((name, None))
            continue
        expected = totals[name]
        report(i, name, 0.0, f"{name}: downloading" + (f" ({_mb(expected)})" if expected else " (exporting)"))

        def on_bytes(b: int, i: int = i, name: str = name, expected: int = expected) -> None:
            if expected:
                report(i, name, min(b / expected, 0.99), f"{name}: {_mb(b)} / {_mb(max(expected, b))}", b)

        err: Optional[str] = None
        with _ByteWatcher(watch_paths(name) if expected else [], on_bytes) as watcher:
            try:
                fn()
            except Exception as exc:
                log.debug("download step %s failed", name, exc_info=True)
                err = f"{type(exc).__name__}: {exc}"
        got = watcher.bytes
        results.append((name, err))
        report(i, name, 1.0, f"{name}: FAILED ({err})" if err else f"{name}: ok" + (f" ({_mb(got)})" if got else ""),
               got, final=True)
        done_w += weight[name]
        done_b += got
    models.release()
    return results


# --------------------------------------------------------------------------- first run of `analyze`


def _present(name: str) -> bool:
    if name == "transnetv2.pt":
        return models.transnet_pt_path().is_file()
    if name == "transnetv2.onnx":
        return models.transnet_onnx_path().is_file()
    if name == models.YOLO_WORLD_FILE:
        return models.yolo_world_path().is_file()
    if name == "clip/ViT-B-32.pt":
        return models.clip_text_path().is_file()
    if name == "raft_small":
        return models.raft_small_path() is not None
    if name == "demucs/htdemucs_ft":
        from . import separate

        return separate.model_present(separate.DEFAULT_MODEL)
    for repo, key_files, _ in models.HF_MODELS:
        if name == repo:
            return any(models.hf_repo_present(repo, f) for f in key_files)
    return True


def models_needed_for_analysis(detector: str, shot_detector: str, prompts: List[str], characters_stale: bool) -> List[str]:
    """Model steps (names of :func:`steps`) an ``analyze`` run with these options will load. Only the
    ML path's models; nothing when torch / ultralytics are not installed (lite mode)."""
    import importlib.util

    def has(mod: str) -> bool:
        try:
            return importlib.util.find_spec(mod) is not None
        except Exception:
            return False

    torch_ok = has("torch")
    need: List[str] = []
    if shot_detector in ("auto", "transnetv2") and (torch_ok or has("onnxruntime")):
        need.append("transnetv2.pt" if torch_ok else "transnetv2.onnx")
    detector = (detector or "hybrid").lower()
    if detector in ("hybrid", "yolo_world", "grounded_sam2") and torch_ok and has("ultralytics"):
        if detector != "grounded_sam2":
            need.append(models.YOLO_WORLD_FILE)
            from .perception import _yolo_text_cache_path

            if not _yolo_text_cache_path(list(prompts), str(models.yolo_world_path())).is_file():
                need.append("clip/ViT-B-32.pt")  # prompt encoder (only until the text features are cached)
        if has("open_clip"):
            need.append(models.OPENCLIP_HF_REPO)
    if torch_ok and has("transformers") and (detector in ("hybrid", "grounded_sam2") or characters_stale):
        need.append(models.GDINO_HF_REPO)
        if detector in ("hybrid", "grounded_sam2"):
            need.append(models.SAM2_HF_REPO)
    if torch_ok and has("torchvision") and os.environ.get("CAPPYCAT_FLOW_BACKEND", "auto").lower() in ("auto", "raft"):
        need.append("raft_small")
    return need


def ensure_models(names: List[str], progress: Progress) -> Tuple[List[str], List[Tuple[str, Optional[str]]]]:
    """Download the missing ones of ``names``; returns ``(missing_before, [(name, error)])``."""
    missing = [n for n in names if not _present(n)]
    if not missing:
        return [], []
    results = download_all(progress, only=None, names=missing)
    return missing, results
