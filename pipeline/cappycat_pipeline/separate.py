"""Voice / background stem separation (CapCut-style "Isolate voice" / "Remove vocals").

Model: Meta's **Demucs v4** hybrid transformer (``htdemucs_ft`` by default, MIT licence), loaded from the
``adefossez/HTDemucs-ft`` Hugging Face repo into ``models_dir()/hf`` (``download-models``). The ``demucs``
package is only used for the network definition and ``apply_model``: audio is decoded with ffmpeg here,
so ``torchaudio`` (which has no build for torch 2.14 / cp314) is never imported.

Two stems per source file:

* ``vocals``     - the model's vocals source (dialogue, singing)
* ``background`` - every other source summed (drums + bass + other: ambience, music, SFX)

Stems are written as 48 kHz stereo float WAVs to
``%LOCALAPPDATA%\\cappycat\\cache\\stems\\<sha1(path|size|mtime|model)>\\{vocals,background}.wav``
(``CAPPYCAT_STEMS_DIR`` overrides the root) and reused when present. Timing follows the exporter's
convention: stem sample 0 is the file's time 0 (``ffmpeg -ss 0``), i.e. the audio stream's start offset
relative to the container start is padded with silence, and the stems end where the source audio ends,
so ``-ss X`` on a stem and ``-ss X`` on the source return the same instant.

Processing is segment-wise (``split=True``, 25 % overlap, the model's 7.8 s segments) with the full mix
kept on the CPU, so VRAM stays bounded (~0.9 GB peak for the ``htdemucs_ft`` bag in fp16 autocast) no
matter how long the clip is. Falls back to the CPU when CUDA is unavailable, the VRAM budget is too
small, or the GPU runs out of memory.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import ffmpeg_util, models

log = logging.getLogger("cappycat.separate")

DEFAULT_MODEL = "htdemucs_ft"
SUPPORTED_MODELS = ("htdemucs_ft", "htdemucs")
DEMUCS_HF_NAMESPACE = "adefossez"
STEM_RATE = 48_000
STEM_NAMES = ("vocals", "background")
OVERLAP = 0.25
# below this much usable VRAM the GPU path is not attempted (measured peak ~0.9 GB for htdemucs_ft)
MIN_VRAM_MB = 1100.0

Progress = Callable[[float, str], None]  # (pct 0..1 within the file, message)


# --------------------------------------------------------------------------- model files


def hf_repo(model: str) -> str:
    """``htdemucs_ft`` -> ``adefossez/HTDemucs-ft`` (same mapping as ``demucs.hf.hf_repo_name``)."""
    if model == "htdemucs":
        name = "HTDemucs"
    elif model.startswith("htdemucs_"):
        name = "HTDemucs-" + model[len("htdemucs_"):]
    else:
        name = "Demucs-" + model
    return f"{DEMUCS_HF_NAMESPACE}/{name}"


def _repo_dir(model: str) -> Path:
    return models.hf_hub_cache() / ("models--" + hf_repo(model).replace("/", "--"))


def local_model_files(model: str = DEFAULT_MODEL) -> Optional[Tuple[Path, List[Path]]]:
    """``(bag yaml, [safetensors...])`` from the local HF cache, or None when incomplete."""
    snaps = _repo_dir(model) / "snapshots"
    if not snaps.is_dir():
        return None
    import yaml  # type: ignore

    for snap in sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        yml = snap / f"{model}.yaml"
        if not yml.is_file():
            continue
        try:
            bag = yaml.safe_load(yml.read_text(encoding="utf-8"))
            files = [snap / f"{sig}.safetensors" for sig in bag["models"]]
        except Exception:
            continue
        if files and all(f.is_file() for f in files):
            return yml, files
    return None


def model_present(model: str = DEFAULT_MODEL) -> bool:
    try:
        return local_model_files(model) is not None
    except Exception:
        return False


def model_size_mb(model: str = DEFAULT_MODEL) -> Optional[float]:
    found = local_model_files(model) if model_present(model) else None
    if not found:
        return None
    return round(sum(p.stat().st_size for p in found[1]) / 2**20, 1)


def ensure_model(model: str = DEFAULT_MODEL) -> Tuple[Path, List[Path]]:
    """Download the bag yaml + weights into ``models_dir()/hf`` (no-op when present)."""
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported separation model {model!r} (choose from {', '.join(SUPPORTED_MODELS)})")
    found = local_model_files(model)
    if found:
        return found
    models.configure_env()
    import yaml  # type: ignore
    from huggingface_hub import hf_hub_download

    repo, cache = hf_repo(model), str(models.hf_hub_cache())
    yml = Path(hf_hub_download(repo, f"{model}.yaml", cache_dir=cache))
    bag = yaml.safe_load(yml.read_text(encoding="utf-8"))
    for sig in bag["models"]:
        hf_hub_download(repo, f"{sig}.safetensors", cache_dir=cache)
    found = local_model_files(model)
    if not found:
        raise RuntimeError(f"Demucs {model} weights were not stored under {_repo_dir(model)}")
    return found


def load_model(model: str = DEFAULT_MODEL):
    """The Demucs ``BagOfModels`` on the CPU (downloads the weights on first use)."""
    yml, files = ensure_model(model)
    import yaml  # type: ignore
    from demucs.apply import BagOfModels  # type: ignore
    from demucs.hf import load_safetensors_model  # type: ignore

    bag = yaml.safe_load(yml.read_text(encoding="utf-8"))
    nets = [load_safetensors_model(f) for f in files]
    out = BagOfModels(nets, bag.get("weights"), bag.get("segment"))
    out.eval()
    return out


_MODEL_CACHE: Dict[str, Any] = {}


def _get_model(model: str):
    if model not in _MODEL_CACHE:
        _MODEL_CACHE[model] = load_model(model)
    return _MODEL_CACHE[model]


def release_model() -> None:
    _MODEL_CACHE.clear()
    models.release()


# --------------------------------------------------------------------------- cache


def stems_root() -> Path:
    env = os.environ.get("CAPPYCAT_STEMS_DIR")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "cappycat" / "cache" / "stems"


def cache_key(path: str | os.PathLike, model: str = DEFAULT_MODEL) -> str:
    p = Path(path)
    st = p.stat()
    src = f"{ffmpeg_util.norm_path(p)}|{st.st_size}|{int(st.st_mtime)}|{model}"
    return hashlib.sha1(src.encode("utf-8")).hexdigest()


def stem_paths(path: str | os.PathLike, model: str = DEFAULT_MODEL, root: Optional[Path] = None) -> Dict[str, Path]:
    d = Path(root or stems_root()) / cache_key(path, model)
    return {name: d / f"{name}.wav" for name in STEM_NAMES}


def _out_path(p: Path) -> str:
    """Absolute, forward slashes, without resolving links (keeps ``%LOCALAPPDATA%`` as the user sees it)."""
    return os.path.abspath(p).replace("\\", "/")


def cached_stems(path: str | os.PathLike, model: str = DEFAULT_MODEL, root: Optional[Path] = None) -> Optional[Dict[str, str]]:
    sp = stem_paths(path, model, root)
    meta = next(iter(sp.values())).parent / "meta.json"
    if meta.is_file() and all(p.is_file() and p.stat().st_size > 44 for p in sp.values()):
        return {k: _out_path(v) for k, v in sp.items()}
    return None


# --------------------------------------------------------------------------- audio I/O


def write_wav_f32(path: Path, samples: np.ndarray, rate: int = STEM_RATE) -> None:
    """Interleaved float32 WAV (WAVE_FORMAT_IEEE_FLOAT). ``samples`` shape ``(n, channels)``."""
    x = np.ascontiguousarray(samples, dtype="<f4")
    ch = 1 if x.ndim == 1 else int(x.shape[1])
    data = x.tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 3, ch, rate, rate * ch * 4, ch * 4, 32))
        f.write(b"data" + struct.pack("<I", len(data)))
        f.write(data)


def read_wav_f32(path: str | os.PathLike) -> Tuple[np.ndarray, int]:
    """Read a WAV written by :func:`write_wav_f32` (``(n, channels)``, rate)."""
    b = Path(path).read_bytes()
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a WAV file")
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(b):
        cid, size = b[pos:pos + 4], struct.unpack("<I", b[pos + 4:pos + 8])[0]
        body = b[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if fmt is None or data is None or fmt[0] != 3 or fmt[5] != 32:
        raise ValueError(f"{path}: expected a 32-bit float WAV")
    ch, rate = fmt[1], fmt[2]
    return np.frombuffer(data, dtype="<f4").reshape(-1, ch), rate


def audio_offset_ms(path: str | os.PathLike) -> Optional[float]:
    """Start of the first audio stream relative to the container start (what ``-ss 0`` means), or
    None when the file has no audio stream."""
    info = ffmpeg_util.probe_raw(str(path))
    audio = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio:
        return None

    def _f(v: Any) -> Optional[float]:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    a0 = _f(audio[0].get("start_time"))
    starts = [x for x in (_f(s.get("start_time")) for s in info.get("streams", [])) if x is not None]
    f0 = _f(info.get("format", {}).get("start_time"))
    if f0 is None:
        f0 = min(starts) if starts else 0.0
    if a0 is None:
        return 0.0
    return max(0.0, (a0 - f0) * 1000.0)


def _resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Polyphase resampling along axis 0 (``(n, ch)``)."""
    if sr_from == sr_to or len(x) == 0:
        return x.astype(np.float32, copy=False)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr_from, sr_to)
    return resample_poly(x, sr_to // g, sr_from // g, axis=0).astype(np.float32)


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros((n - len(x),) + x.shape[1:], dtype=x.dtype)], axis=0)


# --------------------------------------------------------------------------- separation


@dataclass
class SeparationResult:
    path: str
    stems: Dict[str, str]
    cached: bool
    model: str
    device: Optional[str] = None
    seconds: float = 0.0
    audioSeconds: float = 0.0
    offsetMs: float = 0.0
    peakVramMb: Optional[float] = None
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {"path": self.path, "stems": self.stems, "cached": self.cached, "model": self.model,
                "device": self.device, "seconds": round(self.seconds, 3), "audioSeconds": round(self.audioSeconds, 3),
                "offsetMs": round(self.offsetMs, 3), "peakVramMb": self.peakVramMb, "stats": self.stats}


def pick_device(requested: Optional[str] = None) -> str:
    """``cuda`` when requested/available and the VRAM budget allows the model, else ``cpu``."""
    if requested == "cpu":
        return "cpu"
    if not models.cuda_ok():
        if requested == "cuda":
            log.warning("CUDA requested but not available; separating on the CPU")
        return "cpu"
    if models.vram_budget_mb() < MIN_VRAM_MB:
        log.warning("only %.0f MB of VRAM usable; separating on the CPU", models.vram_budget_mb())
        return "cpu"
    return "cuda"


def _run_model(bag, mix: np.ndarray, device: str, progress: Optional[Progress]) -> np.ndarray:
    """``mix`` ``(n, 2)`` at the model rate -> sources ``(S, n, 2)`` (de-normalised)."""
    import torch  # type: ignore
    from demucs.apply import apply_model  # type: ignore

    wav = torch.from_numpy(np.ascontiguousarray(mix.T, dtype=np.float32))
    ref = wav.mean(0)
    mean, std = float(ref.mean()), float(ref.std())
    std = std if std > 1e-8 else 1.0
    x = ((wav - mean) / std)[None]
    n_models = len(getattr(bag, "models", [bag]))
    seg = int(bag.samplerate * float(min(float(getattr(m, "segment", 7.8)) for m in getattr(bag, "models", [bag]))))
    stride = max(1, int((1 - OVERLAP) * seg))
    total = n_models * max(1, len(range(0, x.shape[-1], stride)))
    done = [0]

    def cb(d: Dict[str, Any]) -> None:
        if d.get("state") == "end":
            done[0] += 1
            if progress:
                progress(min(1.0, done[0] / total), f"model {d.get('model_idx_in_bag', 0) + 1}/{n_models}, "
                                                    f"segment {done[0]}/{total}")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
        out = apply_model(bag, x, shifts=0, split=True, overlap=OVERLAP, device=device, callback=cb, progress=False)
    out = out[0].float() * std + mean  # (S, 2, n)
    return out.permute(0, 2, 1).contiguous().numpy()


def separate_file(path: str, model: str = DEFAULT_MODEL, device: Optional[str] = None,
                  cache_root: Optional[Path] = None, progress: Optional[Progress] = None,
                  force: bool = False) -> SeparationResult:
    """Separate one media file into vocals / background stems (cached)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"media not found: {path}")
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported separation model {model!r} (choose from {', '.join(SUPPORTED_MODELS)})")
    t0 = time.perf_counter()
    if not force:
        hit = cached_stems(p, model, cache_root)
        if hit:
            if progress:
                progress(1.0, "stems cached")
            return SeparationResult(path=path, stems=hit, cached=True, model=model, seconds=time.perf_counter() - t0)

    offset_ms = audio_offset_ms(p)
    if offset_ms is None:
        raise ValueError(f"{p.name} has no audio stream")
    if progress:
        progress(0.02, "decoding audio")
    ref48 = ffmpeg_util.decode_pcm(p, sr=STEM_RATE, mono=False)  # exactly what the exporter decodes
    if len(ref48) == 0:
        raise ValueError(f"{p.name}: audio could not be decoded")

    if progress:
        progress(0.05, "loading Demucs " + model)
    bag = _get_model(model)
    sr = int(bag.samplerate)
    mix = ffmpeg_util.decode_pcm(p, sr=sr, mono=False)
    dev = pick_device(device)
    import torch  # type: ignore

    def sub(pct: float, msg: str) -> None:
        if progress:
            progress(0.08 + 0.84 * pct, msg)

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        sources = _run_model(bag, mix, dev, sub)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if dev != "cuda" or "out of memory" not in str(exc).lower():
            raise
        log.warning("GPU out of memory during separation (%s); retrying on the CPU", exc)
        models.release()
        dev = "cpu"
        sources = _run_model(bag, mix, dev, sub)
    peak = models.peak_vram_mb() if dev == "cuda" else None
    if dev == "cuda":
        bag.to("cpu")
        models.release()

    names = list(bag.sources)
    vi = names.index("vocals")
    vocals = sources[vi]
    background = sources.sum(axis=0) - vocals
    if progress:
        progress(0.94, "writing stems")
    n48 = len(ref48)
    pad = int(round(offset_ms * STEM_RATE / 1000.0))
    stems_48 = {}
    for name, x in (("vocals", vocals), ("background", background)):
        y = _fit(_resample(x, sr, STEM_RATE), n48)
        if pad:
            y = np.concatenate([np.zeros((pad, y.shape[1]), np.float32), y], axis=0)
        stems_48[name] = y

    # sanity numbers (logged): how much of the mix the stems explain, and their share of the energy
    def rms(a: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(a, dtype=np.float64)))) if a.size else 0.0

    recon = stems_48["vocals"][pad:] + stems_48["background"][pad:]
    stats = {"mixRms": round(rms(ref48), 5), "vocalsRms": round(rms(stems_48["vocals"]), 5),
             "backgroundRms": round(rms(stems_48["background"]), 5),
             "residualRel": round(rms(recon - ref48) / max(1e-9, rms(ref48)), 4)}

    final = stem_paths(p, model, cache_root)
    out_dir = next(iter(final.values())).parent
    tmp = out_dir.with_name(out_dir.name + f".part{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    for name, y in stems_48.items():
        write_wav_f32(tmp / f"{name}.wav", y, STEM_RATE)
    meta = {"source": ffmpeg_util.norm_path(p), "model": model, "rate": STEM_RATE, "channels": 2,
            "samples": int(n48 + pad), "offsetMs": offset_ms, "device": dev, "stats": stats,
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)
    if progress:
        progress(1.0, "stems ready")
    return SeparationResult(path=path, stems={k: _out_path(v) for k, v in final.items()}, cached=False,
                            model=model, device=dev, seconds=time.perf_counter() - t0,
                            audioSeconds=(n48 + pad) / STEM_RATE, offsetMs=offset_ms, peakVramMb=peak, stats=stats)
