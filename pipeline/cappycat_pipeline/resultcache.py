"""Per-clip analysis result cache.

Re-analysing a folder where most clips did not change should take seconds, not minutes. Every
successfully analysed clip is stored as JSON under
``%LOCALAPPDATA%\\cappycat\\cache\\analysis\\clips\\<key>.json`` (``CAPPYCAT_ANALYSIS_CACHE_DIR``
overrides the folder), written atomically. The key is the sha1 of

* the clip's absolute path, size and mtime (ns),
* every option that changes the per-clip result (detector, prompts, shot detector / threshold,
  similarity threshold, smoothing, audio options, sampling / tracking rates, analysis width ...),
* :data:`ANALYSIS_VERSION` + the package version (bump when the analysis itself changes),
* the main-cast bank hash (the manifest + reference images fingerprint, or "none"),
* which models / ML packages are installed (a run that degraded to lite mode is not reused once
  the models are present).

Cross-clip transitions depend on two neighbouring clips, so they are cached separately under
``pairs/<sha1(keyA|keyB)>.json``. ``--no-cache`` bypasses both (nothing is read or written).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from . import __version__, fsutil

log = logging.getLogger("cappycat.resultcache")

ANALYSIS_VERSION = "2026-09-25.3"


def cache_root() -> Path:
    env = os.environ.get("CAPPYCAT_ANALYSIS_CACHE_DIR")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "cappycat" / "cache" / "analysis" / "clips"


def environment_fingerprint() -> Dict[str, Any]:
    """Which models / ML packages are available (cheap: file stats and ``find_spec`` only)."""
    from . import models

    out: Dict[str, Any] = {}
    for mod in ("torch", "ultralytics", "open_clip", "transformers", "librosa", "onnxruntime"):
        try:
            out[mod] = importlib.util.find_spec(mod) is not None
        except Exception:
            out[mod] = False
    out["transnetv2"] = models.transnet_pt_path().is_file() or models.transnet_onnx_path().is_file()
    out["yolo"] = models.yolo_world_path().is_file()
    out["raft"] = models.raft_small_path() is not None
    for repo, key_files, _ in models.HF_MODELS:
        out[repo] = any(models.hf_repo_present(repo, f) for f in key_files)
    out["device"] = os.environ.get("CAPPYCAT_DEVICE", "")
    return out


def clip_key(path: os.PathLike | str, options: Dict[str, Any], bank_hash: Optional[str],
             env: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Cache key for one clip, or None when the file cannot be stat'ed."""
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return None
    payload = {"path": os.path.abspath(p).replace("\\", "/").lower() if os.name == "nt" else os.path.abspath(p),
               "size": st.st_size, "mtime": st.st_mtime_ns, "options": options, "bank": bank_hash or "none",
               "version": [ANALYSIS_VERSION, __version__], "env": env if env is not None else environment_fingerprint()}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def pair_key(key_a: str, key_b: str) -> str:
    return hashlib.sha1(f"{key_a}|{key_b}|{ANALYSIS_VERSION}".encode("utf-8")).hexdigest()


def _path(key: str, sub: str = "") -> Path:
    root = cache_root() / sub if sub else cache_root()
    return root / f"{key}.json"


def load(key: Optional[str], sub: str = "") -> Optional[Dict[str, Any]]:
    if not key:
        return None
    p = _path(key, sub)
    try:
        if not p.is_file():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("version") != ANALYSIS_VERSION or d.get("key") != key:
            return None
        return d
    except Exception as exc:  # unreadable / truncated entry: treat as a miss
        log.info("analysis cache entry %s unreadable (%s)", p, exc)
        return None


def save(key: Optional[str], payload: Dict[str, Any], sub: str = "") -> Optional[Path]:
    if not key:
        return None
    p = _path(key, sub)
    try:
        fsutil.write_json_atomic(p, {"version": ANALYSIS_VERSION, "key": key, **payload}, separators=(",", ":"))
        return p
    except OSError as exc:
        log.warning("could not write the analysis cache (%s)", exc)
        return None
