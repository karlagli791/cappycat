"""Small file-system helpers shared by every stage.

* **Atomic outputs**: everything the pipeline publishes (analysis JSON, interpolated mp4, stems,
  character cache, per-clip result cache) is written to ``<name>.part`` next to the target and
  moved over it with :func:`os.replace` only once complete, so a killed / cancelled process never
  leaves a truncated file at the real path.
* **Unicode-safe image I/O**: ``cv2.imread`` / ``cv2.imwrite`` silently fail on non-ASCII paths on
  Windows; :func:`imread` / :func:`imwrite` go through ``np.fromfile`` / ``cv2.imencode().tofile``.
"""
from __future__ import annotations

import contextlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np


def part_path(path: os.PathLike | str, suffix: str = "") -> Path:
    """``<path>.<unique>.part<suffix>`` in the same directory (same volume -> ``os.replace`` is atomic).
    ``suffix`` keeps a meaningful extension for tools that pick a format from it (``.part.mp4``)."""
    p = Path(path)
    return p.with_name(f"{p.name}.{os.getpid()}-{uuid.uuid4().hex[:8]}.part{suffix}")


@contextlib.contextmanager
def atomic_path(path: os.PathLike | str, suffix: str = "") -> Iterator[Path]:
    """Yield a temporary path; on success it replaces ``path`` atomically, on error it is removed.

    >>> with atomic_path("out.json") as tmp:
    ...     tmp.write_text("{}")
    """
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = part_path(dst, suffix)
    try:
        yield tmp
        os.replace(tmp, dst)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def write_bytes_atomic(path: os.PathLike | str, data: bytes) -> None:
    with atomic_path(path) as tmp:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())


def write_text_atomic(path: os.PathLike | str, text: str, encoding: str = "utf-8") -> None:
    write_bytes_atomic(path, text.encode(encoding))


def write_json_atomic(path: os.PathLike | str, obj: Any, **dump_kw: Any) -> None:
    dump_kw.setdefault("ensure_ascii", False)
    write_text_atomic(path, json.dumps(obj, **dump_kw))


def save_npz_atomic(path: os.PathLike | str, compressed: bool = True, **arrays: Any) -> None:
    """``np.savez[_compressed]`` through a file handle (numpy would otherwise append ``.npz`` to the
    temporary name) and an atomic replace."""
    with atomic_path(path) as tmp:
        with open(tmp, "wb") as fh:
            (np.savez_compressed if compressed else np.savez)(fh, **arrays)


def imread(path: os.PathLike | str, flags: Optional[int] = None) -> Optional[np.ndarray]:
    """``cv2.imread`` that works with non-ASCII paths (None when unreadable)."""
    import cv2

    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR if flags is None else flags)


def imwrite(path: os.PathLike | str, img: np.ndarray, params: Optional[list] = None) -> bool:
    """``cv2.imwrite`` that works with non-ASCII paths; the extension picks the codec."""
    import cv2

    ext = Path(path).suffix or ".png"
    ok, enc = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    enc.tofile(str(path))
    return True
