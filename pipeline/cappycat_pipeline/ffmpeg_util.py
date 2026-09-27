"""ffmpeg / ffprobe helpers.

Binary discovery order: ``CAPPYCAT_FFMPEG_DIR`` env var -> ``shutil.which`` ->
``%LOCALAPPDATA%\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg*\\ffmpeg-*\\bin``.

Frames are decoded through a rawvideo pipe rather than ``cv2.VideoCapture`` (which
mis-seeks on some codecs), and all ffmpeg stderr is swallowed so the pipeline's stdout
stays a clean JSON-lines stream.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import procs
from .schema import Asset, new_id

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
_EXE = ".exe" if sys.platform == "win32" else ""


class FFmpegNotFound(RuntimeError):
    pass


class FFmpegError(RuntimeError):
    pass


def _candidate_dirs() -> List[Path]:
    dirs: List[Path] = []
    env = os.environ.get("CAPPYCAT_FFMPEG_DIR")
    if env:
        dirs.append(Path(env))
    local = os.environ.get("LOCALAPPDATA")
    if local:
        pattern = os.path.join(local, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*", "ffmpeg-*", "bin")
        for hit in sorted(glob.glob(pattern), reverse=True):
            dirs.append(Path(hit))
    return dirs


def _find_binary(name: str) -> str:
    env = os.environ.get("CAPPYCAT_FFMPEG_DIR")
    if env:
        p = Path(env) / (name + _EXE)
        if p.is_file():
            return str(p)
    found = shutil.which(name)
    if found:
        return found
    for d in _candidate_dirs():
        p = d / (name + _EXE)
        if p.is_file():
            return str(p)
    raise FFmpegNotFound(
        f"{name} not found. Install ffmpeg (winget install Gyan.FFmpeg) or set CAPPYCAT_FFMPEG_DIR to its bin folder."
    )


def find_ffmpeg() -> str:
    return _find_binary("ffmpeg")


def find_ffprobe() -> str:
    return _find_binary("ffprobe")


def ffmpeg_available() -> bool:
    try:
        find_ffmpeg()
        find_ffprobe()
        return True
    except FFmpegNotFound:
        return False


def _run(cmd: Sequence[str], *, capture_stderr: bool = False, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    # through procs.run so the stdin watchdog can kill it when the parent goes away
    return procs.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
                     timeout=timeout)


def norm_path(path: str | os.PathLike) -> str:
    """Absolute path with forward slashes (what the contract examples use)."""
    return str(Path(path).resolve()).replace("\\", "/")


# --------------------------------------------------------------------------- probe


def _parse_rate(s: Optional[str]) -> float:
    if not s:
        return 0.0
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        try:
            return float(s)
        except ValueError:
            return 0.0


def probe_raw(path: str) -> dict:
    cmd = [find_ffprobe(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    proc = _run(cmd, capture_stderr=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed for {path}: {proc.stderr.decode('utf-8', 'replace').strip()}")
    return json.loads(proc.stdout.decode("utf-8", "replace") or "{}")


def probe(path: str | os.PathLike, order: Optional[int] = None) -> Asset:
    """Probe a media file into an :class:`Asset`."""
    p = Path(path)
    info = probe_raw(str(p))
    streams = info.get("streams", [])
    fmt = info.get("format", {})
    video = next((s for s in streams if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic", 0) != 1), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = 0.0
    for src in (video, fmt, audio):
        if src and src.get("duration"):
            try:
                duration = float(src["duration"])
                break
            except ValueError:
                pass

    width = height = 0
    fps = 0.0
    codec = None
    kind = "audio"
    if video:
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
        fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
        codec = video.get("codec_name")
        is_image = (
            video.get("codec_name") in ("png", "mjpeg", "bmp", "webp", "tiff", "gif")
            and (int(video.get("nb_frames") or 1) <= 1 or duration <= 0.05)
        ) or fmt.get("format_name", "").startswith(("image2", "png_pipe"))
        kind = "image" if is_image else "video"
        if kind == "image":
            fps = 0.0
    elif audio:
        codec = audio.get("codec_name")
    else:
        kind = "video"

    return Asset(
        id=new_id("ast"),
        path=norm_path(p),
        name=p.name,
        kind=kind,
        durationMs=round(duration * 1000.0, 3),
        width=width,
        height=height,
        fps=round(fps, 4),
        hasAudio=audio is not None,
        codec=codec,
        sceneTags=[],
        order=order,
    )


# --------------------------------------------------------------------------- frames


def _even(n: int) -> int:
    return max(2, int(round(n / 2.0)) * 2)


def output_size(src_w: int, src_h: int, width: Optional[int] = None) -> Tuple[int, int]:
    if not width or width >= src_w:
        return src_w, src_h
    h = _even(src_h * width / float(src_w))
    return int(width), int(h)


def iter_frames(
    path: str | os.PathLike,
    start_ms: float = 0.0,
    end_ms: Optional[float] = None,
    fps: Optional[float] = None,
    width: Optional[int] = None,
    *,
    src_size: Optional[Tuple[int, int]] = None,
    size: Optional[Tuple[int, int]] = None,
) -> Iterator[np.ndarray]:
    """Yield BGR ``uint8`` frames (H, W, 3) decoded through an ffmpeg rawvideo pipe.

    ``fps`` re-samples the stream (``fps=`` filter); ``width`` down-scales keeping
    aspect (height rounded to even); ``size=(w, h)`` forces an exact output size
    (ignores aspect). ``src_size`` skips the probe when known.
    """
    path = str(path)
    if src_size is None:
        a = probe(path)
        src_size = (a.width, a.height)
    if size is not None:
        out_w, out_h = int(size[0]), int(size[1])
    else:
        out_w, out_h = output_size(src_size[0], src_size[1], width)
    if out_w <= 0 or out_h <= 0:
        return

    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start_ms and start_ms > 0:
        cmd += ["-ss", f"{start_ms / 1000.0:.6f}"]
    cmd += ["-i", path]
    if end_ms is not None and end_ms > start_ms:
        cmd += ["-t", f"{(end_ms - start_ms) / 1000.0:.6f}"]
    filters = []
    if fps:
        filters.append(f"fps={fps:.6f}")
    if (out_w, out_h) != tuple(src_size):
        filters.append(f"scale={out_w}:{out_h}:flags=area")
    if filters:
        cmd += ["-vf", ",".join(filters)]
    if not fps:
        # Emit every decoded frame exactly once. Without this, ffmpeg's rawvideo muxer
        # converts variable-frame-rate sources (common for AI generators: 24 fps content
        # on a 60 fps timebase) to constant r_frame_rate, duplicating frames, so frame
        # index / avg fps no longer maps to the right time.
        cmd += ["-fps_mode", "passthrough"]
    cmd += ["-an", "-sn", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    frame_bytes = out_w * out_h * 3
    proc = procs.popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        bufsize=frame_bytes * 4,
    )
    assert proc.stdout is not None
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(out_h, out_w, 3).copy()
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        procs._unregister(proc)


_SHOWINFO_TB = re.compile(r"config in time_base:\s*(\d+)/(\d+)")
_SHOWINFO_FRAME = re.compile(r"\]\s*n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:")


def format_start_time(path: str | os.PathLike) -> Fraction:
    """The container's ``start_time`` (what ``-ss`` is relative to), as an exact fraction."""
    try:
        st = probe_raw(str(path)).get("format", {}).get("start_time")
        return Fraction(st) if st not in (None, "N/A") else Fraction(0)
    except (ValueError, ZeroDivisionError, FFmpegError):
        return Fraction(0)


def iter_frames_timed(
    path: str | os.PathLike,
    start_ms: float = 0.0,
    *,
    src_size: Optional[Tuple[int, int]] = None,
    start_time: Optional[Fraction] = None,
    timeout: float = 60.0,
) -> Iterator[Tuple[Fraction, np.ndarray]]:
    """Yield ``(t, frame)`` for every decoded frame from ``start_ms`` on, where ``t`` is the frame's
    **real presentation time** in seconds (exact fraction, same origin as ``-ss``: the container's
    start time is subtracted) and ``frame`` a full-resolution BGR ``uint8`` array.

    Frames are emitted exactly once (``-fps_mode passthrough``), so variable-frame-rate sources
    keep their true timing. The timestamps come from a ``showinfo`` filter in the same ffmpeg
    process (``-copyts`` keeps the stream timestamps); its log lines are read from stderr on a
    thread and paired with the rawvideo frames in order."""
    import queue
    import threading

    path = str(path)
    if src_size is None:
        a = probe(path)
        src_size = (a.width, a.height)
    w, h = int(src_size[0]), int(src_size[1])
    if w <= 0 or h <= 0:
        return
    st = format_start_time(path) if start_time is None else start_time
    cmd = [find_ffmpeg(), "-hide_banner", "-nostats", "-loglevel", "info", "-nostdin", "-copyts"]
    if start_ms and start_ms > 0:
        cmd += ["-ss", f"{start_ms / 1000.0:.6f}"]
    cmd += ["-i", path, "-vf", "showinfo=checksum=0", "-fps_mode", "passthrough",
            "-an", "-sn", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    frame_bytes = w * h * 3
    proc = procs.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                       bufsize=frame_bytes * 2)
    assert proc.stdout is not None and proc.stderr is not None
    stamps: "queue.Queue[Optional[Tuple[int, int]]]" = queue.Queue()
    tb: List[Fraction] = []
    tail: List[str] = []

    def _read_stderr() -> None:
        try:
            for raw in iter(proc.stderr.readline, b""):
                line = raw.decode("utf-8", "replace")
                m = _SHOWINFO_FRAME.search(line)
                if m:
                    stamps.put((int(m.group(1)), int(m.group(2))))
                    continue
                m = _SHOWINFO_TB.search(line)
                if m and not tb:
                    tb.append(Fraction(int(m.group(1)), int(m.group(2))))
                    continue
                if "showinfo" not in line:
                    tail.append(line.rstrip())
                    del tail[:-20]
        except Exception:
            pass
        finally:
            stamps.put(None)

    reader = threading.Thread(target=_read_stderr, name="ffmpeg-showinfo", daemon=True)
    reader.start()
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            try:
                item = stamps.get(timeout=timeout)
            except queue.Empty:
                raise FFmpegError(f"{path}: no timestamp for a decoded frame (showinfo)")
            if item is None or not tb:
                raise FFmpegError(f"{path}: frame timestamps unavailable: {' | '.join(tail[-3:])}")
            t = Fraction(item[1]) * tb[0] - st
            yield t, np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()
    finally:
        # kill first: the stderr thread is blocked in readline() and only EOF releases it (closing a
        # buffered pipe another thread is reading would wait for that thread's lock)
        if proc.poll() is None:
            proc.kill()
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()
        reader.join(timeout=5.0)
        try:
            proc.stderr.close()
        except Exception:
            pass
        procs._unregister(proc)


def read_frame(path: str | os.PathLike, at_ms: float, width: Optional[int] = None,
               src_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """Decode a single frame at ``at_ms`` (or None when out of range)."""
    for f in iter_frames(path, start_ms=at_ms, end_ms=None, fps=None, width=width, src_size=src_size):
        return f
    return None


# --------------------------------------------------------------------------- audio


def decode_pcm(path: str | os.PathLike, sr: int = 22050, mono: bool = True) -> np.ndarray:
    """Decode the audio track to float32 PCM in [-1, 1]. Shape ``(n,)`` when mono else ``(n, 2)``."""
    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path), "-vn", "-sn",
           "-ac", "1" if mono else "2", "-ar", str(int(sr)), "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    proc = _run(cmd)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros((0,) if mono else (0, 2), dtype=np.float32)
    pcm = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    if not mono:
        pcm = pcm[: (len(pcm) // 2) * 2].reshape(-1, 2)
    return pcm


# --------------------------------------------------------------------------- segments


def extract_segment(
    path: str | os.PathLike,
    start_ms: float,
    end_ms: float,
    out_path: str | os.PathLike,
    *,
    reencode: bool = False,
) -> str:
    """Cut ``[start_ms, end_ms)`` to ``out_path``. Stream-copies by default (keyframe accurate);
    ``reencode=True`` gives frame-accurate cuts (libx264 / aac)."""
    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-ss", f"{start_ms / 1000.0:.6f}", "-i", str(path), "-t", f"{max(0.0, end_ms - start_ms) / 1000.0:.6f}"]
    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    cmd += [str(out_path)]
    proc = _run(cmd, capture_stderr=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg segment extraction failed: {proc.stderr.decode('utf-8', 'replace').strip()}")
    return str(out_path)


def ffmpeg_version() -> Optional[str]:
    try:
        proc = _run([find_ffmpeg(), "-version"])
    except FFmpegNotFound:
        return None
    first = proc.stdout.decode("utf-8", "replace").splitlines()[:1]
    return first[0] if first else None


# --------------------------------------------------------------------------- frame clock


class FrameClock:
    """Real presentation times of a video's frames, indexed like the frames :func:`iter_frames`
    yields (``-fps_mode passthrough``: every source frame exactly once, in presentation order).

    Variable-frame-rate sources (e.g. 24 fps content on a 60 fps timebase: frame gaps alternate
    33/50 ms) make ``index / avg_fps`` err by tens of milliseconds, enough for a cut to land a frame
    early. Times are in ms from the container start (the origin ``-ss`` and the preview use)."""

    def __init__(self, pts_ms: Sequence[float], fps: float):
        self.pts = np.asarray(sorted(float(t) for t in pts_ms), dtype=np.float64)
        self.fps = float(fps) if fps and fps > 0 else 0.0
        if self.fps <= 0 and len(self.pts) > 1:
            span = self.pts[-1] - self.pts[0]
            self.fps = (len(self.pts) - 1) * 1000.0 / span if span > 0 else 30.0
        if self.fps <= 0:
            self.fps = 30.0

    @classmethod
    def uniform(cls, fps: float) -> "FrameClock":
        """A constant-rate clock (``index / fps``): what callers get when no timestamps exist."""
        return cls([], fps)

    def __len__(self) -> int:
        return len(self.pts)

    def ms(self, i: float) -> float:
        """Presentation time (ms) of frame ``i``; extrapolated at the average rate outside the
        measured range (and linearly interpolated for fractional indices)."""
        n = len(self.pts)
        step = 1000.0 / self.fps
        if n == 0:
            return float(i) * step
        if i <= 0:
            return float(self.pts[0] + i * step)
        if i >= n - 1:
            return float(self.pts[-1] + (i - (n - 1)) * step)
        lo = int(i)
        f = i - lo
        return float(self.pts[lo] + f * (self.pts[lo + 1] - self.pts[lo]))

    def duration_ms(self, i: int) -> float:
        """How long frame ``i`` is on screen (gap to the next frame's time)."""
        return self.ms(i + 1) - self.ms(i)

    def nearest(self, t_ms: float) -> int:
        """The frame whose time is nearest to ``t_ms`` (what ffmpeg's ``fps`` filter picks)."""
        if len(self.pts) == 0:
            return max(0, int(round(t_ms * self.fps / 1000.0)))
        j = int(np.searchsorted(self.pts, t_ms))
        if j <= 0:
            return 0
        if j >= len(self.pts):
            return len(self.pts) - 1
        return j if self.pts[j] - t_ms < t_ms - self.pts[j - 1] else j - 1

    def seek_ms(self, i: int) -> float:
        """A ``-ss`` time from which ffmpeg's accurate seek starts on exactly frame ``i``: halfway
        between frame ``i - 1`` and frame ``i`` (0 for the first frame)."""
        if i <= 0:
            return 0.0
        return (self.ms(i - 1) + self.ms(i)) / 2.0

    def index_at(self, t_ms: float) -> int:
        """The frame showing at ``t_ms`` (last frame whose time is <= t)."""
        if len(self.pts) == 0:
            return max(0, int(np.floor(t_ms * self.fps / 1000.0 + 1e-6)))
        return max(0, int(np.searchsorted(self.pts, t_ms + 1e-6, side="right")) - 1)


_CLOCKS: dict = {}


def frame_clock(path: str | os.PathLike, fps: float = 0.0) -> FrameClock:
    """The :class:`FrameClock` of ``path``'s first video stream, read from the packet timestamps
    (no decode; sorted into presentation order). Cached per file (path, size, mtime). Falls back to
    a uniform ``index / fps`` clock when timestamps are unavailable."""
    p = str(path)
    try:
        stt = os.stat(p)
        key = (norm_path(p), stt.st_size, stt.st_mtime_ns)
    except OSError:
        return FrameClock.uniform(fps)
    hit = _CLOCKS.get(key)
    if hit is not None:
        return hit
    clock = FrameClock.uniform(fps)
    try:
        cmd = [find_ffprobe(), "-v", "error", "-select_streams", "v:0", "-print_format", "json",
               "-show_entries", "stream=time_base,start_time:packet=pts,dts,flags", p]
        proc = _run(cmd, capture_stderr=True, timeout=120.0)
        if proc.returncode == 0:
            info = json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
            streams = info.get("streams") or []
            tb = Fraction(streams[0].get("time_base", "1/1000")) if streams else Fraction(1, 1000)
            st = format_start_time(p)
            ticks = []
            for pk in info.get("packets") or []:
                if "D" in str(pk.get("flags", "")):  # discarded packets never become frames
                    continue
                v = pk.get("pts", pk.get("dts"))
                if v in (None, "N/A"):
                    ticks = []
                    break
                ticks.append(int(v))
            if ticks:
                clock = FrameClock([float((Fraction(t) * tb - st) * 1000) for t in ticks], fps)
    except (FFmpegError, FFmpegNotFound, ValueError, ZeroDivisionError, OSError, subprocess.SubprocessError):
        pass
    if len(_CLOCKS) > 256:
        _CLOCKS.clear()
    _CLOCKS[key] = clock
    return clock
