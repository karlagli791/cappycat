"""``interpolate``: optical-flow frame-rate multiplication of one source segment (for slow motion),
or (``--target-fps``, :func:`interpolate_to_fps`) conversion of a segment to an exact output frame
rate with real source timestamps, exact copies where a source frame coincides and flow-warped
frames at arbitrary fractional positions in between.

Decodes ``[in_ms, out_ms)`` at the source frame rate, writes a video at ``fps * factor`` that
contains every original frame followed by ``factor - 1`` synthesised in-betweens
(:func:`transitions.interpolate_with_flows`, RAFT on CUDA or DIS / Farneback on the CPU).

The in-betweens after the *last* frame of the segment are interpolated towards the first frame
after ``out_ms`` when the source has one (the frame is decoded but not written), otherwise the
last frame is held. The output therefore has exactly ``K * factor`` frames for ``K`` source
frames, i.e. the same duration as the segment, which is what the exporter's time remapping
expects.

Pairs that straddle a hard cut (HSV histogram correlation < 0.8 or mean |difference| > 35 on a
320 px proxy; within-shot pairs of the real clips measured >= 0.92 / <= 18, cuts 0.21-0.68 / >= 49)
are not morphed: the in-betweens hold the outgoing frame until t = 0.5, then the incoming one.

Frames are streamed: at most one flow batch of frames is held in memory, and flow is estimated
at <= 720p with a batch size derived from the VRAM budget.

The encoder writes ``<out>.<pid>-<id>.part.mp4`` and the file is moved to ``out_path`` only after
ffmpeg finished successfully, so a cancelled / killed job never leaves a truncated mp4 at the real
path (the encoder process is registered with :mod:`procs` for the stdin watchdog).
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from . import ffmpeg_util, fsutil, models, procs, transitions

Progress = Callable[[float, str], None]


def source_rates(path: str) -> Tuple[Fraction, Fraction]:
    """``(r_frame_rate, avg_frame_rate)`` of the first video stream as fractions."""
    info = ffmpeg_util.probe_raw(path)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"
                  and s.get("disposition", {}).get("attached_pic", 0) != 1), None)
    if video is None:
        raise ffmpeg_util.FFmpegError(f"{path}: no video stream")

    def frac(key: str) -> Optional[Fraction]:
        try:
            fr = Fraction(video.get(key) or "0")
            return fr.limit_denominator(1_000_000) if fr > 0 else None
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    r, avg = frac("r_frame_rate"), frac("avg_frame_rate")
    if r is None and avg is None:
        raise ffmpeg_util.FFmpegError(f"{path}: unknown frame rate")
    return (r or avg), (avg or r)  # type: ignore[return-value]


def source_rate(path: str) -> Tuple[Fraction, bool]:
    """The frame rate of the uniform grid we work on, and whether the source is variable frame
    rate (``r_frame_rate`` != ``avg_frame_rate`` by > 0.5 %, e.g. 24 fps content stored on a 1/60
    timebase by AI generators). For VFR sources the average rate is used and decoding resamples
    onto that grid (``fps=`` filter) so frame index <-> time stays exact."""
    r, avg = source_rates(path)
    vfr = abs(float(r) - float(avg)) > 0.005 * float(avg)
    return (avg if vfr else r), vfr  # CFR: r_frame_rate is exact (avg can carry duration rounding)


CUT_HIST = 0.8
CUT_MEAN_DIFF = 35.0


def is_cut(a: np.ndarray, b: np.ndarray) -> bool:
    import cv2

    w = 320
    h = max(8, int(round(a.shape[0] * w / float(a.shape[1]))))
    sa = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    sb = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    if float(np.abs(sa.astype(np.int16) - sb.astype(np.int16)).mean()) > CUT_MEAN_DIFF:
        return True
    return transitions.histogram_similarity(sa, sb) < CUT_HIST


def segment_frame_range(in_ms: float, out_ms: float, fps: float) -> tuple:
    """Source frame indices ``[i0, i1)`` whose timestamps fall in ``[in_ms, out_ms)`` (CFR),
    with a 0.1-frame tolerance for rounded millisecond inputs."""
    i0 = max(0, math.ceil(in_ms / 1000.0 * fps - 0.1))
    i1 = max(i0, math.ceil(out_ms / 1000.0 * fps - 0.1))
    return i0, i1


class _Encoder:
    def __init__(self, out_path: str, w: int, h: int, rate: Fraction):
        self.errf = tempfile.TemporaryFile()
        cmd = [ffmpeg_util.find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{rate.numerator}/{rate.denominator}",
               "-i", "-", "-an", "-sn"]
        if w % 2 or h % 2:
            cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(out_path)]
        self.proc = procs.popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.errf)
        self.count = 0

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.count += 1

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        rc = self.proc.wait()
        procs._unregister(self.proc)
        self.errf.seek(0)
        err = self.errf.read().decode("utf-8", "replace").strip()
        self.errf.close()
        if rc != 0:
            raise ffmpeg_util.FFmpegError(f"ffmpeg encode failed ({rc}): {err}")

    def kill(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        procs._unregister(self.proc)
        self.errf.close()


def interpolate_segment(src: str, in_ms: float, out_ms: float, factor: int, out_path: str,
                        progress: Optional[Progress] = None, budget_mb: Optional[float] = None) -> Dict[str, object]:
    """See module docstring. Returns ``{"frames", "sourceFrames", "fps", "backend", "flowScale", "batch"}``."""
    factor = int(factor)
    if factor < 1:
        raise ValueError("--factor must be >= 1")
    if out_ms <= in_ms:
        raise ValueError("--out-ms must be greater than --in-ms")
    progress = progress or (lambda pct, msg: None)
    asset = ffmpeg_util.probe(src)
    if asset.kind != "video" or asset.width <= 0:
        raise ffmpeg_util.FFmpegError(f"{src}: not a video (kind={asset.kind})")
    rate, vfr = source_rate(src)
    fps = float(rate)
    w, h = asset.width, asset.height
    i0, i1 = segment_frame_range(in_ms, out_ms, fps)
    k_expected = i1 - i0
    if k_expected <= 0:
        raise ValueError("segment contains no frames")
    start_ms = max(0.0, (i0 - 0.1) / fps * 1000.0)
    end_ms = start_ms + (k_expected + 1) / fps * 1000.0  # one extra frame (the next one) if it exists
    # CFR: every decoded frame once (passthrough); VFR: resample onto the average-rate grid
    frames = ffmpeg_util.iter_frames(src, start_ms=start_ms, end_ms=end_ms, fps=fps if vfr else None, src_size=(w, h))

    scale, batch = transitions.plan_flow(h, w, budget_mb)
    times = [j / float(factor) for j in range(1, factor)]
    out_rate = rate * factor
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_out = fsutil.part_path(out_path, ".mp4")  # published with os.replace once complete
    enc = _Encoder(str(tmp_out), w, h, out_rate)
    progress(0.0, f"{k_expected} frame(s) @ {fps:.3f} fps{' (VFR source, resampled)' if vfr else ''} -> x{factor} ({float(out_rate):.3f} fps), "
                  f"flow at {int(w * scale)}x{int(h * scale)}, batch {batch}")
    backend = "none"
    done = 0
    n_cuts = 0
    try:
        # chunk holds source frames done .. done+len-1; frame index K (just after the segment) is
        # decoded only as the interpolation target of the last segment frame
        chunk: List[np.ndarray] = []
        first = next(frames, None)
        if first is not None:
            chunk.append(first)
        while chunk and done < k_expected:
            while len(chunk) < batch + 1 and done + len(chunk) <= k_expected:
                f = next(frames, None)
                if f is None:
                    break
                chunk.append(f)
            n_pairs = min(len(chunk) - 1, k_expected - done)
            if n_pairs == 0:  # last available frame and no successor: hold it
                for _ in range(factor):
                    enc.write(chunk[0])
                done += 1
                break
            cuts = [is_cut(chunk[i], chunk[i + 1]) for i in range(n_pairs)] if factor > 1 else [False] * n_pairs
            smooth = [i for i in range(n_pairs) if not cuts[i]]
            flows: Dict[int, tuple] = {}
            if factor > 1 and smooth:
                fl, backend = transitions.pair_flows([chunk[i] for i in smooth], [chunk[i + 1] for i in smooth], scale, batch)
                flows = dict(zip(smooth, fl))
            for i in range(n_pairs):
                enc.write(chunk[i])
                if factor == 1:
                    continue
                if cuts[i]:
                    n_cuts += 1
                    for t in times:
                        enc.write(chunk[i] if t < 0.5 else chunk[i + 1])
                    continue
                f01, f10 = flows[i]
                for mid in transitions.interpolate_with_flows(chunk[i], chunk[i + 1], f01, f10, times):
                    enc.write(mid)
            done += n_pairs
            chunk = chunk[n_pairs:]
            progress(min(done / float(k_expected), 1.0) * 0.98, f"{done}/{k_expected} source frames")
        enc.close()
        if done > 0:
            os.replace(tmp_out, out_path)
    except BaseException:
        enc.kill()
        try:
            tmp_out.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        close = getattr(frames, "close", None)
        if close:
            close()
    k = done
    if k == 0:
        try:
            tmp_out.unlink(missing_ok=True)
        except OSError:
            pass
        raise ffmpeg_util.FFmpegError("no frames decoded for the requested segment")
    progress(1.0, f"wrote {enc.count} frame(s) at {float(out_rate):.3f} fps ({backend}"
                  f"{f', {n_cuts} cut(s) held' if n_cuts else ''})")
    return {"frames": enc.count, "sourceFrames": k, "fps": float(out_rate), "vfrSource": vfr, "backend": backend,
            "cutsHeld": n_cuts,
            "flowScale": round(scale, 4), "batch": batch}


# --------------------------------------------------------------------------- --target-fps

COINCIDE_S = Fraction(1, 2000)  # an output time within 0.5 ms of a source frame copies that frame
_NTSC = (Fraction(24000, 1001), Fraction(30000, 1001), Fraction(48000, 1001), Fraction(60000, 1001),
         Fraction(120000, 1001))


def parse_fps(value: object) -> Fraction:
    """``"60"`` -> 60, ``"29.97"`` -> 30000/1001, ``"30000/1001"`` -> 30000/1001, ``"23.976"`` ->
    24000/1001, ``"12.5"`` -> 25/2. NTSC rates given as decimals (within 0.01) map to their exact
    ``N*1000/1001`` form; any other decimal is taken literally (``limit_denominator(1001)``)."""
    s = str(value).strip()
    try:
        fr = Fraction(s)
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"invalid frame rate {value!r}")
    if fr <= 0 or fr > 1000:
        raise ValueError(f"frame rate out of range: {value!r}")
    if "/" not in s and fr.denominator != 1:
        for exact in _NTSC:
            if abs(float(fr) - float(exact)) < 0.01:
                return exact
        fr = fr.limit_denominator(1001)
    return fr


def fps_label(rate: Fraction) -> str:
    return str(rate.numerator) if rate.denominator == 1 else f"{rate.numerator}/{rate.denominator}"


def target_frame_count(in_ms: float, out_ms: float, rate: Fraction) -> int:
    """Output frames for ``[in_ms, out_ms)`` at ``rate``: ``round(duration * rate)`` (>= 1), so the
    output lasts as long as the segment (to the nearest output frame)."""
    n = (Fraction(out_ms) - Fraction(in_ms)) / 1000 * rate
    return max(1, int(math.floor(n + Fraction(1, 2))))


class TargetFrame:
    """One output frame of :func:`iter_target_fps_frames`. ``kind``: ``"copy"`` (a source frame,
    unchanged), ``"interp"`` (synthesised at ``alpha`` between two source frames), ``"cut"`` (a
    pair across a hard cut: the nearer frame is held), ``"hold"`` (before the first / after the
    last decoded source frame). ``src`` = the source time(s) it was built from."""

    __slots__ = ("index", "time", "frame", "kind", "alpha", "src")

    def __init__(self, index: int, time: Fraction, frame: np.ndarray, kind: str, alpha: float = 0.0,
                 src: Tuple[Fraction, ...] = ()):
        self.index, self.time, self.frame, self.kind, self.alpha, self.src = index, time, frame, kind, alpha, src


def iter_target_fps_frames(src: str, in_ms: float, out_ms: float, rate: Fraction, *, size: Tuple[int, int],
                           scale: float, batch: int, budget_mb: Optional[float] = None,
                           stats: Optional[Dict[str, object]] = None) -> Iterator[TargetFrame]:
    """Yield the :class:`TargetFrame` s of ``[in_ms, out_ms)`` at ``rate``, in order.

    Output frame ``k`` sits at source time ``T_k = in + k / rate``. Source frames are decoded once
    with their real presentation times (:func:`ffmpeg_util.iter_frames_timed`, exact for VFR). For
    each ``T_k`` the bracketing source frames ``t_i <= T_k < t_{i+1}`` are found; within 0.5 ms of
    ``t_i`` the frame is copied, otherwise the pair is flow-interpolated at ``alpha = (T_k - t_i) /
    (t_{i+1} - t_i)``, or, across a hard cut, the nearer frame is held. Frames are streamed in
    windows of ``batch + 1`` source frames whose pairs are interpolated in one batched call."""
    stats = stats if stats is not None else {}
    stats.update({"copied": 0, "interpolated": 0, "held": 0, "cutsHeld": 0, "backend": "none"})
    n_out = target_frame_count(in_ms, out_ms, rate)
    t_in = Fraction(in_ms) / 1000
    times = [t_in + Fraction(k) / rate for k in range(n_out)]
    t_last = times[-1]
    # start a little before `in` so the frame at / before T_0 is decoded (accurate seek drops earlier ones)
    avg = float(source_rates(src)[1])
    margin_ms = max(250.0, 3000.0 / max(avg, 1.0))
    source = ffmpeg_util.iter_frames_timed(src, max(0.0, in_ms - margin_ms), src_size=size)
    buf: List[Tuple[Fraction, np.ndarray]] = []  # consecutive source frames, increasing time
    exhausted = False
    cut_memo: Dict[Tuple[Fraction, Fraction], bool] = {}
    state: dict = {}

    def read_one() -> bool:
        nonlocal exhausted
        if exhausted:
            return False
        item = next(source, None)
        if item is None:
            exhausted = True
            return False
        if not buf or item[0] > buf[-1][0]:  # a non-increasing timestamp (broken stream) is dropped
            buf.append(item)
        return True

    def trim(t: Fraction) -> None:  # drop frames that can no longer bracket t (keep one t_i <= t)
        while len(buf) >= 2 and buf[1][0] <= t + COINCIDE_S:
            buf.pop(0)

    k = 0
    try:
        while k < n_out:
            trim(times[k])
            # fill the window: `batch` pairs, but not past the first frame after the last output time
            while len(buf) < batch + 1 and (not buf or buf[-1][0] <= t_last + COINCIDE_S):
                if not read_one():
                    break
                trim(times[k])
            if not buf:
                raise ffmpeg_util.FFmpegError("no frames decoded for the requested segment")
            # resolve every output time this window can answer: (k, kind, i, alpha)
            plan: List[Tuple[int, str, int, float]] = []
            kk = k
            while kk < n_out:
                t = times[kk]
                if t < buf[0][0] - COINCIDE_S:  # before the first decoded frame
                    plan.append((kk, "hold", 0, 0.0))
                    kk += 1
                    continue
                i = 0
                while i + 1 < len(buf) and buf[i + 1][0] <= t + COINCIDE_S:
                    i += 1
                if abs(t - buf[i][0]) <= COINCIDE_S:
                    plan.append((kk, "copy", i, 0.0))
                elif i + 1 < len(buf):
                    t0, t1 = buf[i][0], buf[i + 1][0]
                    plan.append((kk, "interp", i, float((t - t0) / (t1 - t0))))
                elif exhausted:  # past the end of the source
                    plan.append((kk, "hold", i, 0.0))
                else:
                    break  # needs the next window
                kk += 1
            if not plan:  # a one-frame window that is not the end: read on
                read_one()
                continue
            # pairs across a hard cut are held, the others synthesised in one batched call
            per_pair: Dict[int, List[float]] = {}
            for _, kind, i, alpha in plan:
                if kind != "interp":
                    continue
                key = (buf[i][0], buf[i + 1][0])
                if key not in cut_memo:
                    cut_memo[key] = is_cut(buf[i][1], buf[i + 1][1])
                if not cut_memo[key]:
                    per_pair.setdefault(i, []).append(alpha)
            jobs = sorted(per_pair.items())
            made: Dict[Tuple[int, float], np.ndarray] = {}
            if jobs:
                outs, backend = transitions.interpolate_jobs([f for _, f in buf], jobs, scale, batch, budget_mb, state)
                stats["backend"] = backend
                for (i, alphas), frames in zip(jobs, outs):
                    for a, fr in zip(alphas, frames):
                        made[(i, a)] = fr
            for kk_, kind, i, alpha in plan:
                t = times[kk_]
                if kind in ("copy", "hold"):
                    stats["copied" if kind == "copy" else "held"] += 1  # type: ignore[operator]
                    yield TargetFrame(kk_, t, buf[i][1], kind, 0.0, (buf[i][0],))
                elif (i, alpha) in made:
                    stats["interpolated"] += 1  # type: ignore[operator]
                    yield TargetFrame(kk_, t, made[(i, alpha)], "interp", alpha, (buf[i][0], buf[i + 1][0]))
                else:  # across a cut: the outgoing frame until the midpoint, then the incoming one
                    stats["cutsHeld"] += 1  # type: ignore[operator]
                    j = i if alpha < 0.5 else i + 1
                    yield TargetFrame(kk_, t, buf[j][1], "cut", alpha, (buf[j][0],))
            k = kk
    finally:
        close = getattr(source, "close", None)
        if close:
            close()


def interpolate_to_fps(src: str, in_ms: float, out_ms: float, target_fps: object, out_path: str,
                       progress: Optional[Progress] = None, budget_mb: Optional[float] = None) -> Dict[str, object]:
    """``interpolate --target-fps F``: write ``[in_ms, out_ms)`` as a video at exactly ``F`` fps
    (``round(duration * F)`` frames, encoded with the exact rational rate, e.g. 30000/1001). Output
    frame ``k`` is the source at ``in + k / F``: copied when a source frame sits within 0.5 ms of it,
    otherwise RAFT flow-warped from the two bracketing source frames at the fractional position
    (hard cuts held). Real frame timestamps are used, so VFR sources need no resampling. The mp4 is
    written to ``<out>.<pid>-<id>.part.mp4`` and moved over ``out_path`` once complete."""
    rate = parse_fps(target_fps)
    if out_ms <= in_ms:
        raise ValueError("--out-ms must be greater than --in-ms")
    progress = progress or (lambda pct, msg: None)
    asset = ffmpeg_util.probe(src)
    if asset.kind != "video" or asset.width <= 0:
        raise ffmpeg_util.FFmpegError(f"{src}: not a video (kind={asset.kind})")
    w, h = asset.width, asset.height
    _, vfr = source_rate(src)
    n_out = target_frame_count(in_ms, out_ms, rate)
    scale, batch = transitions.plan_flow(h, w, budget_mb)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_out = fsutil.part_path(out_path, ".mp4")  # published with os.replace once complete
    stats: Dict[str, object] = {}
    progress(0.0, f"{n_out} frame(s) at {fps_label(rate)} fps from a {asset.fps:.3f} fps source"
                  f"{' (VFR: real frame timestamps)' if vfr else ''}, flow at {int(w * scale)}x{int(h * scale)}, "
                  f"batch {batch}")
    models.peak_vram_mb(reset=True)
    t_start = time.perf_counter()
    enc = _Encoder(str(tmp_out), w, h, rate)
    gen = iter_target_fps_frames(src, in_ms, out_ms, rate, size=(w, h), scale=scale, batch=batch,
                                 budget_mb=budget_mb, stats=stats)
    sources: set = set()
    step = max(1, n_out // 50)
    try:
        for tf in gen:
            enc.write(tf.frame)
            sources.update(tf.src)
            if (tf.index + 1) % step == 0 and tf.index + 1 < n_out:
                progress((tf.index + 1) / float(n_out) * 0.98, f"{tf.index + 1}/{n_out} frames")
        enc.close()
        if enc.count != n_out:
            raise ffmpeg_util.FFmpegError(f"wrote {enc.count} of {n_out} frames")
        os.replace(tmp_out, out_path)
    except BaseException:
        enc.kill()
        try:
            tmp_out.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        gen.close()
    seconds = time.perf_counter() - t_start
    stats["seconds"] = round(seconds, 2)
    stats["framesPerSecond"] = round(enc.count / max(seconds, 1e-6), 2)
    stats["peakVramMb"] = models.peak_vram_mb()
    cuts = int(stats["cutsHeld"])  # type: ignore[call-overload]
    progress(1.0, f"wrote {enc.count} frame(s) at {fps_label(rate)} fps: {stats['copied']} copied, "
                  f"{stats['interpolated']} interpolated, {stats['held']} held ({stats['backend']}"
                  f"{f', {cuts} cut(s) held' if cuts else ''})")
    return {"frames": enc.count, "fps": float(rate), "rate": fps_label(rate), "sourceFrames": len(sources),
            "vfrSource": vfr, "flowScale": round(scale, 4), "batch": batch, **stats}
