"""``python -m cappycat_pipeline`` entry point.

stdout carries exactly one JSON object per line (``progress`` / ``log`` / ``result`` events,
see docs/CONTRACTS.md); everything else (logging, library chatter) is routed to stderr.

``--watch-stdin`` (anywhere on the command line) or ``CAPPYCAT_WATCH_STDIN=1`` starts the parent-death
watchdog (:mod:`procs`): when stdin reaches EOF, every child ffmpeg is killed and the process exits.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
import threading
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TextIO

import numpy as np

from . import __version__

log = logging.getLogger("cappycat.cli")

STAGES = ("ingest", "shots", "perception", "reframe", "audio", "transitions", "assemble", "export")
# fraction of a clip's progress span consumed by each per-clip stage (measured on the real clips:
# perception (YOLO-World + Grounded SAM 2 escalations) and duplicate tracking are ~80 % of the time)
_STAGE_SPAN = {"ingest": 0.01, "shots": 0.06, "perception": 0.55, "reframe": 0.30, "audio": 0.04, "transitions": 0.04}
_STAGE_START: Dict[str, float] = {}
_acc = 0.0
for _s in ("ingest", "shots", "perception", "reframe", "audio", "transitions"):
    _STAGE_START[_s] = _acc
    _acc += _STAGE_SPAN[_s]
SETUP_END = 0.02      # overall progress reserved for set-up (downloads, model loading)
CLIPS_END = 0.97      # ... then the clips; the rest is cross-clip transitions + assembly
CACHED_WEIGHT = 0.03  # a cached clip's share of the progress bar relative to an analysed one


class Emitter:
    """Writes contract events as JSON lines to the *real* stdout."""

    def __init__(self, stream: TextIO):
        self.stream = stream
        self._lock = threading.Lock()  # events may come from helper threads (download byte counters)

    def _emit(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=True, separators=(",", ":")) + "\n"
        with self._lock:
            self.stream.write(line)
            self.stream.flush()

    def progress(self, stage: str, clip: Optional[str], pct: float, message: str, **extra: Any) -> None:
        """``extra`` keys follow the contract fields (e.g. the byte counts of ``download`` events)."""
        self._emit({"event": "progress", "stage": stage, "clip": clip, "pct": round(min(max(pct, 0.0), 1.0), 4),
                    "message": message, **extra})

    def log(self, level: str, message: str) -> None:
        self._emit({"event": "log", "level": level, "message": message})

    def result(self, path: str) -> None:
        self._emit({"event": "result", "path": path})


_FD_REDIRECT: Optional[TextIO] = None


def _take_stdout() -> TextIO:
    """Reserve the real stdout for events; redirect ``print`` & friends to stderr.

    When stdout is the process's fd 1, the event stream gets a private duplicate of it and fd 1
    itself is pointed at stderr, so output written by native code (CUDA / onnxruntime / ffmpeg
    libraries, C-level ``printf``) can never corrupt the JSON-lines stream either."""
    global _FD_REDIRECT
    real = sys.stdout
    try:
        real.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        if real.fileno() == 1:
            real.flush()
            stream = os.fdopen(os.dup(1), "w", encoding="utf-8", errors="replace", buffering=1)
            sys.stderr.flush()
            os.dup2(2, 1)
            _FD_REDIRECT = stream
            real = stream
    except Exception:  # no usable fd (e.g. pytest capsys): Python-level redirect only
        pass
    sys.stdout = sys.stderr
    return real


def _restore_stdout(original: TextIO, events: TextIO) -> None:
    global _FD_REDIRECT
    if _FD_REDIRECT is not None and events is _FD_REDIRECT:
        try:
            events.flush()
            os.dup2(events.fileno(), 1)
            events.close()
        except Exception:
            pass
        _FD_REDIRECT = None
    sys.stdout = original


def _print_json(stream: TextIO, obj: Any) -> None:
    stream.write(json.dumps(obj, ensure_ascii=True) + "\n")
    stream.flush()


def quiet_libraries() -> None:
    """Drop known-harmless library chatter from stderr: torch.jit deprecation warnings (the
    YOLO-World prompt encoder is TorchScript) and OpenCV's WARN lines."""
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    for pat in (r".*torch\.jit.*", r".*TorchScript.*", r".*may break.*"):
        warnings.filterwarnings("ignore", message=pat)
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch(\..*)?")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.jit(\..*)?")
    cv2 = sys.modules.get("cv2")
    if cv2 is not None:
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass


# --------------------------------------------------------------------------- argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cappycat-pipeline", description="Cappycat analysis pipeline")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("-v", "--verbose", action="store_true", help="INFO logging on stderr")
    p.add_argument("--watch-stdin", action="store_true",
                   help="exit (killing child processes) when stdin reaches EOF (also CAPPYCAT_WATCH_STDIN=1)")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="full analysis of one or more clips -> AnalysisResult JSON")
    a.add_argument("clips", nargs="+", help="video file paths and/or folders (folders are expanded and ordered by filename)")
    a.add_argument("--keep-order", action="store_true",
                   help="use the clips in the order given instead of ordering them by their filenames")
    a.add_argument("--out", required=True, help="output JSON path")
    a.add_argument("--prompts", nargs="*", default=None, help="open-vocabulary prompts")
    a.add_argument("--detector", choices=["yolo_world", "grounded_sam2", "hybrid", "none"], default="hybrid")
    a.add_argument("--shot-detector", choices=["transnetv2", "pyscenedetect", "auto"], default="auto")
    a.add_argument("--shot-threshold", type=float, default=0.5, help="TransNetV2 probability threshold")
    a.add_argument("--target-duration-ms", type=float, default=None)
    a.add_argument("--target-lufs", type=float, default=-14.0)
    a.add_argument("--similarity-threshold", type=float, default=0.85)
    a.add_argument("--smoothing", choices=["ema", "savgol"], default="ema")
    a.add_argument("--smoothing-alpha", type=float, default=0.15)
    a.add_argument("--smoothing-window", type=int, default=15)
    a.add_argument("--no-normalize-audio", dest="normalize_audio", action="store_false")
    a.add_argument("--no-beats", dest="beats", action="store_false")
    a.add_argument("--sample-fps", type=float, default=4.0, help="perception frame sampling rate")
    a.add_argument("--escalation-fps", type=float, default=2.0,
                   help="Grounded SAM 2 sampling rate of an escalated shot (dense tracking fills in)")
    a.add_argument("--analysis-width", type=int, default=640, help="perception frame width")
    a.add_argument("--track-fps", type=float, default=8.0,
                   help="dense re-scan rate used to follow confirmed duplicates through their shot (0 = off)")
    a.add_argument("--no-hires", dest="hires", action="store_false",
                   help="do not re-run YOLO-World at 1280 px on sparse shots")
    a.add_argument("--no-cache", dest="use_cache", action="store_false",
                   help="neither read nor write the per-clip result cache")
    a.add_argument("--no-download", dest="download", action="store_false",
                   help="do not fetch missing models before the analysis")
    a.add_argument("--director", action="store_true", help="consult the local Ollama AI director")
    a.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    a.add_argument("--ollama-model", default="qwen2.5-vl")
    a.add_argument("--name", default=None, help="project name")
    a.add_argument("--characters", default=None,
                   help="main-cast manifest (default: <repo>/characters/characters.json when it exists)")
    a.add_argument("--no-characters", dest="use_characters", action="store_false",
                   help="do not identify main-cast characters by name")

    pr = sub.add_parser("probe", help="print the Asset JSON for a file")
    pr.add_argument("path")

    sh = sub.add_parser("shots", help="print detected shots")
    sh.add_argument("path")
    sh.add_argument("--shot-detector", choices=["transnetv2", "pyscenedetect", "auto"], default="auto")
    sh.add_argument("--shot-threshold", type=float, default=0.5)

    be = sub.add_parser("beats", help="print beats / tempo for a file")
    be.add_argument("path")

    od = sub.add_parser("order", help="order the media in a folder (or a list of files) by their filenames")
    od.add_argument("paths", nargs="+", help="a folder, or several files")
    od.add_argument("--recursive", action="store_true")
    od.add_argument("--json", action="store_true", help="print one JSON object instead of a table")

    dr = sub.add_parser("doctor", help="report available ffmpeg / runtimes / models / GPU (one JSON object)")
    dr.add_argument("--json", action="store_true", help="accepted for clarity; the report is always one JSON object")

    ip = sub.add_parser("interpolate", help="optical-flow frame interpolation of one segment (slow motion export)")
    ip.add_argument("src", help="source video")
    ip.add_argument("--in-ms", type=float, required=True, help="segment start (inclusive), source ms")
    ip.add_argument("--out-ms", type=float, required=True, help="segment end (exclusive), source ms")
    ipm = ip.add_mutually_exclusive_group(required=True)
    ipm.add_argument("--factor", type=int, help="frame-rate multiplier N (N-1 new frames per pair)")
    ipm.add_argument("--target-fps", default=None,
                     help="output exactly this frame rate (e.g. 60, 50, 40, 30, 29.97, 30000/1001): frame k = source "
                          "at in + k/F, flow-warped between the bracketing source frames (copied when one coincides)")
    ip.add_argument("--out", required=True, help="output .mp4 (libx264 crf 16, yuv420p, no audio)")

    ch = sub.add_parser("characters", help="main-cast reference bank: build / identify")
    chs = ch.add_subparsers(dest="characters_command", required=True)
    cb = chs.add_parser("build", help="(re)build the cached reference embeddings and print the calibration")
    cb.add_argument("--characters", default=None, help="manifest path")
    cb.add_argument("--if-stale", action="store_true", help="only rebuild when images / manifest changed")
    ci = chs.add_parser("identify", help="detect + identify the characters in an image or a video frame")
    ci.add_argument("path")
    ci.add_argument("--at-ms", type=float, default=0.0, help="video time to sample")
    ci.add_argument("--characters", default=None, help="manifest path")
    ci.add_argument("--detector", choices=["yolo_world", "grounded_sam2"], default="yolo_world")

    sp = sub.add_parser("separate", help="split media audio into vocals / background stems (Demucs v4, cached)")
    sp.add_argument("media", nargs="+", help="video / audio files")
    sp.add_argument("--model", choices=["htdemucs_ft", "htdemucs"], default="htdemucs_ft",
                    help="Demucs v4 model (htdemucs_ft: best vocals, 4 fine-tuned nets; htdemucs: ~4x faster)")
    sp.add_argument("--device", choices=["cuda", "cpu"], default=None, help="default: cuda when available")
    sp.add_argument("--force", action="store_true", help="re-separate even when cached stems exist")
    sp.add_argument("--json", action="store_true",
                    help="JSON-lines events on stdout (progress / log / result); otherwise a readable summary")

    dm = sub.add_parser("download-models", help="fetch every ML model into pipeline/models (idempotent)")
    dm.add_argument("--only", nargs="*", default=None, help="substring filter on model names")
    return p


# --------------------------------------------------------------------------- analyze


def _frame_seek_ms(path: str, frame: int, fps: float) -> float:
    """Timestamp that makes ffmpeg's accurate seek land on exactly ``frame`` (from the frames' real
    presentation times, so variable-frame-rate sources land on the right frame too)."""
    from . import ffmpeg_util

    return ffmpeg_util.frame_clock(str(path), fps if fps > 0 else 25.0).seek_ms(int(frame))


def _retime(clock, items) -> None:
    """Set ``timeMs`` of frame-indexed items (findings, reframe keyframes) from the real frame times."""
    for it in items or ():
        it.timeMs = round(clock.ms(int(it.frame)), 3)


def run_analyze(args: argparse.Namespace, em: Emitter) -> int:
    from . import transitions, models

    try:
        return _run_analyze(args, em)
    finally:
        transitions.release_raft()
        models.release()


def _cache_options(args: argparse.Namespace, prompts: Sequence[str]) -> Dict[str, Any]:
    """Every option that changes one clip's analysis (the per-clip cache key)."""
    return {"detector": args.detector, "prompts": list(prompts), "shotDetector": args.shot_detector,
            "shotThreshold": args.shot_threshold, "similarity": args.similarity_threshold, "smoothing": args.smoothing,
            "alpha": args.smoothing_alpha, "window": args.smoothing_window, "normalize": args.normalize_audio,
            "beats": args.beats, "targetLufs": args.target_lufs, "sampleFps": args.sample_fps,
            "escalationFps": getattr(args, "escalation_fps", 2.0), "analysisWidth": args.analysis_width,
            "trackFps": args.track_fps, "hires": getattr(args, "hires", True),
            "characters": bool(getattr(args, "use_characters", True)), "genericGate": not args.prompts}


def _set_hf_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    hub = sys.modules.get("huggingface_hub.constants")
    if hub is not None:
        try:
            hub.HF_HUB_OFFLINE = True  # type: ignore[attr-defined]
        except Exception:
            pass


class _Models:
    """Detector / embedder / bank, built on first use (a fully cached run never loads them)."""

    def __init__(self, args, em: Emitter, prompts: List[str], manifest_path: Optional[Path]):
        self.args, self.em, self.prompts, self.manifest_path = args, em, prompts, manifest_path
        self.ready = False
        self.detector = None
        self.embedder = None
        self.bank = None
        self.lite = True
        self.degraded: List[str] = []  # why results must not be cached (a model the run should have had is missing)
        self.cache = None

    def ensure(self, pct: float = SETUP_END) -> None:
        """Load everything once; ``pct`` = the overall progress to report meanwhile (never backwards)."""
        if self.ready:
            return
        from . import models, perception, shots as shots_mod

        args, em = self.args, self.em
        self.ready = True
        if self.manifest_path is not None:
            from . import characters as characters_mod

            try:
                em.progress("ingest", None, pct, f"loading main-cast references from {self.manifest_path.name}")
                self.bank = characters_mod.load_bank(self.manifest_path, log_fn=lambda m: em.log("info", m))
                em.log("info", f"main cast: {', '.join(f'{c.name} ({n} refs)' for c in self.bank.manifest.characters for k, n in self.bank.counts().items() if k == c.id)}; "
                               f"idThreshold={self.bank.id_threshold} idMargin={self.bank.id_margin}")
            except Exception as exc:
                em.log("warn", f"character references unavailable ({exc}); duplicates use the generic rule only")
                self.bank = None
                self.degraded.append("character bank")
            models.release()
        em.progress("ingest", None, pct, f"initialising detector '{args.detector}'")
        self.detector, warns = perception.build_detector(args.detector, self.prompts)
        for w in warns:
            em.log("warn", w)
        if warns:
            self.degraded.append("detector")
        self.lite = self.detector.name == "none"
        self.embedder, ewarn = perception.build_embedder(prefer_clip=not self.lite)
        if ewarn and not self.lite:
            em.log("warn", ewarn)
            self.degraded.append("embedder")
        if self.bank is not None and (self.lite or self.embedder.name != "open_clip"):
            em.log("warn", "main-cast identification needs the OpenCLIP embedder; skipped")
            self.bank = None
        self.cache = perception.DetectionCache()
        tn = shots_mod.transnet_backend()
        vram = models.vram_info()
        tn_msg = f" (transnetv2 via {tn})" if tn else " (transnetv2 model not found -> pyscenedetect)"
        dev_msg = f"{models.device()} ({vram['device']}, {vram['freeMb']} MB free)" if vram else models.device()
        em.log("info", f"detector={self.detector.name} embedder={self.embedder.name} shot_detector={args.shot_detector}{tn_msg}; "
                       f"device={dev_msg}")

    def close(self) -> None:
        from . import models, perception

        if self.detector is not None:
            perception.close_detector(self.detector)
        if self.embedder is not None and hasattr(self.embedder, "close"):
            self.embedder.close()
        self.detector = self.embedder = None
        models.release()


def _run_analyze(args: argparse.Namespace, em: Emitter) -> int:
    import time as _time

    from . import assemble, ffmpeg_util, models, perception, resultcache
    from .schema import AnalysisResult, TransitionAnalysis, clip_analysis_from_json, new_id, to_json

    t_start = _time.perf_counter()
    models.peak_vram_mb(reset=True)

    try:
        ffmpeg_util.find_ffmpeg()
        ffmpeg_util.find_ffprobe()
    except ffmpeg_util.FFmpegNotFound as exc:
        em.log("error", str(exc))
        return 1

    # ---- main cast manifest (the bank itself is loaded lazily; its hash keys the cache)
    manifest = None
    manifest_path: Optional[Path] = None
    bank_hash: Optional[str] = None
    characters_stale = False
    if getattr(args, "use_characters", True):
        from . import characters as characters_mod

        mp = Path(args.characters) if args.characters else characters_mod.default_manifest_path()
        if mp.is_file():
            try:
                manifest = characters_mod.load_manifest(mp)
                bank_hash = manifest.fingerprint()
                manifest_path = mp
                characters_stale = not characters_mod.cache_state(manifest).get("cacheFresh", False)
            except Exception as exc:
                em.log("warn", f"characters manifest unreadable ({exc}); duplicates use the generic rule only")
        elif args.characters:
            em.log("warn", f"characters manifest not found: {mp}")
    if args.prompts:
        prompts: List[str] = list(args.prompts)
    elif manifest is not None:
        prompts = manifest.detection_prompts()
    else:
        prompts = list(perception.DEFAULT_PROMPTS)

    paths, order_reasons, order_warnings = resolve_clip_order(args.clips, keep_order=args.keep_order)
    for w in order_warnings:
        em.log("warn", f"clip order: {w}")
    if not paths:
        em.log("error", "no video clips found in the given paths")
        return 1
    em.log("info", "clip order: " + " | ".join(f"{i + 1}. {p.name} ({order_reasons.get(str(p), 'as given')})"
                                             for i, p in enumerate(paths)))
    n = len(paths)

    # ---- per-clip result cache lookup
    use_cache = bool(getattr(args, "use_cache", True)) and not args.director
    opts = _cache_options(args, prompts)
    env_fp = resultcache.environment_fingerprint() if use_cache else None
    keys: List[Optional[str]] = [resultcache.clip_key(p, opts, bank_hash, env_fp) if use_cache else None for p in paths]
    cached: List[Optional[dict]] = [resultcache.load(k) for k in keys]
    n_cached = sum(1 for c in cached if c)
    if use_cache:
        em.log("info", f"analysis cache: {n_cached}/{n} clip(s) cached ({resultcache.cache_root()})")

    # ---- first-run downloads (before the clip loop), then offline mode when everything is local
    if n_cached < n or characters_stale:
        from . import downloads

        need = downloads.models_needed_for_analysis(args.detector, args.shot_detector, prompts, characters_stale)
        if getattr(args, "download", True):
            missing, results = downloads.ensure_models(
                need, lambda name, pct, msg: em.progress("download", name, SETUP_END * 0.5 * min(max(pct, 0.0), 1.0), msg))
            for name, err in results:
                if err:
                    em.log("warn", f"model download failed: {name}: {err}")
                else:
                    em.log("info", f"downloaded {name}")
            if not missing:
                _set_hf_offline()
        elif all(downloads._present(x) for x in need):
            _set_hf_offline()
    else:
        _set_hf_offline()

    # ---- progress weights: clip duration (probe uncached clips up front), cached clips ~free
    assets: Dict[int, Any] = {}
    weights: List[float] = []
    for ci, path in enumerate(paths):
        if cached[ci]:
            dur = float(cached[ci]["clip"]["asset"].get("durationMs") or 1000.0)
            weights.append(max(1000.0, dur) * CACHED_WEIGHT)
            continue
        try:
            assets[ci] = ffmpeg_util.probe(path, order=ci)
            weights.append(max(1000.0, assets[ci].durationMs))
        except Exception as exc:
            assets[ci] = exc
            weights.append(1000.0 * CACHED_WEIGHT)
    total_w = sum(weights) or 1.0
    starts = np.concatenate([[0.0], np.cumsum(weights)[:-1]]) / total_w

    mdl = _Models(args, em, prompts, manifest_path)
    analyses = []
    beat_details: Dict[str, list] = {}
    clip_keys: Dict[int, Optional[str]] = {}
    try:
        for ci, path in enumerate(paths):
            base = SETUP_END + (CLIPS_END - SETUP_END) * float(starts[ci])
            span = (CLIPS_END - SETUP_END) * weights[ci] / total_w
            clip_name = path.name

            def prog(stage: str, frac: float, msg: str, base=base, span=span, clip_name=clip_name) -> None:
                em.progress(stage, clip_name, base + span * (_STAGE_START[stage] + _STAGE_SPAN[stage] * min(max(frac, 0.0), 1.0)), msg)

            if cached[ci]:
                try:
                    ca = clip_analysis_from_json(cached[ci]["clip"])
                    ca.asset.id = new_id("ast")
                    ca.asset.order = ci
                    ca.asset.orderReason = order_reasons.get(str(path))
                    markers = cached[ci].get("beatMarkers")
                    if markers is not None:
                        from .schema import BeatMarker

                        beat_details[ca.asset.path] = [BeatMarker(m["timeMs"], m["strength"], m["kind"]) for m in markers]
                    for stage in _STAGE_SPAN:
                        prog(stage, 1.0, "cached")
                    em.log("info", f"{clip_name}: cached analysis reused ({len(ca.shots)} shot(s), "
                                   f"{len(ca.duplicates)} duplicate(s), {len(ca.reframe)} reframe(s)); key {keys[ci][:12]}")
                    analyses.append(ca)
                    clip_keys[len(analyses) - 1] = keys[ci]
                    continue
                except Exception as exc:  # a corrupt entry is just a miss
                    em.log("warn", f"{clip_name}: cached analysis unusable ({exc}); analysing again")
            mdl.ensure(base)
            asset = assets.get(ci)
            ca, markers, ok = _analyze_clip(args, em, mdl, path, ci, asset, order_reasons, prompts, prog)
            if ca is None:
                continue
            if markers is not None:
                beat_details[ca.asset.path] = markers
            analyses.append(ca)
            clip_keys[len(analyses) - 1] = keys[ci]
            if use_cache and ok and not mdl.degraded:
                resultcache.save(keys[ci], {"clip": to_json(ca), "beatMarkers": to_json(markers) if markers is not None else None,
                                            "path": ffmpeg_util.norm_path(path)})
    finally:
        mdl.close()
    peak = models.peak_vram_mb()
    peak_msg = f"; peak VRAM {peak:.0f} MB (torch allocator)" if peak is not None else ""
    cache_msg = f"; detection cache {mdl.cache.hits} hit(s)" if mdl.cache is not None and mdl.cache.hits else ""
    em.log("info", f"analysis took {_time.perf_counter() - t_start:.1f}s ({n_cached} cached){peak_msg}{cache_msg}")
    if not analyses:
        em.log("error", "no clips could be analysed")
        return 1

    # ---- cross-clip transitions (clip i-1's last frame -> clip i's first frame), cached per pair
    em.progress("transitions", None, CLIPS_END, "scoring the cuts between clips")
    for i in range(1, len(analyses)):
        prev_ca, ca = analyses[i - 1], analyses[i]
        if not prev_ca.shots or not ca.shots:
            continue
        ka, kb = clip_keys.get(i - 1), clip_keys.get(i)
        pk = resultcache.pair_key(ka, kb) if (use_cache and ka and kb) else None
        hit = resultcache.load(pk, "pairs") if pk else None
        try:
            if hit:
                mag, smooth = float(hit["flowMagnitude"]), float(hit["smoothness"])
            else:
                from . import transitions

                a = ffmpeg_util.read_frame(prev_ca.path, _frame_seek_ms(prev_ca.path, prev_ca.shots[-1].endFrame, prev_ca.asset.fps),
                                           width=320, src_size=(prev_ca.asset.width, prev_ca.asset.height))
                b = ffmpeg_util.read_frame(ca.path, _frame_seek_ms(ca.path, ca.shots[0].startFrame, ca.asset.fps), width=320,
                                           src_size=(ca.asset.width, ca.asset.height))
                if a is None or b is None:
                    continue
                mag, smooth = transitions.score_transition(a, b, prev_ca.asset.width)
                if pk:
                    resultcache.save(pk, {"flowMagnitude": mag, "smoothness": smooth}, "pairs")
            from . import transitions as tr_mod

            last_idx = prev_ca.shots[-1].index
            prev_s = [t.smoothness for t in prev_ca.transitions]
            # cross-clip transition: toShot = last index + 1 means "first shot of the next clip"
            prev_ca.transitions.append(TransitionAnalysis(fromShot=last_idx, toShot=last_idx + 1, flowMagnitude=mag,
                                                          smoothness=smooth, suggestion=tr_mod.suggest(smooth, prev_s)))
        except Exception as exc:
            em.log("warn", f"{prev_ca.asset.name} -> {ca.asset.name}: transition scoring failed: {exc}")

    # ---- assemble
    em.progress("assemble", None, 1.0 - 0.02, "assembling timeline")
    ref = next((a.asset for a in assemble.order_analyses(analyses) if a.asset.kind == "video"), analyses[0].asset)
    tl_fps = ref.fps if ref.fps > 0 else 24.0
    project = assemble.assemble_timeline(analyses, args.target_duration_ms, tl_fps, ref.width, ref.height, args.name, beat_details)
    result = AnalysisResult(
        generatedAt=_dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        clips=analyses,
        timeline=project,
    )
    out = Path(args.out)
    from . import fsutil

    # atomic: a cancelled / killed run never leaves a truncated JSON at --out
    fsutil.write_json_atomic(out, to_json(result), indent=None, separators=(",", ":"))
    dur = assemble.timeline_duration_ms(project)
    em.progress("assemble", None, 1.0, f"timeline: {sum(len(t.clips) for t in project.tracks if t.kind == 'video')} clips, {dur / 1000:.1f}s")
    em.result(ffmpeg_util.norm_path(out))
    return 0


def _analyze_clip(args, em: Emitter, mdl: _Models, path: Path, ci: int, asset, order_reasons, prompts,
                  prog: Callable[[str, float, str], None]):
    """One clip through ingest -> shots -> perception -> reframe -> audio -> transitions. Returns
    ``(ClipAnalysis | None, beat markers | None, ok)``; ``ok`` is False when a stage failed (the
    result is then not cached)."""
    from . import audio as audio_mod, ffmpeg_util, models, perception, reframe, shots as shots_mod, transitions
    from .schema import AudioAnalysis, ClipAnalysis, Shot, ShotReframe, TransitionAnalysis

    clip_name = path.name
    ok = True
    detector, embedder, bank, lite = mdl.detector, mdl.embedder, mdl.bank, mdl.lite
    # ---- ingest
    prog("ingest", 0.0, f"probing {clip_name}")
    try:
        if asset is None or isinstance(asset, Exception):
            if isinstance(asset, Exception):
                raise asset
            asset = ffmpeg_util.probe(path, order=ci)
        asset.orderReason = order_reasons.get(str(path))
    except Exception as exc:
        em.log("error", f"{clip_name}: probe failed: {exc}")
        return None, None, False
    if asset.kind not in ("video", "image") or asset.width <= 0:
        em.log("error", f"{clip_name}: not a video file (kind={asset.kind})")
        return None, None, False
    ca = ClipAnalysis(path=asset.path, asset=asset)
    fps = asset.fps if asset.fps > 0 else 25.0
    clock = ffmpeg_util.frame_clock(str(path), fps)
    src_size = (asset.width, asset.height)
    prog("ingest", 1.0, f"{asset.width}x{asset.height} @ {asset.fps} fps, {asset.durationMs / 1000:.1f}s, audio={asset.hasAudio}")

    # ---- shots
    prog("shots", 0.0, "detecting shot boundaries")
    try:
        ca.shots = shots_mod.detect_shots(str(path), args.shot_detector, args.shot_threshold, asset=asset)
    except Exception as exc:
        em.log("error", f"{clip_name}: shot detection failed: {exc}")
        ok = False
        n_frames = max(1, int(round(asset.durationMs / 1000.0 * fps)))
        ca.shots = [Shot(0, 0, n_frames - 1, 0.0, round((n_frames - 1) / fps * 1000.0, 3), 0.0, "pyscenedetect")]
    prog("shots", 1.0, f"{len(ca.shots)} shot(s) via {ca.shots[0].method}")

    # shot weights (by length) for per-frame progress inside perception / reframe
    lens = [max(1, s.endFrame - s.startFrame + 1) for s in ca.shots]
    tot = float(sum(lens))
    shot_start = np.concatenate([[0.0], np.cumsum(lens)[:-1]]) / tot

    # ---- perception
    dense_by_shot: Dict[int, list] = {}
    path_by_shot: Dict[int, str] = {}
    clip_cast: List[str] = []
    if lite:
        prog("perception", 1.0, "skipped (lite mode: no detector available)")
    else:
        scale = min(1.0, args.analysis_width / float(asset.width))
        esc_fps = getattr(args, "escalation_fps", perception.ESCALATION_FPS) or args.sample_fps
        for si, shot in enumerate(ca.shots):
            s0, sw = float(shot_start[si]), lens[si] / tot

            def sprog(frac: float, msg: str, s0=s0, sw=sw, si=si) -> None:
                prog("perception", s0 + sw * frac, f"shot {si + 1}/{len(ca.shots)}: {msg}")

            sprog(0.0, f"detecting '{', '.join(prompts)}'")
            try:
                def frames_factory(shot=shot):
                    return perception.sample_shot_frames(str(path), shot, fps, src_size, args.sample_fps, args.analysis_width)

                def esc_factory(shot=shot):
                    return perception.sample_shot_frames(str(path), shot, fps, src_size, esc_fps, args.analysis_width)

                def hires_factory(shot=shot):
                    return perception.sample_shot_frames(str(path), shot, fps, src_size, args.sample_fps, asset.width)

                res = perception.analyze_shot(frames_factory, shot, detector, embedder, args.similarity_threshold,
                                              prompts, fps, scale, args.sample_fps, keep_all=True, bank=bank,
                                              character_like_gate=not args.prompts, escalation_frames_factory=esc_factory,
                                              escalation_fps=esc_fps,
                                              hires_frames_factory=hires_factory if getattr(args, "hires", True) else None,
                                              hires_scale=1.0, cache=mdl.cache, progress=sprog)
                if res.warning:
                    em.log("warn", f"{clip_name} {res.warning}")
                    mdl.degraded.append("escalation")
                dense = res.findings
                _retime(clock, dense)
                st = res.stats
                path_by_shot[shot.index] = res.path
                if bank is not None:
                    shot.cast = res.cast
                    for cid in shot.cast:
                        if cid not in clip_cast:
                            clip_cast.append(cid)
                dd = perception.dedupe_findings(dense)
                named = [f"{f.characterName} x2 @ {f.timeMs / 1000:.2f}s" for f in dd if f.character]
                cast_msg = (f"cast [{', '.join(bank.manifest.name_of(c) or c for c in shot.cast)}]"
                            + (f" (track-level: {', '.join(bank.manifest.name_of(c) or c for c in st.loose_cast())})"
                               if st.loose_cast() else "") + "; " if bank is not None else "")
                em.log("info", f"{clip_name} shot {shot.index}: path={res.path}{' +hires' if res.hires else ''} ({res.reason}); "
                               f"{cast_msg}{st.frames} frame(s), {st.detections} detection(s), {st.masks} mask(s), "
                               f"max same-label IoU {st.max_same_label_iou:.2f} (ambiguous {st.max_ambiguous_iou:.2f}), "
                               f"max similarity {st.max_sim:.3f}, {len(dd)} duplicate(s){': ' + '; '.join(named) if named else ''}"
                               + (f"; {st.dropped_single_frame} single-frame candidate(s) dropped" if st.dropped_single_frame else ""))
                if dense:
                    dense_by_shot[shot.index] = dense
                    ca.duplicates.extend(dd)
            except Exception as exc:
                ok = False
                em.log("error", f"{clip_name} shot {si}: perception failed: {exc}")
                log.debug(traceback.format_exc())
                models.release()
        models.release()  # drop cached activation blocks before the next stage
        if bank is not None:
            order = [c.id for c in bank.manifest.characters]
            ca.asset.sceneTags = [bank.manifest.name_of(c) or c for c in sorted(clip_cast, key=order.index)]
        prog("perception", 1.0, f"{len(ca.duplicates)} duplicate collision(s)")

    # ---- optional AI director
    director_tracks: Dict[int, Any] = {}
    if args.director:
        from . import director as director_mod

        if not director_mod.ollama_reachable(args.ollama_url):
            em.log("warn", f"AI director skipped: Ollama not reachable at {args.ollama_url}")
        else:
            for shot in ca.shots:
                try:
                    mid = (shot.startFrame + shot.endFrame) // 2
                    frame = ffmpeg_util.read_frame(str(path), _frame_seek_ms(str(path), mid, fps), src_size=src_size)
                    if frame is None:
                        continue
                    dets = [] if lite else detector.detect(frame, prompts)
                    ddets = [director_mod.DirectorDetection(f"obj_{i + 1:02d}", d.label, [round(v, 1) for v in d.bbox], round(d.score, 3))
                             for i, d in enumerate(dets)]
                    resp = director_mod.ask_director(asset.width, asset.height, ddets, args.ollama_url, args.ollama_model, frame)
                    if resp is None:
                        continue
                    em.log("info", f"{clip_name} shot {shot.index}: director -> {resp.recommended_action}: {resp.cinematic_reasoning}")
                    if resp.has_duplicate and resp.crop_bounding_box and shot.index not in dense_by_shot:
                        director_tracks[shot.index] = reframe.static_track(asset.width, asset.height, resp.crop_bounding_box, shot, fps,
                                                                           f"AI director: {resp.cinematic_reasoning}")
                except Exception as exc:
                    em.log("warn", f"{clip_name} shot {shot.index}: director failed: {exc}")

    # ---- reframe (dense duplicate tracking + camera planning), weighted by tracked shot length
    prog("reframe", 0.0, "solving crops")
    tracked = [s for s in ca.shots if s.index in dense_by_shot]
    t_tot = float(sum(max(1, s.endFrame - s.startFrame + 1) for s in tracked)) or 1.0
    t_acc = 0.0
    for shot in ca.shots:
        try:
            track = None
            if shot.index in dense_by_shot:
                t0 = t_acc / t_tot
                tw = max(1, shot.endFrame - shot.startFrame + 1) / t_tot
                t_acc += max(1, shot.endFrame - shot.startFrame + 1)

                def rprog(frac: float, msg: str, t0=t0, tw=tw, shot=shot) -> None:
                    prog("reframe", t0 + tw * frac, f"shot {shot.index + 1}: {msg}")

                if args.track_fps and args.track_fps > 0 and not lite:
                    try:
                        track = _tracked_reframe(args, em, clip_name, str(path), shot, dense_by_shot[shot.index],
                                                 path_by_shot.get(shot.index, ""), detector, embedder, fps, src_size,
                                                 asset.width, asset.height, prompts, bank, cache=mdl.cache, progress=rprog)
                    except Exception as exc:
                        ok = False
                        em.log("warn", f"{clip_name} shot {shot.index}: duplicate tracking failed ({exc}); "
                                       "using the sampled-frame reframe")
                        log.debug(traceback.format_exc())
                        models.release()
                        track = None
                if track is None:
                    track = reframe.build_reframe_track(shot, dense_by_shot[shot.index], asset.width, asset.height,
                                                        fps, args.smoothing, args.smoothing_alpha, args.smoothing_window)
            elif shot.index in director_tracks:
                track = director_tracks[shot.index]
            if track is not None:
                _retime(clock, track.keyframes)
                ca.reframe.append(ShotReframe(shotIndex=shot.index, track=track))
        except Exception as exc:
            ok = False
            em.log("error", f"{clip_name} shot {shot.index}: reframe failed: {exc}")
    prog("reframe", 1.0, f"{len(ca.reframe)} reframe track(s)")

    # ---- audio
    markers = None
    prog("audio", 0.0, "measuring loudness")
    if asset.hasAudio:
        try:
            loud = audio_mod.measure_loudness(str(path), args.target_lufs)
            if loud is None:
                em.log("warn", f"{clip_name}: audio stream is silent / unmeasurable")
            else:
                integrated, tp, _lra = loud
                gain = audio_mod.recommended_gain_db(integrated, args.target_lufs, true_peak_db=tp) if args.normalize_audio else 0.0
                ca.audio = AudioAnalysis(integratedLufs=round(integrated, 2), truePeakDb=round(tp, 2),
                                         recommendedGainDb=round(gain, 2), beats=[], tempoBpm=None)
                if args.beats:
                    prog("audio", 0.5, "detecting beats")
                    pcm = ffmpeg_util.decode_pcm(str(path), sr=22050, mono=True)
                    br = audio_mod.analyze_beats(pcm, 22050)
                    ca.audio.beats = br.beats_ms
                    ca.audio.tempoBpm = br.tempo_bpm
                    ca.audio.beatConfidence = br.confidence
                    markers = audio_mod.beat_markers(br.beats_ms, br.strengths)
                    em.log("info", f"{clip_name}: beats {len(br.beats_ms)} (confidence {br.confidence:.2f}; "
                                   f"periodicity {br.periodicity:.2f}, numpy {br.numpy_bpm} / librosa {br.librosa_bpm} BPM: {br.reason})")
        except Exception as exc:
            ok = False
            em.log("error", f"{clip_name}: audio analysis failed: {exc}")
    msg = "no audio stream" if not asset.hasAudio else (
        f"{ca.audio.integratedLufs} LUFS, gain {ca.audio.recommendedGainDb:+.1f} dB (true peak {ca.audio.truePeakDb:+.1f} dBTP), "
        f"{len(ca.audio.beats)} beats{f' @ {ca.audio.tempoBpm} BPM' if ca.audio.tempoBpm else ''}" if ca.audio else "unmeasurable")
    prog("audio", 1.0, msg)

    # ---- transitions (within the clip; cross-clip ones are scored after all clips)
    prog("transitions", 0.0, "scoring cuts")
    try:
        frames_cache: Dict[int, Any] = {}

        def frame_at(fr: int):
            if fr not in frames_cache:
                frames_cache[fr] = ffmpeg_util.read_frame(str(path), _frame_seek_ms(str(path), fr, fps), width=320, src_size=src_size)
            return frames_cache[fr]

        for i in range(len(ca.shots) - 1):
            a = frame_at(ca.shots[i].endFrame)
            b = frame_at(ca.shots[i + 1].startFrame)
            if a is None or b is None:
                continue
            mag, smooth = transitions.score_transition(a, b, asset.width)
            ca.transitions.append(TransitionAnalysis(fromShot=i, toShot=i + 1, flowMagnitude=mag, smoothness=smooth,
                                                     suggestion=transitions.suggest(smooth)))
            prog("transitions", (i + 1) / max(1, len(ca.shots) - 1), f"cut {i}->{i + 1}: {transitions.suggest(smooth)}")
        # suggestions relative to this clip's own cuts (hard cuts are the norm)
        clip_s = [t.smoothness for t in ca.transitions]
        for t in ca.transitions:
            t.suggestion = transitions.suggest(t.smoothness, clip_s)
    except Exception as exc:
        ok = False
        em.log("error", f"{clip_name}: transition scoring failed: {exc}")
    prog("transitions", 1.0, f"{len(ca.transitions)} transition(s)")
    return ca, markers, ok


# --------------------------------------------------------------------------- small commands


def _tracked_reframe(args, em, clip_name, path, shot, findings, shot_path, detector, embedder, fps, src_size,
                     frame_w, frame_h, prompts, bank=None, cache=None, progress=None):
    """Follow every confirmed duplicate of ``shot`` densely and build the per-frame reframe."""
    import time as _time

    from . import dupetrack, models, reframe

    t0 = _time.perf_counter()
    say = progress or (lambda f, m: None)
    pairs = dupetrack.track_shot(path, shot, findings, detector, embedder, prompts, fps, src_size, shot_path, bank,
                                 args.track_fps, args.analysis_width, cache=cache,
                                 progress=lambda f, m: say(0.85 * f, m))
    for pt in pairs:
        em.log("info", f"{clip_name} shot {shot.index}: tracked duplicate {pt.label} at {args.track_fps:g} fps: {pt.stats}")
    models.release()
    say(0.9, "planning the camera path")
    track = reframe.build_tracked_reframe(shot, pairs, frame_w, frame_h, fps, args.smoothing, args.smoothing_alpha,
                                          args.smoothing_window)
    say(1.0, "camera path planned")
    log.info("tracked reframe %s shot %d took %.1fs", clip_name, shot.index, _time.perf_counter() - t0)
    return track


def resolve_clip_order(inputs: Sequence[str], keep_order: bool = False):
    """Expand folders and order clips by filename. Returns (paths, {path: reason}, warnings)."""
    from . import ordering

    files: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            files.extend(ordering.scan_folder(p, kinds=("video",)))
        else:
            files.append(p)
    # de-duplicate while keeping the first occurrence
    seen: set = set()
    uniq = []
    for f in files:
        k = str(f.resolve()) if f.exists() else str(f)
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    if keep_order:
        return uniq, {str(f): "order given on the command line" for f in uniq}, []
    infos, warnings_ = ordering.order_paths(uniq)
    return [Path(i.path) for i in infos], {i.path: i.reason for i in infos}, warnings_


def run_order(args: argparse.Namespace, out: TextIO) -> int:
    from . import ordering

    targets = [Path(p) for p in args.paths]
    if len(targets) == 1 and targets[0].is_dir():
        folder = targets[0]
        infos, warnings_ = ordering.order_folder(folder, recursive=args.recursive)
    else:
        folder = targets[0].parent if targets else Path(".")
        infos, warnings_ = ordering.order_paths(targets)
    if args.json:
        _print_json(out, {"folder": str(folder), "files": [i.to_json() for i in infos], "warnings": warnings_})
    else:
        for i in infos:
            out.write(f"{i.order + 1:>3}. {i.name}    <- {i.reason}\n")
        for w in warnings_:
            out.write(f"warning: {w}\n")
        out.flush()
    return 0


def run_probe(args: argparse.Namespace, out: TextIO) -> int:
    from . import ffmpeg_util
    from .schema import to_json

    _print_json(out, to_json(ffmpeg_util.probe(args.path)))
    return 0


def run_shots(args: argparse.Namespace, out: TextIO) -> int:
    from . import shots as shots_mod
    from .schema import to_json

    _print_json(out, to_json(shots_mod.detect_shots(args.path, args.shot_detector, args.shot_threshold)))
    return 0


def run_beats(args: argparse.Namespace, out: TextIO) -> int:
    from . import audio as audio_mod, ffmpeg_util
    from .schema import to_json

    pcm = ffmpeg_util.decode_pcm(args.path, sr=22050, mono=True)
    br = audio_mod.analyze_beats(pcm, 22050)
    _print_json(out, {"beats": br.beats_ms, "tempoBpm": br.tempo_bpm, "confidence": br.confidence,
                      "periodicity": br.periodicity, "numpyBpm": br.numpy_bpm, "librosaBpm": br.librosa_bpm,
                      "reason": br.reason, "markers": to_json(audio_mod.beat_markers(br.beats_ms, br.strengths))})
    return 0


def _version_of(dist: str, module: Optional[str] = None) -> Optional[str]:
    """Installed version without importing the package (``importlib.metadata``; importing ultralytics,
    transformers, open_clip, librosa and demucs took ~15 s of the old doctor)."""
    import importlib.metadata as md
    import importlib.util

    try:
        return md.version(dist)
    except Exception:
        pass
    try:
        if module and importlib.util.find_spec(module) is not None:
            return "installed"
    except Exception:
        pass
    return None


def run_doctor(args: argparse.Namespace, out: TextIO) -> int:
    from . import ffmpeg_util, models, shots as shots_mod, transitions
    from .director import ollama_reachable

    report: Dict[str, Any] = {"python": sys.version.split()[0], "pipelineVersion": __version__}
    try:
        report["ffmpeg"] = ffmpeg_util.find_ffmpeg()
        report["ffprobe"] = ffmpeg_util.find_ffprobe()
        report["ffmpegVersion"] = ffmpeg_util.ffmpeg_version()
    except ffmpeg_util.FFmpegNotFound as exc:
        report["ffmpeg"] = None
        report["ffmpegError"] = str(exc)
    torch_v = _version_of("torch", "torch")
    report["onnxruntime"] = _version_of("onnxruntime-gpu") or _version_of("onnxruntime", "onnxruntime")
    report["onnxruntimeProviders"] = shots_mod.preferred_providers()
    report["scenedetect"] = _version_of("scenedetect", "scenedetect")
    report["opencv"] = (_version_of("opencv-python-headless") or _version_of("opencv-python")
                        or _version_of("opencv-contrib-python", "cv2"))
    report["scipy"] = _version_of("scipy", "scipy")
    report["torch"] = torch_v
    report["torchCuda"] = False
    if torch_v:
        try:
            import torch  # type: ignore  (needed for the CUDA / VRAM facts)

            report["torchCuda"] = bool(torch.cuda.is_available())
            report["torchCudaVersion"] = torch.version.cuda
            report["torchDevice"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        except Exception:
            pass
    report["device"] = models.device()
    report["vram"] = models.vram_info()  # {"device", "freeMb", "totalMb"} via torch.cuda.mem_get_info
    report["vramBudgetMb"] = round(models.vram_budget_mb()) if report["vram"] else None
    report["torchvision"] = _version_of("torchvision", "torchvision")
    report["ultralytics"] = _version_of("ultralytics", "ultralytics")
    report["open_clip"] = _version_of("open_clip_torch", "open_clip")
    report["transformers"] = _version_of("transformers", "transformers")
    report["huggingface_hub"] = _version_of("huggingface_hub", "huggingface_hub")
    report["librosa"] = _version_of("librosa", "librosa")
    report["clip"] = _version_of("clip", "clip")
    report["clarabel"] = _version_of("clarabel", "clarabel")
    report["demucs"] = _version_of("demucs", "demucs")
    md_ = models.models_dir()
    report["modelsDir"] = str(md_)
    report["hfHome"] = os.environ.get("HF_HOME")
    status = models.model_status()
    report["models"] = {m.name: m.to_json() for m in status}
    report["modelsReady"] = all(m.present for m in status)
    report["shotBackend"] = shots_mod.transnet_backend() or "pyscenedetect"
    report["flowBackend"] = ("raft" if (torch_v and report["models"].get("raft_small", {}).get("present")
                                        and transitions.flow_backend_pref() in ("auto", "raft")) else
                             ("farneback" if transitions.flow_backend_pref() == "farneback" else "dis"))
    try:
        from . import characters as characters_mod

        mp = characters_mod.default_manifest_path()
        report["characters"] = (characters_mod.cache_state(characters_mod.load_manifest(mp)) if mp.is_file()
                                else {"manifest": str(mp), "present": False})
    except Exception as exc:
        report["characters"] = {"error": str(exc)}
    try:
        from . import separate as sep

        report["separation"] = {"model": sep.DEFAULT_MODEL, "present": sep.model_present(sep.DEFAULT_MODEL),
                                "sizeMb": sep.model_size_mb(sep.DEFAULT_MODEL), "stemsDir": str(sep.stems_root()),
                                "backend": ("demucs-" + ("cuda" if models.cuda_ok() else "cpu")) if report["demucs"] else None}
    except Exception as exc:
        report["separation"] = {"error": str(exc)}
    try:
        from . import resultcache

        report["analysisCache"] = str(resultcache.cache_root())
    except Exception:
        pass
    report["ollama"] = ollama_reachable()
    report["mode"] = "ml" if (report["ultralytics"] and torch_v) else "lite"
    _setup_summary(report, status)
    _print_json(out, report)
    return 0


# (report key, distribution names in order of preference, import name): what the ML path needs
REQUIRED_PACKAGES = (
    ("numpy", ("numpy",), "numpy"),
    ("opencv", ("opencv-python-headless", "opencv-python", "opencv-contrib-python"), "cv2"),
    ("scenedetect", ("scenedetect",), "scenedetect"),
    ("scipy", ("scipy",), "scipy"),
    ("onnxruntime", ("onnxruntime-gpu", "onnxruntime"), "onnxruntime"),
    ("clarabel", ("clarabel",), "clarabel"),
    ("osqp", ("osqp",), "osqp"),
    ("torch", ("torch",), "torch"),
    ("torchvision", ("torchvision",), "torchvision"),
    ("ultralytics", ("ultralytics",), "ultralytics"),
    ("clip", ("clip",), "clip"),
    ("open_clip_torch", ("open_clip_torch",), "open_clip"),
    ("transformers", ("transformers",), "transformers"),
    ("accelerate", ("accelerate",), "accelerate"),
    ("huggingface_hub", ("huggingface_hub",), "huggingface_hub"),
    ("onnx", ("onnx",), "onnx"),
    ("onnxscript", ("onnxscript",), "onnxscript"),
    ("librosa", ("librosa",), "librosa"),
    ("demucs", ("demucs",), "demucs"),
    ("julius", ("julius",), "julius"),
    ("einops", ("einops",), "einops"),
)


def _nvidia_gpu() -> Optional[Dict[str, Any]]:
    """``{"name", "driver"}`` from ``nvidia-smi`` (an NVIDIA GPU + driver, whatever torch can do)."""
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    from . import procs

    try:
        p = procs.run([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"], timeout=15,
                      stderr=subprocess.DEVNULL)
        line = p.stdout.decode("utf-8", "replace").strip().splitlines()[0] if p.returncode == 0 else ""
    except Exception:
        return None
    if not line:
        return None
    name, _, driver = line.partition(",")
    return {"name": name.strip(), "driver": driver.strip() or None}


def _setup_summary(report: Dict[str, Any], status) -> None:
    """Machine-readable facts for the installer's AI-setup wizard: ``packages`` (version or None),
    ``cudaAvailable``, ``nvidiaGpu``, ``missing`` (``"ffmpeg"``, ``"package:<name>"``,
    ``"model:<name>"``), ``warnings`` and ``ready`` (nothing missing). CUDA is not required: without an
    NVIDIA GPU the installer puts CPU torch wheels in and everything runs on the CPU."""
    packages: Dict[str, Optional[str]] = {}
    for key, dists, module in REQUIRED_PACKAGES:
        v = None
        for d in dists:
            v = _version_of(d)
            if v:
                break
        if v is None:
            v = _version_of("__none__", module)  # importable without metadata -> "installed"
        packages[key] = v
    report["packages"] = packages
    report["cudaAvailable"] = bool(report.get("torchCuda"))
    gpu = _nvidia_gpu()
    report["nvidiaGpu"] = gpu
    missing: List[str] = []
    if not report.get("ffmpeg"):
        missing.append("ffmpeg")
    missing += [f"package:{k}" for k, v in packages.items() if not v]
    missing += [f"model:{m.name}" for m in status if not m.present]
    warnings_: List[str] = []
    if gpu and packages.get("torch") and not report["cudaAvailable"]:
        warnings_.append(f"an NVIDIA GPU ({gpu['name']}) is present but torch has no CUDA "
                         f"(torch {packages['torch']}): everything runs on the CPU; reinstall torch from the CUDA index")
    if report.get("device") == "cuda" and (report.get("vram") or {}).get("freeMb", 1e9) < 1200:
        warnings_.append(f"only {report['vram']['freeMb']} MB of VRAM free: some stages fall back to the CPU")
    report["missing"] = missing
    report["warnings"] = warnings_
    report["ready"] = not missing


def run_interpolate(args: argparse.Namespace, em: Emitter) -> int:
    from . import ffmpeg_util, interpolate, transitions

    name = Path(args.src).name
    try:
        prog = lambda pct, msg: em.progress("export", name, pct, msg)  # noqa: E731
        if args.target_fps is not None:
            info = interpolate.interpolate_to_fps(args.src, args.in_ms, args.out_ms, args.target_fps, args.out,
                                                  progress=prog)
        else:
            info = interpolate.interpolate_segment(args.src, args.in_ms, args.out_ms, args.factor, args.out,
                                                   progress=prog)
    except (ValueError, ffmpeg_util.FFmpegError, ffmpeg_util.FFmpegNotFound, OSError) as exc:
        em.log("error", f"interpolate failed: {exc}")
        return 1
    except Exception as exc:
        em.log("error", f"interpolate crashed: {exc}")
        log.error(traceback.format_exc())
        return 1
    finally:
        transitions.release_raft()
    log.info("interpolate: %s", info)
    if args.target_fps is not None:
        em.log("info", f"interpolate {name} -> {info['rate']} fps: {info['frames']} frames in {info['seconds']} s "
                       f"({info['framesPerSecond']} fps), {info['copied']} copied / {info['interpolated']} interpolated / "
                       f"{info['held']} held / {info['cutsHeld']} cut-held, backend {info['backend']}"
                       + (f", peak VRAM {info['peakVramMb']:.0f} MB" if info.get('peakVramMb') else ""))
    em.result(ffmpeg_util.norm_path(args.out))
    return 0


def run_characters(args: argparse.Namespace, out: TextIO) -> int:
    from . import characters as characters_mod, ffmpeg_util, models, perception

    mp = Path(args.characters) if args.characters else characters_mod.default_manifest_path()
    if not mp.is_file():
        _print_json(out, {"error": f"characters manifest not found: {mp}"})
        return 1
    manifest = characters_mod.load_manifest(mp)
    if args.characters_command == "build":
        bank = characters_mod.build_bank(manifest, force=not args.if_stale, log_fn=lambda m: log.info(m))
        cal = dict(bank.calibration)
        rows = cal.pop("rows", [])
        _print_json(out, {"manifest": str(mp), "references": bank.counts(), "crops": str(manifest.cache_dir / "crops"),
                          "idThreshold": bank.id_threshold, "idMargin": bank.id_margin, "calibration": cal,
                          "negatives": {"texts": [t for t, o in zip(bank.texts, bank.text_owner) if o == characters_mod.NEG_ID],
                                        "images": bank.neg_sources},
                          "misidentified": [r for r in rows if r["true"] != r["predicted"]]})
        return 0
    # identify
    bank = characters_mod.build_bank(manifest, log_fn=lambda m: log.info(m))
    path = args.path
    ext = Path(path).suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        frame = characters_mod._read_image(Path(path))
    else:
        frame = ffmpeg_util.read_frame(path, args.at_ms)
        if frame is None:
            _print_json(out, {"error": f"no frame at {args.at_ms} ms"})
            return 1
    prompts = manifest.detection_prompts()
    det = (perception.GroundedSam2Detector() if args.detector == "grounded_sam2"
           else perception.YoloWorldDetector(prompts=prompts))
    emb = perception.OpenClipEmbedder()
    dets = perception.nms(det.detect(frame, prompts), 0.6, class_agnostic=True)
    embs = emb.embed_many(frame, [d.bbox for d in dets]) if dets else np.zeros((0, 512), np.float32)
    idents = perception.identify_detections(bank, emb, frame, dets, list(embs)) if len(dets) else []
    rows = []
    for d, e, ident in zip(dets, embs, idents):
        sc = bank.scores(e)
        zs = bank.zero_shot(e)
        rows.append({"bbox": [round(v, 1) for v in d.bbox], "label": d.label, "score": round(d.score, 4),
                     "character": ident.character, "characterName": manifest.name_of(ident.character),
                     "identityScore": ident.score, "margin": ident.margin, "best": ident.best,
                     "castTag": ident.loose(), "rejected": ident.rejected,
                     "zeroShot": {k: round(v, 3) for k, v in sorted(zs.items(), key=lambda kv: -kv[1])} if zs else None,
                     "negativeScore": round(bank.negative_score(e), 4),
                     "scores": {k: round(v, 4) for k, v in sorted(sc.items(), key=lambda kv: -kv[1])}})
    perception.close_detector(det)
    emb.close()
    models.release()
    _print_json(out, {"path": path, "atMs": args.at_ms, "width": int(frame.shape[1]), "height": int(frame.shape[0]),
                      "idThreshold": bank.id_threshold, "idMargin": bank.id_margin, "detections": rows})
    return 0


def run_separate(args: argparse.Namespace, out: TextIO) -> int:
    """``separate``: vocals / background stems per file. With ``--json`` stdout carries
    ``{"event":"progress","stage":"separate","clip":name,"pct":overall,"clipPct":..,"message":..}`` lines and one
    ``{"event":"result","path":<source as given>,"stems":{"vocals":..,"background":..}}`` per separated file."""
    from . import separate as sep

    em = Emitter(out) if args.json else None
    n = len(args.media)
    failed = 0

    def say(text: str) -> None:
        if em is None:
            out.write(text + "\n")
            out.flush()

    try:
        for i, path in enumerate(args.media):
            name = Path(path).name

            def prog(pct: float, msg: str, i=i, name=name) -> None:
                if em is not None:
                    overall = (i + min(max(pct, 0.0), 1.0)) / n
                    em._emit({"event": "progress", "stage": "separate", "clip": name, "pct": round(overall, 4),
                              "clipPct": round(min(max(pct, 0.0), 1.0), 4), "message": f"{name}: {msg}"})

            try:
                res = sep.separate_file(path, model=args.model, device=args.device, progress=prog, force=args.force)
            except Exception as exc:  # one bad file must not stop the others
                failed += 1
                log.debug(traceback.format_exc())
                if em is not None:
                    em.log("error", f"{name}: separation failed: {type(exc).__name__}: {exc}")
                say(f"{name}: FAILED ({exc})")
                continue
            info = (f"{name}: {'cached' if res.cached else f'separated on {res.device} in {res.seconds:.1f}s'}"
                    + (f", {res.audioSeconds:.1f}s audio" if not res.cached else "")
                    + (f", peak VRAM {res.peakVramMb:.0f} MB" if res.peakVramMb else "")
                    + (f", stats {json.dumps(res.stats)}" if res.stats else ""))
            if em is not None:
                em.log("info", info)
                em._emit({"event": "result", "path": path, "stems": res.stems})
            else:
                say(info)
                for k, v in res.stems.items():
                    say(f"  {k:<10} {v}")
    finally:
        sep.release_model()
    return 1 if failed else 0


def run_download_models(args: argparse.Namespace, em: Emitter) -> int:
    from . import downloads, models

    def prog(name: str, pct: float, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        em.progress("download", name, pct, msg, **(extra or {}))

    results = downloads.download_all(prog, only=args.only)
    failed = [(n, e) for n, e in results if e]
    for n, e in failed:
        em.log("error", f"{n}: {e}")
    em.log("info", "models: " + ", ".join(f"{m.name}={'ok' if m.present else 'missing'}" for m in models.model_status()))
    em.result(str(models.models_dir()).replace("\\", "/"))
    return 1 if failed else 0


# --------------------------------------------------------------------------- main


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # --watch-stdin is accepted anywhere on the command line (before or after the sub-command)
    watch_flag = "--watch-stdin" in argv
    argv = [a for a in argv if a != "--watch-stdin"]
    parser = build_parser()
    args = parser.parse_args(argv)
    quiet_libraries()
    from . import procs

    if procs.watch_stdin_enabled(watch_flag):
        procs.start_stdin_watchdog()
    original_stdout = sys.stdout
    real_stdout = _take_stdout()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("YOLO_VERBOSE", "False")
    try:
        if args.command == "analyze":
            em = Emitter(real_stdout)
            try:
                return run_analyze(args, em)
            except Exception as exc:
                em.log("error", f"pipeline crashed: {exc}")
                log.error(traceback.format_exc())
                return 1
        if args.command == "probe":
            return run_probe(args, real_stdout)
        if args.command == "shots":
            return run_shots(args, real_stdout)
        if args.command == "beats":
            return run_beats(args, real_stdout)
        if args.command == "order":
            return run_order(args, real_stdout)
        if args.command == "doctor":
            return run_doctor(args, real_stdout)
        if args.command == "interpolate":
            return run_interpolate(args, Emitter(real_stdout))
        if args.command == "download-models":
            return run_download_models(args, Emitter(real_stdout))
        if args.command == "separate":
            return run_separate(args, real_stdout)
        if args.command == "characters":
            return run_characters(args, real_stdout)
        parser.print_help(sys.stderr)
        return 2
    finally:
        _restore_stdout(original_stdout, real_stdout)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
