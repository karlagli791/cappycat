"""Loudness (EBU R128 via ffmpeg ``ebur128=peak=true``, ``loudnorm`` as fallback) and beat detection.

Loudness: ``ebur128`` agrees with ``loudnorm``'s measurement pass within 0.6 LU on the real clips
and is ~3.7x faster. The recommended gain brings the clip to the target loudness **but never pushes
the true peak above -1 dBTP**: ``gain = min(target - I, -1 - TP)`` (clamped to +-12 dB).

Beats: a numpy periodicity gate (spectral-flux onset envelope -> normalised autocorrelation at the
best tempo lag must be >= :data:`BEAT_MIN_PERIODICITY`) decides whether the audio has a pulse at
all; dialogue / ambience scores 0.03-0.16, music with a beat well above 0.3. When librosa is
installed its tracker must also *agree* with the numpy tempo (ratio ~1, 2 or 1/2); otherwise no
beats are emitted. The result carries a confidence in [0, 1]. Without librosa the numpy tracker
places the beats (phase by peak-picking the onset envelope on the tempo grid). Strong / weak beats
(``beat1`` / ``beat2``) are split by onset-strength quantile.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import ffmpeg_util, procs
from .schema import BeatMarker

log = logging.getLogger("cappycat.audio")

GAIN_CLAMP_DB = 12.0
TRUE_PEAK_CEILING_DB = -1.0
BEAT_MIN_PERIODICITY = 0.25   # dialogue / ambience measured 0.03-0.16 on the real clips
TEMPO_AGREEMENT = 0.06        # numpy vs librosa tempo: ratio within 6 % of 1, 2 or 1/2


# --------------------------------------------------------------------------- loudness


def _parse_loudnorm_json(stderr_text: str) -> Optional[dict]:
    """The loudnorm filter prints a JSON block at the end of stderr."""
    matches = list(re.finditer(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr_text, flags=re.S))
    if not matches:
        return None
    try:
        return json.loads(matches[-1].group(0))
    except json.JSONDecodeError:
        return None


_EBUR_SUMMARY = re.compile(r"Summary:(.*)", re.S)


def _parse_ebur128(stderr_text: str) -> Optional[Tuple[float, float, float]]:
    """``(I, true_peak, LRA)`` from the ``ebur128`` filter's summary block (None when absent)."""
    m = _EBUR_SUMMARY.search(stderr_text)
    if not m:
        return None
    body = m.group(1)

    def last(pattern: str) -> Optional[float]:
        hits = re.findall(pattern, body)
        if not hits:
            return None
        v = hits[-1].strip().lower()
        if v in ("-inf", "inf", "nan"):
            return float("-inf") if v != "nan" else None
        try:
            return float(v)
        except ValueError:
            return None

    integrated = last(r"I:\s+(-?inf|-?[\d.]+)\s+LUFS")
    peak = last(r"Peak:\s+(-?inf|-?[\d.]+)\s+dBFS")
    lra = last(r"LRA:\s+(-?inf|-?[\d.]+)\s+LU(?!FS)")
    if integrated is None:
        return None
    return integrated, (peak if peak is not None else -99.0), (lra if lra is not None else 0.0)


def measure_loudness(path: str | os.PathLike, target_lufs: float = -14.0, true_peak: float = -1.0,
                     lra: float = 11.0, method: str = "ebur128") -> Optional[Tuple[float, float, float]]:
    """Return ``(integrated_lufs, true_peak_db, loudness_range)`` or ``None`` when the file
    has no measurable audio. ``method="ebur128"`` (default, fast) falls back to ``loudnorm``."""
    ffmpeg = ffmpeg_util.find_ffmpeg()
    if method == "ebur128":
        cmd = [ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-i", str(path), "-vn", "-sn",
               "-af", "ebur128=peak=true:framelog=verbose", "-f", "null", "-"]
        proc = procs.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        res = _parse_ebur128(proc.stderr.decode("utf-8", "replace"))
        if res is not None:
            integrated, tp, rng = res
            if not math.isfinite(integrated) or integrated <= -70.0:  # silence / gated out entirely
                return None
            return integrated, (tp if math.isfinite(tp) else -99.0), (rng if math.isfinite(rng) else 0.0)
        log.info("ebur128 summary not found for %s; using loudnorm", path)
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-i", str(path), "-vn", "-sn",
           "-af", f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}:print_format=json", "-f", "null", "-"]
    proc = procs.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    data = _parse_loudnorm_json(proc.stderr.decode("utf-8", "replace"))
    if not data:
        return None

    def _f(key: str) -> Optional[float]:
        v = data.get(key)
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(x) else x

    integrated, tp, rng = _f("input_i"), _f("input_tp"), _f("input_lra")
    if integrated is None or integrated <= -70.0:  # silence / no audio
        return None
    return integrated, (tp if tp is not None else -99.0), (rng if rng is not None else 0.0)


def recommended_gain_db(integrated_lufs: float, target_lufs: float = -14.0, clamp: float = GAIN_CLAMP_DB,
                        true_peak_db: Optional[float] = None, peak_ceiling_db: float = TRUE_PEAK_CEILING_DB) -> float:
    """Gain to reach ``target_lufs``, limited so the true peak stays at or below ``peak_ceiling_db``
    (``min(target - I, ceiling - TP)``), clamped to +-``clamp`` dB. A clip already peaking above the
    ceiling is turned down."""
    gain = target_lufs - integrated_lufs
    if true_peak_db is not None and math.isfinite(true_peak_db) and true_peak_db > -90.0:
        gain = min(gain, peak_ceiling_db - true_peak_db)
    return float(min(max(gain, -clamp), clamp))


# --------------------------------------------------------------------------- onset envelope (numpy)


def _stft_mag(y: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    pad = n_fft // 2
    y = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + (len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = y[idx] * np.hanning(n_fft).astype(np.float32)[None, :]
    return np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)  # (frames, bins)


def onset_envelope(pcm: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256) -> Tuple[np.ndarray, float]:
    """Spectral-flux onset strength (positive log-magnitude differences, mel-ish band pooling
    by simple log-spaced averaging). Returns ``(envelope, frames_per_second)``."""
    mag = _stft_mag(pcm, n_fft, hop)
    # pool bins into ~64 log-spaced bands to de-emphasise high-frequency noise
    n_bins = mag.shape[1]
    edges = np.unique(np.round(np.geomspace(1, n_bins - 1, 65)).astype(int))
    bands = np.stack([mag[:, a:b].mean(axis=1) if b > a else mag[:, a] for a, b in zip(edges[:-1], edges[1:])], axis=1)
    logb = np.log1p(1000.0 * bands / (bands.max() + 1e-9))
    flux = np.diff(logb, axis=0, prepend=logb[:1])
    env = np.maximum(flux, 0.0).mean(axis=1)
    # local mean removal (as in librosa's onset_strength), ~0.2 s window
    win = max(3, int(0.2 * sr / hop)) | 1
    kernel = np.ones(win) / win
    local = np.convolve(np.pad(env, win // 2, mode="edge"), kernel, mode="valid")
    env = np.maximum(env - local, 0.0)
    if env.max() > 0:
        env = env / env.max()
    return env.astype(np.float32), sr / float(hop)


# --------------------------------------------------------------------------- tempo + beats (numpy)


def estimate_tempo_detailed(env: np.ndarray, fps: float, bpm_min: float = 60.0, bpm_max: float = 200.0,
                            prior_bpm: float = 120.0) -> Tuple[Optional[float], float]:
    """``(tempo_bpm, periodicity)``: autocorrelation tempo and the normalised autocorrelation at
    that lag (0 = no pulse, 1 = perfectly periodic). Tempo is None when the envelope is too short
    or empty."""
    if len(env) < int(fps * 2) or float(env.max(initial=0.0)) <= 0.0:
        return None, 0.0
    # onset peaks are ~1 frame wide and periods are fractional: smooth slightly and analyse
    # the autocorrelation at 4x lag resolution so a period of e.g. 34.45 frames is measured
    # as well as an integer one.
    up = 4
    x = np.convolve(env - env.mean(), np.array([0.25, 0.5, 0.25]), mode="same")
    x = np.interp(np.arange(0, len(x) - 1, 1.0 / up), np.arange(len(x)), x)
    fps_up = fps * up
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / (ac[0] + 1e-9)
    lag_min = int(math.floor(fps_up * 60.0 / bpm_max))
    lag_max = int(math.ceil(fps_up * 60.0 / bpm_min))
    lag_max = min(lag_max, n - 1)
    if lag_max <= lag_min + 1:
        return None, 0.0
    lags = np.arange(lag_min, lag_max + 1)
    bpms = 60.0 * fps_up / lags
    # log-normal prior around ``prior_bpm`` (like librosa) to discourage octave errors
    prior = np.exp(-0.5 * ((np.log2(bpms) - np.log2(prior_bpm)) / 1.0) ** 2)
    score = ac[lags] * prior
    best = int(np.argmax(score))
    periodicity = float(max(0.0, ac[lags[best]]))
    # parabolic refinement of the lag
    lag = float(lags[best])
    if 0 < best < len(lags) - 1:
        y0, y1, y2 = score[best - 1], score[best], score[best + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-12:
            lag += 0.5 * (y0 - y2) / denom
    return float(60.0 * fps_up / lag), periodicity


def estimate_tempo(env: np.ndarray, fps: float, bpm_min: float = 60.0, bpm_max: float = 200.0,
                   prior_bpm: float = 120.0, min_periodicity: float = 0.12) -> Optional[float]:
    """Autocorrelation tempo estimate; ``None`` when the onset envelope is too short or shows
    no periodicity (normalised autocorrelation at the best lag < ``min_periodicity``), e.g. a
    sustained tone or silence."""
    tempo, periodicity = estimate_tempo_detailed(env, fps, bpm_min, bpm_max, prior_bpm)
    if tempo is None or periodicity < min_periodicity:
        return None
    return tempo


def tempos_agree(a: Optional[float], b: Optional[float], tol: float = TEMPO_AGREEMENT) -> bool:
    """Same pulse: ratio within ``tol`` of 1, 2 or 1/2 (octave errors are the same beat grid)."""
    if not a or not b or a <= 0 or b <= 0:
        return False
    r = a / b
    return any(abs(r / k - 1.0) <= tol for k in (1.0, 2.0, 0.5))


def track_beats(env: np.ndarray, fps: float, tempo_bpm: float, min_strength: float = 0.02) -> np.ndarray:
    """Beat frame positions on the tempo grid, phase chosen to maximise onset energy, each
    beat snapped to the local onset peak within +-1/8 period."""
    period = fps * 60.0 / tempo_bpm
    n = len(env)
    if period <= 1 or n == 0:
        return np.zeros((0,), dtype=np.float64)
    n_phase = max(8, int(round(period)))
    best_phase, best_score = 0.0, -1.0
    grid = np.arange(0, n, period)
    for k in range(n_phase):
        phi = period * k / n_phase
        pos = grid + phi
        pos = pos[pos < n]
        s = np.interp(pos, np.arange(n), env).sum()
        if s > best_score:
            best_score, best_phase = s, phi
    beats = grid + best_phase
    beats = beats[beats < n]
    radius = max(1, int(period / 8))
    snapped: List[float] = []
    for b in beats:
        c = int(round(b))
        lo, hi = max(0, c - radius), min(n, c + radius + 1)
        if hi <= lo:
            continue
        p = lo + int(np.argmax(env[lo:hi]))
        if env[p] < min_strength:  # grid point past the last onset / in silence
            continue
        pos = float(p)
        if 0 < p < n - 1:  # parabolic sub-frame refinement
            y0, y1, y2 = env[p - 1], env[p], env[p + 1]
            denom = y0 - 2 * y1 + y2
            if abs(denom) > 1e-12:
                pos += float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
        if not snapped or pos - snapped[-1] > period / 2:
            snapped.append(pos)
    return np.array(snapped, dtype=np.float64)


@dataclass
class BeatResult:
    beats_ms: List[float] = field(default_factory=list)
    tempo_bpm: Optional[float] = None
    strengths: List[float] = field(default_factory=list)
    confidence: float = 0.0            # 0..1; beats are only emitted when the pulse is trusted
    periodicity: float = 0.0           # numpy autocorrelation at the tempo lag
    numpy_bpm: Optional[float] = None
    librosa_bpm: Optional[float] = None
    reason: str = ""


def beat_confidence(periodicity: float, agree: Optional[bool]) -> float:
    """Confidence from the numpy periodicity (0.25 -> 0.5, >= 0.5 -> 1.0) and the numpy / librosa
    agreement (``None`` = librosa unavailable: numpy alone, capped at 0.8)."""
    base = float(min(1.0, max(0.0, periodicity) / 0.5))
    if agree is None:
        return round(0.8 * base, 3)
    return round(base if agree else 0.0, 3)


def analyze_beats(pcm: np.ndarray, sr: int, use_librosa: bool = True,
                  min_periodicity: float = BEAT_MIN_PERIODICITY) -> BeatResult:
    """Beat grid with a confidence; no beats when the audio has no trustworthy pulse (see module
    docstring)."""
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if len(pcm) < sr // 2 or float(np.abs(pcm).max(initial=0.0)) < 1e-5:
        return BeatResult(reason="silent / too short")
    librosa = None
    if use_librosa:
        try:
            import librosa  # type: ignore
        except Exception:
            librosa = None  # type: ignore
    env, fps = onset_envelope(pcm, sr)
    tempo, periodicity = estimate_tempo_detailed(env, fps)
    res = BeatResult(periodicity=round(periodicity, 4), numpy_bpm=round(tempo, 2) if tempo else None)
    if tempo is None or periodicity < min_periodicity:
        res.reason = f"no pulse (periodicity {periodicity:.2f} < {min_periodicity:.2f})"
        return res
    if librosa is not None:
        try:
            lenv = librosa.onset.onset_strength(y=pcm, sr=sr)
            ltempo, frames = librosa.beat.beat_track(onset_envelope=lenv, sr=sr, units="frames")
            ltempo = float(np.atleast_1d(ltempo)[0])
            res.librosa_bpm = round(ltempo, 2) if ltempo > 0 else None
            agree = tempos_agree(tempo, ltempo)
            res.confidence = beat_confidence(periodicity, agree)
            if not agree:
                res.reason = f"numpy {tempo:.1f} BPM and librosa {ltempo:.1f} BPM disagree"
                return res
            times = librosa.frames_to_time(frames, sr=sr) * 1000.0
            lenv = lenv / (lenv.max() + 1e-9)
            res.beats_ms = [round(float(t), 3) for t in times]
            res.tempo_bpm = round(ltempo, 3) if ltempo > 0 else None
            res.strengths = [float(lenv[min(int(f), len(lenv) - 1)]) for f in frames]
            res.reason = "librosa beat grid, tempo confirmed by the numpy periodicity"
            return res
        except Exception as exc:  # pragma: no cover - only when librosa is installed but fails
            log.warning("librosa beat tracking failed (%s); using numpy fallback", exc)
    res.confidence = beat_confidence(periodicity, None)
    beats = track_beats(env, fps, tempo)
    res.beats_ms = [round(float(b / fps * 1000.0), 3) for b in beats]
    res.strengths = [float(env[int(b)]) for b in beats]
    res.tempo_bpm = round(float(tempo), 3)
    res.reason = "numpy beat grid"
    return res


def detect_beats_detailed(pcm: np.ndarray, sr: int, use_librosa: bool = True
                          ) -> Tuple[List[float], Optional[float], List[float]]:
    """Return ``(beat_ms, tempo_bpm, strengths_0_1)`` (empty when unconfident; see :func:`analyze_beats`)."""
    r = analyze_beats(pcm, sr, use_librosa)
    return r.beats_ms, r.tempo_bpm, r.strengths


def detect_beats(pcm: np.ndarray, sr: int) -> Tuple[List[float], Optional[float]]:
    beat_ms, tempo, _ = detect_beats_detailed(pcm, sr)
    return beat_ms, tempo


def classify_beats(strengths: Sequence[float], quantile: float = 0.6) -> List[str]:
    """``beat1`` for strong beats (>= quantile of onset strength), ``beat2`` otherwise."""
    if not strengths:
        return []
    s = np.asarray(strengths, dtype=np.float64)
    thr = float(np.quantile(s, quantile)) if len(s) > 1 else s[0]
    return ["beat1" if v >= thr and v > 0 else "beat2" for v in s]


def beat_markers(beat_ms: Sequence[float], strengths: Sequence[float], offset_ms: float = 0.0) -> List[BeatMarker]:
    kinds = classify_beats(strengths)
    out = []
    for i, t in enumerate(beat_ms):
        strength = float(strengths[i]) if i < len(strengths) else 0.5
        kind = kinds[i] if i < len(kinds) else "beat2"
        out.append(BeatMarker(timeMs=round(float(t) + offset_ms, 3), strength=round(min(max(strength, 0.0), 1.0), 4), kind=kind))
    return out


# --------------------------------------------------------------------------- synthetic helpers (tests)


def synth_click_track(bpm: float = 120.0, seconds: float = 8.0, sr: int = 22050, accent_every: int = 4) -> np.ndarray:
    """Click track with an accented click every ``accent_every`` beats."""
    n = int(seconds * sr)
    y = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    click_len = int(0.02 * sr)
    t = np.arange(click_len) / sr
    click = (np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 200)).astype(np.float32)
    i = 0
    while True:
        start = int(round(i * period * sr))
        if start + click_len > n:
            break
        y[start:start + click_len] += click * (1.0 if i % accent_every == 0 else 0.4)
        i += 1
    return y
