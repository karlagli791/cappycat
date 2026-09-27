"""Voice / background separation (``separate.py``, ``python -m cappycat_pipeline separate``).

The base tests cover the cache, the WAV I/O, the audio-offset convention and the CLI contract without
loading a model. ``@pytest.mark.gpu`` tests run Demucs on a synthetic mix of a formant-synthesised
"voice" (glottal pulse train with vibrato through vowel formant filters, syllable envelope, phrase
pauses) and a drum loop + noise bed; they skip when demucs / the weights are missing
(``python -m cappycat_pipeline download-models``) and use CUDA when available.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import lfilter, resample_poly

from cappycat_pipeline import ffmpeg_util, models
from cappycat_pipeline import separate as sep

PIPELINE_DIR = Path(__file__).resolve().parent.parent
SR = 44100
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# --------------------------------------------------------------------------- synthetic signals


def formant_voice(dur: float, sr: int = SR, seed: int = 0) -> np.ndarray:
    """Speech-like signal: Rosenberg glottal pulses (f0 ~160 Hz, declination + 5.5 Hz vibrato + jitter)
    through three vowel formant resonators per 0.26 s syllable, syllable envelope, 0.8 s phrase pauses."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    f0 = 160 + 30 * np.sin(2 * np.pi * 0.35 * t) + 4 * np.sin(2 * np.pi * 5.5 * t)
    f0 *= 1 + 0.01 * np.convolve(rng.standard_normal(n), np.ones(441) / 441, "same")
    ph = np.cumsum(f0 / sr) % 1.0
    g = np.where(ph < 0.4, 0.5 * (1 - np.cos(np.pi * ph / 0.4)), np.where(ph < 0.6, np.cos(np.pi * (ph - 0.4) / 0.4), 0.0))
    src = np.diff(g, prepend=0.0) + 0.02 * rng.standard_normal(n)
    vowels = [[(730, 90), (1090, 110), (2440, 170)], [(270, 60), (2290, 100), (3010, 170)],
              [(300, 60), (870, 90), (2240, 170)], [(530, 70), (1840, 100), (2480, 170)],
              [(570, 70), (840, 90), (2410, 170)]]
    syl = 0.26
    out = np.zeros(n)
    for k in range(int(np.ceil(dur / syl))):
        a, b = int(k * syl * sr), min(n, int((k + 1) * syl * sr))
        y = src[a:b]
        for f, bw in vowels[int(rng.integers(len(vowels)))]:
            r, th = np.exp(-np.pi * bw / sr), 2 * np.pi * f / sr
            y = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], y)
        out[a:b] = y
    env = np.clip(np.sin(np.pi * ((t / syl) % 1.0)), 0, 1) ** 0.6
    env *= np.convolve(((t % 3.0) < 2.2).astype(float), np.ones(2205) / 2205, "same")
    v = out * env
    return (v / np.max(np.abs(v)) * 0.5).astype(np.float32)


def drum_bed(dur: float, sr: int = SR, seed: int = 1, bpm: float = 100) -> np.ndarray:
    """Kick on every beat, snare on 2 and 4, off-beat hi-hats, plus a rain-like noise bed."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    out = np.zeros(n)
    beat = 60 / bpm
    for k in range(int(dur / beat) + 1):
        a = int(k * beat * sr)
        if a >= n:
            break
        tt = np.arange(min(n - a, int(0.35 * sr))) / sr
        out[a:a + len(tt)] += 0.9 * np.sin(2 * np.pi * (50 * tt + 2 * (1 - np.exp(-tt * 30)))) * np.exp(-tt * 9)
        h = int((k + 0.5) * beat * sr)
        if h < n:
            m = min(n - h, int(0.06 * sr))
            out[h:h + m] += 0.5 * np.diff(rng.standard_normal(m) * np.exp(-np.arange(m) / sr * 60), prepend=0)
        if k % 2 == 1:
            m = min(n - a, int(0.2 * sr))
            out[a:a + m] += 0.4 * rng.standard_normal(m) * np.exp(-np.arange(m) / sr * 18)
    out += np.convolve(rng.standard_normal(n), np.ones(8) / 8, "same") * 0.03
    return (out / np.max(np.abs(out)) * 0.5).astype(np.float32)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-20))


def _ffmpeg(*args: str) -> None:
    subprocess.run([ffmpeg_util.find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args], check=True,
                   capture_output=True, creationflags=_NO_WINDOW)


def _decode_range(path: Path, start_s: float, dur_s: float) -> np.ndarray:
    """What the exporter does: ``-ss start -i path -t dur``, 48 kHz stereo f32."""
    proc = subprocess.run([ffmpeg_util.find_ffmpeg(), "-v", "error", "-nostdin", "-ss", f"{start_s:.6f}", "-i", str(path),
                           "-t", f"{dur_s:.6f}", "-vn", "-ac", "2", "-ar", "48000", "-f", "f32le", "pipe:1"],
                          capture_output=True, check=True, creationflags=_NO_WINDOW)
    return np.frombuffer(proc.stdout, "<f4").reshape(-1, 2)


@pytest.fixture()
def stems_dir(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "stems"
    monkeypatch.setenv("CAPPYCAT_STEMS_DIR", str(d))
    return d


@pytest.fixture(scope="module")
def delayed_audio_clip(tmp_path_factory) -> Path:
    """2 s video from t=0 with a 440 Hz AAC track starting at 0.5 s."""
    if not ffmpeg_util.ffmpeg_available():
        pytest.skip("ffmpeg not found")
    out = tmp_path_factory.mktemp("sep") / "delayed.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x90:rate=24:duration=2",
            "-itsoffset", "0.5", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=1.5",
            "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(out))
    return out


# --------------------------------------------------------------------------- base tests


def test_hf_repo_names():
    assert sep.hf_repo("htdemucs_ft") == "adefossez/HTDemucs-ft"
    assert sep.hf_repo("htdemucs") == "adefossez/HTDemucs"
    assert sep.DEFAULT_MODEL in sep.SUPPORTED_MODELS


def test_wav_round_trip(tmp_path):
    x = np.random.default_rng(0).uniform(-1.2, 1.2, (4801, 2)).astype(np.float32)
    p = tmp_path / "x.wav"
    sep.write_wav_f32(p, x, 48000)
    y, rate = sep.read_wav_f32(p)
    assert rate == 48000 and y.shape == x.shape and np.array_equal(x, y)
    # ffmpeg reads it back identically (the exporter decodes stems with ffmpeg)
    assert np.allclose(_decode_range(p, 0.0, 1.0), x, atol=1e-7)


def test_cache_key_depends_on_file_and_model(tmp_path):
    p = tmp_path / "a.wav"
    sep.write_wav_f32(p, np.zeros((100, 2), np.float32))
    k1 = sep.cache_key(p, "htdemucs_ft")
    assert k1 == sep.cache_key(p, "htdemucs_ft") and len(k1) == 40
    assert sep.cache_key(p, "htdemucs") != k1
    os.utime(p, (time.time() + 10, time.time() + 10))
    assert sep.cache_key(p, "htdemucs_ft") != k1
    paths = sep.stem_paths(p, root=tmp_path / "root")
    assert set(paths) == {"vocals", "background"} and paths["vocals"].parent.parent == tmp_path / "root"


def test_audio_offset_convention(delayed_audio_clip, tmp_path):
    assert abs(sep.audio_offset_ms(delayed_audio_clip) - 500.0) < 30.0
    silent = tmp_path / "silent.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent))
    assert sep.audio_offset_ms(silent) is None
    with pytest.raises(ValueError, match="no audio"):
        sep.separate_file(str(silent), cache_root=tmp_path / "stems")


def _fake_cache(src: Path, root: Path) -> dict:
    sp = sep.stem_paths(src, sep.DEFAULT_MODEL, root)
    d = sp["vocals"].parent
    d.mkdir(parents=True)
    for p in sp.values():
        sep.write_wav_f32(p, np.zeros((480, 2), np.float32))
    (d / "meta.json").write_text("{}", encoding="utf-8")
    return sp


def test_cached_stems_skip_the_model(tmp_path, monkeypatch):
    src = tmp_path / "src.wav"
    sep.write_wav_f32(src, np.zeros((4800, 2), np.float32))
    root = tmp_path / "stems"
    assert sep.cached_stems(src, root=root) is None
    sp = _fake_cache(src, root)

    def boom(*_a, **_k):
        raise AssertionError("model must not load on a cache hit")

    monkeypatch.setattr(sep, "_get_model", boom)
    res = sep.separate_file(str(src), cache_root=root)
    assert res.cached and res.stems["vocals"].endswith("/vocals.wav")
    assert Path(res.stems["background"]) == sp["background"]
    (sp["vocals"].parent / "meta.json").unlink()  # incomplete cache entry -> miss
    assert sep.cached_stems(src, root=root) is None


def test_cli_contract_cache_hit_and_missing_file(tmp_path):
    src = tmp_path / "src.wav"
    sep.write_wav_f32(src, np.zeros((4800, 2), np.float32))
    root = tmp_path / "stems"
    _fake_cache(src, root)
    env = {**os.environ, "CAPPYCAT_STEMS_DIR": str(root)}
    proc = subprocess.run([sys.executable, "-m", "cappycat_pipeline", "separate", str(src), str(tmp_path / "missing.mp4"),
                           "--json"], cwd=str(PIPELINE_DIR), capture_output=True, text=True, encoding="utf-8", env=env,
                          timeout=300, creationflags=_NO_WINDOW)
    events = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]  # stdout = JSON lines only
    assert proc.returncode == 1  # one of two files failed
    prog = [e for e in events if e["event"] == "progress"]
    assert prog and all(e["stage"] == "separate" and 0 <= e["pct"] <= 1 for e in prog)
    assert {e["clip"] for e in prog} == {"src.wav"}
    results = [e for e in events if e["event"] == "result"]
    assert len(results) == 1 and results[0]["path"] == str(src)
    assert set(results[0]["stems"]) == {"vocals", "background"}
    errors = [e for e in events if e["event"] == "log" and e["level"] == "error"]
    assert len(errors) == 1 and "missing.mp4" in errors[0]["message"]


# --------------------------------------------------------------------------- model tests


def _need_model(model: str = sep.DEFAULT_MODEL) -> None:
    try:
        import demucs  # noqa: F401
    except Exception:
        pytest.skip("demucs not installed")
    if not sep.model_present(model):
        pytest.skip(f"Demucs {model} weights missing (python -m cappycat_pipeline download-models)")


@pytest.mark.gpu
@pytest.mark.parametrize("model", ["htdemucs_ft", "htdemucs"])
def test_separates_voice_from_drum_bed(tmp_path, model):
    _need_model(model)
    dur = 12.0
    voice, bed = formant_voice(dur), drum_bed(dur)
    src = tmp_path / "mix.wav"
    sep.write_wav_f32(src, np.stack([voice + bed, 0.9 * voice + bed], 1), SR)
    root = tmp_path / "stems"
    res = sep.separate_file(str(src), model=model, cache_root=root)
    assert not res.cached and res.device in ("cuda", "cpu")
    if models.cuda_ok():
        assert res.device == "cuda" and res.peakVramMb is not None and res.peakVramMb < 2048
    voc, r1 = sep.read_wav_f32(res.stems["vocals"])
    bg, r2 = sep.read_wav_f32(res.stems["background"])
    assert r1 == r2 == 48000 and voc.shape == bg.shape and voc.shape[1] == 2
    assert len(voc) == len(ffmpeg_util.decode_pcm(src, 48000, mono=False)) == int(dur * 48000)
    v48, b48 = resample_poly(voice, 160, 147), resample_poly(bed, 160, 147)
    vm, bm = voc.mean(1), bg.mean(1)
    c = {"voc~voice": _corr(vm, v48), "voc~bed": _corr(vm, b48), "bg~voice": _corr(bm, v48), "bg~bed": _corr(bm, b48)}
    assert c["voc~voice"] > 0.5 and abs(c["voc~bed"]) < 0.15, c
    assert c["bg~bed"] > 0.9 and abs(c["bg~voice"]) < 0.2, c
    assert c["voc~voice"] > 4 * abs(c["voc~bed"]) and c["bg~bed"] > 4 * abs(c["bg~voice"]), c

    t = time.perf_counter()
    again = sep.separate_file(str(src), model=model, cache_root=root)
    assert again.cached and again.stems == res.stems and time.perf_counter() - t < 1.0


@pytest.mark.gpu
def test_stems_follow_the_source_timeline(delayed_audio_clip, tmp_path):
    """Stems start at the file's t=0 (audio stream offset padded) and end with the source audio, so
    ``-ss X`` on the stems returns the same instant as ``-ss X`` on the source."""
    _need_model()
    res = sep.separate_file(str(delayed_audio_clip), cache_root=tmp_path / "stems")
    voc, _ = sep.read_wav_f32(res.stems["vocals"])
    bg, _ = sep.read_wav_f32(res.stems["background"])
    assert abs(res.offsetMs - 500) < 30
    assert len(voc) == len(bg) == int(round(res.offsetMs * 48)) + len(ffmpeg_util.decode_pcm(delayed_audio_clip, 48000, False))
    lead = int(0.45 * 48000)
    assert np.abs(voc[:lead]).max() == 0 and np.abs(bg[:lead]).max() == 0
    ref = _decode_range(delayed_audio_clip, 1.0, 0.8)
    both = _decode_range(Path(res.stems["vocals"]), 1.0, 0.8) + _decode_range(Path(res.stems["background"]), 1.0, 0.8)
    n = min(len(ref), len(both))
    assert n > 0.75 * 48000
    assert _corr(ref[:n, 0], both[:n, 0]) > 0.9  # a pure tone: whatever the split, the sum stays aligned
