import numpy as np
import pytest

from cappycat_pipeline import ffmpeg_util
from cappycat_pipeline.audio import (_parse_loudnorm_json, beat_markers, classify_beats, detect_beats, detect_beats_detailed,
                                     measure_loudness, recommended_gain_db, synth_click_track)

SR = 22050


@pytest.mark.parametrize("bpm", [90.0, 120.0, 150.0])
def test_click_track_tempo(bpm):
    y = synth_click_track(bpm=bpm, seconds=10.0, sr=SR)
    beats, tempo = detect_beats(y, SR)
    assert tempo is not None and abs(tempo - bpm) <= 3.0
    assert len(beats) >= 10
    spacing = np.diff(beats)
    assert abs(np.median(spacing) - 60000.0 / bpm) < 30.0  # within ~1 STFT hop


def test_beats_land_on_clicks():
    y = synth_click_track(bpm=120.0, seconds=8.0, sr=SR)
    beats, tempo, strengths = detect_beats_detailed(y, SR, use_librosa=False)
    assert len(beats) == len(strengths)
    for b in beats:
        assert min(abs(b - k * 500.0) for k in range(20)) < 40.0
    kinds = classify_beats(strengths)
    assert set(kinds) <= {"beat1", "beat2"} and "beat1" in kinds and "beat2" in kinds


def test_no_beats_for_tone_or_silence():
    t = np.arange(SR * 6) / SR
    sine = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    assert detect_beats(sine, SR) == ([], None)
    assert detect_beats(np.zeros(SR * 3, dtype=np.float32), SR) == ([], None)
    assert detect_beats(np.zeros(10, dtype=np.float32), SR) == ([], None)


def test_beat_markers_offset_and_bounds():
    m = beat_markers([0.0, 500.0, 1000.0], [1.0, 0.2, 1.5], offset_ms=250.0)
    assert [x.timeMs for x in m] == [250.0, 750.0, 1250.0]
    assert all(0.0 <= x.strength <= 1.0 for x in m)
    assert m[1].kind == "beat2"


def test_recommended_gain_clamp():
    assert recommended_gain_db(-23.0, -14.0) == 9.0
    assert recommended_gain_db(-40.0, -14.0) == 12.0
    assert recommended_gain_db(-2.0, -14.0) == -12.0


def test_parse_loudnorm_json():
    text = 'garbage\n[Parsed_loudnorm_0 @ 0x1] \n{\n\t"input_i" : "-23.10",\n\t"input_tp" : "-1.20",\n\t"input_lra" : "6.50",\n\t"input_thresh" : "-33.5"\n}\n'
    d = _parse_loudnorm_json(text)
    assert d and d["input_i"] == "-23.10"
    assert _parse_loudnorm_json("nothing here") is None


def test_measure_loudness_on_synthetic(synthetic_video):
    res = measure_loudness(synthetic_video, -14.0)
    assert res is not None
    integrated, tp, lra = res
    assert -40.0 < integrated < -5.0
    assert tp <= 0.0
    pcm = ffmpeg_util.decode_pcm(synthetic_video, sr=SR)
    assert pcm.dtype == np.float32 and abs(len(pcm) - 6 * SR) < SR * 0.2
