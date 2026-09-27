import pytest

from cappycat_pipeline.assemble import assemble_timeline, fit_durations, natural_key, order_analyses, timeline_duration_ms
from cappycat_pipeline.schema import (Asset, AudioAnalysis, BeatMarker, ClipAnalysis, ReframeKeyframe, ReframeTrack, Shot,
                                      ShotReframe, to_json)


def _asset(name, order=None, dur=6000.0, has_audio=True):
    return Asset(id="ast_" + name, path="C:/x/" + name, name=name, kind="video", durationMs=dur, width=320, height=180,
                 fps=24.0, hasAudio=has_audio, codec="h264", sceneTags=[], order=order)


def _shots(n, frames_each=48, fps=24.0):
    out = []
    for i in range(n):
        s, e = i * frames_each, (i + 1) * frames_each - 1
        out.append(Shot(i, s, e, s / fps * 1000, e / fps * 1000, 1.0, "pyscenedetect"))
    return out


def test_natural_sort_and_order():
    assert natural_key("clip10.mp4") > natural_key("clip2.mp4")
    a = ClipAnalysis(path="C:/x/clip10.mp4", asset=_asset("clip10.mp4"))
    b = ClipAnalysis(path="C:/x/clip2.mp4", asset=_asset("clip2.mp4"))
    c = ClipAnalysis(path="C:/x/zzz.mp4", asset=_asset("zzz.mp4", order=0))
    assert [x.asset.name for x in order_analyses([a, b, c])] == ["zzz.mp4", "clip2.mp4", "clip10.mp4"]


def test_fit_durations_water_fills_longest_first():
    assert fit_durations([2000, 5000, 9000], None) == [2000, 5000, 9000]
    out = fit_durations([2000, 5000, 9000], 12000)
    assert sum(out) == pytest.approx(12000, abs=1)
    assert out[0] == 2000 and out[1] == pytest.approx(5000) and out[2] == pytest.approx(5000)
    out = fit_durations([2000, 5000, 9000], 8000)
    assert sum(out) == pytest.approx(8000, abs=1) and out[0] == 2000 and out[1] == pytest.approx(3000) and out[2] == pytest.approx(3000)
    # never below 1500 ms; overflow is dropped (the later entry on ties) and the rest re-fitted, so
    # the result fills the target instead of falling short of it
    out = fit_durations([3000, 3000, 3000, 3000], 4000)
    assert out == [2000, 2000, 0, 0]
    assert fit_durations([800, 800], 1000) == [800, 0]
    # with values, the lowest-value entries go first (the first entry is always kept)
    out = fit_durations([3000, 3000, 3000, 3000], 4000, values=[0, 2, 0, 1])
    assert out[0] == pytest.approx(2000) and out[1] == pytest.approx(2000) and out[2] == 0 and out[3] == 0


def test_assemble_timeline_structure_and_trimming():
    a = ClipAnalysis(path="C:/x/b.mp4", asset=_asset("b.mp4"), shots=_shots(3),
                     audio=AudioAnalysis(-20.0, -1.0, 6.0, beats=[100.0, 2100.0, 4100.0], tempoBpm=120))
    track = ReframeTrack(320, 180, [ReframeKeyframe(0, 0, [0, 0, 160, 90], 2.0, -0.5, -0.5)], "test")
    a.reframe.append(ShotReframe(1, track))
    b = ClipAnalysis(path="C:/x/a.mp4", asset=_asset("a.mp4", has_audio=False), shots=_shots(2))
    beat_details = {"C:/x/b.mp4": [BeatMarker(100.0, 0.9, "beat1"), BeatMarker(2100.0, 0.4, "beat2"), BeatMarker(4100.0, 0.7, "beat1")]}
    proj = assemble_timeline([a, b], target_duration_ms=None, fps=24, width=320, height=180, beat_details=beat_details)
    j = to_json(proj)
    assert j["version"] == 1 and j["id"].startswith("proj_") and j["fps"] == 24
    assert [t["name"] for t in j["tracks"]] == ["Video 1", "FX", "Audio 1"]
    assert [t["kind"] for t in j["tracks"]] == ["video", "fx", "audio"]
    video, fx, audio = j["tracks"]
    assert len(video["clips"]) == 5 and fx["clips"] == [] and len(audio["clips"]) == 3
    # a.mp4 sorts first (natural name order), sequential starts
    assert [c["label"] for c in video["clips"]][:2] == ["a.mp4 \u00b7 Shot 1", "a.mp4 \u00b7 Shot 2"]
    starts = [c["startMs"] for c in video["clips"]]
    assert starts == sorted(starts) and starts[0] == 0
    for prev, nxt in zip(video["clips"], video["clips"][1:]):
        assert nxt["startMs"] == pytest.approx(prev["startMs"] + prev["outMs"] - prev["inMs"])
    # the b.mp4 clips carry gain and the reframe on shot 2
    b_clips = [c for c in video["clips"] if c["label"].startswith("b.mp4")]
    assert all(c["audio"] == {"gainDb": 6.0, "normalize": True, "muted": False, "voice": "original"} for c in b_clips)
    assert b_clips[1]["reframe"]["keyframes"][0]["zoom"] == 2.0 and b_clips[0]["reframe"] is None
    assert all(c["trackId"] == audio["id"] and c["assetId"] == "ast_b.mp4" for c in audio["clips"])
    assert [ast["order"] for ast in j["assets"]] == [0, 1]
    # beat markers land on the timeline where their source time falls inside the b.mp4 clips (the
    # clips are a few ms shorter than their shots: cut points sit safely on each side of a cut)
    expected = [round(c["startMs"] + beat - c["inMs"], 3) for beat in (100.0, 2100.0, 4100.0)
                for c in b_clips if c["inMs"] <= beat < c["outMs"]]
    assert [m["timeMs"] for m in j["beatMarkers"]] == expected and len(expected) == 3
    assert [m["kind"] for m in j["beatMarkers"]] == ["beat1", "beat2", "beat1"]
    assert timeline_duration_ms(proj) == pytest.approx(10000, abs=4 * 45)
    # first shot starts at 0, last one ends at its end; consecutive shots lose < 1 frame at the cut
    assert b_clips[0]["inMs"] == 0 and b_clips[-1]["outMs"] == pytest.approx(6000, abs=1)
    for k, (c, n) in enumerate(zip(b_clips, b_clips[1:]), start=1):
        cut = 2000.0 * k
        assert c["outMs"] < cut <= n["inMs"] and n["inMs"] - c["outMs"] < 1000.0 / 24

    # trimming to a target duration
    proj2 = assemble_timeline([a, b], target_duration_ms=8000, fps=24, width=320, height=180)
    assert timeline_duration_ms(proj2) == pytest.approx(8000, abs=1)
    assert all(c.outMs - c.inMs >= 1500 for t in proj2.tracks for c in t.clips)


def test_snap_fps_to_standard_rates():
    from cappycat_pipeline.assemble import snap_fps

    assert snap_fps(24.04) == 24.0
    assert snap_fps(23.98) == 23.976 and snap_fps(29.97) == 29.97 and snap_fps(59.95) == 59.94
    assert snap_fps(25.0) == 25.0 and snap_fps(60.0) == 60.0
    assert snap_fps(27.0) == 27.0  # not near a standard rate: unchanged
    assert snap_fps(24.3) == 24.3  # 1.25 % off: unchanged


def test_linked_video_and_audio_clips_share_a_link_id():
    a = ClipAnalysis(path="C:/x/b.mp4", asset=_asset("b.mp4"), shots=_shots(3))
    b = ClipAnalysis(path="C:/x/a.mp4", asset=_asset("a.mp4", has_audio=False), shots=_shots(2))
    j = to_json(assemble_timeline([a, b], None, 24, 320, 180))
    video, _, audio = j["tracks"]
    links = {c["linkId"] for c in audio["clips"]}
    assert len(links) == 3 and all(lid.startswith("lnk_") for lid in links)
    by_link = {c["linkId"]: c for c in video["clips"] if "linkId" in c}
    assert set(by_link) == links
    for ac in audio["clips"]:
        vc = by_link[ac["linkId"]]
        assert (vc["startMs"], vc["inMs"], vc["outMs"], vc["assetId"]) == (ac["startMs"], ac["inMs"], ac["outMs"], ac["assetId"])
    # a.mp4 has no audio: its video clips are not linked (field omitted)
    assert all("linkId" not in c for c in video["clips"] if c["label"].startswith("a.mp4"))


def test_trimming_keeps_the_middle_and_snaps_to_beats():
    from cappycat_pipeline.assemble import trim_window

    assert trim_window(0.0, 4000.0, 2000.0) == (1000.0, 3000.0)  # both ends trimmed
    assert trim_window(0.0, 4000.0, 4000.0) == (0.0, 4000.0)
    assert trim_window(0.0, 4000.0, 2000.0, beats=[500.0, 2700.0, 3900.0]) == (700.0, 2700.0)  # out point on a beat
    assert trim_window(0.0, 4000.0, 2000.0, beats=[500.0]) == (1000.0, 3000.0)  # no beat fits: centred
    shots = [Shot(0, 0, 95, 0.0, 95 / 24 * 1000, 1.0, "transnetv2")]  # one 4 s shot
    ca = ClipAnalysis(path="C:/x/c.mp4", asset=_asset("c.mp4", dur=4000.0), shots=shots)
    proj = assemble_timeline([ca], 2000, 24, 320, 180)
    v = proj.tracks[0].clips[0]
    assert v.outMs - v.inMs == pytest.approx(2000) and v.inMs == pytest.approx(1000)


def test_lowest_value_shots_are_dropped_not_the_last_ones():
    # 4 x 3 s shots, 4 s target: floors (1.5 s) do not fit; the shots without cast go first
    shots = _shots(4, frames_each=72)
    for s, cast in zip(shots, (["bunny"], [], ["bunny", "turtle"], [])):
        s.cast = cast
    ca = ClipAnalysis(path="C:/x/d.mp4", asset=_asset("d.mp4", dur=12000.0), shots=shots)
    proj = assemble_timeline([ca], 4000, 24, 320, 180)
    labels = [c.label for c in proj.tracks[0].clips]
    assert labels == ["d.mp4 \u00b7 Shot 1", "d.mp4 \u00b7 Shot 3"]
    assert timeline_duration_ms(proj) == pytest.approx(4000, abs=1)


def test_cut_points_are_safe_on_every_export_grid():
    """For any cut time (incl. exact rounding ties on 1/60 and 1/90000 timebases), the outgoing clip
    never samples the new frame and the incoming one never samples an old frame, on every grid the
    exporter may use (with and without its 1 ms seek lead), and the preview agrees."""
    import math
    import random

    from cappycat_pipeline.assemble import EXPORT_GRIDS, SEEK_LEAD_MS, cut_points

    rng = random.Random(7)
    times = [k * 1000.0 / 60.0 for k in range(1, 1200)] + [rng.randrange(1, 1_350_000) * 1000.0 / 90000.0 for _ in range(3000)]
    for p in times:
        p3 = round(p, 3)  # shots carry 3 decimals
        out_ms, in_ms = cut_points(p3)
        assert out_ms < p <= in_ms and in_ms - p < 25.0 and p - out_ms < 25.0
        for g in EXPORT_GRIDS:
            for lead in (0.0, SEEK_LEAD_MS):
                for bias in (-1e-6, 1e-6):
                    tick_of_new = math.floor((p + lead) * g / 1000.0 + 0.5 + bias)
                    assert math.floor(in_ms * g / 1000.0 + 1e-3) >= tick_of_new, (p, g, lead)
                    assert math.floor((out_ms - 0.01) * g / 1000.0 + 1e-3) < tick_of_new, (p, g, lead)
