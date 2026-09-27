"""End-to-end: run the real CLI (lite mode) on the synthetic video and validate the output
against the contract shape."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ASSET_KEYS = {"id", "path", "name", "kind", "durationMs", "width", "height", "fps", "hasAudio"}
SHOT_KEYS = {"index", "startFrame", "endFrame", "startMs", "endMs", "confidence", "method"}
CLIP_KEYS = {"id", "assetId", "trackId", "startMs", "inMs", "outMs", "speed", "transform", "color", "audio", "mask",
             "blendMode", "reframe", "freezeFrame", "reversed"}
TRACK_KEYS = {"id", "kind", "name", "locked", "muted", "clips"}
PROJECT_KEYS = {"version", "id", "name", "fps", "width", "height", "assets", "tracks", "beatMarkers"}
CLIP_ANALYSIS_KEYS = {"path", "asset", "shots", "duplicates", "reframe", "audio", "transitions"}
AUDIO_KEYS = {"integratedLufs", "truePeakDb", "recommendedGainDb", "beats", "tempoBpm"}
TRANSITION_KEYS = {"fromShot", "toShot", "flowMagnitude", "smoothness", "suggestion"}
COLOR_KEYS = {"exposure", "brilliance", "contrast", "brightness", "highlights", "shadows", "saturation", "vibrance", "sharpness",
              "temperature", "tint", "lift", "gamma", "gain", "offset", "hsl", "curves", "lutAssetId", "lutIntensity",
              "vignette", "grain"}


def _run_cli(args, cwd):
    proc = subprocess.run([sys.executable, "-m", "cappycat_pipeline", *args], cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", timeout=300)
    return proc


def test_analyze_lite_mode_end_to_end(synthetic_video, tmp_path):
    out = tmp_path / "analysis.json"
    pipeline_dir = Path(__file__).resolve().parent.parent
    proc = _run_cli(["analyze", str(synthetic_video), "--out", str(out), "--detector", "none", "--target-duration-ms", "5000"],
                    cwd=str(pipeline_dir))
    assert proc.returncode == 0, proc.stderr

    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    events = [json.loads(ln) for ln in lines]  # every stdout line must be JSON
    assert all(e["event"] in ("progress", "log", "result") for e in events)
    stages = [e["stage"] for e in events if e["event"] == "progress"]
    for s in ("ingest", "shots", "perception", "reframe", "audio", "transitions", "assemble"):
        assert s in stages
    for e in events:
        if e["event"] == "progress":
            assert set(e) == {"event", "stage", "clip", "pct", "message"} and 0.0 <= e["pct"] <= 1.0
        elif e["event"] == "log":
            assert set(e) == {"event", "level", "message"} and e["level"] in ("info", "warn", "error")
    assert not any(e["event"] == "log" and e["level"] == "error" for e in events)
    assert events[-1]["event"] == "result"
    assert Path(events[-1]["path"]) == out.resolve() or events[-1]["path"] == str(out.resolve()).replace("\\", "/")

    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == {"version", "generatedAt", "clips", "timeline"} and data["version"] == 1
    assert data["generatedAt"].endswith("Z")
    assert len(data["clips"]) == 1
    clip = data["clips"][0]
    assert set(clip) == CLIP_ANALYSIS_KEYS
    assert ASSET_KEYS <= set(clip["asset"]) and clip["asset"]["hasAudio"] is True
    assert len(clip["shots"]) == 3 and all(set(s) == SHOT_KEYS for s in clip["shots"])
    assert clip["duplicates"] == [] and clip["reframe"] == []
    assert set(clip["audio"]) - {"beatConfidence"} == AUDIO_KEYS and clip["audio"]["recommendedGainDb"] <= 12.0
    # true-peak-limited gain: the recommended gain never pushes the true peak above -1 dBTP
    assert clip["audio"]["truePeakDb"] + clip["audio"]["recommendedGainDb"] <= -1.0 + 0.01
    assert len(clip["transitions"]) == 2 and all(set(t) == TRANSITION_KEYS for t in clip["transitions"])
    assert all(t["suggestion"] in ("cut", "dissolve") for t in clip["transitions"])

    tl = data["timeline"]
    assert set(tl) == PROJECT_KEYS and tl["version"] == 1 and tl["id"].startswith("proj_")
    assert tl["width"] == 320 and tl["height"] == 180 and tl["fps"] == 24.0
    assert [t["kind"] for t in tl["tracks"]] == ["video", "fx", "audio"]
    assert all(set(t) == TRACK_KEYS for t in tl["tracks"])
    video = tl["tracks"][0]
    assert len(video["clips"]) == 3
    for c in video["clips"]:
        assert set(c) == CLIP_KEYS | {"label", "linkId"}
        assert set(c["color"]) == COLOR_KEYS
        assert c["assetId"] == tl["assets"][0]["id"] and c["trackId"] == video["id"]
        assert c["audio"]["normalize"] is True
    total = sum(c["outMs"] - c["inMs"] for c in video["clips"])
    assert total == pytest.approx(5000, abs=1)
    assert len(tl["tracks"][2]["clips"]) == 3
    # each shot's video clip and its mirrored audio clip share a linkId
    assert [c["linkId"] for c in video["clips"]] == [c["linkId"] for c in tl["tracks"][2]["clips"]]
    assert all(set(m) == {"timeMs", "strength", "kind"} for m in tl["beatMarkers"])


def test_probe_shots_doctor_commands(synthetic_video):
    pipeline_dir = Path(__file__).resolve().parent.parent
    p = _run_cli(["probe", str(synthetic_video)], cwd=str(pipeline_dir))
    assert p.returncode == 0 and json.loads(p.stdout)["width"] == 320
    s = _run_cli(["shots", str(synthetic_video), "--shot-detector", "pyscenedetect"], cwd=str(pipeline_dir))
    assert s.returncode == 0 and len(json.loads(s.stdout)) == 3
    d = _run_cli(["doctor"], cwd=str(pipeline_dir))
    assert d.returncode == 0
    rep = json.loads(d.stdout)
    assert rep["ffmpeg"] and rep["onnxruntime"] and rep["mode"] in ("lite", "ml")


def test_bad_input_reports_error_event(tmp_path):
    pipeline_dir = Path(__file__).resolve().parent.parent
    missing = tmp_path / "nope.mp4"
    p = _run_cli(["analyze", str(missing), "--out", str(tmp_path / "o.json"), "--detector", "none"], cwd=str(pipeline_dir))
    assert p.returncode == 1
    events = [json.loads(ln) for ln in p.stdout.splitlines() if ln.strip()]
    assert any(e["event"] == "log" and e["level"] == "error" for e in events)
