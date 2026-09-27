"""Round-trip a hand-written AnalysisResult that matches docs/CONTRACTS.md."""
import copy
import json

from cappycat_pipeline import schema
from cappycat_pipeline.schema import (analysis_result_from_json, default_color_grade, default_speed, default_transform,
                                      new_id, to_json)


def _keyframed(static):
    return {"static": static, "keyframes": []}


COLOR = {
    "exposure": 0, "brilliance": 0, "contrast": 0, "brightness": 0, "highlights": 0, "shadows": 0, "saturation": 0, "vibrance": 0,
    "sharpness": 0, "temperature": 0, "tint": 0, "lift": [0, 0, 0], "gamma": [0, 0, 0], "gain": [0, 0, 0],
    "offset": [0, 0, 0],
    "hsl": {c: {"h": 0, "s": 0, "l": 0} for c in ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta")},
    "curves": {"master": [[0, 0], [1, 1]], "r": [[0, 0], [1, 1]], "g": [[0, 0], [1, 1]], "b": [[0, 0], [1, 1]]},
    "lutAssetId": None, "lutIntensity": 1.0, "vignette": 0, "grain": 0,
}

ASSET = {
    "id": "ast_1", "path": "C:/abs/Clip_01.mp4", "name": "Clip_01.mp4", "kind": "video", "durationMs": 50000,
    "width": 1920, "height": 1080, "fps": 24, "hasAudio": True, "codec": "h264", "sceneTags": ["kitchen", "raccoon"],
    "order": 0,
}

REFRAME = {
    "sourceWidth": 1920, "sourceHeight": 1080,
    "keyframes": [{"frame": 0, "timeMs": 0, "crop": [120, 0, 1800, 945], "zoom": 1.1429, "tx": -0.0, "ty": -0.125}],
    "reason": "duplicate raccoon excluded on right boundary",
}

CLIP = {
    "id": "clp_1", "assetId": "ast_1", "trackId": "trk_1", "startMs": 0, "inMs": 0, "outMs": 5000,
    "speed": {"preset": "normal", "points": [{"t": 0.0, "speed": 1.0}, {"t": 1.0, "speed": 1.0}], "opticalFlow": True},
    "transform": {"position": _keyframed([0, 0]), "scale": _keyframed(1), "rotation": _keyframed(0),
                  "opacity": _keyframed(1), "blur": _keyframed(0)},
    "color": COLOR,
    "audio": {"gainDb": 9.0, "normalize": True, "muted": False, "voice": "voice"},
    "mask": {"shape": "rectangle", "feather": 0.1, "rect": {"static": [0, 0, 1, 1], "keyframes": [
        {"timeMs": 0, "value": [0, 0, 1, 1], "easing": "bezier", "bezier": [0.4, 0, 0.2, 1]}]}, "inverted": False},
    "blendMode": "normal", "reframe": REFRAME, "label": "Shot 1",
    "freezeFrame": {"atMs": 1200, "holdMs": 800}, "reversed": False,
}

PROJECT = {
    "version": 1, "id": "proj_1", "name": "Ep_01", "fps": 24, "width": 1920, "height": 1080,
    "assets": [ASSET],
    "tracks": [
        {"id": "trk_1", "kind": "video", "name": "Video 1", "locked": False, "muted": False, "clips": [CLIP]},
        {"id": "trk_2", "kind": "fx", "name": "FX", "locked": False, "muted": False, "clips": []},
        {"id": "trk_3", "kind": "audio", "name": "Audio 1", "locked": False, "muted": False, "clips": []},
    ],
    "beatMarkers": [{"timeMs": 1234.5, "strength": 0.8, "kind": "beat1"}, {"timeMs": 1734.5, "strength": 0.3, "kind": "beat2"}],
}

RESULT = {
    "version": 1,
    "generatedAt": "2026-09-24T20:00:00Z",
    "clips": [{
        "path": "C:/abs/Clip_01.mp4", "asset": ASSET,
        "shots": [{"index": 0, "startFrame": 0, "endFrame": 119, "startMs": 0, "endMs": 4958.3, "confidence": 0.93,
                   "method": "transnetv2"}],
        "duplicates": [{"shotIndex": 0, "frame": 40, "timeMs": 1666,
                        "primary": {"trackId": 1, "label": "raccoon in hoodie", "bbox": [100, 100, 500, 900], "score": 0.9},
                        "duplicate": {"trackId": 2, "label": "raccoon in hoodie", "bbox": [1500, 100, 1900, 900], "score": 0.8},
                        "similarity": 0.94}],
        "reframe": [{"shotIndex": 0, "track": REFRAME}],
        "audio": {"integratedLufs": -23.1, "truePeakDb": -1.2, "recommendedGainDb": 9.0, "beats": [500, 1000], "tempoBpm": 120},
        "transitions": [{"fromShot": 0, "toShot": 1, "flowMagnitude": 3.2, "smoothness": 0.71, "suggestion": "cut"}],
    }],
    "timeline": PROJECT,
}


def test_round_trip_matches_contract():
    src = copy.deepcopy(RESULT)
    obj = analysis_result_from_json(src)
    out = to_json(obj)
    assert json.loads(json.dumps(out)) == json.loads(json.dumps(RESULT))


def test_character_fields_round_trip_and_are_optional():
    src = copy.deepcopy(RESULT)
    src["clips"][0]["shots"][0]["cast"] = ["bunny", "turtle"]
    src["clips"][0]["duplicates"][0]["character"] = "bunny"
    src["clips"][0]["duplicates"][0]["characterName"] = "Bunny"
    out = to_json(analysis_result_from_json(copy.deepcopy(src)))
    assert json.loads(json.dumps(out)) == json.loads(json.dumps(src))
    # absent -> omitted (older consumers see the exact previous shape)
    plain = to_json(analysis_result_from_json(copy.deepcopy(RESULT)))
    assert "cast" not in plain["clips"][0]["shots"][0]
    assert "character" not in plain["clips"][0]["duplicates"][0] and "characterName" not in plain["clips"][0]["duplicates"][0]


def test_voice_mode_and_stems_round_trip_and_default():
    src = copy.deepcopy(RESULT)
    stems = {"vocals": "C:/cache/stems/abc/vocals.wav", "background": "C:/cache/stems/abc/background.wav"}
    src["timeline"]["assets"][0]["stems"] = stems
    src["timeline"]["tracks"][0]["clips"][0]["audio"]["voice"] = "background"
    out = to_json(analysis_result_from_json(copy.deepcopy(src)))
    assert json.loads(json.dumps(out)) == json.loads(json.dumps(src))
    assert out["timeline"]["assets"][0]["stems"] == stems
    # absent -> "original" (always written) / stems omitted
    old = copy.deepcopy(RESULT)
    del old["timeline"]["tracks"][0]["clips"][0]["audio"]["voice"]
    plain = to_json(analysis_result_from_json(old))
    assert plain["timeline"]["tracks"][0]["clips"][0]["audio"]["voice"] == "original"
    assert "stems" not in plain["timeline"]["assets"][0]
    assert to_json(schema.new_clip("a", "t", 0, 0, 10))["audio"] == {"gainDb": 0.0, "normalize": True, "muted": False,
                                                                     "voice": "original"}


def test_key_order_and_optional_fields():
    out = to_json(analysis_result_from_json(copy.deepcopy(RESULT)))
    assert list(out)[:2] == ["version", "generatedAt"]
    assert list(out["timeline"])[0] == "version"
    # optional fields omitted when None
    a = schema.Asset(id="ast_x", path="p", name="n", kind="video", durationMs=1, width=1, height=1, fps=1, hasAudio=False)
    assert "codec" not in to_json(a) and "order" not in to_json(a)
    kf = schema.Keyframe(timeMs=0, value=1)
    assert "bezier" not in to_json(kf)
    clip = schema.new_clip("a", "t", 0, 0, 10)
    j = to_json(clip)
    assert j["mask"] is None and j["reframe"] is None and j["freezeFrame"] is None and "label" not in j


def test_defaults_match_contract_shapes():
    cg = to_json(default_color_grade())
    assert set(cg) == set(COLOR)
    assert set(cg["hsl"]) == set(COLOR["hsl"])
    assert cg["curves"]["master"] == [[0.0, 0.0], [1.0, 1.0]]
    tr = to_json(default_transform())
    assert set(tr) == {"position", "scale", "rotation", "opacity", "blur"}
    assert tr["position"] == {"static": [0.0, 0.0], "keyframes": []}
    sp = to_json(default_speed())
    assert sp["preset"] == "normal" and sp["opticalFlow"] is True and len(sp["points"]) == 2


def test_new_id_prefixes():
    for p in ("ast", "trk_", "clp", "proj"):
        i = new_id(p)
        assert i.startswith(p.rstrip("_") + "_") and len(i) > len(p) + 6
    assert new_id("clp") != new_id("clp")


def test_numpy_values_serialise():
    import numpy as np

    kf = schema.ReframeKeyframe(frame=np.int64(3), timeMs=np.float32(125.0), crop=np.array([0, 0, 10, 5]), zoom=np.float64(2),
                                tx=0.0, ty=-0.0)
    j = to_json(kf)
    assert j["frame"] == 3 and j["crop"] == [0, 0, 10, 5] and isinstance(j["zoom"], float)
    json.dumps(j)


def test_link_id_round_trip_and_optional():
    src = copy.deepcopy(RESULT)
    src["timeline"]["tracks"][0]["clips"][0]["linkId"] = "lnk_0123456789ab"
    src["clips"][0]["audio"]["beatConfidence"] = 0.83
    out = to_json(analysis_result_from_json(copy.deepcopy(src)))
    assert json.loads(json.dumps(out)) == json.loads(json.dumps(src))
    assert out["timeline"]["tracks"][0]["clips"][0]["linkId"] == "lnk_0123456789ab"
    # absent -> omitted (older consumers see the previous shape) and parsed as None
    plain = analysis_result_from_json(copy.deepcopy(RESULT))
    assert plain.timeline.tracks[0].clips[0].linkId is None
    j = to_json(plain)
    assert "linkId" not in j["timeline"]["tracks"][0]["clips"][0]
    assert "beatConfidence" not in j["clips"][0]["audio"]
    assert "linkId" not in to_json(schema.new_clip("a", "t", 0, 0, 10))


# --------------------------------------------------------------------------- feature set v2 (docs/FEATURES_V2.md)


def _v2_result():
    src = copy.deepcopy(RESULT)
    tl = src["timeline"]
    tl["frameInterpolation"] = "frameBlend"
    tl["universalAdjust"] = {"enabled": True, "name": "Warm", "values": {"temperature": 8, "saturation": 5}}
    tl["assets"].append({"id": "ast_v", "path": "C:/cache/stems/abc/vocals.wav", "name": "Clip_01.mp4 · Voice",
                         "kind": "audio", "durationMs": 50000, "width": 0, "height": 0, "fps": 0, "hasAudio": True,
                         "stemOf": {"assetId": "ast_1", "stem": "vocals"}})
    clip = tl["tracks"][0]["clips"][0]
    clip["audio"].update({"keepPitch": False, "fadeInMs": 250.0, "fadeOutMs": 400.0,
                          "volume": {"static": 0, "keyframes": [
                              {"timeMs": 0, "value": 0, "easing": "linear"},
                              {"timeMs": 1000, "value": -6.5, "easing": "easeInOut"}]}})
    clip["transitionIn"] = {"type": "dipToBlack", "durationMs": 800}
    clip["fadeInMs"] = 120.0
    clip["fadeOutMs"] = 300.0
    fx = copy.deepcopy(CLIP)
    fx.update({"id": "clp_fx", "assetId": "", "trackId": "trk_2", "reframe": None, "mask": None, "label": "Shake",
               "freezeFrame": None, "effect": {"type": "shake", "intensity": 0.7, "params": {"amplitude": 0.02, "frequency": 9}}})
    fx2 = copy.deepcopy(fx)
    fx2.update({"id": "clp_fx2", "effect": {"type": "cameraSnap", "intensity": 1.0}})
    tl["tracks"][1]["clips"] = [fx, fx2]
    tl["tracks"][2]["role"] = "voice"
    return src


def test_v2_fields_round_trip():
    src = _v2_result()
    obj = analysis_result_from_json(copy.deepcopy(src))
    out = to_json(obj)
    assert json.loads(json.dumps(out)) == json.loads(json.dumps(src))
    tl = obj.timeline
    assert tl.frameInterpolation == "frameBlend" and tl.universalAdjust["name"] == "Warm"
    assert tl.assets[1].stemOf == schema.AssetStemOf("ast_1", "vocals")
    c = tl.tracks[0].clips[0]
    assert c.audio.keepPitch is False and c.audio.fadeInMs == 250 and c.audio.volume.keyframes[1].value == -6.5
    assert c.transitionIn == schema.ClipTransition("dipToBlack", 800) and (c.fadeInMs, c.fadeOutMs) == (120, 300)
    e = tl.tracks[1].clips[0].effect
    assert e.type == "shake" and e.intensity == 0.7 and e.params == {"amplitude": 0.02, "frequency": 9}
    assert tl.tracks[1].clips[1].effect.params is None and "params" not in out["timeline"]["tracks"][1]["clips"][1]["effect"]
    assert tl.tracks[2].role == "voice" and tl.tracks[0].role is None


def test_v2_fields_absent_are_omitted_and_default():
    obj = analysis_result_from_json(copy.deepcopy(RESULT))
    out = to_json(obj)
    assert json.loads(json.dumps(out)) == json.loads(json.dumps(RESULT))  # old projects: exact previous shape
    tl = out["timeline"]
    assert "frameInterpolation" not in tl and "universalAdjust" not in tl
    assert "stemOf" not in tl["assets"][0] and all("role" not in t for t in tl["tracks"])
    c = tl["tracks"][0]["clips"][0]
    for k in ("transitionIn", "effect", "fadeInMs", "fadeOutMs"):
        assert k not in c
    for k in ("keepPitch", "fadeInMs", "fadeOutMs", "volume"):
        assert k not in c["audio"]
    # spec defaults
    assert schema.effective_frame_interpolation(obj.timeline) == "opticalFlow"
    assert schema.effective_keep_pitch(obj.timeline.tracks[0].clips[0].audio) is True
    assert schema.effective_keep_pitch(schema.ClipAudio(keepPitch=False)) is False
    assert schema.DEFAULTS_V2["ClipEffect.intensity"] == 1.0 and schema.DEFAULTS_V2["ClipTransition.durationMs"] == 500
    new = to_json(schema.new_clip("a", "t", 0, 0, 10))
    assert not {"transitionIn", "effect", "fadeInMs", "fadeOutMs"} & set(new)
    # explicit null transition (allowed by the TS type) reads as "no transition"
    src = copy.deepcopy(RESULT)
    src["timeline"]["tracks"][0]["clips"][0]["transitionIn"] = None
    assert analysis_result_from_json(src).timeline.tracks[0].clips[0].transitionIn is None
    assert schema.ClipEffect("sepia").intensity == 1.0 and schema.ClipTransition().durationMs == 500.0


def test_v2_enums_match_typescript():
    """The enum tuples mirror the unions in src/types/project.ts (skipped when the file is absent)."""
    import re
    from pathlib import Path

    ts = Path(__file__).resolve().parents[2] / "src" / "types" / "project.ts"
    if not ts.is_file():
        import pytest

        pytest.skip("src/types/project.ts not found")
    text = ts.read_text(encoding="utf-8")

    def union(name):
        m = re.search(r"export type " + name + r"\s*=([^;]+);", text)
        assert m, name
        return tuple(re.findall(r"'([^']+)'", m.group(1)))

    assert union("TransitionType") == schema.TRANSITION_TYPES
    assert union("EffectType") == schema.EFFECT_TYPES
    assert union("TrackRole") == schema.TRACK_ROLES
    assert union("StemKind") == schema.STEM_KINDS
    if re.search(r"export type FrameInterpolation\s*=", text):
        assert set(union("FrameInterpolation")) == set(schema.FRAME_INTERPOLATION_MODES)


def test_assemble_keeps_v2_fields_of_read_projects(tmp_path):
    """A project read from JSON (e.g. a cached / re-loaded analysis result) keeps its v2 fields when
    written again, and a freshly assembled timeline has none of them (they stay None / omitted)."""
    from cappycat_pipeline import assemble, fsutil

    src = _v2_result()
    p = tmp_path / "r.json"
    fsutil.write_json_atomic(p, to_json(analysis_result_from_json(copy.deepcopy(src))))
    again = to_json(analysis_result_from_json(json.loads(p.read_text(encoding="utf-8"))))
    assert json.loads(json.dumps(again)) == json.loads(json.dumps(src))
    res = analysis_result_from_json(copy.deepcopy(RESULT))
    proj = assemble.assemble_timeline(res.clips, None, 24, 1920, 1080, name="t")
    j = to_json(proj)
    assert "frameInterpolation" not in j
    for t in j["tracks"]:
        for c in t["clips"]:
            assert "transitionIn" not in c and "effect" not in c and "keepPitch" not in c["audio"]
