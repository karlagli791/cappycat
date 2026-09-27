"""Behaviours added by the pipeline review fixes (no ML models needed): hybrid escalation rule,
OOM fallback, duplicate evidence, group boxes, track-level cast tagging, detection cache,
open-set rejection, reference dedupe, audio gain / beat gating, atomic writes, the stdin
watchdog and the per-clip result cache."""
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from cappycat_pipeline import audio, fsutil, perception
from cappycat_pipeline.characters import (NEG_ID, Character, CharacterBank, Identity, Manifest, crop_group,
                                          dedupe_references)
from cappycat_pipeline.perception import (DetectionCache, HybridDetector, ShotStats, analyze_shot, find_duplicates,
                                          shot_cast, tracker_for)
from cappycat_pipeline.schema import Shot
from cappycat_pipeline.tracking import Detection

from test_perception_logic import LEFT, RIGHT, SHOT, StubDetector, StubEmbedder, frames

W, H = 640, 360


def _bank(threshold=0.7, texts=None):
    chars = [Character("bunny", "Bunny", "rabbit"), Character("turtle", "Turtle", "turtle")]
    m = Manifest(Path("characters.json"), chars, {}, ["animal character"])
    refs = np.array([[1, 0, 0, 0], [0.95, 0.3, 0, 0], [0, 1, 0, 0], [0.3, 0.95, 0, 0]], np.float32)
    refs /= np.linalg.norm(refs, axis=1, keepdims=True)
    kw = {}
    if texts is not None:
        t = np.array([v for v, _ in texts], np.float32)
        kw = dict(text_embeddings=t / np.linalg.norm(t, axis=1, keepdims=True), text_owner=[o for _, o in texts],
                  texts=[f"t{i}" for i in range(len(texts))])
    return CharacterBank(m, ["bunny", "bunny", "turtle", "turtle"], refs, ["a#0", "a#1", "b#0", "b#1"], threshold, 0.05, **kw)


# --------------------------------------------------------------------------- escalation rule (P0-1)


def _hybrid(boxes):
    return HybridDetector(StubDetector(boxes, "yolo_world"), fallback_factory=lambda: StubDetector(boxes, "grounded_sam2"))


def test_overlap_of_two_different_cast_members_does_not_escalate():
    overlap = [[((60, 40, 260, 340), "person", 0.6), ((120, 40, 320, 340), "person", 0.6)]]
    # left box = Bunny, right box = Turtle, both confidently identified
    emb = StubEmbedder([((0, 190), [1, 0.05, 0, 0]), ((190, 640), [0.05, 1, 0, 0])])
    res = analyze_shot(frames(4), SHOT, _hybrid(overlap), emb, 0.85, ["person"], 24.0, 1.0, 4.0, bank=_bank())
    assert res.path == "yolo_world", res.reason
    assert res.stats.max_same_label_iou > 0.3 and res.stats.max_ambiguous_iou == 0.0


def test_overlap_of_possibly_same_character_still_escalates():
    overlap = [[((60, 40, 260, 340), "person", 0.6), ((120, 40, 320, 340), "person", 0.6)]]
    same = StubEmbedder([((0, 640), [1, 0.05, 0, 0])])  # both Bunny
    res = analyze_shot(frames(4), SHOT, _hybrid(overlap), same, 0.85, ["person"], 24.0, 1.0, 4.0, bank=_bank())
    assert res.path == "grounded_sam2" and "overlap" in res.reason
    unknown = StubEmbedder([((0, 190), [0, 0, 1, 0]), ((190, 640), [0, 0, 0.9, 0.4])])  # neither identified
    res = analyze_shot(frames(4), SHOT, _hybrid(overlap), unknown, 0.85, ["person"], 24.0, 1.0, 4.0, bank=_bank())
    assert res.path == "grounded_sam2"


def test_head_inside_body_does_not_escalate():
    body, head = (100, 40, 260, 340), (130, 40, 230, 130)
    boxes = [[(body, "person", 0.8), (head, "animal character", 0.5)]]
    res = analyze_shot(frames(4), SHOT, _hybrid(boxes), StubEmbedder([((0, 640), [0, 0, 1, 0])]), 0.85, ["x"], 24.0,
                       1.0, 4.0)
    assert res.path == "yolo_world" and res.stats.max_ambiguous_iou == 0.0


# --------------------------------------------------------------------------- OOM fallback (P0-3)


class _Boom:
    name = "grounded_sam2"
    box_threshold = 0.35

    def __init__(self):
        self.calls = 0

    def detect(self, frame, prompts):
        self.calls += 1
        torch = pytest.importorskip("torch")
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    def close(self):
        pass


def test_oom_during_escalation_keeps_yolo_result_and_disables_escalation():
    boom = _Boom()
    primary = StubDetector([[]], name="yolo_world")  # finds nothing -> escalation
    hyb = HybridDetector(primary, fallback_factory=lambda: boom)
    emb = StubEmbedder([((0, 640), [1, 0, 0, 0])])
    res = analyze_shot(frames(4), SHOT, hyb, emb, 0.85, ["person"], 24.0, 1.0, 4.0)
    assert res.path == "yolo_world" and "escalation failed" in res.reason and "out of memory" in res.reason.lower()
    assert res.warning and "disabled escalation" in res.warning
    assert not hyb.fallback_possible and boom.calls == 1
    # the next shot does not try again
    res2 = analyze_shot(frames(4), SHOT, hyb, emb, 0.85, ["person"], 24.0, 1.0, 4.0)
    assert res2.path == "yolo_world" and "escalation skipped" in res2.reason and boom.calls == 1


def test_any_error_during_escalation_is_contained():
    class Err(StubDetector):
        def detect(self, frame, prompts):
            raise ValueError("bad frame")

    hyb = HybridDetector(StubDetector([[]], name="yolo_world"), fallback_factory=lambda: Err([[]], "grounded_sam2"))
    res = analyze_shot(frames(2), SHOT, hyb, StubEmbedder([((0, 640), [1, 0, 0, 0])]), 0.85, ["p"], 24.0, 1.0, 4.0)
    assert res.path == "yolo_world" and "ValueError" in res.reason and not hyb.fallback_possible


# --------------------------------------------------------------------------- evidence / identity (P1-8, P1-6, P1-7)


def test_single_frame_duplicate_is_not_reported():
    pair = [(LEFT, "person", 0.6), (RIGHT, "person", 0.5)]
    boxes = [pair, [(LEFT, "person", 0.6)], [(LEFT, "person", 0.6)], [(LEFT, "person", 0.6)]]
    det = StubDetector(boxes)
    emb = StubEmbedder([((0, 640), [1, 0.1, 0, 0])])
    st = ShotStats()
    f = find_duplicates(frames(4)(), SHOT, det, tracker_for(det, 4.0), emb, 0.85, ["person"], 24.0, bank=_bank(), stats=st)
    assert f == [] and st.dropped_single_frame == 1
    det = StubDetector([pair, pair, [(LEFT, "person", 0.6)], [(LEFT, "person", 0.6)]])
    assert find_duplicates(frames(4)(), SHOT, det, tracker_for(det, 4.0), emb, 0.85, ["person"], 24.0, bank=_bank())


def test_group_box_gets_no_identity_or_cast():
    group = (20, 0, 630, 360)  # > 55 % of the frame
    det = StubDetector([[(group, "person", 0.3)]] * 4)
    st = ShotStats()
    find_duplicates(frames(4)(), SHOT, det, tracker_for(det, 4.0), StubEmbedder([((0, 640), [1, 0.1, 0, 0])]), 0.85,
                    ["person"], 24.0, bank=_bank(), stats=st)
    assert st.cast_frames == {} and st.rejected == 4 and shot_cast(st) == []
    assert perception.is_group_box((96, 2, 1278, 720), 1280, 720)  # the clip9 deer / turtle / elephant box
    assert not perception.is_group_box((716, 106, 1082, 720), 1280, 720)  # a full-height single character


def test_track_level_cast_tagging_with_the_looser_rule():
    # a character scoring 0.68 (under the 0.70 id threshold) with a clear margin, 4 frames on one track
    det = StubDetector([[(LEFT, "person", 0.6)]] * 4)
    emb = StubEmbedder([((0, 640), [0.62, 1, 0, 0])])
    bank = _bank(threshold=0.99)
    st = ShotStats()
    find_duplicates(frames(4)(), SHOT, det, tracker_for(det, 4.0), emb, 0.85, ["person"], 24.0, bank=bank, stats=st)
    assert st.cast_frames == {}  # never confidently identified
    assert st.loose_cast() == ["turtle"] and shot_cast(st) == ["turtle"]
    # two frames only: not enough
    st2 = ShotStats()
    det = StubDetector([[(LEFT, "person", 0.6)]] * 2)
    find_duplicates(frames(2)(), SHOT, det, tracker_for(det, 4.0), emb, 0.85, ["person"], 24.0, bank=bank, stats=st2)
    assert shot_cast(st2) == []


def test_identity_loose_rule_and_zero_shot_agreement():
    assert Identity(None, 0.70, 0.06, "bunny").loose() == "bunny"
    assert Identity(None, 0.70, 0.02, "bunny").loose() is None
    assert Identity(None, 0.70, 0.02, "bunny", zs_best="bunny", zs_prob=0.9).loose() == "bunny"  # text tower agrees
    assert Identity(None, 0.64, 0.2, "bunny").loose() is None
    assert Identity(None, 0.9, 0.2, "bunny", rejected="zero-shot").loose() is None


def test_open_set_rejection_by_negative_text():
    # text 0 "rabbit" (bunny), text 1 "turtle" (turtle), text 2 "a cartoon deer" (negative)
    texts = [([1, 0, 0, 0], "bunny"), ([0, 1, 0, 0], "turtle"), ([0.6, 0, 1, 0], NEG_ID)]
    bank = _bank(texts=texts)
    ok = bank.identify_one(np.array([1, 0.1, 0.0, 0], np.float32))
    assert ok.character == "bunny" and ok.rejected is None and ok.zs_best == "bunny"
    deer = bank.identify_one(np.array([1, 0.1, 0.9, 0], np.float32))  # looks like Bunny to the refs, "deer" in text
    assert deer.character is None and deer.rejected and "not-cast" in deer.rejected and deer.loose() is None


def test_reference_dedupe_counts_each_crop_once():
    assert crop_group("sheet.webp#3rhf") == "sheet.webp#3r" and crop_group("x.webp#12f") == "x.webp#12"
    e = np.array([[1, 0, 0], [1, 0.01, 0], [1, 0.02, 0], [0, 1, 0]], np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    ids = ["b", "b", "b", "b"]
    src = ["s#0", "s#0f", "s#1", "s#2"]  # crop 1 is a near copy of crop 0; crop 2 differs
    keep = dedupe_references(ids, src, e, cos=0.97)
    assert keep == [0, 1, 3]  # crop 0 with its flip, crop 2; crop 1 dropped


# --------------------------------------------------------------------------- detection cache (P1-4)


def test_detection_cache_reuses_only_identical_frames_at_a_low_enough_floor():
    c = DetectionCache()
    f = np.random.default_rng(0).integers(0, 255, (36, 64, 3), dtype=np.uint8)
    raw = [Detection((0, 0, 10, 10), "person", 0.3), Detection((20, 0, 30, 10), "person", 0.12)]
    c.put("grounded_sam2", 42, f, 0.1, raw)
    got = c.get("grounded_sam2", 42, f, 0.25)
    assert [d.score for d in got] == [0.3] and c.hits == 1
    assert c.get("grounded_sam2", 42, f, 0.05) is None  # stored floor 0.1 > requested 0.05
    g = f.copy()
    g[0, 0, 0] ^= 0xFF
    assert c.get("grounded_sam2", 42, g, 0.25) is None  # different pixels
    assert c.get("yolo_world", 42, f, 0.25) is None


def test_find_duplicates_fills_the_cache_with_raw_detections():
    class RawStub(StubDetector):
        def detect(self, frame, prompts):
            out = super().detect(frame, prompts)
            self.last_raw, self.last_floor = out + [Detection((300, 300, 310, 310), "person", 0.05)], 0.05
            return out

    det = RawStub([[(LEFT, "person", 0.6)]] * 3)
    cache = DetectionCache()
    find_duplicates(frames(3)(), SHOT, det, tracker_for(det, 4.0), StubEmbedder([((0, 640), [1, 0, 0, 0])]), 0.85,
                    ["p"], 24.0, cache=cache)
    assert len(cache) == 3
    assert len(cache.get("stub", 6, np.zeros((H, W, 3), np.uint8), 0.05)) == 2


# --------------------------------------------------------------------------- audio (P2-11, P2-12, P3-25)


def test_gain_is_limited_by_the_true_peak():
    assert audio.recommended_gain_db(-23.0, -14.0, true_peak_db=-3.0) == pytest.approx(2.0)  # -1 dBTP ceiling
    assert audio.recommended_gain_db(-23.0, -14.0, true_peak_db=-15.0) == pytest.approx(9.0)  # loudness-limited
    assert audio.recommended_gain_db(-10.0, -14.0, true_peak_db=0.5) == pytest.approx(-4.0)
    assert audio.recommended_gain_db(-12.0, -14.0, true_peak_db=1.5) == pytest.approx(-2.5)  # already clipping
    assert audio.recommended_gain_db(-23.0, -14.0) == 9.0  # no peak known: unchanged behaviour


def test_ebur128_summary_parsing():
    text = textwrap.dedent("""\
        [Parsed_ebur128_0 @ 0x1] t: 1.2 TARGET:-23 LUFS M: -20.1 S:-120.7 I: -19.9 LUFS LRA: 0.0 LU
        [Parsed_ebur128_0 @ 0x1] Summary:

          Integrated loudness:
            I:         -19.6 LUFS
            Threshold: -30.0 LUFS

          Loudness range:
            LRA:         6.4 LU
            Threshold: -40.1 LUFS

          True peak:
            Peak:       -0.4 dBFS
        """)
    assert audio._parse_ebur128(text) == (-19.6, -0.4, 6.4)
    assert audio._parse_ebur128("no summary here") is None


def test_beats_are_gated_on_dialogue_like_audio_and_carry_a_confidence():
    sr = 22050
    rng = np.random.default_rng(3)
    # speech-like: noise bursts with random lengths / gaps (no pulse)
    y = np.zeros(sr * 10, np.float32)
    t = 0
    while t < len(y) - sr // 2:
        n = int(rng.integers(sr // 20, sr // 3))
        y[t:t + n] = rng.normal(0, 0.2, n) * np.hanning(n)
        t += n + int(rng.integers(sr // 30, sr // 2))
    r = audio.analyze_beats(y, sr)
    assert r.beats_ms == [] and r.confidence < 0.5 and r.reason
    click = audio.synth_click_track(bpm=120.0, seconds=10.0, sr=sr)
    r = audio.analyze_beats(click, sr)
    assert len(r.beats_ms) >= 10 and r.confidence >= 0.5 and abs(r.tempo_bpm - 120) < 4
    r = audio.analyze_beats(click, sr, use_librosa=False)
    assert len(r.beats_ms) >= 10 and 0.4 <= r.confidence <= 0.8  # numpy alone: capped


def test_tempo_agreement_accepts_octaves_only():
    assert audio.tempos_agree(120.0, 121.0) and audio.tempos_agree(146.0, 73.5) and audio.tempos_agree(60.0, 118.0)
    assert not audio.tempos_agree(146.0, 107.7) and not audio.tempos_agree(82.0, 129.0)
    assert not audio.tempos_agree(None, 120.0)


# --------------------------------------------------------------------------- atomic outputs (P2-17)


def test_atomic_path_replaces_only_on_success(tmp_path):
    dst = tmp_path / "out.json"
    fsutil.write_json_atomic(dst, {"a": 1})
    assert json.loads(dst.read_text()) == {"a": 1}
    with pytest.raises(RuntimeError):
        with fsutil.atomic_path(dst) as tmp:
            tmp.write_text("{\"a\": 2, trunc")
            raise RuntimeError("killed mid-write")
    assert json.loads(dst.read_text()) == {"a": 1}  # untouched
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]  # no .part left behind
    fsutil.save_npz_atomic(tmp_path / "x.npz", a=np.arange(3))
    assert np.load(tmp_path / "x.npz")["a"].tolist() == [0, 1, 2]


def test_unicode_paths_for_images(tmp_path):
    d = tmp_path / "ünï_日本"
    d.mkdir()
    img = np.random.default_rng(1).integers(0, 255, (20, 30, 3), dtype=np.uint8)
    assert fsutil.imwrite(d / "crop.png", img)
    back = fsutil.imread(d / "crop.png")
    assert back is not None and np.array_equal(back, img)
    assert fsutil.imread(d / "missing.png") is None


def test_interpolate_cancelled_mid_way_leaves_no_output(synthetic_video, tmp_path):
    from cappycat_pipeline import interpolate

    out = tmp_path / "slow.mp4"
    calls = []

    def progress(pct, msg):
        calls.append(pct)
        if len(calls) == 2:
            raise KeyboardInterrupt("cancelled")

    os.environ["CAPPYCAT_FLOW_BACKEND"] = "dis"
    try:
        with pytest.raises(KeyboardInterrupt):
            interpolate.interpolate_segment(str(synthetic_video), 0, 1500, 2, str(out), progress=progress)
    finally:
        os.environ.pop("CAPPYCAT_FLOW_BACKEND", None)
    assert not out.exists() and list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- stdin watchdog (P2-17)


def _alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_watchdog_kills_children_and_exits_on_stdin_eof():
    script = textwrap.dedent("""\
        import sys, time
        from cappycat_pipeline import procs
        child = procs.popen([sys.executable, "-c", "import time; time.sleep(60)"])
        print(child.pid, flush=True)
        procs.start_stdin_watchdog()
        time.sleep(60)
        """)
    p = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         cwd=str(Path(__file__).resolve().parent.parent))
    child_pid = int(p.stdout.readline())
    assert _alive(child_pid)
    t0 = time.perf_counter()
    p.stdin.close()  # the parent "dies": EOF on the pipe
    assert p.wait(timeout=15) == 3
    assert time.perf_counter() - t0 < 10
    deadline = time.time() + 5
    while _alive(child_pid) and time.time() < deadline:
        time.sleep(0.1)
    assert not _alive(child_pid)


def test_watchdog_is_opt_in(monkeypatch):
    from cappycat_pipeline import procs

    monkeypatch.delenv("CAPPYCAT_WATCH_STDIN", raising=False)
    assert not procs.watch_stdin_enabled(False) and procs.watch_stdin_enabled(True)
    monkeypatch.setenv("CAPPYCAT_WATCH_STDIN", "1")
    assert procs.watch_stdin_enabled(False)


# --------------------------------------------------------------------------- per-clip result cache (P1-10)


def _cli(args, env=None):
    proc = subprocess.run([sys.executable, "-m", "cappycat_pipeline", *args], cwd=str(Path(__file__).resolve().parent.parent),
                          capture_output=True, text=True, encoding="utf-8", timeout=300, env=env)
    return proc, [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]


def test_result_cache_hit_miss_and_bypass(synthetic_video, tmp_path):
    env = dict(os.environ, CAPPYCAT_ANALYSIS_CACHE_DIR=str(tmp_path / "cache"))
    base = ["analyze", str(synthetic_video), "--detector", "none", "--no-characters"]
    p1, ev1 = _cli(base + ["--out", str(tmp_path / "a.json")], env)
    assert p1.returncode == 0, p1.stderr
    assert any("0/1 clip(s) cached" in e.get("message", "") for e in ev1)
    pcts = [e["pct"] for e in ev1 if e["event"] == "progress" and e["stage"] != "download"]
    assert pcts == sorted(pcts) and pcts[-1] == 1.0  # re-weighted progress never goes backwards
    entries = list((tmp_path / "cache").glob("*.json"))
    assert len(entries) == 1
    p2, ev2 = _cli(base + ["--out", str(tmp_path / "b.json")], env)
    assert p2.returncode == 0, p2.stderr
    assert any(e["event"] == "log" and "cached analysis reused" in e["message"] for e in ev2)
    assert any(e["event"] == "progress" and e["message"] == "cached" for e in ev2)
    a = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))["clips"][0]
    b = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))["clips"][0]
    for k in ("shots", "duplicates", "reframe", "audio", "transitions"):
        assert a[k] == b[k]
    # a changed option is a miss; --no-cache neither reads nor writes
    p3, ev3 = _cli(base + ["--out", str(tmp_path / "c.json"), "--similarity-threshold", "0.8"], env)
    assert not any("cached analysis reused" in e.get("message", "") for e in ev3)
    n_before = len(list((tmp_path / "cache").glob("*.json")))
    p4, ev4 = _cli(base + ["--out", str(tmp_path / "d.json"), "--no-cache"], env)
    assert p4.returncode == 0
    assert not any("cached analysis reused" in e.get("message", "") or "analysis cache:" in e.get("message", "")
                   for e in ev4)
    assert len(list((tmp_path / "cache").glob("*.json"))) == n_before
    # touching the file (new mtime) invalidates the entry
    os.utime(synthetic_video, None)
    p5, ev5 = _cli(base + ["--out", str(tmp_path / "e.json")], env)
    assert not any("cached analysis reused" in e.get("message", "") for e in ev5)


# --------------------------------------------------------------------------- hi-res re-run (P1-7)


class _FakeYolo(perception.YoloWorldDetector):
    """YOLO-World stand-in: one detection at 640 px, three (small background figures) at 1280 px."""

    def __init__(self):  # no model
        self.name, self.conf, self.imgsz, self.calls = "yolo_world", 0.2, 640, []

    def detect(self, frame, prompts, imgsz=None):
        self.calls.append(imgsz or self.imgsz)
        s = frame.shape[1] / 640.0
        boxes = [(60, 40, 200, 340)] + ([(420, 250, 440, 300), (500, 250, 520, 300)] if imgsz == 1280 else [])
        return [Detection(tuple(v * s for v in b), "person", 0.5) for b in boxes]


def test_sparse_shots_are_re_run_at_high_resolution():
    yolo = _FakeYolo()
    hyb = HybridDetector(yolo)
    hires = lambda: ((i * 6, np.zeros((720, 1280, 3), np.uint8)) for i in range(4))
    res = analyze_shot(frames(4), SHOT, hyb, StubEmbedder([((0, 2000), [1, 0, 0, 0])]), 0.85, ["person"], 24.0, 1.0, 4.0,
                       hires_frames_factory=hires, hires_scale=1.0)
    assert res.hires and res.stats.detections == 12 and yolo.calls.count(1280) == 4
    # a shot that is not sparse is not re-run
    yolo2 = _FakeYolo()
    dense = StubDetector([[(LEFT, "person", 0.6), (RIGHT, "person", 0.6)]], name="yolo_world")
    res = analyze_shot(frames(4), SHOT, HybridDetector(dense), StubEmbedder([((0, 320), [1, 0, 0, 0]), ((320, 640), [0, 1, 0, 0])]),
                       0.85, ["person"], 24.0, 1.0, 4.0, hires_frames_factory=hires)
    assert not res.hires


def test_strong_zero_shot_agreement_tags_back_views():
    assert Identity(None, 0.64, 0.004, "bunny", zs_best="bunny", zs_prob=0.998).loose() == "bunny"
    assert Identity(None, 0.64, 0.004, "bunny", zs_best="bunny", zs_prob=0.9).loose() is None
    assert Identity(None, 0.64, 0.004, "bunny", zs_best="raccoon", zs_prob=0.99).loose() is None
    assert Identity(None, 0.60, 0.1, "bunny", zs_best="bunny", zs_prob=0.999).loose() is None


def test_watchdog_does_not_crash_a_normal_exit_with_stdin_held_open():
    """Regression: a daemon thread blocked in the buffered sys.stdin crashed the interpreter at
    shutdown (Fatal Python error: _enter_buffered_busy, 0xC0000005), so every Rust-spawned job
    looked failed. The watchdog must read a raw duplicated fd instead."""
    import os
    import subprocess
    import sys

    env = dict(os.environ, CAPPYCAT_WATCH_STDIN="1")
    code = "from cappycat_pipeline import procs; procs.start_stdin_watchdog(); print('ok')"
    p = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, env=env)
    try:
        out = p.stdout.read()  # read to EOF while our end of stdin stays open
        rc = p.wait(timeout=60)
    finally:
        if p.stdin:
            p.stdin.close()
    assert rc == 0, p.stderr.read().decode(errors="replace")
    assert b"ok" in out
