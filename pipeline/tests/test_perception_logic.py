"""Duplicate-finding / hybrid-escalation / identity logic with stub detectors and embedders
(no ML models needed)."""
from pathlib import Path

import numpy as np

from cappycat_pipeline import perception
from cappycat_pipeline.characters import CharacterBank, Character, Manifest, labels_compatible
from cappycat_pipeline.perception import (HybridDetector, analyze_shot, dedupe_findings, find_duplicates, nms,
                                          tracker_for)
from cappycat_pipeline.schema import Shot
from cappycat_pipeline.tracking import Detection
from cappycat_pipeline.transitions import DISSOLVE_THRESHOLD, suggest

W, H = 640, 360
SHOT = Shot(0, 0, 47, 0.0, 1958.3, 1.0, "transnetv2")


class StubDetector:
    """Returns the boxes given per frame index (``boxes[i]`` -> list of (bbox, label, score))."""

    name = "stub"
    conf = 0.2

    def __init__(self, boxes, name="stub"):
        self.boxes = boxes
        self.name = name
        self.i = 0

    def detect(self, frame, prompts):
        out = [Detection(tuple(float(v) for v in b), lab, s) for b, lab, s in self.boxes[min(self.i, len(self.boxes) - 1)]]
        self.i += 1
        return out


class StubEmbedder:
    """Embedding chosen by the box's x-centre: ``table`` maps an x-range to a vector."""

    name = "stub"

    def __init__(self, table):
        self.table = table

    def embed_many(self, frame, bboxes):
        out = []
        for b in bboxes:
            cx = (b[0] + b[2]) / 2
            for (x0, x1), v in self.table:
                if x0 <= cx < x1:
                    v = np.asarray(v, np.float32)
                    out.append(v / np.linalg.norm(v))
                    break
            else:
                out.append(np.eye(4, dtype=np.float32)[3])
        return np.stack(out)


def frames(n=8):
    return lambda: ((i * 6, np.zeros((H, W, 3), np.uint8)) for i in range(n))


LEFT, RIGHT = (60, 40, 200, 340), (420, 40, 560, 340)


def test_nms_class_agnostic_keeps_best_box():
    d = [Detection((0, 0, 100, 200), "cartoon rabbit", 0.55), Detection((2, 1, 101, 199), "animal character", 0.34)]
    assert len(nms(d, 0.6)) == 2
    kept = nms(d, 0.6, class_agnostic=True)
    assert len(kept) == 1 and kept[0].label == "cartoon rabbit"


def test_labels_compatible_generic():
    assert labels_compatible("cartoon rabbit", "animal character")
    assert labels_compatible("person", "cartoon turtle")
    assert not labels_compatible("cartoon rabbit", "cartoon turtle")


def test_generic_rule_per_frame_without_confirmed_tracks():
    """Low scores (0.25) and label flips must not hide a duplicate: pairs are tested per frame."""
    boxes = [[(LEFT, "animal character", 0.25), (RIGHT, "cartoon rabbit", 0.24)]] * 8
    det = StubDetector(boxes)
    emb = StubEmbedder([((0, 320), [1, 0.05, 0, 0]), ((320, 640), [1, 0.0, 0.05, 0])])
    f = find_duplicates(frames()(), SHOT, det, tracker_for(det, 4.0), emb, 0.85, ["x"], 24.0, 1.0, keep_all=True)
    assert len(f) == 8 and all(x.similarity > 0.99 and x.character is None for x in f)
    assert len(dedupe_findings(f)) == 1


def test_nested_and_overlapping_boxes_are_not_duplicates():
    body, head = (100, 40, 260, 340), (130, 40, 230, 130)
    boxes = [[(body, "person", 0.8), (head, "animal character", 0.5)]] * 4
    det = StubDetector(boxes)
    emb = StubEmbedder([((0, 640), [1, 0, 0, 0])])
    assert find_duplicates(frames(4)(), SHOT, det, tracker_for(det, 4.0), emb, 0.85, ["x"], 24.0) == []


def _bank(threshold=0.7):
    chars = [Character("bunny", "Bunny", "rabbit"), Character("turtle", "Turtle", "turtle")]
    m = Manifest(Path("characters.json"), chars, {}, ["animal character"])
    refs = np.array([[1, 0, 0, 0], [0.95, 0.3, 0, 0], [0, 1, 0, 0], [0.3, 0.95, 0, 0]], np.float32)
    refs /= np.linalg.norm(refs, axis=1, keepdims=True)
    return CharacterBank(m, ["bunny", "bunny", "turtle", "turtle"], refs, ["a#0", "a#1", "b#0", "b#1"], threshold, 0.02)


def test_named_rule_same_character_twice():
    bank = _bank()
    boxes = [[(LEFT, "person", 0.6), (RIGHT, "person", 0.5)]] * 4
    det = StubDetector(boxes)
    emb = StubEmbedder([((0, 320), [1, 0.1, 0, 0]), ((320, 640), [1, 0.12, 0, 0])])
    f = find_duplicates(frames(4)(), SHOT, det, tracker_for(det, 4.0), emb, 0.85, ["person"], 24.0, bank=bank,
                        keep_all=True)
    assert f and all(x.character == "bunny" and x.characterName == "Bunny" for x in f)
    d = dedupe_findings(f)
    assert len(d) == 1  # one named finding per (shot, character)


def test_named_rule_needs_near_identical_pair():
    """Two instances that both identify as Turtle but do not look alike (young vs elderly turtle)."""
    bank = _bank()
    boxes = [[(LEFT, "person", 0.6), (RIGHT, "person", 0.5)]] * 4
    det = StubDetector(boxes)
    emb = StubEmbedder([((0, 320), [0.25, 1, 0.3, 0]), ((320, 640), [0.3, 1, -0.35, 0])])
    st = perception.ShotStats()
    f = find_duplicates(frames(4)(), SHOT, det, tracker_for(det, 4.0), emb, 0.85, ["person"], 24.0, bank=bank,
                        stats=st)
    assert f == [] and st.cast_frames.get("turtle") == 4


def test_different_characters_are_never_duplicates():
    bank = _bank()
    boxes = [[(LEFT, "person", 0.6), (RIGHT, "person", 0.5)]] * 4
    det = StubDetector(boxes)
    emb = StubEmbedder([((0, 320), [1, 0.5, 0, 0]), ((320, 640), [0.5, 1, 0, 0])])  # sim 0.8, bunny vs turtle
    assert find_duplicates(frames(4)(), SHOT, det, tracker_for(det, 4.0), emb, 0.75, ["person"], 24.0, bank=bank) == []


def test_hybrid_escalates_when_primary_finds_nothing():
    primary = StubDetector([[]], name="yolo_world")
    fallback = StubDetector([[(LEFT, "person", 0.6), (RIGHT, "person", 0.6)]], name="grounded_sam2")
    hyb = HybridDetector(primary, fallback_factory=lambda: fallback)
    emb = StubEmbedder([((0, 640), [1, 0, 0, 0])])
    res = analyze_shot(frames(4), SHOT, hyb, emb, 0.85, ["person"], 24.0, 1.0, 4.0)
    assert res.path == "grounded_sam2" and "found nothing" in res.reason
    assert res.primary_stats is not None and res.primary_stats.detections == 0
    assert len(dedupe_findings(res.findings)) == 1


def test_hybrid_escalates_on_overlap_and_borderline_but_not_when_clean():
    emb = StubEmbedder([((0, 320), [1, 0, 0, 0]), ((320, 640), [0.8, 0.6, 0, 0])])  # sim 0.8 = borderline for 0.85
    boxes = [[(LEFT, "person", 0.6), (RIGHT, "person", 0.6)]]
    res = analyze_shot(frames(4), SHOT, HybridDetector(StubDetector(boxes, "yolo_world"),
                                                       fallback_factory=lambda: StubDetector(boxes, "grounded_sam2")),
                       emb, 0.85, ["person"], 24.0, 1.0, 4.0)
    assert res.path == "grounded_sam2" and "borderline" in res.reason
    overlap = [[((60, 40, 260, 340), "person", 0.6), ((120, 40, 320, 340), "person", 0.6)]]
    res = analyze_shot(frames(4), SHOT, HybridDetector(StubDetector(overlap, "yolo_world"),
                                                       fallback_factory=lambda: StubDetector(overlap, "grounded_sam2")),
                       StubEmbedder([((0, 640), [1, 0, 0, 0])]), 0.85, ["person"], 24.0, 1.0, 4.0)
    assert res.path == "grounded_sam2" and "overlap" in res.reason
    clean_emb = StubEmbedder([((0, 320), [1, 0, 0, 0]), ((320, 640), [0, 1, 0, 0])])
    res = analyze_shot(frames(4), SHOT, HybridDetector(StubDetector(boxes, "yolo_world"),
                                                       fallback_factory=lambda: StubDetector(boxes, "grounded_sam2")),
                       clean_emb, 0.85, ["person"], 24.0, 1.0, 4.0)
    assert res.path == "yolo_world" and res.findings == []


def test_hybrid_without_fallback_reports_skipped_escalation():
    def boom():
        raise perception.MissingDependency("no transformers")

    hyb = HybridDetector(StubDetector([[]], "yolo_world"), fallback_factory=boom)
    res = analyze_shot(frames(2), SHOT, hyb, StubEmbedder([((0, 640), [1, 0, 0, 0])]), 0.85, ["p"], 24.0, 1.0, 4.0)
    assert res.path == "yolo_world" and "escalation skipped" in res.reason and not hyb.fallback_possible


def test_bank_identify_threshold_and_margin():
    bank = _bank(threshold=0.9)
    ids = bank.identify(np.array([[1, 0.1, 0, 0], [0.7, 0.7, 0, 0]], np.float32))
    assert ids[0].character == "bunny" and ids[0].best == "bunny"
    assert ids[1].character is None  # ambiguous: margin too small / below threshold


def test_transition_suggestion_relative_to_clip():
    assert suggest(DISSOLVE_THRESHOLD - 0.01) == "dissolve" and suggest(DISSOLVE_THRESHOLD) == "cut"
    clip = [0.28, 0.31, 0.29, 0.33, 0.13]
    assert suggest(0.13, clip) == "dissolve"   # < 0.2 and < half the clip median (0.29)
    assert suggest(0.25, clip) == "cut"        # ordinary camera cut
    assert suggest(0.13, [0.13, 0.15, 0.2]) == "cut"  # a clip made of such cuts: none stands out
