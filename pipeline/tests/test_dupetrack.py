"""Following a confirmed duplicate through its shot + the tracked, clamped reframe (CPU, stubs)."""
import numpy as np

from cappycat_pipeline.dupetrack import PairTrack, group_findings, track_pair
from cappycat_pipeline.reframe import build_tracked_reframe, clamp_crop, head_region
from cappycat_pipeline.schema import DetectedInstance, DuplicateFinding, Shot
from cappycat_pipeline.tracking import Detection

W, H = 1280, 720
FPS = 24.0


def _inter(a, b):
    return a[0] < b[2] - 0.5 and a[2] > b[0] + 0.5 and a[1] < b[3] - 0.5 and a[3] > b[1] + 0.5


def _contains(c, b):
    return c[0] <= b[0] + 0.5 and c[1] <= b[1] + 0.5 and c[2] >= b[2] - 0.5 and c[3] >= b[3] - 0.5


def _finding(frame, prim, dup, char="bunny"):
    return DuplicateFinding(0, frame, frame / FPS * 1000, DetectedInstance(1, "person", list(prim), 0.6),
                            DetectedInstance(2, "person", list(dup), 0.5), 0.95, char, char.capitalize() if char else None)


def test_track_pair_follows_moving_copies_and_drops_after_exit():
    """Two identical instances moving apart; the duplicate leaves the frame on the right. The
    pair is confirmed once (frame 24) and followed both ways without any identity score."""
    n = 16
    frames, boxes = [], []
    for k in range(n):
        prim = (200 + 5 * k, 150, 400 + 5 * k, 650)
        dup = (700 + 60 * k, 150, 900 + 60 * k, 650)
        frames.append((k * 3, np.zeros((H // 2, W // 2, 3), np.uint8)))
        bs = [prim] + ([dup] if dup[0] < W else [])
        boxes.append([tuple(v / 2 for v in b) for b in bs])  # analysis at half resolution
    anchor = _finding(24, (240, 150, 440, 650), (1180, 150, 1380, 650))  # sample 8
    it = iter(range(n))

    def detect(img):
        return [Detection(b, "person", 0.3) for b in boxes[next(it)]]

    emb = lambda img, bs: np.tile(np.array([1.0, 0, 0, 0], np.float32), (len(bs), 1))  # identical look
    pt = track_pair(frames, [anchor], detect, emb, 0.5, 8.0, hold_s=0.25)
    assert pt is not None and pt.anchors == [8]
    for k in range(n):
        assert pt.primary[k] is not None and abs(pt.primary[k][0] - (200 + 5 * k)) < 1
        if 700 + 60 * k < W:  # still visible: tracked, never confused with the primary
            assert pt.duplicate[k] is not None and abs(pt.duplicate[k][0] - (700 + 60 * k)) < 1 and not pt.dup_held[k]
    # after the exit: held for hold_s (2 samples at 8 fps), then gone
    gone = [k for k in range(n) if 700 + 60 * k >= W]
    assert pt.dup_held[gone[0]] and pt.duplicate[gone[-1]] is None


def test_identity_veto_blocks_other_cast_members():
    class Ident:
        def __init__(self, c):
            self.character = c

    frames = [(k * 3, np.zeros((360, 640, 3), np.uint8)) for k in range(4)]
    boxes = [[(100, 75, 200, 325), (350, 75, 450, 325)], [(100, 75, 200, 325), (360, 75, 460, 325)]] * 2
    it = iter(range(4))
    anchor = _finding(0, (200, 150, 400, 650), (700, 150, 900, 650))
    pt = track_pair(frames, [anchor], lambda img: [Detection(b, "person", 0.3) for b in boxes[next(it)]],
                    lambda img, bs: np.tile(np.array([1.0, 0, 0, 0], np.float32), (len(bs), 1)), 0.5, 8.0, hold_s=0.1,
                    identify=lambda embs: [Ident(None), Ident("raccoon")])  # 2nd box is confidently someone else
    assert pt.duplicate[1] is not None and pt.dup_held[1]  # not re-found on the vetoed box: held
    assert pt.duplicate[3] is None


def test_group_findings_by_character_and_pair():
    fs = [_finding(3, (0, 0, 1, 1), (2, 0, 3, 1)), _finding(6, (0, 0, 1, 1), (2, 0, 3, 1)),
          _finding(6, (0, 0, 1, 1), (2, 0, 3, 1), char=None)]
    groups = group_findings(fs)
    assert sorted(len(g) for g in groups) == [1, 2]


def test_clamp_crop_moves_off_duplicate_keeping_head():
    head = (300, 100, 420, 250)
    dup = (700, 80, 950, 700)
    c = clamp_crop((200, 50, 1000, 500), [dup], head, W, H, W / H)
    assert not _inter(c, dup) and _contains(c, head)
    assert abs((c[2] - c[0]) / (c[3] - c[1]) - W / H) < 1e-6
    # head at the very top edge is pulled back into the crop
    c = clamp_crop((0, 1, 740, 417), [(776, 0, 1280, 720)], (96, 0, 384, 216), W, H, W / H)
    assert _contains(c, (96, 0, 384, 216))


def _pair(n=33, dup_from=0):
    frames = [288 + 3 * k for k in range(n)]
    prim = [(436 - 2 * k, 404 - 1 * k, 586 - 2 * k, 720) for k in range(n)]
    dup = [None if k < dup_from else (1010 - 4 * k, 386 - 1 * k, 1190 - 4 * k, 720) for k in range(n)]
    return PairTrack(frames, prim, dup, [False] * n, [n // 2], "Felix", "felix",
                     {"samples": n, "observedDuplicate": n - dup_from, "held": 0})


def test_tracked_reframe_excludes_duplicate_every_frame_and_relaxes():
    shot = Shot(4, 288, 384, 12000.0, 16000.0, 0.95, "transnetv2")
    track, tf = build_tracked_reframe(shot, [_pair(dup_from=12)], W, H, FPS, return_frames=True)
    assert len(track.keyframes) == 97 and "Felix" in track.reason
    for i in range(len(tf.frames)):
        for d in tf.duplicates[i]:
            assert not _inter(tf.crops[i], d)
        if tf.heads[i] is not None:
            assert _contains(tf.crops[i], tf.heads[i])
    # before the duplicate appears the crop relaxes towards the full frame
    widths = tf.crops[:, 2] - tf.crops[:, 0]
    assert widths[0] > 1200 and widths[-1] < 1100
    assert np.abs(np.diff(widths[:30])).max() < 40  # smooth, no jump


def test_head_region_is_top_30_percent():
    assert head_region((100, 100, 200, 400)) == (120.0, 100.0, 180.0, 190.0)
