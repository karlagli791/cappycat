import numpy as np

from cappycat_pipeline.tracking import (ByteTracker, Detection, KalmanFilter, iou_matrix, linear_assignment, xyah_to_xyxy,
                                        xyxy_to_xyah)


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [100, 100, 110, 110]])
    iou = iou_matrix(a, b)
    assert iou.shape == (2, 3)
    assert iou[0, 0] == 1.0
    assert abs(iou[0, 1] - 25 / 175) < 1e-9
    assert iou[0, 2] == 0.0 and iou[1, 0] == 0.0
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_xyah_round_trip():
    b = (10.0, 20.0, 50.0, 100.0)
    assert np.allclose(xyah_to_xyxy(xyxy_to_xyah(b)), b)


def test_linear_assignment_respects_threshold():
    cost = np.array([[0.1, 0.9], [0.95, 0.2]])
    m, ur, uc = linear_assignment(cost, 0.5)
    assert sorted(m) == [(0, 0), (1, 1)] and ur == [] and uc == []
    m, ur, uc = linear_assignment(np.array([[0.9]]), 0.5)
    assert m == [] and ur == [0] and uc == [0]
    m, ur, uc = linear_assignment(np.zeros((0, 2)), 0.5)
    assert m == [] and ur == [] and uc == [0, 1]


def test_kalman_predicts_constant_velocity():
    kf = KalmanFilter()
    mean, cov = kf.initiate(xyxy_to_xyah((0, 0, 20, 40)))
    for i in range(1, 6):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, xyxy_to_xyah((i * 10, 0, i * 10 + 20, 40)))
    mean, cov = kf.predict(mean, cov)
    assert abs(mean[0] - (60 + 10)) < 3.0  # predicted centre x continues moving ~10 px/frame
    assert mean[4] > 5.0  # positive x velocity


def _box(x, y, w=50, h=80):
    return (x, y, x + w, y + h)


def test_single_object_keeps_identity():
    tr = ByteTracker(frame_rate=24)
    ids = set()
    for f in range(40):
        out = tr.update([Detection(_box(10 + f * 4, 20), "cat", 0.9)])
        ids |= {t.track_id for t in out}
        assert len(out) == 1
    assert ids == {1}


def test_two_objects_no_id_swap():
    tr = ByteTracker(frame_rate=24)
    a_id = b_id = None
    for f in range(40):
        a = _box(10 + f * 3, 10)
        b = _box(400 - f * 3, 200)
        out = tr.update([Detection(a, "cat", 0.9), Detection(b, "cat", 0.85)])
        by_x = sorted(out, key=lambda t: t.bbox[0])
        assert len(out) == 2
        if f == 1:
            a_id, b_id = by_x[0].track_id, by_x[1].track_id
        elif f > 1:
            assert (by_x[0].track_id, by_x[1].track_id) == (a_id, b_id)


def test_low_score_detections_keep_track_alive():
    """ByteTrack's 2nd stage: occluded (low-score) boxes are still associated."""
    tr = ByteTracker(track_thresh=0.5, frame_rate=24)
    ids_by_frame = []
    for f in range(30):
        score = 0.2 if 8 <= f < 18 else 0.9
        out = tr.update([Detection(_box(10 + f * 5, 20), "person", score)])
        ids_by_frame.append([t.track_id for t in out])
    flat = {i for ids in ids_by_frame for i in ids}
    assert flat == {1}
    # still tracked during the low-score stretch
    assert all(ids_by_frame[f] == [1] for f in range(8, 18))


def test_lost_track_is_reidentified_within_buffer_and_new_after():
    tr = ByteTracker(frame_rate=30, track_buffer=10)
    for f in range(5):
        tr.update([Detection(_box(100, 100), "dog", 0.9)])
    for _ in range(5):  # short disappearance (< buffer)
        tr.update([])
    out = tr.update([Detection(_box(102, 101), "dog", 0.9)])
    assert [t.track_id for t in out] == [1]
    for _ in range(25):  # long disappearance (> buffer) -> removed
        tr.update([])
    out = tr.update([Detection(_box(100, 100), "dog", 0.9)])
    out = tr.update([Detection(_box(100, 100), "dog", 0.9)])
    assert [t.track_id for t in out] == [2]


def test_new_tracks_need_confirmation_and_min_score():
    tr = ByteTracker(track_thresh=0.5, frame_rate=24)
    tr.update([Detection(_box(0, 0), "cat", 0.9)])  # frame 1 activates immediately
    out = tr.update([Detection(_box(2, 0), "cat", 0.9), Detection(_box(300, 300), "cat", 0.9)])
    assert [t.track_id for t in out] == [1]  # the new track is tentative on its first frame
    out = tr.update([Detection(_box(4, 0), "cat", 0.9), Detection(_box(302, 300), "cat", 0.9)])
    assert sorted(t.track_id for t in out) == [1, 2]
    # detections below det_thresh (0.6) never spawn tracks
    tr2 = ByteTracker(track_thresh=0.5)
    for _ in range(5):
        assert tr2.update([Detection(_box(0, 0), "cat", 0.55)]) == []


def test_labels_do_not_mix():
    tr = ByteTracker(frame_rate=24)
    tr.update([Detection(_box(0, 0), "cat", 0.9)])
    out = tr.update([Detection(_box(1, 0), "dog", 0.9)])
    assert out == [] or all(t.label == "cat" for t in out)
    out = tr.update([Detection(_box(2, 0), "dog", 0.9)])
    assert {t.label for t in out} <= {"cat", "dog"}
    assert any(t.label == "dog" and t.track_id == 2 for t in out)
