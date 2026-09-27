"""ML-path tests on the real models (``@pytest.mark.gpu``). Each test skips when torch / CUDA /
the model files (``python -m cappycat_pipeline download-models``) or the user's media folders
(``<repo>/cappycat- clips``, ``<repo>/characters``) are unavailable. Run only these with
``pytest -m gpu``."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cappycat_pipeline import ffmpeg_util, models

pytestmark = pytest.mark.gpu

PIPELINE_DIR = Path(__file__).resolve().parent.parent
REPO = PIPELINE_DIR.parent
CLIPS = REPO / "cappycat- clips"
CHARACTERS = REPO / "characters" / "characters.json"


def _cuda() -> bool:
    return models.cuda_ok()


def need(*conds_and_reasons):
    for ok, why in conds_and_reasons:
        if not ok:
            pytest.skip(why)


@pytest.fixture(scope="module")
def cuda():
    need((_cuda(), "torch with CUDA not available"))


def _asset(name: str) -> Path:
    try:
        import ultralytics  # noqa: F401
    except Exception:
        pytest.skip("ultralytics not installed")
    p = Path(ultralytics.__file__).parent / "assets" / name
    need((p.is_file(), f"ultralytics asset {name} missing"))
    return p


def _img(name: str) -> np.ndarray:
    import cv2

    return cv2.imread(str(_asset(name)))


def _run_cli(args, timeout=900):
    proc = subprocess.run([sys.executable, "-m", "cappycat_pipeline", *args], cwd=str(PIPELINE_DIR), capture_output=True,
                          text=True, encoding="utf-8", timeout=timeout)
    events = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]  # stdout = JSON lines only
    return proc, events


# --------------------------------------------------------------------------- TransNetV2


def _scene(img: np.ndarray, n: int) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    return np.stack([cv2.cvtColor(cv2.resize(img[: h * 3 // 4, int(i * 0.5): int(i * 0.5) + w * 3 // 4], (48, 27),
                                             interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for i in range(n)])


@pytest.fixture(scope="module")
def transnet(cuda):
    need((models.transnet_pt_path().is_file(), "models/transnetv2.pt missing"))
    from cappycat_pipeline.shots import TransNetV2Torch

    m = TransNetV2Torch(device="cuda")
    yield m
    m.close()


def test_transnet_hard_cut_dissolve_and_static(transnet):
    a, b = _scene(_img("bus.jpg"), 150), _scene(_img("zidane.jpg"), 150)
    p = transnet.predict_frames(np.concatenate([a[:60], b[60:]]))
    assert int(p.argmax()) == 59 and p[59] > 0.9  # boundary frame of the hard cut
    assert np.delete(p, [58, 59, 60]).max() < 0.1
    fade = a.astype(np.float32).copy()
    for i in range(150):
        t = min(1.0, max(0.0, (i - 60) / 24))
        fade[i] = (1 - t) * a[i] + t * b[i]
    p = transnet.predict_frames(fade.astype(np.uint8))
    assert p[58:88].max() > 0.5 and 62 <= 58 + int(p[58:88].argmax()) <= 84  # peak inside the dissolve
    assert transnet.predict_frames(a).max() < 0.1  # slow pan, no transition


def test_transnet_onnx_matches_torch(transnet):
    need((models.transnet_onnx_path().is_file(), "models/transnetv2.onnx missing"))
    from cappycat_pipeline.shots import TransNetV2Onnx

    frames = np.concatenate([_scene(_img("bus.jpg"), 80), _scene(_img("zidane.jpg"), 80)])
    pt = transnet.predict_frames(frames)
    ox = TransNetV2Onnx(providers=["CPUExecutionProvider"]).predict_frames(frames)
    assert np.abs(pt - ox).max() < 0.02


def test_transnet_detects_synthetic_cuts(transnet, synthetic_video):
    from cappycat_pipeline.shots import detect_shots

    shots = detect_shots(synthetic_video, method="transnetv2")
    assert [s.method for s in shots] == ["transnetv2"] * 3
    assert abs(shots[1].startFrame - 48) <= 1 and abs(shots[2].startFrame - 96) <= 1


# --------------------------------------------------------------------------- detectors / embedder / flow


def test_yolo_world_fp16_finds_people(cuda):
    need((models.yolo_world_path().is_file(), "yolov8s-worldv2.pt missing"))
    from cappycat_pipeline.perception import YoloWorldDetector

    det = YoloWorldDetector(prompts=["person", "bus"])
    assert det.half and det.device == "cuda"
    dets = det.detect(_img("bus.jpg"), ["person", "bus"])
    assert sum(d.label == "person" and d.score > 0.5 for d in dets) >= 3
    assert any(d.label == "bus" for d in dets)
    det.close()


def test_grounded_sam2_boxes_and_masks(cuda):
    need((models.hf_repo_present(models.GDINO_HF_REPO, "model.safetensors"), "Grounding DINO weights missing"),
         (models.hf_repo_present(models.SAM2_HF_REPO, "model.safetensors"), "SAM 2.1 weights missing"))
    from cappycat_pipeline.perception import GroundedSam2Detector

    det = GroundedSam2Detector()
    img = _img("bus.jpg")
    dets = det.detect(img, ["person"])
    people = [d for d in dets if d.label == "person"]
    assert len(people) >= 3
    for d in people:
        assert d.mask is not None and d.mask.shape == img.shape[:2] and d.mask.sum() > 500
        ys, xs = np.nonzero(d.mask)
        # the bbox is the tight box around the mask
        assert abs(xs.min() - d.bbox[0]) <= 2 and abs(ys.max() + 1 - d.bbox[3]) <= 2
    det.close()


def test_openclip_fp16_duplicate_similarity(cuda):
    need((models.hf_repo_present(models.OPENCLIP_HF_REPO, "open_clip_model.safetensors"), "OpenCLIP weights missing"))
    from cappycat_pipeline.perception import OpenClipEmbedder, cosine

    emb = OpenClipEmbedder()
    assert emb.dtype.__repr__().endswith("float16")
    img = _img("bus.jpg")
    person, other_person, bus = (222, 400, 346, 862), (668, 394, 810, 879), (0, 230, 808, 749)
    e = emb.embed_many(img, [person, person, other_person, bus])
    assert cosine(e[0], e[1]) > 0.99
    assert cosine(e[0], e[2]) < cosine(e[0], e[1]) and cosine(e[0], e[3]) < 0.85
    emb.close()


def test_raft_flow_recovers_shift(cuda):
    need((models.raft_small_path() is not None, "RAFT weights missing"))
    from cappycat_pipeline import transitions

    img = np.ascontiguousarray(_img("zidane.jpg")[100:452, 200:840])
    shifted = np.roll(img, 8, axis=1)
    flows, backend = transitions.estimate_flows([img], [shifted])
    assert backend == "raft"
    fx = flows[0][40:-40, 40:-40, 0]
    assert abs(float(np.median(fx)) - 8.0) < 1.0
    mids = transitions.interpolate_frames(img, shifted, 1)
    assert len(mids) == 1 and mids[0].shape == img.shape
    transitions.release_raft()


def test_doctor_reports_models_and_vram(cuda):
    proc, _ = _run_cli(["doctor"], timeout=300)
    rep = json.loads(proc.stdout)
    assert rep["mode"] == "ml" and rep["torchCuda"] and rep["device"] == "cuda"
    assert rep["vram"]["totalMb"] > 0 and rep["vram"]["freeMb"] > 0
    assert set(rep["models"]) >= {"transnetv2.pt", "yolov8s-worldv2.pt", "raft_small", models.GDINO_HF_REPO}


def test_download_models_is_idempotent(cuda):
    need((models.raft_small_path() is not None, "RAFT weights missing"))
    proc, events = _run_cli(["download-models", "--only", "raft", "transnetv2.pt"], timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert all(e["event"] in ("progress", "log", "result") for e in events)
    assert [e["clip"] for e in events if e["event"] == "progress" and e["pct"] == 1.0][-1] in ("raft_small", "transnetv2.pt")
    assert events[-1]["event"] == "result"


# --------------------------------------------------------------------------- end to end


def _make_duplicate_clip(out: Path) -> tuple:
    """Shot 1 (4 s): the bus.jpg pedestrian pasted twice (slight motion); hard cut; shot 2 (2 s):
    one pedestrian. 1280x720 @ 24 fps."""
    import cv2

    person = _img("bus.jpg")[400:862, 222:346]
    H, W, ph = 720, 1280, 520
    pw = int(round(person.shape[1] * ph / person.shape[0]))
    person = cv2.resize(person, (pw, ph), interpolation=cv2.INTER_CUBIC)

    def bg(c0, c1, seed):
        g = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
        b = (np.array(c0, np.float32) * (1 - g) + np.array(c1, np.float32) * g) * np.ones((H, 1, 1), np.float32)
        rng = np.random.default_rng(seed)
        for _ in range(40):
            cv2.circle(b, (int(rng.integers(0, W)), int(rng.integers(0, H // 2))), int(rng.integers(10, 60)),
                       tuple(float(v) for v in rng.integers(60, 200, 3)), -1)
        return cv2.GaussianBlur(b, (0, 0), 3).clip(0, 255).astype(np.uint8)

    bg1, bg2 = bg((150, 110, 60), (200, 170, 120), 1), bg((60, 140, 90), (40, 110, 200), 2)
    cmd = [ffmpeg_util.find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", "24", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
           "-pix_fmt", "yuv420p", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    y0 = H - ph - 60
    for i in range(96):
        f = bg1.copy()
        dx, dy = int(round(12 * math.sin(2 * math.pi * i / 48))), int(round(4 * math.sin(2 * math.pi * i / 32)))
        for x0 in (220 + dx, 1000 - dx):
            f[y0 + dy:y0 + dy + ph, x0:x0 + pw] = person
        proc.stdin.write(f.tobytes())
    for i in range(48):
        f = bg2.copy()
        x0 = (W - pw) // 2 + int(round(8 * math.sin(2 * math.pi * i / 48)))
        f[y0:y0 + ph, x0:x0 + pw] = person
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    return out


@pytest.mark.parametrize("detector", ["hybrid", "grounded_sam2"])
def test_analyze_synthetic_duplicate_person(cuda, tmp_path, has_ffmpeg, detector):
    need((has_ffmpeg, "ffmpeg missing"), (models.yolo_world_path().is_file(), "YOLO-World weights missing"),
         (models.transnet_pt_path().is_file(), "TransNetV2 weights missing"))
    if detector == "grounded_sam2":
        need((models.hf_repo_present(models.GDINO_HF_REPO, "model.safetensors"), "Grounding DINO weights missing"))
    clip = _make_duplicate_clip(tmp_path / "dup_person.mp4")
    out = tmp_path / "a.json"
    proc, events = _run_cli(["analyze", str(clip), "--out", str(out), "--detector", detector, "--prompts", "person",
                             "--no-characters"])
    assert proc.returncode == 0, proc.stderr
    assert any(e["event"] == "log" and f"path={detector if detector != 'hybrid' else 'yolo_world'}" in e["message"]
               for e in events)
    c = json.loads(out.read_text(encoding="utf-8"))["clips"][0]
    assert [s["method"] for s in c["shots"]] == ["transnetv2", "transnetv2"]
    assert abs(c["shots"][1]["startFrame"] - 96) <= 1
    dups = c["duplicates"]
    assert dups and all(d["shotIndex"] == 0 for d in dups)  # none in the single-person shot
    assert all(d["similarity"] >= 0.85 for d in dups)
    tracks = {r["shotIndex"]: r["track"] for r in c["reframe"]}
    assert set(tracks) == {0}
    by_frame = {d["frame"]: d for d in dups}
    for kf in tracks[0]["keyframes"]:
        d = by_frame[min(by_frame, key=lambda f: abs(f - kf["frame"]))]
        cr, dup, prim = kf["crop"], d["duplicate"]["bbox"], d["primary"]["bbox"]
        assert cr[2] <= dup[0] + 2 or cr[0] >= dup[2] - 2  # excludes the duplicate
        hx1, hy1, hx2, hy2 = prim[0] + 0.2 * (prim[2] - prim[0]), prim[1], prim[2] - 0.2 * (prim[2] - prim[0]),             prim[1] + 0.3 * (prim[3] - prim[1])
        assert cr[0] <= hx1 + 2 and cr[2] >= hx2 - 2 and cr[1] <= hy1 + 2 and cr[3] >= hy2 - 2  # head kept
        ix = max(0.0, min(cr[2], prim[2]) - max(cr[0], prim[0])) * max(0.0, min(cr[3], prim[3]) - max(cr[1], prim[1]))
        assert ix >= 0.95 * (prim[2] - prim[0]) * (prim[3] - prim[1])  # and essentially the whole body


# --------------------------------------------------------------------------- main cast (user media)


@pytest.fixture(scope="module")
def bank(cuda):
    need((CHARACTERS.is_file(), "characters/characters.json not present"),
         (models.hf_repo_present(models.OPENCLIP_HF_REPO, "open_clip_model.safetensors"), "OpenCLIP weights missing"))
    from cappycat_pipeline import characters

    return characters.load_bank(CHARACTERS)


def test_character_reference_calibration(bank):
    cal = bank.calibration
    counts = bank.counts()
    assert set(counts) == {c.id for c in bank.manifest.characters} and min(counts.values()) >= 4
    assert cal["queries"] >= 10 and cal["top1Accuracy"] >= 0.7
    assert 0.6 <= bank.id_threshold <= 0.8 and 0.0 < bank.id_margin <= 0.05
    crops = CHARACTERS.parent / ".cache" / "crops"
    assert len(list(crops.glob("*.jpg"))) >= len(counts)


def _analyze_clip(tmp_path, name):
    clip = CLIPS / name
    need((clip.is_file(), f"{clip} not present"))
    out = tmp_path / f"{name}.json"
    proc, events = _run_cli(["analyze", str(clip), "--out", str(out), "--detector", "hybrid", "--keep-order"], timeout=1200)
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text(encoding="utf-8"))["clips"][0], events


def test_clip7_duplicate_bunny_is_found_and_reframed(bank, tmp_path):
    c, _ = _analyze_clip(tmp_path, "clip7.mp4")
    late = [s for s in c["shots"] if s["startMs"] > 8500]
    assert late, "expected a shot starting around 9.1 s"
    shot = late[0]
    bunny = [d for d in c["duplicates"] if d["shotIndex"] == shot["index"] and d.get("character") == "bunny"]
    assert bunny and bunny[0]["characterName"] == "Bunny" and bunny[0]["similarity"] >= 0.85
    track = {r["shotIndex"]: r["track"] for r in c["reframe"]}[shot["index"]]
    d = bunny[0]
    mid = min(track["keyframes"], key=lambda k: abs(k["frame"] - d["frame"]))
    cr, dup, prim = mid["crop"], d["duplicate"]["bbox"], d["primary"]["bbox"]
    assert cr[2] <= dup[0] + 5 or cr[0] >= dup[2] - 5  # one bunny cropped out
    pcx, pcy = (prim[0] + prim[2]) / 2, (prim[1] + prim[3]) / 2
    assert cr[0] <= pcx <= cr[2] and cr[1] <= pcy <= cr[3]  # the other one kept
    assert "bunny" in (shot.get("cast") or [])
    assert "Bunny" in (c["asset"].get("sceneTags") or [])


def test_clip9_two_different_turtles_are_not_duplicates(bank, tmp_path):
    c, _ = _analyze_clip(tmp_path, "clip9.mp4")
    assert not [d for d in c["duplicates"] if d.get("character") == "turtle"]
    assert all(set(s.get("cast") or []) <= {b.id for b in bank.manifest.characters} for s in c["shots"])


# --------------------------------------------------------------------------- duplicate tracking through a shot


@pytest.fixture(scope="module")
def tracking_models(bank):
    need((models.hf_repo_present(models.GDINO_HF_REPO, "model.safetensors"), "Grounding DINO weights missing"),
         (models.yolo_world_path().is_file(), "YOLO-World weights missing"))
    from cappycat_pipeline import perception

    det = perception.HybridDetector(perception.YoloWorldDetector(prompts=bank.manifest.detection_prompts()),
                                    fallback_factory=perception.GroundedSam2Detector)
    emb = perception.OpenClipEmbedder()
    yield det, emb
    det.close()
    emb.close()


@pytest.mark.parametrize("clip,start_after_ms,character", [("clip10.mp4", 10000, "felix"), ("clip7.mp4", 8500, "bunny")])
def test_tracked_reframe_on_real_shot(tracking_models, bank, clip, start_after_ms, character):
    """The duplicate is confirmed at a few frames; followed through the whole shot, the crop must
    never intersect it (any frame) and must keep the primary's head region wherever it is tracked."""
    from cappycat_pipeline import dupetrack, perception, reframe, shots

    path = CLIPS / clip
    need((path.is_file(), f"{path} not present"))
    det, emb = tracking_models
    a = ffmpeg_util.probe(path)
    sh = [s for s in shots.detect_shots(path, "auto", asset=a) if s.startMs > start_after_ms][0]
    prompts = bank.manifest.detection_prompts()
    fac = lambda: perception.sample_shot_frames(str(path), sh, a.fps, (a.width, a.height), 4.0, 640)
    res = perception.analyze_shot(fac, sh, det, emb, 0.85, prompts, a.fps, 640 / a.width, 4.0, keep_all=True, bank=bank)
    found = [f for f in res.findings if f.character == character]
    assert found, f"no {character} duplicate confirmed in {clip} shot {sh.index}"
    pairs = dupetrack.track_shot(str(path), sh, found, det, emb, prompts, a.fps, (a.width, a.height), res.path, bank)
    assert pairs and pairs[0].stats["observedDuplicate"] >= 8
    track, tf = reframe.build_tracked_reframe(sh, pairs, a.width, a.height, a.fps, return_frames=True)
    assert len(tf.frames) == sh.endFrame - sh.startFrame + 1
    assert sum(1 for d in tf.duplicates if d) >= 0.6 * len(tf.frames)
    for i in range(len(tf.frames)):
        c = tf.crops[i]
        for d in tf.duplicates[i]:
            assert not (c[0] < d[2] - 0.5 and c[2] > d[0] + 0.5 and c[1] < d[3] - 0.5 and c[3] > d[1] + 0.5),                 f"frame {tf.frames[i]}: crop {c} intersects duplicate {d}"
        h = tf.heads[i]
        if h is not None:
            assert c[0] <= h[0] + 0.5 and c[1] <= h[1] + 0.5 and c[2] >= h[2] - 0.5 and c[3] >= h[3] - 0.5,                 f"frame {tf.frames[i]}: crop {c} cuts the primary's head {h}"
    # the keyframes the editor receives are the per-frame crops
    assert len(track.keyframes) == len(tf.frames)
    assert all(abs(k.crop[0] - round(tf.crops[i][0], 2)) < 0.02 for i, k in enumerate(track.keyframes))


def test_grounding_dino_batched_prompts_match_one_forward_per_prompt(cuda):
    """All captions of a frame in one batched forward (padded text) give the same boxes as one
    forward per caption, and the raw low-threshold set is kept for the detection cache."""
    need((models.hf_repo_present(models.GDINO_HF_REPO, "model.safetensors"), "Grounding DINO weights missing"))
    import time

    import torch

    from cappycat_pipeline.perception import GroundedSam2Detector
    from cappycat_pipeline.tracking import iou_matrix

    det = GroundedSam2Detector(with_masks=False)
    img = _img("bus.jpg")
    prompts = ["person", "bus", "animal character"]
    det.batch_prompts = False
    seq = det.detect(img, prompts)
    det.batch_prompts = True
    bat = det.detect(img, prompts)
    assert len(seq) == len(bat) and len(seq) >= 4
    for a in seq:
        same = [b for b in bat if b.label == a.label]
        assert same, a
        iou = iou_matrix(np.array([a.bbox]), np.array([b.bbox for b in same]))[0]
        k = int(iou.argmax())
        assert iou[k] > 0.97 and abs(same[k].score - a.score) < 0.02
    assert det.last_raw is not None and len(det.last_raw) >= len(bat) and det.last_floor <= 0.25

    def timed(flag):
        det.batch_prompts = flag
        det.detect(img, prompts)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            det.detect(img, prompts)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 3

    assert timed(True) < timed(False)
    det.close()
