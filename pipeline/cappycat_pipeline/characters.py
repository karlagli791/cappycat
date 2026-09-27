"""Main-cast identity matching against the user's character reference images.

``<repo>/characters/characters.json`` lists the cast (id, name, species, detection prompts,
reference images; group pictures under ``lineups`` with the characters left to right). This
module turns those images into OpenCLIP reference embeddings and identifies detections by name:

1. **Reference crops** (:func:`build_bank`): every referenced image is run through an
   open-vocabulary detector (Grounding DINO when available - cleaner boxes on illustrations -
   else YOLO-World) with the character's prompts + species + the generic prompts. Small candidate
   boxes (< 25 % of the sheet's height) are kept only when OpenCLIP zero-shot says "a cartoon
   <species> character" rather than an accessory (backpack / cap / ball / book / text / shoe ...);
   full-figure boxes skip that filter (a back view is mostly backpack and used to be rejected).
   A reference entry may also give manual crop boxes (``"boxes": [[x1, y1, x2, y2], ...]`` in
   image pixels; detection is then skipped unless ``"detect": true``). Lineup images keep the N
   best non-overlapping head boxes, sorted by x-centre and assigned in the listed order. Each
   crop is embedded with the *same* OpenCLIP model as the duplicate embedder, plus its horizontal
   flip and (tall crops) its head; near-identical references (cosine > :data:`DEDUPE_COS`) are
   dropped so a sheet's repeated front views don't inflate the top-3 mean.
2. **Open-set rejection**: text embeddings of every character's species ("a cartoon rabbit
   character") and of the manifest's ``negatives`` (text, and/or image crops of supporting
   characters that are *not* cast) plus each character's ``notLike`` text are stored with the
   bank. A detection whose zero-shot label is a negative (or that looks more like a negative
   image reference than like its best cast match) is not given a cast identity.
3. **Cache**: ``characters/.cache/refs.npz`` keyed by the sha1 of the manifest + images + build
   version (written atomically); the chosen crops are written to ``characters/.cache/crops/`` for
   review.
4. **identify** (:meth:`CharacterBank.identify`): per character, score = mean of the top-3
   cosine similarities to its references; a detection is that character when
   ``score >= id_threshold`` and ``score - second_best >= id_margin`` (and it is not rejected).
   Cast tagging along a track uses the looser :data:`CAST_THRESHOLD` / :data:`CAST_MARGIN`.
5. **Calibration** (:func:`calibrate`): leave-one-image-out over the references (lineup crops
   identified with the sheets only, sheet crops with the lineup only) gives the positive /
   best-wrong score distributions; the threshold is placed between them and clamped to
   :data:`VIDEO_THRESHOLD_RANGE` (video frames score lower than clean sheets because the crops
   include scene background), the margin to :data:`MARGIN_RANGE`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import fsutil, models

log = logging.getLogger("cappycat.characters")

BUILD_VERSION = "7"
TALL_ASPECT = 1.3  # only boxes at least this tall (h / w) get a separate head crop
HEAD_FRACTION = 0.45  # "head" crop = top 45 % of a box (face + upper body)
TOPK = 3
DEFAULT_ID_THRESHOLD = 0.72
DEFAULT_ID_MARGIN = 0.05
# identity margin clamp: 0.02 let a group box (deer + turtle + elephant) through as "Suzie" by 0.021
MARGIN_RANGE = (0.05, 0.08)
# Clean character sheets resemble each other (same style, plain background), so the reference-only
# calibration lands high (~0.77). On the real clips correct identities scored 0.72-0.89 with the
# runner-up at 0.59-0.75, so the threshold is clamped into this range.
VIDEO_THRESHOLD_RANGE = (0.68, 0.74)
# Cast tagging (Shot.cast) along a ByteTrack track: looser than the duplicate rule's identity, but
# needs the same best guess in several frames of the track.
CAST_THRESHOLD = 0.66
CAST_MARGIN = 0.05
CAST_MIN_TRACK_FRAMES = 3
DEDUPE_COS = 0.97          # references closer than this to a kept one (same character) are dropped
ACCESSORY_MAX_HEIGHT = 0.25  # the accessory filter only applies to boxes shorter than this (x image height)
ACCESSORY_TEXTS = ["a backpack", "a baseball cap", "a soccer ball", "a notebook", "a book", "text on paper",
                   "a sneaker", "a pair of glasses", "a hair bow", "a color palette", "goggles", "a pencil", "a plain background"]
GENERIC_WORDS = ("character", "creature", "animal", "person", "people", "figure")
NEG_ID = "__negative__"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def default_manifest_path() -> Path:
    env = os.environ.get("CAPPYCAT_CHARACTERS")
    return Path(env) if env else repo_root() / "characters" / "characters.json"


# --------------------------------------------------------------------------- manifest


@dataclass
class Character:
    id: str
    name: str
    species: str
    unique: bool = True
    prompts: List[str] = field(default_factory=list)
    features: str = ""
    references: List[Dict[str, Any]] = field(default_factory=list)
    notLike: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    zero_shot: List[str] = field(default_factory=list)  # manifest "zeroShot": positive descriptions

    @property
    def kinds(self) -> List[str]:
        """Species plus aliases (e.g. Felix: dog, earlier design "fox") for zero-shot filtering."""
        return list(dict.fromkeys([self.species] + list(self.aliases)))


@dataclass
class Manifest:
    path: Path
    characters: List[Character]
    lineups: Dict[str, Dict[str, Any]]
    generic_prompts: List[str]
    # figures that are NOT main cast: [{"text": "a cartoon deer"}, {"image": "x.webp", "boxes": [[..]]}]
    negatives: List[Dict[str, Any]] = field(default_factory=list)

    def zero_shot_texts(self) -> Tuple[List[str], List[str]]:
        """``(texts, owners)`` for the open-set check: per character its species ("a cartoon rabbit
        character"), its ``zeroShot`` descriptions or else "a cartoon <species> character: <features>"
        (the description is what separates e.g. the young Turtle from an elderly turtle), then the
        negatives (owner :data:`NEG_ID`)."""
        texts: List[str] = []
        owners: List[str] = []
        for c in self.characters:
            for kind in c.kinds:
                texts.append(f"a cartoon {kind} character")
                owners.append(c.id)
            desc = list(c.zero_shot) or ([f"a cartoon {c.species} character: {c.features}"] if c.features else [])
            for t in desc:
                texts.append(t)
                owners.append(c.id)
        for t in self.negative_texts():
            texts.append(t)
            owners.append(NEG_ID)
        return texts, owners

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def cache_dir(self) -> Path:
        return self.root / ".cache"

    def by_id(self) -> Dict[str, Character]:
        return {c.id: c for c in self.characters}

    def name_of(self, cid: Optional[str]) -> Optional[str]:
        c = self.by_id().get(cid or "")
        return c.name if c else None

    def detection_prompts(self) -> List[str]:
        """Prompts used to *find* instances in video frames: the manifest's generic prompts plus
        "person" / "cartoon character". Identity comes from the OpenCLIP reference match, not from
        the prompt, and on the real clips species prompts ("cartoon rabbit", ...) had far lower
        recall (YOLO-World mostly returned nothing; Grounding DINO scored < 0.35 with a long
        caption) than generic ones (Grounding DINO "animal character" 0.55-0.83, YOLO-World
        "person" 0.75-0.86)."""
        out: List[str] = []
        for p in list(self.generic_prompts) + ["person", "cartoon character"]:
            if p not in out:
                out.append(p)
        return out

    def character_prompts(self) -> List[str]:
        """Every character's own prompts (used for the reference sheets)."""
        out: List[str] = []
        for c in self.characters:
            for p in c.prompts:
                if p not in out:
                    out.append(p)
        return out

    def images(self) -> List[str]:
        seen: List[str] = []
        for c in self.characters:
            for r in c.references:
                if r.get("image") and r["image"] not in seen:
                    seen.append(r["image"])
        for n in self.negatives:
            if n.get("image") and n["image"] not in seen:
                seen.append(n["image"])
        return seen

    def negative_texts(self) -> List[str]:
        """Zero-shot texts of non-cast figures: the manifest's text negatives + every ``notLike``."""
        out: List[str] = []
        for n in self.negatives:
            if n.get("text") and n["text"] not in out:
                out.append(str(n["text"]))
        for c in self.characters:
            if c.notLike and c.notLike not in out:
                out.append(c.notLike)
        return out

    def fingerprint(self) -> str:
        h = hashlib.sha1()
        h.update(BUILD_VERSION.encode())
        h.update(self.path.read_bytes())
        for im in self.images():
            p = self.root / im
            h.update(im.encode())
            h.update(p.read_bytes() if p.is_file() else b"missing")
        return h.hexdigest()


def load_manifest(path: Optional[os.PathLike] = None) -> Manifest:
    p = Path(path) if path else default_manifest_path()
    d = json.loads(p.read_text(encoding="utf-8"))
    chars = [Character(id=c["id"], name=c.get("name", c["id"]), species=c.get("species", c["id"]),
                       unique=bool(c.get("unique", True)), prompts=list(c.get("prompts", [])),
                       features=c.get("features", ""), references=list(c.get("references", [])), notLike=c.get("notLike"),
                       aliases=list(c.get("aliases", [])), zero_shot=list(c.get("zeroShot", [])))
             for c in d.get("characters", [])]
    negs = []
    for n in d.get("negatives", []):
        negs.append({"text": n} if isinstance(n, str) else dict(n))
    return Manifest(path=p, characters=chars, lineups=dict(d.get("lineups", {})),
                    generic_prompts=list(d.get("genericPrompts", ["animal character"])), negatives=negs)


def is_generic_label(label: str) -> bool:
    lab = (label or "").lower()
    return any(w in lab for w in GENERIC_WORDS)


def labels_compatible(a: str, b: str) -> bool:
    """Same prompt, or either is a generic prompt ("animal character", "person", ...)."""
    return a == b or is_generic_label(a) or is_generic_label(b)


# --------------------------------------------------------------------------- bank


@dataclass
class Identity:
    character: Optional[str]
    score: float
    margin: float
    best: Optional[str] = None  # best-scoring character even when not assigned
    rejected: Optional[str] = None  # open-set rejection reason (the detection is not a cast member)
    zs_best: Optional[str] = None   # zero-shot winner (character id / NEG_ID) over the bank's texts
    zs_prob: float = 0.0

    def loose(self, threshold: float = CAST_THRESHOLD, margin: float = CAST_MARGIN) -> Optional[str]:
        """The best guess when it passes the looser cast-tagging rule: ``score >= threshold`` and a
        clear ``margin`` - or, instead of the margin, a zero-shot vote for the same character
        (>= :data:`ZS_AGREE`), e.g. a back view that scores 0.67 vs 0.63 on the references but is
        unmistakably "a cartoon rabbit" for the text tower. None otherwise."""
        if self.rejected or not self.best:
            return None
        agree = self.zs_best == self.best
        if self.score >= threshold and (self.margin >= margin or (agree and self.zs_prob >= ZS_AGREE)):
            return self.best
        # a back view scores low on the references (0.64 for Bunny's back in clip4) while the text
        # tower is certain: accept it one tier lower when the two agree strongly
        if agree and self.score >= CAST_THRESHOLD_ZS and self.zs_prob >= ZS_STRONG:
            return self.best
        return None


ZS_AGREE = 0.8  # zero-shot probability that stands in for the identity margin in the cast-tagging rule
ZS_STRONG = 0.95  # ... and that lowers the cast-tagging threshold to CAST_THRESHOLD_ZS
CAST_THRESHOLD_ZS = 0.62
ZS_REJECT = 0.5  # zero-shot probability of "not cast" (negatives) that rejects an identity


@dataclass
class CharacterBank:
    manifest: Manifest
    ids: List[str]                 # per reference embedding
    embeddings: np.ndarray         # (n, d) L2-normalised
    sources: List[str]             # per reference: "<image>#<n>[f]"
    id_threshold: float = DEFAULT_ID_THRESHOLD
    id_margin: float = DEFAULT_ID_MARGIN
    calibration: Dict[str, Any] = field(default_factory=dict)
    # open-set rejection: text embeddings (species of each character + negative texts) and image
    # embeddings of non-cast figures
    text_embeddings: Optional[np.ndarray] = None   # (t, d)
    text_owner: List[str] = field(default_factory=list)  # per text: character id or NEG_ID
    texts: List[str] = field(default_factory=list)
    neg_embeddings: Optional[np.ndarray] = None    # (k, d)
    neg_sources: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._ids_arr = np.array(self.ids)
        self._uniq = list(dict.fromkeys(self.ids))
        self._masks = {cid: self._ids_arr == cid for cid in self._uniq}
        # reference crop each embedding belongs to (a crop and its flipped / head variants count once
        # in the top-k mean, so a symmetric pose does not fill the top 3 with copies of itself)
        groups = [crop_group(s) for s in self.sources]
        index = {g: i for i, g in enumerate(dict.fromkeys(groups))}
        self._group = np.array([index[g] for g in groups], dtype=np.int64)
        self._n_groups = len(index)

    @property
    def character_ids(self) -> List[str]:
        return [c.id for c in self.manifest.characters if c.id in set(self.ids)]

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for i in self.ids:
            out[i] = out.get(i, 0) + 1
        return out

    def _scores_one(self, emb: np.ndarray, mask: Optional[np.ndarray]) -> Dict[str, float]:
        e = np.asarray(emb, dtype=np.float32).ravel()
        e = e / (np.linalg.norm(e) + 1e-9)
        sims = self.embeddings @ e
        out: Dict[str, float] = {}
        for cid in self._uniq:
            sel = self._masks[cid]
            if mask is not None:
                sel = sel & mask
            if not sel.any():
                continue
            best = np.full(self._n_groups, -np.inf, dtype=np.float32)
            np.maximum.at(best, self._group[sel], sims[sel])
            best = best[np.isfinite(best)]
            s = np.sort(best)[::-1][:TOPK]
            out[cid] = float(s.mean())
        return out

    def scores(self, emb: np.ndarray, mask: Optional[np.ndarray] = None,
               head_emb: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Per-character score = mean of the top-``TOPK`` cosine similarities to its references
        (full-body, head and flipped crops); with ``head_emb`` (the top 45 % of the detection) the
        max of the full-box and head scores, so a costume change or a partly hidden body still
        identifies by the face."""
        out = self._scores_one(emb, mask)
        if head_emb is not None:
            for cid, v in self._scores_one(head_emb, mask).items():
                out[cid] = max(out.get(cid, 0.0), v)
        return out

    def zero_shot(self, emb: np.ndarray) -> Optional[Dict[str, float]]:
        """Softmax over the stored texts, summed per owner (character id / NEG_ID); None without texts."""
        if self.text_embeddings is None or not len(self.text_embeddings):
            return None
        e = np.asarray(emb, dtype=np.float32).ravel()
        e = e / (np.linalg.norm(e) + 1e-9)
        logits = 100.0 * (self.text_embeddings @ e)
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        out: Dict[str, float] = {}
        for owner, v in zip(self.text_owner, p):
            out[owner] = out.get(owner, 0.0) + float(v)
        return out

    def negative_score(self, emb: np.ndarray, head_emb: Optional[np.ndarray] = None) -> float:
        """Top-``TOPK`` mean similarity to the negative (non-cast) image references (0 without)."""
        if self.neg_embeddings is None or not len(self.neg_embeddings):
            return 0.0
        best = 0.0
        for x in (emb, head_emb):
            if x is None:
                continue
            e = np.asarray(x, dtype=np.float32).ravel()
            e = e / (np.linalg.norm(e) + 1e-9)
            s = np.sort(self.neg_embeddings @ e)[::-1][:TOPK]
            best = max(best, float(s.mean()))
        return best

    def open_set_reject(self, emb: np.ndarray, best_id: Optional[str], best: float,
                        head_emb: Optional[np.ndarray] = None) -> Optional[str]:
        """Reason why a detection whose best cast match is ``best_id`` is *not* a cast member, or None:
        its zero-shot label is a negative text rather than the character's species, or it is closer
        to a negative image reference than to ``best_id``'s references."""
        if best_id is None:
            return None
        zs = self.zero_shot(emb)
        if zs is not None and NEG_ID in zs:
            p_neg, p_own = zs.get(NEG_ID, 0.0), zs.get(best_id, 0.0)
            if p_neg >= ZS_REJECT and p_neg > p_own:
                return f"zero-shot: not-cast {p_neg:.2f} vs {best_id} {p_own:.2f}"
        ns = self.negative_score(emb, head_emb)
        if ns > 0 and ns >= best:
            return f"closer to a negative reference ({ns:.3f} >= {best:.3f})"
        return None

    def identify_one(self, emb: np.ndarray, mask: Optional[np.ndarray] = None, threshold: Optional[float] = None,
                     margin: Optional[float] = None, head_emb: Optional[np.ndarray] = None) -> Identity:
        sc = self.scores(emb, mask, head_emb)
        if not sc:
            return Identity(None, 0.0, 0.0)
        ranked = sorted(sc.items(), key=lambda kv: -kv[1])
        best_id, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        thr = self.id_threshold if threshold is None else threshold
        mar = self.id_margin if margin is None else margin
        ok = best >= thr and (best - second) >= mar
        rejected = None
        zs_best, zs_prob = None, 0.0
        if best >= CAST_THRESHOLD_ZS:
            zs = self.zero_shot(emb)
            if zs:
                zs_best, zs_prob = max(zs.items(), key=lambda kv: kv[1])
            rejected = self.open_set_reject(emb, best_id, best, head_emb)
        return Identity(best_id if ok and not rejected else None, round(best, 4), round(best - second, 4), best_id,
                        rejected, zs_best, round(float(zs_prob), 4))

    def identify(self, embeddings: np.ndarray, head_embeddings: Optional[Sequence[Optional[np.ndarray]]] = None
                 ) -> List[Identity]:
        if len(embeddings) == 0:
            return []
        d = self.embeddings.shape[1]
        embs = np.asarray(embeddings, dtype=np.float32).reshape(-1, d)
        heads = list(head_embeddings) if head_embeddings is not None else [None] * len(embs)
        return [self.identify_one(e, head_emb=hd) for e, hd in zip(embs, heads)]

    def save(self, path: Path, key: str) -> None:
        d = self.embeddings.shape[1] if self.embeddings.ndim == 2 else 512
        fsutil.save_npz_atomic(
            path, key=np.array(key), ids=np.array(self.ids), embeddings=self.embeddings.astype(np.float32),
            sources=np.array(self.sources), id_threshold=np.array(self.id_threshold),
            id_margin=np.array(self.id_margin), calibration=np.array(json.dumps(self.calibration)),
            text_embeddings=(self.text_embeddings if self.text_embeddings is not None else np.zeros((0, d))).astype(np.float32),
            text_owner=np.array(self.text_owner, dtype=str), texts=np.array(self.texts, dtype=str),
            neg_embeddings=(self.neg_embeddings if self.neg_embeddings is not None else np.zeros((0, d))).astype(np.float32),
            neg_sources=np.array(self.neg_sources, dtype=str))


def _load_cached(manifest: Manifest, key: str) -> Optional[CharacterBank]:
    path = manifest.cache_dir / "refs.npz"
    if not path.is_file():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if str(z["key"]) != key:
            return None
        te = z["text_embeddings"].astype(np.float32) if "text_embeddings" in z else None
        ne = z["neg_embeddings"].astype(np.float32) if "neg_embeddings" in z else None
        return CharacterBank(manifest, [str(x) for x in z["ids"]], z["embeddings"].astype(np.float32),
                             [str(x) for x in z["sources"]], float(z["id_threshold"]), float(z["id_margin"]),
                             json.loads(str(z["calibration"])),
                             text_embeddings=te if te is not None and len(te) else None,
                             text_owner=[str(x) for x in z["text_owner"]] if "text_owner" in z else [],
                             texts=[str(x) for x in z["texts"]] if "texts" in z else [],
                             neg_embeddings=ne if ne is not None and len(ne) else None,
                             neg_sources=[str(x) for x in z["neg_sources"]] if "neg_sources" in z else [])
    except Exception as exc:
        log.warning("character cache unreadable (%s); rebuilding", exc)
        return None


def cache_state(manifest: Manifest) -> Dict[str, Any]:
    path = manifest.cache_dir / "refs.npz"
    out: Dict[str, Any] = {"manifest": str(manifest.path), "characters": [c.id for c in manifest.characters],
                           "cache": str(path), "cacheFresh": False}
    try:
        out["missingImages"] = [im for im in manifest.images() if not (manifest.root / im).is_file()]
        bank = _load_cached(manifest, manifest.fingerprint())
        if bank is not None:
            out.update(cacheFresh=True, references=bank.counts(), idThreshold=bank.id_threshold, idMargin=bank.id_margin,
                       negatives={"texts": len(bank.texts) - sum(1 for o in bank.text_owner if o != NEG_ID),
                                  "images": len(bank.neg_sources)})
    except Exception as exc:
        out["error"] = str(exc)
    return out


# --------------------------------------------------------------------------- building


def is_tall(b: Sequence[float]) -> bool:
    return (b[3] - b[1]) >= TALL_ASPECT * max(b[2] - b[0], 1e-6)


def head_box(b: Sequence[float]) -> Tuple[float, float, float, float]:
    """Top ``HEAD_FRACTION`` of a tall (full-body) box; the box itself otherwise (a bust / face
    box already is the head)."""
    x1, y1, x2, y2 = (float(v) for v in b)
    if not is_tall(b):
        return (x1, y1, x2, y2)
    return (x1, y1, x2, y1 + HEAD_FRACTION * (y2 - y1))


def head_crop(patch: np.ndarray) -> Optional[np.ndarray]:
    h, w = patch.shape[:2]
    if not is_tall((0, 0, w, h)):
        return None
    return patch[: max(8, int(round(h * HEAD_FRACTION)))]


def _read_image(path: Path) -> np.ndarray:
    import cv2

    img = fsutil.imread(path)  # non-ASCII-safe (cv2.imread fails silently on such paths)
    if img is None:  # e.g. an OpenCV build without webp
        from PIL import Image  # type: ignore

        img = cv2.cvtColor(np.asarray(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    return img


def _containment(a: Sequence[float], b: Sequence[float]) -> float:
    """Fraction of ``a``'s area inside ``b``."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


@dataclass
class _Cand:
    bbox: Tuple[float, float, float, float]
    score: float
    label: str
    char_prob: float = 0.0


def _candidates(img: np.ndarray, prompts: Sequence[str], detector, embedder, species: Sequence[str],
                min_area_frac: float = 0.012) -> List[_Cand]:
    from .perception import _crop, nms

    h, w = img.shape[:2]
    # one caption per prompt (better recall with Grounding DINO; batched into one forward), then
    # class-agnostic NMS
    dets = nms(list(detector.detect(img, list(dict.fromkeys(prompts)))), 0.7, class_agnostic=True)
    out: List[_Cand] = []
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        bw, bh = x2 - x1, y2 - y1
        if bw * bh < min_area_frac * w * h or bw / max(bh, 1) > 2.5 or bh / max(bw, 1) > 5.0:
            continue
        out.append(_Cand(tuple(float(v) for v in d.bbox), float(d.score), d.label))
    if not out:
        return out
    texts = [f"a cartoon {s} character" for s in species] + ["a cartoon animal character"] + ACCESSORY_TEXTS
    n_char = len(species) + 1
    probs = embedder.zero_shot([_crop(img, c.bbox, pad=0.02) for c in out], texts)
    for c, p in zip(out, probs):
        c.char_prob = float(p[:n_char].sum())
    # accessories (ball, cap, backpack in the "accessories" panel) are small; a full figure seen from
    # the back is mostly backpack for CLIP, so the filter only applies to small boxes
    return [c for c in out if c.char_prob >= 0.5 or (c.bbox[3] - c.bbox[1]) >= ACCESSORY_MAX_HEIGHT * h]


def _dedupe_nested(cands: List[_Cand], contain: float = 0.8) -> List[_Cand]:
    """Drop boxes mostly inside a bigger kept box (head inside body)."""
    keep: List[_Cand] = []
    for c in sorted(cands, key=lambda c: -(c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1])):
        if any(_containment(c.bbox, k.bbox) > contain for k in keep):
            continue
        keep.append(c)
    return keep


HEAD_PROMPTS = ["animal face", "cartoon animal head"]


def _lineup_heads(img: np.ndarray, n: int, detector, embedder) -> List[_Cand]:
    """The ``n`` best non-nested head / face boxes, left to right (heads stay separable in a
    group picture where bodies overlap and get merged into one box)."""
    from .perception import nms

    h, w = img.shape[:2]
    # one caption per prompt: Grounding DINO's boxes change when prompts share a caption
    dets = nms(list(detector.detect(img, HEAD_PROMPTS)), 0.5, class_agnostic=True)
    cands = [_Cand(tuple(float(v) for v in d.bbox), float(d.score), d.label) for d in dets
             if 0.004 * w * h <= (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) <= 0.10 * w * h]
    keep: List[_Cand] = []
    for c in sorted(cands, key=lambda c: -(c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1])):
        if any(_containment(c.bbox, k.bbox) > 0.6 for k in keep):
            continue
        keep.append(c)
    keep = sorted(keep, key=lambda c: -c.score)[:n]
    return sorted(keep, key=lambda c: (c.bbox[0] + c.bbox[2]) / 2.0)


def _lineup_crops(img: np.ndarray, order: Sequence[str], chars: Dict[str, Character], detector, embedder,
                  say: Callable[[str], None], name: str) -> List[Tuple[str, np.ndarray]]:
    """[(character_id, crop)] for a lineup: a head crop per character (heads sorted left to right
    and assigned in the listed order). Body boxes are not used: in a group picture they include
    the neighbours and pulled identities towards whoever stands next to them."""
    from .perception import _crop

    heads = _lineup_heads(img, len(order), detector, embedder)
    if len(heads) != len(order):
        say(f"{name}: {len(heads)} head(s) for a lineup of {len(order)}; using per-character body boxes instead")
        return [(cid, _crop(img, c.bbox, pad=0.02))
                for cid, c in zip(order, _assign_lineup(img, order, chars, detector, embedder)) if c is not None]
    return [(cid, _crop(img, hd.bbox, pad=0.12)) for cid, hd in zip(order, heads)]


def _assign_lineup(img: np.ndarray, order: Sequence[str], chars: Dict[str, Character], detector, embedder
                   ) -> List[Optional[_Cand]]:
    """Fallback: one box per lineup character. Each character is searched with its *own* prompts (the
    text-conditioned detector ranks the right animal first), then a DP picks one candidate per
    character maximising the summed score subject to strictly increasing x-centres (the listed
    left-to-right order). Characters without a consistent box get None."""
    per: List[List[_Cand]] = []
    for cid in order:
        ch = chars[cid]
        cands = _candidates(img, list(ch.prompts) + [f"cartoon {ch.species}"], detector, embedder, ch.kinds,
                            min_area_frac=0.02)
        per.append(sorted(cands, key=lambda c: -c.score)[:6])
    n = len(order)
    NEG = -1e9
    # best[i][j]: best total for characters[:i+1] with character i using candidate j
    best: List[List[float]] = []
    back: List[List[Tuple[int, int]]] = []  # (previous character index, previous candidate) or (-1, -1)
    for i in range(n):
        row, brow = [], []
        for j, c in enumerate(per[i]):
            cx = (c.bbox[0] + c.bbox[2]) / 2.0
            val, arg = c.score * c.char_prob, (-1, -1)
            # link to the closest assigned predecessor k < i (predecessors may be skipped)
            top = 0.0
            for k in range(i - 1, -1, -1):
                for jj, pc in enumerate(per[k]):
                    if best[k][jj] > NEG / 2 and (pc.bbox[0] + pc.bbox[2]) / 2.0 < cx - 1.0 and best[k][jj] > top:
                        top, arg = best[k][jj], (k, jj)
            row.append(val + top)
            brow.append(arg)
        best.append(row)
        back.append(brow)
    # trace back from the best end state
    end, end_val = (-1, -1), NEG
    for i in range(n):
        for j, v in enumerate(best[i]):
            if v > end_val:
                end, end_val = (i, j), v
    chosen: List[Optional[_Cand]] = [None] * n
    while end[0] >= 0:
        i, j = end
        chosen[i] = per[i][j]
        end = back[i][j]
    return chosen


def _manual_boxes(ref: Dict[str, Any], img: np.ndarray) -> List[Tuple[float, float, float, float]]:
    """Manual crop boxes of a reference entry (image pixels, clamped; degenerate boxes dropped)."""
    h, w = img.shape[:2]
    out = []
    for b in ref.get("boxes") or []:
        try:
            x1, y1, x2, y2 = (float(v) for v in b)
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((max(0.0, min(w, x1)), max(0.0, min(w, x2))))
        y1, y2 = sorted((max(0.0, min(h, y1)), max(0.0, min(h, y2))))
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            out.append((x1, y1, x2, y2))
    return out


def extract_reference_crops(manifest: Manifest, detector, embedder,
                            log_fn: Optional[Callable[[str], None]] = None) -> List[Tuple[str, str, np.ndarray, bool]]:
    """[(character_id, source_image, crop_bgr, is_query)] for every reference image. ``is_query``:
    the crop is also a calibration query (lineup heads and sheet crops that zero-shot recognised as
    the character); full figures that only got in because they are big (back views) and manual
    boxes are references only, so the calibration measures the same population as before."""
    from .perception import _crop

    say = log_fn or (lambda m: log.info(m))
    chars = manifest.by_id()
    generic = list(manifest.generic_prompts)
    crops: List[Tuple[str, str, np.ndarray, bool]] = []
    cast_images = [im for c in manifest.characters for r in c.references for im in [r.get("image")] if im]
    for im_name in dict.fromkeys(cast_images):
        path = manifest.root / im_name
        if not path.is_file():
            say(f"reference image missing: {path}")
            continue
        img = _read_image(path)
        lineup = manifest.lineups.get(im_name)
        if lineup:
            order = [cid for cid in lineup.get("order", []) if cid in chars]
            for cid, patch in _lineup_crops(img, order, chars, detector, embedder, say, im_name):
                crops.append((cid, im_name, patch, True))
            continue
        for ch in manifest.characters:
            refs = [r for r in ch.references if r.get("image") == im_name and "lineupIndex" not in r]
            if not refs:
                continue
            manual = [b for r in refs for b in _manual_boxes(r, img)]
            detect = not manual or any(bool(r.get("detect")) for r in refs)
            boxes: List[Tuple[float, float, float, float]] = []
            query: List[bool] = []
            if detect:
                prompts = list(ch.prompts) + [k for kind in ch.kinds for k in (kind, f"cartoon {kind}")] + generic
                for c in _candidates(img, prompts, detector, embedder, ch.kinds):
                    boxes.append(c.bbox)
                    query.append(c.char_prob >= 0.5)
            # manual boxes are added unless a detected box already covers them (IoU > 0.7)
            from .tracking import iou_matrix

            for b in manual:
                if not boxes or iou_matrix(np.array([b]), np.array(boxes)).max() <= 0.7:
                    boxes.append(b)
                    query.append(False)
            for b, q in zip(boxes, query):
                crops.append((ch.id, im_name, _crop(img, b, pad=0.02), q))
            if manual:
                say(f"{im_name}: {len(manual)} manual box(es) for {ch.id}")
    return crops


def extract_negative_crops(manifest: Manifest) -> List[Tuple[str, np.ndarray]]:
    """[(source, crop)] for image negatives (manifest ``negatives`` entries with an ``image``; the
    whole image when no ``boxes`` are given)."""
    from .perception import _crop

    out: List[Tuple[str, np.ndarray]] = []
    for n in manifest.negatives:
        im_name = n.get("image")
        if not im_name:
            continue
        path = manifest.root / im_name
        if not path.is_file():
            log.warning("negative reference image missing: %s", path)
            continue
        img = _read_image(path)
        boxes = _manual_boxes(n, img) or [(0.0, 0.0, float(img.shape[1]), float(img.shape[0]))]
        for k, b in enumerate(boxes):
            out.append((f"{im_name}#{k}", _crop(img, b, pad=0.0)))
    return out


def calibrate(ids: Sequence[str], embeddings: np.ndarray, sources: Sequence[str], manifest: Manifest) -> Dict[str, Any]:
    """Leave-one-image-out identification over the reference crops: lineup crops are identified
    with the sheets only, sheet crops with the lineup only (references from the query's own image
    are always excluded). Queries are the un-flipped full crops, scored with their head crop like
    a video detection. Returns the score distributions, per-character accuracy and the chosen
    threshold / margin."""
    ids_a = np.array(ids)
    src_img = np.array([s.split("#")[0] for s in sources])
    tags = [s.split("#")[1] if "#" in s else "" for s in sources]
    is_lineup = np.array([im in manifest.lineups for im in src_img])
    index = {s: i for i, s in enumerate(sources)}
    bank = CharacterBank(manifest, list(ids), embeddings, list(sources))
    pos, neg, margins, rows = [], [], [], []
    per: Dict[str, List[int]] = {}
    for i in range(len(ids)):
        if not tags[i].isdigit():
            continue  # flipped / head copies are references, not queries
        mask = ~is_lineup if is_lineup[i] else is_lineup.copy()
        mask &= src_img != src_img[i]
        if not (mask & (ids_a == ids_a[i])).any():
            continue  # no reference of this character on the other side
        hi = index.get(f"{src_img[i]}#{tags[i]}h")
        sc = bank.scores(embeddings[i], mask, embeddings[hi] if hi is not None else None)
        ranked = sorted(sc.items(), key=lambda kv: -kv[1])
        true_score = sc.get(ids_a[i], 0.0)
        wrong = max((v for k, v in sc.items() if k != ids_a[i]), default=0.0)
        pos.append(true_score)
        neg.append(wrong)
        margins.append(true_score - wrong)
        ok = ranked[0][0] == ids_a[i]
        per.setdefault(str(ids_a[i]), []).append(int(ok))
        rows.append({"source": sources[i], "true": str(ids_a[i]), "predicted": ranked[0][0], "score": round(true_score, 4),
                     "bestWrong": round(wrong, 4)})
    if not pos:
        return {"idThreshold": DEFAULT_ID_THRESHOLD, "idMargin": DEFAULT_ID_MARGIN, "queries": 0}
    pos_a, neg_a = np.array(pos), np.array(neg)
    # between the 10th percentile of correct scores and the 90th of best-wrong scores, clamped for video
    raw_thr = float((np.percentile(pos_a, 10) + np.percentile(neg_a, 90)) / 2.0)
    thr = float(np.clip(raw_thr, *VIDEO_THRESHOLD_RANGE))
    good = np.array(margins)[np.array(margins) > 0]
    margin = float(np.clip(np.percentile(good, 10) / 2.0 if len(good) else DEFAULT_ID_MARGIN, *MARGIN_RANGE))
    total = len(rows)
    correct = sum(sum(v) for v in per.values())
    return {"queries": total, "top1Accuracy": round(correct / total, 4),
            "perCharacter": {k: {"queries": len(v), "top1Accuracy": round(sum(v) / len(v), 4)} for k, v in per.items()},
            "positive": {"min": round(float(pos_a.min()), 4), "p10": round(float(np.percentile(pos_a, 10)), 4),
                         "median": round(float(np.median(pos_a)), 4)},
            "bestWrong": {"median": round(float(np.median(neg_a)), 4), "p90": round(float(np.percentile(neg_a, 90)), 4),
                          "max": round(float(neg_a.max()), 4)},
            "referenceThreshold": round(raw_thr, 4), "videoRange": list(VIDEO_THRESHOLD_RANGE),
            "idThreshold": round(thr, 4), "idMargin": round(margin, 4), "rows": rows}


def crop_group(source: str) -> str:
    """``"sheet.webp#3rhf"`` -> ``"sheet.webp#3r"``: the reference crop an embedding variant came from."""
    im, _, tag = source.partition("#")
    m = re.match(r"\d+r?", tag)
    return f"{im}#{m.group(0) if m else tag}"


def dedupe_references(ids: Sequence[str], sources: Sequence[str], embs: np.ndarray,
                      cos: float = DEDUPE_COS) -> List[int]:
    """Indices to keep: a reference *crop* (with all its flipped / head variants) is dropped when the
    un-flipped full view of an already kept crop of the same character is more similar than ``cos``
    (a sheet's "front" and "neutral / pleasant" views are near-identical and would otherwise count
    twice in the top-3 mean). Calibration-query crops are visited first so they survive."""
    e = np.asarray(embs, dtype=np.float32)
    groups = [crop_group(s) for s in sources]
    full = {}  # group -> index of its un-flipped full view
    for i, s in enumerate(sources):
        tag = s.partition("#")[2]
        if re.fullmatch(r"\d+r?", tag):
            full[groups[i]] = i

    def rank(g: str) -> Tuple[int, int]:
        i = full[g]
        return (0 if sources[i].partition("#")[2].isdigit() else 1, i)

    kept_groups: List[str] = []
    for g in sorted(full, key=rank):
        i = full[g]
        same = [full[k] for k in kept_groups if ids[full[k]] == ids[i]]
        if same and float((e[same] @ e[i]).max()) > cos:
            continue
        kept_groups.append(g)
    kg = set(kept_groups)
    return [i for i in range(len(ids)) if groups[i] in kg or groups[i] not in full]


def _reference_detector():
    """Grounding DINO boxes (no masks) when available, else YOLO-World at a low confidence."""
    from . import perception

    ok, _ = perception.gsam_available()
    if ok:
        try:
            return perception.GroundedSam2Detector(box_threshold=0.25, text_threshold=0.2, with_masks=False)
        except perception.MissingDependency as exc:
            log.info("Grounding DINO unavailable for reference crops (%s); using YOLO-World", exc)
    return perception.YoloWorldDetector(conf=0.05)


def build_bank(manifest: Manifest, force: bool = False, embedder=None,
               log_fn: Optional[Callable[[str], None]] = None) -> CharacterBank:
    """Load the cached reference bank, or (re)build it. Loads its own detector and a
    text-capable OpenCLIP for the zero-shot filter, and frees both afterwards."""
    import cv2

    from . import perception

    key = manifest.fingerprint()
    if not force:
        cached = _load_cached(manifest, key)
        if cached is not None:
            return cached
    say = log_fn or (lambda m: log.info(m))
    say(f"building character references from {manifest.path}")
    clip = perception.OpenClipEmbedder(keep_text=True)
    det = _reference_detector()
    try:
        crops = extract_reference_crops(manifest, det, clip, say)
    finally:
        perception.close_detector(det)
    ids: List[str] = []
    sources: List[str] = []
    patches: List[np.ndarray] = []
    per_char: Dict[str, int] = {}
    full_crops: List[Tuple[str, np.ndarray]] = []
    for cid, im_name, patch, is_query in crops:
        n = per_char.get(cid, 0)
        per_char[cid] = n + 1
        full_crops.append((f"{cid}_{n}", patch))
        head = head_crop(patch)
        variants = [("", patch), ("f", cv2.flip(patch, 1))]
        if head is not None:
            variants += [("h", head), ("hf", cv2.flip(head, 1))]
        nt = f"{n}" if is_query else f"{n}r"  # "r": reference only (not a calibration query)
        for tag, im in variants:
            ids.append(cid)
            sources.append(f"{im_name}#{nt}{tag}")
            patches.append(im)
    embs = clip.embed_images(patches) if patches else np.zeros((0, 512), np.float32)
    keep = dedupe_references(ids, sources, embs)
    dropped = len(ids) - len(keep)
    ids = [ids[i] for i in keep]
    sources = [sources[i] for i in keep]
    embs = embs[keep] if len(keep) else embs[:0]
    if dropped:
        say(f"character references: dropped {dropped} near-duplicate reference(s) (cos > {DEDUPE_COS})")
    # open-set rejection data: species / negative texts and negative image crops
    texts, owners = manifest.zero_shot_texts()
    text_embs = clip.embed_texts(texts) if texts else None
    neg = extract_negative_crops(manifest)
    neg_embs = None
    neg_sources: List[str] = []
    if neg:
        neg_patches = [p for _, p in neg] + [cv2.flip(p, 1) for _, p in neg]
        neg_sources = [s for s, _ in neg] + [s + "f" for s, _ in neg]
        neg_embs = clip.embed_images(neg_patches)
    clip.close()
    del clip
    models.release()
    cal = calibrate(ids, embs, sources, manifest)
    bank = CharacterBank(manifest, ids, embs.astype(np.float32), sources, float(cal["idThreshold"]),
                         float(cal["idMargin"]), cal, text_embeddings=text_embs,
                         text_owner=owners if text_embs is not None else [], texts=texts if text_embs is not None else [],
                         neg_embeddings=neg_embs, neg_sources=neg_sources)
    manifest.cache_dir.mkdir(parents=True, exist_ok=True)
    ignore = manifest.cache_dir / ".gitignore"
    if not ignore.is_file():
        fsutil.write_text_atomic(ignore, "*\n")
    bank.save(manifest.cache_dir / "refs.npz", key)
    # review crops (non-ASCII-safe writes); written after the bank so a failure here cannot lose it
    crop_dir = manifest.cache_dir / "crops"
    if crop_dir.is_dir():
        for old in crop_dir.glob("*.jpg"):
            try:
                old.unlink()
            except OSError:
                pass
    crop_dir.mkdir(parents=True, exist_ok=True)
    kept_full = {s for s in sources if "#" in s and s.split("#")[1].rstrip("r").isdigit()}
    for (name, patch), (cid, im_name, _, is_query) in zip(full_crops, crops):
        n = name.rsplit("_", 1)[1] + ("" if is_query else "r")
        tag = ("" if is_query else "_ref") + ("" if f"{im_name}#{n}" in kept_full else "_dropped")
        fsutil.imwrite(crop_dir / f"{name}{tag}.jpg", patch)
    for s, p in neg:
        fsutil.imwrite(crop_dir / f"negative_{Path(s.split('#')[0]).stem}_{s.split('#')[1]}.jpg", p)
    say(f"character references: {bank.counts()}; idThreshold={bank.id_threshold} idMargin={bank.id_margin}")
    return bank


def load_bank(path: Optional[os.PathLike] = None, force: bool = False,
              log_fn: Optional[Callable[[str], None]] = None) -> Optional[CharacterBank]:
    """Manifest + bank, or None when there is no manifest."""
    p = Path(path) if path else default_manifest_path()
    if not p.is_file():
        return None
    return build_bank(load_manifest(p), force=force, log_fn=log_fn)
