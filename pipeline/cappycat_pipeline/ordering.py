"""Clip ordering from filenames.

Users name their clips so the editor knows the story order. This module reads that
intent from the filename and explains every decision, instead of relying on a plain
alphabetical sort (which puts ``clip10`` before ``clip2`` and ignores words like
"intro", "second" or "Part II").

Signals, strongest first:

1. **Hierarchical markers** - ``ep1 scene2 shot3``, ``S01E02``, ``act II``, ``part 3``,
   ``clip_04``, ``#5``, ``chapter three``. Numbers may be digits, Roman numerals or words.
   Missing lower levels count as 0, so ``scene2`` sorts before ``scene2_shot1``.
2. **Leading number** - ``01 intro.mp4``, ``3-kitchen.mp4``, ``002_raccoon.mp4``.
3. **Ordinal words** - ``first``, ``second`` ... ``twentieth``, ``1st``, ``2nd``.
4. **Trailing number** - ``raccoon_kitchen_3.mp4``, ``Clip (2).mp4``.
5. **Timestamps** - generator names like ``kling_20260924_153012.mp4``.
6. **Position words** - ``intro/opening/prologue/teaser`` go first, ``outro/ending/finale/
   epilogue/credits`` go last, when a file has no number.

The scheme is chosen for the whole folder: the strongest signal that every file (apart
from pure position-word files) carries wins. Ties and gaps are reported as warnings.
Letter suffixes (``1a``, ``1b``) and a natural filename sort break remaining ties.
Version tags (``v2``, ``final_v3``), resolutions (``1080p``, ``4k``), frame rates
(``24fps``), dimensions (``1920x1080``) and long hex ids are ignored.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".ts", ".mts"}
AUDIO_EXTS = {".mp3", ".wav", ".aac", ".flac", ".m4a", ".ogg", ".opus"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
LUT_EXTS = {".cube"}
MEDIA_EXTS = VIDEO_EXTS | AUDIO_EXTS | IMAGE_EXTS | LUT_EXTS

# marker word -> hierarchy level (lower = more significant)
MARKERS: Dict[str, int] = {}
for _lvl, _words in enumerate([
    ["season"],
    ["ep", "episode", "eps"],
    ["act", "book", "volume", "vol"],
    ["chapter", "ch", "chap"],
    ["scene", "sc", "scn", "sequence", "seq", "sq"],
    ["part", "pt", "section", "sec", "segment", "seg", "block"],
    ["shot", "sh", "clip", "cut", "beat", "step", "panel", "vid", "video", "no", "num", "number", "nr", "#"],
    ["take", "tk"],
]):
    for _w in _words:
        MARKERS[_w] = _lvl
LEVEL_NAMES = ["season", "episode", "act", "chapter", "scene", "part", "shot", "take"]

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20,
}
START_WORDS = {"intro", "introduction", "opening", "opener", "open", "prologue", "teaser", "coldopen", "beginning",
               "begin", "start", "hook", "cold"}
END_WORDS = {"outro", "ending", "end", "finale", "final", "epilogue", "credits", "closing", "conclusion", "last",
             "wrap", "wrapup", "fin"}
ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}

_IGNORE_PATTERNS = [
    re.compile(r"\b\d{3,4}x\d{3,4}\b"),        # 1920x1080
    re.compile(r"\b\d{1,2}x\d{1,2}\b"),         # 16x9 aspect
    re.compile(r"\b\d{3,4}p\b"),                # 1080p
    re.compile(r"\b[248]k\b"),                  # 4k
    re.compile(r"\b\d{2,3}\s?fps\b"),           # 24fps
    re.compile(r"\bv(?:er(?:sion)?)?\s?\d+\b"), # v2, ver3, version 4
    re.compile(r"\b[0-9a-f]*[a-f][0-9a-f]*\d[0-9a-f]*\b|\b[0-9a-f]*\d[0-9a-f]*[a-f][0-9a-f]*\b"),  # hex-ish ids (filtered by length below)
]
_TIMESTAMP = re.compile(r"(?<!\d)(20\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?([0-2]\d|3[01])(?:[-_. t]?([01]\d|2[0-3])[-_.:]?([0-5]\d)(?:[-_.:]?([0-5]\d))?)?(?!\d)")
_SXXEYY = re.compile(r"\bs(\d{1,3})\s?e(\d{1,4})\b")


@dataclass
class OrderInfo:
    path: str
    name: str
    order: int = 0
    reason: str = ""
    key: List[float] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"path": self.path, "name": self.name, "order": self.order, "reason": self.reason, "key": self.key}


@dataclass
class _Parsed:
    path: Path
    markers: Dict[int, float]            # level -> number
    marker_words: Dict[int, str]
    leading: Optional[float]
    ordinal: Optional[float]
    trailing: Optional[float]
    timestamp: Optional[float]
    position: int                        # -1 start word, +1 end word, 0 none
    suffix: str                          # letter suffix after the chosen number ("a", "b")
    natural: List[object]


def natural_key(name: str) -> List[object]:
    return [(0, int(t)) if t.isdigit() else (1, t.lower()) for t in re.split(r"(\d+)", name) if t]


def _roman(tok: str) -> Optional[int]:
    tok = tok.lower()
    if not tok or any(ch not in ROMAN for ch in tok) or len(tok) > 6:
        return None
    total, prev = 0, 0
    for ch in reversed(tok):
        v = ROMAN[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total if 0 < total <= 100 else None


def _number(tok: str, allow_roman: bool) -> Optional[float]:
    if tok.isdigit():
        return float(int(tok))
    if tok in NUMBER_WORDS:
        return float(NUMBER_WORDS[tok])
    if tok in ORDINAL_WORDS:
        return float(ORDINAL_WORDS[tok])
    m = re.fullmatch(r"(\d+)(st|nd|rd|th)", tok)
    if m:
        return float(int(m.group(1)))
    if allow_roman:
        r = _roman(tok)
        if r is not None:
            return float(r)
    return None


def _clean(stem: str) -> Tuple[str, Optional[float]]:
    """Lower-case, strip ignorable tokens, return (clean text, timestamp)."""
    s = stem.lower()
    ts: Optional[float] = None
    m = _TIMESTAMP.search(s)
    if m:
        y, mo, d, hh, mm, ss = (int(g) if g else 0 for g in m.groups())
        ts = float(((((y * 12 + mo) * 31 + d) * 24 + hh) * 60 + mm) * 60 + ss)
        s = s[: m.start()] + " " + s[m.end():]
    s = re.sub(r"[_\-.,+~=\[\]{}()]+", " ", s)
    for pat in _IGNORE_PATTERNS[:-1]:
        s = pat.sub(" ", s)
    s = " ".join(t for t in s.split() if not _is_random_id(t))
    return s, ts


_KNOWN_WORDS = set(MARKERS) | START_WORDS | END_WORDS | set(NUMBER_WORDS) | set(ORDINAL_WORDS) | {"st", "nd", "rd", "th"}


def _is_random_id(tok: str) -> bool:
    """Generator/download ids like 'a3f9c2e1b7' or 'x7k2m9q4p1z8' - not ordering information."""
    if len(tok) < 8 or not re.fullmatch(r"[0-9a-z]+", tok) or not re.search(r"\d", tok) or not re.search(r"[a-z]", tok):
        return False
    runs = re.findall(r"[a-z]+", tok)
    if all(r in _KNOWN_WORDS or len(r) <= 1 for r in runs):
        return False  # e.g. scene2shot3, clip0004b, ep01sc02
    if re.fullmatch(r"[a-z]{3,}\d{1,4}[a-z]?", tok):
        return False  # e.g. raccoon12, kitchen3b
    return True


def _tokens(text: str) -> List[str]:
    # split letter/digit boundaries: "clip04b" -> "clip 04 b", keep ordinals like "2nd"
    text = re.sub(r"#", " # ", text)
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", r" \1\2 ", text)
    text = re.sub(r"(?<=[a-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[a-z])(?!(?:st|nd|rd|th)\b)", " ", text)
    return text.split()


def parse_name(path: Path) -> _Parsed:
    stem = path.stem
    text, ts = _clean(stem)
    markers: Dict[int, float] = {}
    marker_words: Dict[int, str] = {}
    suffix = ""

    sxe = _SXXEYY.search(text)
    if sxe:
        markers[0], markers[1] = float(sxe.group(1)), float(sxe.group(2))
        marker_words[0], marker_words[1] = "S", "E"
        text = text[: sxe.start()] + " " + text[sxe.end():]

    toks = _tokens(text)
    for i, tok in enumerate(toks):
        if tok in MARKERS and i + 1 < len(toks):
            lvl = MARKERS[tok]
            num = _number(toks[i + 1], allow_roman=True)
            if num is not None and lvl not in markers:
                markers[lvl] = num
                marker_words[lvl] = tok
                if i + 2 < len(toks) and re.fullmatch(r"[a-z]", toks[i + 2]) and toks[i + 2] not in MARKERS:
                    suffix = suffix or toks[i + 2]

    leading = None
    if toks and toks[0].isdigit():
        leading = float(int(toks[0]))
        if len(toks) > 1 and re.fullmatch(r"[a-z]", toks[1]):
            suffix = suffix or toks[1]

    ordinal = None
    for tok in toks:
        if tok in ORDINAL_WORDS:
            ordinal = float(ORDINAL_WORDS[tok])
            break
        m = re.fullmatch(r"(\d+)(st|nd|rd|th)", tok)
        if m:
            ordinal = float(int(m.group(1)))
            break

    trailing = None
    for j in range(len(toks) - 1, -1, -1):
        if toks[j].isdigit():
            trailing = float(int(toks[j]))
            if j + 1 < len(toks) and re.fullmatch(r"[a-z]", toks[j + 1]):
                suffix = suffix or toks[j + 1]
            break
        if toks[j] in NUMBER_WORDS:
            trailing = float(NUMBER_WORDS[toks[j]])
            break

    joined = "".join(toks)
    position = 0
    tokset = set(toks)
    if tokset & START_WORDS or "coldopen" in joined:
        position = -1
    elif tokset & END_WORDS:
        position = 1

    return _Parsed(path=path, markers=markers, marker_words=marker_words, leading=leading, ordinal=ordinal,
                   trailing=trailing, timestamp=ts, position=position, suffix=suffix, natural=natural_key(path.name))


def _fmt(n: float) -> str:
    return str(int(n)) if float(n).is_integer() else f"{n:g}"


def order_paths(paths: Iterable[Path | str]) -> Tuple[List[OrderInfo], List[str]]:
    """Order media files by the sequence encoded in their names.

    Returns (ordered infos, warnings).
    """
    parsed = [parse_name(Path(p)) for p in paths]
    warnings: List[str] = []
    if not parsed:
        return [], warnings

    # files that only carry a position word don't need to share the numeric scheme
    def numeric(p: _Parsed) -> bool:
        return bool(p.markers) or p.leading is not None or p.ordinal is not None or p.trailing is not None

    core = [p for p in parsed if numeric(p) or p.position == 0 or p.timestamp is not None]
    schemes = [
        ("markers", lambda p: bool(p.markers)),
        ("leading", lambda p: p.leading is not None),
        ("ordinal", lambda p: p.ordinal is not None),
        ("trailing", lambda p: p.trailing is not None),
        ("timestamp", lambda p: p.timestamp is not None),
    ]
    scheme = "natural"
    for name, has in schemes:
        if core and all(has(p) for p in core):
            scheme = name
            break
    if scheme == "natural" and core:
        # partial coverage: use the scheme covering most files, others follow by natural order
        best, best_n = "natural", 0
        for name, has in schemes:
            n = sum(1 for p in core if has(p))
            if n > best_n:
                best, best_n = name, n
        if best_n >= max(2, (len(core) + 1) // 2):
            scheme = best
            missing = [p.path.name for p in core if not dict(schemes)[best](p)]
            warnings.append(f"{len(missing)} file(s) have no {best} number and were placed after numbered clips: "
                            + ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else ""))
        else:
            warnings.append("no consistent numbering found in the filenames; using natural filename order. "
                            "Name clips like '01_intro.mp4', '02_kitchen.mp4' to control the order.")

    levels = sorted({lvl for p in parsed for lvl in p.markers})
    BIG = 1e12

    def key_and_reason(p: _Parsed) -> Tuple[Tuple, List[float], str]:
        pos_bucket = 0 if p.position == -1 else 2 if p.position == 1 else 1
        nums: List[float] = []
        reason = ""
        if scheme == "markers" and p.markers:
            nums = [p.markers.get(l, 0.0) for l in levels]
            reason = " ".join(f"{p.marker_words.get(l, LEVEL_NAMES[l])} {_fmt(p.markers[l])}" for l in levels if l in p.markers)
            pos_bucket = 1
        elif scheme == "leading" and p.leading is not None:
            nums, reason, pos_bucket = [p.leading], f"leading number {_fmt(p.leading)}", 1
        elif scheme == "ordinal" and p.ordinal is not None:
            nums, reason, pos_bucket = [p.ordinal], f"ordinal '{_fmt(p.ordinal)}'", 1
        elif scheme == "trailing" and p.trailing is not None:
            nums, reason, pos_bucket = [p.trailing], f"number {_fmt(p.trailing)} in name", 1
        elif scheme == "timestamp" and p.timestamp is not None:
            nums, reason, pos_bucket = [p.timestamp], "timestamp in name", 1
        else:
            nums = [BIG]
            if p.position == -1:
                reason = "opening word (intro/opening/prologue) -> placed first"
            elif p.position == 1:
                reason = "ending word (outro/finale/credits) -> placed last"
            else:
                reason = "no sequence number -> natural filename order"
        if p.suffix:
            reason += f" + suffix '{p.suffix}'"
        sfx = (ord(p.suffix) - 96) / 100.0 if p.suffix else 0.0
        key = (pos_bucket, tuple(nums), sfx, p.natural)
        json_key = [float(pos_bucket), *[float(n) for n in nums], sfx]
        return key, json_key, reason

    items = []
    for p in parsed:
        k, jk, r = key_and_reason(p)
        items.append((k, jk, r, p))
    items.sort(key=lambda t: t[0])

    # warnings: ties and gaps in the primary sequence
    seen: Dict[Tuple, List[str]] = {}
    for k, _, _, p in items:
        if k[1] and k[1][0] != BIG:
            seen.setdefault((k[0], k[1], k[2]), []).append(p.path.name)
    for names in seen.values():
        if len(names) > 1:
            warnings.append(f"clips share the same position and were ordered by name: {', '.join(names)}")
    if scheme in ("leading", "trailing", "ordinal") or (scheme == "markers" and len(levels) == 1):
        vals = sorted({int(k[1][0]) for k, *_ in items if k[1] and k[1][0] != BIG and float(k[1][0]).is_integer()})
        if vals and len(vals) > 1:
            gaps = [v for v in range(vals[0], vals[-1]) if v not in vals]
            if gaps and len(gaps) <= 10:
                warnings.append(f"numbering skips {', '.join(map(str, gaps))} - is a clip missing?")

    out: List[OrderInfo] = []
    for i, (_, jk, reason, p) in enumerate(items):
        out.append(OrderInfo(path=str(p.path), name=p.path.name, order=i, reason=reason, key=jk))
    return out, warnings


def scan_folder(folder: Path | str, recursive: bool = False, kinds: Sequence[str] = ("video",)) -> List[Path]:
    """List media files in a folder (non-recursive by default), skipping hidden/temp files."""
    folder = Path(folder)
    exts = set()
    for k in kinds:
        exts |= {"video": VIDEO_EXTS, "audio": AUDIO_EXTS, "image": IMAGE_EXTS, "lut": LUT_EXTS}[k]
    it = folder.rglob("*") if recursive else folder.iterdir()
    files = []
    for f in it:
        if not f.is_file() or f.name.startswith((".", "~$")) or f.suffix.lower() not in exts:
            continue
        files.append(f)
    return files


def order_folder(folder: Path | str, recursive: bool = False,
                 kinds: Sequence[str] = ("video", "audio", "image", "lut")) -> Tuple[List[OrderInfo], List[str]]:
    """Order every media file in a folder. Videos are ordered among themselves; other kinds follow."""
    files = scan_folder(folder, recursive, kinds)
    videos = [f for f in files if f.suffix.lower() in VIDEO_EXTS]
    others = [f for f in files if f.suffix.lower() not in VIDEO_EXTS]
    ordered, warnings = order_paths(videos)
    extra = sorted(others, key=lambda f: natural_key(f.name))
    base = len(ordered)
    for i, f in enumerate(extra):
        kind = "LUT" if f.suffix.lower() in LUT_EXTS else "audio" if f.suffix.lower() in AUDIO_EXTS else "image"
        ordered.append(OrderInfo(path=str(f), name=f.name, order=base + i, reason=f"{kind} asset (not part of the cut order)",
                                 key=[9.0, float(i)]))
    return ordered, warnings
