"""Python mirror of ``src/types/project.ts`` / ``docs/CONTRACTS.md``.

Every dataclass serialises to the exact camelCase JSON contract via :func:`to_json`.
Times are milliseconds, geometry is in source pixels, bboxes are ``[x1, y1, x2, y2]``.
"""
from __future__ import annotations

import dataclasses
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

BBox = Tuple[float, float, float, float]

# --------------------------------------------------------------------------- ids


def new_id(prefix: str) -> str:
    """Create a random id such as ``clp_3f9a1c2d4e5f``. Accepts ``clp`` or ``clp_``."""
    prefix = prefix if prefix.endswith("_") else prefix + "_"
    return prefix + secrets.token_hex(6)


# --------------------------------------------------------------------------- keyframes


@dataclass
class Keyframe:
    timeMs: float
    value: Any
    easing: str = "linear"
    bezier: Optional[List[float]] = None


@dataclass
class Keyframed:
    static: Any
    keyframes: List[Keyframe] = field(default_factory=list)


def keyframed(value: Any) -> Keyframed:
    return Keyframed(static=value, keyframes=[])


# --------------------------------------------------------------------------- clip sub-structures


@dataclass
class ClipTransform:
    position: Keyframed = field(default_factory=lambda: keyframed([0.0, 0.0]))
    scale: Keyframed = field(default_factory=lambda: keyframed(1.0))
    rotation: Keyframed = field(default_factory=lambda: keyframed(0.0))
    opacity: Keyframed = field(default_factory=lambda: keyframed(1.0))
    blur: Keyframed = field(default_factory=lambda: keyframed(0.0))


@dataclass
class SpeedPoint:
    t: float
    speed: float


@dataclass
class SpeedCurve:
    preset: str = "normal"
    points: List[SpeedPoint] = field(default_factory=lambda: [SpeedPoint(0.0, 1.0), SpeedPoint(1.0, 1.0)])
    opticalFlow: bool = True


@dataclass
class HslOffset:
    h: float = 0.0
    s: float = 0.0
    l: float = 0.0


HSL_CHANNELS = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta")


def _identity_curve() -> List[List[float]]:
    return [[0.0, 0.0], [1.0, 1.0]]


@dataclass
class ColorCurves:
    master: List[List[float]] = field(default_factory=_identity_curve)
    r: List[List[float]] = field(default_factory=_identity_curve)
    g: List[List[float]] = field(default_factory=_identity_curve)
    b: List[List[float]] = field(default_factory=_identity_curve)


@dataclass
class ColorGrade:
    exposure: float = 0.0
    brilliance: float = 0.0
    contrast: float = 0.0
    brightness: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    saturation: float = 0.0
    vibrance: float = 0.0
    sharpness: float = 0.0
    temperature: float = 0.0
    tint: float = 0.0
    lift: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gamma: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gain: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    hsl: Dict[str, HslOffset] = field(default_factory=lambda: {c: HslOffset() for c in HSL_CHANNELS})
    curves: ColorCurves = field(default_factory=ColorCurves)
    lutAssetId: Optional[str] = None
    lutIntensity: float = 1.0
    vignette: float = 0.0
    grain: float = 0.0


@dataclass
class ClipAudio:
    gainDb: float = 0.0
    normalize: bool = True
    muted: bool = False
    # "original" | "voice" (isolate voice: vocals stem) | "background" (remove vocals: background stem)
    voice: str = "original"
    # --- feature set v2: optional, omitted from the JSON when None (spec defaults in DEFAULTS_V2) ---
    keepPitch: Optional[bool] = None     # default True: pitch-preserving time-stretch when speed != 1
    fadeInMs: Optional[float] = None     # default 0: equal-power fade-in, timeline ms
    fadeOutMs: Optional[float] = None    # default 0: equal-power fade-out, timeline ms
    volume: Optional[Keyframed] = None   # dB offset added to gainDb, clip-local keyframes (static 0)


@dataclass
class ClipMask:
    shape: str = "rectangle"
    feather: float = 0.1
    rect: Keyframed = field(default_factory=lambda: keyframed([0.0, 0.0, 1.0, 1.0]))
    inverted: bool = False


@dataclass
class FreezeFrame:
    atMs: float
    holdMs: float


# --------------------------------------------------------------------------- reframe


@dataclass
class ReframeKeyframe:
    frame: int
    timeMs: float
    crop: List[float]
    zoom: float
    tx: float
    ty: float


@dataclass
class ReframeTrack:
    sourceWidth: int
    sourceHeight: int
    keyframes: List[ReframeKeyframe] = field(default_factory=list)
    reason: Optional[str] = None


# --------------------------------------------------------------------------- feature set v2 (docs/FEATURES_V2.md)

PROJECT_FPS = (24, 25, 30, 40, 48, 50, 60)  # plus 23.976 / 29.97 for sources
EXPORT_FPS_CHOICES = (24, 30, 40, 50, 60)   # what the UI offers
FRAME_INTERPOLATION_MODES = ("frameBlend", "opticalFlow", "none")
STEM_KINDS = ("vocals", "background")
TRACK_ROLES = ("voice", "background")
TRANSITION_TYPES = ("dissolve", "dipToBlack", "dipToWhite", "wipeLeft", "wipeRight", "wipeUp", "wipeDown",
                    "slideLeft", "slideRight", "pushLeft", "pushRight", "zoomIn", "zoomOut", "blurDissolve",
                    "flash", "circleOpen")
EFFECT_TYPES = ("cameraSnap", "fadeFromBlack", "fadeToBlack", "fadeFromWhite", "fadeToWhite", "blackAndWhite",
                "sepia", "letterbox", "shake", "zoomPunch", "blurIn", "blurOut", "rgbSplit", "vhs",
                "vignettePulse", "flashWhite")
TRANSITION_MIN_MS, TRANSITION_MAX_MS, TRANSITION_DEFAULT_MS = 100.0, 3000.0, 500.0

# Spec defaults of the optional v2 fields: an absent field means exactly this value. The dataclasses
# keep None for "absent", so re-serialising an older project gives back its exact previous shape.
DEFAULTS_V2: Dict[str, Any] = {
    "Project.frameInterpolation": "opticalFlow",
    "ClipAudio.keepPitch": True,
    "ClipAudio.fadeInMs": 0.0,
    "ClipAudio.fadeOutMs": 0.0,
    "ClipAudio.volume": 0.0,  # static dB offset
    "Clip.fadeInMs": 0.0,
    "Clip.fadeOutMs": 0.0,
    "ClipEffect.intensity": 1.0,
    "ClipTransition.durationMs": TRANSITION_DEFAULT_MS,
}


@dataclass
class AssetStemOf:
    """``Asset.stemOf``: this (audio) asset is the ``stem`` of asset ``assetId`` ("Separate to tracks")."""
    assetId: str
    stem: str  # "vocals" | "background"


@dataclass
class ClipTransition:
    """``Clip.transitionIn`` (on the incoming clip), centred on the cut."""
    type: str = "dissolve"
    durationMs: float = TRANSITION_DEFAULT_MS


@dataclass
class ClipEffect:
    """``Clip.effect`` of an effect clip (``fx`` track, ``assetId: ''``)."""
    type: str
    intensity: float = 1.0
    params: Optional[Dict[str, float]] = None


# --------------------------------------------------------------------------- timeline document


@dataclass
class Asset:
    id: str
    path: str
    name: str
    kind: str
    durationMs: float
    width: int
    height: int
    fps: float
    hasAudio: bool
    codec: Optional[str] = None
    sceneTags: Optional[List[str]] = None
    order: Optional[int] = None
    orderReason: Optional[str] = None
    # {"vocals": path, "background": path} once `separate` has run for this file
    stems: Optional[Dict[str, str]] = None
    # set on the audio assets created by "Separate to tracks"
    stemOf: Optional[AssetStemOf] = None


@dataclass
class Clip:
    id: str
    assetId: str
    trackId: str
    startMs: float
    inMs: float
    outMs: float
    speed: SpeedCurve = field(default_factory=SpeedCurve)
    transform: ClipTransform = field(default_factory=ClipTransform)
    color: ColorGrade = field(default_factory=ColorGrade)
    audio: ClipAudio = field(default_factory=ClipAudio)
    mask: Optional[ClipMask] = None
    blendMode: str = "normal"
    reframe: Optional[ReframeTrack] = None
    label: Optional[str] = None
    freezeFrame: Optional[FreezeFrame] = None
    reversed: bool = False
    # clips that move / trim / delete together (a shot's video clip, its mirrored audio clip and any
    # stem clips); omitted from the JSON when None
    linkId: Optional[str] = None
    # --- feature set v2: optional, omitted when None ---
    transitionIn: Optional[ClipTransition] = None  # from the previous clip on the same track
    effect: Optional[ClipEffect] = None            # effect clips on fx tracks
    fadeInMs: Optional[float] = None               # video fade from black, default 0
    fadeOutMs: Optional[float] = None              # video fade to black, default 0


@dataclass
class Track:
    id: str
    kind: str
    name: str
    locked: bool = False
    muted: bool = False
    clips: List[Clip] = field(default_factory=list)
    role: Optional[str] = None  # "voice" | "background" (tracks created by "Separate to tracks")


@dataclass
class BeatMarker:
    timeMs: float
    strength: float
    kind: str  # "beat1" | "beat2"


@dataclass
class Project:
    id: str
    name: str
    fps: float
    width: int
    height: int
    assets: List[Asset] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    beatMarkers: List[BeatMarker] = field(default_factory=list)
    version: int = 1
    # UniversalAdjust ({enabled, name, values}), kept as-is so re-serialising a project never drops it
    universalAdjust: Optional[Dict[str, Any]] = None
    frameInterpolation: Optional[str] = None  # "frameBlend" | "opticalFlow" (default) | "none"


# --------------------------------------------------------------------------- analysis result


@dataclass
class Shot:
    index: int
    startFrame: int
    endFrame: int
    startMs: float
    endMs: float
    confidence: float
    method: str  # "transnetv2" | "pyscenedetect"
    cast: Optional[List[str]] = None  # main-cast character ids seen in >= 2 sampled frames (characters.json)


@dataclass
class DetectedInstance:
    trackId: int
    label: str
    bbox: List[float]
    score: float


@dataclass
class DuplicateFinding:
    shotIndex: int
    frame: int
    timeMs: float
    primary: DetectedInstance
    duplicate: DetectedInstance
    similarity: float
    character: Optional[str] = None      # main-cast id when both instances were identified as it
    characterName: Optional[str] = None  # its display name ("Bunny")


@dataclass
class ShotReframe:
    shotIndex: int
    track: ReframeTrack


@dataclass
class AudioAnalysis:
    integratedLufs: float
    truePeakDb: float
    recommendedGainDb: float
    beats: List[float] = field(default_factory=list)
    tempoBpm: Optional[float] = None
    # 0..1 trust in the beat grid (periodicity + numpy / librosa tempo agreement); beats are only
    # emitted when confident. Optional, omitted when None.
    beatConfidence: Optional[float] = None


@dataclass
class TransitionAnalysis:
    fromShot: int
    toShot: int
    flowMagnitude: float
    smoothness: float
    suggestion: str  # "cut" | "dissolve"


@dataclass
class ClipAnalysis:
    path: str
    asset: Asset
    shots: List[Shot] = field(default_factory=list)
    duplicates: List[DuplicateFinding] = field(default_factory=list)
    reframe: List[ShotReframe] = field(default_factory=list)
    audio: Optional[AudioAnalysis] = None
    transitions: List[TransitionAnalysis] = field(default_factory=list)


@dataclass
class AnalysisResult:
    generatedAt: str
    clips: List[ClipAnalysis]
    timeline: Project
    version: int = 1


# --------------------------------------------------------------------------- factories


def default_color_grade() -> ColorGrade:
    return ColorGrade()


def default_transform() -> ClipTransform:
    return ClipTransform()


def default_speed() -> SpeedCurve:
    return SpeedCurve()


def default_audio() -> ClipAudio:
    return ClipAudio()


def new_clip(asset_id: str, track_id: str, start_ms: float, in_ms: float, out_ms: float, **kw: Any) -> Clip:
    return Clip(
        id=new_id("clp"),
        assetId=asset_id,
        trackId=track_id,
        startMs=start_ms,
        inMs=in_ms,
        outMs=out_ms,
        speed=default_speed(),
        transform=default_transform(),
        color=default_color_grade(),
        audio=default_audio(),
        **kw,
    )


def new_track(kind: str, name: str) -> Track:
    return Track(id=new_id("trk"), kind=kind, name=name)


def new_project(name: str, fps: float, width: int, height: int) -> Project:
    return Project(id=new_id("proj"), name=name, fps=fps, width=width, height=height)


def effective_frame_interpolation(project: Project) -> str:
    """``Project.frameInterpolation`` with its spec default (``opticalFlow``; unknown values too)."""
    v = project.frameInterpolation
    return v if v in FRAME_INTERPOLATION_MODES else DEFAULTS_V2["Project.frameInterpolation"]


def effective_keep_pitch(audio: ClipAudio) -> bool:
    """``ClipAudio.keepPitch`` with its spec default (True)."""
    return DEFAULTS_V2["ClipAudio.keepPitch"] if audio.keepPitch is None else bool(audio.keepPitch)


# --------------------------------------------------------------------------- serialisation

# Fields that are *optional* in the TypeScript contract (``field?:``). When None they are
# omitted from the JSON rather than emitted as null.
_OMIT_WHEN_NONE = {
    (Asset, "codec"),
    (Asset, "sceneTags"),
    (Asset, "order"),
    (Asset, "orderReason"),
    (Asset, "stems"),
    (Asset, "stemOf"),
    (ClipAudio, "keepPitch"),
    (ClipAudio, "fadeInMs"),
    (ClipAudio, "fadeOutMs"),
    (ClipAudio, "volume"),
    (Clip, "transitionIn"),
    (Clip, "effect"),
    (Clip, "fadeInMs"),
    (Clip, "fadeOutMs"),
    (ClipEffect, "params"),
    (Track, "role"),
    (Project, "universalAdjust"),
    (Project, "frameInterpolation"),
    (Keyframe, "bezier"),
    (ReframeTrack, "reason"),
    (Clip, "label"),
    (Clip, "linkId"),
    (AudioAnalysis, "beatConfidence"),
    (Shot, "cast"),
    (DuplicateFinding, "character"),
    (DuplicateFinding, "characterName"),
}

# Key order of the top-level objects (``version`` first, like the contract examples).
_KEY_ORDER = {
    Project: ("version", "id", "name", "fps", "width", "height", "assets", "tracks", "beatMarkers"),
    AnalysisResult: ("version", "generatedAt", "clips", "timeline"),
}


def _scalar(v: Any) -> Any:
    if hasattr(v, "item") and not isinstance(v, (list, tuple, dict, str, bytes)):
        try:
            v = v.item()
        except Exception:  # pragma: no cover
            pass
    if isinstance(v, float) and v == 0:
        return 0.0  # avoid "-0.0"
    return v


def to_json(obj: Any) -> Any:
    """Recursively convert dataclasses / numpy / tuples into JSON-ready python objects."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj)
        out: Dict[str, Any] = {}
        names = [f.name for f in dataclasses.fields(obj)]
        order = _KEY_ORDER.get(cls)
        if order:
            names = [n for n in order if n in names] + [n for n in names if n not in order]
        for name in names:
            val = getattr(obj, name)
            if val is None and (cls, name) in _OMIT_WHEN_NONE:
                continue
            out[name] = to_json(val)
        return out
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        return to_json(obj.tolist())
    return _scalar(obj)


# --------------------------------------------------------------------------- deserialisation


def _kf_from_json(d: Dict[str, Any]) -> Keyframed:
    return Keyframed(
        static=d.get("static"),
        keyframes=[
            Keyframe(k["timeMs"], k["value"], k.get("easing", "linear"), k.get("bezier"))
            for k in d.get("keyframes", [])
        ],
    )


def asset_from_json(d: Dict[str, Any]) -> Asset:
    return Asset(
        id=d["id"], path=d["path"], name=d["name"], kind=d["kind"], durationMs=d["durationMs"],
        width=d["width"], height=d["height"], fps=d["fps"], hasAudio=d["hasAudio"],
        codec=d.get("codec"), sceneTags=d.get("sceneTags"), order=d.get("order"),
        orderReason=d.get("orderReason"), stems=dict(d["stems"]) if d.get("stems") else None,
        stemOf=AssetStemOf(d["stemOf"]["assetId"], d["stemOf"]["stem"]) if d.get("stemOf") else None,
    )


def reframe_track_from_json(d: Dict[str, Any]) -> ReframeTrack:
    return ReframeTrack(
        sourceWidth=d["sourceWidth"],
        sourceHeight=d["sourceHeight"],
        keyframes=[
            ReframeKeyframe(k["frame"], k["timeMs"], list(k["crop"]), k["zoom"], k["tx"], k["ty"])
            for k in d.get("keyframes", [])
        ],
        reason=d.get("reason"),
    )


def color_grade_from_json(d: Dict[str, Any]) -> ColorGrade:
    cg = ColorGrade()
    for f in dataclasses.fields(ColorGrade):
        if f.name in ("hsl", "curves"):
            continue
        if f.name in d:
            setattr(cg, f.name, d[f.name])
    if "hsl" in d:
        cg.hsl = {c: HslOffset(**d["hsl"].get(c, {})) for c in HSL_CHANNELS}
    if "curves" in d:
        cg.curves = ColorCurves(**{k: d["curves"][k] for k in ("master", "r", "g", "b") if k in d["curves"]})
    return cg


def clip_from_json(d: Dict[str, Any]) -> Clip:
    t = d.get("transform", {})
    tr = ClipTransform(**{k: _kf_from_json(t[k]) for k in ("position", "scale", "rotation", "opacity", "blur") if k in t})
    sp = d.get("speed", {})
    speed = SpeedCurve(
        preset=sp.get("preset", "normal"),
        points=[SpeedPoint(p["t"], p["speed"]) for p in sp.get("points", [])] or default_speed().points,
        opticalFlow=sp.get("opticalFlow", True),
    )
    a = d.get("audio", {})
    mask = d.get("mask")
    ff = d.get("freezeFrame")
    tri = d.get("transitionIn")
    fx = d.get("effect")
    return Clip(
        id=d["id"], assetId=d["assetId"], trackId=d["trackId"],
        startMs=d["startMs"], inMs=d["inMs"], outMs=d["outMs"],
        speed=speed, transform=tr, color=color_grade_from_json(d.get("color", {})),
        audio=ClipAudio(gainDb=a.get("gainDb", 0.0), normalize=a.get("normalize", True), muted=a.get("muted", False),
                        voice=a.get("voice", "original"), keepPitch=a.get("keepPitch"), fadeInMs=a.get("fadeInMs"),
                        fadeOutMs=a.get("fadeOutMs"),
                        volume=None if a.get("volume") is None else _kf_from_json(a["volume"])),
        mask=None if mask is None else ClipMask(mask["shape"], mask["feather"], _kf_from_json(mask["rect"]), mask["inverted"]),
        blendMode=d.get("blendMode", "normal"),
        reframe=None if d.get("reframe") is None else reframe_track_from_json(d["reframe"]),
        label=d.get("label"),
        freezeFrame=None if ff is None else FreezeFrame(ff["atMs"], ff["holdMs"]),
        reversed=d.get("reversed", False),
        linkId=d.get("linkId"),
        transitionIn=None if tri is None else ClipTransition(tri.get("type", "dissolve"),
                                                             tri.get("durationMs", TRANSITION_DEFAULT_MS)),
        effect=None if fx is None else ClipEffect(fx["type"], fx.get("intensity", 1.0),
                                                  dict(fx["params"]) if fx.get("params") is not None else None),
        fadeInMs=d.get("fadeInMs"),
        fadeOutMs=d.get("fadeOutMs"),
    )


def project_from_json(d: Dict[str, Any]) -> Project:
    return Project(
        id=d["id"], name=d["name"], fps=d["fps"], width=d["width"], height=d["height"],
        assets=[asset_from_json(a) for a in d.get("assets", [])],
        tracks=[
            Track(t["id"], t["kind"], t["name"], t.get("locked", False), t.get("muted", False),
                  [clip_from_json(c) for c in t.get("clips", [])], t.get("role"))
            for t in d.get("tracks", [])
        ],
        beatMarkers=[BeatMarker(b["timeMs"], b["strength"], b["kind"]) for b in d.get("beatMarkers", [])],
        version=d.get("version", 1),
        universalAdjust=d.get("universalAdjust"),
        frameInterpolation=d.get("frameInterpolation"),
    )


def _instance_from_json(d: Dict[str, Any]) -> DetectedInstance:
    return DetectedInstance(d["trackId"], d["label"], list(d["bbox"]), d["score"])


def clip_analysis_from_json(c: Dict[str, Any]) -> ClipAnalysis:
    au = c.get("audio")
    return ClipAnalysis(
        path=c["path"],
        asset=asset_from_json(c["asset"]),
        shots=[Shot(**s) for s in c.get("shots", [])],
        duplicates=[
            DuplicateFinding(x["shotIndex"], x["frame"], x["timeMs"], _instance_from_json(x["primary"]),
                             _instance_from_json(x["duplicate"]), x["similarity"], x.get("character"),
                             x.get("characterName"))
            for x in c.get("duplicates", [])
        ],
        reframe=[ShotReframe(r["shotIndex"], reframe_track_from_json(r["track"])) for r in c.get("reframe", [])],
        audio=None if au is None else AudioAnalysis(au["integratedLufs"], au["truePeakDb"], au["recommendedGainDb"],
                                                     list(au.get("beats", [])), au.get("tempoBpm"),
                                                     au.get("beatConfidence")),
        transitions=[TransitionAnalysis(**t) for t in c.get("transitions", [])],
    )


def analysis_result_from_json(d: Dict[str, Any]) -> AnalysisResult:
    clips = [clip_analysis_from_json(c) for c in d.get("clips", [])]
    return AnalysisResult(generatedAt=d["generatedAt"], clips=clips, timeline=project_from_json(d["timeline"]),
                          version=d.get("version", 1))
