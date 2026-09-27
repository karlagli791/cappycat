"""Timeline assembly: ClipAnalysis[] -> Project.

* Assets ordered by ``order`` then natural filename sort.
* Tracks: "Video 1" (video), "FX" (fx, empty), "Audio 1" (audio, mirrors video clips of
  assets that have audio).
* Every shot becomes a Clip (``inMs``/``outMs`` from the shot, ``outMs`` end-exclusive),
  sequential ``startMs``, reframe attached when the shot has one, loudness gain applied. A shot's
  video clip and its mirrored audio clip share a ``linkId`` (they move / trim / delete together).
* If the total exceeds ``target_duration_ms`` the longest shots are trimmed (water-filling
  from the top, never below ``MIN_CLIP_MS``). A trimmed shot loses frames at **both ends** (the
  middle of the shot is kept); when the clip has a (confident) beat grid, the trimmed window is
  shifted within the shot so its out point lands on a source beat. If even the floors don't fit,
  the **lowest-value** shots are dropped (fewest cast members, shortest, duplicate-artifact shots
  first; ties drop the later shot) and the rest is re-fitted, so the timeline fills the target.
* Beat markers are the clips' beats offset to their timeline position.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .schema import (Asset, BeatMarker, Clip, ClipAnalysis, ClipAudio, Project, ReframeTrack, Shot, Track, new_clip,
                     new_id, new_project, new_track)

MIN_CLIP_MS = 1500.0


def natural_key(name: str) -> List[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def order_analyses(analyses: Sequence[ClipAnalysis]) -> List[ClipAnalysis]:
    def key(ca: ClipAnalysis) -> Tuple[int, List[object]]:
        o = ca.asset.order if ca.asset.order is not None else 1_000_000
        return (o, natural_key(Path(ca.path).name))

    return sorted(analyses, key=key)


def _frame_ms(fps: float) -> float:
    return 1000.0 / fps if fps and fps > 0 else 40.0


def _shot_range(shot: Shot, fps: float) -> Tuple[float, float]:
    """(inMs, outMs) with outMs end-exclusive (last frame + one frame duration)."""
    return float(shot.startMs), float(shot.endMs) + _frame_ms(fps)


# Frame grids an export may decode a source on: every project rate the app allows (model.rs
# ALLOWED_FPS). Rounding makes a multiple of a safe grid not necessarily safe, so all are listed.
EXPORT_GRIDS = (24000 / 1001, 24.0, 25.0, 30000 / 1001, 30.0, 40.0, 48.0, 50.0, 60000 / 1001, 60.0)
# timestamps within this many ticks of a rounding tie count as either side (shot times carry 3
# decimals of ms and float noise)
TIE_TOLERANCE = 1e-4
# The exporter seeks to ``i / grid - SEEK_LEAD_MS`` (0 for a decode starting at frame 0), which
# shifts every source timestamp by up to this much before ffmpeg's ``fps`` filter rounds it.
SEEK_LEAD_MS = 1.0


def _first_tick(t_ms: float, grid: float, shift_ms: float, bias: float) -> int:
    """Grid tick ffmpeg's ``fps`` filter (``round=near``: half away from zero) assigns a frame
    presented at ``t_ms``; ``bias`` breaks float-noise ties towards the safe side."""
    return int(math.floor((t_ms + shift_ms) * grid / 1000.0 + 0.5 + bias))


def cut_points(first_new_ms: float, grids: Sequence[float] = EXPORT_GRIDS, margin_ms: float = 0.25) -> Tuple[float, float]:
    """``(outMs, inMs)`` for a hard cut whose first new frame is presented at ``first_new_ms``.

    The preview (a ``<video>`` element) shows the frame whose display interval contains the time;
    the exporter decodes on a uniform grid (tick ``floor(t * grid)``; ffmpeg's ``fps`` filter shows
    on tick ``k`` the latest frame whose rounded timestamp is ``<= k``). The outgoing clip ends
    before the new frame's earliest tick on every grid (and before the frame itself); the incoming
    one starts at its latest tick (and after the frame). So neither angle leaks across the cut in
    either renderer; each side gives up at most about half a frame."""
    out_ms = first_new_ms - margin_ms
    in_ms = first_new_ms + margin_ms
    for g in grids:
        if g and g > 0:
            lo = _first_tick(first_new_ms, g, 0.0, -TIE_TOLERANCE)
            hi = _first_tick(first_new_ms, g, SEEK_LEAD_MS, TIE_TOLERANCE)
            out_ms = min(out_ms, lo * 1000.0 / g - margin_ms)
            in_ms = max(in_ms, hi * 1000.0 / g + margin_ms)
    return round(out_ms, 3), round(in_ms, 3)


def shot_ranges(shots: Sequence[Shot], fps: float, duration_ms: float = 0.0,
                grids: Sequence[float] = EXPORT_GRIDS) -> List[Tuple[float, float]]:
    """``(inMs, outMs)`` per shot (``outMs`` end-exclusive). Where two shots are consecutive
    (``prev.endFrame + 1 == next.startFrame``) the boundary uses :func:`cut_points`, so each clip
    shows only its own shot's frames in both the preview and the export."""
    out: List[Tuple[float, float]] = []
    for i, shot in enumerate(shots):
        a, b = _shot_range(shot, fps)
        if i > 0 and shots[i - 1].endFrame + 1 == shot.startFrame and shot.startMs > shots[i - 1].endMs:
            a = cut_points(shot.startMs, grids)[1]
        if i + 1 < len(shots) and shot.endFrame + 1 == shots[i + 1].startFrame and shots[i + 1].startMs > shot.endMs:
            b = cut_points(shots[i + 1].startMs, grids)[0]
        if duration_ms > 0:
            b = min(b, duration_ms)
        out.append((round(a, 3), round(b, 3)))
    return out


def _water_fill(d: List[float], target_ms: float, min_ms: float) -> List[float]:
    floors = [min(x, min_ms) for x in d]
    # binary search the level L such that sum(clamp(x, floor, L)) == target
    lo, hi = 0.0, max(d)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        total = sum(max(f, min(x, mid)) for x, f in zip(d, floors))
        if total > target_ms:
            hi = mid
        else:
            lo = mid
    return [max(f, min(x, lo)) for x, f in zip(d, floors)]


def fit_durations(durations: Sequence[float], target_ms: Optional[float], min_ms: float = MIN_CLIP_MS,
                  values: Optional[Sequence[float]] = None) -> List[float]:
    """Water-fill trim: reduce the longest entries down to a common level until the total
    fits ``target_ms``; entries never go below ``min_ms`` (or their own length if shorter).
    When even the floors do not fit, entries are dropped (returned as 0), lowest ``values`` first
    (ties / no values: the later entry; the first entry is always kept), and the remaining ones are
    re-fitted, so the result fills ``target_ms`` instead of falling short of it."""
    d = [float(x) for x in durations]
    if target_ms is None or target_ms <= 0 or sum(d) <= target_ms or not d:
        return d
    vals = [0.0] * len(d) if values is None else [float(v) for v in values]
    keep = list(range(len(d)))
    while True:
        fitted = _water_fill([d[i] for i in keep], target_ms, min_ms)
        if sum(fitted) <= target_ms + 1e-6 or len(keep) <= 1:
            break
        drop = min((i for i in keep if i != 0), key=lambda i: (vals[i], -i))
        keep.remove(drop)
    out = [0.0] * len(d)
    for i, x in zip(keep, fitted):
        out[i] = x
    return out


def shot_value(ca: ClipAnalysis, shot: Shot, dur_ms: float) -> float:
    """How much a shot is worth keeping when the timeline must lose some: cast members on screen,
    then length (up to 4 s); a shot that carried a duplicate-character artifact is worth a little less."""
    has_dup = any(f.shotIndex == shot.index for f in ca.duplicates)
    return len(shot.cast or []) + 0.25 * min(dur_ms / 1000.0, 4.0) - (0.5 if has_dup else 0.0)


def trim_window(in_ms: float, out_ms: float, dur: float, beats: Sequence[float] = ()) -> Tuple[float, float]:
    """Source window of length ``dur`` inside ``[in_ms, out_ms)``: centred (both ends trimmed), or
    ending on the source beat nearest to the centred end when such a window fits in the shot."""
    if dur >= out_ms - in_ms - 1e-6:
        return in_ms, out_ms
    centred_end = in_ms + (out_ms - in_ms + dur) / 2.0
    cands = [b for b in beats if in_ms + dur <= b <= out_ms]
    end = min(cands, key=lambda b: abs(b - centred_end)) if cands else centred_end
    return round(end - dur, 3), round(end, 3)


STANDARD_RATES = (24000 / 1001, 24.0, 25.0, 30000 / 1001, 30.0, 48.0, 50.0, 60000 / 1001, 60.0)


def snap_fps(fps: float, tolerance: float = 0.005) -> float:
    """Nearest standard frame rate when within ``tolerance`` (relative), else ``fps`` unchanged.
    AI generators store 24 fps content on odd timebases (average rate 24.04): the project should
    be 24, not 24.04."""
    if not fps or fps <= 0:
        return fps
    best = min(STANDARD_RATES, key=lambda r: abs(r - fps))
    if abs(best - fps) <= tolerance * best:
        return round(best, 3) if best != round(best) else float(best)
    return fps


def assemble_timeline(clip_analyses: Sequence[ClipAnalysis], target_duration_ms: Optional[float], fps: float, width: int,
                      height: int, name: Optional[str] = None,
                      beat_details: Optional[Dict[str, List[BeatMarker]]] = None) -> Project:
    """Build the Project document. ``beat_details`` optionally maps asset path -> BeatMarkers
    (with strengths / kinds); otherwise ``audio.beats`` are used with default strength."""
    ordered = order_analyses(clip_analyses)
    proj = new_project(name or (Path(ordered[0].path).stem if ordered else "Untitled"), snap_fps(fps), width, height)
    video = new_track("video", "Video 1")
    fx = new_track("fx", "FX")
    audio = new_track("audio", "Audio 1")
    proj.tracks = [video, fx, audio]

    # collect shots in order
    entries: List[Tuple[ClipAnalysis, Shot, float, float, Optional[ReframeTrack]]] = []
    for order_idx, ca in enumerate(ordered):
        asset = ca.asset
        if asset.order is None:
            asset.order = order_idx
        proj.assets.append(asset)
        reframes = {r.shotIndex: r.track for r in ca.reframe}
        shots = ca.shots or [Shot(0, 0, 0, 0.0, asset.durationMs, 1.0, "pyscenedetect")]
        grids = tuple(EXPORT_GRIDS) + (snap_fps(fps),)
        for shot, (in_ms, out_ms) in zip(shots, shot_ranges(shots, asset.fps or fps, asset.durationMs, grids)):
            if out_ms <= in_ms:
                continue
            entries.append((ca, shot, in_ms, out_ms, reframes.get(shot.index)))

    durations = [e[3] - e[2] for e in entries]
    values = [shot_value(e[0], e[1], d) for e, d in zip(entries, durations)]
    fitted = fit_durations(durations, target_duration_ms, values=values)
    cursor = 0.0
    for (ca, shot, in_ms, out_ms, reframe), dur in zip(entries, fitted):
        if dur <= 0:
            continue
        markers = (beat_details or {}).get(ca.path)
        if markers is None and ca.audio:
            markers = [BeatMarker(timeMs=b, strength=0.5, kind="beat2") for b in ca.audio.beats]
        in_ms, out_ms = trim_window(in_ms, out_ms, dur, [m.timeMs for m in markers or []])
        asset = ca.asset
        gain = ca.audio.recommendedGainDb if ca.audio else 0.0
        clip = new_clip(asset.id, video.id, cursor, in_ms, out_ms)
        clip.label = f"{asset.name} · Shot {shot.index + 1}"
        clip.reframe = reframe
        clip.audio = ClipAudio(gainDb=round(gain, 3), normalize=True, muted=False)
        video.clips.append(clip)
        if asset.hasAudio:
            aclip = new_clip(asset.id, audio.id, cursor, in_ms, out_ms)
            aclip.label = clip.label
            aclip.audio = ClipAudio(gainDb=round(gain, 3), normalize=True, muted=False)
            # the shot's video clip and its mirrored audio clip are one unit in the editor
            clip.linkId = aclip.linkId = new_id("lnk")
            audio.clips.append(aclip)
            # beats inside this source range, shifted onto the timeline
            for m in markers or []:
                if in_ms <= m.timeMs < out_ms:
                    proj.beatMarkers.append(BeatMarker(round(cursor + (m.timeMs - in_ms), 3), m.strength, m.kind))
        cursor += dur

    proj.beatMarkers.sort(key=lambda m: m.timeMs)
    return proj


def timeline_duration_ms(project: Project) -> float:
    end = 0.0
    for t in project.tracks:
        for c in t.clips:
            end = max(end, c.startMs + (c.outMs - c.inMs))
    return end
