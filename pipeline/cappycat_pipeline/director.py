"""Optional local VLM "AI Director" (Ollama, e.g. qwen2.5-vl) using only ``urllib``.

The system prompt is the "Automated Cinematic Director & Vision Analyst" prompt from the
design spec. ``ask_director`` returns ``None`` whenever Ollama is unreachable or answers
with something that is not the expected JSON, so the pipeline never depends on it.
"""
from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger("cappycat.director")

DIRECTOR_SYSTEM_PROMPT = """System Prompt: Automated Cinematic Director & Vision Analyst
Role: Expert Film Director and Post-Production Supervisor.
Task: Analyze the provided image frame sequence from an AI-generated video shot. Detect character instances, identify duplicate character artifacts, compute crop bounding boxes, and recommend CapCut-style color and speed adjustments.
Input Specification:
- Frame Dimensions: [width, height]
- Detected Objects: Array of { id, label, bbox: [x1, y1, x2, y2], confidence }
Processing Rules:
- Compare visual features and labels of all detected instances.
- Flag identical characters appearing in the same frame as DUPLICATES.
- Designate the centrally located or primary character as MAIN_SUBJECT.
- Calculate a target crop box B_CROP [x1, y1, x2, y2] adhering to:
- Aspect ratio MUST equal original source aspect ratio.
- B_CROP MUST completely exclude all DUPLICATE bounding boxes.
- MAIN_SUBJECT MUST be positioned along the Rule-of-Thirds vertical grid lines.
- Recommend color adjustments and speed curve profiles to enhance visual mood.
- Return JSON format ONLY matching the strict schema below.
Output Schema:
{
  "has_duplicate": true,
  "primary_subject_id": "obj_01",
  "duplicate_ids": ["obj_02"],
  "recommended_action": "ZOOM_CROP",
  "crop_bounding_box": [120, 0, 1800, 1080],
  "zoom_factor": 1.45,
  "color_grade_preset": "Cinematic_Teal_Orange",
  "speed_curve_preset": "Hero_Time",
  "cinematic_reasoning": "Excludes duplicate raccoon on right boundary while applying a dynamic speed ramp on lead character."
}"""


@dataclass
class DirectorDetection:
    id: str
    label: str
    bbox: List[float]
    confidence: float


@dataclass
class DirectorRequest:
    frame_dimensions: List[int]
    detected_objects: List[DirectorDetection] = field(default_factory=list)

    def to_user_message(self) -> str:
        return json.dumps({
            "frame_dimensions": self.frame_dimensions,
            "detected_objects": [asdict(d) for d in self.detected_objects],
        })


@dataclass
class DirectorResponse:
    has_duplicate: bool
    primary_subject_id: Optional[str]
    duplicate_ids: List[str]
    recommended_action: str
    crop_bounding_box: Optional[List[float]]
    zoom_factor: Optional[float]
    color_grade_preset: Optional[str]
    speed_curve_preset: Optional[str]
    cinematic_reasoning: str
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "DirectorResponse":
        crop = d.get("crop_bounding_box")
        if isinstance(crop, (list, tuple)) and len(crop) == 4:
            try:
                crop = [float(v) for v in crop]
            except (TypeError, ValueError):
                crop = None
        else:
            crop = None
        zoom = d.get("zoom_factor")
        try:
            zoom = float(zoom) if zoom is not None else None
        except (TypeError, ValueError):
            zoom = None
        dup_ids = d.get("duplicate_ids") or []
        return cls(
            has_duplicate=bool(d.get("has_duplicate", False)),
            primary_subject_id=d.get("primary_subject_id"),
            duplicate_ids=[str(x) for x in dup_ids] if isinstance(dup_ids, list) else [],
            recommended_action=str(d.get("recommended_action", "NONE")),
            crop_bounding_box=crop,
            zoom_factor=zoom,
            color_grade_preset=d.get("color_grade_preset"),
            speed_curve_preset=d.get("speed_curve_preset"),
            cinematic_reasoning=str(d.get("cinematic_reasoning", "")),
            raw=d,
        )


def _encode_jpeg(frame_bgr: np.ndarray, max_width: int = 768) -> Optional[str]:
    try:
        import cv2

        h, w = frame_bgr.shape[:2]
        if w > max_width:
            frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None


def ollama_reachable(ollama_url: str = "http://127.0.0.1:11434", timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(ollama_url.rstrip("/") + "/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def ask_director(frame_w: int, frame_h: int, detections: Sequence[DirectorDetection],
                 ollama_url: str = "http://127.0.0.1:11434", model: str = "qwen2.5-vl",
                 frame_bgr: Optional[np.ndarray] = None, timeout: float = 120.0) -> Optional[DirectorResponse]:
    """POST to Ollama ``/api/chat`` with ``format: "json"``; ``None`` if unreachable / invalid."""
    req = DirectorRequest([int(frame_w), int(frame_h)], list(detections))
    user: Dict[str, Any] = {"role": "user", "content": req.to_user_message()}
    if frame_bgr is not None:
        img = _encode_jpeg(frame_bgr)
        if img:
            user["images"] = [img]
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
        "messages": [{"role": "system", "content": DIRECTOR_SYSTEM_PROMPT}, user],
    }
    data = json.dumps(payload).encode("utf-8")
    http_req = urllib.request.Request(ollama_url.rstrip("/") + "/api/chat", data=data,
                                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(http_req, timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
        log.info("AI director skipped (Ollama not reachable / invalid reply): %s", exc)
        return None
    content = (body.get("message") or {}).get("content", "")
    try:
        parsed = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    return DirectorResponse.from_json(parsed)
