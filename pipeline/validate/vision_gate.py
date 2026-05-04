"""
vision_gate.py — Stage 2 validator. One frame, one Vision call, cached.

Runs only on candidates the text gate scored 60–84 (configurable).
Caches by (video_id, start_ts, beat_hash, model).
"""
from __future__ import annotations

import base64
import hashlib
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..cache import Cache, beat_hash, get_cache
from ..config import (
    VISION_GATE_FACE_REJECT,
    VISION_GATE_FRAMES,
    VISION_GATE_STATIC_REJECT,
    VISION_MODEL,
)
from ..gemini_client import GeminiError, call, call_json

_PROMPT = textwrap.dedent("""\
    Look at this single frame from a candidate video clip and decide if it
    matches the beat brief.

    --- BEAT BRIEF ---
    Place:        __GEO__
    Strictness:   __STRICTNESS__   (strict = exact place; loose = right region/biome OK)
    Visual want:  __SUBJECT__
    Visual kind:  __KIND__

    Pass criteria:
    - Strict beats: the frame must plausibly be from the named place. Look for
      landmarks, signage, vegetation, climate, architecture cues. If the place
      is obscure (small city), accept any plausible match — do NOT reject just
      because you can't recognize it.
    - Loose beats: the frame must show the right BIOME and visual KIND. The
      exact place doesn't matter.

    Hard fails:
    - Frame is mostly UI/text/intro card (not actual footage)
    - Watermark covers >25% of frame
    - Wrong continent (e.g. Sahara when beat asks for Greenland)
    - AI-generated artifacts (impossible geometry, melted faces)
    - Visual kind is "drone_aerial" but frame is clearly NOT aerial. Aerial
      means: shot from significant elevation (drone, helicopter, rooftop,
      mountainside) showing landscape/city from above OR a level horizon from
      high vantage. Reject if the frame shows: ground-level/eye-level street
      view, interior shots, a person filming with phone, or a low-angle handheld
      perspective. The horizon should be at or above mid-frame and the camera
      should clearly be elevated.

    Return STRICT JSON: {"passed": true|false, "reason": "one short clause"}
""")

_PROMINENT_FACE_PROMPT = (
    "Does this frame contain a person's face that takes up more than a small "
    "portion of the image? Answer only YES or NO."
)
_SIMILAR_FRAME_PROMPT = (
    "Is this frame visually similar to the previous one? Answer only YES or NO."
)


def _extract_frame(clip_path: Path, at_sec: float) -> Optional[Path]:
    """Pull one JPEG at `at_sec` into a temp file. Caller must delete.

    Uses a uuid-named path that we DON'T pre-create — ffmpeg writes a fresh
    file. Pre-creating with mkstemp + ffmpeg -y (overwrite) races with the
    Windows file lock and trips WinError 32 under thread concurrency.
    """
    import uuid
    tmp = Path(tempfile.gettempdir()) / f"v2vg_{uuid.uuid4().hex}.jpg"

    cmd = [
        "ffmpeg", "-y", "-ss", f"{at_sec:.2f}", "-i", str(clip_path),
        "-frames:v", "1", "-q:v", "3", "-f", "image2", str(tmp),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if r.returncode != 0 or not tmp.exists():
            return None
    except subprocess.TimeoutExpired:
        return None
    return tmp


def _sample_times(duration: float, n: int) -> List[float]:
    n = max(1, int(n or 1))
    duration = max(0.5, float(duration or 5.0))
    if n == 1:
        return [max(0.5, min(duration / 2.0, 4.0))]
    if n == 2:
        raw = [0.6, max(0.6, duration * 0.55)]
    else:
        raw = [0.6, max(0.6, duration / 2.0), max(0.6, duration - 0.6)]
    out: List[float] = []
    for t in raw[:n]:
        t = max(0.05, min(duration - 0.05, t))
        if not out or abs(t - out[-1]) > 0.15:
            out.append(t)
    return out or [0.5]


def _read_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _yes_no_images(
    prompt: str,
    image_paths: List[Path],
    video_id: str,
    cache_key: str,
    *,
    cache: Cache,
) -> bool:
    prompt_hash = hashlib.sha256(f"{cache_key}:{prompt}".encode("utf-8")).hexdigest()[:16]
    cached = cache.get_vision_verdict(video_id, 0.0, prompt_hash, VISION_MODEL)
    if cached:
        return bool(cached["passed"])
    try:
        images = [
            {"mime_type": "image/jpeg", "data": _read_b64(p)}
            for p in image_paths
        ]
        text, _ = call(
            VISION_MODEL,
            prompt,
            images=images,
            temperature=0.0,
            max_retries=2,
            timeout=30,
        )
        passed = text.strip().upper().startswith("YES")
    except (OSError, GeminiError):
        passed = False
    cache.put_vision_verdict(
        video_id, 0.0, prompt_hash, passed, "YES" if passed else "NO", VISION_MODEL
    )
    return passed


def check(clip_path: Path, beat: Dict[str, Any], video_id: str,
          start_ts: float, *, cache: Optional[Cache] = None) -> Dict[str, Any]:
    """Returns {passed, reason, cached, elapsed_sec}."""
    cache = cache or get_cache()
    bh = (
        f"{beat_hash(beat)}:frames{VISION_GATE_FRAMES}:"
        f"face{int(bool(VISION_GATE_FACE_REJECT))}:static{int(bool(VISION_GATE_STATIC_REJECT))}"
    )

    cached = cache.get_vision_verdict(video_id, start_ts, bh, VISION_MODEL)
    if cached:
        return {
            "passed": bool(cached["passed"]),
            "reason": cached["reason"],
            "cached": True,
            "elapsed_sec": 0.0,
        }

    duration = float(beat.get("duration_sec") or 5.0)
    frame_paths: List[Path] = []
    for frame_at in _sample_times(duration, VISION_GATE_FRAMES):
        frame_path = _extract_frame(clip_path, frame_at)
        if frame_path:
            frame_paths.append(frame_path)
    if not frame_paths:
        return {"passed": False, "reason": "frame extract failed",
                "cached": False, "elapsed_sec": 0.0}

    visual = beat.get("visual") or {}
    prompt = (_PROMPT
        .replace("__GEO__", str(visual.get("geo") or ""))
        .replace("__STRICTNESS__", str(visual.get("strictness") or "loose"))
        .replace("__SUBJECT__", str(visual.get("subject") or ""))
        .replace("__KIND__", str(visual.get("kind") or ""))
    )

    elapsed_total = 0.0
    try:
        face_count = 0
        if VISION_GATE_FACE_REJECT:
            for idx, path in enumerate(frame_paths):
                if _yes_no_images(
                    _PROMINENT_FACE_PROMPT,
                    [path],
                    video_id,
                    f"{bh}:face:{idx}:{start_ts:.2f}",
                    cache=cache,
                ):
                    face_count += 1
            if len(frame_paths) >= 2 and face_count >= min(2, len(frame_paths)):
                reason = f"prominent faces in {face_count}/{len(frame_paths)} frames"
                print(f"[vision-gate] reject faces={face_count}/{len(frame_paths)} video_id={video_id}", flush=True)
                cache.put_vision_verdict(video_id, start_ts, bh, False, reason, VISION_MODEL)
                return {"passed": False, "reason": reason, "cached": False, "elapsed_sec": 0.0}

        similar_count = 0
        if VISION_GATE_STATIC_REJECT and len(frame_paths) >= 2:
            for idx in range(1, len(frame_paths)):
                if _yes_no_images(
                    _SIMILAR_FRAME_PROMPT,
                    [frame_paths[idx - 1], frame_paths[idx]],
                    video_id,
                    f"{bh}:similar:{idx}:{start_ts:.2f}",
                    cache=cache,
                ):
                    similar_count += 1
            if similar_count >= max(1, len(frame_paths) - 1):
                reason = f"static scene {similar_count}/{len(frame_paths) - 1} frame pairs"
                print(f"[vision-gate] reject static={similar_count}/{len(frame_paths) - 1} video_id={video_id}", flush=True)
                cache.put_vision_verdict(video_id, start_ts, bh, False, reason, VISION_MODEL)
                return {"passed": False, "reason": reason, "cached": False, "elapsed_sec": 0.0}

        passed_frames = 0
        reasons: List[str] = []
        for path in frame_paths:
            parsed, elapsed = call_json(
                VISION_MODEL, prompt,
                images=[{"mime_type": "image/jpeg", "data": _read_b64(path)}],
                temperature=0.1, max_retries=2, timeout=30,
            )
            elapsed_total += float(elapsed or 0.0)
            if bool(parsed.get("passed")):
                passed_frames += 1
            reasons.append(str(parsed.get("reason") or "")[:60])
    except GeminiError as e:
        return {"passed": False, "reason": f"vision error: {e}",
                "cached": False, "elapsed_sec": 0.0}
    finally:
        for frame_path in frame_paths:
            for _ in range(3):
                try:
                    frame_path.unlink(missing_ok=True)
                    break
                except PermissionError:
                    import time as _t; _t.sleep(0.1)

    passed = passed_frames >= 1
    reason = f"relevant frames {passed_frames}/{len(frame_paths)}"
    if reasons:
        reason = f"{reason}: {reasons[0]}"[:120]
    cache.put_vision_verdict(video_id, start_ts, bh, passed, reason, VISION_MODEL)

    return {"passed": passed, "reason": reason, "cached": False,
            "elapsed_sec": round(elapsed_total, 2)}


def relevance_check(image_path: Path, prompt: str, cache_key: str,
                    *, cache: Optional[Cache] = None) -> Dict[str, Any]:
    """YES/NO relevance gate for a still image, cached by image URL + prompt.

    The prompt may ask for a short reason after YES/NO; preserve that answer so
    callers can log why geographic context-card candidates passed or failed.
    """
    cache = cache or get_cache()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    verdict_key = f"context-card:{hashlib.sha256(cache_key.encode('utf-8')).hexdigest()[:32]}"

    cached = cache.get_vision_verdict(verdict_key, 0.0, prompt_hash, VISION_MODEL)
    if cached:
        return {
            "passed": bool(cached["passed"]),
            "reason": cached["reason"],
            "cached": True,
            "elapsed_sec": 0.0,
        }

    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        text, elapsed = call(
            VISION_MODEL,
            prompt,
            images=[{"mime_type": "image/jpeg", "data": b64}],
            temperature=0.0,
            max_retries=2,
            timeout=30,
        )
    except (OSError, GeminiError) as e:
        reason = f"vision error: {e}"[:120]
        cache.put_vision_verdict(verdict_key, 0.0, prompt_hash, False, reason, VISION_MODEL)
        return {"passed": False, "reason": reason, "cached": False, "elapsed_sec": 0.0}

    answer = text.strip()
    passed = answer.upper().startswith("YES")
    reason = re.sub(r"\s+", " ", answer)[:120] if answer else ("YES" if passed else "NO")
    cache.put_vision_verdict(verdict_key, 0.0, prompt_hash, passed, reason, VISION_MODEL)
    return {"passed": passed, "reason": reason, "cached": False,
            "elapsed_sec": round(elapsed, 2)}
