"""
text_gate.py — Stage 1 validator. Cheap LLM scores video metadata vs beat brief.

Inputs:
  - video_meta: dict with title/channel/channel_id/description/tags/duration
  - beat: dict with visual.subject, visual.geo, visual.queries, visual.strictness

Decision rule (caller's job):
  score >= TEXT_GATE_ACCEPT  → accept, skip vision
  score >= TEXT_GATE_REJECT  → send to vision_gate
  score <  TEXT_GATE_REJECT  → reject

Allowlist channels get a +20 boost so they reliably skip vision.
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional, Set

from ..cache import Cache, beat_hash, get_cache
from ..config import (
    CHANNEL_ALLOWLIST, CHANNEL_BLOCKLIST, TEXT_GATE_ACCEPT, TEXT_GATE_MODEL,
    TEXT_GATE_REJECT,
)
from ..gemini_client import GeminiError, call_json

_ALLOWLIST_CACHE: Optional[Dict[str, Any]] = None
_CHANNEL_BLOCKLIST_CACHE: Optional[Dict[str, Any]] = None

_PERSONALITY_REJECT_RE = re.compile(
    r"(?i)(?:"
    r"\bmeet(?:\s|-)?up\b|"
    r"\bhang(?:\s|-)?out\b|\bhanging\s+out\b|"
    r"\bmy\s+channel\b|\bour\s+channel\b|"
    r"\blet'?s\s+(?:see|go)\b|"
    r"\bmy\s+friends\b|\bwith\s+my\b|\bgroup\s+of\b|"
    r"\bguys\b|\bbros\b|\bhomies\b|"
    r"\blike\s+and\s+subscribe\b|\bsubscribed?\b|"
    r"\bpilots?\b|\bpilot\s+meet(?:\s|-)?up\b|\bpilot\s+meet\b|"
    r"\bpodcast\b|\blive\s+stream\b|\blivestream\b|\bstream\s+with\b|"
    r"\bq\s*&\s*a\b|\bq\s+and\s+a\b|\bama\b|\bask\s+me\s+anything\b|"
    r"\breacts?\b|\breaction\b|\breacting\b|"
    r"#?shorts?\b|\bshorts\s+compilation\b|"
    r"\bwalking\b|\bpov\s+walk\b|\bstreet\s+walk\b|\bvlog\b|"
    r"\bi\s+went\s+to\b|\bmy\s+trip\b|\bwe\s+visited\b|"
    r"\btrying\b.+\bfood\b|\binterview\b"
    r")"
)


def _load_allowlist() -> Dict[str, Any]:
    global _ALLOWLIST_CACHE
    if _ALLOWLIST_CACHE is not None:
        return _ALLOWLIST_CACHE
    p = Path(CHANNEL_ALLOWLIST)
    if not p.exists():
        _ALLOWLIST_CACHE = {"topic_classes": {}, "blocklist": []}
        return _ALLOWLIST_CACHE
    try:
        _ALLOWLIST_CACHE = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        _ALLOWLIST_CACHE = {"topic_classes": {}, "blocklist": []}
    return _ALLOWLIST_CACHE


def _load_channel_blocklist() -> Dict[str, Any]:
    global _CHANNEL_BLOCKLIST_CACHE
    if _CHANNEL_BLOCKLIST_CACHE is not None:
        return _CHANNEL_BLOCKLIST_CACHE
    p = Path(CHANNEL_BLOCKLIST)
    if not p.exists():
        _CHANNEL_BLOCKLIST_CACHE = {"blocked_channel_ids": [], "blocked_patterns": []}
        return _CHANNEL_BLOCKLIST_CACHE
    try:
        _CHANNEL_BLOCKLIST_CACHE = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        _CHANNEL_BLOCKLIST_CACHE = {"blocked_channel_ids": [], "blocked_patterns": []}
    return _CHANNEL_BLOCKLIST_CACHE


def _allowlist_boost(channel_id: str, topic_class: str) -> int:
    al = _load_allowlist()
    groups = list((al.get("topic_classes") or {}).values())
    groups.extend((al.get("kinds") or {}).values())
    for group in groups:
        for ch in group.get("channels", []):
            cid = ch.get("channel_id") or ch.get("id") or ""
            if cid == channel_id:
                return 20
    return 0


def is_blocklisted(channel_id: str) -> bool:
    al = _load_allowlist()
    for ch in al.get("blocklist", []):
        cid = ch.get("channel_id") or ch.get("id") or ""
        if cid and cid == channel_id:
            return True
    return False


def _channel_blocklist_reason(video_meta: Dict[str, Any]) -> Optional[str]:
    data = _load_channel_blocklist()
    blocked_ids = {str(x).lower() for x in data.get("blocked_channel_ids") or []}
    channel_fields = [
        video_meta.get("channel_id"),
        video_meta.get("channel"),
        video_meta.get("channel_handle"),
        video_meta.get("uploader_handle"),
        video_meta.get("uploader_id"),
    ]
    for value in channel_fields:
        key = str(value or "").lower().strip("@")
        if key and (key in blocked_ids or f"@{key}" in blocked_ids):
            return f'channel blocklisted channel="{value}"'

    blob = " ".join(str(video_meta.get(k) or "") for k in ("title", "channel", "description")).lower()
    for pattern in data.get("blocked_patterns") or []:
        pat = str(pattern or "").strip().lower()
        if pat and pat in blob:
            return f'channel pattern blocklisted pattern="{pat}"'
    return None


def _metadata_hard_reject_reason(video_meta: Dict[str, Any], beat: Dict[str, Any]) -> Optional[str]:
    visual = beat.get("visual") or {}
    kind = str(visual.get("kind") or "").strip().lower()
    if kind not in {"drone_aerial", "landmark"}:
        return None
    title = str(video_meta.get("title") or "")
    match = _PERSONALITY_REJECT_RE.search(title)
    if not match:
        return None
    snippet = title.replace('"', "'")[:120]
    return f'reject meetup=true token="{match.group(0)}" title="{snippet}"'


_PROMPT = textwrap.dedent("""\
    You are a footage-relevance judge for a YouTube Shorts pipeline.
    Score how well the candidate video MATCHES the beat brief on a 0-100 scale.

    --- BEAT BRIEF ---
    Place (geography):     __GEO__
    Strictness:            __STRICTNESS__   (strict = exact place required; loose = same biome/region OK)
    Visual we want:        __SUBJECT__
    Visual kind:            __KIND__
    Search queries used:   __QUERIES__

    --- CANDIDATE VIDEO ---
    Title:        __TITLE__
    Channel:      __CHANNEL__
    Description:  __DESC__
    Tags:         __TAGS__
    Duration:     __DURATION__ seconds

    Scoring guide (be GENEROUS — we'd rather waste a vision call than miss good footage):
    100 = title names the exact place AND describes the visual we want
     85 = title clearly mentions the place, visual kind matches
     70 = title mentions the place OR a clearly nearby/equivalent place
     55 = right country/region, visual kind plausible (route to vision gate)
     40 = generic match, usually too weak unless other metadata supports it
     20 = probably wrong place or wrong visual kind
      0 = obviously unrelated content (e.g. cooking video for a city beat)

    HARD REJECTS (score 0) — be conservative, only when CERTAIN:
    - Title explicitly names a different country/region than the beat asks for
    - Video duration is < 15s or > 7200s
    - Channel name screams reupload-spam compilation farm ("Top10Lists", "BestOf2025Videos")
    - Visual kind is "drone_aerial" or "landmark" AND title clearly indicates
      handheld/personality content: "walking", "POV walk", "street walk",
      "vlog", "I went to", "my trip", "we visited", "meetup", "meet up",
      "meet-up", "hangout", "hang out", "hanging out", "my channel",
      "our channel", "let's see", "lets see", "let's go", "lets go",
      "my friends", "with my", "group of", "guys", "bros", "homies",
      "subscribe", "subscribed", "like and subscribe", "pilots",
      "pilot meet", "pilot meetup", "podcast", "live stream", "livestream",
      "stream with", "Q&A", "q and a", "AMA", "ask me anything", "react",
      "reacts", "reaction", "reacting", "trying ___ food", "interview",
      "shorts", "shorts compilation"
      (these will be shaky ground-level footage, not aerial). NOTE: "tour" by
      itself is fine — many drone showreels are titled "aerial tour" / "city tour".

    BIG BONUS (+15) when kind="drone_aerial" and title mentions any of:
    "drone", "aerial", "bird's eye", "from above", "overhead", "4K cinematic",
    "DJI", "Mavic" — these are the smooth elevated shots we actually want.

    DO NOT reject for:
    - Non-Latin script in title (Japanese/Chinese/Korean/Cyrillic channels are LEGIT)
    - Generic-sounding channel names — we judge by the title content
    - Lack of "4K" or "HD" in title — quality is checked at download time

    Return STRICT JSON: {"score": int 0-100, "reason": "one short clause"}
""")


def _negative_keyword_penalty(video_meta: Dict[str, Any], negative_terms: Set[str]) -> tuple[int, list[str]]:
    if not negative_terms:
        return 0, []
    title = str(video_meta.get("title") or "").lower()
    hits = [term for term in sorted(negative_terms) if term and term.lower() in title]
    if not hits:
        return 0, []
    return -30, hits[:3]


def score(video_meta: Dict[str, Any], beat: Dict[str, Any], topic_class: str,
          cache: Optional[Cache] = None,
          negative_terms: Optional[Set[str]] = None) -> Dict[str, Any]:
    """Returns {score, reason, accept, send_to_vision, reject, cached}."""
    cache = cache or get_cache()
    bh = beat_hash(beat)
    video_id = video_meta["video_id"]
    channel_id = video_meta.get("channel_id") or ""
    negative_terms = negative_terms or set()

    channel_penalty = 0
    channel_reason = ""
    if is_blocklisted(channel_id):
        channel_penalty = -100
        channel_reason = "channel blocklisted"
    block_reason = _channel_blocklist_reason(video_meta)
    if block_reason:
        channel_penalty = -100
        channel_reason = block_reason
    if channel_reason:
        print(f"[text-gate] {channel_reason}", flush=True)
    hard_reason = _metadata_hard_reject_reason(video_meta, beat)
    if hard_reason:
        print(f"[text-gate] {hard_reason}", flush=True)
        return _decision(0, hard_reason, cached=False)

    # Cache hit
    cached = cache.get_text_verdict(video_id, bh, TEXT_GATE_MODEL)
    if cached:
        boost = _allowlist_boost(channel_id, topic_class)
        neg_penalty, neg_hits = _negative_keyword_penalty(video_meta, negative_terms)
        reason = str(cached["reason"] or "")
        if neg_hits:
            reason = f"{reason}; negative keyword {', '.join(neg_hits)}"[:120]
        if channel_reason:
            reason = f"{reason}; {channel_reason}"[:120]
        decision = _decision(int(cached["score"]) + boost + channel_penalty + neg_penalty, reason, cached=True)
        decision["negative_keyword_penalty"] = neg_penalty
        decision["negative_keyword_hits"] = neg_hits
        return decision

    visual = beat.get("visual") or {}
    prompt = (_PROMPT
        .replace("__GEO__", str(visual.get("geo") or ""))
        .replace("__STRICTNESS__", str(visual.get("strictness") or "loose"))
        .replace("__SUBJECT__", str(visual.get("subject") or ""))
        .replace("__KIND__", str(visual.get("kind") or ""))
        .replace("__QUERIES__", json.dumps(visual.get("queries") or []))
        .replace("__TITLE__", str(video_meta.get("title") or "")[:200])
        .replace("__CHANNEL__", str(video_meta.get("channel") or ""))
        .replace("__DESC__", str(video_meta.get("description") or "")[:600])
        .replace("__TAGS__", json.dumps(video_meta.get("tags") or [])[:300])
        .replace("__DURATION__", str(int(video_meta.get("duration") or 0)))
    )

    try:
        parsed, _ = call_json(TEXT_GATE_MODEL, prompt, temperature=0.1, max_retries=2)
    except GeminiError as e:
        return _decision(0, f"gate error: {e}", cached=False)

    raw = max(0, min(100, int(parsed.get("score") or 0)))
    reason = str(parsed.get("reason") or "")[:120]
    cache.put_text_verdict(video_id, bh, raw, reason, TEXT_GATE_MODEL)

    neg_penalty, neg_hits = _negative_keyword_penalty(video_meta, negative_terms)
    final = raw + _allowlist_boost(channel_id, topic_class) + channel_penalty + neg_penalty
    if neg_hits:
        reason = f"{reason}; negative keyword {', '.join(neg_hits)}"[:120]
    if channel_reason:
        reason = f"{reason}; {channel_reason}"[:120]
    decision = _decision(final, reason, cached=False)
    decision["negative_keyword_penalty"] = neg_penalty
    decision["negative_keyword_hits"] = neg_hits
    return decision


def _decision(score_val: int, reason: str, cached: bool) -> Dict[str, Any]:
    return {
        "score": score_val,
        "reason": reason,
        "accept": score_val >= TEXT_GATE_ACCEPT,
        "send_to_vision": TEXT_GATE_REJECT <= score_val < TEXT_GATE_ACCEPT,
        "reject": score_val < TEXT_GATE_REJECT,
        "cached": cached,
    }
