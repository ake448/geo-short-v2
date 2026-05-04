"""
Parallel multi-tier footage sourcing.

Per-beat flow:
  1. Generate candidate video_ids (Tier 1 allowlist + Tier 2 broad search)
  2. For each candidate, in parallel up to SOURCING_PARALLEL_CANDIDATES:
       a. fetch_metadata (cached)
       b. text_gate.score
          - score >= ACCEPT          → download → return
          - REJECT <= score < ACCEPT → download → vision_gate
                                         → pass → return
                                         → fail → next candidate
          - score < REJECT           → skip (no download)
  3. If no YouTube candidate wins AND beat is loose → try Pexels/Pixabay
  4. If still nothing → return None (caller falls back to generated cinematic)

Cross-beat: SOURCING_PARALLEL_BEATS beats run concurrently via ThreadPoolExecutor.
"""
from __future__ import annotations

import time
import threading
import io
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..cache import get_cache
from ..config import (
    DATA_DIR,
    SOURCING_PARALLEL_CANDIDATES,
    SOURCING_MAX_CANDIDATES, TEXT_GATE_ACCEPT, TEXT_GATE_REJECT,
    VISUAL_DEDUP_PHASH, VISUAL_DEDUP_THRESHOLD,
)
from ..validate import text_gate, vision_gate
from . import pexels, youtube


# Kinds that are always rendered by cinematics.make_fallback — never sourced.
GENERATED_KINDS = {
    "map_zoom",
    "map_animation",
    "density_infographic",
    "dynamic_infographic",
    "wikipedia_photo",
    "archival_annotated",
}
# Real-footage kinds sourced from YouTube / Pexels.
FOOTAGE_KINDS = {"street_level", "drone_aerial", "landmark"}


@dataclass
class SourcingContext:
    used_video_ids: Set[str]
    claimed_video_beats: Dict[str, List[int]]
    used_visual_terms: Set[str]
    claimed_phashes: List[Dict[str, Any]]
    used_winner_titles: List[Dict[str, Any]]
    used_lock: threading.Lock


PHASH_CACHE = DATA_DIR / "phash_cache.json"
VIDEO_REUSE_MIN_BEAT_GAP = 5
VISUAL_DUP_MIN_BEAT_GAP = 4


_TERM_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "over", "under", "near",
    "centralia", "pennsylvania", "lake", "lanier", "georgia", "america",
    "usa", "us", "abandoned", "exploring", "explore", "drone", "aerial",
    "tour", "walk", "walking", "drive", "driving", "footage", "video",
    "cinematic", "documentary", "shorts", "short", "history", "dark",
    "real", "city", "town", "county", "state", "4k", "hd", "uhd",
}
_TERM_CUE_WORDS = {
    "highway", "route", "road", "street", "avenue", "turnpike", "bridge",
    "tunnel", "trail", "mine", "cemetery", "church", "school", "park",
    "river", "creek", "dam", "falls", "island", "mountain", "mill",
}

_VISUAL_SUBJECT_GATE_RE = re.compile(
    r"\b("
    r"smoke|smoking|steam|steaming|vent|vents|"
    r"crack(?:ed|s)?|charred|burn(?:ing|ed)?|fire|"
    r"sinkhole|collapsed?|demolished|demolition|razed|"
    r"empty lots?|overgrown|rubble|debris|"
    r"row houses?|mailbox|post office|cemetery|graveyard|"
    r"highway|route|mine|quarry|"
    r"abandoned|desolate|scarred|ruins?|dilapidated|"
    r"flooded|submerged|underwater|dried up|"
    r"toxic|contaminated|polluted|superfund|"
    r"crater|erosion|eroded|landslide|"
    r"ghost town|boarded up|vacant|shuttered|"
    r"barren|wasteland|scorched"
    r")\b",
    re.IGNORECASE,
)


def _requires_visual_subject_gate(beat: Dict[str, Any]) -> bool:
    visual = beat.get("visual") if isinstance(beat.get("visual"), dict) else {}
    kind = str(visual.get("kind") or "").strip().lower()
    if kind not in FOOTAGE_KINDS:
        return False
    strict = str(visual.get("strictness") or "").strip().lower() == "strict"
    subject = str(visual.get("subject") or "")
    if strict and _VISUAL_SUBJECT_GATE_RE.search(subject):
        return True
    geo = str(visual.get("geo") or "").strip().lower()
    if kind in ("drone_aerial", "street_level", "landmark") and subject and geo:
        subject_lower = subject.lower()
        if subject_lower != geo and not subject_lower.startswith(geo):
            if _VISUAL_SUBJECT_GATE_RE.search(subject):
                return True
    return False


def _geo_ladder(geo: str) -> List[str]:
    """Yield progressively broader geo scopes.
    "Summerhill, Atlanta, GA" -> ["Summerhill, Atlanta, GA", "Atlanta, GA", "GA"].
    """
    parts = [p.strip() for p in geo.split(",") if p.strip()]
    return [", ".join(parts[i:]) for i in range(len(parts))]


def _mutate_beat_for_kind(beat: Dict[str, Any], new_kind: str, geo: str) -> Dict[str, Any]:
    """Clone beat with a different visual.kind + regenerated queries for that kind."""
    new_beat = dict(beat)
    new_visual = dict(beat.get("visual") or {})
    new_visual["kind"] = new_kind
    new_visual["geo"] = geo
    # Manufacture queries from the per-kind templates
    queries: List[str] = []
    for tmpl in youtube.search_templates_for_kind(new_kind)[:3]:
        try:
            queries.append(tmpl.format(
                place=geo, city=geo, region=geo, feature=geo,
                subject=new_visual.get("subject", geo),
            ))
        except KeyError:
            queries.append(f"{geo} {new_kind.replace('_', ' ')}")
    
    # BROADEN SUBJECT: If kind or geo changed, the hyperspecific L0 subject description
    # might cause the Text Gate to reject good generic footage (e.g. "aerial" vs "walk").
    # We replace it with a cleaner, high-level subject for the fallback seekers.
    if new_kind != beat.get("visual", {}).get("kind") or geo != beat.get("visual", {}).get("geo"):
        new_visual["subject"] = f"Cinematic {new_kind.replace('_', ' ')} of {geo}"
        
    new_visual["queries"] = queries
    new_beat["visual"] = new_visual
    return new_beat


def _load_phash_cache() -> Dict[str, Any]:
    if not PHASH_CACHE.exists():
        return {}
    try:
        return json.loads(PHASH_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_phash_cache(data: Dict[str, Any]) -> None:
    PHASH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PHASH_CACHE.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _phash_key(video_id: str, segment_start: float) -> str:
    return f"{video_id}|{segment_start:.2f}"


def _compute_frame_phash(frame_path: Path) -> Optional[int]:
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return None


def _compute_frame_phash_bytes(frame_data: bytes) -> Optional[int]:
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(io.BytesIO(frame_data)) as src:
            img = src.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.array(img, dtype=np.float32, copy=True)
        img.close()
        n = 32
        u = np.arange(n)
        x = np.arange(n)
        basis = np.cos(((2 * x[:, None] + 1) * u[None, :] * np.pi) / (2 * n))
        coeff = basis.T @ pixels @ basis
        low = coeff[:8, :8].copy()
        vals = low.flatten()
        median = float(np.median(vals[1:]))
        bits = vals > median
        value = 0
        for bit in bits:
            value = (value << 1) | int(bool(bit))
        return value
    except Exception:
        return None
    try:
        with Image.open(frame_path) as src:
            img = src.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.array(img, dtype=np.float32, copy=True)
        img.close()
        n = 32
        u = np.arange(n)
        x = np.arange(n)
        basis = np.cos(((2 * x[:, None] + 1) * u[None, :] * np.pi) / (2 * n))
        coeff = basis.T @ pixels @ basis
        low = coeff[:8, :8].copy()
        vals = low.flatten()
        median = float(np.median(vals[1:]))
        bits = vals > median
        value = 0
        for bit in bits:
            value = (value << 1) | int(bool(bit))
        return value
    except Exception:
        return None


def _segment_phash(video_id: str, segment_start: float, clip_path: Path, dur: float) -> Optional[str]:
    key = _phash_key(video_id, segment_start)
    cache = _load_phash_cache()
    hit = cache.get(key)
    if isinstance(hit, str) and re.fullmatch(r"[0-9a-fA-F]{16}", hit):
        return hit.lower()

    sample_ts = max(0.1, min(max(0.1, dur - 0.1), dur / 2.0))
    cmd = [
        "ffmpeg", "-y", "-ss", f"{sample_ts:.3f}", "-i", str(clip_path.resolve()),
        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        if r.returncode != 0 or not r.stdout:
            return None
        value = _compute_frame_phash_bytes(r.stdout)
        if value is None:
            return None
        hex_hash = f"{value:016x}"
        cache[key] = hex_hash
        _save_phash_cache(cache)
        return hex_hash
    except Exception:
        return None


def _hamming_distance(a_hex: str, b_hex: str) -> int:
    return (int(a_hex, 16) ^ int(b_hex, 16)).bit_count()


def _context_negative_terms(context: Optional[SourcingContext]) -> Set[str]:
    if not context:
        return set()
    with context.used_lock:
        return set(context.used_visual_terms)


def _title_repeat_penalty(
    title: str, beat: Dict[str, Any], context: Optional[SourcingContext],
) -> int:
    if not context or not title:
        return 0
    title_words = set(re.findall(r"[a-z0-9']+", title.lower()))
    title_words -= _TERM_STOPWORDS
    if len(title_words) < 2:
        return 0
    current_subject = str((beat.get("visual") or {}).get("subject") or "").lower()
    with context.used_lock:
        for prior in context.used_winner_titles:
            prior_title_words = set(re.findall(r"[a-z0-9']+", prior["title"].lower()))
            prior_title_words -= _TERM_STOPWORDS
            if not prior_title_words:
                continue
            overlap = title_words & prior_title_words
            overlap_ratio = len(overlap) / max(len(prior_title_words), 1)
            if overlap_ratio < 0.5:
                continue
            prior_subject = prior.get("subject", "").lower()
            if prior_subject and current_subject and prior_subject != current_subject:
                return -50
    return 0


_SUBJECT_MISMATCH_TITLE_RE = re.compile(
    r"\b(graffiti|highway|road\s*trip|tour|review|reaction|"
    r"vlog|meetup|drive|driving|walking|hike|hiking|"
    r"fishing|kayak|camping|boating|swimming)\b",
    re.IGNORECASE,
)


def _subject_title_mismatch_penalty(
    title: str, beat: Dict[str, Any],
) -> int:
    visual = beat.get("visual") if isinstance(beat.get("visual"), dict) else {}
    subject = str(visual.get("subject") or "").lower()
    if not subject or not _VISUAL_SUBJECT_GATE_RE.search(subject):
        return 0
    title_lower = title.lower()
    subject_cue_matches = _VISUAL_SUBJECT_GATE_RE.findall(subject)
    if not subject_cue_matches:
        return 0
    for cue in subject_cue_matches:
        if cue.lower() in title_lower:
            return 0
    if _SUBJECT_MISMATCH_TITLE_RE.search(title_lower):
        return -25
    return 0


def _extract_visual_terms(title: str, beat: Dict[str, Any]) -> Set[str]:
    clean = re.sub(r"[\[\](){}|/\\:;,_-]+", " ", title or "")
    words = re.findall(r"[A-Za-z0-9']+", clean)
    if not words:
        return set()

    geo_words: Set[str] = set()
    visual = beat.get("visual") or {}
    ctx = beat.get("_script_context") or {}
    for value in (visual.get("geo"), ctx.get("place"), ctx.get("region"), ctx.get("subject")):
        geo_words.update(w.lower() for w in re.findall(r"[A-Za-z0-9']+", str(value or "")) if len(w) > 2)

    terms: List[str] = []
    for size in (3, 2):
        for i in range(0, len(words) - size + 1):
            chunk = words[i:i + size]
            lows = [w.lower() for w in chunk]
            if all(w in _TERM_STOPWORDS or w in geo_words for w in lows):
                continue
            if not (any(w in _TERM_CUE_WORDS for w in lows) or any(re.search(r"\d", w) for w in lows)):
                continue
            trimmed = [w for w in chunk if w.lower() not in geo_words and w.lower() not in _TERM_STOPWORDS]
            if len(trimmed) < 2 and not any(re.search(r"\d", w) for w in trimmed):
                continue
            term = " ".join(trimmed or chunk).strip()
            term_l = term.lower()
            if len(term_l) >= 5 and term_l not in {t.lower() for t in terms}:
                terms.append(term_l)
            if len(terms) >= 3:
                return set(terms)
    return set(terms[:3])


def source_beat(beat: Dict[str, Any], topic_class: str, run_dir: Path,
                verbose: bool = True,
                sourcing_context: Optional[SourcingContext] = None) -> Dict[str, Any]:
    """Source one beat with a degradation ladder:
       L0  exact beat (YouTube Tier 0+1+2)
       L1  widen geo progressively, same kind
       L2  swap kind within footage group, same widened place
       L3  Pexels/Pixabay (loose beats only)
       L4  mutate kind to map_animation so caller renders Mapbox cinematic
    Cinematic kinds (map_zoom etc.) short-circuit straight to L4.
    """
    t0 = time.time()
    beat_id = beat.get("beat_id", "?")
    visual = beat.get("visual") or {}
    kind = (visual.get("kind") or "").lower()
    dur = float(beat.get("duration_sec") or 5.0)

    out_path = run_dir / "clips" / f"beat_{beat_id:02d}.mp4" \
        if isinstance(beat_id, int) else run_dir / "clips" / f"beat_{beat_id}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gates: List[Dict[str, Any]] = []

    if verbose:
        print(f"  [beat {beat_id}] {kind} | {visual.get('subject', '')[:55]}", flush=True)

    # Short-circuit: generated kinds are never sourced — caller renders them.
    if kind in GENERATED_KINDS:
        if verbose:
            print(f"    [skip] kind={kind} renders via cinematics.make_fallback", flush=True)
        return _wrap(None, None, gates, t0, fallback=True)

    # L0: try exact beat as-is.
    candidates = youtube.candidates_for_beat(
        beat, topic_class, n_per_query=3, max_total=SOURCING_MAX_CANDIDATES,
    )
    if verbose:
        print(f"    {len(candidates)} candidates ({topic_class} tier 1+2)", flush=True)
    yt_winner = _try_youtube_candidates(
        candidates, beat, topic_class, out_path, gates, verbose, sourcing_context
    )
    if yt_winner:
        return _wrap(yt_winner, out_path, gates, t0, fallback=False)

    geo = str(visual.get("geo") or "")

    # L1: multi-level geo widening, same kind.
    ladder = _geo_ladder(geo)[1:]  # skip index 0 (already tried at L0)
    for broader in ladder:
        if verbose:
            print(f"    [L1] widening geo to {broader!r}", flush=True)
        widened = _mutate_beat_for_kind(beat, kind, broader)
        c = youtube.candidates_for_beat(widened, topic_class, n_per_query=3, max_total=SOURCING_MAX_CANDIDATES)
        w = _try_youtube_candidates(c, widened, topic_class, out_path, gates, verbose, sourcing_context)
        if w:
            return _wrap(w, out_path, gates, t0, fallback=True)

    # L2: cross-kind swap within footage group. Try other footage kinds at the
    # widest geo (which is most likely to have abundant b-roll).
    if kind in FOOTAGE_KINDS:
        widest_geo = ladder[-1] if ladder else geo
        alt_kinds = [k for k in ("drone_aerial", "street_level", "landmark") if k != kind]
        for alt_kind in alt_kinds:
            if verbose:
                print(f"    [L2] swap kind {kind}->{alt_kind} @ {widest_geo!r}", flush=True)
            swapped = _mutate_beat_for_kind(beat, alt_kind, widest_geo)
            c = youtube.candidates_for_beat(swapped, topic_class, n_per_query=3, max_total=SOURCING_MAX_CANDIDATES)
            w = _try_youtube_candidates(c, swapped, topic_class, out_path, gates, verbose, sourcing_context)
            if w:
                w["kind_swapped_to"] = alt_kind
                return _wrap(w, out_path, gates, t0, fallback=True)

    # L3: Pexels/Pixabay — loose beats only, footage kinds only.
    if kind in FOOTAGE_KINDS and (visual.get("strictness") or "loose") != "strict":
        if verbose:
            print(f"    [L3] trying Pexels/Pixabay", flush=True)
        px = _try_pexels_candidates(beat, out_path, gates, verbose)
        if px:
            return _wrap(px, out_path, gates, t0, fallback=True)

    # L4: mutate the beat's kind to map_animation so the caller's cinematic
    # renderer draws a Mapbox zoom of the actual location instead of a black gap.
    if verbose:
        print(f"    [L4] no footage; overriding kind -> map_animation @ {geo!r}", flush=True)
    beat["visual"]["kind"] = "map_animation"
    # Preserve the original subject + geo so the map renderer can label it.
    return _wrap(None, None, gates, t0, fallback=True)


def _try_youtube_candidates(candidates: List[str], beat: Dict[str, Any],
                            topic_class: str, out_path: Path,
                            gates: List[Dict[str, Any]],
                            verbose: bool,
                            sourcing_context: Optional[SourcingContext] = None,
                            allow_claimed_last_resort: bool = True,
                            allow_video_reuse: bool = False) -> Optional[Dict[str, Any]]:
    """Walk candidates serially with parallel metadata prefetch."""
    cache = get_cache()
    dur = float(beat.get("duration_sec") or 5.0)
    deferred_claimed: List[str] = []
    deferred_visual_dups: List[Dict[str, Any]] = []

    # Prefetch metadata for ALL candidates in parallel — this is cheap and
    # surfaces blocklist/cached-reject decisions immediately.
    metas: Dict[str, Optional[Dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=SOURCING_PARALLEL_CANDIDATES) as ex:
        futures = {ex.submit(youtube.fetch_metadata, vid): vid for vid in candidates}
        for fut in as_completed(futures):
            vid = futures[fut]
            try:
                metas[vid] = fut.result()
            except Exception:
                metas[vid] = None

    # Now walk in original order — Tier 1 channels were emitted first.
    for i, vid in enumerate(candidates):
        meta = metas.get(vid)
        if not meta:
            gates.append({"video_id": vid, "stage": "meta", "reason": "no metadata"})
            continue
            
        handle = meta.get("channel_handle") or meta.get("uploader_handle") or meta.get("channel") or ""
        if youtube.is_explainer_channel(handle):
            if i < len(candidates) - 1:
                gates.append({"video_id": vid, "stage": "meta", "reason": "explainer channel skipped"})
                if verbose:
                    print(f"      [SKIP] {vid} explainer channel penalty", flush=True)
                continue
            # If it's the last candidate, we let it pass through (text_gate might still reject it naturally)
                
        if meta.get("duration", 0) < youtube.MIN_VIDEO_SEC:
            gates.append({"video_id": vid, "stage": "meta", "reason": "too short"})
            continue

        # Stage 1: text gate
        negative_terms = _context_negative_terms(sourcing_context)
        verdict = text_gate.score(meta, beat, topic_class, cache=cache, negative_terms=negative_terms)
        already_claimed = _video_already_claimed(vid, sourcing_context)
        reuse_gap = _video_reuse_gap(vid, beat, sourcing_context) if already_claimed else None
        reuse_too_close = already_claimed and (
            reuse_gap is None or reuse_gap < VIDEO_REUSE_MIN_BEAT_GAP
        )
        duplicate_penalty = 0 if (
            already_claimed and allow_video_reuse and not reuse_too_close
        ) else (80 if already_claimed else 0)
        repeat_penalty = _title_repeat_penalty(
            str(meta.get("title") or ""), beat, sourcing_context,
        )
        mismatch_penalty = _subject_title_mismatch_penalty(
            str(meta.get("title") or ""), beat,
        )
        effective_score = int(verdict["score"]) - duplicate_penalty + repeat_penalty + mismatch_penalty
        gates.append({
            "video_id": vid, "stage": "text", "score": verdict["score"],
            "effective_score": effective_score,
            "duplicate_penalty": duplicate_penalty,
            "repeat_penalty": repeat_penalty,
            "mismatch_penalty": mismatch_penalty,
            "negative_keyword_penalty": verdict.get("negative_keyword_penalty", 0),
            "negative_keyword_hits": verdict.get("negative_keyword_hits", []),
            "reason": verdict["reason"], "cached": verdict["cached"],
        })
        needs_subject_gate = _requires_visual_subject_gate(beat)
        if verbose:
            tag = "VISION" if (needs_subject_gate and effective_score >= TEXT_GATE_ACCEPT) else \
                  "ACCEPT" if effective_score >= TEXT_GATE_ACCEPT else \
                  "VISION" if effective_score >= TEXT_GATE_REJECT else "REJECT"
            penalties = []
            if duplicate_penalty:
                penalties.append("dup=-80")
            if repeat_penalty:
                penalties.append(f"repeat={repeat_penalty}")
            if mismatch_penalty:
                penalties.append(f"mismatch={mismatch_penalty}")
            suffix = f" -> {effective_score:3d} {' '.join(penalties)}" if penalties else ""
            print(f"      [{tag}] {vid} score={verdict['score']:3d}{suffix}  {meta.get('channel','?')[:25]}", flush=True)
        if already_claimed:
            reason = "video_id already claimed by another beat"
            if reuse_too_close:
                reason = (
                    f"video_id reused too close to prior beat "
                    f"(gap={reuse_gap}, min={VIDEO_REUSE_MIN_BEAT_GAP})"
                )
            gates.append({
                "video_id": vid, "stage": "dedup",
                "reason": reason,
                "penalty": -duplicate_penalty,
                "reuse_gap": reuse_gap,
            })
            if verbose:
                print(
                    f"      [sourcing] dup reject video_id={vid} "
                    f"gap={reuse_gap} score={verdict['score']} adjusted={effective_score}",
                    flush=True,
                )
            if reuse_too_close:
                continue
            if not allow_video_reuse:
                if allow_claimed_last_resort and not verdict["reject"]:
                    deferred_claimed.append(vid)
                continue
            if allow_claimed_last_resort and not verdict["reject"]:
                deferred_claimed.append(vid)

        if effective_score < TEXT_GATE_REJECT:
            continue

        # Pick a start offset somewhere in the middle to avoid intros
        start_sec = max(0.0, (meta.get("duration", dur) - dur) / 3.0)

        if effective_score >= TEXT_GATE_ACCEPT and not needs_subject_gate:
            if youtube.download_trim(vid, start_sec, dur, out_path):
                # Harvest related videos
                kind = beat.get("visual", {}).get("kind", "unknown")
                rel = youtube.harvest_related(vid)
                if rel:
                    youtube._HARVESTED_POOL.setdefault(kind, []).extend(rel)
                if not _claim_youtube_winner(
                    vid, sourcing_context, out_path, gates, verbose, beat,
                    start_sec, meta, dur, allow_video_reuse=allow_video_reuse,
                ):
                    if gates and gates[-1].get("stage") == "visual-dedup":
                        deferred_visual_dups.append({
                            "video_id": vid, "start_sec": start_sec, "meta": meta,
                            "matched_beat": gates[-1].get("matched_beat"),
                        })
                    continue
                return {"video_id": vid, "source": "youtube", "start_sec": start_sec}
            gates.append({"video_id": vid, "stage": "download", "reason": "trim failed"})
            if verbose:
                print(f"        [download fail] {vid}", flush=True)
            continue

        # send_to_vision: download first, then validate
        if needs_subject_gate and effective_score >= TEXT_GATE_ACCEPT:
            gates.append({
                "video_id": vid,
                "stage": "visual-subject",
                "reason": "strict beat requires frame-level subject validation despite high text score",
            })
            if verbose:
                print(f"        [visual-subject] forcing vision gate for strict subject", flush=True)
        if not youtube.download_trim(vid, start_sec, dur, out_path):
            gates.append({"video_id": vid, "stage": "download", "reason": "trim failed"})
            if verbose:
                print(f"        [download fail] {vid}", flush=True)
            continue

        v_verdict = vision_gate.check(out_path, beat, vid, start_sec, cache=cache)
        gates.append({
            "video_id": vid, "stage": "vision",
            "passed": v_verdict["passed"], "reason": v_verdict["reason"],
            "cached": v_verdict["cached"],
        })
        if v_verdict["passed"]:
            # Harvest related videos
            kind = beat.get("visual", {}).get("kind", "unknown")
            rel = youtube.harvest_related(vid)
            if rel:
                youtube._HARVESTED_POOL.setdefault(kind, []).extend(rel)
            if not _claim_youtube_winner(
                vid, sourcing_context, out_path, gates, verbose, beat,
                start_sec, meta, dur, allow_video_reuse=allow_video_reuse,
            ):
                if gates and gates[-1].get("stage") == "visual-dedup":
                    deferred_visual_dups.append({
                        "video_id": vid, "start_sec": start_sec, "meta": meta,
                        "matched_beat": gates[-1].get("matched_beat"),
                    })
                continue
            return {"video_id": vid, "source": "youtube", "start_sec": start_sec}
        # vision rejected — drop the file and try next
        out_path.unlink(missing_ok=True)

    if deferred_claimed:
        if verbose:
            print(f"      [sourcing] trying claimed YouTube ids as last resort: {deferred_claimed}", flush=True)
        return _try_youtube_candidates(
            deferred_claimed, beat, topic_class, out_path, gates, verbose,
            sourcing_context=sourcing_context, allow_claimed_last_resort=False,
            allow_video_reuse=True,
        )

    if deferred_visual_dups:
        dup = deferred_visual_dups[0]
        vid = dup["video_id"]
        start_sec = float(dup["start_sec"])
        matched_beat = dup.get("matched_beat")
        if not _visual_dup_allowed(beat, matched_beat):
            gates.append({
                "video_id": vid,
                "stage": "visual-dedup",
                "reason": (
                    f"visual duplicate too close to beat {matched_beat}; "
                    f"min gap={VISUAL_DUP_MIN_BEAT_GAP}"
                ),
                "matched_beat": matched_beat,
            })
            if verbose:
                print(
                    f"      [visual-dedup] beat {beat.get('beat_id', '?')} "
                    f"blocked near duplicate {vid} from beat {matched_beat}",
                    flush=True,
                )
            return None
        if not _video_reuse_allowed(vid, beat, sourcing_context):
            if verbose:
                print(
                    f"      [visual-dedup] beat {beat.get('beat_id', '?')} "
                    f"blocked near video reuse {vid}",
                    flush=True,
                )
            return None
        if verbose:
            print(f"      [visual-dedup] beat {beat.get('beat_id', '?')} all remaining candidates collided; allowing dup {vid}", flush=True)
        if youtube.download_trim(vid, start_sec, dur, out_path):
            _claim_youtube_winner(
                vid, sourcing_context, out_path, gates, verbose, beat, start_sec,
                dup.get("meta") or {}, dur, allow_visual_dup=True,
                allow_video_reuse=True,
            )
            return {"video_id": vid, "source": "youtube", "start_sec": start_sec}

    return None


def _video_already_claimed(video_id: str, context: Optional[SourcingContext]) -> bool:
    if not context or not video_id:
        return False
    with context.used_lock:
        return video_id in context.used_video_ids


def _beat_number(beat: Optional[Dict[str, Any]]) -> Optional[int]:
    try:
        value = (beat or {}).get("beat_id")
        return int(value)
    except Exception:
        return None


def _min_gap_to_prior(priors: List[int], beat_id: Optional[int]) -> Optional[int]:
    if beat_id is None or not priors:
        return None
    return min(abs(int(beat_id) - int(prior)) for prior in priors)


def _video_reuse_gap(video_id: str, beat: Optional[Dict[str, Any]],
                     context: Optional[SourcingContext]) -> Optional[int]:
    if not context or not video_id:
        return None
    beat_id = _beat_number(beat)
    with context.used_lock:
        return _min_gap_to_prior(context.claimed_video_beats.get(video_id, []), beat_id)


def _video_reuse_allowed(video_id: str, beat: Optional[Dict[str, Any]],
                         context: Optional[SourcingContext]) -> bool:
    gap = _video_reuse_gap(video_id, beat, context)
    return gap is None or gap >= VIDEO_REUSE_MIN_BEAT_GAP


def _visual_dup_allowed(beat: Optional[Dict[str, Any]], matched_beat: Any) -> bool:
    beat_id = _beat_number(beat)
    try:
        prior = int(matched_beat)
    except Exception:
        return False
    if beat_id is None:
        return False
    return abs(beat_id - prior) >= VISUAL_DUP_MIN_BEAT_GAP


def _claim_youtube_winner(video_id: str, context: Optional[SourcingContext],
                          out_path: Path, gates: List[Dict[str, Any]],
                          verbose: bool, beat: Optional[Dict[str, Any]] = None,
                          start_sec: float = 0.0, meta: Optional[Dict[str, Any]] = None,
                          dur: float = 5.0, allow_visual_dup: bool = False,
                          allow_video_reuse: bool = False) -> bool:
    if not context or not video_id:
        return True
    beat_id = (beat or {}).get("beat_id", "?")
    phash: Optional[str] = None
    if VISUAL_DEDUP_PHASH and out_path.exists():
        phash = _segment_phash(video_id, start_sec, out_path, dur)
    title = str((meta or {}).get("title") or "")
    visual_terms = _extract_visual_terms(title, beat or {})
    with context.used_lock:
        beat_num = _beat_number(beat)
        prior_video_beats = context.claimed_video_beats.get(video_id, [])
        reuse_gap = _min_gap_to_prior(prior_video_beats, beat_num)
        video_reuse = video_id in context.used_video_ids
        if video_reuse and (
            not allow_video_reuse
            or reuse_gap is None
            or reuse_gap < VIDEO_REUSE_MIN_BEAT_GAP
        ):
            claimed = False
        else:
            if phash and not allow_visual_dup:
                for prior in context.claimed_phashes:
                    prior_hash = prior.get("phash")
                    if not prior_hash:
                        continue
                    dist = _hamming_distance(phash, prior_hash)
                    if dist <= VISUAL_DEDUP_THRESHOLD:
                        gates.append({
                            "video_id": video_id,
                            "stage": "visual-dedup",
                            "reason": f"pHash dist={dist} to beat {prior.get('beat_id')} winner",
                            "phash": phash,
                            "distance": dist,
                            "matched_beat": prior.get("beat_id"),
                            "matched_video_id": prior.get("video_id"),
                        })
                        try:
                            out_path.unlink(missing_ok=True)
                        except PermissionError:
                            pass
                        if verbose:
                            print(
                                f"      [visual-dedup] beat {beat_id} candidate {video_id} rejected, "
                                f"pHash dist={dist} to beat {prior.get('beat_id')} winner",
                                flush=True,
                            )
                            print(f"      [visual-dedup] beat {beat_id} fell through to next candidate", flush=True)
                        return False
            context.used_video_ids.add(video_id)
            if beat_num is not None:
                context.claimed_video_beats.setdefault(video_id, []).append(beat_num)
            if phash:
                context.claimed_phashes.append({
                    "beat_id": beat_id,
                    "video_id": video_id,
                    "start_sec": round(float(start_sec), 3),
                    "phash": phash,
                })
            for term in visual_terms:
                context.used_visual_terms.add(term)
            context.used_winner_titles.append({
                "beat_id": beat_id,
                "video_id": video_id,
                "title": title,
                "subject": str((beat or {}).get("visual", {}).get("subject") or ""),
            })
            claimed = True
    if claimed:
        if phash and verbose:
            mode = "allowed_dup" if allow_visual_dup else "claimed"
            print(f"      [visual-dedup] beat {beat_id} {mode} {video_id} phash={phash}", flush=True)
        if visual_terms and verbose:
            print(f"      [text-gate] beat {beat_id} negative terms += {sorted(visual_terms)}", flush=True)
        return True
    gates.append({
        "video_id": video_id,
        "stage": "dedup",
        "reason": "video_id claimed too close during finalization; trying next candidate",
    })
    try:
        out_path.unlink(missing_ok=True)
    except PermissionError:
        pass
    if verbose:
        print(f"      [sourcing] dup reject video_id={video_id} at finalize; trying next candidate", flush=True)
    return False


def _try_pexels_candidates(beat: Dict[str, Any], out_path: Path,
                           gates: List[Dict[str, Any]],
                           verbose: bool) -> Optional[Dict[str, Any]]:
    visual = beat.get("visual") or {}
    queries = list(visual.get("queries") or [])
    dur = float(beat.get("duration_sec") or 5.0)
    for q in queries[:2]:
        items = pexels.search(q, per_page=4)
        for item in items:
            if item.get("duration", 0) < dur:
                continue
            if pexels.fetch_and_trim(item, dur, out_path):
                gates.append({
                    "video_id": f"{item['source']}:{item['id']}",
                    "stage": "pexels", "reason": "downloaded",
                })
                return {
                    "video_id": item["id"],
                    "source": item["source"],
                    "credit": item.get("credit", ""),
                }
            gates.append({
                "video_id": f"{item['source']}:{item['id']}",
                "stage": "pexels", "reason": "trim failed",
            })
    return None


def _wrap(winner: Optional[Dict[str, Any]], clip_path: Optional[Path],
          gates: List[Dict[str, Any]], t0: float, fallback: bool) -> Dict[str, Any]:
    return {
        "clip_path": str(clip_path) if (winner and clip_path and clip_path.exists()) else None,
        "winner": winner,
        "gates": gates,
        "elapsed_sec": round(time.time() - t0, 2),
        "fallback_used": fallback,
    }


def _dedup_same_video(
    results_list: List[Dict[str, Any]],
    beats: List[Dict[str, Any]],
    verbose: bool,
) -> None:
    """Ensure no two beats use the same video_id at the same start_sec.

    When multiple beats land on the same video, keep the first user's clip as-is
    and re-trim subsequent users from evenly-spaced positions in the video.
    If a re-trim fails, that beat's clip is cleared so the cinematic fallback runs.
    """
    from . import youtube as _yt

    # Map video_id -> [(result_idx, orig_start_sec), ...]
    vid_users: Dict[str, List[tuple]] = {}
    for idx, res in enumerate(results_list):
        winner = (res or {}).get("winner") or {}
        if winner.get("source") == "youtube" and winner.get("video_id"):
            vid = winner["video_id"]
            start = float(winner.get("start_sec") or 0.0)
            vid_users.setdefault(vid, []).append((idx, start))

    for vid, users in vid_users.items():
        if len(users) <= 1:
            continue
        if verbose:
            print(f"  [dedup] video {vid} used by {len(users)} beats — retrimming duplicates", flush=True)
        meta = _yt.fetch_metadata(vid)
        vid_dur = float((meta or {}).get("duration") or 600.0)

        for rank, (idx, _orig_start) in enumerate(users):
            if rank == 0:
                continue  # first user keeps the original segment
            beat = beats[idx]
            beat_dur = float(beat.get("duration_sec") or 5.0)
            segment = max(beat_dur + 10.0, (vid_dur - beat_dur) / max(1, len(users)))
            new_start = max(0.0, min(vid_dur - beat_dur - 1.0, rank * segment))
            out_path = Path(results_list[idx]["clip_path"])
            if _yt.download_trim(vid, new_start, beat_dur, out_path):
                results_list[idx]["winner"]["start_sec"] = new_start
                if verbose:
                    beat_id = beat.get("beat_id", idx + 1)
                    print(f"    [dedup] beat {beat_id} retrimmed at {new_start:.1f}s", flush=True)
            else:
                # Can't get a different segment — clear the clip so cinematic fallback runs
                results_list[idx]["clip_path"] = None
                results_list[idx]["winner"] = None
                if verbose:
                    beat_id = beat.get("beat_id", idx + 1)
                    print(f"    [dedup] beat {beat_id} retrim failed, clearing (cinematic fallback)", flush=True)


def source_all_beats(script: Dict[str, Any], run_dir: Path,
                     verbose: bool = True) -> List[Dict[str, Any]]:
    """Source every beat in the script in beat order.

    Visual pHash claims and negative title terms are order-dependent, so beat
    finalization is deliberately sequential while per-candidate metadata fetch
    remains parallel inside each beat.
    """
    topic_class = script["topic_class"]
    beats = script["beats"]
    sourcing_context = SourcingContext(
        used_video_ids=set(),
        claimed_video_beats={},
        used_visual_terms=set(),
        claimed_phashes=[],
        used_winner_titles=[],
        used_lock=threading.Lock(),
    )

    results: List[Optional[Dict[str, Any]]] = []
    if verbose:
        print("  [sourcing] ordered beat sourcing enabled for visual dedup + negative terms", flush=True)
    for b in beats:
        try:
            results.append(source_beat(b, topic_class, run_dir, verbose, sourcing_context))
        except Exception as e:
            results.append({"clip_path": None, "winner": None,
                            "gates": [], "elapsed_sec": 0.0,
                            "fallback_used": True, "error": str(e)})

    results_list = [r or {"clip_path": None, "winner": None} for r in results]

    # Dedup: prevent multiple beats from using the same video segment
    _dedup_same_video(results_list, beats, verbose)

    # Cinematic variety pass: avoid consecutive generated-kind fallbacks of the same type
    for i in range(1, len(results_list)):
        curr_res = results_list[i]
        prev_res = results_list[i - 1]
        if not curr_res.get("winner") and not prev_res.get("winner"):
            curr_beat = beats[i]
            prev_beat = beats[i - 1]
            curr_kind = curr_beat.get("visual", {}).get("kind")
            prev_kind = prev_beat.get("visual", {}).get("kind")
            if curr_kind == prev_kind and curr_kind not in GENERATED_KINDS:
                new_kind = "wikipedia_photo" if curr_kind != "wikipedia_photo" else "map_zoom"
                if "overlay" in curr_beat and curr_beat["overlay"] and curr_beat["overlay"].get("kind") == "stat":
                    new_kind = "density_infographic"
                elif curr_kind == "map_zoom":
                    new_kind = "map_animation"
                curr_beat["visual"]["kind"] = new_kind
                if verbose:
                    print(f"      [variety pass] beat {i+1} changed {curr_kind} -> {new_kind}", flush=True)

    return results_list
