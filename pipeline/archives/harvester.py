"""
harvester.py — fan anchor specs out to archive adapters, score, dedupe,
and emit a PoolManifest the script layer can bind overlays to.

Flow per AnchorSpec:
  1. Routing picks which adapters to call (DLG gates on Georgia).
  2. For each (adapter, query) pair, run in parallel.
  3. Score every returned ArchivalAsset against the anchor.
  4. Assign evidence_status (direct / contextual / substitute).
  5. Dedupe across adapters by image_url / title.
  6. Keep top-K per beat.
  7. Anchors with nothing above threshold become NegativeFinding entries.

The harvester is deliberately stateless — the caller drives a batch of
anchors in, gets a manifest out. Cache / persistence lives one layer up.
"""
from __future__ import annotations

import concurrent.futures as cf
import re
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from . import dlg, internet_archive, loc, wikimedia, wikipedia
from .models import (
    AnchorSpec,
    ArchivalAsset,
    EvidenceStatus,
    NegativeFinding,
    PoolEntry,
    PoolManifest,
)

# ── Tunables ─────────────────────────────────────────────────────────────────
MAX_ADAPTER_WORKERS = 6        # parallel (adapter, query) calls
MAX_QUERIES_PER_ANCHOR = 6     # context-card prompts generate 4-6 targeted catalog queries
TOP_K_PER_BEAT = 6             # pool entries kept per beat
DIRECT_THRESHOLD = 70          # score >= this = DIRECT
CONTEXTUAL_THRESHOLD = 45      # CONTEXTUAL <= score < DIRECT
MIN_KEEP_THRESHOLD = 25        # below this = discard (SUBSTITUTE only above)

# (adapter_name, search_callable)
_ADAPTERS: List[Tuple[str, Callable[[AnchorSpec, str], List[ArchivalAsset]]]] = [
    ("wikipedia", wikipedia.search),
    ("loc", loc.search),
    ("wikimedia", wikimedia.search),
    ("internet_archive", internet_archive.search),
    ("dlg", dlg.search),          # no-op for non-Georgia via dlg.applies_to
]


# ── Scoring ──────────────────────────────────────────────────────────────────
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> List[str]:
    return _WORD_RE.findall((s or "").lower())


def _content_tokens(s: str) -> set:
    """Tokens minus stopwords too short to be discriminating."""
    stop = {"the", "a", "an", "of", "in", "on", "at", "for", "to", "and",
            "or", "by", "with", "from", "near"}
    return {t for t in _tokens(s) if len(t) > 2 and t not in stop}


def _score(anchor: AnchorSpec, asset: ArchivalAsset) -> Tuple[float, List[str]]:
    """Return (score 0-100, list of signals matched)."""
    score = 0.0
    matched: List[str] = []

    subject_tokens = _content_tokens(anchor.subject)
    geo_tokens = _content_tokens(anchor.geo)
    query_token_sets = [_content_tokens(q) for q in anchor.queries if q]

    title_blob = (asset.title or "").lower()
    desc_blob = (asset.description or "").lower()
    place_blob = (asset.place or "").lower()
    all_blob = " ".join((title_blob, desc_blob, place_blob))

    # Exact subject phrase in title.
    if anchor.subject and anchor.subject.lower() in title_blob:
        score += 35
        matched.append("title_exact_phrase")

    # Subject tokens in title.
    title_tokens = set(_tokens(title_blob))
    subj_overlap = subject_tokens & title_tokens
    if subj_overlap:
        # Proportional credit — scales from 0 to 20.
        frac = len(subj_overlap) / max(1, len(subject_tokens))
        bonus = round(20 * frac, 2)
        score += bonus
        if bonus >= 10:
            matched.append("title_subject_tokens")

    # Query tokens in title. Context-card searches often use concise catalog
    # phrases ("Centralia Pennsylvania mine fire") while anchor.subject can be
    # a verbose visual description. Credit the query that actually surfaced
    # the asset, or obvious Commons hits like "Hartwell Dam" score too low.
    best_query_bonus = 0.0
    for query_tokens in query_token_sets:
        if not query_tokens:
            continue
        if len(query_tokens & subject_tokens) < 2 and len(query_tokens & geo_tokens) < 2:
            continue
        query_overlap = query_tokens & title_tokens
        if not query_overlap:
            continue
        frac = len(query_overlap) / max(1, len(query_tokens))
        best_query_bonus = max(best_query_bonus, round(20 * frac, 2))
    if best_query_bonus:
        score += best_query_bonus
        if best_query_bonus >= 10:
            matched.append("title_query_tokens")

    # Subject tokens anywhere.
    if subject_tokens and subject_tokens.issubset(set(_tokens(all_blob))):
        score += 10
        matched.append("desc_subject_tokens")

    # Geo check — place or title mentions geo.
    geo_hit = False
    if geo_tokens:
        if geo_tokens & set(_tokens(place_blob)):
            score += 15
            matched.append("place_match")
            geo_hit = True
        elif geo_tokens & set(_tokens(title_blob)):
            score += 10
            matched.append("title_geo_match")
            geo_hit = True

    # Wrong-geo penalty — if the anchor specifies geo but the asset has a
    # place field that contradicts it, cut hard. Research warned
    # Marietta Ohio polluting Marietta Georgia queries.
    if geo_tokens and place_blob and not geo_hit:
        # Asset has a place string but none of our geo tokens matched.
        # Only penalize if the asset's place looks state-like (comma sep).
        if "," in place_blob:
            score -= 30
            matched.append("wrong_geo_penalty")

    # Era overlap. When the anchor allows substitutes, don't penalize era
    # misses — a modern reference shot of a landmark is a legitimate
    # SUBSTITUTE-tier overlay. Strict anchors (allow_substitute=False)
    # still get the miss penalty.
    if anchor.era and len(anchor.era) == 2 and asset.year:
        lo, hi = anchor.era
        if lo - 5 <= asset.year <= hi + 5:
            score += 12
            matched.append("era_match")
        elif not anchor.allow_substitute:
            score -= 15
            matched.append("era_miss_penalty")

    # Media type fit. For landmark/event/person anchors we prefer photos.
    if asset.media_hint == "photo":
        score += 8
    elif asset.media_hint in ("postcard", "aerial"):
        score += 5
    elif asset.media_hint == "map":
        # Maps are great for geography anchors, mediocre for events/people.
        if anchor.anchor_type in ("map", "vanished_place", "stat"):
            score += 8
        else:
            score += 2
    elif asset.media_hint == "document":
        score -= 8   # newspaper/book/directory — not usable as overlay imagery

    # No direct image URL = degraded candidate (DLG landing-only case).
    if not asset.image_url:
        score -= 12
        matched.append("landing_only")

    # Clamp.
    score = max(0.0, min(100.0, score))
    return score, matched


def _evidence_status(score: float, asset: ArchivalAsset) -> str:
    if score >= DIRECT_THRESHOLD:
        return EvidenceStatus.DIRECT
    if asset.media_hint in ("map", "document") and score >= CONTEXTUAL_THRESHOLD:
        return EvidenceStatus.SUBSTITUTE
    if score >= CONTEXTUAL_THRESHOLD:
        return EvidenceStatus.CONTEXTUAL
    return EvidenceStatus.SUBSTITUTE


# ── Dedupe ───────────────────────────────────────────────────────────────────
def _dedupe_key(asset: ArchivalAsset) -> str:
    if asset.image_url:
        return asset.image_url
    return f"{asset.source}:{asset.source_record_id}"


# ── One-anchor harvest ───────────────────────────────────────────────────────
def _queries_for(anchor: AnchorSpec) -> List[str]:
    """Prefer LLM-supplied queries; fall back to synthesizing from
    subject + geo so an under-briefed anchor still gets tried."""
    qs = list(anchor.queries)[:MAX_QUERIES_PER_ANCHOR]
    if qs:
        return qs
    # Fallback: subject alone, subject+geo, subject+era.
    fallback = []
    if anchor.subject:
        fallback.append(anchor.subject)
        if anchor.geo:
            fallback.append(f"{anchor.subject} {anchor.geo}")
        if anchor.era:
            fallback.append(f"{anchor.subject} {anchor.era[0]}")
    return fallback[:MAX_QUERIES_PER_ANCHOR]


def _harvest_one(
    anchor: AnchorSpec, executor: cf.ThreadPoolExecutor
) -> Tuple[List[PoolEntry], Optional[NegativeFinding]]:
    """Run all applicable adapters × queries for one anchor, return
    ranked PoolEntries (capped at TOP_K_PER_BEAT) plus a NegativeFinding
    if nothing survived."""
    queries = _queries_for(anchor)
    if not queries:
        return [], NegativeFinding(
            beat_id=anchor.beat_id,
            anchor_type=anchor.anchor_type,
            subject=anchor.subject,
            queried_sources=[],
            reason="no_queries",
        )

    # Build (name, func, query) tasks, filtering DLG on geo.
    tasks: List[Tuple[str, Callable, str]] = []
    for name, fn in _ADAPTERS:
        if name == "dlg" and not dlg.applies_to(anchor):
            continue
        for q in queries:
            tasks.append((name, fn, q))

    queried_sources: List[str] = sorted({t[0] for t in tasks})

    futures = {
        executor.submit(fn, anchor, q): (name, q)
        for (name, fn, q) in tasks
    }

    hits: List[ArchivalAsset] = []
    for fut in cf.as_completed(futures):
        try:
            result = fut.result(timeout=30)
        except Exception:
            result = []
        if result:
            hits.extend(result)

    if not hits:
        return [], NegativeFinding(
            beat_id=anchor.beat_id,
            anchor_type=anchor.anchor_type,
            subject=anchor.subject,
            queried_sources=queried_sources,
            reason="no_hits",
        )

    # Score + dedupe.
    scored: Dict[str, Tuple[float, List[str], ArchivalAsset]] = {}
    for asset in hits:
        key = _dedupe_key(asset)
        s, matched = _score(anchor, asset)
        if s < MIN_KEEP_THRESHOLD:
            continue
        prev = scored.get(key)
        if prev is None or s > prev[0]:
            scored[key] = (s, matched, asset)

    if not scored:
        return [], NegativeFinding(
            beat_id=anchor.beat_id,
            anchor_type=anchor.anchor_type,
            subject=anchor.subject,
            queried_sources=queried_sources,
            reason="all_below_threshold",
        )

    ranked = sorted(scored.values(), key=lambda t: t[0], reverse=True)[:TOP_K_PER_BEAT]

    entries = [
        PoolEntry(
            beat_id=anchor.beat_id,
            anchor_type=anchor.anchor_type,
            asset=asset,
            relevance_score=round(score, 2),
            evidence_status=_evidence_status(score, asset),
            matched_on=matched,
        )
        for (score, matched, asset) in ranked
    ]
    return entries, None


# ── Public API ───────────────────────────────────────────────────────────────
def harvest(anchors: Iterable[AnchorSpec]) -> PoolManifest:
    """Harvest a batch of anchors. One ThreadPoolExecutor is shared across
    all anchors — stays polite to the various archive rate limits while
    parallelizing the network-bound work.
    """
    anchors = list(anchors)
    manifest = PoolManifest()

    if not anchors:
        manifest.stats = {"anchors": 0, "entries": 0, "negative": 0}
        return manifest

    with cf.ThreadPoolExecutor(max_workers=MAX_ADAPTER_WORKERS) as ex:
        for anchor in anchors:
            entries, negative = _harvest_one(anchor, ex)
            manifest.entries.extend(entries)
            if negative:
                manifest.negative.append(negative)

    # Cheap per-source tally for observability.
    by_source: Dict[str, int] = {}
    for e in manifest.entries:
        by_source[e.asset.source] = by_source.get(e.asset.source, 0) + 1

    manifest.stats = {
        "anchors": len(anchors),
        "entries": len(manifest.entries),
        "negative": len(manifest.negative),
        "by_source": by_source,
    }
    return manifest
