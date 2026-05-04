"""
loc.py — Library of Congress /photos/ adapter.

Docs referenced:
  https://www.loc.gov/apis/json-and-yaml/
  Per ARCHIVAL_IMAGE_API_FINDINGS.md (2026-04-18 research).

No API key. 20 req/min on JSON endpoint; 1-hour block on overrun. We
stay well under by limiting result counts and relying on the harvester's
per-adapter concurrency cap.

Known gotchas from the research:
  - /search/ returns books, newspapers, events — we use /photos/ only.
  - "Georgia" matches both the US state and the country; filter via the
    location facet when we can resolve it.
  - result-row metadata is often incomplete; we do NOT follow each item
    to its detail endpoint here — that's 1 extra call per hit, not worth
    it for an overlay-selection pool. If the row lacks image_url we skip.
"""
from __future__ import annotations

from typing import List, Optional

from ._http import get_json, parse_year, strip_html
from .models import AnchorSpec, ArchivalAsset

SEARCH_URL = "https://www.loc.gov/photos/"

# Rows we request per query. Kept small on purpose — harvester issues
# multiple queries per anchor, and we don't want to blow rate limits.
RESULTS_PER_QUERY = 10


def _build_params(anchor: AnchorSpec, query: str) -> dict:
    params = {
        "q": query,
        "fo": "json",
        "c": str(RESULTS_PER_QUERY),
        "at": "results",
    }

    # Era filter — LOC accepts YYYY/YYYY.
    if anchor.era and len(anchor.era) == 2:
        params["dates"] = f"{anchor.era[0]}/{anchor.era[1]}"

    # Geography filter. LOC's `fa=location:<token>` disambiguates state
    # vs. country for "georgia" and similar collisions. We add the last
    # comma-segment of `geo` as a location facet when it looks like a
    # state/country word. If the user gave just a city, we skip — the
    # text query carries the signal.
    geo_tokens = [t.strip() for t in (anchor.geo or "").split(",") if t.strip()]
    if geo_tokens:
        tail = geo_tokens[-1].lower()
        # Rough heuristic: facets work best on state/country-ish tokens.
        # LOC accepts e.g. location:georgia, location:new york, location:japan.
        if 2 <= len(tail) <= 30 and " " not in tail or tail in {
            "new york", "new jersey", "new mexico", "north carolina",
            "south carolina", "north dakota", "south dakota",
            "west virginia", "rhode island",
        }:
            params["fa"] = f"location:{tail}"

    return params


def _extract_image_url(row: dict) -> Optional[str]:
    """Prefer the highest-resolution service URL we can find."""
    imgs = row.get("image_url") or []
    if not isinstance(imgs, list) or not imgs:
        return None
    # LOC image_url arrays tend to be ordered small -> large thumbnails.
    # Pick the last entry that looks like a real file.
    for candidate in reversed(imgs):
        if isinstance(candidate, str) and candidate.startswith("http"):
            return candidate
    return None


def _extract_thumbnail(row: dict) -> str:
    imgs = row.get("image_url") or []
    if isinstance(imgs, list) and imgs:
        first = imgs[0]
        if isinstance(first, str):
            return first
    return ""


def _row_to_asset(row: dict) -> Optional[ArchivalAsset]:
    image_url = _extract_image_url(row)
    if not image_url:
        # Per the plan: rows without a reachable image are evidence-only
        # and shouldn't enter an overlay pool. Drop.
        return None

    # Record URL — the human landing page.
    record_url = str(row.get("url") or "").strip()
    if not record_url:
        return None

    # Prefer id from URL; LOC results don't always expose a clean id field.
    source_record_id = record_url.rstrip("/").rsplit("/", 1)[-1]

    date_str = ""
    # LOC returns "date" as a string like "1936" or "1936-05-14".
    if isinstance(row.get("date"), str):
        date_str = row["date"]
    elif isinstance(row.get("dates"), list) and row["dates"]:
        date_str = str(row["dates"][0])

    place = ""
    locs = row.get("location") or []
    if isinstance(locs, list) and locs:
        place = ", ".join(str(x) for x in locs if isinstance(x, str))

    original_format = row.get("original_format") or []
    media_hint = "photo"
    if isinstance(original_format, list):
        fmt_str = " ".join(str(f).lower() for f in original_format)
        if "map" in fmt_str:
            media_hint = "map"
        elif "newspaper" in fmt_str or "book" in fmt_str:
            media_hint = "document"
        elif "aerial" in fmt_str:
            media_hint = "aerial"

    rights = ""
    # LOC puts rights in different fields per division; grab whatever's present.
    for k in ("rights", "rights_advisory", "access_advisory"):
        v = row.get(k)
        if v:
            rights = strip_html(v if isinstance(v, str) else " ".join(str(x) for x in v if x))
            break

    return ArchivalAsset(
        source="loc",
        source_record_id=source_record_id,
        record_url=record_url,
        image_url=image_url,
        thumbnail_url=_extract_thumbnail(row),
        title=strip_html(str(row.get("title") or "")),
        description=strip_html(str(row.get("description") or "")),
        creator=", ".join(str(c) for c in (row.get("creator") or []) if c)
                if isinstance(row.get("creator"), list)
                else str(row.get("creator") or ""),
        date=date_str,
        year=parse_year(date_str),
        place=place,
        rights=rights,
        media_hint=media_hint,
        raw={"original_format": original_format},
    )


def search(anchor: AnchorSpec, query: str) -> List[ArchivalAsset]:
    """Run one query against LOC /photos/. Empty list on any failure —
    the harvester records that as a negative finding."""
    if not query or not query.strip():
        return []

    params = _build_params(anchor, query)
    try:
        payload = get_json(SEARCH_URL, params=params)
    except Exception:
        return []

    rows = payload.get("results") or []
    out: List[ArchivalAsset] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        asset = _row_to_asset(row)
        if asset:
            out.append(asset)
    return out
