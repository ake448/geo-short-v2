"""
dlg.py — Digital Library of Georgia adapter (conditional).

Docs:
  https://dlg.usg.edu/about/api
  Per ARCHIVAL_IMAGE_API_FINDINGS.md (2026-04-18 research).

Only runs when the harvester decides the anchor's geo is in Georgia —
DLG is the single best Georgia-specific source, but irrelevant for
other regions. The harvester's router calls `applies_to(anchor)`
before `search(...)` so off-target anchors don't pay the round-trip.

Known gotchas from the research:
  - DLG is a discovery/aggregation layer. Many records only expose
    landing-page URLs; the master image often lives on a partner
    CONTENTdm server. We accept landing-only hits (record_url populated,
    image_url blank) so the harvester can at least surface them — the
    script writer treats image_url="" as a degraded candidate.
  - We explicitly skip partner gsu_ajc records — AJC copyright is
    restrictive per the research, not safe for automated publication.
"""
from __future__ import annotations

from typing import List, Optional

from ._http import get_json, parse_year
from .models import AnchorSpec, ArchivalAsset

SEARCH_URL = "https://dlg.usg.edu/records.json"
RESULTS_PER_QUERY = 10

# Partner collection IDs we refuse to surface (rights issues).
BLOCKED_COLLECTIONS = {"gsu_ajc"}


def applies_to(anchor: AnchorSpec) -> bool:
    """DLG is Georgia-only. Check anchor.geo for a Georgia signal."""
    g = (anchor.geo or "").lower()
    if not g:
        return False
    if "georgia" in g and "georgia, europe" not in g:
        return True
    # Accept common GA city/county names — but only when paired with a
    # state-ish token OR a ", ga" suffix, to avoid false positives on
    # e.g. "Atlanta, Illinois" or "Savannah, TN".
    if ", ga" in g or ", ga." in g:
        return True
    return False


def _build_params(anchor: AnchorSpec, query: str) -> dict:
    params = {
        "q": query,
        "per_page": str(RESULTS_PER_QUERY),
        # Restrict to still images + photographs where possible.
        "f[type_facet][]": "Still Image",
    }
    return params


def _row_to_asset(doc: dict) -> Optional[ArchivalAsset]:
    identifier = str(doc.get("id") or "").strip()
    if not identifier:
        return None

    # Filter blocked partner collections before building the record.
    for blocked in BLOCKED_COLLECTIONS:
        if identifier.startswith(blocked):
            return None
    coll = doc.get("collection_name") or doc.get("collection_titles") or ""
    if isinstance(coll, list):
        coll_blob = " ".join(str(c).lower() for c in coll)
    else:
        coll_blob = str(coll).lower()
    for blocked in BLOCKED_COLLECTIONS:
        if blocked in coll_blob:
            return None

    record_url = str(doc.get("record_url") or f"https://dlg.usg.edu/record/{identifier}")

    # DLG exposes a thumbnail URL on many records but not a reliable
    # master image URL (partner CONTENTdm handles that). We surface the
    # thumb when present; image_url stays "" if no direct file.
    thumb = ""
    for k in ("thumbnail_url", "thumbnail", "object_url"):
        v = doc.get(k)
        if v and isinstance(v, str) and v.startswith("http"):
            thumb = v
            break

    # Location: DLG uses a "location" array like
    # "United States, Georgia, Fulton County, Atlanta".
    locs = doc.get("location") or []
    if isinstance(locs, list):
        place = " / ".join(str(x) for x in locs if isinstance(x, str))
    else:
        place = str(locs or "")

    date_str = str(doc.get("date") or (doc.get("date_range") or [""])[0] if isinstance(doc.get("date_range"), list) else doc.get("date") or "")

    medium_field = doc.get("medium") or []
    if isinstance(medium_field, list):
        medium_blob = " ".join(str(m).lower() for m in medium_field)
    else:
        medium_blob = str(medium_field).lower()

    media_hint = "photo"
    if "map" in medium_blob:
        media_hint = "map"
    elif "postcard" in medium_blob:
        media_hint = "postcard"
    elif "newspaper" in medium_blob or "directory" in medium_blob:
        media_hint = "document"
    elif "aerial" in medium_blob:
        media_hint = "aerial"

    return ArchivalAsset(
        source="dlg",
        source_record_id=identifier,
        record_url=record_url,
        image_url="",  # DLG rarely gives a direct master — landing page only
        thumbnail_url=thumb,
        title=str(doc.get("title") or ""),
        description=str(doc.get("description") or ""),
        creator=str(doc.get("creator") or ""),
        date=date_str,
        year=parse_year(date_str),
        place=place,
        rights=str(doc.get("rights") or ""),
        media_hint=media_hint,
        raw={"collection": coll_blob[:200]},
    )


def search(anchor: AnchorSpec, query: str) -> List[ArchivalAsset]:
    """Run one query against DLG. Empty list on failure or off-geo anchor."""
    if not applies_to(anchor):
        return []
    if not query or not query.strip():
        return []

    try:
        payload = get_json(SEARCH_URL, params=_build_params(anchor, query))
    except Exception:
        return []

    docs = ((payload.get("response") or {}).get("docs")) or []
    # Older shape puts docs at top level.
    if not docs:
        docs = payload.get("docs") or []

    out: List[ArchivalAsset] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        asset = _row_to_asset(doc)
        if asset:
            out.append(asset)
    return out
