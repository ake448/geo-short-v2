"""
internet_archive.py — archive.org advancedsearch + metadata adapter.

Docs referenced:
  https://archive.org/advancedsearch.php
  Per ARCHIVAL_IMAGE_API_FINDINGS.md (2026-04-18 research).

No API key. Two-step flow per the research:
  1. advancedsearch returns identifiers matching mediatype:image
  2. /metadata/{id} returns files[] — we pick the best non-derivative image

We skip step 2 for identifiers we'd reject anyway (wrong mediatype,
empty title). Step 2 adds one request per hit, so RESULTS_PER_QUERY
stays small.

Known gotchas from the research:
  - mediatype:image includes album covers, scans, maps, and misc uploads.
  - Items hold many derivative files (_thumb, _files.xml, OCR) — filter.
  - "Georgia" matches country + state + author surnames; we rely on the
    harvester's relevance scorer to reject off-topic hits.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

from ._http import get_json, parse_year
from .models import AnchorSpec, ArchivalAsset

SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/"

RESULTS_PER_QUERY = 6   # keep low — we do a metadata fetch per hit

# Extensions we'd accept as a "real" image. Ordered by preference.
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2")

# Filename suffixes that indicate a derivative we should skip.
_DERIVATIVE_SUFFIXES = ("_thumb", "_small", "_medium", "_itemimage", ".thumbs")


def _build_search_params(anchor: AnchorSpec, query: str) -> dict:
    # Lucene-style query. The title+subject OR lets us hit both how items
    # are catalogued (title-forward historic uploads, subject-forward
    # institutional uploads).
    q_parts = [
        f'(title:"{query}" OR subject:"{query}" OR description:"{query}")',
        "mediatype:image",
    ]
    if anchor.era and len(anchor.era) == 2:
        q_parts.append(f"date:[{anchor.era[0]}-01-01 TO {anchor.era[1]}-12-31]")

    return {
        "q": " AND ".join(q_parts),
        "fl[]": ["identifier", "title", "date", "mediatype", "creator",
                 "description", "subject", "licenseurl", "rights"],
        "rows": str(RESULTS_PER_QUERY),
        "page": "1",
        "output": "json",
    }


def _pick_best_file(files: Iterable[dict]) -> Optional[dict]:
    """From a metadata files[] list, pick the best full-size image."""
    best = None
    best_rank = -1
    for f in files:
        if not isinstance(f, dict):
            continue
        name = str(f.get("name") or "").lower()
        if not name:
            continue
        if any(suf in name for suf in _DERIVATIVE_SUFFIXES):
            continue
        # Rank by extension preference.
        rank = -1
        for i, ext in enumerate(_IMG_EXTS):
            if name.endswith(ext):
                rank = len(_IMG_EXTS) - i
                break
        if rank < 0:
            continue
        if rank > best_rank:
            best = f
            best_rank = rank
    return best


def _fetch_asset(doc: dict) -> Optional[ArchivalAsset]:
    identifier = str(doc.get("identifier") or "").strip()
    if not identifier:
        return None

    try:
        meta = get_json(METADATA_URL + identifier)
    except Exception:
        return None

    files = meta.get("files") or []
    if not isinstance(files, list):
        return None

    picked = _pick_best_file(files)
    if not picked:
        return None

    filename = str(picked.get("name") or "")
    image_url = f"https://archive.org/download/{identifier}/{filename}"
    record_url = f"https://archive.org/details/{identifier}"

    metadata = meta.get("metadata") or {}

    # IA returns title as string OR list; normalize.
    def _first(v):
        if isinstance(v, list):
            return str(v[0]) if v else ""
        return str(v or "")

    title = _first(metadata.get("title") or doc.get("title"))
    description = _first(metadata.get("description") or doc.get("description"))
    creator = _first(metadata.get("creator") or doc.get("creator"))
    date_str = _first(metadata.get("date") or doc.get("date"))
    subject_field = metadata.get("subject") or doc.get("subject") or ""
    if isinstance(subject_field, list):
        place = ", ".join(str(s) for s in subject_field if s)[:200]
    else:
        place = str(subject_field)[:200]

    rights = _first(metadata.get("rights") or doc.get("rights"))
    license_url = _first(metadata.get("licenseurl") or doc.get("licenseurl"))
    rights_combined = rights or license_url

    # Media hint from IA's own collection/subject hints.
    blob = (title + " " + place + " " + description).lower()
    media_hint = "photo"
    if "map" in blob or "sanborn" in blob:
        media_hint = "map"
    elif "aerial" in blob:
        media_hint = "aerial"
    elif "postcard" in blob:
        media_hint = "postcard"
    elif "newspaper" in blob or "directory" in blob or "book" in blob:
        media_hint = "document"

    width_raw = picked.get("width")
    height_raw = picked.get("height")

    def _to_int(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return None

    return ArchivalAsset(
        source="internet_archive",
        source_record_id=identifier,
        record_url=record_url,
        image_url=image_url,
        thumbnail_url=f"https://archive.org/services/img/{identifier}",
        title=title,
        description=description[:500],
        creator=creator,
        date=date_str,
        year=parse_year(date_str),
        place=place,
        rights=rights_combined,
        media_hint=media_hint,
        width=_to_int(width_raw),
        height=_to_int(height_raw),
        raw={"file": filename, "format": picked.get("format")},
    )


def search(anchor: AnchorSpec, query: str) -> List[ArchivalAsset]:
    """Run one query against archive.org. Empty list on failure."""
    if not query or not query.strip():
        return []

    try:
        payload = get_json(SEARCH_URL, params=_build_search_params(anchor, query))
    except Exception:
        return []

    docs = ((payload.get("response") or {}).get("docs")) or []
    out: List[ArchivalAsset] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if doc.get("mediatype") != "image":
            continue
        asset = _fetch_asset(doc)
        if asset:
            out.append(asset)
    return out
