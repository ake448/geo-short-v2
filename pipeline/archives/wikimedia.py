"""
wikimedia.py — Wikimedia Commons adapter.

Docs:
  https://commons.wikimedia.org/wiki/Commons:API/MediaWiki
  Per ARCHIVAL_IMAGE_API_FINDINGS.md (2026-04-18 research).

No API key for public reads. Uses generator=search over namespace 6
(files) + prop=imageinfo to get the direct upload.wikimedia.org URL
and license metadata in one request.

Known gotchas from the research:
  - Commons holds SVGs, logos, diagrams, and maps alongside archival
    photos. We filter by mime prefix and media_hint heuristics.
  - extmetadata is expensive; we request it only once per search call,
    not per-item.
  - License comes back with inline HTML — strip before storing.
"""
from __future__ import annotations

from typing import List, Optional

from ._http import get_json, parse_year, strip_html
from .models import AnchorSpec, ArchivalAsset

API_URL = "https://commons.wikimedia.org/w/api.php"
RESULTS_PER_QUERY = 10


def _build_params(query: str) -> dict:
    # gsrsearch includes filetype:bitmap to cut SVG/PDF noise up front.
    # We still see maps and scans; those get classified in _row_to_asset.
    return {
        "action": "query",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrsearch": f"{query} filetype:bitmap",
        "gsrlimit": str(RESULTS_PER_QUERY),
        "prop": "imageinfo",
        "iiprop": "url|mime|mediatype|size|extmetadata",
        "format": "json",
        "formatversion": "2",
        "origin": "*",
    }


def _classify_media(title: str, mime: str, categories_hint: str = "") -> str:
    title_l = (title or "").lower()
    cat_l = (categories_hint or "").lower()
    blob = title_l + " " + cat_l

    if "map" in blob or "sanborn" in blob or "topograph" in blob:
        return "map"
    if "aerial" in blob or "bird's eye" in blob:
        return "aerial"
    if "postcard" in blob:
        return "postcard"
    if "newspaper" in blob or "directory" in blob or "book" in blob:
        return "document"
    if "diagram" in blob or "logo" in blob or "coat of arms" in blob or "flag of" in blob:
        return "document"
    if mime and not mime.startswith("image/"):
        return "document"
    return "photo"


def _row_to_asset(page: dict) -> Optional[ArchivalAsset]:
    title = str(page.get("title") or "")
    iis = page.get("imageinfo") or []
    if not iis:
        return None
    info = iis[0]
    image_url = str(info.get("url") or "")
    if not image_url:
        return None

    mime = str(info.get("mime") or "")
    # Hard filter: skip vector graphics, PDFs, and other non-photo assets
    # that slip past gsrsearch. filetype:bitmap covers most of this but
    # not all (some PNG diagrams survive).
    if mime in {"image/svg+xml", "application/pdf"}:
        return None

    description_url = str(info.get("descriptionurl") or "")
    source_record_id = title.replace(" ", "_")

    # extmetadata is a dict of {field: {value, source, ...}}.
    ext = info.get("extmetadata") or {}

    def _ext(field: str) -> str:
        v = ext.get(field)
        if isinstance(v, dict):
            return strip_html(str(v.get("value") or ""))
        return ""

    license_short = _ext("LicenseShortName")
    usage_terms = _ext("UsageTerms")
    rights = license_short or usage_terms or _ext("Copyrighted") or ""

    date_str = _ext("DateTimeOriginal") or _ext("DateTime") or ""
    credit = _ext("Credit")
    artist = _ext("Artist")
    categories_blob = _ext("Categories")
    description = _ext("ImageDescription")
    object_name = _ext("ObjectName")

    return ArchivalAsset(
        source="wikimedia",
        source_record_id=source_record_id,
        record_url=description_url or f"https://commons.wikimedia.org/wiki/{source_record_id}",
        image_url=image_url,
        thumbnail_url="",  # API can return thumbs with iiurlwidth, but not needed for pool selection
        title=object_name or title.replace("File:", ""),
        description=description,
        creator=artist or credit,
        date=date_str,
        year=parse_year(date_str),
        place="",  # Commons rarely has a clean place field; the harvester ranks via title+description anyway
        rights=rights,
        media_hint=_classify_media(title, mime, categories_blob),
        width=info.get("width") if isinstance(info.get("width"), int) else None,
        height=info.get("height") if isinstance(info.get("height"), int) else None,
        raw={"mime": mime},
    )


def search(anchor: AnchorSpec, query: str) -> List[ArchivalAsset]:
    """Run one query against Commons. Empty list on failure."""
    if not query or not query.strip():
        return []

    try:
        payload = get_json(API_URL, params=_build_params(query))
    except Exception:
        return []

    pages = ((payload.get("query") or {}).get("pages")) or []
    # formatversion=2 returns pages as a list; guard against older shape.
    if isinstance(pages, dict):
        pages = list(pages.values())

    out: List[ArchivalAsset] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        asset = _row_to_asset(page)
        if asset:
            out.append(asset)
    return out
