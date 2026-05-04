"""
wikipedia.py - Wikipedia article lead-image adapter.

This is deliberately separate from the Commons file-search adapter. Commons
search is broad and often returns logos, maps, diagrams, or weak keyword hits.
For place/event beats, a resolved Wikipedia article lead image is usually a
cleaner visual: it has article context, a stable page URL, and a Commons file
record we can credit via imageinfo/extmetadata when available.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from ._http import get_json, parse_year, strip_html
from .models import AnchorSpec, ArchivalAsset

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
RESULTS_PER_QUERY = 5

_BAD_TITLE_RE = re.compile(
    r"\b("
    r"logo|seal|flag|coat of arms|locator map|location map|blank map|"
    r"route map|diagram|chart|icon|symbol|svg|pdf"
    r")\b",
    re.IGNORECASE,
)


def _page_query_params(query: str) -> Dict[str, str]:
    return {
        "action": "query",
        "generator": "search",
        "gsrnamespace": "0",
        "gsrsearch": query,
        "gsrlimit": str(RESULTS_PER_QUERY),
        "prop": "pageimages|info|extracts",
        "piprop": "name|original|thumbnail",
        "pithumbsize": "1400",
        "inprop": "url",
        "exintro": "1",
        "explaintext": "1",
        "exsentences": "2",
        "format": "json",
        "formatversion": "2",
        "origin": "*",
    }


def _commons_metadata(file_title: str) -> Dict[str, Any]:
    if not file_title:
        return {}
    title = file_title if file_title.lower().startswith("file:") else f"File:{file_title}"
    params = {
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|mime|mediatype|size|extmetadata",
        "iiurlwidth": "1400",
        "format": "json",
        "formatversion": "2",
        "origin": "*",
    }
    try:
        payload = get_json(COMMONS_API_URL, params=params)
    except Exception:
        return {}
    pages = ((payload.get("query") or {}).get("pages")) or []
    if isinstance(pages, dict):
        pages = list(pages.values())
    if not pages:
        return {}
    infos = pages[0].get("imageinfo") or []
    return infos[0] if infos else {}


def _ext_value(ext: Dict[str, Any], field: str) -> str:
    raw = ext.get(field)
    if isinstance(raw, dict):
        return strip_html(str(raw.get("value") or ""))
    return ""


def _classify_media(file_title: str, page_title: str, mime: str, desc: str) -> str:
    blob = " ".join([file_title, page_title, desc]).lower()
    if "map" in blob or "sanborn" in blob or "topograph" in blob:
        return "map"
    if "aerial" in blob or "satellite" in blob:
        return "aerial"
    if "postcard" in blob:
        return "postcard"
    if _BAD_TITLE_RE.search(blob):
        return "document"
    if mime and not mime.startswith("image/"):
        return "document"
    return "photo"


def _looks_clean_file(file_title: str, mime: str, width: Any, height: Any) -> bool:
    if not file_title or _BAD_TITLE_RE.search(file_title):
        return False
    if mime in {"image/svg+xml", "application/pdf"}:
        return False
    if mime and not mime.startswith("image/"):
        return False
    try:
        w = int(width or 0)
        h = int(height or 0)
    except Exception:
        return True
    if w and h and min(w, h) < 300:
        return False
    return True


def _page_to_asset(page: Dict[str, Any], anchor: AnchorSpec) -> Optional[ArchivalAsset]:
    page_title = str(page.get("title") or "")
    file_name = str(page.get("pageimage") or "")
    original = page.get("original") if isinstance(page.get("original"), dict) else {}
    thumbnail = page.get("thumbnail") if isinstance(page.get("thumbnail"), dict) else {}
    image_url = str(original.get("source") or thumbnail.get("source") or "")
    if not image_url or not file_name:
        return None

    info = _commons_metadata(file_name)
    ext = info.get("extmetadata") if isinstance(info.get("extmetadata"), dict) else {}
    mime = str(info.get("mime") or "")
    width = info.get("width") or original.get("width") or thumbnail.get("width")
    height = info.get("height") or original.get("height") or thumbnail.get("height")
    if not _looks_clean_file(file_name, mime, width, height):
        return None

    image_url = str(info.get("thumburl") or info.get("url") or image_url)
    record_url = str(info.get("descriptionurl") or page.get("fullurl") or "")
    description = _ext_value(ext, "ImageDescription") or str(page.get("extract") or "")
    object_name = _ext_value(ext, "ObjectName")
    date_str = _ext_value(ext, "DateTimeOriginal") or _ext_value(ext, "DateTime")
    rights = _ext_value(ext, "LicenseShortName") or _ext_value(ext, "UsageTerms")
    creator = _ext_value(ext, "Artist") or _ext_value(ext, "Credit")

    title = object_name or file_name.replace("_", " ")
    return ArchivalAsset(
        source="wikipedia",
        source_record_id=file_name.replace(" ", "_"),
        record_url=record_url or f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
        image_url=image_url,
        thumbnail_url=str(thumbnail.get("source") or ""),
        title=title,
        description=f"{page_title}. {description}".strip(),
        creator=creator,
        date=date_str,
        year=parse_year(date_str),
        place=anchor.geo,
        rights=rights,
        media_hint=_classify_media(file_name, page_title, mime, description),
        width=int(width) if str(width or "").isdigit() else None,
        height=int(height) if str(height or "").isdigit() else None,
        raw={
            "article_title": page_title,
            "article_url": page.get("fullurl") or "",
            "file": file_name,
            "mime": mime,
        },
    )


def _queries(anchor: AnchorSpec, query: str) -> Iterable[str]:
    seen: set[str] = set()
    candidates = [
        anchor.subject,
        f"{anchor.subject} {anchor.geo}".strip(),
        query,
    ]
    candidates.extend(anchor.queries or [])
    for item in candidates:
        cleaned = re.sub(r"\s+", " ", str(item or "")).strip(" ,.-")
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            yield cleaned


def search(anchor: AnchorSpec, query: str) -> List[ArchivalAsset]:
    out: List[ArchivalAsset] = []
    seen_files: set[str] = set()
    for q in _queries(anchor, query):
        try:
            payload = get_json(WIKI_API_URL, params=_page_query_params(q))
        except Exception:
            continue
        pages = ((payload.get("query") or {}).get("pages")) or []
        if isinstance(pages, dict):
            pages = list(pages.values())
        for page in pages:
            if not isinstance(page, dict):
                continue
            file_name = str(page.get("pageimage") or "").lower()
            if not file_name or file_name in seen_files:
                continue
            asset = _page_to_asset(page, anchor)
            if asset:
                seen_files.add(file_name)
                out.append(asset)
        if out:
            break
    return out
