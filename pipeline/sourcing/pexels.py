"""
pexels.py — Pexels + Pixabay video fallback. Loose beats only.

Used when YouTube tier-1/2 search produces no acceptable footage. Pexels and
Pixabay are stock libraries — no exact-place footage, but reliable for biome
and visual-kind matches (city night, river aerial, market crowd, etc.).
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import OUT_H, OUT_W, PEXELS_API_KEY, PIXABAY_API_KEY

PEXELS_URL = "https://api.pexels.com/videos/search"
PEXELS_PHOTOS_URL = "https://api.pexels.com/v1/search"
PIXABAY_URL = "https://pixabay.com/api/videos/"
PIXABAY_PHOTOS_URL = "https://pixabay.com/api/"
DOWNLOAD_TIMEOUT = 90
TRIM_TIMEOUT = 60


def _has_negative_terms(blob: str, negative_terms: Optional[List[str]]) -> bool:
    if not negative_terms:
        return False
    blob_l = (blob or "").lower()
    return any(term.lower().lstrip("-") in blob_l for term in negative_terms if term)


def _http_json(url: str, headers: Optional[Dict[str, str]] = None,
               timeout: int = 20) -> Optional[Dict[str, Any]]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def search_pexels(query: str, per_page: int = 8) -> List[Dict[str, Any]]:
    if not PEXELS_API_KEY:
        return []
    qs = urllib.parse.urlencode({"query": query, "per_page": per_page,
                                  "orientation": "portrait", "size": "large"})
    data = _http_json(f"{PEXELS_URL}?{qs}", headers={"Authorization": PEXELS_API_KEY})
    if not data:
        return []
    out = []
    for v in data.get("videos", []):
        files = sorted(v.get("video_files", []),
                       key=lambda f: (f.get("height") or 0), reverse=True)
        if not files:
            continue
        best = files[0]
        out.append({
            "source": "pexels",
            "id": str(v.get("id")),
            "duration": float(v.get("duration") or 0.0),
            "width": int(best.get("width") or 0),
            "height": int(best.get("height") or 0),
            "url": best.get("link"),
            "credit": v.get("user", {}).get("name", ""),
        })
    return out


def search_pexels_photos(query: str, per_page: int = 8,
                         negative_terms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Search Pexels still photos for a single-card contextual image."""
    if not PEXELS_API_KEY:
        return []
    qs = urllib.parse.urlencode({
        "query": query,
        "per_page": per_page,
        "orientation": "landscape",
        "size": "large",
    })
    data = _http_json(f"{PEXELS_PHOTOS_URL}?{qs}", headers={"Authorization": PEXELS_API_KEY})
    if not data:
        return []
    out = []
    for p in data.get("photos", []):
        src = p.get("src") or {}
        url = src.get("large2x") or src.get("large") or src.get("original")
        if not url:
            continue
        alt = p.get("alt") or ""
        if _has_negative_terms(" ".join([alt, p.get("photographer") or "", url]), negative_terms):
            continue
        out.append({
            "source": "pexels",
            "id": str(p.get("id")),
            "width": int(p.get("width") or 0),
            "height": int(p.get("height") or 0),
            "url": url,
            "credit": (p.get("photographer") or ""),
            "tags": alt,
        })
    return out


def search_pixabay_photos(query: str, per_page: int = 8,
                          negative_terms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if not PIXABAY_API_KEY:
        return []
    pixabay_query = query
    if negative_terms:
        negatives = " ".join(f"-{term.lower().lstrip('-')}" for term in negative_terms if term)
        pixabay_query = f"{query} {negatives}".strip()
    qs = urllib.parse.urlencode({
        "key": PIXABAY_API_KEY,
        "q": pixabay_query,
        "per_page": per_page,
        "image_type": "photo",
        "orientation": "horizontal",
        "safesearch": "true",
    })
    data = _http_json(f"{PIXABAY_PHOTOS_URL}?{qs}")
    if not data:
        return []
    out = []
    for p in data.get("hits", []):
        url = p.get("largeImageURL") or p.get("webformatURL")
        if not url:
            continue
        tags = p.get("tags") or ""
        if _has_negative_terms(" ".join([tags, p.get("user") or "", url]), negative_terms):
            continue
        out.append({
            "source": "pixabay",
            "id": str(p.get("id")),
            "width": int(p.get("imageWidth") or p.get("webformatWidth") or 0),
            "height": int(p.get("imageHeight") or p.get("webformatHeight") or 0),
            "url": url,
            "credit": (p.get("user") or ""),
            "tags": tags,
        })
    return out


def search_photos(query: str, per_page: int = 8,
                  negative_terms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Combined still-photo search: Pexels first, Pixabay fallback."""
    return (
        search_pexels_photos(query, per_page, negative_terms=negative_terms)
        + search_pixabay_photos(query, per_page, negative_terms=negative_terms)
    )


def search_pixabay(query: str, per_page: int = 8) -> List[Dict[str, Any]]:
    if not PIXABAY_API_KEY:
        return []
    qs = urllib.parse.urlencode({"key": PIXABAY_API_KEY, "q": query,
                                  "per_page": per_page, "video_type": "film"})
    data = _http_json(f"{PIXABAY_URL}?{qs}")
    if not data:
        return []
    out = []
    for v in data.get("hits", []):
        videos = v.get("videos") or {}
        # Prefer "medium" (~1080p) — "large" is often 4K and OOMs ffmpeg.
        for size in ("medium", "small", "large"):
            f = videos.get(size)
            if f and f.get("url"):
                out.append({
                    "source": "pixabay",
                    "id": str(v.get("id")),
                    "duration": float(v.get("duration") or 0.0),
                    "width": int(f.get("width") or 0),
                    "height": int(f.get("height") or 0),
                    "url": f.get("url"),
                    "credit": v.get("user") or "",
                })
                break
    return out


def search(query: str, per_page: int = 8) -> List[Dict[str, Any]]:
    """Combined: Pexels first (better quality), then Pixabay."""
    return search_pexels(query, per_page) + search_pixabay(query, per_page)


_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def fetch_and_trim(item: Dict[str, Any], dur_sec: float, out_path: Path) -> bool:
    """Download the source mp4 and re-encode to 9:16. Trim to dur_sec from start."""
    url = item.get("url")
    if not url:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="v2px_"))
    raw = tmp_dir / "raw.mp4"
    try:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as r, open(raw, "wb") as f:
                f.write(r.read())
        except Exception:
            return False

        # 4K sources OOM ffmpeg on Windows: H.264 frame-threading creates one
        # DPB per thread. Force slice threading (single DPB) and downscale
        # to 1280-wide before the split. -x264-params and -tune cause encoder
        # init failures here, so keep flags minimal.
        fc = (
            f"[0:v]scale=1280:-2:flags=bilinear,split=2[a][b];"
            f"[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{OUT_H},boxblur=15:1[bg];"
            f"[b]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-threads", "1", "-thread_type", "slice",
            "-ss", "0", "-i", str(raw), "-t", f"{dur_sec:.2f}",
            "-filter_complex", fc, "-map", "[v]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "ultrafast", "-crf", "23", "-an",
            str(out_path),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=TRIM_TIMEOUT)
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0 and out_path.exists()
    finally:
        try:
            for f in tmp_dir.glob("*"):
                f.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except Exception:
            pass
