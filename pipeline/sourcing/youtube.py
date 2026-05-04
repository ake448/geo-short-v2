"""
youtube.py — yt-dlp search + metadata + trim.

Two-tier search:
  Tier 1: trusted-channel allowlist (fast, high-precision)
  Tier 2: broad ytsearch  (wide net)

All metadata is cached in SQLite via Cache.upsert_video_meta so repeat queries
are free.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..cache import Cache, get_cache
from ..config import CHANNEL_ALLOWLIST, COOKIES_YT, OUT_H, OUT_W, YTDLP_PATH, NEGATIVE_SEARCH_TERMS

# ── Constants ────────────────────────────────────────────────────────────────
SEARCH_TIMEOUT = 45
META_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 180
TRIM_TIMEOUT = 60
MIN_VIDEO_SEC = 12
MAX_VIDEO_SEC = 600


# ── Allowlist helpers ────────────────────────────────────────────────────────
def trusted_channels_for_kind(kind: str) -> List[Dict[str, str]]:
    """Return b-roll channels matching this visual.kind. Empty if kind is not YouTube-sourced."""
    p = Path(CHANNEL_ALLOWLIST)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data.get("kinds", {}).get(kind, {}).get("channels", [])


def search_templates_for_kind(kind: str) -> List[str]:
    """Per-kind query templates with {place} / {subject} slots."""
    p = Path(CHANNEL_ALLOWLIST)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data.get("kinds", {}).get(kind, {}).get("search_templates", [])


def is_explainer_channel(handle: str) -> bool:
    """True if the channel is in explainers_reference_only."""
    if not handle:
        return False
    p = Path(CHANNEL_ALLOWLIST)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    
    explainers = data.get("explainers_reference_only", {}).get("channels", [])
    handle_lower = handle.lower()
    for ch in explainers:
        h = ch.get("handle", "").lower()
        n = ch.get("name", "").lower()
        if handle_lower == h or handle_lower == n:
            return True
        # Sometimes handle comes prepended with @ or without
        if h and h.strip("@") == handle_lower.strip("@"):
            return True
    return False


# ── yt-dlp wrappers ──────────────────────────────────────────────────────────
def _ytdlp_base() -> List[str]:
    cmd = [YTDLP_PATH, "--no-warnings", "--no-playlist", "--ignore-config"]
    if COOKIES_YT:
        cmd += ["--cookies", COOKIES_YT]
    return cmd


def search(query: str, n: int = 5, *, channel_handle: Optional[str] = None) -> List[str]:
    """Return up to N video IDs for the query.

    If channel_handle given (e.g. '@ProwalkTours'), prepends it to the query —
    YouTube's search engine treats this as a soft channel filter.
    """
    full_q = f"{channel_handle} {query}" if channel_handle else query
    
    # Append negative terms from config to filter low-quality/off-topic hits
    if NEGATIVE_SEARCH_TERMS:
        full_q += " " + " ".join(NEGATIVE_SEARCH_TERMS)
    
    flat_url = f"ytsearch{n}:{full_q}"
    cmd = _ytdlp_base() + ["--flat-playlist", "--print", "%(id)s", "-i", flat_url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=SEARCH_TIMEOUT)
    except subprocess.TimeoutExpired:
        return []
    if r.returncode != 0:
        return []
    ids = [line.strip() for line in r.stdout.splitlines() if re.fullmatch(r"[A-Za-z0-9_-]{11}", line.strip())]
    return ids[:n]


def list_channel_videos(channel_id: str, limit: int = 30) -> List[Dict[str, str]]:
    """Pull recent videos from a channel directly. Used when search by channel
    handle is unreliable — we get video IDs and let the text gate score them."""
    url = f"https://www.youtube.com/channel/{channel_id}/videos"
    cmd = _ytdlp_base() + [
        "--flat-playlist", "--playlist-end", str(limit),
        "--print", "%(id)s\t%(title)s", "-i", url,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=SEARCH_TIMEOUT)
    except subprocess.TimeoutExpired:
        return []
    if r.returncode != 0:
        return []
    results = []
    for line in r.stdout.splitlines():
        parts = line.strip().split("\t", 1)
        if len(parts) == 2 and re.fullmatch(r"[A-Za-z0-9_-]{11}", parts[0]):
            results.append({"video_id": parts[0], "title": parts[1]})
    return results[:limit]


def fetch_metadata(video_id: str, *, cache: Optional[Cache] = None) -> Optional[Dict[str, Any]]:
    """Get + cache metadata for one video. Returns None on failure."""
    cache = cache or get_cache()
    hit = cache.get_video_meta(video_id)
    if hit:
        return hit

    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = _ytdlp_base() + ["--dump-single-json", "--skip-download", url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=META_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    try:
        raw = json.loads(r.stdout)
    except Exception:
        return None

    meta = {
        "video_id": video_id,
        "title": raw.get("title") or "",
        "channel": raw.get("uploader") or raw.get("channel") or "",
        "channel_id": raw.get("channel_id") or raw.get("uploader_id") or "",
        "description": (raw.get("description") or "")[:2000],
        "tags": raw.get("tags") or [],
        "duration": float(raw.get("duration") or 0.0),
        "view_count": int(raw.get("view_count") or 0),
        "upload_date": raw.get("upload_date") or "",
    }
    cache.upsert_video_meta(meta)
    return meta


def download_trim(video_id: str, start_sec: float, dur_sec: float,
                  out_path: Path) -> bool:
    """Download a windowed segment via yt-dlp then trim + reframe to 9:16
    1080x1920 with our own ffmpeg pass.

    Strategy: yt-dlp's --download-sections is incompatible with DASH bv+ba
    merging (the merge step crashes with code 143). So we either:
      A) sectioned download of a single-file progressive format (22=720p, 18=360p)
         using the ffmpeg downloader (fast, ~5s)
      B) full-video download with a height cap, then ffmpeg-trim the window
         (slower but works for everything)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="v2dl_"))
    raw_template = str(tmp_dir / "raw.%(ext)s")

    # ±3s padding so our trim doesn't land on a non-keyframe boundary.
    pad = 3.0
    seg_start = max(0.0, start_sec - pad)
    seg_end = seg_start + dur_sec + pad * 2
    section = f"*{seg_start:.2f}-{seg_end:.2f}"

    url = f"https://www.youtube.com/watch?v={video_id}"

    # Strategy A: sectioned download of a progressive single-file format.
    # Format 22 = 720p mp4, format 18 = 360p mp4 — both have audio+video in
    # one stream, so no merge is required and --download-sections works.
    cmd_a = _ytdlp_base() + [
        "--download-sections", section,
        "--downloader", "ffmpeg",
        "--downloader-args", "ffmpeg:-loglevel error",
        "-f", "22/18",
        "-o", raw_template,
        url,
    ]
    try:
        subprocess.run(cmd_a, capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT)
    except subprocess.TimeoutExpired:
        pass

    raw_files = [f for f in tmp_dir.glob("raw.*") if f.stat().st_size > 1024]

    # Strategy B fallback: full-video download with a height cap. Slower but
    # works for videos that have no progressive format.
    if not raw_files:
        for f in tmp_dir.glob("raw.*"):
            f.unlink(missing_ok=True)
        cmd_b = _ytdlp_base() + [
            "-f", "best[ext=mp4][height<=720]/best[height<=720]/best[ext=mp4]/best",
            "-o", raw_template,
            url,
        ]
        try:
            subprocess.run(cmd_b, capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT)
        except subprocess.TimeoutExpired:
            _rmtree(tmp_dir)
            return False
        raw_files = [f for f in tmp_dir.glob("raw.*") if f.stat().st_size > 1024]
        if not raw_files:
            _rmtree(tmp_dir)
            return False
        raw = raw_files[0]
        # When we downloaded the full video, our window is at the absolute
        # start_sec, not at the in-segment offset.
        ss_offset = start_sec
    else:
        raw = raw_files[0]
        # In sectioned mode, content begins `pad` seconds into the segment.
        ss_offset = pad

    # Trim + reframe to 9:16 with blur-bar for non-vertical sources.
    # Note: must use -filter_complex (not -vf) for split — the -vf path
    # triggers an x264 malloc failure on Windows ffmpeg 62.
    fc = (
        f"[0:v]split=2[a][b];"
        f"[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
        f"crop={OUT_W}:{OUT_H},boxblur=25:5[bg];"
        f"[b]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
    )
    cmd2 = [
        "ffmpeg", "-y", "-ss", f"{ss_offset:.2f}", "-i", str(raw),
        "-t", f"{dur_sec:.2f}",
        "-filter_complex", fc, "-map", "[v]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "20",
        "-an", str(out_path),
    ]
    try:
        r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=TRIM_TIMEOUT)
        if r2.returncode != 0:
            import sys
            print(f"      [ffmpeg trim fail rc={r2.returncode}] {video_id}: {r2.stderr[-300:]}", file=sys.stderr, flush=True)
    finally:
        _rmtree(tmp_dir)
    return r2.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1024


def _rmtree(p: Path) -> None:
    try:
        for f in p.glob("*"):
            f.unlink(missing_ok=True)
        p.rmdir()
    except Exception:
        pass


_HARVESTED_POOL: Dict[str, List[str]] = {}

def harvest_related(video_id: str) -> List[str]:
    """Grab related videos from a winning candidate."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = _ytdlp_base() + ["--flat-playlist", "--print", "%(id)s", url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=SEARCH_TIMEOUT)
    except subprocess.TimeoutExpired:
        return []
    res = []
    if r.returncode == 0:
        for line in r.stdout.splitlines():
            vid = line.strip()
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid) and vid != video_id:
                res.append(vid)
    return res[:10]

# ── Candidate generators ─────────────────────────────────────────────────────
def _get_channel_videos_cached(channel_id: str, limit: int = 20) -> List[Dict[str, str]]:
    import time
    cache_path = Path(CHANNEL_ALLOWLIST).parent / "channel_videos_cache.json"
    cache_data = {}
    if cache_path.exists():
        try:
            cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    
    now = time.time()
    hit = cache_data.get(channel_id)
    if hit and (now - hit.get("ts", 0)) < 86400:
        return hit.get("videos", [])[:limit]
    
    videos = list_channel_videos(channel_id, limit=30)
    if videos:
        cache_data[channel_id] = {"ts": now, "videos": videos}
        try:
            cache_path.write_text(json.dumps(cache_data), encoding="utf-8")
        except Exception:
            pass
    return videos[:limit]


def candidates_for_beat(beat: Dict[str, Any], topic_class: str,
                        n_per_query: int = 3, max_total: int = 12) -> List[str]:
    """Return ordered list of candidate video_ids:
       Tier 1 (allowlist channels) first, then Tier 2 (broad search)."""
    visual = beat.get("visual") or {}
    kind = visual.get("kind", "unknown")
    queries = list(visual.get("queries") or [])
    geo = str(visual.get("geo") or "")

    seen: List[str] = []

    if kind not in ["street_level", "drone_aerial", "landmark"]:
        return []

    geo_parts = geo.lower().replace(",", " ").split()
    place_words = [w for w in geo_parts if len(w) > 3]
    
    # Tier 0: allowlist channels absolute sweep
    for ch in trusted_channels_for_kind(kind)[:10]:
        cid = ch.get("channel_id") or ch.get("id")
        if not cid:
            continue
        recent_videos = _get_channel_videos_cached(cid, limit=20)
        found_tier0 = 0
        for vid_obj in recent_videos:
            vid = vid_obj["video_id"]
            title = vid_obj["title"].lower()
            # Lightweight title regex pre-filter
            if (place_words and any(pw in title for pw in place_words)) or (geo.lower() in title and len(geo) > 3):
                if vid not in seen:
                    seen.append(vid)
                    found_tier0 += 1
                if len(seen) >= max_total:
                    print(f"      [tier 0] fetched {found_tier0} exact matches from {ch.get('handle')}")
                    return seen
        if found_tier0 > 0:
            print(f"      [tier 0] fetched {found_tier0} exact matches from {ch.get('handle')}")

    # Tier 1: allowlist channels — handle prepended to query
    for ch in trusted_channels_for_kind(kind)[:8]:
        handle = ch.get("handle") or ""
        if not handle:
            continue
        for q in queries[:2]:
            for vid in search(q, n=n_per_query, channel_handle=handle):
                if vid not in seen:
                    seen.append(vid)
                if len(seen) >= max_total:
                    return seen

    # Tier 1.5: allowlist search templates instantiated with the place
    for tmpl in search_templates_for_kind(kind)[:3]:
        if "{" not in tmpl:
            q = tmpl
        else:
            try:
                q = tmpl.format(city=geo, region=geo, place=geo, feature=geo, subject=visual.get("subject", geo))
            except KeyError:
                q = tmpl
        for vid in search(q, n=n_per_query):
            if vid not in seen:
                seen.append(vid)
            if len(seen) >= max_total:
                return seen

    # Tier 1.5b: Harvested related videos for this visual kind
    for vid in _HARVESTED_POOL.get(kind, [])[:5]:
        if vid not in seen:
            seen.append(vid)
        if len(seen) >= max_total:
            return seen

    # Tier 2: broad beat queries
    for q in queries:
        for vid in search(q, n=n_per_query):
            if vid not in seen:
                seen.append(vid)
            if len(seen) >= max_total:
                return seen

    return seen
