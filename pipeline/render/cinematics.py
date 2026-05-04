"""
Synthetic V2 cinematic fallbacks.

Public contract:
    make_fallback(beat, out_path) -> bool

The generated files are silent 1080x1920 H.264 clips at the beat duration.
Network-backed renderers cache their expensive inputs under DATA_DIR.
"""
from __future__ import annotations

import io
import json
import math
import re
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from ..config import (
    DATA_DIR,
    FONTS_DIR,
    FPS,
    INFOGRAPHIC_BG_FROM_BROLL,
    MAPBOX_TOKEN,
    OUT_H,
    OUT_W,
    RENDER_ARCHIVAL_CALLOUTS,
)
from ..sourcing.context_card import (
    fetch_context_backdrop_image,
    fetch_context_card_image,
    fetch_thematic_archival_image,
)

BG_NAVY = "#0a0e27"
BG_NAVY_LIGHT = "#161b3d"
ACCENT_YELLOW = "#FFD700"
WHITE = "#f5f7ff"
MUTED = "#aab2d5"

MAPBOX_STYLE = "satellite-streets-v12"
TILE_SIZE = 512
TILE_SCALE = "@2x"
USER_AGENT = "UrbanAtlasV2/1.0 (synthetic-video-renderer)"
_GEOCODE_LOCK = Lock()


@dataclass(frozen=True)
class GeoResult:
    name: str
    lat: float
    lon: float
    bbox: Tuple[float, float, float, float]


def make_fallback(
    beat: dict,
    out_path: Path,
    sources_for_all_beats: Optional[List[Dict[str, Any]]] = None,
    source_index: Optional[int] = None,
) -> bool:
    """Synthesise a 1080x1920 30fps mp4 for ``beat``.

    Returns False for unsupported visual kinds or recoverable renderer failures.
    """
    visual = beat.get("visual") or {}
    kind = str(visual.get("kind") or "").strip().lower()
    try:
        if kind == "map_zoom":
            return _render_map_zoom(beat, Path(out_path), animated_path=False)
        if kind == "map_animation":
            return _render_map_zoom(beat, Path(out_path), animated_path=True)
        if kind == "density_infographic":
            return _render_density_infographic(beat, Path(out_path), sources_for_all_beats, source_index)
        if kind == "dynamic_infographic":
            return _render_dynamic_infographic(beat, Path(out_path), sources_for_all_beats, source_index)
        if kind == "archival_annotated":
            return _render_archival_annotated(beat, Path(out_path), sources_for_all_beats, source_index)
        if kind == "wikipedia_photo":
            return _render_wikipedia_photo(beat, Path(out_path))
    except Exception as exc:
        print(f"[cinematics] {kind or 'unknown'} failed: {exc}", flush=True)
        _safe_unlink(Path(out_path))
        return False
    return False


# ---------------------------------------------------------------------------
# ffmpeg frame pipe


def _frame_count(duration_sec: Any) -> int:
    try:
        dur = float(duration_sec)
    except Exception:
        dur = 5.0
    return max(1, int(round(max(0.05, dur) * FPS)))


def _write_video(
    frames: Iterable[Image.Image],
    out_path: Path,
    frame_count: int,
    target_duration: float,
) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not _safe_unlink(out_path):
        return False
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{OUT_W}x{OUT_H}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "ultrafast",
        "-crf",
        "22",
        str(out_path),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return False

    assert proc.stdin is not None
    try:
        written = 0
        for frame in frames:
            if written >= frame_count:
                break
            if frame.size != (OUT_W, OUT_H):
                frame = frame.resize((OUT_W, OUT_H), Image.Resampling.LANCZOS)
            if frame.mode != "RGB":
                frame = frame.convert("RGB")
            proc.stdin.write(frame.tobytes())
            written += 1
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        try:
            proc.stdin.close()
        except Exception:
            pass
    stderr = b""
    if proc.stderr is not None:
        stderr = proc.stderr.read()
    rc = proc.wait()
    if rc != 0 or not out_path.exists() or out_path.stat().st_size < 1024:
        msg = stderr.decode("utf-8", errors="replace").strip()
        if msg:
            print(f"[cinematics] ffmpeg failed: {msg[:500]}", flush=True)
        _safe_unlink(out_path)
        return False
    return _duration_ok(out_path, target_duration)


def _duration_ok(path: Path, target_duration: float) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    try:
        got = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=10)
        dur = float(got.strip())
    except Exception:
        return True
    if abs(dur - target_duration) <= 0.05:
        return True
    print(f"[cinematics] duration mismatch: got {dur:.3f}s target {target_duration:.3f}s", flush=True)
    _safe_unlink(path)
    return False


def _safe_unlink(path: Path) -> bool:
    try:
        path.unlink(missing_ok=True)
        return True
    except PermissionError:
        return False
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Mapbox geocoding and tile cache


def _render_map_zoom(beat: dict, out_path: Path, animated_path: bool) -> bool:
    visual = beat.get("visual") or {}
    geo_text = str(visual.get("geo") or visual.get("subject") or "").strip()
    if not geo_text:
        return False

    frame_count = _frame_count(beat.get("duration_sec"))
    duration = frame_count / FPS

    # Primary path: Earth-cinematic (raycast sphere + blue highlight of the region
    # outline). Looks dramatically better than a flat tile crop-zoom. Only viable
    # when geo_text resolves to a region with a real polygon boundary.
    if _earth_highlight_viable(geo_text):
        if _render_earth_highlight(geo_text, out_path, duration):
            return True

    # Fallback: flat Mapbox tile crop + pan + pin overlay.
    if not MAPBOX_TOKEN:
        return False

    geo = _geocode(geo_text)
    if geo is None:
        return False

    zoom = _zoom_for_bbox(geo.bbox, geo_text, animated_path=animated_path)
    canvas = _stitched_map(geo.lat, geo.lon, zoom)
    if canvas is None:
        return False

    label = _short_place_label(geo_text or geo.name)
    frames = _map_frames(canvas, frame_count, label, animated_path)
    return _write_video(frames, out_path, frame_count, duration)


# ── Earth-cinematic (sphere raycast) path ────────────────────────────────────

_CINEMATICS_MOD = None
_CINEMATICS_TRIED = False


def _earth_cinematics():
    """Lazy-load generate_cinematics.py from repo root (heavy numpy/PIL module)."""
    global _CINEMATICS_MOD, _CINEMATICS_TRIED
    if _CINEMATICS_TRIED:
        return _CINEMATICS_MOD
    _CINEMATICS_TRIED = True
    try:
        import importlib.util
        from pathlib import Path as _P
        root = _P(__file__).resolve().parents[2]
        gc_path = root / "generate_cinematics.py"
        if not gc_path.exists():
            return None
        spec = importlib.util.spec_from_file_location("generate_cinematics", str(gc_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _CINEMATICS_MOD = mod
    except Exception as exc:
        print(f"[cinematics] earth-highlight load failed: {exc}", flush=True)
    return _CINEMATICS_MOD


def _earth_highlight_viable(geo_text: str) -> bool:
    # Heuristic: a comma-delimited region name ("Tochigi, Japan", "Kanto, Japan")
    # is what fetch_boundary needs. Pure city-only strings without a country can
    # still succeed, but regions are the sweet spot.
    return bool(geo_text and len(geo_text) >= 3)


def _render_earth_highlight(geo_text: str, out_path: Path, duration: float) -> bool:
    mod = _earth_cinematics()
    if mod is None or not hasattr(mod, "render_region_highlight_to"):
        return False
    try:
        return bool(mod.render_region_highlight_to(geo_text, out_path, duration))
    except Exception as exc:
        print(f"[cinematics] earth highlight failed: {exc}", flush=True)
        return False


def _geocode(place: str) -> Optional[GeoResult]:
    cache_path = DATA_DIR / "geocode.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    key = place.strip()
    with _GEOCODE_LOCK:
        cache = _read_geocode_cache(cache_path)
        cached = cache.get(key)
        if isinstance(cached, dict):
            try:
                return GeoResult(
                    name=str(cached.get("name") or key),
                    lat=float(cached["lat"]),
                    lon=float(cached["lon"]),
                    bbox=tuple(float(x) for x in cached["bbox"]),  # type: ignore[arg-type]
                )
            except Exception:
                pass

    url = (
        "https://api.mapbox.com/geocoding/v5/mapbox.places/"
        f"{urllib.parse.quote(place)}.json?"
        f"access_token={urllib.parse.quote(MAPBOX_TOKEN)}&limit=1"
    )
    try:
        data = _read_json(url, timeout=20)
        feature = (data.get("features") or [])[0]
        lon, lat = feature["center"][:2]
        bbox_raw = feature.get("bbox")
        if not bbox_raw:
            bbox_raw = _fallback_bbox(float(lat), float(lon))
        bbox = tuple(float(x) for x in bbox_raw[:4])
        result = GeoResult(
            name=str(feature.get("place_name") or place),
            lat=float(lat),
            lon=float(lon),
            bbox=bbox,  # type: ignore[arg-type]
        )
    except Exception as exc:
        print(f"[cinematics] geocode failed for {place!r}: {exc}", flush=True)
        return None

    with _GEOCODE_LOCK:
        cache = _read_geocode_cache(cache_path)
        cache[key] = {
            "name": result.name,
            "lat": result.lat,
            "lon": result.lon,
            "bbox": list(result.bbox),
        }
        try:
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return result


def _read_geocode_cache(cache_path: Path) -> Dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_json(url: str, timeout: int = 20) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _fallback_bbox(lat: float, lon: float) -> Tuple[float, float, float, float]:
    lat_delta = 0.22
    lon_delta = lat_delta / max(0.25, math.cos(math.radians(max(-80.0, min(80.0, lat)))))
    return (lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta)


def _zoom_for_bbox(
    bbox: Tuple[float, float, float, float],
    place: str = "",
    animated_path: bool = False,
) -> int:
    min_lon, min_lat, max_lon, max_lat = bbox
    span = max(abs(max_lon - min_lon), abs(max_lat - min_lat))
    if "," in place and span <= 25:
        return 10 if animated_path else 11
    if span <= 1.5:
        return 11
    if span <= 8:
        return 8
    if span <= 30:
        return 6
    return 5


def _stitched_map(lat: float, lon: float, zoom: int) -> Optional[Image.Image]:
    cx, cy = _lonlat_to_tile(lon, lat, zoom)
    tile_px = TILE_SIZE * 2
    canvas = Image.new("RGB", (tile_px * 3, tile_px * 3), BG_NAVY)
    n = 2**zoom
    for row, dy in enumerate((-1, 0, 1)):
        for col, dx in enumerate((-1, 0, 1)):
            x = (cx + dx) % n
            y = max(0, min(n - 1, cy + dy))
            tile = _mapbox_tile(zoom, x, y)
            if tile is None:
                return None
            canvas.paste(tile.resize((tile_px, tile_px), Image.Resampling.LANCZOS), (col * tile_px, row * tile_px))
    return canvas


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    lat = max(-85.05112878, min(85.05112878, lat))
    lat_rad = math.radians(lat)
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def _mapbox_tile(zoom: int, x: int, y: int) -> Optional[Image.Image]:
    tile_dir = DATA_DIR / "tiles" / "mapbox" / MAPBOX_STYLE / str(zoom)
    tile_dir.mkdir(parents=True, exist_ok=True)
    cache_path = tile_dir / f"{x}_{y}_2x.png"
    if cache_path.exists() and cache_path.stat().st_size > 512:
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            cache_path.unlink(missing_ok=True)

    url = (
        "https://api.mapbox.com/styles/v1/mapbox/"
        f"{MAPBOX_STYLE}/tiles/{TILE_SIZE}/{zoom}/{x}/{y}{TILE_SCALE}"
        f"?access_token={urllib.parse.quote(MAPBOX_TOKEN)}"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.save(cache_path, "PNG")
        return img
    except (urllib.error.URLError, OSError) as exc:
        print(f"[cinematics] tile fetch failed z{zoom}/{x}/{y}: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Frame composition


def _map_frames(
    canvas: Image.Image,
    frame_count: int,
    label: str,
    animated_path: bool,
) -> Iterator[Image.Image]:
    cw, ch = canvas.size
    aspect = OUT_W / OUT_H
    start_h = ch * (0.98 if animated_path else 0.92)
    end_h = ch * (0.48 if animated_path else 0.58)
    start_h = min(start_h, cw / aspect, ch)
    end_h = min(end_h, cw / aspect, ch)
    end_h = max(end_h, OUT_H * 0.7)
    start_w = start_h * aspect
    end_w = end_h * aspect

    zoom_frames = frame_count if not animated_path else max(1, min(frame_count, int(round(3.0 * FPS))))
    for idx in range(frame_count):
        if animated_path:
            t = min(1.0, idx / max(1, zoom_frames - 1))
        else:
            t = idx / max(1, frame_count - 1)
        e = _ease_in_out_cubic(t)
        crop_w = _lerp(start_w, end_w, e)
        crop_h = _lerp(start_h, end_h, e)
        cx = cw / 2
        cy = ch / 2
        left = int(round(cx - crop_w / 2))
        top = int(round(cy - crop_h / 2))
        crop = canvas.crop((left, top, int(round(left + crop_w)), int(round(top + crop_h))))
        frame = crop.resize((OUT_W, OUT_H), Image.Resampling.LANCZOS)
        frame = _grade_map_frame(frame)
        _draw_map_overlay(frame, label)
        yield frame


def _grade_map_frame(frame: Image.Image) -> Image.Image:
    frame = ImageOps.autocontrast(frame, cutoff=1)
    overlay = Image.new("RGB", frame.size, BG_NAVY)
    frame = Image.blend(frame, overlay, 0.12)
    top = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(top)
    d.rectangle((0, 0, OUT_W, 260), fill=(0, 0, 0, 70))
    d.rectangle((0, OUT_H - 420, OUT_W, OUT_H), fill=(0, 0, 0, 95))
    return Image.alpha_composite(frame.convert("RGBA"), top).convert("RGB")


def _draw_map_overlay(frame: Image.Image, label: str) -> None:
    draw = ImageDraw.Draw(frame)
    pin_x, pin_y = OUT_W // 2, int(OUT_H * 0.47)
    draw.line((pin_x, pin_y + 18, pin_x, pin_y + 86), fill=(0, 0, 0), width=10)
    draw.line((pin_x, pin_y + 18, pin_x, pin_y + 86), fill=ACCENT_YELLOW, width=5)
    draw.ellipse((pin_x - 30, pin_y - 30, pin_x + 30, pin_y + 30), fill=(0, 0, 0), outline=None)
    draw.ellipse((pin_x - 22, pin_y - 22, pin_x + 22, pin_y + 22), fill=ACCENT_YELLOW, outline=(0, 0, 0), width=4)

    font = _font(78)
    small = _font(30)
    text = _fit_text(label.upper(), font, OUT_W - 140)
    text_w = _text_bbox(draw, text, font)[0]
    x = (OUT_W - text_w) // 2
    y = int(OUT_H * 0.855)
    draw.text((x, y), text, font=font, fill=WHITE, stroke_width=6, stroke_fill=(0, 0, 0))
    sub = "MAP CONTEXT"
    sub_w = _text_bbox(draw, sub, small)[0]
    draw.text(((OUT_W - sub_w) // 2, y - 48), sub, font=small, fill=ACCENT_YELLOW, stroke_width=2, stroke_fill=(0, 0, 0))


def _render_density_infographic(
    beat: dict,
    out_path: Path,
    sources_for_all_beats: Optional[List[Dict[str, Any]]] = None,
    source_index: Optional[int] = None,
) -> bool:
    frame_count = _frame_count(beat.get("duration_sec"))
    duration = frame_count / FPS
    visual = beat.get("visual") or {}
    geo = str(visual.get("geo") or "")
    subject = str(visual.get("subject") or beat.get("caption_text") or beat.get("narration") or "").strip()
    bg_spec = _infographic_backdrop_spec(beat, {"geo": geo, "context_visual": subject})
    kind = _classify_infographic(beat)
    # Heatmap needs a less-blurred background so map geography shows through
    if kind == "heatmap":
        bg = _build_infographic_bg(
            beat,
            bg_spec,
            out_path,
            sources_for_all_beats,
            source_index,
            map_blur=10,
            map_navy_blend=0.38,
        )
    else:
        bg = _build_infographic_bg(
            beat,
            bg_spec,
            out_path,
            sources_for_all_beats,
            source_index,
            map_blur=30,
            map_navy_blend=0.65,
        )
    try:
        if kind == "heatmap":
            frames = _infographic_heatmap(beat, bg, frame_count)
        elif kind == "bar_chart":
            frames = _infographic_bar_chart(beat, bg, frame_count)
        elif kind == "border_shift":
            frames = _infographic_border_shift(beat, bg, frame_count)
        else:
            frames = _infographic_stat_card(beat, bg, frame_count)
        return _write_video(frames, out_path, frame_count, duration)
    except Exception as exc:
        print(f"[cinematics] infographic {kind} failed: {exc}", flush=True)
        still = _density_still(beat)
        frames_fb = _ken_burns_still_frames(still, frame_count, zoom_start=1.0, zoom_end=1.055)
        return _write_video(frames_fb, out_path, frame_count, duration)


# ---------------------------------------------------------------------------
# Dynamic Infographic renderer — motion-based animated fact card
# (group enter/drift/exit animation; complementary to the PIL classifier above)


def _dyn_ease_out(t: float) -> float:
    return 1.0 if t >= 1.0 else 1.0 - pow(2.0, -10.0 * t)


def _dyn_ease_in(t: float) -> float:
    return 0.0 if t <= 0.0 else pow(2.0, 10.0 * (t - 1.0))


_DYN_OFFSCREEN = 1260


def _dyn_offscreen(direction: str) -> Tuple[int, int]:
    return {
        "left": (-_DYN_OFFSCREEN, 0),
        "right": (_DYN_OFFSCREEN, 0),
        "top": (0, -_DYN_OFFSCREEN),
        "bottom": (0, _DYN_OFFSCREEN),
    }.get(direction, (-_DYN_OFFSCREEN, 0))


def _dyn_translate(layer: Image.Image, dx: int, dy: int) -> Image.Image:
    out = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    out.paste(layer, (dx, dy))
    return out


def _dyn_group_offset(
    direction: str,
    exit_dir: str,
    frame_idx: int,
    start: int,
    enter: int,
    hold: int,
    exit_f: int,
    drift_px: int = 28,
) -> Optional[Tuple[int, int]]:
    drift_map = {
        "left": (-drift_px, 0), "right": (drift_px, 0),
        "top": (0, -drift_px), "bottom": (0, drift_px),
    }
    enter_off = _dyn_offscreen(direction)
    exit_off = _dyn_offscreen(exit_dir)
    ds = drift_map.get(direction, (0, 0))
    de = (-ds[0], -ds[1])

    if frame_idx < start:
        return None
    f = frame_idx - start
    if f < enter:
        p = _dyn_ease_out(f / max(1, enter - 1))
        return (
            int(enter_off[0] + (ds[0] - enter_off[0]) * p),
            int(enter_off[1] + (ds[1] - enter_off[1]) * p),
        )
    f -= enter
    if f < hold:
        t = f / max(1, hold - 1)
        return (int(ds[0] + (de[0] - ds[0]) * t), int(ds[1] + (de[1] - ds[1]) * t))
    f -= hold
    if f < exit_f:
        p = _dyn_ease_in(f / max(1, exit_f - 1))
        return (
            int(de[0] + (exit_off[0] - de[0]) * p),
            int(de[1] + (exit_off[1] - de[1]) * p),
        )
    return None


def _dyn_context_card(
    bg_light: Image.Image,
    spec: Dict[str, Any],
    context_image_path: Optional[Path] = None,
) -> Image.Image:
    """Render a 700×410 context card as an RGBA layer on OUT_W×OUT_H canvas."""
    c1 = _hex_to_rgb(spec["colors"][0])
    c2 = _hex_to_rgb(spec["colors"][1])
    c3 = _hex_to_rgb(spec["colors"][2])
    card_w, card_h = 700, 410
    cx = (OUT_W - card_w) // 2
    cy = 220

    layer = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))

    shadow = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (cx + 10, cy + 14, cx + card_w + 10, cy + card_h + 14),
        radius=32, fill=(0, 0, 0, 140),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(14))
    layer = Image.alpha_composite(layer, shadow)

    # Center-crop bg_light (1080x1920) to card aspect (700:410 ≈ 1.71:1)
    used_context_image = False
    used_synthetic_context = False
    card_img: Optional[Image.Image] = None
    concept_kind = _dyn_context_concept_kind(spec)
    if concept_kind:
        card_img = _dyn_context_concept_image(spec, card_w, card_h, concept_kind)
        used_context_image = True
        used_synthetic_context = True
    elif context_image_path and context_image_path.exists():
        try:
            card_img = ImageOps.fit(
                Image.open(context_image_path).convert("RGB"),
                (card_w, card_h),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            ).convert("RGBA")
            used_context_image = True
        except Exception:
            card_img = None

    if card_img is None:
        src_w, src_h = bg_light.size
        tgt_ratio = card_w / card_h
        new_h = int(src_w / tgt_ratio)
        crop_y0 = max(0, (src_h - new_h) // 4)
        crop_y1 = min(src_h, crop_y0 + new_h)
        card_img = bg_light.crop((0, crop_y0, src_w, crop_y1)).resize(
            (card_w, card_h), Image.Resampling.LANCZOS
        ).convert("RGBA")

    cd = ImageDraw.Draw(card_img, "RGBA")
    cd.rectangle((0, 0, card_w, 80), fill=(0, 0, 0, 110))
    cd.rectangle((0, card_h - 104, card_w, card_h), fill=(0, 0, 0, 150))
    cd.rectangle((0, card_h - 12, card_w // 2, card_h), fill=(*c1, 255))
    cd.rectangle((card_w // 2, card_h - 12, card_w, card_h), fill=(*c2, 255))

    cd.text((26, 24), spec["category"], fill=(*c3, 255), font=_font(32))
    cd.text((card_w - 26, 26), spec["badge"], fill=(245, 247, 255, 255), font=_font(26), anchor="ra")

    context_vis = _dyn_context_caption(spec, synthetic=used_synthetic_context or not used_context_image)
    if context_vis and not used_context_image:
        cd.text(
            (card_w // 2, card_h - 68), context_vis[:48].upper(),
            fill=(245, 247, 255, 220), font=_font(28), anchor="ma",
        )
    elif context_vis:
        caption = context_vis[:62].upper()
        cd.text(
            (26, card_h - 52), caption,
            fill=(245, 247, 255, 190),
            font=_fit_font(caption, _font(22), card_w - 52, 16),
        )

    mask = Image.new("L", (card_w, card_h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, card_w, card_h), radius=32, fill=255)
    layer.paste(card_img, (cx, cy), mask)
    return layer


def _dyn_context_concept_kind(spec: Dict[str, Any]) -> str:
    text = " ".join(
        str(spec.get(k) or "") for k in ("context_visual", "headline", "support", "category")
    ).lower()
    if re.search(r"\b(money|dollars?|funding|buyouts?|relocation|evacuation|costs?|houses?|homes?)\b", text):
        if re.search(r"\b(money|dollars?|funding|buyouts?|relocation|evacuation|costs?)\b", text):
            return "money_house"
    if re.search(r"\b(zip|postal|post office|mailbox|mail carrier)\b", text):
        return "postal"
    if re.search(r"\b(coal|mine|underground|seam)\b", text) and re.search(r"\b(fire|burn(?:ing|ed)?|smoke|vents?)\b", text):
        return "coal_fire"
    if re.search(r"\b(anthrax|spores?|microscopic|bacteria|virus|pathogen)\b", text):
        if re.search(r"\b(burial|pit|buried|shallow pits?)\b", text):
            return "spores_burial"
        return "microscope"
    if re.search(r"\b(burial|pit|buried|cross[- ]?section)\b", text):
        return "burial"
    if re.search(r"\b(timeline|sequence|chronolog)\b", text):
        return "timeline"
    if re.search(
        r"\b(diagram|schematic|cross[- ]?section|cutaway|"
        r"geological layers?|soil layers?|sediment|strata|"
        r"chemical|contamina\w+|toxic\w*|radiation|"
        r"flood stage|water level|water table|"
        r"population graph|displacement|migration chart|"
        r"comparison chart|scale comparison|"
        r"process flow|supply chain|pipeline diagram|"
        r"anatomy|cellular|molecular|"
        r"elevation profile|topograph\w+|depth chart)\b",
        text,
    ):
        return "generic_diagram"
    return ""


def _dyn_context_caption(spec: Dict[str, Any], *, synthetic: bool = False) -> str:
    text = " ".join(
        str(spec.get(k) or "") for k in ("context_visual", "headline", "support", "category")
    ).lower()
    if re.search(r"\b(buyouts?|relocation|evacuation)\b", text):
        if "federal" in text or "congress" in text:
            return "Federal buyout funding"
        return "Relocation funding"
    if re.search(r"\b(money|dollars?|funding|costs?)\b", text):
        return "Cost / funding diagram"
    if re.search(r"\b(zip|postal|post office|mailbox)\b", text):
        return "Postal record"
    if re.search(r"\b(coal|mine|underground|seam)\b", text) and re.search(r"\b(fire|burn(?:ing|ed)?|smoke|vents?)\b", text):
        return "Underground coal fire"
    if not synthetic:
        return _dyn_clean_context_label(spec)
    if "anthrax" in text and re.search(r"\b(burial|pit|buried)\b", text):
        return "Anthrax spores / burial pit diagram"
    if "anthrax" in text or "spore" in text:
        return "Microscopic spore diagram"
    if re.search(r"\b(burial|pit|buried)\b", text):
        return "Burial pit diagram"
    if "timeline" in text:
        return "Timeline diagram"
    headline = str(spec.get("headline") or spec.get("context_visual") or "").strip()
    if headline:
        clean = re.sub(
            r"^(infographic|chart|graph|diagram|illustration|image|showing|of|the|a|an)\s+",
            "", headline, flags=re.IGNORECASE,
        ).strip()
        if clean:
            return clean[:60]
    return _dyn_clean_context_label(spec) or "Concept diagram"


def _dyn_clean_context_label(spec: Dict[str, Any]) -> str:
    raw = str(spec.get("context_visual") or "").strip()
    headline = str(spec.get("headline") or "").strip()
    source = raw or headline
    clean = re.sub(
        r"\b("
        r"what the top card should depict|context visual|stock photo|"
        r"photo|picture|image|icon|symbol|clipart|thumbnail|showing|of|"
        r"the|a|an"
        r")\b",
        " ",
        source,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"\b(or|and)\b", " / ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*/\s*", " / ", clean)
    clean = re.sub(r"\s+", " ", clean).strip(" /,.-")
    if not clean and headline:
        clean = re.sub(r"\s+", " ", headline).strip(" /,.-")
    return clean[:60]


def _dyn_context_concept_image(
    spec: Dict[str, Any],
    card_w: int,
    card_h: int,
    kind: str,
) -> Image.Image:
    """Draw a small explanatory image when the script asks for a diagram.

    This avoids pairing an unrelated archival/map photo with a label like
    "microscopic spores" or "burial pit diagram".
    """
    bg = Image.new("RGBA", (card_w, card_h), (8, 13, 24, 255))
    d = ImageDraw.Draw(bg, "RGBA")
    c1 = _hex_to_rgb(spec["colors"][0])
    c2 = _hex_to_rgb(spec["colors"][1])
    gold = _hex_to_rgb(ACCENT_YELLOW)

    for y in range(card_h):
        alpha = int(40 + 80 * y / max(1, card_h - 1))
        d.line((0, y, card_w, y), fill=(10, 18, 36, alpha))
    for x in range(0, card_w, 70):
        d.line((x, 82, x + 130, card_h - 110), fill=(255, 255, 255, 10), width=1)

    if kind == "money_house":
        d.rounded_rectangle((34, 94, card_w - 34, 298), radius=18, fill=(2, 8, 16, 185), outline=(*c1, 175), width=2)
        # House silhouette.
        house_x, house_y = 88, 176
        d.polygon(
            [(house_x, house_y), (house_x + 112, house_y - 76), (house_x + 224, house_y), (house_x + 204, house_y), (house_x + 204, house_y + 86), (house_x + 22, house_y + 86), (house_x + 22, house_y)],
            fill=(*c2, 145),
            outline=(245, 247, 255, 180),
        )
        d.rectangle((house_x + 94, house_y + 24, house_x + 134, house_y + 86), fill=(8, 13, 24, 210), outline=(*gold, 180), width=2)
        d.rectangle((house_x + 36, house_y + 20, house_x + 76, house_y + 58), fill=(8, 13, 24, 180), outline=(245, 247, 255, 115), width=2)
        # Funding/transfer arrow.
        d.line((360, 214, 484, 214), fill=(*gold, 240), width=6)
        d.polygon([(484, 214), (454, 194), (454, 234)], fill=(*gold, 240))
        # Coins / dollar marker.
        for idx, x in enumerate((534, 586, 638)):
            y = 176 + idx * 22
            d.ellipse((x - 34, y - 34, x + 34, y + 34), fill=(*gold, 155), outline=(*gold, 235), width=3)
            d.text((x, y - 19), "$", font=_font(42), fill=(8, 13, 24, 230), anchor="ma")
        d.text((card_w // 2, 324), "RELOCATION FUNDING", font=_font(25), fill=(245, 247, 255, 220), anchor="ma")

    if kind == "coal_fire":
        d.rounded_rectangle((34, 94, card_w - 34, 298), radius=18, fill=(6, 10, 14, 190), outline=(*c1, 170), width=2)
        ground_y = 154
        d.rectangle((58, 98, card_w - 58, ground_y), fill=(28, 45, 38, 180))
        d.line((58, ground_y, card_w - 58, ground_y), fill=(*gold, 230), width=4)
        layers = [
            (ground_y, ground_y + 40, (112, 88, 54, 220)),
            (ground_y + 40, ground_y + 78, (72, 54, 42, 230)),
            (ground_y + 78, 284, (27, 27, 32, 240)),
        ]
        for y0, y1, fill in layers:
            d.rectangle((58, y0, card_w - 58, y1), fill=fill)
        for x in (182, 338, 494):
            d.line((x, ground_y + 6, x - 18, 126), fill=(230, 230, 230, 90), width=5)
            d.line((x - 8, 124, x + 14, 108), fill=(230, 230, 230, 65), width=4)
        flame_pts = [(236, 272), (272, 202), (306, 272), (340, 214), (382, 272)]
        d.polygon(flame_pts, fill=(220, 60, 40, 210))
        d.polygon([(272, 272), (302, 226), (330, 272)], fill=(*gold, 225))
        d.text((card_w // 2, 324), "UNDERGROUND COAL FIRE", font=_font(25), fill=(245, 247, 255, 220), anchor="ma")

    if kind == "postal":
        d.rounded_rectangle((52, 112, card_w - 52, 292), radius=18, fill=(245, 247, 255, 225), outline=(*c1, 185), width=3)
        d.line((62, 122, card_w // 2, 220), fill=(*c1, 180), width=4)
        d.line((card_w - 62, 122, card_w // 2, 220), fill=(*c1, 180), width=4)
        d.line((62, 282, 264, 206), fill=(*c2, 170), width=3)
        d.line((card_w - 62, 282, 436, 206), fill=(*c2, 170), width=3)
        d.rounded_rectangle((96, 134, 224, 174), radius=8, fill=(*gold, 210))
        d.text((card_w // 2, 324), "POSTAL RECORD", font=_font(25), fill=(245, 247, 255, 220), anchor="ma")

    if kind in {"spores_burial", "microscope"}:
        # Left: microscope/spore field.
        d.rounded_rectangle((26, 96, 322, 296), radius=18, fill=(2, 8, 16, 185), outline=(*c1, 180), width=2)
        d.ellipse((58, 124, 290, 264), outline=(*c2, 170), width=3)
        spores = [
            (104, 178, 26), (158, 150, 18), (211, 194, 22),
            (242, 158, 14), (136, 222, 16), (82, 210, 13),
        ]
        for sx, sy, r in spores:
            d.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(*gold, 90), outline=(*gold, 220), width=2)
            d.arc((sx - r + 5, sy - r + 5, sx + r - 5, sy + r - 5), 20, 230, fill=(255, 255, 255, 120), width=2)
        d.text((44, 314), "MICROSCOPIC SPORES", font=_font(22), fill=(245, 247, 255, 210))

    if kind in {"spores_burial", "burial"}:
        # Right: burial pit cross-section.
        left = 360 if kind == "spores_burial" else 86
        right = card_w - 34 if kind == "spores_burial" else card_w - 86
        d.rounded_rectangle((left, 96, right, 296), radius=18, fill=(6, 10, 14, 188), outline=(*gold, 185), width=2)
        ground_y = 154
        d.polygon(
            [(left + 18, ground_y), (right - 18, ground_y), (right - 42, 268), (left + 54, 268)],
            fill=(96, 73, 45, 210),
            outline=(180, 137, 72, 210),
        )
        d.line((left + 18, ground_y, right - 18, ground_y), fill=(*gold, 230), width=4)
        pit = (left + 92, 184, right - 86, 246)
        d.rounded_rectangle(pit, radius=12, fill=(20, 23, 28, 230), outline=(245, 247, 255, 120), width=2)
        for x in range(pit[0] + 22, pit[2] - 10, 38):
            d.line((x, pit[1] + 10, x + 18, pit[3] - 10), fill=(245, 247, 255, 70), width=2)
        d.text((left + 22, 314), "SHALLOW BURIAL PIT", font=_font(22), fill=(245, 247, 255, 210))

    if kind == "timeline":
        y = card_h // 2
        d.line((70, y, card_w - 70, y), fill=(*gold, 230), width=5)
        for idx, x in enumerate((110, 260, 430, 590)):
            d.ellipse((x - 18, y - 18, x + 18, y + 18), fill=(*c1, 255), outline=(*gold, 255), width=3)
            d.text((x, y + 38), str(idx + 1), font=_font(26), fill=(245, 247, 255, 220), anchor="ma")
        d.text((card_w // 2, 314), "EVENT TIMELINE", font=_font(26), fill=(245, 247, 255, 220), anchor="ma")

    if kind == "generic_diagram":
        d.rounded_rectangle(
            (36, 96, card_w - 36, card_h - 56), radius=18,
            fill=(6, 10, 14, 188), outline=(255, 255, 255, 40), width=2,
        )
        bar_labels = ["A", "B", "C", "D"]
        bar_widths = [0.85, 0.55, 0.70, 0.40]
        bar_y = 130
        bar_h = 36
        bar_gap = 12
        max_bar_w = card_w - 160
        for i, (label, frac) in enumerate(zip(bar_labels, bar_widths)):
            y0 = bar_y + i * (bar_h + bar_gap)
            bw = int(max_bar_w * frac)
            bar_color = c1 if i % 2 == 0 else c2
            d.rounded_rectangle(
                (80, y0, 80 + bw, y0 + bar_h), radius=8,
                fill=(*bar_color, 200),
            )
            d.text((60, y0 + bar_h // 2), label, font=_font(22),
                   fill=(245, 247, 255, 200), anchor="ma")
        d.line((80, bar_y - 10, 80, bar_y + 4 * (bar_h + bar_gap)),
               fill=(*gold, 160), width=2)
        caption_text = _dyn_context_caption(spec, synthetic=True)
        d.text(
            (card_w // 2, card_h - 80),
            caption_text[:40].upper(),
            font=_font(22), fill=(245, 247, 255, 210), anchor="ma",
        )

    d.rectangle((0, 0, card_w, card_h), outline=(255, 255, 255, 18), width=2)
    return bg


def _dyn_type_panel(spec: Dict[str, Any]) -> Image.Image:
    """Render the 856×645 data panel as an RGBA layer on OUT_W×OUT_H canvas."""
    c1 = _hex_to_rgb(spec["colors"][0])
    c2 = _hex_to_rgb(spec["colors"][1])
    bg_r, bg_g, bg_b = _hex_to_rgb(BG_NAVY)
    px, py, pw, ph = 112, 662, 856, 645

    layer = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))

    shadow = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (px + 8, py + 10, px + pw + 8, py + ph + 10), radius=26, fill=(0, 0, 0, 128),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    layer = Image.alpha_composite(layer, shadow)

    d = ImageDraw.Draw(layer, "RGBA")
    d.rounded_rectangle(
        (px, py, px + pw, py + ph), radius=26,
        fill=(bg_r, bg_g, bg_b, 216),
        outline=(255, 255, 255, 28), width=2,
    )

    chip_w = min(300, 120 + len(spec["category"]) * 13)
    d.rounded_rectangle(
        (px + 32, py + 32, px + 32 + chip_w, py + 80), radius=15, fill=(*c1, 255),
    )
    d.text((px + 50, py + 44), spec["category"], fill=(245, 247, 255, 255), font=_font(27))
    d.text((px + pw - 42, py + 44), spec["badge"], fill=(*c2, 255), font=_font(27), anchor="ra")
    d.line((px + 32, py + 108, px + pw - 32, py + 108), fill=(255, 255, 255, 36), width=2)

    layout = spec["layout"]
    headline = spec["headline"]
    number = spec["number"]
    label = spec["label"]
    support = spec.get("support") or ""
    acc_rgb = _hex_to_rgb(ACCENT_YELLOW)

    if layout == "statement" or not number:
        d.text(
            (px + pw // 2, py + 230), headline,
            fill=(245, 247, 255, 255), font=_fit_font(headline, _font(66), pw - 100, 36),
            anchor="mm",
        )
        if label:
            lw = min(640, 200 + len(label) * 16)
            d.rounded_rectangle(
                (px + (pw - lw) // 2, py + 350, px + (pw + lw) // 2, py + 422),
                radius=22, fill=(255, 255, 255, 22), outline=(*c2, 255), width=3,
            )
            d.text(
                (px + pw // 2, py + 386), label,
                fill=(245, 247, 255, 255), font=_fit_font(label, _font(38), lw - 40, 20),
                anchor="mm",
            )
        support_y = py + 510
    elif layout == "year":
        d.text(
            (px + pw // 2, py + 168), headline,
            fill=(245, 247, 255, 255), font=_fit_font(headline, _font(50), pw - 100, 32),
            anchor="mm",
        )
        d.text(
            (px + pw // 2, py + 318), number,
            fill=(*acc_rgb, 255), font=_fit_font(number, _font(128), pw - 120, 60),
            anchor="mm",
        )
        lw = min(640, 200 + len(label) * 16)
        d.rounded_rectangle(
            (px + (pw - lw) // 2, py + 410, px + (pw + lw) // 2, py + 480),
            radius=22, fill=(255, 255, 255, 22), outline=(*c2, 255), width=3,
        )
        d.text(
            (px + pw // 2, py + 445), label,
            fill=(245, 247, 255, 255), font=_fit_font(label, _font(38), lw - 40, 20),
            anchor="mm",
        )
        support_y = py + 546
    elif layout == "comparison":
        d.text(
            (px + pw // 2, py + 166), headline,
            fill=(245, 247, 255, 255), font=_fit_font(headline, _font(50), pw - 100, 32),
            anchor="mm",
        )
        d.text(
            (px + pw // 2, py + 316), number,
            fill=(*acc_rgb, 255), font=_fit_font(number, _font(146), pw - 120, 60),
            anchor="mm",
        )
        lw = min(640, 200 + len(label) * 16)
        d.rounded_rectangle(
            (px + (pw - lw) // 2, py + 424, px + (pw + lw) // 2, py + 492),
            radius=22, fill=(255, 255, 255, 22), outline=(*c2, 255), width=3,
        )
        d.text(
            (px + pw // 2, py + 458), label,
            fill=(245, 247, 255, 255), font=_fit_font(label, _font(38), lw - 40, 20),
            anchor="mm",
        )
        support_y = py + 554
    else:  # stat
        d.text(
            (px + pw // 2, py + 166), headline,
            fill=(245, 247, 255, 255), font=_fit_font(headline, _font(50), pw - 100, 32),
            anchor="mm",
        )
        num_size = 122 if len(number) >= 6 else 142
        d.text(
            (px + pw // 2, py + 312), number,
            fill=(*acc_rgb, 255), font=_fit_font(number, _font(num_size), pw - 120, 60),
            anchor="mm",
        )
        lw = min(670, 200 + len(label) * 16)
        d.rounded_rectangle(
            (px + (pw - lw) // 2, py + 428, px + (pw + lw) // 2, py + 498),
            radius=22, fill=(255, 255, 255, 22), outline=(*c2, 255), width=3,
        )
        d.text(
            (px + pw // 2, py + 463), label,
            fill=(245, 247, 255, 255), font=_fit_font(label, _font(38), lw - 40, 20),
            anchor="mm",
        )
        support_y = py + 556

    if support:
        dummy = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        sup_lines = _fit_multiline(dummy, support, _font(30), pw - 110, max_lines=3)
        muted_rgb = _hex_to_rgb(MUTED)
        for idx, line in enumerate(sup_lines):
            d.text(
                (px + pw // 2, support_y + idx * 42), line,
                fill=(*muted_rgb, 230), font=_font(30), anchor="mm",
            )

    return layer


def _dyn_color_defaults(layout: str) -> List[str]:
    return {
        "stat":       ["#2f80ed", "#FFD700", "#f5f7ff"],
        "year":       ["#c8102e", "#fdb515", "#f5f7ff"],
        "comparison": ["#00843d", "#ffcd00", "#f5f7ff"],
        "statement":  ["#38c8ff", "#55d17a", "#f5f7ff"],
    }.get(layout, ["#2f80ed", "#FFD700", "#f5f7ff"])


def _dyn_extract_spec(beat: dict) -> Dict[str, Any]:
    overlay = beat.get("overlay") or {}
    data = overlay.get("data") or {}
    visual = beat.get("visual") or {}
    geo = str(visual.get("geo") or "").strip()
    visual_subject = str(visual.get("subject") or "").strip()
    caption = str(beat.get("caption_text") or "").strip()
    narration = str(beat.get("narration") or "").strip()
    layout = str(data.get("layout") or "stat").lower()
    colors_raw = data.get("colors") or []
    colors = [str(c) for c in colors_raw[:3]] if len(colors_raw) >= 3 else _dyn_color_defaults(layout)
    return {
        "category": str(data.get("category") or geo or "URBAN ATLAS").upper(),
        "badge":    str(data.get("badge") or "FACT").upper(),
        "layout":   layout,
        "headline": str(data.get("headline") or caption or narration[:60]).upper(),
        "number":   str(data.get("number") or "").strip(),
        "label":    str(data.get("label") or "").upper(),
        "support":  str(data.get("support") or "").strip(),
        "context_visual": str(data.get("context_visual") or visual_subject or caption or narration[:80]).strip(),
        "context_credit": str(data.get("context_credit") or "").strip(),
        "colors":   colors,
        "geo":      geo,
    }


def _render_dynamic_infographic(
    beat: dict,
    out_path: Path,
    sources_for_all_beats: Optional[List[Dict[str, Any]]] = None,
    source_index: Optional[int] = None,
) -> bool:
    frame_count = _frame_count(beat.get("duration_sec"))
    duration = frame_count / FPS
    spec = _dyn_extract_spec(beat)

    beat_id = beat.get("beat_id")
    beat_key = f"{beat_id:02d}" if isinstance(beat_id, int) else "unknown"
    if _dyn_context_concept_kind(spec):
        context_image_path = None
        print(
            f"[context-card] using generated concept diagram beat={beat_id} "
            f"context={spec.get('context_visual')!r}",
            flush=True,
        )
    else:
        context_image_path = fetch_context_card_image(
            beat,
            spec,
            out_path.parent / f"context_{beat_key}.jpg",
            archive_only=True,
        )
    bg_full = _build_infographic_bg(
        beat,
        spec,
        out_path,
        sources_for_all_beats,
        source_index,
        context_image_path=context_image_path,
        map_blur=22,
        map_navy_blend=0.58,
    )
    # Lighter version for context card so geography is recognisable
    bg_light = _infographic_bg(spec["geo"], blur=8, navy_blend=0.22) if spec["geo"] else bg_full

    # Motion timing
    clear_tail = min(18, max(10, int(0.5 * FPS)))
    active_end = max(30, frame_count - clear_tail)
    start = 5
    enter = max(12, min(22, int(active_end * 0.18)))
    hold = active_end - start - enter * 2
    if hold < 20:
        hold = 20
        enter = max(10, (active_end - start - hold) // 2)
    exit_f = enter

    # Deterministic variety based on beat_id
    # Patterns: 0: L/R, 1: R/L, 2: L/L, 3: R/R
    bid = beat.get("beat_id") or 0
    patterns = [
        ("left", "right", "right", "left"),  # Opposing 1
        ("right", "left", "left", "right"),  # Opposing 2
        ("left", "right", "left", "right"),  # Both from left
        ("right", "left", "right", "left"),  # Both from right
    ]
    c_dir, c_exit, p_dir, p_exit = patterns[bid % 4]

    card_layer = _dyn_context_card(bg_light, spec, context_image_path)
    panel_layer = _dyn_type_panel(spec)
    veil = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 96))

    def _frames() -> Iterator[Image.Image]:
        for idx in range(frame_count):
            if idx < active_end:
                base = Image.alpha_composite(bg_full.convert("RGBA"), veil)
                card_off = _dyn_group_offset(c_dir, c_exit, idx, start, enter, hold, exit_f)
                if card_off is not None:
                    base = Image.alpha_composite(base, _dyn_translate(card_layer, *card_off))
                panel_off = _dyn_group_offset(p_dir, p_exit, idx, start, enter, hold, exit_f)
                if panel_off is not None:
                    base = Image.alpha_composite(base, _dyn_translate(panel_layer, *panel_off))
                base = Image.alpha_composite(base, _vignette_layer())
                yield base.convert("RGB")
            else:
                yield bg_light.convert("RGB")

    try:
        return _write_video(_frames(), out_path, frame_count, duration)
    except Exception as exc:
        print(f"[cinematics] dynamic_infographic failed: {exc}", flush=True)
        frames_fb = _ken_burns_still_frames(bg_full, frame_count, zoom_start=1.0, zoom_end=1.04)
        return _write_video(frames_fb, out_path, frame_count, duration)


# ---------------------------------------------------------------------------
# Infographic system: classifier + blurred-map background + 4 animated renderers


def _infographic_backdrop_spec(beat: dict, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Build a subject-leading still-image spec for blurred infographic backdrops."""
    visual = beat.get("visual") if isinstance(beat.get("visual"), dict) else {}
    subject = str(
        spec.get("context_visual")
        or visual.get("subject")
        or beat.get("caption_text")
        or beat.get("narration")
        or ""
    ).strip()
    geo = str(spec.get("geo") or visual.get("geo") or "").strip()
    thematic = _thematic_backdrop_query(subject)
    context_visual = thematic or subject
    return {
        **spec,
        "context_visual": context_visual,
        "headline": str(spec.get("headline") or beat.get("caption_text") or subject).strip(),
        "category": str(spec.get("category") or "").strip(),
        "geo": geo,
    }


def _thematic_backdrop_query(subject: str) -> str:
    """Translate abstract infographic subjects into stock-photo-friendly terms."""
    text = (subject or "").lower()
    if re.search(r"\b(zip|postal|post office|mailbox|mail carrier)\b", text):
        return "abandoned post office mailbox"
    if re.search(r"\b(coal|mine|mining|strip mine|shaft)\b", text):
        return "abandoned coal mine landscape"
    if re.search(r"\b(fire|smoke|burning|landfill)\b", text):
        return "smoky abandoned landscape"
    if re.search(r"\b(row house|rowhouse|relocation|demolished|demolition)\b", text):
        return "abandoned row houses"
    if re.search(r"\b(population|density|census|decline|evacuation)\b", text):
        return "empty street abandoned town"
    cleaned = re.sub(
        r"\b(infographic|showcasing|showing|chart|graph|diagram|illustration|image|of|the|a|an)\b",
        " ",
        subject or "",
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
    return cleaned[:90]


def _source_clip_path(src: Optional[Dict[str, Any]]) -> Optional[Path]:
    if not src:
        return None
    winner = src.get("winner") or {}
    if winner.get("source") not in {"youtube", "pexels", "pixabay"}:
        return None
    raw = src.get("clip_path")
    if not raw:
        return None
    path = Path(raw)
    return path if path.exists() else None


def _video_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    try:
        got = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=10)
        return max(0.1, float(got.strip()))
    except Exception:
        return 5.0


def _extract_mid_video_frame(path: Path) -> Optional[Image.Image]:
    import uuid

    tmp = Path(tempfile.gettempdir()) / f"v2infobg_{uuid.uuid4().hex}.jpg"
    at_sec = max(0.1, _video_duration(path) / 2.0)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{at_sec:.2f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        "-f",
        "image2",
        str(tmp),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if r.returncode != 0 or not tmp.exists():
            return None
        with Image.open(tmp) as im:
            return im.convert("RGB").copy()
    except Exception:
        return None
    finally:
        for _ in range(3):
            try:
                tmp.unlink(missing_ok=True)
                break
            except PermissionError:
                import time as _t; _t.sleep(0.1)


def _blur_and_tint_image(image: Image.Image, blur: int, navy_blend: float) -> Image.Image:
    img = ImageOps.fit(
        image.convert("RGB"),
        (OUT_W, OUT_H),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    img = img.filter(ImageFilter.GaussianBlur(blur))
    dark = Image.new("RGB", (OUT_W, OUT_H), BG_NAVY)
    img = Image.blend(img, dark, navy_blend)
    return Image.alpha_composite(img.convert("RGBA"), _vignette_layer()).convert("RGB")


def _build_infographic_bg(
    beat: dict,
    spec: Dict[str, Any],
    out_path: Path,
    sources_for_all_beats: Optional[List[Dict[str, Any]]] = None,
    source_index: Optional[int] = None,
    *,
    context_image_path: Optional[Path] = None,
    map_blur: int = 22,
    map_navy_blend: float = 0.58,
) -> Image.Image:
    beat_id = beat.get("beat_id", "?")
    backdrop_spec = _infographic_backdrop_spec(beat, spec)
    geo = str(backdrop_spec.get("geo") or (beat.get("visual") or {}).get("geo") or "")

    sources = sources_for_all_beats or []
    if INFOGRAPHIC_BG_FROM_BROLL:
        own_src = None
        if source_index is not None and 0 <= source_index < len(sources):
            own_src = sources[source_index]
        own_clip = _source_clip_path(own_src)
        if own_clip:
            frame = _extract_mid_video_frame(own_clip)
            if frame is not None:
                print(f"[infographic-bg] using own_broll beat={beat_id} path={own_clip}", flush=True)
                return _blur_and_tint_image(frame, blur=26, navy_blend=0.55)
    else:
        print(f"[infographic-bg] broll disabled beat={beat_id}; trying subject backdrop before mapbox", flush=True)

    if context_image_path and context_image_path.exists():
        try:
            with Image.open(context_image_path) as im:
                print(f"[infographic-bg] using context_image beat={beat_id} path={context_image_path}", flush=True)
                return _blur_and_tint_image(im.convert("RGB"), blur=40, navy_blend=0.60)
        except Exception:
            pass

    beat_key = f"{beat_id:02d}" if isinstance(beat_id, int) else "unknown"
    subject_path = fetch_context_backdrop_image(
        beat,
        backdrop_spec,
        out_path.parent / f"subject_bg_{beat_key}.jpg",
    )
    if subject_path and subject_path.exists():
        try:
            with Image.open(subject_path) as im:
                print(
                    f"[infographic-bg] using subject_backdrop beat={beat_id} "
                    f"query={backdrop_spec.get('context_visual')!r} path={subject_path}",
                    flush=True,
                )
                return _blur_and_tint_image(im.convert("RGB"), blur=40, navy_blend=0.60)
        except Exception:
            pass

    if INFOGRAPHIC_BG_FROM_BROLL:
        for idx, src in enumerate(sources):
            if source_index is not None and idx == source_index:
                continue
            clip = _source_clip_path(src)
            if not clip:
                continue
            frame = _extract_mid_video_frame(clip)
            if frame is not None:
                print(f"[infographic-bg] using neighbor_broll beat={beat_id} source_beat={idx + 1} path={clip}", flush=True)
                return _blur_and_tint_image(frame, blur=32, navy_blend=0.60)

    print(f"[infographic-bg] using mapbox beat={beat_id} reason=no_broll_or_context", flush=True)
    return _infographic_bg(geo, blur=map_blur, navy_blend=map_navy_blend)


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ---------------------------------------------------------------------------
# Depth effects — vignette, film grain, drop shadow (lazy-init, cached at module level)

_VIGNETTE_CACHE: Optional[Image.Image] = None
_GRAIN_CACHE: Optional[Image.Image] = None


def _vignette_layer() -> Image.Image:
    global _VIGNETTE_CACHE
    if _VIGNETTE_CACHE is None:
        mask = Image.new("L", (OUT_W, OUT_H), 0)
        d = ImageDraw.Draw(mask)
        d.ellipse((OUT_W // 6, OUT_H // 7, OUT_W * 5 // 6, OUT_H * 6 // 7), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=OUT_W // 3))
        mask = mask.point(lambda p: int((255 - p) * 0.50))
        _VIGNETTE_CACHE = Image.merge("RGBA", (
            Image.new("L", (OUT_W, OUT_H), 0),
            Image.new("L", (OUT_W, OUT_H), 0),
            Image.new("L", (OUT_W, OUT_H), 0),
            mask,
        ))
    return _VIGNETTE_CACHE


def _grain_layer() -> Image.Image:
    global _GRAIN_CACHE
    if _GRAIN_CACHE is None:
        import random as _r
        rng = _r.Random(42)
        sw, sh = OUT_W // 3, OUT_H // 3
        tile = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
        pix = tile.load()
        for y in range(sh):
            for x in range(sw):
                if rng.random() < 0.22:
                    v = rng.randint(160, 255)
                    a = rng.randint(0, 10)
                    pix[x, y] = (v, v, v, a)
        _GRAIN_CACHE = tile.resize((OUT_W, OUT_H), Image.Resampling.NEAREST)
    return _GRAIN_CACHE


def _apply_depth_fx(img: Image.Image) -> Image.Image:
    """Apply vignette + grain over an RGB image. Returns RGB."""
    rgba = Image.alpha_composite(img.convert("RGBA"), _vignette_layer())
    return Image.alpha_composite(rgba, _grain_layer()).convert("RGB")


def _card_shadow(frame: Image.Image, xy: Tuple[int, int, int, int], radius: int) -> Image.Image:
    """Blurred drop shadow behind a card. Returns updated RGB."""
    x0, y0, x1, y1 = xy
    sh = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle(
        (x0 + 7, y0 + 9, x1 + 7, y1 + 9), radius=radius, fill=(0, 0, 0, 95)
    )
    sh = sh.filter(ImageFilter.GaussianBlur(11))
    return Image.alpha_composite(frame.convert("RGBA"), sh).convert("RGB")


def _draw_bar_sheen(
    frame: Image.Image,
    x0: int, y0: int, x1: int, y1: int,
    radius: int,
    rgb: Tuple[int, int, int],
    alpha: int,
) -> Image.Image:
    """Rounded bar with a subtle top-sheen gradient. Returns updated RGB."""
    r, g, b = rgb
    bl = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bl)
    bd.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=(r, g, b, alpha))
    # White highlight in upper 40%
    sheen_h = max(4, int((y1 - y0) * 0.42))
    sh = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.rounded_rectangle(
        (x0 + 2, y0 + 2, x1 - 2, y0 + sheen_h),
        radius=max(2, radius - 2),
        fill=(255, 255, 255, 28),
    )
    bl = Image.alpha_composite(bl, sh)
    return Image.alpha_composite(frame.convert("RGBA"), bl).convert("RGB")


def _count_up_num(numeric_val: float, source_str: str, progress: float) -> str:
    """Format a count-up value to match the source number's format."""
    val = numeric_val * progress
    if ',' in source_str:
        return f"{val:,.0f}"
    if numeric_val >= 1_000_000:
        return f"{val / 1_000_000:.1f}M"
    if numeric_val >= 10_000:
        return f"{val / 1_000:.0f}K"
    if '.' in source_str:
        dec = max(1, len(source_str.rstrip('0').split('.')[-1]))
        return f"{val:.{dec}f}"
    return str(int(round(val)))


def _classify_infographic(beat: dict) -> str:
    """Route to: 'stat_card' | 'heatmap' | 'bar_chart' | 'border_shift'."""
    subject = str((beat.get("visual") or {}).get("subject") or "").lower()
    narration = str(beat.get("narration") or "").lower()
    text = f"{subject} {narration}"

    if any(w in text for w in (
        "heat map", "heatmap", "heat-map", "gang control", "spreading",
        "density cluster", "population cluster", "choropleth", "hot spot",
    )):
        return "heatmap"
    if any(w in text for w in (
        "border", "territory", "pre-1879", "pre-war", "annexed",
        "lost land", "war of the pacific", "border shift", "before and after map",
        "split-screen", "1879", "ceded",
    )):
        return "border_shift"
    if any(w in text for w in (
        "bar chart", "bar graph", "ranking", "ranked", "47th", "versus",
        "compared to", "bar race", "rising cost", "growing cost",
        "comparison", "compared with",
    )):
        return "bar_chart"
    return "stat_card"


def _infographic_bg(geo_text: str, blur: int = 30, navy_blend: float = 0.65) -> Image.Image:
    """Return a blurred satellite map crop, or a dark grid fallback."""
    if geo_text and MAPBOX_TOKEN:
        try:
            geo = _geocode(geo_text)
            if geo:
                zoom = _zoom_for_bbox(geo.bbox, geo_text)
                canvas = _stitched_map(geo.lat, geo.lon, zoom)
                if canvas:
                    cw, ch = canvas.size
                    aspect = OUT_W / OUT_H
                    crop_h = min(ch, cw / aspect)
                    crop_w = crop_h * aspect
                    x0 = int((cw - crop_w) / 2)
                    y0 = int((ch - crop_h) / 2)
                    img = canvas.crop((x0, y0, int(x0 + crop_w), int(y0 + crop_h)))
                    img = img.resize((OUT_W, OUT_H), Image.Resampling.LANCZOS)
                    img = img.filter(ImageFilter.GaussianBlur(blur))
                    dark = Image.new("RGB", (OUT_W, OUT_H), BG_NAVY)
                    img = Image.blend(img, dark, navy_blend)
                    return Image.alpha_composite(img.convert("RGBA"), _vignette_layer()).convert("RGB")
        except Exception:
            pass
    img = Image.new("RGB", (OUT_W, OUT_H), BG_NAVY)
    _draw_background_grid(ImageDraw.Draw(img))
    return img


def _infographic_subject(beat: dict) -> str:
    visual = beat.get("visual") or {}
    caption = str(beat.get("caption_text") or "").strip()
    raw = str(visual.get("subject") or "").strip()
    is_alttext = bool(re.match(
        r"^(infographic|chart|graph|diagram|illustration|image)\b", raw, re.IGNORECASE
    ))
    return caption or (raw if not is_alttext else str(beat.get("narration") or "KEY STAT").strip())


def _fmt_stat(num_str: str, suffix: str) -> str:
    """Join number and suffix with correct spacing: '80%', '47th', '38,000 ACRES'."""
    if not suffix:
        return num_str.upper()
    s = suffix.strip()
    # Symbols and ordinals attach directly; words get a space
    if s[:1] in ('%', '°', "'", '"') or s.lower()[:2] in ('th', 'st', 'nd', 'rd'):
        return (num_str + s).upper()
    return (num_str + ' ' + s).upper()


def _infographic_primary(beat: dict) -> Tuple[str, str, str]:
    """Returns (number_str, suffix_str, label_str)."""
    data = ((beat.get("overlay") or {}).get("data") or {})
    number = str(data.get("number") or "").strip()
    suffix = str(data.get("suffix") or "").strip()
    label = str(data.get("label") or "KEY FIGURE").strip()
    if not number:
        facts = _extract_numbers(beat, "")
        if facts:
            number, label = facts[0]
    return number, suffix, label


def _infographic_stat_card(
    beat: dict, bg: Image.Image, frame_count: int
) -> Iterator[Image.Image]:
    """Animated stat card: drop shadow card + count-up number + tracked label + grain."""
    subject = _infographic_subject(beat)
    number, suffix, label = _infographic_primary(beat)
    numeric_val = _number_value(number) if number else None

    title_font = _font(62)
    number_font = _font(172)
    label_font = _font(44)
    micro_font = _font(24)
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    title_lines = _fit_multiline(dummy_draw, subject.upper(), title_font, OUT_W - 140, max_lines=3)

    CARD_TOP, CARD_BOT = 490, 1150
    CARD_XY = (70, CARD_TOP, OUT_W - 70, CARD_BOT)
    center_y = CARD_TOP + int((CARD_BOT - CARD_TOP) * 0.42)

    for idx in range(frame_count):
        t = idx / max(1, frame_count - 1)
        frame = bg.copy()

        # Drop shadow + card background
        card_t = _ease_in_out_cubic(min(1.0, max(0.0, (t - 0.18) / 0.26)))
        if card_t > 0:
            frame = _card_shadow(frame, CARD_XY, 18)
            cl = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
            cd = ImageDraw.Draw(cl)
            cr, cg, cb = _hex_to_rgb(BG_NAVY_LIGHT)
            cd.rounded_rectangle(CARD_XY, radius=18, fill=(cr, cg, cb, int(card_t * 218)))
            if card_t >= 0.88:
                cd.rounded_rectangle(CARD_XY, radius=18, outline=(255, 215, 0, 255), width=3)
            frame = Image.alpha_composite(frame.convert("RGBA"), cl).convert("RGB")

        draw = ImageDraw.Draw(frame)

        # Yellow accent bar wipes across
        bar_w = int(OUT_W * _ease_in_out_cubic(min(1.0, t / 0.08)))
        draw.rectangle((0, 0, bar_w, 18), fill=ACCENT_YELLOW)

        # Brand label + title slide down from top
        title_t = _ease_in_out_cubic(min(1.0, max(0.0, (t - 0.05) / 0.26)))
        y_off = int((1.0 - title_t) * -210)
        draw.text((70, 108 + y_off), "URBAN ATLAS", font=micro_font, fill=ACCENT_YELLOW,
                  stroke_width=1, stroke_fill=(0, 0, 0))
        y = 166 + y_off
        for line in title_lines:
            draw.text((70, y), line, font=title_font, fill=WHITE,
                      stroke_width=3, stroke_fill=(0, 0, 14))
            y += 74

        # Number counts up (28% → 68%), then holds
        num_raw = min(1.0, max(0.0, (t - 0.28) / 0.40))
        num_t = _ease_in_out_cubic(num_raw)
        if num_t > 0:
            if numeric_val is not None and num_raw < 1.0:
                display_num = _count_up_num(numeric_val, number, num_t)
            else:
                display_num = number or "?"
            disp = _fmt_stat(display_num, suffix)
            nf = _fit_font(disp, number_font, OUT_W - 210, min_size=92)
            draw.text((OUT_W // 2, center_y), disp, font=nf, fill=ACCENT_YELLOW,
                      anchor="mm", stroke_width=5, stroke_fill=(0, 0, 0))

        # Label fades + slides up once number is settled (55%)
        if t >= 0.54:
            lab_t = _ease_in_out_cubic(min(1.0, (t - 0.54) / 0.24))
            lab = _fit_text(label.upper(), label_font, OUT_W - 220)
            lab_y = center_y + 152 + int((1.0 - lab_t) * 30)
            mr, mg, mb = _hex_to_rgb(MUTED)
            ll = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
            ImageDraw.Draw(ll).text(
                (OUT_W // 2, lab_y), lab, font=label_font,
                fill=(mr, mg, mb, int(lab_t * 255)), anchor="mm",
            )
            frame = Image.alpha_composite(frame.convert("RGBA"), ll).convert("RGB")
            draw = ImageDraw.Draw(frame)

        # Bottom yellow bar grows in
        if t >= 0.74:
            bot_w = int((OUT_W - 140) * _ease_in_out_cubic(min(1.0, (t - 0.74) / 0.18)))
            draw.rectangle((70, OUT_H - 68, 70 + bot_w, OUT_H - 58), fill=ACCENT_YELLOW)

        yield _apply_depth_fx(frame)


def _infographic_heatmap(
    beat: dict, bg: Image.Image, frame_count: int
) -> Iterator[Image.Image]:
    """Glowing radial hotspots expand over blurred map bg + stat overlay."""
    subject = _infographic_subject(beat)
    number, suffix, label = _infographic_primary(beat)
    numeric_val = _number_value(number) if number else None

    import hashlib, random as _rnd
    seed = int(hashlib.md5(subject.encode()).hexdigest()[:8], 16)
    rng = _rnd.Random(seed)

    # Gaussian-distributed hotspots: clustered toward center where the city is,
    # with a few outliers — mirrors how population/event density actually looks.
    cx_img, cy_img = OUT_W // 2, int(OUT_H * 0.47)
    sx, sy = int(OUT_W * 0.20), int(OUT_H * 0.16)
    hotspots = []
    for _ in range(4):
        hx = int(rng.gauss(cx_img, sx))
        hy = int(rng.gauss(cy_img, sy))
        hx = max(120, min(OUT_W - 120, hx))
        hy = max(int(OUT_H * 0.22), min(int(OUT_H * 0.72), hy))
        hotspots.append((hx, hy))
    heat_colors = [(220, 35, 35), (255, 100, 10), (210, 30, 70), (200, 60, 20)]
    max_radii = [rng.randint(260, 360) for _ in range(4)]

    title_font = _font(62)
    number_font = _font(150)
    label_font = _font(48)
    micro_font = _font(24)
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    title_lines = _fit_multiline(dummy_draw, subject.upper(), title_font, OUT_W - 140, max_lines=3)

    CARD_TOP = OUT_H - 580

    for idx in range(frame_count):
        t = idx / max(1, frame_count - 1)
        frame = bg.copy()

        # Heat circles grow in with stagger
        heat_layer = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
        hd = ImageDraw.Draw(heat_layer)
        for i, ((hx, hy), color, max_r) in enumerate(zip(hotspots, heat_colors, max_radii)):
            delay = i * 0.11
            ht = _ease_in_out_cubic(min(1.0, max(0.0, (t - delay) / max(0.1, 0.58 - delay))))
            if ht <= 0:
                continue
            cur_r = int(max_r * ht)
            for r in range(cur_r, 0, -10):
                frac = 1.0 - (r / max_r)
                alpha = int(frac * frac * 150 * ht)
                hd.ellipse((hx - r, hy - r, hx + r, hy + r), fill=(*color, alpha))
            # Bright core dot
            core_r = max(8, int(32 * ht))
            hd.ellipse(
                (hx - core_r, hy - core_r, hx + core_r, hy + core_r),
                fill=(*color, int(230 * ht)),
            )
        frame = Image.alpha_composite(frame.convert("RGBA"), heat_layer).convert("RGB")

        # Stat card fades in at bottom
        stat_t = _ease_in_out_cubic(min(1.0, max(0.0, (t - 0.45) / 0.30)))
        if stat_t > 0:
            cl = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
            cd = ImageDraw.Draw(cl)
            r, g, b = _hex_to_rgb(BG_NAVY_LIGHT)
            cd.rounded_rectangle(
                (70, CARD_TOP, OUT_W - 70, OUT_H - 100), radius=18,
                fill=(r, g, b, int(stat_t * 220)),
            )
            if stat_t >= 0.9:
                cd.rounded_rectangle(
                    (70, CARD_TOP, OUT_W - 70, OUT_H - 100), radius=18,
                    outline=(255, 215, 0, 255), width=3,
                )
            frame = Image.alpha_composite(frame.convert("RGBA"), cl).convert("RGB")

        draw = ImageDraw.Draw(frame)
        bar_w = int(OUT_W * _ease_in_out_cubic(min(1.0, t / 0.06)))
        draw.rectangle((0, 0, bar_w, 18), fill=ACCENT_YELLOW)

        title_t = _ease_in_out_cubic(min(1.0, max(0.0, (t - 0.04) / 0.22)))
        y_off = int((1.0 - title_t) * -200)
        draw.text((70, 110 + y_off), "URBAN ATLAS", font=micro_font, fill=ACCENT_YELLOW)
        y = 170 + y_off
        for line in title_lines:
            draw.text((70, y), line, font=title_font, fill=WHITE,
                      stroke_width=2, stroke_fill=(0, 0, 14))
            y += 72

        if stat_t > 0 and number:
            num_t = _ease_in_out_cubic(min(1.0, max(0.0, (t - 0.55) / 0.30)))
            if num_t > 0:
                if numeric_val is not None and num_t < 1.0:
                    dv = numeric_val * num_t
                    display_num = str(int(round(dv))) if numeric_val >= 10 else f"{dv:.1f}"
                else:
                    display_num = number
                disp = _fmt_stat(display_num, suffix)
                center_y = CARD_TOP + int((OUT_H - 100 - CARD_TOP) * 0.40)
                nf = _fit_font(disp, number_font, OUT_W - 210, min_size=80)
                draw.text((OUT_W // 2, center_y), disp, font=nf, fill=ACCENT_YELLOW,
                          anchor="mm", stroke_width=4, stroke_fill=(0, 0, 0))

            lab_t = max(0.0, (t - 0.70) / 0.20)
            if lab_t > 0:
                lab = _fit_text(label.upper(), label_font, OUT_W - 220)
                center_y = CARD_TOP + int((OUT_H - 100 - CARD_TOP) * 0.40)
                r2, g2, b2 = _hex_to_rgb(WHITE)
                ll = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
                ImageDraw.Draw(ll).text(
                    (OUT_W // 2, center_y + 130), lab, font=label_font,
                    fill=(r2, g2, b2, int(min(1.0, lab_t) * 255)), anchor="mm",
                )
                frame = Image.alpha_composite(frame.convert("RGBA"), ll).convert("RGB")
                draw = ImageDraw.Draw(frame)

        if t >= 0.82:
            bot_w = int((OUT_W - 140) * _ease_in_out_cubic(min(1.0, (t - 0.82) / 0.15)))
            draw.rectangle((70, OUT_H - 60, 70 + bot_w, OUT_H - 50), fill=ACCENT_YELLOW)

        yield _apply_depth_fx(frame)


def _infographic_bar_chart(
    beat: dict, bg: Image.Image, frame_count: int
) -> Iterator[Image.Image]:
    """Bar race: gradient bars, smart label placement (inside/above/right), grain."""
    subject = _infographic_subject(beat)
    number, suffix, label = _infographic_primary(beat)

    facts = _extract_numbers(beat, number)
    bars: List[Tuple[str, str, float, str]] = []  # (label, display, value, color)
    if number:
        v = _number_value(number) or 1.0
        bars.append((label or "PRIMARY", _fmt_stat(number, suffix), v, ACCENT_YELLOW))
    for num_s, lbl in facts[:3]:
        v = _number_value(num_s) or 0.5
        bars.append((lbl, num_s.upper(), v, "#58c7ff"))

    if not bars:
        bars = [(subject[:22].upper(), "—", 1.0, ACCENT_YELLOW)]

    if len(bars) == 1:
        val0 = bars[0][2]
        if suffix.lower() in ('th', 'st', 'nd', 'rd') or (number.isdigit() and int(number) <= 100):
            # Ordinal ranking: show a rank ladder — worst → avg → best
            bars.append(("PREF. AVERAGE", "24TH", val0 * 0.50, "#58c7ff"))
            bars.append(("BEST RANKED", "#1", val0 * 0.021, "#aab2d5"))
        else:
            bars.append(("COMPARISON", "—", val0 * 0.54, "#58c7ff"))
            bars.append(("REFERENCE", "—", val0 * 0.28, "#aab2d5"))

    max_val = max(b[2] for b in bars) or 1.0
    n = len(bars)

    BAR_H = 96
    BAR_SPACING = 52
    BAR_X0, BAR_X1 = 90, OUT_W - 90
    BAR_MAX_W = BAR_X1 - BAR_X0

    # Center bar group vertically between title area and bottom margin
    group_h = n * BAR_H + (n - 1) * BAR_SPACING
    title_end_y = 420
    bottom_margin_y = OUT_H - 200
    group_start = title_end_y + max(40, (bottom_margin_y - title_end_y - group_h) // 2)

    title_font = _font(56)
    bar_label_font = _font(32)
    val_font = _font(34)
    micro_font = _font(24)
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    title_lines = _fit_multiline(dummy_draw, subject.upper(), title_font, OUT_W - 140, max_lines=3)

    # Pre-compute final widths + label placement strategy for each bar
    bar_meta = []
    for blabel, bdisplay, bval, bcolor in bars:
        final_fw = max(BAR_H, int(BAR_MAX_W * (bval / max_val)))
        lw, lh = _text_bbox(dummy_draw, blabel, bar_label_font)
        vw, vh = _text_bbox(dummy_draw, bdisplay, val_font)
        if final_fw >= lw + 32:
            lpos = "inside"
        elif final_fw >= BAR_H:
            lpos = "above"
        else:
            lpos = "right"
        bar_meta.append({
            "label": blabel, "display": bdisplay, "value": bval, "color": bcolor,
            "final_fw": final_fw, "lw": lw, "lh": lh, "vw": vw, "vh": vh, "lpos": lpos,
        })

    for idx in range(frame_count):
        t = idx / max(1, frame_count - 1)
        frame = bg.copy()

        # Ghost track bars appear first
        track_t = _ease_in_out_cubic(min(1.0, max(0.0, (t - 0.10) / 0.16)))
        if track_t > 0:
            tl = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
            td = ImageDraw.Draw(tl)
            for i in range(n):
                by = group_start + i * (BAR_H + BAR_SPACING)
                td.rounded_rectangle(
                    (BAR_X0, by, BAR_X1, by + BAR_H),
                    radius=BAR_H // 2,
                    fill=(255, 255, 255, int(track_t * 22)),
                )
            frame = Image.alpha_composite(frame.convert("RGBA"), tl).convert("RGB")

        draw = ImageDraw.Draw(frame)

        # Yellow accent bar
        acw = int(OUT_W * _ease_in_out_cubic(min(1.0, t / 0.07)))
        draw.rectangle((0, 0, acw, 18), fill=ACCENT_YELLOW)

        # Title slides in
        title_t = _ease_in_out_cubic(min(1.0, max(0.0, (t - 0.04) / 0.24)))
        y_off = int((1.0 - title_t) * -210)
        draw.text((70, 100 + y_off), "URBAN ATLAS", font=micro_font, fill=ACCENT_YELLOW,
                  stroke_width=1, stroke_fill=(0, 0, 0))
        y = 152 + y_off
        for line in title_lines:
            draw.text((70, y), line, font=title_font, fill=WHITE,
                      stroke_width=3, stroke_fill=(0, 0, 14))
            y += 68

        # Bars race in with stagger
        for i, meta in enumerate(bar_meta):
            delay = 0.17 + i * 0.13
            bt = _ease_in_out_cubic(min(1.0, max(0.0, (t - delay) / 0.36)))
            if bt <= 0:
                continue

            by = group_start + i * (BAR_H + BAR_SPACING)
            fill_w = max(8, int(meta["final_fw"] * bt))
            rgb = _hex_to_rgb(meta["color"])

            # Gradient bar
            frame = _draw_bar_sheen(frame, BAR_X0, by, BAR_X0 + fill_w, by + BAR_H,
                                    BAR_H // 2, rgb, 232)
            draw = ImageDraw.Draw(frame)

            lab = meta["label"]
            lpos = meta["lpos"]
            lh = meta["lh"]
            label_y = by + (BAR_H - lh) // 2

            if lpos == "inside" and fill_w >= meta["lw"] + 32:
                # Dark text inside the bright bar
                draw.text((BAR_X0 + 18, label_y), lab, font=bar_label_font,
                          fill=BG_NAVY)
            elif lpos == "above":
                # White text floats above the bar, fades in early
                if bt > 0.12:
                    at = min(1.0, (bt - 0.12) / 0.25)
                    al = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
                    wr, wg, wb = _hex_to_rgb(WHITE)
                    ImageDraw.Draw(al).text(
                        (BAR_X0, by - lh - 10), lab, font=bar_label_font,
                        fill=(wr, wg, wb, int(at * 230)),
                        stroke_width=2, stroke_fill=(0, 0, 0, int(at * 255)),
                    )
                    frame = Image.alpha_composite(frame.convert("RGBA"), al).convert("RGB")
                    draw = ImageDraw.Draw(frame)
            else:
                # Label to the right of the bar — always readable, no truncation
                if bt > 0.28:
                    at = min(1.0, (bt - 0.28) / 0.30)
                    al = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
                    wr, wg, wb = _hex_to_rgb(WHITE)
                    ImageDraw.Draw(al).text(
                        (BAR_X1 + 14, label_y), lab, font=bar_label_font,
                        fill=(wr, wg, wb, int(at * 230)),
                        stroke_width=2, stroke_fill=(0, 0, 0, int(at * 255)),
                    )
                    frame = Image.alpha_composite(frame.convert("RGBA"), al).convert("RGB")
                    draw = ImageDraw.Draw(frame)

            # Value at bar tip (fades in after bar is 60% done)
            if bt > 0.55 and meta["final_fw"] > 50:
                vt = min(1.0, (bt - 0.55) / 0.35)
                vl = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
                vx = BAR_X0 + fill_w + 10
                if vx + meta["vw"] > BAR_X1 - 4:
                    vx = max(BAR_X0 + 8, BAR_X0 + fill_w - meta["vw"] - 10)
                wr, wg, wb = _hex_to_rgb(WHITE)
                ImageDraw.Draw(vl).text(
                    (vx, label_y), meta["display"], font=val_font,
                    fill=(wr, wg, wb, int(vt * 245)),
                )
                frame = Image.alpha_composite(frame.convert("RGBA"), vl).convert("RGB")
                draw = ImageDraw.Draw(frame)

        if t >= 0.83:
            bot_w = int((OUT_W - 140) * _ease_in_out_cubic(min(1.0, (t - 0.83) / 0.14)))
            draw.rectangle((70, OUT_H - 68, 70 + bot_w, OUT_H - 58), fill=ACCENT_YELLOW)

        yield _apply_depth_fx(frame)


def _infographic_border_shift(
    beat: dict, bg: Image.Image, frame_count: int
) -> Iterator[Image.Image]:
    """Before/after territory: old area in red fades in, new area in blue wipes over it."""
    subject = _infographic_subject(beat)
    number, suffix, label = _infographic_primary(beat)
    numeric_val = _number_value(number) if number else None

    title_font = _font(62)
    number_font = _font(150)
    label_font = _font(46)
    badge_font = _font(36)
    micro_font = _font(24)
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    title_lines = _fit_multiline(dummy_draw, subject.upper(), title_font, OUT_W - 140, max_lines=3)

    CX = OUT_W // 2
    TERR_CY = int(OUT_H * 0.52)
    BEFORE_RX, BEFORE_RY = 290, 220
    AFTER_RX, AFTER_RY = 200, 155
    CARD_TOP = OUT_H - 530

    for idx in range(frame_count):
        t = idx / max(1, frame_count - 1)
        frame = bg.copy()

        # BEFORE territory (red blob)
        before_t = _ease_in_out_cubic(min(1.0, max(0.0, (t - 0.12) / 0.28)))
        if before_t > 0:
            bl = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
            bd = ImageDraw.Draw(bl)
            rx = int(BEFORE_RX * before_t)
            ry = int(BEFORE_RY * before_t)
            bd.ellipse((CX - rx, TERR_CY - ry, CX + rx, TERR_CY + ry),
                       fill=(210, 50, 50, int(before_t * 170)))
            bd.ellipse((CX - rx, TERR_CY - ry, CX + rx, TERR_CY + ry),
                       outline=(255, 80, 80, int(before_t * 255)), width=4)
            frame = Image.alpha_composite(frame.convert("RGBA"), bl).convert("RGB")

        # AFTER territory (blue blob, offset to show loss)
        after_t = _ease_in_out_cubic(min(1.0, max(0.0, (t - 0.45) / 0.32)))
        if after_t > 0:
            al = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
            ad = ImageDraw.Draw(al)
            rx = int(AFTER_RX * after_t)
            ry = int(AFTER_RY * after_t)
            offset_x = int(50 * after_t)
            ad.ellipse((CX - rx + offset_x, TERR_CY - ry,
                        CX + rx + offset_x, TERR_CY + ry),
                       fill=(70, 110, 240, int(after_t * 200)))
            ad.ellipse((CX - rx + offset_x, TERR_CY - ry,
                        CX + rx + offset_x, TERR_CY + ry),
                       outline=(130, 170, 255, int(after_t * 255)), width=4)
            frame = Image.alpha_composite(frame.convert("RGBA"), al).convert("RGB")

        draw = ImageDraw.Draw(frame)
        bar_w = int(OUT_W * _ease_in_out_cubic(min(1.0, t / 0.07)))
        draw.rectangle((0, 0, bar_w, 18), fill=ACCENT_YELLOW)

        title_t = _ease_in_out_cubic(min(1.0, max(0.0, (t - 0.04) / 0.22)))
        y_off = int((1.0 - title_t) * -200)
        draw.text((70, 110 + y_off), "URBAN ATLAS", font=micro_font, fill=ACCENT_YELLOW)
        y = 170 + y_off
        for line in title_lines:
            draw.text((70, y), line, font=title_font, fill=WHITE,
                      stroke_width=2, stroke_fill=(0, 0, 14))
            y += 72

        # BEFORE / AFTER badges — centered above/below each territory
        if before_t >= 0.7:
            bt_alpha = int(min(1.0, (before_t - 0.7) / 0.25) * 255)
            bl2 = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
            ImageDraw.Draw(bl2).text(
                (CX, TERR_CY - BEFORE_RY - 44), "BEFORE", font=badge_font,
                fill=(255, 100, 100, bt_alpha), anchor="mm",
                stroke_width=2, stroke_fill=(0, 0, 0, bt_alpha),
            )
            frame = Image.alpha_composite(frame.convert("RGBA"), bl2).convert("RGB")
            draw = ImageDraw.Draw(frame)
        if after_t >= 0.7:
            at_alpha = int(min(1.0, (after_t - 0.7) / 0.25) * 255)
            al2 = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
            ImageDraw.Draw(al2).text(
                (CX + 40, TERR_CY + AFTER_RY + 44), "AFTER", font=badge_font,
                fill=(130, 170, 255, at_alpha), anchor="mm",
                stroke_width=2, stroke_fill=(0, 0, 0, at_alpha),
            )
            frame = Image.alpha_composite(frame.convert("RGBA"), al2).convert("RGB")
            draw = ImageDraw.Draw(frame)

        # Stat card at bottom
        card_t = _ease_in_out_cubic(min(1.0, max(0.0, (t - 0.62) / 0.28)))
        if card_t > 0:
            cl = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
            cd = ImageDraw.Draw(cl)
            r, g, b = _hex_to_rgb(BG_NAVY_LIGHT)
            cd.rounded_rectangle(
                (70, CARD_TOP, OUT_W - 70, OUT_H - 100), radius=18,
                fill=(r, g, b, int(card_t * 220)),
            )
            if card_t >= 0.9:
                cd.rounded_rectangle(
                    (70, CARD_TOP, OUT_W - 70, OUT_H - 100), radius=18,
                    outline=(255, 215, 0, 255), width=3,
                )
            frame = Image.alpha_composite(frame.convert("RGBA"), cl).convert("RGB")
            draw = ImageDraw.Draw(frame)

            if number:
                num_t = _ease_in_out_cubic(min(1.0, max(0.0, (t - 0.70) / 0.25)))
                if num_t > 0:
                    if numeric_val is not None and num_t < 1.0:
                        dv = numeric_val * num_t
                        display_num = str(int(round(dv))) if numeric_val >= 10 else f"{dv:.1f}"
                    else:
                        display_num = number
                    disp = _fmt_stat(display_num, suffix)
                    center_y = CARD_TOP + int((OUT_H - 100 - CARD_TOP) * 0.40)
                    nf = _fit_font(disp, number_font, OUT_W - 210, min_size=80)
                    draw.text((OUT_W // 2, center_y), disp, font=nf, fill=ACCENT_YELLOW,
                              anchor="mm", stroke_width=4, stroke_fill=(0, 0, 0))

                lab_t = max(0.0, (t - 0.82) / 0.15)
                if lab_t > 0:
                    lab = _fit_text(label.upper(), label_font, OUT_W - 220)
                    center_y = CARD_TOP + int((OUT_H - 100 - CARD_TOP) * 0.40)
                    r2, g2, b2 = _hex_to_rgb(WHITE)
                    ll = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
                    ImageDraw.Draw(ll).text(
                        (OUT_W // 2, center_y + 120), lab, font=label_font,
                        fill=(r2, g2, b2, int(min(1.0, lab_t) * 255)), anchor="mm",
                    )
                    frame = Image.alpha_composite(frame.convert("RGBA"), ll).convert("RGB")
                    draw = ImageDraw.Draw(frame)

        if t >= 0.86:
            bot_w = int((OUT_W - 140) * _ease_in_out_cubic(min(1.0, (t - 0.86) / 0.12)))
            draw.rectangle((70, OUT_H - 70, 70 + bot_w, OUT_H - 60), fill=ACCENT_YELLOW)

        yield _apply_depth_fx(frame)


def _density_still(beat: dict) -> Image.Image:
    visual = beat.get("visual") or {}
    overlay = beat.get("overlay") or {}
    data = overlay.get("data") or {}

    # Prefer the caption_text (a real headline) over visual.subject (often an
    # alt-text style description like "Infographic showing X ranked 47th...").
    caption = str(beat.get("caption_text") or "").strip()
    raw_subject = str(visual.get("subject") or "").strip()
    subject_is_alttext = bool(re.match(
        r"^(infographic|chart|graph|diagram|illustration|image)\b",
        raw_subject, re.IGNORECASE,
    ))
    if caption:
        subject = caption
    elif raw_subject and not subject_is_alttext:
        subject = raw_subject
    else:
        subject = str(beat.get("narration") or "Key statistic").strip()
    primary_number = str(data.get("number") or "").strip()
    primary_label = str(data.get("label") or data.get("title") or "KEY FIGURE").strip()

    facts = _extract_numbers(beat, primary_number)
    if not primary_number and facts:
        primary_number, primary_label = facts.pop(0)
    if not primary_number:
        primary_number, primary_label = "?", primary_label or "KEY FIGURE"

    secondary = facts[0] if facts else None

    img = Image.new("RGB", (OUT_W, OUT_H), BG_NAVY)
    draw = ImageDraw.Draw(img)
    _draw_background_grid(draw)

    title_font = _font(62)
    number_font = _font(172)
    label_font = _font(46)
    small_font = _font(34)
    micro_font = _font(24)

    draw.rectangle((0, 0, OUT_W, 18), fill=ACCENT_YELLOW)
    draw.text((70, 110), "URBAN ATLAS", font=micro_font, fill=ACCENT_YELLOW)

    title = _fit_multiline(draw, subject.upper(), title_font, OUT_W - 140, max_lines=3)
    y = 170
    for line in title:
        draw.text((70, y), line, font=title_font, fill=WHITE)
        y += 72

    card_top = 500
    card_bottom = 1065 if secondary else 1140
    _rounded_rect(draw, (70, card_top, OUT_W - 70, card_bottom), radius=18, fill=BG_NAVY_LIGHT, outline=(255, 215, 0), width=3)

    num = primary_number.upper()
    num_font = _fit_font(num, number_font, OUT_W - 210, min_size=92)
    center_y = card_top + int((card_bottom - card_top) * 0.42)
    draw.text((OUT_W // 2, center_y), num, font=num_font, fill=ACCENT_YELLOW, anchor="mm")
    lab = primary_label.upper()
    lab = _fit_text(lab, label_font, OUT_W - 220)
    draw.text((OUT_W // 2, center_y + 148), lab, font=label_font, fill=WHITE, anchor="mm")

    if secondary:
        sec_num, sec_label = secondary
        _draw_comparison_bars(draw, (110, 1165, OUT_W - 110, 1470), (num, lab), (sec_num, sec_label))
    else:
        note = _fit_multiline(draw, _clean_sentence(str(beat.get("narration") or subject)), small_font, OUT_W - 160, max_lines=4)
        ny = 1220
        for line in note:
            w, _ = _text_bbox(draw, line, small_font)
            draw.text(((OUT_W - w) // 2, ny), line, font=small_font, fill=MUTED)
            ny += 46

    footer = "SYNTHETIC FALLBACK"
    fw, _ = _text_bbox(draw, footer, micro_font)
    draw.text(((OUT_W - fw) // 2, OUT_H - 110), footer, font=micro_font, fill=MUTED)
    draw.rectangle((70, OUT_H - 70, OUT_W - 70, OUT_H - 60), fill=ACCENT_YELLOW)
    return img


def _draw_background_grid(draw: ImageDraw.ImageDraw) -> None:
    grid_color = (26, 32, 70)
    for x in range(70, OUT_W, 90):
        draw.line((x, 0, x, OUT_H), fill=grid_color, width=1)
    for y in range(80, OUT_H, 90):
        draw.line((0, y, OUT_W, y), fill=grid_color, width=1)
    draw.rectangle((0, 0, OUT_W - 1, OUT_H - 1), outline=(255, 215, 0), width=8)


def _draw_comparison_bars(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    primary: Tuple[str, str],
    secondary: Tuple[str, str],
) -> None:
    x0, y0, x1, _ = box
    small = _font(34)
    nums = [primary[0], secondary[0]]
    labels = [primary[1], secondary[1].upper()]
    values = [_number_value(nums[0]), _number_value(nums[1])]
    max_val = max([v for v in values if v is not None] or [1.0])
    for i, (num, label, val) in enumerate(zip(nums, labels, values)):
        y = y0 + i * 130
        draw.text((x0, y), _fit_text(label, small, x1 - x0), font=small, fill=WHITE)
        bar_y = y + 50
        draw.rounded_rectangle((x0, bar_y, x1, bar_y + 42), radius=16, fill="#242a55")
        pct = 0.72 if val is None else max(0.12, min(1.0, val / max_val))
        fill = ACCENT_YELLOW if i == 0 else "#58c7ff"
        draw.rounded_rectangle((x0, bar_y, x0 + int((x1 - x0) * pct), bar_y + 42), radius=16, fill=fill)
        draw.text((x1 - 260, bar_y - 5), _fit_text(num.upper(), small, 250), font=small, fill=WHITE)


def _extract_numbers(beat: dict, primary_number: str) -> List[Tuple[str, str]]:
    visual = beat.get("visual") or {}
    data = (beat.get("overlay") or {}).get("data") or {}
    explicit: List[Tuple[str, str]] = []
    for key in ("secondary_number", "comparison_number", "value2"):
        if data.get(key):
            label = str(data.get("secondary_label") or data.get("comparison_label") or "COMPARISON")
            explicit.append((str(data[key]), label))
    if explicit:
        return explicit

    text = " ".join(
        str(x or "")
        for x in (
            visual.get("subject"),
            beat.get("narration"),
            " ".join(visual.get("queries") or []),
        )
    )
    pattern = re.compile(
        r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*"
        r"(km2|km²|sq\s*km|square\s*kilometers|million|billion|thousand|bn|people|percent|m|k|%)?",
        re.IGNORECASE,
    )
    out: List[Tuple[str, str]] = []
    primary_clean = re.sub(r"\s+", "", primary_number.lower())
    primary_value = _number_value(primary_number)
    seen = set()
    seen_values: List[float] = []
    for match in pattern.finditer(text):
        raw_num = match.group(1)
        unit = (match.group(2) or "").strip()
        display = f"{raw_num}{('%' if unit == '%' else ' ' + unit.upper()) if unit else ''}".strip()
        key = re.sub(r"\s+", "", display.lower())
        numeric = _number_value(display)
        duplicate_primary = (
            numeric is not None
            and primary_value is not None
            and abs(numeric - primary_value) <= max(1.0, abs(primary_value) * 0.01)
        )
        duplicate_seen = any(
            numeric is not None and abs(numeric - prev) <= max(1.0, abs(prev) * 0.01)
            for prev in seen_values
        )
        if key in seen or key == primary_clean or duplicate_primary or duplicate_seen:
            continue
        seen.add(key)
        if numeric is not None:
            seen_values.append(numeric)
        out.append((display, _label_for_unit(unit)))
        if len(out) >= 2:
            break
    return out


def _label_for_unit(unit: str) -> str:
    u = unit.lower()
    if "km" in u or "square" in u:
        return "LAND AREA"
    if u in ("people", "million", "billion", "thousand", "m", "bn", "k"):
        return "POPULATION"
    if "%" in u or "percent" in u:
        return "SHARE"
    return "COMPARISON"


def _render_archival_annotated(
    beat: dict,
    out_path: Path,
    sources_for_all_beats: Optional[List[Dict[str, Any]]] = None,
    source_index: Optional[int] = None,
) -> bool:
    frame_count = _frame_count(beat.get("duration_sec"))
    duration = frame_count / FPS
    spec = _archival_context_spec(beat)

    beat_id = beat.get("beat_id")
    beat_key = f"{beat_id:02d}" if isinstance(beat_id, int) else "unknown"
    image_path = fetch_context_card_image(
        beat,
        spec,
        out_path.parent / f"archival_{beat_key}.jpg",
        archive_only=True,
    )

    caption = _archival_caption(beat, spec)
    image: Optional[Image.Image] = None
    is_archival_source = False
    if image_path and image_path.exists():
        try:
            image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
            is_archival_source = True
        except Exception:
            image = None

    if image is None:
        image_path = fetch_context_card_image(
            beat,
            spec,
            out_path.parent / f"archival_context_{beat_key}.jpg",
            archive_only=False,
        )
        if image_path and image_path.exists():
            try:
                image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
                is_archival_source = _archival_source_banner(image_path) == "ARCHIVAL SOURCE"
            except Exception:
                image = None

    # Bug A fix (V4.1): when the archive + tight stock both fail, ask Gemini
    # to generate thematic stock-photo queries from the narration. Far better
    # than the old hardcoded "vintage town" tokens — for "underground heat"
    # narration it'll suggest "lava cracks", "smoking ground", "coal mine
    # fire" rather than blandly searching for the literal phrase.
    if image is None:
        image_path = fetch_thematic_archival_image(
            beat,
            spec,
            out_path.parent / f"archival_thematic_{beat_key}.jpg",
        )
        if image_path and image_path.exists():
            try:
                image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
                is_archival_source = False
                print(
                    f"[archival] beat {beat_id} thematic stock "
                    f"path={image_path.name}",
                    flush=True,
                )
            except Exception:
                image = None

    # Final loosened-keyword fallback (kept as a safety net under the Gemini
    # path in case Gemini is unreachable or returned nothing).
    if image is None:
        loosened = _loosen_archival_spec(spec)
        if loosened:
            image_path = fetch_context_card_image(
                beat,
                loosened,
                out_path.parent / f"archival_loose_{beat_key}.jpg",
                archive_only=False,
            )
            if image_path and image_path.exists():
                try:
                    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
                    is_archival_source = False
                    print(
                        f"[archival] beat {beat_id} loosened-query stock "
                        f"path={image_path.name}",
                        flush=True,
                    )
                except Exception:
                    image = None

    if image is None:
        frames = _archival_placeholder_frames(
            beat, spec, caption, frame_count,
            sources_for_all_beats, source_index,
        )
        return _write_video(frames, out_path, frame_count, duration)

    # Apply sepia/desaturate treatment to non-archival images so they still
    # read as historical context rather than modern stock.
    if not is_archival_source:
        image = _apply_sepia_treatment(image)

    credit = _archival_credit(image_path, beat)
    callouts = _archival_callouts(beat) if RENDER_ARCHIVAL_CALLOUTS else []
    source_label = _archival_source_banner(image_path)
    if source_label != "ARCHIVAL SOURCE":
        callouts = []
    frames = _archival_annotated_frames(image, caption, credit, callouts, frame_count, source_label)
    return _write_video(frames, out_path, frame_count, duration)


# Tokens stripped before building a thematic-fallback query for archival beats.
_ARCHIVAL_QUERY_FILLER = re.compile(
    r"\b(archival|archive|photo(?:graph)?|picture|image|footage|"
    r"black\s+and\s+white|b&w|sepia|"
    r"vintage|historic(?:al)?|old|original|"
    r"19[0-9]{2}s?|18[0-9]{2}s?|20[0-2][0-9]s?|"
    r"early|late|circa|c\.)\b",
    re.IGNORECASE,
)


def _loosen_archival_spec(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a spec with a broader thematic query when the precise archival
    one fails. Strips era/'archival photo of' tokens and adds period-flavor
    words so generic stock is more likely to hit something usable."""
    raw = " ".join(
        str(spec.get(k) or "") for k in ("context_visual", "headline", "category")
    )
    cleaned = _ARCHIVAL_QUERY_FILLER.sub(" ", raw)
    cleaned = re.sub(r"[^A-Za-z0-9 ,'-]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
    if not cleaned:
        return None

    # Pick at most a handful of keyword tokens so the query stays searchable.
    tokens = [t for t in cleaned.split() if len(t) >= 3][:6]
    if not tokens:
        return None

    geo = str(spec.get("geo") or "").strip()
    flavor = "abandoned vintage town"  # neutral period-flavor seed
    loosened_visual = f"{flavor} {' '.join(tokens)}".strip()

    new_spec = dict(spec)
    new_spec["context_visual"] = loosened_visual
    if geo:
        new_spec["geo"] = geo
    return new_spec


def _apply_sepia_treatment(image: Image.Image) -> Image.Image:
    """Grade a modern stock image so it reads as period/archival context.
    Desaturates, applies a warm sepia tone, and crushes blacks slightly."""
    desat = ImageOps.grayscale(image).convert("RGB")
    color = Image.blend(image.convert("RGB"), desat, 0.65)
    sepia = Image.new("RGB", color.size, (112, 88, 60))
    toned = Image.blend(color, sepia, 0.20)
    toned = ImageOps.autocontrast(toned, cutoff=2)
    # Slight darken so it groups visually with archival prints
    return Image.blend(toned, Image.new("RGB", toned.size, (0, 0, 0)), 0.08)


def _archival_context_spec(beat: dict) -> Dict[str, Any]:
    visual = beat.get("visual") or {}
    anchor = beat.get("anchor_spec") if isinstance(beat.get("anchor_spec"), dict) else {}
    subject = str(anchor.get("subject") or visual.get("subject") or beat.get("caption_text") or "").strip()
    geo = str(anchor.get("geo") or visual.get("geo") or "").strip()
    caption = str(beat.get("caption_text") or subject or beat.get("narration") or "").strip()
    era = anchor.get("era") if isinstance(anchor.get("era"), list) else None
    support = "archival source"
    if era:
        support = f"{support} {' '.join(str(y) for y in era[:2])}"
    return {
        "category": geo or subject or "URBAN ATLAS",
        "headline": caption or subject or "ARCHIVAL CONTEXT",
        "context_visual": subject or caption or geo,
        "support": support,
        "geo": geo,
    }


def _archival_caption(beat: dict, spec: Dict[str, Any]) -> str:
    visual = beat.get("visual") or {}
    anchor = beat.get("anchor_spec") if isinstance(beat.get("anchor_spec"), dict) else {}
    raw = (
        beat.get("caption_text")
        or anchor.get("subject")
        or visual.get("subject")
        or spec.get("headline")
        or "ARCHIVAL CONTEXT"
    )
    return _clean_sentence(str(raw)).upper()


def _archival_credit(image_path: Path, beat: dict) -> str:
    data: Dict[str, Any] = {}
    try:
        meta_path = image_path.with_suffix(".json")
        if meta_path.exists():
            data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    raw_source = str(data.get("source") or "").lower().strip()
    # Only show credit for verified archival sources. Stock providers
    # (pexels/pixabay) shouldn't be branded on-screen as if they were the
    # historical source — that's misleading and reads as debug watermarking.
    if raw_source not in {"loc", "wikimedia", "wikipedia", "internet_archive", "dlg"}:
        return ""

    source = _archive_source_label(raw_source)
    date = str(data.get("date") or "").strip()
    year = data.get("year")
    if not date and year:
        date = f"circa {year}"
    if not date:
        anchor = beat.get("anchor_spec") if isinstance(beat.get("anchor_spec"), dict) else {}
        era = anchor.get("era") if isinstance(anchor.get("era"), list) else None
        if era:
            date = f"circa {era[0]}" if len(era) == 1 or era[0] == era[-1] else f"{era[0]}-{era[-1]}"
    return f"{source}, {date}" if date else source


def _archival_source_banner(image_path: Optional[Path]) -> str:
    source = ""
    if image_path:
        try:
            meta_path = image_path.with_suffix(".json")
            if meta_path.exists():
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                source = str(data.get("source") or "").lower().strip()
        except Exception:
            source = ""
    if source in {"loc", "wikimedia", "wikipedia", "internet_archive", "dlg"}:
        return "ARCHIVAL SOURCE"
    # Non-archive (stock / placeholder): draw nothing. Treating sepia-toned
    # stock photos as authoritative archival sources misleads viewers, and
    # the bare "CONTEXT IMAGE" / "MAP CONTEXT" labels were reading as debug
    # watermarks in finished renders.
    return ""


def _archive_source_label(source: str) -> str:
    return {
        "loc": "LOC",
        "wikimedia": "WIKIMEDIA",
        "wikipedia": "WIKIPEDIA",
        "internet_archive": "INTERNET ARCHIVE",
        "dlg": "DIGITAL LIBRARY OF GEORGIA",
    }.get(source.lower().strip(), source.upper().replace("_", " "))


def _archival_callouts(beat: dict) -> List[Dict[str, Any]]:
    visual = beat.get("visual") or {}
    raw = visual.get("callouts") or []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = _clean_sentence(str(item.get("text") or "")).upper()
        xy = item.get("xy") or item.get("point")
        if not text or not isinstance(xy, list) or len(xy) < 2:
            continue
        try:
            x = max(0.03, min(0.97, float(xy[0])))
            y = max(0.03, min(0.97, float(xy[1])))
        except Exception:
            continue
        words = text.split()
        if len(words) > 5:
            text = " ".join(words[:5])
        out.append({"text": text[:52], "xy": (x, y), "style": str(item.get("style") or "circle")})
        if len(out) >= 3:
            break
    return out


def _archival_annotated_frames(
    image: Image.Image,
    caption: str,
    credit: str,
    callouts: List[Dict[str, Any]],
    frame_count: int,
    source_label: str = "ARCHIVAL SOURCE",
) -> Iterator[Image.Image]:
    for idx in range(frame_count):
        t = idx / max(1, frame_count - 1)
        e = _ease_in_out_cubic(t)
        frame, transform = _archival_cover_frame(image, e)
        frame = _archival_grade(frame)
        frame = _draw_archival_callouts(frame, callouts, transform, t)
        _draw_archival_caption(frame, caption, credit, source_label)
        yield _apply_depth_fx(frame)


def _archival_cover_frame(
    image: Image.Image,
    progress: float,
) -> Tuple[Image.Image, Tuple[float, float, float, float]]:
    sw, sh = image.size
    base_scale = max(OUT_W / sw, OUT_H / sh)
    scale = base_scale * _lerp(1.0, 1.085, progress)
    rw = max(OUT_W, int(round(sw * scale)))
    rh = max(OUT_H, int(round(sh * scale)))
    resized = image.resize((rw, rh), Image.Resampling.LANCZOS)

    x_shift = 0.055 if rw > OUT_W else 0.0
    y_shift = 0.045 if rh > OUT_H else 0.0
    center_x = 0.5 + x_shift * (progress - 0.5)
    center_y = 0.5 - y_shift * (progress - 0.5)
    left = int(round(rw * center_x - OUT_W / 2))
    top = int(round(rh * center_y - OUT_H / 2))
    left = max(0, min(max(0, rw - OUT_W), left))
    top = max(0, min(max(0, rh - OUT_H), top))
    crop = resized.crop((left, top, left + OUT_W, top + OUT_H))
    return crop, (float(rw), float(rh), float(left), float(top))


def _archival_grade(frame: Image.Image) -> Image.Image:
    frame = ImageOps.autocontrast(frame, cutoff=1)
    frame = Image.blend(frame, Image.new("RGB", frame.size, BG_NAVY), 0.10)
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    d.rectangle((0, 0, OUT_W, 220), fill=(0, 0, 0, 64))
    d.rectangle((0, OUT_H - 420, OUT_W, OUT_H), fill=(0, 0, 0, 128))
    return Image.alpha_composite(frame.convert("RGBA"), layer).convert("RGB")


def _draw_archival_callouts(
    frame: Image.Image,
    callouts: List[Dict[str, Any]],
    transform: Tuple[float, float, float, float],
    progress: float,
) -> Image.Image:
    if not callouts:
        return frame

    rw, rh, left, top = transform
    layer = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    font = _font(29)
    accent = _hex_to_rgb(ACCENT_YELLOW)
    bg = _hex_to_rgb(BG_NAVY)

    for idx, item in enumerate(callouts):
        appear = _ease_in_out_cubic(min(1.0, max(0.0, (progress - 0.16 - idx * 0.12) / 0.22)))
        if appear <= 0:
            continue
        alpha = int(appear * 255)
        nx, ny = item["xy"]
        px = int(round(nx * rw - left))
        py = int(round(ny * rh - top))
        px = max(34, min(OUT_W - 34, px))
        py = max(140, min(OUT_H - 460, py))
        text = str(item["text"])
        text = _fit_text(text, font, 360)
        tw, th = _text_bbox(d, text, font)
        bw, bh = tw + 34, th + 26
        side = 1 if px < OUT_W * 0.56 else -1
        bx0 = px + side * 74
        if side < 0:
            bx0 -= bw
        by0 = py - bh // 2
        bx0 = max(34, min(OUT_W - bw - 34, bx0))
        by0 = max(130, min(OUT_H - 470, by0))
        bx1, by1 = bx0 + bw, by0 + bh
        tx = bx0 if side > 0 else bx1
        ty = by0 + bh // 2

        d.line((px, py, tx, ty), fill=(0, 0, 0, int(alpha * 0.72)), width=9)
        d.line((px, py, tx, ty), fill=(*accent, alpha), width=4)
        d.ellipse((px - 22, py - 22, px + 22, py + 22), outline=(0, 0, 0, alpha), width=8)
        d.ellipse((px - 17, py - 17, px + 17, py + 17), outline=(*accent, alpha), width=5)
        d.rounded_rectangle(
            (bx0, by0, bx1, by1),
            radius=8,
            fill=(*bg, int(alpha * 0.82)),
            outline=(*accent, alpha),
            width=2,
        )
        d.text((bx0 + 17, by0 + 12), text, font=font, fill=(245, 247, 255, alpha))

    return Image.alpha_composite(frame.convert("RGBA"), layer).convert("RGB")


def _draw_archival_caption(
    frame: Image.Image,
    caption: str,
    credit: str,
    source_label: str = "ARCHIVAL SOURCE",
) -> None:
    draw = ImageDraw.Draw(frame)
    micro = _font(25)
    title_font = _font(54)
    credit_font = _font(28)

    draw.rectangle((0, OUT_H - 18, OUT_W, OUT_H), fill=ACCENT_YELLOW)
    if source_label:
        draw.text((70, OUT_H - 318), source_label, font=micro, fill=ACCENT_YELLOW,
                  stroke_width=2, stroke_fill=(0, 0, 0))
    lines = _fit_multiline(draw, caption, title_font, OUT_W - 140, max_lines=2)
    y = OUT_H - 262
    for line in lines:
        draw.text((70, y), line, font=title_font, fill=WHITE,
                  stroke_width=4, stroke_fill=(0, 0, 0))
        y += 64

    if credit:
        credit_text = _fit_text(credit.upper(), credit_font, OUT_W - 140)
        cw, _ = _text_bbox(draw, credit_text, credit_font)
        draw.text((OUT_W - 70 - cw, OUT_H - 86), credit_text, font=credit_font, fill=MUTED,
                  stroke_width=2, stroke_fill=(0, 0, 0))


def _archival_placeholder_frames(
    beat: dict,
    spec: Dict[str, Any],
    caption: str,
    frame_count: int,
    sources_for_all_beats: Optional[List[Dict[str, Any]]] = None,
    source_index: Optional[int] = None,
) -> Iterator[Image.Image]:
    # Bug A fix: when no archival/stock image is available, prefer a heavy-
    # blurred frame from the beat's own or a neighbor beat's b-roll (then
    # sepia-treat it) so the screen has thematic texture instead of just a
    # blurred Mapbox tile + caption text.
    bg = _archival_placeholder_bg(beat, spec, sources_for_all_beats, source_index)
    title_font = _font(58)
    small = _font(30)
    for idx in range(frame_count):
        t = idx / max(1, frame_count - 1)
        e = _ease_in_out_cubic(t)
        z = _lerp(1.0, 1.035, e)
        crop_w = OUT_W / z
        crop_h = OUT_H / z
        x0 = int((OUT_W - crop_w) / 2)
        y0 = int((OUT_H - crop_h) / 2)
        frame = bg.crop((x0, y0, int(x0 + crop_w), int(y0 + crop_h))).resize(
            (OUT_W, OUT_H), Image.Resampling.LANCZOS
        )
        veil = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 88))
        frame = Image.alpha_composite(frame.convert("RGBA"), veil).convert("RGB")
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, int(OUT_W * min(1.0, t / 0.18)), 18), fill=ACCENT_YELLOW)
        # Removed the "ARCHIVAL CONTEXT" / "NO ARCHIVAL SOURCE" debug text —
        # they read as watermarks on the final render. Just show the caption.
        lines = _fit_multiline(draw, caption, title_font, OUT_W - 140, max_lines=3)
        y = 250
        for line in lines:
            draw.text((70, y), line, font=title_font, fill=WHITE,
                      stroke_width=4, stroke_fill=(0, 0, 0))
            y += 68
        yield _apply_depth_fx(frame)


def _archival_placeholder_bg(
    beat: dict,
    spec: Dict[str, Any],
    sources_for_all_beats: Optional[List[Dict[str, Any]]],
    source_index: Optional[int],
) -> Image.Image:
    """Build a thematic backdrop for the archival placeholder. Prefers neighbor
    b-roll (so the empty card visually groups with the rest of the video),
    falls back to the Mapbox geo blur. Always sepia-toned so the result reads
    as historical context rather than 'we have no idea what this beat is'."""
    beat_id = beat.get("beat_id", "?")
    sources = sources_for_all_beats or []

    candidates: List[Path] = []
    own_clip = None
    if source_index is not None and 0 <= source_index < len(sources):
        own_clip = _source_clip_path(sources[source_index])
    if own_clip:
        candidates.append(own_clip)
    for idx, src in enumerate(sources):
        if source_index is not None and idx == source_index:
            continue
        clip = _source_clip_path(src)
        if clip:
            candidates.append(clip)

    for clip in candidates:
        frame = _extract_mid_video_frame(clip)
        if frame is None:
            continue
        kind = "own_broll" if clip == own_clip else "neighbor_broll"
        print(
            f"[archival-placeholder] using {kind} beat={beat_id} path={clip}",
            flush=True,
        )
        toned = _apply_sepia_treatment(frame)
        blurred = toned.filter(ImageFilter.GaussianBlur(28))
        veil = Image.new("RGB", blurred.size, BG_NAVY)
        return Image.blend(blurred, veil, 0.30)

    print(
        f"[archival-placeholder] using mapbox beat={beat_id} reason=no_broll",
        flush=True,
    )
    bg = _infographic_bg(str(spec.get("geo") or ""), blur=18, navy_blend=0.54)
    return _apply_sepia_treatment(bg)


def _render_wikipedia_photo(beat: dict, out_path: Path) -> bool:
    image = None
    used_title = ""
    for title in _wiki_title_candidates(beat):
        image = _fetch_wikipedia_image(title)
        if image is not None:
            used_title = title
            break
    if image is None:
        return False
    frame_count = _frame_count(beat.get("duration_sec"))
    duration = frame_count / FPS
    caption = _short_place_label(used_title)
    frames = _photo_frames(image, frame_count, caption)
    return _write_video(frames, out_path, frame_count, duration)


# Adjectives/modifiers that bury a Wikipedia hit. "Summerhill residential" has
# no page; "Summerhill" does. Stripped progressively on lookup retries.
_WIKI_NOISE_WORDS = {
    "residential", "commercial", "downtown", "historic", "modern", "old",
    "new", "central", "north", "south", "east", "west", "upper", "lower",
    "neighborhood", "neighbourhood", "district", "area", "zone", "block",
    "view", "scene", "shot", "footage", "photo",
}


def _wiki_title_candidates(beat: dict) -> List[str]:
    """Yield title candidates from most specific to most generic."""
    visual = beat.get("visual") or {}
    data = (beat.get("overlay") or {}).get("data") or {}
    seeds = [
        data.get("title"),
        visual.get("subject"),
        visual.get("geo"),
    ]
    out: List[str] = []
    seen: set = set()

    def _add(s: Optional[str]) -> None:
        if not s:
            return
        s = str(s).strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)

    for s in seeds:
        if not s:
            continue
        s = str(s).strip()
        _add(s)
        # Strip noise words progressively
        tokens = [t for t in re.split(r"\s+", s) if t]
        kept = [t for t in tokens if t.lower() not in _WIKI_NOISE_WORDS]
        if kept and kept != tokens:
            _add(" ".join(kept))
        # First N tokens (drop trailing modifiers like "park trees at sunset")
        if len(kept) > 2:
            _add(" ".join(kept[:2]))
        if kept:
            _add(kept[0])
    # Fall back to widening on geo: "Summerhill, Atlanta" -> "Atlanta"
    geo = str(visual.get("geo") or "")
    if "," in geo:
        for part in [p.strip() for p in geo.split(",") if p.strip()][1:]:
            _add(part)
    return out


def _fetch_wikipedia_image(title: str) -> Optional[Image.Image]:
    wiki_dir = DATA_DIR / "wiki_images"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", title).strip("_")[:120] or "wiki"
    cache_path = wiki_dir / f"{key}.jpg"
    if cache_path.exists() and cache_path.stat().st_size > 1024:
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            cache_path.unlink(missing_ok=True)

    summary_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title)
    try:
        data = _read_json(summary_url, timeout=20)
        source = ((data.get("originalimage") or {}).get("source") or (data.get("thumbnail") or {}).get("source"))
        if not source:
            return None
        req = urllib.request.Request(str(source), headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.save(cache_path, "JPEG", quality=92)
        return img
    except Exception as exc:
        print(f"[cinematics] wikipedia image failed for {title!r}: {exc}", flush=True)
        return None


def _photo_frames(image: Image.Image, frame_count: int, caption: str) -> Iterator[Image.Image]:
    bg = ImageOps.fit(image, (OUT_W, OUT_H), method=Image.Resampling.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(24))
    bg = Image.blend(bg, Image.new("RGB", (OUT_W, OUT_H), BG_NAVY), 0.34)
    portrait = image.height >= image.width
    for idx in range(frame_count):
        t = idx / max(1, frame_count - 1)
        e = _ease_in_out_cubic(t)
        scale = _lerp(1.02, 1.10, e)
        contain = _contain_size(image.size, (OUT_W - 80, OUT_H - 260), scale)
        fg = image.resize(contain, Image.Resampling.LANCZOS)
        if portrait:
            x = (OUT_W - fg.width) // 2
            travel = max(0, fg.height - (OUT_H - 260))
            y = 120 - int(travel * e)
        else:
            travel = max(0, fg.width - (OUT_W - 80))
            x = 40 - int(travel * e)
            y = (OUT_H - fg.height) // 2
        frame = bg.copy()
        frame.paste(fg, (x, y))
        _draw_photo_caption(frame, caption)
        yield frame


def _draw_photo_caption(frame: Image.Image, caption: str) -> None:
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle((0, OUT_H - 360, OUT_W, OUT_H), fill=(0, 0, 0, 120))
    frame.paste(Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB"))
    draw = ImageDraw.Draw(frame)
    font = _font(64)
    small = _font(28)
    text = _fit_text(caption.upper(), font, OUT_W - 140)
    tw, _ = _text_bbox(draw, text, font)
    y = OUT_H - 260
    draw.text(((OUT_W - tw) // 2, y), text, font=font, fill=WHITE, stroke_width=5, stroke_fill=(0, 0, 0))
    credit = "WIKIPEDIA"
    cw, _ = _text_bbox(draw, credit, small)
    draw.text(((OUT_W - cw) // 2, y + 88), credit, font=small, fill=ACCENT_YELLOW, stroke_width=2, stroke_fill=(0, 0, 0))


def _ken_burns_still_frames(
    still: Image.Image,
    frame_count: int,
    zoom_start: float,
    zoom_end: float,
) -> Iterator[Image.Image]:
    still = still.convert("RGB")
    for idx in range(frame_count):
        t = idx / max(1, frame_count - 1)
        z = _lerp(zoom_start, zoom_end, _ease_in_out_cubic(t))
        crop_w = OUT_W / z
        crop_h = OUT_H / z
        x0 = (OUT_W - crop_w) / 2
        y0 = (OUT_H - crop_h) / 2
        crop = still.crop((int(x0), int(y0), int(x0 + crop_w), int(y0 + crop_h)))
        yield crop.resize((OUT_W, OUT_H), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# Text and drawing helpers


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        FONTS_DIR / "Montserrat-Bold.ttf",
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ]
    for path in candidates:
        try:
            if path.exists():
                return ImageFont.truetype(str(path), size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _fit_font(text: str, font: ImageFont.ImageFont, max_width: int, min_size: int) -> ImageFont.ImageFont:
    size = getattr(font, "size", min_size)
    while size > min_size:
        trial = _font(size)
        if _text_bbox(ImageDraw.Draw(Image.new("RGB", (10, 10))), text, trial)[0] <= max_width:
            return trial
        size -= 4
    return _font(min_size)


def _fit_text(text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    if _text_bbox(draw, text, font)[0] <= max_width:
        return text
    ellipsis = "..."
    out = text
    while out and _text_bbox(draw, out + ellipsis, font)[0] > max_width:
        out = out[:-1].rstrip()
    return (out + ellipsis) if out else text[:8]


def _fit_multiline(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        trial = " ".join(current + [word])
        if current and _text_bbox(draw, trial, font)[0] > max_width:
            lines.append(" ".join(current))
            current = [word]
            if len(lines) >= max_lines:
                break
        else:
            current.append(word)
    if current and len(lines) < max_lines:
        lines.append(" ".join(current))
    if len(lines) == max_lines and len(" ".join(words)) > len(" ".join(lines)):
        lines[-1] = _fit_text(lines[-1], font, max_width)
    return lines or [text[:30]]


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _rounded_rect(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int, int, int],
    radius: int,
    fill: str,
    outline: Optional[Tuple[int, int, int]] = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _short_place_label(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) >= 2:
            return f"{parts[0]}, {parts[-1]}"
    return text[:42]


def _clean_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:180]


def _number_value(text: str) -> Optional[float]:
    m = re.search(r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)", text)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    lower = text.lower()
    suffix = lower[m.end():].strip()
    if suffix.startswith(("km", "sq", "square")):
        return value
    if suffix.startswith("billion") or suffix.startswith("bn"):
        value *= 1_000_000_000
    elif suffix.startswith("million") or re.match(r"m\b", suffix):
        value *= 1_000_000
    elif suffix.startswith("thousand") or re.match(r"k\b", suffix):
        value *= 1_000
    return value


def _contain_size(src: Tuple[int, int], dst: Tuple[int, int], scale: float = 1.0) -> Tuple[int, int]:
    sw, sh = src
    dw, dh = dst
    factor = min(dw / sw, dh / sh) * scale
    return max(1, int(round(sw * factor))), max(1, int(round(sh * factor)))


def _ease_in_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4 * t * t * t
    return 1 - pow(-2 * t + 2, 3) / 2


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t
