"""
branding.py - Yellow border overlay + Urban Vectors watermark for V2 shorts.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..config import DATA_DIR, FPS, OUT_H, OUT_W, ROOT, BRAND_LOGO


_LOGO_SRC = BRAND_LOGO
if not _LOGO_SRC.exists():
    raise FileNotFoundError(
        f"Branding logo asset not found: {_LOGO_SRC}. "
        "Expected logo.png in assets/branding/"
    )

_CACHE_DIR = DATA_DIR / "branding_cache"

BORDER_COLOR = (255, 215, 0)  # Gold yellow #FFD700
BORDER_WIDTH = 10
WATERMARK_OPACITY = 0.65
WATERMARK_HEIGHT = 100
WATERMARK_MARGIN = 20

_ANIMATED_WATERMARK_CANDIDATES = [
    ROOT / "assets" / "watermarks" / "Smoky_Unveiling_Logo_Reveal_0p_watermark.mov",
    ROOT / "assets" / "watermarks" / "1d406e4f9ca744b41449364a6b3f1514_watermark.mov",
]


def _ensure_cache_dir() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _extract_logo_transparent() -> Path:
    """Extract logo from purple gradient background -> white logo on transparent."""
    out = _CACHE_DIR / "logo_transparent.png"
    if out.exists():
        return out
    _ensure_cache_dir()

    img = Image.open(str(_LOGO_SRC)).convert("RGBA")
    arr = np.array(img, dtype=np.float32)

    # The logo is dark charcoal on light purple gradient.
    # Background ~155 lum, logo text/arrow ~100-120 lum, shadow ~130 lum.
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # Sample background from all 4 corners (large patches for accuracy)
    corners = [lum[:20, :20], lum[:20, -20:], lum[-20:, :20], lum[-20:, -20:]]
    bg_lum = np.mean([c.mean() for c in corners])
    _ = bg_lum

    # The source logo is already a transparent PNG, but the RGB channels
    # contain a purple background gradient. We keep the original high-quality
    # alpha mask and simply force the logo to be pure white.
    orig_alpha = arr[:, :, 3].astype(np.uint8)

    # REMOVE GHOST PIXELS: The source logo has alpha values of 1-2 at the
    # very edges of the frame, which prevents getbbox() from cropping tightly.
    # We threshold anything below alpha 10 to zero to ensure a crisp box.
    orig_alpha[orig_alpha < 10] = 0

    # CALCULATE BBOX ON ALPHA ONLY: getbbox() uses all channels.
    # Since we make RGB pure white (255) below, it would incorrectly
    # consider the whole frame as bounds.
    mask_img = Image.fromarray(orig_alpha, "L")
    bbox = mask_img.getbbox()

    # Create white logo with original alpha
    result = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    result[:, :, 0] = 255  # R
    result[:, :, 1] = 255  # G
    result[:, :, 2] = 255  # B
    result[:, :, 3] = orig_alpha

    out_img = Image.fromarray(result, "RGBA")

    # Crop to content using the alpha bounding box
    if bbox:
        out_img = out_img.crop(bbox)

    out_img.save(str(out), "PNG")
    print(f"  [branding] Extracted logo -> {out}", flush=True)
    return out


def _create_border_layer(
    color: tuple[int, int, int] = BORDER_COLOR,
    alpha: int = 255,
    progress: float = 1.0,
) -> Image.Image:
    """Create an RGBA border layer; progress reveals edges toward center."""
    progress = max(0.0, min(1.0, float(progress)))
    alpha = max(0, min(255, int(alpha)))
    ss = 2
    w = OUT_W * ss
    h = OUT_H * ss
    bw = BORDER_WIDTH * ss
    fill = color + (alpha,)

    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    if progress >= 1.0:
        draw.rectangle([0, 0, w - 1, h - 1], outline=fill, width=bw)
    else:
        x_reveal = max(bw, int((w / 2) * progress))
        y_reveal = max(bw, int((h / 2) * progress))

        # Top and bottom borders reveal from left/right edges toward center.
        draw.rectangle([0, 0, x_reveal, bw - 1], fill=fill)
        draw.rectangle([w - x_reveal, 0, w - 1, bw - 1], fill=fill)
        draw.rectangle([0, h - bw, x_reveal, h - 1], fill=fill)
        draw.rectangle([w - x_reveal, h - bw, w - 1, h - 1], fill=fill)

        # Left and right borders reveal from top/bottom edges toward center.
        draw.rectangle([0, 0, bw - 1, y_reveal], fill=fill)
        draw.rectangle([0, h - y_reveal, bw - 1, h - 1], fill=fill)
        draw.rectangle([w - bw, 0, w - 1, y_reveal], fill=fill)
        draw.rectangle([w - bw, h - y_reveal, w - 1, h - 1], fill=fill)

    return layer.resize((OUT_W, OUT_H), Image.LANCZOS)


def _load_watermark() -> Image.Image:
    logo_path = _extract_logo_transparent()
    logo = Image.open(str(logo_path)).convert("RGBA")

    scale = WATERMARK_HEIGHT / logo.height
    new_w = int(logo.width * scale)
    logo = logo.resize((new_w, WATERMARK_HEIGHT), Image.LANCZOS)

    logo_arr = np.array(logo, dtype=np.float32)
    logo_arr[:, :, 3] *= WATERMARK_OPACITY
    return Image.fromarray(logo_arr.astype(np.uint8), "RGBA")


def _paste_watermark(overlay: Image.Image) -> None:
    logo = _load_watermark()
    x = OUT_W - logo.width - WATERMARK_MARGIN
    y = OUT_H - WATERMARK_HEIGHT - WATERMARK_MARGIN - BORDER_WIDTH
    overlay.paste(logo, (x, y), logo)


def _create_branding_overlay(include_watermark: bool = True) -> Path:
    """Create a single RGBA overlay with the border and optional static watermark."""
    out = _CACHE_DIR / ("branding_overlay.png" if include_watermark else "branding_border_overlay.png")
    if out.exists():
        return out
    _ensure_cache_dir()

    overlay = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    border_layer = _create_border_layer()
    overlay.paste(border_layer, (0, 0), border_layer)
    if include_watermark:
        _paste_watermark(overlay)

    overlay.save(str(out), "PNG")
    print(f"  [branding] Created overlay -> {out}", flush=True)
    return out


def _ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return 1.0 - (1.0 - t) ** 3


def _intro_cache_meta(intro_duration: float, frame_count: int, include_watermark: bool) -> dict[str, Any]:
    return {
        "version": 3,
        "out_w": OUT_W,
        "out_h": OUT_H,
        "fps": FPS,
        "intro_duration": round(float(intro_duration), 4),
        "frame_count": frame_count,
        "border_color": list(BORDER_COLOR),
        "border_width": BORDER_WIDTH,
        "watermark_height": WATERMARK_HEIGHT,
        "watermark_opacity": WATERMARK_OPACITY,
        "include_watermark": include_watermark,
    }


def _build_intro_frames(intro_duration: float, include_watermark: bool = True) -> Path:
    """Build cached transparent PNG sequence for the animated intro border."""
    _ensure_cache_dir()
    intro_dir = _CACHE_DIR / ("intro" if include_watermark else "intro_border")
    intro_dir.mkdir(parents=True, exist_ok=True)

    frame_count = max(1, int(round(float(intro_duration) * FPS)))
    expected_meta = _intro_cache_meta(intro_duration, frame_count, include_watermark)
    meta_path = intro_dir / "metadata.json"
    frames = [intro_dir / f"frame_{idx:02d}.png" for idx in range(frame_count)]

    if meta_path.exists() and all(frame.exists() for frame in frames):
        try:
            if json.loads(meta_path.read_text(encoding="utf-8")) == expected_meta:
                return intro_dir
        except json.JSONDecodeError:
            pass

    for old_frame in intro_dir.glob("frame_*.png"):
        old_frame.unlink(missing_ok=True)

    denominator = max(1, frame_count - 1)
    for idx, frame_path in enumerate(frames):
        raw_t = idx / denominator
        eased = _ease_out_cubic(raw_t)

        brightness = 0.30 + 0.70 * eased
        alpha = int(round(255 * (0.15 + 0.85 * eased)))
        color = tuple(max(0, min(255, int(round(c * brightness)))) for c in BORDER_COLOR)
        motion_progress = 0.03 + 0.97 * eased

        frame = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
        border_layer = _create_border_layer(color=color, alpha=alpha, progress=motion_progress)
        frame = Image.alpha_composite(frame, border_layer)
        if include_watermark:
            _paste_watermark(frame)
        frame.save(str(frame_path), "PNG")

    meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    print(f"  [branding] Created intro frames -> {intro_dir}", flush=True)
    return intro_dir


def _animated_watermark_path() -> Path | None:
    """Return the preferred animated watermark if one has been generated."""
    for path in _ANIMATED_WATERMARK_CANDIDATES:
        if path.exists():
            return path
    return None


def apply_branding(video_path: Path, out_path: Path, intro_duration: float = 0.6) -> bool:
    """Overlay yellow border + Urban Vectors watermark onto video_path.

    The border animates in across `intro_duration` seconds at the start,
    then is static for the rest of the video. Watermark is static throughout.
    Returns True on success.
    """
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"  [branding] FAILED missing input: {video_path}", flush=True)
        return False

    animated_watermark = _animated_watermark_path()
    static_overlay = _create_branding_overlay(include_watermark=animated_watermark is None)
    intro_duration = max(0.0, float(intro_duration))

    if intro_duration <= 0:
        if animated_watermark:
            filter_complex = (
                "[0:v][2:v]overlay=0:0:format=auto:eof_action=repeat[wm];"
                "[wm][1:v]overlay=0:0:format=auto[v]"
            )
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path.resolve()),
                "-i",
                str(static_overlay.resolve()),
                "-i",
                str(animated_watermark.resolve()),
                "-filter_complex",
                filter_complex,
                "-map",
                "[v]",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "ultrafast",
                "-crf",
                "20",
                "-c:a",
                "copy",
                str(out_path.resolve()),
            ]
        else:
            cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path.resolve()),
            "-i",
            str(static_overlay.resolve()),
            "-filter_complex",
            "[0:v][1:v]overlay=0:0:format=auto[v]",
            "-map",
            "[v]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "ultrafast",
            "-crf",
            "20",
            "-c:a",
            "copy",
            str(out_path.resolve()),
            ]
    else:
        intro_dir = _build_intro_frames(intro_duration, include_watermark=animated_watermark is None)
        intro_pattern = intro_dir / "frame_%02d.png"
        if animated_watermark:
            filter_complex = (
                "[0:v][3:v]overlay=0:0:format=auto:eof_action=repeat[wm];"
                f"[wm][2:v]overlay=0:0:format=auto:enable='lt(t,{intro_duration:.4f})'[intro];"
                f"[intro][1:v]overlay=0:0:format=auto:enable='gte(t,{intro_duration:.4f})'[v]"
            )
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path.resolve()),
                "-i",
                str(static_overlay.resolve()),
                "-framerate",
                str(FPS),
                "-start_number",
                "0",
                "-i",
                str(intro_pattern.resolve()),
                "-i",
                str(animated_watermark.resolve()),
                "-filter_complex",
                filter_complex,
                "-map",
                "[v]",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "ultrafast",
                "-crf",
                "20",
                "-c:a",
                "copy",
                str(out_path.resolve()),
            ]
        else:
            filter_complex = (
                f"[0:v][2:v]overlay=0:0:format=auto:enable='lt(t,{intro_duration:.4f})'[intro];"
                f"[intro][1:v]overlay=0:0:format=auto:enable='gte(t,{intro_duration:.4f})'[v]"
            )
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path.resolve()),
                "-i",
                str(static_overlay.resolve()),
                "-framerate",
                str(FPS),
                "-start_number",
                "0",
                "-i",
                str(intro_pattern.resolve()),
                "-filter_complex",
                filter_complex,
                "-map",
                "[v]",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "ultrafast",
                "-crf",
                "20",
                "-c:a",
                "copy",
                str(out_path.resolve()),
            ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode == 0 and out_path.exists():
        if animated_watermark:
            print(f"  [branding] Animated watermark -> {animated_watermark.name}", flush=True)
        print(f"  [branding] Applied to {out_path.name}", flush=True)
        return True

    msg = (result.stderr or result.stdout or "").strip()
    print(f"  [branding] FAILED for {video_path.name}: {msg[-800:]}", flush=True)
    return False
