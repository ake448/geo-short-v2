"""
config.py — Single source of truth for V2 knobs.

Loads .env once, then exposes typed constants. Reuses V1's loader contract so
both pipelines see the same environment.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V2_ROOT = Path(__file__).resolve().parent


def _load_dotenv(p: Path) -> None:
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv(ROOT / ".env")


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default

# ─── Paths ────────────────────────────────────────────────────────────────────
# Static assets (version-controlled, read-only at runtime)
ASSETS_DIR = ROOT / "assets"
FONTS_DIR = Path(os.environ.get("GEO_FONTS_DIR", str(ASSETS_DIR / "fonts"))).resolve()
MUSIC_DIR = Path(os.environ.get("GEO_MUSIC_DIR", str(ASSETS_DIR / "music"))).resolve()
SFX_DIR = ASSETS_DIR / "sfx"
VOICES_DIR = ASSETS_DIR / "voices"
LUTS_DIR = ASSETS_DIR / "luts"
BRAND_LOGO = ASSETS_DIR / "branding" / "logo.png"

# Runtime data (generated, cached, gitignored)
DATA_DIR = ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"
CACHE_DB = DATA_DIR / "cache.db"

# Source-controlled reference data (inside the package)
SEEDDATA_DIR = V2_ROOT / "seeddata"
CHANNEL_ALLOWLIST = SEEDDATA_DIR / "channel_allowlist.json"
CHANNEL_BLOCKLIST = SEEDDATA_DIR / "channel_blocklist.json"

DEFAULT_MUSIC_FILE = "delosound-nature-documentary-442828.mp3"
FINAL_EXPORT_DIR = Path(
    os.environ.get("GEO_FINAL_EXPORT_DIR", "").strip() or str(ROOT / "output" / "final_videos_v2")
).resolve()

# Mapping of friendly names to voice sample files in VOICES_DIR
VOICE_MAP = {
    "david": "david_attenborough.mp3",
    "guy":   "guy_michaels.mp3",
    "simon": "simon_whistler.mp3",
    "viral": "viral_generic.mp3",
}
DEFAULT_VOICE = "david"

for _p in (RUNS_DIR, DATA_DIR, FINAL_EXPORT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ─── Render constants (must match V1 for asset compatibility) ────────────────
OUT_W, OUT_H = 1080, 1920
FPS = 30
CAPTION_FONT_NAME = os.environ.get("CAPTION_FONT_NAME", "Montserrat Bold").strip() or "Montserrat Bold"

# ─── Pacing (tighter cuts for Shorts retention) ─────────────────────────────
BEAT_MIN_SEC = 2.0
BEAT_MAX_SEC = 3.5
TARGET_RUNTIME_SEC = 45
BEATS_PER_VIDEO = (12, 16)

# ─── Sourcing knobs ──────────────────────────────────────────────────────────
SOURCING_PARALLEL_BEATS = 3        # process this many beats concurrently
SOURCING_PARALLEL_CANDIDATES = 4   # candidates per beat fetched in parallel
SOURCING_MAX_CANDIDATES = 6        # hard cap before bailing to fallback tier
VISUAL_DEDUP_PHASH = True
VISUAL_DEDUP_THRESHOLD = 10        # max pHash Hamming distance before reject
NEGATIVE_SEARCH_TERMS = [
    "-real_estate", "-interior", "-zillow", "-for_sale",  # Filter "Zillow Noise"
    "-paris", "-nice", "-france", "-florida", "-miami",   # Filter common SEO pollution
    "-advertising", "-commercial", "-promotion"
]
TEXT_GATE_ACCEPT = 85              # score >= → skip vision
TEXT_GATE_REJECT = 55              # score <  → skip download
VISION_GATE_FRAMES = max(1, _int_env("V2_VISION_GATE_FRAMES", 3))
VISION_GATE_FACE_REJECT = _bool_env("V2_VISION_GATE_FACE_REJECT", True)
VISION_GATE_STATIC_REJECT = _bool_env("V2_VISION_GATE_STATIC_REJECT", True)
CONTEXT_CARD_VISION_GATE = True    # gate stock context-card stills before caching
INFOGRAPHIC_BG_FROM_BROLL = _bool_env("V2_INFOGRAPHIC_BG_FROM_BROLL", True)
ARCHIVE_MIN_SCORE = _float_env("V2_ARCHIVE_MIN_SCORE", 25.0)  # was 60; small/historical topics rarely score above 30
RENDER_ARCHIVAL_CALLOUTS = _bool_env("V2_RENDER_ARCHIVAL_CALLOUTS", False)  # disabled — Gemini xy coords are random until we add vision-derived positioning

# ─── Models ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
SCRIPT_MODEL = os.environ.get("V2_SCRIPT_MODEL", "gemini-3-flash-preview")
TEXT_GATE_MODEL = os.environ.get("V2_TEXT_GATE_MODEL", "gemini-3.1-flash-lite-preview")
VISION_MODEL = os.environ.get("V2_VISION_MODEL", "gemini-3-flash-preview")
TOPIC_MODEL = os.environ.get("V2_TOPIC_MODEL", "gemini-3.1-flash-lite-preview")

# ─── External APIs ───────────────────────────────────────────────────────────
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "").strip()
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY", "").strip()
MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN", "").strip()
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
CESIUM_ION_TOKEN = os.environ.get("CESIUM_ION_TOKEN", "").strip()
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "").strip()

# ─── yt-dlp ──────────────────────────────────────────────────────────────────
YTDLP_PATH = os.environ.get("YTDLP_PATH", "yt-dlp").strip() or "yt-dlp"
COOKIES_YT = os.environ.get("YOUTUBE_COOKIES_PATH", "").strip()

# ─── Whisper ─────────────────────────────────────────────────────────────────
WHISPER_BACKEND = os.environ.get("V2_WHISPER_BACKEND", "faster").strip().lower() or "faster"
WHISPER_DEVICE = os.environ.get("V2_WHISPER_DEVICE", "cpu").strip() or "cpu"  # "cuda" if available
_WHISPER_CPU = WHISPER_DEVICE.lower() == "cpu"
WHISPER_MODEL = os.environ.get(
    "V2_WHISPER_MODEL",
    "base.en" if _WHISPER_CPU else "large-v3",
).strip() or ("base.en" if _WHISPER_CPU else "large-v3")
WHISPER_COMPUTE_TYPE = os.environ.get(
    "V2_WHISPER_COMPUTE",
    "int8" if _WHISPER_CPU else "float16",
).strip() or ("int8" if _WHISPER_CPU else "float16")
WHISPER_CPU_THREADS = _int_env("V2_WHISPER_CPU_THREADS", max(1, (os.cpu_count() or 4) - 1))
WHISPER_NUM_WORKERS = _int_env("V2_WHISPER_NUM_WORKERS", 1)
WHISPER_BEAM_SIZE = _int_env("V2_WHISPER_BEAM_SIZE", 1)
WHISPER_LANGUAGE = os.environ.get("V2_WHISPER_LANGUAGE", "en").strip() or "en"

# ─── Color Grade ─────────────────────────────────────────────────────────────
# Drop Q-DDL .cube files into pipeline_v2/data/luts/ then reference by name.
# When LUT_FILE is set and exists, color_grade applies it via lut3d at
# LUT_STRENGTH (0.0=no grade, 1.0=full LUT). Per GPT-5.4 recommendation:
# Lopez @ 0.45-0.55 default; raise to 0.60 if too subtle, drop to 0.35 if
# too stylized. Falls back to manual house grade if file missing.
LUTS_DIR = DATA_DIR / "luts"
LUT_FILE = os.environ.get("V2_LUT_FILE", "Lopez.cube").strip()
LUT_STRENGTH = float(os.environ.get("V2_LUT_STRENGTH", "0.50"))

# ─── Branding ────────────────────────────────────────────────────────────────
BRAND_NAME = "Urban Atlas"
BRAND_BORDER_COLOR = "#FFD500"  # yellow
BRAND_BORDER_WIDTH = 8
WATERMARK_TINT = "#FFD500"      # color the watermark, per user
WATERMARK_DARKEN = 0.35         # 0=no darken, 1=full black overlay
