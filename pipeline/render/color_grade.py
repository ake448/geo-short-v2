"""
color_grade.py — Per-clip color grading profiles for V2.

Returns ffmpeg filter strings to be inlined into the assembly filter_complex
chain. No standalone encode pass — the grade rides along with the existing
scale/crop/fps normalization.

Grades are derived from findings.md (2026-04-18 research). The house grade
targets an Atlas Pro + Wendover hybrid: cool shadows, warm highlights,
muted greens, lifted blacks, smooth highlight rolloff.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from ..config import LUTS_DIR, LUT_FILE, LUT_STRENGTH


# Conservative pre-normalization to flatten exposure/WB drift across sourced
# YouTube/Pexels clips before the creative grade. Per findings.md: keep
# strength low and independence near 0 to avoid color shift.
PRE_NORMALIZE = (
    "normalize=blackpt=0x040404:whitept=0xf6f6f6:smoothing=50:"
    "independence=0.10:strength=0.20"
)


def lut_path() -> Optional[str]:
    """Return the configured LUT path if it exists on disk, else None."""
    if not LUT_FILE:
        return None
    p = LUTS_DIR / LUT_FILE
    return str(p.resolve()).replace("\\", "/") if p.exists() else None


def _ff_quote_path(p: str) -> str:
    """Wrap a path for use as a quoted ffmpeg filter argument value.

    Per ffmpeg filtergraph escaping: wrap in single quotes, escape any
    internal single quote, and escape colons (the option separator) even
    inside the quotes — Windows drive letters like 'C:' otherwise terminate
    the value at the colon.
    """
    inner = p.replace("'", "\\'").replace(":", "\\:")
    return f"'{inner}'"


def lut_chain_for(in_label: str, out_label: str,
                  strength: Optional[float] = None) -> Optional[str]:
    """Build a split+lut3d+blend filter_complex fragment for a single clip.

    Returns the fragment as `;`-joined statements ending in [out_label], or
    None if no LUT is configured. The caller stitches this in place of the
    inline grade chain.
    """
    p = lut_path()
    if not p:
        return None
    s = float(strength if strength is not None else LUT_STRENGTH)
    s = max(0.0, min(1.0, s))
    quoted = _ff_quote_path(p)
    a, b, g = f"{in_label}_a", f"{in_label}_b", f"{in_label}_g"
    # lut3d's positional first arg is `clut` (a stream input), so use
    # `file=` explicitly for the cube path. Quote+escape per Windows.
    return (
        f"[{in_label}]split[{a}][{b}];"
        f"[{b}]lut3d=file={quoted}:interp=tetrahedral[{g}];"
        f"[{a}][{g}]blend=all_mode=normal:all_opacity={s:.2f}[{out_label}]"
    )


# Each profile is a comma-joined ffmpeg filter chain (no leading/trailing comma).
GRADES: Dict[str, str] = {
    # Urban Atlas house grade — Atlas Pro + Wendover hybrid.
    # Teal/cyan shadows, slightly warm highlights, muted greens, lifted blacks.
    "cinematic": (
        "eq=brightness=0.000:contrast=1.10:saturation=0.90:gamma=1.02:gamma_weight=0.82,"
        "curves=interp=pchip:"
        "m='0/0.035 0.18/0.155 0.50/0.505 0.84/0.895 1/0.975':"
        "r='0/0.030 0.50/0.505 1/0.985':"
        "g='0/0.035 0.50/0.490 1/0.965':"
        "b='0/0.055 0.50/0.530 1/0.990',"
        "colorchannelmixer=rr=1.02:rg=0.00:rb=-0.01:gr=0.015:gg=0.965:gb=0.005:"
        "br=-0.025:bg=0.045:bb=1.045:pc=lum:pa=0.32,"
        "colorbalance=rs=-0.04:gs=0.00:bs=0.07:rm=0.00:gm=0.00:bm=0.02:"
        "rh=0.05:gh=0.01:bh=-0.05:pl=1"
    ),

    # Heavier blue lift, darker midtones — for night-city beats.
    "urban_night": (
        "eq=contrast=1.22:brightness=-0.10:saturation=1.05,"
        "colorchannelmixer=rr=0.92:gg=0.96:bb=1.14:rb=0.03,"
        "curves=preset=darker,"
        "unsharp=3:3:0.5:3:3:0.0,"
        "vignette=angle=PI/4"
    ),

    # Light touch for cinematic/generated content (map_zoom, map_animation,
    # wikipedia_photo) — they're already stylized; just match overall tone
    # so they sit alongside graded footage without jumping.
    "generated": (
        "eq=contrast=1.04:saturation=0.94:gamma=1.02,"
        "colorchannelmixer=rr=1.01:gg=0.98:bb=1.04:bg=0.02,"
        "colorbalance=rs=-0.02:bs=0.04:rh=0.02:bh=-0.02:pl=1"
    ),

    "none": "",
}

DEFAULT_GRADE = "cinematic"

# Sources where pre-normalize helps (variable exposure/WB).
# Cinematic renders are already controlled — skip normalize for them.
_NORMALIZE_SOURCES = {"youtube", "pexels", "pixabay"}


def pick_profile(source: Optional[str], kind: Optional[str],
                 render_mode: Optional[str] = None) -> str:
    """Choose a grade profile for a clip based on its source/kind/mode."""
    rm = (render_mode or "").strip().lower()
    if rm == "night":
        return "urban_night"

    src = (source or "").strip().lower()
    if src == "cinematic":
        return "generated"

    return DEFAULT_GRADE


def filter_for(profile: str) -> str:
    """Return the ffmpeg filter chain for a profile name (empty if none)."""
    return GRADES.get(profile, GRADES[DEFAULT_GRADE])


def needs_normalize(source: Optional[str]) -> bool:
    """Whether to apply the pre-normalize pass for this source kind."""
    return (source or "").strip().lower() in _NORMALIZE_SOURCES
