"""
captions.py — ASS caption generation and burning for Geography Shorts V2.
Ported from V1 with stability improvements for Windows.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from ..config import ROOT, OUT_W, OUT_H, FONTS_DIR, CAPTION_FONT_NAME

CAPTION_MARGIN_V = 240
CAPTION_FONT_SIZE = 50


def _format_ass_time(seconds: float) -> str:
    """Format seconds into ASS time format H:MM:SS.cc"""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def generate_ass(
    whisper_segments: List[Dict],
    hook_duration: float = 0.0,
    beats: List[Dict[str, Any]] | None = None,
) -> str:
    """
    Generate two-line karaoke ASS captions with smooth word highlighting.
    
    Ported byte-for-byte in style from V1, but cleaner code structure.
    """
    words: List[Dict[str, Any]] = []
    for seg in whisper_segments or []:
        for w in seg.get("words", []) or []:
            text = str(w.get("word", "")).strip()
            if not text:
                continue
            try:
                start = float(w.get("start", 0.0))
                end = float(w.get("end", start + 0.2))
            except (ValueError, TypeError):
                continue
            if end <= start:
                end = start + 0.2
            words.append({"word": text, "start": start, "end": end})

    if not words:
        return ""

    beat_ends: List[float] = []
    cursor = 0.0
    for beat in beats or []:
        try:
            cursor += max(0.0, float(beat.get("duration_sec") or 0.0))
        except (TypeError, ValueError):
            continue
        beat_ends.append(cursor)

    def _caption_end_limit(start: float) -> float | None:
        if not beat_ends:
            return None
        guard = 0.10
        for end in beat_ends:
            if start < end - guard:
                return max(0.0, end - guard)
        return None

    ass = (
        "[Script Info]\n"
        "Title: Geography Short Captions V2\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n"
        f"PlayResX: {OUT_W}\n"
        f"PlayResY: {OUT_H}\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # Bottom-center captions. Lower MarginV moves them closer to the bottom,
        # below the infographic panels while staying above the border/watermark.
        f"Style: Default,{CAPTION_FONT_NAME},{CAPTION_FONT_SIZE},&H0000D5FF,&H0000D5FF,"
        f"&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,30,30,{CAPTION_MARGIN_V},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )

    def _clean(raw: str) -> str:
        return (
            str(raw)
            .replace("\\N", " ")
            .replace("\n", " ")
            .replace("\r", " ")
            .replace("{", "")
            .replace("}", "")
            .strip()
        )

    def _chunk_words(in_words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        chunks: List[List[Dict[str, Any]]] = []
        cur: List[Dict[str, Any]] = []
        max_words = 14
        max_chars = 72
        for item in in_words:
            wtxt = _clean(item["word"])
            if not wtxt:
                continue
            candidate = cur + [{"word": wtxt, "start": item["start"], "end": item["end"]}]
            char_count = len(" ".join(x["word"] for x in candidate))
            if cur and (len(candidate) > max_words or char_count > max_chars):
                chunks.append(cur)
                cur = [{"word": wtxt, "start": item["start"], "end": item["end"]}]
            else:
                cur = candidate
        if cur:
            chunks.append(cur)
        return chunks

    for chunk in _chunk_words(words):
        # Hook mutex: skip chunks overlap with the intro hook
        if hook_duration > 0 and float(chunk[0]["start"]) < hook_duration:
            continue
            
        tokens = [_clean(w["word"]).upper() for w in chunk]
        if not any(tokens):
            continue
            
        full_text_str = " ".join(tokens)
        split_idx = len(tokens)
        
        # Two-line splitting strategy (aim for roughly equal halves)
        if len(full_text_str) > 20 and len(tokens) > 2:
            best_delta = 1_000_000
            for k in range(1, len(tokens)):
                left_len = len(" ".join(tokens[:k]))
                right_len = len(" ".join(tokens[k:]))
                delta = abs(left_len - right_len)
                if delta < best_delta:
                    best_delta = delta
                    split_idx = k

        for i in range(len(chunk)):
            start = float(chunk[i]["start"])
            if i < len(chunk) - 1:
                end = float(chunk[i+1]["start"])
            else:
                end = max(start + 0.1, float(chunk[i]["end"]))
            limit = _caption_end_limit(start)
            if limit is not None and end > limit:
                print(
                    f"[captions] clamped chunk at beat boundary start={start:.2f} "
                    f"old_end={end:.2f} new_end={limit:.2f}",
                    flush=True,
                )
                end = limit
            if end <= start + 0.03:
                continue
                
            start_t = _format_ass_time(start)
            end_t = _format_ass_time(end)

            out_str = ""
            for j, t in enumerate(tokens):
                prefix = ""
                if j > 0 and j != split_idx:
                    prefix = " "
                elif j == split_idx:
                    prefix = "\\N"
                
                fade_ms = 150 # soft fade duration in ms
                
                if j < i:
                    # Already-spoken word: fully visible
                    out_str += prefix + t
                elif j == i:
                    # Active word: soft fade in from invisible
                    out_str += prefix + f"{{\\alpha&HFF&\\t(0,{fade_ms},\\alpha&H00&)}}" + t
                else:
                    # Upcoming word: invisible
                    out_str += prefix + "{\\alpha&HFF&}" + t + "{\\alpha&H00&}"

            anim = "{\\an2}"
            ass += f"Dialogue: 0,{start_t},{end_t},Default,,0,0,0,,{anim}{out_str}\n"

    return ass


def burn(video_path: Path, ass_text: str, out_path: Path) -> bool:
    """
    Burn ASS subtitles into the video using FFmpeg.
    Uses ultrafast preset and crf 22 for Windows stability.
    """
    if not ass_text:
        # Just copy if no captions
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-c", "copy", str(out_path)]
        subprocess.run(cmd, capture_output=True)
        return True

    base_dir = ROOT

    # Write temp ASS to base_dir directly so the relative path has no special
    # chars (commas in run dir names like "Summerhill,_Atlanta" break the
    # ffmpeg subtitles filter parser).
    import uuid as _uuid
    temp_ass_path = base_dir / f"_tmp_caption_{_uuid.uuid4().hex[:8]}.ass"
    temp_ass_path.write_text(ass_text, encoding="utf-8")

    def _ff_escape(s: str) -> str:
        # ffmpeg filter value escaping: backslash, colon, comma, quote, brackets
        return (s.replace("\\", "\\\\")
                 .replace(":", "\\:")
                 .replace(",", "\\,")
                 .replace("'", "\\'")
                 .replace("[", "\\[")
                 .replace("]", "\\]")
                 .replace(";", "\\;"))

    try:
        ass_rel = temp_ass_path.relative_to(base_dir).as_posix()
        try:
            fonts_rel = FONTS_DIR.relative_to(base_dir).as_posix()
        except ValueError:
            fonts_rel = FONTS_DIR.as_posix()

        vf = f"subtitles={_ff_escape(ass_rel)}:fontsdir={_ff_escape(fonts_rel)}"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path.resolve()),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            str(out_path.resolve()),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(base_dir)
        )
        
        if result.returncode != 0:
            print(f"FFmpeg Error: {result.stderr}")
            return False
        return True
    
    finally:
        try:
            temp_ass_path.unlink(missing_ok=True)
        except PermissionError:
            pass
