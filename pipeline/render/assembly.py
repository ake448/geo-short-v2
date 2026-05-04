"""
assembly.py — Concat per-beat clips to beat durations and mux with final audio.

Input: list of (clip_path, duration_sec) tuples + final_audio.mp3
Output: 1080x1920, 30fps, H.264, yuv420p MP4 at run_dir/assembled.mp4

Uses a filter_complex pipeline: each clip is trimmed, rescaled/padded to 1080x1920,
then concatenated. Audio is taken from the final_audio track and truncated to the
assembled video length.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List, Tuple

from ..config import OUT_W, OUT_H, FPS
from . import color_grade


def _probe_aspect(clip_path: Path) -> float:
    """Return width/height ratio of the first video stream."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(Path(clip_path).resolve()),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return 0.0
        data = json.loads(r.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        w = float(stream.get("width") or 0)
        h = float(stream.get("height") or 0)
        return (w / h) if w > 0 and h > 0 else 0.0
    except Exception:
        return 0.0


def _probe_duration(clip_path: Path) -> float:
    """Return the duration in seconds of a video file's first stream."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(Path(clip_path).resolve()),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return 0.0
        return float((r.stdout or "0").strip() or 0.0)
    except Exception:
        return 0.0


def _normalize_chain(input_idx: int, dur: float, out_label: str,
                     aspect: float, norm_part: str,
                     source_dur: float = 0.0) -> List[str]:
    # Pad strategy: when source is long enough we just trim — no loop, no
    # clone. When source is shorter than the beat, fill the gap by holding
    # the LAST frame (tpad clone). The previous unconditional `loop` made
    # short clips visibly restart from frame 0 at the beat boundary.
    if source_dur and source_dur + 0.05 >= dur:
        stem = f"[{input_idx}:v]trim=duration={dur:.3f},setpts=PTS-STARTPTS"
    else:
        stem = (
            f"[{input_idx}:v]tpad=stop_mode=clone:stop_duration={dur:.3f},"
            f"trim=duration={dur:.3f},setpts=PTS-STARTPTS"
        )
    if aspect >= 1.4:
        return [
            (
                f"{stem},scale=-2:{OUT_H}:force_original_aspect_ratio=increase,"
                f"crop={OUT_W}:{OUT_H},fps={FPS},setsar=1{norm_part}[{out_label}]"
            )
        ]
    return [
        (
            f"{stem},scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{OUT_H},fps={FPS},setsar=1{norm_part}[{out_label}]"
        )
    ]


def assemble(
    clips: List[Tuple[Path, float]],
    audio_path: Path,
    out_path: Path,
    grade_profiles: List[str] | None = None,
    normalize_flags: List[bool] | None = None,
) -> bool:
    """Concatenate clips with exact per-beat durations, then mux audio.

    Args:
        clips: list of (video_path, duration_sec) in beat order
        audio_path: final_audio.mp3 (already mixed VO + music + SFX)
        out_path: destination mp4
        grade_profiles: optional per-clip color grade profile names (parallel
            to clips). Falls back to DEFAULT_GRADE for any None/missing entry.

    Returns True on success.
    """
    clips = [(Path(p), float(d)) for p, d in clips if p and Path(p).exists() and d > 0]
    if not clips:
        print("  [assembly] FAILED: no valid clips", flush=True)
        return False

    if grade_profiles is None:
        grade_profiles = [color_grade.DEFAULT_GRADE] * len(clips)
    while len(grade_profiles) < len(clips):
        grade_profiles.append(color_grade.DEFAULT_GRADE)

    if normalize_flags is None:
        normalize_flags = [False] * len(clips)
    while len(normalize_flags) < len(clips):
        normalize_flags.append(False)

    audio_path = Path(audio_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    inputs: List[str] = []
    for clip, _ in clips:
        inputs += ["-i", str(clip.resolve())]
    inputs += ["-i", str(audio_path.resolve())]
    audio_idx = len(clips)

    # Per-clip normalization: trim to exact duration, scale+pad to 1080x1920, fix fps+sar
    segs: List[str] = []
    concat_labels: List[str] = []
    lut_active = color_grade.lut_path() is not None
    for i, (clip, dur) in enumerate(clips):
        lbl = f"v{i}"
        norm_part = f",{color_grade.PRE_NORMALIZE}" if normalize_flags[i] else ""
        aspect = _probe_aspect(clip)
        source_dur = _probe_duration(clip)
        mode = "portrait_crop" if aspect >= 1.4 else "vertical"
        pad_mode = "trim" if source_dur + 0.05 >= dur else "freeze-pad"
        print(
            f"  [reframe] {clip.name} aspect={aspect:.3f} mode={mode} "
            f"source={source_dur:.2f}s beat={dur:.2f}s pad={pad_mode}",
            flush=True,
        )

        if lut_active:
            # Per GPT-5.4: don't stack manual house grade on top of LUT.
            # Pre-normalize → split → lut3d → blend at partial strength.
            pre_label = f"v{i}_pre"
            pre_stmts = _normalize_chain(i, dur, pre_label, aspect, norm_part, source_dur)
            lut_stmt = color_grade.lut_chain_for(pre_label, lbl)
            segs.extend(pre_stmts)
            segs.append(lut_stmt)
        else:
            # Manual house grade chain inline.
            grade_chain = color_grade.filter_for(grade_profiles[i])
            norm_label = f"v{i}_norm"
            segs.extend(_normalize_chain(i, dur, norm_label, aspect, norm_part, source_dur))
            if grade_chain:
                segs.append(f"[{norm_label}]{grade_chain}[{lbl}]")
            else:
                segs.append(f"[{norm_label}]copy[{lbl}]")
        concat_labels.append(f"[{lbl}]")

    concat_stmt = "".join(concat_labels) + f"concat=n={len(clips)}:v=1:a=0[vout]"
    fc = ";".join(segs + [concat_stmt])

    total_dur = sum(d for _, d in clips)

    cmd = [
        "ffmpeg", "-y",
        "-filter_threads", "1",
        "-filter_complex_threads", "1",
        *inputs,
        "-filter_complex", fc,
        "-map", "[vout]",
        "-map", f"{audio_idx}:a",
        "-t", f"{total_dur:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "ultrafast", "-crf", "20",
        "-threads", "1",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(out_path.resolve()),
    ]

    print(f"  [assembly] Concatenating {len(clips)} clips -> {out_path.name} ({total_dur:.1f}s)", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode == 0 and out_path.exists():
        return True

    msg = (result.stderr or result.stdout or "").strip()
    print(f"  [assembly] FAILED: {msg[-1000:]}", flush=True)
    return False
