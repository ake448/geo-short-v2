"""
sfx.py — Cinematic sound design: SFX layering and sidechain music ducking.
"""
from __future__ import annotations

import subprocess
import os
from pathlib import Path
from typing import Dict, List, Optional

from ..config import ROOT, MUSIC_DIR, SFX_DIR, DEFAULT_MUSIC_FILE

def process_sfx(script: dict, vo_path: Path, run_dir: Path) -> Path:
    """Layers SFX and Music onto the voiceover to produce the final audio.
    Returns the path to `final_audio.mp3`.
    """
    total_dur = script.get("total_duration_sec", 60.0)
    out_path = run_dir / "final_audio.mp3"
    
    # 1. Resolve Music
    music_path = MUSIC_DIR / DEFAULT_MUSIC_FILE
    if not music_path.exists():
        # Fallback to any mp3 in music dir
        music_files = list(MUSIC_DIR.glob("*.mp3"))
        if music_files:
            music_path = music_files[0]
        else:
            # Last resort: just VO
            import shutil
            shutil.copy(vo_path, out_path)
            return out_path

    # 2. Plan SFX
    beats = script.get("beats", [])
    sfx_commands = []
    
    # Assets
    swoosh_path = SFX_DIR / "swoosh.mp3"
    ping_path = SFX_DIR / "ping.mp3"
    riser_path = SFX_DIR / "riser.mp3"

    # FFmpeg inputs and filters
    # Input 0: VO (already exists)
    # Input 1: Music (looping or trimmed)
    # Input 2+: SFX
    
    inputs = [
        "-i", str(vo_path.resolve()),
        "-stream_loop", "-1", "-i", str(music_path.resolve()) # Loop music
    ]
    
    # Filter fragments
    # Sidechain VO -> Music
    # [0:a] is VO. [1:a] is Music.
    # We want [music] to duck when [vo] is peaked.
    
    # Start with VO and Music
    filter_graph = [
        f"[0:a]asplit=2[vo][vo_sidechain]",
        f"[1:a]volume=0.4[music]",
        f"[music][vo_sidechain]sidechaincompress=threshold=0.01:ratio=20:level_sc=1[music_ducked]"
    ]
    
    amix_inputs = ["[vo]", "[music_ducked]"]
    
    sfx_idx = 2
    for i, beat in enumerate(beats):
        start = beat.get("audio_start", 0.0)
        end = beat.get("audio_end", 0.0)
        
        # swoosh on transitions
        if i > 0 and swoosh_path.exists():
            inputs += ["-i", str(swoosh_path.resolve())]
            # peak on transition: offset swoosh slightly before cut
            # swoosh duration ~300ms, start ~150ms before cut
            offset = max(0, start - 0.15)
            filter_graph.append(f"[{sfx_idx}:a]adelay={int(offset*1000)}|{int(offset*1000)}[sfx{sfx_idx}]")
            amix_inputs.append(f"[sfx{sfx_idx}]")
            sfx_idx += 1
            
        # ping on overlays
        if beat.get("overlay") and ping_path.exists():
            inputs += ["-i", str(ping_path.resolve())]
            # ping 100ms after start
            offset = start + 0.1
            filter_graph.append(f"[{sfx_idx}:a]adelay={int(offset*1000)}|{int(offset*1000)}[sfx{sfx_idx}]")
            amix_inputs.append(f"[sfx{sfx_idx}]")
            sfx_idx += 1

        # riser at end of hook (beat 0)
        if i == 0 and riser_path.exists():
            inputs += ["-i", str(riser_path.resolve())]
            # peak on transition to beat 1
            offset = max(0, end - 1.0) # riser starts 1s before end
            filter_graph.append(f"[{sfx_idx}:a]adelay={int(offset*1000)}|{int(offset*1000)}[sfx{sfx_idx}]")
            amix_inputs.append(f"[sfx{sfx_idx}]")
            sfx_idx += 1

    # Mix everything
    filter_graph.append(f"{''.join(amix_inputs)}amix=inputs={len(amix_inputs)}:duration=first:dropout_transition=0[out]")
    
    # Fade out at end
    fade_start = max(0, total_dur - 1.0)
    filter_graph.append(f"[out]afade=t=out:st={fade_start}:d=1.0[final]")

    # Run FFmpeg
    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(filter_graph),
        "-map", "[final]",
        "-c:a", "libmp3lame", "-q:a", "2",
        "-t", str(total_dur + 0.5), # Add small buffer
        str(out_path.resolve())
    ]
    
    print(f"  [SFX] Mixing {len(amix_inputs)} layers...")
    subprocess.run(cmd, check=True, capture_output=True)
    
    return out_path
