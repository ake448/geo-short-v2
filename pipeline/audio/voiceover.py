"""
voiceover.py — Voiceover synthesis, Whisper alignment, and beat sync.
"""
from __future__ import annotations

import json
import os
import copy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..config import (
    VOICES_DIR, ROOT, WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    WHISPER_BACKEND, WHISPER_CPU_THREADS, WHISPER_NUM_WORKERS,
    WHISPER_BEAM_SIZE, WHISPER_LANGUAGE, VOICE_MAP, DEFAULT_VOICE
)

@dataclass
class VoiceoverResult:
    audio_path: Path                   # .../voiceover.mp3
    whisper_segments: List[Dict]       # same shape as V1
    updated_script: Dict               # script with beat durations corrected
    total_duration_sec: float          # final spoken length

def synthesize(script: Dict, run_dir: Path,
               voice_path: Optional[Path] = None) -> Optional[VoiceoverResult]:
    """Generate voiceover.mp3 + whisper alignment for the script.
    Supports multi-voice beats if "voice" field is present in beats.
    """
    from ..providers import qwen_tts as tts_module
    import subprocess
    import tempfile
    import shutil

    audio_path = run_dir / "voiceover.mp3"
    beats = script.get("beats", [])
    if not beats:
        return None
    
    # 0. Resolve CLI override if any
    cli_voice_key = voice_path.name if (voice_path and voice_path.name in VOICE_MAP) else None
    cli_voice_path = voice_path if (voice_path and voice_path.exists()) else None

    # 1. Prepare synthesis plan: group consecutive beats with same voice
    groups = []
    if beats:
        # Priority: cli override -> beat field -> default
        first_voice = cli_voice_key or beats[0].get("voice") or DEFAULT_VOICE
        current_voice = first_voice
        current_narration = [beats[0].get("narration", "").strip()]
        
        for b in beats[1:]:
            v = cli_voice_key or b.get("voice") or DEFAULT_VOICE
            nar = b.get("narration", "").strip()
            if not nar: continue
            
            if v == current_voice:
                current_narration.append(nar)
            else:
                groups.append({
                    "voice": current_voice,
                    "text": " ".join(current_narration).strip()
                })
                current_voice = v
                current_narration = [nar]
        
        if current_narration:
            groups.append({
                "voice": current_voice,
                "text": " ".join(current_narration).strip()
            })

    # 2. Synthesize each group
    tmp_vox = Path(tempfile.mkdtemp(prefix="multi_voice_"))
    part_files = []
    
    try:
        for i, group in enumerate(groups):
            voice_key = group["voice"]
            text = group["text"]
            if not text: continue
            
            # Resolve voice sample path
            v_sample = cli_voice_path
            if not v_sample:
                v_filename = VOICE_MAP.get(voice_key, VOICE_MAP[DEFAULT_VOICE])
                v_sample = VOICES_DIR / v_filename
            
            print(f"  [VOICE] Group {i}: speaker={voice_key} text='{text[:30]}...'")
            voice_id = tts_module.create_cloned_voice(file_path=str(v_sample) if (v_sample and v_sample.exists()) else None)
            
            part_path = tmp_vox / f"part_{i:03d}.mp3"
            tts_module.generate_long_form_audio(text, voice_id, str(part_path))
            
            if part_path.exists():
                part_files.append(part_path)

        # 3. Concatenate all parts
        if not part_files:
            return None
            
        if len(part_files) == 1:
            shutil.copy(part_files[0], audio_path)
        else:
            # Use ffmpeg concat demuxer for seamless merging of same-codec files
            list_path = tmp_vox / "list.txt"
            list_content = "\n".join([f"file '{p.name}'" for p in part_files])
            list_path.write_text(list_content, encoding="utf-8")
            
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_path), "-c", "copy", str(audio_path)
            ], check=True, cwd=str(tmp_vox), capture_output=True)

        if not audio_path.exists():
            return None

    finally:
        shutil.rmtree(tmp_vox, ignore_errors=True)

    # 4. Run Whisper Alignment (same as V1/V2 default)
    whisper_segments = _run_whisper_alignment(audio_path)
    if not whisper_segments:
        return None

    # 5. Reconcile beat durations
    updated_script = _update_beat_durations_from_whisper(copy.deepcopy(script), whisper_segments)
    
    return VoiceoverResult(
        audio_path=audio_path,
        whisper_segments=whisper_segments,
        updated_script=updated_script,
        total_duration_sec=updated_script.get("total_duration_sec", 0.0)
    )

def _run_whisper_alignment(audio_path: Path) -> Optional[List[Dict]]:
    """Run Whisper on voiceover to get word-level timestamps."""
    json_cache = audio_path.parent / "whisper_segments.json"
    meta_cache = audio_path.parent / "whisper_segments.meta.json"
    cache_meta = _whisper_cache_meta(audio_path)
    if json_cache.exists() and _whisper_cache_matches(meta_cache, cache_meta):
        return json.loads(json_cache.read_text(encoding="utf-8"))

    segments = _run_faster_whisper(audio_path)
    if segments is None:
        if WHISPER_BACKEND == "faster":
            print("  [WHISPER] faster-whisper unavailable; falling back to openai-whisper")
        segments = _run_openai_whisper(audio_path)

    if segments:
        json_cache.write_text(json.dumps(segments, indent=2), encoding="utf-8")
        meta_cache.write_text(json.dumps(cache_meta, indent=2), encoding="utf-8")
        
    return segments


def _whisper_cache_meta(audio_path: Path) -> Dict[str, Any]:
    stat = audio_path.stat()
    return {
        "backend": _resolved_whisper_backend(),
        "model": WHISPER_MODEL,
        "device": WHISPER_DEVICE,
        "compute_type": WHISPER_COMPUTE_TYPE,
        "beam_size": WHISPER_BEAM_SIZE,
        "language": WHISPER_LANGUAGE,
        "audio_name": audio_path.name,
        "audio_size": stat.st_size,
        "audio_mtime": round(stat.st_mtime, 3),
    }


def _resolved_whisper_backend() -> str:
    if WHISPER_BACKEND in {"auto", "faster", "faster-whisper"}:
        import importlib.util
        if importlib.util.find_spec("faster_whisper") is not None:
            return "faster"
    return "openai"


def _whisper_cache_matches(meta_cache: Path, expected: Dict[str, Any]) -> bool:
    if not meta_cache.exists():
        return False
    try:
        cached = json.loads(meta_cache.read_text(encoding="utf-8"))
    except Exception:
        return False
    return all(cached.get(k) == v for k, v in expected.items())


def _run_faster_whisper(audio_path: Path) -> Optional[List[Dict]]:
    if WHISPER_BACKEND not in {"auto", "faster", "faster-whisper"}:
        return None

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None

    print(
        "  [WHISPER] Using faster-whisper "
        f"model={WHISPER_MODEL} device={WHISPER_DEVICE} "
        f"compute={WHISPER_COMPUTE_TYPE} threads={WHISPER_CPU_THREADS}"
    )
    model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
        cpu_threads=WHISPER_CPU_THREADS,
        num_workers=WHISPER_NUM_WORKERS,
    )
    segments_gen, _ = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language=WHISPER_LANGUAGE,
        beam_size=WHISPER_BEAM_SIZE,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
    )

    segments = []
    for s in segments_gen:
        segments.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text.strip(),
            "words": [
                {
                    "word": w.word,
                    "start": float(w.start if w.start is not None else s.start),
                    "end": float(w.end if w.end is not None else s.end),
                }
                for w in (s.words or [])
            ],
        })
    return segments


def _run_openai_whisper(audio_path: Path) -> Optional[List[Dict]]:
    import gc
    import whisper

    # OOM-aware model loading (V1 legacy)
    fallbacks = []
    for m_name in (WHISPER_MODEL, "base.en", "tiny.en"):
        if m_name not in fallbacks:
            fallbacks.append(m_name)

    for m_name in fallbacks:
        try:
            print(f"  [WHISPER] Using openai-whisper model={m_name} device={WHISPER_DEVICE}")
            model = whisper.load_model(m_name, device=WHISPER_DEVICE)
            break
        except (RuntimeError, Exception) as e:
            if "mem" in str(e).lower() or "alloc" in str(e).lower():
                gc.collect()
                continue
            raise
    else:
        return None

    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language=WHISPER_LANGUAGE,
        verbose=False,
        beam_size=WHISPER_BEAM_SIZE,
        temperature=0.0,
        fp16=WHISPER_DEVICE.lower() != "cpu",
    )

    return [
        {
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": s["text"].strip(),
            "words": [
                {"word": w["word"], "start": float(w["start"]), "end": float(w["end"])}
                for w in s.get("words", [])
            ],
        }
        for s in result.get("segments", [])
    ]

def _update_beat_durations_from_whisper(
    script: Dict[str, Any], whisper_segs: List[Dict]
) -> Dict[str, Any]:
    """Ported from V1: Sync beat durations with real spoken timing."""
    beats = script.get("beats", [])
    if not beats or not whisper_segs:
        return script

    word_synced = _update_beat_durations_from_words(script, whisper_segs)
    if word_synced is not None:
        return word_synced

    BUFFER_SEC = 0.5
    MIN_DUR = 3.0
    MAX_DUR = 9.0  # BEAT_MAX_SEC (8.0) + 1s tolerance; enforces word-budget upstream
    LOOKAHEAD = 6
    MIN_SCORE = 0.12

    def _word_overlap(a: str, b: str) -> float:
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    seg_idx = 0
    last_known_end = 0.0

    for bi, beat in enumerate(beats):
        narration = beat.get("narration", "").lower().strip()

        if not narration:
            beat["audio_start"] = round(last_known_end, 3)
            beat["audio_end"] = round(last_known_end + 4.0, 3)
            beat["duration_sec"] = 4.0
            last_known_end = beat["audio_end"]
            continue

        if seg_idx >= len(whisper_segs):
            beat["audio_start"] = round(last_known_end, 3)
            beat["audio_end"] = round(last_known_end + float(beat.get("duration_sec", 7.0)), 3)
            last_known_end = beat["audio_end"]
            continue

        best_end_idx = seg_idx
        best_score = 0.0
        accumulated_text = ""

        for si in range(seg_idx, min(seg_idx + LOOKAHEAD, len(whisper_segs))):
            accumulated_text += " " + whisper_segs[si].get("text", "")
            score = _word_overlap(narration, accumulated_text)
            if score > best_score:
                best_score = score
                best_end_idx = si

        audio_start = float(whisper_segs[seg_idx]["start"])
        audio_end = float(whisper_segs[best_end_idx]["end"])

        if best_score >= MIN_SCORE:
            real_dur = max(MIN_DUR, min(MAX_DUR, audio_end - audio_start + BUFFER_SEC))
            beat["audio_start"] = round(audio_start, 3)
            beat["audio_end"] = round(audio_end, 3)
            beat["duration_sec"] = round(real_dur, 2)
            last_known_end = audio_end
            seg_idx = best_end_idx + 1
        else:
            fallback_dur = float(beat.get("duration_sec", 7.0))
            beat["audio_start"] = round(last_known_end, 3)
            beat["audio_end"] = round(last_known_end + fallback_dur, 3)
            beat["duration_sec"] = round(fallback_dur, 2)
            last_known_end = beat["audio_end"]
            seg_idx += 1

    script["total_duration_sec"] = round(
        sum(float(b.get("duration_sec", 7)) for b in beats), 1
    )
    return script


def _norm_tokens(text: str) -> List[str]:
    import re
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _update_beat_durations_from_words(
    script: Dict[str, Any], whisper_segs: List[Dict]
) -> Optional[Dict[str, Any]]:
    """Align beats against sequential word timestamps, not whole segments.

    faster-whisper often merges several short narration lines into one segment.
    Segment-level matching then stretches the first beat across multiple lines.
    The script text is synthesized in beat order, so a sequential word alignment
    is more stable for the short, one-clause pacing used by V2.
    """
    beats = script.get("beats", [])
    flat_words: List[Dict[str, Any]] = []
    for seg in whisper_segs or []:
        for item in seg.get("words", []) or []:
            raw = str(item.get("word") or "")
            tokens = _norm_tokens(raw)
            if not tokens:
                continue
            try:
                start = float(item.get("start", seg.get("start", 0.0)))
                end = float(item.get("end", seg.get("end", start + 0.2)))
            except Exception:
                continue
            if end <= start:
                end = start + 0.2
            for token in tokens:
                flat_words.append({"token": token, "start": start, "end": end})

    if len(flat_words) < max(1, len(beats) // 2):
        return None

    spans: List[tuple[float, float] | None] = []
    cursor = 0

    def _window_score(target: List[str], got: List[str]) -> float:
        if not target or not got:
            return 0.0
        target_set = set(target)
        got_set = set(got)
        overlap = len(target_set & got_set) / max(1, len(target_set | got_set))
        ordered = sum(1 for a, b in zip(target, got) if a == b) / max(len(target), len(got), 1)
        return overlap * 0.65 + ordered * 0.35

    for beat in beats:
        target = _norm_tokens(str(beat.get("narration") or ""))
        if not target:
            spans.append(None)
            continue

        remaining = len(flat_words) - cursor
        if remaining <= 0:
            spans.append(None)
            continue

        n = len(target)
        best: tuple[float, int, int] | None = None
        max_start = min(cursor + 3, len(flat_words) - 1)
        min_len = max(1, n - max(2, n // 3))
        max_len = min(remaining, n + max(3, n // 2))
        for start_idx in range(cursor, max_start + 1):
            for length in range(min_len, max_len + 1):
                end_idx = min(len(flat_words), start_idx + length)
                if end_idx <= start_idx:
                    continue
                got = [w["token"] for w in flat_words[start_idx:end_idx]]
                score = _window_score(target, got)
                score -= abs(length - n) * 0.015
                score -= (start_idx - cursor) * 0.04
                if best is None or score > best[0]:
                    best = (score, start_idx, end_idx)

        if best is None or best[0] < 0.10:
            spans.append(None)
            cursor = min(len(flat_words), cursor + n)
            continue

        _, start_idx, end_idx = best
        start = float(flat_words[start_idx]["start"])
        end = float(flat_words[end_idx - 1]["end"])
        spans.append((start, end))
        cursor = end_idx

    if not any(spans):
        return None

    last_end = 0.0
    for idx, beat in enumerate(beats):
        span = spans[idx] if idx < len(spans) else None
        if span is None:
            fallback = float(beat.get("duration_sec", 3.0) or 3.0)
            beat["audio_start"] = round(last_end, 3)
            beat["audio_end"] = round(last_end + fallback, 3)
            beat["duration_sec"] = round(fallback, 2)
            last_end = float(beat["audio_end"])
            continue

        start, word_end = span
        next_start = None
        for next_span in spans[idx + 1:]:
            if next_span is not None:
                next_start = float(next_span[0])
                break
        # Cap dead-air hold: never extend the visual clip more than 0.6s past
        # the last spoken word. Previously, extending to next_start could create
        # 1-2s of silence at the end of a beat where the clip played but nothing
        # was being said.
        _MAX_POST_SPEECH_GAP = 0.6
        if next_start is not None and next_start > start:
            visual_end = min(next_start, word_end + _MAX_POST_SPEECH_GAP)
        else:
            visual_end = word_end + 0.30
        visual_end = max(visual_end, word_end + 0.08)
        dur = max(2.0, visual_end - start)
        beat["audio_start"] = round(start, 3)
        beat["audio_end"] = round(word_end, 3)
        beat["duration_sec"] = round(dur, 2)
        last_end = visual_end

    script["total_duration_sec"] = round(
        sum(float(b.get("duration_sec", 3.0)) for b in beats), 1
    )
    return script
