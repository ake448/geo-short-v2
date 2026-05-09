#!/usr/bin/env python3
"""
Qwen voice cloning + synthesis helper used by geo_short_maker.py.

Implements:
- Open enrollment voice creation via Qwen Voice-Enrollment API
- Local voice cache keyed by (audio hash + target model)
- Reuse cached voice ID for synthesis calls
- Natural text splitting for long scripts
- Chunk synthesis + ffmpeg concatenation

Public API expected by geo_short_maker.py:
- create_cloned_voice(file_path: Optional[str]) -> str
- generate_long_form_audio(text: str, voice_id: str, output_path: str) -> None
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests


# ──────────────────────────────────────────────────────────────────────────────
# Paths and env
# ──────────────────────────────────────────────────────────────────────────────
from ..config import ROOT, VOICES_DIR

CACHE_FILE = VOICES_DIR / ".qwen_voice_cache.json"

_default_voice_candidates = [
    VOICES_DIR / "david_attenborough.mp3",
    ROOT / "assets" / "fonts" / "David Attenborough describes humanity.mp3", # Fallback for old layout
]
_env_voice = os.environ.get("DEFAULT_VOICE_FILE", "").strip()
if _env_voice:
    DEFAULT_VOICE_FILE = Path(_env_voice).resolve()
else:
    DEFAULT_VOICE_FILE = next((p for p in _default_voice_candidates if p.exists()), _default_voice_candidates[0]).resolve()

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "").strip()

# Enrollment model is fixed by Qwen docs
VOICE_ENROLLMENT_MODEL = os.environ.get("QWEN_VOICE_ENROLLMENT_MODEL", "qwen-voice-enrollment").strip() or "qwen-voice-enrollment"

# Must match target_model used during enrollment
# Per Qwen docs: realtime models use WebSocket, non-streaming uses HTTP.
# We use the realtime model since it works with the QwenTtsRealtime SDK.
QWEN_TTS_MODEL = os.environ.get("QWEN_TTS_MODEL", "qwen3-tts-vc-realtime-2026-01-15").strip() or "qwen3-tts-vc-realtime-2026-01-15"

# Qwen region endpoint (intl default)
DASHSCOPE_TTS_CUSTOMIZATION_URL = os.environ.get(
    "DASHSCOPE_TTS_CUSTOMIZATION_URL",
    "https://dashscope-intl.aliyuncs.com/api/v1/services/audio/tts/customization",
).strip()

# Audio synthesis uses SDK; optional base url override
DASHSCOPE_BASE_HTTP_API_URL = os.environ.get("DASHSCOPE_BASE_HTTP_API_URL", "").strip()

TTS_CHAR_LIMIT = int(os.environ.get("TTS_CHAR_LIMIT", "300"))
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg").strip() or "ffmpeg"
CROSSFADE_MS = int(os.environ.get("TTS_CROSSFADE_MS", "80"))

# Practical local cache expiry. Voices themselves can live much longer server-side,
# but we refresh local enrollment periodically.
CACHE_TTL_SEC = int(os.environ.get("VOICE_CACHE_TTL_SEC", str(14 * 24 * 3600)))


def _load_dotenv() -> None:
    for candidate in (ROOT / ".env", ROOT.parent / ".env"):
        if not candidate.exists():
            continue
        try:
            for raw in candidate.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception:
            pass


_load_dotenv()
DASHSCOPE_API_KEY = DASHSCOPE_API_KEY or os.environ.get("DASHSCOPE_API_KEY", "").strip()


# ──────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────────────────────────────────────
def _load_cache() -> Dict[str, Dict]:
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Dict]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _audio_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()[:24]


def _cache_key(path: Path, target_model: str) -> str:
    return f"{target_model}::{_audio_hash(path)}"


def _is_cache_valid(entry: Dict) -> bool:
    created_at = float(entry.get("created_at", 0))
    return (time.time() - created_at) < CACHE_TTL_SEC


# ──────────────────────────────────────────────────────────────────────────────
# Enrollment API
# ──────────────────────────────────────────────────────────────────────────────
def _mime_for_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".m4a":
        return "audio/mp4"
    raise ValueError(f"Unsupported voice file format: {path.suffix}. Use wav/mp3/m4a")


def _to_data_uri(path: Path) -> str:
    mime = _mime_for_file(path)
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _enroll_voice_with_qwen(file_path: Path, target_model: str) -> str:
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY is missing")

    preferred_name = re.sub(r"[^a-zA-Z0-9_]", "_", file_path.stem)[:16] or "geo_voice"

    payload = {
        "model": VOICE_ENROLLMENT_MODEL,
        "input": {
            "action": "create",
            "target_model": target_model,
            "preferred_name": preferred_name,
            "audio": {
                "data": _to_data_uri(file_path)
            },
        },
    }

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(
        DASHSCOPE_TTS_CUSTOMIZATION_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Voice enrollment failed {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    voice = data.get("output", {}).get("voice")
    output_target_model = data.get("output", {}).get("target_model")

    if not voice:
        raise RuntimeError(f"Voice enrollment response missing output.voice: {data}")

    if output_target_model and output_target_model != target_model:
        raise RuntimeError(
            f"Enrollment target_model mismatch. expected={target_model} got={output_target_model}"
        )

    return voice


def create_cloned_voice(file_path: Optional[str] = None) -> str:
    """Create or reuse a cached cloned voice, returning the voice token."""
    voice_file = Path(file_path).resolve() if file_path else DEFAULT_VOICE_FILE
    if not voice_file.exists():
        raise FileNotFoundError(f"Voice file not found: {voice_file}")

    # Basic guardrails from Qwen docs
    size_mb = voice_file.stat().st_size / (1024 * 1024)
    if size_mb > 10:
        raise RuntimeError(f"Voice file is {size_mb:.2f}MB; must be < 10MB")

    cache = _load_cache()
    key = _cache_key(voice_file, QWEN_TTS_MODEL)
    entry = cache.get(key)

    if entry and _is_cache_valid(entry):
        voice_id = entry.get("voice_id", "")
        if voice_id:
            age_hours = (time.time() - float(entry.get("created_at", time.time()))) / 3600
            print(f"  [CACHED] Reusing voice_id={voice_id} ({age_hours:.1f}h old)")
            return voice_id

    print(f"  [ENROLL] Creating Qwen voice from {voice_file.name} using model={QWEN_TTS_MODEL}")
    voice_id = _enroll_voice_with_qwen(voice_file, QWEN_TTS_MODEL)

    cache[key] = {
        "voice_id": voice_id,
        "voice_file": str(voice_file),
        "target_model": QWEN_TTS_MODEL,
        "created_at": time.time(),
    }
    _save_cache(cache)
    print(f"  [OK] Enrolled voice_id={voice_id}")
    return voice_id


# ──────────────────────────────────────────────────────────────────────────────
# Natural script splitting
# ──────────────────────────────────────────────────────────────────────────────
def split_text_naturally(text: str, max_chars: int = TTS_CHAR_LIMIT) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text or len(text) <= max_chars:
        return [text] if text else []

    # Step 1: split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        # Sentence itself too long — must sub-split
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            # Sub-split on clause boundaries
            sub_chunks = _split_long_sentence(sentence, max_chars)
            chunks.extend(sub_chunks)
            continue

        # Adding this sentence would exceed limit — flush current
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence).strip() if current else sentence

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]


def _split_long_sentence(sentence: str, max_chars: int) -> List[str]:
    """Last resort: split a single long sentence on clause boundaries."""
    chunks = []
    remaining = sentence
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut = _last_boundary(window, r"[,;:—–](?:\s|$)")
        if cut < 0:
            cut = window.rfind(" ")
        if cut < 0:
            cut = max_chars - 1
        chunks.append(remaining[:cut + 1].strip())
        remaining = remaining[cut + 1:].lstrip()
    if remaining:
        chunks.append(remaining.strip())
    return chunks


def _last_boundary(text: str, pattern: str) -> int:
    last = -1
    for m in re.finditer(pattern, text):
        last = m.end() - 1
    return last


# ──────────────────────────────────────────────────────────────────────────────
# Synthesis — uses QwenTtsRealtime (bidirectional WebSocket) per Qwen docs.
# The realtime model streams PCM audio which we save as WAV chunks then merge.
# ──────────────────────────────────────────────────────────────────────────────
import struct
import threading

# International WebSocket endpoint for realtime TTS
QWEN_TTS_WS_URL = os.environ.get(
    "QWEN_TTS_WS_URL",
    "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime",
).strip()


def _configure_dashscope():
    import dashscope
    dashscope.api_key = DASHSCOPE_API_KEY
    dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"


def _pcm_to_wav(pcm: bytes, sample_rate: int = 24000, channels: int = 1, bits: int = 16) -> bytes:
    """Wrap raw PCM bytes in a WAV header."""
    byte_rate = sample_rate * channels * (bits // 8)
    block_align = channels * (bits // 8)
    data_size = len(pcm)
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE',
        b'fmt ', 16, 1, channels, sample_rate, byte_rate, block_align, bits,
        b'data', data_size,
    )
    return header + pcm


def _synthesize_chunk(text: str, voice_id: str, out_path: Path) -> Path:
    """Synthesize text to audio file using QwenTtsRealtime WebSocket API."""
    _configure_dashscope()
    import base64 as b64
    from dashscope.audio.qwen_tts_realtime import (
        QwenTtsRealtime, QwenTtsRealtimeCallback, AudioFormat,
    )

    audio_chunks: List[bytes] = []
    done_event = threading.Event()
    error_holder: List[str] = []

    class _Collector(QwenTtsRealtimeCallback):
        def on_open(self) -> None:
            pass

        def on_close(self, code, msg) -> None:
            done_event.set()

        def on_event(self, response: dict) -> None:
            evt = response.get('type', '')
            if evt == 'response.audio.delta':
                audio_chunks.append(b64.b64decode(response['delta']))
            elif evt == 'session.finished':
                done_event.set()
            elif evt == 'error':
                error_holder.append(str(response))
                done_event.set()

        def on_error(self, message) -> None:
            error_holder.append(str(message))
            done_event.set()

    callback = _Collector()
    tts = QwenTtsRealtime(
        model=QWEN_TTS_MODEL,
        callback=callback,
        url=QWEN_TTS_WS_URL,
    )
    tts.connect()
    tts.update_session(
        voice=voice_id,
        response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
        mode='server_commit',
    )
    tts.append_text(text)
    tts.finish()
    done_event.wait(timeout=120)

    if error_holder:
        raise RuntimeError(f"Qwen TTS error: {error_holder[0][:300]}")
    if not audio_chunks:
        raise RuntimeError("Qwen TTS returned no audio")

    pcm = b''.join(audio_chunks)
    wav_bytes = _pcm_to_wav(pcm, sample_rate=24000)
    out_path.write_bytes(wav_bytes)
    return out_path


def _synthesize_chunk_with_retry(
    text: str, voice_id: str, out_path: Path, retries: int = 3
) -> Path:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return _synthesize_chunk(text, voice_id, out_path)
        except Exception as e:
            last_err = e
            print(f"    [RETRY {attempt}/{retries}] TTS chunk failed: {e}")
            out_path.unlink(missing_ok=True)
            time.sleep(2 ** attempt)  # exponential backoff
    raise RuntimeError(f"TTS chunk failed after {retries} attempts: {last_err}")


def _concat_simple(paths: List[Path], output: Path) -> None:
    concat_list = output.parent / "_tts_concat_list.txt"
    concat_list.write_text("\n".join(f"file '{p}'" for p in paths), encoding="utf-8")
    try:
        subprocess.run(
            [
                FFMPEG_BIN,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "2",
                str(output),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        concat_list.unlink(missing_ok=True)


def _get_trailing_silence_ms(path: Path) -> float:
    """Detect how much silence exists at the end of an audio chunk."""
    result = subprocess.run([
        FFMPEG_BIN, "-i", str(path),
        "-af", "silencedetect=noise=-40dB:d=0.05",
        "-f", "null", "-"
    ], capture_output=True, text=True)
    
    # Parse silence_end from stderr
    matches = re.findall(r"silence_end: ([\d.]+)", result.stderr)
    if not matches:
        return 0.0
    
    # Get total duration
    dur_match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", result.stderr)
    if not dur_match:
        return 0.0
    
    total = (int(dur_match.group(1)) * 3600 +
             int(dur_match.group(2)) * 60 +
             float(dur_match.group(3)))
    last_silence_end = float(matches[-1])
    trailing = (total - last_silence_end) * 1000  # convert to ms
    return max(0.0, trailing)


def _dynamic_crossfade_ms(chunk_a: Path, chunk_b: Path) -> int:
    """
    If the chunk already ends in silence, use 0ms crossfade (let it breathe).
    If it ends abruptly, use a short crossfade to avoid clicks.
    """
    trailing_ms = _get_trailing_silence_ms(chunk_a)
    
    if trailing_ms > 150:
        return 0      # Attenborough-style: natural pause, don't trim it
    elif trailing_ms > 60:
        return 40     # Some natural gap, light crossfade
    else:
        return 80     # Abrupt end (fast speaker), standard crossfade


def _concat_crossfade(paths: List[Path], output: Path, ms: int) -> None:
    if len(paths) == 1:
        subprocess.run(
            [FFMPEG_BIN, "-y", "-i", str(paths[0]), "-codec:a", "libmp3lame", "-q:a", "2", str(output)],
            check=True, capture_output=True,
        )
        return

    cmd = [FFMPEG_BIN, "-y"]
    for p in paths:
        cmd += ["-i", str(p)]

    filters: List[str] = []
    prev = "[0:a]"
    for i in range(1, len(paths)):
        # Dynamically measure crossfade per-junction
        xfade = _dynamic_crossfade_ms(paths[i - 1], paths[i]) / 1000.0
        out_tag = f"[a{i:02d}]"
        if xfade > 0:
            filters.append(
                f"{prev}[{i}:a]acrossfade=d={xfade}:c1=tri:c2=tri{out_tag}"
            )
        else:
            # No crossfade: just concatenate cleanly
            filters.append(f"{prev}[{i}:a]concat=n=2:v=0:a=1{out_tag}")
        prev = out_tag

    cmd += ["-filter_complex", ";".join(filters), "-map", prev,
            "-codec:a", "libmp3lame", "-q:a", "2", str(output)]
    subprocess.run(cmd, check=True, capture_output=True)


def _concat_audio(paths: List[Path], output: Path) -> None:
    if not paths:
        raise RuntimeError("No chunk audio files to concatenate")

    try:
        if CROSSFADE_MS > 0 and len(paths) <= 25:
            _concat_crossfade(paths, output, CROSSFADE_MS)
        else:
            _concat_simple(paths, output)
    except subprocess.CalledProcessError:
        _concat_simple(paths, output)


def generate_long_form_audio(text: str, voice_id: str, output_path: str) -> None:
    """Generate full narration audio with natural chunking + merge."""
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY is missing")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    chunks = split_text_naturally(text, TTS_CHAR_LIMIT)
    print(f"  TTS chunks: {len(chunks)} (max {TTS_CHAR_LIMIT} chars each)")

    tmp_dir = Path(tempfile.mkdtemp(prefix="qwen_tts_chunks_"))
    created: List[Path] = []

    try:
        for i, chunk in enumerate(chunks, 1):
            chunk_path = tmp_dir / f"chunk_{i:03d}.mp3"
            print(f"    [{i}/{len(chunks)}] {len(chunk)} chars")
            _synthesize_chunk_with_retry(chunk, voice_id, chunk_path)
            created.append(chunk_path)

        _concat_audio(created, out)
        if not out.exists() or out.stat().st_size == 0:
            raise RuntimeError("Failed to produce final voiceover file")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen voice clone + TTS test")
    parser.add_argument("--voice", default="", help="Path to reference voice file")
    parser.add_argument("--text", default="This is a quick test of cloned voice generation.")
    parser.add_argument("--out", default="test_output.mp3")
    args = parser.parse_args()

    vid = create_cloned_voice(file_path=args.voice or None)
    generate_long_form_audio(args.text, vid, args.out)
    print(f"Done: {args.out}")
