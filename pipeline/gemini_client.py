"""
gemini_client.py — Minimal Gemini REST wrapper (no SDK dependency).

Mirrors V1's urllib approach. Adds:
  - retries with exponential backoff
  - JSON-mode forcing via responseMimeType
  - per-call timing returned alongside the response
"""
from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .config import GEMINI_API_KEY

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_SSL_CTX = ssl.create_default_context()


class GeminiError(RuntimeError):
    pass


def call(
    model: str,
    prompt: str,
    *,
    temperature: float = 0.7,
    json_mode: bool = False,
    system: Optional[str] = None,
    images: Optional[List[Dict[str, str]]] = None,
    timeout: int = 60,
    max_retries: int = 3,
) -> Tuple[str, float]:
    """Call Gemini and return (text, elapsed_seconds).

    images: list of {"mime_type": "image/jpeg", "data": <base64>} for vision.
    """
    if not GEMINI_API_KEY:
        raise GeminiError("GEMINI_API_KEY not set")

    parts: List[Dict[str, Any]] = [{"text": prompt}]
    for img in images or []:
        parts.append({"inline_data": {"mime_type": img["mime_type"], "data": img["data"]}})

    body: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": float(temperature)},
    }
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}

    url = f"{_BASE}/{model}:generateContent?key={GEMINI_API_KEY}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
                payload = json.loads(r.read().decode("utf-8"))
            elapsed = time.time() - t0
            text = _extract_text(payload)
            return text, elapsed
        except urllib.error.HTTPError as e:
            body_txt = ""
            try:
                body_txt = e.read().decode("utf-8", errors="ignore")[:300]
            except Exception:
                pass
            last_err = GeminiError(f"HTTP {e.code}: {body_txt}")
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            raise last_err
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = GeminiError(f"network: {e}")
            time.sleep(2 ** attempt)
            continue

    raise last_err or GeminiError("unknown failure")


def call_json(model: str, prompt: str, **kwargs: Any) -> Tuple[Dict[str, Any], float]:
    """Call with json_mode=True and parse the response. Raises on parse error."""
    kwargs["json_mode"] = True
    text, elapsed = call(model, prompt, **kwargs)
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        return json.loads(text), elapsed
    except json.JSONDecodeError as e:
        raise GeminiError(f"invalid JSON response: {e}\n--- response ---\n{text[:500]}") from e


def _extract_text(payload: Dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        block = payload.get("promptFeedback", {}).get("blockReason")
        raise GeminiError(f"no candidates (blocked: {block})")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts if "text" in p).strip()
