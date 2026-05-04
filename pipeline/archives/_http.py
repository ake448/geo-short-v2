"""
_http.py — stdlib HTTP helper for archive adapters.

Everything goes through urllib (matches gemini_client.py convention —
no `requests` dep). Handles retries, 429 backoff, and a polite
User-Agent since several archive APIs reject generic python-urllib.
"""
from __future__ import annotations

import json
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

_SSL_CTX = ssl.create_default_context()

USER_AGENT = (
    "UrbanAtlasArchivalHarvester/0.1 "
    "(geography/history shorts pipeline; contact via github)"
)

DEFAULT_TIMEOUT = 20


class ArchiveHTTPError(RuntimeError):
    pass


def get_json(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """GET url, return parsed JSON. Retries 429/5xx with exponential backoff."""
    if params:
        qs = urllib.parse.urlencode(params, doseq=True, quote_via=urllib.parse.quote)
        url = url + ("&" if "?" in url else "?") + qs

    req_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, headers=req_headers, method="GET")

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
                body = r.read()
            try:
                return json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as e:
                raise ArchiveHTTPError(f"non-JSON response from {url}: {e}") from e
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                # LOC's documented block is 1 hour on exceed; we can't outwait
                # that, but we can honor short throttles politely.
                sleep = min(2 ** attempt * 1.5, 30)
                time.sleep(sleep)
                last_err = ArchiveHTTPError(f"HTTP {e.code} on {url}")
                continue
            body_txt = ""
            try:
                body_txt = e.read().decode("utf-8", errors="ignore")[:200]
            except Exception:
                pass
            raise ArchiveHTTPError(f"HTTP {e.code} on {url}: {body_txt}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                last_err = ArchiveHTTPError(f"network error on {url}: {e}")
                continue
            raise ArchiveHTTPError(f"network error on {url}: {e}") from e

    raise last_err or ArchiveHTTPError(f"unknown failure on {url}")


# ── Small shared utilities ───────────────────────────────────────────────────
_YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")


def parse_year(s: str) -> Optional[int]:
    """Extract a plausible 4-digit year from a free-form date string.
    Returns None if nothing 1600-2099 is found."""
    if not s:
        return None
    m = _YEAR_RE.search(str(s))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def strip_html(s: str) -> str:
    """Crude tag stripper for description/rights fields returned with HTML."""
    if not s:
        return ""
    return re.sub(r"<[^>]+>", "", s).strip()
