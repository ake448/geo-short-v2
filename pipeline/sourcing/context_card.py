"""
Context-card image sourcing for dynamic_infographic overlays.

This is intentionally narrow: fetch one still image that matches the
script-provided context_visual, cache it under the run's clips directory, and
return the local path to the renderer. Failures return None so cinematics can
fall back to its old blurred-map card.
"""
from __future__ import annotations

import io
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image

from ..archives import AnchorSpec, harvest
from ..archives.models import PoolEntry
from ..config import ARCHIVE_MIN_SCORE, CONTEXT_CARD_VISION_GATE, TEXT_GATE_MODEL
from ..gemini_client import GeminiError, call_json
from ..validate import vision_gate
from . import pexels

USER_AGENT = "UrbanAtlasV2/1.0 (context-card-image-fetcher)"
_FILLER_RE = re.compile(
    r"\b("
    r"black\s+and\s+white|b&w|photo(?:graph)?\s+of|image\s+of|picture\s+of|"
    r"shot\s+of|view\s+of|archival|historic(?:al)?|vintage|old"
    r")\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})s?\b")
_DECADE_RE = re.compile(r"\b(1[6-9]\d0|20\d0)s\b")
_HISTORICAL_RE = re.compile(
    r"\b(archival|historic(?:al)?|vintage|old|black\s+and\s+white|b&w|"
    r"19[0-9]{2}s?|18[0-9]{2}s?|20[0-2][0-9]s?)\b",
    re.IGNORECASE,
)
AMBIGUOUS_US_STATES = {
    "georgia", "washington", "new york", "indiana",
    "jordan", "new mexico", "virginia",
}
AMBIGUOUS_US_CITIES = {
    "athens", "paris", "rome", "moscow", "berlin",
    "manchester", "dublin", "cairo", "lima", "memphis",
}
NEGATIVE_GEO_TERMS = ["caucasus", "tbilisi", "batumi", "soviet", "ussr"]
_US_STATE_TOKENS = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming",
}
_US_STATE_ABBRS = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga",
    "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
    "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
    "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy",
}
_US_STATE_ABBR_BY_NAME = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct",
    "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi",
    "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo",
    "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy",
}
_US_STATE_NAMES_BY_LEN = sorted(_US_STATE_ABBR_BY_NAME, key=len, reverse=True)
_COMMON_WORD_STATE_ABBRS = {"as", "he", "hi", "id", "in", "is", "it", "me", "no", "or", "we"}


def fetch_context_card_image(
    beat: Dict[str, Any],
    spec: Dict[str, Any],
    cache_path: Path,
    archive_only: bool = False,
) -> Optional[Path]:
    """Fetch one still image for a rendered context card or archival beat."""
    if _valid_cached_image(cache_path):
        cached_source = _cached_image_source(cache_path)
        if archive_only and cached_source in {"pexels", "pixabay"}:
            print(
                f"[context-card] rejecting cached foreground stock for archive-only {cache_path.name}",
                flush=True,
            )
            _safe_unlink(cache_path)
            _safe_unlink(cache_path.with_suffix(".json"))
        elif not _cached_stock_vision_passed(cache_path):
            print(
                f"[context-card] rejecting cached ungated foreground stock {cache_path.name}",
                flush=True,
            )
            _safe_unlink(cache_path)
            _safe_unlink(cache_path.with_suffix(".json"))
        else:
            print(f"[context-card] cache hit {cache_path.name}", flush=True)
            return cache_path

    query = _build_query(spec, beat)
    if not query:
        return None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    historical = _looks_historical(spec)
    us_state_topic = _has_us_state_token(_geo_blob(beat, spec))
    if archive_only:
        routes = ("archive",)
    else:
        routes = ("archive", "stock") if (historical or us_state_topic) else ("stock", "archive")
    print(f"[context-card] query={query!r} route={'->'.join(routes)}", flush=True)

    for route in routes:
        try:
            if route == "archive" and _try_archive(beat, spec, query, cache_path):
                return cache_path
            if route == "stock" and _try_stock(query, spec, beat, cache_path):
                return cache_path
        except Exception as exc:
            print(f"[context-card] {route} failed: {exc}", flush=True)
    return None


def _cached_image_source(cache_path: Path) -> str:
    try:
        meta_path = cache_path.with_suffix(".json")
        if not meta_path.exists():
            return ""
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return str(data.get("source") or "").lower().strip()
    except Exception:
        return ""


def _cached_stock_vision_passed(cache_path: Path) -> bool:
    """Foreground stock images must not be reused if they bypassed vision.

    Archive images and legacy images without sidecars are allowed; stock
    sidecars are explicit enough to enforce fail-closed behavior.
    """
    try:
        meta_path = cache_path.with_suffix(".json")
        if not meta_path.exists():
            return True
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    source = str(data.get("source") or "").lower().strip()
    if source in {"pexels", "pixabay"}:
        return str(data.get("vision_gate") or "").lower().strip() == "passed"
    return True


def _valid_thematic_foreground_cache(cache_path: Path) -> bool:
    if not _valid_cached_image(cache_path):
        return False
    if _cached_stock_vision_passed(cache_path):
        print(f"[context-card] cache hit {cache_path.name}", flush=True)
        return True
    print(
        f"[context-card] rejecting cached ungated thematic foreground {cache_path.name}",
        flush=True,
    )
    _safe_unlink(cache_path)
    _safe_unlink(cache_path.with_suffix(".json"))
    return False


def fetch_context_backdrop_image(
    beat: Dict[str, Any],
    spec: Dict[str, Any],
    cache_path: Path,
) -> Optional[Path]:
    """Fetch a loose subject-themed stock backdrop without vision gating.

    Tries the literal subject-leading query first, then falls through to the
    Gemini-derived thematic queries (same as `fetch_thematic_archival_image`)
    so a beat about 'land cleared in 1955' gets a backdrop of bulldozers /
    cleared lots instead of, say, a serene lake-with-bench Pixabay match on
    the loose 'forest landscape' keywords.
    """
    if _valid_cached_image(cache_path):
        print(f"[context-card] backdrop cache hit {cache_path.name}", flush=True)
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    literal_query = _build_query(spec, beat, subject_leading=True)
    if literal_query:
        print(
            f"[context-card] backdrop query={literal_query!r} "
            f"route=stock vision_gate=disabled",
            flush=True,
        )
        try:
            if _try_stock(literal_query, spec, beat, cache_path, use_vision_gate=False):
                return cache_path
        except Exception as exc:
            print(f"[context-card] backdrop stock failed: {exc}", flush=True)

    # Thematic Gemini queries — same path the foreground archival fetch uses.
    # Backdrop is heavily blurred so vision gate stays disabled, but we still
    # want the IMAGE to be conceptually related to the beat's subject (e.g.
    # "bulldozers clearing trees" for a "land cleared in 1955" beat).
    queries = _gemini_thematic_queries(beat, spec)
    for q in queries:
        per_query_spec = dict(spec)
        per_query_spec["context_visual"] = q
        per_query_spec["headline"] = q
        per_query_spec["category"] = ""
        print(
            f"[context-card] backdrop thematic query={q!r} "
            f"route=stock vision_gate=disabled",
            flush=True,
        )
        try:
            if _try_stock(q, per_query_spec, beat, cache_path, use_vision_gate=False):
                return cache_path
        except Exception as exc:
            print(
                f"[context-card] backdrop thematic stock failed for {q!r}: {exc}",
                flush=True,
            )
    return None


def fetch_thematic_archival_image(
    beat: Dict[str, Any],
    spec: Dict[str, Any],
    cache_path: Path,
) -> Optional[Path]:
    """Fetch a stock image whose subject was inferred from the beat's narration
    by a small Gemini call. Used as a foreground archival fallback when the
    archive-only and tight-stock fetches both miss. No vision relevance gate
    (the foreground will be sepia-graded so loose match is fine), but each
    query is tried in order so the first match wins."""
    if _valid_thematic_foreground_cache(cache_path):
        print(f"[context-card] thematic cache hit {cache_path.name}", flush=True)
        return cache_path

    queries = _gemini_thematic_queries(beat, spec)
    if not queries:
        return None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for q in queries:
        # Override spec.context_visual so the downstream _stock_queries rebuilder
        # actually uses our thematic phrase. Without this, _stock_queries
        # silently replaces our query with variants of spec.context_visual.
        per_query_spec = dict(spec)
        per_query_spec["context_visual"] = q
        per_query_spec["headline"] = q
        per_query_spec["category"] = ""
        # Vision gate stays ENABLED here — without it, "aerial view neighborhood"
        # returned a thriving modern suburb for a beat about razed homes, and
        # "smoking landfill" returned a modern third-world dump for a 1962
        # Pennsylvania beat. The vision check is what catches era/place mismatch.
        print(f"[context-card] thematic query={q!r} route=stock", flush=True)
        try:
            if _try_stock(q, per_query_spec, beat, cache_path, use_vision_gate=True):
                return cache_path
        except Exception as exc:
            print(f"[context-card] thematic stock failed for {q!r}: {exc}", flush=True)
    return None


_ARCHIVE_QUERY_PROMPT = (
    "You generate archive search queries for catalog librarians (Library of "
    "Congress, Wikimedia Commons, Internet Archive, Digital Library of Georgia). "
    "These catalogs are indexed by formal subject headings, place names with "
    "state/parens disambiguation, exact dates, and noun-led titles — NOT "
    "narrative phrases.\n\n"
    "Beat narration: {narration}\n"
    "Beat subject:   {subject}\n"
    "Beat caption:   {caption}\n"
    "Beat geo:       {geo}\n"
    "Era hint:       {era}\n\n"
    "BEFORE writing queries, IDENTIFY any famous namesakes that could pollute "
    "the search results, and write queries that explicitly disambiguate. "
    "Examples of dangerous collisions:\n"
    "  • 'Andersonville' → famous Civil War POW camp in Sumter County GA. If "
    "    the beat is about a different Andersonville (e.g. the 1801 town near "
    "    Lake Hartwell on the Tugaloo/Savannah), DO NOT issue queries that "
    "    would surface POW/prison/Civil War content. Add 'Hart County' or "
    "    'Tugaloo' or '1801 frontier' to disambiguate.\n"
    "  • 'Athens' → Ohio + Greece + many others. Add state.\n"
    "  • 'Springfield' → 30+ towns. Always add state.\n"
    "  • 'Forsyth County' → exists in both GA and NC. Add state.\n"
    "  • 'Plymouth' → MA + UK + Montserrat. Add disambiguator.\n"
    "  • 'Centralia' → PA mine fire is famous; avoid 'Centralia WA' results.\n"
    "If the beat names a small/obscure place that shares its name with a "
    "much-more-famous landmark, add a county, river, or year to EVERY query "
    "and avoid words associated with the famous one (prison, war, POW, "
    "battle, etc.) unless those words actually appear in the beat narration.\n\n"
    "Return 4–6 short (2–6 word) queries optimized for ARCHIVE catalog search. "
    "Each query should match one or more of these styles:\n"
    "  • Place + state + medium: 'Andersonville Hart County Georgia map'\n"
    "  • Subject heading style: 'Centralia Pennsylvania mine fire'\n"
    "  • Date + place: 'Buford Dam 1956 construction'\n"
    "  • Named landmark only: 'Hartwell Dam'\n"
    "  • Broader regional category: 'Tugaloo River frontier', 'upcountry Georgia'\n"
    "Avoid: filler words ('archival', 'historical', 'old', 'vintage', "
    "'photograph of'), sentence fragments, decade-only dates ('1950s' — use "
    "'1955' instead), narrative phrasing, and any word strongly associated "
    "with a famous-but-wrong namesake.\n\n"
    'Return ONLY a JSON array of strings, e.g.: ["Andersonville Hart County Georgia","Tugaloo River 1801","Lake Hartwell relocation 1955","Hartwell Dam construction"]'
)


def _gemini_archive_queries(
    beat: Dict[str, Any],
    spec: Dict[str, Any],
    max_queries: int = 6,
) -> List[str]:
    visual = beat.get("visual") if isinstance(beat.get("visual"), dict) else {}
    anchor = beat.get("anchor_spec") if isinstance(beat.get("anchor_spec"), dict) else {}
    narration = str(beat.get("narration") or "").strip()
    subject = str(anchor.get("subject") or visual.get("subject") or spec.get("context_visual") or "").strip()
    caption = str(beat.get("caption_text") or spec.get("headline") or "").strip()
    geo = str(visual.get("geo") or spec.get("geo") or "").strip()
    era_raw = anchor.get("era") if isinstance(anchor.get("era"), list) else None
    era = ""
    if era_raw:
        era = f"{era_raw[0]}-{era_raw[-1]}" if len(era_raw) >= 2 else str(era_raw[0])
    if not (narration or subject or caption):
        return []

    prompt = _ARCHIVE_QUERY_PROMPT.format(
        narration=narration[:300] or "(none)",
        subject=subject[:200] or "(none)",
        caption=caption[:80] or "(none)",
        geo=geo[:80] or "(none)",
        era=era or "(none)",
    )
    try:
        parsed, _ = call_json(TEXT_GATE_MODEL, prompt, temperature=0.3, max_retries=1)
    except GeminiError as exc:
        print(f"[context-card] archive-query gemini failed: {exc}", flush=True)
        return []
    except Exception as exc:
        print(f"[context-card] archive-query unexpected: {exc}", flush=True)
        return []

    raw_list: List[str] = []
    if isinstance(parsed, list):
        raw_list = [str(x) for x in parsed if x]
    elif isinstance(parsed, dict):
        for key in ("queries", "query", "results", "items"):
            v = parsed.get(key)
            if isinstance(v, list):
                raw_list = [str(x) for x in v if x]
                break

    out: List[str] = []
    seen: set = set()
    for q in raw_list:
        cleaned = re.sub(r"[^A-Za-z0-9 ,'-]+", " ", q)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned[:60])
        if len(out) >= max_queries:
            break
    return out


_THEMATIC_QUERY_PROMPT = (
    "You generate stock-photo search queries for a YouTube Shorts video beat. "
    "The previous archival/specific search returned nothing — now we want the "
    "BEST thematically-related stock image we can find.\n\n"
    "Beat narration: {narration}\n"
    "Beat subject:   {subject}\n"
    "Beat caption:   {caption}\n"
    "Beat geo:       {geo}\n\n"
    "Return 4–6 short stock-photo search queries (each 2–4 words). Focus on "
    "concrete VISUAL CONCEPTS: nouns, settings, objects, atmosphere, materials. "
    "Lean evocative — if the beat is about underground heat, queries like "
    "'lava cracks', 'glowing embers', 'smoking ground', 'coal mine fire' beat "
    "abstract terms like 'underground heat'. Order from most-likely-to-find-a-"
    "good-match to most-generic. Avoid named places (those already failed).\n\n"
    'Return ONLY a JSON array of strings, e.g.: ["lava cracks","glowing embers","smoking earth","coal mine fire"]'
)


def _gemini_thematic_queries(
    beat: Dict[str, Any],
    spec: Dict[str, Any],
    max_queries: int = 6,
) -> List[str]:
    visual = beat.get("visual") if isinstance(beat.get("visual"), dict) else {}
    narration = str(beat.get("narration") or "").strip()
    subject = str(visual.get("subject") or spec.get("context_visual") or "").strip()
    caption = str(beat.get("caption_text") or spec.get("headline") or "").strip()
    geo = str(visual.get("geo") or spec.get("geo") or "").strip()
    if not (narration or subject or caption):
        return []

    prompt = _THEMATIC_QUERY_PROMPT.format(
        narration=narration[:300] or "(none)",
        subject=subject[:200] or "(none)",
        caption=caption[:80] or "(none)",
        geo=geo[:80] or "(none)",
    )
    fallback_queries = _deterministic_thematic_queries(beat, spec)
    try:
        parsed, _ = call_json(TEXT_GATE_MODEL, prompt, temperature=0.4, max_retries=1)
    except GeminiError as exc:
        print(f"[context-card] thematic-query gemini failed: {exc}", flush=True)
        return fallback_queries[:max_queries]
    except Exception as exc:
        print(f"[context-card] thematic-query unexpected: {exc}", flush=True)
        return fallback_queries[:max_queries]

    raw_list: List[str] = []
    if isinstance(parsed, list):
        raw_list = [str(x) for x in parsed if x]
    elif isinstance(parsed, dict):
        for key in ("queries", "query", "results", "items"):
            v = parsed.get(key)
            if isinstance(v, list):
                raw_list = [str(x) for x in v if x]
                break

    out: List[str] = []
    seen: set = set()
    for q in raw_list:
        cleaned = re.sub(r"[^A-Za-z0-9 ,'-]+", " ", q)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned[:60])
        if len(out) >= max_queries:
            break
    for q in fallback_queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            out.append(q)
        if len(out) >= max_queries:
            break
    return out


def _deterministic_thematic_queries(beat: Dict[str, Any], spec: Dict[str, Any]) -> List[str]:
    visual = beat.get("visual") if isinstance(beat.get("visual"), dict) else {}
    text = " ".join(
        str(v or "")
        for v in (
            spec.get("context_visual"),
            spec.get("headline"),
            spec.get("category"),
            beat.get("caption_text"),
            beat.get("narration"),
            visual.get("subject"),
        )
    ).lower()
    queries: List[str] = []
    patterns = [
        (r"\b(lead poisoning|lead contamination|blood lead|children.*lead)\b", [
            "lead contamination warning",
            "blood test laboratory",
            "environmental health warning",
            "toxic waste warning sign",
        ]),
        (r"\b(chat pile|mine waste|mining waste|tailings)\b", [
            "mine tailings landscape",
            "mining waste piles",
            "toxic mine waste",
        ]),
        (r"\b(buyout|relocation|evacuation|condemned)\b", [
            "abandoned town street",
            "boarded up houses",
            "empty neighborhood",
        ]),
        (r"\b(zip|postal|post office|mailbox)\b", [
            "abandoned post office",
            "mailbox empty street",
            "postal service building",
        ]),
        (r"\b(fire|smoke|burning|landfill)\b", [
            "smoking ground",
            "burning landfill",
            "coal mine fire",
        ]),
        (r"\b(demolished|razed|row house|empty lots?)\b", [
            "demolished neighborhood",
            "abandoned row houses",
            "empty lots street",
        ]),
    ]
    for pattern, items in patterns:
        if re.search(pattern, text):
            queries.extend(items)
    cleaned = re.sub(
        r"\b(infographic|dynamic|showing|showcasing|photo|image|of|the|a|an|"
        r"for|with|near|during|from)\b",
        " ",
        str(spec.get("context_visual") or visual.get("subject") or ""),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[^A-Za-z0-9 ,'-]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
    if cleaned:
        queries.append(cleaned[:60])
    out: List[str] = []
    seen: set[str] = set()
    for q in queries:
        q = re.sub(r"\s+", " ", q).strip(" ,.-")
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            out.append(q)
    return out


def _valid_cached_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 1024:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        path.unlink(missing_ok=True)
        return False


def _build_query(
    spec: Dict[str, Any],
    beat: Optional[Dict[str, Any]] = None,
    *,
    subject_leading: bool = False,
) -> str:
    context = _clean_query(str(spec.get("context_visual") or ""))
    headline = _clean_query(str(spec.get("headline") or ""))
    category = _clean_query(str(spec.get("category") or ""))
    geo = _clean_query(str(spec.get("geo") or ""))

    parts: List[str] = []
    seen = set()
    order = (context, headline, category, geo)
    if subject_leading:
        order = (context, geo, headline, category)
    for part in order:
        key = part.lower()
        if part and key not in seen:
            parts.append(part)
            seen.add(key)
    query = " ".join(parts)[:180].strip()
    if _needs_us_geo_disambiguation(beat or {}, spec):
        query = _harden_us_query(query)
    return query


def _clean_query(value: str) -> str:
    value = _FILLER_RE.sub(" ", value or "")
    value = re.sub(r"[^A-Za-z0-9,.' -]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" ,.-")
    return value


def _looks_historical(spec: Dict[str, Any]) -> bool:
    blob = " ".join(str(spec.get(k) or "") for k in ("context_visual", "headline", "support"))
    return bool(_HISTORICAL_RE.search(blob))


def _geo_blob(beat: Optional[Dict[str, Any]], spec: Dict[str, Any]) -> str:
    beat = beat or {}
    visual = beat.get("visual") if isinstance(beat.get("visual"), dict) else {}
    anchor = beat.get("anchor_spec") if isinstance(beat.get("anchor_spec"), dict) else {}
    script_ctx = beat.get("_script_context") if isinstance(beat.get("_script_context"), dict) else {}
    values = [
        spec.get("geo"), visual.get("geo"), anchor.get("geo"),
        script_ctx.get("place"), script_ctx.get("region"),
    ]
    return " ".join(str(v or "") for v in values)


def _contains_phrase(blob: str, phrases: Iterable[str]) -> bool:
    normalized = f" {re.sub(r'[^a-z0-9]+', ' ', (blob or '').lower())} "
    return any(f" {phrase.lower()} " in normalized for phrase in phrases)


def _has_us_state_token(blob: str) -> bool:
    if _contains_phrase(blob, _US_STATE_TOKENS):
        return True
    tokens = set(re.findall(r"\b[a-z]{2}\b", (blob or "").lower()))
    return bool(tokens & _US_STATE_ABBRS)


def _state_abbrev(state: str) -> Optional[str]:
    s = (state or "").strip().lower().replace(".", "")
    if not s:
        return None
    if s in _US_STATE_ABBRS:
        return s
    return _US_STATE_ABBR_BY_NAME.get(s)


def _find_us_states(text: str) -> set[str]:
    """Return canonical two-letter state abbreviations explicitly named in text."""
    blob = f" {re.sub(r'[^a-z0-9]+', ' ', (text or '').lower())} "
    found: set[str] = set()
    for name in _US_STATE_NAMES_BY_LEN:
        if f" {name} " in blob:
            found.add(_US_STATE_ABBR_BY_NAME[name])
    for abbr in set(re.findall(r"\b[a-z]{2}\b", blob)) & _US_STATE_ABBRS:
        if abbr not in _COMMON_WORD_STATE_ABBRS:
            found.add(abbr)
    return found


def _extract_us_state(target_geo: str) -> Optional[str]:
    states = _find_us_states(target_geo)
    if len(states) == 1:
        return next(iter(states))
    return None


def _extract_geo_city(target_geo: str) -> str:
    city = str(target_geo or "").split(",", 1)[0]
    city = re.sub(r"[^A-Za-z0-9 -]+", " ", city)
    city = re.sub(r"\s+", " ", city).strip().lower()
    if len(city) < 4 or city in {"lake", "county", "united states"}:
        return ""
    return city


def _metadata_geo_matches(asset_title: str, asset_record_url: str, target_geo: str) -> bool:
    """Reject obvious US state mismatches via metadata text alone."""
    target_state = _extract_us_state(target_geo)
    if not target_state:
        return True
    metadata_text = f"{asset_title or ''} {asset_record_url or ''}"
    metadata_states = _find_us_states(metadata_text)
    other_states = metadata_states - {target_state}
    if other_states:
        return False

    target_city = _extract_geo_city(target_geo)
    if target_city and target_state in metadata_states:
        metadata_lower = metadata_text.lower()
        blob = f" {re.sub(r'[^a-z0-9]+', ' ', metadata_lower)} "
        state_names = [name for name, abbr in _US_STATE_ABBR_BY_NAME.items() if abbr == target_state]
        state_pattern = "|".join([re.escape(n) for n in state_names] + [re.escape(target_state)])
        for m in re.finditer(rf"\b([a-z][a-z .'-]{{2,40}}),\s*(?:{state_pattern})\b", metadata_lower):
            city = re.sub(r"[^a-z0-9]+", " ", m.group(1)).strip()
            if city and target_city not in city:
                return False
        has_state_pair = any(f" {name} " in blob for name in state_names) or f" {target_state} " in blob
        if has_state_pair and f" {target_city} " not in blob:
            return False

    return True


def _archive_asset_matches_beat(asset_title: str, beat: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    """Reject obvious namesake/catalog collisions from title metadata."""
    title_blob = f" {re.sub(r'[^a-z0-9]+', ' ', (asset_title or '').lower())} "
    beat_blob = " ".join(
        str(v or "")
        for v in (
            beat.get("narration"),
            (beat.get("anchor_spec") or {}).get("subject") if isinstance(beat.get("anchor_spec"), dict) else "",
            spec.get("context_visual"),
            spec.get("support"),
        )
    ).lower()

    wrong_namesake_terms = (
        " prison ", " pow ", " prisoner ", " civil war ",
        " confederate ", " soldiers ", " legion ", " war department ",
    )
    if any(term in title_blob for term in wrong_namesake_terms):
        if not any(term.strip() in beat_blob for term in wrong_namesake_terms):
            return False

    if " plaque " in title_blob and "plaque" not in beat_blob:
        return False

    return True


def _is_us_topic(beat: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    script_ctx = beat.get("_script_context") if isinstance(beat.get("_script_context"), dict) else {}
    topic_class = str(script_ctx.get("topic_class") or "").lower()
    region = str(script_ctx.get("region") or "")
    place = str(script_ctx.get("place") or "")
    script_blob = " ".join([topic_class, region, place])
    if re.search(r"\b(united states|usa|u\.s\.|southeastern united states)\b", script_blob, re.I):
        return True
    return _has_us_state_token(_geo_blob(beat, spec))


def _needs_us_geo_disambiguation(beat: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    if not _is_us_topic(beat, spec):
        return False
    geo_blob = _geo_blob(beat, spec)
    return (
        _contains_phrase(geo_blob, AMBIGUOUS_US_STATES)
        or _contains_phrase(geo_blob, AMBIGUOUS_US_CITIES)
    )


def _harden_us_query(query: str) -> str:
    if not query:
        return query
    if re.search(r"\b(usa|united states|u\.s\.|us state)\b", query, re.I):
        return query
    return f"{query} USA"


def _era_from_spec(spec: Dict[str, Any]) -> Optional[List[int]]:
    blob = " ".join(str(spec.get(k) or "") for k in ("context_visual", "headline", "support"))
    decade = _DECADE_RE.search(blob)
    if decade:
        start = int(decade.group(1))
        return [start, start + 9]
    years = [int(m.group(1)) for m in _YEAR_RE.finditer(blob)]
    if years:
        y = years[0]
        return [max(1600, y - 3), y + 3]
    return None


def _archive_queries(query: str, spec: Dict[str, Any]) -> List[str]:
    context = _clean_query(str(spec.get("context_visual") or ""))
    geo = _clean_query(str(spec.get("geo") or ""))
    headline = _clean_query(str(spec.get("headline") or ""))
    candidates = [
        context,
        f"{context} {geo}".strip(),
        f"{headline} {geo}".strip(),
        query,
    ]
    out: List[str] = []
    seen = set()
    for q in candidates:
        q = re.sub(r"\s+", " ", q).strip()
        key = q.lower()
        if q and key not in seen:
            out.append(q)
            seen.add(key)
    return out[:3]


def _state_name_from_geo(geo: str) -> str:
    state = _extract_us_state(geo)
    if not state:
        return ""
    for name, abbr in _US_STATE_ABBR_BY_NAME.items():
        if abbr == state:
            return name.title()
    return ""


def _expand_geo_state_name(geo: str) -> str:
    expanded = _clean_query(geo)
    for name, abbr in _US_STATE_ABBR_BY_NAME.items():
        expanded = re.sub(rf"\b{re.escape(abbr.upper())}\b", name.title(), expanded)
    return expanded


def _deterministic_archive_queries(
    beat: Dict[str, Any],
    spec: Dict[str, Any],
    query: str,
) -> List[str]:
    """Build catalog-shaped archive queries without LLM access."""
    anchor = beat.get("anchor_spec") if isinstance(beat.get("anchor_spec"), dict) else {}
    subject = _clean_query(str(anchor.get("subject") or ""))
    context = _clean_query(str(spec.get("context_visual") or ""))
    geo = _clean_query(str(spec.get("geo") or anchor.get("geo") or ""))
    category = _clean_query(str(spec.get("category") or ""))
    state_name = _state_name_from_geo(geo)
    expanded_geo = _expand_geo_state_name(geo)
    era = anchor.get("era") if isinstance(anchor.get("era"), list) else None
    year = ""
    if era:
        year = str(era[0])
    else:
        m = _YEAR_RE.search(" ".join([subject, context, str(spec.get("support") or "")]))
        if m:
            year = m.group(1)

    candidates: List[str] = []
    if subject:
        candidates.append(subject)
        no_year = re.sub(r"\b(1[6-9]\d{2}|20\d{2})s?\b", "", subject).strip()
        if no_year and no_year != subject:
            candidates.append(no_year)
        if state_name and state_name.lower() not in subject.lower():
            candidates.append(f"{subject} {state_name}")
    if category and geo:
        candidates.append(f"{category} {geo}")
    if category and state_name:
        candidates.append(f"{category} {state_name}")
    landmark_blob = " ".join([subject, context, category])
    for match in re.finditer(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\s+Dam)\b", landmark_blob):
        landmark = _clean_query(match.group(1))
        if landmark:
            candidates.append(landmark)
            if state_name:
                candidates.append(f"{landmark} {state_name}")
    if expanded_geo:
        candidates.append(expanded_geo)
    if expanded_geo and year:
        candidates.append(f"{expanded_geo} {year}")
    if expanded_geo and state_name and year:
        candidates.append(f"{expanded_geo.replace(',', '')} {year}")

    candidates.extend(_archive_queries(query, spec))

    out: List[str] = []
    seen = set()
    for q in candidates:
        q = re.sub(r"\s+", " ", q).strip(" ,.-")
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            out.append(q)
        if len(out) >= 6:
            break
    return out


def _stock_queries(query: str, spec: Dict[str, Any],
                   beat: Optional[Dict[str, Any]] = None) -> List[str]:
    context = _clean_query(str(spec.get("context_visual") or ""))
    geo = _clean_query(str(spec.get("geo") or ""))
    category = _clean_query(str(spec.get("category") or ""))
    thematic = _deterministic_thematic_queries(beat or {}, spec)
    candidates = [
        context,
        re.sub(r"\b(18|19|20)\d{2}s?\b", "", context).strip(),
        f"{context} {geo}".strip(),
        f"{category} {geo}".strip(),
        query,
    ]
    candidates.extend(thematic[:4])
    out: List[str] = []
    seen = set()
    harden = _needs_us_geo_disambiguation(beat or {}, spec)
    for q in candidates:
        q = re.sub(r"\s+", " ", q).strip()
        if harden:
            q = _harden_us_query(q)
        key = q.lower()
        if q and key not in seen:
            out.append(q)
            seen.add(key)
    return out[:4]


def _try_archive(beat: Dict[str, Any], spec: Dict[str, Any], query: str, cache_path: Path) -> bool:
    anchor_raw = beat.get("anchor_spec") or {}
    beat_id = beat.get("beat_id") if isinstance(beat.get("beat_id"), int) else 0
    anchor_queries: List[str] = []
    for q in anchor_raw.get("queries") or []:
        cleaned = _clean_query(str(q))
        if cleaned:
            anchor_queries.append(cleaned)

    # Better archive prompting: ask Gemini for catalog-librarian-shaped
    # queries instead of feeding archive APIs the raw narrative phrase
    # ("archival sketch or early map showing the layout of Andersonville").
    # Catalogs index by subject heading + place + date, not narrative.
    gemini_archive_qs = _gemini_archive_queries(beat, spec)
    combined_qs: List[str] = []
    seen: set = set()
    deterministic_qs = _deterministic_archive_queries(beat, spec, query)
    for q in (deterministic_qs + anchor_queries[:3] + gemini_archive_qs):
        key = q.lower().strip()
        if key and key not in seen:
            seen.add(key)
            combined_qs.append(q)
        if len(combined_qs) >= 6:
            break
    if gemini_archive_qs:
        print(
            f"[context-card] archive gemini queries={gemini_archive_qs[:6]}",
            flush=True,
        )

    anchor = AnchorSpec(
        beat_id=beat_id,
        anchor_type=str(anchor_raw.get("anchor_type") or "stat"),
        subject=_clean_query(str(anchor_raw.get("subject") or spec.get("context_visual") or spec.get("headline") or query)),
        geo=str(spec.get("geo") or anchor_raw.get("geo") or ""),
        era=anchor_raw.get("era") or _era_from_spec(spec),
        queries=combined_qs or _archive_queries(query, spec),
        allow_substitute=True,
    )
    manifest = harvest([anchor])
    entries = _rank_archive_entries(manifest.entries)
    print(
        f"[context-card] archive hits={len(entries)} stats={json.dumps(manifest.stats, sort_keys=True)}",
        flush=True,
    )
    best_score = max((float(e.relevance_score or 0.0) for e in entries), default=0.0)
    eligible = [e for e in entries if float(e.relevance_score or 0.0) >= ARCHIVE_MIN_SCORE]
    if entries and not eligible:
        print(
            f"[context-card] archive best_score={best_score / 100:.2f} "
            f"below {ARCHIVE_MIN_SCORE / 100:.2f} - falling to stock",
            flush=True,
        )
    for entry in eligible:
        asset = entry.asset
        url = asset.image_url or asset.thumbnail_url
        if not url:
            continue
        target_geo = str(spec.get("geo") or (beat.get("visual") or {}).get("geo") or "")
        if not _metadata_geo_matches(asset.title, asset.record_url, target_geo):
            print(
                f"[context-card] archive reject metadata_geo title={asset.title!r} "
                f"target={target_geo!r}",
                flush=True,
            )
            continue
        if not _archive_asset_matches_beat(asset.title, beat, spec):
            print(
                f"[context-card] archive reject namesake title={asset.title!r}",
                flush=True,
            )
            continue
        if _download_image(url, cache_path):
            _write_credit(cache_path, {
                "source": asset.source,
                "id": asset.source_record_id,
                "title": asset.title,
                "record_url": asset.record_url,
                "date": asset.date,
                "year": asset.year,
                "creator": asset.creator,
                "rights": asset.rights,
                "media_hint": asset.media_hint,
                "evidence_status": entry.evidence_status,
                "score": entry.relevance_score,
            })
            print(
                f"[context-card] archive image {asset.source}:{asset.source_record_id} "
                f"score={entry.relevance_score}",
                flush=True,
            )
            return True
    return False


def _rank_archive_entries(entries: Iterable[PoolEntry]) -> List[PoolEntry]:
    media_rank = {"photo": 0, "postcard": 1, "aerial": 2, "unknown": 3, "map": 4, "document": 5}
    return sorted(
        entries,
        key=lambda e: (media_rank.get(e.asset.media_hint, 4), -float(e.relevance_score or 0)),
    )


_STOCK_VISION_BUDGET_FOREGROUND = 4
_STOCK_VISION_BUDGET_BACKDROP = 2


def _try_stock(
    query: str,
    spec: Dict[str, Any],
    beat: Dict[str, Any],
    cache_path: Path,
    *,
    use_vision_gate: bool = True,
) -> bool:
    vision_calls = 0
    vision_budget = _STOCK_VISION_BUDGET_FOREGROUND if use_vision_gate else _STOCK_VISION_BUDGET_BACKDROP
    for q in _stock_queries(query, spec, beat):
        items = pexels.search_photos(q, per_page=6, negative_terms=NEGATIVE_GEO_TERMS)
        print(f"[context-card] stock photo query={q!r} hits={len(items)}", flush=True)
        for item in items:
            if _item_has_negative_geo(item):
                print(
                    f"[context-card] stock reject {item.get('source')}:{item.get('id')} negative geo tags",
                    flush=True,
                )
                continue
            url = item.get("url")
            if not url:
                continue
            tmp_path = cache_path.with_name(f".{cache_path.stem}.candidate.jpg")
            if _download_image(str(url), tmp_path):
                if use_vision_gate and CONTEXT_CARD_VISION_GATE and vision_calls < vision_budget:
                    prompt = _stock_vision_prompt(spec, beat)
                    verdict = vision_gate.relevance_check(tmp_path, prompt, str(url))
                    if not verdict.get("cached"):
                        vision_calls += 1
                    status = "passed" if verdict["passed"] else "rejected"
                    print(
                        f"[context-card] vision gate {status} "
                        f"{item.get('source')}:{item.get('id')} cached={verdict.get('cached')} "
                        f"reason={verdict.get('reason')}",
                        flush=True,
                    )
                    if not verdict["passed"]:
                        _safe_unlink(tmp_path)
                        continue
                elif use_vision_gate and CONTEXT_CARD_VISION_GATE:
                    print(
                        f"[context-card] vision gate budget exhausted ({vision_budget} calls); "
                        f"rejecting ungated stock candidate {item.get('source')}:{item.get('id')}",
                        flush=True,
                    )
                    _safe_unlink(tmp_path)
                    continue
                _safe_replace(tmp_path, cache_path)
                _write_credit(cache_path, {
                    "source": item.get("source"),
                    "id": item.get("id"),
                    "credit": item.get("credit"),
                    "query": q,
                    "tags": item.get("tags") or "",
                    "vision_gate": "passed" if (use_vision_gate and CONTEXT_CARD_VISION_GATE) else "disabled",
                })
                print(f"[context-card] stock image {item.get('source')}:{item.get('id')}", flush=True)
                return True
            _safe_unlink(tmp_path)
    return False


def _item_has_negative_geo(item: Dict[str, Any]) -> bool:
    blob = " ".join(str(item.get(k) or "") for k in ("tags", "credit", "url"))
    blob_l = blob.lower()
    return any(term in blob_l for term in NEGATIVE_GEO_TERMS)


def _stock_vision_prompt(spec: Dict[str, Any], beat: Dict[str, Any]) -> str:
    claim = str(spec.get("support") or spec.get("context_visual") or "").strip()
    geo = str(spec.get("geo") or (beat.get("visual") or {}).get("geo") or "").strip()
    script_ctx = beat.get("_script_context") if isinstance(beat.get("_script_context"), dict) else {}
    topic = str(script_ctx.get("subject") or beat.get("narration") or "").strip()
    narration = str(beat.get("narration") or "").strip()
    anchor = beat.get("anchor_spec") if isinstance(beat.get("anchor_spec"), dict) else {}
    era_raw = anchor.get("era") if isinstance(anchor.get("era"), list) else None
    era_line = ""
    if era_raw:
        era_line = (
            f"3. The image's apparent era must match {era_raw[0]}-{era_raw[-1]}. "
            "Modern smartphones, contemporary cars, current fashion, "
            "modern signage, solar panels — all NO if the beat is historical. "
            "Sepia/black-and-white/film-grain styling preferred for pre-1980 beats.\n"
        )
    return (
        "This image is being considered as illustration for a beat about\n"
        f'"{claim or topic}" set in "{geo}".\n'
        f'Beat narration: "{narration[:200]}"\n\n'
        "HARD REJECT — answer NO only if the image DIRECTLY CONTRADICTS the\n"
        "narration. We are looking for 'good enough', not perfection.\n\n"
        "1. CONTRADICTION CHECK (the only hard reject):\n"
        "   Does the image visually REVERSE the narration's claim?\n"
        "   - A modern thriving suburb CONTRADICTS 'homes were razed' → NO\n"
        "   - A modern third-world landfill CONTRADICTS '1962 Pennsylvania' → NO\n"
        "   - An intact highway CONTRADICTS 'demolished neighborhood' → NO\n"
        "   - A colorful graffiti road CONTRADICTS 'charred desolate landscape' → NO\n"
        "   - A Southeast Asian dump CONTRADICTS a beat set in Pennsylvania → NO\n"
        "   If the image merely doesn't match perfectly but doesn't CONTRADICT\n"
        "   the narration, answer YES. A generic moody landscape for a 'desolate\n"
        "   town' beat is acceptable — a lively populated town is not.\n"
        "2. WRONG LOCATION (hard reject only when obvious):\n"
        f'   If the image clearly shows a DIFFERENT CONTINENT or COUNTRY than\n'
        f'   "{geo}", answer NO. Do NOT reject just because you can\'t confirm\n'
        "   the exact state — only reject when it's visibly wrong.\n"
        f"{era_line}"
        "\n"
        "Be GENEROUS. A usable-but-imperfect image is far better than no image.\n"
        "Only reject when the image would actively mislead the viewer.\n\n"
        "Answer only YES or NO with one short reason."
    )


def _safe_unlink(path: Path) -> None:
    for _ in range(3):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(0.1)


def _safe_replace(src: Path, dst: Path) -> None:
    for attempt in range(3):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.1)


def _download_image(url: str, out_path: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=35) as resp:
            raw = resp.read(20 * 1024 * 1024)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        if img.width < 240 or img.height < 160:
            return False
        img.save(out_path, "JPEG", quality=92)
        return out_path.exists() and out_path.stat().st_size > 1024
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _write_credit(image_path: Path, data: Dict[str, Any]) -> None:
    try:
        image_path.with_suffix(".json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass
