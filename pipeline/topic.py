"""
topic.py — Classifies a user prompt into one of 4 topic classes and extracts
the primary mappable place, plus a hook formula hint.

Classes:
  city_profile        → "Tokyo", "Albany Georgia"
  region_demographic  → "80% of X live in Y"
  geography_feature   → "this river feeds billions"
  urban_culture       → "the ghettos of X", "favelas of Y"
"""
from __future__ import annotations

import textwrap
from typing import Any, Dict

from .config import TOPIC_MODEL
from .gemini_client import call_json, GeminiError

TOPIC_CLASSES = ("city_profile", "region_demographic", "geography_feature", "urban_culture")

HOOK_FORMULAS = (
    "why_x_live_in_y",          # "Why 80% of Canadians Live in the South"
    "location_paradox",          # "This Country Shouldn't Exist"
    "how_a_did_b",               # "How a B2 Bombed Afghanistan from Missouri"
    "superlative_location",      # "Deadliest Road in the World"
    "difference_between",        # "The Difference Between Dubai & Abu Dhabi"
)

_PROMPT_TEMPLATE = textwrap.dedent(f"""\
    Classify this YouTube Shorts topic and extract its mappable geography.

    User prompt: __PROMPT__

    Return STRICT JSON with these exact keys:

    {{
      "topic_class": one of {list(TOPIC_CLASSES)},
      "place": "specific mappable proper noun (e.g. 'Tokyo, Japan', 'Pine Ridge Reservation, SD'). Never a common word from the prompt.",
      "region": "broader region (e.g. 'East Asia', 'Great Plains, USA')",
      "subject": "what the video is ABOUT (e.g. 'population density', 'street life', 'the Nile')",
      "hook_formula": one of {list(HOOK_FORMULAS)},
      "hook_text": "<=7 word opening hook, ALL CAPS allowed, matches hook_formula",
      "confidence": 0.0-1.0,
      "reasoning": "one sentence explaining the classification"
    }}

    Classification rules:
    - city_profile: the prompt names a specific city or town as the primary subject
    - region_demographic: the prompt is about WHERE people live or WHERE activity happens ("80%", "majority", "nobody", "empty")
    - geography_feature: the prompt is about a river, mountain, strait, canyon, desert, or similar natural feature
    - urban_culture: the prompt mentions ghettos, favelas, slums, megacity life, or class/poverty contrast

    If multiple fit, pick the dominant one and explain in reasoning.
""")


def classify(prompt: str) -> Dict[str, Any]:
    """Returns a dict with topic_class, place, region, subject, hook_formula,
    hook_text, confidence, reasoning, elapsed_sec."""
    prompt_clean = (prompt or "").strip()
    if not prompt_clean:
        raise ValueError("empty prompt")

    full_prompt = _PROMPT_TEMPLATE.replace("__PROMPT__", prompt_clean)
    parsed, elapsed = call_json(
        TOPIC_MODEL, full_prompt, temperature=0.2, max_retries=2
    )

    tc = str(parsed.get("topic_class", "")).strip()
    if tc not in TOPIC_CLASSES:
        raise GeminiError(f"invalid topic_class: {tc!r}")

    hf = str(parsed.get("hook_formula", "")).strip()
    if hf not in HOOK_FORMULAS:
        hf = "superlative_location"

    place = str(parsed.get("place") or "").strip()
    if not place or len(place) < 2:
        raise GeminiError(f"no valid place extracted: {place!r}")

    return {
        "topic_class": tc,
        "place": place,
        "region": str(parsed.get("region") or place).strip(),
        "subject": str(parsed.get("subject") or "").strip(),
        "hook_formula": hf,
        "hook_text": str(parsed.get("hook_text") or "").strip()[:60],
        "confidence": float(parsed.get("confidence") or 0.5),
        "reasoning": str(parsed.get("reasoning") or "").strip(),
        "elapsed_sec": round(elapsed, 2),
    }
