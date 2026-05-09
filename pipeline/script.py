"""
script.py — Template-driven beat-plan generation.

Flow:
  1. topic.classify(prompt) picks a topic class + extracts place/region/subject/hook.
  2. TEMPLATES[topic_class] provides the beat skeleton (what visual types, in what order).
  3. Gemini Flash fills narration + specific visual briefs per beat, respecting pacing rules.

Output schema (stable contract for sourcing + render layers):

{
  "topic_class": "...",
  "place": "...", "region": "...", "subject": "...",
  "title": "...", "hook_text": "...",
  "total_duration_sec": float,
  "beats": [
    {
      "beat_id": 1,
      "role": "HOOK|ESTABLISH|FACT|CONTRAST|PAYOFF|EXIT",
      "narration": "...",
      "duration_sec": 5.0,
      "visual": {
        "kind": "drone_aerial|street_level|landmark|map_zoom|map_animation|
                 density_infographic|dynamic_infographic|wikipedia_photo|archival_annotated|stock_bland",
        "subject": "what viewer sees (specific)",
        "geo": "strict place name for geo-validation",
        "strictness": "strict|loose",
        "queries": ["youtube search terms", ...]
      },
      "caption_text": "ALL CAPS 2-5 words",
      "voice": "david|guy|simon|viral",
      "overlay": {"kind": "...", "data": {...}} | null
    }
  ],
  "youtube_metadata": { "title": "...", "description": "...", "tags": [...] }
}
"""
from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, List

from .archives import AnchorSpec
from .config import (
    BEAT_MAX_SEC, BEAT_MIN_SEC, BEATS_PER_VIDEO, SCRIPT_MODEL,
    TARGET_RUNTIME_SEC,
)
from .gemini_client import call_json, GeminiError
from .topic import classify

# Archival-primary visuals need a concrete source object. Without one the
# renderer has to guess, which is how abstract stats become unavailable cards.
_ARCHIVAL_ARTIFACT_TERMS = (
    "aerial",
    "article",
    "clipping",
    "construction",
    "document",
    "map",
    "newspaper",
    "photo",
    "photograph",
    "plaque",
    "postcard",
)

# ── Templates ─────────────────────────────────────────────────────────────────
# Each template is a list of beat "slots" — visual type + role. The LLM fills
# narration and specific subjects; the template enforces structure.

VisualKind = str
BeatSlot = Dict[str, Any]

TEMPLATES: Dict[str, List[BeatSlot]] = {
    "city_profile": [
        {"role": "HOOK",       "kinds": ["map_zoom", "landmark"],       "strict": "loose"},
        {"role": "ESTABLISH",  "kinds": ["drone_aerial"],               "strict": "strict"},
        {"role": "FACT",       "kinds": ["street_level", "landmark"],   "strict": "strict"},
        {"role": "FACT",       "kinds": ["archival_annotated", "street_level"], "strict": "strict"},
        {"role": "CONTRAST",   "kinds": ["dynamic_infographic", "density_infographic"], "strict": "loose"},
        {"role": "FACT",       "kinds": ["drone_aerial", "street_level"], "strict": "strict"},
        {"role": "FACT",       "kinds": ["landmark", "street_level"],   "strict": "strict"},
        {"role": "CONTRAST",   "kinds": ["dynamic_infographic", "map_zoom"], "strict": "loose"},
        {"role": "FACT",       "kinds": ["archival_annotated", "wikipedia_photo"], "strict": "strict"},
        {"role": "FACT",       "kinds": ["drone_aerial", "street_level"], "strict": "strict"},
        {"role": "PAYOFF",     "kinds": ["drone_aerial", "landmark"],   "strict": "strict"},
        {"role": "EXIT",       "kinds": ["map_zoom"],                   "strict": "loose"},
    ],
    "region_demographic": [
        {"role": "HOOK",       "kinds": ["map_animation"],              "strict": "loose"},
        {"role": "ESTABLISH",  "kinds": ["map_zoom"],                   "strict": "loose"},
        {"role": "FACT",       "kinds": ["dynamic_infographic", "density_infographic"], "strict": "loose"},
        {"role": "FACT",       "kinds": ["drone_aerial", "street_level"], "strict": "strict"},
        {"role": "FACT",       "kinds": ["street_level", "landmark"],   "strict": "strict"},
        {"role": "CONTRAST",   "kinds": ["archival_annotated", "drone_aerial"], "strict": "strict"},
        {"role": "FACT",       "kinds": ["dynamic_infographic"],        "strict": "loose"},
        {"role": "FACT",       "kinds": ["street_level", "drone_aerial"], "strict": "strict"},
        {"role": "FACT",       "kinds": ["drone_aerial", "landmark"],   "strict": "strict"},
        {"role": "CONTRAST",   "kinds": ["density_infographic", "map_zoom"], "strict": "loose"},
        {"role": "PAYOFF",     "kinds": ["map_animation", "drone_aerial"], "strict": "loose"},
        {"role": "EXIT",       "kinds": ["drone_aerial", "map_zoom"],   "strict": "loose"},
    ],
    "geography_feature": [
        {"role": "HOOK",       "kinds": ["map_animation", "drone_aerial"], "strict": "loose"},
        {"role": "ESTABLISH",  "kinds": ["drone_aerial"],               "strict": "strict"},
        {"role": "FACT",       "kinds": ["drone_aerial", "street_level"], "strict": "strict"},
        {"role": "FACT",       "kinds": ["archival_annotated", "street_level"], "strict": "strict"},
        {"role": "FACT",       "kinds": ["dynamic_infographic", "density_infographic"], "strict": "loose"},
        {"role": "CONTRAST",   "kinds": ["street_level", "landmark"],   "strict": "strict"},
        {"role": "FACT",       "kinds": ["drone_aerial"],               "strict": "strict"},
        {"role": "FACT",       "kinds": ["archival_annotated", "drone_aerial"], "strict": "strict"},
        {"role": "CONTRAST",   "kinds": ["dynamic_infographic", "map_zoom"], "strict": "loose"},
        {"role": "FACT",       "kinds": ["street_level", "landmark"],   "strict": "strict"},
        {"role": "PAYOFF",     "kinds": ["drone_aerial", "map_zoom"],   "strict": "loose"},
        {"role": "EXIT",       "kinds": ["map_animation"],              "strict": "loose"},
    ],
    "urban_culture": [
        {"role": "HOOK",       "kinds": ["map_zoom", "drone_aerial"],   "strict": "loose"},
        {"role": "ESTABLISH",  "kinds": ["drone_aerial"],               "strict": "strict"},
        {"role": "FACT",       "kinds": ["street_level"],               "strict": "strict"},
        {"role": "FACT",       "kinds": ["drone_aerial", "street_level"], "strict": "strict"},
        {"role": "CONTRAST",   "kinds": ["archival_annotated", "street_level"], "strict": "strict"},
        {"role": "FACT",       "kinds": ["dynamic_infographic", "density_infographic"], "strict": "loose"},
        {"role": "FACT",       "kinds": ["street_level", "landmark"],   "strict": "strict"},
        {"role": "CONTRAST",   "kinds": ["street_level", "drone_aerial"], "strict": "strict"},
        {"role": "FACT",       "kinds": ["archival_annotated", "wikipedia_photo"], "strict": "strict"},
        {"role": "FACT",       "kinds": ["drone_aerial"],               "strict": "strict"},
        {"role": "PAYOFF",     "kinds": ["drone_aerial", "landmark"],   "strict": "strict"},
        {"role": "EXIT",       "kinds": ["map_zoom"],                   "strict": "loose"},
    ],
}

_BEAT_PROMPT = textwrap.dedent("""\
    You are writing the narration for a YouTube Shorts video in the style of
    @urban_atlas — slow, cinematic, education-meets-intrigue. Each beat is one
    narrated line paired with one visual clip.

    Topic class: __TOPIC_CLASS__
    Place:       __PLACE__
    Region:      __REGION__
    Subject:     __SUBJECT__
    Hook hint:   __HOOK_TEXT__

    NARRATIVE ARC (CRITICAL — this separates viral from forgettable):
    This video must tell a SINGLE STORY. The viewer should be able to complete
    the sentence "This video is about how ___" in one clause after watching.
    Bad: "here are interesting facts about Tokyo's lights"
    Good: "Tokyo is so addicted to artificial light that it erased the night sky"

    Story structure:
      HOOK  → Drop the viewer into the mystery or shock. No setup.
      ESTABLISH → One grounding image. The scale of what we're dealing with.
      FACT beats → Each must ADVANCE the thesis. Ask: "Does this fact make the
                   story bigger, darker, stranger, or more ironic?" If not, cut it.
      CONTRAST → The pivot. Introduce the price, the paradox, or the hidden cost.
      PAYOFF → The emotional landing. What does this mean for people who live here?
      EXIT → Leave them unsettled or curious enough to comment.

    CONNECTIVE TISSUE (required between every pair of beats):
    Write each narration line so it answers an implicit question the previous
    beat posed. Use openers like:
      "But that's not the strangest part —"
      "And yet —"
      "Which is why —"
      "What nobody mentions —"
      "The reason it started —"
      "Here's what it costs —"
    Never write two consecutive beats that could be reordered without losing meaning.
    If you can swap beat 5 and beat 6 and nothing breaks, one of them is wrong.

    PACING VARIATION (required):
    Never use the same `duration_sec` for more than 2 consecutive beats.
    Punchy revelations: 2.0-2.5s. Context and atmosphere: 3.0-3.5s.
    The rhythm should feel like breathing — short-short-long, not flat.

    PACING RULES (HARD CONSTRAINTS):
    - Total runtime: __TARGET__s ±3s
    - Each beat: __BMIN__-__BMAX__ seconds
    - __BCOUNT_MIN__-__BCOUNT_MAX__ beats total
    - Urban Atlas pacing: cinematic but PUNCHY. Each beat is one thought, one
      cut. Viewers swipe in 2 seconds — every beat must earn the next.

    HOOK RULES (CRITICAL):
    - `hook_text` MUST be ≤ 7 words. This is burned as large on-screen text.
    - Use a number, superlative, mystery trigger, or paradox. Patterns:
        MYSTERY:   "This city hasn't seen stars since 1964"
        NUMBER:    "40 million people. Zero stars."
        PARADOX:   "The world's brightest city is going dark"
        ACCUSATION: "Tokyo stole the night sky"
        SUPERLATIVE: "700 deaths in one lake"
    - The hook should PROMISE the rest of the video. Viewers should think
      "wait, how?" or "that can't be right" — not just "cool fact".
    - Do NOT write a full question. Do NOT exceed 7 words.

    NARRATION LENGTH BUDGET (CRITICAL — voice consistency required):
    - USE A SINGLE VOICE FOR EVERY BEAT IN THE VIDEO. No mixing. Default to
      `david` (Attenborough, slow/cinematic) unless every beat clearly fits a
      faster register — in which case pick ONE faster voice and use it across
      all beats. Switching voices mid-video breaks the cinematic feel.
    - Voice rates (use to size each beat's narration to its `duration_sec`):
        • `david`  (slow/cinematic)        → ~2.0 words/sec  (default)
        • `guy`    (energetic mid-tempo)   → ~2.6 words/sec
        • `simon`  (rapid factual)         → ~3.1 words/sec
        • `viral`  (generic uptempo)       → ~2.8 words/sec
    - Word budget per beat = `duration_sec` × wps. Examples for `david`:
        • 2.0s beat → ~4 words max
        • 2.5s beat → ~5 words max
        • 3.0s beat → ~6 words max
        • 3.5s beat → ~7 words max
    - Each narration line is ONE clause. No compound sentences. No "and then."
    - COUNT THE WORDS in each narration line before finalizing. If over budget,
      either (a) shorten the line, or (b) increase duration_sec (within __BMIN__-__BMAX__).
    - If narration runs long at synthesis time, the video gets truncated —
      the last beat's narration will literally be cut off mid-sentence. This is
      unacceptable. Stay within the word budget.
    - When narration names a specific place, person, or year, prefer a visual kind
      that can pulse or label it (`archival_annotated`, `dynamic_infographic`,
      `density_infographic`, `map_zoom`) over a generic `drone_aerial`.

    BEAT STRUCTURE (FILL IN):
    You are given __N__ beat slots below. For each slot, write:
      - `narration`: the spoken line (matches the slot's role and the chosen visual kind)
      - `duration_sec`: respecting pacing rules; sum to ~__TARGET__s
      - `visual.kind`: pick ONE from the slot's allowed kinds
      - `visual.subject`: CONCRETE description of what viewer sees (e.g. "drone shot of Shibuya crossing at dusk", NOT "Tokyo busy")
      - `visual.geo`: the exact mappable place for geo-validation
      - `visual.queries`: 3-4 YouTube search queries. If `drone_aerial`, you MUST append "4K drone", "aerial", or "from above". If `landmark`, append "4K cinematic" or "walking tour" to find smooth footage.
      - `visual.callouts`: optional, only for `archival_annotated`; 1-3 objects like {"text":"Oscarville", "xy":[0.34,0.52], "style":"circle"}. `xy` is normalized image position from 0 to 1. Keep callout text <= 5 words.
      - `caption_text`: 2-5 ALL CAPS words for on-screen emphasis
      - `voice`: pick ONE: "david" (slow/contemplative), "guy" (energetic), "simon" (rapid/factual), "viral" (generic)
      - `overlay`: optional — one of:
          {"kind":"stat","data":{"number":"80","suffix":"%","label":"LIVE HERE"}}
          {"kind":"wiki_photo","data":{"title":"Exact Wikipedia Title"}}
          {"kind":"dynamic_infographic","data":{
            "layout":"stat|year|comparison|statement",
            "category":"PLACE OR TOPIC (short, caps)",
            "badge":"HISTORY|GEOGRAPHY|SCALE|FACT (one word, caps)",
            "headline":"PLAIN-LANGUAGE MEANING (short, caps)",
            "number":"the big stat/year/ratio (or omit for statement)",
            "label":"UNIT OR SHORT CLARIFIER (caps)",
            "support":"one sentence grounding the fact",
            "context_visual":"viewer-facing label for the top card, not a search query",
            "colors":["#hex1","#hex2","#fff"] or omit for auto
          }}
          null
        Use `dynamic_infographic` overlay when the beat's visual kind is `dynamic_infographic`.
        Choose layout: `stat` for big numbers, `year` for dates, `comparison` for ratios/X times,
        `statement` when no big number fits. Never force a number into `statement` layout.
        For `context_visual`, write a concrete, on-screen-safe phrase like
        "Federal buyout funding" or "Underground coal fire diagram". Never write raw search
        syntax, alternatives, or prop words such as "money icon or house", "photo of",
        "stock image", or "image showing".
      - `anchor_spec`: optional — emit ONLY when the beat's narration
        references something that deserves archival sourcing, either as an
        `archival_annotated` primary visual or as an archival-image overlay
        on top of primary b-roll. Fire it when the line names a specific
        historical landmark, dated event, historical person, a statistic
        tied to a year, a vanished/altered place, or a map-worthy claim.
        DO NOT emit one for generic establishing lines, vibe shots, or
        present-tense observations.
        Shape:
          {
            "anchor_type": "landmark|event|person|stat|vanished_place|map",
            "subject": "concrete thing to depict — 'Cobb Parkway strip 1970s', 'Shibuya Crossing 1964 Olympics opening'",
            "geo": "strict place for routing — 'Smyrna, GA' or 'Tokyo, Japan'",
            "era": [start_year, end_year] or null,
            "queries": ["2-4 archive-native search strings (see QUERY RULES below)"],
            "allow_substitute": true/false  // may a map/aerial/document stand in if no exact photo exists?
          }
        QUERY RULES (these hit archive catalogs like LOC, Wikimedia
        Commons, Internet Archive — catalog librarians, not TikTok):
          - Use FULL formal names, not nicknames. "Martin Luther King",
            not "MLK". "Western and Atlantic Railroad", not "W&A".
          - Use EXACT YEARS, not decades. "Atlanta 1864", not "Atlanta 1860s".
          - NEVER append filler words: no "historical", "old", "vintage",
            "archival", "early", "classic", "nostalgic". Archives don't
            tag material that way. A plain "Atlanta 1864" outperforms
            "Atlanta 1864 historical archival" every time.
          - Put the proper noun FIRST. "Sweet Auburn Atlanta 1955" not
            "1955 historic Sweet Auburn".
          - One of your queries should be the bare subject noun only
            ("Zero Mile Post Atlanta") — catches items with incomplete
            date metadata.
          - Prefer common-language forms archivists use: "ruins" not
            "destruction aftermath", "parade" not "public celebration".
        Set to null on beats that don't need one. Budget: aim for 2-4
        anchor_specs across the whole video, not one per beat. When the
        slot allows `archival_annotated` and the narration names a specific
        historical event, person, dated landmark, or vanished/altered place,
        prefer `archival_annotated` over generic b-roll and include an
        anchor_spec. Otherwise archival overlays are a decoration, not the backbone.

    VISUAL KIND RULES:
    - `archival_annotated`: choose when the beat names a specific historical event,
      person, dated landmark, vanished/altered place, or archival source-worthy claim.
      It uses an archive photo as the whole beat with slow Ken Burns motion, 1-3
      callouts, and source credit. MUST include `anchor_spec`. Add `visual.callouts`
      only when you can name visible features; each text <= 5 words.
      Do NOT use for abstract statistics, death counts, hazard summaries, or general
      claims unless `visual.subject` names a concrete archival artifact such as a
      specific map, newspaper clipping, plaque, aerial photo, document, or photograph.
      For statistics without a source artifact, use `dynamic_infographic` or
      `density_infographic`.
    - `dynamic_infographic`: choose when the beat delivers ONE key fact — a big number, a year,
      a ratio/scale comparison, or a striking statement. The renderer animates a context card
      (top) + data panel (bottom) sliding in from opposite sides. Always pair with overlay
      kind="dynamic_infographic". Use for punchy single-fact moments.
    - `density_infographic`: choose when the fact involves multiple bars, a heat distribution,
      territorial shifts, rankings, or any multi-data comparison. The PIL renderer handles
      heatmaps, bar charts, border shifts, and stat cards automatically from narration keywords.

    NARRATION RULES:
    - First beat (HOOK): expand the hook_text into one vivid sentence.
      It must land in the first 2 seconds and make the viewer physically pause.
    - EXIT beat: do NOT ask a yes/no question. Instead, leave an unresolved
      tension, drop a reveal that reframes everything, or issue a challenge:
        UNRESOLVED: "And they're about to turn them all off."
        REVEAL: "The darkest place in Tokyo? A 200-year-old garden."
        CHALLENGE: "Name a city that's brighter. You can't."
      The exit line should make people want to comment OR immediately rewatch.
    - Every beat flows into the next with implicit connective logic.
    - Place-specific, not generic. Name real neighborhoods, rivers, districts.

    FACTUAL ACCURACY (HARD CONSTRAINTS — treat as non-negotiable):

    GOAL: Vivid, specific, *defensible* facts. Never bland, never fabricated.
    The truth is almost always sharper than the viral myth — use it.

    MUST INCLUDE (every video):
    - At least 2-3 verifiable specific facts: named towns, named rivers/dams/
      agencies, dated events, documented counts, named people, exact acreage
      or elevation. Specificity is REQUIRED, not optional. "Hazardous maze"
      is a failure; "barbed wire fences from pre-1956 farms" is the bar.
    - At least one ear-catching number or date where the topic supports one
      (death tolls, displacement counts, drops in elevation, years of
      construction). Omitting the juicy stat is a worse failure than getting
      it slightly wrong — but hedge or cite a source type when confidence
      is not total.
    - For topics with documented controversial history (racial, political,
      ecological), include the documented context. It is MORE viral than the
      sanitized version, not less. Lake Lanier's Oscarville was emptied in
      the 1912 Forsyth County racial cleansing, not just "flooded" — that
      fact is the story. Do not launder history into generic "communities
      were displaced" phrasing.

    MUST NOT:
    - Invent round-number statistics. "700 families", "500 deaths", "1000
      acres" — round tidy numbers are a red flag for fabrication. If the
      real figure is 247, say "nearly 250" or "about 250". If unknown,
      hedge ("an estimated") or drop the number.
    - Burn an uncertain number into a `dynamic_infographic` overlay's
      `number` field. Infographic stat cards are read as authoritative.
      Only use figures you can defend against a Wikipedia / news-archive
      check. If unsure, use `layout: "statement"` with no number.
    - Soften documented truth to avoid controversy. Racial history,
      disputed borders, environmental damage, casualty counts — include
      them when they are the story.
    - Use superlatives ("deadliest", "most haunted", "largest", "first
      ever") unless factually established, not just internet lore.
    - Replace a dropped stat with vague atmospheric language ("hazardous
      maze", "dark secrets", "the water hides something"). Vague = boring.
      Specific truth or nothing.

    HEDGING VOCABULARY (use when confidence is partial):
    - "an estimated", "reportedly", "by some counts", "according to the
      Army Corps of Engineers", "as tracked by the AJC", "per USGS",
      "roughly", "nearly", "just under", "just over".

    SELF-CHECK (silent, before emitting JSON):
    For each number you include, ask: "Could I name the source type
    (agency, news outlet, Wikipedia article)?" If no, hedge it or drop it.
    For each overlay stat card, ask: "Would a local resident correct this
    in the comments?" If yes, fix it.

    SLOTS:
    __SLOTS_JSON__

    Return STRICT JSON. Output MUST be a JSON OBJECT (top-level `{ ... }`),
    NOT a JSON array. The beats array lives INSIDE the object under the
    "beats" key, along with title, hook_text, total_duration_sec, and
    youtube_metadata. Do not return a bare list of beats.

    {
      "title": "<50 char YouTube title using the hook formula",
      "hook_text": "<= 7 words, burned as on-screen text at frame 0",
      "total_duration_sec": number,
      "beats": [
        { "beat_id": 1, "role": "...", "narration": "...", "duration_sec": 5.0,
          "visual": {"kind":"...","subject":"...","geo":"...","strictness":"strict|loose","queries":["...","..."],"callouts":[]},
          "caption_text": "...",
          "voice": "david",
          "overlay": null,
          "anchor_spec": null
        }
      ],
      "youtube_metadata": {
        "title": "string",
        "description": "1-2 sentences + 3-5 hashtags",
        "tags": ["tag1","tag2","tag3"]
      }
    }
""")


def _clamp_hook(raw: str) -> str:
    raw = raw.strip()
    words = raw.split()
    if len(words) <= 7:
        return raw[:60]
    return " ".join(words[:7])[:60]


def generate(prompt: str) -> Dict[str, Any]:
    """Full pipeline: classify → pick template → fill beats → validate."""
    topic = classify(prompt)
    tc = topic["topic_class"]
    template = TEMPLATES[tc]

    slots_for_prompt = [
        {"slot": i + 1, "role": s["role"], "allowed_kinds": s["kinds"],
         "strictness": s["strict"]}
        for i, s in enumerate(template)
    ]

    filled_prompt = (_BEAT_PROMPT
        .replace("__TOPIC_CLASS__", tc)
        .replace("__PLACE__", topic["place"])
        .replace("__REGION__", topic["region"])
        .replace("__SUBJECT__", topic["subject"] or topic["place"])
        .replace("__HOOK_TEXT__", topic["hook_text"])
        .replace("__TARGET__", str(TARGET_RUNTIME_SEC))
        .replace("__BMIN__", str(BEAT_MIN_SEC))
        .replace("__BMAX__", str(BEAT_MAX_SEC))
        .replace("__BCOUNT_MIN__", str(BEATS_PER_VIDEO[0]))
        .replace("__BCOUNT_MAX__", str(BEATS_PER_VIDEO[1]))
        .replace("__N__", str(len(template)))
        .replace("__SLOTS_JSON__", json.dumps(slots_for_prompt, indent=2))
    )

    parsed, elapsed = call_json(
        SCRIPT_MODEL, filled_prompt, temperature=0.75, max_retries=2
    )

    # Gemini occasionally returns a bare list of beats instead of the full
    # envelope. Rewrap it so downstream code sees the expected shape.
    if isinstance(parsed, list):
        parsed = {"beats": parsed}

    beats = parsed.get("beats") or []
    if not beats:
        raise GeminiError("script returned no beats")

    # Enforce pacing on our side, not just trust the model.
    anchor_specs: List[AnchorSpec] = []
    for i, b in enumerate(beats):
        d = float(b.get("duration_sec") or 0.0)
        if d < BEAT_MIN_SEC:
            b["duration_sec"] = BEAT_MIN_SEC
        elif d > BEAT_MAX_SEC:
            b["duration_sec"] = BEAT_MAX_SEC
        b["beat_id"] = i + 1
        # Fill strictness from template if the model dropped it.
        if i < len(template):
            b.setdefault("visual", {})
            b["visual"].setdefault("strictness", template[i]["strict"])

        # Normalize anchor_spec: present+usable -> AnchorSpec, else drop.
        raw_anchor = b.get("anchor_spec")
        spec = _coerce_anchor_spec(raw_anchor, beat_id=b["beat_id"])
        if spec is not None:
            anchor_specs.append(spec)
            b["anchor_spec"] = spec.to_dict()
        else:
            b["anchor_spec"] = None

        if _should_downgrade_archival_visual(b):
            visual = b.setdefault("visual", {})
            visual["kind"] = "density_infographic"
            visual["strictness"] = "loose"
            visual.pop("callouts", None)


    # Pacing variation enforcement: break monotonous same-duration runs.
    # The LLM is instructed to vary pacing but sometimes ignores it. Never
    # allow 3+ consecutive beats with the same duration_sec — it kills rhythm.
    for i in range(2, len(beats)):
        d0 = float(beats[i - 2].get("duration_sec") or BEAT_MIN_SEC)
        d1 = float(beats[i - 1].get("duration_sec") or BEAT_MIN_SEC)
        d2 = float(beats[i].get("duration_sec") or BEAT_MIN_SEC)
        if abs(d0 - d1) < 0.05 and abs(d1 - d2) < 0.05:
            # Shorten the middle beat if there's room, else lengthen it.
            if d1 > BEAT_MIN_SEC + 0.5:
                beats[i - 1]["duration_sec"] = BEAT_MIN_SEC
            elif d1 < BEAT_MAX_SEC - 0.5:
                beats[i - 1]["duration_sec"] = BEAT_MAX_SEC

    total = sum(float(b["duration_sec"]) for b in beats)


    return {
        "topic_class": tc,
        "place": topic["place"],
        "region": topic["region"],
        "subject": topic["subject"],
        "title": str(parsed.get("title") or "").strip()[:60],
        "hook_text": _clamp_hook(str(parsed.get("hook_text") or topic["hook_text"])),
        "total_duration_sec": round(total, 2),
        "beats": beats,
        "anchor_specs": [a.to_dict() for a in anchor_specs],
        "youtube_metadata": parsed.get("youtube_metadata") or {},
        "topic_meta": topic,
        "script_elapsed_sec": round(elapsed, 2),
    }


def _coerce_anchor_spec(raw: Any, *, beat_id: int) -> "AnchorSpec | None":
    """Gemini can emit anchor_spec as null, {}, or a partial dict. Only
    keep it if subject is populated — everything else has sensible
    defaults in AnchorSpec.from_dict."""
    if not isinstance(raw, dict):
        return None
    if not str(raw.get("subject") or "").strip():
        return None
    raw = dict(raw)
    raw["beat_id"] = beat_id
    try:
        return AnchorSpec.from_dict(raw)
    except Exception:
        return None


def _should_downgrade_archival_visual(beat: Dict[str, Any]) -> bool:
    visual = beat.get("visual") or {}
    if str(visual.get("kind") or "").strip().lower() != "archival_annotated":
        return False

    anchor = beat.get("anchor_spec") if isinstance(beat.get("anchor_spec"), dict) else {}
    if not anchor:
        return True

    anchor_type = str(anchor.get("anchor_type") or "").strip().lower()
    blob_parts = [
        visual.get("subject"),
        visual.get("geo"),
        beat.get("caption_text"),
        anchor.get("subject"),
        anchor.get("geo"),
    ]
    blob_parts.extend(visual.get("queries") or [])
    blob_parts.extend(anchor.get("queries") or [])
    blob = " ".join(str(p or "") for p in blob_parts).lower()

    if anchor_type == "stat" and not any(term in blob for term in _ARCHIVAL_ARTIFACT_TERMS):
        return True
    return False
