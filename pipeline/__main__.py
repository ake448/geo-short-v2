"""
CLI entry: python -m pipeline_v2 "<topic prompt>"

Currently a stub that exercises the cache + config wiring. Each build step
fills in another stage.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from . import __version__
from .cache import get_cache
from .config import (
    BEAT_MIN_SEC, BEAT_MAX_SEC, GEMINI_API_KEY, MAPBOX_TOKEN,
    PEXELS_API_KEY, PIXABAY_API_KEY, RUNS_DIR, SOURCING_PARALLEL_BEATS,
)


def _check_env() -> int:
    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not (PEXELS_API_KEY or PIXABAY_API_KEY):
        missing.append("PEXELS_API_KEY or PIXABAY_API_KEY")
    if not MAPBOX_TOKEN:
        missing.append("MAPBOX_TOKEN")
    if missing:
        print(f"[FAIL] missing env vars: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pipeline_v2", description="Geo Shorts V2")
    p.add_argument("prompt", nargs="?", help="Topic prompt (e.g. 'Tokyo')")
    p.add_argument("--check", action="store_true", help="Validate env + cache only")
    p.add_argument("--cache-stats", action="store_true", help="Print cache stats and exit")
    p.add_argument("--invalidate-model", help="Drop verdict rows for a given model id")
    p.add_argument("--resume", help="Resume from an existing run directory")
    p.add_argument("--beats", help="Comma-separated beat IDs to process (e.g. 1,2,5)")
    p.add_argument("--voice", help="Voice keyword (david, guy, simon, viral) or path to mp3")
    args = p.parse_args(argv)

    print(f"pipeline_v2 v{__version__}")
    print(f"  runs dir: {RUNS_DIR}")
    print(f"  beat pacing: {BEAT_MIN_SEC}-{BEAT_MAX_SEC}s")
    print(f"  parallelism: {SOURCING_PARALLEL_BEATS} beats concurrent")

    if args.check:
        rc = _check_env()
        cache = get_cache()
        print(f"  cache: {cache.stats()}")
        return rc

    if args.cache_stats:
        print(get_cache().stats())
        return 0

    if args.invalidate_model:
        n = get_cache().invalidate_model(args.invalidate_model)
        print(f"invalidated {n} verdict rows for model {args.invalidate_model!r}")
        return 0

    if not args.prompt and not args.resume:
        p.print_help()
        return 2

    from .script import generate
    from .sourcing import source_all_beats
    import json as _json
    import time as _time

    if args.resume:
        run_dir = Path(args.resume).resolve()
        if not run_dir.exists():
            print(f"[FAIL] resume directory not found: {run_dir}", file=sys.stderr)
            return 1
        print(f"\n[step 1] resuming from: {run_dir}")
        script_path = run_dir / "script.json"
        if not script_path.exists():
            print(f"[FAIL] script.json not found in {run_dir}", file=sys.stderr)
            return 1
        script = _json.loads(script_path.read_text(encoding="utf-8"))
    else:
        print(f"\n[step 2] generating script for: {args.prompt!r}")
        script = generate(args.prompt)
        print(f"  topic_class: {script['topic_class']} (conf {script['topic_meta']['confidence']})")
        print(f"  place: {script['place']}")
        print(f"  title: {script['title']}")
        print(f"  hook:  {script['hook_text']}")
        print(f"  beats: {len(script['beats'])}, total {script['total_duration_sec']}s")

        slug = args.prompt[:40].replace(" ", "_").replace("/", "_")
        run_dir = RUNS_DIR / f"{int(_time.time())}_{slug}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "script.json").write_text(
            _json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ─── Filter beats if requested ───────────────────────────────────────────
    if args.beats:
        try:
            target_ids = [int(x.strip()) for x in args.beats.split(",") if x.strip()]
            original_len = len(script["beats"])
            script["beats"] = [b for b in script["beats"] if b.get("beat_id") in target_ids]
            if not script["beats"]:
                print(f"[FAIL] no beats matched IDs: {target_ids}", file=sys.stderr)
                return 1
            print(f"  mode: selective beats {target_ids} ({len(script['beats'])}/{original_len})")
        except ValueError:
            print(f"[FAIL] invalid --beats format: {args.beats}", file=sys.stderr)
            return 1

    script_context = {
        "topic_class": script.get("topic_class"),
        "place": script.get("place"),
        "region": script.get("region"),
        "subject": script.get("subject"),
    }
    for beat in script.get("beats", []):
        beat["_script_context"] = script_context

    print(f"\n[step 3] sourcing footage for {len(script['beats'])} beats...")
    src_t0 = _time.time()
    sources = source_all_beats(script, run_dir, verbose=True)
    src_elapsed = _time.time() - src_t0

    n_ok = sum(1 for s in sources if s.get("winner"))
    n_yt = sum(1 for s in sources if (s.get("winner") or {}).get("source") == "youtube")
    n_px = sum(1 for s in sources if (s.get("winner") or {}).get("source") in ("pexels", "pixabay"))
    print(f"\n  sourced {n_ok}/{len(sources)} beats in {src_elapsed:.1f}s "
          f"(yt={n_yt}, px={n_px}, fail={len(sources)-n_ok})")

    (run_dir / "sources.json").write_text(
        _json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  written: {run_dir}")

    # ─── Step 4: fill cinematic fallbacks for beats without a clip ───────────
    from .render import cinematics, captions, branding, assembly
    from .audio import voiceover as vo_mod, sfx as sfx_mod

    print(f"\n[step 4] filling cinematic fallbacks...")
    clip_dir = run_dir / "clips"
    clip_dir.mkdir(exist_ok=True)
    n_cinematic = 0
    for i, (beat, src) in enumerate(zip(script["beats"], sources)):
        clip = src.get("clip_path")
        if clip and Path(clip).exists():
            continue
        out = clip_dir / f"beat_{i+1:02d}_cinematic.mp4"
        ok = cinematics.make_fallback(
            beat,
            out,
            sources_for_all_beats=sources,
            source_index=i,
        )
        if ok:
            src["clip_path"] = str(out)
            src["winner"] = src.get("winner") or {"source": "cinematic", "kind": beat["visual"]["kind"]}
            n_cinematic += 1
            print(f"  beat {i+1}: cinematic {beat['visual']['kind']} -> {out.name}")
        else:
            print(f"  beat {i+1}: [FAIL] cinematic failed for kind={beat['visual']['kind']}")
    print(f"  cinematic fallbacks: {n_cinematic}")

    # ─── Step 5: voiceover + whisper alignment ───────────────────────────────
    print(f"\n[step 5] synthesizing voiceover...")
    vo_voice = Path(args.voice) if args.voice else None
    vo_result = vo_mod.synthesize(script, run_dir, voice_path=vo_voice)
    if not vo_result:
        print("[FAIL] voiceover synthesis failed", file=sys.stderr)
        return 1
    script = vo_result.updated_script  # durations now reflect real speech
    print(f"  voiceover: {vo_result.audio_path.name} ({vo_result.total_duration_sec:.1f}s)")

    # ─── Step 6: mix audio (VO + music + SFX) ────────────────────────────────
    print(f"\n[step 6] mixing final audio...")
    final_audio = sfx_mod.process_sfx(script, vo_result.audio_path, run_dir)
    print(f"  final audio: {final_audio.name}")

    # ─── Step 7a: assemble clips to beat durations + mux audio ───────────────
    print(f"\n[step 7a] assembling clips...")
    from .render import color_grade
    beat_clips = []
    grade_profiles = []
    normalize_flags = []
    for i, (beat, src) in enumerate(zip(script["beats"], sources)):
        cp = src.get("clip_path")
        if cp and Path(cp).exists():
            beat_clips.append((Path(cp), float(beat.get("duration_sec", 5.0))))
            winner = src.get("winner") or {}
            profile = color_grade.pick_profile(
                source=winner.get("source"),
                kind=winner.get("kind") or beat.get("visual", {}).get("kind"),
                render_mode=beat.get("render_mode"),
            )
            grade_profiles.append(profile)
            norm = color_grade.needs_normalize(winner.get("source"))
            normalize_flags.append(norm)
            print(f"  beat {i+1}: grade={profile} norm={norm}", flush=True)
        else:
            print(f"  [warn] beat {i+1} has no clip, skipping", flush=True)
    assembled = run_dir / "assembled.mp4"
    if not assembly.assemble(beat_clips, final_audio, assembled,
                             grade_profiles=grade_profiles,
                             normalize_flags=normalize_flags):
        print("[FAIL] assembly failed", file=sys.stderr)
        return 1

    # ─── Step 7b: burn captions ──────────────────────────────────────────────
    print(f"\n[step 7b] burning captions...")
    hook_dur = 0.0
    ass_text = captions.generate_ass(
        vo_result.whisper_segments,
        hook_duration=hook_dur,
        beats=script.get("beats") or [],
    )
    captioned = run_dir / "captioned.mp4"
    if not captions.burn(assembled, ass_text, captioned):
        print("[FAIL] caption burn failed", file=sys.stderr)
        return 1

    # ─── Step 7c: apply Urban Atlas branding (border + watermark + intro) ────
    print(f"\n[step 7c] applying branding...")
    from .config import FINAL_EXPORT_DIR
    final_out = FINAL_EXPORT_DIR / f"{run_dir.name}.mp4"
    if not branding.apply_branding(captioned, final_out, intro_duration=0.6):
        print("[FAIL] branding failed", file=sys.stderr)
        return 1

    print(f"\n[done] final video: {final_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
