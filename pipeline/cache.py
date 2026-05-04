"""
cache.py — SQLite-backed cache for V2.

Tables:
  video_meta      yt-dlp metadata per video_id
  text_verdict    cheap-LLM score per (video_id, beat_hash)
  vision_verdict  vision pass/fail per (video_id, start_ts, beat_hash)
  clips           on-disk clip registry (sha256 → path)

Verdict tables include `model` so a model bump can invalidate stale rows.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .config import CACHE_DB

_SCHEMA = """
CREATE TABLE IF NOT EXISTS video_meta (
    video_id     TEXT PRIMARY KEY,
    title        TEXT,
    channel      TEXT,
    channel_id   TEXT,
    description  TEXT,
    tags         TEXT,
    duration     REAL,
    view_count   INTEGER,
    upload_date  TEXT,
    fetched_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS text_verdict (
    video_id   TEXT NOT NULL,
    beat_hash  TEXT NOT NULL,
    score      INTEGER NOT NULL,
    reason     TEXT,
    model      TEXT NOT NULL,
    ts         REAL NOT NULL,
    PRIMARY KEY (video_id, beat_hash, model)
);

CREATE TABLE IF NOT EXISTS vision_verdict (
    video_id   TEXT NOT NULL,
    start_ts   REAL NOT NULL,
    beat_hash  TEXT NOT NULL,
    passed     INTEGER NOT NULL,
    reason     TEXT,
    model      TEXT NOT NULL,
    ts         REAL NOT NULL,
    PRIMARY KEY (video_id, start_ts, beat_hash, model)
);

CREATE TABLE IF NOT EXISTS clips (
    sha256     TEXT PRIMARY KEY,
    clip_path  TEXT NOT NULL,
    video_id   TEXT,
    start_ts   REAL,
    end_ts     REAL,
    duration   REAL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_text_verdict_beat ON text_verdict(beat_hash);
CREATE INDEX IF NOT EXISTS idx_vision_verdict_beat ON vision_verdict(beat_hash);
CREATE INDEX IF NOT EXISTS idx_clips_video ON clips(video_id);
"""


def beat_hash(beat: Dict[str, Any]) -> str:
    """Stable hash of the parts of a beat that affect validation outcome."""
    keyed = {
        "narration": beat.get("narration", ""),
        "visual_brief": beat.get("visual_brief", ""),
        "geo": beat.get("geo", ""),
        "strictness": beat.get("strictness", "loose"),
    }
    blob = json.dumps(keyed, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class Cache:
    """Thread-safe SQLite wrapper. One instance per process is plenty."""

    def __init__(self, db_path: Path = CACHE_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ── video_meta ────────────────────────────────────────────────────────────
    def upsert_video_meta(self, meta: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO video_meta
                  (video_id, title, channel, channel_id, description, tags,
                   duration, view_count, upload_date, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                  title=excluded.title, channel=excluded.channel,
                  channel_id=excluded.channel_id, description=excluded.description,
                  tags=excluded.tags, duration=excluded.duration,
                  view_count=excluded.view_count, upload_date=excluded.upload_date,
                  fetched_at=excluded.fetched_at
                """,
                (
                    meta["video_id"],
                    meta.get("title"),
                    meta.get("channel"),
                    meta.get("channel_id"),
                    meta.get("description"),
                    json.dumps(meta.get("tags") or []),
                    meta.get("duration"),
                    meta.get("view_count"),
                    meta.get("upload_date"),
                    time.time(),
                ),
            )
            self._conn.commit()

    def get_video_meta(self, video_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM video_meta WHERE video_id = ?", (video_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["tags"] = json.loads(d.get("tags") or "[]")
        except Exception:
            d["tags"] = []
        return d

    # ── text_verdict ──────────────────────────────────────────────────────────
    def get_text_verdict(self, video_id: str, beat_h: str, model: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM text_verdict WHERE video_id=? AND beat_hash=? AND model=?",
                (video_id, beat_h, model),
            ).fetchone()
        return dict(row) if row else None

    def put_text_verdict(self, video_id: str, beat_h: str, score: int,
                         reason: str, model: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO text_verdict
                  (video_id, beat_hash, score, reason, model, ts)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (video_id, beat_h, int(score), reason, model, time.time()),
            )
            self._conn.commit()

    # ── vision_verdict ────────────────────────────────────────────────────────
    def get_vision_verdict(self, video_id: str, start_ts: float,
                           beat_h: str, model: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM vision_verdict
                WHERE video_id=? AND start_ts=? AND beat_hash=? AND model=?
                """,
                (video_id, float(start_ts), beat_h, model),
            ).fetchone()
        return dict(row) if row else None

    def put_vision_verdict(self, video_id: str, start_ts: float, beat_h: str,
                           passed: bool, reason: str, model: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO vision_verdict
                  (video_id, start_ts, beat_hash, passed, reason, model, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (video_id, float(start_ts), beat_h, 1 if passed else 0,
                 reason, model, time.time()),
            )
            self._conn.commit()

    # ── clips registry ────────────────────────────────────────────────────────
    def register_clip(self, clip_path: Path, video_id: Optional[str] = None,
                      start_ts: Optional[float] = None,
                      end_ts: Optional[float] = None) -> str:
        sha = _sha256_file(clip_path)
        duration = (end_ts - start_ts) if (start_ts is not None and end_ts is not None) else None
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO clips
                  (sha256, clip_path, video_id, start_ts, end_ts, duration, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (sha, str(clip_path), video_id, start_ts, end_ts, duration, time.time()),
            )
            self._conn.commit()
        return sha

    def find_clip_by_hash(self, sha: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM clips WHERE sha256=?", (sha,)
            ).fetchone()
        return dict(row) if row else None

    # ── maintenance ───────────────────────────────────────────────────────────
    def invalidate_model(self, model: str) -> int:
        """Drop all verdicts for a given model id. Returns rows deleted."""
        with self._lock:
            n = self._conn.execute(
                "DELETE FROM text_verdict WHERE model=?", (model,)
            ).rowcount
            n += self._conn.execute(
                "DELETE FROM vision_verdict WHERE model=?", (model,)
            ).rowcount
            self._conn.commit()
        return n

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "videos": self._conn.execute("SELECT COUNT(*) FROM video_meta").fetchone()[0],
                "text_verdicts": self._conn.execute("SELECT COUNT(*) FROM text_verdict").fetchone()[0],
                "vision_verdicts": self._conn.execute("SELECT COUNT(*) FROM vision_verdict").fetchone()[0],
                "clips": self._conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0],
            }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


_SINGLETON: Optional[Cache] = None


def get_cache() -> Cache:
    """Process-wide singleton. Avoids opening multiple SQLite connections."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = Cache()
    return _SINGLETON
