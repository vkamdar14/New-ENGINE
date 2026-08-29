"""SQLite snapshot store.

Velocity is not observable from one API pull. A single call says a video has
900,000 views; only a second call, hours later, says whether it is still
climbing at 40k/hour or has flatlined. Those are opposite decisions - one is a
topic to publish into tomorrow, the other is a wave that already broke.

So the trend tracker is a daemon-shaped thing: snapshot on a schedule, and the
differences between snapshots are the signal.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .models import Snapshot, Video

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id      TEXT PRIMARY KEY,
    channel_id    TEXT NOT NULL,
    channel_title TEXT,
    title         TEXT,
    published_at  TEXT NOT NULL,
    duration_s    INTEGER,
    tags          TEXT
);
CREATE TABLE IF NOT EXISTS snapshots (
    video_id    TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    views       INTEGER NOT NULL,
    likes       INTEGER NOT NULL,
    comments    INTEGER NOT NULL,
    PRIMARY KEY (video_id, observed_at)
);
CREATE INDEX IF NOT EXISTS idx_snap_video ON snapshots(video_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_vid_channel ON videos(channel_id);
"""


class SnapshotStore:
    def __init__(self, path: str | Path = "ytengine.db"):
        self.path = str(path)
        with closing(self._conn()) as c:
            c.executescript(SCHEMA)
            c.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def record(self, videos: Sequence[Video], observed_at: Optional[datetime] = None) -> int:
        """Persist one observation of each video. Idempotent per timestamp."""
        ts = (observed_at or datetime.now(timezone.utc)).isoformat()
        with closing(self._conn()) as c:
            c.executemany(
                "INSERT OR REPLACE INTO videos VALUES (?,?,?,?,?,?,?)",
                [
                    (v.video_id, v.channel_id, v.channel_title, v.title,
                     v.published_at.isoformat(), v.duration_s, ",".join(v.tags))
                    for v in videos
                ],
            )
            c.executemany(
                "INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?)",
                [(v.video_id, ts, v.views, v.likes, v.comments) for v in videos],
            )
            c.commit()
            return len(videos)

    def history(self, video_id: str) -> list[Snapshot]:
        with closing(self._conn()) as c:
            rows = c.execute(
                "SELECT * FROM snapshots WHERE video_id=? ORDER BY observed_at", (video_id,)
            ).fetchall()
        return [
            Snapshot(
                video_id=r["video_id"],
                observed_at=datetime.fromisoformat(r["observed_at"]),
                views=r["views"],
                likes=r["likes"],
                comments=r["comments"],
            )
            for r in rows
        ]

    def tracked_ids(self) -> list[str]:
        with closing(self._conn()) as c:
            return [r[0] for r in c.execute("SELECT video_id FROM videos").fetchall()]

    def titles(self) -> dict[str, str]:
        with closing(self._conn()) as c:
            return {r["video_id"]: r["title"] for r in c.execute("SELECT video_id,title FROM videos")}

    def videos_with_history(self, min_snapshots: int = 2) -> list[str]:
        with closing(self._conn()) as c:
            rows = c.execute(
                "SELECT video_id FROM snapshots GROUP BY video_id HAVING COUNT(*) >= ?",
                (min_snapshots,),
            ).fetchall()
        return [r[0] for r in rows]

    def prune(self, keep_per_video: int = 200) -> int:
        """Cap history per video so a long-running tracker stays bounded."""
        removed = 0
        with closing(self._conn()) as c:
            for (vid,) in c.execute("SELECT DISTINCT video_id FROM snapshots").fetchall():
                stale = c.execute(
                    "SELECT observed_at FROM snapshots WHERE video_id=? "
                    "ORDER BY observed_at DESC LIMIT -1 OFFSET ?",
                    (vid, keep_per_video),
                ).fetchall()
                if stale:
                    c.executemany(
                        "DELETE FROM snapshots WHERE video_id=? AND observed_at=?",
                        [(vid, r[0]) for r in stale],
                    )
                    removed += len(stale)
            c.commit()
        return removed
