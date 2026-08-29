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

from .models import Channel, Snapshot, Video

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


class CorpusStore:
    """Persistent video corpus for large harvests.

    Separate from SnapshotStore because the two have opposite shapes: snapshots
    are many rows per video over time, the corpus is one row per video that
    must survive across many days of resumed harvesting.

    Everything here is written to be resumable. A 200k-video harvest spans
    several daily quota resets, and a crash on day three must not cost days one
    and two - so channels are marked done only after their uploads are fully
    walked, and every video insert is idempotent.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS corpus_videos (
        video_id      TEXT PRIMARY KEY,
        channel_id    TEXT NOT NULL,
        channel_title TEXT,
        title         TEXT,
        description   TEXT,
        published_at  TEXT NOT NULL,
        duration_s    INTEGER NOT NULL,
        views         INTEGER NOT NULL,
        likes         INTEGER NOT NULL,
        comments      INTEGER NOT NULL,
        tags          TEXT,
        thumbnail_url TEXT,
        is_short      INTEGER NOT NULL,
        harvested_at  TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_corpus_channel ON corpus_videos(channel_id);
    CREATE INDEX IF NOT EXISTS idx_corpus_short   ON corpus_videos(is_short);

    CREATE TABLE IF NOT EXISTS corpus_channels (
        channel_id   TEXT PRIMARY KEY,
        title        TEXT,
        subscribers  INTEGER,
        video_count  INTEGER,
        uploads_playlist TEXT,
        expanded     INTEGER NOT NULL DEFAULT 0,
        discovered_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_chan_expanded ON corpus_channels(expanded);

    CREATE TABLE IF NOT EXISTS harvest_slices (
        slice_key TEXT PRIMARY KEY,
        done_at   TEXT NOT NULL
    );
    """

    def __init__(self, path: str | Path = "corpus.db"):
        self.path = str(path)
        with closing(self._conn()) as c:
            c.executescript(self.SCHEMA)
            c.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        # A 200k-row harvest is write-heavy; WAL keeps reads from blocking it.
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ---------- videos ----------

    def add_videos(self, videos: Sequence[Video], now: Optional[datetime] = None) -> list[str]:
        """Insert videos, ignoring ones already stored. Returns the new ids.

        INSERT OR IGNORE rather than REPLACE: a video seen again on a later day
        would otherwise have its original counters overwritten with fresher
        ones, silently destroying the age-vs-views relationship the whole
        engine is built on. Re-observation belongs in SnapshotStore.

        Returning the ids rather than a bare count matters for reporting: a
        harvest re-encounters the same video constantly, and a caller that
        counts what it *submitted* rather than what was *stored* will report
        more shorts than it has videos.
        """
        ts = (now or datetime.now(timezone.utc)).isoformat()
        rows = [
            (v.video_id, v.channel_id, v.channel_title, v.title, v.description,
             v.published_at.isoformat(), v.duration_s, v.views, v.likes, v.comments,
             ",".join(v.tags), v.thumbnail_url, 1 if v.is_short else 0, ts)
            for v in videos
        ]
        ids = [v.video_id for v in videos]
        with closing(self._conn()) as c:
            existing = set()
            for i in range(0, len(ids), 400):  # keep the SQL variable count sane
                chunk = ids[i:i + 400]
                q = f"SELECT video_id FROM corpus_videos WHERE video_id IN ({','.join('?' * len(chunk))})"
                existing.update(r[0] for r in c.execute(q, chunk))
            c.executemany(
                "INSERT OR IGNORE INTO corpus_videos VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
            )
            c.commit()
        seen, new_ids = set(), []
        for vid in ids:
            if vid not in existing and vid not in seen:
                seen.add(vid)
                new_ids.append(vid)
        return new_ids

    def load_videos(self, shorts_only: bool = False, limit: Optional[int] = None) -> list[Video]:
        q = "SELECT * FROM corpus_videos"
        if shorts_only:
            q += " WHERE is_short=1"
        if limit:
            q += f" LIMIT {int(limit)}"
        with closing(self._conn()) as c:
            rows = c.execute(q).fetchall()
        return [
            Video(
                video_id=r["video_id"], channel_id=r["channel_id"],
                channel_title=r["channel_title"] or "", title=r["title"] or "",
                description=r["description"] or "",
                published_at=datetime.fromisoformat(r["published_at"]),
                duration_s=r["duration_s"], views=r["views"], likes=r["likes"],
                comments=r["comments"],
                tags=[t for t in (r["tags"] or "").split(",") if t],
                thumbnail_url=r["thumbnail_url"] or "",
            )
            for r in rows
        ]

    def counts(self) -> dict[str, int]:
        with closing(self._conn()) as c:
            return {
                "videos": c.execute("SELECT COUNT(*) FROM corpus_videos").fetchone()[0],
                "shorts": c.execute("SELECT COUNT(*) FROM corpus_videos WHERE is_short=1").fetchone()[0],
                "channels": c.execute("SELECT COUNT(*) FROM corpus_channels").fetchone()[0],
                "channels_expanded": c.execute(
                    "SELECT COUNT(*) FROM corpus_channels WHERE expanded=1").fetchone()[0],
                "slices_done": c.execute("SELECT COUNT(*) FROM harvest_slices").fetchone()[0],
            }

    # ---------- channels ----------

    def add_channels(self, channels: Sequence[Channel], now: Optional[datetime] = None) -> int:
        ts = (now or datetime.now(timezone.utc)).isoformat()
        with closing(self._conn()) as c:
            before = c.execute("SELECT COUNT(*) FROM corpus_channels").fetchone()[0]
            c.executemany(
                "INSERT OR IGNORE INTO corpus_channels "
                "(channel_id,title,subscribers,video_count,uploads_playlist,expanded,discovered_at) "
                "VALUES (?,?,?,?,?,0,?)",
                [(ch.channel_id, ch.title, ch.subscribers, ch.video_count,
                  ch.uploads_playlist, ts) for ch in channels],
            )
            c.commit()
            after = c.execute("SELECT COUNT(*) FROM corpus_channels").fetchone()[0]
        return after - before

    def pending_channels(self, limit: int = 100) -> list[Channel]:
        """Channels discovered but not yet walked - the resume queue."""
        with closing(self._conn()) as c:
            rows = c.execute(
                "SELECT * FROM corpus_channels WHERE expanded=0 ORDER BY subscribers DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            Channel(channel_id=r["channel_id"], title=r["title"] or "",
                    subscribers=r["subscribers"] or 0, total_views=0,
                    video_count=r["video_count"] or 0,
                    uploads_playlist=r["uploads_playlist"] or "")
            for r in rows
        ]

    def mark_expanded(self, channel_id: str) -> None:
        with closing(self._conn()) as c:
            c.execute("UPDATE corpus_channels SET expanded=1 WHERE channel_id=?", (channel_id,))
            c.commit()

    def known_channel_ids(self) -> set[str]:
        with closing(self._conn()) as c:
            return {r[0] for r in c.execute("SELECT channel_id FROM corpus_channels")}

    # ---------- slices ----------

    def slice_done(self, key: str) -> bool:
        with closing(self._conn()) as c:
            return c.execute(
                "SELECT 1 FROM harvest_slices WHERE slice_key=?", (key,)
            ).fetchone() is not None

    def mark_slice(self, key: str, now: Optional[datetime] = None) -> None:
        with closing(self._conn()) as c:
            c.execute("INSERT OR REPLACE INTO harvest_slices VALUES (?,?)",
                      (key, (now or datetime.now(timezone.utc)).isoformat()))
            c.commit()
