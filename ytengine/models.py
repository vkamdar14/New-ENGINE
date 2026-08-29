"""Core record types.

Kept as plain dataclasses so every stage of the pipeline - fetch, score,
report - passes the same objects around and tests can build them by hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?$"
)


def parse_duration(iso: str) -> int:
    """Turn an ISO-8601 duration (YouTube's `PT12M30S`) into seconds.

    Returns 0 for anything unparseable - live streams and premieres report
    durations we don't care about, and a zero sorts them into their own bucket
    rather than crashing a run midway through a 500-video pull.
    """
    if not iso:
        return 0
    m = _DURATION_RE.match(iso)
    if not m:
        return 0
    parts = {k: int(v) if v else 0 for k, v in m.groupdict().items()}
    return parts["days"] * 86400 + parts["hours"] * 3600 + parts["minutes"] * 60 + parts["seconds"]


def parse_ts(iso: str) -> datetime:
    """Parse an RFC-3339 timestamp into an aware UTC datetime."""
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class Video:
    video_id: str
    channel_id: str
    channel_title: str
    title: str
    description: str
    published_at: datetime
    duration_s: int
    views: int
    likes: int
    comments: int
    tags: list[str] = field(default_factory=list)
    thumbnail_url: str = ""

    @property
    def is_short(self) -> bool:
        """Shorts play by different distribution rules, so they never share a
        baseline with long-form. 180s is the current upper bound."""
        return 0 < self.duration_s <= 180

    def age_days(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(timezone.utc)
        return max((now - self.published_at).total_seconds() / 86400.0, 0.0)

    def views_per_day(self, now: Optional[datetime] = None) -> float:
        # Floor the age at half a day: a video three hours old would otherwise
        # report an enormous rate off a handful of views.
        return self.views / max(self.age_days(now), 0.5)

    def engagement_rate(self) -> float:
        """Likes+comments per view. A weak CTR proxy, but a strong proxy for
        whether the people who did arrive actually cared."""
        return (self.likes + self.comments) / self.views if self.views else 0.0

    @classmethod
    def from_api(cls, item: dict) -> "Video":
        sn = item.get("snippet", {})
        st = item.get("statistics", {})
        cd = item.get("contentDetails", {})
        thumbs = sn.get("thumbnails", {})
        best = thumbs.get("maxres") or thumbs.get("high") or thumbs.get("medium") or {}
        return cls(
            video_id=item.get("id", ""),
            channel_id=sn.get("channelId", ""),
            channel_title=sn.get("channelTitle", ""),
            title=sn.get("title", ""),
            description=sn.get("description", ""),
            published_at=parse_ts(sn.get("publishedAt", "1970-01-01T00:00:00Z")),
            duration_s=parse_duration(cd.get("duration", "")),
            views=int(st.get("viewCount", 0) or 0),
            likes=int(st.get("likeCount", 0) or 0),
            comments=int(st.get("commentCount", 0) or 0),
            tags=list(sn.get("tags", []) or []),
            thumbnail_url=best.get("url", ""),
        )


@dataclass
class Channel:
    channel_id: str
    title: str
    subscribers: int
    total_views: int
    video_count: int
    uploads_playlist: str

    @classmethod
    def from_api(cls, item: dict) -> "Channel":
        sn = item.get("snippet", {})
        st = item.get("statistics", {})
        rel = item.get("contentDetails", {}).get("relatedPlaylists", {})
        return cls(
            channel_id=item.get("id", ""),
            title=sn.get("title", ""),
            subscribers=int(st.get("subscriberCount", 0) or 0),
            total_views=int(st.get("viewCount", 0) or 0),
            video_count=int(st.get("videoCount", 0) or 0),
            uploads_playlist=rel.get("uploads", ""),
        )


@dataclass
class Snapshot:
    """One observation of a video's counters at a point in time.

    Velocity needs two of these. A single API pull can tell you a video has
    900k views; only a second pull tells you whether it is still climbing.
    """

    video_id: str
    observed_at: datetime
    views: int
    likes: int
    comments: int
