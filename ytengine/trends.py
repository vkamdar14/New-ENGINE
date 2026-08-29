"""Velocity and acceleration from the snapshot history.

The decision this supports: publish into a rising curve, not a peaked one.

By the time a topic is obviously big it is saturated - every competitor has
already shipped, and you are arriving into a feed that has stopped rewarding
the subject. What you want is the topic whose *second derivative* is positive
while its absolute numbers are still modest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .models import Snapshot
from .stats import median
from .store import SnapshotStore


@dataclass
class Velocity:
    video_id: str
    title: str
    views: int
    vph_recent: float     # views/hour over the latest interval
    vph_prior: float      # views/hour over the interval before that
    acceleration: float   # ratio recent/prior; >1 means still building
    hours_tracked: float
    n_snapshots: int

    @property
    def phase(self) -> str:
        """Where the video sits on its own curve."""
        if self.n_snapshots < 3:
            return "unknown"     # need three points to see a second derivative
        if self.acceleration > 1.25:
            return "accelerating"
        if self.acceleration < 0.6:
            return "decaying"
        return "plateau"


def _rate(a: Snapshot, b: Snapshot) -> float:
    hours = (b.observed_at - a.observed_at).total_seconds() / 3600.0
    # Two snapshots minutes apart produce a rate dominated by counter jitter -
    # YouTube's public view count updates in lumps, not continuously.
    if hours < 0.5:
        return 0.0
    return max(b.views - a.views, 0) / hours


def velocity_from_history(history: Sequence[Snapshot], title: str = "") -> Optional[Velocity]:
    if len(history) < 2:
        return None
    h = sorted(history, key=lambda s: s.observed_at)
    recent = _rate(h[-2], h[-1])
    prior = _rate(h[-3], h[-2]) if len(h) >= 3 else 0.0
    hours = (h[-1].observed_at - h[0].observed_at).total_seconds() / 3600.0
    # Guard the ratio: a prior rate of zero would make acceleration infinite.
    accel = recent / prior if prior > 0 else (1.0 if recent == 0 else 2.0)
    return Velocity(
        video_id=h[-1].video_id,
        title=title,
        views=h[-1].views,
        vph_recent=recent,
        vph_prior=prior,
        acceleration=accel,
        hours_tracked=hours,
        n_snapshots=len(h),
    )


def rank_rising(store: SnapshotStore, limit: int = 25) -> list[Velocity]:
    """Videos still building, hottest first.

    Ranked by recent rate *and* acceleration together, so a video doing 50k/hr
    and slowing loses to one doing 20k/hr and doubling.
    """
    titles = store.titles()
    out = []
    for vid in store.videos_with_history(min_snapshots=2):
        v = velocity_from_history(store.history(vid), titles.get(vid, ""))
        if v and v.vph_recent > 0:
            out.append(v)
    out.sort(key=lambda v: v.vph_recent * min(v.acceleration, 3.0), reverse=True)
    return out[:limit]


@dataclass
class TopicTrend:
    topic: str
    n_videos: int
    median_vph: float
    accelerating_share: float

    @property
    def verdict(self) -> str:
        if self.n_videos < 3:
            return "too thin to call"
        if self.accelerating_share > 0.5:
            return "RISING - publish into this now"
        if self.accelerating_share < 0.2:
            return "saturated - arriving late"
        return "steady"


def topic_trends(vels: Sequence[Velocity], keywords: Sequence[str]) -> list[TopicTrend]:
    """Group tracked videos by keyword and judge each topic's phase."""
    out = []
    for kw in keywords:
        hits = [v for v in vels if kw.lower() in v.title.lower()]
        if not hits:
            continue
        accel = sum(1 for v in hits if v.phase == "accelerating") / len(hits)
        out.append(
            TopicTrend(
                topic=kw,
                n_videos=len(hits),
                median_vph=median([v.vph_recent for v in hits]),
                accelerating_share=accel,
            )
        )
    out.sort(key=lambda t: (t.accelerating_share, t.median_vph), reverse=True)
    return out
