"""Channel audit: what to make more of, what to kill.

Groups a channel's own uploads into formats, measures each format's median
multiplier, and turns that into a keep/kill call.

The bar is deliberately the channel's *own* median, not any external
benchmark. A format doing 0.7x is not bad television - it may be perfectly
good - but it is worse than the average thing this channel could have spent
that week making, and a week is the real currency.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

from .metrics import MATURITY_DAYS, Scored, score_channel
from .models import Video
from .stats import median, quantile

DURATION_BUCKETS = [
    ("short (<3m)", 0, 180),
    ("brief (3-8m)", 180, 480),
    ("standard (8-15m)", 480, 900),
    ("long (15-30m)", 900, 1800),
    ("extended (30m+)", 1800, 10 ** 9),
]


def duration_bucket(seconds: int) -> str:
    for label, lo, hi in DURATION_BUCKETS:
        if lo <= seconds < hi:
            return label
    return "unknown"


@dataclass
class FormatVerdict:
    label: str
    n: int
    median_multiplier: float
    best_multiplier: float
    median_views: float
    median_engagement: float

    @property
    def call(self) -> str:
        if self.n < 3:
            return "TOO FEW - test 2-3 more before judging"
        if self.median_multiplier >= 1.5:
            return "DOUBLE DOWN"
        if self.median_multiplier >= 1.0:
            return "keep"
        if self.median_multiplier >= 0.7:
            return "marginal - only if cheap to make"
        return "KILL"


@dataclass
class CadenceFinding:
    uploads_per_week: float
    consistency: str
    note: str


@dataclass
class ChannelAudit:
    channel_title: str
    n_videos: int
    n_mature: int
    formats: list[FormatVerdict]
    cadence: CadenceFinding
    best: list[Scored]
    worst: list[Scored]
    engagement_median: float
    hit_rate: float  # share of mature videos beating their own baseline


def _engagement(vs: Sequence[Video]) -> float:
    return median([v.engagement_rate() for v in vs]) if vs else 0.0


def audit_cadence(videos: Sequence[Video], now: Optional[datetime] = None) -> CadenceFinding:
    """Upload rhythm, and whether it is steady enough to build a habit.

    Consistency is measured as the spread of gaps between uploads relative to
    the typical gap. A channel posting every 7 days is training an audience; a
    channel posting after 2, then 31, then 5 days is not, even at the same
    average rate.
    """
    now = now or datetime.now(timezone.utc)
    if len(videos) < 4:
        return CadenceFinding(0.0, "unknown", "too few uploads to read a rhythm")

    dates = sorted(v.published_at for v in videos)
    span_days = (dates[-1] - dates[0]).total_seconds() / 86400.0
    if span_days <= 0:
        return CadenceFinding(0.0, "unknown", "all uploads share a timestamp")

    per_week = (len(dates) - 1) / (span_days / 7.0)
    gaps = [
        (b - a).total_seconds() / 86400.0
        for a, b in zip(dates, dates[1:])
    ]
    typical = median(gaps)
    spread = (quantile(gaps, 0.75) - quantile(gaps, 0.25)) / typical if typical > 0 else 0.0

    if spread < 0.4:
        consistency, note = "steady", "predictable rhythm - keep it"
    elif spread < 1.0:
        consistency, note = "uneven", "gaps vary enough to weaken return-viewer habit"
    else:
        consistency, note = "erratic", "no rhythm an audience can anticipate"

    silent = (now - dates[-1]).total_seconds() / 86400.0
    if silent > max(typical * 3, 21):
        note += f"; dormant {silent:.0f}d - recent uploads will underperform on restart"
    return CadenceFinding(per_week, consistency, note)


def audit_channel(
    videos: Sequence[Video],
    channel_title: str = "",
    now: Optional[datetime] = None,
) -> ChannelAudit:
    now = now or datetime.now(timezone.utc)
    scored = score_channel(videos, now)
    settled = [s for s in scored if s.trustworthy]

    by_bucket: dict[str, list[Scored]] = {}
    for s in settled:
        by_bucket.setdefault(duration_bucket(s.video.duration_s), []).append(s)

    formats = [
        FormatVerdict(
            label=label,
            n=len(rows),
            median_multiplier=median([r.multiplier for r in rows]),
            best_multiplier=max(r.multiplier for r in rows),
            median_views=median([float(r.video.views) for r in rows]),
            median_engagement=_engagement([r.video for r in rows]),
        )
        for label, rows in by_bucket.items()
    ]
    formats.sort(key=lambda f: f.median_multiplier, reverse=True)

    hits = sum(1 for s in settled if s.multiplier > 1.0)
    return ChannelAudit(
        channel_title=channel_title or (videos[0].channel_title if videos else ""),
        n_videos=len(videos),
        n_mature=len(settled),
        formats=formats,
        cadence=audit_cadence(videos, now),
        best=settled[:5],
        worst=sorted(settled, key=lambda s: s.multiplier)[:5],
        engagement_median=_engagement([s.video for s in settled]),
        hit_rate=hits / len(settled) if settled else 0.0,
    )
