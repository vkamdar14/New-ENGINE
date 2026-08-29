"""Baselines, age adjustment, and the outlier multiplier.

The one number this engine is built around is the *multiplier*: how many times
its own channel's normal performance a video achieved.

Raw view counts are useless for deciding what to make. A 200k-view video is a
catastrophe on a channel that normally does 2M and a career-defining breakout
on a channel that normally does 5k. The multiplier removes channel size, which
is exactly the variable you cannot copy, and leaves format and packaging, which
are exactly the variables you can.

Two corrections make it honest:

1. Format split. Shorts and long-form are different distribution systems. A
   Short doing 10x a channel's long-form median says nothing.

2. Age adjustment. A 4-day-old video has not finished earning its views, so
   comparing it to mature videos scores every recent upload as a failure.
   We correct with an empirical age curve fitted on the corpus itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

from .models import Video
from .stats import median, quantile

# Below this age a video is still accumulating fast enough that its multiplier
# is provisional rather than settled.
MATURITY_DAYS = 30.0

# Age buckets (in days) for the empirical growth curve.
_AGE_BUCKETS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 14), (14, 21), (21, 30)]


@dataclass
class AgeCurve:
    """Fraction of mature (30-day) views a video has typically earned by age t.

    Fitted cross-sectionally: we take the median views of corpus videos in each
    age bucket and divide by the median views of mature videos.

    The assumption this makes - and it is a real one, so it is stated rather
    than buried - is that the channels in the corpus were producing
    comparable-performing videos across the whole window. If a channel doubled
    in size two weeks ago, its recent videos are genuinely better, and this
    curve will read that growth as "videos mature fast". Over a corpus of many
    channels those idiosyncratic trends largely cancel; over a single small
    channel they do not, which is why `fit` refuses to fit below `min_n` and
    falls back to a conservative default curve instead.
    """

    points: list[tuple[float, float]]  # (age_days, fraction) ascending
    fitted: bool = False

    @classmethod
    def default(cls) -> "AgeCurve":
        # A typical YouTube accumulation shape: roughly half of the first
        # month's views land in the first three days, then it flattens.
        return cls(
            points=[(0.5, 0.20), (1.5, 0.34), (2.5, 0.45), (4, 0.55),
                    (6, 0.64), (10.5, 0.76), (17.5, 0.87), (25.5, 0.95), (30, 1.0)],
            fitted=False,
        )

    @classmethod
    def fit(
        cls,
        videos: Sequence[Video],
        now: Optional[datetime] = None,
        min_n: int = 8,
        min_per_bucket: int = 4,
    ) -> "AgeCurve":
        """Fit the curve on within-channel view ratios.

        The naive fit - median views per age bucket, divided by the median
        views of all mature videos - is badly unstable, because channel size
        varies by three orders of magnitude inside a normal corpus. Whether a
        given age bucket happens to contain uploads from the big channels or
        the small ones then moves the fitted fraction far more than age does.

        So we normalise first: every video becomes a ratio against its own
        channel-and-format mature median, which removes channel size entirely.
        Only then do we bucket by age. What is left in the bucket spread is
        accumulation shape, which is the thing we were trying to measure.
        """
        now = now or datetime.now(timezone.utc)

        # Per channel and format, the mature median that defines "1.0".
        refs: dict[tuple[str, bool], float] = {}
        groups: dict[tuple[str, bool], list[Video]] = {}
        for v in videos:
            groups.setdefault((v.channel_id, v.is_short), []).append(v)
        for key, vids in groups.items():
            mature = [float(x.views) for x in vids if x.age_days(now) >= MATURITY_DAYS]
            if len(mature) >= 3:
                m = median(mature)
                if m > 0:
                    refs[key] = m
        if len(refs) < 2:
            return cls.default()

        ratios: list[tuple[float, float]] = []
        for v in videos:
            ref = refs.get((v.channel_id, v.is_short))
            if ref:
                ratios.append((v.age_days(now), v.views / ref))
        if len(ratios) < min_n:
            return cls.default()

        pts: list[tuple[float, float]] = []
        for lo, hi in _AGE_BUCKETS:
            bucket = [r for age, r in ratios if lo <= age < hi]
            if len(bucket) >= min_per_bucket:
                pts.append(((lo + hi) / 2.0, min(median(bucket), 1.0)))
        if len(pts) < 3:
            return cls.default()

        # Views only accumulate, so the curve must be non-decreasing. Sampling
        # noise can invert adjacent buckets; enforce monotonicity.
        pts.sort()
        out, running = [], 0.0
        for age, frac in pts:
            running = max(running, frac)
            out.append((age, running))
        out.append((MATURITY_DAYS, 1.0))
        return cls(points=out, fitted=True)

    def fraction_at(self, age_days: float) -> float:
        """Linear interpolation, clamped to (0, 1]."""
        if age_days >= MATURITY_DAYS:
            return 1.0
        pts = self.points
        if age_days <= pts[0][0]:
            # Never divide by ~0 - it would turn a 6-hour-old video into an
            # infinite multiplier and dominate every ranking.
            return max(pts[0][1], 0.02)
        for (a0, f0), (a1, f1) in zip(pts, pts[1:]):
            if a0 <= age_days <= a1:
                span = a1 - a0
                t = (age_days - a0) / span if span else 0.0
                return max(f0 + (f1 - f0) * t, 0.02)
        return 1.0


@dataclass
class Baseline:
    """A channel's normal performance for one format."""

    channel_id: str
    is_short: bool
    median_views: float
    p25: float
    p75: float
    n: int
    reliable: bool

    @property
    def spread_ratio(self) -> float:
        """p75/p25. High means the channel is already streaky - its outliers
        are less likely to be a repeatable format and more likely to be luck."""
        return self.p75 / self.p25 if self.p25 > 0 else 0.0


def build_baseline(
    videos: Sequence[Video],
    is_short: bool,
    now: Optional[datetime] = None,
    exclude_id: str = "",
    min_n: int = 5,
) -> Optional[Baseline]:
    """Median performance of one channel's mature videos in one format.

    Only mature videos define the baseline. Including a 2-day-old upload would
    drag the median down and inflate every multiplier measured against it.
    """
    now = now or datetime.now(timezone.utc)
    cohort = [
        v for v in videos
        if v.is_short == is_short
        and v.age_days(now) >= MATURITY_DAYS
        and v.video_id != exclude_id
    ]
    if not cohort:
        return None
    views = [float(v.views) for v in cohort]
    return Baseline(
        channel_id=cohort[0].channel_id,
        is_short=is_short,
        median_views=median(views),
        p25=quantile(views, 0.25),
        p75=quantile(views, 0.75),
        n=len(cohort),
        reliable=len(cohort) >= min_n,
    )


@dataclass
class Scored:
    video: Video
    multiplier: float
    expected_views: float
    provisional: bool       # too young for a settled verdict
    baseline_reliable: bool  # channel had enough mature videos to compare against

    @property
    def trustworthy(self) -> bool:
        return self.baseline_reliable and not self.provisional


def score_video(
    video: Video,
    baseline: Baseline,
    curve: AgeCurve,
    now: Optional[datetime] = None,
) -> Scored:
    """Multiplier = actual views / views this channel would normally have by now."""
    now = now or datetime.now(timezone.utc)
    age = video.age_days(now)
    expected = baseline.median_views * curve.fraction_at(age)
    multiplier = video.views / expected if expected > 0 else 0.0
    return Scored(
        video=video,
        multiplier=multiplier,
        expected_views=expected,
        provisional=age < MATURITY_DAYS,
        baseline_reliable=baseline.reliable,
    )


def score_channel(
    videos: Sequence[Video],
    now: Optional[datetime] = None,
    curve: Optional[AgeCurve] = None,
) -> list[Scored]:
    """Score every video on a channel against its own format's baseline."""
    now = now or datetime.now(timezone.utc)
    curve = curve or AgeCurve.fit(videos, now)
    out: list[Scored] = []
    for fmt in (False, True):
        fmt_videos = [v for v in videos if v.is_short == fmt]
        if not fmt_videos:
            continue
        # Leave-one-out: a breakout must not be allowed to raise the bar it is
        # being measured against. On a 20-video channel a single 10x video
        # shifts the median enough to disguise itself as a 6x.
        for v in fmt_videos:
            base = build_baseline(videos, fmt, now, exclude_id=v.video_id)
            if base:
                out.append(score_video(v, base, curve, now))
    out.sort(key=lambda s: s.multiplier, reverse=True)
    return out
