"""Extra pre-publication signals, and the machinery to prove they earn a place.

The backtest already established the ceiling: a static channel median explains
almost everything, and structural title features explain nothing. So new
features only matter if they capture something a *static* median cannot -
which means change over time, or context outside the channel.

Five families here, each chosen against that bar:

  momentum   a channel's median is a level; these are its derivative. A
             channel that tripled over its last five uploads is not the same
             bet as one that has been flat for a year at the same median.
  fatigue    the same joke posted eleven times decays. Novelty against a
             channel's own recent titles measures how much of that is left.
  cadence    the gap since the last upload, and whether posting is regular.
  saturation how crowded this video's topic was in the corpus at publish time.
  format     hashtag and tag usage, which are choices rather than outcomes.

Every one is computed strictly from information available before publishing:
earlier videos only, matured earlier videos where a view count is involved.

None of this is assumed to work. `signals.py` exists so the claim can be
tested, and the honest outcome of that test is reported either way.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence

from .models import Video
from .stats import median

_TOKEN = re.compile(r"[a-z0-9']{3,}")
_HASHTAG = re.compile(r"#\w+")

SIGNAL_FEATURES = [
    "mom_ratio", "mom_slope", "mom_best_recent", "mom_volatility",
    "fatigue_novelty", "fatigue_repeat_rate",
    "cad_gap_days", "cad_regularity", "cad_uploads_30d",
    "sat_topic_density", "sat_topic_competition",
    "fmt_hashtags", "fmt_tag_count", "fmt_tag_overlap", "fmt_title_is_reused",
]


def tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class CorpusIndex:
    """Topic density over time, for the saturation signals.

    Built once over the whole corpus and queried per video with a strict
    `before` cutoff, so a video is never told how crowded its topic *became*.
    """

    by_token_day: dict[str, list[float]]

    @classmethod
    def build(cls, videos: Sequence[Video]) -> "CorpusIndex":
        idx: dict[str, list[float]] = defaultdict(list)
        for v in videos:
            ts = v.published_at.timestamp()
            for t in tokens(v.title):
                idx[t].append(ts)
        for t in idx:
            idx[t].sort()
        return cls(by_token_day=idx)

    def count_before(self, token: str, before: datetime, window_days: float = 30.0) -> int:
        stamps = self.by_token_day.get(token)
        if not stamps:
            return 0
        hi = before.timestamp()
        lo = hi - window_days * 86400
        # Bisect twice rather than scanning: this is called once per token per
        # video, which on a 200k-video corpus is tens of millions of calls.
        import bisect
        return bisect.bisect_left(stamps, hi) - bisect.bisect_left(stamps, lo)


def momentum(prior_log_views: Sequence[float], recent_n: int = 5) -> dict[str, float]:
    """Is the channel rising, flat, or sliding?

    `mom_ratio` compares the last few uploads to the channel's whole history.
    A static median cannot express this, which is the entire reason these
    features might add something the backtest has not already captured.
    """
    if len(prior_log_views) < 3:
        return {"mom_ratio": 0.0, "mom_slope": 0.0,
                "mom_best_recent": 0.0, "mom_volatility": 0.0}
    recent = prior_log_views[-recent_n:]
    overall = median(prior_log_views)
    # Difference of logs, i.e. a log ratio - stays finite when overall is 0.
    ratio = median(recent) - overall

    n = len(prior_log_views)
    xs = list(range(n))
    mx, my = (n - 1) / 2.0, sum(prior_log_views) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, prior_log_views)) / denom
             if denom else 0.0)

    dev = median([abs(v - overall) for v in prior_log_views])
    return {
        "mom_ratio": ratio,
        "mom_slope": slope,
        "mom_best_recent": max(recent) - overall,
        "mom_volatility": dev,
    }


def fatigue(title: str, prior_titles: Sequence[str], window: int = 10) -> dict[str, float]:
    """How much has this channel already used this idea?"""
    if not prior_titles:
        return {"fatigue_novelty": 1.0, "fatigue_repeat_rate": 0.0}
    me = tokens(title)
    recent = [tokens(t) for t in prior_titles[-window:]]
    sims = [jaccard(me, other) for other in recent]
    peak = max(sims) if sims else 0.0
    return {
        "fatigue_novelty": 1.0 - peak,
        # Share of recent uploads that are near-restatements of this one.
        "fatigue_repeat_rate": sum(1 for s in sims if s > 0.6) / len(sims),
    }


def cadence(published: datetime, prior_dates: Sequence[datetime]) -> dict[str, float]:
    if not prior_dates:
        return {"cad_gap_days": 0.0, "cad_regularity": 0.0, "cad_uploads_30d": 0.0}
    gap = (published - prior_dates[-1]).total_seconds() / 86400.0
    gaps = [(b - a).total_seconds() / 86400.0
            for a, b in zip(prior_dates, prior_dates[1:])][-20:]
    typical = median(gaps) if gaps else 0.0
    spread = median([abs(g - typical) for g in gaps]) if gaps else 0.0
    # High regularity means an audience can anticipate the next upload.
    regularity = 1.0 / (1.0 + spread / typical) if typical > 0 else 0.0
    cutoff = published - timedelta(days=30)
    return {
        "cad_gap_days": min(gap, 90.0),   # clipped: a 3-year gap is not 40x a 30-day one
        "cad_regularity": regularity,
        "cad_uploads_30d": float(sum(1 for d in prior_dates if d >= cutoff)),
    }


def saturation(title: str, published: datetime, index: CorpusIndex,
               top_k: int = 4) -> dict[str, float]:
    """How crowded was this topic when the video went out?"""
    tk = sorted(tokens(title))[:top_k]
    if not tk:
        return {"sat_topic_density": 0.0, "sat_topic_competition": 0.0}
    counts = [index.count_before(t, published) for t in tk]
    return {
        "sat_topic_density": math.log1p(sum(counts)),
        "sat_topic_competition": math.log1p(max(counts)),
    }


def format_signals(video: Video, prior: Sequence[Video]) -> dict[str, float]:
    prior_tags = set()
    for p in prior[-10:]:
        prior_tags.update(t.lower() for t in p.tags)
    mine = {t.lower() for t in video.tags}
    prior_titles = {p.title.strip().lower() for p in prior}
    return {
        "fmt_hashtags": float(len(_HASHTAG.findall(video.title))),
        "fmt_tag_count": float(len(video.tags)),
        "fmt_tag_overlap": jaccard(mine, prior_tags),
        "fmt_title_is_reused": 1.0 if video.title.strip().lower() in prior_titles else 0.0,
    }


def compute(video: Video, prior: Sequence[Video], prior_log_views: Sequence[float],
            index: Optional[CorpusIndex]) -> dict[str, float]:
    """All signal features for one video, from its causal past."""
    out: dict[str, float] = {}
    out.update(momentum(prior_log_views))
    out.update(fatigue(video.title, [p.title for p in prior]))
    out.update(cadence(video.published_at, [p.published_at for p in prior]))
    if index is not None:
        out.update(saturation(video.title, video.published_at, index))
    else:
        out.update({"sat_topic_density": 0.0, "sat_topic_competition": 0.0})
    out.update(format_signals(video, prior))
    return out
