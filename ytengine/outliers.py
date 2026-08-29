"""The opportunity finder.

Scores every video in a multi-channel corpus against its own channel's
baseline, then looks for what the winners have in common.

The two filters that make the output actionable rather than merely interesting:

  * Small-channel bias. A 5x on a 2M-subscriber channel is mostly the
    algorithm re-serving an existing audience. A 5x on a 20k-subscriber
    channel means the video reached people who had never heard of them -
    which is the only thing that transfers to you.

  * Repeatability. One breakout is an anecdote. A format that produced three
    separate outliers, on more than one channel, is a pattern worth copying.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

from .metrics import AgeCurve, Scored, score_channel
from .models import Video
from .packaging import title_features
from .stats import median


@dataclass
class Opportunity:
    scored: Scored
    subscribers: int
    reach_ratio: float  # views per subscriber - did it escape the existing audience?

    @property
    def video(self) -> Video:
        return self.scored.video

    @property
    def multiplier(self) -> float:
        return self.scored.multiplier

    @property
    def escaped_audience(self) -> bool:
        """Views exceeding the subscriber count means genuine outward reach."""
        return self.reach_ratio > 1.0


def find_opportunities(
    corpus: dict[str, Sequence[Video]],
    subscribers: dict[str, int],
    now: Optional[datetime] = None,
    min_multiplier: float = 2.0,
    include_provisional: bool = False,
    max_subscribers: Optional[int] = None,
) -> list[Opportunity]:
    """Rank outliers across many channels.

    `corpus` maps channel_id -> that channel's uploads. The age curve is fitted
    once across the pooled corpus rather than per channel: a single channel
    rarely has enough videos per age bucket to fit a stable curve, while the
    pooled set does, and accumulation shape is far more a property of the
    platform than of any one creator.
    """
    now = now or datetime.now(timezone.utc)
    pooled = [v for vids in corpus.values() for v in vids]
    curve = AgeCurve.fit(pooled, now)

    out: list[Opportunity] = []
    for cid, vids in corpus.items():
        subs = subscribers.get(cid, 0)
        if max_subscribers is not None and subs > max_subscribers:
            continue
        for s in score_channel(vids, now, curve):
            if not s.baseline_reliable:
                continue
            if s.provisional and not include_provisional:
                continue
            if s.multiplier < min_multiplier:
                continue
            out.append(
                Opportunity(
                    scored=s,
                    subscribers=subs,
                    reach_ratio=s.video.views / subs if subs > 0 else 0.0,
                )
            )
    out.sort(key=lambda o: o.multiplier, reverse=True)
    return out


@dataclass
class Pattern:
    feature: str
    outlier_rate: float   # share of outliers with this feature
    baseline_rate: float  # share of everything else with it
    lift: float           # outlier_rate / baseline_rate
    n_outliers: int

    @property
    def meaningful(self) -> bool:
        """Needs both a real effect and enough support to not be noise."""
        return self.n_outliers >= 4 and (self.lift > 1.4 or self.lift < 0.7)

    @property
    def direction(self) -> str:
        return "over-represented" if self.lift > 1 else "under-represented"


def extract_patterns(
    opportunities: Sequence[Opportunity],
    corpus: dict[str, Sequence[Video]],
) -> list[Pattern]:
    """What do the outliers share that the ordinary videos don't?

    Reported as lift over the base rate, not as a raw percentage. "80% of
    outliers have a number in the title" is meaningless if 80% of all videos
    do. Only the gap between those two rates carries information.
    """
    winners = {o.video.video_id for o in opportunities}
    if len(winners) < 4:
        return []

    all_videos = [v for vids in corpus.values() for v in vids]
    win_v = [v for v in all_videos if v.video_id in winners]
    rest_v = [v for v in all_videos if v.video_id not in winners]
    if not rest_v:
        return []

    out: list[Pattern] = []
    binary = ["has_number", "has_year", "is_question", "has_bracket",
              "has_colon", "first_person", "second_person", "starts_with_verb"]
    for feat in binary:
        wr = sum(title_features(v.title)[feat] for v in win_v) / len(win_v)
        br = sum(title_features(v.title)[feat] for v in rest_v) / len(rest_v)
        if br <= 0.01:
            continue  # too rare overall for a lift ratio to be stable
        out.append(
            Pattern(feature=feat, outlier_rate=wr, baseline_rate=br,
                    lift=wr / br, n_outliers=int(wr * len(win_v)))
        )

    # Duration is continuous, so compare medians as a ratio instead of a rate.
    for label, sel in (("long-form", False), ("shorts", True)):
        w = [v.duration_s for v in win_v if v.is_short == sel]
        r = [v.duration_s for v in rest_v if v.is_short == sel]
        if len(w) >= 4 and len(r) >= 4 and median(r) > 0:
            out.append(
                Pattern(feature=f"{label} duration", outlier_rate=median(w),
                        baseline_rate=median(r), lift=median(w) / median(r),
                        n_outliers=len(w))
            )

    out.sort(key=lambda p: abs(p.lift - 1.0), reverse=True)
    return out
