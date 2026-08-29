"""Synthetic corpus generator, for offline runs and tests.

Two jobs:

  * Let someone run every analysis end to end without an API key, so the tool
    can be evaluated before anyone provisions quota.

  * Give the test suite a corpus with a *known planted signal*, so we can
    assert the engine recovers it. A scorer that cannot rediscover an effect
    we deliberately injected is not measuring anything.

This data is fake and labelled as such everywhere it surfaces. It is a
harness, never a source of real-world conclusions.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Optional

from .models import Video

_TOPICS = [
    "beginner mistakes", "gear setup", "full tutorial", "workflow tour",
    "budget build", "pro teardown", "common myths", "field test",
    "first impressions", "one year later", "cheap vs expensive", "speedrun",
]
_HOOKS = ["Why", "How", "The Truth About", "Stop Making", "I Tried", "Nobody Tells You About"]


def make_corpus(
    n_channels: int = 12,
    videos_per_channel: int = 45,
    seed: int = 7,
    now: Optional[datetime] = None,
    number_boost: float = 1.9,
) -> tuple[dict[str, list[Video]], dict[str, int]]:
    """Build a corpus with a planted effect.

    Videos whose title carries a number are given a `number_boost` multiplier
    on views. Everything else - channel size, upload cadence, per-video luck -
    is noise drawn around that. `fit_packaging` and `extract_patterns` should
    both be able to find the planted feature and no other.
    """
    rng = random.Random(seed)
    now = now or datetime.now(timezone.utc)
    corpus: dict[str, list[Video]] = {}
    subs: dict[str, int] = {}

    for c in range(n_channels):
        cid = f"UC{c:022d}"
        # Channel sizes span three orders of magnitude, as a real niche does.
        subscribers = int(10 ** rng.uniform(3.3, 6.3))
        base_views = subscribers * rng.uniform(0.05, 0.5)
        cadence_days = rng.uniform(3.0, 12.0)
        subs[cid] = subscribers

        vids = []
        age = rng.uniform(0.4, 3.0)
        for i in range(videos_per_channel):
            age += rng.uniform(cadence_days * 0.5, cadence_days * 1.5)
            published = now - timedelta(days=age)

            # Each packaging feature is drawn independently. If they were
            # correlated - every numbered title also carrying a year, say -
            # the regression could not separate them, and a test asserting
            # that the engine recovers `has_number` would in fact be passing
            # on any of the three. Independence is what makes the fixture a
            # real test rather than a tautology.
            has_number = rng.random() < 0.35
            has_year = rng.random() < 0.30
            has_bracket = rng.random() < 0.30
            topic = rng.choice(_TOPICS)
            hook = rng.choice(_HOOKS)

            # Every title gets a hook; the number is layered on top rather
            # than replacing it. Otherwise `has_number` would be the exact
            # complement of the hook-word features, and the planted effect
            # would be indistinguishable from "lacks a curiosity word".
            head = f"{hook} {topic}"
            if has_number:
                head = f"{rng.choice([3,5,7,10,12])} {head}"
            if has_year:
                head += f" {rng.choice(['2024','2025'])}"
            title = f"{head} [{rng.choice(['Full Guide','Update','Part 2'])}]" if has_bracket else head

            is_short = rng.random() < 0.2
            duration = rng.randint(20, 90) if is_short else rng.randint(300, 1900)

            # Lognormal luck: most videos cluster, a few break out hard.
            luck = rng.lognormvariate(0.0, 0.55)
            boost = number_boost if has_number else 1.0
            fmt_scale = 0.6 if is_short else 1.0
            mature_views = base_views * fmt_scale * luck * boost

            # Apply the same accumulation shape the age curve expects to find.
            frac = 1.0 if age >= 30 else min(1.0, 0.2 + 0.8 * (age / 30) ** 0.55)
            views = max(int(mature_views * frac), 1)

            er = rng.uniform(0.02, 0.06)
            likes = int(views * er * 0.9)
            comments = int(views * er * 0.1)

            vids.append(
                Video(
                    video_id=f"vid{c:03d}{i:04d}",
                    channel_id=cid,
                    channel_title=f"Demo Channel {c}",
                    title=title,
                    description="synthetic fixture data",
                    published_at=published,
                    duration_s=duration,
                    views=views,
                    likes=likes,
                    comments=comments,
                    tags=[topic],
                    thumbnail_url="",
                )
            )
        corpus[cid] = vids
    return corpus, subs
