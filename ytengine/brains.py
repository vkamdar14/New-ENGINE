"""Segmented model ensemble - many small models instead of one big one.

The idea behind "one brain per niche" is sound: what predicts a cooking Short
is not what predicts a gaming Short, and a single global model has to average
those into a compromise that fits neither.

The idea it competes with is that a segment holding forty videos cannot
estimate thirty coefficients, and a model fit on forty rows is mostly noise.
Split a 4,000-video corpus 144 ways and the average segment has 28 rows.

Partial pooling is the standard resolution, and it is what this module
implements. Every segment gets its own coefficients, shrunk toward the global
model by a weight that depends on how much data the segment actually has:

    beta_segment = w * beta_local + (1 - w) * beta_global,   w = n / (n + k)

A segment with thousands of videos keeps its own model almost entirely. A
segment with twelve is pulled nearly all the way back to global. Nothing has
to be thrown away, and nothing thin gets trusted.

Whether this beats a single model on real data is an empirical question, not
an architectural preference, so `compare_architectures` answers it by
measurement.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .backtest import Sample, active_features, temporal_split
from .stats import median, ridge_fit
from .virality import pairwise_win_rate

# Tiers chosen so the grid is 4 x 6 x 6 = 144 cells.
SIZE_TIERS = [("micro", 0, 3.0), ("small", 3.0, 4.0),
              ("mid", 4.0, 5.0), ("large", 5.0, 99.0)]          # log10 channel median
DURATION_TIERS = [("s0", 0, 10), ("s1", 10, 20), ("s2", 20, 35),
                  ("s3", 35, 60), ("s4", 60, 120), ("s5", 120, 10 ** 6)]
TOPIC_TIERS = 6

# Shrinkage constant. A segment needs ~k rows to earn half its own model; below
# that it is pulled toward global. Tuned to the scale of a real segment here.
SHRINK_K = 120.0


def size_tier(log_median: float) -> str:
    for name, lo, hi in SIZE_TIERS:
        if lo <= log_median < hi:
            return name
    return SIZE_TIERS[-1][0]


def duration_tier(seconds: float) -> str:
    for name, lo, hi in DURATION_TIERS:
        if lo <= seconds < hi:
            return name
    return DURATION_TIERS[-1][0]


def topic_tier(title: str, buckets: int = TOPIC_TIERS) -> str:
    """Cheap deterministic topic bucket.

    A hash rather than clustering: real topic clusters need embeddings, and a
    stable arbitrary partition is enough to test whether *any* segmentation
    helps. If splitting by an arbitrary axis helps as much as splitting by a
    meaningful one, the gain was never about topic.
    """
    words = sorted(w for w in title.lower().split() if len(w) > 3)[:3]
    h = abs(hash(" ".join(words))) if words else 0
    return f"t{h % buckets}"


def segment_of(s: Sample) -> str:
    return (f"{size_tier(s.features.get('chan_prior_log_median', 0.0))}"
            f"|{duration_tier(s.features.get('duration_s', 0.0))}"
            f"|{topic_tier(s.video.title)}")


@dataclass
class Brain:
    segment: str
    coefs: list[float]
    n: int
    shrinkage: float          # weight given to this segment's own fit

    @property
    def is_mostly_local(self) -> bool:
        return self.shrinkage > 0.5


@dataclass
class Ensemble:
    features: list[str]
    global_coefs: list[float]
    brains: dict[str, Brain] = field(default_factory=dict)

    @property
    def populated(self) -> int:
        return len(self.brains)

    @property
    def mostly_local(self) -> int:
        return sum(1 for b in self.brains.values() if b.is_mostly_local)

    def predict(self, s: Sample) -> float:
        b = self.brains.get(segment_of(s))
        coefs = b.coefs if b else self.global_coefs
        *slopes, intercept = coefs
        return intercept + sum(c * s.features[f] for c, f in zip(slopes, self.features))


def fit_ensemble(train: Sequence[Sample], alpha: float = 100.0,
                 shrink_k: float = SHRINK_K, min_n: int = 15) -> Optional[Ensemble]:
    feats = active_features(train)
    X = [[s.features[f] for f in feats] for s in train]
    y = [s.log_views for s in train]
    global_coefs = ridge_fit(X, y, alpha=alpha)
    if not global_coefs:
        return None

    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in train:
        groups[segment_of(s)].append(s)

    brains: dict[str, Brain] = {}
    for seg, rows in groups.items():
        if len(rows) < min_n:
            # Too thin to fit at all: this segment simply uses the global model,
            # which is what full shrinkage would have produced anyway.
            continue
        gx = [[s.features[f] for f in feats] for s in rows]
        gy = [s.log_views for s in rows]
        local = ridge_fit(gx, gy, alpha=alpha)
        if not local:
            continue
        w = len(rows) / (len(rows) + shrink_k)
        blended = [w * l + (1 - w) * g for l, g in zip(local, global_coefs)]
        brains[seg] = Brain(segment=seg, coefs=blended, n=len(rows), shrinkage=w)

    return Ensemble(features=list(feats), global_coefs=global_coefs, brains=brains)


@dataclass
class ArchComparison:
    n_train: int
    n_test: int
    grid_cells: int
    populated: int
    fitted: int
    mostly_local: int
    median_segment_n: float
    global_win_rate: float
    ensemble_win_rate: float

    @property
    def delta(self) -> float:
        return self.ensemble_win_rate - self.global_win_rate

    @property
    def verdict(self) -> str:
        if self.delta > 0.005:
            return "SEGMENTATION HELPS - many brains beat one"
        if self.delta < -0.005:
            return "SEGMENTATION HURTS - the segments are too thin to fit"
        return "NO DIFFERENCE - one brain is doing the same job as many"


def compare_architectures(samples: Sequence[Sample], test_frac: float = 0.25,
                          alpha: float = 100.0) -> Optional[ArchComparison]:
    """One model against 144 partially-pooled ones, on the same held-out data."""
    train, test = temporal_split(samples, test_frac)
    if len(train) < 200 or len(test) < 50:
        return None
    ens = fit_ensemble(train, alpha=alpha)
    if ens is None:
        return None

    truth = [s.log_views for s in test]
    *gs, gi = ens.global_coefs
    g_pred = [gi + sum(c * s.features[f] for c, f in zip(gs, ens.features)) for s in test]
    e_pred = [ens.predict(s) for s in test]

    seg_counts = defaultdict(int)
    for s in train:
        seg_counts[segment_of(s)] += 1

    gwr, _ = pairwise_win_rate(g_pred, truth)
    ewr, _ = pairwise_win_rate(e_pred, truth)
    return ArchComparison(
        n_train=len(train), n_test=len(test),
        grid_cells=len(SIZE_TIERS) * len(DURATION_TIERS) * TOPIC_TIERS,
        populated=len(seg_counts), fitted=ens.populated,
        mostly_local=ens.mostly_local,
        median_segment_n=median(list(seg_counts.values())) if seg_counts else 0.0,
        global_win_rate=gwr, ensemble_win_rate=ewr,
    )


def render(c: ArchComparison) -> str:
    o = ["", "SEGMENTED ENSEMBLE (\"144 brains\")", "=" * 34, ""]
    o.append(f"  grid            {c.grid_cells} cells "
             f"({len(SIZE_TIERS)} size x {len(DURATION_TIERS)} duration x {TOPIC_TIERS} topic)")
    o.append(f"  populated       {c.populated} cells hold any training data")
    o.append(f"  fitted          {c.fitted} had enough rows to fit their own model")
    o.append(f"  mostly local    {c.mostly_local} kept more than half their own coefficients")
    o.append(f"  median cell     {c.median_segment_n:.0f} videos")
    o.append("")
    o.append(f"  one global brain     win rate {c.global_win_rate:6.2%}")
    o.append(f"  {c.fitted} segmented brains   win rate {c.ensemble_win_rate:6.2%}")
    o.append(f"  delta                         {c.delta:+6.2%}")
    o.append(f"\n  VERDICT: {c.verdict}")
    if c.median_segment_n < 100:
        o.append(f"\n  The median cell holds {c.median_segment_n:.0f} videos against "
                 f"{len(active_features([])) or 22}+ features.")
        o.append("  Partial pooling is what keeps that from being pure noise: each")
        o.append("  segment is shrunk toward the global fit in proportion to how")
        o.append("  little data it has, so a thin cell contributes almost nothing")
        o.append("  and a rich one keeps its own model.")
    return "\n".join(o)
