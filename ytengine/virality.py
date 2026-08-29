"""Virality predictor: a 0-100 score, a bucket, and honest accuracy for both.

Two outputs per video, because they answer different questions:

  score 0-100  where this video's predicted performance sits in the corpus
               distribution. Comparative. Good for ranking your own drafts.
  bucket       an absolute view band. Harder, and the honest answer is that
               nobody hits it reliably from public data.

The headline metric is **pairwise win rate**: shown two videos, how often does
the model correctly say which got more views? It is reported because plain
accuracy is misleading on a skewed corpus - if half your videos flop, always
saying FLOP scores 50% while knowing nothing. Win rate has a fixed, meaningful
null: 50% is a coin flip, no matter how lopsided the classes are.

Calibration is reported separately from accuracy, because they fail
independently. A model can rank perfectly (win rate 0.75) while every number
it prints is wrong by 10x. Ranking is what you use to choose between drafts;
calibration is what you would need to promise someone a view count. This
engine is built to be trusted for the first and not the second.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .backtest import (CHANNEL_PRIOR_FEATURES, FEATURES, Sample, _fit, _predict,
                       build_vocabulary, temporal_split)
from .stats import median, quantile, spearman

# The bands, in absolute views.
BANDS = [
    ("FLOP", 0, 5_000),
    ("LOW", 5_000, 15_000),
    ("JAIL", 15_000, 50_000),
    ("MID", 50_000, 500_000),
    ("BIG", 500_000, 2_000_000),
    ("VIRAL", 2_000_000, 10 ** 15),
]
BAND_NAMES = [b[0] for b in BANDS]


def band_of(views: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= views < hi:
            return name
    return BAND_NAMES[-1]


def band_index(name: str) -> int:
    return BAND_NAMES.index(name)


@dataclass
class Prediction:
    video_id: str
    title: str
    score: int              # 0-100
    band: str
    predicted_views: int
    actual_views: Optional[int] = None

    @property
    def actual_band(self) -> Optional[str]:
        return band_of(self.actual_views) if self.actual_views is not None else None

    @property
    def correct(self) -> Optional[bool]:
        ab = self.actual_band
        return None if ab is None else ab == self.band

    @property
    def bands_off(self) -> Optional[int]:
        ab = self.actual_band
        return None if ab is None else abs(band_index(ab) - band_index(self.band))


@dataclass
class ViralityReport:
    n_train: int
    n_test: int
    win_rate: float               # pairwise ranking accuracy
    win_rate_pairs: int
    exact: float
    within_one: float
    majority_exact: float
    prior_exact: float
    spearman: float
    decile_actual: list[tuple[int, float, int]] = field(default_factory=list)
    band_rows: list[tuple[str, int, float, float]] = field(default_factory=list)
    calibration: list[tuple[str, float, float, int]] = field(default_factory=list)
    top_lift: float = 0.0

    @property
    def verdict(self) -> str:
        if self.win_rate < 0.55:
            return "COIN FLIP - the ranking carries no usable information"
        if self.exact <= self.prior_exact + 0.01:
            return (f"RANKS BUT DOES NOT FORECAST - win rate {self.win_rate:.0%} is real, "
                    "but absolute bands beat no baseline")
        return "USABLE - beats both the ranking null and the band baselines"


def pairwise_win_rate(pred: Sequence[float], truth: Sequence[float],
                      max_pairs: int = 400_000) -> tuple[float, int]:
    """Of all comparable pairs, how often is the predicted order right?

    Ties in truth are skipped rather than counted as wins - scoring them as
    correct inflates the rate on any corpus with repeated view counts.
    Sub-sampled deterministically above `max_pairs` so a big test set does not
    turn an O(n^2) sweep into minutes of compute.
    """
    n = len(pred)
    if n < 2:
        return 0.0, 0
    total_pairs = n * (n - 1) // 2
    stride = max(1, total_pairs // max_pairs)

    wins = 0.0
    comparable = k = 0
    for i in range(n):
        for j in range(i + 1, n):
            k += 1
            if k % stride:
                continue
            if truth[i] == truth[j]:
                continue
            comparable += 1
            if pred[i] == pred[j]:
                # Half credit for a tied prediction, the standard AUC
                # convention. Scoring ties as correct is not a rounding
                # choice - it means a model that outputs one constant for
                # every video scores a perfect 100%, because `False == False`
                # reads as agreement. The metric has to punish indecision,
                # not reward it.
                wins += 0.5
            elif (pred[i] > pred[j]) == (truth[i] > truth[j]):
                wins += 1
    return (wins / comparable if comparable else 0.0), comparable


def _scores_from(preds: Sequence[float], reference: Sequence[float]) -> list[int]:
    """Map predicted log-views onto 0-100 by percentile within the reference.

    A percentile rather than a scaled raw prediction: the score is meant to be
    read as "better than N% of what this corpus produces", which stays
    interpretable across niches whose absolute view scales differ by 100x.
    """
    ref = sorted(reference)
    if not ref:
        return [50] * len(preds)
    out = []
    for p in preds:
        lo, hi = 0, len(ref)
        while lo < hi:
            mid = (lo + hi) // 2
            if ref[mid] < p:
                lo = mid + 1
            else:
                hi = mid
        out.append(max(0, min(100, round(100 * lo / len(ref)))))
    return out


def run_virality(samples: Sequence[Sample], test_frac: float = 0.25,
                 alpha: float = 100.0, use_topics: bool = False,
                 use_channel_prior: bool = True
                 ) -> Optional[tuple[ViralityReport, list[Prediction]]]:
    feats = [f for f in FEATURES if use_channel_prior or f not in CHANNEL_PRIOR_FEATURES]
    train, test = temporal_split(samples, test_frac)
    if len(train) < 100 or len(test) < 30:
        return None

    vocab = build_vocabulary(train) if use_topics else []
    coefs = _fit(train, feats, alpha, vocab)
    if coefs is None:
        return None

    train_pred = [_predict(coefs, s, feats, vocab) for s in train]
    pred_log = [_predict(coefs, s, feats, vocab) for s in test]
    truth_log = [s.log_views for s in test]
    scores = _scores_from(pred_log, train_pred)

    preds = [
        Prediction(video_id=s.video.video_id, title=s.video.title,
                   score=sc, band=band_of(10 ** p), predicted_views=int(10 ** p),
                   actual_views=s.video.views)
        for s, p, sc in zip(test, pred_log, scores)
    ]

    wr, pairs = pairwise_win_rate(pred_log, truth_log)
    exact = sum(1 for p in preds if p.correct) / len(preds)
    within1 = sum(1 for p in preds if p.bands_off <= 1) / len(preds)

    counts: dict[str, int] = {}
    for s in train:
        counts[band_of(s.video.views)] = counts.get(band_of(s.video.views), 0) + 1
    majority = max(counts, key=counts.get) if counts else BAND_NAMES[0]
    maj = sum(1 for p in preds if p.actual_band == majority) / len(preds)
    prior = sum(1 for s, p in zip(test, preds)
                if band_of(10 ** s.features["chan_prior_log_median"]) == p.actual_band) / len(preds)

    # Decile lift: does a higher score actually mean more views?
    order = sorted(zip(scores, truth_log), key=lambda x: x[0])
    decile = []
    per = max(len(order) // 10, 1)
    for d in range(10):
        chunk = order[d * per:(d + 1) * per] if d < 9 else order[9 * per:]
        if chunk:
            decile.append((d + 1, 10 ** median([t for _, t in chunk]), len(chunk)))
    if len(decile) >= 2 and decile[0][1] > 0:
        report_lift = decile[-1][1] / decile[0][1]
    else:
        report_lift = 0.0

    band_rows = []
    for b in BAND_NAMES:
        n = sum(1 for p in preds if p.actual_band == b)
        got = sum(1 for p in preds if p.band == b)
        tp = sum(1 for p in preds if p.band == b and p.actual_band == b)
        band_rows.append((b, n, tp / got if got else 0.0, tp / n if n else 0.0))

    # Calibration: within each predicted band, what did those videos really do?
    calib = []
    for b in BAND_NAMES:
        got = [p for p in preds if p.band == b]
        if got:
            calib.append((b, median([float(p.predicted_views) for p in got]),
                          median([float(p.actual_views) for p in got]), len(got)))

    return ViralityReport(
        n_train=len(train), n_test=len(test), win_rate=wr, win_rate_pairs=pairs,
        exact=exact, within_one=within1, majority_exact=maj, prior_exact=prior,
        spearman=spearman(pred_log, truth_log), decile_actual=decile,
        band_rows=band_rows, calibration=calib, top_lift=report_lift,
    ), preds


def render(r: ViralityReport, preds: Sequence[Prediction], show: int = 12) -> str:
    o = ["", "VIRALITY PREDICTOR", "==================", ""]
    o.append(f"  trained {r.n_train:,}  tested {r.n_test:,} (strictly later videos)")
    o.append("")
    o.append(f"  WIN RATE          {r.win_rate:6.1%}   "
             f"(coin flip = 50%, {r.win_rate_pairs:,} pairs)")
    o.append(f"  rank correlation  {r.spearman:6.3f}")
    o.append(f"  exact band        {r.exact:6.1%}")
    o.append(f"  within one band   {r.within_one:6.1%}")
    o.append("")
    o.append(f"  baselines:  always-commonest {r.majority_exact:.1%}   "
             f"channel-history {r.prior_exact:.1%}")
    o.append(f"\n  VERDICT: {r.verdict}")

    o.append("\n  DOES A HIGHER SCORE MEAN MORE VIEWS?")
    o.append("    score decile   median actual views      n")
    for d, med, n in r.decile_actual:
        o.append(f"      {d:>2}          {med:>15,.0f}   {n:>5}")
    if r.top_lift:
        o.append(f"    top decile earns {r.top_lift:,.1f}x the bottom decile")

    o.append("\n  CALIBRATION (is the printed number itself trustworthy?)")
    o.append("    predicted band   median predicted   median actual      n")
    for b, p, a, n in r.calibration:
        flag = "" if 0.5 <= (p / a if a else 0) <= 2.0 else "   <-- off"
        o.append(f"    {b:<14} {p:>16,.0f} {a:>15,.0f} {n:>6}{flag}")

    o.append("\n  PER BAND        actual n   precision   recall")
    for b, n, p, rec in r.band_rows:
        o.append(f"    {b:<8} {n:>12}   {p:>8.1%} {rec:>8.1%}")

    o.append(f"\n  SAMPLE PREDICTIONS (highest scored)")
    o.append(f"    {'score':>5} {'predicted':>10} {'actual':>12}  {'hit':>4}  title")
    for p in sorted(preds, key=lambda x: -x.score)[:show]:
        hit = "OK" if p.correct else f"{p.bands_off} off"
        o.append(f"    {p.score:>5} {p.band:>10} {p.actual_views:>12,}  {hit:>6}  "
                 f"{p.title[:42]}")
    return "\n".join(o)
