"""Prediction backtest: guess the view bucket before publishing, then check.

This is the file that decides whether any of the rest is real. Everything else
describes the past; this makes a falsifiable forward claim and scores it.

Three rules make the score mean something. Break any one and the numbers get
better while the model gets worse:

1. **Pre-publication features only.** Views, likes and comments are outcomes,
   not inputs. Using them predicts views from views - perfect scores, zero use.
   Only what you know before you hit publish is allowed: title, duration, tags,
   timing, and how the channel has done historically.

2. **Temporal split, never random.** Train on the past, test on the future. A
   random split lets the model see a channel's July results while predicting
   its June ones, which is not a situation you will ever be in.

3. **Scored against baselines that need no model.** Always-guess-the-commonest
   bucket already scores ~49% here. A model at 52% has learned almost nothing,
   and would look impressive quoted alone. The number that matters is the gap
   over the baselines, and the hardest baseline - channel history alone -
   is the one worth beating.

The channel-history ablation is the interesting result: it separates "this
channel reliably gets 5k views" from "this packaging earned more than usual".
Only the second is something you can act on.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from .models import Video
from .packaging import CURIOSITY, STAKES
from .signals import SIGNAL_FEATURES, CorpusIndex, compute as compute_signals
from .stats import median, quantile, ridge_fit, spearman

# The buckets. Absolute view counts, because for a channel starting from zero
# "did it beat my own median" is not the question - "did it break out" is.
BUCKETS = [
    ("FLOP", 0, 5_000),
    ("JAIL", 5_000, 25_000),
    ("OK", 25_000, 75_000),
    ("GOOD", 75_000, 250_000),
    ("VIRAL", 250_000, 10 ** 15),
]
BUCKET_NAMES = [b[0] for b in BUCKETS]

# Views only settle after ~30 days; younger videos would be scored as failures
# for the crime of being recent.
MATURITY_DAYS = 30.0

_EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
_WORD = re.compile(r"[A-Za-z']+")


def bucket_of(views: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= views < hi:
            return name
    return BUCKET_NAMES[-1]


def bucket_index(name: str) -> int:
    return BUCKET_NAMES.index(name)


@dataclass
class Sample:
    video: Video
    features: dict[str, float]
    log_views: float
    bucket: str


# Features knowable before publishing. Nothing here is an outcome.
FEATURES = [
    "chan_prior_log_median", "chan_prior_n", "chan_prior_spread",
    "duration_s", "log_duration", "is_micro",
    "title_len", "title_words", "has_number", "caps_words", "emoji_count",
    "hashtag_count", "is_question", "curiosity", "stakes", "has_at_mention",
    "tag_count", "desc_len", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
CHANNEL_PRIOR_FEATURES = {"chan_prior_log_median", "chan_prior_n", "chan_prior_spread"}


def _title_feats(v: Video) -> dict[str, float]:
    t = v.title
    words = _WORD.findall(t)
    return {
        "title_len": float(len(t)),
        "title_words": float(len(words)),
        "has_number": 1.0 if any(c.isdigit() for c in t) else 0.0,
        "caps_words": float(sum(1 for w in words if len(w) > 2 and w.isupper())),
        "emoji_count": float(len(_EMOJI.findall(t))),
        "hashtag_count": float(t.count("#")),
        "is_question": 1.0 if t.strip().endswith("?") else 0.0,
        "curiosity": float(sum(1 for w in words if w.lower() in CURIOSITY)),
        "stakes": float(sum(1 for w in words if w.lower() in STAKES)),
        "has_at_mention": 1.0 if "@" in t else 0.0,
    }


def build_samples(videos: Sequence[Video], now: Optional[datetime] = None,
                  with_signals: bool = False) -> list[Sample]:
    """Turn videos into training rows, with a strictly-causal channel prior.

    The channel prior is the median of that channel's videos published at least
    MATURITY_DAYS *before* this one. Two constraints, both load-bearing:
    earlier, so we never use the future; and matured, because a video published
    a day earlier had not settled at the moment this one went out, so its count
    was not knowable either.
    """
    now = now or datetime.now(timezone.utc)
    mature = [v for v in videos if v.age_days(now) >= MATURITY_DAYS and v.views > 0]
    index = CorpusIndex.build(mature) if with_signals else None

    by_channel: dict[str, list[Video]] = defaultdict(list)
    for v in mature:
        by_channel[v.channel_id].append(v)

    samples: list[Sample] = []
    for vids in by_channel.values():
        vids.sort(key=lambda v: v.published_at)
        for i, v in enumerate(vids):
            cutoff = v.published_at - timedelta(days=MATURITY_DAYS)
            prior = [math.log10(p.views) for p in vids[:i] if p.published_at <= cutoff]
            if len(prior) < 3:
                # Without a usable history there is no honest channel prior, and
                # inventing one (global mean, say) leaks the corpus average into
                # a per-channel feature.
                continue
            feats = {
                "chan_prior_log_median": median(prior),
                "chan_prior_n": float(len(prior)),
                "chan_prior_spread": quantile(prior, 0.75) - quantile(prior, 0.25),
                "duration_s": float(v.duration_s),
                "log_duration": math.log10(max(v.duration_s, 1)),
                "is_micro": 1.0 if v.duration_s <= 15 else 0.0,
                "tag_count": float(len(v.tags)),
                "desc_len": float(len(v.description)),
                "hour_sin": math.sin(2 * math.pi * v.published_at.hour / 24),
                "hour_cos": math.cos(2 * math.pi * v.published_at.hour / 24),
                "dow_sin": math.sin(2 * math.pi * v.published_at.weekday() / 7),
                "dow_cos": math.cos(2 * math.pi * v.published_at.weekday() / 7),
            }
            feats.update(_title_feats(v))
            if with_signals:
                # Prior videos only, and prior *matured* ones for anything that
                # touches a view count - same causality rule as the channel
                # prior itself.
                earlier = [p for p in vids[:i] if p.published_at <= cutoff]
                feats.update(compute_signals(v, earlier, prior, index))
            samples.append(Sample(video=v, features=feats,
                                  log_views=math.log10(v.views), bucket=bucket_of(v.views)))
    samples.sort(key=lambda s: s.video.published_at)
    return samples


_TOKEN = re.compile(r"[a-z']{3,}")


def title_tokens(title: str) -> set[str]:
    return set(_TOKEN.findall(title.lower()))


def build_vocabulary(train: Sequence[Sample], size: int = 250, min_uses: int = 15) -> list[str]:
    """Topic vocabulary, built from the TRAINING SLICE ONLY.

    This restriction is the whole reason this is a function rather than a
    comprehension over the corpus. Vocabulary drawn from all the data encodes
    which words exist in the test period - a subtle leak that inflates the
    score without improving a single real prediction.

    Topic matters more than structure: measured on a real corpus, adding these
    tokens moved out-of-sample R-squared from -0.23 to -0.07 and rank
    correlation from 0.16 to 0.22, where every structural feature (title
    length, digit counts, emoji) together had achieved nothing. What a video is
    *about* carries signal that how its title is punctuated does not.
    """
    counts: dict[str, int] = defaultdict(int)
    for s in train:
        for w in title_tokens(s.video.title):
            counts[w] += 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return [w for w, n in ranked if n >= min_uses][:size]


def temporal_split(samples: Sequence[Sample], test_frac: float = 0.25):
    """Past trains, future tests. Never random."""
    cut = int(len(samples) * (1 - test_frac))
    return list(samples[:cut]), list(samples[cut:])


@dataclass
class Report:
    n_train: int
    n_test: int
    features_used: list[str]
    accuracy: float
    majority_accuracy: float
    prior_only_accuracy: float
    within_one_accuracy: float
    mae_log: float
    baseline_mae_log: float
    spearman: float
    confusion: dict[str, dict[str, int]]
    per_class: dict[str, tuple[int, float, float]]  # n, precision, recall
    train_span: tuple[str, str] = ("", "")
    test_span: tuple[str, str] = ("", "")

    @property
    def lift(self) -> float:
        return self.accuracy - self.majority_accuracy

    @property
    def verdict(self) -> str:
        if self.accuracy <= self.majority_accuracy + 0.01:
            return "NO SIGNAL - does not beat guessing the commonest bucket"
        if self.accuracy <= self.prior_only_accuracy + 0.01:
            return "NO SIGNAL BEYOND CHANNEL HISTORY - packaging adds nothing here"
        return "REAL SIGNAL - beats both baselines"


def _row(s: Sample, feats: Sequence[str], vocab: Sequence[str]) -> list[float]:
    base = [s.features[f] for f in feats]
    if not vocab:
        return base
    tk = title_tokens(s.video.title)
    return base + [1.0 if w in tk else 0.0 for w in vocab]


def _fit(train: Sequence[Sample], feats: Sequence[str], alpha: float,
         vocab: Sequence[str] = ()):
    X = [_row(s, feats, vocab) for s in train]
    y = [s.log_views for s in train]
    return ridge_fit(X, y, alpha=alpha) or None


def _predict(coefs, s: Sample, feats: Sequence[str], vocab: Sequence[str] = ()) -> float:
    *slopes, intercept = coefs
    return intercept + sum(c * v for c, v in zip(slopes, _row(s, feats, vocab)))


def active_features(samples: Sequence[Sample]) -> list[str]:
    """FEATURES plus any signal features the samples actually carry.

    Keyed off the data rather than a flag so a caller cannot ask for a model
    over features its samples were never built with - that mismatch produces a
    KeyError deep in the fit, where it is far harder to read than here.
    """
    if samples and all(f in samples[0].features for f in SIGNAL_FEATURES):
        return FEATURES + SIGNAL_FEATURES
    return list(FEATURES)


def run_backtest(samples: Sequence[Sample], test_frac: float = 0.25,
                 alpha: float = 1.0, use_channel_prior: bool = True,
                 use_topics: bool = True, vocab_size: int = 250) -> Optional[Report]:
    feats = [f for f in active_features(samples)
             if use_channel_prior or f not in CHANNEL_PRIOR_FEATURES]
    train, test = temporal_split(samples, test_frac)
    if len(train) < 100 or len(test) < 30:
        return None

    vocab = build_vocabulary(train, size=vocab_size) if use_topics else []
    coefs = _fit(train, feats, alpha, vocab)
    if coefs is None:
        return None

    preds_log = [_predict(coefs, s, feats, vocab) for s in test]
    preds = [bucket_of(int(10 ** p)) for p in preds_log]
    truth = [s.bucket for s in test]

    # Baseline 1: always guess the commonest bucket in the training period.
    counts = defaultdict(int)
    for s in train:
        counts[s.bucket] += 1
    majority = max(counts, key=counts.get)
    maj_acc = sum(1 for t in truth if t == majority) / len(truth)

    # Baseline 2, the hard one: predict the channel's own prior median. If the
    # model cannot beat this, it has learned nothing except who posted it.
    prior_preds = [bucket_of(int(10 ** s.features["chan_prior_log_median"])) for s in test]
    prior_acc = sum(1 for p, t in zip(prior_preds, truth) if p == t) / len(truth)
    prior_mae = sum(abs(s.features["chan_prior_log_median"] - s.log_views)
                    for s in test) / len(test)

    acc = sum(1 for p, t in zip(preds, truth) if p == t) / len(truth)
    within1 = sum(1 for p, t in zip(preds, truth)
                  if abs(bucket_index(p) - bucket_index(t)) <= 1) / len(truth)
    mae = sum(abs(p - s.log_views) for p, s in zip(preds_log, test)) / len(test)

    confusion = {a: {b: 0 for b in BUCKET_NAMES} for a in BUCKET_NAMES}
    for p, t in zip(preds, truth):
        confusion[t][p] += 1

    per_class = {}
    for b in BUCKET_NAMES:
        n = sum(1 for t in truth if t == b)
        tp = confusion[b][b]
        predicted = sum(1 for p in preds if p == b)
        per_class[b] = (n, tp / predicted if predicted else 0.0, tp / n if n else 0.0)

    return Report(
        n_train=len(train), n_test=len(test),
        features_used=list(feats) + [f"topic:{w}" for w in vocab],
        accuracy=acc, majority_accuracy=maj_acc, prior_only_accuracy=prior_acc,
        within_one_accuracy=within1, mae_log=mae, baseline_mae_log=prior_mae,
        spearman=spearman(preds_log, [s.log_views for s in test]),
        confusion=confusion, per_class=per_class,
        train_span=(train[0].video.published_at.date().isoformat(),
                    train[-1].video.published_at.date().isoformat()),
        test_span=(test[0].video.published_at.date().isoformat(),
                   test[-1].video.published_at.date().isoformat()),
    )


def render(r: Report, title: str = "") -> str:
    out = ["", f"BACKTEST{' - ' + title if title else ''}",
           "=" * (8 + (len(title) + 3 if title else 0)), ""]
    out.append(f"  train {r.n_train:,} videos ({r.train_span[0]} -> {r.train_span[1]})")
    out.append(f"  test  {r.n_test:,} videos ({r.test_span[0]} -> {r.test_span[1]})   [strictly later]")
    out.append(f"  features: {len(r.features_used)} pre-publication only")
    out.append("")
    out.append(f"  exact bucket        {r.accuracy:6.1%}")
    out.append(f"  within one bucket   {r.within_one_accuracy:6.1%}")
    out.append(f"  MAE (log10 views)   {r.mae_log:6.2f}   "
               f"(x{10 ** r.mae_log:.1f} typical error)")
    out.append(f"  rank correlation    {r.spearman:6.3f}")
    out.append("")
    out.append("  BASELINES - the numbers that decide whether the above means anything")
    out.append(f"    always guess '{max(r.per_class, key=lambda b: r.per_class[b][0])}'"
               f"{'':<8} {r.majority_accuracy:6.1%}")
    out.append(f"    channel history only        {r.prior_only_accuracy:6.1%}"
               f"   (MAE {r.baseline_mae_log:.2f})")
    out.append(f"\n  VERDICT: {r.verdict}")

    out.append("\n  confusion (rows = actual, cols = predicted)")
    out.append("      " + "".join(f"{b:>8}" for b in BUCKET_NAMES))
    for a in BUCKET_NAMES:
        row = "".join(f"{r.confusion[a][b]:>8}" for b in BUCKET_NAMES)
        out.append(f"  {a:<4}{row}")

    out.append("\n  per class        n   precision   recall")
    for b in BUCKET_NAMES:
        n, p, rec = r.per_class[b]
        out.append(f"    {b:<6} {n:>6}   {p:>8.1%} {rec:>8.1%}")
    return "\n".join(out)
