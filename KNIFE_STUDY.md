# The Falling-Knife Study

*Catch the falling knife, turn it into a fork, shove it back up — when is
that actually possible?* 8,346 distinct crash events over the 5-year test
window (500 names; crash = −15%/5d, −20%/10d, or 8-of-10 down days ≥ −10%;
10-day cooldown). Every metric and the shape of the chart on the way down,
vs what happened next. Files: `results/knife_events.csv` (full table),
`knife_cohorts.csv`, `knife_forks.csv`, charts `knife_cohorts.png`,
`knife_paths.png`.

## The adage is true by default

Catching **every** knife: −7.8 bp market-adjusted over the next 5 days,
−21 bp after costs (t = −2.7), bounce probability 48.6%, and a **32%
chance the knife cuts another −10% within 10 days** (42% for fast
knives). Indiscriminate bottom-fishing is a losing strategy, full stop.

## WHY it was falling is everything

| Cohort | 5d after (bp) | 10d (bp) | 2nd-knife risk |
|---|---|---|---|
| News-driven crash | −24 | **−55** | 30% |
| News + climax volume (**the trap**) | **−52** | **−106** | 29% |
| No-news crash | −5 | **+5** | 32% |
| Reversal bar (close top-40% of range, RVOL≥2) | **+20** | −12 | 30% |
| Gap-down exhaustion (opened ≤−3%, recovered half) | **+160** | **+372** | 27% — n = 22 only |
| **FORK RULE: no-news AND (reversal bar OR climax)** | **+50** | **+136** | 39% |

The same climax volume that lore calls "capitulation" is a **trap when
news drives the crash** — heavy volume on real bad news means the
information is still repricing (negative drift continues; the average
news-crash chart keeps sliding for 20+ days). Capitulation signatures only
mean something **without a catalyst**. Shape matters too, with the same
sign everywhere: parabolic finishes and accelerating (convex-down) paths
keep falling harder than grinds.

## The fork exists — but it is rare and unproven

The rule fork (no-news + reversal-bar-or-climax) earned **+54 bp net per
5-day trade, 54% hit rate — on 50 trades in 5 years (≈ 1/month), t = 0.4**.
Right sign, economically sensible, *statistically unproven at this
frequency*: this is a garnish, not a meal. Honest negative results beside
it: a walk-forward GBM on all 16 event features has IC ≈ −0.02 (its
threshold portfolio LOSES 51 bp/trade), and the **reference-class analog
forecaster** — your "look at the IPOs before this IPO" method, implemented
as strictly backward-looking 25-nearest-neighbor cohorts in feature
space — has IC ≈ 0.00. Beyond the news/no-news split and the two rare
reversal signatures, crash outcomes are not predictable from crash
anatomy.¹

## The IPO analogy, honestly

The kNN cohort machinery in `engine/experiments/knife.py` IS the
"analyst-insider by analogy" method: judge an event with no history by the
distribution of outcomes of its nearest past analogs. It runs and is
calibrated here on crashes; pointing it at actual IPO cohorts (first-day
pop/fade vs offer-size, sector, bookbuild, lockup) requires listing data
no free blocked-sandbox source provides — queued for the armed real-data
run. Expectation to hold it to: the IPO literature (Ritter's long-run
underperformance; first-day pop persistence) says cohort features explain
the *average* pop, not which individual IPO doubles — same lesson as the
knives.

¹ Walk-forward footnote: monthly retraining uses all events dated before
the test month; events in the last week of a training month have 5-day
outcomes bleeding slightly past the boundary. This can only OVERSTATE
model skill — and measured skill was ≈ zero — so the negative conclusion
is robust.
