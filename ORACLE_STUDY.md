# The Look-Ahead Winner Study

*"Look ahead at the past year, take the top-3 performers each day, put them
in a table, and make a strategy that would have said those stocks."*

We did exactly that — with the look-ahead firewalled to where it belongs.
Universe: the 150 most liquid names ("S&P 150" proxy; synthetic
literature-calibrated market — real-data run is armed for when the network
opens). Window: the final 252 trading days.

## 1. The table

`results/oracle_top3_table.csv` — 756 rows (252 days × 3 winners): date,
rank, ticker, return, day-of context (news flag, RVOL, gap) and the full
**day-before** observable profile of each winner. Average top-3 winner made
**+8.2% in its day**; 16.5% had a news catalyst that day (8.6× the 1.9%
base rate); 22% gapped >3% that morning; day-of RVOL ran 3–8×.

## 2. What the winners looked like the day before: like everyone else

`results/oracle_winner_profile.csv` and the information-timing chart. On
every ex-ante observable — RVOL, gap, momentum (5d/21d/12m), 52-week
position, compression, quality — tomorrow's top-3 sit at the **43rd–64th
percentile** of the universe: statistically almost indistinguishable. The
one mild tilt is 20-day volatility (64th percentile): volatile names are
mechanically over-represented in *both* tails. The information that makes
a top-3 winner — the catalyst, the volume surge, the gap — **arrives the
same day**, not the day before.

## 3. The "strategy that would have said those stocks"

| Strategy | Avg daily return | What it proves |
|---|---|---|
| **Oracle** — buy each day's actual top-3 (look-ahead) | **+8.23%/day** (≈ ×10⁸ in a year) | the ceiling if tomorrow's table were known today |
| **Honest predictor** — walk-forward GBM trained to predict top-3 membership from day-before observables, top-3 probabilities bought at next open, net of costs | **−0.50%/day** (t = −1.9) | hindsight targets do not invert |
| Random 3 names/day | −0.21%/day | the honest predictor is *worse* than random after costs |

The subtle and important detail: the classifier is **not useless at its
own job** — its picks land in the actual top-3 at a **4.9% rate vs 2.0%
random (2.4×)**. It genuinely finds the candidates. It still loses money,
because the features that predict "will be in the top-3" (high volatility,
news-proneness) equally predict "will be in the bottom-3": the tail is
symmetric ex ante, the long side pays the costs, and the day's sign is
decided by information that does not exist the evening before. Capture
ratio of the oracle: **−6%**.

## The lesson

A strategy "that would have said those stocks" is a strategy that knows
day-of news and day-of volume the night before. That is not a chart
pattern; it is tomorrow's newspaper. This is the same conclusion the main
catalog reaches from the other direction: the tradable residue of the
winner phenomenon is the *news × volume × gap interaction* (catch the
drift AFTER the catalyst is public — worth basis points, not percent), plus
slow anomalies. Charts: `results/oracle_curves.png`,
`results/oracle_information_timing.png`; picks log:
`results/oracle_honest_picks.csv`.
