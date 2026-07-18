# The Random Day Trader Study

*"What about being a day trader, hopping in random stocks and selling?"*
Learn first, then backtest. Files: `results/daytrader_summary.csv`,
`daytrader_required_skill.csv`, `daytrader_lifetimes.png`,
`daytrader_required_skill.png`; code `engine/experiments/daytrader.py`.

## Learn: what the evidence already says

* **Barber, Lee, Liu & Odean (Taiwan, complete market data)**: ~80% of day
  traders lose money; fewer than 1% are predictably profitable.
* **Chague, De-Losso & Giovannetti (Brazil, all futures day traders
  2013-2015)**: of those who persisted 300+ sessions, **97% lost**; ~1.1%
  earned more than minimum wage.
* **Barber & Odean (2000)**: household returns fall monotonically with
  turnover — "trading is hazardous to your wealth."
* **Lou, Polk & Skouras (2019)**: the equity premium accrues almost
  entirely **overnight**. The intraday segment — the only segment a day
  trader holds — has ≈ zero drift. A day trader is structurally long the
  worthless half of the day.

## Backtest: 2,000 simulated trader-years (3 random stocks/day, retail 5 bp RT)

| Variant | Mean/yr | P(profitable year) | 5th pctile |
|---|---|---|---|
| Sim upper bound (sim puts drift intraday — disclosed artifact) | +5.6% | 63% | −64% |
| **Overnight-adjusted (realistic)** | **−2.7%** | **53%** | **−70%** |
| Hot-mover picks (attention bias) | +5.3%* | 62% | −66% |
| Random entry time ("hop in whenever") | **−8.2%** | **34%** | −44% |

The realistic case: negative expectancy, a coin-flip year, and a 1-in-20
chance of losing ~70%. "Hopping in whenever" is strictly worse — random
timing forfeits half of every move but pays full costs. And this is the
FRIENDLY version: no leverage, no overtrading on tilt, no borrow fees, a
symmetric simulator, and only 5 bp of costs. The Brazilian 97% figure is
what happens when real frictions and real behavior are added.

\* The hot-mover panel is the one place our simulator is **generous**: it
embeds news-drift continuation but no intraday overreaction, so chasing
movers accidentally rides PEAD here. In real data, attention-driven buying
has *negative* alpha (Barber-Odean attention effect) — treat that +5.3% as
an upper bound with the wrong sign vs reality.

## What +1%/day would actually require

Per-trade edge needed = (100 bp + k×costs)/k on full-capital round trips:

| Cycles/day | Edge/trade (retail) | Hit rate needed |
|---|---|---|
| 1 | 105 bp | **73.7%** |
| 3 | 38 bp | **58.6%** |
| 10 | 15 bp | 53.4% |
| 20 | 10 bp | **52.3%** |

Reference line: Renaissance's Medallion — the best sustained edge ever
documented — reportedly wins **50.75%** of the time. A day trader flipping
3 positions a day needs to be *right 58.6% of the time, every day, for
years* to average 1%: roughly 15× Medallion's edge per bet, sustained by a
human without Medallion's infrastructure. At 20 cycles/day the required
hit rate looks "small" (52.3%) — but 20 full-capital round trips daily IS
high-frequency trading, which is precisely the industry that exists
because no human can do this manually.

## Verdict

Random-stock day trading is a **negative-sum lottery ticket with daily
fees**: zero captured drift (wrong half of the day), costs × turnover as a
certain loss, and a P&L distribution wide enough to keep the winners'
stories circulating. The only versions that work are the ones from the
main catalog — holding through the drift the day trader gives away
(news-gap continuation, PEAD, overnight premium) — or industrializing the
frequency until the law of large numbers does the work, which is a firm,
not a lifestyle.
