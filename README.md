# New-ENGINE — algorithmic chart-pattern edge catalog

Catalogs every chart-based edge and tests it **algorithmically** — kernel-
regression classical patterns (Lo–Mamaysky–Wang), candlesticks, support/
resistance (swings, round numbers, anchored VWAP, MAs), 52-week-high
anchoring, breakout systems (ORB, Donchian, VCP), a full gap taxonomy split
by news catalyst and volume, volume–price signals, trend systems, a
Jiang–Kelly–Xiu-style CNN on chart images, quality/moat fundamentals, six
engine-original indicators, and a walk-forward fusion meta-model over all of
them. Deliverables: per-edge event studies with Newey–West t-stats, decay
tables, the news × charts interaction study, equity/drawdown/bootstrap/
attribution/sample-trade charts, and `CATALOG.md` with formal definitions,
original evidence, and honest verdicts.

## Data honesty (read this first)

This build ran in a sandbox whose egress policy **blocks all market-data
hosts** (stooq, Yahoo, GDELT — verified 403). Therefore:

* Simulator **dynamics** are calibrated on the **real S&P 500 daily series
  1999–2018** (bundled with the `arch` package): GARCH(1,1)-t, overnight
  variance share, volume–|return| elasticity.
* **Effect sizes** are embedded at published magnitudes, with deliberate
  placebo families (candlesticks, round numbers) — see `CATALOG.md`.
* All empirical tables/charts in `results/` are therefore **machinery
  validation on a literature-calibrated synthetic market**, not real-market
  P&L. Real-data adapters ship in `engine/data/loaders.py`; from any
  network-enabled machine:

```bash
pip install -r requirements.txt
echo "AAPL MSFT ..." > tickers.txt
python -m engine.run_all --source real
```

## Run

```bash
pip install -r requirements.txt          # + `pip install torch` for the CNN
python -m engine.run_all                 # full synthetic run -> results/
python -m engine.run_all --fast --no-ml  # 60-stock smoke test
python -m pytest tests/ -q               # incl. anti-lookahead tests
```

## Layout

```
engine/data        calibration (real SP500), simulator w/ ground truth, real loaders
engine/patterns    11 detector families, one shared point-in-time signal contract
engine/backtest    forward-return engine, cost model, NW stats, decay, bootstrap
engine/analysis    news x charts interaction tables
engine/fusion      walk-forward GBM meta-model + linear baseline
engine/report      chart deliverables (dataviz-validated palette)
results/           per_signal_stats.csv, decay_by_year.csv, interaction_*.csv,
                   ground_truth_recovery.csv, summary.json, *.png, sample_trades/
```
