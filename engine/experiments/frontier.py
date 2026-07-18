"""The 'keep going until 1%/day' study: how far can HONEST fusion be pushed?

Sweeps the levers that actually move average daily return:
  * selectivity: top-K per side (10 -> 5 -> 3 -> 1)
  * conviction threshold: trade only days/names with |score| above the
    trailing q-th percentile of scores (skip flat days entirely)
  * holding period: 1-5 days (amortizes the round trip over h days)
  * universe restriction: catalyst (news) names only -- the interaction cells
  * cost scenarios: 15 bp round trip (base), 8 bp (aggressive institutional),
    2 bp (near-frictionless bound; unrealistic, shown as the ceiling)

Uses cached signals (results/signals.pkl) + the same walk-forward GBM scores
as run_all; every variant stays strictly walk-forward.  Output:
results/frontier.csv + a bar/line chart, and the summary line the report
quotes.  Run AFTER run_all.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from ..backtest.engine import Forward
from ..backtest.stats import newey_west_t
from ..config import RunConfig
from ..fusion.meta import build_features
from ..patterns.base import Wide


def scores_walk_forward(X, fwd, seed=3, min_train_days=320):
    """Same monthly-retrain GBM as fusion.meta, returning scores only."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    pos_d = {d: i for i, d in enumerate(fwd.dates)}
    di = X.index.get_level_values("date").map(pos_d).values
    tj_map = {t: j for j, t in enumerate(fwd.tickers)}
    tj = X.index.get_level_values("ticker").map(tj_map).values
    y = fwd.fwd_no[1][di, tj] - fwd.mkt_h[1][di]
    Xv = X.values.astype(np.float64)
    Xv[~np.isfinite(Xv)] = 0.0
    ok = np.isfinite(y)
    months = pd.PeriodIndex(X.index.get_level_values("date"), freq="M")
    active = X["n_active"].values > 0
    score = np.full(len(X), np.nan)
    for m in pd.period_range(months.min(), months.max(), freq="M"):
        tr = ok & (months < m) & active
        te = (months == m) & active
        if tr.sum() < min_train_days * 5 or te.sum() == 0:
            continue
        idx = np.where(tr)[0][-200_000:]
        mod = HistGradientBoostingRegressor(
            max_iter=150, max_depth=4, learning_rate=0.06,
            l2_regularization=1.0, random_state=seed)
        mod.fit(Xv[idx], y[idx])
        score[te] = mod.predict(Xv[te])
    return pd.Series(score, index=X.index, name="score").dropna()


def portfolio(scores: pd.Series, fwd: Forward, top_k: int, hold: int,
              rt_bps: float, conviction_q: float = 0.0,
              news_only: pd.Series | None = None) -> pd.Series:
    """Daily net returns of a top-K/bottom-K walk-forward portfolio.

    hold > 1: capital splits into `hold` overlapping tranches, each tranche
    rebalances every `hold` days; round trip charged once per tranche cycle.
    conviction_q: within each day, require |score| >= that day's q-quantile
    of |score| (0 = no filter); if fewer than 2 names pass, stay flat.
    news_only: boolean per (date,ticker); restrict candidates to it.
    """
    dates = fwd.dates
    pos_d = {d: i for i, d in enumerate(dates)}
    pos_t = {t: j for j, t in enumerate(fwd.tickers)}
    T = len(dates)
    rt = rt_bps / 1e4
    num = np.zeros(T)
    sc = scores if news_only is None else scores[news_only.reindex(
        scores.index, fill_value=False)]
    for d, g in sc.groupby(level="date"):
        g = g.droplevel("date")
        if conviction_q > 0:
            thr = g.abs().quantile(conviction_q)
            g = g[g.abs() >= thr]
        if len(g) < 2:
            continue
        g = g.sort_values()
        k = min(top_k, len(g) // 2)
        longs = [pos_t[t] for t in g.index[-k:]]
        shorts = [pos_t[t] for t in g.index[:k]]
        t0 = pos_d[d]
        # tranche: enter next open, hold `hold` days, mark daily
        for kk in range(1, hold + 1):
            day = t0 + kk
            if day >= T:
                break
            if kk == 1:
                rl = np.nanmean(fwd.oc[day, longs])
                rs = np.nanmean(fwd.oc[day, shorts])
            else:
                rl = np.nanmean(fwd.cc[day, longs])
                rs = np.nanmean(fwd.cc[day, shorts])
            r = (rl - rs) / 2.0
            if kk == 1:
                r -= rt
            if np.isfinite(r):
                num[day] += r / hold      # 1/hold of capital per tranche
    return pd.Series(num, index=dates)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args(argv)
    cfg = RunConfig()

    print("[frontier] rebuilding panel + features (cached signals)...")
    from ..data.calibration import calibrate
    from ..data.synthetic import simulate
    calib = calibrate(verbose=False)
    data = simulate(cfg.sim, calib)
    panel = data["panel"]
    w = Wide(panel)
    fwd = Forward(w.open, w.close, cfg.horizons)
    signals = pd.read_pickle(f"{args.outdir}/signals.pkl")
    X = build_features(signals, w)
    print("[frontier] walk-forward scores...")
    sc = scores_walk_forward(X, fwd)
    test_start = data["dates"][-1] - pd.DateOffset(years=cfg.test_years)
    sc = sc[sc.index.get_level_values("date") >= test_start]

    news = (panel["catalyst"].rolling(3, min_periods=1).max() > 0)

    rows = []
    grid_k = [10, 5, 3, 1]
    grid_h = [1, 2, 5]
    grid_q = [0.0, 0.8, 0.95]
    grid_c = [15.0, 8.0, 2.0]
    total = len(grid_k) * len(grid_h) * len(grid_q) + 4
    done = 0
    for k in grid_k:
        for h in grid_h:
            for q in grid_q:
                net = portfolio(sc, fwd, k, h, 15.0, q)
                net = net[net.index >= test_start]
                rows.append({"variant": "all", "top_k": k, "hold": h,
                             "conviction_q": q, "rt_bps": 15.0,
                             "net_bps": 1e4 * net.mean(),
                             "sharpe": float(np.sqrt(252) * net.mean()
                                             / (net.std() + 1e-12)),
                             "nw_t": newey_west_t(net.values),
                             "traded_days": int((net != 0).sum())})
                done += 1
        print(f"[frontier] {done}/{total}")
    # news-only and cost scenarios on the best liquid config
    for rt in grid_c:
        net = portfolio(sc, fwd, 3, 2, rt, 0.8)
        net = net[net.index >= test_start]
        rows.append({"variant": "all", "top_k": 3, "hold": 2,
                     "conviction_q": 0.8, "rt_bps": rt,
                     "net_bps": 1e4 * net.mean(),
                     "sharpe": float(np.sqrt(252) * net.mean()
                                     / (net.std() + 1e-12)),
                     "nw_t": newey_west_t(net.values),
                     "traded_days": int((net != 0).sum())})
    net = portfolio(sc, fwd, 3, 2, 15.0, 0.0, news_only=news)
    net = net[net.index >= test_start]
    rows.append({"variant": "news_only", "top_k": 3, "hold": 2,
                 "conviction_q": 0.0, "rt_bps": 15.0,
                 "net_bps": 1e4 * net.mean(),
                 "sharpe": float(np.sqrt(252) * net.mean()
                                 / (net.std() + 1e-12)),
                 "nw_t": newey_west_t(net.values),
                 "traded_days": int((net != 0).sum())})

    df = pd.DataFrame(rows).sort_values("net_bps", ascending=False)
    df.to_csv(f"{args.outdir}/frontier.csv", index=False)
    print(df.head(12).to_string(index=False))
    best = df.iloc[0]
    print(f"\n[frontier] BEST HONEST CONFIG: {best['net_bps']:.1f} bp/day net "
          f"(target 100) -- gap to target: {100 / max(best['net_bps'], 0.01):.0f}x")
    return df


if __name__ == "__main__":
    main()
