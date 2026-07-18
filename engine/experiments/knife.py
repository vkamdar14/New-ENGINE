"""The falling-knife study: when can a crash be caught?

"Catch the falling knife, turn it into a fork" -- algorithmically:

1. EVENTS.  Every fresh crash in the 5y test window (500 names):
     crash5 : <= -15% over 5 days      (fast knife)
     crash10: <= -20% over 10 days     (extended knife)
     grind  : >= 8 of last 10 days down and <= -10% total (slow bleed)
   10-day per-ticker cooldown so events are distinct.

2. EVERY METRIC + HOW THE CHART LOOKED at the event day t:
   news-driven vs no-news; climax volume (max RVOL of last 2 days); crash
   sigma (depth / own vol); acceleration (share of the fall in the last 2
   days -- parabolic finish vs front-loaded); path convexity (quadratic
   coefficient of the 10d log-price path); reversal bar (close in top 40%
   of range on RVOL>=2); gap-down exhaustion (opened <=-3% but recovered
   half intraday); depth below MA50 in ATRs; 52wk range position; quality;
   prior 6m trend.

3. OUTCOMES.  Forward 1/5/10/20-day market-adjusted returns from next
   open, bounce probability, upside/downside percentiles (P90/P10), and
   second-knife risk P(another -10% low within 10 days).

4. THE FORK.  Two honest, walk-forward catchers, net of costs:
     a. rule fork: no-news AND (reversal bar OR climax volume>=3)
     b. model fork: monthly-retrained GBM on event features, buy events
        with predicted 5d abnormal > +100 bp
   vs the baseline "catch every knife".

5. THE IPO ANALOGY (reference-class forecaster).  Each event is judged by
   its 25 nearest PAST events in standardized feature space (kNN, strictly
   backward-looking): forecast = cohort mean outcome.  This is "look at
   the IPOs before this IPO" as machinery; on real data the same class
   runs on actual IPO cohorts (needs listing data -- flagged for the
   armed real-data run).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..backtest.engine import Forward
from ..backtest.stats import newey_west_t
from ..config import RunConfig
from ..patterns.base import Wide

FEATS = ["depth5", "sigma_crash", "accel", "convexity", "news_crash",
         "climax_rvol", "reversal_bar", "gap_exhaust", "below_ma50_atr",
         "rangepos52", "quality", "mom126", "vol20", "kind_fast",
         "kind_ext", "kind_grind"]


def extract_events(w: Wide, test_start) -> pd.DataFrame:
    c, h, l, o, v = w.close, w.high, w.low, w.open, w.volume
    ret5 = c.pct_change(5)
    ret10 = c.pct_change(10)
    down = (w.ret_cc < 0).rolling(10).sum()
    crash_fast = ret5 <= -0.15
    crash_ext = ret10 <= -0.20
    crash_grind = (down >= 8) & (ret10 <= -0.10)
    any_crash = crash_fast | crash_ext | crash_grind

    dates = c.index
    ma50 = c.rolling(50, min_periods=50).mean()
    rangepos = (c - w.lo52) / (w.hi52 - w.lo52)
    mom126 = c.pct_change(126)
    news5 = w.catalyst.rolling(5, min_periods=1).max()
    rvol2 = w.rvol.rolling(2, min_periods=1).max()
    clv = (c - l) / (h - l).replace(0, np.nan)
    gap = w.gap

    rows = []
    arr = any_crash.fillna(False).values
    tickers = list(c.columns)
    logc = np.log(c.values)
    tt = np.arange(10)
    A = np.vstack([tt**2, tt, np.ones_like(tt)]).T
    pinv = np.linalg.pinv(A)
    for j, tkr in enumerate(tickers):
        idx = np.where(arr[:, j])[0]
        last = -99
        for t in idx:
            if t - last < 10:
                continue
            last = t
            d = dates[t]
            if d < test_start or t < 260 or t + 21 >= len(dates):
                continue
            r5 = ret5.iat[t, j]
            r2 = c.iat[t, j] / c.iat[t - 2, j] - 1
            path = logc[t - 9:t + 1, j]
            coef = pinv @ (path - path[0])
            rows.append({
                "date": d, "ticker": tkr,
                "kind_fast": int(bool(crash_fast.iat[t, j])),
                "kind_ext": int(bool(crash_ext.iat[t, j])),
                "kind_grind": int(bool(crash_grind.iat[t, j])),
                "depth5": float(r5),
                "sigma_crash": float(r5 / (w.vol20.iat[t, j] * np.sqrt(5)
                                           + 1e-9)),
                "accel": float(r2 / min(r5, -1e-6)),
                "convexity": float(coef[0]),
                "news_crash": int(news5.iat[t, j] > 0),
                "climax_rvol": float(rvol2.iat[t, j]),
                "reversal_bar": int((clv.iat[t, j] >= 0.6)
                                    and (w.rvol.iat[t, j] >= 2)),
                "gap_exhaust": int((gap.iat[t, j] <= -0.03)
                                   and (c.iat[t, j] >= o.iat[t, j]
                                        - 0.5 * (o.iat[t, j]
                                                 - c.iat[t - 1, j]))),
                "below_ma50_atr": float((ma50.iat[t, j] - c.iat[t, j])
                                        / (w.atr.iat[t, j] + 1e-9)),
                "rangepos52": float(rangepos.iat[t, j]),
                "quality": float(w.quality.iat[t, j]),
                "mom126": float(mom126.iat[t, j]),
                "vol20": float(w.vol20.iat[t, j]),
            })
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args(argv)
    cfg = RunConfig()
    rt = cfg.costs.round_trip_bps / 1e4

    print("[knife] building panel...")
    from ..data.calibration import calibrate
    from ..data.synthetic import simulate
    data = simulate(cfg.sim, calibrate(verbose=False))
    panel = data["panel"]
    w = Wide(panel)
    fwd = Forward(w.open, w.close, (1, 5, 10, 20))
    test_start = data["dates"][-1] - pd.DateOffset(years=cfg.test_years)

    ev = extract_events(w, test_start)
    print(f"[knife] {len(ev)} distinct crash events "
          f"({ev.news_crash.mean():.0%} news-driven)")

    # ---- outcomes ----
    pos_d = {d: i for i, d in enumerate(fwd.dates)}
    pos_t = {t: j for j, t in enumerate(fwd.tickers)}
    ti = ev["date"].map(pos_d).values
    tj = ev["ticker"].map(pos_t).values
    for h in (1, 5, 10, 20):
        ev[f"fwd{h}"] = fwd.fwd_no[h][ti, tj] - fwd.mkt_h_no[h][ti]
    lo = w.low.values
    cl = w.close.values
    second = []
    for t, j in zip(ti, tj):
        fut_min = lo[t + 1:t + 11, j].min()
        second.append(fut_min / cl[t, j] - 1 <= -0.10)
    ev["second_knife"] = np.array(second, dtype=int)
    ev.to_csv(f"{args.outdir}/knife_events.csv", index=False)

    # ---- cohort table: every metric vs outcome ----
    def coh(mask, name):
        g = ev[mask]
        if len(g) < 15:
            return None
        return {
            "cohort": name, "n": len(g),
            "fwd5_bps": 1e4 * g["fwd5"].mean(),
            "fwd10_bps": 1e4 * g["fwd10"].mean(),
            "p_bounce5": float((g["fwd5"] > 0).mean()),
            "p90_5d_pct": 100 * g["fwd5"].quantile(0.9),
            "p10_5d_pct": 100 * g["fwd5"].quantile(0.1),
            "p_second_knife": g["second_knife"].mean(),
        }

    cohorts = [
        coh(ev.index == ev.index, "ALL knives"),
        coh(ev.news_crash == 1, "news-driven crash"),
        coh(ev.news_crash == 0, "no-news crash"),
        coh(ev.kind_fast == 1, "fast (-15%/5d)"),
        coh(ev.kind_grind == 1, "grind (8/10 down)"),
        coh(ev.climax_rvol >= 3, "climax volume >=3x"),
        coh(ev.reversal_bar == 1, "reversal bar"),
        coh(ev.gap_exhaust == 1, "gap-down exhaustion"),
        coh((ev.news_crash == 0) & ((ev.reversal_bar == 1)
                                    | (ev.climax_rvol >= 3)),
            "FORK RULE: no-news + (reversal|climax)"),
        coh((ev.news_crash == 1) & (ev.climax_rvol >= 3),
            "news + climax (the trap)"),
        coh(ev.sigma_crash <= -4, "crash >4 sigma"),
        coh(ev.quality > 0.5, "high quality"),
        coh(ev.quality < -0.5, "low quality"),
        coh(ev.accel > 0.5, "parabolic finish"),
        coh(ev.convexity < 0, "accelerating path (convex down)"),
    ]
    cohort_df = pd.DataFrame([c for c in cohorts if c])
    cohort_df.to_csv(f"{args.outdir}/knife_cohorts.csv", index=False)
    print(cohort_df.round(3).to_string(index=False))

    # ---- walk-forward forks + analog forecaster ----
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.neighbors import NearestNeighbors
    ev = ev.sort_values("date").reset_index(drop=True)
    months = pd.PeriodIndex(ev["date"], freq="M")
    X = ev[FEATS].values.astype(float)
    X[~np.isfinite(X)] = 0.0
    y = ev["fwd5"].values
    gbm_pred = np.full(len(ev), np.nan)
    knn_pred = np.full(len(ev), np.nan)
    for m in pd.period_range(months.min(), months.max(), freq="M"):
        te = np.where(months == m)[0]
        tr = np.where(months < m)[0]
        if len(te) == 0 or len(tr) < 300:
            continue
        gbm = HistGradientBoostingRegressor(max_iter=120, max_depth=3,
                                            learning_rate=0.07,
                                            random_state=2)
        gbm.fit(X[tr], y[tr])
        gbm_pred[te] = gbm.predict(X[te])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        nn = NearestNeighbors(n_neighbors=min(25, len(tr)))
        nn.fit((X[tr] - mu) / sd)
        _, nbr = nn.kneighbors((X[te] - mu) / sd)
        knn_pred[te] = y[tr][nbr].mean(axis=1)
    ev["gbm_pred"] = gbm_pred
    ev["knn_pred"] = knn_pred

    scored = ev.dropna(subset=["gbm_pred"])
    strat = {
        "catch ALL knives": scored["fwd5"] - rt,
        "rule fork (no-news + reversal|climax)": scored.loc[
            (scored.news_crash == 0) & ((scored.reversal_bar == 1)
                                        | (scored.climax_rvol >= 3)),
            "fwd5"] - rt,
        "model fork (GBM pred > +100bp)": scored.loc[
            scored.gbm_pred > 0.01, "fwd5"] - rt,
        "analog fork (kNN cohort > +100bp)": scored.loc[
            scored.knn_pred > 0.01, "fwd5"] - rt,
    }
    summ = []
    for name, s in strat.items():
        if len(s) < 10:
            continue
        summ.append({"strategy": name, "n_trades": len(s),
                     "avg_5d_net_bps": 1e4 * s.mean(),
                     "hit": float((s > 0).mean()),
                     "nw_t": newey_west_t(s.values, lags=6),
                     "daily_equiv_bps": 1e4 * s.mean() / 5})
    summ = pd.DataFrame(summ)
    summ.to_csv(f"{args.outdir}/knife_forks.csv", index=False)
    print("\n", summ.round(3).to_string(index=False))

    # analog calibration
    okk = ev.dropna(subset=["knn_pred"])
    ic = float(np.corrcoef(okk["knn_pred"], okk["fwd5"])[0, 1])
    icg = float(np.corrcoef(scored["gbm_pred"], scored["fwd5"])[0, 1])
    print(f"\n[knife] analog kNN IC={ic:.3f}  GBM IC={icg:.3f}")

    # ---- average chart around events (how the charts looked) ----
    paths = {}
    for label, mask in (("news", ev.news_crash == 1),
                        ("nonews", ev.news_crash == 0)):
        g = ev[mask]
        acc = []
        for d, tkr in zip(g["date"], g["ticker"]):
            t, j = pos_d[d], pos_t[tkr]
            if t < 15 or t + 21 >= len(fwd.dates):
                continue
            seg = np.log(cl[t - 15:t + 21, j] / cl[t, j])
            acc.append(seg)
        paths[label] = np.nanmean(np.vstack(acc), axis=0)
    np.save(f"{args.outdir}/knife_paths.npy", paths, allow_pickle=True)

    return {"events": ev, "cohorts": cohort_df, "forks": summ,
            "paths": paths, "ic": {"knn": ic, "gbm": icg}}


if __name__ == "__main__":
    main()
