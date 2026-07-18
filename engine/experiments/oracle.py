"""The look-ahead winner study ("what would have picked the top-3?").

Three strictly separated layers:

1. ORACLE TABLE (hindsight, descriptive).  For each day of the final year,
   the top-3 close-to-close performers among the 150 most liquid names,
   with their day-of context (news? RVOL? gap?) and their EX-ANTE (t-1)
   observable profile.  This table looks ahead BY CONSTRUCTION and is
   labeled as such -- it is data, not a strategy.

2. ORACLE "STRATEGY" (cheating, the ceiling).  Buy each day's top-3 at the
   prior close with perfect foresight.  Not tradable; quantifies what
   perfect prediction of the extreme tail would earn.

3. HONEST PREDICTOR (the real strategy).  A walk-forward GBM classifier
   P(stock lands in tomorrow's top-3) trained ONLY on features observable
   at today's close (RVOL, recent news, gap, vol, momentum, 52wk position,
   compression, quality), retrained monthly on an expanding window that
   ends strictly before the prediction month.  Portfolio: buy the 3 highest
   predicted probabilities at next open, hold to close, full costs.
   The oracle-vs-honest gap is the value of information the market does
   not publish in advance.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..backtest.engine import Forward
from ..backtest.stats import newey_west_t
from ..config import RunConfig
from ..patterns.base import Wide


def build_ex_ante_features(w: Wide) -> dict[str, pd.DataFrame]:
    """All observable at day t's close; used to predict day t+1's winners."""
    c = w.close
    feats = {
        "rvol": w.rvol,
        "gap_abs": w.gap.abs(),
        "vol20": w.vol20,
        "mom5": c.pct_change(5),
        "mom21": c.pct_change(21),
        "mom252": c.pct_change(252),
        "prox52": c / w.hi52,
        "news_today": w.catalyst.astype(float),
        "news_3d": (w.catalyst.rolling(3, min_periods=1).max()).astype(float),
        "compression": (w.atr / c) / (w.vol20 + 1e-9),
        "quality": w.quality,
        "ret_today": w.ret_cc,
    }
    return feats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--n-universe", type=int, default=150)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--year-days", type=int, default=252)
    args = ap.parse_args(argv)
    cfg = RunConfig()
    rt = cfg.costs.round_trip_bps / 1e4

    print("[oracle] building panel (deterministic seed)...")
    from ..data.calibration import calibrate
    from ..data.synthetic import simulate
    data = simulate(cfg.sim, calibrate(verbose=False))
    panel = data["panel"]

    # --- "S&P 150" proxy: the 150 highest-ADV names ---
    adv = panel["volume"].groupby("ticker").mean().nlargest(args.n_universe)
    univ = sorted(adv.index)
    panel = panel[panel.index.get_level_values("ticker").isin(univ)]
    w = Wide(panel)
    ret = w.ret_cc
    dates = ret.index
    year = dates[-args.year_days:]
    print(f"[oracle] universe={len(univ)} names, window "
          f"{year[0].date()}..{year[-1].date()}")

    feats = build_ex_ante_features(w)

    # ---------------- 1. the oracle table ----------------
    rows = []
    for d in year:
        day = ret.loc[d].dropna().sort_values(ascending=False)
        for rank, (tkr, r) in enumerate(day.head(args.top_n).items(), 1):
            row = {"date": d.date(), "rank": rank, "ticker": tkr,
                   "return_pct": 100 * r,
                   "news_that_day": int(w.catalyst.loc[d, tkr] == 1),
                   "rvol_that_day": float(w.rvol.loc[d, tkr]),
                   "gap_that_day_pct": 100 * float(w.gap.loc[d, tkr])}
            iloc = dates.get_loc(d)
            if iloc > 0:
                dprev = dates[iloc - 1]
                for name in ("rvol", "news_3d", "mom21", "prox52", "vol20",
                             "compression"):
                    v = feats[name].loc[dprev, tkr]
                    row[f"exante_{name}"] = float(v) if np.isfinite(v) else np.nan
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(f"{args.outdir}/oracle_top3_table.csv", index=False)
    print(f"[oracle] table written: {len(table)} rows "
          f"({args.year_days} days x top-{args.top_n})")

    # ---------------- winner profile: ex-ante lifts ----------------
    winner_mask = pd.DataFrame(False, index=dates, columns=ret.columns)
    for d in year:
        top = ret.loc[d].dropna().nlargest(args.top_n).index
        winner_mask.loc[d, top] = True
    lifts = {}
    in_year = winner_mask.loc[year]
    for name, f in feats.items():
        f_prev = f.shift(1).loc[year]
        if name in ("news_today", "news_3d"):
            base = float(f_prev.stack().mean())
            wv = float(f_prev.where(in_year).stack().mean())
            lifts[name] = {"winners": wv, "universe": base,
                           "lift": wv / max(base, 1e-9)}
        else:
            base_med = float(f_prev.stack().median())
            wv_med = float(f_prev.where(in_year).stack().median())
            # rank-based lift: winners' median percentile in universe
            pct = f_prev.rank(axis=1, pct=True)
            wpct = float(pct.where(in_year).stack().median())
            lifts[name] = {"winners_median": wv_med, "universe_median": base_med,
                           "winner_median_pctile": wpct}
    pd.DataFrame(lifts).T.to_csv(f"{args.outdir}/oracle_winner_profile.csv")

    # day-of attribution: what made winners win
    news_rate = table["news_that_day"].mean()
    print(f"[oracle] {100 * news_rate:.1f}% of top-3 slots had a news "
          f"catalyst THAT day (universe base rate "
          f"{100 * float(w.catalyst.loc[year].stack().mean()):.1f}%)")

    # ---------------- 2. oracle ceiling ----------------
    oracle_daily = pd.Series(
        [ret.loc[d].dropna().nlargest(args.top_n).mean() for d in year],
        index=year)
    # ---------------- random-3 baseline ----------------
    rng = np.random.default_rng(0)
    rand_daily = pd.Series(
        [ret.loc[d].dropna().sample(args.top_n, random_state=int(s)).mean()
         for d, s in zip(year, rng.integers(0, 2**31, len(year)))], index=year)

    # ---------------- 3. honest walk-forward predictor ----------------
    print("[oracle] training walk-forward top-3 membership classifier...")
    from sklearn.ensemble import HistGradientBoostingClassifier
    fnames = sorted(feats)
    stacked = {n: feats[n].shift(1).stack() for n in fnames}  # t-1 features
    X = pd.DataFrame(stacked)
    # label: in top-3 on day t
    y = winner_mask.stack().reindex(X.index).astype(int)
    # full-universe winners for training history too (pre-year days)
    all_win = pd.DataFrame(False, index=dates, columns=ret.columns)
    for d in dates[30:]:
        top = ret.loc[d].dropna().nlargest(args.top_n).index
        all_win.loc[d, top] = True
    y = all_win.stack().reindex(X.index).astype(int)

    day_idx = X.index.get_level_values(0)
    months = pd.PeriodIndex(day_idx, freq="M")
    Xv = X.values.copy()
    Xv[~np.isfinite(Xv)] = 0.0
    proba = pd.Series(np.nan, index=X.index)
    for m in pd.period_range(months.min(), months.max(), freq="M"):
        te = months == m
        tr = months < m
        if te.sum() == 0 or tr.sum() < 20000:
            continue
        tr_i = np.where(tr)[0][-400_000:]
        clf = HistGradientBoostingClassifier(
            max_iter=120, max_depth=4, learning_rate=0.07, random_state=1)
        clf.fit(Xv[tr_i], y.values[tr_i])
        proba.iloc[np.where(te)[0]] = clf.predict_proba(Xv[te])[:, 1]

    fwd = Forward(w.open, w.close, (1,))
    picks_rows = []
    strat_daily = {}
    hits = 0
    slots = 0
    pos_d = {d: i for i, d in enumerate(dates)}
    pos_t = {t: j for j, t in enumerate(fwd.tickers)}
    for d in year:
        # proba row at label-date d uses features from d-1's close and a
        # model trained on months strictly before month(d) -- both fully
        # known before d's open
        if d not in proba.index.get_level_values(0):
            continue
        p = proba.loc[d].dropna()
        if len(p) < 10:
            continue
        picked = p.nlargest(args.top_n)
        actual = set(ret.loc[d].dropna().nlargest(args.top_n).index)
        # enter at d's open, exit at d's close (oc), costs
        rr = [fwd.oc[pos_d[d], pos_t[t]] for t in picked.index]
        r_net = float(np.nanmean(rr)) - rt
        strat_daily[d] = r_net
        hit_today = len(set(picked.index) & actual)
        hits += hit_today
        slots += args.top_n
        picks_rows.append({"date": d.date(),
                           "picked": ",".join(picked.index),
                           "prob": ",".join(f"{v:.3f}" for v in picked.values),
                           "hits_in_actual_top3": hit_today,
                           "net_return_pct": 100 * r_net})
    strat_daily = pd.Series(strat_daily)
    pd.DataFrame(picks_rows).to_csv(f"{args.outdir}/oracle_honest_picks.csv",
                                    index=False)

    res = {
        "oracle_avg_daily_pct": 100 * float(oracle_daily.mean()),
        "honest_avg_daily_net_pct": 100 * float(strat_daily.mean()),
        "honest_nw_t": newey_west_t(strat_daily.values),
        "random3_avg_daily_pct": 100 * float(rand_daily.mean()),
        "honest_hit_rate": hits / max(slots, 1),
        "random_hit_rate": args.top_n / len(univ),
        "capture_ratio": float(strat_daily.mean() / oracle_daily.mean()),
        "news_share_of_top3": float(news_rate),
    }
    pd.Series(res).to_csv(f"{args.outdir}/oracle_summary.csv")
    print("\n[oracle] ================ RESULTS ================")
    for k, v in res.items():
        print(f"  {k:28s} {v:.4f}")
    return {"table": table, "oracle": oracle_daily, "honest": strat_daily,
            "random": rand_daily, "res": res, "lifts": lifts}


if __name__ == "__main__":
    main()
