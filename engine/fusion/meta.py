"""Signal fusion: a walk-forward meta-model over every edge family.

Design (formulas in CATALOG.md #12):
  * Feature vector x_{it} per (date, ticker):
      - family scores: sum of direction*strength of active signals per
        family (11 features)
      - context: gap, RVOL, catalyst flag, news sign, 52wk proximity,
        12-1 momentum, 20d vol, quality
  * Label y_{it} = next-day tradable return (open_{t+1} -> close_{t+1}),
    market-adjusted.
  * Model: HistGradientBoostingRegressor, retrained monthly on ALL data
    strictly before the prediction month (expanding window, >=15 months);
    plus a transparent linear baseline (z-scored family-score sum).
  * Portfolio: each day, among names with >=1 active signal, long top_k
    and short bottom_k scores, equal weight; full round-trip cost charged
    every day (positions assumed fully turned over daily -- conservative).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.engine import Forward
from ..config import CostModel
from ..patterns.base import Wide

CONTEXT = ["gap", "rvol", "catalyst", "news_sign", "prox52", "mom121",
           "vol20", "quality"]


def build_features(signals: pd.DataFrame, w: Wide) -> pd.DataFrame:
    """(date,ticker)-indexed features: per-family net score + context."""
    s = signals.copy()
    s["score"] = s["direction"] * s["strength"].clip(-10, 10)
    fam = (s.pivot_table(index=["date", "ticker"], columns="family",
                         values="score", aggfunc="sum")
           .add_prefix("f_"))
    ctx = pd.DataFrame({
        "gap": w.flat(w.gap),
        "rvol": w.flat(w.rvol),
        "catalyst": w.flat(w.catalyst.astype(float)),
        "news_sign": w.flat(w.news_sign.astype(float)),
        "prox52": w.flat(w.close / w.hi52),
        "mom121": w.flat(w.close.pct_change(252).shift(21)),
        "vol20": w.flat(w.vol20),
        "quality": w.flat(w.quality),
    })
    X = fam.reindex(ctx.index).fillna(0.0).join(ctx)
    X["n_active"] = (fam.reindex(ctx.index).notna()).sum(axis=1)
    return X


def walk_forward_fusion(X: pd.DataFrame, fwd: Forward, costs: CostModel,
                        top_k: int = 10, min_train_days: int = 320,
                        seed: int = 3) -> dict:
    """Returns dict with daily net/gross return series, per-day picks,
    feature importances."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance

    dates = fwd.dates
    pos_d = {d: i for i, d in enumerate(dates)}
    pos_t = {t: j for j, t in enumerate(fwd.tickers)}
    di = X.index.get_level_values("date").map(pos_d).values
    tj = X.index.get_level_values("ticker").map(pos_t).values
    y_raw = fwd.fwd_no[1][di, tj]
    y = y_raw - fwd.mkt_h_no[1][di]
    feat_cols = list(X.columns)
    Xv = X.values.astype(np.float64)
    Xv[~np.isfinite(Xv)] = 0.0
    ok = np.isfinite(y)

    months = pd.PeriodIndex(X.index.get_level_values("date"), freq="M")
    uniq_months = pd.period_range(months.min(), months.max(), freq="M")
    active = X["n_active"].values > 0

    score = np.full(len(X), np.nan)
    model = None
    importances = None
    day_of = X.index.get_level_values("date")
    max_train_rows = 200_000
    for m in uniq_months:
        tr = ok & (months < m) & active
        te = (months == m) & active
        if tr.sum() < min_train_days * 5 or te.sum() == 0:
            continue
        tr_idx = np.where(tr)[0]
        if len(tr_idx) > max_train_rows:      # keep the most recent rows
            tr_idx = tr_idx[-max_train_rows:]
        # date-based expanding window; retrain monthly
        model = HistGradientBoostingRegressor(
            max_iter=150, max_depth=4, learning_rate=0.06,
            l2_regularization=1.0, random_state=seed)
        model.fit(Xv[tr_idx], y[tr_idx])
        score[te] = model.predict(Xv[te])
    if model is not None:
        samp = np.where(ok & active)[0]
        samp = samp[np.linspace(0, len(samp) - 1,
                                min(20000, len(samp))).astype(int)]
        imp = permutation_importance(model, Xv[samp], y[samp], n_repeats=3,
                                     random_state=seed)
        importances = pd.Series(imp.importances_mean, index=feat_cols)

    # ---- portfolio construction ----
    sc = pd.Series(score, index=X.index, name="score").dropna()
    gross = pd.Series(0.0, index=dates)
    net = pd.Series(0.0, index=dates)
    picks = []
    rt = costs.round_trip_bps / 1e4
    y_ser = pd.Series(y_raw, index=X.index)
    for d, g in sc.groupby(level="date"):
        g = g.droplevel("date").sort_values()
        n_side = min(top_k, len(g) // 2)
        if n_side < 1:
            continue
        longs = g.index[-n_side:]
        shorts = g.index[:n_side]
        rl = y_ser.loc[(d, longs)].values
        rs = y_ser.loc[(d, shorts)].values
        r_g = (np.nanmean(rl) - np.nanmean(rs)) / 2.0   # 100% gross exposure
        if not np.isfinite(r_g):
            continue
        gross[d] = r_g
        net[d] = r_g - rt
        picks.append((d, list(longs), list(shorts)))
    # z-score-sum linear baseline -- WITHIN-DAY cross-sectional z-scores
    # (no time-series moments, hence no look-ahead)
    fam_cols = [c for c in feat_cols if c.startswith("f_")]
    fam_df = X[fam_cols]
    day = X.index.get_level_values("date")
    mu_d = fam_df.groupby(day).transform("mean")
    sd_d = fam_df.groupby(day).transform("std")
    lin = pd.Series(
        np.where(active,
                 np.nan_to_num(((fam_df - mu_d) / (sd_d + 1e-9))
                               .sum(axis=1).values), np.nan),
        index=X.index).dropna()
    lin_net = pd.Series(0.0, index=dates)
    for d, g in lin.groupby(level="date"):
        g = g.droplevel("date").sort_values()
        n_side = min(top_k, len(g) // 2)
        if n_side < 1:
            continue
        r_g = (np.nanmean(y_ser.loc[(d, g.index[-n_side:])].values)
               - np.nanmean(y_ser.loc[(d, g.index[:n_side])].values)) / 2.0
        if np.isfinite(r_g):
            lin_net[d] = r_g - rt
    return {"gross": gross, "net": net, "linear_net": lin_net,
            "picks": picks, "importances": importances, "scores": sc}
