"""Per-edge statistics: significance, decay, cost survival, bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CostModel


def newey_west_t(x: np.ndarray, lags: int | None = None) -> float:
    """t-stat of mean(x) with Newey-West (Bartlett) HAC standard errors."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return np.nan
    if lags is None:
        lags = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    e = x - x.mean()
    g0 = (e @ e) / n
    s = g0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1 - l / (lags + 1)
        s += 2 * w * (e[l:] @ e[:-l]) / n
    se = np.sqrt(max(s, 1e-18) / n)
    return float(x.mean() / se)


def per_signal_stats(sig_fwd: pd.DataFrame, costs: CostModel,
                     horizons=(1, 2, 3, 5, 10, 20)) -> pd.DataFrame:
    """Event-study table: one row per (family, signal)."""
    rows = []
    rt = costs.round_trip_bps / 1e4
    for (fam, name), g in sig_fwd.groupby(["family", "signal"]):
        row = {"family": fam, "signal": name, "n": len(g),
               "direction": int(g["direction"].mode().iloc[0])}
        h0 = int(g["horizon"].mode().iloc[0])
        row["horizon"] = h0
        for h in horizons:
            a = g[f"abn_{h}"].values
            a = a[np.isfinite(a)]
            if len(a) < 8:
                row[f"abn_{h}"] = np.nan
                continue
            row[f"abn_{h}"] = a.mean()
            if h == h0:
                row["hit"] = float((a > 0).mean())
                # HAC lag ~ horizon (overlapping event windows)
                row["tstat"] = newey_west_t(a, lags=h0 + 1)
                row["net_per_trade"] = a.mean() - rt
                row["net_daily_equiv"] = (a.mean() - rt) / h0
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("tstat", ascending=False, na_position="last")


def decay_by_year(sig_fwd: pd.DataFrame) -> pd.DataFrame:
    """Mean abnormal return at each signal's stamped horizon, by calendar
    year -- the decay table."""
    g = sig_fwd.copy()
    g["year"] = pd.to_datetime(g["date"]).dt.year

    def own_abn(r):
        return r[f"abn_{int(r['horizon'])}"]

    g["abn_own"] = g.apply(own_abn, axis=1)
    out = (g.pivot_table(index=["family", "signal"], columns="year",
                         values="abn_own", aggfunc="mean"))
    out["n_total"] = g.groupby(["family", "signal"]).size()
    return out


def bootstrap_avg_daily(daily: pd.Series, n_draws: int = 1000,
                        seed: int = 11) -> np.ndarray:
    """1,000 random-draw averages: stationary block bootstrap (mean block
    ~21 days) of the daily return series; each draw's statistic is its
    average daily return."""
    r = daily.values
    r = r[np.isfinite(r)]
    n = len(r)
    rng = np.random.default_rng(seed)
    p = 1 / 21
    out = np.empty(n_draws)
    for b in range(n_draws):
        idx = np.empty(n, dtype=np.int64)
        idx[0] = rng.integers(n)
        jumps = rng.random(n) < p
        steps = rng.integers(0, n, size=n)
        for i in range(1, n):
            idx[i] = steps[i] if jumps[i] else (idx[i - 1] + 1) % n
        out[b] = r[idx].mean()
    return out


def ground_truth_recovery(sig: pd.DataFrame, truth: pd.DataFrame,
                          panel_index: pd.MultiIndex) -> pd.DataFrame:
    """Did detectors fire where the simulator actually embedded edge?

    For each (family, signal): mean hidden next-10-day drift (sum of pead +
    anchor + alpha + quality + gap_fade terms over the 10 days AFTER the
    stamp, signed by trade direction) vs the unconditional mean.  Placebo
    families should show ~zero."""
    tw = truth.copy()
    tw["total"] = tw.sum(axis=1)
    wide = tw["total"].unstack("ticker")
    fwd10 = wide[::-1].rolling(10, min_periods=1).sum()[::-1].shift(-1)
    flat = fwd10.stack().reindex(panel_index)
    base = float(np.nanmean(np.abs(flat.values)))
    rows = []
    idx = pd.MultiIndex.from_arrays([sig["date"], sig["ticker"]])
    vals = flat.reindex(idx).values * sig["direction"].values
    sig2 = sig.assign(truth_fwd=vals)
    for (fam, name), g in sig2.groupby(["family", "signal"]):
        rows.append({"family": fam, "signal": name,
                     "mean_true_drift_bps": 1e4 * np.nanmean(g["truth_fwd"]),
                     "n": len(g)})
    df = pd.DataFrame(rows)
    df["baseline_abs_bps"] = 1e4 * base
    return df.sort_values("mean_true_drift_bps", ascending=False)
