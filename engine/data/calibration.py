"""Calibrate simulator dynamics on REAL market data.

The `arch` package ships the actual S&P 500 daily OHLCV series 1999-2018
(source: Yahoo Finance, bundled with the package -- no network needed).
We fit the market-factor GARCH, the overnight/intraday return split, the
gap-size distribution, and the volume-|return| elasticity on it, so the
synthetic cross-section inherits real index dynamics rather than invented
ones.  Everything here is measurement of real data; nothing is invented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_real_sp500() -> pd.DataFrame:
    """Real S&P 500 daily OHLCV, 1999-2018, from the arch package."""
    from arch.data import sp500

    df = sp500.load()
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index.name = "date"
    return df


def fit_market_dynamics(px: pd.DataFrame) -> dict:
    """Fit GARCH(1,1)-t on real index returns + measure stylized facts.

    Returns a dict of parameters consumed by SimConfig / the simulator.
    """
    from arch import arch_model

    r = np.log(px["close"]).diff().dropna()
    am = arch_model(100 * r, vol="GARCH", p=1, q=1, dist="t", mean="Constant")
    res = am.fit(disp="off")
    p = res.params

    # overnight vs intraday split
    r_on = np.log(px["open"] / px["close"].shift(1)).dropna()
    r_id = np.log(px["close"] / px["open"]).dropna()

    # gap sizes (index gaps are small; single stocks scale these up via idio vol)
    gap = r_on
    # volume response to |return| (log-volume on |r| regression)
    lv = np.log(px["volume"]).diff().dropna()
    ar = r.abs().reindex(lv.index).dropna()
    lv = lv.reindex(ar.index)
    beta_vol = float(np.polyfit(ar.values, lv.values, 1)[0])

    # intraday range vs close-close vol (for synthesizing high/low)
    rng = (np.log(px["high"]) - np.log(px["low"])).dropna()
    parkinson = float((rng**2).mean() / (4 * np.log(2)))
    cc_var = float(r.var())

    return {
        "mu": float(p["mu"]) / 100.0,
        "omega": float(p["omega"]) / 100.0**2,
        "alpha": float(p["alpha[1]"]),
        "beta": float(p["beta[1]"]),
        "t_dof": float(p["nu"]),
        "overnight_var_share": float(r_on.var() / (r_on.var() + r_id.var())),
        "gap_std": float(gap.std()),
        "gap_kurtosis": float(gap.kurtosis()),
        "volume_abs_ret_beta": beta_vol,
        "range_var_ratio": parkinson / cc_var,  # ~1 for GBM; >1 with intraday noise
        "n_obs": int(len(r)),
        "sample": f"{px.index.min():%Y-%m-%d}..{px.index.max():%Y-%m-%d}",
    }


def calibrate(verbose: bool = True) -> dict:
    px = load_real_sp500()
    cal = fit_market_dynamics(px)
    if verbose:
        print("[calibration] real S&P 500", cal["sample"], f"n={cal['n_obs']}")
        for k in ("mu", "omega", "alpha", "beta", "t_dof",
                  "overnight_var_share", "gap_std", "volume_abs_ret_beta",
                  "range_var_ratio"):
            print(f"  {k:>22s} = {cal[k]:.6g}")
    return cal
