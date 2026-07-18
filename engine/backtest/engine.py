"""Event-study and portfolio backtester with explicit cost model.

TIMING CONTRACT (the anti-lookahead core):
  * A signal stamped `date` uses information available through `date`'s
    close (ORB/gap signals: through `date`'s intraday session -- still
    strictly before any entry price used here).
  * Default entry: next day's OPEN.  Signals whose name starts with
    "gap_" or "orb_" enter at `date`'s CLOSE (their information exists at
    the open / first 30 minutes of `date`; close entry is conservative).
  * Exit: close of `date` + horizon trading days.
  * Abnormal return = raw minus the equal-weight universe return over the
    identical window (beta=1 market adjustment, disclosed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CostModel


class Forward:
    """Precomputed forward/daily returns from both entry conventions.

    Matrices (T x N):
      cc[t] = close[t]/close[t-1] - 1   (day-t close-to-close return)
      oc[t] = close[t]/open[t] - 1      (day-t open-to-close return)
      fwd_no[h][t] = close[t+h]/open[t+1] - 1   (next-open entry)
      fwd_sc[h][t] = close[t+h]/close[t] - 1    (same-close entry)
    """

    def __init__(self, wide_open: pd.DataFrame, wide_close: pd.DataFrame,
                 horizons=(1, 2, 3, 5, 10, 20)):
        self.dates = wide_close.index
        self.horizons = tuple(horizons)
        self.tickers = list(wide_close.columns)
        o, c = wide_open.values, wide_close.values
        T = len(self.dates)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.cc = np.vstack([np.full((1, c.shape[1]), np.nan),
                                 c[1:] / c[:-1] - 1])
            self.oc = c / o - 1
        with np.errstate(invalid="ignore"):
            self.mkt_cc = np.where(
                np.isnan(self.cc).all(axis=1), 0.0,
                np.nanmean(self.cc, axis=1))
            self.mkt_oc = np.where(
                np.isnan(self.oc).all(axis=1), 0.0,
                np.nanmean(self.oc, axis=1))
        cum = np.concatenate([[1.0], np.cumprod(1 + self.mkt_cc)])
        self.fwd_no, self.fwd_sc = {}, {}
        self.mkt_h = {}      # same-close benchmark: mkt close[t] -> close[t+h]
        self.mkt_h_no = {}   # next-open benchmark: mkt open[t+1] -> close[t+h]
        for h in self.horizons:
            f_no = np.full_like(c, np.nan)
            f_sc = np.full_like(c, np.nan)
            if T > h:
                f_no[:T - h] = c[h:] / o[1:T - h + 1] - 1
                f_sc[:T - h] = c[h:] / c[:T - h] - 1
            self.fwd_no[h], self.fwd_sc[h] = f_no, f_sc
            m = np.full(T, np.nan)
            m[:T - h] = cum[1 + h:] / cum[1:T - h + 1] - 1     # days t+1..t+h
            self.mkt_h[h] = m
            # next-open window excludes the t -> t+1 overnight market move:
            # (1 + mkt_oc[t+1]) * prod(cc over t+2..t+h) - 1
            m_no = np.full(T, np.nan)
            m_no[:T - h] = ((1 + self.mkt_oc[1:T - h + 1])
                            * (cum[1 + h:] / cum[2:T - h + 2]) - 1)
            self.mkt_h_no[h] = m_no

    def locate(self, signals: pd.DataFrame):
        pos_d = {d: i for i, d in enumerate(self.dates)}
        pos_t = {t: j for j, t in enumerate(self.tickers)}
        ti = signals["date"].map(pos_d).values
        tj = signals["ticker"].map(pos_t).values
        return ti.astype(np.int64), tj.astype(np.int64)


def entry_mode(signal_name: str) -> str:
    return ("same_close" if signal_name.startswith(("gap_", "orb_"))
            else "next_open")


def attach_forward(signals: pd.DataFrame, fwd: Forward) -> pd.DataFrame:
    """Attach per-horizon raw and abnormal directional forward returns."""
    if not len(signals):
        return signals.copy()
    s = signals.reset_index(drop=True).copy()
    ti, tj = fwd.locate(s)
    same_close = s["signal"].str.startswith(("gap_", "orb_")).values
    d = s["direction"].values
    for h in fwd.horizons:
        raw = np.where(same_close, fwd.fwd_sc[h][ti, tj],
                       fwd.fwd_no[h][ti, tj])
        mkt = np.where(same_close, fwd.mkt_h[h][ti], fwd.mkt_h_no[h][ti])
        s[f"ret_{h}"] = raw * d
        s[f"abn_{h}"] = (raw - mkt) * d
    return s


def family_daily_returns(signals: pd.DataFrame, fwd: Forward,
                         costs: CostModel,
                         horizon: int | None = None) -> pd.Series:
    """Daily returns of an equal-weight portfolio of all active trades.

    Each signal is one trade: entry per its convention, held `horizon`
    days (its stamp unless overridden), exit at close.  Day-k marks:
      next_open : k=1 -> oc[t0+1]; k>=2 -> cc[t0+k]
      same_close: k>=1 -> cc[t0+k]
    Round-trip costs are charged on the entry-day mark.  Capital is split
    equally across trades active that day; days with no trades return 0.
    """
    daily = np.zeros(len(fwd.dates))
    if not len(signals):
        return pd.Series(daily, index=fwd.dates)
    s = signals.reset_index(drop=True)
    ti, tj = fwd.locate(s)
    dirs = s["direction"].values.astype(float)
    hs = (np.full(len(s), horizon) if horizon
          else s["horizon"].values).astype(int)
    same_close = s["signal"].str.startswith(("gap_", "orb_")).values
    T = len(fwd.dates)
    num = np.zeros(T)
    den = np.zeros(T)
    rt_cost = costs.round_trip_bps / 1e4
    hmax = int(hs.max())
    for k in range(1, hmax + 1):
        live = hs >= k
        day = ti + k
        ok = live & (day < T)
        if not ok.any():
            continue
        dk, jk = day[ok], tj[ok]
        first_open = (~same_close[ok]) & (k == 1)
        r = np.where(first_open, fwd.oc[dk, jk], fwd.cc[dk, jk])
        r = r * dirs[ok]
        r = np.where(k == 1, r - rt_cost, r)
        good = np.isfinite(r)
        np.add.at(num, dk[good], r[good])
        np.add.at(den, dk[good], 1.0)
    np.divide(num, den, out=daily, where=den > 0)
    return pd.Series(daily, index=fwd.dates)
