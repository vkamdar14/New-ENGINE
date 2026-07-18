"""Shared signal contract for every detector.

detect(panel, ctx) -> DataFrame with columns:
    date      : pd.Timestamp   (information date: signal uses data <= this close)
    ticker    : str
    family    : str            (edge family, e.g. "gaps")
    signal    : str            (specific rule, e.g. "gap_up_3_7_news")
    direction : int            (+1 long, -1 short)
    strength  : float          (comparable within a signal; larger = stronger)
    horizon   : int            (intended holding days; backtester also tests all)

HARD RULES (enforced by tests + adversarial review):
  * No row may use any data after `date`'s close (ORB signals may use the
    same day's first-30-minute fields ONLY, and are flagged intraday=True
    via signal name prefix "orb_" -- their entry is same-day, after the
    first half hour, at or30 breakout level).
  * Detectors receive the full panel for vectorization but must only use
    shifted/rolling constructs that respect the information date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["date", "ticker", "family", "signal", "direction", "strength", "horizon"]


def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def pack(index: pd.MultiIndex, mask: pd.Series | np.ndarray, family: str,
         signal: str, direction, strength, horizon: int) -> pd.DataFrame:
    """Build a signal frame from a boolean mask over a (date,ticker) index."""
    m = np.asarray(mask, dtype=bool)
    if m.sum() == 0:
        return empty()
    sel = index[m]
    dirs = (np.asarray(direction)[m] if np.ndim(direction) else
            np.full(m.sum(), direction))
    str_ = (np.asarray(strength)[m] if np.ndim(strength) else
            np.full(m.sum(), strength, dtype=float))
    return pd.DataFrame({
        "date": sel.get_level_values("date"),
        "ticker": sel.get_level_values("ticker"),
        "family": family, "signal": signal,
        "direction": dirs.astype(int), "strength": str_.astype(float),
        "horizon": horizon,
    })


def by_ticker(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    """Unstack one column to a (date x ticker) wide frame."""
    return panel[col].unstack("ticker")


class Wide:
    """Cached wide views of the panel + common derived series.

    Everything here is point-in-time safe when used with .shift(1) semantics
    documented per attribute.  All rolling stats are trailing.
    """

    def __init__(self, panel: pd.DataFrame):
        self.panel = panel
        self.open = by_ticker(panel, "open")
        self.high = by_ticker(panel, "high")
        self.low = by_ticker(panel, "low")
        self.close = by_ticker(panel, "close")
        self.volume = by_ticker(panel, "volume")
        self.catalyst = by_ticker(panel, "catalyst")
        self.news_sign = by_ticker(panel, "news_sign")
        self.quality = by_ticker(panel, "quality")
        self.or30_high = by_ticker(panel, "or30_high")
        self.or30_low = by_ticker(panel, "or30_low")

        self.ret_cc = self.close.pct_change()
        self.logret = np.log(self.close).diff()
        # trailing 63d ADV and RVOL (today's volume vs trailing 63d mean EXCL today)
        adv = self.volume.rolling(63, min_periods=20).mean().shift(1)
        self.rvol = self.volume / adv
        self.atr = _atr(self.high, self.low, self.close, 14)
        self.vol20 = self.ret_cc.rolling(20, min_periods=10).std()
        # 52-week trailing high/low of daily highs/lows, EXCLUDING today
        self.hi52 = self.high.rolling(252, min_periods=60).max().shift(1)
        self.lo52 = self.low.rolling(252, min_periods=60).min().shift(1)
        self.gap = self.open / self.close.shift(1) - 1.0
        self.index = panel.index

    def mask_to_index(self, wide_bool: pd.DataFrame) -> pd.Series:
        """Convert a (date x ticker) boolean frame to a flat mask aligned
        with panel.index."""
        s = wide_bool.stack()
        return s.reindex(self.index, fill_value=False).astype(bool)

    def flat(self, wide_vals: pd.DataFrame) -> pd.Series:
        return wide_vals.stack().reindex(self.index)


def _atr(high, low, close, n) -> pd.DataFrame:
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()],
                   keys=["a", "b", "c"]).groupby(level=1).max()
    return tr.rolling(n, min_periods=5).mean()
