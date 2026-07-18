"""The required news x charts interaction study.

For every overnight gap and every Donchian-20 breakout, measure
continuation vs fade split by:
    catalyst present / absent  x  gap-size bin  x  RVOL tercile

Definitions (all measurable, no eyeballing):
  gap day t, gap g_t = open_t/close_{t-1} - 1
  * intraday continuation: sign(close_t - open_t) == sign(g_t)
  * gap filled by close:  close_t crosses back to/beyond close_{t-1}
  * next-day drift: (close_{t+1}/open_{t+1} - 1) * sign(g_t)  [tradable]
  * 5-day drift:    (close_{t+5}/open_{t+1} - 1) * sign(g_t)
For breakouts on day t (close beyond prior 20d high/low):
  * next-day and 5-day drift in breakout direction, by catalyst x RVOL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..patterns.base import Wide

GAP_BINS = [(0.005, 0.015, "small"), (0.015, 0.03, "mid"),
            (0.03, 0.07, "large"), (0.07, np.inf, "huge")]


def _tercile(x: pd.DataFrame) -> pd.DataFrame:
    r = x.rank(axis=1, pct=True)
    return pd.DataFrame(np.select([r <= 1 / 3, r <= 2 / 3], [0, 1], 2),
                        index=x.index, columns=x.columns).where(x.notna())


def gap_interaction_table(panel: pd.DataFrame) -> pd.DataFrame:
    w = Wide(panel)
    g = w.gap
    sign = np.sign(g)
    news = w.catalyst.fillna(0).astype(bool)
    rvol_t = _tercile(w.rvol)

    o, c, c1 = w.open, w.close, w.close.shift(1)
    intraday = (c - o) * sign > 0
    filled = ((sign > 0) & (c <= c1)) | ((sign < 0) & (c >= c1))
    nd = (w.close.shift(-1) / w.open.shift(-1) - 1) * sign
    d5 = (w.close.shift(-5) / w.open.shift(-1) - 1) * sign

    rows = []
    for lo, hi, size in GAP_BINS:
        in_bin = (g.abs() >= lo) & (g.abs() < hi)
        for has_news in (True, False):
            base = in_bin & (news if has_news else ~news)
            for tv, tname in ((None, "all"), (0, "lowV"), (1, "midV"),
                              (2, "highV")):
                m = base if tv is None else (base & (rvol_t == tv))
                n = int(m.sum().sum())
                if n < 30:
                    continue
                rows.append({
                    "size": size, "news": has_news, "rvol": tname, "n": n,
                    "p_intraday_cont": float(intraday.where(m).stack().mean()),
                    "p_filled_same_day": float(filled.where(m).stack().mean()),
                    "nextday_drift_bps": 1e4 * float(nd.where(m).stack().mean()),
                    "d5_drift_bps": 1e4 * float(d5.where(m).stack().mean()),
                })
    return pd.DataFrame(rows)


def breakout_interaction_table(panel: pd.DataFrame) -> pd.DataFrame:
    w = Wide(panel)
    c = w.close
    hi20 = c.rolling(20, min_periods=20).max().shift(1)
    lo20 = c.rolling(20, min_periods=20).min().shift(1)
    up = (c > hi20) & ~(c > hi20).shift(1, fill_value=False)
    dn = (c < lo20) & ~(c < lo20).shift(1, fill_value=False)
    sign = pd.DataFrame(np.where(up, 1.0, np.where(dn, -1.0, np.nan)),
                        index=c.index, columns=c.columns)
    news2 = (w.catalyst.rolling(2, min_periods=1).max() > 0)  # news today or yday
    rvol_t = _tercile(w.rvol)
    nd = (w.close.shift(-1) / w.open.shift(-1) - 1) * sign
    d5 = (w.close.shift(-5) / w.open.shift(-1) - 1) * sign
    any_brk = sign.notna()

    rows = []
    for has_news in (True, False):
        base = any_brk & (news2 if has_news else ~news2)
        for tv, tname in ((None, "all"), (0, "lowV"), (1, "midV"), (2, "highV")):
            m = base if tv is None else (base & (rvol_t == tv))
            n = int(m.sum().sum())
            if n < 30:
                continue
            rows.append({
                "news": has_news, "rvol": tname, "n": n,
                "nextday_drift_bps": 1e4 * float(nd.where(m).stack().mean()),
                "d5_drift_bps": 1e4 * float(d5.where(m).stack().mean()),
                "p_nextday_cont": float((nd > 0).where(m).stack().mean()),
            })
    return pd.DataFrame(rows)
