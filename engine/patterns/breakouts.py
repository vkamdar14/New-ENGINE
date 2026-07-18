"""Breakout systems: opening-range, Donchian channels, volatility
contraction (VCP), with news/volume qualifiers.

  * orb_30min_{up|dn}: the daily CLOSE finishes beyond the first-30-minute
    range (EOD-confirmed opening-range hold; entry at that close).  This is
    deliberately NOT the intraday-executed ORB of Crabel (1990) -- honest
    intraday execution needs intraday data (see loaders.py).  Requires
    or30 fields; synthesized on the simulator, NaN -> skipped on real
    daily data.
  * donchian{20,55}_{up|dn}: Turtle/Donchian channel breakout on close vs
    prior N-day extreme of closes.  Faith (2007); Moskowitz et al. TS-mom.
  * vcp_breakout: >=3 successively tighter 5-day ranges (each < 75% of the
    prior), dry volume (RVOL<0.8 in the pinch), then close above the
    contraction high on RVOL>1.5 (Minervini's rules, algorithmized).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Wide, empty, pack


def detect(panel: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
    w = Wide(panel)
    c, h, l = w.close, w.high, w.low
    frames = []

    # ---------- opening-range breakout (intraday-actionable) ----------
    if w.or30_high.notna().any().any():
        up = (c > w.or30_high) & (w.gap.abs() < 0.005)   # exclude big-gap days
        dn = (c < w.or30_low) & (w.gap.abs() < 0.005)
        or_rng = (w.or30_high - w.or30_low) / w.atr
        # narrow OR (NR-style) breakouts are the canonical setup
        narrow = or_rng < or_rng.rolling(20, min_periods=10).median()
        for m, name, d in ((up & narrow, "orb_30min_up", 1),
                           (dn & narrow, "orb_30min_dn", -1)):
            frames.append(pack(w.index, w.mask_to_index(m.fillna(False)).values,
                               "breakouts", name, d,
                               w.flat((1 / or_rng.clip(0.1)).where(m))
                               .fillna(0).values, 1))

    # ---------- Donchian ----------
    for n in (20, 55):
        hi_n = c.rolling(n, min_periods=n).max().shift(1)
        lo_n = c.rolling(n, min_periods=n).min().shift(1)
        up = c > hi_n
        dn = c < lo_n
        fresh_up = up & ~up.shift(1, fill_value=False)
        fresh_dn = dn & ~dn.shift(1, fill_value=False)
        stretch = ((c - hi_n) / w.atr)
        for m, name, d, s in ((fresh_up, f"donchian{n}_up", 1, stretch),
                              (fresh_dn, f"donchian{n}_dn", -1,
                               (lo_n - c) / w.atr)):
            frames.append(pack(w.index, w.mask_to_index(m.fillna(False)).values,
                               "breakouts", name, d,
                               w.flat(s.where(m)).fillna(0).values, 10))
        # volume-confirmed variant
        conf_up = fresh_up & (w.rvol > 1.5)
        frames.append(pack(w.index, w.mask_to_index(conf_up.fillna(False)).values,
                           "breakouts", f"donchian{n}_up_hotvol", 1,
                           w.flat(stretch.where(conf_up)).fillna(0).values, 10))

    # ---------- volatility contraction pattern ----------
    rng5 = (h.rolling(5).max() - l.rolling(5).min())
    r0 = rng5
    r1 = rng5.shift(5)
    r2 = rng5.shift(10)
    contracting = (r0 < 0.75 * r1) & (r1 < 0.75 * r2)
    dry = w.rvol.rolling(5, min_periods=3).mean() < 0.8
    pivot = h.rolling(15, min_periods=15).max().shift(1)
    vcp = (contracting.shift(1) & dry.shift(1) & (c > pivot)
           & (w.rvol > 1.5))
    tightness = (1 - r0 / r2.replace(0, np.nan)).clip(0, 1)
    frames.append(pack(w.index, w.mask_to_index(vcp.fillna(False)).values,
                       "breakouts", "vcp_breakout", 1,
                       w.flat(tightness.where(vcp)).fillna(0).values, 10))

    out = [f for f in frames if len(f)]
    return pd.concat(out, ignore_index=True) if out else empty()
