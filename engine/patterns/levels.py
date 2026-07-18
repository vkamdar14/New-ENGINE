"""Support/resistance levels: prior swing highs/lows, round numbers,
anchored VWAP, and moving averages -- with bounce and break signals.

Level sources (all trailing / point-in-time):
  * swing levels: fractal highs/lows (5-day center) clustered within
    0.5*ATR, weight = number of touches (Osler 2000-style)
  * round numbers: $1/$5/$10 multiples depending on price magnitude
    (placebo in the simulator: zero embedded effect)
  * anchored VWAP: cumulative typical-price VWAP anchored at the latest
    catalyst day (news anchor), plus a 20-day rolling VWAP
  * moving averages: SMA 20/50/200 as dynamic support/resistance

Signals: `bounce_*` = touch level and close back away from it (fade);
`break_*` = close through level with confirmation (continuation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Wide, empty, pack


def _round_levels(price: pd.DataFrame) -> pd.DataFrame:
    """Nearest round number below/above: $5 grid under $50, $10 to $200,
    $25 above."""
    grid = pd.DataFrame(np.select(
        [price < 50, price < 200], [5.0, 10.0], default=25.0),
        index=price.index, columns=price.columns)
    return (price / grid).round() * grid


def detect(panel: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
    w = Wide(panel)
    o, h, l, c = w.open, w.high, w.low, w.close
    atr = w.atr
    frames = []

    # ---------- swing levels (fractal + touch count) ----------
    # fractal high at t-2: high[t-2] is max of the 5 highs ending at t
    frac_hi = (h.shift(2) == h.rolling(5).max()).shift(0)
    frac_lo = (l.shift(2) == l.rolling(5).min()).shift(0)
    # last confirmed swing level (as of t, uses data through t)
    swing_hi = h.shift(2).where(frac_hi).ffill()
    swing_lo = l.shift(2).where(frac_lo).ffill()
    near = 0.25  # within 0.25*ATR counts as a touch

    touch_res = (h >= swing_hi - near * atr) & (c < swing_hi) & (swing_hi > 0)
    brk_res = (c > swing_hi + near * atr) & (c.shift(1) <= swing_hi.shift(1))
    touch_sup = (l <= swing_lo + near * atr) & (c > swing_lo)
    brk_sup = (c < swing_lo - near * atr) & (c.shift(1) >= swing_lo.shift(1))

    dist_r = ((swing_hi - c) / atr).clip(0, 5)
    dist_s = ((c - swing_lo) / atr).clip(0, 5)
    frames += [
        pack(w.index, w.mask_to_index(touch_res.fillna(False)).values,
             "levels", "bounce_resistance", -1,
             w.flat((5 - dist_r).where(touch_res)).fillna(0).values, 3),
        pack(w.index, w.mask_to_index(brk_res.fillna(False)).values,
             "levels", "break_resistance", 1,
             w.flat(((c - swing_hi) / atr).where(brk_res)).fillna(0).values, 5),
        pack(w.index, w.mask_to_index(touch_sup.fillna(False)).values,
             "levels", "bounce_support", 1,
             w.flat((5 - dist_s).where(touch_sup)).fillna(0).values, 3),
        pack(w.index, w.mask_to_index(brk_sup.fillna(False)).values,
             "levels", "break_support", -1,
             w.flat(((swing_lo - c) / atr).where(brk_sup)).fillna(0).values, 5),
    ]

    # ---------- round numbers (placebo family) ----------
    rl = _round_levels(c)
    near_round = (c - rl).abs() <= 0.15 * atr
    above = c > rl
    frames += [
        pack(w.index, w.mask_to_index((near_round & above).fillna(False)).values,
             "levels", "round_number_above", 1, 1.0, 3),
        pack(w.index, w.mask_to_index((near_round & ~above).fillna(False)).values,
             "levels", "round_number_below", -1, 1.0, 3),
    ]

    # ---------- anchored VWAP at latest catalyst ----------
    tp = (h + l + c) / 3.0
    pv = tp * w.volume
    cat = w.catalyst.fillna(0).astype(bool)
    # group id increments at each catalyst day, per ticker
    gid = cat.cumsum()
    avwap = pd.DataFrame(index=c.index, columns=c.columns, dtype=float)
    for t in c.columns:
        g = gid[t]
        avwap[t] = pv[t].groupby(g).cumsum() / w.volume[t].groupby(g).cumsum()
    cross_up = (c > avwap) & (c.shift(1) <= avwap.shift(1)) & (gid > 0)
    cross_dn = (c < avwap) & (c.shift(1) >= avwap.shift(1)) & (gid > 0)
    frames += [
        pack(w.index, w.mask_to_index(cross_up.fillna(False)).values,
             "levels", "avwap_reclaim", 1,
             w.flat(((c - avwap) / atr).where(cross_up)).fillna(0).values, 5),
        pack(w.index, w.mask_to_index(cross_dn.fillna(False)).values,
             "levels", "avwap_loss", -1,
             w.flat(((avwap - c) / atr).where(cross_dn)).fillna(0).values, 5),
    ]

    # ---------- moving averages as dynamic S/R ----------
    for n in (20, 50, 200):
        ma = c.rolling(n, min_periods=n).mean()
        above_ma = c > ma
        # bounce: touched MA intraday from above, closed back above
        bounce = above_ma & (l <= ma) & (c.shift(1) > ma.shift(1))
        lose = (~above_ma) & above_ma.shift(1, fill_value=False)
        frames += [
            pack(w.index, w.mask_to_index(bounce.fillna(False)).values,
                 "levels", f"ma{n}_bounce", 1,
                 w.flat(((c - ma) / atr).where(bounce)).fillna(0).values, 5),
            pack(w.index, w.mask_to_index(lose.fillna(False)).values,
                 "levels", f"ma{n}_loss", -1,
                 w.flat(((ma - c) / atr).where(lose)).fillna(0).values, 5),
        ]

    out = [f for f in frames if len(f)]
    return pd.concat(out, ignore_index=True) if out else empty()
