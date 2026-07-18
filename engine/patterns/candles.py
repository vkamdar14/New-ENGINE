"""Algorithmic candlestick patterns.

Definitions follow the quantified conventions of Marshall, Young & Rose
(2006, JBF) and Bulkowski's measured rules -- every pattern is a strict
inequality system on O/H/L/C, no eyeballing.  Body/shadow thresholds are
scaled by the 20-day ATR so they adapt to each stock's volatility.

The simulator embeds ZERO intrinsic candlestick effect, matching the
published evidence that candlestick signals carry no exploitable
predictability in US equities after costs; this family doubles as a
placebo control for the whole pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Wide, empty, pack

HORIZON = 3   # candlestick literature tests 1-10 day holding; 3 is standard


def detect(panel: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
    w = Wide(panel)
    o, h, l, c = w.open, w.high, w.low, w.close
    o1, h1, l1, c1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
    o2, c2 = o.shift(2), c.shift(2)

    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    up_sh = h - np.maximum(o, c)
    dn_sh = np.minimum(o, c) - l
    atr = w.atr
    trend_dn = c.shift(1) < c.shift(6)     # 5-day downtrend context
    trend_up = c.shift(1) > c.shift(6)

    sigs = {}

    # --- single-candle ---
    sigs["hammer"] = (trend_dn & (dn_sh >= 2 * body) & (up_sh <= 0.1 * rng)
                      & (body > 0.05 * rng), 1)
    sigs["shooting_star"] = (trend_up & (up_sh >= 2 * body)
                             & (dn_sh <= 0.1 * rng) & (body > 0.05 * rng), -1)
    sigs["doji"] = ((body <= 0.05 * rng) & (rng > 0.5 * atr), 0)
    sigs["bull_marubozu"] = ((c > o) & (body >= 0.95 * rng)
                             & (body > 0.8 * atr), 1)
    sigs["bear_marubozu"] = ((c < o) & (body >= 0.95 * rng)
                             & (body > 0.8 * atr), -1)

    # --- two-candle ---
    sigs["bull_engulf"] = (trend_dn & (c1 < o1) & (c > o)
                           & (o <= c1) & (c >= o1) & (body > body.shift(1)), 1)
    sigs["bear_engulf"] = (trend_up & (c1 > o1) & (c < o)
                           & (o >= c1) & (c <= o1) & (body > body.shift(1)), -1)
    sigs["bull_harami"] = (trend_dn & (c1 < o1) & (c > o)
                           & (np.maximum(o, c) <= o1)
                           & (np.minimum(o, c) >= c1), 1)
    sigs["bear_harami"] = (trend_up & (c1 > o1) & (c < o)
                           & (np.maximum(o, c) <= c1)
                           & (np.minimum(o, c) >= o1), -1)
    sigs["piercing"] = (trend_dn & (c1 < o1) & (o < l1)
                        & (c > 0.5 * (o1 + c1)) & (c < o1), 1)
    sigs["dark_cloud"] = (trend_up & (c1 > o1) & (o > h1)
                          & (c < 0.5 * (o1 + c1)) & (c > o1), -1)

    # --- three-candle ---
    body2 = (c.shift(2) - o.shift(2)).abs()
    small_mid = body.shift(1) < 0.3 * body2
    sigs["morning_star"] = (trend_dn.shift(2) & (c2 < o2) & small_mid
                            & (c > o) & (c > 0.5 * (o2 + c2)), 1)
    sigs["evening_star"] = (trend_up.shift(2) & (c2 > o2) & small_mid
                            & (c < o) & (c < 0.5 * (o2 + c2)), -1)
    up_body = (c > o) & (body > 0.5 * atr)
    sigs["three_white_soldiers"] = (up_body & up_body.shift(1)
                                    & up_body.shift(2) & (c > c1)
                                    & (c1 > c.shift(2)), 1)
    dn_body = (c < o) & (body > 0.5 * atr)
    sigs["three_black_crows"] = (dn_body & dn_body.shift(1)
                                 & dn_body.shift(2) & (c < c1)
                                 & (c1 < c.shift(2)), -1)

    frames = []
    for name, (mask, d) in sigs.items():
        mask = mask.fillna(False)
        if d == 0:
            continue  # doji is context, not a directional trade
        flat = w.mask_to_index(mask)
        strength = w.flat((body / atr).where(mask)).fillna(0.0)
        frames.append(pack(w.index, flat.values, "candles", name, d,
                           strength.values, HORIZON))
    out = pd.concat([f for f in frames if len(f)], ignore_index=True)
    return out if len(out) else empty()
