"""52-week-high proximity -- the George & Hwang (2004) anchoring effect.

Evidence base: George & Hwang, "The 52-Week High and Momentum Investing",
Journal of Finance 59(5), 2004: nearness to the 52-week high predicts
higher future returns (anchoring: traders under-react near a salient
reference point).  This is real, replicated anchoring evidence -- unlike
most visual patterns.

Signals:
  * anchor_near_high : close within 2% of trailing 252d high (ex today) -> long
  * anchor_far_high  : close in the bottom quintile of the 52wk range -> short
  * anchor_break_high: first close ABOVE the prior 252d high in 60+ days -> long
  * strength = proximity ratio close/hi52 (or range position)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Wide, empty, pack


def detect(panel: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
    w = Wide(panel)
    c = w.close
    prox = c / w.hi52                       # <=1 typically; >1 = fresh breakout
    rangepos = (c - w.lo52) / (w.hi52 - w.lo52)

    near = (prox >= 0.98) & (prox <= 1.0)
    far = rangepos <= 0.2
    # fresh 52wk-high breakout: above prior 252d high, not within last 60d
    above = c > w.hi52
    fresh = above & (~above.rolling(60, min_periods=1).max().shift(1)
                     .fillna(0).astype(bool))

    frames = [
        pack(w.index, w.mask_to_index(near.fillna(False)).values, "anchors",
             "near_52wk_high", 1, w.flat(prox.where(near)).fillna(0).values, 20),
        pack(w.index, w.mask_to_index(far.fillna(False)).values, "anchors",
             "far_52wk_high", -1,
             w.flat((1 - rangepos).where(far)).fillna(0).values, 20),
        pack(w.index, w.mask_to_index(fresh.fillna(False)).values, "anchors",
             "fresh_52wk_breakout", 1,
             w.flat((prox - 1).where(fresh)).fillna(0).values, 10),
    ]
    out = [f for f in frames if len(f)]
    return pd.concat(out, ignore_index=True) if out else empty()
