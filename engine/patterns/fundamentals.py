"""Business-quality / moat factor and its interaction with chart signals.

Evidence: Novy-Marx (2013) gross profitability; Asness-Frazzini-Pedersen
(2019) Quality-Minus-Junk; moat persistence (stable margins/ROIC) is the
economic story behind both.  Effects are REAL but small and slow (a few
percent a year on decile spreads) -- the simulator embeds them at that
scale via a static observable `quality` score.

Signals:
  * quality_long / quality_short: top/bottom quality decile, monthly grid
  * quality_breakout: high quality AND fresh Donchian-20 breakout (the
    interaction: chart entries filtered by moat quality)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Wide, empty, pack


def detect(panel: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
    w = Wide(panel)
    q = w.quality
    if q.isna().all().all():
        return empty()
    c = w.close
    qrank = q.rank(axis=1, pct=True)

    month_end = pd.Series(c.index, index=c.index).dt.month.diff().fillna(0) != 0
    grid = pd.DataFrame(np.tile(month_end.values[:, None], (1, c.shape[1])),
                        index=c.index, columns=c.columns)
    top = (qrank >= 0.9) & grid
    bot = (qrank <= 0.1) & grid

    hi20 = c.rolling(20, min_periods=20).max().shift(1)
    brk = (c > hi20)
    fresh = brk & ~brk.shift(1, fill_value=False)
    q_brk = fresh & (qrank >= 0.8)

    frames = [
        pack(w.index, w.mask_to_index(top.fillna(False)).values, "fundamentals",
             "quality_long", 1, w.flat(qrank.where(top)).fillna(0).values, 21),
        pack(w.index, w.mask_to_index(bot.fillna(False)).values, "fundamentals",
             "quality_short", -1,
             w.flat((1 - qrank).where(bot)).fillna(0).values, 21),
        pack(w.index, w.mask_to_index(q_brk.fillna(False)).values, "fundamentals",
             "quality_breakout", 1,
             w.flat(qrank.where(q_brk)).fillna(0).values, 10),
    ]
    out = [f for f in frames if len(f)]
    return pd.concat(out, ignore_index=True) if out else empty()
