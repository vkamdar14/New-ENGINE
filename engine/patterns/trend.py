"""Trend-following MA and channel systems + time-series/cross-sectional
momentum.

Evidence: Brock-Lakonishok-LeBaron (1992) MA rules (pre-cost, decayed
post-publication -- Sullivan-Timmermann-White 1999 data-snooping
correction); Moskowitz-Ooi-Pedersen (2012) time-series momentum;
Jegadeesh-Titman (1993) cross-sectional momentum; Keltner/Bollinger
channel systems from the CTA literature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Wide, empty, pack


def detect(panel: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
    w = Wide(panel)
    c = w.close
    frames = []

    # ---------- MA cross systems ----------
    for fast, slow in ((5, 20), (20, 50), (50, 200)):
        mf = c.rolling(fast, min_periods=fast).mean()
        ms = c.rolling(slow, min_periods=slow).mean()
        above = mf > ms
        gold = above & ~above.shift(1).fillna(False)
        death = ~above & above.shift(1).fillna(True)
        spread = ((mf - ms) / ms).abs()
        frames += [
            pack(w.index, w.mask_to_index(gold.fillna(False)).values, "trend",
                 f"ma_cross_{fast}_{slow}_up", 1,
                 w.flat(spread.where(gold)).fillna(0).values, 20),
            pack(w.index, w.mask_to_index(death.fillna(False)).values, "trend",
                 f"ma_cross_{fast}_{slow}_dn", -1,
                 w.flat(spread.where(death)).fillna(0).values, 20),
        ]

    # ---------- time-series momentum (12-1) ----------
    tsm = c.pct_change(252) - c.pct_change(21)     # 12m ex last month
    pos = tsm > 0.10
    neg = tsm < -0.10
    frames += [
        pack(w.index, w.mask_to_index(pos.fillna(False)).values, "trend",
             "tsmom_12_1_up", 1, w.flat(tsm.where(pos)).fillna(0).values, 20),
        pack(w.index, w.mask_to_index(neg.fillna(False)).values, "trend",
             "tsmom_12_1_dn", -1, w.flat((-tsm).where(neg)).fillna(0).values, 20),
    ]

    # ---------- cross-sectional momentum (decile ranks, monthly grid) ----------
    r_12_1 = c.pct_change(252).shift(21)
    rank = r_12_1.rank(axis=1, pct=True)
    month_end = pd.Series(c.index, index=c.index).dt.month.diff().fillna(0) != 0
    grid = pd.DataFrame(np.tile(month_end.values[:, None], (1, c.shape[1])),
                        index=c.index, columns=c.columns)
    win = (rank >= 0.9) & grid
    lose = (rank <= 0.1) & grid
    frames += [
        pack(w.index, w.mask_to_index(win.fillna(False)).values, "trend",
             "xs_momentum_winner", 1, w.flat(rank.where(win)).fillna(0).values, 21),
        pack(w.index, w.mask_to_index(lose.fillna(False)).values, "trend",
             "xs_momentum_loser", -1,
             w.flat((1 - rank).where(lose)).fillna(0).values, 21),
    ]

    # ---------- Bollinger channel system ----------
    ma20 = c.rolling(20, min_periods=20).mean()
    sd20 = c.rolling(20, min_periods=20).std()
    upper, lower = ma20 + 2 * sd20, ma20 - 2 * sd20
    bb_break_up = (c > upper) & (c.shift(1) <= upper.shift(1))
    bb_revert_lo = (c < lower) & (w.rvol < 1.5)    # quiet drift under band -> revert
    frames += [
        pack(w.index, w.mask_to_index(bb_break_up.fillna(False)).values, "trend",
             "bollinger_breakout_up", 1,
             w.flat(((c - upper) / sd20).where(bb_break_up)).fillna(0).values, 5),
        pack(w.index, w.mask_to_index(bb_revert_lo.fillna(False)).values, "trend",
             "bollinger_revert_long", 1,
             w.flat(((lower - c) / sd20).where(bb_revert_lo)).fillna(0).values, 5),
    ]

    # ---------- Keltner channel ----------
    kma = c.ewm(span=20, min_periods=20).mean()
    kup = kma + 2 * w.atr
    kbrk = (c > kup) & (c.shift(1) <= kup.shift(1))
    frames.append(pack(w.index, w.mask_to_index(kbrk.fillna(False)).values,
                       "trend", "keltner_breakout_up", 1,
                       w.flat(((c - kup) / w.atr).where(kbrk)).fillna(0).values, 10))

    out = [f for f in frames if len(f)]
    return pd.concat(out, ignore_index=True) if out else empty()
