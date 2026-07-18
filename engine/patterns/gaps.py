"""Full gap taxonomy: direction x size x catalyst x relative volume.

The interaction the user asked for lives here: a gap WITH a fresh news
catalyst and abnormal volume is treated as a different signal from the
same-size gap on no news.  Literature: post-earnings-announcement drift
(Ball-Brown 1968; Bernard-Thomas 1989) says news-backed gaps CONTINUE;
overnight noise gaps tend to FADE/fill (gap-fill studies, e.g. Plastun
et al. 2019-2021).  The engine measures continuation vs fade for every
cell of the taxonomy.

Signal grammar:  gap_{up|dn}_{small|mid|large|huge}_{news|nonews}
  small: 0.5-1.5%   mid: 1.5-3%   large: 3-7%   huge: >7%
Direction convention encodes the HYPOTHESIS being traded:
  news gaps    -> continuation (long gap-up, short gap-down)
  no-news gaps -> fade (short gap-up, long gap-down)
The backtester scores each; where the hypothesis is wrong the stats will
say so -- the taxonomy itself stays neutral.

ENTRY TIMING: gap signals are stamped with the gap day as `date` and are
INTRADAY-ACTIONABLE: the gap is observable at the open of `date`.  The
backtester enters intraday signals at `date`'s close by default (fully
conservative: every input known by then) and additionally reports
open-entry stats for gap fades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Wide, empty, pack

SIZE_BINS = [(0.005, 0.015, "small"), (0.015, 0.03, "mid"),
             (0.03, 0.07, "large"), (0.07, np.inf, "huge")]
RVOL_HOT = 2.0


def detect(panel: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
    w = Wide(panel)
    gap = w.gap                     # open / prior close - 1, known at the open
    news = w.catalyst.fillna(0).astype(bool)
    hot = (w.rvol >= RVOL_HOT).fillna(False)

    frames = []
    for lo, hi, size in SIZE_BINS:
        for up in (True, False):
            g = (gap >= lo) & (gap < hi) if up else (gap <= -lo) & (gap > -hi)
            for has_news in (True, False):
                m = g & (news if has_news else ~news)
                if has_news:
                    d = 1 if up else -1          # continuation hypothesis
                else:
                    d = -1 if up else 1          # fade hypothesis
                name = (f"gap_{'up' if up else 'dn'}_{size}"
                        f"_{'news' if has_news else 'nonews'}")
                strength = w.flat(gap.abs().where(m)).fillna(0)
                frames.append(pack(w.index, w.mask_to_index(m.fillna(False)).values,
                                   "gaps", name, d, strength.values, 2))
                # volume-qualified variant of the news cells
                if has_news:
                    mh = m & hot
                    frames.append(pack(
                        w.index, w.mask_to_index(mh.fillna(False)).values,
                        "gaps", name + "_hotvol", d,
                        w.flat(gap.abs().where(mh)).fillna(0).values, 2))

    out = [f for f in frames if len(f)]
    return pd.concat(out, ignore_index=True) if out else empty()
