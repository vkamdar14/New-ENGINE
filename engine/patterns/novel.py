"""ENGINE-ORIGINAL indicators and patterns (built for this project).

These are NEW constructs -- no prior literature, therefore no prior
evidence.  They are stated as precise formulas, tested identically to the
classical families, and flagged in the catalog with an explicit
multiple-testing discount (novel rules get a higher significance bar:
|t| > 3 rather than 2, per Harvey-Liu-Zhu 2016).

1. Trend Purity Index (TPI)
     TPI_t = sum_{k=1..20} w_k * 1{sign(r_{t-k}) == sign(P_t - P_{t-20})}
     with volume weights w_k = V_{t-k}/sum(V).  Purity > 0.75 with an
     up-trend and a 1-3 day pullback -> long ("clean-trend pullback").

2. Compression-Expansion Ratio (CER)
     CER_t = ATR5_t / ATR50_t.  Fire when CER crosses up through its own
     20th percentile (trailing 1y) after >=5 days below, with a
     directional close (|close-open| > 0.6 * range) -> trade close's sign.
     Generalizes VCP/NR7 into one continuous statistic.

3. Anchored-Memory Oscillator (AMO)
     A_t = sum over last 3 catalyst days a of w_a * close_a, with
     w_a proportional to RVOL_a * 0.5^{(t-a)/63}  (the market's decaying
     "memory anchors").  AMO_t = (close_t - A_t)/ATR14_t.
     Reclaim of the anchor from below (AMO crosses 0 upward) -> long;
     stretch beyond +4 -> fade short.

4. Shadow Pressure Index (SPI)
     SPI_t = EMA_10 of (upper_shadow - lower_shadow)/range.
     SPI < -0.25 in an uptrend = persistent dip-buying (lower rejections)
     -> long; SPI > +0.25 in a downtrend -> short.

5. Liquidity Vacuum (LVX)
     Range expansion without participation: range_t > 1.5*ATR14 and
     RVOL_t < 0.7 -> fade the day's direction (moves without volume are
     unbacked).

6. Gap Echo (GEX)
     After a >=3% news gap, count closes retaining >=50% of the gap over
     the following 3 days ("echo").  Echo==3 -> continuation long/short
     with the gap on day 4 (institutional absorption signature).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Wide, empty, pack


def detect(panel: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
    w = Wide(panel)
    o, h, l, c, v = w.open, w.high, w.low, w.close, w.volume
    atr = w.atr
    frames = []

    # ---------- 1. Trend Purity Index ----------
    trend_sign = np.sign(c - c.shift(20))
    match = (np.sign(w.ret_cc) == trend_sign).astype(float)
    vw = v / v.rolling(20).sum()
    tpi = (match * vw).rolling(20).sum()
    pullback = (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    tpi_long = (tpi > 0.75) & (trend_sign > 0) & pullback
    tpi_short = (tpi > 0.75) & (trend_sign < 0) & (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    frames += [
        pack(w.index, w.mask_to_index(tpi_long.fillna(False)).values, "novel",
             "tpi_pullback_long", 1, w.flat(tpi.where(tpi_long)).fillna(0).values, 5),
        pack(w.index, w.mask_to_index(tpi_short.fillna(False)).values, "novel",
             "tpi_rally_short", -1, w.flat(tpi.where(tpi_short)).fillna(0).values, 5),
    ]

    # ---------- 2. Compression-Expansion Ratio ----------
    atr5 = _atr_n(h, l, c, 5)
    atr50 = _atr_n(h, l, c, 50)
    cer = atr5 / atr50
    q20 = cer.rolling(252, min_periods=120).quantile(0.20)
    below = (cer < q20).rolling(5, min_periods=5).min().shift(1) == 1
    crossing = (cer >= q20) & below
    body_dir = np.sign(c - o)
    decisive = (c - o).abs() > 0.6 * (h - l).replace(0, np.nan)
    cer_up = crossing & decisive & (body_dir > 0)
    cer_dn = crossing & decisive & (body_dir < 0)
    frames += [
        pack(w.index, w.mask_to_index(cer_up.fillna(False)).values, "novel",
             "cer_expansion_long", 1,
             w.flat((1 / cer.clip(0.05)).where(cer_up)).fillna(0).values, 5),
        pack(w.index, w.mask_to_index(cer_dn.fillna(False)).values, "novel",
             "cer_expansion_short", -1,
             w.flat((1 / cer.clip(0.05)).where(cer_dn)).fillna(0).values, 5),
    ]

    # ---------- 3. Anchored-Memory Oscillator ----------
    cat = w.catalyst.fillna(0).astype(bool)
    anchor_px = c.where(cat)
    anchor_wt = w.rvol.where(cat)
    lam = 0.5 ** (1 / 63)
    num_a = np.zeros(c.shape); den_a = np.zeros(c.shape)
    num_s = np.zeros(c.shape[1]); den_s = np.zeros(c.shape[1])
    ap, aw = anchor_px.values, anchor_wt.values
    for t in range(c.shape[0]):
        num_s *= lam; den_s *= lam
        newa = np.isfinite(ap[t])
        wgt = np.where(newa, np.nan_to_num(aw[t], nan=1.0), 0.0)
        num_s = num_s + wgt * np.nan_to_num(ap[t])
        den_s = den_s + wgt
        num_a[t] = num_s
        den_a[t] = den_s
    den = pd.DataFrame(den_a, index=c.index, columns=c.columns)
    anchor = pd.DataFrame(num_a, index=c.index, columns=c.columns) / \
        den.replace(0, np.nan)
    amo = (c - anchor) / atr
    reclaim = (amo > 0) & (amo.shift(1) <= 0) & (den > 0.5)
    stretch = (amo > 4) & (den > 0.5)
    frames += [
        pack(w.index, w.mask_to_index(reclaim.fillna(False)).values, "novel",
             "amo_reclaim_long", 1, w.flat(amo.where(reclaim)).fillna(0).values, 5),
        pack(w.index, w.mask_to_index(stretch.fillna(False)).values, "novel",
             "amo_stretch_short", -1, w.flat(amo.where(stretch)).fillna(0).values, 5),
    ]

    # ---------- 4. Shadow Pressure Index ----------
    rng = (h - l).replace(0, np.nan)
    spi = ((h - np.maximum(o, c)) - (np.minimum(o, c) - l)) / rng
    spi = spi.ewm(span=10, min_periods=10).mean()
    up_tr = c > c.rolling(50, min_periods=50).mean()
    spi_long = (spi < -0.25) & up_tr
    spi_short = (spi > 0.25) & ~up_tr
    frames += [
        pack(w.index, w.mask_to_index(spi_long.fillna(False)).values, "novel",
             "spi_dipbuy_long", 1, w.flat((-spi).where(spi_long)).fillna(0).values, 5),
        pack(w.index, w.mask_to_index(spi_short.fillna(False)).values, "novel",
             "spi_rejection_short", -1,
             w.flat(spi.where(spi_short)).fillna(0).values, 5),
    ]

    # ---------- 5. Liquidity Vacuum ----------
    lvx = ((h - l) > 1.5 * atr) & (w.rvol < 0.7)
    d = -np.sign(c - o)   # fade the day's direction
    nz = lvx & (d != 0)
    frames.append(pack(w.index, w.mask_to_index(nz.fillna(False)).values, "novel",
                       "liquidity_vacuum_fade",
                       w.flat(d.where(nz)).fillna(0).values.astype(int),
                       w.flat(((h - l) / atr).where(nz)).fillna(0).values, 3))

    # ---------- 6. Gap Echo ----------
    g = w.gap
    big_news_gap = (g.abs() >= 0.03) & w.catalyst.fillna(0).astype(bool)
    gap_sign = np.sign(g)
    # gap-day references, seen from stamp date t = gap_day + 2
    pc0 = c.shift(1).shift(2)      # prior close before the gap
    og = o.shift(2)                # gap-day open
    esign = gap_sign.shift(2)
    half_gap = 0.5 * (og - pc0)    # signed: >0 for up gaps

    def _ret(day_close):
        return (day_close - pc0) * esign >= half_gap * esign

    echo = (_ret(c.shift(2)) & _ret(c.shift(1)) & _ret(c)
            & big_news_gap.shift(2, fill_value=False))
    m = echo & (esign != 0)
    frames.append(pack(w.index, w.mask_to_index(m.fillna(False)).values, "novel",
                       "gap_echo_continuation",
                       w.flat(esign.where(m)).fillna(0).values.astype(int),
                       w.flat(g.abs().shift(2).where(m)).fillna(0).values, 10))

    out = [f for f in frames if len(f)]
    return pd.concat(out, ignore_index=True) if out else empty()


def _atr_n(h, l, c, n):
    pc = c.shift(1)
    tr = np.maximum.reduce([(h - l).values, (h - pc).abs().values,
                            (l - pc).abs().values])
    return pd.DataFrame(tr, index=h.index, columns=h.columns).rolling(
        n, min_periods=max(3, n // 3)).mean()
