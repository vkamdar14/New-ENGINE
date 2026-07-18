"""Volume-price signals: relative volume spikes, accumulation/distribution
divergence, and on-balance-volume trend confirmation.

Evidence: volume is a *conditioner*, not a standalone signal -- Gervais,
Kaniel & Mingelgrin (2001, JF) "high-volume return premium" is the one
robust standalone effect (stocks with abnormally high volume outperform
over the next weeks); OBV/AD-line divergences per se have thin evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Wide, empty, pack


def detect(panel: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
    w = Wide(panel)
    c, h, l, v = w.close, w.high, w.low, w.volume
    frames = []

    # ---------- high-volume return premium (GKM 2001) ----------
    # extreme RVOL day with small |return|: information arrival without
    # resolution -> long (their finding: visibility premium)
    small_move = w.ret_cc.abs() < 0.01
    hv = (w.rvol > 3.0) & small_move
    frames.append(pack(w.index, w.mask_to_index(hv.fillna(False)).values,
                       "volume", "high_volume_premium", 1,
                       w.flat(w.rvol.where(hv)).fillna(0).values, 10))

    # ---------- RVOL spike with directional close ----------
    strong_close = (c - l) / (h - l).replace(0, np.nan)
    spike_up = (w.rvol > 2.5) & (strong_close > 0.8) & (w.ret_cc > 0.01)
    spike_dn = (w.rvol > 2.5) & (strong_close < 0.2) & (w.ret_cc < -0.01)
    frames += [
        pack(w.index, w.mask_to_index(spike_up.fillna(False)).values, "volume",
             "rvol_spike_up", 1, w.flat(w.rvol.where(spike_up)).fillna(0).values, 5),
        pack(w.index, w.mask_to_index(spike_dn.fillna(False)).values, "volume",
             "rvol_spike_dn", -1, w.flat(w.rvol.where(spike_dn)).fillna(0).values, 5),
    ]

    # ---------- accumulation/distribution divergence ----------
    clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    ad = (clv * v).cumsum()
    ad_slope = ad.diff(20)
    px_slope = c.diff(20)
    ad_z = ad_slope / ad_slope.rolling(120, min_periods=40).std()
    accum_div = (px_slope < 0) & (ad_z > 1.0)      # price down, A/D up -> long
    distr_div = (px_slope > 0) & (ad_z < -1.0)     # price up, A/D down -> short
    frames += [
        pack(w.index, w.mask_to_index(accum_div.fillna(False)).values, "volume",
             "accumulation_divergence", 1,
             w.flat(ad_z.where(accum_div)).fillna(0).values, 10),
        pack(w.index, w.mask_to_index(distr_div.fillna(False)).values, "volume",
             "distribution_divergence", -1,
             w.flat((-ad_z).where(distr_div)).fillna(0).values, 10),
    ]

    # ---------- OBV confirmation of 20d breakout ----------
    obv = (np.sign(w.ret_cc).fillna(0) * v).cumsum()
    brk = c > c.rolling(20, min_periods=20).max().shift(1)
    obv_conf = brk & (obv > obv.rolling(20, min_periods=20).max().shift(1))
    frames.append(pack(w.index, w.mask_to_index(obv_conf.fillna(False)).values,
                       "volume", "obv_confirmed_breakout", 1,
                       w.flat(w.rvol.where(obv_conf)).fillna(0).values, 10))

    out = [f for f in frames if len(f)]
    return pd.concat(out, ignore_index=True) if out else empty()
