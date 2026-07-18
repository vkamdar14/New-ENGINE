"""Classical chart patterns via the Lo-Mamaysky-Wang (2000) method.

Reference: Lo, Mamaysky & Wang, "Foundations of Technical Analysis:
Computational Algorithms, Statistical Inference, and Empirical
Implementation", Journal of Finance 55(4), 2000.

Method (faithful, with disclosed practical adaptations):
  1. For each stock and each day t, take the trailing L=38-day price window
     (LMW roll 38-day windows daily).
  2. Smooth with a Nadaraya-Watson Gaussian-kernel regression on time.
     LMW pick h by cross-validation x 0.3; we use a fixed h=2.5 days,
     which sits in the range their CV procedure selects for daily data
     (disclosed simplification -- fixed h makes the smoother a single
     precomputed weight matrix, so the whole panel vectorizes).
  3. Local extrema of the smoothed curve, alternation enforced.
  4. A pattern completes when its defining 5 consecutive extrema
     E1..E5 (3 for double tops/bottoms) satisfy LMW's inequality
     conditions and the last extremum sits d=3 days before the window
     end (their detection lag).  The signal date is t: only data through
     t's close is used -- the smoother never sees beyond the window.

Patterns: HS, IHS, BTOP, BBOT, TTOP, TBOT, RTOP, RBOT, DTOP, DBOT
(the 10 LMW geometries), plus formal cup-and-handle and bull/bear flag
rules (post-LMW additions, definitions below).

Directions follow classical TA priors: HS/DTOP/BTOP bearish; IHS/DBOT/BBOT
bullish; triangles and rectangles trade the boundary BREAK direction on the
completion day.  LMW themselves only test conditional-distribution shifts;
the directional priors are what a practitioner trades, and the catalog
scores them honestly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .base import COLUMNS, Wide, empty

L = 38          # window length (LMW)
H_BW = 2.5      # kernel bandwidth in days
D_LAG = 3       # completion lag: last extremum this many days before window end
REL_TOL = 0.015  # 1.5% equality tolerance (LMW)
RECT_TOL = 0.0075  # 0.75% tolerance for rectangles (LMW)
DT_MIN_SEP = 22  # min separation for double tops/bottoms (LMW: > 22 days)


def _kernel_matrix(n: int = L, h: float = H_BW) -> np.ndarray:
    t = np.arange(n)
    w = np.exp(-0.5 * ((t[None, :] - t[:, None]) / h) ** 2)
    return w / w.sum(axis=1, keepdims=True)


_W = _kernel_matrix()


def _extrema(sm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices and kinds (+1 max, -1 min) of local extrema of one smoothed
    window, alternation enforced by keeping the more extreme of any
    same-kind run."""
    d = np.diff(sm)
    s = np.sign(d)
    s[s == 0] = 1
    turn = np.where(np.diff(s) != 0)[0] + 1
    if turn.size == 0:
        return np.empty(0, int), np.empty(0, int)
    kinds = np.where(s[turn - 1] > 0, 1, -1)
    # enforce alternation
    keep_idx: list[int] = []
    keep_kind: list[int] = []
    for i, k in zip(turn, kinds):
        if keep_kind and keep_kind[-1] == k:
            prev = keep_idx[-1]
            better = (sm[i] > sm[prev]) if k == 1 else (sm[i] < sm[prev])
            if better:
                keep_idx[-1] = i
        else:
            keep_idx.append(i)
            keep_kind.append(k)
    return np.asarray(keep_idx), np.asarray(keep_kind)


def _within(a: float, b: float, tol: float) -> bool:
    m = 0.5 * (a + b)
    return abs(a - m) <= tol * m and abs(b - m) <= tol * m


def _classify(e_val: np.ndarray, e_kind: np.ndarray,
              e_idx: np.ndarray) -> list[tuple[str, int, float]]:
    """Apply LMW conditions to the LAST extrema of a window.

    Returns list of (pattern, direction, strength).  Directions for
    triangles/rectangles are resolved by the caller from the breakout side
    (direction 0 here).
    """
    out = []
    n = len(e_val)
    if n >= 5:
        v = e_val[-5:]
        k = e_kind[-5:]
        if k[0] == 1:  # E1 max: HS / BTOP / TTOP / RTOP
            e1, e2, e3, e4, e5 = v
            if (e3 > e1 and e3 > e5 and _within(e1, e5, REL_TOL)
                    and _within(e2, e4, REL_TOL)):
                head = e3 / (0.5 * (e1 + e5)) - 1.0
                out.append(("head_shoulders", -1, head))
            if e1 < e3 < e5 and e2 > e4:
                out.append(("broadening_top", -1, (e5 - e1) / e1))
            if e1 > e3 > e5 and e2 < e4:
                out.append(("triangle_top", 0, (e1 - e5) / e1))
            tops, bots = v[::2], v[1::2]
            if (_within(tops.min(), tops.max(), RECT_TOL)
                    and _within(bots.min(), bots.max(), RECT_TOL)
                    and tops.min() > bots.max()):
                out.append(("rectangle_top", 0,
                            (tops.mean() - bots.mean()) / bots.mean()))
        else:  # E1 min: IHS / BBOT / TBOT / RBOT
            e1, e2, e3, e4, e5 = v
            if (e3 < e1 and e3 < e5 and _within(e1, e5, REL_TOL)
                    and _within(e2, e4, REL_TOL)):
                head = 1.0 - e3 / (0.5 * (e1 + e5))
                out.append(("inv_head_shoulders", 1, head))
            if e1 > e3 > e5 and e2 < e4:
                out.append(("broadening_bottom", 1, (e1 - e5) / e5))
            if e1 < e3 < e5 and e2 > e4:
                out.append(("triangle_bottom", 0, (e5 - e1) / e1))
            bots, tops = v[::2], v[1::2]
            if (_within(tops.min(), tops.max(), RECT_TOL)
                    and _within(bots.min(), bots.max(), RECT_TOL)
                    and tops.min() > bots.max()):
                out.append(("rectangle_bottom", 0,
                            (tops.mean() - bots.mean()) / bots.mean()))
    if n >= 3:
        v3, k3, i3 = e_val[-3:], e_kind[-3:], e_idx[-3:]
        if (k3[0] == 1 and k3[2] == 1 and i3[2] - i3[0] >= DT_MIN_SEP
                and _within(v3[0], v3[2], REL_TOL) and v3[1] < min(v3[0], v3[2])):
            out.append(("double_top", -1, 1 - v3[1] / v3[0]))
        if (k3[0] == -1 and k3[2] == -1 and i3[2] - i3[0] >= DT_MIN_SEP
                and _within(v3[0], v3[2], REL_TOL) and v3[1] > max(v3[0], v3[2])):
            out.append(("double_bottom", 1, v3[1] / v3[0] - 1))
    return out


def _detect_ticker(prices: np.ndarray, dates: pd.DatetimeIndex,
                   ticker: str) -> list[dict]:
    T = len(prices)
    if T < L + 5:
        return []
    win = sliding_window_view(prices, L)          # (T-L+1, L)
    sm = win @ _W.T                               # smoothed windows
    rows: list[dict] = []
    seen: set[tuple] = set()
    for wi in range(sm.shape[0]):
        t = wi + L - 1                            # info date index
        ei, ek = _extrema(sm[wi])
        if ei.size < 3:
            continue
        # completion condition: last extremum exactly D_LAG bars before end
        if ei[-1] != L - 1 - D_LAG:
            continue
        ev = win[wi][ei]                          # raw prices at extrema
        for name, direction, strength in _classify(ev, ek, ei):
            if direction == 0:
                # resolve break direction for triangles/rectangles:
                # close beyond the last boundary extremum's level
                boundary_hi = ev[ek == 1].min() if (ek == 1).any() else np.inf
                boundary_lo = ev[ek == -1].max() if (ek == -1).any() else -np.inf
                c = prices[t]
                if c > boundary_hi:
                    direction = 1
                elif c < boundary_lo:
                    direction = -1
                else:
                    continue                      # no break yet -> no trade
            key = (ticker, name, t - D_LAG)       # dedupe same completion
            if key in seen:
                continue
            seen.add(key)
            rows.append({"date": dates[t], "ticker": ticker, "family": "lmw",
                         "signal": name, "direction": int(direction),
                         "strength": float(strength), "horizon": 10})
    return rows


# ---------------- cup-and-handle and flags (formal, non-LMW) ---------------

CUP_LEN = 60          # cup window
HANDLE_MAX = 15       # handle length bound
CUP_DEPTH = (0.12, 0.35)


def _cup_handle(close: pd.DataFrame, high: pd.DataFrame) -> pd.DataFrame:
    """Formal cup-and-handle.  A breakout day tb qualifies iff:
      * close[tb] exceeds the max close of the prior (handle+cup) span
        (rim breakout) -- candidates pre-filtered on a fresh ~60d closing
        high, which keeps the quadratic fits tractable;
      * handle = the hl days ending at tb-1: stays in the upper half of the
        cup and retraces < 1/3 of cup depth;
      * cup = CUP_LEN days before the handle: quadratic U-fit with positive
        curvature and R^2 > 0.5, depth in CUP_DEPTH, rims within 3%.
    Signal date = tb (uses tb's close only)."""
    rows = []
    tt = np.arange(CUP_LEN)
    A = np.vstack([tt**2, tt, np.ones_like(tt)]).T
    pinv = np.linalg.pinv(A)
    fresh_high = close > close.rolling(60, min_periods=60).max().shift(1)
    for ticker in close.columns:
        c = close[ticker].values
        cand = np.where(fresh_high[ticker].values)[0]
        for tb in cand:
            for hl in (5, 8, 12, HANDLE_MAX):
                cs, ce = tb - hl - CUP_LEN, tb - hl
                if cs < 0:
                    continue
                seg = c[cs:ce]
                if np.any(~np.isfinite(seg)):
                    continue
                coef = pinv @ seg
                fit = A @ coef
                ssr = ((seg - fit) ** 2).sum()
                sst = ((seg - seg.mean()) ** 2).sum() + 1e-12
                if coef[0] <= 0 or 1 - ssr / sst < 0.5:
                    continue
                rim = max(seg[0], seg[-1])
                depth = 1 - seg.min() / rim
                if not (CUP_DEPTH[0] <= depth <= CUP_DEPTH[1]):
                    continue
                if abs(seg[0] - seg[-1]) / rim > 0.03:
                    continue
                handle = c[ce:tb]
                if handle.min() < seg.min() + 0.5 * (rim - seg.min()):
                    continue                      # handle left upper half
                if (rim - handle.min()) / rim > depth / 3:
                    continue                      # handle too deep
                if c[tb] > rim:
                    rows.append({"date": close.index[tb], "ticker": ticker,
                                 "family": "lmw", "signal": "cup_handle",
                                 "direction": 1, "strength": float(depth),
                                 "horizon": 20})
                    break
    return pd.DataFrame(rows, columns=COLUMNS) if rows else empty()


def _flags(w: Wide) -> pd.DataFrame:
    """Bull/bear flag: pole = |20d return| >= 4*ATR14/price cumulated
    (proxy: 10-day |return| >= 12%); flag = next 5-10 days with range
    < 40% of pole and drift against the pole; signal on close breaking the
    flag extreme in the pole direction."""
    c = w.close
    pole = c.pct_change(10).shift(6)              # pole ended ~6 days ago
    flag_hi = w.high.rolling(6).max().shift(1)
    flag_lo = w.low.rolling(6).min().shift(1)
    flag_rng = (flag_hi - flag_lo) / c
    pole_ok_up = pole > 0.12
    pole_ok_dn = pole < -0.12
    tight = flag_rng < 0.4 * pole.abs()
    brk_up = c > flag_hi
    brk_dn = c < flag_lo
    up = pole_ok_up & tight & brk_up
    dn = pole_ok_dn & tight & brk_dn
    frames = []
    for mask, sig, d in ((up, "bull_flag", 1), (dn, "bear_flag", -1)):
        st = mask.stack()
        st = st[st]
        if len(st):
            frames.append(pd.DataFrame({
                "date": st.index.get_level_values(0),
                "ticker": st.index.get_level_values(1),
                "family": "lmw", "signal": sig, "direction": d,
                "strength": pole.abs().stack().reindex(st.index).values,
                "horizon": 10}))
    return pd.concat(frames, ignore_index=True) if frames else empty()


def detect(panel: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
    w = Wide(panel)
    dates = w.close.index
    rows: list[dict] = []
    for ticker in w.close.columns:
        rows.extend(_detect_ticker(w.close[ticker].values, dates, ticker))
    frames = [pd.DataFrame(rows, columns=COLUMNS) if rows else empty(),
              _cup_handle(w.close, w.high), _flags(w)]
    return pd.concat(frames, ignore_index=True)
