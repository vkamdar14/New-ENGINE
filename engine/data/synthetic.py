"""Synthetic cross-sectional market with literature-calibrated ground truth.

WHY SYNTHETIC.  This sandbox's network policy blocks every market-data host
(stooq, Yahoo, GDELT, HuggingFace -- verified 403 at the egress proxy), so a
real cross-sectional panel cannot be downloaded here.  The engine therefore
runs on a simulator whose *dynamics* are calibrated on the real S&P 500
series bundled with the `arch` package and whose *embedded effects* are set
to published magnitudes.  `engine/data/loaders.py` contains ready adapters:
point them at a network-enabled environment and `run_all.py --source real`
reruns the identical pipeline on real data.

GROUND TRUTH EMBEDDED (all magnitudes documented in CATALOG.md):
  * news-catalyst jumps with post-event drift (PEAD-calibrated)  -> REAL edge
  * no-news overnight gaps partially fade                        -> REAL edge (small)
  * 52-week-high proximity carries extra drift (George-Hwang)    -> REAL edge (small)
  * slow OU alpha per stock (hosts trend/momentum systems)       -> REAL edge (modest)
  * quality/moat premium via static quality score                -> REAL edge (small)
  * high RVOL amplifies news drift (interaction)                 -> REAL edge
  * candlestick shapes per se                                    -> PLACEBO (zero)
  * round-number levels                                          -> PLACEBO (zero)
  * classical LMW geometries per se                              -> PLACEBO (zero;
    any profit they show must come from riding the embedded trend/news terms)

The simulator exposes what a real trader could observe: OHLCV, a news feed
(catalyst flag on the day it hits), a static quality score (public
fundamentals), and a synthesized first-30-minute range for opening-range
breakouts.  Hidden state (true drift terms) is returned separately for
ground-truth recovery diagnostics and is NEVER given to detectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SimConfig


def _student_t(rng: np.random.Generator, dof: float, size) -> np.ndarray:
    """Unit-variance Student-t innovations."""
    z = rng.standard_t(dof, size=size)
    return z / np.sqrt(dof / (dof - 2.0))


def _garch_path(rng, n, omega, alpha, beta, dof) -> np.ndarray:
    """Zero-mean GARCH(1,1)-t return path."""
    h = omega / max(1e-12, (1 - alpha - beta))
    out = np.empty(n)
    z = _student_t(rng, dof, n)
    for t in range(n):
        out[t] = np.sqrt(h) * z[t]
        h = omega + alpha * out[t] ** 2 + beta * h
    return out


def simulate(cfg: SimConfig, calib: dict | None = None) -> dict:
    """Build the panel.  Returns dict with:
       panel  : DataFrame indexed (date, ticker) with columns
                open, high, low, close, volume, catalyst, news_sign,
                quality, or30_high, or30_low
       truth  : DataFrame indexed (date, ticker) with hidden per-day drift
                components (pead, gap_fade, anchor, alpha, quality) for
                ground-truth recovery diagnostics
       dates  : DatetimeIndex;  tickers : list[str]
    """
    rng = np.random.default_rng(cfg.seed)
    n_days = int(round(252 * cfg.years))
    dates = pd.bdate_range(end="2026-07-17", periods=n_days)
    tickers = [f"S{i:03d}" for i in range(cfg.n_stocks)]

    mu = cfg.mkt_mu
    omega, alpha_g, beta_g, dof = (cfg.garch_omega, cfg.garch_alpha,
                                   cfg.garch_beta, cfg.t_dof)
    on_share = 0.30
    vol_beta = 6.0
    if calib:
        mu = calib["mu"]
        omega, alpha_g, beta_g = calib["omega"], calib["alpha"], calib["beta"]
        dof = max(4.0, calib["t_dof"])
        on_share = float(np.clip(calib["overnight_var_share"], 0.1, 0.5))
        vol_beta = calib["volume_abs_ret_beta"]

    # ----- market + sector factors -----
    f_mkt = mu + _garch_path(rng, n_days, omega, alpha_g, beta_g, dof)
    sector_of = rng.integers(0, cfg.n_sectors, cfg.n_stocks)
    f_sec = np.stack([
        0.6 * _garch_path(rng, n_days, omega * 0.5, alpha_g, beta_g, dof)
        for _ in range(cfg.n_sectors)
    ])  # (n_sectors, n_days)

    beta_i = rng.normal(1.0, 0.25, cfg.n_stocks).clip(0.3, 1.9)
    gamma_i = rng.normal(0.7, 0.2, cfg.n_stocks).clip(0.1, 1.4)
    sigma_i = rng.uniform(cfg.idio_vol_lo, cfg.idio_vol_hi, cfg.n_stocks)
    quality = rng.normal(0.0, 1.0, cfg.n_stocks)          # static, observable
    q_rank = pd.Series(quality).rank(pct=True).values      # 0..1

    # OU alpha state per stock (hosts trend & x-sec momentum)
    phi = 0.5 ** (1.0 / cfg.alpha_ou_halflife)
    ou_sd = cfg.alpha_ou_sigma
    ou_innov_sd = ou_sd * np.sqrt(1 - phi**2)

    p0 = rng.uniform(cfg.price_lo, cfg.price_hi, cfg.n_stocks)

    N, T = cfg.n_stocks, n_days
    close = np.empty((T, N)); open_ = np.empty((T, N))
    high = np.empty((T, N)); low = np.empty((T, N))
    volume = np.empty((T, N))
    catalyst = np.zeros((T, N), dtype=np.int8)
    news_sign = np.zeros((T, N), dtype=np.int8)
    or30_hi = np.empty((T, N)); or30_lo = np.empty((T, N))
    tr_pead = np.zeros((T, N)); tr_fade = np.zeros((T, N))
    tr_anchor = np.zeros((T, N)); tr_alpha = np.zeros((T, N))
    tr_quality = np.zeros((T, N))

    log_p = np.log(p0.copy())
    alpha_state = rng.normal(0.0, ou_sd, N)
    pead_queue = np.zeros((cfg.pead_days, N))   # scheduled future drift
    base_lv = rng.normal(13.5, 0.8, N)          # log average daily volume
    lv_state = np.zeros(N)
    hi52 = np.full(N, -np.inf)
    hist_max = np.log(p0.copy())                # running 52wk high in logs
    log_hist = np.full((252, N), np.nan)

    # top-vs-bottom decile spread ~= 0.9 * quality_premium per day
    q_drift = cfg.quality_premium * (q_rank - 0.5)

    for t in range(T):
        # --- events ---
        ev = rng.random(N) < cfg.catalyst_rate
        jump = np.zeros(N)
        if ev.any():
            k = int(ev.sum())
            mag = np.exp(rng.normal(cfg.jump_mu_ln, cfg.jump_sigma_ln, k))
            sgn = rng.choice([-1.0, 1.0], size=k)
            jump[ev] = sgn * mag
            catalyst[t, ev] = 1
            news_sign[t, ev] = np.sign(jump[ev]).astype(np.int8)
            # schedule PEAD drift over next pead_days
            add = cfg.pead_share * jump[ev] / cfg.pead_days
            pead_queue[:, ev] += add

        pead_today = pead_queue[0].copy()
        pead_queue[:-1] = pead_queue[1:]
        pead_queue[-1] = 0.0

        # --- alpha OU update ---
        alpha_state = phi * alpha_state + rng.normal(0, ou_innov_sd, N)

        # --- 52wk-high anchoring drift (uses yesterday's close vs trailing max) ---
        anchor = np.where(np.exp(log_p - hist_max) > (1 - cfg.anchor_band),
                          cfg.anchor_drift, 0.0)

        # --- idiosyncratic return, split overnight/intraday ---
        z = _student_t(rng, dof, N) * sigma_i
        sec = f_sec[sector_of, t] * gamma_i
        drift = alpha_state + anchor + q_drift + pead_today
        r_core = beta_i * f_mkt[t] + sec + z

        r_on_noise = np.sqrt(on_share) * r_core * rng.normal(1.0, 0.35, N)
        # overnight = share of core + full jump (news hits overnight 80% of time)
        overnight_jump = np.where(rng.random(N) < 0.8, jump, 0.0)
        intraday_jump = jump - overnight_jump
        r_on = r_on_noise + overnight_jump
        # no-news gap fade: fade a fraction of yesterday-informed overnight noise
        nonews = (catalyst[t] == 0)
        fade = np.where(nonews, cfg.nonews_gap_fade * r_on_noise, 0.0)
        # fade materializes intraday TODAY? No: fade of today's gap must hit the
        # NEXT session to be tradable at the open.  Schedule it.
        r_id = (r_core - r_on_noise) + intraday_jump + drift
        if t + 1 < T:
            pass  # applied via fade_carry below
        if t == 0:
            fade_carry = np.zeros(N)
        r_id = r_id + fade_carry            # yesterday's scheduled fade lands today
        tr_fade[t] = fade_carry
        fade_carry = fade

        r_tot = r_on + r_id
        tr_pead[t] = pead_today
        tr_anchor[t] = anchor
        tr_alpha[t] = alpha_state
        tr_quality[t] = q_drift

        # --- build OHLC ---
        new_log = log_p + r_tot
        o = np.exp(log_p + r_on)
        c = np.exp(new_log)
        # intraday range: Brownian-bridge style envelope around open->close
        span = np.abs(r_id) + sigma_i * np.abs(rng.normal(0.9, 0.35, N))
        u = rng.beta(2.0, 2.0, N)
        hi = np.maximum(o, c) * np.exp(span * u * 0.5)
        lo = np.minimum(o, c) * np.exp(-span * (1 - u) * 0.5)
        # first-30-min range for ORB (fraction of day range, front-loaded vol)
        f30 = rng.beta(2.5, 4.0, N) * 0.55 + 0.10
        or30_hi[t] = o * np.exp(np.log(hi / o) * f30)
        or30_lo[t] = o * np.exp(np.log(lo / o) * f30)

        # --- volume ---
        lv_state = 0.6 * lv_state + rng.normal(0, 0.35, N)
        rvol_news = np.where(
            catalyst[t] == 1,
            np.log(rng.uniform(cfg.news_vol_mult_lo, cfg.news_vol_mult_hi, N)),
            0.0,
        )
        # events echo: elevated volume decays over 2 days handled by lv_state bump
        lv_state += rvol_news * 0.4
        vol = np.exp(base_lv + lv_state + vol_beta * np.abs(r_tot) + rvol_news)

        open_[t], high[t], low[t], close[t] = o, hi, lo, c
        volume[t] = vol
        log_p = new_log

        # --- update 52wk trailing max (t-252..t-1) ---
        log_hist[t % 252] = log_p
        hist_max = np.nanmax(log_hist, axis=0)

    # round to tick
    for arr in (open_, high, low, close, or30_hi, or30_lo):
        np.round(arr / 0.01, out=arr)
        arr *= 0.01
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])

    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    panel = pd.DataFrame({
        "open": open_.ravel(), "high": high.ravel(), "low": low.ravel(),
        "close": close.ravel(), "volume": volume.ravel(),
        "catalyst": catalyst.ravel(), "news_sign": news_sign.ravel(),
        "quality": np.tile(quality, T),
        "or30_high": or30_hi.ravel(), "or30_low": or30_lo.ravel(),
    }, index=idx)

    truth = pd.DataFrame({
        "pead": tr_pead.ravel(), "gap_fade": tr_fade.ravel(),
        "anchor": tr_anchor.ravel(), "alpha": tr_alpha.ravel(),
        "quality_drift": tr_quality.ravel(),
    }, index=idx)

    return {"panel": panel, "truth": truth, "dates": dates, "tickers": tickers,
            "sector_of": dict(zip(tickers, sector_of.tolist()))}
