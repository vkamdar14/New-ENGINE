"""End-to-end run: data -> detectors -> stats -> interaction -> fusion ->
visuals -> REPORT.md.

  python -m engine.run_all                 # synthetic, calibrated on real SP500
  python -m engine.run_all --source real   # real data (network required)
  python -m engine.run_all --fast          # smaller universe for smoke tests
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .analysis.interaction import breakout_interaction_table, gap_interaction_table
from .backtest.engine import Forward, attach_forward, family_daily_returns
from .backtest.stats import (bootstrap_avg_daily, decay_by_year,
                             ground_truth_recovery, newey_west_t,
                             per_signal_stats)
from .config import RunConfig
from .fusion.meta import build_features, walk_forward_fusion
from .patterns import ALL_DETECTORS
from .patterns.base import Wide
from .patterns.ml_cnn import detect_ml
from .report import visuals


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic",
                    choices=["synthetic", "real"])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-ml", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    cfg = RunConfig()
    cfg.outdir = args.outdir
    os.makedirs(cfg.outdir, exist_ok=True)

    # ---------- data ----------
    if args.source == "synthetic":
        from .data.calibration import calibrate
        from .data.synthetic import simulate
        log("calibrating simulator on real S&P 500 (arch package)...")
        calib = calibrate()
        cfg.sim.seed = args.seed
        if args.fast:
            cfg.sim.n_stocks = 60
            cfg.sim.years = 3.0
        log(f"simulating {cfg.sim.n_stocks} stocks x {cfg.sim.years} years...")
        data = simulate(cfg.sim, calib)
    else:
        from .data.loaders import load_real_panel
        tickers = open("tickers.txt").read().split()
        data = load_real_panel(tickers, "2021-07-01", "2026-07-17")
        calib = None
    panel, truth = data["panel"], data["truth"]
    w = Wide(panel)
    fwd = Forward(w.open, w.close, cfg.horizons)
    log(f"panel: {panel.shape[0]:,} rows, "
        f"{len(data['tickers'])} tickers, {len(data['dates'])} days")

    # ---------- detectors ----------
    all_sigs = []
    for name, det in ALL_DETECTORS.items():
        t0 = time.time()
        s = det(panel, None)
        log(f"detector {name:>13s}: {len(s):7,} signals "
            f"({time.time() - t0:5.1f}s)")
        all_sigs.append(s)
    if not args.no_ml:
        t0 = time.time()
        s = detect_ml(panel, seed=args.seed)
        log(f"detector {'ml':>13s}: {len(s):7,} signals "
            f"({time.time() - t0:5.1f}s)")
        all_sigs.append(s)
    signals = pd.concat(all_sigs, ignore_index=True)
    signals["date"] = pd.to_datetime(signals["date"])

    # restrict scoring to the contiguous test window (last test_years)
    test_start = data["dates"][-1] - pd.DateOffset(years=cfg.test_years)
    in_test = signals["date"] >= test_start
    log(f"signals total {len(signals):,}; in 5y test window {in_test.sum():,} "
        f"(window starts {test_start.date()})")

    # ---------- event-study stats ----------
    sig_fwd = attach_forward(signals[in_test], fwd)
    stats = per_signal_stats(sig_fwd, cfg.costs, cfg.horizons)
    stats.to_csv(f"{cfg.outdir}/per_signal_stats.csv", index=False)
    decay = decay_by_year(sig_fwd)
    decay.to_csv(f"{cfg.outdir}/decay_by_year.csv")
    log(f"event-study stats written ({len(stats)} signals)")

    if truth is not None:
        rec = ground_truth_recovery(signals[in_test], truth, panel.index)
        rec.to_csv(f"{cfg.outdir}/ground_truth_recovery.csv", index=False)

    # ---------- per-family standalone portfolios ----------
    family_daily = {}
    for fam, g in sig_fwd.groupby("family"):
        family_daily[fam] = family_daily_returns(g, fwd, cfg.costs).loc[
            lambda s: s.index >= test_start]
    fam_summary = {f: {"avg_daily_bps": 1e4 * s.mean(),
                       "ann_sharpe": float(np.sqrt(252) * s.mean()
                                           / (s.std() + 1e-12)),
                       "active_days": int((s != 0).sum())}
                   for f, s in family_daily.items()}

    # ---------- interaction study ----------
    log("running news x charts interaction study...")
    gap_tab = gap_interaction_table(panel[panel.index.get_level_values(0)
                                          >= test_start])
    brk_tab = breakout_interaction_table(panel[panel.index.get_level_values(0)
                                               >= test_start])
    gap_tab.to_csv(f"{cfg.outdir}/interaction_gaps.csv", index=False)
    brk_tab.to_csv(f"{cfg.outdir}/interaction_breakouts.csv", index=False)

    # ---------- fusion ----------
    log("building fusion features + walk-forward meta-model...")
    X = build_features(signals, w)
    fusion = walk_forward_fusion(X, fwd, cfg.costs, top_k=cfg.top_k)
    net = fusion["net"].loc[lambda s: s.index >= test_start]
    gross = fusion["gross"].loc[lambda s: s.index >= test_start]
    lin_net = fusion["linear_net"].loc[lambda s: s.index >= test_start]
    achieved = float(net.mean())
    log(f"FUSED NET avg daily: {achieved * 1e4:.1f} bp "
        f"(target {cfg.target_daily * 1e4:.0f} bp)")

    draws = bootstrap_avg_daily(net, cfg.n_bootstrap)

    # ---------- visuals ----------
    log("rendering visual deliverables...")
    visuals.equity_curve(net, lin_net, cfg.outdir)
    visuals.drawdown_chart(net, cfg.outdir)
    visuals.bootstrap_hist(draws, achieved, cfg.target_daily, cfg.outdir)
    visuals.attribution_bars(family_daily, cfg.outdir)
    visuals.interaction_heat(gap_tab, cfg.outdir)
    n_charts = render_sample_trades(panel, sig_fwd, fwd, cfg)

    # ---------- summary json ----------
    summary = {
        "source": args.source,
        "calibration": calib,
        "n_signals_test": int(in_test.sum()),
        "families": fam_summary,
        "fused": {
            "avg_daily_net_bps": achieved * 1e4,
            "avg_daily_gross_bps": 1e4 * float(gross.mean()),
            "ann_sharpe_net": float(np.sqrt(252) * net.mean()
                                    / (net.std() + 1e-12)),
            "nw_tstat": newey_west_t(net.values),
            "linear_baseline_bps": 1e4 * float(lin_net.mean()),
            "target_daily_bps": cfg.target_daily * 1e4,
            "target_hit": bool(achieved >= cfg.target_daily),
            "p_draws_above_target": float((draws >= cfg.target_daily).mean()),
            "bootstrap_p5_bps": 1e4 * float(np.percentile(draws, 5)),
            "bootstrap_p95_bps": 1e4 * float(np.percentile(draws, 95)),
        },
        "importances": (fusion["importances"].sort_values(ascending=False)
                        .head(15).to_dict()
                        if fusion["importances"] is not None else None),
        "sample_trade_charts": n_charts,
    }
    with open(f"{cfg.outdir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"summary.json written; target_hit={summary['fused']['target_hit']}")
    return summary


def render_sample_trades(panel, sig_fwd, fwd, cfg, n=8):
    """Pick 5-10 representative trades across families (mix of winners and
    losers, news and no-news) and render annotated charts."""
    pos_d = {d: i for i, d in enumerate(fwd.dates)}
    cat_w = panel["catalyst"].unstack("ticker")
    picks = []
    prefer = ["gaps", "lmw", "breakouts", "anchors", "ml", "novel", "candles",
              "trend"]
    for fam in prefer:
        g = sig_fwd[(sig_fwd["family"] == fam)
                    & sig_fwd["ret_5"].notna()].sort_values("strength",
                                                            ascending=False)
        if not len(g):
            continue
        win = g[g["ret_5"] > 0].head(1)
        lose = g[g["ret_5"] < 0].head(1)
        picks.append(win)
        if len(picks) < n:
            picks.append(lose)
        if sum(len(p) for p in picks) >= n:
            break
    picks = pd.concat(picks).drop_duplicates(
        subset=["date", "ticker", "signal"]).head(n)
    count = 0
    for _, r in picks.iterrows():
        t0 = pos_d[r["date"]]
        same_close = str(r["signal"]).startswith(("gap_", "orb_"))
        h = int(r["horizon"]) if int(r["horizon"]) in fwd.horizons else 5
        t_entry = fwd.dates[min(t0 + (0 if same_close else 1),
                                len(fwd.dates) - 1)]
        t_exit = fwd.dates[min(t0 + h, len(fwd.dates) - 1)]
        df_t = panel.xs(r["ticker"], level="ticker")
        entry_px = (df_t.loc[t_entry, "close"] if same_close
                    else df_t.loc[t_entry, "open"])
        exit_px = df_t.loc[t_exit, "close"]
        had_news = bool(cat_w.loc[r["date"], r["ticker"]] == 1)
        tag = (f"{count:02d}_{r['family']}_{r['signal']}_{r['ticker']}"
               f"_{pd.Timestamp(r['date']).date()}")
        visuals.sample_trade_chart(
            panel, r["ticker"], r["date"], t_entry, t_exit, r["signal"],
            int(r["direction"]), float(entry_px), float(exit_px), had_news,
            cfg.outdir, tag)
        count += 1
    return count


if __name__ == "__main__":
    main()
