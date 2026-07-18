"""The random day trader study: hop into random stocks, sell by the close.

LEARN (evidence this backtest is designed to reproduce/measure):
  * Barber-Lee-Liu-Odean (Taiwan, 2014): ~80% of day traders lose; <1%
    predictably profitable.
  * Chague-De-Losso-Giovannetti (Brazil, 2020): 97% of persistent futures
    day traders lost over 300+ sessions.
  * Barber-Odean (2000): net returns fall monotonically with turnover.
  * Lou-Polk-Skouras (2019): the equity premium accrues OVERNIGHT; the
    intraday segment a day trader holds carries ~zero drift.

BACKTEST.  On the engine's panel (500 names, 5y test window):
  1. 2,000 simulated trader-years: each day pick k random stocks, buy at
     the open, sell at the close, pay retail costs.  Distribution of
     annual outcomes, P(profitable year).
  2. Same, but with the day-trader ATTENTION BIAS: picks drawn from the
     top-RVOL decile ("hot movers") instead of uniformly.
  3. Random ENTRY TIME variant: entry uniformly inside the day (half the
     intraday exposure on average) -- the "hopping in whenever" version.
  4. The turnover treadmill: gross vs net per trades-per-day.
  5. Required skill: per-trade hit rate needed to average +1%/day at k
     trades/day and given costs.

DRIFT-PLACEMENT CAVEAT (disclosed, quantified): this simulator puts the
market factor's drift intraday (an artifact of calibrating on index data
with stale opens).  Real markets put it overnight.  We therefore also
report an OVERNIGHT-ADJUSTED variant where the market factor's mean is
removed from the intraday leg -- that variant is the realistic one; the
raw variant is the day-trader-friendly upper bound.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..config import RunConfig
from ..patterns.base import Wide

RETAIL_RT_BPS = 5.0     # commission-free retail: effective spread ~2.5bp/side
ACTIVE_RT_BPS = 15.0    # the engine's institutional base for comparison


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--n-traders", type=int, default=2000)
    ap.add_argument("--k", type=int, default=3, help="stocks per day")
    args = ap.parse_args(argv)
    cfg = RunConfig()

    print("[daytrader] building panel...")
    from ..data.calibration import calibrate
    from ..data.synthetic import simulate
    calib = calibrate(verbose=False)
    data = simulate(cfg.sim, calib)
    panel = data["panel"]
    w = Wide(panel)
    test_start = data["dates"][-1] - pd.DateOffset(years=cfg.test_years)

    oc = (w.close / w.open - 1)
    keep = oc.index >= test_start
    oc = oc[keep]
    rvol = w.rvol[keep]
    on = (w.open / w.close.shift(1) - 1)[keep]

    ocv = oc.values
    T, N = ocv.shape
    mkt_mu_intraday = float(np.nanmean(ocv))       # the artifact drift
    print(f"[daytrader] sim intraday universe drift {1e4*mkt_mu_intraday:+.1f} "
          f"bp/day; overnight {1e4*float(np.nanmean(on.values)):+.1f} bp/day "
          "(real markets: intraday ~0, overnight positive)")

    rng = np.random.default_rng(42)
    hot = (rvol.rank(axis=1, pct=True) >= 0.9).values

    def trader_years(pick_hot: bool, adj_overnight: bool, entry_random: bool,
                     rt_bps: float, n_traders: int, days: int = 252):
        rt = rt_bps / 1e4
        out = np.zeros(n_traders)
        yr_starts = rng.integers(0, T - days, n_traders)
        adj = mkt_mu_intraday if adj_overnight else 0.0
        for i in range(n_traders):
            s = yr_starts[i]
            pnl = 0.0
            for t in range(s, s + days):
                row = ocv[t]
                if pick_hot:
                    cand = np.where(hot[t] & np.isfinite(row))[0]
                else:
                    cand = np.where(np.isfinite(row))[0]
                if len(cand) < args.k:
                    continue
                picks = rng.choice(cand, args.k, replace=False)
                r = row[picks] - adj
                if entry_random:
                    r = r * rng.uniform(0, 1, args.k)   # entered mid-move
                pnl += float(np.mean(r)) - rt
            out[i] = pnl
        return out

    print(f"[daytrader] simulating {args.n_traders} trader-years x 4 variants...")
    variants = {
        "random, sim-world (upper bound)": trader_years(
            False, False, False, RETAIL_RT_BPS, args.n_traders),
        "random, overnight-adjusted (realistic)": trader_years(
            False, True, False, RETAIL_RT_BPS, args.n_traders),
        "hot movers (attention bias), adj": trader_years(
            True, True, False, RETAIL_RT_BPS, args.n_traders),
        "random entry time, adj": trader_years(
            False, True, True, RETAIL_RT_BPS, args.n_traders),
    }
    rows = []
    for name, v in variants.items():
        rows.append({
            "variant": name,
            "mean_annual_pct": 100 * v.mean(),
            "median_annual_pct": 100 * np.median(v),
            "p_profitable_year": float((v > 0).mean()),
            "p5_pct": 100 * np.percentile(v, 5),
            "p95_pct": 100 * np.percentile(v, 95),
            "daily_bps": 1e4 * v.mean() / 252,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{args.outdir}/daytrader_summary.csv", index=False)
    print(summary.round(2).to_string(index=False))
    np.save(f"{args.outdir}/daytrader_dists.npy", variants, allow_pickle=True)

    # ---- turnover treadmill: k trades/day, gross vs net ----
    tread = []
    for k in (1, 2, 3, 5, 10, 20):
        gross_bps = 1e4 * 0.0                     # adj: gross ~ 0 realistic
        for rt_name, rt in (("retail 5bp", RETAIL_RT_BPS),
                            ("active 15bp", ACTIVE_RT_BPS)):
            tread.append({"trades_per_day": k, "costs": rt_name,
                          "net_daily_bps": gross_bps - rt * 1.0,
                          "net_annual_pct": 100 * (gross_bps - rt) * 252 / 1e4})
    # note: with ~zero gross drift, net/day = -(round trips/day * rt); the
    # table shows one full round trip of exposure per day regardless of k
    # (k parallel positions share the day's capital; cost scales with k
    # only if positions are cycled) -- cycled version:
    tread = []
    for k in (1, 2, 3, 5, 10, 20):
        for rt_name, rt in (("retail 5bp", RETAIL_RT_BPS),
                            ("active 15bp", ACTIVE_RT_BPS)):
            tread.append({"cycles_per_day": k, "costs": rt_name,
                          "net_daily_bps": -k * rt,
                          "net_annual_pct": 100 * (-k * rt) * 252 / 1e4})
    pd.DataFrame(tread).to_csv(f"{args.outdir}/daytrader_treadmill.csv",
                               index=False)

    # ---- required skill for +1%/day ----
    sd_oc = float(np.nanstd(ocv))
    req = []
    for k in (1, 2, 3, 5, 10, 20):
        for rt_name, rt in (("retail 5bp", RETAIL_RT_BPS / 1e4),
                            ("active 15bp", ACTIVE_RT_BPS / 1e4)):
            need_gross = 0.01 + k * rt            # per day, cycled k times
            per_trade = need_gross / k
            # directional coin on a move of typical size sd_oc:
            # E[r] = (2p-1)*E|move| ; E|move| ~ sd*sqrt(2/pi)
            e_abs = sd_oc * np.sqrt(2 / np.pi)
            p_needed = 0.5 + per_trade / (2 * e_abs)
            req.append({"cycles_per_day": k, "costs": rt_name,
                        "per_trade_edge_bps": 1e4 * per_trade,
                        "hit_rate_needed": p_needed})
    req = pd.DataFrame(req)
    req.to_csv(f"{args.outdir}/daytrader_required_skill.csv", index=False)
    print("\n", req.round(4).to_string(index=False))
    return {"summary": summary, "dists": variants, "required": req,
            "sd_oc": sd_oc}


if __name__ == "__main__":
    main()
