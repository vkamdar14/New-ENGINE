"""Phase-2 visual deliverables.

Palette/chrome follow the validated reference dataviz palette (light mode):
series blue #2a78d6, comparison violet #4a3aa7, critical red #d03b3b for
the target line, green #008300 / red #e34948 for up/down candles, ink
#0b0b0b, muted #898781, hairline grid #e1e0d9, surface #fcfcfb.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#0b0b0b"; MUTED = "#898781"; GRID = "#e1e0d9"; SURFACE = "#fcfcfb"
BLUE = "#2a78d6"; VIOLET = "#4a3aa7"; CRIT = "#d03b3b"
UP = "#008300"; DN = "#e34948"; YELLOW = "#eda100"; AQUA = "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] wrote {path}")


def equity_curve(net: pd.Series, linear_net: pd.Series, outdir: str,
                 title_suffix: str = ""):
    eq = (1 + net).cumprod()
    eq_lin = (1 + linear_net).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eq.index, eq.values, color=BLUE, lw=2, label="Fused meta-model (net)")
    ax.plot(eq_lin.index, eq_lin.values, color=VIOLET, lw=2, alpha=0.85,
            label="Linear z-sum baseline (net)")
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_title(f"Contiguous 5-year equity curve, after costs{title_suffix}",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    # direct label at the end
    ax.annotate(f"  {eq.iloc[-1]:,.2f}x", (eq.index[-1], eq.iloc[-1]),
                color=BLUE, fontweight="bold", va="center")
    _save(fig, os.path.join(outdir, "equity_curve.png"))


def drawdown_chart(net: pd.Series, outdir: str):
    eq = (1 + net).cumprod()
    dd = eq / eq.cummax() - 1
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.fill_between(dd.index, dd.values * 100, 0, color=DN, alpha=0.35, lw=0)
    ax.plot(dd.index, dd.values * 100, color=DN, lw=1.2)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown, fused strategy (net)", loc="left", fontweight="bold")
    worst = dd.min()
    ax.annotate(f"max DD {worst * 100:.1f}%", (dd.idxmin(), worst * 100),
                color=INK, fontsize=9, xytext=(10, -12),
                textcoords="offset points")
    _save(fig, os.path.join(outdir, "drawdown.png"))


def bootstrap_hist(draws: np.ndarray, achieved: float, target: float,
                   outdir: str):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bps = draws * 1e4
    ax.hist(bps, bins=60, color=BLUE, alpha=0.9, edgecolor=SURFACE, lw=0.4)
    ax.axvline(achieved * 1e4, color=INK, lw=1.5, ls="--")
    ax.annotate(f"realized mean {achieved * 1e4:.1f} bp/day",
                (achieved * 1e4, ax.get_ylim()[1] * 0.92), color=INK,
                fontsize=9, ha="left", xytext=(6, 0), textcoords="offset points")
    ax.axvline(target * 1e4, color=CRIT, lw=2)
    ax.annotate(f"TARGET {target * 1e4:.0f} bp/day (1%)",
                (target * 1e4, ax.get_ylim()[1] * 0.45), color=CRIT,
                fontweight="bold", fontsize=10, ha="right", rotation=90,
                xytext=(-8, 0), textcoords="offset points")
    ax.set_xlabel("Average daily net return per bootstrap draw (basis points)")
    ax.set_ylabel("Draws")
    ax.set_title("1,000 random-draw averages (block bootstrap of daily net "
                 "returns) vs the 1%/day target", loc="left", fontweight="bold")
    _save(fig, os.path.join(outdir, "bootstrap_hist.png"))


def attribution_bars(family_daily: dict[str, pd.Series], outdir: str):
    rows = []
    for fam, s in family_daily.items():
        active = s[s != 0]
        rows.append((fam, 1e4 * s.mean(), 1e4 * (active.mean() if len(active)
                                                 else 0), len(active)))
    df = pd.DataFrame(rows, columns=["family", "bps_all", "bps_active",
                                     "active_days"]).sort_values("bps_all")
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(df) + 1.5))
    colors = [UP if v > 0 else DN for v in df["bps_all"]]
    ax.barh(df["family"], df["bps_all"], color=colors, height=0.62)
    ax.axvline(0, color="#c3c2b7", lw=1)
    lo, hi = df["bps_all"].min(), df["bps_all"].max()
    span = max(hi - lo, 1.0)
    ax.set_xlim(lo - 0.35 * span, hi + 0.35 * span)
    for i, (v, n) in enumerate(zip(df["bps_all"], df["active_days"])):
        ax.annotate(f" {v:+.1f} bp  (n={n})", (v, i), va="center", fontsize=8.5,
                    ha="left" if v >= 0 else "right", color=INK,
                    xytext=(4 if v >= 0 else -4, 0), textcoords="offset points")
    ax.set_xlabel("Average daily net contribution (basis points, "
                  "standalone equal-weight family portfolio)")
    ax.set_title("Per-edge attribution, after costs", loc="left",
                 fontweight="bold")
    _save(fig, os.path.join(outdir, "attribution.png"))


def _candles(ax, seg: pd.DataFrame):
    x = np.arange(len(seg))
    up = seg["close"] >= seg["open"]
    ax.vlines(x, seg["low"], seg["high"], color=MUTED, lw=0.8, zorder=2)
    ax.bar(x[up], (seg["close"] - seg["open"])[up], bottom=seg["open"][up],
           width=0.62, color=UP, zorder=3)
    ax.bar(x[~up], (seg["open"] - seg["close"])[~up], bottom=seg["close"][~up],
           width=0.62, color=DN, zorder=3)
    return x


def sample_trade_chart(panel: pd.DataFrame, ticker: str,
                       t_signal: pd.Timestamp, t_entry: pd.Timestamp,
                       t_exit: pd.Timestamp, signal_name: str, direction: int,
                       entry_px: float, exit_px: float, had_news: bool,
                       outdir: str, tag: str):
    df = panel.xs(ticker, level="ticker")
    dts = df.index
    i_sig = dts.get_loc(t_signal)
    lo = max(0, i_sig - 40)
    hi = min(len(dts), dts.get_loc(t_exit) + 12)
    seg = df.iloc[lo:hi]
    x = np.arange(len(seg))
    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.06})
    _candles(ax, seg)

    def xi(ts):
        return seg.index.get_loc(ts)

    ax.axvline(xi(t_signal), color=YELLOW, lw=1.2, ls=":", zorder=1)
    ax.annotate(f"signal: {signal_name}", (xi(t_signal), seg["high"].max()),
                color=INK, fontsize=9, fontweight="bold", ha="center",
                xytext=(0, 10), textcoords="offset points")
    ax.scatter([xi(t_entry)], [entry_px], marker="^" if direction > 0 else "v",
               s=130, color=BLUE, zorder=5, edgecolor=SURFACE, lw=1)
    ax.annotate(f"entry {entry_px:.2f}", (xi(t_entry), entry_px), color=BLUE,
                fontsize=9, xytext=(6, -14), textcoords="offset points")
    ax.scatter([xi(t_exit)], [exit_px], marker="x", s=110, color=VIOLET,
               zorder=5, lw=2.5)
    ax.annotate(f"exit {exit_px:.2f}", (xi(t_exit), exit_px), color=VIOLET,
                fontsize=9, xytext=(6, 8), textcoords="offset points")
    news_days = seg.index[seg["catalyst"] == 1]
    for nd in news_days:
        ax.annotate("news", (xi(nd), seg.loc[nd, "low"]), color=CRIT,
                    fontsize=8, fontweight="bold", ha="center",
                    xytext=(0, -16), textcoords="offset points")
        ax.scatter([xi(nd)], [seg.loc[nd, "low"]], marker="D", s=36,
                   color=CRIT, zorder=5)
    ret = direction * (exit_px / entry_px - 1)
    side = "LONG" if direction > 0 else "SHORT"
    ax.set_title(f"{ticker}  {side}  {signal_name}   "
                 f"{'with news catalyst' if had_news else 'no news'}   "
                 f"P&L {ret * 100:+.2f}%", loc="left", fontweight="bold")
    axv.bar(x, seg["volume"], width=0.62, color=BLUE, alpha=0.55)
    axv.set_ylabel("Volume", fontsize=8)
    axv.grid(False)
    step = max(1, len(seg) // 8)
    axv.set_xticks(x[::step])
    axv.set_xticklabels([d.strftime("%Y-%m-%d") for d in seg.index[::step]],
                        rotation=30, ha="right", fontsize=8)
    _save(fig, os.path.join(outdir, "sample_trades", f"{tag}.png"))


def interaction_heat(table: pd.DataFrame, outdir: str):
    """News x size continuation heat-map (next-day drift bps, all-volume)."""
    t = table[table["rvol"] == "all"]
    piv = t.pivot_table(index="news", columns="size", values="nextday_drift_bps")
    piv = piv.reindex(index=[True, False],
                      columns=["small", "mid", "large", "huge"])
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    v = np.nanmax(np.abs(piv.values)) or 1.0
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(4), ["small\n0.5-1.5%", "mid\n1.5-3%",
                             "large\n3-7%", "huge\n>7%"])
    ax.set_yticks([0, 1], ["news catalyst", "no news"])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            if np.isfinite(val):
                ax.annotate(f"{val:+.0f}", (j, i), ha="center", va="center",
                            color=INK, fontsize=10, fontweight="bold")
    ax.set_title("Gap next-day drift in gap direction (bp) -- news vs no news",
                 loc="left", fontweight="bold")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="bp")
    _save(fig, os.path.join(outdir, "interaction_heatmap.png"))
