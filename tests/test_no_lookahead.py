"""Anti-lookahead and correctness tests.

The core test: truncation invariance.  Any signal stamped `date` must be
identical whether the detector saw data only up to `date` or the full
sample.  If a detector peeks past its stamp, truncating the panel changes
(or removes) signals stamped inside the kept window.
"""

import numpy as np
import pandas as pd
import pytest

from engine.config import CostModel, SimConfig
from engine.data.synthetic import simulate
from engine.patterns import ALL_DETECTORS
from engine.backtest.engine import Forward, attach_forward, family_daily_returns
from engine.backtest.stats import newey_west_t


@pytest.fixture(scope="module")
def data():
    cfg = SimConfig()
    cfg.n_stocks = 25
    cfg.years = 2.2
    cfg.seed = 123
    return simulate(cfg, None)


def _cut(panel: pd.DataFrame, n_last: int) -> pd.DataFrame:
    dates = panel.index.get_level_values("date").unique().sort_values()
    keep = dates[:-n_last]
    return panel[panel.index.get_level_values("date").isin(keep)]


@pytest.mark.parametrize("name", sorted(ALL_DETECTORS))
def test_truncation_invariance(data, name):
    panel = data["panel"]
    full = ALL_DETECTORS[name](panel, None)
    cut_panel = _cut(panel, 30)
    cut_dates = cut_panel.index.get_level_values("date").unique()
    part = ALL_DETECTORS[name](cut_panel, None)
    # compare signals stamped strictly inside the truncated window, minus a
    # small right edge (rolling min_periods effects only ever ADD signals at
    # the edge, never change interior ones)
    edge = cut_dates.sort_values()[-1]
    key = ["date", "ticker", "signal", "direction", "strength"]
    f = (full[full["date"] < edge].sort_values(key)[key]
         .reset_index(drop=True))
    p = (part[part["date"] < edge].sort_values(key)[key]
         .reset_index(drop=True))
    f["strength"] = f["strength"].round(9)
    p["strength"] = p["strength"].round(9)
    pd.testing.assert_frame_equal(f, p)


def test_forward_returns_are_forward(data):
    panel = data["panel"]
    from engine.patterns.base import Wide
    w = Wide(panel)
    fwd = Forward(w.open, w.close, (1, 5))
    dates, tickers = fwd.dates, fwd.tickers
    t0, j = 100, 3
    o = w.open.values, 0
    exp = w.close.values[t0 + 1, j] / w.open.values[t0 + 1, j] - 1
    assert np.isclose(fwd.fwd_no[1][t0, j], exp)
    exp5 = w.close.values[t0 + 5, j] / w.open.values[t0 + 1, j] - 1
    assert np.isclose(fwd.fwd_no[5][t0, j], exp5)
    # same-close entry
    exp_sc = w.close.values[t0 + 1, j] / w.close.values[t0, j] - 1
    assert np.isclose(fwd.fwd_sc[1][t0, j], exp_sc)


def test_costs_reduce_returns(data):
    panel = data["panel"]
    from engine.patterns.base import Wide
    w = Wide(panel)
    fwd = Forward(w.open, w.close, (1, 2, 3, 5, 10, 20))
    sig = pd.DataFrame({
        "date": [fwd.dates[50], fwd.dates[60]],
        "ticker": [fwd.tickers[0], fwd.tickers[1]],
        "family": "x", "signal": ["donchian20_up", "donchian20_up"],
        "direction": [1, -1], "strength": 1.0, "horizon": 5})
    free = family_daily_returns(sig, fwd, CostModel(0, 0, 0))
    paid = family_daily_returns(sig, fwd, CostModel(5, 5, 1))
    assert paid.sum() < free.sum()
    diff = free.sum() - paid.sum()
    assert np.isclose(diff, 2 * 2 * 11 / 1e4, atol=1e-10)


def test_newey_west_reasonable():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 2000)
    t = newey_west_t(x)
    assert 2 < t < 8


def test_attach_forward_direction_sign(data):
    panel = data["panel"]
    from engine.patterns.base import Wide
    w = Wide(panel)
    fwd = Forward(w.open, w.close, (1,))
    sig = pd.DataFrame({
        "date": [fwd.dates[50]], "ticker": [fwd.tickers[0]],
        "family": "x", "signal": ["s"], "direction": [-1],
        "strength": [1.0], "horizon": [1]})
    out = attach_forward(sig, fwd)
    raw = fwd.fwd_no[1][50, 0]
    assert np.isclose(out["ret_1"].iloc[0], -raw)


def test_fusion_walk_forward_no_lookahead(data):
    """GBM scores for month m must be unchanged when later months' data are
    deleted -- i.e., no future row influences a past prediction."""
    from engine.backtest.engine import Forward
    from engine.fusion.meta import build_features, walk_forward_fusion
    from engine.patterns import ALL_DETECTORS
    from engine.patterns.base import Wide
    from engine.config import CostModel

    panel = data["panel"]
    sigs = pd.concat([ALL_DETECTORS[k](panel, None)
                      for k in ("gaps", "trend", "anchors")],
                     ignore_index=True)
    w = Wide(panel)
    fwd = Forward(w.open, w.close, (1, 2, 3, 5, 10, 20))
    X = build_features(sigs, w)
    full = walk_forward_fusion(X, fwd, CostModel(), top_k=3,
                               min_train_days=40)["scores"]

    dates = panel.index.get_level_values("date").unique().sort_values()
    cut = dates[-45]
    panel2 = panel[panel.index.get_level_values("date") < cut]
    sigs2 = sigs[sigs["date"] < cut]
    w2 = Wide(panel2)
    fwd2 = Forward(w2.open, w2.close, (1, 2, 3, 5, 10, 20))
    X2 = build_features(sigs2, w2)
    part = walk_forward_fusion(X2, fwd2, CostModel(), top_k=3,
                               min_train_days=40)["scores"]

    # compare on months fully inside both runs, excluding the boundary month
    last_full_month = (pd.Period(cut, freq="M") - 1)
    keep = [d for d in part.index.get_level_values("date").unique()
            if pd.Period(d, freq="M") < last_full_month]
    a = full[full.index.get_level_values("date").isin(keep)]
    b = part[part.index.get_level_values("date").isin(keep)]
    common = a.index.intersection(b.index)
    assert len(common) > 100
    assert np.allclose(a.loc[common].values, b.loc[common].values,
                       atol=1e-10), "fusion scores changed when future removed"


def test_ml_signals_only_out_of_sample(data):
    """ML signals must all be dated after the train split point."""
    from engine.patterns.ml_cnn import detect_ml, TRAIN_FRAC
    panel = data["panel"]
    s = detect_ml(panel, max_train=4000, grid_step=15, use_torch=False)
    dates = panel.index.get_level_values("date").unique().sort_values()
    split_date = dates[int(len(dates) * TRAIN_FRAC)]
    assert len(s) > 0
    assert pd.to_datetime(s["date"]).min() >= split_date


def test_simulator_gap_independence(data):
    """No-news gaps must NOT predict same-day intraday direction."""
    panel = data["panel"]
    from engine.patterns.base import Wide
    w = Wide(panel)
    news = w.catalyst.astype(bool)
    g = w.gap.where(~news)
    intr = (w.close / w.open - 1).where(~news)
    m = (g.abs() > 0.005)
    same = (np.sign(g) == np.sign(intr)).where(m).stack().mean()
    assert 0.40 < same < 0.60, f"gap leaks intraday direction: {same}"
