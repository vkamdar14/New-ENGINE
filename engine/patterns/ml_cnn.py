"""ML pattern recognition on price-chart IMAGES, after Jiang, Kelly & Xiu,
"(Re-)Imag(in)ing Price Trends" (Journal of Finance, 2023).

Faithful elements:
  * inputs are rendered chart images, not numeric features: 20-day windows
    drawn as JKX do -- per day a 3px column: open tick (left), high-low bar
    (center), close tick (right); 20-day MA line overlaid; volume bars in a
    bottom strip; binary image, height 64 x width 60.
  * label: 1{forward 5-day return > 0} (JKX's short-horizon variant).
  * model: small CNN (conv-pool blocks then dense), trained on an initial
    window, applied strictly out-of-sample afterwards.
  * output signal: cnn_long / cnn_short on top/bottom prediction deciles.

Disclosed deviations (compute-bounded): 2 conv blocks (JKX use 3),
a weekly out-of-sample grid, and a capped training sample.  If PyTorch is
unavailable the same images (4x4-pooled) feed a HistGradientBoosting
classifier -- reported as `gbm` variant so the two are never conflated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import COLUMNS, Wide, empty

IMG_H, IMG_W = 64, 60
WIN = 20                 # days per image (3px each)
PRICE_H = 51             # top strip for price, bottom 12 for volume, 1 gap
FWD = 5                  # label horizon
TRAIN_FRAC = 0.4
DECILE = 0.1


def render_images(open_, high, low, close, volume, ends, ticker_cols):
    """Render binary chart images for windows ending at index positions
    `ends` (inclusive), for every ticker column.  Returns (images, keys)
    where keys = list of (end_pos, ticker)."""
    T, N = close.shape
    imgs, keys = [], []
    ma = pd.DataFrame(close).rolling(WIN, min_periods=1).mean().values
    for e in ends:
        s = e - WIN + 1
        if s < 0:
            continue
        o = open_[s:e + 1]; h = high[s:e + 1]; lo = low[s:e + 1]
        c = close[s:e + 1]; v = volume[s:e + 1]; m = ma[s:e + 1]
        valid = ~np.isnan(c).any(axis=0)
        pmax = np.nanmax(h, axis=0); pmin = np.nanmin(lo, axis=0)
        rng = np.where(pmax - pmin <= 0, 1.0, pmax - pmin)
        vmax = np.nanmax(v, axis=0)
        for j in np.where(valid)[0]:
            img = np.zeros((IMG_H, IMG_W), dtype=np.float32)
            ys = lambda p: np.clip(((pmax[j] - p) / rng[j]
                                    * (PRICE_H - 1)).astype(int), 0, PRICE_H - 1)
            yo, yh, yl, yc, ym = (ys(o[:, j]), ys(h[:, j]), ys(lo[:, j]),
                                  ys(c[:, j]), ys(m[:, j]))
            for d in range(WIN):
                x = 3 * d
                img[yo[d], x] = 1.0                          # open tick
                img[min(yh[d], yl[d]):max(yh[d], yl[d]) + 1, x + 1] = 1.0
                img[yc[d], x + 2] = 1.0                      # close tick
                img[ym[d], x:x + 3] = np.maximum(img[ym[d], x:x + 3], 0.5)
                vh = int(np.round(v[d, j] / max(vmax[j], 1e-9) * 11))
                if vh > 0:
                    img[IMG_H - vh:IMG_H, x + 1] = 1.0       # volume bar
            imgs.append(img)
            keys.append((e, j))
    if not imgs:
        return np.empty((0, IMG_H, IMG_W), np.float32), []
    return np.stack(imgs), keys


def _train_torch(Xtr, ytr, seed=0, epochs=4, batch=256):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    dev = "cpu"

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(1, 16, (5, 3), padding=(2, 1)), nn.ReLU(),
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(16, 32, (5, 3), padding=(2, 1)), nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Flatten(),
                nn.Linear(32 * 16 * 30, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, 1))

        def forward(self, x):
            return self.f(x)

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    X = torch.tensor(Xtr[:, None]); y = torch.tensor(ytr[:, None].astype(np.float32))
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            out = net(X[idx])
            loss = lossf(out, y[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        print(f"    [cnn] epoch {ep + 1}/{epochs} loss={tot / n:.4f}")

    def predict(Xte, batch=512):
        net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(Xte), batch):
                out = net(torch.tensor(Xte[i:i + batch, None]))
                outs.append(torch.sigmoid(out).numpy().ravel())
        return np.concatenate(outs) if outs else np.empty(0)

    return predict, "cnn"


def _train_sklearn(Xtr, ytr, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    def pool(X):
        # 4x4 max pool -> 16x15 = 240 features
        n, hh, ww = X.shape
        X = X[:, :64, :60].reshape(n, 16, 4, 15, 4)
        return X.max(axis=(2, 4)).reshape(n, -1)

    clf = HistGradientBoostingClassifier(max_iter=300, random_state=seed,
                                         early_stopping=True)
    clf.fit(pool(Xtr), ytr)

    def predict(Xte):
        return (clf.predict_proba(pool(Xte))[:, 1]
                if len(Xte) else np.empty(0))

    return predict, "gbm"


def detect_ml(panel: pd.DataFrame, seed: int = 0,
              max_train: int = 120_000, use_torch: bool | None = None,
              grid_step: int = 5) -> pd.DataFrame:
    """Walk-forward ML image signals.  Train once on the first TRAIN_FRAC of
    dates, predict on a `grid_step`-day grid afterwards."""
    w = Wide(panel)
    dates = w.close.index
    tickers = list(w.close.columns)
    O, H, Lo, C, V = (w.open.values, w.high.values, w.low.values,
                      w.close.values, w.volume.values)
    T = len(dates)
    split = int(T * TRAIN_FRAC)

    fwd = pd.DataFrame(C).pct_change(FWD).shift(-FWD).values  # label only (train side)

    rng = np.random.default_rng(seed)
    train_ends = np.arange(WIN, split - FWD)
    if len(train_ends) * len(tickers) > max_train:
        keep = max(1, max_train // len(tickers))
        train_ends = rng.choice(train_ends, size=keep, replace=False)
    print(f"  [ml] rendering train images ({len(train_ends)} days x "
          f"{len(tickers)} tickers)...")
    Xtr, ktr = render_images(O, H, Lo, C, V, np.sort(train_ends), tickers)
    ytr = np.array([fwd[e, j] > 0 for e, j in ktr], dtype=np.int64)
    ok = np.array([np.isfinite(fwd[e, j]) for e, j in ktr])
    Xtr, ytr = Xtr[ok], ytr[ok]
    print(f"  [ml] train set: {len(Xtr)} images, base rate {ytr.mean():.3f}")

    if use_torch is None:
        try:
            import torch  # noqa: F401
            use_torch = True
        except ImportError:
            use_torch = False
    predict, tag = (_train_torch(Xtr, ytr, seed) if use_torch
                    else _train_sklearn(Xtr, ytr, seed))

    rows = []
    test_ends = np.arange(split, T, grid_step)
    print(f"  [ml] predicting {len(test_ends)} out-of-sample grid days...")
    for chunk in np.array_split(test_ends, max(1, len(test_ends) // 25)):
        Xte, kte = render_images(O, H, Lo, C, V, chunk, tickers)
        if not len(Xte):
            continue
        p = predict(Xte)
        dfp = pd.DataFrame({"e": [k[0] for k in kte],
                            "j": [k[1] for k in kte], "p": p})
        for e, grp in dfp.groupby("e"):
            qlo, qhi = grp["p"].quantile([DECILE, 1 - DECILE])
            for _, r in grp.iterrows():
                if r["p"] >= qhi:
                    rows.append((dates[int(e)], tickers[int(r['j'])],
                                 "ml", f"{tag}_long", 1, float(r["p"]), FWD))
                elif r["p"] <= qlo:
                    rows.append((dates[int(e)], tickers[int(r['j'])],
                                 "ml", f"{tag}_short", -1,
                                 float(1 - r["p"]), FWD))
    out = pd.DataFrame(rows, columns=COLUMNS)
    return out if len(out) else empty()
