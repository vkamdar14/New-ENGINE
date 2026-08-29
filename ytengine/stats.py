"""Small statistics kit, pure stdlib.

View distributions are heavy-tailed and small-n: one 10M-view breakout inside
forty 20k-view uploads is normal. Every estimator here is chosen to survive
that - medians over means, MAD over standard deviation, rank correlation over
Pearson, ridge over plain least squares.
"""

from __future__ import annotations

import math
from typing import Sequence


def median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def quantile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile. q in [0, 1]."""
    s = sorted(xs)
    if not s:
        return 0.0
    if len(s) == 1:
        return float(s[0])
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def mad(xs: Sequence[float]) -> float:
    """Median absolute deviation, scaled to be comparable to a std dev."""
    if not xs:
        return 0.0
    m = median(xs)
    return 1.4826 * median([abs(x - m) for x in xs])


def robust_z(x: float, xs: Sequence[float]) -> float:
    """How many robust sigmas x sits above the cohort's centre."""
    spread = mad(xs)
    if spread <= 0:
        return 0.0
    return (x - median(xs)) / spread


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation, with ties averaged.

    Used instead of Pearson because we care whether a title feature moves a
    video up the ranking, not whether it moves views by a linear amount.
    """
    n = len(xs)
    if n < 3 or n != len(ys):
        return 0.0
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else 0.0


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def ridge_fit(X: Sequence[Sequence[float]], y: Sequence[float], alpha: float = 1.0) -> list[float]:
    """Ridge regression via the normal equations, solved by Gauss-Jordan.

    Returns coefficients for the columns of X plus a trailing intercept term.
    Ridge rather than OLS because title features are collinear by nature -
    "has a number" and "has a digit in the first three words" overlap heavily,
    and unpenalised least squares hands back wild offsetting coefficients that
    read as insight but are noise.
    """
    if not X or not y:
        return []
    n, p = len(X), len(X[0])
    # Standardise so one penalty is fair across features on different scales,
    # then append a constant column for the intercept (left unpenalised).
    means = [sum(row[j] for row in X) / n for j in range(p)]
    sds = []
    for j in range(p):
        var = sum((row[j] - means[j]) ** 2 for row in X) / max(n - 1, 1)
        sds.append(math.sqrt(var) or 1.0)
    Z = [[(row[j] - means[j]) / sds[j] for j in range(p)] + [1.0] for row in X]
    m = p + 1

    ata = [[sum(Z[i][a] * Z[i][b] for i in range(n)) for b in range(m)] for a in range(m)]
    aty = [sum(Z[i][a] * y[i] for i in range(n)) for a in range(m)]
    for a in range(p):  # penalise slopes only, never the intercept
        ata[a][a] += alpha

    coef_std = _solve(ata, aty)
    if not coef_std:
        return []
    # Map standardised coefficients back onto the raw feature scale.
    coefs = [coef_std[j] / sds[j] for j in range(p)]
    intercept = coef_std[p] - sum(coefs[j] * means[j] for j in range(p))
    return coefs + [intercept]


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gauss-Jordan with partial pivoting. Returns [] if singular."""
    m = len(b)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return []
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(m):
            if r != col and aug[r][col]:
                f = aug[r][col]
                aug[r] = [v - f * w for v, w in zip(aug[r], aug[col])]
    return [row[m] for row in aug]


def r_squared(y: Sequence[float], pred: Sequence[float]) -> float:
    if not y:
        return 0.0
    mean = sum(y) / len(y)
    ss_tot = sum((v - mean) ** 2 for v in y)
    ss_res = sum((v - p) ** 2 for v, p in zip(y, pred))
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0
