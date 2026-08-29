"""Title and thumbnail scoring, fitted rather than asserted.

Most "title checkers" ship a fixed list of power words and a arbitrary score.
That is astrology. The same title that wins in true-crime dies in woodworking.

This module instead *learns* which packaging features predict overperformance
inside the corpus you give it, by regressing log-multiplier on title features.
The output is a model plus the honest diagnostics - how much variance it
actually explains, and how many videos it saw - so you can tell the difference
between a real signal and a pattern fitted to forty data points.

Expect low R-squared. Packaging genuinely is a minority of the variance;
topic and audience fit dominate. An R-squared of 0.15 is a useful model here,
and any tool claiming 0.9 is fitting noise.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .metrics import Scored
from .stats import r_squared, ridge_fit, spearman

CURIOSITY = {
    "why", "how", "what", "secret", "truth", "reason", "actually", "really",
    "nobody", "everyone", "never", "always", "stop", "mistake", "wrong",
    "hidden", "myth", "before", "until", "finally",
}
STAKES = {
    "worst", "best", "biggest", "insane", "brutal", "crazy", "impossible",
    "perfect", "ultimate", "destroyed", "exposed", "banned", "ruined",
}

_NUM_RE = re.compile(r"\d")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_BRACKET_RE = re.compile(r"[\[\(][^\]\)]{1,25}[\]\)]")
_WORD_RE = re.compile(r"[A-Za-z']+")

FEATURES = [
    "char_len", "word_count", "has_number", "has_year", "is_question",
    "curiosity_words", "stakes_words", "caps_words", "has_bracket",
    "has_colon", "first_person", "second_person", "starts_with_verb",
]


def title_features(title: str) -> dict[str, float]:
    """Structural features of a title. No judgement, just measurement."""
    words = _WORD_RE.findall(title)
    lower = [w.lower() for w in words]
    # A "caps word" means shouting (THIS), not an initialism (AI, NASA).
    caps = [w for w in words if len(w) > 2 and w.isupper()]
    return {
        "char_len": float(len(title)),
        "word_count": float(len(words)),
        "has_number": 1.0 if _NUM_RE.search(title) else 0.0,
        "has_year": 1.0 if _YEAR_RE.search(title) else 0.0,
        "is_question": 1.0 if title.strip().endswith("?") else 0.0,
        "curiosity_words": float(sum(1 for w in lower if w in CURIOSITY)),
        "stakes_words": float(sum(1 for w in lower if w in STAKES)),
        "caps_words": float(len(caps)),
        "has_bracket": 1.0 if _BRACKET_RE.search(title) else 0.0,
        "has_colon": 1.0 if ":" in title else 0.0,
        "first_person": 1.0 if any(w in ("i", "my", "me", "we", "our") for w in lower) else 0.0,
        "second_person": 1.0 if any(w in ("you", "your", "yours") for w in lower) else 0.0,
        "starts_with_verb": 1.0 if lower and lower[0] in _COMMON_VERBS else 0.0,
    }


_COMMON_VERBS = {
    "make", "build", "stop", "watch", "learn", "get", "do", "try", "use",
    "fix", "start", "avoid", "master", "read", "buy", "play", "turn",
}


@dataclass
class PackagingModel:
    coefs: dict[str, float]
    intercept: float
    r2: float
    n: int
    rank_corr: dict[str, float] = field(default_factory=dict)

    @property
    def reliable(self) -> bool:
        """Roughly 8 videos per feature before coefficients mean much."""
        return self.n >= 8 * len(FEATURES) and self.r2 > 0.02

    def predict_log_multiplier(self, title: str) -> float:
        f = title_features(title)
        return self.intercept + sum(self.coefs.get(k, 0.0) * f[k] for k in FEATURES)

    def score(self, title: str) -> float:
        """Predicted multiplier for a draft title, in the corpus's own terms."""
        return math.exp(self.predict_log_multiplier(title))

    def top_drivers(self, k: int = 6) -> list[tuple[str, float]]:
        """Features with the largest rank correlation to overperformance.

        Reported from Spearman rather than the fitted coefficients, because a
        ridge coefficient on a collinear feature set is not directly readable
        as importance, whereas a rank correlation is.
        """
        ranked = sorted(self.rank_corr.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return ranked[:k]


def fit_packaging(scored: Sequence[Scored], alpha: float = 1.0) -> Optional[PackagingModel]:
    """Fit title features against log-multiplier.

    Only trustworthy rows are used: an unreliable baseline or a provisional
    (too-young) multiplier would teach the model noise.

    The target is log-multiplier because multipliers are multiplicative and
    heavy-tailed - a 20x and a 0.05x are symmetric departures in log space but
    would let the 20x dominate a fit in linear space.
    """
    rows = [s for s in scored if s.trustworthy and s.multiplier > 0]
    if len(rows) < 20:
        return None

    feats = [title_features(s.video.title) for s in rows]
    X = [[f[k] for k in FEATURES] for f in feats]
    y = [math.log(s.multiplier) for s in rows]

    coefs = ridge_fit(X, y, alpha=alpha)
    if not coefs:
        return None
    *slopes, intercept = coefs
    pred = [sum(c * xi for c, xi in zip(slopes, row)) + intercept for row in X]

    rank_corr = {
        k: spearman([f[k] for f in feats], y)
        for k in FEATURES
        # A feature that is constant across the corpus has no rank information.
        if len({f[k] for f in feats}) > 1
    }

    return PackagingModel(
        coefs=dict(zip(FEATURES, slopes)),
        intercept=intercept,
        r2=r_squared(y, pred),
        n=len(rows),
        rank_corr=rank_corr,
    )


# ---------------- thumbnails ----------------

@dataclass
class ThumbStats:
    brightness: float      # 0-1
    saturation: float      # 0-1
    edge_density: float    # 0-1, a proxy for visual clutter
    available: bool = True


def thumbnail_stats(path: str) -> ThumbStats:
    """Cheap visual statistics for a thumbnail.

    Deliberately limited to properties that survive being shrunk to the size a
    thumbnail is actually viewed at on a phone. Face detection and OCR would
    need heavyweight dependencies; brightness, saturation and clutter already
    separate thumbnails that read at 120px wide from ones that do not.

    Degrades to `available=False` when Pillow is not installed, so the rest of
    the engine keeps working.
    """
    try:
        from PIL import Image, ImageFilter  # type: ignore
    except ImportError:
        return ThumbStats(0.0, 0.0, 0.0, available=False)

    img = Image.open(path).convert("RGB").resize((160, 90))
    px = list(img.getdata())
    n = len(px)
    brightness = sum((r + g + b) / 3 for r, g, b in px) / (255 * n)
    saturation = sum((max(p) - min(p)) / (max(p) or 1) for p in px) / n
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_density = sum(edges.getdata()) / (255 * n)
    return ThumbStats(brightness, saturation, edge_density)


def thumbnail_notes(t: ThumbStats) -> list[str]:
    """Flag the failure modes that actually cost clicks on a small screen."""
    if not t.available:
        return ["thumbnail analysis unavailable (pip install Pillow)"]
    notes = []
    if t.brightness < 0.25:
        notes.append("very dark - loses to bright neighbours in a crowded feed")
    if t.brightness > 0.88:
        notes.append("blown out - little contrast left for a focal point")
    if t.saturation < 0.18:
        notes.append("washed out - low colour separation from the UI chrome")
    if t.edge_density > 0.34:
        notes.append("visually busy - unlikely to resolve at phone-feed size")
    if t.edge_density < 0.05:
        notes.append("almost no detail - may read as empty or unfinished")
    return notes or ["no structural problems detected"]
