"""Console/markdown rendering.

Every report states its own sample size and reliability inline. A ranking
computed from nine videos and one from nine hundred look identical once
they are formatted into a table, and the difference is the whole story.
"""

from __future__ import annotations

import math
from typing import Sequence

from .audit import ChannelAudit
from .metrics import AgeCurve
from .outliers import Opportunity, Pattern
from .packaging import PackagingModel, thumbnail_notes, ThumbStats
from .trends import TopicTrend, Velocity


def _num(n: float) -> str:
    for cutoff, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= cutoff:
            return f"{n / cutoff:.1f}{suffix}"
    return f"{n:.0f}"


def _rule(title: str) -> str:
    return f"\n{title}\n{'=' * len(title)}"


def render_opportunities(ops: Sequence[Opportunity], limit: int = 20) -> str:
    out = [_rule("OUTLIERS - videos that beat their own channel")]
    if not ops:
        return "\n".join(out + ["", "  none cleared the threshold.",
                                "  Lower --min-multiplier, or widen the corpus."])
    out.append(f"\n{'mult':>6}  {'views':>8}  {'subs':>7}  {'reach':>6}  title")
    out.append("  " + "-" * 76)
    for o in ops[:limit]:
        flag = "*" if o.escaped_audience else " "
        prov = "~" if o.scored.provisional else " "
        out.append(
            f"{o.multiplier:6.1f}x {_num(o.video.views):>8}  {_num(o.subscribers):>7}  "
            f"{o.reach_ratio:5.1f}{flag} {prov}{o.video.title[:52]}"
        )
    out.append("\n  * views exceed subscriber count - genuinely reached new people")
    out.append("  ~ provisional: too young for a settled multiplier")
    return "\n".join(out)


def render_patterns(pats: Sequence[Pattern]) -> str:
    out = [_rule("WHAT THE OUTLIERS SHARE")]
    real = [p for p in pats if p.meaningful]
    if not real:
        return "\n".join(out + ["", "  No feature separated winners from the rest.",
                                "  That is a result: in this corpus, packaging structure is not",
                                "  the lever. Topic selection probably is."])
    for p in real:
        out.append(
            f"\n  {p.feature:<22} {p.lift:.2f}x lift  ({p.direction} among outliers)"
            f"\n  {'':<22} {p.outlier_rate:.0%} of outliers vs {p.baseline_rate:.0%} baseline, n={p.n_outliers}"
        )
    return "\n".join(out)


def render_packaging(model: PackagingModel | None) -> str:
    out = [_rule("PACKAGING MODEL")]
    if model is None:
        return "\n".join(out + ["", "  Not enough settled videos to fit (need 20+).",
                                "  Pull more channels or let recent uploads mature."])
    out.append(f"\n  fitted on {model.n} videos, R^2 = {model.r2:.3f}")
    if not model.reliable:
        out.append("  WEAK FIT - treat the drivers below as hypotheses, not findings.")
    out.append("\n  strongest correlates of overperformance (Spearman):")
    for feat, corr in model.top_drivers(6):
        arrow = "+" if corr > 0 else "-"
        out.append(f"    {arrow} {feat:<20} {corr:+.3f}")
    out.append(
        "\n  R^2 is low by nature here. Packaging is a minority of the variance;"
        "\n  topic and audience fit dominate. A model over ~0.4 is overfitted."
    )
    return "\n".join(out)


def render_title_scores(model: PackagingModel, titles: Sequence[str]) -> str:
    out = [_rule("TITLE DRAFTS")]
    if not model.reliable:
        out.append("\n  WARNING: model fit is weak - these rankings are indicative only.")
    scored = sorted(((model.score(t), t) for t in titles), reverse=True)
    out.append("")
    for score, t in scored:
        out.append(f"  {score:5.2f}x  {t}")
    out.append("\n  Scores are predicted multipliers in the corpus's own terms.")
    out.append("  Use them to rank drafts against each other, not as absolute forecasts.")
    return "\n".join(out)


def render_audit(a: ChannelAudit) -> str:
    out = [_rule(f"CHANNEL AUDIT - {a.channel_title}")]
    out.append(f"\n  {a.n_videos} uploads, {a.n_mature} with settled multipliers")
    out.append(f"  hit rate: {a.hit_rate:.0%} of videos beat their own baseline")
    out.append(f"  median engagement: {a.engagement_median:.2%} (likes+comments per view)")
    out.append(f"\n  cadence: {a.cadence.uploads_per_week:.1f}/week, {a.cadence.consistency}")
    out.append(f"           {a.cadence.note}")

    out.append(f"\n  {'format':<18} {'n':>3} {'median':>8} {'best':>7}  call")
    out.append("  " + "-" * 62)
    for f in a.formats:
        out.append(
            f"  {f.label:<18} {f.n:>3} {f.median_multiplier:7.2f}x {f.best_multiplier:6.1f}x  {f.call}"
        )

    if a.best:
        out.append("\n  best performers:")
        for s in a.best:
            out.append(f"    {s.multiplier:5.1f}x  {s.video.title[:58]}")
    if a.worst:
        out.append("\n  worst performers:")
        for s in a.worst:
            out.append(f"    {s.multiplier:5.1f}x  {s.video.title[:58]}")
    return "\n".join(out)


def render_trends(vels: Sequence[Velocity], topics: Sequence[TopicTrend]) -> str:
    out = [_rule("VELOCITY")]
    if not vels:
        return "\n".join(out + ["", "  No video has two snapshots yet.",
                                "  Velocity is a difference, so it needs a second observation:",
                                "  run `track` again in a few hours."])
    out.append(f"\n{'views/hr':>9} {'accel':>6}  {'phase':<13} title")
    out.append("  " + "-" * 74)
    for v in vels:
        out.append(f"{v.vph_recent:9.0f} {v.acceleration:5.2f}x  {v.phase:<13} {v.title[:44]}")

    if topics:
        out.append(_rule("TOPICS"))
        for t in topics:
            out.append(
                f"\n  {t.topic:<24} {t.verdict}"
                f"\n  {'':<24} {t.n_videos} videos, {t.median_vph:.0f} views/hr median,"
                f" {t.accelerating_share:.0%} still climbing"
            )
    return "\n".join(out)


def render_thumbnail(stats: ThumbStats) -> str:
    out = [_rule("THUMBNAIL")]
    if not stats.available:
        return "\n".join(out + ["", "  Pillow not installed - skipping (pip install Pillow)."])
    out.append(f"\n  brightness {stats.brightness:.2f}  saturation {stats.saturation:.2f}"
               f"  clutter {stats.edge_density:.2f}")
    for n in thumbnail_notes(stats):
        out.append(f"    - {n}")
    return "\n".join(out)


def render_curve(curve: AgeCurve) -> str:
    kind = "fitted from corpus" if curve.fitted else "DEFAULT (corpus too thin to fit)"
    pts = "  ".join(f"d{a:.0f}:{f:.0%}" for a, f in curve.points)
    return f"\n  age curve [{kind}]\n    {pts}"
