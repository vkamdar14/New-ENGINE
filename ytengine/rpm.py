"""RPM-weighted niche value: views are not income.

The engine's other rankings sort by views, which optimises the wrong variable
if the goal is money. Ad rates vary by more than an order of magnitude across
niches, so a smaller audience in an expensive niche routinely out-earns a
larger one in a cheap niche.

Rough long-form RPM by niche, in USD per 1,000 monetised views. These are
order-of-magnitude figures gathered from widely reported creator ranges, not
measurements - they move with season, audience geography and ad market, and
Q1 typically runs 30-40% below Q4. Treat them as relative weights, and replace
them with your own YouTube Studio numbers the moment you have any.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

# Long-form RPM bands. Shorts earn far less - see SHORTS_RPM_FACTOR.
NICHE_RPM = {
    "personal finance": 18.0, "investing": 17.0, "insurance": 22.0,
    "real estate": 14.0, "business": 13.0, "b2b software": 20.0,
    "marketing": 12.0, "career": 9.0, "productivity": 8.0,
    "tech reviews": 7.0, "programming": 7.5, "ai tools": 9.0,
    "health": 6.5, "fitness": 5.0, "cooking": 4.0, "diy": 4.5,
    "travel": 4.0, "education": 5.0, "parenting": 6.0,
    "beauty": 5.5, "fashion": 5.0, "automotive": 6.0,
    "pets": 3.0, "comedy": 2.5, "gaming": 2.0, "music": 1.8,
    "vlog": 2.5, "reaction": 1.5, "entertainment": 2.0, "shorts": 1.0,
}
DEFAULT_RPM = 4.0

# Shorts monetise through a separate revenue pool and land far below long-form
# on the same view count. This factor is the single most consequential number
# in the file: it is why a million Shorts views and a million long-form views
# are not remotely the same business.
SHORTS_RPM_FACTOR = 0.02


@dataclass
class NicheValue:
    niche: str
    rpm: float
    monthly_views: int
    is_shorts: bool

    @property
    def effective_rpm(self) -> float:
        return self.rpm * (SHORTS_RPM_FACTOR if self.is_shorts else 1.0)

    @property
    def monthly_revenue(self) -> float:
        return self.monthly_views / 1000.0 * self.effective_rpm

    def views_needed_for(self, target_monthly: float) -> int:
        """How many views a month this niche needs to hit an income target."""
        if self.effective_rpm <= 0:
            return 0
        return int(target_monthly * 1000.0 / self.effective_rpm)


def rpm_for(niche: str) -> float:
    key = niche.strip().lower()
    if key in NICHE_RPM:
        return NICHE_RPM[key]
    # Substring fallback: "beginner personal finance tips" should find its band.
    for k, v in NICHE_RPM.items():
        if k in key or key in k:
            return v
    return DEFAULT_RPM


def value_of(niche: str, monthly_views: int, is_shorts: bool = True) -> NicheValue:
    return NicheValue(niche=niche, rpm=rpm_for(niche),
                      monthly_views=monthly_views, is_shorts=is_shorts)


def compare(niches: Sequence[str], monthly_views: int,
            is_shorts: bool = True) -> list[NicheValue]:
    vals = [value_of(n, monthly_views, is_shorts) for n in niches]
    vals.sort(key=lambda v: v.monthly_revenue, reverse=True)
    return vals


def render(vals: Sequence[NicheValue], target_monthly: float = 2000.0) -> str:
    if not vals:
        return "\n  no niches given"
    o = ["", "NICHE VALUE (views x RPM, not views)", "=" * 36, ""]
    fmt = "shorts" if vals[0].is_shorts else "long-form"
    o.append(f"  at {vals[0].monthly_views:,} {fmt} views/month\n")
    o.append(f"  {'niche':<20} {'RPM':>7} {'eff.RPM':>9} {'$/month':>10}"
             f" {'views for $' + str(int(target_monthly)):>16}")
    o.append("  " + "-" * 68)
    for v in vals:
        o.append(f"  {v.niche[:20]:<20} {v.rpm:>7.1f} {v.effective_rpm:>9.2f} "
                 f"{v.monthly_revenue:>10,.0f} {v.views_needed_for(target_monthly):>16,}")
    best, worst = vals[0], vals[-1]
    if worst.monthly_revenue > 0:
        o.append(f"\n  '{best.niche}' earns {best.monthly_revenue / worst.monthly_revenue:,.0f}x "
                 f"'{worst.niche}' on identical view counts.")
    if vals[0].is_shorts:
        o.append(f"\n  Shorts are weighted at {SHORTS_RPM_FACTOR:.0%} of long-form RPM. That factor,")
        o.append("  not the niche, is usually the biggest number on this page: the same")
        o.append("  views as long-form earn roughly one fiftieth as much.")
    o.append("\n  RPMs are order-of-magnitude industry ranges, not measurements.")
    o.append("  Replace them with your own Studio figures as soon as you have any.")
    return "\n".join(o)
