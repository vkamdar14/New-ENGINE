"""Clip mining: finding the moments in a long video worth cutting into Shorts.

The problem with clipping a 4-hour stream is not editing, it is *search*. There
are roughly 480 candidate 30-second windows in four hours and maybe six of them
are worth posting. Watching the whole thing to find them does not scale past
one creator.

YouTube knows exactly which moments get re-watched - that is the `mostReplayed`
heatmap on the scrubber - but it is not exposed in the Data API. The closest
public proxy is remarkably good and nearly free at 1 quota unit per 100
comments: **viewers timestamp the moments they want to re-watch**. A comment
saying "3:47 killed me" is a human vote for a specific second, and on a video
with a few thousand comments those votes concentrate hard on the handful of
moments that actually land.

The domain detail that makes or breaks this: viewers timestamp the **payoff**,
not the setup. A clip that starts at the timestamped second opens on a
punchline with no context and dies in the first two seconds. So every candidate
window here starts before its peak - see `LEAD_SECONDS`.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

# Matches 1:23, 12:34, 1:02:03, and 83:20 - the forms people actually type.
#
# The minute field allows up to 99 rather than stopping at 59. On a four-hour
# stream plenty of viewers write "83:20" instead of "1:23:20", and capping
# minutes at 59 silently discards every one of those votes - precisely on the
# long VODs where clip mining matters most. The looser bound admits some
# non-timestamps ("83:20" could be something else entirely), which is why
# `mentions_from_comments` discards anything past the video's duration; that
# bound is what keeps precision, not the regex.
#
# It stops at two digits: past 99 minutes essentially everyone writes 2:00:00,
# so accepting "120:00" would add almost no genuine marks while admitting far
# more scores, prices and model numbers.
_TS_RE = re.compile(r"(?<![\d:])(?:(\d{1,2}):)?(\d{1,2}):([0-5]\d)(?![\d:])")

# Viewers mark the payoff. Open a clip there and it lands with no setup, so the
# window starts this many seconds earlier. Tuned to the length of a typical
# spoken setup - long enough to carry context, short enough that the payoff
# still arrives inside a Shorts viewer's patience.
LEAD_SECONDS = 14.0

# How far past the peak to run. Reactions and the laugh land after the beat.
TAIL_SECONDS = 16.0

# Shorts are capped at 180s; below ~8s there is nothing to watch.
MIN_CLIP_S = 8.0
MAX_CLIP_S = 60.0


@dataclass
class Mention:
    """One viewer pointing at one second of the video."""

    seconds: float
    author: str
    likes: int
    text: str


def parse_timestamps(text: str) -> list[float]:
    """Every timestamp in a comment, in seconds.

    Bounded to plausible values: `99:99` is not a timestamp, and a bare `1:00`
    inside "1:00 scale model" is indistinguishable from a real mark, so the
    caller filters against the video's actual duration.
    """
    out = []
    for h, m, s in _TS_RE.findall(text or ""):
        total = int(m) * 60 + int(s) + (int(h) * 3600 if h else 0)
        out.append(float(total))
    return out


def mentions_from_comments(items: Iterable[dict], duration_s: int) -> list[Mention]:
    """Extract timestamp votes from raw commentThreads items.

    Anything past the video's end is discarded: those are prices, scores, and
    "2:1 ratio", not marks. This filter is the single biggest source of
    precision in the whole module.
    """
    out: list[Mention] = []
    for item in items:
        sn = (item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {}))
        text = sn.get("textDisplay") or sn.get("textOriginal") or ""
        author = sn.get("authorDisplayName", "")
        likes = int(sn.get("likeCount", 0) or 0)
        for t in parse_timestamps(text):
            if 0 <= t <= duration_s:
                out.append(Mention(seconds=t, author=author, likes=likes, text=text[:200]))
    return out


@dataclass
class ClipCandidate:
    peak_s: float
    start_s: float
    end_s: float
    mentions: int
    unique_authors: int
    like_weight: int
    sharpness: float          # peak height over local background
    score: float
    samples: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def confidence(self) -> str:
        """Sample size, stated plainly - three people is not a signal."""
        if self.unique_authors >= 15:
            return "high"
        if self.unique_authors >= 6:
            return "medium"
        return "low"

    def timestamp(self) -> str:
        m, s = divmod(int(self.start_s), 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _kde(mentions: Sequence[Mention], duration_s: float, bandwidth: float,
         step: float) -> tuple[list[float], list[float]]:
    """Gaussian kernel density over mention times.

    A histogram would be wrong here: viewers' clocks disagree by a few seconds,
    so the same moment gets marked at 3:45, 3:47 and 3:48. Hard bin edges split
    that one moment across two bins and can hide the strongest peak in the
    video. A kernel smooths the disagreement back into a single peak.

    Each mention is weighted by log(1+likes): a heavily-liked comment is
    evidence that many people agreed with the mark, but linear weighting would
    let one viral comment outvote fifty independent viewers.
    """
    n = max(int(duration_s / step) + 1, 1)
    grid = [i * step for i in range(n)]
    dens = [0.0] * n
    inv = 1.0 / (2 * bandwidth * bandwidth)
    for m in mentions:
        w = 1.0 + math.log1p(max(m.likes, 0))
        lo = max(0, int((m.seconds - 3 * bandwidth) / step))
        hi = min(n, int((m.seconds + 3 * bandwidth) / step) + 1)
        for i in range(lo, hi):
            d = grid[i] - m.seconds
            dens[i] += w * math.exp(-d * d * inv)
    return grid, dens


def find_clips(
    mentions: Sequence[Mention],
    duration_s: int,
    max_clips: int = 12,
    bandwidth: float = 8.0,
    step: float = 1.0,
    min_authors: int = 3,
) -> list[ClipCandidate]:
    """Rank the moments most worth cutting.

    Peaks are found on the smoothed density, then filtered by *unique authors*
    rather than raw mention count - one enthusiast posting the same timestamp
    twelve times is not twelve votes, and without this the top of the ranking
    fills with single-fan moments.
    """
    if not mentions or duration_s <= 0:
        return []

    grid, dens = _kde(mentions, duration_s, bandwidth, step)
    if not any(dens):
        return []

    # Background level for the sharpness ratio. Median rather than mean: on a
    # video with a few huge peaks the mean is dragged up by the peaks
    # themselves and every moment then looks unremarkable.
    ordered = sorted(dens)
    background = ordered[len(ordered) // 2] or (sum(dens) / len(dens)) or 1e-9

    # Local maxima, strongest first.
    peaks = [
        i for i in range(1, len(dens) - 1)
        if dens[i] >= dens[i - 1] and dens[i] > dens[i + 1] and dens[i] > background
    ]
    peaks.sort(key=lambda i: dens[i], reverse=True)

    chosen: list[ClipCandidate] = []
    for i in peaks:
        if len(chosen) >= max_clips:
            break
        peak_s = grid[i]
        start = max(0.0, peak_s - LEAD_SECONDS)
        end = min(float(duration_s), peak_s + TAIL_SECONDS)
        if end - start < MIN_CLIP_S:
            continue
        # Non-maximum suppression: two peaks 4s apart are one moment, and
        # without this the whole ranking is a dozen slices of one joke.
        if any(not (end <= c.start_s or start >= c.end_s) for c in chosen):
            continue

        window = [m for m in mentions if start - bandwidth <= m.seconds <= end + bandwidth]
        authors = {m.author for m in window if m.author}
        if len(authors) < min_authors:
            continue

        sharpness = dens[i] / background
        # Score rewards breadth of agreement first (unique authors), then
        # how sharply the moment stands out from the video's background.
        score = len(authors) * math.log1p(sharpness)
        chosen.append(
            ClipCandidate(
                peak_s=peak_s, start_s=start, end_s=min(end, start + MAX_CLIP_S),
                mentions=len(window), unique_authors=len(authors),
                like_weight=sum(m.likes for m in window),
                sharpness=sharpness, score=score,
                samples=[m.text for m in sorted(window, key=lambda x: -x.likes)[:3]],
            )
        )

    chosen.sort(key=lambda c: c.score, reverse=True)
    return chosen


def render_clips(cands: Sequence[ClipCandidate], title: str = "") -> str:
    out = ["", f"CLIP CANDIDATES{' - ' + title if title else ''}",
           "=" * (15 + (len(title) + 3 if title else 0))]
    if not cands:
        out += ["", "  No timestamp clusters found.",
                "  Either comments are disabled, the video is too new for viewers to",
                "  have marked it up, or nothing in it stood out enough to timestamp.",
                "  That last case is itself a finding: there may be no clip here."]
        return "\n".join(out)

    out.append(f"\n  {'start':>8} {'len':>5} {'votes':>6} {'people':>7} {'sharp':>6}  confidence")
    out.append("  " + "-" * 60)
    for c in cands:
        out.append(
            f"  {c.timestamp():>8} {c.duration_s:4.0f}s {c.mentions:6d} "
            f"{c.unique_authors:7d} {c.sharpness:5.1f}x  {c.confidence}"
        )
    out.append("\n  Windows open ~%ds before the marked second: viewers timestamp the" % LEAD_SECONDS)
    out.append("  payoff, and a clip that opens on the punchline has no setup.")
    top = cands[0]
    if top.samples:
        out.append(f"\n  what viewers said about {top.timestamp()}:")
        for s in top.samples:
            out.append(f"    \"{s[:70]}\"")
    return "\n".join(out)
