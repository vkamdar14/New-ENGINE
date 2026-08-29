"""Craft engines: the decisions that make an edit work, checked before render.

`editor.py` renders whatever it is handed. This module decides *what* to hand
it, and catches the mistakes that are invisible in a spec and obvious in the
finished video - a caption two characters too wide, a hook nobody can read in
time, a clip that stops a beat before the laugh.

The rules are drawn from how the format is actually consumed rather than from
taste, and each carries its reason, because a rule whose reason you cannot
state is one you cannot sensibly break:

  - Sound-off is the default viewing mode, so captions are load-bearing.
  - The first second decides whether the rest is watched at all.
  - Shorts autoplay in a loop, so the last frame is followed by the first;
    a clip that ends where it began buys a second viewing for free.
  - Text that overflows 1080px wraps mid-phrase and reads as amateur.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .editor import STYLES, WIDTH, CaptionStyle, Cue, EditSpec

# Reading speed for burned-in captions. Deliberately below prose reading rates:
# a viewer is also watching the picture, and the text is moving.
READ_CPS = 16.0

# Impact-like condensed faces average ~0.48 of point size per character; the
# wider Helvetica/Montserrat faces run nearer 0.55. Approximate on purpose -
# exact metrics need the font file, and the point of this is to catch the
# 30%-over case, not to typeset.
CHAR_WIDTH_RATIO = {"Impact": 0.46, "Arial Black": 0.58,
                    "Montserrat ExtraBold": 0.55, "Helvetica": 0.50}
DEFAULT_RATIO = 0.52

SAFE_WIDTH = WIDTH - 120           # 60px margin each side, matching the ASS styles
MIN_CUE_S = 0.45                   # below this a caption flickers rather than reads
MAX_CUE_S = 3.0                    # above this the caption is stale against the audio
SHORTS_MAX_S = 180.0
IDEAL_MIN_S, IDEAL_MAX_S = 15.0, 45.0


def estimate_width_px(text: str, style: CaptionStyle) -> float:
    ratio = CHAR_WIDTH_RATIO.get(style.font, DEFAULT_RATIO)
    return len(text) * style.size * ratio


def readable_seconds(text: str) -> float:
    return len(text) / READ_CPS


@dataclass
class Issue:
    severity: str      # "error" | "warn"
    where: str
    message: str
    fix: str = ""


@dataclass
class PacingReport:
    duration_s: float
    cue_count: int
    words: int
    words_per_minute: float
    cues_per_minute: float
    covered_fraction: float     # share of the clip with a caption on screen
    longest_gap_s: float

    @property
    def notes(self) -> list[str]:
        out = []
        if self.covered_fraction < 0.6 and self.cue_count:
            out.append(f"captions cover only {self.covered_fraction:.0%} of the clip - "
                       "sound-off viewers lose the thread in the gaps")
        if self.longest_gap_s > 3.0:
            out.append(f"{self.longest_gap_s:.1f}s with no caption - the longest silence "
                       "is where viewers leave")
        if self.words_per_minute and self.words_per_minute < 90:
            out.append(f"{self.words_per_minute:.0f} words/min is slow for the format; "
                       "consider tightening the cut")
        if self.words_per_minute > 260:
            out.append(f"{self.words_per_minute:.0f} words/min outruns comfortable reading")
        return out


def pacing(cues: Sequence[Cue], duration_s: float) -> PacingReport:
    if duration_s <= 0:
        return PacingReport(0, 0, 0, 0, 0, 0, 0)
    words = sum(len(c.text.split()) for c in cues)
    # Merge overlapping cues before measuring coverage, or two captions on
    # screen together would count that second twice and report >100%.
    spans = sorted((max(c.start_s, 0.0), min(c.end_s, duration_s)) for c in cues)
    merged: list[list[float]] = []
    for a, b in spans:
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    covered = sum(b - a for a, b in merged)

    gaps, cursor = [], 0.0
    for a, b in merged:
        gaps.append(a - cursor)
        cursor = b
    gaps.append(duration_s - cursor)

    return PacingReport(
        duration_s=duration_s, cue_count=len(cues), words=words,
        words_per_minute=words / (duration_s / 60.0),
        cues_per_minute=len(cues) / (duration_s / 60.0),
        covered_fraction=covered / duration_s,
        longest_gap_s=max(gaps) if gaps else duration_s,
    )


def validate(spec: EditSpec) -> list[Issue]:
    """Catch the failures that are invisible in a spec and obvious on screen."""
    issues: list[Issue] = []
    d = spec.duration_s

    if d <= 0:
        issues.append(Issue("error", "window", "end is not after start"))
        return issues
    if d > SHORTS_MAX_S:
        issues.append(Issue("error", "window", f"{d:.0f}s exceeds the {SHORTS_MAX_S:.0f}s "
                                               "Shorts limit", "shorten the window"))
    elif not IDEAL_MIN_S <= d <= IDEAL_MAX_S:
        issues.append(Issue("warn", "window",
                            f"{d:.0f}s is outside the {IDEAL_MIN_S:.0f}-{IDEAL_MAX_S:.0f}s "
                            "range most Shorts land in"))

    for i, c in enumerate(spec.cues):
        if estimate_width_px(c.text, spec.style) > SAFE_WIDTH:
            issues.append(Issue(
                "warn", f"cue {i}",
                f"\"{c.text[:28]}\" is ~{estimate_width_px(c.text, spec.style):.0f}px wide "
                f"against {SAFE_WIDTH}px safe area",
                "lower words-per-cue or pick a smaller style"))
        length = c.end_s - c.start_s
        if 0 < length < MIN_CUE_S:
            issues.append(Issue("warn", f"cue {i}",
                                f"on screen {length:.2f}s - flickers rather than reads"))
        elif length > MAX_CUE_S:
            issues.append(Issue("warn", f"cue {i}",
                                f"held {length:.1f}s - goes stale against the audio"))
        if length > 0 and readable_seconds(c.text) > length * 1.4:
            issues.append(Issue("warn", f"cue {i}",
                                f"needs ~{readable_seconds(c.text):.1f}s to read but shows "
                                f"for {length:.1f}s"))

    if spec.hook:
        if readable_seconds(spec.hook) > 2.5:
            issues.append(Issue("warn", "hook",
                                f"\"{spec.hook}\" needs ~{readable_seconds(spec.hook):.1f}s "
                                "to read but holds for 2.5s", "shorten it"))
        if estimate_width_px(spec.hook, spec.style) > SAFE_WIDTH:
            issues.append(Issue("warn", "hook", "wider than the safe area"))
        early = [c for c in spec.cues if c.start_s < 2.5]
        if early:
            issues.append(Issue("warn", "hook",
                                f"{len(early)} caption(s) share the screen with the hook",
                                "delay the first cue or drop the hook"))
    else:
        issues.append(Issue("warn", "hook", "no hook text in the first seconds",
                            "the opening second decides whether the rest is watched"))

    if not spec.cues:
        issues.append(Issue("error", "captions", "no captions",
                            "most Shorts viewing is sound-off; captions are load-bearing"))
    return issues


@dataclass
class LoopPlan:
    loops_cleanly: bool
    tail_gap_s: float
    advice: str


def plan_loop(spec: EditSpec) -> LoopPlan:
    """Shorts autoplay end-to-start, so the seam is a real edit decision.

    A clip whose last frame flows into its first gets a second view at no cost,
    and replays are among the strongest signals the format has. The cheap
    version of this is simply not leaving dead air at the end: trailing silence
    makes the seam obvious and gives the viewer a moment to swipe.
    """
    if not spec.cues:
        return LoopPlan(False, spec.duration_s,
                        "no captions, so no way to judge the seam")
    last_end = max(c.end_s for c in spec.cues)
    tail = spec.duration_s - last_end
    if tail > 1.5:
        return LoopPlan(False, tail,
                        f"{tail:.1f}s of dead air after the last caption - trim the tail so "
                        "the loop closes on the payoff instead of on silence")
    if tail < 0.2:
        return LoopPlan(False, tail,
                        "the clip cuts on the final word; leave ~0.4s so the last line "
                        "is readable before the loop restarts")
    return LoopPlan(True, tail,
                    f"clean seam: {tail:.1f}s after the last caption, tight enough to loop")


def suggest_style(cues: Sequence[Cue], duration_s: float) -> CaptionStyle:
    """Pick a caption style from the material rather than by preference.

    Dense speech needs smaller text and longer cues or it overflows; sparse,
    punchy speech can carry the heavy look that reads best in a feed.
    """
    p = pacing(cues, duration_s)
    if p.words_per_minute > 200:
        return STYLES["docu"]
    if p.words_per_minute > 150:
        return STYLES["clean"]
    if p.cue_count and p.words / max(p.cue_count, 1) <= 1.5:
        return STYLES["karaoke"]
    return STYLES["punch"]


_HOOK_PATTERNS = [
    ("question", "{}?"),
    ("stakes", "WATCH WHAT HAPPENS"),
    ("countdown", "WAIT FOR IT"),
    ("contradiction", "THIS SHOULDN'T WORK"),
    ("stakes_named", "{}"),
]


def hook_candidates(first_words: str, max_len: int = 24) -> list[str]:
    """Short hook options, length-capped so they can be read in 2.5s.

    Kept deliberately generic and few: a hook that misrepresents the clip
    costs more than no hook at all, because the viewer who stays for a promise
    you do not keep leaves in a way the algorithm notices.
    """
    words = re.findall(r"[\w']+", first_words)
    lead = " ".join(words[:4]).upper()
    out = ["WAIT FOR IT", "WATCH THIS"]
    if lead and len(lead) <= max_len:
        out.append(lead + "...")
    return [h for h in dict.fromkeys(out) if len(h) <= max_len]


def render(spec: EditSpec, issues: Sequence[Issue], p: PacingReport,
           loop: LoopPlan) -> str:
    o = ["", "CRAFT REVIEW", "============", ""]
    o.append(f"  {p.duration_s:.1f}s   {p.cue_count} cues   {p.words} words")
    o.append(f"  {p.words_per_minute:.0f} words/min   "
             f"captions cover {p.covered_fraction:.0%}   "
             f"longest silence {p.longest_gap_s:.1f}s")
    for n in p.notes:
        o.append(f"    - {n}")

    o.append(f"\n  LOOP: {'clean' if loop.loops_cleanly else 'needs work'}")
    o.append(f"    {loop.advice}")

    errors = [i for i in issues if i.severity == "error"]
    warns = [i for i in issues if i.severity == "warn"]
    o.append(f"\n  ISSUES: {len(errors)} error(s), {len(warns)} warning(s)")
    for i in errors + warns:
        tag = "ERROR" if i.severity == "error" else "warn "
        o.append(f"    [{tag}] {i.where}: {i.message}")
        if i.fix:
            o.append(f"            -> {i.fix}")
    if not issues:
        o.append("    none - spec is render-ready")
    return "\n".join(o)
