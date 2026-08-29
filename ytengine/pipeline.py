"""Clip -> edit -> gate -> revise, until the clip is fit to publish.

The loop the whole engine was building toward: mine a moment, cut it, check
it, and send it back to editing until it passes.

WHAT THE GATE CAN AND CANNOT CERTIFY. It is tempting to gate on a predicted
view count - "reject unless the model says 50k". The backtest is direct
evidence that this does not work, for two reasons that compound:

  1. Calibration runs ~1.7x high, so a predicted 50,000 is really nearer
     30,000. The model's numbers rank correctly and are individually wrong.
  2. Nearly all its predictive power comes from *channel history*, and a new
     clip channel has none. Stripped of that feature the model scored 42%
     against a 35% baseline, with rank correlation 0.068 - barely above noise.

So a view-count gate on a brand-new channel would be measuring almost nothing
while sounding authoritative. This module therefore gates on the two signals
that are real at publication time:

  crowd    how many *distinct* people independently marked this moment, and
           how sharply it stands out. This is measurement, not prediction -
           those people already watched it and said so.
  craft    the mechanical failures that reliably cost retention: captions
           that overflow or flicker, silence, a broken loop seam.

The virality score is still reported, as a *comparative* ranking across your
own candidates, which is the job it demonstrably does at 82.5% win rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .clips import MAX_CLIP_S, MIN_CLIP_S, ClipCandidate
from .craft import Issue, pacing, plan_loop, suggest_style, validate
from .editor import STYLES, Cue, EditSpec, build_ass

# A moment nobody but a handful of people marked is not a moment.
MIN_CROWD_AUTHORS = 5
MIN_SHARPNESS = 8.0

# Pacing floors. These are gate failures rather than advisory notes, because
# they are exactly what `revise` knows how to fix - passing a clip with 14
# seconds of dead air while holding an automated fix for it is the pipeline
# declining to do its job.
MIN_CAPTION_COVERAGE = 0.40
MAX_DEAD_AIR_S = 1.5


@dataclass
class GateResult:
    passed: bool
    crowd_ok: bool
    craft_ok: bool
    authors: int
    sharpness: float
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        if self.passed:
            return "passed"
        bits = []
        if not self.crowd_ok:
            bits.append(f"crowd too thin ({self.authors} people, {self.sharpness:.1f}x)")
        if not self.craft_ok:
            bits.append(f"{len(self.errors)} craft error(s)" if self.errors
                        else "pacing below floor (coverage or dead air)")
        return "; ".join(bits)


def gate(spec: EditSpec, cand: ClipCandidate) -> GateResult:
    issues = validate(spec)
    errors = [i for i in issues if i.severity == "error"]
    warns = [i for i in issues if i.severity == "warn"]
    p = pacing(spec.cues, spec.duration_s)
    loop = plan_loop(spec)

    crowd_ok = (cand.unique_authors >= MIN_CROWD_AUTHORS
                and cand.sharpness >= MIN_SHARPNESS)

    pacing_ok = True
    notes = list(p.notes)
    if spec.cues and p.covered_fraction < MIN_CAPTION_COVERAGE:
        pacing_ok = False
        notes.append(f"caption coverage {p.covered_fraction:.0%} is below the "
                     f"{MIN_CAPTION_COVERAGE:.0%} floor")
    if loop.tail_gap_s > MAX_DEAD_AIR_S:
        pacing_ok = False
    if not loop.loops_cleanly:
        notes.append(loop.advice)

    craft_ok = not errors and pacing_ok
    return GateResult(passed=crowd_ok and craft_ok, crowd_ok=crowd_ok, craft_ok=craft_ok,
                      authors=cand.unique_authors, sharpness=cand.sharpness,
                      errors=errors, warnings=warns, notes=notes)


def revise(spec: EditSpec, result: GateResult) -> tuple[EditSpec, list[str]]:
    """Apply the fixes the gate asked for. Returns the new spec and what changed.

    Only mechanical corrections are automated - trimming dead air, restyling
    for overflow, shortening an unreadable hook. Nothing here can invent
    captions or make a boring moment interesting, and pretending otherwise
    would just loop forever on a clip that was never going to work.
    """
    changes: list[str] = []
    new = EditSpec(source=spec.source, start_s=spec.start_s, end_s=spec.end_s,
                   style=spec.style, cues=list(spec.cues), hook=spec.hook,
                   crop_mode=spec.crop_mode)

    # Trim trailing dead air so the loop closes on the payoff.
    if new.cues:
        last_end = max(c.end_s for c in new.cues)
        tail = new.duration_s - last_end
        if tail > 1.5:
            # Trim to just past the last caption, but never below the minimum
            # watchable length. Skipping the trim entirely when the captions
            # end early - as an earlier version did - left the clip carrying
            # all of its dead air rather than as little as the floor allows.
            wanted = last_end + 0.5
            keep = max(wanted, MIN_CLIP_S)
            if keep < new.duration_s - 0.5:
                removed = new.duration_s - keep
                new.end_s = new.start_s + keep
                changes.append(f"trimmed {removed:.1f}s of dead air off the tail")

    # Overflowing captions: a lighter style fits more characters per line.
    if any("safe area" in i.message for i in result.warnings) and new.style.name != "docu":
        new.style = STYLES["docu"]
        changes.append("switched to the 'docu' style so captions fit the safe area")

    # A hook that cannot be read inside its window is worse than none.
    if any(i.where == "hook" and "to read" in i.message for i in result.warnings):
        if len(new.hook) > 14:
            new.hook = new.hook[:14].rstrip()
            changes.append(f"shortened the hook to \"{new.hook}\"")

    # A hook sharing the screen with captions reads as clutter.
    if any(i.where == "hook" and "share the screen" in i.message for i in result.warnings):
        new.hook = ""
        changes.append("dropped the hook - captions already occupy the opening")

    # Low coverage on a long clip means the window is padded around a short
    # moment. Tighten it to what the captions actually cover.
    if new.cues:
        first, last = min(c.start_s for c in new.cues), max(c.end_s for c in new.cues)
        covered = last - first
        if covered > 0 and covered / new.duration_s < 0.40:
            lead, tail = 2.0, 1.0
            ns = new.start_s + max(first - lead, 0.0)
            ne = new.start_s + min(last + tail, new.duration_s)
            if ne - ns >= MIN_CLIP_S and (ne - ns) < new.duration_s - 0.5:
                shift = ns - new.start_s
                new.cues = [Cue(c.start_s - shift, c.end_s - shift, c.text) for c in new.cues]
                new.start_s, new.end_s = ns, ne
                changes.append(f"tightened the window to {ne - ns:.0f}s around the captions")

    if new.duration_s > MAX_CLIP_S:
        new.end_s = new.start_s + MAX_CLIP_S
        changes.append(f"capped the clip at {MAX_CLIP_S:.0f}s")

    return new, changes


@dataclass
class PipelineRun:
    spec: EditSpec
    result: GateResult
    rounds: int
    history: list[list[str]] = field(default_factory=list)


def run(spec: EditSpec, cand: ClipCandidate, max_rounds: int = 4) -> PipelineRun:
    """Gate, revise, repeat - until it passes or nothing more can be fixed."""
    history: list[list[str]] = []
    for rnd in range(1, max_rounds + 1):
        res = gate(spec, cand)
        if res.passed:
            return PipelineRun(spec=spec, result=res, rounds=rnd, history=history)
        spec, changes = revise(spec, res)
        history.append(changes)
        if not changes:
            # Nothing left to fix mechanically; looping again would just burn
            # rounds producing the identical spec.
            return PipelineRun(spec=spec, result=res, rounds=rnd, history=history)
    return PipelineRun(spec=spec, result=gate(spec, cand),
                       rounds=max_rounds, history=history)


def render(run_: PipelineRun, score: Optional[int] = None) -> str:
    r = run_.result
    o = ["", "PUBLISH GATE", "============", ""]
    o.append(f"  rounds run    {run_.rounds}")
    for i, changes in enumerate(run_.history, 1):
        for c in changes:
            o.append(f"    round {i}: {c}")
    o.append("")
    o.append(f"  crowd   {'PASS' if r.crowd_ok else 'FAIL'}   "
             f"{r.authors} distinct people marked it, {r.sharpness:.1f}x sharper than "
             "the video's background")
    o.append(f"  craft   {'PASS' if r.craft_ok else 'FAIL'}   "
             f"{len(r.errors)} error(s), {len(r.warnings)} warning(s)")
    for i in r.errors:
        o.append(f"    ERROR {i.where}: {i.message}")
    for n in r.notes:
        o.append(f"    note: {n}")
    o.append(f"\n  RESULT: {'PUBLISH' if r.passed else 'BACK TO EDIT'} - {r.reason}")

    if score is not None:
        o.append(f"\n  virality score: {score}/100 (comparative rank among candidates)")
    o.append("\n  This gate does not certify a view count, and no gate built on public")
    o.append("  data can. The predictor's absolute numbers run ~1.7x high, and on a")
    o.append("  new channel its dominant feature - channel history - does not exist,")
    o.append("  which drops it to near noise. What it does well is rank your own")
    o.append("  candidates against each other, so use the score to choose which clip")
    o.append("  to cut next, not to promise what it will do.")
    return "\n".join(o)
