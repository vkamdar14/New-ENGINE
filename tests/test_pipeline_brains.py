"""Tests for the publish gate, the revise loop, and the segmented ensemble."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from ytengine.backtest import build_samples
from ytengine.brains import (SIZE_TIERS, TOPIC_TIERS, DURATION_TIERS, compare_architectures,
                             duration_tier, fit_ensemble, segment_of, size_tier, topic_tier)
from ytengine.clips import ClipCandidate
from ytengine.editor import STYLES, Cue, EditSpec
from ytengine.models import Video
from ytengine.pipeline import (MAX_DEAD_AIR_S, MIN_CAPTION_COVERAGE, MIN_CROWD_AUTHORS,
                               gate, revise, run)

NOW = datetime(2026, 8, 1, tzinfo=timezone.utc)


def cand(authors=10, sharp=20.0, start=100.0, end=130.0):
    return ClipCandidate(peak_s=start + 14, start_s=start, end_s=end, mentions=authors,
                         unique_authors=authors, like_weight=0, sharpness=sharp, score=0.0)


def spec(**kw):
    base = dict(source="v.mp4", start_s=100.0, end_s=130.0, style=STYLES["punch"],
                cues=[Cue(2.0, 20.0, "SOMETHING HAPPENS")], hook="WAIT")
    base.update(kw)
    return EditSpec(**base)


class TestGate(unittest.TestCase):
    def test_thin_crowd_fails(self):
        g = gate(spec(), cand(authors=MIN_CROWD_AUTHORS - 1))
        self.assertFalse(g.passed)
        self.assertFalse(g.crowd_ok)
        self.assertIn("crowd too thin", g.reason)

    def test_flat_moment_fails_even_with_many_marks(self):
        # Lots of marks spread flat across a video is not a moment.
        self.assertFalse(gate(spec(), cand(authors=50, sharp=1.2)).crowd_ok)

    def test_no_captions_fails(self):
        self.assertFalse(gate(spec(cues=[]), cand()).craft_ok)

    def test_dead_air_blocks_rather_than_warns(self):
        """A gate that passes a clip while holding an automated fix for it is
        declining to do its job."""
        s = spec(cues=[Cue(0.0, 5.0, "HI")], end_s=130.0)   # 25s of tail
        self.assertFalse(gate(s, cand()).craft_ok)

    def test_low_coverage_blocks(self):
        s = spec(cues=[Cue(0.0, 2.0, "HI")], end_s=130.0)
        g = gate(s, cand())
        self.assertFalse(g.craft_ok)
        self.assertTrue(any("coverage" in n for n in g.notes))

    def test_well_formed_clip_passes(self):
        s = spec(cues=[Cue(0.5, 14.0, "A LINE"), Cue(14.0, 28.5, "ANOTHER")], end_s=129.0)
        self.assertTrue(gate(s, cand()).passed)


class TestRevise(unittest.TestCase):
    def test_trims_dead_air(self):
        s = spec(cues=[Cue(0.0, 5.0, "HI")], end_s=130.0)
        new, changes = revise(s, gate(s, cand()))
        self.assertLess(new.duration_s, s.duration_s)
        self.assertTrue(any("dead air" in c for c in changes))

    def test_never_trims_below_the_minimum_clip_length(self):
        s = spec(start_s=0.0, end_s=12.0, cues=[Cue(0.0, 1.0, "HI")])
        new, _ = revise(s, gate(s, cand()))
        self.assertGreaterEqual(new.duration_s, 8.0)

    def test_cues_shift_with_a_retimed_window(self):
        """Tightening the window without rebasing the cues desyncs every
        caption by the amount trimmed."""
        s = spec(start_s=100.0, end_s=160.0,
                 cues=[Cue(30.0, 34.0, "A"), Cue(34.0, 38.0, "B")])
        new, changes = revise(s, gate(s, cand(end=160.0)))
        if new.start_s != s.start_s:
            shift = new.start_s - s.start_s
            self.assertAlmostEqual(new.cues[0].start_s, 30.0 - shift, places=3)

    def test_caps_over_long_clips(self):
        s = spec(start_s=0.0, end_s=400.0, cues=[Cue(0, 390, "X")])
        new, _ = revise(s, gate(s, cand(end=400.0)))
        self.assertLessEqual(new.duration_s, 60.0)

    def test_drops_a_hook_that_collides_with_captions(self):
        s = spec(cues=[Cue(0.2, 20.0, "EARLY LINE")], hook="WAIT")
        new, changes = revise(s, gate(s, cand()))
        self.assertTrue(new.hook == "" or any("hook" in c for c in changes))


class TestRunLoop(unittest.TestCase):
    def test_stops_when_nothing_more_can_be_fixed(self):
        """A clip with no captions cannot be repaired mechanically, and looping
        would just reproduce the same spec four times."""
        r = run(spec(cues=[]), cand(), max_rounds=4)
        self.assertLess(r.rounds, 4)
        self.assertFalse(r.result.passed)

    def test_terminates_within_max_rounds(self):
        r = run(spec(cues=[Cue(0, 1, "HI")], end_s=200.0), cand(end=200.0), max_rounds=3)
        self.assertLessEqual(r.rounds, 3)

    def test_passing_clip_needs_one_round(self):
        s = spec(cues=[Cue(0.5, 14.0, "A"), Cue(14.0, 28.5, "B")], end_s=129.0)
        self.assertEqual(run(s, cand()).rounds, 1)

    def test_revision_actually_improves_the_spec(self):
        s = spec(cues=[Cue(0.0, 6.0, "HI")], end_s=130.0)
        r = run(s, cand(), max_rounds=4)
        self.assertLess(r.spec.duration_s, s.duration_s)


class TestSegments(unittest.TestCase):
    def test_grid_is_144_cells(self):
        self.assertEqual(len(SIZE_TIERS) * len(DURATION_TIERS) * TOPIC_TIERS, 144)

    def test_tiers_partition_their_ranges(self):
        self.assertEqual(size_tier(2.0), "micro")
        self.assertEqual(size_tier(4.5), "mid")
        self.assertEqual(size_tier(99.0), "large")
        self.assertEqual(duration_tier(5), "s0")
        self.assertEqual(duration_tier(10 ** 7), "s5")

    def test_topic_tier_is_stable(self):
        self.assertEqual(topic_tier("some title here"), topic_tier("some title here"))

    def test_topic_tier_ignores_word_order(self):
        self.assertEqual(topic_tier("alpha beta gamma"), topic_tier("gamma beta alpha"))


class TestEnsemble(unittest.TestCase):
    def _corpus(self):
        out = []
        for c in range(14):
            for i in range(45):
                out.append(Video(
                    video_id=f"c{c}-{i}", channel_id=f"ch{c}", channel_title="C",
                    title=f"title {i % 7} word{c}", description="",
                    published_at=NOW - timedelta(days=1400 - i * 28),
                    duration_s=15 + (i % 5) * 20, views=1000 * (c + 1) * (1 + i % 3),
                    likes=10, comments=1))
        return out

    def test_thin_segments_are_shrunk_toward_global(self):
        """The whole point of partial pooling: a segment with twelve videos
        must not be trusted to estimate its own coefficients."""
        s = build_samples(self._corpus(), now=NOW)
        ens = fit_ensemble(s)
        self.assertIsNotNone(ens)
        for b in ens.brains.values():
            if b.n < 30:
                self.assertLess(b.shrinkage, 0.35)

    def test_shrinkage_rises_with_segment_size(self):
        s = build_samples(self._corpus(), now=NOW)
        ens = fit_ensemble(s)
        pairs = sorted((b.n, b.shrinkage) for b in ens.brains.values())
        if len(pairs) >= 2:
            self.assertLessEqual(pairs[0][1], pairs[-1][1])

    def test_comparison_reports_both_architectures(self):
        c = compare_architectures(build_samples(self._corpus(), now=NOW))
        self.assertIsNotNone(c)
        self.assertEqual(c.grid_cells, 144)
        self.assertGreater(c.global_win_rate, 0.0)
        self.assertIn(c.verdict.split(" ")[0], ("SEGMENTATION", "NO"))

    def test_unseen_segment_falls_back_to_global(self):
        s = build_samples(self._corpus(), now=NOW)
        ens = fit_ensemble(s)
        odd = s[0]
        odd.features["duration_s"] = 999999.0   # a tier nothing was fitted for
        self.assertIsInstance(ens.predict(odd), float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
