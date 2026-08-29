"""Tests for the signal engines and the craft (pre-render) engines."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from ytengine.craft import (IDEAL_MAX_S, MIN_CUE_S, SAFE_WIDTH, estimate_width_px,
                            hook_candidates, pacing, plan_loop, suggest_style, validate)
from ytengine.editor import STYLES, Cue, EditSpec
from ytengine.models import Video
from ytengine.signals import (CorpusIndex, cadence, fatigue, format_signals, jaccard,
                              momentum, saturation, tokens)

NOW = datetime(2026, 8, 1, tzinfo=timezone.utc)


def vid(vid_id="v", title="a title", days_ago=100, views=1000, tags=()):
    return Video(video_id=vid_id, channel_id="c", channel_title="C", title=title,
                 description="", published_at=NOW - timedelta(days=days_ago),
                 duration_s=40, views=views, likes=1, comments=1, tags=list(tags))


class TestMomentum(unittest.TestCase):
    def test_rising_channel_scores_positive(self):
        m = momentum([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
        self.assertGreater(m["mom_ratio"], 0)
        self.assertGreater(m["mom_slope"], 0)

    def test_declining_channel_scores_negative(self):
        m = momentum([3.0, 3.0, 3.0, 2.0, 2.0, 2.0])
        self.assertLess(m["mom_ratio"], 0)
        self.assertLess(m["mom_slope"], 0)

    def test_flat_channel_is_near_zero(self):
        m = momentum([2.5] * 8)
        self.assertAlmostEqual(m["mom_ratio"], 0.0, places=6)
        self.assertAlmostEqual(m["mom_slope"], 0.0, places=6)

    def test_two_channels_same_median_differ_on_momentum(self):
        """The whole justification for this family: a static median cannot tell
        a channel that tripled last month from one flat for a year."""
        rising = momentum([1.0, 1.5, 2.0, 2.5, 3.0])
        flat = momentum([2.0, 2.0, 2.0, 2.0, 2.0])
        self.assertNotAlmostEqual(rising["mom_slope"], flat["mom_slope"])

    def test_too_little_history_is_zeros_not_noise(self):
        self.assertEqual(set(momentum([1.0]).values()), {0.0})


class TestFatigueAndCadence(unittest.TestCase):
    def test_repeated_title_scores_low_novelty(self):
        f = fatigue("my sourdough starter day 3", ["my sourdough starter day 3"] * 5)
        self.assertLess(f["fatigue_novelty"], 0.2)
        self.assertGreater(f["fatigue_repeat_rate"], 0.9)

    def test_fresh_title_scores_high_novelty(self):
        f = fatigue("completely unrelated topic here",
                    ["my sourdough starter", "bread proofing tips"])
        self.assertGreater(f["fatigue_novelty"], 0.8)

    def test_no_history_is_maximally_novel(self):
        self.assertEqual(fatigue("anything", [])["fatigue_novelty"], 1.0)

    def test_regular_cadence_scores_higher_than_erratic(self):
        base = NOW - timedelta(days=200)
        regular = [base + timedelta(days=7 * i) for i in range(10)]
        erratic = [base + timedelta(days=d) for d in (0, 1, 40, 42, 90, 91, 92, 150, 151, 190)]
        r = cadence(NOW, regular)["cad_regularity"]
        e = cadence(NOW, erratic)["cad_regularity"]
        self.assertGreater(r, e)

    def test_gap_is_clipped(self):
        """A three-year absence is not forty times worse than a thirty-day one,
        and left unclipped it dominates the standardised feature."""
        self.assertLessEqual(cadence(NOW, [NOW - timedelta(days=1000)])["cad_gap_days"], 90.0)

    def test_uploads_in_last_30_days(self):
        dates = [NOW - timedelta(days=d) for d in (5, 10, 20, 200)]
        self.assertEqual(cadence(NOW, dates)["cad_uploads_30d"], 3.0)


class TestSaturation(unittest.TestCase):
    def test_counts_only_earlier_videos(self):
        """A video must never learn how crowded its topic later became."""
        vids = [vid(f"a{i}", "sourdough bread", days_ago=50) for i in range(5)]
        vids += [vid(f"b{i}", "sourdough bread", days_ago=1) for i in range(20)]
        idx = CorpusIndex.build(vids)
        early = saturation("sourdough bread", NOW - timedelta(days=45), idx)
        late = saturation("sourdough bread", NOW, idx)
        self.assertLess(early["sat_topic_density"], late["sat_topic_density"])

    def test_unknown_topic_is_zero(self):
        idx = CorpusIndex.build([vid()])
        self.assertEqual(saturation("zzz qqq", NOW, idx)["sat_topic_density"], 0.0)

    def test_empty_title_does_not_crash(self):
        self.assertEqual(saturation("", NOW, CorpusIndex.build([vid()]))["sat_topic_density"], 0.0)


class TestFormatSignals(unittest.TestCase):
    def test_counts_hashtags_and_tags(self):
        f = format_signals(vid(title="cool #shorts #viral", tags=("a", "b")), [])
        self.assertEqual(f["fmt_hashtags"], 2.0)
        self.assertEqual(f["fmt_tag_count"], 2.0)

    def test_detects_a_reused_title(self):
        prior = [vid(title="Same Title")]
        self.assertEqual(format_signals(vid(title="same title"), prior)["fmt_title_is_reused"], 1.0)

    def test_jaccard_bounds(self):
        self.assertEqual(jaccard(set(), {"a"}), 0.0)
        self.assertEqual(jaccard({"a"}, {"a"}), 1.0)


class TestPacing(unittest.TestCase):
    def test_coverage_never_exceeds_one_with_overlaps(self):
        """Overlapping cues counted separately would report >100% coverage."""
        cues = [Cue(0, 5, "a"), Cue(1, 6, "b"), Cue(2, 7, "c")]
        self.assertLessEqual(pacing(cues, 10.0).covered_fraction, 1.0)

    def test_finds_the_longest_silence(self):
        p = pacing([Cue(0, 1, "a"), Cue(9, 10, "b")], 10.0)
        self.assertAlmostEqual(p.longest_gap_s, 8.0)

    def test_trailing_silence_counts_as_a_gap(self):
        self.assertAlmostEqual(pacing([Cue(0, 1, "a")], 10.0).longest_gap_s, 9.0)

    def test_zero_duration_is_safe(self):
        self.assertEqual(pacing([], 0.0).cue_count, 0)


class TestValidate(unittest.TestCase):
    def _spec(self, **kw):
        base = dict(source="s.mp4", start_s=0.0, end_s=25.0, style=STYLES["punch"],
                    cues=[Cue(3.0, 5.0, "HELLO THERE")], hook="WAIT")
        base.update(kw)
        return EditSpec(**base)

    def test_missing_captions_is_an_error(self):
        issues = validate(self._spec(cues=[]))
        self.assertTrue(any(i.severity == "error" and i.where == "captions" for i in issues))

    def test_over_long_clip_is_an_error(self):
        self.assertTrue(any(i.severity == "error" for i in validate(self._spec(end_s=200.0))))

    def test_wide_caption_is_flagged(self):
        wide = "THIS IS A VERY LONG CAPTION LINE THAT WILL CERTAINLY OVERFLOW THE FRAME"
        self.assertGreater(estimate_width_px(wide, STYLES["punch"]), SAFE_WIDTH)
        issues = validate(self._spec(cues=[Cue(3, 6, wide)]))
        self.assertTrue(any("safe area" in i.message for i in issues))

    def test_flickering_cue_is_flagged(self):
        issues = validate(self._spec(cues=[Cue(3.0, 3.0 + MIN_CUE_S / 2, "HI")]))
        self.assertTrue(any("flicker" in i.message for i in issues))

    def test_unreadably_fast_cue_is_flagged(self):
        issues = validate(self._spec(cues=[Cue(3.0, 3.4, "A VERY LONG LINE TO READ QUICKLY")]))
        self.assertTrue(any("to read" in i.message for i in issues))

    def test_hook_colliding_with_captions_is_flagged(self):
        issues = validate(self._spec(cues=[Cue(0.5, 2.0, "EARLY")]))
        self.assertTrue(any(i.where == "hook" and "share the screen" in i.message
                            for i in issues))

    def test_missing_hook_is_flagged(self):
        self.assertTrue(any(i.where == "hook" for i in validate(self._spec(hook=""))))

    def test_clean_spec_has_no_errors(self):
        spec = self._spec(cues=[Cue(3.0, 5.0, "SHORT LINE"), Cue(5.0, 7.0, "NEXT LINE")])
        self.assertEqual([i for i in validate(spec) if i.severity == "error"], [])


class TestLoopAndStyle(unittest.TestCase):
    def test_dead_air_breaks_the_loop(self):
        spec = EditSpec("s.mp4", 0, 30.0, STYLES["punch"], cues=[Cue(0, 5, "A")])
        plan = plan_loop(spec)
        self.assertFalse(plan.loops_cleanly)
        self.assertIn("dead air", plan.advice)

    def test_cutting_on_the_final_word_is_flagged(self):
        spec = EditSpec("s.mp4", 0, 5.0, STYLES["punch"], cues=[Cue(0, 5.0, "A")])
        self.assertFalse(plan_loop(spec).loops_cleanly)

    def test_tight_tail_loops_cleanly(self):
        spec = EditSpec("s.mp4", 0, 5.5, STYLES["punch"], cues=[Cue(0, 5.0, "A")])
        self.assertTrue(plan_loop(spec).loops_cleanly)

    def test_dense_speech_gets_a_smaller_style(self):
        dense = [Cue(i * 0.5, i * 0.5 + 0.5, "word word word word") for i in range(40)]
        self.assertEqual(suggest_style(dense, 20.0).name, "docu")

    def test_sparse_punchy_speech_gets_the_heavy_style(self):
        sparse = [Cue(0, 2, "WOW"), Cue(5, 7, "NO")]
        self.assertIn(suggest_style(sparse, 30.0).name, ("punch", "karaoke"))

    def test_hooks_fit_the_readable_window(self):
        for h in hook_candidates("okay so this is the thing that happened"):
            self.assertLessEqual(len(h), 24)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestFontSubstitution(unittest.TestCase):
    """Regression: a substituted font breaks the overflow check.

    libass silently swaps in a different face when the requested one is
    missing. If the width estimate keeps using the *requested* font's metrics,
    a narrow spec (Impact, 0.46) that renders wide (DejaVu Sans, 0.62) passes
    the overflow check while visibly running off the frame. Measured on a real
    1080x1920 render: "WALKED INTO HIS" was predicted at 662px and rendered
    near 850px.
    """

    def test_wider_substitute_gives_a_wider_estimate(self):
        from ytengine.craft import estimate_width_px
        t = "WALKED INTO HIS"
        narrow = estimate_width_px(t, STYLES["punch"])                   # Impact
        wide = estimate_width_px(t, STYLES["punch"], "DejaVu Sans")
        self.assertGreater(wide, narrow * 1.25)

    def test_resolve_font_reports_substitution(self):
        from ytengine.editor import resolve_font
        got, exact = resolve_font("Impact", {"DejaVu Sans"})
        self.assertEqual(got, "DejaVu Sans")
        self.assertFalse(exact)

    def test_resolve_font_prefers_an_exact_match(self):
        from ytengine.editor import resolve_font
        got, exact = resolve_font("Impact", {"Impact", "DejaVu Sans"})
        self.assertEqual(got, "Impact")
        self.assertTrue(exact)

    def test_unknown_fontconfig_does_not_claim_substitution(self):
        from ytengine.editor import resolve_font
        got, exact = resolve_font("Impact", set())
        self.assertEqual((got, exact), ("Impact", True))

    def test_validate_warns_about_a_missing_font(self):
        spec = EditSpec("s.mp4", 0.0, 25.0, STYLES["punch"],
                        cues=[Cue(3.0, 5.0, "HI")], hook="WAIT")
        issues = validate(spec, rendered_font=None)
        # This container has no Impact, so the warning must appear.
        from ytengine.editor import resolve_font
        if not resolve_font("Impact")[1]:
            self.assertTrue(any(i.where == "font" for i in issues))

    def test_explicit_rendered_font_skips_the_lookup(self):
        spec = EditSpec("s.mp4", 0.0, 25.0, STYLES["punch"],
                        cues=[Cue(3.0, 5.0, "HI")], hook="WAIT")
        self.assertFalse(any(i.where == "font" for i in validate(spec, rendered_font="Impact")))
