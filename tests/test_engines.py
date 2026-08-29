"""Tests for the virality, RSS, RPM and editor engines."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from ytengine.editor import (CAPTION_Y, HEIGHT, STYLES, WIDTH, EditSpec, build_ass,
                             cues_from_words, ffmpeg_command, _ts)
from ytengine.rpm import SHORTS_RPM_FACTOR, compare, rpm_for, value_of
from ytengine.rss import FeedItem, parse_feed
from ytengine.virality import (BAND_NAMES, Prediction, ViralityReport, _scores_from,
                               band_of, pairwise_win_rate)

NOW = datetime(2026, 8, 1, tzinfo=timezone.utc)

FEED_XML = """<?xml version="1.0"?><feed>
<entry><id>x</id><yt:videoId>abc123</yt:videoId><yt:channelId>UC1</yt:channelId>
<title>A &amp; B test</title><author><name>Chan</name></author>
<published>2026-07-30T10:00:00+00:00</published>
<media:group><media:community><media:statistics views="1500"/>
<media:starRating count="42"/></media:community></media:group></entry>
<entry><id>y</id><yt:videoId>def456</yt:videoId><yt:channelId>UC1</yt:channelId>
<title>Second</title><author><name>Chan</name></author>
<published>2026-07-29T10:00:00+00:00</published></entry>
</feed>"""


class TestWinRate(unittest.TestCase):
    """Win rate is the headline metric, so its null must be exactly 50%."""

    def test_perfect_ranking(self):
        wr, n = pairwise_win_rate([1, 2, 3, 4], [10, 20, 30, 40])
        self.assertEqual(wr, 1.0)
        self.assertEqual(n, 6)

    def test_inverted_ranking(self):
        wr, _ = pairwise_win_rate([4, 3, 2, 1], [10, 20, 30, 40])
        self.assertEqual(wr, 0.0)

    def test_ties_in_truth_are_skipped_not_counted_as_wins(self):
        """Counting ties as correct inflates the rate on any corpus with
        repeated view counts - and repeated small view counts are everywhere."""
        wr, n = pairwise_win_rate([1, 2, 3], [5, 5, 5])
        self.assertEqual(n, 0)
        self.assertEqual(wr, 0.0)

    def test_constant_prediction_scores_exactly_a_coin_flip(self):
        """A model that says the same thing about every video knows nothing,
        and must score the 50% null - not 100%, which is what counting tied
        predictions as agreement produces."""
        wr, n = pairwise_win_rate([1, 1, 1, 1], [10, 20, 30, 40])
        self.assertEqual(n, 6)
        self.assertEqual(wr, 0.5)

    def test_partial_ties_get_half_credit(self):
        wr, _ = pairwise_win_rate([1, 1, 3], [10, 20, 30])
        self.assertAlmostEqual(wr, (0.5 + 1 + 1) / 3)

    def test_degenerate_inputs(self):
        self.assertEqual(pairwise_win_rate([], []), (0.0, 0))
        self.assertEqual(pairwise_win_rate([1], [1]), (0.0, 0))


class TestBandsAndScores(unittest.TestCase):
    def test_band_boundaries(self):
        self.assertEqual(band_of(4_999), "FLOP")
        self.assertEqual(band_of(5_000), "LOW")
        self.assertEqual(band_of(15_000), "JAIL")
        self.assertEqual(band_of(50_000), "MID")
        self.assertEqual(band_of(500_000), "BIG")
        self.assertEqual(band_of(2_000_000), "VIRAL")

    def test_scores_are_percentiles_in_range(self):
        ref = list(range(100))
        s = _scores_from([0, 50, 99, 200, -5], ref)
        self.assertTrue(all(0 <= x <= 100 for x in s))
        self.assertEqual(s[4], 0)
        self.assertEqual(s[3], 100)

    def test_scores_are_monotonic_in_prediction(self):
        ref = list(range(100))
        s = _scores_from([10, 20, 30], ref)
        self.assertEqual(s, sorted(s))

    def test_empty_reference_does_not_crash(self):
        self.assertEqual(_scores_from([1, 2], []), [50, 50])

    def test_prediction_scores_itself_against_truth(self):
        p = Prediction("v", "t", 80, "MID", 60_000, actual_views=70_000)
        self.assertTrue(p.correct)
        self.assertEqual(p.bands_off, 0)
        p2 = Prediction("v", "t", 80, "VIRAL", 3_000_000, actual_views=1_000)
        self.assertFalse(p2.correct)
        self.assertEqual(p2.bands_off, 5)

    def test_unscored_prediction_reports_none(self):
        p = Prediction("v", "t", 50, "LOW", 9_000)
        self.assertIsNone(p.correct)


class TestVerdict(unittest.TestCase):
    def _r(self, **kw):
        base = dict(n_train=1, n_test=1, win_rate=0.8, win_rate_pairs=10, exact=0.6,
                    within_one=0.9, majority_exact=0.5, prior_exact=0.59, spearman=0.8)
        base.update(kw)
        return ViralityReport(**base)

    def test_coin_flip_is_called_out(self):
        self.assertIn("COIN FLIP", self._r(win_rate=0.51).verdict)

    def test_ranking_without_forecasting_is_called_out(self):
        self.assertIn("RANKS BUT DOES NOT FORECAST", self._r(exact=0.59).verdict)

    def test_beating_both_is_accepted(self):
        self.assertIn("USABLE", self._r(exact=0.75, prior_exact=0.6).verdict)


class TestRSS(unittest.TestCase):
    def test_parses_entries(self):
        items = parse_feed(FEED_XML)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].video_id, "abc123")
        self.assertEqual(items[0].views, 1500)
        self.assertEqual(items[0].likes, 42)

    def test_unescapes_titles(self):
        self.assertEqual(parse_feed(FEED_XML)[0].title, "A & B test")

    def test_missing_statistics_are_none_not_zero(self):
        """None means 'the feed did not say'; zero would mean 'nobody watched'."""
        self.assertIsNone(parse_feed(FEED_XML)[1].views)
        self.assertIsNone(parse_feed(FEED_XML)[1].views_per_hour(NOW))

    def test_velocity_floors_age_at_one_hour(self):
        i = FeedItem("v", "c", "C", "t", NOW - timedelta(minutes=6), views=600)
        self.assertLessEqual(i.views_per_hour(NOW), 600)

    def test_malformed_feed_yields_nothing(self):
        self.assertEqual(parse_feed("<feed><entry>junk</entry></feed>"), [])
        self.assertEqual(parse_feed(""), [])


class TestRPM(unittest.TestCase):
    def test_known_and_unknown_niches(self):
        self.assertGreater(rpm_for("personal finance"), rpm_for("gaming"))
        self.assertEqual(rpm_for("completely made up niche"), 4.0)

    def test_substring_match(self):
        self.assertEqual(rpm_for("beginner GAMING tips"), rpm_for("gaming"))

    def test_shorts_earn_a_fraction_of_long_form(self):
        s = value_of("gaming", 1_000_000, is_shorts=True)
        l = value_of("gaming", 1_000_000, is_shorts=False)
        self.assertAlmostEqual(s.monthly_revenue / l.monthly_revenue, SHORTS_RPM_FACTOR, places=6)

    def test_ranking_is_by_revenue_not_views(self):
        vals = compare(["gaming", "personal finance"], 1_000_000)
        self.assertEqual(vals[0].niche, "personal finance")

    def test_views_needed_scales_inversely_with_rpm(self):
        cheap = value_of("gaming", 1_000).views_needed_for(2000)
        rich = value_of("personal finance", 1_000).views_needed_for(2000)
        self.assertGreater(cheap, rich)


class TestEditor(unittest.TestCase):
    def _spec(self, **kw):
        base = dict(source="in.mp4", start_s=10.0, end_s=30.0, style=STYLES["punch"])
        base.update(kw)
        return EditSpec(**base)

    def test_ass_timestamps(self):
        self.assertEqual(_ts(0), "0:00:00.00")
        self.assertEqual(_ts(3725.5), "1:02:05.50")
        self.assertEqual(_ts(-5), "0:00:00.00")

    def test_cues_are_rebased_to_clip_start(self):
        """A word spoken at 3721s in the stream must appear at 1s in a clip
        that starts at 3720s, or every caption is minutes out of sync."""
        words = [(3720.0, 3720.5, "a"), (3720.5, 3721.0, "b")]
        cues = cues_from_words(words, STYLES["karaoke"], offset_s=3720.0)
        self.assertAlmostEqual(cues[0].start_s, 0.0)
        self.assertAlmostEqual(cues[1].start_s, 0.5)

    def test_cue_grouping_respects_words_per_cue(self):
        words = [(float(i), i + 0.5, f"w{i}") for i in range(9)]
        self.assertEqual(len(cues_from_words(words, STYLES["punch"], 0.0)), 3)  # 3/cue

    def test_uppercase_style_applies(self):
        c = cues_from_words([(0, 1, "quiet")], STYLES["punch"], 0.0)
        self.assertEqual(c[0].text, "QUIET")
        c2 = cues_from_words([(0, 1, "quiet")], STYLES["docu"], 0.0)
        self.assertEqual(c2[0].text, "quiet")

    def test_ass_has_required_sections_and_resolution(self):
        a = build_ass(self._spec())
        for token in ("[Script Info]", "[V4+ Styles]", "[Events]",
                      f"PlayResX: {WIDTH}", f"PlayResY: {HEIGHT}"):
            self.assertIn(token, a)

    def test_captions_sit_above_the_player_chrome(self):
        style_line = [l for l in build_ass(self._spec()).splitlines()
                      if l.startswith("Style: Caption")][0]
        margin_v = int(style_line.split(",")[-2])
        self.assertAlmostEqual(margin_v, int(HEIGHT * (1 - CAPTION_Y)), delta=2)
        self.assertGreater(margin_v, HEIGHT * 0.18, "caption would sit under the UI")

    def test_hook_only_when_given(self):
        self.assertNotIn("Style: Hook", build_ass(self._spec()))
        self.assertIn("Style: Hook", build_ass(self._spec(hook="WAIT")))

    def test_braces_escaped_so_they_are_not_override_blocks(self):
        spec = self._spec(hook="{drop}")
        self.assertNotIn("{drop}", build_ass(spec))

    def test_cues_outside_the_window_are_dropped(self):
        spec = self._spec(end_s=15.0)  # 5s clip
        spec.cues = cues_from_words([(100.0, 101.0, "late")], STYLES["punch"], 10.0)
        self.assertNotIn("LATE", build_ass(spec))

    def test_ffmpeg_command_fills_vertical_without_letterboxing(self):
        cmd = ffmpeg_command(self._spec(), "c.ass", "o.mp4")
        vf = cmd[cmd.index("-vf") + 1]
        self.assertIn("force_original_aspect_ratio=increase", vf)
        self.assertIn(f"crop={WIDTH}:{HEIGHT}", vf)
        self.assertIn("subtitles=", vf)

    def test_ffmpeg_seeks_before_input_for_speed(self):
        cmd = ffmpeg_command(self._spec(), "c.ass", "o.mp4")
        self.assertLess(cmd.index("-ss"), cmd.index("-i"))

    def test_crop_modes(self):
        for mode in ("center", "left", "right"):
            cmd = ffmpeg_command(self._spec(crop_mode=mode), "c.ass", "o.mp4")
            self.assertTrue(any("crop=" in a for a in cmd))


if __name__ == "__main__":
    unittest.main(verbosity=2)
