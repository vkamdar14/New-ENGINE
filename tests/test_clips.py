"""Clip-miner tests.

The load-bearing test plants known moments in a synthetic comment stream and
asserts the miner recovers them, plus a negative control on uniformly-scattered
timestamps where it must find nothing. A peak-finder that always returns twelve
"top moments" is worthless - it would send you to edit six clips that nobody
actually marked.
"""

from __future__ import annotations

import random
import unittest

from ytengine.clips import (LEAD_SECONDS, MAX_CLIP_S, Mention, find_clips,
                            mentions_from_comments, parse_timestamps, render_clips)


def comment(text, author="a", likes=0):
    return {"snippet": {"topLevelComment": {"snippet": {
        "textDisplay": text, "authorDisplayName": author, "likeCount": likes}}}}


def fmt(t, style="auto"):
    """Format seconds the way viewers actually type timestamps.

    Real comment threads mix both conventions on the same long video: some
    people write 1:23:20, others write 83:20. The miner has to handle both.
    """
    m, s = divmod(int(t), 60)
    if style == "mmss" or (style == "auto" and m < 60):
        return f"{m}:{s:02d}"
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def synth(peaks, n_per_peak=25, noise=60, duration=7200, seed=0, jitter=4.0):
    """Comments clustering on `peaks`, plus scattered noise timestamps."""
    rng = random.Random(seed)
    out = []
    for pi, p in enumerate(peaks):
        for i in range(n_per_peak):
            t = max(0, min(duration, rng.gauss(p, jitter)))
            # Alternate conventions, as a real thread does.
            style = "mmss" if i % 3 == 0 else "auto"
            out.append(comment(f"{fmt(t, style)} lol", author=f"p{pi}u{i}",
                               likes=rng.randint(0, 5)))
    for i in range(noise):
        out.append(comment(fmt(rng.uniform(0, duration)), author=f"n{i}"))
    return out


class TestParsing(unittest.TestCase):
    def test_common_forms(self):
        self.assertEqual(parse_timestamps("3:47"), [227.0])
        self.assertEqual(parse_timestamps("1:02:03"), [3723.0])
        self.assertEqual(parse_timestamps("at 0:05 and 12:34"), [5.0, 754.0])

    def test_rejects_impossible_clock_values(self):
        self.assertEqual(parse_timestamps("99:99"), [])
        self.assertEqual(parse_timestamps("7:75"), [])

    def test_ignores_bare_numbers(self):
        self.assertEqual(parse_timestamps("part 2 was better, 1000 views"), [])

    def test_out_of_range_marks_are_dropped(self):
        # "$5:00" and "2:1 ratio" parse as clock-shaped; the duration bound is
        # what keeps prices and ratios out of the ranking.
        items = [comment("costs $5:00 for the set")]
        self.assertEqual(mentions_from_comments(items, duration_s=120), [])
        self.assertEqual(len(mentions_from_comments(items, duration_s=600)), 1)

    def test_handles_missing_fields(self):
        self.assertEqual(mentions_from_comments([{}, {"snippet": {}}], 600), [])


class TestFindClips(unittest.TestCase):
    def test_recovers_planted_moments(self):
        peaks = [600.0, 2400.0, 5000.0]
        m = mentions_from_comments(synth(peaks), duration_s=7200)
        found = find_clips(m, duration_s=7200, max_clips=6)
        self.assertGreaterEqual(len(found), 3)
        # Every planted peak should sit inside one of the top windows.
        for p in peaks:
            self.assertTrue(
                any(c.start_s <= p <= c.end_s + 5 for c in found[:5]),
                f"missed planted peak at {p}s",
            )

    def test_finds_nothing_in_uniform_noise(self):
        """Negative control: scattered timestamps are not moments."""
        m = mentions_from_comments(synth([], n_per_peak=0, noise=300), duration_s=7200)
        found = find_clips(m, duration_s=7200, min_authors=8)
        self.assertLessEqual(len(found), 1, f"invented {len(found)} moments from noise")

    def test_windows_open_before_the_marked_second(self):
        """Viewers mark the payoff; a clip opening there has no setup."""
        m = mentions_from_comments(synth([1200.0]), duration_s=7200)
        c = find_clips(m, duration_s=7200)[0]
        self.assertLess(c.start_s, c.peak_s)
        self.assertAlmostEqual(c.peak_s - c.start_s, LEAD_SECONDS, delta=3.0)

    def test_one_moment_is_not_returned_as_many(self):
        """Non-max suppression: nearby peaks are one joke, not six clips."""
        m = mentions_from_comments(synth([1200.0], n_per_peak=60, jitter=6.0), duration_s=7200)
        found = find_clips(m, duration_s=7200, max_clips=12)
        overlapping = [
            (a, b) for i, a in enumerate(found) for b in found[i + 1:]
            if not (a.end_s <= b.start_s or a.start_s >= b.end_s)
        ]
        self.assertEqual(overlapping, [])

    def test_one_person_spamming_is_not_a_moment(self):
        """Unique authors, not raw mentions - otherwise one fan sets the agenda."""
        items = [comment("5:00 best part", author="superfan") for _ in range(50)]
        m = mentions_from_comments(items, duration_s=7200)
        self.assertEqual(find_clips(m, duration_s=7200, min_authors=3), [])

    def test_clips_never_exceed_the_cap(self):
        m = mentions_from_comments(synth([600.0, 2400.0]), duration_s=7200)
        self.assertTrue(all(c.duration_s <= MAX_CLIP_S for c in find_clips(m, 7200)))

    def test_clips_stay_inside_the_video(self):
        # A moment marked at 0:03 must not produce a negative start.
        m = mentions_from_comments(synth([3.0, 7195.0]), duration_s=7200)
        for c in find_clips(m, duration_s=7200):
            self.assertGreaterEqual(c.start_s, 0.0)
            self.assertLessEqual(c.end_s, 7200)

    def test_stronger_moment_outranks_weaker(self):
        m = mentions_from_comments(
            synth([600.0], n_per_peak=60, seed=1) + synth([3000.0], n_per_peak=12, seed=2),
            duration_s=7200)
        found = find_clips(m, duration_s=7200)
        self.assertLess(abs(found[0].peak_s - 600.0), 30)

    def test_likes_are_log_weighted(self):
        """One viral comment must not outvote many independent viewers."""
        viral = [comment("10:00 this", author="v", likes=100_000)]
        crowd = [comment("20:00 this", author=f"u{i}", likes=1) for i in range(30)]
        found = find_clips(mentions_from_comments(viral + crowd, 7200), 7200, min_authors=2)
        self.assertLess(abs(found[0].peak_s - 1200.0), 30, "one viral comment outvoted 30 people")

    def test_empty_and_degenerate_inputs(self):
        self.assertEqual(find_clips([], 7200), [])
        self.assertEqual(find_clips([Mention(10, "a", 0, "")], 0), [])

    def test_confidence_reflects_sample_size(self):
        m = mentions_from_comments(synth([600.0], n_per_peak=40), duration_s=7200)
        self.assertEqual(find_clips(m, 7200)[0].confidence, "high")

    def test_render_is_honest_when_empty(self):
        self.assertIn("No timestamp clusters", render_clips([]))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestLongVideoTimestamps(unittest.TestCase):
    """Regression: long-stream timestamp conventions.

    On a four-hour VOD viewers mix "1:23:20" and "83:20" freely. Capping the
    minute field at 59 drops every mark of the second kind - and does it
    silently, on exactly the long videos clip mining exists for.
    """

    def test_minutes_past_59_are_parsed(self):
        self.assertEqual(parse_timestamps("83:20"), [5000.0])
        self.assertEqual(parse_timestamps("99:59"), [5999.0])

    def test_three_digit_minutes_are_deliberately_not_parsed(self):
        """The boundary is two digits, and it is a choice rather than an
        oversight. Past 99 minutes essentially everyone writes 2:00:00, so
        accepting "120:00" would buy almost no real marks while admitting
        scores, prices and model numbers into the ranking."""
        self.assertEqual(parse_timestamps("120:00"), [])
        self.assertEqual(parse_timestamps("2:00:00"), [7200.0])

    def test_both_conventions_agree(self):
        self.assertEqual(parse_timestamps("1:23:20"), parse_timestamps("83:20"))

    def test_duration_bound_is_what_guards_precision(self):
        # The loose minute field admits non-timestamps; the video's length is
        # what rejects them.
        items = [comment("83:20 great")]
        self.assertEqual(mentions_from_comments(items, duration_s=600), [])
        self.assertEqual(len(mentions_from_comments(items, duration_s=7200)), 1)

    def test_mixed_convention_thread_finds_one_moment(self):
        items = ([comment("83:20 lol", author=f"a{i}") for i in range(10)]
                 + [comment("1:23:20 lol", author=f"b{i}") for i in range(10)])
        found = find_clips(mentions_from_comments(items, 7200), 7200, min_authors=5)
        self.assertEqual(len(found), 1, "same moment split across two conventions")
        self.assertEqual(found[0].unique_authors, 20)
