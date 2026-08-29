"""Backtest tests, which are mostly leakage guards.

Every bug in a backtest makes the score go UP. That asymmetry is why these
tests exist: a broken metric looks like success, so nothing about the output
warns you. The guards below pin the three ways this file could quietly start
lying - outcome features, future information, and a non-temporal split.
"""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone

from ytengine.backtest import (BUCKET_NAMES, CHANNEL_PRIOR_FEATURES, FEATURES,
                               MATURITY_DAYS, Report, bucket_of, build_samples,
                               build_vocabulary, run_backtest, temporal_split,
                               title_tokens)
from ytengine.models import Video

NOW = datetime(2026, 8, 1, tzinfo=timezone.utc)


def vid(vid_id, channel, views, age_days, title="a title", duration=40):
    return Video(video_id=vid_id, channel_id=channel, channel_title="C", title=title,
                 description="", published_at=NOW - timedelta(days=age_days),
                 duration_s=duration, views=views, likes=1, comments=1)


def build_samples_now(videos):
    """build_samples with the clock pinned, so fixtures do not drift."""
    return build_samples(videos, now=NOW)


def channel_run(cid, n, views=5000, start_age=800, step=7, title="a title"):
    return [vid(f"{cid}-{i}", cid, views, start_age - i * step, title) for i in range(n)]


class TestBuckets(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(bucket_of(0), "FLOP")
        self.assertEqual(bucket_of(4_999), "FLOP")
        self.assertEqual(bucket_of(5_000), "JAIL")
        self.assertEqual(bucket_of(24_999), "JAIL")
        self.assertEqual(bucket_of(250_000), "VIRAL")
        self.assertEqual(bucket_of(10 ** 12), "VIRAL")


class TestNoOutcomeLeakage(unittest.TestCase):
    """The most basic way to fake a good score: predict views from views."""

    def test_no_outcome_field_is_a_feature(self):
        banned = {"views", "likes", "comments", "log_views", "engagement",
                  "view_count", "like_count", "multiplier"}
        for f in FEATURES:
            self.assertNotIn(f, banned, f"{f} is an outcome, not an input")

    def test_features_are_all_knowable_before_publishing(self):
        s = build_samples_now(channel_run("c", 12))
        self.assertTrue(s)
        # Every feature must be computable from title/duration/tags/timing/history.
        for f in FEATURES:
            self.assertIn(f, s[0].features)


class TestChannelPriorCausality(unittest.TestCase):
    def test_prior_uses_only_earlier_videos(self):
        """A channel's later hits must not inform an earlier video's prior."""
        vids = channel_run("c", 6, views=1000, start_age=400, step=30)
        vids.append(vid("c-late", "c", 10_000_000, 40))  # a monster, published last
        samples = {s.video.video_id: s for s in build_samples_now(vids)}
        for s in samples.values():
            if s.video.video_id != "c-late":
                # 10M would drag any prior that included it far above log10(1000)=3.
                self.assertLess(s.features["chan_prior_log_median"], 4.0)

    def test_prior_excludes_videos_not_yet_matured_at_publish_time(self):
        """A video published two days earlier had not settled when this one went
        out, so its count was not knowable either."""
        vids = [vid(f"c-{i}", "c", 1000, 500 - i) for i in range(6)]  # 1 day apart
        self.assertEqual(build_samples_now(vids), [],
                         "used neighbours that had not matured at publish time")

    def test_channels_without_history_are_dropped_not_imputed(self):
        # Filling a missing prior with the corpus average would leak the global
        # distribution into a per-channel feature.
        self.assertEqual(build_samples_now(channel_run("c", 2)), [])

    def test_immature_videos_are_never_labelled(self):
        vids = channel_run("c", 10, start_age=400, step=30) + [vid("fresh", "c", 5, 2)]
        # `now` must be pinned: left to the real clock, a fixture video that was
        # "2 days old" when written silently matures as the calendar advances
        # and the test stops testing anything.
        self.assertNotIn("fresh", {s.video.video_id for s in build_samples(vids, now=NOW)})


class TestTemporalSplit(unittest.TestCase):
    def test_train_is_strictly_before_test(self):
        s = build_samples_now(channel_run("c", 40, start_age=1200, step=25))
        train, test = temporal_split(s, 0.25)
        self.assertTrue(train and test)
        self.assertLessEqual(train[-1].video.published_at, test[0].video.published_at)

    def test_split_is_not_random(self):
        s = build_samples_now(channel_run("c", 40, start_age=1200, step=25))
        a, _ = temporal_split(s, 0.25)
        b, _ = temporal_split(s, 0.25)
        self.assertEqual([x.video.video_id for x in a], [x.video.video_id for x in b])


class TestVocabulary(unittest.TestCase):
    def test_built_from_training_slice_only(self):
        """Vocabulary from the whole corpus encodes which words exist in the
        test period - a leak that raises the score without helping anything."""
        train = build_samples_now(channel_run("c", 30, start_age=1200, step=25,
                                          title="common word here"))
        vocab = build_vocabulary(train, min_uses=2)
        self.assertIn("common", vocab)
        self.assertNotIn("neverseen", vocab)

    def test_respects_min_uses(self):
        train = build_samples_now(channel_run("c", 30, start_age=1200, step=25))
        self.assertEqual(build_vocabulary(train, min_uses=10_000), [])

    def test_tokenizer_drops_short_words_and_case(self):
        # 3-char minimum: "it", "of" and "a" are stop-word-shaped noise that
        # would occupy vocabulary slots without carrying topic.
        self.assertEqual(title_tokens("The BIG a of It"), {"the", "big"})


class TestBaselinesAndVerdict(unittest.TestCase):
    def test_verdict_calls_out_a_model_that_only_knows_the_channel(self):
        r = Report(n_train=1, n_test=1, features_used=[], accuracy=0.67,
                   majority_accuracy=0.35, prior_only_accuracy=0.669,
                   within_one_accuracy=0.9, mae_log=0.36, baseline_mae_log=0.32,
                   spearman=0.8, confusion={}, per_class={})
        self.assertIn("NO SIGNAL BEYOND CHANNEL HISTORY", r.verdict)

    def test_verdict_calls_out_a_model_that_beats_nothing(self):
        r = Report(n_train=1, n_test=1, features_used=[], accuracy=0.35,
                   majority_accuracy=0.35, prior_only_accuracy=0.30,
                   within_one_accuracy=0.9, mae_log=0.5, baseline_mae_log=0.5,
                   spearman=0.0, confusion={}, per_class={})
        self.assertIn("NO SIGNAL", r.verdict)

    def test_verdict_accepts_a_model_that_beats_both(self):
        r = Report(n_train=1, n_test=1, features_used=[], accuracy=0.80,
                   majority_accuracy=0.35, prior_only_accuracy=0.60,
                   within_one_accuracy=0.95, mae_log=0.2, baseline_mae_log=0.32,
                   spearman=0.9, confusion={}, per_class={})
        self.assertIn("REAL SIGNAL", r.verdict)


class TestRunBacktest(unittest.TestCase):
    def _corpus(self):
        out = []
        for c in range(12):
            out += channel_run(f"ch{c}", 45, views=1000 * (c + 1) * 3,
                               start_age=1400, step=28)
        return out

    def test_end_to_end(self):
        r = run_backtest(build_samples_now(self._corpus()), use_topics=False)
        self.assertIsNotNone(r)
        self.assertGreater(r.n_train, 0)
        self.assertGreater(r.n_test, 0)

    def test_confusion_totals_match_the_test_set(self):
        r = run_backtest(build_samples_now(self._corpus()), use_topics=False)
        total = sum(v for row in r.confusion.values() for v in row.values())
        self.assertEqual(total, r.n_test)

    def test_ablation_removes_channel_prior(self):
        s = build_samples_now(self._corpus())
        r = run_backtest(s, use_channel_prior=False, use_topics=False)
        for f in CHANNEL_PRIOR_FEATURES:
            self.assertNotIn(f, r.features_used)

    def test_returns_none_rather_than_a_meaningless_score(self):
        self.assertIsNone(run_backtest(build_samples_now(channel_run("c", 12))))

    def test_accuracy_and_within_one_are_consistent(self):
        r = run_backtest(build_samples_now(self._corpus()), use_topics=False)
        self.assertLessEqual(r.accuracy, r.within_one_accuracy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
