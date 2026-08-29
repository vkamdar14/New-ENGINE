"""Test suite. Stdlib unittest, no external deps.

The important tests here are not the unit checks on arithmetic - those catch
typos. The ones that matter assert that the engine *recovers a signal we
planted*, and that it *stays quiet when we plant nothing*. A scorer that
always finds a pattern is worse than useless, so both directions are tested.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from ytengine.audit import audit_cadence, audit_channel, duration_bucket
from ytengine.fixtures import make_corpus
from ytengine.metrics import (MATURITY_DAYS, AgeCurve, build_baseline,
                              score_channel, score_video)
from ytengine.models import Snapshot, Video, parse_duration, parse_ts
from ytengine.outliers import extract_patterns, find_opportunities
from ytengine.packaging import fit_packaging, title_features
from ytengine.stats import mad, median, quantile, r_squared, ridge_fit, spearman
from ytengine.store import SnapshotStore
from ytengine.trends import rank_rising, topic_trends, velocity_from_history

NOW = datetime(2025, 6, 1, tzinfo=timezone.utc)


def mkvideo(vid="v1", views=1000, age_days=60.0, title="t", duration=600,
            channel="UC1", likes=50, comments=5) -> Video:
    return Video(
        video_id=vid, channel_id=channel, channel_title="C", title=title,
        description="", published_at=NOW - timedelta(days=age_days),
        duration_s=duration, views=views, likes=likes, comments=comments,
    )


class TestModels(unittest.TestCase):
    def test_duration_parsing(self):
        self.assertEqual(parse_duration("PT12M30S"), 750)
        self.assertEqual(parse_duration("PT1H2M3S"), 3723)
        self.assertEqual(parse_duration("P1DT2H"), 93600)
        self.assertEqual(parse_duration("PT45S"), 45)

    def test_unparseable_duration_is_zero_not_crash(self):
        # Live streams and premieres report shapes we don't handle; a 500-video
        # pull must not die on one of them.
        for bad in ("", "garbage", "P", None and ""):
            self.assertEqual(parse_duration(bad), 0)

    def test_shorts_boundary(self):
        self.assertTrue(mkvideo(duration=180).is_short)
        self.assertFalse(mkvideo(duration=181).is_short)
        # Zero duration means unparseable, which must not be classed a Short.
        self.assertFalse(mkvideo(duration=0).is_short)

    def test_timestamp_parsing_is_utc_aware(self):
        self.assertIsNotNone(parse_ts("2024-01-01T00:00:00Z").tzinfo)

    def test_views_per_day_floors_age(self):
        # A three-hour-old video must not report an astronomical daily rate.
        v = mkvideo(views=100, age_days=0.01)
        self.assertLessEqual(v.views_per_day(NOW), 200)

    def test_engagement_rate_handles_zero_views(self):
        self.assertEqual(mkvideo(views=0).engagement_rate(), 0.0)

    def test_from_api_tolerates_missing_statistics(self):
        # Creators can hide like counts; the field simply vanishes.
        v = Video.from_api({
            "id": "x", "snippet": {"title": "T", "publishedAt": "2024-01-01T00:00:00Z"},
            "statistics": {"viewCount": "500"}, "contentDetails": {"duration": "PT5M"},
        })
        self.assertEqual((v.views, v.likes, v.duration_s), (500, 0, 300))


class TestStats(unittest.TestCase):
    def test_median_and_quantile(self):
        self.assertEqual(median([1, 2, 3, 4]), 2.5)
        self.assertEqual(median([]), 0.0)
        self.assertEqual(quantile([0, 10, 20, 30, 40], 0.5), 20.0)

    def test_mad_resists_outliers(self):
        # One 1000x value must not move the spread estimate much.
        self.assertLess(mad([1, 2, 3, 4, 5000]), 5.0)

    def test_ridge_recovers_known_coefficients(self):
        import random
        rng = random.Random(0)
        X = [[rng.random(), rng.random()] for _ in range(300)]
        y = [3 * a - 2 * b + 5 for a, b in X]
        c = ridge_fit(X, y, alpha=1e-8)
        self.assertAlmostEqual(c[0], 3.0, places=2)
        self.assertAlmostEqual(c[1], -2.0, places=2)
        self.assertAlmostEqual(c[2], 5.0, places=2)

    def test_ridge_shrinks_toward_zero_with_penalty(self):
        import random
        rng = random.Random(1)
        X = [[rng.random()] for _ in range(100)]
        y = [4 * r[0] for r in X]
        weak = ridge_fit(X, y, alpha=1e-8)[0]
        strong = ridge_fit(X, y, alpha=500.0)[0]
        self.assertLess(abs(strong), abs(weak))

    def test_ridge_survives_a_constant_column(self):
        # A feature nobody in the corpus uses has zero variance; the solver
        # must not divide by it.
        X = [[1.0, float(i)] for i in range(30)]
        y = [float(i) for i in range(30)]
        self.assertTrue(ridge_fit(X, y, alpha=1.0))

    def test_spearman_ties_and_degenerate(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [2, 4, 6]), 1.0, places=6)
        self.assertAlmostEqual(spearman([1, 2, 3], [6, 4, 2]), -1.0, places=6)
        self.assertEqual(spearman([1], [1]), 0.0)
        self.assertEqual(spearman([1, 1, 1], [1, 2, 3]), 0.0)

    def test_r_squared_perfect_and_empty(self):
        self.assertEqual(r_squared([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertEqual(r_squared([], []), 0.0)


class TestAgeCurve(unittest.TestCase):
    def test_default_is_monotonic_and_bounded(self):
        c = AgeCurve.default()
        fracs = [c.fraction_at(t) for t in range(0, 40)]
        self.assertEqual(fracs, sorted(fracs))
        self.assertTrue(all(0 < f <= 1.0 for f in fracs))

    def test_mature_is_full_credit(self):
        self.assertEqual(AgeCurve.default().fraction_at(MATURITY_DAYS + 10), 1.0)

    def test_never_returns_near_zero(self):
        # Otherwise a six-hour-old video divides by ~0 and takes over every
        # ranking with an effectively infinite multiplier.
        self.assertGreaterEqual(AgeCurve.default().fraction_at(0.001), 0.02)

    def test_thin_corpus_falls_back_to_default(self):
        self.assertFalse(AgeCurve.fit([mkvideo(vid=f"v{i}") for i in range(3)], NOW).fitted)

    def test_fits_on_a_real_corpus_and_stays_monotonic(self):
        corpus, _ = make_corpus(n_channels=10, videos_per_channel=45, now=NOW)
        curve = AgeCurve.fit([v for vs in corpus.values() for v in vs], NOW)
        self.assertTrue(curve.fitted)
        fracs = [curve.fraction_at(t) for t in range(0, 35)]
        self.assertEqual(fracs, sorted(fracs))


class TestBaseline(unittest.TestCase):
    def test_immature_videos_excluded_from_baseline(self):
        vids = [mkvideo(vid=f"m{i}", views=1000, age_days=90) for i in range(5)]
        vids += [mkvideo(vid=f"y{i}", views=10, age_days=2) for i in range(5)]
        base = build_baseline(vids, is_short=False, now=NOW)
        # Young videos would have dragged the median far below 1000.
        self.assertEqual(base.median_views, 1000)
        self.assertEqual(base.n, 5)

    def test_formats_do_not_share_a_baseline(self):
        vids = [mkvideo(vid=f"l{i}", views=1000, age_days=90) for i in range(5)]
        vids += [mkvideo(vid=f"s{i}", views=50, age_days=90, duration=30) for i in range(5)]
        self.assertEqual(build_baseline(vids, False, NOW).median_views, 1000)
        self.assertEqual(build_baseline(vids, True, NOW).median_views, 50)

    def test_baseline_marked_unreliable_when_thin(self):
        vids = [mkvideo(vid=f"m{i}", views=1000, age_days=90) for i in range(3)]
        self.assertFalse(build_baseline(vids, False, NOW).reliable)

    def test_returns_none_with_no_cohort(self):
        self.assertIsNone(build_baseline([mkvideo(age_days=1)], False, NOW))

    def test_leave_one_out_prevents_self_inflation(self):
        # A breakout must not raise the bar it is measured against.
        vids = [mkvideo(vid=f"m{i}", views=1000, age_days=90) for i in range(5)]
        vids.append(mkvideo(vid="hit", views=20000, age_days=90))
        scored = {s.video.video_id: s for s in score_channel(vids, NOW)}
        self.assertAlmostEqual(scored["hit"].multiplier, 20.0, places=1)


class TestScoring(unittest.TestCase):
    def test_multiplier_is_views_over_expected(self):
        vids = [mkvideo(vid=f"m{i}", views=1000, age_days=90) for i in range(6)]
        base = build_baseline(vids, False, NOW)
        s = score_video(mkvideo(vid="x", views=3000, age_days=90), base, AgeCurve.default(), NOW)
        self.assertAlmostEqual(s.multiplier, 3.0, places=3)
        self.assertFalse(s.provisional)
        self.assertTrue(s.trustworthy)

    def test_young_video_is_age_adjusted_and_flagged(self):
        vids = [mkvideo(vid=f"m{i}", views=1000, age_days=90) for i in range(6)]
        base = build_baseline(vids, False, NOW)
        curve = AgeCurve.default()
        young = mkvideo(vid="y", views=500, age_days=2.5)
        s = score_video(young, base, curve, NOW)
        self.assertTrue(s.provisional)
        self.assertFalse(s.trustworthy)
        # Unadjusted this would score 0.5x; adjusted it is correctly above 1x.
        self.assertGreater(s.multiplier, 1.0)


class TestOutliers(unittest.TestCase):
    def test_recovers_the_planted_effect(self):
        """The load-bearing test: we boost numbered titles 1.9x, and the
        pattern extractor must independently rediscover that."""
        corpus, subs = make_corpus(n_channels=14, videos_per_channel=50,
                                   now=NOW, number_boost=1.9)
        ops = find_opportunities(corpus, subs, now=NOW, min_multiplier=2.0)
        self.assertGreater(len(ops), 20)
        pats = {p.feature: p for p in extract_patterns(ops, corpus)}
        self.assertIn("has_number", pats)
        self.assertGreater(pats["has_number"].lift, 1.2)
        self.assertTrue(pats["has_number"].meaningful)

    def test_stays_quiet_when_nothing_is_planted(self):
        """The other direction. With no effect injected, has_number must not
        show up as a meaningful pattern - a finder that always finds something
        is a random number generator with a UI."""
        corpus, subs = make_corpus(n_channels=14, videos_per_channel=50,
                                   now=NOW, number_boost=1.0)
        ops = find_opportunities(corpus, subs, now=NOW, min_multiplier=2.0)
        pats = {p.feature: p for p in extract_patterns(ops, corpus)}
        if "has_number" in pats:
            self.assertLess(abs(pats["has_number"].lift - 1.0), 0.4)

    def test_max_subscribers_filter(self):
        corpus, subs = make_corpus(n_channels=10, now=NOW)
        ops = find_opportunities(corpus, subs, now=NOW, max_subscribers=50_000)
        self.assertTrue(all(o.subscribers <= 50_000 for o in ops))

    def test_provisional_excluded_by_default(self):
        corpus, subs = make_corpus(n_channels=10, now=NOW)
        self.assertTrue(all(not o.scored.provisional
                            for o in find_opportunities(corpus, subs, now=NOW)))

    def test_patterns_need_support(self):
        corpus, subs = make_corpus(n_channels=2, videos_per_channel=12, now=NOW)
        ops = find_opportunities(corpus, subs, now=NOW, min_multiplier=50.0)
        self.assertEqual(extract_patterns(ops, corpus), [])


class TestPackaging(unittest.TestCase):
    def test_title_features(self):
        f = title_features("Why NOBODY Beat This 7-Year Record (2024)?")
        self.assertEqual(f["has_number"], 1.0)
        self.assertEqual(f["has_year"], 1.0)
        self.assertEqual(f["is_question"], 1.0)
        self.assertEqual(f["has_bracket"], 1.0)
        self.assertEqual(f["caps_words"], 1.0)   # NOBODY, not a 2-letter initialism
        self.assertGreaterEqual(f["curiosity_words"], 2)

    def test_short_initialisms_are_not_shouting(self):
        self.assertEqual(title_features("AI vs ML")["caps_words"], 0.0)

    def test_model_finds_the_planted_driver(self):
        corpus, _ = make_corpus(n_channels=14, videos_per_channel=50,
                                now=NOW, number_boost=1.9)
        curve = AgeCurve.fit([v for vs in corpus.values() for v in vs], NOW)
        scored = [s for vs in corpus.values() for s in score_channel(vs, NOW, curve)]
        model = fit_packaging(scored)
        self.assertIsNotNone(model)
        self.assertEqual(model.top_drivers(1)[0][0], "has_number")
        # Ridge shrinks the estimate, so we assert direction and rough size
        # rather than exact recovery of 1.9x.
        self.assertGreater(math.exp(model.coefs["has_number"]), 1.2)

    def test_refuses_to_fit_on_too_little_data(self):
        corpus, _ = make_corpus(n_channels=1, videos_per_channel=8, now=NOW)
        scored = [s for vs in corpus.values() for s in score_channel(vs, NOW)]
        self.assertIsNone(fit_packaging(scored))

    def test_scoring_ranks_drafts(self):
        corpus, _ = make_corpus(n_channels=14, videos_per_channel=50,
                                now=NOW, number_boost=2.5)
        curve = AgeCurve.fit([v for vs in corpus.values() for v in vs], NOW)
        scored = [s for vs in corpus.values() for s in score_channel(vs, NOW, curve)]
        model = fit_packaging(scored)
        self.assertGreater(model.score("7 Ways To Fix This"), model.score("Ways To Fix This"))


class TestAudit(unittest.TestCase):
    def test_duration_buckets(self):
        self.assertEqual(duration_bucket(60), "short (<3m)")
        self.assertEqual(duration_bucket(600), "standard (8-15m)")
        self.assertEqual(duration_bucket(99999), "extended (30m+)")

    def test_steady_cadence_detected(self):
        vids = [mkvideo(vid=f"v{i}", age_days=7.0 * i + 1) for i in range(12)]
        c = audit_cadence(vids, NOW)
        self.assertEqual(c.consistency, "steady")
        self.assertAlmostEqual(c.uploads_per_week, 1.0, places=1)

    def test_erratic_cadence_detected(self):
        gaps = [1, 2, 40, 3, 60, 2, 1, 55, 4, 2]
        ages, run = [], 1.0
        for g in gaps:
            run += g
            ages.append(run)
        self.assertEqual(audit_cadence([mkvideo(vid=f"v{i}", age_days=a)
                                        for i, a in enumerate(ages)], NOW).consistency,
                         "erratic")

    def test_dormant_channel_flagged(self):
        vids = [mkvideo(vid=f"v{i}", age_days=200 + 7.0 * i) for i in range(10)]
        self.assertIn("dormant", audit_cadence(vids, NOW).note)

    def test_too_few_uploads(self):
        self.assertEqual(audit_cadence([mkvideo()], NOW).consistency, "unknown")

    def test_kill_and_double_down_calls(self):
        vids = [mkvideo(vid=f"g{i}", views=5000, age_days=60 + i, duration=600) for i in range(8)]
        vids += [mkvideo(vid=f"b{i}", views=200, age_days=60 + i, duration=2000) for i in range(8)]
        a = audit_channel(vids, now=NOW)
        calls = {f.label: f.call for f in a.formats}
        self.assertEqual(calls["standard (8-15m)"], "DOUBLE DOWN")
        self.assertEqual(calls["extended (30m+)"], "KILL")

    def test_hit_rate_is_a_fraction(self):
        corpus, _ = make_corpus(n_channels=1, videos_per_channel=40, now=NOW)
        a = audit_channel(list(corpus.values())[0], now=NOW)
        self.assertTrue(0.0 <= a.hit_rate <= 1.0)


class TestTrends(unittest.TestCase):
    def _hist(self, views, hours):
        return [
            Snapshot("v1", NOW - timedelta(hours=h), v, 0, 0)
            for v, h in zip(views, hours)
        ]

    def test_velocity_needs_two_points(self):
        self.assertIsNone(velocity_from_history(self._hist([100], [0])))

    def test_acceleration_and_phase(self):
        # 1000/hr then 3000/hr - still building.
        v = velocity_from_history(self._hist([0, 6000, 24000], [12, 6, 0]))
        self.assertAlmostEqual(v.vph_recent, 3000, places=0)
        self.assertAlmostEqual(v.vph_prior, 1000, places=0)
        self.assertEqual(v.phase, "accelerating")

    def test_decay_detected(self):
        v = velocity_from_history(self._hist([0, 30000, 33000], [12, 6, 0]))
        self.assertEqual(v.phase, "decaying")

    def test_snapshots_too_close_together_give_no_rate(self):
        # Public view counts update in lumps; a 6-minute gap is pure jitter.
        v = velocity_from_history(self._hist([1000, 1005], [0.1, 0]))
        self.assertEqual(v.vph_recent, 0.0)

    def test_counter_never_goes_backwards(self):
        v = velocity_from_history(self._hist([5000, 4000], [6, 0]))
        self.assertGreaterEqual(v.vph_recent, 0.0)

    def test_two_points_is_unknown_phase(self):
        self.assertEqual(velocity_from_history(self._hist([0, 6000], [6, 0])).phase, "unknown")

    def test_topic_verdicts(self):
        from ytengine.trends import Velocity
        rising = [Velocity(f"v{i}", "espresso guide", 1000, 500, 100, 5.0, 24, 5)
                  for i in range(4)]
        stale = [Velocity(f"w{i}", "latte art", 1000, 500, 5000, 0.1, 24, 5)
                 for i in range(4)]
        t = {x.topic: x for x in topic_trends(rising + stale, ["espresso", "latte"])}
        self.assertIn("RISING", t["espresso"].verdict)
        self.assertIn("saturated", t["latte"].verdict)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.store = SnapshotStore(self.tmp.name)

    def test_roundtrip(self):
        v = mkvideo(views=100)
        self.store.record([v], NOW - timedelta(hours=6))
        self.store.record([mkvideo(views=700)], NOW)
        h = self.store.history("v1")
        self.assertEqual([s.views for s in h], [100, 700])

    def test_same_timestamp_is_idempotent(self):
        self.store.record([mkvideo(views=100)], NOW)
        self.store.record([mkvideo(views=100)], NOW)
        self.assertEqual(len(self.store.history("v1")), 1)

    def test_velocity_over_stored_history(self):
        self.store.record([mkvideo(views=0)], NOW - timedelta(hours=12))
        self.store.record([mkvideo(views=6000)], NOW - timedelta(hours=6))
        self.store.record([mkvideo(views=24000)], NOW)
        v = rank_rising(self.store)[0]
        self.assertEqual(v.phase, "accelerating")

    def test_prune_caps_history(self):
        for i in range(10):
            self.store.record([mkvideo(views=i * 100)], NOW - timedelta(hours=10 - i))
        self.store.prune(keep_per_video=4)
        self.assertEqual(len(self.store.history("v1")), 4)

    def test_videos_with_history_filters(self):
        self.store.record([mkvideo(vid="a")], NOW - timedelta(hours=6))
        self.store.record([mkvideo(vid="a")], NOW)
        self.store.record([mkvideo(vid="b")], NOW)
        self.assertEqual(self.store.videos_with_history(2), ["a"])


class TestCLI(unittest.TestCase):
    def test_every_offline_command_runs(self):
        import contextlib, io
        from ytengine.cli import main
        with tempfile.NamedTemporaryFile(suffix=".db") as db:
            for argv in (
                ["outliers", "--offline", "--limit", "3"],
                ["packaging", "--offline", "--title", "5 Things"],
                ["audit", "--offline"],
                ["track", "--offline", "--db", db.name],
                ["trends", "--offline", "--db", db.name],
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(argv), 0, f"{argv} failed")

    def test_live_mode_requires_a_target(self):
        import contextlib, io
        from ytengine.cli import main
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["outliers"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
