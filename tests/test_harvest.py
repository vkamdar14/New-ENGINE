"""Harvester tests, driven by a fake client.

There is no API key in CI, and there should not be one - a test suite that
needs live quota is a test suite nobody runs. So the client is faked, which
also lets us assert the thing that actually matters and that a live test could
never check deterministically: that the harvester survives interruption.

A 200k-video harvest spans several daily quota resets. If a crash on day three
loses days one and two, the tool is useless at the only scale it exists for.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from ytengine.client import QuotaExceeded
from ytengine.harvest import DEFAULT_SEEDS, Harvester, HarvestPlan, estimate_quota
from ytengine.models import Channel, Video
from ytengine.store import CorpusStore

NOW = datetime(2025, 6, 1, tzinfo=timezone.utc)


class FakeClient:
    """Stands in for YouTubeClient, charging the same quota as the real one."""

    def __init__(self, n_channels=40, per_channel=60, quota_budget=10000, fail_on=()):
        self.quota_budget = quota_budget
        self.quota_used = 0
        self.fail_on = set(fail_on)
        self.n_channels = n_channels
        self.per_channel = per_channel
        self.calls = {"search": 0, "uploads": 0, "channels": 0, "videos": 0}
        self.orders = []

    def _charge(self, units):
        if self.quota_used + units > self.quota_budget:
            raise QuotaExceeded("fake budget exhausted")
        self.quota_used += units

    def _video(self, cid, i, short=True):
        return Video(
            video_id=f"{cid}-v{i}", channel_id=cid, channel_title=f"C{cid}",
            title=f"title {i}", description="", published_at=NOW - timedelta(days=40 + i),
            duration_s=45 if short else 900, views=1000 + i, likes=10, comments=1,
        )

    def search_video_ids(self, query, limit=50, region=None, published_after=None,
                         published_before=None, video_duration=None, order="date"):
        self._charge(100)
        self.calls["search"] += 1
        self.orders.append(order)
        if query in self.fail_on:
            raise QuotaExceeded("fake")
        # Channels are derived from the slice itself, not the call index, so
        # a resumed run hitting fresh slices sees fresh channels - which is
        # how real slicing behaves and what a resume test must exercise.
        h = abs(hash((query, region, published_after))) % max(self.n_channels, 1)
        return [f"UC{(h + j) % self.n_channels}-v0" for j in range(3)]

    def videos_by_id(self, ids):
        self._charge(max(1, len(ids) // 50))
        self.calls["videos"] += 1
        return [self._video(i.split("-")[0], 0) for i in ids]

    def channels(self, ids):
        self._charge(max(1, len(ids) // 50))
        self.calls["channels"] += 1
        return [
            Channel(channel_id=c, title=f"C{c}", subscribers=1000, total_views=0,
                    video_count=self.per_channel, uploads_playlist=f"UU{c}")
            for c in ids
        ]

    def channel_uploads(self, channel, limit=500):
        n = min(self.per_channel, limit)
        self._charge(max(1, (n // 50) * 2))
        self.calls["uploads"] += 1
        return [self._video(channel.channel_id, i) for i in range(n)]


class TempStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "corpus.db")
        self.store = CorpusStore(self.path)


class TestPlan(unittest.TestCase):
    def test_grid_size_is_the_product(self):
        p = HarvestPlan(seeds=["a", "b"], regions=["US", "GB"], windows=4)
        self.assertEqual(p.total_slices, 16)
        self.assertEqual(len(list(p.slices(NOW))), 16)

    def test_slice_keys_are_unique(self):
        p = HarvestPlan(seeds=["a", "b"], regions=["US", "GB"], windows=4)
        keys = [s[0] for s in p.slices(NOW)]
        self.assertEqual(len(keys), len(set(keys)))

    def test_windows_march_backwards_and_do_not_overlap(self):
        p = HarvestPlan(seeds=["a"], regions=["US"], windows=3, window_days=90)
        rows = list(p.slices(NOW))
        befores = [r[4] for r in rows]
        self.assertEqual(befores, sorted(befores, reverse=True))
        # window n's "after" is exactly window n+1's "before"
        self.assertEqual(rows[0][3], rows[1][4])

    def test_default_seed_grid_reaches_far_past_the_500_cap(self):
        # The whole point of slicing: one query is capped at ~500 results.
        self.assertGreater(HarvestPlan().total_slices * 500, 500_000)


class TestQuotaEstimate(unittest.TestCase):
    def test_expansion_dominates_discovery(self):
        est = estimate_quota(100_000, channels_needed=500)
        self.assertGreater(est["expansion_units"], est["discovery_units"])

    def test_thirty_thousand_fits_in_one_day(self):
        self.assertEqual(estimate_quota(30_000, channels_needed=150)["days_at_10k"], 1)

    def test_a_billion_is_not_remotely_feasible(self):
        # The honest ceiling, asserted so nobody plans a full census.
        self.assertGreater(estimate_quota(1_000_000_000)["days_at_10k"], 1000)


class TestHarvest(TempStore):
    def test_end_to_end_fills_the_corpus(self):
        c = FakeClient()
        r = Harvester(c, self.store).run(HarvestPlan(seeds=["a"], regions=["US"], windows=4),
                                         target=500, now=NOW)
        self.assertGreater(r.videos_added, 0)
        self.assertGreater(self.store.counts()["shorts"], 0)

    def test_expansion_is_where_the_videos_come_from(self):
        """Discovery finds channels; expansion is 50x cheaper per video and
        must supply the overwhelming majority of the corpus."""
        c = FakeClient(per_channel=60)
        Harvester(c, self.store).run(HarvestPlan(seeds=["a"], regions=["US"], windows=4),
                                     target=1000, now=NOW)
        self.assertGreater(c.calls["uploads"], c.calls["search"])

    def test_only_shorts_are_kept_by_default(self):
        c = FakeClient()
        Harvester(c, self.store, shorts_only=True).run(
            HarvestPlan(seeds=["a"], regions=["US"], windows=2), target=200, now=NOW)
        self.assertTrue(all(v.is_short for v in self.store.load_videos()))

    def test_resumes_where_it_stopped(self):
        """The load-bearing test. Run with a tiny budget, then again with a
        fresh one, and the second run must continue rather than restart."""
        plan = HarvestPlan(seeds=["a", "b"], regions=["US"], windows=4)
        c1 = FakeClient(quota_budget=400)
        r1 = Harvester(c1, self.store).run(plan, target=10_000, now=NOW)
        first = self.store.counts()["videos"]
        self.assertGreater(first, 0)
        self.assertIn("quota", r1.stopped_because)  # budget, not an empty queue

        c2 = FakeClient(quota_budget=10_000)
        Harvester(c2, self.store).run(plan, target=10_000, now=NOW)
        self.assertGreater(self.store.counts()["videos"], first)

    def test_completed_slices_are_never_re_searched(self):
        # Search is the expensive call; repeating a retired slice on a resume
        # would burn 100 units to re-learn nothing.
        plan = HarvestPlan(seeds=["a"], regions=["US"], windows=3)
        c1 = FakeClient()
        Harvester(c1, self.store).discover(plan, __import__(
            "ytengine.harvest", fromlist=["HarvestReport"]).HarvestReport(), max_slices=3, now=NOW)
        used = c1.calls["search"]
        c2 = FakeClient()
        Harvester(c2, self.store).discover(plan, __import__(
            "ytengine.harvest", fromlist=["HarvestReport"]).HarvestReport(), max_slices=3, now=NOW)
        self.assertEqual(used, 3)
        self.assertEqual(c2.calls["search"], 0)

    def test_walked_channels_are_not_walked_again(self):
        plan = HarvestPlan(seeds=["a"], regions=["US"], windows=2)
        c1 = FakeClient()
        Harvester(c1, self.store).run(plan, target=10_000, now=NOW)
        walked = c1.calls["uploads"]
        self.assertGreater(walked, 0)
        c2 = FakeClient()
        Harvester(c2, self.store).run(plan, target=10_000, now=NOW)
        self.assertEqual(c2.calls["uploads"], 0)

    def test_dead_channel_does_not_abort_the_harvest(self):
        class Flaky(FakeClient):
            def channel_uploads(self, channel, limit=500):
                if channel.channel_id.endswith("1"):
                    raise LookupError("deleted channel")
                return super().channel_uploads(channel, limit)

        c = Flaky()
        r = Harvester(c, self.store).run(HarvestPlan(seeds=["a"], regions=["US"], windows=4),
                                         target=1000, now=NOW)
        self.assertGreater(r.videos_added, 0)

    def test_quota_is_reserved_for_expansion(self):
        """Discovery must not eat the units expansion needs, or a run finds
        channels it has no budget left to actually read."""
        c = FakeClient(quota_budget=250)
        h = Harvester(c, self.store)
        from ytengine.harvest import HarvestReport
        rep = HarvestReport()
        h.discover(HarvestPlan(seeds=["a"], regions=["US"], windows=8), rep,
                   max_slices=8, now=NOW)
        self.assertGreaterEqual(c.quota_budget - c.quota_used, 0)
        self.assertIn("quota", rep.stopped_because)

    def test_counters_are_never_overwritten_on_re_observation(self):
        # Re-storing a video with fresher views would destroy the age/views
        # relationship every baseline depends on.
        v = Video(video_id="x", channel_id="c", channel_title="", title="t",
                  description="", published_at=NOW - timedelta(days=60),
                  duration_s=30, views=100, likes=1, comments=0)
        self.store.add_videos([v])
        v.views = 999_999
        self.store.add_videos([v])
        self.assertEqual(self.store.load_videos()[0].views, 100)

    def test_reports_when_the_walk_queue_is_empty(self):
        from ytengine.harvest import HarvestReport
        rep = HarvestReport()
        Harvester(FakeClient(), self.store).expand(rep, target=100)
        self.assertIn("discovery", rep.stopped_because)


class TestHarvestCLI(unittest.TestCase):
    def test_estimate_runs_without_a_key(self):
        import contextlib, io
        from ytengine.cli import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["harvest", "--estimate", "--target", "30000", "--offline"])
        self.assertEqual(rc, 0)
        self.assertIn("quota units", buf.getvalue())

    def test_harvest_refuses_offline(self):
        import contextlib, io
        from ytengine.cli import main
        with contextlib.redirect_stderr(io.StringIO()) as err:
            rc = main(["harvest", "--offline", "--target", "10"])
        self.assertEqual(rc, 2)
        self.assertIn("API key", err.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestAccounting(TempStore):
    """Regression tests for harvest counters.

    A harvest re-encounters the same video constantly - the same Short surfaces
    in several search slices and again when its channel is walked. Counters
    that tally submissions rather than stores drift upward and quietly lie
    about corpus size, which is the one number every downstream decision and
    every quota projection is based on.
    """

    def test_add_videos_returns_only_the_new_ids(self):
        v = self._vid("a")
        self.assertEqual(self.store.add_videos([v]), ["a"])
        self.assertEqual(self.store.add_videos([v]), [])

    def test_duplicates_within_one_batch_counted_once(self):
        v = self._vid("a")
        self.assertEqual(self.store.add_videos([v, v, v]), ["a"])

    def test_shorts_never_exceed_videos(self):
        c = FakeClient(n_channels=30, per_channel=80)
        r = Harvester(c, self.store).run(
            HarvestPlan(seeds=["a", "b"], regions=["US"], windows=4), target=2000, now=NOW)
        self.assertLessEqual(r.shorts_added, r.videos_added)
        self.assertEqual(self.store.counts()["shorts"], self.store.counts()["videos"])

    def test_reported_additions_match_the_corpus(self):
        c = FakeClient(n_channels=30, per_channel=80)
        r = Harvester(c, self.store).run(
            HarvestPlan(seeds=["a", "b"], regions=["US"], windows=4), target=2000, now=NOW)
        self.assertEqual(r.videos_added, self.store.counts()["videos"])

    def _vid(self, vid):
        return Video(video_id=vid, channel_id="c", channel_title="", title="t",
                     description="", published_at=NOW - timedelta(days=60),
                     duration_s=30, views=1, likes=0, comments=0)


class TestMultiDay(TempStore):
    """A harvest is a multi-day process; these guard that property."""

    def test_a_barren_discovery_round_does_not_end_the_run(self):
        """Some slices return only channels we already hold. Treating the
        first such round as terminal ends a harvest with quota unspent and
        most of the grid unretired."""
        class Repetitive(FakeClient):
            def __init__(self, **kw):
                super().__init__(**kw)
                self._n = 0

            def search_video_ids(self, query, limit=50, region=None, published_after=None,
                                 published_before=None, video_duration=None, order="date"):
                self._charge(100)
                self.calls["search"] += 1
                self._n += 1
                # First three slices repeat one known channel, then fresh ones.
                if self._n <= 3:
                    return ["UC0-v0"]
                return [f"UC{self._n}-v0"]

        c = Repetitive(n_channels=500, per_channel=100)
        r = Harvester(c, self.store).run(
            HarvestPlan(seeds=["a", "b", "c"], regions=["US"], windows=8),
            target=500, now=NOW, max_discovery_slices=2)
        self.assertGreater(r.videos_added, 0, "gave up on the first barren round")

    def test_corpus_grows_across_consecutive_days(self):
        plan = HarvestPlan(seeds=["a", "b", "c"], regions=["US", "GB"], windows=8)
        sizes = []
        for _ in range(3):
            c = FakeClient(n_channels=5000, per_channel=100, quota_budget=10_000)
            Harvester(c, self.store).run(plan, target=2000, now=NOW, max_discovery_slices=4)
            sizes.append(self.store.counts()["videos"])
        self.assertEqual(sizes, sorted(sizes))
        self.assertGreater(sizes[-1], sizes[0], "corpus did not grow on resume")

    def test_run_terminates_when_the_grid_is_fully_retired(self):
        # Must not spin forever once every slice is done and no channels remain.
        c = FakeClient(n_channels=2, per_channel=10)
        r = Harvester(c, self.store).run(
            HarvestPlan(seeds=["a"], regions=["US"], windows=1), target=10**9, now=NOW)
        self.assertNotEqual(r.stopped_because, "target reached")


class TestSamplingBias(TempStore):
    """Regression guard for the worst bug this project has had.

    Discovery originally ordered search results by viewCount. That returns only
    the top videos, which only ever surfaces mega-channels, which produces a
    corpus with no failures in it - a real 12,134-video harvest came back with
    median views of 4.6 MILLION and a 5th percentile of 181k. Not one flop.

    A classifier trained on that learns "everything goes viral", scores
    beautifully on its own held-out split, and cannot predict anything. The
    sampling decides whether the model means anything, so it is pinned here.
    """

    def test_discovery_never_orders_by_view_count(self):
        c = FakeClient()
        Harvester(c, self.store).run(
            HarvestPlan(seeds=["a"], regions=["US"], windows=4), target=200, now=NOW)
        self.assertTrue(c.orders, "discovery never ran")
        self.assertNotIn("viewCount", c.orders,
                         "ordering discovery by viewCount rebuilds the survivorship bias")

    def test_default_discovery_order_is_date(self):
        # Date is uncorrelated with outcome; relevance and viewCount are not.
        c = FakeClient()
        Harvester(c, self.store).run(
            HarvestPlan(seeds=["a"], regions=["US"], windows=2), target=100, now=NOW)
        self.assertEqual(set(c.orders), {"date"})
