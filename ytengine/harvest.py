"""Large-scale Shorts harvesting.

The goal is the biggest honest corpus obtainable, resumed across days.

What is not possible, stated plainly so nobody plans around it: there is no
endpoint that enumerates every Short. The Data API has no "list all videos"
call, and `search.list` stops returning new results after roughly 500 per
query however hard you paginate. YouTube's own index is not exposed. So a
complete census of Shorts is out of reach for any third party, at any budget.

What is possible is large, and it compounds:

    discovery   search.list   100 units / 50 results   <- expensive, capped
    expansion   uploads walk    2 units / 50 videos    <- cheap, uncapped

Discovery is 50x the price of expansion and hits the 500-result wall;
expansion is nearly free and has no wall. So the strategy is: spend a little
quota finding *channels*, then spend the rest walking their entire back
catalogues. One search page (100 units, 50 channels) can unlock tens of
thousands of videos at 2 units per 50.

Rough daily arithmetic on a default 10,000-unit key:

    1,000 units discovery  ->  ~500 channels
    9,000 units expansion  ->  ~225,000 videos

so ~30k Shorts is comfortably one day's work, and the corpus keeps growing
every day the harvester is re-run because state is checkpointed.

The 500-result cap is per *query*, not per key, so breadth comes from slicing:
the same seed term across regions and publish windows is many distinct
queries, each with its own 500-result allowance.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional, Sequence

from .client import QuotaExceeded, YouTubeClient
from .models import Video
from .store import CorpusStore

# Broad seeds spanning the format's main content families. Not exhaustive -
# nothing can be - but wide enough that the discovered channel set is not all
# one genre, which would poison every downstream baseline.
DEFAULT_SEEDS = [
    "shorts", "tutorial", "recipe", "workout", "gaming", "comedy sketch",
    "science explained", "history facts", "car review", "makeup tutorial",
    "guitar lesson", "coding tips", "travel vlog", "pet", "diy project",
    "finance tips", "language learning", "magic trick", "dance", "street food",
]

DEFAULT_REGIONS = ["US", "GB", "IN", "CA", "AU", "DE", "BR", "PH", "NG", "ID"]


@dataclass
class HarvestPlan:
    """The slicing grid that breaks the 500-result-per-query ceiling."""

    seeds: Sequence[str] = field(default_factory=lambda: list(DEFAULT_SEEDS))
    regions: Sequence[str] = field(default_factory=lambda: list(DEFAULT_REGIONS))
    window_days: int = 90
    windows: int = 8          # how far back to slice, in window_days steps
    per_slice: int = 50       # results per slice; one search page

    def slices(self, now: Optional[datetime] = None) -> Iterator[tuple[str, str, str, str, str]]:
        """Yield (key, query, region, published_after, published_before).

        Each combination is a distinct query as far as the API's result cap is
        concerned, so the reachable total scales with the size of this grid
        rather than being pinned at 500.
        """
        now = now or datetime.now(timezone.utc)
        for seed, region, w in itertools.product(self.seeds, self.regions, range(self.windows)):
            before = now - timedelta(days=self.window_days * w)
            after = before - timedelta(days=self.window_days)
            key = f"{seed}|{region}|{w}"
            yield (key, seed, region,
                   after.strftime("%Y-%m-%dT%H:%M:%SZ"),
                   before.strftime("%Y-%m-%dT%H:%M:%SZ"))

    @property
    def total_slices(self) -> int:
        return len(self.seeds) * len(self.regions) * self.windows


@dataclass
class HarvestReport:
    videos_added: int = 0
    shorts_added: int = 0
    channels_found: int = 0
    channels_expanded: int = 0
    slices_done: int = 0
    quota_spent: int = 0
    stopped_because: str = "target reached"

    def render(self, totals: dict[str, int]) -> str:
        return "\n".join([
            "",
            "HARVEST",
            "=======",
            f"  stopped: {self.stopped_because}",
            f"  this run:  +{self.videos_added} videos ({self.shorts_added} shorts), "
            f"{self.channels_found} channels found, {self.channels_expanded} walked",
            f"  slices:    {self.slices_done} completed this run",
            f"  quota:     {self.quota_spent} units",
            "",
            f"  corpus now: {totals['videos']:,} videos / {totals['shorts']:,} shorts",
            f"              {totals['channels']:,} channels "
            f"({totals['channels_expanded']:,} fully walked)",
            f"              {totals['slices_done']:,} slices retired",
        ])


class Harvester:
    """Two-phase, checkpointed, quota-bounded.

    Phase 1 discovers channels through sliced search. Phase 2 walks each
    discovered channel's uploads playlist. Phase 2 is where nearly all the
    videos come from, because it is 50x cheaper per video.
    """

    def __init__(self, client: YouTubeClient, store: CorpusStore, shorts_only: bool = True):
        self.client = client
        self.store = store
        self.shorts_only = shorts_only

    def _quota_left(self) -> int:
        return self.client.quota_budget - self.client.quota_used

    def discover(self, plan: HarvestPlan, report: HarvestReport,
                 max_slices: int = 10, now: Optional[datetime] = None) -> None:
        """Spend a bounded slice of quota finding channels.

        Deliberately capped: discovery is the expensive phase, and a run that
        spends everything on search finds thousands of channels it then has no
        quota left to actually read.
        """
        done = 0
        for key, query, region, after, before in plan.slices(now):
            if done >= max_slices:
                return
            if self.store.slice_done(key):
                continue
            # Reserve headroom so a search never consumes the last units that
            # expansion needs to turn those channels into videos.
            if self._quota_left() < 200:
                report.stopped_because = "quota reserved for expansion"
                return
            try:
                video_ids = self.client.search_video_ids(
                    query, limit=plan.per_slice, region=region,
                    published_after=after, published_before=before,
                    video_duration="short" if self.shorts_only else None,
                )
            except QuotaExceeded:
                report.stopped_because = "quota exhausted during discovery"
                return

            if video_ids:
                # Hydrating the search hits costs 1 unit per 50 and gives us
                # both the videos themselves and their channel ids.
                vids = self.client.videos_by_id(video_ids)
                self._store(vids, report)
                new_ids = [v.channel_id for v in vids] 
                unknown = list({c for c in new_ids if c} - self.store.known_channel_ids())
                if unknown:
                    chans = self.client.channels(unknown)
                    report.channels_found += self.store.add_channels(chans)

            self.store.mark_slice(key)
            report.slices_done += 1
            done += 1

    def expand(self, report: HarvestReport, target: int,
               per_channel: int = 500, batch: int = 200) -> None:
        """Walk pending channels' uploads until target or quota runs out."""
        while report.videos_added < target:
            pending = self.store.pending_channels(limit=batch)
            if not pending:
                report.stopped_because = "no channels left to walk - run discovery again"
                return
            for ch in pending:
                if report.videos_added >= target:
                    report.stopped_because = "target reached"
                    return
                # Each 50 videos costs ~2 units; keep a small floor so the
                # final channel completes rather than aborting mid-playlist.
                if self._quota_left() < 10:
                    report.stopped_because = "daily quota exhausted - rerun tomorrow to resume"
                    return
                try:
                    vids = self.client.channel_uploads(ch, limit=per_channel)
                except QuotaExceeded:
                    report.stopped_because = "quota exhausted during expansion"
                    return
                except (LookupError, RuntimeError):
                    # A deleted or private channel must not kill a multi-day
                    # harvest; mark it done so we never retry it.
                    self.store.mark_expanded(ch.channel_id)
                    continue
                self._store(vids, report)
                self.store.mark_expanded(ch.channel_id)
                report.channels_expanded += 1

    def _store(self, vids: Sequence[Video], report: HarvestReport) -> None:
        keep = [v for v in vids if v.is_short] if self.shorts_only else list(vids)
        new_ids = set(self.store.add_videos(keep))
        report.videos_added += len(new_ids)
        # Count shorts among what was actually stored. Counting them among
        # what was submitted double-counts every re-encountered video and can
        # report more shorts than the corpus holds videos.
        report.shorts_added += sum(1 for v in keep if v.is_short and v.video_id in new_ids)

    def run(self, plan: HarvestPlan, target: int = 30000,
            max_discovery_slices: int = 10, per_channel: int = 500,
            now: Optional[datetime] = None, max_barren_rounds: int = 5) -> HarvestReport:
        """Alternate discovery and expansion until target, quota, or grid runs out.

        The two phases have to interleave rather than run once each. Expansion
        drains the walk queue and then has nothing left to do, while quota and
        unretired slices may both remain - a single discover-then-expand pass
        would stop at a few thousand videos on a 30k target and hand back
        most of the day's budget unspent. So whenever the queue empties short
        of target, we buy more channels and keep going.
        """
        report = HarvestReport()
        barren = 0
        while report.videos_added < target:
            if self._quota_left() < 10:
                report.stopped_because = "daily quota exhausted - rerun tomorrow to resume"
                break

            if len(self.store.pending_channels(limit=1)) == 0:
                before_slices = report.slices_done
                self.discover(plan, report, max_slices=max_discovery_slices, now=now)
                if report.slices_done == before_slices:
                    # Discovery bought nothing: either the grid is fully
                    # retired or quota stopped it. Either way there is no
                    # further progress to make this run.
                    if "quota" not in report.stopped_because:
                        report.stopped_because = (
                            "search grid exhausted - widen --seeds/--regions/--windows"
                        )
                    break
                if len(self.store.pending_channels(limit=1)) == 0:
                    # A barren round is normal, not terminal: a slice whose
                    # channels we already hold yields nothing while the next
                    # slice over is untouched. Giving up on the first barren
                    # round ends multi-day harvests early, with quota unspent
                    # and most of the grid still unretired.
                    barren += 1
                    if barren >= max_barren_rounds:
                        report.stopped_because = (
                            f"no new channels across {barren} discovery rounds - "
                            "widen --seeds/--regions/--windows"
                        )
                        break
                    continue
                barren = 0

            before_videos = report.videos_added
            self.expand(report, target=target, per_channel=per_channel)
            if "quota" in report.stopped_because:
                break
            if report.videos_added == before_videos and \
                    len(self.store.pending_channels(limit=1)) == 0:
                # A full pass that added nothing and left nothing queued would
                # spin forever; only discovery can help, and we just tried.
                continue

        report.quota_spent = self.client.quota_used
        return report


def estimate_quota(n_videos: int, channels_needed: int = 0) -> dict[str, int]:
    """What a harvest of this size costs, before committing to it."""
    expansion = (n_videos // 50) * 2          # playlistItems page + videos batch
    discovery = (channels_needed // 50) * 101  # search page + channels hydrate
    total = expansion + discovery
    return {
        "expansion_units": expansion,
        "discovery_units": discovery,
        "total_units": total,
        "days_at_10k": max(1, -(-total // 10000)),
    }
