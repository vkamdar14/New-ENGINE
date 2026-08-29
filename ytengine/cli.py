"""Command line interface.

    ytengine outliers  --niche "espresso"          find what overperforms
    ytengine packaging --channel UC... --title "…"  score drafts
    ytengine audit     --channel UC...              keep/kill by format
    ytengine track     --channel UC...              snapshot for velocity
    ytengine trends    --topics a,b,c               what is accelerating

Every command takes --offline, which runs the whole pipeline against a
synthetic corpus so the tool can be evaluated before anyone provisions an
API key.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from . import __version__, report
from .audit import audit_channel
from .backtest import build_samples, render as render_backtest, run_backtest
from .check import check_key, render as render_check
from .client import QuotaExceeded, YouTubeClient
from .clips import find_clips, mentions_from_comments, render_clips
from .fixtures import make_corpus
from .harvest import DEFAULT_REGIONS, DEFAULT_SEEDS, Harvester, HarvestPlan, estimate_quota
from .metrics import AgeCurve, score_channel
from .outliers import extract_patterns, find_opportunities
from .packaging import fit_packaging, thumbnail_stats
from .store import CorpusStore, SnapshotStore
from .trends import rank_rising, topic_trends

OFFLINE_BANNER = (
    "  [OFFLINE MODE] Synthetic fixture data. Numbers below are generated,\n"
    "  not observed - useful for checking the pipeline, useless as advice.\n"
)


def _corpus(args, client: YouTubeClient | None):
    """Return (corpus, subscribers) either from fixtures or the live API."""
    if args.offline:
        return make_corpus(seed=args.seed)

    assert client is not None
    if args.channel:
        channel_ids = args.channel.split(",")
    else:
        print(f"  discovering channels for '{args.niche}' (search costs 100 units)...")
        channel_ids = client.search_channels(args.niche, limit=args.channels)

    channels = client.channels(channel_ids)
    corpus, subs = {}, {}
    for ch in channels:
        print(f"  pulling {ch.title} ({ch.subscribers:,} subs)...")
        corpus[ch.channel_id] = client.channel_uploads(ch, limit=args.videos)
        subs[ch.channel_id] = ch.subscribers
    return corpus, subs


def cmd_outliers(args, client):
    corpus, subs = _corpus(args, client)
    ops = find_opportunities(
        corpus, subs,
        min_multiplier=args.min_multiplier,
        include_provisional=args.include_provisional,
        max_subscribers=args.max_subs,
    )
    pooled = [v for vs in corpus.values() for v in vs]
    print(report.render_curve(AgeCurve.fit(pooled)))
    print(report.render_opportunities(ops, limit=args.limit))
    print(report.render_patterns(extract_patterns(ops, corpus)))
    return 0


def cmd_packaging(args, client):
    corpus, _ = _corpus(args, client)
    curve = AgeCurve.fit([v for vs in corpus.values() for v in vs])
    scored = [s for vs in corpus.values() for s in score_channel(vs, curve=curve)]
    model = fit_packaging(scored)
    print(report.render_packaging(model))
    if args.title and model:
        print(report.render_title_scores(model, args.title))
    if args.thumbnail:
        print(report.render_thumbnail(thumbnail_stats(args.thumbnail)))
    return 0


def cmd_audit(args, client):
    corpus, _ = _corpus(args, client)
    for cid, vids in corpus.items():
        if vids:
            print(report.render_audit(audit_channel(vids)))
    return 0


def cmd_track(args, client):
    """Snapshot current counters. Velocity needs at least two runs."""
    corpus, _ = _corpus(args, client)
    store = SnapshotStore(args.db)
    n = sum(store.record(v) for v in corpus.values())
    ready = len(store.videos_with_history(min_snapshots=2))
    print(f"\n  recorded {n} videos into {args.db}")
    print(f"  {ready} now have 2+ snapshots and can be ranked for velocity")
    if not ready:
        print("  run this again in a few hours to get the first velocity read")
    return 0


def cmd_trends(args, client):
    store = SnapshotStore(args.db)
    vels = rank_rising(store, limit=args.limit)
    topics = topic_trends(vels, args.topics.split(",")) if args.topics else []
    print(report.render_trends(vels, topics))
    return 0


def cmd_harvest(args, client):
    """Build a large corpus, resumably, across as many days as it takes."""
    store = CorpusStore(args.corpus)
    before = store.counts()

    if args.estimate:
        est = estimate_quota(args.target, channels_needed=args.target // 200)
        print(f"\n  {args.target:,} videos would cost ~{est['total_units']:,} quota units")
        print(f"    expansion {est['expansion_units']:,}u + discovery {est['discovery_units']:,}u")
        print(f"    ~{est['days_at_10k']} day(s) on a default 10,000-unit key")
        return 0

    if client is None:
        print("error: harvesting needs a real API key (YOUTUBE_API_KEY); "
              "--offline cannot invent a corpus", file=sys.stderr)
        return 2

    plan = HarvestPlan(
        seeds=args.seeds.split(",") if args.seeds else list(DEFAULT_SEEDS),
        regions=args.regions.split(",") if args.regions else list(DEFAULT_REGIONS),
        windows=args.windows,
    )
    print(f"\n  plan: {plan.total_slices:,} query slices "
          f"({len(plan.seeds)} seeds x {len(plan.regions)} regions x {plan.windows} windows)")
    print(f"  already held: {before['videos']:,} videos / {before['shorts']:,} shorts")

    harvester = Harvester(client, store, shorts_only=not args.include_long)
    report = harvester.run(plan, target=args.target,
                           max_discovery_slices=args.discovery_slices,
                           per_channel=args.per_channel)
    print(report.render(store.counts()))
    return 0


def cmd_clips(args, client):
    """Mine a video (or a channel's recent uploads) for clippable moments."""
    if client is None:
        print("error: clip mining reads real comments; needs YOUTUBE_API_KEY",
              file=sys.stderr)
        return 2

    if args.video:
        video_ids = args.video.split(",")
    else:
        ch = client.channel(args.channel)
        print(f"  scanning {ch.title}: {args.scan} most recent uploads")
        uploads = client.channel_uploads(ch, limit=args.scan)
        # Long videos only. A Short has nothing to clip out of it, and
        # timestamp comments are a long-form behaviour to begin with.
        longform = [v for v in uploads if v.duration_s >= args.min_duration]
        print(f"  {len(longform)} of {len(uploads)} are long enough to clip")
        video_ids = [v.video_id for v in longform]

    if not video_ids:
        print("  nothing to mine.")
        return 0

    videos = {v.video_id: v for v in client.videos_by_id(video_ids)}
    any_found = False
    for vid in video_ids:
        v = videos.get(vid)
        if not v or v.duration_s <= 0:
            continue
        raw = client.comments(vid, limit=args.comments)
        mentions = mentions_from_comments(raw, v.duration_s)
        cands = find_clips(mentions, v.duration_s, max_clips=args.limit,
                           min_authors=args.min_authors)
        if cands or args.video:
            any_found = any_found or bool(cands)
            print(render_clips(cands, v.title[:60]))
            print(f"  source: https://youtu.be/{vid}  "
                  f"({len(raw)} comments -> {len(mentions)} timestamp votes)")
    if not any_found:
        print("\n  No clippable moments cleared the bar on any video scanned.")
    return 0


def cmd_check(args, client):
    """Verify the API key end to end before committing to a long run."""
    import os
    print(render_check(check_key(os.environ.get("YOUTUBE_API_KEY"))))
    return 0


def cmd_backtest(args, client):
    """Predict view buckets before publication, then score against reality."""
    store = CorpusStore(args.corpus)
    videos = store.load_videos(shorts_only=not args.include_long)
    if not videos:
        print(f"error: {args.corpus} is empty - run 'harvest' first", file=sys.stderr)
        return 2

    samples = build_samples(videos)
    print(f"\n  {len(videos):,} videos -> {len(samples):,} with a usable channel history")
    if not samples:
        print("  No video has 3+ matured earlier uploads on its channel, so there is")
        print("  no honest channel prior to predict from. Harvest more per channel.")
        return 1

    r = run_backtest(samples, test_frac=args.test_frac, alpha=args.alpha,
                     use_channel_prior=not args.no_channel_prior)
    if r is None:
        print("  Not enough data for a temporal split (need 100 train / 30 test).")
        return 1
    print(render_backtest(r, args.corpus))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ytengine", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"ytengine {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, needs_corpus=True):
        sp.add_argument("--offline", action="store_true",
                        help="run on synthetic fixture data, no API key needed")
        sp.add_argument("--seed", type=int, default=7, help="fixture seed")
        sp.add_argument("--db", default="ytengine.db", help="snapshot database path")
        if needs_corpus:
            sp.add_argument("--niche", help="search phrase to discover channels")
            sp.add_argument("--channel", help="comma-separated channel ids (skips search)")
            sp.add_argument("--channels", type=int, default=15, help="channels to discover")
            sp.add_argument("--videos", type=int, default=100, help="uploads per channel")
            sp.add_argument("--quota", type=int, default=10000, help="quota budget for this run")
            sp.add_argument("--cache", default=".cache/yt", help="response cache dir")

    sp = sub.add_parser("outliers", help="find videos beating their own channel")
    common(sp)
    sp.add_argument("--min-multiplier", type=float, default=2.0)
    sp.add_argument("--max-subs", type=int, default=None,
                    help="ignore channels above this size")
    sp.add_argument("--include-provisional", action="store_true",
                    help="include videos too young for a settled multiplier")
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(fn=cmd_outliers)

    sp = sub.add_parser("packaging", help="fit and apply a title model")
    common(sp)
    sp.add_argument("--title", action="append", help="draft title to score (repeatable)")
    sp.add_argument("--thumbnail", help="path to a thumbnail image to check")
    sp.set_defaults(fn=cmd_packaging)

    sp = sub.add_parser("audit", help="keep/kill verdict per format")
    common(sp)
    sp.set_defaults(fn=cmd_audit)

    sp = sub.add_parser("backtest", help="predict view buckets, then score against reality")
    common(sp, needs_corpus=False)
    sp.add_argument("--corpus", default="corpus.db", help="corpus database to backtest on")
    sp.add_argument("--test-frac", type=float, default=0.25, help="most-recent share held out")
    sp.add_argument("--alpha", type=float, default=1.0, help="ridge penalty")
    sp.add_argument("--no-channel-prior", action="store_true",
                    help="ablation: drop channel history, leaving packaging and timing only")
    sp.add_argument("--include-long", action="store_true")
    sp.set_defaults(fn=cmd_backtest)

    sp = sub.add_parser("check", help="verify YOUTUBE_API_KEY works, and say what it unlocks")
    common(sp, needs_corpus=False)
    sp.set_defaults(fn=cmd_check)

    sp = sub.add_parser("clips", help="find clippable moments via comment timestamps")
    common(sp)
    sp.add_argument("--video", help="comma-separated video ids to mine")
    sp.add_argument("--scan", type=int, default=25, help="recent uploads to scan on a channel")
    sp.add_argument("--comments", type=int, default=500, help="comments to pull per video")
    sp.add_argument("--limit", type=int, default=10, help="clips to report per video")
    sp.add_argument("--min-authors", type=int, default=4,
                    help="distinct people who must mark a moment for it to count")
    sp.add_argument("--min-duration", type=int, default=600,
                    help="ignore uploads shorter than this many seconds")
    sp.set_defaults(fn=cmd_clips)

    sp = sub.add_parser("harvest", help="build a large corpus, resumably")
    common(sp)
    sp.add_argument("--target", type=int, default=30000, help="videos to add this run")
    sp.add_argument("--corpus", default="corpus.db", help="corpus database path")
    sp.add_argument("--seeds", help="comma-separated search seeds")
    sp.add_argument("--regions", help="comma-separated region codes")
    sp.add_argument("--windows", type=int, default=8, help="publish-window slices per seed/region")
    sp.add_argument("--discovery-slices", type=int, default=10,
                    help="max search slices per run (search costs 100u each)")
    sp.add_argument("--per-channel", type=int, default=500, help="uploads to walk per channel")
    sp.add_argument("--include-long", action="store_true", help="keep long-form too, not just shorts")
    sp.add_argument("--estimate", action="store_true", help="print quota cost and exit")
    sp.set_defaults(fn=cmd_harvest)

    sp = sub.add_parser("track", help="snapshot counters for velocity")
    common(sp)
    sp.set_defaults(fn=cmd_track)

    sp = sub.add_parser("trends", help="what is accelerating right now")
    common(sp, needs_corpus=False)
    sp.add_argument("--topics", help="comma-separated keywords to group by")
    sp.add_argument("--limit", type=int, default=25)
    sp.set_defaults(fn=cmd_trends)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if getattr(args, "offline", False) and args.cmd not in ("harvest", "clips", "check", "backtest"):
        print(OFFLINE_BANNER)
    elif args.cmd not in ("harvest", "clips", "check", "backtest") and hasattr(args, "niche") \
            and not (args.niche or args.channel):
        print("error: need --niche or --channel (or --offline to try it out)", file=sys.stderr)
        return 2

    if args.cmd == "clips" and not (args.video or args.channel):
        print("error: clips needs --video or --channel", file=sys.stderr)
        return 2

    client = None
    if not args.offline and hasattr(args, "quota"):
        client = YouTubeClient(cache_dir=args.cache, quota_budget=args.quota)

    try:
        rc = args.fn(args, client)
    except QuotaExceeded as e:
        print(f"\nquota exhausted: {e}", file=sys.stderr)
        return 3
    except (LookupError, RuntimeError) as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 1

    if client and client.quota_used:
        print(f"\n  quota spent this run: {client.quota_used}/{client.quota_budget} units")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
