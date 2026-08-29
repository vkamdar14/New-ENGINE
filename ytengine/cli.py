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
from .client import QuotaExceeded, YouTubeClient
from .fixtures import make_corpus
from .metrics import AgeCurve, score_channel
from .outliers import extract_patterns, find_opportunities
from .packaging import fit_packaging, thumbnail_stats
from .store import SnapshotStore
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

    if getattr(args, "offline", False):
        print(OFFLINE_BANNER)
    elif hasattr(args, "niche") and not (args.niche or args.channel):
        print("error: need --niche or --channel (or --offline to try it out)", file=sys.stderr)
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
