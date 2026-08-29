"""Zero-quota channel monitoring via RSS.

`youtube.com/feeds/videos.xml?channel_id=...` returns a channel's 15 most
recent uploads and costs **no API quota at all**. That changes what monitoring
can afford: watching 1,000 competitors through the Data API means burning
quota every cycle, while through RSS it is free and can run hourly forever.

The feed is deliberately thin - id, title, published, author, and rough
statistics. That is exactly enough to answer "did anything new appear, and is
it moving", and the expensive Data API call can then be spent only on the
handful of videos that turn out to be worth hydrating.

The pattern this enables: RSS discovers and triages for free, the API is
reserved for depth on the few things that matter.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

FEED = "https://www.youtube.com/feeds/videos.xml?channel_id={}"

# The feed is small, well-formed Atom with a fixed shape, so targeted regexes
# beat pulling in an XML parser and the attack surface that comes with it.
_ENTRY = re.compile(r"<entry>(.*?)</entry>", re.S)
_FIELDS = {
    "video_id": re.compile(r"<yt:videoId>(.*?)</yt:videoId>"),
    "channel_id": re.compile(r"<yt:channelId>(.*?)</yt:channelId>"),
    "title": re.compile(r"<title>(.*?)</title>", re.S),
    "published": re.compile(r"<published>(.*?)</published>"),
    "author": re.compile(r"<name>(.*?)</name>", re.S),
}
_VIEWS = re.compile(r'<media:statistics views="(\d+)"')
_RATING = re.compile(r'<media:starRating[^>]*count="(\d+)"')


def _unescape(s: str) -> str:
    for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&quot;", '"'), ("&#39;", "'")):
        s = s.replace(a, b)
    return s.strip()


@dataclass
class FeedItem:
    video_id: str
    channel_id: str
    channel_title: str
    title: str
    published_at: datetime
    views: Optional[int] = None
    likes: Optional[int] = None

    @property
    def url(self) -> str:
        return f"https://youtu.be/{self.video_id}"

    def age_hours(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(timezone.utc)
        return max((now - self.published_at).total_seconds() / 3600.0, 0.0)

    def views_per_hour(self, now: Optional[datetime] = None) -> Optional[float]:
        """Early velocity. None when the feed omitted view counts.

        Age is floored at one hour: a video published twelve minutes ago would
        otherwise report a rate five times its true pace and dominate any
        ranking built on this.
        """
        if self.views is None:
            return None
        return self.views / max(self.age_hours(now), 1.0)


def parse_feed(xml: str) -> list[FeedItem]:
    out: list[FeedItem] = []
    for block in _ENTRY.findall(xml):
        got = {}
        for key, rx in _FIELDS.items():
            m = rx.search(block)
            if not m:
                break
            got[key] = _unescape(m.group(1))
        else:
            v = _VIEWS.search(block)
            r = _RATING.search(block)
            try:
                published = datetime.fromisoformat(got["published"].replace("Z", "+00:00"))
            except ValueError:
                continue
            out.append(FeedItem(
                video_id=got["video_id"], channel_id=got["channel_id"],
                channel_title=got["author"], title=got["title"],
                published_at=published.astimezone(timezone.utc),
                views=int(v.group(1)) if v else None,
                likes=int(r.group(1)) if r else None,
            ))
    return out


def fetch_channel(channel_id: str, timeout: int = 20) -> list[FeedItem]:
    """One channel's recent uploads. Costs nothing.

    Returns empty rather than raising on a dead or private channel: a monitor
    sweeping a thousand feeds must not stop at the first one that 404s.
    """
    try:
        with urllib.request.urlopen(FEED.format(channel_id), timeout=timeout) as resp:
            return parse_feed(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return []


def sweep(channel_ids: Iterable[str], timeout: int = 20) -> list[FeedItem]:
    out: list[FeedItem] = []
    for cid in channel_ids:
        out.extend(fetch_channel(cid, timeout))
    out.sort(key=lambda i: i.published_at, reverse=True)
    return out


def render(items: list[FeedItem], limit: int = 30,
           now: Optional[datetime] = None) -> str:
    o = ["", "RSS SWEEP (zero quota)", "======================", ""]
    if not items:
        return "\n".join(o + ["  No feeds returned entries.",
                              "  Check the channel ids - RSS needs the UC... form."])
    o.append(f"  {len(items)} uploads across "
             f"{len({i.channel_id for i in items})} channels\n")
    o.append(f"  {'age':>7} {'views':>10} {'v/hr':>9}  channel / title")
    o.append("  " + "-" * 74)
    for i in items[:limit]:
        vph = i.views_per_hour(now)
        o.append(f"  {i.age_hours(now):6.0f}h {('-' if i.views is None else f'{i.views:,}'):>10} "
                 f"{('-' if vph is None else f'{vph:,.0f}'):>9}  "
                 f"{i.channel_title[:18]:<18} {i.title[:34]}")
    o.append("\n  Feeds carry the 15 most recent uploads only, so a sweep run less")
    o.append("  often than a channel posts will miss videos. Hourly is safe for")
    o.append("  almost anyone; daily is not, for a channel posting several a day.")
    return "\n".join(o)
