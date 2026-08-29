"""YouTube Data API v3 client: quota-aware, cached, stdlib-only.

Quota is the binding constraint on this whole engine. A default project gets
10,000 units/day, and the costs are wildly uneven:

    search.list         100 units   <- returns 50 items
    playlistItems.list    1 unit    <- returns 50 items
    videos.list           1 unit    <- returns up to 50 items, fully hydrated
    channels.list         1 unit

So search.list is 100x the price of the same data reached another way. Pulling
a 200-video channel through search costs 400 units; reaching it through the
channel's uploads playlist costs 4 + 4 = 8. That is the difference between
auditing 25 channels a day and auditing 1,000.

This client therefore never uses search.list for anything reachable by
playlist, and every response is cached to disk so re-running an analysis on
the same corpus costs zero quota.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterator, Optional

from .models import Channel, Video

API_ROOT = "https://www.googleapis.com/youtube/v3"

QUOTA_COST = {"search": 100, "videos": 1, "channels": 1, "playlistItems": 1}


class QuotaExceeded(RuntimeError):
    pass


class YouTubeClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str | Path = ".cache/yt",
        quota_budget: int = 10000,
        ttl_s: int = 6 * 3600,
    ):
        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY", "")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.quota_budget = quota_budget
        self.quota_used = 0
        self.ttl_s = ttl_s

    # ---------------- transport ----------------

    def _cache_path(self, endpoint: str, params: dict) -> Path:
        blob = endpoint + json.dumps(params, sort_keys=True)
        return self.cache_dir / f"{hashlib.sha256(blob.encode()).hexdigest()[:24]}.json"

    def _get(self, endpoint: str, params: dict) -> dict:
        params = {k: v for k, v in params.items() if v is not None}
        cache_key = dict(params)
        cache_key.pop("key", None)
        path = self._cache_path(endpoint, cache_key)

        if path.exists() and (time.time() - path.stat().st_mtime) < self.ttl_s:
            return json.loads(path.read_text())

        cost = QUOTA_COST.get(endpoint, 1)
        if self.quota_used + cost > self.quota_budget:
            raise QuotaExceeded(
                f"{endpoint} would cost {cost}u; {self.quota_used}/{self.quota_budget} already spent. "
                "Raise --quota, wait for the midnight-Pacific reset, or narrow the pull."
            )
        if not self.api_key:
            raise RuntimeError(
                "No API key. Set YOUTUBE_API_KEY, or run against a fixture corpus with --offline."
            )

        params["key"] = self.api_key
        url = f"{API_ROOT}/{endpoint}?{urllib.parse.urlencode(params)}"
        data = self._request_with_retry(url)
        self.quota_used += cost
        path.write_text(json.dumps(data))
        return data

    def _request_with_retry(self, url: str, attempts: int = 4) -> dict:
        delay = 2.0
        last: Exception | None = None
        for _ in range(attempts):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:400]
                # 403 here is usually quotaExceeded, which retrying cannot fix.
                if e.code == 403 and "quota" in body.lower():
                    raise QuotaExceeded(f"API reports quota exhausted: {body}") from e
                if e.code in (500, 503):
                    last = e
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"HTTP {e.code} from {url.split('?')[0]}: {body}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"gave up after {attempts} attempts: {last}")

    def _paginate(self, endpoint: str, params: dict, cap: int) -> Iterator[dict]:
        token, seen = None, 0
        while seen < cap:
            page = self._get(endpoint, {**params, "pageToken": token, "maxResults": 50})
            for item in page.get("items", []):
                yield item
                seen += 1
                if seen >= cap:
                    return
            token = page.get("nextPageToken")
            if not token:
                return

    # ---------------- public surface ----------------

    def channel(self, channel_id: str) -> Channel:
        data = self._get("channels", {"part": "snippet,statistics,contentDetails", "id": channel_id})
        items = data.get("items", [])
        if not items:
            raise LookupError(f"no channel {channel_id}")
        return Channel.from_api(items[0])

    def channels(self, channel_ids: list[str]) -> list[Channel]:
        """Batched - 50 channels for a single quota unit."""
        out = []
        for i in range(0, len(channel_ids), 50):
            chunk = channel_ids[i : i + 50]
            data = self._get(
                "channels", {"part": "snippet,statistics,contentDetails", "id": ",".join(chunk)}
            )
            out.extend(Channel.from_api(it) for it in data.get("items", []))
        return out

    def videos_by_id(self, video_ids: list[str]) -> list[Video]:
        """Hydrate ids into full Video records, 50 per unit."""
        out = []
        for i in range(0, len(video_ids), 50):
            chunk = video_ids[i : i + 50]
            data = self._get(
                "videos", {"part": "snippet,statistics,contentDetails", "id": ",".join(chunk)}
            )
            out.extend(Video.from_api(it) for it in data.get("items", []))
        return out

    def channel_uploads(self, channel: Channel, limit: int = 200) -> list[Video]:
        """Every upload, newest first - the cheap path.

        Two units per 50 videos (one playlistItems page + one videos batch),
        versus 100 units per 50 through search.list.
        """
        if not channel.uploads_playlist:
            return []
        ids = [
            item["contentDetails"]["videoId"]
            for item in self._paginate(
                "playlistItems",
                {"part": "contentDetails", "playlistId": channel.uploads_playlist},
                cap=limit,
            )
            if item.get("contentDetails", {}).get("videoId")
        ]
        return self.videos_by_id(ids)

    def search_video_ids(
        self,
        query: str,
        limit: int = 50,
        region: str | None = None,
        published_after: str | None = None,
        published_before: str | None = None,
        video_duration: str | None = None,
    ) -> list[str]:
        """Search for video ids. The expensive call - 100 units per 50 results.

        `search.list` refuses to paginate past roughly 500 results for any one
        query, however many pageTokens you feed it. That cap is per *query*,
        not per key, so the way to breadth is many narrow queries - the same
        term sliced by region and by publish window - rather than one broad
        query paginated harder. `harvest.py` builds those slices.

        `video_duration="short"` asks YouTube for sub-4-minute videos, which
        is the closest server-side filter to Shorts; the real <=180s test still
        happens client-side after hydration.
        """
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "regionCode": region,
            "publishedAfter": published_after,
            "publishedBefore": published_before,
            "videoDuration": video_duration,
            "order": "viewCount",
        }
        out, seen = [], set()
        for it in self._paginate("search", params, cap=limit):
            vid = it.get("id", {}).get("videoId")
            if vid and vid not in seen:
                seen.add(vid)
                out.append(vid)
        return out

    def search_channels(self, query: str, limit: int = 25) -> list[str]:
        """The one place search.list is unavoidable: discovering a niche.

        Costs 100 units per page, so it is called once to seed a corpus and
        never again - afterwards you work from the cached channel ids.
        """
        items = list(
            self._paginate("search", {"part": "snippet", "q": query, "type": "channel"}, cap=limit)
        )
        seen, ids = set(), []
        for it in items:
            cid = it.get("snippet", {}).get("channelId") or it.get("id", {}).get("channelId")
            if cid and cid not in seen:
                seen.add(cid)
                ids.append(cid)
        return ids
