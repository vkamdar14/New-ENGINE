"""Preflight: does the key work, and what does it unlock?

Every failure mode here produces the same unhelpful symptom in normal use - a
403 - so this module's whole job is telling them apart. The four common causes
need four completely different fixes, and guessing wrong costs an afternoon:

    key absent          -> nothing was ever set
    API not enabled     -> key is valid, YouTube Data API v3 is off in the project
    key invalid/deleted -> the string is wrong
    key restricted      -> HTTP-referrer or IP restriction blocks server-side use
    quota exhausted     -> key is fine, come back after midnight Pacific

The referrer case is the one that wastes the most time: a key created with
"HTTP referrers" restriction works perfectly in a browser and fails from every
script, with an error that does not say so.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

PROBE_VIDEO = "dQw4w9WgXcQ"  # a video certain to exist; probe costs 1 unit


@dataclass
class CheckResult:
    ok: bool
    status: str
    detail: str
    fix: str = ""
    quota_spent: int = 0
    unlocked: list[str] = field(default_factory=list)


def _classify(code: int, body: str) -> tuple[str, str]:
    """Map an API error to a cause and the specific fix for it."""
    low = body.lower()
    if "has not been used in project" in low or "is disabled" in low or "accessnotconfigured" in low:
        return ("API not enabled",
                "Enable 'YouTube Data API v3' for this project: "
                "console.cloud.google.com -> APIs & Services -> Library -> "
                "search YouTube Data API v3 -> Enable. It can take a minute to take effect.")
    if "quota" in low:
        return ("quota exhausted",
                "The key works. Daily quota resets at midnight Pacific. "
                "Run 'harvest --estimate' to size runs against the 10,000-unit budget.")
    if "api key not valid" in low or "badrequest" in low or code == 400:
        return ("key invalid",
                "The key string is wrong or was deleted. Recreate it under "
                "APIs & Services -> Credentials -> Create credentials -> API key.")
    if "referer" in low or "referrer" in low or "blocked" in low or "ipreferrer" in low:
        return ("key restricted",
                "This key is restricted to HTTP referrers or specific IPs, so it works "
                "in a browser but not from a script. Edit the key and set Application "
                "restrictions to 'None' (keep the API restriction to YouTube Data API v3).")
    if code == 403:
        return ("forbidden",
                "403 without a recognised reason. Usually the API is not enabled on the "
                "project the key belongs to, or the key has application restrictions.")
    return (f"HTTP {code}", "Unexpected response; the raw body is shown above.")


def check_key(api_key: Optional[str], timeout: int = 20) -> CheckResult:
    """One cheap live call that proves the whole path works."""
    if not api_key:
        return CheckResult(
            ok=False, status="no key",
            detail="YOUTUBE_API_KEY is not set in this environment.",
            fix="Create one (free, no billing) at console.cloud.google.com, then "
                "export YOUTUBE_API_KEY=... before running any command.",
        )

    url = ("https://www.googleapis.com/youtube/v3/videos?"
           + urllib.parse.urlencode({"part": "snippet,statistics", "id": PROBE_VIDEO,
                                     "key": api_key}))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:2000]
        status, fix = _classify(e.code, body)
        return CheckResult(ok=False, status=status, detail=_message(body), fix=fix, quota_spent=1)
    except Exception as e:  # network, DNS, proxy
        return CheckResult(
            ok=False, status="unreachable",
            detail=f"{type(e).__name__}: {e}",
            fix="googleapis.com could not be reached. Check outbound network access.",
        )

    items = data.get("items", [])
    if not items:
        return CheckResult(
            ok=False, status="empty response", quota_spent=1,
            detail="The API answered but returned no items for a video that exists.",
            fix="Unusual - typically a region or availability restriction on the probe video.",
        )

    return CheckResult(
        ok=True, status="working", quota_spent=1,
        detail=f"Read '{items[0]['snippet']['title'][:50]}' "
               f"({int(items[0]['statistics'].get('viewCount', 0)):,} views).",
        unlocked=[
            "harvest   build a large Shorts corpus (~30k/day on one key)",
            "clips     mine comment timestamps for clippable moments",
            "outliers  find videos beating their own channel baseline",
            "packaging fit a title model on a real niche",
            "audit     keep/kill verdict per format",
            "track     snapshot counters so velocity becomes measurable",
        ],
    )


def _message(body: str) -> str:
    """Pull the human sentence out of Google's error envelope.

    Dumping raw JSON at someone buries the one line that says what went wrong
    under twenty that do not.
    """
    try:
        err = json.loads(body).get("error", {})
        msg = err.get("message") or ""
        reason = ""
        details = err.get("errors") or []
        if details and isinstance(details, list):
            reason = details[0].get("reason", "")
        return f"{msg} [{reason}]" if reason else msg or body[:200]
    except (ValueError, AttributeError, TypeError):
        return body[:200]


def render(r: CheckResult) -> str:
    out = ["", "API KEY CHECK", "=============", ""]
    out.append(f"  status: {r.status.upper()}")
    if r.detail:
        out.append(f"  {r.detail}")
    if r.quota_spent:
        out.append(f"  quota spent by this check: {r.quota_spent} unit")
    if r.fix:
        out.append("")
        for line in _wrap(r.fix, 72):
            out.append(f"  fix: {line}" if line == _wrap(r.fix, 72)[0] else f"       {line}")
    if r.unlocked:
        out.append("\n  commands now available:")
        for u in r.unlocked:
            out.append(f"    {u}")
        out.append("\n  Note: views, likes, comments and durations are all public data and")
        out.append("  reachable with this key. Click-through rate, retention and impressions")
        out.append("  are NOT - they live in the YouTube Analytics API, need OAuth, and only")
        out.append("  ever cover channels you own.")
    return "\n".join(out)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines
