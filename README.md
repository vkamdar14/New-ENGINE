# ytengine

A YouTube growth engine. It measures what actually earns views so you can
decide what to make next.

It does **not** inflate views. There are no bots, proxies, autoplay loops or
retention fakers here, and there will not be. Those get channels demonetised
and terminated, and the views they produce are worthless anyway - inflated
watch time teaches the recommendation system that your video disappoints real
audiences, which suppresses the distribution you were trying to buy. The way
to a million views is a video the algorithm can profitably show to strangers.
This tool helps you find out what that is.

## What it does

Four analyses, all built on one number: the **multiplier** - how many times its
own channel's normal performance a video achieved.

Raw view counts cannot guide decisions. 200k views is a disaster on a channel
that normally does 2M and a career-maker on one that normally does 5k.
Dividing by the channel's own baseline removes channel size - the variable you
cannot copy - and leaves format and packaging, which are the variables you can.

| Command | Question it answers |
|---|---|
| `outliers` | Which videos in this niche beat their own channel, and what do they share? |
| `packaging` | Which title features actually predict overperformance *here*? Score my drafts. |
| `audit` | On my channel, which formats should I double down on and which should I kill? |
| `trends` | Which topics are still accelerating, so I publish into a rising curve? |
| `harvest` | Build a large Shorts corpus, resumably, across as many days as it takes. |
| `clips` | Which moments in a long video are worth cutting into Shorts? |
| `backtest` | Predict a video's view bucket before publishing, then score it against reality. |

## Getting an API key

Free, no billing, about three minutes. The YouTube Data API has no paid tier -
you get 10,000 quota units/day per project and cannot buy more with a card.

1. Go to **console.cloud.google.com** and sign in with any Google account.
2. Create a project (top-left project dropdown -> New Project). Any name.
3. **APIs & Services -> Library**, search **YouTube Data API v3**, click **Enable**.
4. **APIs & Services -> Credentials -> Create credentials -> API key**. Copy it.
5. Restrict it: click the key, set **API restrictions** to *YouTube Data API v3*.
   Leave **Application restrictions** on **None** - the "HTTP referrers" option
   makes the key work in a browser and fail from every script, with an error
   that never says so.

```bash
export YOUTUBE_API_KEY=AIza...
python -m ytengine check
```

`check` spends 1 unit proving the whole path works, and tells apart the four
failures that all look like a bare 403: key absent, API not enabled on the
project, key invalid, key restricted, quota exhausted. Each needs a different
fix and guessing wrong costs an afternoon.

### What a key does and does not reach

| Data | Source | Needs |
|---|---|---|
| views, likes, comments, duration, titles, tags | Data API v3 | API key |
| comment text and timestamps | Data API v3 | API key |
| **CTR, retention, impressions, traffic sources** | **Analytics API** | **OAuth, own channel only** |

That second row is the one to plan around: click-through rate and retention are
never available for other people's videos, at any price. Every competitor-facing
number in this engine is derived from public counts, and the reports label the
engagement-rate proxy as a proxy rather than passing it off as CTR.

### If 10,000 units/day is not enough

It probably is - `harvest --estimate` puts 30k Shorts at ~1,500 units. If you
genuinely outgrow it, the legitimate route is the **YouTube API Services quota
extension** form, which requires an audit of your application. Spinning up
extra Google Cloud projects to multiply quota is a Terms of Service violation
and gets keys revoked; it is not a scaling strategy.

## Quickstart

No API key needed to try it - every command takes `--offline` and runs the
full pipeline against synthetic data:

```bash
python -m ytengine outliers  --offline
python -m ytengine audit     --offline
python -m ytengine packaging --offline --title "7 Espresso Mistakes" --title "Why Espresso Tastes Bad"
```

Against real data:

```bash
export YOUTUBE_API_KEY=...

# what overperforms in a niche, biased toward small channels
python -m ytengine outliers --niche "home espresso" --max-subs 100000

# fit a title model on that niche, then rank your drafts against it
python -m ytengine packaging --niche "home espresso" \
    --title "7 Espresso Mistakes Killing Your Shots" \
    --title "Why Your Espresso Tastes Bad" \
    --thumbnail ./draft.jpg

# keep/kill by format on your own channel
python -m ytengine audit --channel UCxxxxxxxxxxxxxxxxxxxxxx

# velocity needs two observations - snapshot on a schedule, then read trends
python -m ytengine track  --channel UCxxxx --db yt.db   # cron this, hourly
python -m ytengine trends --db yt.db --topics "espresso,grinder,latte"
```

Python 3.11+. No dependencies. Pillow is optional and only enables
thumbnail checks.

## Backtest: does any of this actually predict anything?

```bash
python -m ytengine backtest --corpus corpus.db
python -m ytengine backtest --corpus corpus.db --no-channel-prior   # ablation
```

Everything else in this repo describes the past. This makes a falsifiable
forward claim and scores it. Buckets are absolute, because for a channel
starting from zero the question is "did it break out", not "did it beat my own
median": FLOP <5k, JAIL 5k-25k, OK 25k-75k, GOOD 75k-250k, VIRAL 250k+.

Three rules keep the score meaningful. Break any one and the number goes *up*
while the model gets worse - which is why they are pinned by tests:

1. **Pre-publication features only.** Views, likes and comments are outcomes.
   Using them predicts views from views: a perfect score and no use at all.
2. **Temporal split, never random.** Train on the past, test on the future. A
   random split lets the model see a channel's July while predicting its June.
3. **Scored against baselines that need no model.** Always-guess-the-commonest
   already scores 35%. The baseline that matters is *channel history alone*.

### The result, on 6,114 real Shorts

| | exact bucket | MAE (log10) |
|---|---|---|
| always guess FLOP | 34.8% | - |
| **channel history alone** | **66.9%** | **0.325** |
| full model, 22 features | 67.6% | 0.361 |

**Verdict: no signal beyond channel history.** The model crushes the naive
baseline (67.6% vs 34.8%) and that number would look great quoted on its own.
It is not: essentially all of it comes from knowing *which channel posted the
video*. Every packaging and timing feature together adds ~0.7 points of
accuracy and makes the magnitude error worse.

Modelling the *residual* - over/under-performance against a channel's own
median, which is the only part you can actually change - gives out-of-sample
R-squared of **-0.07** and rank correlation **+0.22**. Weak, real, and nowhere
near enough to forecast a view count.

One thing did help. All the structural features (title length, digit counts,
emoji, hashtags) achieved nothing; adding *topic* tokens - what the video is
about - moved residual R-squared from -0.23 to -0.07 and rank correlation from
0.16 to 0.22. Topic beats packaging, which is exactly what the packaging
section of this README predicted and what most title-optimiser tools deny.

This is the honest state. Reporting 67.6% without the 66.9% next to it would be
the single most misleading thing this project could do.

### Sampling decides everything

The first real harvest returned 12,134 Shorts with **median views of 4.6
million** and a 5th percentile of 181k - not one flop in the entire corpus,
because discovery ordered search results by `viewCount` and so only ever found
mega-channels. A classifier trained on that learns "everything goes viral" and
scores beautifully on its own held-out split.

Ordering by date helped only marginally (94% still VIRAL). What actually fixed
it was **seed choice**: swapping broad seeds ("shorts", "viral", "comedy") for
long-tail ones ("sourdough starter day 3", "excel pivot table tutorial")
produced a corpus that is 48.7% FLOP, 33.8% JAIL, 1.8% VIRAL - a real
distribution with real failures. Popular-sounding seeds were a bigger source of
bias than the sort order was.

## Clip mining

```bash
# every clippable moment in one VOD
python -m ytengine clips --video VIDEOID

# scan a creator's recent long uploads and rank moments across all of them
python -m ytengine clips --channel UCxxxx --scan 25 --min-authors 5
```

Clipping a four-hour stream is not an editing problem, it is a *search*
problem: there are ~480 candidate 30-second windows in four hours and maybe
six are worth posting.

YouTube knows exactly which moments get re-watched - that is the
`mostReplayed` heatmap on the scrubber - but it is not in the Data API. The
best public proxy is nearly free at **1 unit per 100 comments**: viewers
timestamp the moments they want to re-watch. A comment reading "3:47 killed
me" is a human vote for a specific second, and those votes concentrate hard.

Details that decide whether the output is usable:

- **Windows open ~14s *before* the marked second.** Viewers timestamp the
  payoff, not the setup. A clip that opens on the punchline has no context and
  dies in its first two seconds. This is the single highest-impact choice here.
- **Kernel density, not a histogram.** Viewers' clocks disagree by a few
  seconds, so one moment gets marked at 3:45, 3:47 and 3:48. Hard bin edges
  split that across two bins and can hide the best moment in the video.
- **Unique authors, not mention count.** One enthusiast posting the same
  timestamp twelve times is one vote. Without this the ranking fills with
  single-fan moments.
- **Likes weighted logarithmically.** A heavily-liked comment means many people
  agreed, but linear weighting lets one viral comment outvote fifty independent
  viewers.
- **Non-maximum suppression.** Two peaks four seconds apart are one joke, not
  two clips.
- **Both timestamp conventions.** On long VODs viewers mix `1:23:20` and
  `83:20`. Capping minutes at 59 silently discards every mark of the second
  kind, precisely on the long videos clip mining exists for.

Against a simulated 4-hour VOD with five planted moments, all five come back as
the top five - 20-32 distinct people and 100-178x sharpness - cleanly separated
from noise peaks at 4-5 people and ~10x.

Precision comes from the duration bound, not the regex: `$5:00` and `2:1` parse
as clock-shaped, and are rejected because they fall outside the video's length.

### On clipping other people's content

Check the creator's policy before building a channel on it. Many streamers
explicitly welcome clip channels, some require credit, and some do not permit
monetized reuploads. That is a permissions question rather than a technical
one, and it is what decides whether the channel survives.

## Harvesting at scale

```bash
python -m ytengine harvest --estimate --target 30000     # cost check, no key needed
python -m ytengine harvest --target 30000 --corpus corpus.db   # then run it daily
```

**What is not possible:** there is no way to get every Short. The Data API has
no enumerate-all endpoint, `search.list` stops returning new results after
roughly 500 per query however hard you paginate, and YouTube's index is not
exposed. A complete census is out of reach for any third party at any budget -
a billion Shorts would be ~20M quota units, about 2,000 days on a default key.

**What is possible is large and compounds.** The two phases have wildly
different economics:

| phase | call | cost | capped? |
|---|---|---|---|
| discovery | `search.list` | 100 units / 50 results | yes, ~500 per query |
| expansion | uploads playlist walk | 2 units / 50 videos | no |

Expansion is 50x cheaper per video and has no ceiling, so the strategy is to
spend a little quota finding *channels* and the rest walking their entire back
catalogues. The 500-result cap is per *query*, not per key, so breadth comes
from slicing the same seed across regions and publish windows - the default
grid is 20 seeds x 10 regions x 8 windows = 1,600 distinct queries.

Measured against a simulated API, one default 10,000-unit key sustains roughly
**30,000 Shorts per day** using about two thirds of the budget, and the corpus
keeps growing every day the harvester is re-run:

```
day 1: +30,018   quota 6,912u   corpus  30,018
day 2: +30,192   quota 6,104u   corpus  60,210
day 3: +30,191   quota 6,104u   corpus  90,401
day 5: +30,190   quota 6,104u   corpus 150,607
```

Everything is checkpointed in SQLite - retired query slices, walked channels,
stored videos - so a crash or a quota reset costs nothing but the current
batch. Re-running resumes rather than restarting.

Videos are inserted with `INSERT OR IGNORE`, never `REPLACE`: a Short
re-encountered next week must keep its original counters, or the age-vs-views
relationship every baseline depends on is silently destroyed. Re-observation
is a different job and belongs in the snapshot store.

## Quota is the real constraint

A default API project gets 10,000 units/day, and endpoint costs are wildly
uneven:

| Endpoint | Cost | Returns |
|---|---|---|
| `search.list` | **100** | 50 items |
| `playlistItems.list` | 1 | 50 items |
| `videos.list` | 1 | up to 50 items, fully hydrated |
| `channels.list` | 1 | up to 50 channels |

So pulling a 200-video channel through `search` costs 400 units; reaching the
same videos through the channel's uploads playlist costs 8. That is the
difference between auditing 25 channels a day and auditing 1,000.

This client never uses `search.list` for anything reachable another way - it
appears once, to discover channels in a niche, and never again. Every response
is cached to disk, so re-running an analysis on the same corpus is free.

## How the multiplier is computed

1. **Split by format.** Shorts and long-form are different distribution
   systems and never share a baseline.
2. **Baseline on mature videos only.** A 2-day-old upload in the baseline
   would drag the median down and inflate every multiplier measured against it.
3. **Leave-one-out.** A breakout is excluded from the baseline it is scored
   against, so it cannot raise its own bar and disguise a 10x as a 6x.
4. **Age-adjust.** A 4-day-old video has not finished earning its views.
   We divide by an empirical age curve fitted on the corpus.

The age curve is fitted on *within-channel* view ratios rather than raw
medians. Channel sizes in a normal corpus span three orders of magnitude, so
whether an age bucket happens to contain big channels or small ones would
otherwise move the fitted fraction far more than age does.

## What it deliberately will not tell you

- **Precise forecasts.** Title scores rank drafts against each other. They are
  not predictions of absolute views, and the tool says so wherever it prints one.
- **A high R².** Packaging is a minority of the variance in video performance;
  topic and audience fit dominate. Expect R² around 0.1-0.25. Anything above
  ~0.4 on this feature set is overfitting, and the reports flag weak fits
  rather than hiding them.
- **Findings from thin data.** Baselines below 5 mature videos are marked
  unreliable, patterns below 4 supporting outliers are dropped, and the
  packaging model refuses to fit under 20 videos.
- **CTR, retention, or impressions.** Those live in YouTube Analytics and are
  only available to a channel's owner via OAuth. Everything here is computed
  from public Data API fields. Engagement rate is used as a weak proxy and is
  labelled as one.

## Assumptions worth knowing

The age curve assumes channels in the corpus were producing
comparable-performing videos across the whole window. A channel that doubled
in size last month genuinely has better recent videos, and the curve reads
that as "videos mature fast". Across many channels this largely cancels; on a
single small channel it does not, which is why the fit refuses below a minimum
sample and falls back to a conservative default instead.

## Tests

```bash
python -m unittest discover -s tests -v
```

138 tests. The load-bearing ones are not the arithmetic checks - they are the
pair that plant a known effect in synthetic data and assert the engine
recovers it, *and* plant nothing and assert it stays quiet. A pattern finder
that always finds a pattern is a random number generator with a table.

The harvest suite runs against a fake client that charges real quota costs -
a test needing live quota is a test nobody runs - and asserts the property a
live test could never check deterministically: that a run interrupted by a
dead quota resumes rather than restarts.
