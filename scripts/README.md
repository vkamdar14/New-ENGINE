# Three tech Shorts scripts

Fast, second-person, no intro, one idea each — the format that survives a feed.
All three validated by the craft engine: **~150 wpm, 89-90% caption coverage,
clean loop seam, zero errors.**

Timed transcripts (`*.txt`) drop straight into the editor:

```bash
python -m ytengine edit --source yourfootage.mp4 --start 0 --end 25.9 \
  --style punch --words-per-cue 2 --font-size 84 \
  --hook "YOUR CHARGER LIES" --transcript scripts/charger.txt
```

---

## 1. "Your charger is lying" — 25.9s

> your phone charger is lying to you
> it says sixty five watts on the brick
> your phone pulls eighteen
> because USB C negotiates
> the cable the phone and the charger
> all agree on the lowest one
> and that five dollar cable you bought
> caps the whole thing
> check for e marker on the cable
> no e marker means sixty watts max
> **your charger was never the problem**

Strongest of the three. Names a thing the viewer owns, blames an object they
did not suspect, and ends on a line that reframes the whole clip — which is
also what makes it loop.

## 2. "8GB is dead" — 25.0s

> eight gigs of RAM is done
> not slow, done
> chrome alone eats four
> windows takes three before you open anything
> so every tab now hits your SSD instead
> that stutter you blame on your CPU
> is your computer using storage as memory
> sixteen gigs costs thirty dollars used
> it is the cheapest upgrade you will ever make
> **and the only one you feel instantly**

"not slow, done" at 2s is the hook doing real work — a correction lands harder
than a claim.

## 3. "Stop buying consumer SSDs" — 23.4s

> stop buying consumer SSDs
> enterprise drives get dumped on eBay
> when data centers cycle them out
> you get power loss protection
> and ten times the write endurance
> for less than a samsung
> check the health hours before you buy
> under twenty thousand is basically new
> these were built to run for a decade
> **somebody already paid for that**

Most niche, so the smallest audience — but the highest save rate, because it
is actionable the moment it ends.

---

## Two notes from the engine

**2 words per cue, not 3.** At 3 the captions ran ~1,070-1,420px against a
960px safe area and would have wrapped mid-phrase. The craft validator caught
it; `--words-per-cue 2 --font-size 84` fixes it.

**Install Impact or Anton.** These rendered in DejaVu Sans because this
machine has neither, and libass substitutes silently. Impact is narrow enough
that 3 words per cue fits again.
