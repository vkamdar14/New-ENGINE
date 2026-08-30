# Tech scripts v2 — rebuilt against the CarterPCs data

The v1 scripts opened second-person ("your charger is lying to you"). Measured
on his 480 mature Shorts, second person runs **0.79x** while first person runs
**2.50x** and the specific "I \<verb\>ed…" opener runs **2.37x**. So these are
rewritten as *I did a physical thing to an object you own, here is what
happened*.

The bigger constraint is the one that came out of the engagement analysis:
his engagement rate correlates with views at **−0.636**, monotonic across all
five quintiles — lowest-engagement Shorts get 4x the views of his highest.
High engagement means a video stayed inside an audience that already cared.
So the test for every line below is **can a stranger understand the promise
without already being interested in tech.** Everyone owns a cable. Nobody
needs to know what an e-marker is first.

All three validated: 163–167 wpm, 91–92% caption coverage, clean loop seams,
zero craft errors.

```bash
python -m ytengine edit --source yourfootage.mp4 --start 0 --end 27.6 \
  --style punch --words-per-cue 2 --font-size 84 \
  --hook "SIX CABLES ONE PHONE" --transcript scripts/v2_cables.txt
```

---

## 1. "Six cables, one phone" — 27.6s  ← lead with this one

> i plugged six different cables
> into the same phone and the same charger
> same wall same everything
> **the three dollar one hit sixty watts**
> the twenty dollar braided one
> capped at eighteen
> because the expensive one had no e marker chip
> it is a wire, thats it
> the chip is what tells the charger
> hey you can send more power
> so half the cables in your drawer
> **are throttling a charger you already paid for**

Strongest of the three, and it is the one closest to *"I swapped my keyboard's
WASD with a joystick.."* (28M): a visible physical test on an object in
everyone's drawer, with a result that inverts the expected answer. The cheap
one winning is the whole video.

Shoot it as six cables in frame. The visual carries the first second.

## 2. "I bought the cheapest SSD on Amazon" — 27.3s

> i put the cheapest SSD on amazon
> into my main pc
> twenty two dollars, no brand name
> it benchmarked at four thousand megabytes a second
> same as a samsung
> then i checked the health after a month
> **eleven percent of its life, gone**
> the speed was real
> the endurance was not
> cheap drives use the fast cells as a buffer
> then fall off a cliff when its full
> **it is quick until the day it isnt**

Two-act structure: it looks like a win, then the turn at "then i checked the
health". The reversal is what earns the second half.

## 3. "I swapped his RAM and said nothing" — 28.5s

> i swapped my friends thirty two gigs of ram
> down to eight and said nothing
> he messaged me in two days
> asking if his gpu was dying
> **it was not the gpu**
> windows takes three gigs before you open anything
> chrome takes four
> so every tab was hitting his SSD instead of memory
> that is the stutter everybody blames on their graphics card
> sixteen gigs is thirty dollars used
> i told him after a week
> **he is still mad**

The only one with a person in it, which is why it ends on a joke rather than a
spec. "he is still mad" is the loop line — it sends you back to the setup.

---

## What changed from v1

| | v1 | v2 |
|---|---|---|
| frame | second person ("your charger…") | first person ("i plugged six…") |
| measured lift | 0.79x | 2.37–2.50x |
| length | 23–26s | 27–29s (his median is 32s) |
| pace | 149–155 wpm | 163–167 wpm |
| ending | a fact | a reversal or a joke |

Dropped entirely: `??` (0.62x in his data), `$` in the hook (0% of his top
decile), and anything shaped like a debate prompt — measured at 0.86x views
for no engagement gain.
