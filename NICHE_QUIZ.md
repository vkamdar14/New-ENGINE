# The format that needs nothing: football quiz Shorts

You do not have FC 26 or a PS5. This one needs **no footage, no photos, no
game, no camera, no football ability** — a question, a countdown, a list. It
is generated entirely from text, which is why `ytengine/quizgen.py` can output
the finished video.

```bash
python -c "
from ytengine.quizgen import Quiz, build_ass, ffmpeg_command
q = Quiz(question=r'Can you name the top 5\NPremier League goalscorers?',
         answers=['...','...','...','...','...'],
         subtitles=['260','208','187','177','175'])
open('q.ass','w').write(build_ass(q))
print(' '.join(ffmpeg_command(q,'q.ass','q.mp4')))"
```

## What the data says — both halves of it

**The format works.** In the 9,002-video football harvest, "Can you name..."
quizzes ran **7.78x the corpus median** — 267,751 views against 34,425.

**But that number is not proof you can do it.** Every one of those 15 quiz
videos came from a single channel: Tifo Football, 1.81M subs, owned by The
Athletic. A big publisher doing well is not evidence a format travels cold.

So I measured small channels running it separately — 118 quiz Shorts from the
last 60 days:

| | |
|---|---|
| median competitor size | **112 subs** |
| median views | **1,103** |
| best from a channel under 20k | **271,862** (from **672 subs**) |

**Read that honestly: the floor is brutal and the ceiling is real.** Most quiz
Shorts do a thousand views. The format has a very low floor and a very high
top end, which means volume is the only strategy that reaches the top end.

## The one thing the winner did differently

The 672-sub channel that hit 271k and 160k was not posting generic quizzes:

> ⚽Guess the Football Player: **TURKEY** Edition 🇹🇷
> ⚽Guess the Football Player: **ENGLAND** Edition 🏴

**Country editions.** A national audience shows up for its own flag in a way
nobody shows up for "guess the player". Same production cost, and it is the
difference between 1,100 and 271,000 in the same 60-day window.

## The structure, from the videos that hit

```
0.0s   question on screen, nothing else
1.5s   countdown starts — ●●●●●● 6
~6s    answers reveal one at a time, ranked
end    hold the full list
```

The countdown is the mechanism, not decoration. It turns a scroll into a
commitment: the viewer stays because they are trying to beat it. That is the
retention the whole format runs on.

## Before you post anything

**Verify every fact against a current source.** My knowledge stops in May 2026
and a World Cup was scheduled for June–July 2026, so any all-time football list
I generate may be a tournament out of date. The demo video uses the pre-2026
top scorers and could already be wrong.

Getting a stat wrong in a quiz is worse than in any other format — the comments
exist to correct you, and the correction becomes the video's story.

## Realistic economics

Sports/football Shorts RPM is low — call it $2-3 long-form, and Shorts earn
roughly 2% of that. This is an audience-building format, not an income one. It
is worth doing because it costs you nothing but time and it compounds.
