# 16 football quiz Shorts — generated and scheduled

All 16 rendered, 1080x1920, 13.5s each, ~150KB. No footage, no photos, no
game — every frame is generated from a list.

## Files

- `out/01..16_*.mp4` — the videos, in posting order
- `schedule.csv` — publish times, titles, tags, per-video verification risk
- `manifest.json` — the full answer list for each video

## The schedule is built around who each video is for

These are country editions, which was the measured differentiator: the 672-sub
channel that hit 271k ran "TURKEY Edition" and "ENGLAND Edition" while generic
quizzes sat at a ~1,100-view median.

So each video publishes into **its own audience's evening**, not the
uploader's. A Turkey edition posted at 19:00 UK time lands at 22:00 in Turkey,
past the peak it was made for.

16 posts over 9 days, 2 per day, 12.4/week — comparable to the model channels
(NextGenFootball2 runs 9.2/week, the FC26 model 10.6).

## Verify the facts before posting

**This is the one thing that matters.** My knowledge stops in May 2026 and a
World Cup ran that June-July, so any all-time list may be a tournament stale.
`schedule.csv` carries a `verify` column per video:

- **HIGH** — an active player or a 2026 competition is involved. Check it.
  Brazil WC scorers, Argentina caps, France, Portugal, Netherlands, Liverpool,
  Champions League, Ballon d'Or.
- **medium** — Germany, Turkey.
- **low** — everyone on the list has retired. Italy, Spain, Real Madrid,
  Man United, Barcelona.

Getting a stat wrong here is worse than in any other format: the comments exist
to correct you, and the correction becomes the video's story.

## A ranking bug worth knowing about

Five of the first sixteen shipped with lists whose numbers contradicted their
own order — Turkey showed "4. Cenk Tosun 20 / 5. Küçükandonyadis 21". All five
are fixed, and `quizgen.Quiz` now refuses to build a mis-ranked list at all,
with the rule under test. Regenerating from `questions.py` cannot reproduce it.
