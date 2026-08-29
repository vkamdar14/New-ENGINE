# DrDonut clip package

14 moments mined from 3 DrDonut VODs by the clip engine.

Each moment was found by mining viewer comments for timestamps. YouTube's
`mostReplayed` heatmap is not in the public API, so this uses the closest
public proxy: viewers timestamp what they want to re-watch. Every clip below
was marked independently by the number of distinct people shown.

## What is in here

- `manifest.json` - every clip with timing, crowd evidence and viewer quotes
- `clipNN_*.ass` - burned-in caption styling for each clip (1080x1920)
- `render_all.sh` - renders every clip once you have the source VODs

## To render

```bash
# download each source VOD to <VIDEO_ID>.mp4 in this directory, then:
./render_all.sh
```

Needs `ffmpeg`. The container these were mined in has none, which is why you
get specs and commands rather than finished mp4s.

## Read this before cutting

**Every clip is marked BACK TO EDIT, and that is correct.** The publish gate
requires captions covering at least 40% of the clip, and these carry a single
`[PAYOFF]` placeholder cue. Real captions need a transcript, and caption
download requires OAuth on the owning channel - which I do not have for
DrDonut. Run the source through any transcription tool, feed it back with
`--transcript`, and the gate passes.

**Windows open ~14s before the marked second.** Viewers timestamp the payoff,
not the setup, so a clip starting on the marked second opens on a punchline
with no context. The engine already applied this offset.

**Some marks are navigational, not reactions.** Comments like "beginning of
the mod at 1:29:36" are viewers indexing a long VOD rather than flagging a
great moment. They cluster just as tightly, so the engine cannot yet tell
them apart. The quotes are included per clip so you can judge - the top few
below are genuine beats, the 4-person ones are more likely index marks.

## The moments, strongest first

### 1. `247:52` - 9 people, 218.2x sharpness (medium confidence)

- **Source**: https://youtu.be/bULfieZael0?t=14872
- **Window**: 14872s -> 14888s (16s)
- **Gate**: BACK TO EDIT - pacing below floor (coverage or dead air)
- **Auto-revisions applied**: trimmed 13.7s of dead air off the tail
- **What viewers said**:
  - "4:08:07 donut base gets raided if anyone is wondering"
  - "4:08:00 truezay appears in donuts base"
  - "4:08:08 Truezay destroys Dr Donuts Base"

### 2. `155:45` - 10 people, 78.9x sharpness (medium confidence)

- **Source**: https://youtu.be/9HM5HJRqqtQ?t=9345
- **Window**: 9345s -> 9361s (16s)
- **Gate**: BACK TO EDIT - pacing below floor (coverage or dead air)
- **Auto-revisions applied**: trimmed 13.7s of dead air off the tail
- **What viewers said**:
  - "2:36:00 if you pause at right time at 0.25 speed you see a player"
  - "At 2:36:00 if you pause it right, you can see whlspy"
  - "The verity guys username is Whlspy for anyone wondering itf u pause when he dissapears on 2:35:50"

### 3. `168:48` - 9 people, 68.6x sharpness (medium confidence)

- **Source**: https://youtu.be/9HM5HJRqqtQ?t=10128
- **Window**: 10128s -> 10144s (16s)
- **Gate**: BACK TO EDIT - pacing below floor (coverage or dead air)
- **Auto-revisions applied**: trimmed 13.7s of dead air off the tail
- **What viewers said**:
  - "2:49:03 NOOO VERITY MISSES MOB 🥹💔

Someone tell him Mob is gonna make him a marketable plush

Can yall stop bl"
  - "2:49:03 OMGG BRO, HE MISSES MOB😢
Oh wow TX for 100 likes!"
  - "2:48:40 ouh did bro got flashback on the Lab👀"

### 4. `110:20` - 4 people, 46.3x sharpness (low confidence)

- **Source**: https://youtu.be/9HM5HJRqqtQ?t=6620
- **Window**: 6620s -> 6636s (16s)
- **Gate**: BACK TO EDIT - crowd too thin (4 people, 46.3x); pacing below floor (coverage or dead air)
- **Auto-revisions applied**: trimmed 13.7s of dead air off the tail
- **What viewers said**:
  - "- 1:50:33 Dr Donutt Find Verity
- 3:03:48 Dr Donutt Last Part of Verity Mod"
  - "1:50:33  Also this is Exact moment Dr Donutt find Verity inside the box."
  - "verity spawns at 1:50:36 for the people wondering"

### 5. `89:20` - 4 people, 39.1x sharpness (low confidence)

- **Source**: https://youtu.be/9HM5HJRqqtQ?t=5360
- **Window**: 5360s -> 5376s (16s)
- **Gate**: BACK TO EDIT - crowd too thin (4 people, 39.1x); pacing below floor (coverage or dead air)
- **Auto-revisions applied**: trimmed 13.7s of dead air off the tail
- **What viewers said**:
  - "he started playing the verity mod at 1:29:36 for anyone wondering!"
  - "1:29:30 homer let the barts out is crazy"
  - "Beginning of mod (1:29:36)          End of mod 3:03:45"

### 6. `183:32` - 4 people, 34.0x sharpness (low confidence)

- **Source**: https://youtu.be/9HM5HJRqqtQ?t=11012
- **Window**: 11012s -> 11028s (16s)
- **Gate**: BACK TO EDIT - crowd too thin (4 people, 34.0x); pacing below floor (coverage or dead air)
- **Auto-revisions applied**: trimmed 13.7s of dead air off the tail
- **What viewers said**:
  - "- 1:50:33 Dr Donutt Find Verity
- 3:03:48 Dr Donutt Last Part of Verity Mod"
  - "3:03:40 verity said donuts name “Nothing will work Nate"
  - "Beginning of mod (1:29:36)          End of mod 3:03:45"

### 7. `145:25` - 4 people, 33.0x sharpness (low confidence)

- **Source**: https://youtu.be/9HM5HJRqqtQ?t=8725
- **Window**: 8725s -> 8741s (16s)
- **Gate**: BACK TO EDIT - crowd too thin (4 people, 33.0x); pacing below floor (coverage or dead air)
- **Auto-revisions applied**: trimmed 13.7s of dead air off the tail
- **What viewers said**:
  - "2:25:37 LMAO IM CRINE WISPY WAS NOT HAVING IT😭"
  - "2:25:44 donut with the survivial instinct of a donut sprinkle"
  - "2:25:34 bro sounds like my mom when she tells we are going somewhere I don't wanna go"

### 8. `101:38` - 4 people, 26.9x sharpness (low confidence)

- **Source**: https://youtu.be/9HM5HJRqqtQ?t=6098
- **Window**: 6098s -> 6114s (16s)
- **Gate**: BACK TO EDIT - crowd too thin (4 people, 26.9x); pacing below floor (coverage or dead air)
- **Auto-revisions applied**: trimmed 13.7s of dead air off the tail
- **What viewers said**:
  - "donut is actually tweaking out 1:41:59"
  - "1:41:52 valid crash out blue donut guy"
  - "1:41:44 crashout"

### 9. `4:25` - 10 people, 26.0x sharpness (medium confidence)

- **Source**: https://youtu.be/oafS9P4zFZo?t=265
- **Window**: 265s -> 281s (16s)
- **Gate**: BACK TO EDIT - pacing below floor (coverage or dead air)
- **Auto-revisions applied**: trimmed 13.7s of dead air off the tail
- **What viewers said**:
  - "4:39 “THE FIH”🗣️🔥🔥🔥🔥🔥🔥"
  - "4:38 "THE FIH!""
  - "4:40  THE FISH"

### 10. `10:57` - 5 people, 18.5x sharpness (low confidence)

- **Source**: https://youtu.be/oafS9P4zFZo?t=657
- **Window**: 657s -> 673s (16s)
- **Gate**: BACK TO EDIT - pacing below floor (coverage or dead air)
- **Auto-revisions applied**: trimmed 13.7s of dead air off the tail
- **What viewers said**:
  - "11:10 Video recorded so long ago bro didnt even have me"
  - "11:17 bro WTF chill😭✌️"
  - "11:11 he is speaking french"

### 11. `10:22` - 7 people, 17.8x sharpness (medium confidence)

- **Source**: https://youtu.be/oafS9P4zFZo?t=622
- **Window**: 622s -> 638s (16s)
- **Gate**: BACK TO EDIT - pacing below floor (coverage or dead air)
- **Auto-revisions applied**: trimmed 13.7s of dead air off the tail
- **What viewers said**:
  - "10:36 fym get out of my face"
  - "10:35 "Can I get a screenshot?" "Get out of my face bro"  your editor did you so dirty
12:52
20:14
28:38
32:56"
  - "10:35 dang bro its just a screenshot"

### 12. `118:52` - 5 people, 16.5x sharpness (low confidence)

- **Source**: https://youtu.be/9HM5HJRqqtQ?t=7132
- **Window**: 7132s -> 7148s (16s)
- **Gate**: BACK TO EDIT - pacing below floor (coverage or dead air)
- **Auto-revisions applied**: trimmed 13.7s of dead air off the tail
- **What viewers said**:
  - "1:59:06 HOLY HELL THATS APOCALYPSE BIRDS SCREAM?? PROJECT MOON REFERENCEEE"
  - "1:59:05  my heart stopped for a sec 😭😭 i was watching at 1:43am"
  - "1:59:05 bro you can RECOGNISE mob bro LMAO"

### 13. `0:00` - 8 people, 16.2x sharpness (medium confidence)

- **Source**: https://youtu.be/oafS9P4zFZo?t=0
- **Window**: 0s -> 8s (8s)
- **Gate**: BACK TO EDIT - pacing below floor (coverage or dead air)
- **Auto-revisions applied**: trimmed 11.0s of dead air off the tail; dropped the hook - captions already occupy the opening
- **What viewers said**:
  - "0:10 I finally can see the clip!"
  - "0:00 THE JAWLINE BRO LIKE DAYYUM."
  - "0:00 holy psl"

### 14. `17:56` - 4 people, 4.3x sharpness (low confidence)

- **Source**: https://youtu.be/oafS9P4zFZo?t=1076
- **Window**: 1076s -> 1092s (16s)
- **Gate**: BACK TO EDIT - crowd too thin (4 people, 4.3x); pacing below floor (coverage or dead air)
- **Auto-revisions applied**: trimmed 13.7s of dead air off the tail
- **What viewers said**:
  - "18:10 Death by glamour reference"
  - "18:07 why does this sounded like death by glamour by mettaton😭"
  - "18:09  w music"
