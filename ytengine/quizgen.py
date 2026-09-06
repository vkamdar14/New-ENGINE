"""Quiz-Short generator: a complete video from a list, no footage required.

The one format in the football harvest that needs nothing you have to own -
no match footage, no game, no camera, no player photos. Text, a timer, and a
reveal. Which means it can be generated end to end.

Measured in the harvest: "Can you name..." quizzes ran 7.78x the corpus median
(267,751 vs 34,425). That number comes from a 1.8M-subscriber publisher
though, so the honest read is in `NICHE_QUIZ.md` - small channels running this
format have a median of ~1,100 views, with the best of them hitting 271k from
672 subs. The format has a very high ceiling and a very low floor.

The structure that works, from the videos that hit:

    0.0s   question on screen, nothing else
    1.5s   countdown bar starts
    ~8s    answers reveal one at a time, ranked
    end    hold the full list

The countdown is the whole mechanism. It converts a scroll into a commitment -
the viewer stays because they are trying to beat it, and that is the retention
the format is built on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .editor import HEIGHT, WIDTH, _escape, _ts

# Layout. The question sits high so the reveal list has room beneath it, and
# everything clears the bottom 18% the player's UI occupies.
QUESTION_Y = 0.14
LIST_TOP_Y = 0.34
ROW_STEP = 0.075
TIMER_Y = 0.26

THINK_SECONDS = 6.0      # time to guess before the first reveal
REVEAL_STEP = 1.1        # gap between answers
HOLD_SECONDS = 2.0       # hold the completed list before the loop


class RankingError(ValueError):
    """A ranked list whose numbers contradict its order."""


def check_ranking(answers: Sequence[str], subtitles: Sequence[str]) -> list[str]:
    """Verify a ranked list actually descends.

    A quiz that shows "4. Cenk Tosun 20 / 5. Küçükandonyadis 21" is wrong on
    its face, and a ranking error is the single most punished mistake in this
    format - the comments exist to correct you and the correction becomes the
    video. Cheap to check, so it is checked every time rather than eyeballed.
    """
    import re as _re
    vals = []
    for s_ in subtitles:
        m = _re.match(r"\s*(\d+)", s_ or "")
        vals.append(int(m.group(1)) if m else None)
    problems = []
    for i in range(len(vals) - 1):
        a, b = vals[i], vals[i + 1]
        if a is None or b is None:
            continue
        if a < b:
            problems.append(
                f"#{i+1} {answers[i]} ({a}) ranks above #{i+2} {answers[i+1]} ({b})")
    return problems


@dataclass
class Quiz:
    question: str
    answers: Sequence[str]          # already in reveal order
    subtitles: Sequence[str] = ()   # optional right-hand column, e.g. "5 goals"
    accent: str = "&H0000E5FF&"     # ASS BGR - amber

    strict: bool = True             # refuse to build a mis-ranked list

    def __post_init__(self):
        if self.strict and self.subtitles:
            problems = check_ranking(self.answers, self.subtitles)
            if problems:
                raise RankingError("; ".join(problems))

    @property
    def duration_s(self) -> float:
        return THINK_SECONDS + len(self.answers) * REVEAL_STEP + HOLD_SECONDS


def build_ass(q: Quiz, font: str = "DejaVu Sans") -> str:
    """A complete ASS file: question, countdown, staged reveal."""
    n = len(q.answers)
    total = q.duration_s
    head = [
        "[Script Info]", "ScriptType: v4.00+",
        f"PlayResX: {WIDTH}", f"PlayResY: {HEIGHT}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
        "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        # Alignment 8 = top-centre for the question, 7 = top-left for rows.
        f"Style: Q,{font},74,&H00FFFFFF&,&H00FFFFFF&,&H00000000&,&H90000000&,-1,0,0,0,"
        f"100,100,0,0,1,6,3,8,70,70,{int(HEIGHT * QUESTION_Y)},1",
        f"Style: Row,{font},60,&H00FFFFFF&,&H00FFFFFF&,&H00000000&,&H90000000&,-1,0,0,0,"
        f"100,100,0,0,1,5,2,7,110,110,0,1",
        f"Style: Num,{font},60,{q.accent},{q.accent},&H00000000&,&H90000000&,-1,0,0,0,"
        f"100,100,0,0,1,5,2,7,70,70,0,1",
        f"Style: Timer,{font},52,{q.accent},{q.accent},&H00000000&,&H90000000&,-1,0,0,0,"
        f"100,100,0,0,1,4,2,8,70,70,{int(HEIGHT * TIMER_Y)},1",
        "", "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    ev = [f"Dialogue: 0,{_ts(0)},{_ts(total)},Q,,0,0,0,,{_escape(q.question)}"]

    # Countdown. Redrawn each second rather than animated: a number ticking
    # down is legible at a glance in a feed, a smooth bar is not.
    for i in range(int(THINK_SECONDS), 0, -1):
        t0 = THINK_SECONDS - i
        ev.append(f"Dialogue: 0,{_ts(t0)},{_ts(t0 + 1)},Timer,,0,0,0,,"
                  f"{'●' * i}{'○' * (int(THINK_SECONDS) - i)}   {i}")

    for idx, ans in enumerate(q.answers):
        start = THINK_SECONDS + idx * REVEAL_STEP
        y = int(HEIGHT * (LIST_TOP_Y + idx * ROW_STEP))
        sub = q.subtitles[idx] if idx < len(q.subtitles) else ""
        text = f"{_escape(ans)}" + (f"   {_escape(sub)}" if sub else "")
        # \pos overrides the style margin so each row lands on its own line.
        ev.append(f"Dialogue: 0,{_ts(start)},{_ts(total)},Num,,0,0,0,,"
                  f"{{\\pos(90,{y})}}{idx + 1}.")
        ev.append(f"Dialogue: 0,{_ts(start)},{_ts(total)},Row,,0,0,0,,"
                  f"{{\\pos(170,{y})}}{text}")
    return "\n".join(head + ev) + "\n"


def ffmpeg_command(q: Quiz, ass_path: str, out_path: str,
                   bg: str = "0x0b1a12", panel: str = "0x14352a") -> list[str]:
    """Render command. Pitch-green background, no external assets."""
    d = f"{q.duration_s:.2f}"
    return [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={bg}:s={WIDTH}x{HEIGHT}:d={d}:r=30",
        "-f", "lavfi", "-i", f"color=c={panel}:s={WIDTH}x{int(HEIGHT*0.60)}:d={d}:r=30",
        "-filter_complex",
        f"[1:v]format=rgb24[p];[0:v][p]overlay=0:{int(HEIGHT*0.28)}[bg];"
        f"[bg]subtitles=filename={ass_path}:fontsdir=/usr/share/fonts[v]",
        "-map", "[v]", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path,
    ]
