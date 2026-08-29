"""Edit-spec engine: turn a clip window into a render-ready Short.

Produces two artefacts from a clip: an ASS subtitle file carrying the caption
styling, and the exact ffmpeg command that burns it into a 9:16 cut. Rendering
happens wherever ffmpeg is installed; nothing here needs the binary present to
produce the spec.

ASS rather than ffmpeg's `drawtext` because the format that actually performs
needs per-word highlight timing, an outline *and* a shadow, and precise
positioning. drawtext can do one word at a time with none of the styling;
ASS does all of it in one filter pass.

The layout constants below are the part that matters, and they come from how
the Shorts player is built rather than from taste:

  - The bottom ~18% of the frame is covered by title, channel, and the
    description sheet. The right ~14% is the like/comment/share rail. Captions
    placed in either are simply not read.
  - So captions sit at ~62% of frame height: clear of the bottom chrome, low
    enough to leave the speaker's face unobstructed.
  - Most viewing is sound-off, which makes burned-in captions the difference
    between a clip that works and one that does not. This is not decoration.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Optional, Sequence

# Vertical canvas. 1080x1920 is the native Shorts resolution.
WIDTH, HEIGHT = 1080, 1920

# Fractions of frame height that the player's own UI occupies.
BOTTOM_CHROME = 0.18
RIGHT_RAIL = 0.14
CAPTION_Y = 0.62          # where captions sit, as a fraction of height
HOOK_Y = 0.18             # hook text rides high, above the face


@dataclass
class CaptionStyle:
    """A caption look, in the terms ASS needs."""

    name: str
    font: str
    size: int
    primary: str          # &HBBGGRR& - ASS is BGR, not RGB
    highlight: str
    outline: str
    outline_w: int
    shadow: int
    bold: int = -1        # ASS uses -1 for true
    uppercase: bool = True
    words_per_cue: int = 3

    def ass_style(self) -> str:
        # Alignment 2 = bottom-centre; MarginV then lifts it off the chrome.
        margin_v = int(HEIGHT * (1 - CAPTION_Y))
        return (
            f"Style: Caption,{self.font},{self.size},{self.primary},{self.highlight},"
            f"{self.outline},&H80000000&,{self.bold},0,0,0,100,100,0,0,1,"
            f"{self.outline_w},{self.shadow},2,60,60,{margin_v},1"
        )


# Presets modelled on the formats that dominate Shorts feeds. Each is a
# coherent look rather than a font choice - size, stroke weight and cue length
# have to move together or the result reads as neither one style nor another.
STYLES = {
    "punch": CaptionStyle(
        name="punch", font="Impact", size=96,
        primary="&H00FFFFFF&", highlight="&H0000FFFF&",
        outline="&H00000000&", outline_w=7, shadow=3, words_per_cue=3),
    "clean": CaptionStyle(
        name="clean", font="Montserrat ExtraBold", size=82,
        primary="&H00FFFFFF&", highlight="&H0000E5FF&",
        outline="&H00000000&", outline_w=5, shadow=2, words_per_cue=4),
    "karaoke": CaptionStyle(
        name="karaoke", font="Arial Black", size=88,
        primary="&H00FFFFFF&", highlight="&H0000FF00&",
        outline="&H00000000&", outline_w=6, shadow=2, words_per_cue=1),
    "docu": CaptionStyle(
        name="docu", font="Helvetica", size=68,
        primary="&H00FFFFFF&", highlight="&H00FFFFFF&",
        outline="&H00000000&", outline_w=3, shadow=1,
        uppercase=False, words_per_cue=6),
}


@dataclass
class Cue:
    start_s: float
    end_s: float
    text: str


@dataclass
class EditSpec:
    source: str
    start_s: float
    end_s: float
    style: CaptionStyle
    cues: list[Cue] = field(default_factory=list)
    hook: str = ""
    crop_mode: str = "center"   # center | left | right

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _ts(seconds: float) -> str:
    """ASS timestamp: H:MM:SS.cc (centiseconds, single-digit hour)."""
    seconds = max(seconds, 0.0)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def cues_from_words(words: Sequence[tuple[float, float, str]],
                    style: CaptionStyle, offset_s: float = 0.0) -> list[Cue]:
    """Group timed words into cues of `words_per_cue`.

    Cue timing runs from the first word's start to the last word's end, so the
    caption is on screen exactly while those words are spoken. Times are
    rebased to the clip start - a cue at 3721s into the stream must appear at
    0.5s into a clip that began at 3720.5s.
    """
    cues: list[Cue] = []
    n = max(style.words_per_cue, 1)
    for i in range(0, len(words), n):
        group = words[i:i + n]
        text = " ".join(w for _, _, w in group)
        if style.uppercase:
            text = text.upper()
        cues.append(Cue(start_s=max(group[0][0] - offset_s, 0.0),
                        end_s=max(group[-1][1] - offset_s, 0.0),
                        text=text))
    return cues


def build_ass(spec: EditSpec) -> str:
    """A complete ASS subtitle file for this clip."""
    head = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {WIDTH}",
        f"PlayResY: {HEIGHT}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
        "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        spec.style.ass_style(),
    ]
    if spec.hook:
        # Alignment 8 = top-centre, so the hook cannot collide with captions.
        head.append(
            f"Style: Hook,{spec.style.font},{int(spec.style.size * 1.05)},"
            f"&H00FFFFFF&,&H0000FFFF&,&H00000000&,&H90000000&,-1,0,0,0,100,100,0,0,1,"
            f"{spec.style.outline_w + 1},4,8,60,60,{int(HEIGHT * HOOK_Y)},1"
        )
    body = ["", "[Events]",
            "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text"]
    if spec.hook:
        text = spec.hook.upper() if spec.style.uppercase else spec.hook
        # The hook holds for the first 2.5s - long enough to read, short enough
        # to clear before the payoff lands.
        body.append(f"Dialogue: 0,{_ts(0)},{_ts(min(2.5, spec.duration_s))},Hook,,0,0,0,,"
                    f"{_escape(text)}")
    for c in spec.cues:
        if c.end_s <= 0 or c.start_s >= spec.duration_s:
            continue
        body.append(
            f"Dialogue: 0,{_ts(c.start_s)},{_ts(min(c.end_s, spec.duration_s))},"
            f"Caption,,0,0,0,,{_escape(c.text)}"
        )
    return "\n".join(head + body) + "\n"


def _escape(text: str) -> str:
    """ASS treats braces as override blocks and newlines as \\N."""
    return text.replace("{", "(").replace("}", ")").replace("\n", "\\N").strip()


def ffmpeg_command(spec: EditSpec, ass_path: str, out_path: str) -> list[str]:
    """The exact render command.

    `-ss` before `-i` seeks fast by keyframe; the trim is then exact because
    the stream is re-encoded anyway for the subtitle burn. Scale-then-crop
    fills 9:16 without letterboxing, which matters because black bars read as
    low effort in a feed and cost the first-second retention that decides
    whether the clip travels.
    """
    xpos = {"center": "(in_w-out_w)/2", "left": "0", "right": "in_w-out_w"}[spec.crop_mode]
    vf = (
        f"scale={WIDTH}:-2:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT}:{xpos}:(in_h-out_h)/2,"
        f"subtitles='{ass_path}'"
    )
    return [
        "ffmpeg", "-y",
        "-ss", f"{spec.start_s:.2f}",
        "-i", spec.source,
        "-t", f"{spec.duration_s:.2f}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        out_path,
    ]


def render_plan(spec: EditSpec, ass_path: str, out_path: str) -> str:
    o = ["", "EDIT SPEC", "=========", ""]
    o.append(f"  source     {spec.source}")
    o.append(f"  window     {spec.start_s:.1f}s -> {spec.end_s:.1f}s  ({spec.duration_s:.1f}s)")
    o.append(f"  style      {spec.style.name}  ({spec.style.font} {spec.style.size}px, "
             f"{spec.style.words_per_cue} words/cue)")
    o.append(f"  canvas     {WIDTH}x{HEIGHT}, crop {spec.crop_mode}")
    o.append(f"  captions   {len(spec.cues)} cues at {CAPTION_Y:.0%} height")
    if spec.hook:
        o.append(f"  hook       \"{spec.hook}\" (first 2.5s, top {HOOK_Y:.0%})")
    o.append(f"\n  wrote {ass_path}")
    o.append("\n  render with:")
    # shlex.quote, not hand-rolled quoting: the filter string contains its own
    # single quotes around the subtitles path, and naive wrapping produces a
    # command that looks right and dies in the shell.
    o.append("    " + shlex.join(ffmpeg_command(spec, ass_path, out_path)))
    o.append("\n  Layout avoids the player's own UI: the bottom "
             f"{BOTTOM_CHROME:.0%} carries title and")
    o.append(f"  description, the right {RIGHT_RAIL:.0%} is the like/comment rail. Captions in")
    o.append("  either are not read. Most Shorts viewing is sound-off, so burned-in")
    o.append("  captions are load-bearing rather than decoration.")
    return "\n".join(o)
