from . import (anchors, base, breakouts, candles, fundamentals, gaps, levels,
               lmw, novel, trend, volume)

ALL_DETECTORS = {
    "lmw": lmw.detect,
    "candles": candles.detect,
    "levels": levels.detect,
    "anchors": anchors.detect,
    "breakouts": breakouts.detect,
    "gaps": gaps.detect,
    "volume": volume.detect,
    "trend": trend.detect,
    "fundamentals": fundamentals.detect,
    "novel": novel.detect,
}
