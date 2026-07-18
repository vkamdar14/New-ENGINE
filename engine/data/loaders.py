"""Real-data adapters (stooq / Yahoo / GDELT news).

STATUS IN THIS SANDBOX: every one of these hosts is blocked by the egress
proxy (verified: CONNECT 403 for stooq.com, query1.finance.yahoo.com,
api.gdeltproject.org).  The adapters are shipped ready-to-run so the
identical pipeline can be re-run on real data from any network-enabled
environment:  `python -m engine.run_all --source real --tickers-file sp500.txt`

The output contract matches engine.data.synthetic.simulate(): a MultiIndex
(date, ticker) panel with open/high/low/close/volume plus, when a news
source is configured, `catalyst` (0/1) and `news_sign` (-1/0/+1).
"""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd


class DataSourceBlocked(RuntimeError):
    pass


def fetch_stooq_daily(ticker: str, start: str, end: str,
                      session=None) -> pd.DataFrame:
    """Daily OHLCV from stooq.com CSV endpoint (free, no key)."""
    import requests

    url = (f"https://stooq.com/q/d/l/?s={ticker.lower()}.us"
           f"&d1={start.replace('-', '')}&d2={end.replace('-', '')}&i=d")
    try:
        r = (session or requests).get(url, timeout=30)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise DataSourceBlocked(
            f"stooq fetch failed for {ticker}: {e}. "
            "This sandbox blocks market-data hosts; run from a "
            "network-enabled environment.") from e
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"])
    df = df.rename(columns=str.lower).set_index("date")
    return df[["open", "high", "low", "close", "volume"]]


def fetch_yahoo_daily(tickers: list[str], start: str, end: str,
                      auto_adjust: bool = False) -> pd.DataFrame:
    """Daily OHLCV via yfinance for a list of tickers -> (date,ticker) panel.

    NOTE: auto_adjust defaults to False because level-dependent detectors
    (round numbers, gap sizes vs actual traded prices, $-grids) need TRADED
    prices; back-adjusted series distort historical levels.  Handle splits/
    dividends explicitly downstream if you enable it.
    """
    try:
        import yfinance as yf
        raw = yf.download(tickers, start=start, end=end, group_by="ticker",
                          auto_adjust=auto_adjust, threads=True,
                          progress=False)
    except Exception as e:  # noqa: BLE001
        raise DataSourceBlocked(f"yahoo fetch failed: {e}") from e
    frames = {}
    for t in tickers:
        try:
            d = raw[t].rename(columns=str.lower)
            frames[t] = d[["open", "high", "low", "close", "volume"]].dropna()
        except KeyError:
            continue
    panel = pd.concat(frames, names=["ticker", "date"]).swaplevel().sort_index()
    return panel


def fetch_gdelt_news_volume(query: str, start: str, end: str,
                            pause: float = 0.3) -> pd.Series:
    """Daily article-volume timeline for a company from GDELT DOC 2.0
    (free, no key).  Used to flag catalyst days: volume z-score > 2."""
    import requests

    url = ("https://api.gdeltproject.org/api/v2/doc/doc?"
           f"query={requests.utils.quote(query)}&mode=timelinevolraw"
           f"&format=json&startdatetime={start.replace('-', '')}000000"
           f"&enddatetime={end.replace('-', '')}235959")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        raise DataSourceBlocked(f"gdelt fetch failed: {e}") from e
    tl = data["timeline"][0]["data"]
    s = pd.Series({pd.Timestamp(x["date"][:8]): x["value"] for x in tl})
    time.sleep(pause)
    return s


def catalyst_flags_from_news_volume(vol: pd.Series, z: float = 2.0,
                                    lookback: int = 63) -> pd.Series:
    """catalyst=1 on days where article volume z-score vs trailing window > z."""
    mu = vol.rolling(lookback, min_periods=20).mean().shift(1)
    sd = vol.rolling(lookback, min_periods=20).std().shift(1)
    return ((vol - mu) / sd.replace(0, np.nan) > z).astype(int)


def load_real_panel(tickers: list[str], start: str, end: str,
                    with_news: bool = True) -> dict:
    """Assemble a real panel in the simulator's output format."""
    panel = fetch_yahoo_daily(tickers, start, end)
    panel["catalyst"] = 0
    panel["news_sign"] = 0
    panel["quality"] = np.nan       # plug in your fundamentals source here
    # ORB fields require intraday data (e.g. Polygon/Alpaca); left NaN on real
    panel["or30_high"] = np.nan
    panel["or30_low"] = np.nan
    if with_news:
        for t in tickers:
            try:
                nv = fetch_gdelt_news_volume(f'"{t}" stock', start, end)
            except DataSourceBlocked:
                break
            flags = catalyst_flags_from_news_volume(nv)
            idx = panel.loc[(slice(None), t), :].index
            aligned = flags.reindex(idx.get_level_values("date")).fillna(0).values
            panel.loc[(slice(None), t), "catalyst"] = aligned.astype(int)
    dates = panel.index.get_level_values("date").unique().sort_values()
    return {"panel": panel.sort_index(), "truth": None, "dates": dates,
            "tickers": tickers, "sector_of": {}}
