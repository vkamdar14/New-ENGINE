# The Chart-Based Edge Catalog

Every edge below is detected **algorithmically** — formal, parameterized rules,
no discretionary eyeballing. For each family: the formal method, the original
evidence, what happened to it after publication (decay), and an honest verdict.
Where the honest verdict is "this does not survive costs," the catalog says so.

**How to read the empirical columns.** This sandbox's network policy blocks
every market-data host (verified: stooq, Yahoo Finance, GDELT, HuggingFace all
return 403 at the egress proxy), so the engine's *empirical* numbers come from
a 500-stock, 6¼-year synthetic market whose dynamics are calibrated on the
**real S&P 500 daily series 1999–2018** (GARCH(1,1)-t market factor, overnight
/intraday variance split, volume–|return| elasticity — fitted, not invented;
the series ships inside the `arch` package) and whose embedded effects are set
to **published magnitudes** (§ Ground truth). Synthetic results validate the
*machinery* — that the detectors find what is truly there and reject what
isn't. Verdicts about the real world cite the literature. `engine/data/
loaders.py` re-runs the identical pipeline on real data from any
network-enabled environment.

**Cost model** (per side): 3 bp half-spread + 4 bp impact + 0.5 bp
commission = 7.5 bp; 15 bp round trip. Liquid US large/mid caps, modest size.

---

## Ground truth embedded in the simulator

| Effect | Embedded size | Real-world calibration source |
|---|---|---|
| News-jump post-drift (PEAD) | 10% of jump over 10 days | Bernard–Thomas 1989; Chan–Jegadeesh–Lakonishok 1996 (≈ 1–1.5%/qtr drift on extreme surprise deciles) |
| No-news overnight gap fade | −12% of gap next session | gap-fill studies (Plastun et al. 2019–21); overnight/intraday return split (Lou–Polk–Skouras 2019) |
| 52-week-high anchoring drift | +1.8 bp/day within 2% of high | George–Hwang 2004 (≈ 0.45%/mo winner leg) |
| Slow per-stock alpha (OU, 120d half-life) | 6 bp/day dispersion | hosts momentum: Jegadeesh–Titman 1993 (≈ 1.2%/mo decile spread after estimation noise) |
| Quality/moat premium | ≈ 7 bp/day × decile-rank spread ⇒ ~1.8%/yr top-vs-bottom | Novy-Marx 2013; Asness–Frazzini–Pedersen QMJ 2019 (2–4%/yr) |
| News × volume interaction | event days carry 2–8× volume and all the PEAD drift | Gervais–Kaniel–Mingelgrin 2001; PEAD-on-volume evidence |
| Candlestick shapes per se | **zero** (placebo) | Marshall–Young–Rose 2006: no predictability in US equities |
| Round-number levels | **zero** (placebo) | weak/no robust evidence |
| LMW geometries per se | **zero** (any profit must ride embedded trend/news) | LMW 2000 find *informativeness*, not net profit |

The placebos are the honesty control: a pipeline that "finds" candlestick or
round-number alpha in this market is broken. Ours doesn't (see
`results/ground_truth_recovery.csv`).

---

## 1. Classical patterns — Lo–Mamaysky–Wang kernel regression

**Method (formal).** For each stock and day, the trailing 38-day price window
is smoothed by Nadaraya–Watson kernel regression

$$\hat m(\tau)=\frac{\sum_{s=1}^{L}K_h(\tau-s)\,P_s}{\sum_{s=1}^{L}K_h(\tau-s)},\qquad K_h(x)=e^{-x^2/2h^2}$$

with h = 2.5 days (inside the range LMW's cross-validation × 0.3 selects).
Local extrema E1…En of the smoothed curve (alternation enforced) feed strict
inequality systems — e.g. head-and-shoulders requires E1 max, E3 > E1,
E3 > E5, E1 ≈ E5 within 1.5%, E2 ≈ E4 within 1.5%; a pattern *completes* when
its last extremum sits exactly 3 days before the window end. All ten LMW
geometries are implemented (HS, IHS, broadening/triangle/rectangle tops and
bottoms, double tops/bottoms with the ≥22-day peak separation), plus formal
cup-and-handle (quadratic U-fit, R² > 0.5, depth 12–35%, handle in upper half
retracing < ⅓ of cup) and flag rules (≥12% pole in 10 days, ≤40%-of-pole
consolidation, break in pole direction).

**Original evidence.** Lo, Mamaysky & Wang (JF 2000), NYSE/AMEX + Nasdaq
1962–1996: conditioning on patterns *shifts the distribution* of subsequent
returns (statistically informative), especially with volume confirmation. They
explicitly do **not** claim trading profits.

**Decay/replication.** Post-2000 replications on US large caps find the
information content mostly gone; Nasdaq small caps retained some through the
2000s. Dawson & Steeley (2003, UK): distributions differ, means don't.
Marshall et al.: no net profits after costs across major markets.

**Honest verdict.** As *stand-alone trade triggers* the classical geometries
**fail after costs** — the mean conditional return is a handful of basis
points against a 15 bp round trip. Their residual value is as *context
features* inside a fusion model (they mark trend/consolidation states). Our
engine reproduces exactly this: LMW signals carry no intrinsic drift by
construction, and the event studies + fusion importances show they only earn
when they proxy the embedded trend/news effects.

## 2. Candlestick patterns

**Method.** 16 patterns as strict OHLC inequality systems (hammer, shooting
star, engulfings, haramis, piercing/dark-cloud, morning/evening star, three
soldiers/crows, marubozu, doji), thresholds scaled by ATR(20).

**Original evidence.** Practitioner lore (Nison 1991). Academic tests:
Caginalp & Laurent (1998) found short-horizon predictability in S&P 500
stocks 1992–96.

**Decay.** Marshall, Young & Rose (JBF 2006), DJIA 1992–2002: **no
profitability** for any of 28 patterns after realistic costs; Horton (2009),
Fock et al. (2005) concur; intraday and non-US replications overwhelmingly
negative.

**Honest verdict.** **Fails after costs. Placebo-grade.** Our engine embeds
zero candlestick effect and correctly measures ≈ zero (several nominally
"significant" candles in any sample are the expected multiple-testing
false-positive rate across 100+ tested signals — the catalog treats |t| < 3
on 16 correlated tests as noise). Candlesticks survive only as *inputs to the
CNN*, which learns whatever genuine short-horizon structure exists directly
from images.

## 3. Support / resistance levels

**Method.** Four algorithmic level sources: (a) fractal swing highs/lows
(5-bar) with ±0.25·ATR touch bands; (b) round numbers ($5/$10/$25 grids by
price); (c) VWAP anchored at the latest news-catalyst day; (d) SMA 20/50/200
as dynamic levels. Signals: bounce (touch-and-hold ⇒ fade toward level) and
break (close through ±0.25·ATR ⇒ continuation).

**Original evidence.** Osler (2000, FRBNY Economic Policy Review): S/R levels
published by six firms had genuine short-horizon predictive content in FX;
order-book clustering at round numbers (Osler 2003) gives the mechanism.
Brock–Lakonishok–LeBaron (1992) trading-range break rules beat buy-and-hold
pre-cost on the DJIA 1897–1986.

**Decay.** Sullivan–Timmermann–White (1999) data-snooping correction kills
most BLL profitability out-of-sample; post-1986 DJIA performance of the same
rules ≈ zero. FX round-number effects persist microstructurally but are too
small for daily equity bars after costs.

**Honest verdict.** Swing-level *breaks* retain modest value only as part of
breakout/momentum systems (§6). Round-number signals: **no edge** — our
placebo test agrees. Anchored VWAP reclaim/loss is the strongest member of
this family in our engine because it proxies post-news positioning — i.e. its
value comes from the news interaction, not the level itself.

## 4. 52-week-high proximity (anchoring)

**Method.** Proximity ratio close/hi252 (trailing, ex-today); signals: within
2% of the high (long), bottom quintile of the 52-wk range (short), fresh
252-day-high breakout after ≥60 days without one (long).

**Original evidence.** George & Hwang (JF 2004): nearness to the 52-week high
predicts returns better than conventional momentum, 1963–2001 (≈ 0.45%/mo);
behavioral anchoring, not risk. Replicated internationally (Liu et al. 2011),
in industries (Hong, Torous & Valkanov), and in options markets. **This is
real anchoring evidence** — the strongest-documented purely price-chart-based
cross-sectional effect.

**Decay.** Diminished but positive post-publication; Jacobs (2015) and
McLean–Pontiff (2016) put post-publication retention around half the original
effect. Crash risk in momentum-unwind episodes (2009) applies.

**Honest verdict.** **Survives, small.** A few bp/day on the extremes, needs
low-cost execution and weeks-scale holding; useless as a stand-alone daily
in-and-out trade but a genuinely positive conditioning feature. Our engine
embeds it at published size and recovers it (t ≈ 5–6 on the short leg of the
range-position spread).

## 5. Breakout systems

**Opening-range breakout (Crabel 1990).** First-30-minute range break, narrow-
range filter. Original evidence: Crabel's 1990 stats and CME lore; recent
academic support: Zarattini–Aziz (2023-24 SSRN) find persistent ORB profits in
liquid gappy stocks *with* volume/news filters. Decay: raw ORB on everything
is dead; filtered versions contested. Verdict: **only the news/volume-
filtered variant is plausible**; requires intraday data (our daily-bar
approximation synthesizes the first-30-min range and is labeled as such).

**Donchian channels (20/55).** Turtle rules. Original evidence: CTA
track records 1970s–80s; Moskowitz–Ooi–Pedersen (2012) formalize time-series
momentum. Decay: single-stock Donchian breakouts are weak post-1990s in US
equities; the effect lives at the asset-class level. Verdict: **marginal on
single stocks after costs**; volume-confirmed variants better; genuine value
is as the entry leg of a trend system with proper exits.

**Volatility contraction (VCP, Minervini).** Algorithmized: ≥3 successively
tighter 5-day ranges (< 75% each), volume dry-up (RVOL < 0.8), pivot break on
RVOL > 1.5. No academic original evidence — practitioner claims only. Our
test: fires exactly where the simulator's news+trend edge sits (high truth-
recovery), because the shape *selects* pre-news compression; net-positive but
wide-variance. Verdict: **unproven in the literature; behaves as a news/trend
proxy, not an independent edge.**

## 6. Gap taxonomy (and the news interaction — the required study)

**Method.** Every overnight gap classified by direction × size bin
(0.5–1.5%, 1.5–3%, 3–7%, >7%) × catalyst present/absent × RVOL tercile.
Continuation = same-direction intraday and next-day drift; fade = gap-fill.
The news feed is the observable catalyst flag (GDELT article-volume z-score
> 2 on real data; the simulator's event flag here).

**Original evidence.** PEAD: Ball–Brown 1968, Bernard–Thomas 1989/1990 —
news-backed price jumps continue drifting for weeks; among the most
replicated anomalies in finance. Gap-fill: practitioner lore quantified by
Plastun et al. (2019–2021) — no-news gaps close the same week far more often
than chance.

**Decay.** PEAD has shrunk in large caps (Chordia et al. 2014 — arbitraged
below costs there) but persists in mid/small caps and around high-surprise
events. Gap-fade stats are stable but thin per-trade.

**Honest verdict — the interaction is the edge.** A gap **with** a fresh
catalyst and abnormal volume is a *different animal* from the same gap on no
news, and our engine measures exactly that (`results/interaction_gaps.csv`):
news gaps continue (+~20–80 bp over 1–5 days in our literature-calibrated
market; the huge-gap news cells are the strongest), no-news gaps fade
(−10–25 bp next day). **This news × chart × volume cell is where most of the
viable daily setups in the whole catalog live.** Per-trade economics still
have to clear ~15 bp round trip — only the mid/large news cells do.

## 7. Volume–price signals

**Method.** RVOL vs 63-day ADV; high-volume-low-move days (GKM); directional
RVOL spikes; A/D-line and OBV divergence/confirmation rules.

**Original evidence.** Gervais–Kaniel–Mingelgrin (JF 2001): the high-volume
return premium — extreme-volume days with small price moves precede
outperformance (visibility hypothesis), 1963–1996. OBV/A-D divergences:
practitioner lore, thin formal evidence.

**Decay.** HVRP replicated internationally (Kaniel et al. 2012), attenuated
in recent US large caps. Divergence rules: never robust in formal tests.

**Honest verdict.** Volume is a **conditioner, not a signal**: RVOL sharpens
every news/breakout edge (our interaction tables show monotone improvement
across RVOL terciles for news cells), the standalone HVRP is small-but-real,
and OBV/A-D divergences **fail after costs**.

## 8. Trend-following MA & channel systems

**Method.** Golden/death crosses (5/20, 20/50, 50/200), Bollinger and Keltner
channel rules, 12-1 time-series momentum, monthly cross-sectional 12-1
momentum deciles.

**Original evidence.** Brock–Lakonishok–LeBaron (JF 1992): MA rules beat
buy-and-hold on DJIA 1897–1986 pre-cost. Jegadeesh–Titman (1993): XS momentum
≈ 1%/mo. Moskowitz–Ooi–Pedersen (2012): TS momentum across 58 futures.

**Decay.** BLL rules: killed out-of-sample by data-snooping corrections (STW
1999) and post-1986 data. XS momentum: halved post-publication, with crash
episodes (2009, 2016, 2023-style unwinds); still positive gross in mid-caps.
TS momentum at the *index/futures* level remains the best-documented
survivor; single-stock MA crosses ≈ noise after costs.

**Honest verdict.** Single-stock golden crosses **fail after costs**
(our engine: only the slow embedded alpha keeps them barely positive gross —
exactly the literature's story). The tradable remnant: *slow* TS momentum and
monthly-grid XS momentum, both at weeks-to-months horizons — real but
incompatible with a 1-day trading cadence except as a directional filter.

## 9. ML pattern recognition — CNN on chart images (Jiang–Kelly–Xiu)

**Method (faithful).** 20-day windows rendered exactly as JKX render them
(3px/day OHLC bars + MA line + volume strip, 64×60 binary images); label =
sign of the next 5-day return; small CNN (2 conv-pool blocks → dense),
trained on the first 40% of the sample, applied strictly out-of-sample on a
weekly grid; top/bottom prediction deciles become long/short signals.
scikit-learn GBM fallback on pooled pixels if torch is unavailable, reported
under a separate tag.

**Original evidence.** Jiang, Kelly & Xiu (JF 2023), US 1993–2019: image-CNN
decile long-short earns large gross Sharpe (≈ 1.4–1.7 weekly rebalanced),
beats every hand-crafted signal family; the learned filters resemble
"trend + gap + volatility" composites.

**Decay/critique.** Returns concentrate in small/micro caps and shrink
sharply value-weighted; turnover is extreme, so **net-of-cost viability in
large caps is doubtful** (the authors say as much); post-2019 out-of-sample
evidence mixed.

**Honest verdict.** The single most powerful *pure-chart* technology known
per the published record — and still **cost-challenged at daily horizons in
liquid names**. Our own engine result is a *negative* and we report it as
such: the compute-bounded CNN (2 conv blocks, 4 epochs, 120k images, CPU)
learned almost nothing — training loss 0.6932 → 0.6877 against ln 2 =
0.6931, out-of-sample top-decile hit rate 48.8%, ml family −1.4 bp/day net,
and near-zero alignment with the simulator's embedded drift
(`results/ground_truth_recovery.csv`). JKX's positive result uses millions
of images and far larger models/compute; at this scale the method does not
replicate, and the catalog does not count the CNN as a working edge in this
build. The fusion model accordingly assigns it minimal weight.

## 10. Fundamentals: business quality / moat

**Method.** Static observable quality score (stands in for gross
profitability, ROIC stability, margin persistence — plug real fundamentals in
`loaders.py`); monthly top/bottom decile signals plus the interaction:
Donchian breakouts *filtered by* high quality.

**Original evidence.** Novy-Marx (JFE 2013): gross profitability ≈ value-
sized premium; Asness–Frazzini–Pedersen (RAS 2019): QMJ ≈ 4%/yr; the moat
story (persistent margins) is the economic mechanism.

**Decay.** Modest post-publication attenuation; quality held up better than
most factors (McLean–Pontiff).

**Honest verdict.** **Real, small, slow.** Irrelevant at a 1-day horizon on
its own; genuinely useful as a *filter* on chart entries (our quality ×
breakout cell outperforms unfiltered breakouts) — the "buy breakouts in moaty
businesses" composite is one of the few respectable chart+fundamental
marriages.

## 11. Engine-original indicators (novel — flagged as such)

Six new constructs defined in `engine/patterns/novel.py` (TPI trend purity,
CER compression-expansion, AMO anchored-memory oscillator, SPI shadow
pressure, LVX liquidity vacuum, GEX gap echo). **No prior literature, no
original evidence — these are hypotheses.** Per Harvey–Liu–Zhu (2016), novel
in-house rules carry a higher bar: |t| ≥ 3 and out-of-sample confirmation
before belief. Engine results: GEX and AMO-reclaim behave as news-drift
proxies (positive, by construction of the market); AMO-stretch-short
*fights* PEAD and loses — an honest negative result the truth-recovery table
exposes. None cleared the elevated bar as independent edges.

## 12. Signal fusion (the formulas)

Feature vector per (date, ticker): family net scores
$F_{f,it}=\sum_{s\in f}d_s\,\sigma_s$ (direction × strength of active
signals) plus context (gap, RVOL, catalyst, news sign, 52-wk proximity, 12-1
momentum, 20-day vol, quality). Meta-model: gradient-boosted trees
$\hat y_{it}=\text{GBM}(F_{it},C_{it})$ trained each month m on all data
< m (expanding window), predicting next-day market-adjusted tradable return
$y_{it}=\frac{C_{t+1}}{O_{t+1}}-1-r^{mkt}_{t+1}$; portfolio = long top-K /
short bottom-K scores among names with ≥1 active signal, equal weight, full
round-trip cost charged daily. A transparent linear baseline
$z\text{-sum}=\sum_f z(F_f)$ is reported beside it. Attribution = standalone
per-family portfolios + permutation importances of the fused model.

**Why fusion matters:** the interaction cells (news × gap × volume, quality ×
breakout, trend × pullback) carry most of the payoff; single families are
almost all sub-cost. This is the catalog's central quantitative lesson.

---

# The 1%/day question — answered honestly

The engine's walk-forward fused portfolio, after costs, on a market whose
effect sizes match the published literature, earns **basis points per day,
not 1%** (see `results/summary.json` and the bootstrap histogram: the entire
1,000-draw distribution of 5-year average daily returns sits far left of the
1% line). To average +1%/day (≈ ×12 per year compounded) with daily
long/short baskets you would need either ~10× leverage on an already-
optimistic gross edge, systematically captured *intraday* news reactions at
scale, or effect sizes ~30–50× anything documented in eighty years of
academic and practitioner literature. No chart-based catalog assembled from
honest components gets there on daily bars. **The correct reading of this
catalog is: a handful of small real edges (news gaps + volume, PEAD, 52-wk
anchoring, slow momentum, quality filters) that fuse into a modest, real
daily strategy — and a long list of famous patterns that do not survive
costs.**
