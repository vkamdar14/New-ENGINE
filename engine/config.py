"""Global configuration for the engine."""

from dataclasses import dataclass, field


@dataclass
class CostModel:
    """Per-side trading costs in basis points of notional.

    Defaults model a liquid US large/mid cap traded with marketable limit
    orders at modest size: half the quoted spread plus temporary impact,
    plus (optionally) commissions/fees.  These are deliberately on the
    conservative-but-realistic side; the catalog reports gross and net.
    """

    half_spread_bps: float = 3.0     # half quoted spread, liquid names
    impact_bps: float = 4.0          # temporary impact at small participation
    commission_bps: float = 0.5      # per-side commissions + fees

    @property
    def per_side_bps(self) -> float:
        return self.half_spread_bps + self.impact_bps + self.commission_bps

    @property
    def round_trip_bps(self) -> float:
        return 2.0 * self.per_side_bps


@dataclass
class SimConfig:
    """Synthetic market configuration.

    Effect sizes are calibrated to published estimates (see CATALOG.md,
    "Ground truth embedded in the simulator").  Families with no credible
    real-world effect (candlesticks per se, round numbers) are embedded
    with ZERO intrinsic effect and act as placebo controls for the
    detection machinery.
    """

    n_stocks: int = 500
    n_sectors: int = 10
    years: float = 6.25              # ~1 year warm-up + contiguous 5y test
    seed: int = 7

    # market factor dynamics (overridden by real-data calibration if used)
    mkt_mu: float = 0.00035          # ~9% p.a. drift
    garch_omega: float = 2.0e-6
    garch_alpha: float = 0.10
    garch_beta: float = 0.87
    t_dof: float = 6.0

    # idiosyncratic vol
    idio_vol_lo: float = 0.012
    idio_vol_hi: float = 0.035

    # news / catalyst process
    catalyst_rate: float = 1.0 / 55.0    # ~4.6 events / stock / year (earnings + episodic)
    jump_sigma_ln: float = 0.55          # lognormal sigma of |jump|
    jump_mu_ln: float = -3.15            # median |jump| ~ 4.3%
    pead_share: float = 0.10             # share of news jump that continues as drift
    pead_days: int = 10                  # over this many days (PEAD-calibrated)
    news_vol_mult_lo: float = 2.0        # RVOL multiplier on event day
    news_vol_mult_hi: float = 8.0

    # no-news overnight gaps partially fade (Plastun et al.-style)
    nonews_gap_fade: float = -0.12       # E[next-day r] = fade * gap size

    # 52-week-high anchoring (George-Hwang-calibrated, small)
    anchor_drift: float = 0.00018        # extra daily drift when within 2% of 52wk high
    anchor_band: float = 0.02

    # slow-moving stock alpha (trend / cross-sectional momentum host)
    alpha_ou_halflife: int = 120         # days
    alpha_ou_sigma: float = 0.0006       # stationary daily-alpha dispersion
                                         # (keeps realized momentum decile spreads
                                         #  near the ~1.2%/month of the literature
                                         #  after estimation noise)

    # quality / moat premium (Novy-Marx / QMJ calibrated, small)
    quality_premium: float = 0.00008     # daily drift spread of top-vs-bottom quality decile

    # microstructure
    tick: float = 0.01
    price_lo: float = 8.0
    price_hi: float = 400.0


@dataclass
class RunConfig:
    outdir: str = "results"
    horizons: tuple = (1, 2, 3, 5, 10, 20)
    test_years: int = 5              # contiguous test window at the end
    top_k: int = 10                  # names per side in fused portfolio
    n_bootstrap: int = 1000          # random-draw averages for the histogram
    target_daily: float = 0.01       # the 1%/day target line
    costs: CostModel = field(default_factory=CostModel)
    sim: SimConfig = field(default_factory=SimConfig)
