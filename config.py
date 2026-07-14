from dataclasses import dataclass, field
from typing import Optional, Set

# ---------------------------------------------------------------------------
# Instrument specifications
# ---------------------------------------------------------------------------
INSTRUMENT_SPECS = {
    "MNQ": {"tick_size": 0.25, "tick_value": 0.50,  "commission": 0.85, "margin": 40},
    "MES": {"tick_size": 0.25, "tick_value": 1.25,  "commission": 0.62, "margin": 40},
    "MYM": {"tick_size": 1.00, "tick_value": 0.50,  "commission": 0.62, "margin": 40},
    "M2K": {"tick_size": 0.10, "tick_value": 0.50,  "commission": 0.62, "margin": 40},
}

# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
NY_OPEN_UTC   = "13:30"
SESSION_START = "13:30"
SESSION_END   = "20:00"

# ---------------------------------------------------------------------------
# Execution (DO NOT CHANGE — realistic model)
# ---------------------------------------------------------------------------
SLIPPAGE_TICKS = 1

# ---------------------------------------------------------------------------
# ICT Confirmation methods
# Options: "mss", "displacement", "fvg", "ob"
# ---------------------------------------------------------------------------
CONFIRM_METHODS       = {"mss", "displacement", "fvg", "ob"}
DISPLACEMENT_ATR_MULT = 1.5

# ---------------------------------------------------------------------------
# Stop loss
# STOP_MODE: "sweep" | "swing" | "fixed_points" | "atr"
# ---------------------------------------------------------------------------
STOP_MODE         = "sweep"
STOP_FIXED_POINTS = 10.0
STOP_ATR_MULT     = 1.5
ATR_PERIOD        = 14

# ---------------------------------------------------------------------------
# Take profit
# TP_MODE: "fixed_r" | "session_high_low" | "fixed_points" |
#          "opposite_level" | "midpoint" | "end_of_session" |
#          "trailing" | "partials" | "liquidity_target"
# ---------------------------------------------------------------------------
TP_MODE          = "fixed_r"
TP_FIXED_POINTS  = 20.0
TP_R_MULTIPLE    = 3.0
TRAILING_TICKS   = 8

# -- Break-even --
USE_BREAKEVEN        = False
BREAKEVEN_TRIGGER_R  = 1.0   # move stop to entry after this R is reached

# -- Partial profits --
USE_PARTIALS         = False
PARTIAL_1_R          = 1.0   # take partial at this R
PARTIAL_1_PCT        = 0.50  # fraction of position to close
PARTIAL_2_R          = 2.0   # remainder target (or trailing if PARTIAL_2_TRAIL=True)
PARTIAL_2_TRAIL      = False

# -- ATR trailing stop --
USE_ATR_TRAIL        = False
ATR_TRAIL_MULT       = 1.5

# -- Liquidity target --
# TP_MODE = "liquidity_target" uses the closest opposing liquidity level
# Options: "prev_day_high", "prev_day_low", "session_high", "session_low", "opposing"
LIQUIDITY_TARGET_MODE = "opposing"

# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------
STARTING_EQUITY   = 10_000.0
RISK_MODE         = "fixed_dollar"   # "fixed_dollar" | "pct_equity"
RISK_FIXED_DOLLAR = 100.0
RISK_PCT_EQUITY   = 0.01
MAX_CONTRACTS     = 10

# ---------------------------------------------------------------------------
# Filter 1 — Opening Range Quality
# USE_OR_QUALITY: enable ATR-normalised range filter
# OR_MIN_ATR_MULT / OR_MAX_ATR_MULT: reject if range < min or > max * ATR
# OR_PERCENTILE_WINDOW: rolling window (days) for percentile thresholds
# OR_MIN_PERCENTILE / OR_MAX_PERCENTILE: reject outside these percentiles
# ---------------------------------------------------------------------------
USE_OR_QUALITY        = True
OR_FILTER_MODE        = "atr"        # "atr" | "percentile"
OR_MIN_ATR_MULT       = 0.3
OR_MAX_ATR_MULT       = 3.0
OR_PERCENTILE_WINDOW  = 60           # trading days
OR_MIN_PERCENTILE     = 10
OR_MAX_PERCENTILE     = 90

# ---------------------------------------------------------------------------
# Filter 2 — Sweep Depth
# Minimum distance the sweep must exceed the first-minute H/L
# SWEEP_DEPTH_MODE: "ticks" | "atr_pct"
# ---------------------------------------------------------------------------
USE_SWEEP_DEPTH       = True
SWEEP_DEPTH_MODE      = "ticks"
SWEEP_DEPTH_TICKS     = 2
SWEEP_DEPTH_ATR_PCT   = 0.10         # 10% of ATR

# ---------------------------------------------------------------------------
# Filter 3 — Displacement Quality
# Require the confirmation candle to show genuine displacement
# DISP_BODY_PCT: body / total_range >= this value
# DISP_CLOSE_NEAR_EXTREME_PCT: close within this % of high (bull) or low (bear)
# ---------------------------------------------------------------------------
USE_DISPLACEMENT_FILTER    = True
DISP_BODY_PCT              = 0.50    # body >= 50% of candle range
DISP_CLOSE_NEAR_EXTREME    = True
DISP_CLOSE_NEAR_PCT        = 0.35    # close in top/bottom 35% of candle

# ---------------------------------------------------------------------------
# Filter 4 — Market Structure Shift strength
# Require a confirmed HH (long) or LL (short) after the sweep
# MSS_LOOKBACK: how many post-reclaim candles to look back for the prior swing
# ---------------------------------------------------------------------------
USE_STRONG_MSS        = True
MSS_LOOKBACK          = 3

# ---------------------------------------------------------------------------
# Filter 5 — FVG Quality
# FVG_MIN_SIZE_ATR_PCT: FVG gap must be >= this fraction of ATR
# FVG_MUST_BE_UNFILLED: reject if FVG is immediately filled on the next candle
# FVG_AFTER_DISPLACEMENT: FVG must appear after a displacement candle
# ---------------------------------------------------------------------------
USE_FVG_QUALITY            = True
FVG_MIN_SIZE_ATR_PCT       = 0.10
FVG_MUST_BE_UNFILLED       = True
FVG_AFTER_DISPLACEMENT     = True

# ---------------------------------------------------------------------------
# Filter 6 — Trend Filter (optional)
# TREND_FILTER_MODE: "ema50" | "ema200" | "vwap" | "prev_day" | "none"
# ---------------------------------------------------------------------------
USE_TREND_FILTER      = False
TREND_FILTER_MODE     = "ema50"      # "ema50" | "ema200" | "vwap" | "prev_day"

# ---------------------------------------------------------------------------
# Filter 7 — Volatility Regime
# VOLREG_MODE: "atr" | "or_size" | "prev_day_atr" | "realized_vol"
# ---------------------------------------------------------------------------
USE_VOLREG_FILTER     = False
VOLREG_MODE           = "atr"
VOLREG_MIN_PERCENTILE = 20           # skip bottom 20% volatility days
VOLREG_MAX_PERCENTILE = 90           # skip top 10% volatility days
VOLREG_WINDOW         = 60

# ---------------------------------------------------------------------------
# Filter 8 — Time Filter
# Only accept entries within N minutes of the NY open
# ---------------------------------------------------------------------------
USE_TIME_FILTER       = True
MAX_ENTRY_MINUTES     = 20           # 5 | 10 | 20

# ---------------------------------------------------------------------------
# Filter 9 — News Filter
# NEWS_FILTER_CSV: path to CSV with column "date" (YYYY-MM-DD)
# NEWS_FILTER_TYPES: set of event types to skip (for future calendar integration)
# ---------------------------------------------------------------------------
NEWS_FILTER_CSV       = None
NEWS_FILTER_TYPES: Set[str] = {"CPI", "FOMC", "NFP"}

# ---------------------------------------------------------------------------
# Volume filter
# ---------------------------------------------------------------------------
MIN_VOLUME        = 0

# ---------------------------------------------------------------------------
# Gap filter
# ---------------------------------------------------------------------------
USE_GAP_FILTER    = False
MAX_GAP_POINTS    = 50.0

# ---------------------------------------------------------------------------
# Walk-forward split
# ---------------------------------------------------------------------------
TRAIN_PCT    = 0.60
VALIDATE_PCT = 0.20

# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
MC_SIMULATIONS = 10_000

# ---------------------------------------------------------------------------
# Optimisation grid  (walk-forward aware — train set only)
# Keep small: robustness over exhaustive search
# ---------------------------------------------------------------------------
OPT_STOP_FIXED_POINTS = [5, 8, 10, 12, 15, 20]
OPT_TP_FIXED_POINTS   = [10, 15, 20, 25, 30, 40]
OPT_ATR_MULT          = [1.0, 1.5, 2.0, 2.5]
OPT_SESSION_END_HOURS = ["15:30", "16:00", "17:00", "20:00"]

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR = "output"
