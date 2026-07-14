import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
import config


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DaySetup:
    date:          object
    ny_open_high:  float
    ny_open_low:   float
    ny_open_range: float
    atr:           float
    prev_close:    float
    prev_high:     float
    prev_low:      float
    prev_atr:      float
    valid:         bool
    skip_reason:   str = ""


@dataclass
class Signal:
    date:           object
    signal_time:    pd.Timestamp
    entry_time:     Optional[pd.Timestamp]
    direction:      str               # "long" | "short"
    signal_close:   float
    sweep_extreme:  float             # actual sweep low/high — stop anchor
    swing_ref:      float             # alias for ExecutionEngine
    ny_open_high:   float
    ny_open_low:    float
    atr:            float
    confirm_type:   str = ""
    fvg_top:        float = 0.0
    fvg_bottom:     float = 0.0
    session_high:   float = 0.0
    session_low:    float = 0.0
    # enrichment for analytics
    sweep_depth:    float = 0.0       # how far price swept beyond the level
    or_size:        float = 0.0       # opening range size
    or_atr_ratio:   float = 0.0       # or_size / atr
    entry_minute:   int   = 0         # minutes after NY open
    trend_aligned:  bool  = True
    prev_high:      float = 0.0
    prev_low:       float = 0.0
    displacement_body_pct: float = 0.0
    fvg_size:       float = 0.0


# ---------------------------------------------------------------------------
# Rolling percentile helper (uses only past data — no look-ahead)
# ---------------------------------------------------------------------------

def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Return the percentile rank of each value within the preceding `window` values."""
    def _rank(arr):
        if len(arr) < 2:
            return 50.0
        return float(np.sum(arr[:-1] < arr[-1]) / (len(arr) - 1) * 100)
    return series.rolling(window + 1, min_periods=2).apply(_rank, raw=True)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class Strategy:
    def __init__(self, news_dates: set = None):
        self.news_dates = news_dates or set()
        # Caches for rolling percentile ranks — populated lazily per instrument
        self._or_pct_cache:  dict[str, pd.Series] = {}
        self._vol_pct_cache: dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # Pre-compute rolling percentile ranks for a full instrument DataFrame.
    # Must be called once before running the backtest for that instrument.
    # ------------------------------------------------------------------
    def precompute(self, df: pd.DataFrame, instrument: str):
        # Opening range size per day (at the NY open candle)
        open_candles = df[df["hhmm"] == config.NY_OPEN_UTC].copy()
        or_sizes = (open_candles["High"] - open_candles["Low"]).reset_index(drop=True)
        or_pct = _rolling_percentile_rank(or_sizes, config.OR_PERCENTILE_WINDOW)
        self._or_pct_cache[instrument] = pd.Series(
            or_pct.values, index=open_candles["date"].values
        )

        # Volatility regime: ATR at the open candle
        atr_vals = open_candles["atr"].reset_index(drop=True)
        vol_pct = _rolling_percentile_rank(atr_vals, config.VOLREG_WINDOW)
        self._vol_pct_cache[instrument] = pd.Series(
            vol_pct.values, index=open_candles["date"].values
        )

    # ------------------------------------------------------------------
    # Build per-day setup — all day-level filters applied here
    # ------------------------------------------------------------------
    def build_day_setup(self, day_df: pd.DataFrame, instrument: str) -> DaySetup:
        date = day_df["date"].iloc[0]

        # Filter 9 — News
        if str(date) in self.news_dates:
            return DaySetup(date, 0, 0, 0, 0, 0, 0, 0, 0, False, "news_day")

        open_candle = day_df[day_df["hhmm"] == config.NY_OPEN_UTC]
        if open_candle.empty:
            return DaySetup(date, 0, 0, 0, 0, 0, 0, 0, 0, False, "no_open_candle")

        oc         = open_candle.iloc[0]
        ny_high    = oc["High"]
        ny_low     = oc["Low"]
        ny_range   = ny_high - ny_low
        atr        = oc["atr"]
        prev_close = oc.get("prev_close", np.nan)
        prev_high  = oc.get("prev_high",  np.nan)
        prev_low   = oc.get("prev_low",   np.nan)
        prev_atr   = oc.get("prev_atr",   np.nan)

        # Gap filter
        if config.USE_GAP_FILTER and not np.isnan(prev_close):
            if abs(oc["Open"] - prev_close) > config.MAX_GAP_POINTS:
                return DaySetup(date, ny_high, ny_low, ny_range, atr,
                                prev_close, prev_high, prev_low, prev_atr,
                                False, "gap_too_large")

        # Filter 1 — Opening Range Quality
        if config.USE_OR_QUALITY:
            reason = self._check_or_quality(date, ny_range, atr, instrument)
            if reason:
                return DaySetup(date, ny_high, ny_low, ny_range, atr,
                                prev_close, prev_high, prev_low, prev_atr,
                                False, reason)

        # Filter 7 — Volatility Regime
        if config.USE_VOLREG_FILTER:
            reason = self._check_volreg(date, atr, prev_atr, ny_range, instrument)
            if reason:
                return DaySetup(date, ny_high, ny_low, ny_range, atr,
                                prev_close, prev_high, prev_low, prev_atr,
                                False, reason)

        return DaySetup(date, ny_high, ny_low, ny_range, atr,
                        prev_close, prev_high, prev_low, prev_atr, True)

    # ------------------------------------------------------------------
    # Filter 1 helpers
    # ------------------------------------------------------------------
    def _check_or_quality(self, date, ny_range: float, atr: float,
                          instrument: str) -> str:
        if config.OR_FILTER_MODE == "atr" and atr > 0:
            if ny_range < config.OR_MIN_ATR_MULT * atr:
                return "or_too_small_atr"
            if ny_range > config.OR_MAX_ATR_MULT * atr:
                return "or_too_large_atr"
        elif config.OR_FILTER_MODE == "percentile":
            pct_series = self._or_pct_cache.get(instrument)
            if pct_series is not None and date in pct_series.index:
                pct = pct_series[date]
                if pct < config.OR_MIN_PERCENTILE:
                    return "or_too_small_pct"
                if pct > config.OR_MAX_PERCENTILE:
                    return "or_too_large_pct"
        return ""

    # ------------------------------------------------------------------
    # Filter 7 helpers
    # ------------------------------------------------------------------
    def _check_volreg(self, date, atr: float, prev_atr: float,
                      ny_range: float, instrument: str) -> str:
        mode = config.VOLREG_MODE
        if mode == "atr":
            ref = atr
        elif mode == "prev_day_atr":
            ref = prev_atr if not np.isnan(prev_atr) else atr
        elif mode == "or_size":
            ref = ny_range
        else:
            ref = atr

        vol_series = self._vol_pct_cache.get(instrument)
        if vol_series is not None and date in vol_series.index:
            pct = vol_series[date]
            if pct < config.VOLREG_MIN_PERCENTILE:
                return "volreg_too_low"
            if pct > config.VOLREG_MAX_PERCENTILE:
                return "volreg_too_high"
        return ""

    # ------------------------------------------------------------------
    # Main signal finder
    # ------------------------------------------------------------------
    def find_signal(self, day_df: pd.DataFrame, setup: DaySetup,
                    instrument: str, tick_size: float = 0.25) -> Optional[Signal]:
        self._tick_size = tick_size
        if not setup.valid:
            return None

        open_time = day_df[day_df["hhmm"] == config.NY_OPEN_UTC]["time"].iloc[0]

        session_mask = (
            (day_df["hhmm"] >= config.SESSION_START) &
            (day_df["hhmm"] <  config.SESSION_END) &
            (day_df["time"] >  open_time)
        )
        if config.MIN_VOLUME > 0:
            session_mask &= (day_df["Volume"] >= config.MIN_VOLUME)

        candles = day_df[session_mask].reset_index(drop=True)
        if candles.empty:
            return None

        long_signal  = self._scan_direction(candles, setup, "long",  open_time)
        short_signal = self._scan_direction(candles, setup, "short", open_time)
        self._tick_size = 0.25  # reset

        if long_signal is None and short_signal is None:
            return None
        if long_signal is None:
            return short_signal
        if short_signal is None:
            return long_signal
        return long_signal if long_signal.signal_time <= short_signal.signal_time else short_signal

    # ------------------------------------------------------------------
    def _scan_direction(self, candles: pd.DataFrame, setup: DaySetup,
                        direction: str,
                        open_time: pd.Timestamp) -> Optional[Signal]:
        ny_high = setup.ny_open_high
        ny_low  = setup.ny_open_low
        tick    = getattr(self, "_tick_size", 0.25)
        n = len(candles)

        # Phase 1 & 2: sweep + reclaim
        reclaim_i, sweep_extreme = self._find_sweep_reclaim(
            candles, direction, ny_high, ny_low
        )
        if reclaim_i is None:
            return None

        # Filter 2 — Sweep Depth
        if config.USE_SWEEP_DEPTH:
            if not self._check_sweep_depth(sweep_extreme, direction,
                                           ny_high, ny_low, setup.atr):
                return None

        # Filter 8 — Time Filter
        if config.USE_TIME_FILTER:
            sweep_time = candles.iloc[reclaim_i]["time"]
            minutes_elapsed = (sweep_time - open_time).total_seconds() / 60
            if minutes_elapsed > config.MAX_ENTRY_MINUTES:
                return None

        # Phase 3: confirmation
        post = candles.iloc[reclaim_i + 1:].reset_index(drop=True)
        if post.empty:
            return None

        sess_high = candles.iloc[: reclaim_i + 1]["High"].max()
        sess_low  = candles.iloc[: reclaim_i + 1]["Low"].min()

        confirm = self._find_confirmation(post, direction, ny_high, ny_low, setup)
        if confirm is None:
            return None

        conf_row, conf_type, fvg_top, fvg_bottom, disp_body_pct, fvg_size = confirm

        # Filter 6 — Trend Filter
        if config.USE_TREND_FILTER:
            if not self._check_trend(conf_row, direction):
                return None

        entry_minute = int(
            (conf_row["time"] - open_time).total_seconds() / 60
        )

        if direction == "long":
            sweep_depth = ny_low - sweep_extreme
        else:
            sweep_depth = sweep_extreme - ny_high

        return Signal(
            date=setup.date,
            signal_time=conf_row["time"],
            entry_time=None,
            direction=direction,
            signal_close=conf_row["Close"],
            sweep_extreme=sweep_extreme,
            swing_ref=sweep_extreme,
            ny_open_high=ny_high,
            ny_open_low=ny_low,
            atr=setup.atr,
            confirm_type=conf_type,
            fvg_top=fvg_top,
            fvg_bottom=fvg_bottom,
            session_high=sess_high,
            session_low=sess_low,
            sweep_depth=sweep_depth,
            or_size=setup.ny_open_range,
            or_atr_ratio=setup.ny_open_range / setup.atr if setup.atr > 0 else 0,
            entry_minute=entry_minute,
            trend_aligned=True,
            prev_high=setup.prev_high,
            prev_low=setup.prev_low,
            displacement_body_pct=disp_body_pct,
            fvg_size=fvg_size,
        )

    # ------------------------------------------------------------------
    def _find_sweep_reclaim(
        self, candles: pd.DataFrame, direction: str,
        ny_high: float, ny_low: float
    ) -> Tuple[Optional[int], Optional[float]]:
        n = len(candles)
        for i in range(n):
            row = candles.iloc[i]
            if direction == "long":
                if row["Low"] < ny_low and row["Close"] > ny_low:
                    return i, row["Low"]
                if row["Low"] < ny_low and row["Close"] <= ny_low:
                    if i + 1 < n and candles.iloc[i + 1]["Close"] > ny_low:
                        return i + 1, row["Low"]
            else:
                if row["High"] > ny_high and row["Close"] < ny_high:
                    return i, row["High"]
                if row["High"] > ny_high and row["Close"] >= ny_high:
                    if i + 1 < n and candles.iloc[i + 1]["Close"] < ny_high:
                        return i + 1, row["High"]
        return None, None

    # ------------------------------------------------------------------
    # Filter 2 — Sweep Depth
    # ------------------------------------------------------------------
    def _check_sweep_depth(self, sweep_extreme: float, direction: str,
                           ny_high: float, ny_low: float, atr: float) -> bool:
        if direction == "long":
            depth = ny_low - sweep_extreme
        else:
            depth = sweep_extreme - ny_high

        if config.SWEEP_DEPTH_MODE == "ticks":
            # Use MES tick as reference (0.25); caller should pass instrument tick
            # but we use a conservative 0.25 as minimum unit
            min_depth = config.SWEEP_DEPTH_TICKS * 0.25
        else:
            min_depth = config.SWEEP_DEPTH_ATR_PCT * atr if atr > 0 else 0

        return depth >= min_depth

    # ------------------------------------------------------------------
    # Filter 6 — Trend
    # ------------------------------------------------------------------
    def _check_trend(self, conf_row: pd.Series, direction: str) -> bool:
        mode = config.TREND_FILTER_MODE
        close = conf_row["Close"]

        if mode == "ema50":
            ema = conf_row.get("ema50", np.nan)
            if np.isnan(ema):
                return True
            return (direction == "long" and close > ema) or \
                   (direction == "short" and close < ema)

        elif mode == "ema200":
            ema = conf_row.get("ema200", np.nan)
            if np.isnan(ema):
                return True
            return (direction == "long" and close > ema) or \
                   (direction == "short" and close < ema)

        elif mode == "vwap":
            vwap = conf_row.get("vwap", np.nan)
            if np.isnan(vwap):
                return True
            return (direction == "long" and close > vwap) or \
                   (direction == "short" and close < vwap)

        elif mode == "prev_day":
            prev_close = conf_row.get("prev_close", np.nan)
            if np.isnan(prev_close):
                return True
            return (direction == "long" and close > prev_close) or \
                   (direction == "short" and close < prev_close)

        return True  # "none" or unknown

    # ------------------------------------------------------------------
    # Phase 3 — Confirmation
    # ------------------------------------------------------------------
    def _find_confirmation(
        self, post: pd.DataFrame, direction: str,
        ny_high: float, ny_low: float, setup: DaySetup
    ) -> Optional[Tuple]:
        """
        Returns (row, confirm_type, fvg_top, fvg_bottom, disp_body_pct, fvg_size)
        or None.
        """
        enabled = config.CONFIRM_METHODS
        n = len(post)
        had_displacement = False  # tracks whether a displacement candle was seen

        for i in range(n):
            row  = post.iloc[i]
            atr  = row.get("atr", setup.atr) or setup.atr
            body = abs(row["Close"] - row["Open"])
            rng  = row["High"] - row["Low"]
            body_pct = body / rng if rng > 0 else 0

            # -- MSS (Market Structure Shift) --
            if "mss" in enabled and i > 0:
                prev = post.iloc[i - 1]
                if config.USE_STRONG_MSS:
                    # Require HH (long) or LL (short) vs lookback window
                    lookback = post.iloc[max(0, i - config.MSS_LOOKBACK): i]
                    if direction == "long":
                        prior_high = lookback["High"].max() if not lookback.empty else prev["High"]
                        if row["Close"] > prior_high:
                            return row, "mss", 0.0, 0.0, body_pct, 0.0
                    else:
                        prior_low = lookback["Low"].min() if not lookback.empty else prev["Low"]
                        if row["Close"] < prior_low:
                            return row, "mss", 0.0, 0.0, body_pct, 0.0
                else:
                    if direction == "long" and row["Close"] > prev["High"]:
                        return row, "mss", 0.0, 0.0, body_pct, 0.0
                    if direction == "short" and row["Close"] < prev["Low"]:
                        return row, "mss", 0.0, 0.0, body_pct, 0.0

            # -- Displacement candle --
            if "displacement" in enabled:
                is_displacement = (
                    atr > 0 and body >= config.DISPLACEMENT_ATR_MULT * atr
                )
                # Filter 3 — Displacement Quality
                if config.USE_DISPLACEMENT_FILTER and is_displacement:
                    is_displacement = self._check_displacement_quality(
                        row, direction, body_pct
                    )

                if is_displacement:
                    bullish = row["Close"] > row["Open"]
                    if direction == "long" and bullish:
                        had_displacement = True
                        return row, "displacement", 0.0, 0.0, body_pct, 0.0
                    if direction == "short" and not bullish:
                        had_displacement = True
                        return row, "displacement", 0.0, 0.0, body_pct, 0.0

            # -- Fair Value Gap --
            if "fvg" in enabled and i >= 2:
                c0 = post.iloc[i - 2]
                c2 = row
                fvg_top = fvg_bottom = 0.0
                is_fvg = False

                if direction == "long" and c0["High"] < c2["Low"]:
                    fvg_top, fvg_bottom = c2["Low"], c0["High"]
                    is_fvg = True
                elif direction == "short" and c0["Low"] > c2["High"]:
                    fvg_top, fvg_bottom = c0["Low"], c2["High"]
                    is_fvg = True

                if is_fvg:
                    fvg_size = fvg_top - fvg_bottom
                    # Filter 5 — FVG Quality
                    if config.USE_FVG_QUALITY:
                        if not self._check_fvg_quality(
                            fvg_size, fvg_top, fvg_bottom, direction,
                            post, i, atr, had_displacement
                        ):
                            is_fvg = False

                if is_fvg:
                    return row, "fvg", fvg_top, fvg_bottom, body_pct, fvg_size

            # -- Order Block --
            if "ob" in enabled and i >= 1:
                prev = post.iloc[i - 1]
                body_prev = abs(prev["Close"] - prev["Open"])
                body_curr = abs(row["Close"] - row["Open"])
                if direction == "long":
                    if prev["Close"] < prev["Open"] and \
                       row["Close"] > row["Open"] and body_curr > body_prev:
                        return row, "ob", prev["High"], prev["Low"], body_pct, 0.0
                else:
                    if prev["Close"] > prev["Open"] and \
                       row["Close"] < row["Open"] and body_curr > body_prev:
                        return row, "ob", prev["High"], prev["Low"], body_pct, 0.0

        return None

    # ------------------------------------------------------------------
    # Filter 3 — Displacement Quality
    # ------------------------------------------------------------------
    def _check_displacement_quality(self, row: pd.Series, direction: str,
                                     body_pct: float) -> bool:
        if body_pct < config.DISP_BODY_PCT:
            return False
        if config.DISP_CLOSE_NEAR_EXTREME:
            rng = row["High"] - row["Low"]
            if rng == 0:
                return False
            if direction == "long":
                # close should be in the upper portion of the candle
                close_pos = (row["Close"] - row["Low"]) / rng
                return close_pos >= (1 - config.DISP_CLOSE_NEAR_PCT)
            else:
                close_pos = (row["High"] - row["Close"]) / rng
                return close_pos >= (1 - config.DISP_CLOSE_NEAR_PCT)
        return True

    # ------------------------------------------------------------------
    # Filter 5 — FVG Quality
    # ------------------------------------------------------------------
    def _check_fvg_quality(self, fvg_size: float, fvg_top: float,
                            fvg_bottom: float, direction: str,
                            post: pd.DataFrame, i: int,
                            atr: float, had_displacement: bool) -> bool:
        # Minimum size
        if atr > 0 and fvg_size < config.FVG_MIN_SIZE_ATR_PCT * atr:
            return False

        # Must not be immediately filled on the very next candle
        if config.FVG_MUST_BE_UNFILLED and i + 1 < len(post):
            nxt = post.iloc[i + 1]
            if direction == "long" and nxt["Low"] <= fvg_bottom:
                return False
            if direction == "short" and nxt["High"] >= fvg_top:
                return False

        # Must appear after a displacement candle
        if config.FVG_AFTER_DISPLACEMENT and not had_displacement:
            return False

        return True


# ---------------------------------------------------------------------------
# Instrument-aware sweep depth check (used by Backtester)
# ---------------------------------------------------------------------------
MIN_VOLUME = 0  # kept for backward compat; config.MIN_VOLUME used in strategy
