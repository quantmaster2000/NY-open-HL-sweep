import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
import config
from config import INSTRUMENT_SPECS
from strategy import Signal


@dataclass
class Trade:
    instrument:    str
    date:          object
    direction:     str
    entry_time:    pd.Timestamp
    exit_time:     pd.Timestamp
    entry_price:   float
    exit_price:    float
    stop_price:    float
    target_price:  float
    contracts:     int
    pnl_points:    float
    pnl_ticks:     float
    pnl_dollars:   float
    commission:    float
    slippage_cost: float
    exit_reason:   str
    mae_points:    float
    mfe_points:    float
    equity_before: float
    equity_after:  float
    risk_pct:      float
    hold_bars:     int
    r_multiple:    float
    # Signal enrichment (for analytics)
    confirm_type:  str   = ""
    sweep_depth:   float = 0.0
    or_size:       float = 0.0
    or_atr_ratio:  float = 0.0
    entry_minute:  int   = 0
    trend_aligned: bool  = True
    prev_high:     float = 0.0
    prev_low:      float = 0.0
    displacement_body_pct: float = 0.0
    fvg_size:      float = 0.0


class ExecutionEngine:
    def __init__(self, instrument: str):
        self.instrument = instrument
        self.spec       = INSTRUMENT_SPECS[instrument]
        self.tick       = self.spec["tick_size"]
        self.tick_val   = self.spec["tick_value"]
        self.commission_per_side = self.spec["commission"]

    # ------------------------------------------------------------------
    def execute(self, signal: Signal, day_df: pd.DataFrame,
                contracts: int, equity_before: float) -> Optional[Trade]:

        after = day_df[day_df["time"] > signal.signal_time].copy()
        if after.empty:
            return None

        entry_candle = after.iloc[0]
        entry_time   = entry_candle["time"]

        slip = config.SLIPPAGE_TICKS * self.tick
        entry_price = (entry_candle["Open"] + slip
                       if signal.direction == "long"
                       else entry_candle["Open"] - slip)

        stop_price   = self._compute_stop(signal, entry_price)
        target_price = self._compute_target(signal, entry_price, stop_price)

        session_candles = day_df[
            (day_df["time"] >= entry_time) &
            (day_df["hhmm"] < config.SESSION_END)
        ].copy()

        if session_candles.empty:
            return None

        exit_price, exit_time, exit_reason, mae, mfe, hold_bars = self._simulate(
            signal, entry_price, stop_price, target_price, session_candles
        )

        # Unfavourable slippage on exit
        exit_price = (exit_price - slip if signal.direction == "long"
                      else exit_price + slip)

        pnl_points = (exit_price - entry_price if signal.direction == "long"
                      else entry_price - exit_price)
        pnl_ticks   = pnl_points / self.tick
        pnl_dollars = pnl_ticks * self.tick_val * contracts

        commission    = self.commission_per_side * 2 * contracts
        slippage_cost = (config.SLIPPAGE_TICKS * self.tick_val * 2) * contracts
        net_pnl       = pnl_dollars - commission

        risk_pts     = abs(entry_price - stop_price)
        risk_dollars = risk_pts / self.tick * self.tick_val * contracts
        risk_pct     = risk_dollars / equity_before if equity_before > 0 else 0
        r_multiple   = pnl_points / risk_pts if risk_pts > 0 else 0.0

        return Trade(
            instrument=self.instrument,
            date=signal.date,
            direction=signal.direction,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_price=stop_price,
            target_price=target_price if target_price is not None else 0.0,
            contracts=contracts,
            pnl_points=pnl_points,
            pnl_ticks=pnl_ticks,
            pnl_dollars=net_pnl,
            commission=commission,
            slippage_cost=slippage_cost,
            exit_reason=exit_reason,
            mae_points=mae,
            mfe_points=mfe,
            equity_before=equity_before,
            equity_after=equity_before + net_pnl,
            risk_pct=risk_pct,
            hold_bars=hold_bars,
            r_multiple=r_multiple,
            confirm_type=signal.confirm_type,
            sweep_depth=signal.sweep_depth,
            or_size=signal.or_size,
            or_atr_ratio=signal.or_atr_ratio,
            entry_minute=signal.entry_minute,
            trend_aligned=signal.trend_aligned,
            prev_high=signal.prev_high,
            prev_low=signal.prev_low,
            displacement_body_pct=signal.displacement_body_pct,
            fvg_size=signal.fvg_size,
        )

    # ------------------------------------------------------------------
    def _compute_stop(self, signal: Signal, entry_price: float) -> float:
        tick = self.tick
        if config.STOP_MODE == "sweep":
            return (signal.sweep_extreme - tick if signal.direction == "long"
                    else signal.sweep_extreme + tick)
        elif config.STOP_MODE == "swing":
            return (signal.swing_ref - tick if signal.direction == "long"
                    else signal.swing_ref + tick)
        elif config.STOP_MODE == "fixed_points":
            return (entry_price - config.STOP_FIXED_POINTS if signal.direction == "long"
                    else entry_price + config.STOP_FIXED_POINTS)
        elif config.STOP_MODE == "atr":
            dist = config.STOP_ATR_MULT * signal.atr
            return (entry_price - dist if signal.direction == "long"
                    else entry_price + dist)
        return entry_price

    def _compute_target(self, signal: Signal, entry_price: float,
                        stop_price: float) -> Optional[float]:
        mode = config.TP_MODE
        risk = abs(entry_price - stop_price)

        if mode == "end_of_session":
            return None
        elif mode == "fixed_r":
            dist = risk * config.TP_R_MULTIPLE
        elif mode == "session_high_low":
            return (signal.session_high if signal.direction == "long"
                    else signal.session_low) or None
        elif mode == "fixed_points":
            dist = config.TP_FIXED_POINTS
        elif mode == "opposite_level":
            dist = (abs(signal.ny_open_high - entry_price) if signal.direction == "long"
                    else abs(entry_price - signal.ny_open_low))
        elif mode == "midpoint":
            mid = (signal.ny_open_high + signal.ny_open_low) / 2
            dist = abs(entry_price - mid)
        elif mode in ("trailing", "partials"):
            dist = risk * config.TP_R_MULTIPLE
        elif mode == "liquidity_target":
            return self._liquidity_target(signal, entry_price)
        else:
            return None

        return (entry_price + dist if signal.direction == "long"
                else entry_price - dist)

    def _liquidity_target(self, signal: Signal, entry_price: float) -> Optional[float]:
        mode = config.LIQUIDITY_TARGET_MODE
        if mode == "prev_day_high":
            return signal.prev_high if signal.prev_high > 0 else None
        elif mode == "prev_day_low":
            return signal.prev_low if signal.prev_low > 0 else None
        elif mode == "session_high":
            return signal.session_high if signal.session_high > 0 else None
        elif mode == "session_low":
            return signal.session_low if signal.session_low > 0 else None
        elif mode == "opposing":
            # Long: target previous day high; Short: target previous day low
            if signal.direction == "long":
                return signal.prev_high if signal.prev_high > entry_price else signal.session_high or None
            else:
                return signal.prev_low if signal.prev_low < entry_price else signal.session_low or None
        return None

    # ------------------------------------------------------------------
    def _simulate(self, signal: Signal, entry: float, stop: float,
                  target: Optional[float],
                  candles: pd.DataFrame):
        direction = signal.direction
        risk      = abs(entry - stop)

        # Route to appropriate simulator
        if config.USE_PARTIALS and risk > 0:
            return self._simulate_partials(direction, entry, stop, target, risk, candles)
        if config.USE_ATR_TRAIL:
            return self._simulate_atr_trail(direction, entry, stop, target, candles)
        if config.TP_MODE == "trailing":
            return self._simulate_trailing(direction, entry, stop, target, candles)
        return self._simulate_fixed(direction, entry, stop, target, risk, candles)

    # ------------------------------------------------------------------
    def _simulate_fixed(self, direction: str, entry: float, stop: float,
                        target: Optional[float], risk: float,
                        candles: pd.DataFrame):
        highs  = candles["High"].values
        lows   = candles["Low"].values
        closes = candles["Close"].values
        times  = candles["time"].values
        n      = len(highs)

        be_stop = stop  # break-even stop, updated when triggered
        be_triggered = False

        for i in range(n):
            h, l, c, t = highs[i], lows[i], closes[i], times[i]

            # Break-even logic
            if config.USE_BREAKEVEN and not be_triggered and risk > 0:
                mfe_now = (h - entry if direction == "long" else entry - l)
                if mfe_now >= config.BREAKEVEN_TRIGGER_R * risk:
                    be_stop = entry
                    be_triggered = True

            current_stop = be_stop

            if direction == "long":
                if target is not None and h >= target:
                    mae = min(0.0, l - entry)
                    mfe = max(0.0, h - entry)
                    return target, t, "target", mae, mfe, i + 1
                if l <= current_stop:
                    mae = min(0.0, l - entry)
                    mfe = max(0.0, h - entry)
                    return current_stop, t, "stop", mae, mfe, i + 1
            else:
                if target is not None and l <= target:
                    mae = min(0.0, entry - h)
                    mfe = max(0.0, entry - l)
                    return target, t, "target", mae, mfe, i + 1
                if h >= current_stop:
                    mae = min(0.0, entry - h)
                    mfe = max(0.0, entry - l)
                    return current_stop, t, "stop", mae, mfe, i + 1

        # Session end
        adverse   = (lows  - entry if direction == "long" else entry - highs)
        favorable = (highs - entry if direction == "long" else entry - lows)
        return closes[-1], times[-1], "session_end", float(adverse.min()), float(favorable.max()), n

    # ------------------------------------------------------------------
    def _simulate_partials(self, direction: str, entry: float, stop: float,
                           target: Optional[float], risk: float,
                           candles: pd.DataFrame):
        """
        Two-leg partial exit:
          Leg 1: PARTIAL_1_PCT at PARTIAL_1_R
          Leg 2: remainder at PARTIAL_2_R (or trailing)
        Returns blended exit price.
        """
        highs  = candles["High"].values
        lows   = candles["Low"].values
        closes = candles["Close"].values
        times  = candles["time"].values
        n      = len(highs)

        p1_target = (entry + config.PARTIAL_1_R * risk if direction == "long"
                     else entry - config.PARTIAL_1_R * risk)
        p2_target = (entry + config.PARTIAL_2_R * risk if direction == "long"
                     else entry - config.PARTIAL_2_R * risk)

        p1_pct  = config.PARTIAL_1_PCT
        p2_pct  = 1.0 - p1_pct
        p1_done = False
        p1_exit = None
        trail_stop = stop
        be_triggered = False

        for i in range(n):
            h, l, c, t = highs[i], lows[i], closes[i], times[i]

            # Break-even
            if config.USE_BREAKEVEN and not be_triggered and risk > 0:
                mfe_now = (h - entry if direction == "long" else entry - l)
                if mfe_now >= config.BREAKEVEN_TRIGGER_R * risk:
                    trail_stop = entry
                    be_triggered = True

            # ATR trail on remainder
            if config.PARTIAL_2_TRAIL and p1_done:
                atr_val = candles.iloc[i].get("atr", 0) or 0
                trail_dist = config.ATR_TRAIL_MULT * atr_val if atr_val > 0 else 0
                if direction == "long":
                    trail_stop = max(trail_stop, c - trail_dist)
                else:
                    trail_stop = min(trail_stop, c + trail_dist)

            # Stop hit
            if direction == "long":
                if not p1_done and h >= p1_target:
                    p1_done = True
                    p1_exit = p1_target
                    trail_stop = entry  # move to BE after partial
                if p1_done and h >= p2_target:
                    blended = p1_pct * p1_exit + p2_pct * p2_target
                    mae = min(0.0, l - entry)
                    mfe = max(0.0, h - entry)
                    return blended, t, "target", mae, mfe, i + 1
                if l <= trail_stop:
                    stop_exit = trail_stop
                    if not p1_done:
                        blended = stop_exit
                    else:
                        blended = p1_pct * p1_exit + p2_pct * stop_exit
                    mae = min(0.0, l - entry)
                    mfe = max(0.0, h - entry)
                    return blended, t, "stop", mae, mfe, i + 1
            else:
                if not p1_done and l <= p1_target:
                    p1_done = True
                    p1_exit = p1_target
                    trail_stop = entry
                if p1_done and l <= p2_target:
                    blended = p1_pct * p1_exit + p2_pct * p2_target
                    mae = min(0.0, entry - h)
                    mfe = max(0.0, entry - l)
                    return blended, t, "target", mae, mfe, i + 1
                if h >= trail_stop:
                    stop_exit = trail_stop
                    blended = (stop_exit if not p1_done
                               else p1_pct * p1_exit + p2_pct * stop_exit)
                    mae = min(0.0, entry - h)
                    mfe = max(0.0, entry - l)
                    return blended, t, "stop", mae, mfe, i + 1

        blended = (p1_pct * p1_exit + p2_pct * closes[-1]
                   if p1_done else closes[-1])
        adverse   = (lows  - entry if direction == "long" else entry - highs)
        favorable = (highs - entry if direction == "long" else entry - lows)
        return blended, times[-1], "session_end", float(adverse.min()), float(favorable.max()), n

    # ------------------------------------------------------------------
    def _simulate_atr_trail(self, direction: str, entry: float, stop: float,
                             target: Optional[float], candles: pd.DataFrame):
        highs  = candles["High"].values
        lows   = candles["Low"].values
        closes = candles["Close"].values
        times  = candles["time"].values
        n      = len(highs)
        trail  = stop
        mae = mfe = 0.0

        for i in range(n):
            h, l, c, t = highs[i], lows[i], closes[i], times[i]
            atr_val = candles.iloc[i].get("atr", 0) or 0
            trail_dist = config.ATR_TRAIL_MULT * atr_val if atr_val > 0 else abs(entry - stop)

            if direction == "long":
                mae = min(mae, l - entry); mfe = max(mfe, h - entry)
                if target is not None and h >= target:
                    return target, t, "target", mae, mfe, i + 1
                if l <= trail:
                    return trail, t, "stop", mae, mfe, i + 1
                trail = max(trail, c - trail_dist)
            else:
                mae = min(mae, entry - h); mfe = max(mfe, entry - l)
                if target is not None and l <= target:
                    return target, t, "target", mae, mfe, i + 1
                if h >= trail:
                    return trail, t, "stop", mae, mfe, i + 1
                trail = min(trail, c + trail_dist)

        return closes[-1], times[-1], "session_end", mae, mfe, n

    # ------------------------------------------------------------------
    def _simulate_trailing(self, direction: str, entry: float, stop: float,
                            target: Optional[float], candles: pd.DataFrame):
        highs  = candles["High"].values
        lows   = candles["Low"].values
        closes = candles["Close"].values
        times  = candles["time"].values
        n      = len(highs)
        trail  = stop
        trail_dist = config.TRAILING_TICKS * self.tick
        mae = mfe = 0.0

        for i in range(n):
            h, l, c, t = highs[i], lows[i], closes[i], times[i]
            if direction == "long":
                mae = min(mae, l - entry); mfe = max(mfe, h - entry)
                if target is not None and h >= target:
                    return target, t, "target", mae, mfe, i + 1
                if l <= trail:
                    return trail, t, "stop", mae, mfe, i + 1
                trail = max(trail, c - trail_dist)
            else:
                mae = min(mae, entry - h); mfe = max(mfe, entry - l)
                if target is not None and l <= target:
                    return target, t, "target", mae, mfe, i + 1
                if h >= trail:
                    return trail, t, "stop", mae, mfe, i + 1
                trail = min(trail, c + trail_dist)

        return closes[-1], times[-1], "session_end", mae, mfe, n
