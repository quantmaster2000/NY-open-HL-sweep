import pandas as pd
import numpy as np
from typing import Optional
import config
from strategy import Strategy, Signal
from risk_manager import RiskManager
from execution_engine import ExecutionEngine, Trade


class Backtester:
    def __init__(self, instrument: str, df: pd.DataFrame,
                 news_dates: set = None,
                 override_cfg: dict = None):
        self.instrument = instrument
        self.df         = df.copy()
        self.strategy   = Strategy(news_dates)
        self.risk_mgr   = RiskManager(config.STARTING_EQUITY)
        self.engine     = ExecutionEngine(instrument)
        self.override   = override_cfg or {}

    def run(self) -> list[Trade]:
        # Pre-compute rolling percentile caches (uses only past data)
        self.strategy.precompute(self.df, self.instrument)

        trades  = []
        grouped = self.df.groupby("date")

        for date, day_df in grouped:
            setup = self.strategy.build_day_setup(day_df, self.instrument)
            if not setup.valid:
                continue

            tick = config.INSTRUMENT_SPECS[self.instrument]["tick_size"]
            signal = self.strategy.find_signal(day_df, setup, self.instrument, tick)
            if signal is None:
                continue

            stop_ref  = self._estimate_stop(signal)
            contracts = self.risk_mgr.position_size(self.instrument, stop_ref)

            equity_before = self.risk_mgr.equity
            trade = self.engine.execute(signal, day_df, contracts, equity_before)

            if trade is None:
                continue

            self.risk_mgr.update_equity(trade.pnl_dollars)
            trades.append(trade)

        return trades

    def _estimate_stop(self, signal: Signal) -> float:
        tick = config.INSTRUMENT_SPECS[self.instrument]["tick_size"]
        if config.STOP_MODE in ("sweep", "swing"):
            return abs(signal.signal_close - signal.sweep_extreme) + tick
        elif config.STOP_MODE == "fixed_points":
            return config.STOP_FIXED_POINTS
        elif config.STOP_MODE == "atr":
            return config.STOP_ATR_MULT * signal.atr
        return config.STOP_FIXED_POINTS
