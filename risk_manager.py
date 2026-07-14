import math
import config
from config import INSTRUMENT_SPECS


class RiskManager:
    def __init__(self, starting_equity: float = None):
        self.equity = starting_equity or config.STARTING_EQUITY
        self.peak_equity = self.equity

    def position_size(self, instrument: str, stop_points: float) -> int:
        """Return number of contracts, floored to MAX_CONTRACTS."""
        spec = INSTRUMENT_SPECS[instrument]
        tick = spec["tick_size"]
        tick_val = spec["tick_value"]

        if stop_points <= 0:
            return 1

        stop_ticks = stop_points / tick
        stop_dollars = stop_ticks * tick_val

        if config.RISK_MODE == "fixed_dollar":
            risk_dollars = config.RISK_FIXED_DOLLAR
        else:
            risk_dollars = self.equity * config.RISK_PCT_EQUITY

        contracts = math.floor(risk_dollars / stop_dollars)
        return max(1, min(contracts, config.MAX_CONTRACTS))

    def update_equity(self, pnl_dollars: float):
        self.equity += pnl_dollars
        self.peak_equity = max(self.peak_equity, self.equity)

    def current_drawdown(self) -> float:
        return self.equity - self.peak_equity
