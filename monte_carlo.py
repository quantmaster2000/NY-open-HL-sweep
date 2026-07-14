import numpy as np
import pandas as pd
from execution_engine import Trade
import config


class MonteCarlo:
    def __init__(self, trades: list[Trade], n_sims: int = None):
        self.pnl = np.array([t.pnl_dollars for t in trades])
        self.n_sims = n_sims or config.MC_SIMULATIONS
        self.rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    def run_all(self) -> dict:
        results = {}
        results["bootstrap_trades"]  = self._bootstrap_trades()
        results["bootstrap_days"]    = self._bootstrap_days()
        results["random_slippage"]   = self._random_slippage()
        results["random_skip"]       = self._random_skip()
        return results

    # ------------------------------------------------------------------
    def _bootstrap_trades(self) -> dict:
        """Resample individual trades with replacement."""
        n = len(self.pnl)
        sims = self.rng.choice(self.pnl, size=(self.n_sims, n), replace=True)
        return self._stats(sims, "Bootstrap Trades")

    def _bootstrap_days(self) -> dict:
        """Resample whole days (preserves intra-day correlation)."""
        # Group pnl by index as proxy for days (one trade per day)
        n = len(self.pnl)
        sims = self.rng.choice(self.pnl, size=(self.n_sims, n), replace=True)
        return self._stats(sims, "Bootstrap Days")

    def _random_slippage(self) -> dict:
        """Add random extra slippage ±1 tick per trade."""
        spec_vals = [0.50, 1.25, 0.50, 0.50]  # rough tick values
        avg_tick_val = np.mean(spec_vals)
        noise = self.rng.uniform(-avg_tick_val, avg_tick_val,
                                  size=(self.n_sims, len(self.pnl)))
        sims = self.pnl[np.newaxis, :] + noise
        return self._stats(sims, "Random Slippage")

    def _random_skip(self, skip_pct: float = 0.10) -> dict:
        """Randomly skip 10% of trades (missed signals)."""
        n = len(self.pnl)
        mask = self.rng.random(size=(self.n_sims, n)) > skip_pct
        sims = self.pnl[np.newaxis, :] * mask
        return self._stats(sims, "Random Skip 10%")

    # ------------------------------------------------------------------
    def _stats(self, sims: np.ndarray, label: str) -> dict:
        equity = np.cumsum(sims, axis=1)
        final  = equity[:, -1]
        peaks  = np.maximum.accumulate(equity, axis=1)
        dds    = (equity - peaks).min(axis=1)

        return {
            "label":           label,
            "median_pnl":      np.median(final),
            "pnl_5pct":        np.percentile(final, 5),
            "pnl_95pct":       np.percentile(final, 95),
            "median_max_dd":   np.median(dds),
            "worst_5pct_dd":   np.percentile(dds, 5),
            "prob_profit":     (final > 0).mean() * 100,
            "prob_beat_base":  (final >= np.sum(self.pnl)).mean() * 100,
        }

    # ------------------------------------------------------------------
    def print_results(self, results: dict):
        for key, r in results.items():
            print(f"\n  [{r['label']}]")
            print(f"    Median PnL      : ${r['median_pnl']:>10.2f}")
            print(f"    PnL  5th pct    : ${r['pnl_5pct']:>10.2f}")
            print(f"    PnL 95th pct    : ${r['pnl_95pct']:>10.2f}")
            print(f"    Median Max DD   : ${r['median_max_dd']:>10.2f}")
            print(f"    Worst 5%  DD    : ${r['worst_5pct_dd']:>10.2f}")
            print(f"    Prob Profit     : {r['prob_profit']:>9.1f}%")
            print(f"    Prob >= Actual  : {r['prob_beat_base']:>9.1f}%")
