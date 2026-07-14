import itertools
import pandas as pd
import numpy as np
import config
from backtester import Backtester


class Optimizer:
    def __init__(self, instrument: str, df: pd.DataFrame,
                 train_df: pd.DataFrame, val_df: pd.DataFrame,
                 news_dates: set = None):
        self.instrument = instrument
        self.df         = df
        self.train_df   = train_df
        self.val_df     = val_df
        self.news_dates = news_dates or set()

    # ------------------------------------------------------------------
    def run_grid(self) -> pd.DataFrame:
        """
        Grid search on train set only.
        Each candidate is then validated on val set.
        Only parameter sets that generalise (val PF >= 1.0) are returned.
        Sorted by val Calmar to avoid train-set overfitting.
        """
        stop_mode_orig = config.STOP_MODE
        tp_mode_orig   = config.TP_MODE
        config.STOP_MODE = "fixed_points"
        config.TP_MODE   = "fixed_points"

        results = []
        for stop, tp in itertools.product(
            config.OPT_STOP_FIXED_POINTS,
            config.OPT_TP_FIXED_POINTS,
        ):
            if tp <= stop:
                continue

            config.STOP_FIXED_POINTS = stop
            config.TP_FIXED_POINTS   = tp

            train_stats = self._run_stats(self.train_df)
            if train_stats is None:
                continue

            val_stats = self._run_stats(self.val_df)
            if val_stats is None:
                continue

            # Reject if val PF < 1.0 (does not generalise)
            if val_stats["profit_factor"] < 1.0:
                continue

            results.append({
                "stop":              stop,
                "tp":                tp,
                "train_trades":      train_stats["trades"],
                "train_pf":          train_stats["profit_factor"],
                "train_calmar":      train_stats["calmar"],
                "train_net":         train_stats["net_pnl"],
                "val_trades":        val_stats["trades"],
                "val_pf":            val_stats["profit_factor"],
                "val_calmar":        val_stats["calmar"],
                "val_net":           val_stats["net_pnl"],
                # Generalisation gap: smaller is better
                "pf_gap":            abs(train_stats["profit_factor"] -
                                         val_stats["profit_factor"]),
            })

        config.STOP_MODE = stop_mode_orig
        config.TP_MODE   = tp_mode_orig

        df = pd.DataFrame(results)
        if df.empty:
            return df
        return df.sort_values("val_calmar", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    def sensitivity(self, best_stop: float, best_tp: float,
                    pct: float = 0.20) -> pd.DataFrame:
        """Test ±pct variation around best params on full dataset."""
        stops = [best_stop * (1 + d) for d in [-pct, -pct/2, 0, pct/2, pct]]
        tps   = [best_tp   * (1 + d) for d in [-pct, -pct/2, 0, pct/2, pct]]

        config.STOP_MODE = "fixed_points"
        config.TP_MODE   = "fixed_points"
        results = []

        for stop, tp in itertools.product(stops, tps):
            config.STOP_FIXED_POINTS = stop
            config.TP_FIXED_POINTS   = tp
            stats = self._run_stats(self.df)
            if stats:
                results.append({
                    "stop":    round(stop, 2),
                    "tp":      round(tp, 2),
                    "net_pnl": stats["net_pnl"],
                })

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    def _run_stats(self, df: pd.DataFrame) -> dict | None:
        if df.empty:
            return None
        bt = Backtester(self.instrument, df, self.news_dates)
        trades = bt.run()
        if not trades:
            return None

        pnl    = [t.pnl_dollars for t in trades]
        wins   = [p for p in pnl if p > 0]
        losses = [p for p in pnl if p < 0]
        equity = np.cumsum(pnl)
        peak   = np.maximum.accumulate(equity)
        mdd    = float((equity - peak).min())
        gp     = sum(wins)
        gl     = abs(sum(losses))
        pf     = gp / gl if gl > 0 else np.inf
        net    = sum(pnl)
        calmar = net / abs(mdd) if mdd != 0 else np.inf

        return {
            "trades":        len(trades),
            "profit_factor": round(pf, 3),
            "calmar":        round(calmar, 3),
            "net_pnl":       round(net, 2),
        }
