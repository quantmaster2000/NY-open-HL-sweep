import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from execution_engine import Trade
import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bucket_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute PF, expectancy, win rate, avg R, trade count per bucket."""
    rows = []
    for bucket, grp in df.groupby(group_col, observed=True):
        pnl  = grp["pnl_dollars"]
        wins = pnl[pnl > 0]
        loss = pnl[pnl < 0]
        n    = len(grp)
        wr   = len(wins) / n if n > 0 else 0
        gp   = wins.sum()
        gl   = loss.sum()
        pf   = abs(gp / gl) if gl != 0 else np.inf
        exp  = pnl.mean()
        avg_r = grp["r_multiple"].mean() if "r_multiple" in grp.columns else np.nan
        rows.append({
            "bucket":         bucket,
            "trades":         n,
            "win_rate":       round(wr * 100, 1),
            "profit_factor":  round(pf, 2),
            "expectancy":     round(exp, 2),
            "avg_r":          round(avg_r, 2),
            "net_pnl":        round(pnl.sum(), 2),
        })
    return pd.DataFrame(rows).sort_values("bucket")


def _significance_flag(df: pd.DataFrame, group_col: str,
                        min_trades: int = 20) -> pd.DataFrame:
    """
    One-way ANOVA across buckets; flag buckets whose mean PnL is
    statistically different from the overall mean (t-test, p < 0.05).
    """
    overall_mean = df["pnl_dollars"].mean()
    rows = []
    for bucket, grp in df.groupby(group_col, observed=True):
        if len(grp) < min_trades:
            rows.append({"bucket": bucket, "significant": False, "p_value": np.nan})
            continue
        t, p = scipy_stats.ttest_1samp(grp["pnl_dollars"], overall_mean)
        rows.append({"bucket": bucket, "significant": p < 0.05, "p_value": round(p, 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class PerformanceAnalyzer:
    def __init__(self, trades: list[Trade], instrument: str):
        self.instrument = instrument
        self.trades     = trades
        self.df         = self._to_df(trades)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        df = self.df
        if df.empty:
            return {"instrument": self.instrument, "trades": 0}

        pnl    = df["pnl_dollars"]
        wins   = df[pnl > 0]
        losses = df[pnl < 0]

        equity = df["equity_after"]
        peak   = equity.cummax()
        dd     = equity - peak

        gp = wins["pnl_dollars"].sum()
        gl = losses["pnl_dollars"].sum()
        pf = abs(gp / gl) if gl != 0 else np.inf

        avg_win  = wins["pnl_dollars"].mean()  if len(wins)   else 0
        avg_loss = losses["pnl_dollars"].mean() if len(losses) else 0
        wr       = len(wins) / len(df)
        exp      = wr * avg_win + (1 - wr) * avg_loss

        sharpe  = self._sharpe(pnl)
        sortino = self._sortino(pnl)
        calmar  = self._calmar(pnl, dd)
        ulcer   = self._ulcer(equity)
        max_dd  = dd.min()
        avg_dd  = dd[dd < 0].mean() if (dd < 0).any() else 0
        recovery = pnl.sum() / abs(max_dd) if max_dd != 0 else np.inf

        win_streak, loss_streak = self._streaks(pnl)
        ror = self._risk_of_ruin(wr, avg_win, avg_loss)

        monthly = self._monthly_returns(df)
        yearly  = monthly.groupby(monthly.index.year).sum()

        return {
            "instrument":          self.instrument,
            "trades":              len(df),
            "wins":                len(wins),
            "losses":              len(losses),
            "win_rate":            wr,
            "net_profit":          pnl.sum(),
            "gross_profit":        gp,
            "gross_loss":          gl,
            "profit_factor":       pf,
            "expectancy":          exp,
            "avg_winner":          avg_win,
            "avg_loser":           avg_loss,
            "largest_winner":      wins["pnl_dollars"].max()  if len(wins)   else 0,
            "largest_loser":       losses["pnl_dollars"].min() if len(losses) else 0,
            "avg_hold_bars":       df["hold_bars"].mean(),
            "avg_mae":             df["mae_points"].mean(),
            "avg_mfe":             df["mfe_points"].mean(),
            "avg_r_multiple":      df["r_multiple"].mean(),
            "sharpe":              sharpe,
            "sortino":             sortino,
            "calmar":              calmar,
            "recovery_factor":     recovery,
            "ulcer_index":         ulcer,
            "max_drawdown":        max_dd,
            "avg_drawdown":        avg_dd,
            "longest_win_streak":  win_streak,
            "longest_loss_streak": loss_streak,
            "return_over_maxdd":   pnl.sum() / abs(max_dd) if max_dd != 0 else np.inf,
            "risk_of_ruin":        ror,
            "monthly_returns":     monthly,
            "yearly_returns":      yearly,
            "equity_curve":        equity,
            "drawdown_curve":      dd,
        }

    # ------------------------------------------------------------------
    # Edge decomposition — all bucketed analyses
    # ------------------------------------------------------------------
    def edge_decomposition(self) -> dict[str, pd.DataFrame]:
        df = self.df
        if df.empty:
            return {}

        results = {}

        # -- Day of week --
        df["dow"] = pd.to_datetime(df["entry_time"]).dt.day_name()
        results["by_dow"] = _bucket_stats(df, "dow")

        # -- Month --
        df["month"] = pd.to_datetime(df["entry_time"]).dt.month
        results["by_month"] = _bucket_stats(df, "month")

        # -- Year --
        df["year"] = pd.to_datetime(df["entry_time"]).dt.year
        results["by_year"] = _bucket_stats(df, "year")

        # -- Direction --
        results["by_direction"] = _bucket_stats(df, "direction")

        # -- Confirm type --
        if "confirm_type" in df.columns:
            results["by_confirm_type"] = _bucket_stats(df, "confirm_type")

        # -- Entry minute bucket --
        if "entry_minute" in df.columns:
            df["entry_min_bucket"] = pd.cut(
                df["entry_minute"],
                bins=[0, 5, 10, 20, 60],
                labels=["0-5m", "5-10m", "10-20m", "20m+"]
            )
            results["by_entry_time"] = _bucket_stats(df, "entry_min_bucket")

        # -- OR size quartile --
        if "or_size" in df.columns and df["or_size"].notna().any():
            df["or_quartile"] = pd.qcut(df["or_size"], q=4,
                                         labels=["Q1_small","Q2","Q3","Q4_large"],
                                         duplicates="drop")
            results["by_or_size"] = _bucket_stats(df, "or_quartile")

        # -- Sweep depth quartile --
        if "sweep_depth" in df.columns and df["sweep_depth"].notna().any():
            df["sweep_depth_q"] = pd.qcut(df["sweep_depth"], q=4,
                                            labels=["Q1_shallow","Q2","Q3","Q4_deep"],
                                            duplicates="drop")
            results["by_sweep_depth"] = _bucket_stats(df, "sweep_depth_q")

        # -- Displacement body pct quartile --
        if "displacement_body_pct" in df.columns:
            df["disp_q"] = pd.qcut(df["displacement_body_pct"], q=4,
                                    labels=["Q1_weak","Q2","Q3","Q4_strong"],
                                    duplicates="drop")
            results["by_displacement"] = _bucket_stats(df, "disp_q")

        # -- FVG size quartile --
        if "fvg_size" in df.columns and (df["fvg_size"] > 0).any():
            fvg_df = df[df["fvg_size"] > 0].copy()
            fvg_df["fvg_q"] = pd.qcut(fvg_df["fvg_size"], q=4,
                                        labels=["Q1_small","Q2","Q3","Q4_large"],
                                        duplicates="drop")
            results["by_fvg_size"] = _bucket_stats(fvg_df, "fvg_q")

        # -- Hold time --
        if "hold_bars" in df.columns:
            df["hold_bucket"] = pd.cut(
                df["hold_bars"],
                bins=[0, 5, 15, 30, 9999],
                labels=["1-5","6-15","16-30","30+"]
            )
            results["by_hold_time"] = _bucket_stats(df, "hold_bucket")

        # -- OR/ATR ratio (volatility regime proxy) --
        if "or_atr_ratio" in df.columns and df["or_atr_ratio"].notna().any():
            df["vol_regime"] = pd.qcut(df["or_atr_ratio"], q=3,
                                        labels=["low_vol","mid_vol","high_vol"],
                                        duplicates="drop")
            results["by_vol_regime"] = _bucket_stats(df, "vol_regime")

        # -- Trend aligned --
        if "trend_aligned" in df.columns:
            results["by_trend_aligned"] = _bucket_stats(df, "trend_aligned")

        # -- Statistical significance flags --
        results["significance_dow"]  = _significance_flag(df, "dow")
        results["significance_year"] = _significance_flag(df, "year")

        return results

    # ------------------------------------------------------------------
    def trade_log(self) -> pd.DataFrame:
        return self.df.copy()

    def rolling_metrics(self, window: int = 50) -> pd.DataFrame:
        df = self.df.copy()
        if df.empty:
            return pd.DataFrame()
        pnl = df["pnl_dollars"]
        roll = pd.DataFrame(index=df.index)
        roll["rolling_sharpe"]   = (pnl.rolling(window).mean() /
                                     pnl.rolling(window).std() * np.sqrt(252))
        roll["rolling_win_rate"] = (pnl > 0).astype(float).rolling(window).mean()
        gw = pnl.clip(lower=0).rolling(window).sum()
        gl = pnl.clip(upper=0).rolling(window).sum().abs()
        roll["rolling_pf"]  = gw / gl.replace(0, np.nan)
        peak = df["equity_after"].rolling(window, min_periods=1).max()
        roll["rolling_dd"]  = df["equity_after"] - peak
        roll["entry_time"]  = df["entry_time"]
        return roll

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _to_df(self, trades: list[Trade]) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame()
        df = pd.DataFrame([t.__dict__ for t in trades])
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"]  = pd.to_datetime(df["exit_time"])
        return df.sort_values("entry_time").reset_index(drop=True)

    def _sharpe(self, pnl: pd.Series, periods: int = 252) -> float:
        return (pnl.mean() / pnl.std() * np.sqrt(periods)) if pnl.std() > 0 else 0

    def _sortino(self, pnl: pd.Series, periods: int = 252) -> float:
        ds = pnl[pnl < 0].std()
        return (pnl.mean() / ds * np.sqrt(periods)) if ds > 0 else 0

    def _calmar(self, pnl: pd.Series, dd: pd.Series) -> float:
        mdd = dd.min()
        return pnl.sum() / abs(mdd) if mdd != 0 else 0

    def _ulcer(self, equity: pd.Series) -> float:
        peak   = equity.cummax()
        pct_dd = ((equity - peak) / peak) * 100
        return float(np.sqrt((pct_dd ** 2).mean()))

    def _streaks(self, pnl: pd.Series):
        ws = ls = cw = cl = 0
        for p in pnl:
            if p > 0:   cw += 1; cl = 0
            elif p < 0: cl += 1; cw = 0
            else:       cw = cl = 0
            ws = max(ws, cw); ls = max(ls, cl)
        return ws, ls

    def _monthly_returns(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        df["month"] = df["entry_time"].dt.to_period("M")
        monthly = df.groupby("month")["pnl_dollars"].sum()
        monthly.index = monthly.index.to_timestamp()
        return monthly

    def _risk_of_ruin(self, wr: float, avg_win: float,
                      avg_loss: float, ruin_pct: float = 0.5) -> float:
        if avg_win <= 0 or avg_loss >= 0 or wr <= 0:
            return 1.0
        edge = wr * avg_win + (1 - wr) * avg_loss
        if edge <= 0:
            return 1.0
        kelly = edge / avg_win
        if kelly <= 0:
            return 1.0
        return max(0.0, min(1.0, ((1 - kelly) / (1 + kelly)) ** (1 / ruin_pct)))
