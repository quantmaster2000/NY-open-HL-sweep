import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import config


class ReportGenerator:
    def __init__(self, output_dir: str = None):
        self.out = output_dir or config.OUTPUT_DIR
        os.makedirs(self.out, exist_ok=True)

    # ------------------------------------------------------------------
    def save_trade_log(self, df: pd.DataFrame, instrument: str):
        path = os.path.join(self.out, f"{instrument}_trade_log.csv")
        df.to_csv(path, index=False)
        print(f"  Trade log  -> {path}")

    def save_edge_decomposition(self, edge: dict, instrument: str):
        for name, tbl in edge.items():
            if isinstance(tbl, pd.DataFrame) and not tbl.empty:
                path = os.path.join(self.out, f"{instrument}_edge_{name}.csv")
                tbl.to_csv(path, index=False)
        print(f"  Edge CSVs  -> {self.out}/{instrument}_edge_*.csv")

    # ------------------------------------------------------------------
    def print_summary(self, s: dict):
        inst = s["instrument"]
        if s["trades"] == 0:
            print(f"\n{inst}: No trades.")
            return
        w = 46
        print(f"\n{'='*w}")
        print(f"  {inst}")
        print(f"{'='*w}")
        print(f"  Trades            : {s['trades']}")
        print(f"  Wins / Losses     : {s['wins']} / {s['losses']}")
        print(f"  Win Rate          : {s['win_rate']*100:.1f}%")
        print(f"  --- P&L (dollars) ---")
        print(f"  Net Profit        : ${s['net_profit']:>10.2f}")
        print(f"  Gross Profit      : ${s['gross_profit']:>10.2f}")
        print(f"  Gross Loss        : ${s['gross_loss']:>10.2f}")
        print(f"  Profit Factor     : {s['profit_factor']:>10.2f}")
        print(f"  Expectancy        : ${s['expectancy']:>10.2f}")
        print(f"  Avg Winner        : ${s['avg_winner']:>10.2f}")
        print(f"  Avg Loser         : ${s['avg_loser']:>10.2f}")
        print(f"  Largest Winner    : ${s['largest_winner']:>10.2f}")
        print(f"  Largest Loser     : ${s['largest_loser']:>10.2f}")
        print(f"  Avg R Multiple    : {s['avg_r_multiple']:>10.2f}")
        print(f"  --- Execution ---")
        print(f"  Avg Hold Bars     : {s['avg_hold_bars']:>10.1f}")
        print(f"  Avg MAE (pts)     : {s['avg_mae']:>10.2f}")
        print(f"  Avg MFE (pts)     : {s['avg_mfe']:>10.2f}")
        print(f"  --- Risk ---")
        print(f"  Max Drawdown      : ${s['max_drawdown']:>10.2f}")
        print(f"  Avg Drawdown      : ${s['avg_drawdown']:>10.2f}")
        print(f"  Sharpe Ratio      : {s['sharpe']:>10.2f}")
        print(f"  Sortino Ratio     : {s['sortino']:>10.2f}")
        print(f"  Calmar Ratio      : {s['calmar']:>10.2f}")
        print(f"  Recovery Factor   : {s['recovery_factor']:>10.2f}")
        print(f"  Ulcer Index       : {s['ulcer_index']:>10.2f}")
        print(f"  Return / Max DD   : {s['return_over_maxdd']:>10.2f}")
        print(f"  Risk of Ruin      : {s['risk_of_ruin']*100:>9.2f}%")
        print(f"  --- Streaks ---")
        print(f"  Longest Win Streak: {s['longest_win_streak']}")
        print(f"  Longest Loss Streak:{s['longest_loss_streak']}")
        print(f"\n  Monthly Returns ($):")
        for period, val in s["monthly_returns"].items():
            print(f"    {str(period)[:7]}  ${val:>8.2f}")

    def print_edge_summary(self, edge: dict):
        priority = ["by_direction", "by_confirm_type", "by_entry_time",
                    "by_dow", "by_year", "by_or_size", "by_sweep_depth",
                    "by_vol_regime"]
        for key in priority:
            if key in edge and not edge[key].empty:
                print(f"\n  [{key}]")
                print(edge[key].to_string(index=False))

    def print_wf_summary(self, label: str, s: dict):
        if s["trades"] == 0:
            print(f"  {label}: No trades")
            return
        print(f"  {label:20s} | trades={s['trades']:4d} | "
              f"net=${s['net_profit']:>8.2f} | "
              f"WR={s['win_rate']*100:4.1f}% | "
              f"PF={s['profit_factor']:5.2f} | "
              f"MaxDD=${s['max_drawdown']:>8.2f}")

    # ------------------------------------------------------------------
    def plot_all(self, summary: dict, rolling: pd.DataFrame,
                 instrument: str, sensitivity_df: pd.DataFrame = None):
        eq = summary["equity_curve"]
        dd = summary["drawdown_curve"]
        mr = summary["monthly_returns"]

        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        fig.suptitle(f"{instrument} - NY Open HL Sweep", fontsize=14, fontweight="bold")

        # 1. Equity curve
        ax = axes[0, 0]
        ax.plot(eq.values, color="steelblue")
        ax.set_title("Equity Curve ($)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        # 2. Drawdown
        ax = axes[0, 1]
        ax.fill_between(range(len(dd)), dd.values, 0, color="crimson", alpha=0.6)
        ax.set_title("Drawdown ($)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        # 3. Monthly returns heatmap
        ax = axes[1, 0]
        if not mr.empty:
            mr_df = mr.reset_index()
            mr_df.columns = ["date", "pnl"]
            mr_df["year"]  = mr_df["date"].dt.year
            mr_df["month"] = mr_df["date"].dt.month
            pivot = mr_df.pivot(index="year", columns="month", values="pnl").fillna(0)
            pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]
            sns.heatmap(pivot, ax=ax, annot=True, fmt=".0f", cmap="RdYlGn",
                        center=0, linewidths=0.5, cbar=False)
            ax.set_title("Monthly Returns ($)")
        else:
            ax.set_visible(False)

        # 4. PnL distribution
        ax = axes[1, 1]
        pnl_vals = summary.get("_pnl_series", pd.Series(dtype=float))
        if not pnl_vals.empty:
            ax.hist(pnl_vals, bins=40, color="steelblue", edgecolor="white")
            ax.axvline(0, color="red", linestyle="--")
            ax.set_title("Trade PnL Distribution ($)")

        # 5. Rolling Sharpe
        ax = axes[2, 0]
        if not rolling.empty and "rolling_sharpe" in rolling.columns:
            ax.plot(rolling["rolling_sharpe"].values, color="darkorange")
            ax.axhline(0, color="grey", linestyle="--")
            ax.set_title("Rolling Sharpe (50-trade window)")

        # 6. Rolling Win Rate
        ax = axes[2, 1]
        if not rolling.empty and "rolling_win_rate" in rolling.columns:
            ax.plot(rolling["rolling_win_rate"].values * 100, color="green")
            ax.axhline(50, color="grey", linestyle="--")
            ax.set_title("Rolling Win Rate % (50-trade window)")

        # 7. Rolling Profit Factor
        ax = axes[3, 0]
        if not rolling.empty and "rolling_pf" in rolling.columns:
            ax.plot(rolling["rolling_pf"].values, color="purple")
            ax.axhline(1, color="grey", linestyle="--")
            ax.set_title("Rolling Profit Factor (50-trade window)")

        # 8. Sensitivity heatmap
        ax = axes[3, 1]
        if sensitivity_df is not None and not sensitivity_df.empty:
            try:
                pivot = sensitivity_df.pivot(index="stop", columns="tp", values="net_pnl")
                sns.heatmap(pivot, ax=ax, annot=True, fmt=".0f", cmap="RdYlGn",
                            center=0, linewidths=0.5, cbar=False)
                ax.set_title("Sensitivity: Stop vs TP Net PnL ($)")
            except Exception:
                ax.set_visible(False)
        else:
            ax.set_visible(False)

        plt.tight_layout()
        path = os.path.join(self.out, f"{instrument}_report.png")
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"  Charts     -> {path}")
