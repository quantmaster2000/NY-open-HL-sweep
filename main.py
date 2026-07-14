import os
import pandas as pd
import numpy as np

import config
from market_data import MarketData
from backtester import Backtester
from performance_analyzer import PerformanceAnalyzer
from monte_carlo import MonteCarlo
from optimizer import Optimizer
from report_generator import ReportGenerator


# ---------------------------------------------------------------------------
def load_news_dates() -> set:
    if config.NEWS_FILTER_CSV and os.path.exists(config.NEWS_FILTER_CSV):
        df = pd.read_csv(config.NEWS_FILTER_CSV)
        return set(df["date"].astype(str).tolist())
    return set()


def walk_forward_split(df: pd.DataFrame):
    dates = sorted(df["date"].unique())
    n     = len(dates)
    t_end = int(n * config.TRAIN_PCT)
    v_end = int(n * (config.TRAIN_PCT + config.VALIDATE_PCT))
    train_dates = set(dates[:t_end])
    val_dates   = set(dates[t_end:v_end])
    oos_dates   = set(dates[v_end:])
    return (
        df[df["date"].isin(train_dates)].copy(),
        df[df["date"].isin(val_dates)].copy(),
        df[df["date"].isin(oos_dates)].copy(),
    )


# ---------------------------------------------------------------------------
def run_instrument(instrument: str, df: pd.DataFrame,
                   news_dates: set, reporter: ReportGenerator):

    print(f"\n{'='*55}")
    print(f"  {instrument}")
    print(f"{'='*55}")

    train_df, val_df, oos_df = walk_forward_split(df)

    # ---- Full backtest ------------------------------------------------
    print("  Running full backtest...")
    bt     = Backtester(instrument, df, news_dates)
    trades = bt.run()

    if not trades:
        print("  No trades generated.")
        return

    analyzer = PerformanceAnalyzer(trades, instrument)
    summary  = analyzer.summary()
    rolling  = analyzer.rolling_metrics()
    summary["_pnl_series"] = analyzer.df["pnl_dollars"]

    reporter.print_summary(summary)

    # ---- Edge decomposition ------------------------------------------
    print("\n  Edge Decomposition:")
    edge = analyzer.edge_decomposition()
    reporter.print_edge_summary(edge)
    reporter.save_edge_decomposition(edge, instrument)

    # ---- Walk-forward ------------------------------------------------
    print("\n  Walk-Forward Results:")
    for label, split_df in [("Train", train_df), ("Validate", val_df), ("OOS", oos_df)]:
        if split_df.empty:
            continue
        wf_trades = Backtester(instrument, split_df, news_dates).run()
        if not wf_trades:
            reporter.print_wf_summary(label, {"trades": 0, "net_profit": 0,
                                               "win_rate": 0, "profit_factor": 0,
                                               "max_drawdown": 0})
            continue
        reporter.print_wf_summary(label, PerformanceAnalyzer(wf_trades, instrument).summary())

    # ---- Optimisation (train + val, walk-forward aware) ---------------
    print("\n  Running walk-forward optimisation (train->val)...")
    opt      = Optimizer(instrument, df, train_df, val_df, news_dates)
    grid_df  = opt.run_grid()
    sensitivity_df = None

    if not grid_df.empty:
        best = grid_df.iloc[0]
        print(f"  Best generalising params: stop={best['stop']} tp={best['tp']} "
              f"val_PF={best['val_pf']:.2f} val_calmar={best['val_calmar']:.2f} "
              f"PF_gap={best['pf_gap']:.2f}")
        grid_path = os.path.join(config.OUTPUT_DIR, f"{instrument}_opt_grid.csv")
        grid_df.to_csv(grid_path, index=False)
        print(f"  Opt grid   -> {grid_path}")
        sensitivity_df = opt.sensitivity(best["stop"], best["tp"])
    else:
        print("  No generalising parameter sets found.")

    # ---- Monte Carlo --------------------------------------------------
    print(f"\n  Monte Carlo ({config.MC_SIMULATIONS:,} simulations)...")
    mc = MonteCarlo(trades)
    mc.print_results(mc.run_all())

    # ---- Trade log & charts ------------------------------------------
    reporter.save_trade_log(analyzer.trade_log(), instrument)
    reporter.plot_all(summary, rolling, instrument, sensitivity_df)


# ---------------------------------------------------------------------------
def main():
    data_dir = os.path.join(
        os.path.dirname(__file__),
        "TimeAggregated-20260713T165927Z-2-002", "TimeAggregated"
    )

    print("Loading market data...")
    md     = MarketData(data_dir)
    frames = md.load_all()

    news_dates = load_news_dates()
    reporter   = ReportGenerator()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    for instrument in ["MNQ", "MES", "M2K", "MYM"]:
        if instrument not in frames:
            print(f"  {instrument}: data not found, skipping.")
            continue
        run_instrument(instrument, frames[instrument], news_dates, reporter)

    print(f"\nDone. Outputs saved to: {config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
