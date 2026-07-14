import pandas as pd
import numpy as np
import glob
import os

# NY open = 13:30 UTC (9:30 ET)
NY_OPEN_UTC = "13:30"

def load_csv(path):
    df = pd.read_csv(path, sep=";", usecols=["Time left", "Open", "High", "Low", "Close"])
    df.rename(columns={"Time left": "time"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df

def run_backtest(df, instrument):
    df = df.copy()
    df["date"] = df["time"].dt.date
    df["hhmm"] = df["time"].dt.strftime("%H:%M")

    # Get the NY open candle for each day
    open_candles = df[df["hhmm"] == NY_OPEN_UTC][["date", "High", "Low"]].rename(
        columns={"High": "first_high", "Low": "first_low"}
    )

    # Merge first candle levels onto all subsequent candles
    merged = df.merge(open_candles, on="date", how="inner")

    # Only candles strictly after the open candle
    open_times = df[df["hhmm"] == NY_OPEN_UTC].set_index("date")["time"]
    merged["open_time"] = merged["date"].map(open_times)
    merged = merged[merged["time"] > merged["open_time"]]

    # Identify sweep signals
    short_mask = (merged["High"] > merged["first_high"]) & (merged["Close"] < merged["first_high"])
    long_mask  = (merged["Low"]  < merged["first_low"])  & (merged["Close"] > merged["first_low"])

    merged["signal"] = None
    merged.loc[short_mask, "signal"] = "short"
    merged.loc[long_mask,  "signal"] = "long"

    # Keep only the first signal per day
    signals = (
        merged[merged["signal"].notna()]
        .sort_values("time")
        .groupby("date", as_index=False)
        .first()
    )

    if signals.empty:
        return pd.DataFrame()

    signals["pnl_points"] = signals.apply(
        lambda r: r["Open"] - r["Close"] if r["signal"] == "short" else r["Close"] - r["Open"],
        axis=1
    )

    signals["instrument"] = instrument
    return signals[["instrument", "date", "time", "signal", "Open", "Close",
                    "pnl_points", "first_high", "first_low"]].rename(
        columns={"signal": "direction", "Open": "entry", "Close": "exit"}
    )

def max_drawdown(pnl_series):
    equity = pnl_series.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    return dd.min()


def monte_carlo(pnl_series, n_sims=10000, n_trades=None):
    trades = pnl_series.values
    if n_trades is None:
        n_trades = len(trades)
    rng = np.random.default_rng(42)
    # Shuffle trade order n_sims times, compute final equity and max DD each time
    sims = rng.choice(trades, size=(n_sims, n_trades), replace=True)
    equity_curves = np.cumsum(sims, axis=1)
    final_pnl = equity_curves[:, -1]
    peaks = np.maximum.accumulate(equity_curves, axis=1)
    drawdowns = (equity_curves - peaks).min(axis=1)
    return final_pnl, drawdowns


def print_summary(trades_df, instrument):
    if trades_df.empty:
        print(f"\n{instrument}: No trades found.")
        return

    t = trades_df[trades_df["instrument"] == instrument]
    if t.empty:
        print(f"\n{instrument}: No trades found.")
        return

    total = len(t)
    wins = (t["pnl_points"] > 0).sum()
    losses = (t["pnl_points"] < 0).sum()
    win_rate = wins / total * 100
    total_pnl = t["pnl_points"].sum()
    avg_pnl = t["pnl_points"].mean()
    avg_win = t[t["pnl_points"] > 0]["pnl_points"].mean() if wins > 0 else 0
    avg_loss = t[t["pnl_points"] < 0]["pnl_points"].mean() if losses > 0 else 0
    actual_mdd = max_drawdown(t["pnl_points"])

    # Monte Carlo
    final_pnl, mc_dds = monte_carlo(t["pnl_points"])
    mc_mdd_median = np.median(mc_dds)
    mc_mdd_95     = np.percentile(mc_dds, 5)   # worst 5% (most negative)
    mc_pnl_median = np.median(final_pnl)
    mc_pnl_5      = np.percentile(final_pnl, 5)
    mc_pnl_95     = np.percentile(final_pnl, 95)
    prob_profit   = (final_pnl > 0).mean() * 100

    print(f"\n{'='*45}")
    print(f"  {instrument}")
    print(f"{'='*45}")
    print(f"  Trades        : {total}")
    print(f"  Wins          : {wins}  |  Losses: {losses}")
    print(f"  Win Rate      : {win_rate:.1f}%")
    print(f"  Total PnL     : {total_pnl:.2f} pts")
    print(f"  Avg PnL       : {avg_pnl:.2f} pts")
    print(f"  Avg Win       : {avg_win:.2f} pts")
    print(f"  Avg Loss      : {avg_loss:.2f} pts")
    print(f"  Long trades   : {(t['direction']=='long').sum()}")
    print(f"  Short trades  : {(t['direction']=='short').sum()}")
    print(f"  --- Drawdown ---")
    print(f"  Actual Max DD : {actual_mdd:.2f} pts")
    print(f"  --- Monte Carlo (10,000 simulations) ---")
    print(f"  Median Max DD : {mc_mdd_median:.2f} pts")
    print(f"  Worst 5% DD   : {mc_mdd_95:.2f} pts")
    print(f"  Median PnL    : {mc_pnl_median:.2f} pts")
    print(f"  PnL 5th pct   : {mc_pnl_5:.2f} pts")
    print(f"  PnL 95th pct  : {mc_pnl_95:.2f} pts")
    print(f"  Prob Profit   : {prob_profit:.1f}%")

def main():
    data_dir = os.path.join(os.path.dirname(__file__),
                            "TimeAggregated-20260713T165927Z-2-002", "TimeAggregated")
    files = glob.glob(os.path.join(data_dir, "*.csv"))

    # Exclude ZB (bond, different session logic)
    files = [f for f in files if "ZB" not in os.path.basename(f)]

    all_trades = []

    for path in files:
        name = os.path.basename(path)
        instrument = name.split(" ")[0]
        print(f"Processing {instrument}...")
        df = load_csv(path)
        trades = run_backtest(df, instrument)
        all_trades.append(trades)

    combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    for instrument in ["MNQ", "MES", "M2K", "MYM"]:
        print_summary(combined, instrument)

    if not combined.empty:
        out_path = os.path.join(os.path.dirname(__file__), "ny_sweep_trades.csv")
        combined.to_csv(out_path, index=False)
        print(f"\nAll trades saved to: {out_path}")

if __name__ == "__main__":
    main()
