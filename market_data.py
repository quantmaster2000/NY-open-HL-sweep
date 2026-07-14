import pandas as pd
import numpy as np
import glob
import os
from config import INSTRUMENT_SPECS, ATR_PERIOD


class MarketData:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._frames: dict[str, pd.DataFrame] = {}

    def load_all(self, exclude=("ZB",)) -> dict[str, pd.DataFrame]:
        pattern = os.path.join(self.data_dir, "*.csv")
        for path in sorted(glob.glob(pattern)):
            name = os.path.basename(path)
            instrument = name.split(" ")[0]
            if instrument in exclude or instrument not in INSTRUMENT_SPECS:
                continue
            print(f"  Loading {instrument}...")
            self._frames[instrument] = self._parse(path)
        return self._frames

    def get(self, instrument: str) -> pd.DataFrame:
        return self._frames[instrument]

    # ------------------------------------------------------------------
    def _parse(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(
            path, sep=";",
            usecols=["Time left", "Open", "High", "Low", "Close", "Volume"],
        )
        df.rename(columns={"Time left": "time"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)
        df["date"] = df["time"].dt.date
        df["hhmm"] = df["time"].dt.strftime("%H:%M")

        df = self._add_atr(df)
        df = self._add_prev_day_levels(df)
        df = self._add_ema(df, 50)
        df = self._add_ema(df, 200)
        df = self._add_vwap(df)
        df = self._add_realized_vol(df)
        return df

    # ------------------------------------------------------------------
    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        high, low, close = df["High"], df["Low"], df["Close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
        return df

    def _add_prev_day_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add previous day's OHLC and ATR (computed at day close) to every bar."""
        daily = df.groupby("date").agg(
            prev_open=("Open",  "first"),
            prev_high=("High",  "max"),
            prev_low=("Low",   "min"),
            prev_close=("Close", "last"),
            prev_atr=("atr",   "last"),
        ).shift(1)
        df = df.merge(daily, on="date", how="left")
        return df

    def _add_ema(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        col = f"ema{period}"
        df[col] = df["Close"].ewm(span=period, adjust=False).mean()
        return df

    def _add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Session VWAP — resets each calendar day."""
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        df["_tp_vol"] = typical * df["Volume"]
        df["vwap"] = (
            df.groupby("date")["_tp_vol"].cumsum() /
            df.groupby("date")["Volume"].cumsum()
        )
        df.drop(columns=["_tp_vol"], inplace=True)
        return df

    def _add_realized_vol(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Rolling realized volatility: std of log returns over `window` bars."""
        log_ret = np.log(df["Close"] / df["Close"].shift(1))
        df["realized_vol"] = log_ret.rolling(window).std()
        return df
