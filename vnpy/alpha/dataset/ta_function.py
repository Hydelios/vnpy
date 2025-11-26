"""
Technical Analysis Operators
"""

import talib
import polars as pl
import pandas as pd

from .utility import DataProxy
from .extra_function import ts_ewm


def to_pd_series(feature: DataProxy) -> pd.Series:
    """Convert to pandas.Series data structure"""
    return feature.df.to_pandas().set_index(["datetime", "vt_symbol"])["data"]


def to_pl_dataframe(series: pd.Series) -> pl.DataFrame:
    """Convert to polars.DataFrame data structure"""
    return pl.from_pandas(series.reset_index().rename(columns={0: "data"}))


def ta_rsi(close: DataProxy, window: int) -> DataProxy:
    """Calculate RSI indicator by contract"""
    close_: pd.Series = to_pd_series(close)

    result: pd.Series = talib.RSI(close_, timeperiod=window)   # type: ignore

    df: pl.DataFrame = to_pl_dataframe(result)
    return DataProxy(df)


def ta_atr(high: DataProxy, low: DataProxy, close: DataProxy, window: int) -> DataProxy:
    """Calculate ATR indicator by contract"""
    high_: pd.Series = to_pd_series(high)
    low_: pd.Series = to_pd_series(low)
    close_: pd.Series = to_pd_series(close)

    result: pd.Series = talib.ATR(high_, low_, close_, timeperiod=window)   # type: ignore

    df: pl.DataFrame = to_pl_dataframe(result)
    return DataProxy(df)


def ta_sma(feature: DataProxy, n: int, m: int) -> DataProxy:
    """Tongdaxin/JoinQuant style SMA(X, N, M).

    Recursive definition:
        Y_t = (M * X_t + (N - M) * Y_{t-1}) / N
    which is equivalent to an EWM with alpha = M / N (no adjustment).

    Parameters
    ----------
    feature : DataProxy
        Input time-series per symbol.
    n : int
        Window parameter N (> 0)
    m : int
        Smoothing parameter M (0 < M <= N)
    """
    if n <= 0:
        raise ValueError("ta_sma: n must be > 0")
    if m <= 0 or m > n:
        raise ValueError("ta_sma: m must be in (0, n]")
    alpha = float(m) / float(n)
    return ts_ewm(feature, alpha=alpha, adjust=False)


def ta_linearreg_slope(feature: DataProxy, window: int) -> DataProxy:
    """Linear regression slope over a rolling window (TA-Lib LINEARREG_SLOPE)."""
    series: pd.Series = to_pd_series(feature)
    result: pd.Series = talib.LINEARREG_SLOPE(series, timeperiod=int(window))  # type: ignore
    df: pl.DataFrame = to_pl_dataframe(result)
    return DataProxy(df)
