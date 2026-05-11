"""
DolphinDB-compatible expression entry points.

These functions intentionally keep DolphinDB spelling at the expression layer
and delegate to existing vn.py alpha operators where the semantics are close.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .cs_function import cs_rank
from .extra_function import cs_pct_rank, ts_cov, ts_prod, ts_sign, ts_where
from .stateful_function import ts_beta, ts_count, ts_ratio
from .ts_function import (
    ts_argmax,
    ts_argmin,
    ts_corr,
    ts_delay,
    ts_greater,
    ts_less,
    ts_max,
    ts_mean,
    ts_min,
    ts_rank,
    ts_slope,
    ts_std,
    ts_sum,
)
from .utility import DataProxy


Number = int | float | bool
FeatureOrNumber = DataProxy | Number


def _as_window(window: int | float) -> int:
    """Normalize DolphinDB window arguments."""
    size = int(round(float(window)))
    return max(1, size)


def _clean_infinite(feature: DataProxy) -> DataProxy:
    """Replace +/-inf with null while preserving DataProxy layout."""
    df = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.when(pl.col("data").is_infinite())
        .then(None)
        .otherwise(pl.col("data"))
        .alias("data"),
    )
    return DataProxy(df, interval=feature.interval)


def _series_to_float(series: pl.Series) -> np.ndarray:
    values: list[float] = []
    for value in series.to_list():
        if value is None:
            values.append(np.nan)
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def iif(cond: DataProxy | bool, x: FeatureOrNumber, y: FeatureOrNumber) -> FeatureOrNumber:
    """DolphinDB iif(cond, x, y), mapped to element-wise ts_where."""
    if isinstance(cond, DataProxy):
        return ts_where(cond, x, y)
    return x if bool(cond) else y


def sign(x: DataProxy) -> DataProxy:
    """DolphinDB sign(x), mapped to ts_sign."""
    return ts_sign(x)


def move(x: DataProxy, n: int | float) -> DataProxy:
    """DolphinDB move(x, n), mapped to ts_delay(x, n)."""
    return ts_delay(x, _as_window(n))


def ratios(x: DataProxy) -> DataProxy:
    """DolphinDB ratios(x), mapped to one-period ts_ratio."""
    return ts_ratio(x, 1)


def rowRank(X: DataProxy | None = None, percent: bool = True, **kwargs: DataProxy) -> DataProxy:
    """Cross-sectional DolphinDB rowRank.

    Supports DolphinDB-style named ``X=``. ``percent=True`` maps to
    cs_pct_rank; ``percent=False`` maps to raw cs_rank.
    """
    feature = X if X is not None else kwargs.get("x")
    if feature is None:
        raise TypeError("rowRank requires X")
    return cs_pct_rank(feature) if bool(percent) else cs_rank(feature)


def msum(x: DataProxy, window: int | float) -> DataProxy:
    return ts_sum(x, _as_window(window))


def mavg(x: DataProxy, window: int | float) -> DataProxy:
    return ts_mean(x, _as_window(window))


def mstd(x: DataProxy, window: int | float) -> DataProxy:
    return ts_std(x, _as_window(window))


def mcorr(x: DataProxy, y: DataProxy, window: int | float) -> DataProxy:
    return ts_corr(x, y, _as_window(window))


def mcovar(x: DataProxy, y: DataProxy, window: int | float) -> DataProxy:
    return ts_cov(x, y, _as_window(window))


def mcov(x: DataProxy, y: DataProxy, window: int | float) -> DataProxy:
    return mcovar(x, y, window)


def mbeta(y: DataProxy, x: DataProxy, window: int | float) -> DataProxy:
    return ts_beta(y, x, _as_window(window))


def mmin(x: DataProxy, window: int | float) -> DataProxy:
    return ts_min(x, _as_window(window))


def mmax(x: DataProxy, window: int | float) -> DataProxy:
    return ts_max(x, _as_window(window))


def mcount(x: DataProxy, window: int | float) -> DataProxy:
    return ts_count(x, _as_window(window))


def mprod(x: DataProxy, window: int | float) -> DataProxy:
    return ts_prod(x, _as_window(window))


def mfirst(x: DataProxy, window: int | float) -> DataProxy:
    """DolphinDB formula-style first value as delay(x, window - 1)."""
    size = _as_window(window)
    if size <= 1:
        return x
    return ts_delay(x, size - 1)


def mrank(x: DataProxy, ascending: bool | int | float = True, window: int | float | None = None) -> DataProxy:
    """Rolling rank compatible entry.

    The underlying ts_rank is percentile-like with scipy percentileofscore
    tie behavior. For descending ranks we mirror the percentile scale with
    ``1 - ts_rank``; this is an approximation of DolphinDB tie handling.
    """
    if window is None:
        window = ascending  # allow mrank(x, window)
        ascending = True
    result = ts_rank(x, _as_window(window))
    if bool(ascending):
        return result
    return 1.0 - result


def mimin(x: DataProxy, window: int | float) -> DataProxy:
    """Rolling argmin entry.

    Maps to ts_argmin for now; DolphinDB position origin/window-edge semantics
    still need numerical alignment against reference output.
    """
    return ts_argmin(x, _as_window(window))


def mimax(x: DataProxy, window: int | float) -> DataProxy:
    """Rolling argmax entry.

    Maps to ts_argmax for now; DolphinDB position origin/window-edge semantics
    still need numerical alignment against reference output.
    """
    return ts_argmax(x, _as_window(window))


def elem_min(x: DataProxy, y: DataProxy | float) -> DataProxy:
    """Element-wise minimum."""
    return ts_less(x, y)


def elem_max(x: DataProxy, y: DataProxy | float) -> DataProxy:
    """Element-wise maximum."""
    return ts_greater(x, y)


ddb_min = elem_min
ddb_max = elem_max


def rowMax(x: DataProxy) -> DataProxy:
    """Cross-sectional maximum per datetime, matching panel rowMax usage."""
    base = x.df.with_columns(
        pl.col("data").cast(pl.Float64, strict=False).fill_nan(None).alias("data")
    )
    df = base.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").max().over("datetime").alias("data"),
    )
    return DataProxy(df, interval=x.interval)


def rowMin(x: DataProxy) -> DataProxy:
    """Cross-sectional minimum per datetime, matching panel rowMin usage."""
    base = x.df.with_columns(
        pl.col("data").cast(pl.Float64, strict=False).fill_nan(None).alias("data")
    )
    df = base.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").min().over("datetime").alias("data"),
    )
    return DataProxy(df, interval=x.interval)


def rollingOlsResidual3(
    y: DataProxy,
    x1: DataProxy,
    x2: DataProxy,
    x3: DataProxy,
    window: int | float,
) -> DataProxy:
    """Rolling OLS current-row residual with intercept and three exposures."""
    size = _as_window(window)
    joined = (
        y.df.with_row_index("_row")
        .rename({"data": "y"})
        .join(x1.df.rename({"data": "x1"}), on=["datetime", "vt_symbol"], how="left")
        .join(x2.df.rename({"data": "x2"}), on=["datetime", "vt_symbol"], how="left")
        .join(x3.df.rename({"data": "x3"}), on=["datetime", "vt_symbol"], how="left")
        .sort("_row")
    )
    output = np.full(joined.height, np.nan, dtype=float)

    for group in joined.partition_by("vt_symbol", maintain_order=True):
        rows = group["_row"].to_numpy()
        y_values = _series_to_float(group["y"])
        x1_values = _series_to_float(group["x1"])
        x2_values = _series_to_float(group["x2"])
        x3_values = _series_to_float(group["x3"])

        for index in range(len(group)):
            current = np.asarray(
                [1.0, x1_values[index], x2_values[index], x3_values[index]],
                dtype=float,
            )
            if not np.isfinite(y_values[index]) or not np.all(np.isfinite(current)):
                continue

            start = max(0, index - size + 1)
            y_window = y_values[start:index + 1]
            x1_window = x1_values[start:index + 1]
            x2_window = x2_values[start:index + 1]
            x3_window = x3_values[start:index + 1]
            valid = (
                np.isfinite(y_window)
                & np.isfinite(x1_window)
                & np.isfinite(x2_window)
                & np.isfinite(x3_window)
            )
            if int(valid.sum()) < 5:
                continue

            x_matrix = np.column_stack(
                [
                    np.ones(int(valid.sum()), dtype=float),
                    x1_window[valid],
                    x2_window[valid],
                    x3_window[valid],
                ]
            )
            try:
                coeffs = np.linalg.lstsq(x_matrix, y_window[valid], rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            output[int(rows[index])] = y_values[index] - float(current @ coeffs)

    df = joined.select("datetime", "vt_symbol").with_columns(pl.Series("data", output))
    return DataProxy(df, interval=y.interval)


rolling_ols_residual3 = rollingOlsResidual3


def conditionalCumprod(
    condition: DataProxy,
    value: DataProxy | Number,
    initial: int | float = 1.0,
) -> DataProxy:
    """Recursive state: state *= value when condition is true, else hold."""
    joined = condition.df.with_row_index("_row").rename({"data": "condition"})
    if isinstance(value, DataProxy):
        joined = joined.join(value.df.rename({"data": "value"}), on=["datetime", "vt_symbol"], how="left")
    else:
        joined = joined.with_columns(pl.lit(float(value)).alias("value"))
    joined = joined.sort("_row")
    output = np.full(joined.height, np.nan, dtype=float)

    for group in joined.partition_by("vt_symbol", maintain_order=True):
        rows = group["_row"].to_numpy()
        conditions = group["condition"].to_list()
        values = _series_to_float(group["value"])
        state = float(initial)
        for index, flag in enumerate(conditions):
            if bool(flag) and np.isfinite(values[index]):
                state *= values[index]
            output[int(rows[index])] = state

    df = joined.select("datetime", "vt_symbol").with_columns(pl.Series("data", output))
    return DataProxy(df, interval=condition.interval)


conditional_cumprod = conditionalCumprod


def signed_power(x: DataProxy, a: FeatureOrNumber) -> DataProxy:
    """DolphinDB signedPower: sign(x) * abs(x) ** a."""
    return _clean_infinite(ts_sign(x) * (abs(x) ** a))


def signedPower(x: DataProxy, a: FeatureOrNumber) -> DataProxy:
    """Camel-case DolphinDB alias for signed_power."""
    return signed_power(x, a)


def linearTimeTrend(x: DataProxy, window: int | float, index: int = 1) -> DataProxy:
    """DolphinDB linearTimeTrend slope entry.

    DolphinDB can expose multiple regression outputs; this compatibility
    entry intentionally returns the slope only.
    """
    return ts_slope(x, _as_window(window))


linear_time_trend = linearTimeTrend
