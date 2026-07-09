"""
Stateful time-series operators.

These operators extend the expression engine with DolphinDB-style stateful
windows that are not covered by the original regular rolling functions.
Implementations favor correctness and clear semantics over maximum throughput;
hot operators can later be specialized with native Polars expressions.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import math
import re
from typing import Callable

import numpy as np
import polars as pl

from .utility import DataProxy


_DURATION_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z]+)?\s*$")
_NS_PER_UNIT: dict[str, float] = {
    "ns": 1.0,
    "us": 1_000.0,
    "ms": 1_000_000.0,
    "s": 1_000_000_000.0,
    "sec": 1_000_000_000.0,
    "secs": 1_000_000_000.0,
    "second": 1_000_000_000.0,
    "seconds": 1_000_000_000.0,
    "m": 60_000_000_000.0,
    "min": 60_000_000_000.0,
    "mins": 60_000_000_000.0,
    "minute": 60_000_000_000.0,
    "minutes": 60_000_000_000.0,
    "h": 3_600_000_000_000.0,
    "hour": 3_600_000_000_000.0,
    "hours": 3_600_000_000_000.0,
    "d": 86_400_000_000_000.0,
    "day": 86_400_000_000_000.0,
    "days": 86_400_000_000_000.0,
    "w": 604_800_000_000_000.0,
    "week": 604_800_000_000_000.0,
    "weeks": 604_800_000_000_000.0,
}
_SECONDS_PER_UNIT: dict[str, float] = {
    key: value / 1_000_000_000.0 for key, value in _NS_PER_UNIT.items()
}


def _as_window(window: int | float) -> int:
    """Normalize expression window arguments to positive integers."""
    result = int(round(float(window)))
    if result < 1:
        raise ValueError("window must be positive")
    return result


def _as_float(value: object) -> float:
    """Convert scalar values to float, mapping invalid/missing values to NaN."""
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan


def _series_to_float(series: pl.Series) -> np.ndarray:
    """Convert a Polars series to a float array with nulls represented as NaN."""
    return np.array([_as_float(value) for value in series.to_list()], dtype=float)


def _series_to_bool(series: pl.Series) -> np.ndarray:
    """Convert a Polars series to bool, treating null/NaN as False."""
    values: list[bool] = []
    for value in series.to_list():
        if value is None:
            values.append(False)
            continue
        try:
            if isinstance(value, float) and math.isnan(value):
                values.append(False)
            else:
                values.append(bool(value))
        except TypeError:
            values.append(False)
    return np.array(values, dtype=bool)


def _valid(values: np.ndarray) -> np.ndarray:
    """Return finite values."""
    return values[np.isfinite(values)]


def _valid_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned finite pairs."""
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _safe_result(value: float | np.floating) -> float:
    """Normalize scalar calculation output."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan
    return result if math.isfinite(result) else np.nan


def _result_from_values(
    base_df: pl.DataFrame,
    values: list[float] | np.ndarray,
    interval: str | None = None,
) -> DataProxy:
    """Build a DataProxy result from a base frame and computed values."""
    result = base_df.select("datetime", "vt_symbol").with_columns(
        pl.Series("data", [_safe_result(value) for value in values])
    )
    return DataProxy(result, interval=interval)


def _per_symbol_unary(
    feature: DataProxy,
    func: Callable[[np.ndarray], np.ndarray],
) -> DataProxy:
    """Apply a vector function independently per vt_symbol."""
    work = feature.df.with_row_index("_result_id")
    out = np.full(work.height, np.nan, dtype=float)

    for group in work.partition_by("vt_symbol", maintain_order=True):
        rows = group["_result_id"].to_numpy()
        values = _series_to_float(group["data"])
        result = np.asarray(func(values), dtype=float)
        out[rows] = result

    return _result_from_values(feature.df, out, interval=feature.interval)


def _join_two(feature1: DataProxy, feature2: DataProxy) -> pl.DataFrame:
    """Join two features while preserving feature1 row order."""
    left = feature1.df.with_row_index("_left_id").rename({"data": "x"})
    right = feature2.df.rename({"data": "y"})
    return (
        left.join(right, on=["datetime", "vt_symbol"], how="inner")
        .sort("_left_id")
        .with_row_index("_result_id")
    )


def _join_three(feature1: DataProxy, feature2: DataProxy, feature3: DataProxy) -> pl.DataFrame:
    """Join three features while preserving feature1 row order."""
    left = feature1.df.with_row_index("_left_id").rename({"data": "x"})
    second = feature2.df.rename({"data": "y"})
    third = feature3.df.rename({"data": "z"})
    return (
        left.join(second, on=["datetime", "vt_symbol"], how="inner")
        .join(third, on=["datetime", "vt_symbol"], how="inner")
        .sort("_left_id")
        .with_row_index("_result_id")
    )


def _per_symbol_joined(
    joined: pl.DataFrame,
    columns: tuple[str, ...],
    func: Callable[..., np.ndarray],
) -> DataProxy:
    """Apply a vector function to joined columns independently per vt_symbol."""
    out = np.full(joined.height, np.nan, dtype=float)

    for group in joined.partition_by("vt_symbol", maintain_order=True):
        rows = group["_result_id"].to_numpy()
        arrays = [_series_to_float(group[column]) for column in columns]
        result = np.asarray(func(*arrays), dtype=float)
        out[rows] = result

    return _result_from_values(joined, out)


def _rolling_unary(
    values: np.ndarray,
    window: int,
    reducer: Callable[[np.ndarray], float],
) -> np.ndarray:
    """Rolling reducer on a single vector."""
    out = np.full(len(values), np.nan, dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        out[index] = reducer(values[start:index + 1])
    return out


def _rolling_pair(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
    reducer: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Rolling reducer on two aligned vectors."""
    out = np.full(len(x), np.nan, dtype=float)
    for index in range(len(x)):
        start = max(0, index - window + 1)
        out[index] = reducer(x[start:index + 1], y[start:index + 1])
    return out


def _cov_population(x: np.ndarray, y: np.ndarray) -> float:
    """Population covariance on aligned non-NaN pairs."""
    x_valid, y_valid = _valid_pair(x, y)
    if len(x_valid) == 0:
        return np.nan
    return float(np.mean(x_valid * y_valid) - np.mean(x_valid) * np.mean(y_valid))


def _corr_population(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation on aligned non-NaN pairs."""
    x_valid, y_valid = _valid_pair(x, y)
    if len(x_valid) < 2:
        return np.nan
    x_std = float(np.std(x_valid, ddof=0))
    y_std = float(np.std(y_valid, ddof=0))
    if x_std == 0.0 or y_std == 0.0:
        return np.nan
    return _cov_population(x_valid, y_valid) / (x_std * y_std)


def _beta_population(y: np.ndarray, x: np.ndarray) -> float:
    """Beta of y against x on aligned non-NaN pairs."""
    y_valid, x_valid = _valid_pair(y, x)
    if len(x_valid) < 2:
        return np.nan
    var_x = float(np.var(x_valid, ddof=0))
    if var_x == 0.0:
        return np.nan
    return _cov_population(y_valid, x_valid) / var_x


def _sum_valid(values: np.ndarray) -> float:
    """Sum finite values."""
    valid = _valid(values)
    if len(valid) == 0:
        return np.nan
    return float(np.sum(valid))


def _prod_valid(values: np.ndarray) -> float:
    """Product of finite values."""
    valid = _valid(values)
    if len(valid) == 0:
        return np.nan
    return float(np.prod(valid))


def _mean_valid(values: np.ndarray) -> float:
    """Mean of finite values."""
    valid = _valid(values)
    if len(valid) == 0:
        return np.nan
    return float(np.mean(valid))


def _min_valid(values: np.ndarray) -> float:
    """Minimum finite value."""
    valid = _valid(values)
    if len(valid) == 0:
        return np.nan
    return float(np.min(valid))


def _max_valid(values: np.ndarray) -> float:
    """Maximum finite value."""
    valid = _valid(values)
    if len(valid) == 0:
        return np.nan
    return float(np.max(valid))


def _median_valid(values: np.ndarray) -> float:
    """Median of finite values."""
    valid = _valid(values)
    if len(valid) == 0:
        return np.nan
    return float(np.median(valid))


def _first_valid(values: np.ndarray) -> float:
    """First finite value preserving window order."""
    valid = _valid(values)
    if len(valid) == 0:
        return np.nan
    return float(valid[0])


def _last_valid(values: np.ndarray) -> float:
    """Last finite value preserving window order."""
    valid = _valid(values)
    if len(valid) == 0:
        return np.nan
    return float(valid[-1])


def _var_valid(values: np.ndarray, ddof: int) -> float:
    """Variance of finite values."""
    valid = _valid(values)
    if len(valid) <= ddof:
        return np.nan
    return float(np.var(valid, ddof=ddof))


def _std_valid(values: np.ndarray, ddof: int) -> float:
    """Standard deviation of finite values."""
    valid = _valid(values)
    if len(valid) <= ddof:
        return np.nan
    return float(np.std(valid, ddof=ddof))


def _skew_valid(values: np.ndarray) -> float:
    """Population skewness of finite values."""
    valid = _valid(values)
    if len(valid) < 3:
        return np.nan
    centered = valid - np.mean(valid)
    second = float(np.mean(centered ** 2))
    if second == 0.0:
        return np.nan
    third = float(np.mean(centered ** 3))
    return third / (second ** 1.5)


def _kurtosis_valid(values: np.ndarray) -> float:
    """Population excess kurtosis of finite values."""
    valid = _valid(values)
    if len(valid) < 4:
        return np.nan
    centered = valid - np.mean(valid)
    second = float(np.mean(centered ** 2))
    if second == 0.0:
        return np.nan
    fourth = float(np.mean(centered ** 4))
    return fourth / (second ** 2) - 3.0


def _sum2_valid(values: np.ndarray) -> float:
    """Sum of squares for finite values."""
    valid = _valid(values)
    if len(valid) == 0:
        return np.nan
    return float(np.sum(valid ** 2))


def _rank_current(values: np.ndarray) -> float:
    """Percent rank of the current value among finite window values."""
    current = values[-1]
    if not np.isfinite(current):
        return np.nan
    valid = _valid(values)
    count = len(valid)
    if count == 0:
        return np.nan
    less = float(np.sum(valid < current))
    equal = float(np.sum(valid == current))
    return (less + (equal + 1.0) / 2.0) / count


def _quantile_valid(values: np.ndarray, quantile: float) -> float:
    """Quantile of finite values."""
    q = float(quantile)
    if q < 0.0 or q > 1.0:
        raise ValueError("quantile must be between 0 and 1")
    valid = _valid(values)
    if len(valid) == 0:
        return np.nan
    return float(np.quantile(valid, q))


def _wsum_valid(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted sum on finite value/weight pairs."""
    x_valid, w_valid = _valid_pair(values, weights)
    if len(x_valid) == 0:
        return np.nan
    return float(np.sum(x_valid * w_valid))


def _wavg_valid(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted average on finite value/weight pairs."""
    x_valid, w_valid = _valid_pair(values, weights)
    if len(x_valid) == 0:
        return np.nan
    denominator = float(np.sum(w_valid))
    if denominator == 0.0:
        return np.nan
    return float(np.sum(x_valid * w_valid) / denominator)


def ts_count(feature: DataProxy, window: int | float) -> DataProxy:
    """Rolling non-NaN count."""
    size = _as_window(window)
    return _per_symbol_unary(
        feature,
        lambda values: _rolling_unary(values, size, lambda item: float(len(_valid(item)))),
    )


def ts_var(feature: DataProxy, window: int | float) -> DataProxy:
    """Rolling sample variance, ignoring NaN."""
    size = _as_window(window)
    return _per_symbol_unary(
        feature,
        lambda values: _rolling_unary(values, size, lambda item: _var_valid(item, ddof=1)),
    )


def ts_varp(feature: DataProxy, window: int | float) -> DataProxy:
    """Rolling population variance, ignoring NaN."""
    size = _as_window(window)
    return _per_symbol_unary(
        feature,
        lambda values: _rolling_unary(values, size, lambda item: _var_valid(item, ddof=0)),
    )


def ts_stdp(feature: DataProxy, window: int | float) -> DataProxy:
    """Rolling population standard deviation, ignoring NaN."""
    size = _as_window(window)
    return _per_symbol_unary(
        feature,
        lambda values: _rolling_unary(values, size, lambda item: _std_valid(item, ddof=0)),
    )


def ts_skew(feature: DataProxy, window: int | float) -> DataProxy:
    """Rolling population skewness, ignoring NaN."""
    size = _as_window(window)
    return _per_symbol_unary(
        feature,
        lambda values: _rolling_unary(values, size, _skew_valid),
    )


def ts_kurtosis(feature: DataProxy, window: int | float) -> DataProxy:
    """Rolling excess kurtosis, ignoring NaN."""
    size = _as_window(window)
    return _per_symbol_unary(
        feature,
        lambda values: _rolling_unary(values, size, _kurtosis_valid),
    )


def ts_median(feature: DataProxy, window: int | float) -> DataProxy:
    """Rolling median, ignoring NaN."""
    size = _as_window(window)
    return _per_symbol_unary(feature, lambda values: _rolling_unary(values, size, _median_valid))


def ts_first(feature: DataProxy, window: int | float) -> DataProxy:
    """First non-NaN value in the rolling window."""
    size = _as_window(window)
    return _per_symbol_unary(feature, lambda values: _rolling_unary(values, size, _first_valid))


def ts_last(feature: DataProxy, window: int | float) -> DataProxy:
    """Last non-NaN value in the rolling window."""
    size = _as_window(window)
    return _per_symbol_unary(feature, lambda values: _rolling_unary(values, size, _last_valid))


def ts_wsum(feature: DataProxy, weight: DataProxy, window: int | float) -> DataProxy:
    """Rolling weighted sum: sum(feature * weight)."""
    size = _as_window(window)
    joined = _join_two(feature, weight)

    def reducer(x: np.ndarray, y: np.ndarray) -> float:
        x_valid, y_valid = _valid_pair(x, y)
        if len(x_valid) == 0:
            return np.nan
        return float(np.sum(x_valid * y_valid))

    return _per_symbol_joined(
        joined,
        ("x", "y"),
        lambda x, y: _rolling_pair(x, y, size, reducer),
    )


def ts_wavg(feature: DataProxy, weight: DataProxy, window: int | float) -> DataProxy:
    """Rolling weighted average: sum(feature * weight) / sum(weight)."""
    size = _as_window(window)
    joined = _join_two(feature, weight)

    def reducer(x: np.ndarray, y: np.ndarray) -> float:
        x_valid, y_valid = _valid_pair(x, y)
        if len(x_valid) == 0:
            return np.nan
        denominator = float(np.sum(y_valid))
        if denominator == 0.0:
            return np.nan
        return float(np.sum(x_valid * y_valid) / denominator)

    return _per_symbol_joined(
        joined,
        ("x", "y"),
        lambda x, y: _rolling_pair(x, y, size, reducer),
    )


def ts_mdd(feature: DataProxy, window: int | float) -> DataProxy:
    """Rolling maximum drawdown as a positive ratio.

    For each rolling window, this returns max((peak - later_trough) / peak).
    Invalid or non-positive peaks are ignored.
    """
    size = _as_window(window)

    def reducer(values: np.ndarray) -> float:
        valid = _valid(values)
        if len(valid) == 0:
            return np.nan
        peak = np.nan
        max_drawdown = 0.0
        for value in valid:
            if not np.isfinite(peak) or value > peak:
                peak = value
            if np.isfinite(peak) and peak > 0.0:
                max_drawdown = max(max_drawdown, float((peak - value) / peak))
        return max_drawdown

    return _per_symbol_unary(feature, lambda values: _rolling_unary(values, size, reducer))


def ts_avedev(feature: DataProxy, window: int | float) -> DataProxy:
    """Rolling average absolute deviation from the window mean."""
    size = _as_window(window)

    def reducer(values: np.ndarray) -> float:
        valid = _valid(values)
        if len(valid) == 0:
            return np.nan
        mean = float(np.mean(valid))
        return float(np.mean(np.abs(valid - mean)))

    return _per_symbol_unary(feature, lambda values: _rolling_unary(values, size, reducer))


def rq_dma(
    feature: DataProxy,
    alpha: DataProxy | int | float,
    alpha_scale: int | float = 100.0,
    clip: bool = True,
) -> DataProxy:
    """Ricequant-style dynamic moving average.

    The rq_tech definitions pass turnover rates as percentages, e.g.
    ``100 * volume / capital``.  By default those values are divided by 100
    before applying the recursive update:

        y_t = a_t * x_t + (1 - a_t) * y_{t-1}
    """
    base = feature.df.with_row_index("_left_id").rename({"data": "x"})
    joined, alpha_name = _aligned_value_array(base, alpha, "_alpha")
    joined = joined.sort("_left_id").with_row_index("_result_id")
    scale = float(alpha_scale)
    use_clip = bool(clip)

    def func(values: np.ndarray, alphas: np.ndarray) -> np.ndarray:
        out = np.full(len(values), np.nan, dtype=float)
        state = np.nan
        for index, value in enumerate(values):
            if not np.isfinite(value):
                out[index] = state
                continue

            raw_alpha = alphas[index]
            if not np.isfinite(raw_alpha):
                out[index] = state
                continue

            coeff = raw_alpha / scale if scale != 0.0 else raw_alpha
            if use_clip:
                coeff = min(1.0, max(0.0, coeff))

            if not np.isfinite(state):
                state = value
            else:
                state = coeff * value + (1.0 - coeff) * state
            out[index] = state
        return out

    return _per_symbol_joined(joined, ("x", alpha_name), func)


def ts_beta(feature_y: DataProxy, feature_x: DataProxy, window: int | float) -> DataProxy:
    """Rolling beta of feature_y against feature_x."""
    size = _as_window(window)
    joined = _join_two(feature_y, feature_x)
    return _per_symbol_joined(
        joined,
        ("x", "y"),
        lambda y, x: _rolling_pair(y, x, size, _beta_population),
    )


def ts_ffill(feature: DataProxy) -> DataProxy:
    """Forward-fill non-NaN values per symbol."""
    def func(values: np.ndarray) -> np.ndarray:
        out = np.full(len(values), np.nan, dtype=float)
        last = np.nan
        for index, value in enumerate(values):
            if np.isfinite(value):
                last = value
            out[index] = last
        return out

    return _per_symbol_unary(feature, func)


def ts_ratio(feature: DataProxy, window: int | float) -> DataProxy:
    """Ratio to the value window bars ago: x_t / x_{t-window}."""
    size = _as_window(window)

    def func(values: np.ndarray) -> np.ndarray:
        out = np.full(len(values), np.nan, dtype=float)
        for index in range(size, len(values)):
            current = values[index]
            previous = values[index - size]
            if not np.isfinite(current) or not np.isfinite(previous) or previous == 0.0:
                continue
            out[index] = current / previous
        return out

    return _per_symbol_unary(feature, func)


def ts_pct_change(feature: DataProxy, window: int | float = 1) -> DataProxy:
    """Percentage change to the value window bars ago."""
    ratio = ts_ratio(feature, window)
    return ratio - 1.0


def ts_prev_state(feature: DataProxy) -> DataProxy:
    """Previous observed state, compressing consecutive identical states."""
    def func(values: np.ndarray) -> np.ndarray:
        out = np.full(len(values), np.nan, dtype=float)
        last_state = np.nan
        previous_distinct = np.nan
        for index, value in enumerate(values):
            if not np.isfinite(value):
                out[index] = last_state
            elif not np.isfinite(last_state):
                last_state = value
            elif value == last_state:
                out[index] = previous_distinct
            else:
                previous_distinct = last_state
                last_state = value
                out[index] = previous_distinct
        return out

    return _per_symbol_unary(feature, func)


def ts_cumsum(feature: DataProxy) -> DataProxy:
    """Cumulative sum per symbol, ignoring NaN in the state update."""
    def func(values: np.ndarray) -> np.ndarray:
        out = np.full(len(values), np.nan, dtype=float)
        total = 0.0
        seen = False
        for index, value in enumerate(values):
            if np.isfinite(value):
                total += value
                seen = True
            if seen:
                out[index] = total
        return out

    return _per_symbol_unary(feature, func)


def ts_cumprod(feature: DataProxy) -> DataProxy:
    """Cumulative product per symbol, ignoring NaN in the state update."""
    def func(values: np.ndarray) -> np.ndarray:
        out = np.full(len(values), np.nan, dtype=float)
        total = 1.0
        seen = False
        for index, value in enumerate(values):
            if np.isfinite(value):
                total *= value
                seen = True
            if seen:
                out[index] = total
        return out

    return _per_symbol_unary(feature, func)


def ts_cumcount(feature: DataProxy) -> DataProxy:
    """Cumulative non-NaN count per symbol."""
    def func(values: np.ndarray) -> np.ndarray:
        out = np.zeros(len(values), dtype=float)
        count = 0.0
        for index, value in enumerate(values):
            if np.isfinite(value):
                count += 1.0
            out[index] = count
        return out

    return _per_symbol_unary(feature, func)


def ts_cummin(feature: DataProxy) -> DataProxy:
    """Cumulative minimum per symbol, ignoring NaN."""
    def func(values: np.ndarray) -> np.ndarray:
        out = np.full(len(values), np.nan, dtype=float)
        current = np.nan
        for index, value in enumerate(values):
            if np.isfinite(value):
                current = value if not np.isfinite(current) else min(current, value)
            out[index] = current
        return out

    return _per_symbol_unary(feature, func)


def ts_cummax(feature: DataProxy) -> DataProxy:
    """Cumulative maximum per symbol, ignoring NaN."""
    def func(values: np.ndarray) -> np.ndarray:
        out = np.full(len(values), np.nan, dtype=float)
        current = np.nan
        for index, value in enumerate(values):
            if np.isfinite(value):
                current = value if not np.isfinite(current) else max(current, value)
            out[index] = current
        return out

    return _per_symbol_unary(feature, func)


def ts_cummean(feature: DataProxy) -> DataProxy:
    """Cumulative mean per symbol, ignoring NaN."""
    def func(values: np.ndarray) -> np.ndarray:
        out = np.full(len(values), np.nan, dtype=float)
        total = 0.0
        count = 0.0
        for index, value in enumerate(values):
            if np.isfinite(value):
                total += value
                count += 1.0
            if count > 0.0:
                out[index] = total / count
        return out

    return _per_symbol_unary(feature, func)


def _cum_var(values: np.ndarray, ddof: int) -> np.ndarray:
    """Online cumulative variance."""
    out = np.full(len(values), np.nan, dtype=float)
    count = 0
    mean = 0.0
    m2 = 0.0
    for index, value in enumerate(values):
        if np.isfinite(value):
            count += 1
            delta = value - mean
            mean += delta / count
            m2 += delta * (value - mean)
        if count > ddof:
            out[index] = m2 / (count - ddof)
    return out


def ts_cumvar(feature: DataProxy) -> DataProxy:
    """Cumulative sample variance per symbol."""
    return _per_symbol_unary(feature, lambda values: _cum_var(values, ddof=1))


def ts_cumstd(feature: DataProxy) -> DataProxy:
    """Cumulative sample standard deviation per symbol."""
    return _per_symbol_unary(feature, lambda values: np.sqrt(_cum_var(values, ddof=1)))


def ts_cumvarp(feature: DataProxy) -> DataProxy:
    """Cumulative population variance per symbol."""
    return _per_symbol_unary(feature, lambda values: _cum_var(values, ddof=0))


def ts_cumstdp(feature: DataProxy) -> DataProxy:
    """Cumulative population standard deviation per symbol."""
    return _per_symbol_unary(feature, lambda values: np.sqrt(_cum_var(values, ddof=0)))


def ts_cumfirst_not(feature: DataProxy) -> DataProxy:
    """First finite value observed so far per symbol."""
    def func(values: np.ndarray) -> np.ndarray:
        out = np.full(len(values), np.nan, dtype=float)
        first = np.nan
        for index, value in enumerate(values):
            if np.isfinite(value) and not np.isfinite(first):
                first = value
            out[index] = first
        return out

    return _per_symbol_unary(feature, func)


def ts_cumlast_not(feature: DataProxy) -> DataProxy:
    """Last finite value observed so far per symbol."""
    return ts_ffill(feature)


def ts_cummedian(feature: DataProxy) -> DataProxy:
    """Cumulative median per symbol."""
    def func(values: np.ndarray) -> np.ndarray:
        out = np.full(len(values), np.nan, dtype=float)
        history: list[float] = []
        for index, value in enumerate(values):
            if np.isfinite(value):
                history.append(float(value))
            if history:
                out[index] = float(np.median(np.asarray(history, dtype=float)))
        return out

    return _per_symbol_unary(feature, func)


def ts_cumpercentile(feature: DataProxy, quantile: float) -> DataProxy:
    """Cumulative percentile per symbol."""
    q = float(quantile)
    if q < 0.0 or q > 1.0:
        raise ValueError("quantile must be between 0 and 1")

    def func(values: np.ndarray) -> np.ndarray:
        out = np.full(len(values), np.nan, dtype=float)
        history: list[float] = []
        for index, value in enumerate(values):
            if np.isfinite(value):
                history.append(float(value))
            if history:
                out[index] = float(np.quantile(np.asarray(history, dtype=float), q))
        return out

    return _per_symbol_unary(feature, func)


def ts_cumnunique(feature: DataProxy) -> DataProxy:
    """Cumulative number of distinct finite values per symbol."""
    def func(values: np.ndarray) -> np.ndarray:
        out = np.zeros(len(values), dtype=float)
        seen: set[float] = set()
        for index, value in enumerate(values):
            if np.isfinite(value):
                seen.add(float(value))
            out[index] = float(len(seen))
        return out

    return _per_symbol_unary(feature, func)


def ts_cum_positive_streak(feature: DataProxy) -> DataProxy:
    """Consecutive positive finite-value count per symbol."""
    def func(values: np.ndarray) -> np.ndarray:
        out = np.zeros(len(values), dtype=float)
        streak = 0.0
        for index, value in enumerate(values):
            if np.isfinite(value) and value > 0.0:
                streak += 1.0
            else:
                streak = 0.0
            out[index] = streak
        return out

    return _per_symbol_unary(feature, func)


def _cum_pair(values1: np.ndarray, values2: np.ndarray, metric: str) -> np.ndarray:
    """Online cumulative covariance/correlation/beta."""
    out = np.full(len(values1), np.nan, dtype=float)
    count = 0
    mean1 = 0.0
    mean2 = 0.0
    c12 = 0.0
    m2_1 = 0.0
    m2_2 = 0.0

    for index, (value1, value2) in enumerate(zip(values1, values2)):
        if np.isfinite(value1) and np.isfinite(value2):
            count += 1
            delta1 = value1 - mean1
            mean1 += delta1 / count
            delta2 = value2 - mean2
            mean2 += delta2 / count
            c12 += delta1 * (value2 - mean2)
            m2_1 += delta1 * (value1 - mean1)
            m2_2 += delta2 * (value2 - mean2)

        if metric == "cov":
            if count > 0:
                out[index] = c12 / count
        elif metric == "corr":
            if count > 1 and m2_1 > 0.0 and m2_2 > 0.0:
                out[index] = c12 / math.sqrt(m2_1 * m2_2)
        elif metric == "beta":
            if count > 1 and m2_2 > 0.0:
                out[index] = c12 / m2_2
        else:
            raise ValueError(f"unsupported cumulative pair metric: {metric}")

    return out


def ts_cumcov(feature1: DataProxy, feature2: DataProxy) -> DataProxy:
    """Cumulative population covariance per symbol."""
    joined = _join_two(feature1, feature2)
    return _per_symbol_joined(joined, ("x", "y"), lambda x, y: _cum_pair(x, y, "cov"))


def ts_cumcorr(feature1: DataProxy, feature2: DataProxy) -> DataProxy:
    """Cumulative correlation per symbol."""
    joined = _join_two(feature1, feature2)
    return _per_symbol_joined(joined, ("x", "y"), lambda x, y: _cum_pair(x, y, "corr"))


def ts_cumbeta(feature_y: DataProxy, feature_x: DataProxy) -> DataProxy:
    """Cumulative beta of feature_y against feature_x per symbol."""
    joined = _join_two(feature_y, feature_x)
    return _per_symbol_joined(joined, ("x", "y"), lambda y, x: _cum_pair(y, x, "beta"))


def ts_cumwsum(feature: DataProxy, weight: DataProxy) -> DataProxy:
    """Cumulative weighted sum per symbol."""
    joined = _join_two(feature, weight)

    def func(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        out = np.full(len(values), np.nan, dtype=float)
        total = 0.0
        seen = False
        for index, (value, weight_value) in enumerate(zip(values, weights)):
            if np.isfinite(value) and np.isfinite(weight_value):
                total += value * weight_value
                seen = True
            if seen:
                out[index] = total
        return out

    return _per_symbol_joined(joined, ("x", "y"), func)


def ts_cumwavg(feature: DataProxy, weight: DataProxy) -> DataProxy:
    """Cumulative weighted average per symbol."""
    joined = _join_two(feature, weight)

    def func(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        out = np.full(len(values), np.nan, dtype=float)
        numerator = 0.0
        denominator = 0.0
        for index, (value, weight_value) in enumerate(zip(values, weights)):
            if np.isfinite(value) and np.isfinite(weight_value):
                numerator += value * weight_value
                denominator += weight_value
            if denominator != 0.0:
                out[index] = numerator / denominator
        return out

    return _per_symbol_joined(joined, ("x", "y"), func)


def _looks_temporal(value: object) -> bool:
    """Infer whether a scalar represents time."""
    if isinstance(value, (datetime, date, np.datetime64)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return False
        except ValueError:
            return True
    return False


def _time_scalar_to_float(value: object, temporal: bool) -> float:
    """Convert time values to numeric coordinates."""
    if value is None:
        return np.nan
    if temporal:
        try:
            return float(np.datetime64(value, "ns").astype("int64"))
        except (TypeError, ValueError, OverflowError):
            return np.nan
    return _as_float(value)


def _time_series_to_float(series: pl.Series) -> tuple[np.ndarray, bool]:
    """Convert a time series to numeric coordinates and return temporal flag."""
    values = series.to_list()
    temporal = False
    for value in values:
        if value is not None:
            temporal = _looks_temporal(value)
            break
    return (
        np.array([_time_scalar_to_float(value, temporal) for value in values], dtype=float),
        temporal,
    )


def _duration_amount_to_units(amount: float, unit: str, temporal: bool) -> float:
    """Convert a positive duration amount and unit to the time vector units."""
    unit = unit.lower()
    if temporal:
        if unit not in _NS_PER_UNIT:
            raise ValueError(f"unsupported temporal duration unit: {unit!r}")
        return amount * _NS_PER_UNIT[unit]

    if unit not in _SECONDS_PER_UNIT:
        raise ValueError(f"unsupported numeric duration unit: {unit!r}")
    return amount * _SECONDS_PER_UNIT[unit]


def _interval_to_units(interval: str | None, temporal: bool) -> float:
    """Convert an AlphaDataset interval string to a base real-time duration."""
    if interval is None:
        raise ValueError(
            "numeric tm window requires an interval context or an explicit duration unit"
        )

    normalized = str(interval).strip().lower()
    aliases = {
        "d": "1d",
        "day": "1d",
        "daily": "1d",
        "minute": "1m",
        "min": "1m",
        "hour": "1h",
        "h": "1h",
    }
    normalized = aliases.get(normalized, normalized)

    match = _DURATION_RE.match(normalized)
    if match is None or match.group(2) is None:
        raise ValueError(f"unsupported interval for numeric tm window: {interval!r}")

    amount = float(match.group(1))
    unit = match.group(2)
    if amount <= 0.0:
        raise ValueError(f"interval must be positive: {interval!r}")
    return _duration_amount_to_units(amount, unit, temporal)


def _duration_to_units(
    duration: str | int | float | timedelta,
    temporal: bool,
    interval: str | None,
) -> float:
    """Normalize duration to the same units as the converted time vector."""
    unit: str | None = None
    amount: float

    if isinstance(duration, timedelta):
        amount = duration.total_seconds()
        unit = "s"
    elif isinstance(duration, str):
        match = _DURATION_RE.match(duration)
        if match is None:
            raise ValueError(f"invalid duration: {duration!r}")
        amount = float(match.group(1))
        unit = match.group(2)
    else:
        amount = float(duration)

    if amount <= 0.0:
        raise ValueError("duration must be positive")

    if unit is None:
        return amount * _interval_to_units(interval, temporal)

    return _duration_amount_to_units(amount, unit, temporal)


def _same_key_order(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    """Return whether two feature frames are row-aligned by key."""
    if left.height != right.height:
        return False
    return left.select(["datetime", "vt_symbol"]).equals(
        right.select(["datetime", "vt_symbol"])
    )


def _join_time_feature(time: DataProxy, feature: DataProxy) -> pl.DataFrame:
    """Join an explicit time vector with one feature."""
    base = time.df.with_row_index("_left_id").rename({"data": "_time"})
    data = feature.df.rename({"data": "x"})
    if _same_key_order(time.df, feature.df):
        return (
            base.with_columns(data["x"].alias("x"))
            .sort("_left_id")
            .with_row_index("_result_id")
        )
    return (
        base.join(data, on=["datetime", "vt_symbol"], how="inner")
        .sort("_left_id")
        .with_row_index("_result_id")
    )


def _join_time_pair(time: DataProxy, feature1: DataProxy, feature2: DataProxy) -> pl.DataFrame:
    """Join an explicit time vector with two features."""
    base = time.df.with_row_index("_left_id").rename({"data": "_time"})
    first = feature1.df.rename({"data": "x"})
    second = feature2.df.rename({"data": "y"})
    if _same_key_order(time.df, feature1.df) and _same_key_order(time.df, feature2.df):
        return (
            base.with_columns(
                first["x"].alias("x"),
                second["y"].alias("y"),
            )
            .sort("_left_id")
            .with_row_index("_result_id")
        )
    return (
        base.join(first, on=["datetime", "vt_symbol"], how="inner")
        .join(second, on=["datetime", "vt_symbol"], how="inner")
        .sort("_left_id")
        .with_row_index("_result_id")
    )


def _join_time_three(
    time: DataProxy,
    feature1: DataProxy,
    feature2: DataProxy,
    feature3: DataProxy,
) -> pl.DataFrame:
    """Join an explicit time vector with three features."""
    base = time.df.with_row_index("_left_id").rename({"data": "_time"})
    first = feature1.df.rename({"data": "x"})
    second = feature2.df.rename({"data": "y"})
    third = feature3.df.rename({"data": "z"})
    if (
        _same_key_order(time.df, feature1.df)
        and _same_key_order(time.df, feature2.df)
        and _same_key_order(time.df, feature3.df)
    ):
        return (
            base.with_columns(
                first["x"].alias("x"),
                second["y"].alias("y"),
                third["z"].alias("z"),
            )
            .sort("_left_id")
            .with_row_index("_result_id")
        )
    return (
        base.join(first, on=["datetime", "vt_symbol"], how="inner")
        .join(second, on=["datetime", "vt_symbol"], how="inner")
        .join(third, on=["datetime", "vt_symbol"], how="inner")
        .sort("_left_id")
        .with_row_index("_result_id")
    )


def _time_window_unary(
    time: DataProxy,
    feature: DataProxy,
    duration: str | int | float | timedelta,
    reducer: Callable[[np.ndarray], float],
) -> DataProxy:
    """Apply a left-open, right-closed explicit time window reducer."""
    joined = _join_time_feature(time, feature)
    out = np.full(joined.height, np.nan, dtype=float)

    for group in joined.partition_by("vt_symbol", maintain_order=True):
        result_rows = group["_result_id"].to_numpy()
        times, temporal = _time_series_to_float(group["_time"])
        values = _series_to_float(group["x"])
        valid_time_rows = np.flatnonzero(np.isfinite(times))
        if len(valid_time_rows) == 0:
            continue

        order = valid_time_rows[np.lexsort((result_rows[valid_time_rows], times[valid_time_rows]))]
        sorted_times = times[order]
        sorted_values = values[order]
        sorted_rows = result_rows[order]
        duration_value = _duration_to_units(duration, temporal, time.interval)

        left = 0
        for right, current_time in enumerate(sorted_times):
            lower = current_time - duration_value
            while left <= right and sorted_times[left] <= lower:
                left += 1
            out[sorted_rows[right]] = reducer(sorted_values[left:right + 1])

    return _result_from_values(joined, out, interval=time.interval)


def _time_window_pair(
    time: DataProxy,
    feature1: DataProxy,
    feature2: DataProxy,
    duration: str | int | float | timedelta,
    reducer: Callable[[np.ndarray, np.ndarray], float],
) -> DataProxy:
    """Apply a left-open, right-closed explicit time window pair reducer."""
    joined = _join_time_pair(time, feature1, feature2)
    out = np.full(joined.height, np.nan, dtype=float)

    for group in joined.partition_by("vt_symbol", maintain_order=True):
        result_rows = group["_result_id"].to_numpy()
        times, temporal = _time_series_to_float(group["_time"])
        x = _series_to_float(group["x"])
        y = _series_to_float(group["y"])
        valid_time_rows = np.flatnonzero(np.isfinite(times))
        if len(valid_time_rows) == 0:
            continue

        order = valid_time_rows[np.lexsort((result_rows[valid_time_rows], times[valid_time_rows]))]
        sorted_times = times[order]
        sorted_x = x[order]
        sorted_y = y[order]
        sorted_rows = result_rows[order]
        duration_value = _duration_to_units(duration, temporal, time.interval)

        left = 0
        for right, current_time in enumerate(sorted_times):
            lower = current_time - duration_value
            while left <= right and sorted_times[left] <= lower:
                left += 1
            out[sorted_rows[right]] = reducer(
                sorted_x[left:right + 1],
                sorted_y[left:right + 1],
            )

    return _result_from_values(joined, out, interval=time.interval)


def _time_window_topn_unary(
    time: DataProxy,
    feature: DataProxy,
    sort_by: DataProxy,
    duration: str | int | float | timedelta,
    top: int | float,
    ascending: bool,
    ties_method: str,
    reducer: Callable[[np.ndarray], float],
) -> DataProxy:
    """Apply a time-window TopN reducer on one value vector."""
    joined = _join_time_pair(time, feature, sort_by)
    out = np.full(joined.height, np.nan, dtype=float)

    for group in joined.partition_by("vt_symbol", maintain_order=True):
        result_rows = group["_result_id"].to_numpy()
        times, temporal = _time_series_to_float(group["_time"])
        values = _series_to_float(group["x"])
        sort_values = _series_to_float(group["y"])
        valid_time_rows = np.flatnonzero(np.isfinite(times))
        if len(valid_time_rows) == 0:
            continue

        order = valid_time_rows[np.lexsort((result_rows[valid_time_rows], times[valid_time_rows]))]
        sorted_times = times[order]
        sorted_values = values[order]
        sorted_sort = sort_values[order]
        sorted_rows = result_rows[order]
        duration_value = _duration_to_units(duration, temporal, time.interval)

        left = 0
        for right, current_time in enumerate(sorted_times):
            lower = current_time - duration_value
            while left <= right and sorted_times[left] <= lower:
                left += 1
            chosen = _top_indices(sorted_sort[left:right + 1], top, ascending, ties_method)
            if len(chosen) > 0:
                out[sorted_rows[right]] = reducer(sorted_values[left:right + 1][chosen])

    return _result_from_values(joined, out, interval=time.interval)


def _time_window_topn_pair(
    time: DataProxy,
    feature1: DataProxy,
    feature2: DataProxy,
    sort_by: DataProxy,
    duration: str | int | float | timedelta,
    top: int | float,
    ascending: bool,
    ties_method: str,
    reducer: Callable[[np.ndarray, np.ndarray], float],
) -> DataProxy:
    """Apply a time-window TopN reducer on two value vectors."""
    joined = _join_time_three(time, feature1, feature2, sort_by)
    out = np.full(joined.height, np.nan, dtype=float)

    for group in joined.partition_by("vt_symbol", maintain_order=True):
        result_rows = group["_result_id"].to_numpy()
        times, temporal = _time_series_to_float(group["_time"])
        x = _series_to_float(group["x"])
        y = _series_to_float(group["y"])
        sort_values = _series_to_float(group["z"])
        valid_time_rows = np.flatnonzero(np.isfinite(times))
        if len(valid_time_rows) == 0:
            continue

        order = valid_time_rows[np.lexsort((result_rows[valid_time_rows], times[valid_time_rows]))]
        sorted_times = times[order]
        sorted_x = x[order]
        sorted_y = y[order]
        sorted_sort = sort_values[order]
        sorted_rows = result_rows[order]
        duration_value = _duration_to_units(duration, temporal, time.interval)

        left = 0
        for right, current_time in enumerate(sorted_times):
            lower = current_time - duration_value
            while left <= right and sorted_times[left] <= lower:
                left += 1
            chosen = _top_indices(sorted_sort[left:right + 1], top, ascending, ties_method)
            if len(chosen) > 0:
                out[sorted_rows[right]] = reducer(
                    sorted_x[left:right + 1][chosen],
                    sorted_y[left:right + 1][chosen],
                )

    return _result_from_values(joined, out, interval=time.interval)


def ts_time_sum(
    time: DataProxy,
    feature: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Explicit time-window sum."""
    return _time_window_unary(time, feature, duration, _sum_valid)


def ts_time_sum2(
    time: DataProxy,
    feature: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Explicit time-window sum of squares."""
    return _time_window_unary(time, feature, duration, _sum2_valid)


def ts_time_prod(
    time: DataProxy,
    feature: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Explicit time-window product."""
    return _time_window_unary(time, feature, duration, _prod_valid)


def ts_time_mean(
    time: DataProxy,
    feature: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Explicit time-window mean."""
    return _time_window_unary(time, feature, duration, _mean_valid)


def ts_time_count(
    time: DataProxy,
    feature: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Explicit time-window non-NaN count."""
    return _time_window_unary(time, feature, duration, lambda values: float(len(_valid(values))))


def ts_time_min(
    time: DataProxy,
    feature: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Explicit time-window minimum."""
    return _time_window_unary(time, feature, duration, _min_valid)


def ts_time_max(
    time: DataProxy,
    feature: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Explicit time-window maximum."""
    return _time_window_unary(time, feature, duration, _max_valid)


def ts_time_first(
    time: DataProxy,
    feature: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """First finite value in an explicit time window."""
    return _time_window_unary(time, feature, duration, _first_valid)


def ts_time_last(
    time: DataProxy,
    feature: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Last finite value in an explicit time window."""
    return _time_window_unary(time, feature, duration, _last_valid)


def ts_time_var(
    time: DataProxy,
    feature: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Explicit time-window sample variance."""
    return _time_window_unary(time, feature, duration, lambda values: _var_valid(values, ddof=1))


def ts_time_varp(
    time: DataProxy,
    feature: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Explicit time-window population variance."""
    return _time_window_unary(time, feature, duration, lambda values: _var_valid(values, ddof=0))


def ts_time_std(
    time: DataProxy,
    feature: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Explicit time-window sample standard deviation."""
    return _time_window_unary(time, feature, duration, lambda values: _std_valid(values, ddof=1))


def ts_time_stdp(
    time: DataProxy,
    feature: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Explicit time-window population standard deviation."""
    return _time_window_unary(time, feature, duration, lambda values: _std_valid(values, ddof=0))


def ts_time_skew(
    time: DataProxy,
    feature: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Explicit time-window skewness."""
    return _time_window_unary(time, feature, duration, _skew_valid)


def ts_time_kurtosis(
    time: DataProxy,
    feature: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Explicit time-window excess kurtosis."""
    return _time_window_unary(time, feature, duration, _kurtosis_valid)


def ts_time_median(
    time: DataProxy,
    feature: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Explicit time-window median."""
    return _time_window_unary(time, feature, duration, _median_valid)


def ts_time_rank(
    time: DataProxy,
    feature: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Percent rank of the current value in an explicit time window."""
    return _time_window_unary(time, feature, duration, _rank_current)


def ts_time_percentile(
    time: DataProxy,
    feature: DataProxy,
    duration: str | int | float | timedelta,
    quantile: float,
) -> DataProxy:
    """Explicit time-window percentile."""
    return _time_window_unary(
        time,
        feature,
        duration,
        lambda values: _quantile_valid(values, quantile),
    )


def ts_time_cov(
    time: DataProxy,
    feature1: DataProxy,
    feature2: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Explicit time-window population covariance."""
    return _time_window_pair(time, feature1, feature2, duration, _cov_population)


def ts_time_corr(
    time: DataProxy,
    feature1: DataProxy,
    feature2: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Explicit time-window correlation."""
    return _time_window_pair(time, feature1, feature2, duration, _corr_population)


def ts_time_beta(
    time: DataProxy,
    feature_y: DataProxy,
    feature_x: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Explicit time-window beta of feature_y against feature_x."""
    return _time_window_pair(time, feature_y, feature_x, duration, _beta_population)


def ts_time_wsum(
    time: DataProxy,
    feature: DataProxy,
    weight: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Explicit time-window weighted sum."""
    return _time_window_pair(time, feature, weight, duration, _wsum_valid)


def ts_time_wavg(
    time: DataProxy,
    feature: DataProxy,
    weight: DataProxy,
    duration: str | int | float | timedelta,
) -> DataProxy:
    """Explicit time-window weighted average."""
    return _time_window_pair(time, feature, weight, duration, _wavg_valid)


def ts_time_sum_topn(
    time: DataProxy,
    feature: DataProxy,
    sort_by: DataProxy,
    duration: str | int | float | timedelta,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Explicit time-window TopN sum."""
    return _time_window_topn_unary(
        time, feature, sort_by, duration, top, ascending, ties_method, _sum_valid
    )


def ts_time_mean_topn(
    time: DataProxy,
    feature: DataProxy,
    sort_by: DataProxy,
    duration: str | int | float | timedelta,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Explicit time-window TopN mean."""
    return _time_window_topn_unary(
        time, feature, sort_by, duration, top, ascending, ties_method, _mean_valid
    )


def ts_time_var_topn(
    time: DataProxy,
    feature: DataProxy,
    sort_by: DataProxy,
    duration: str | int | float | timedelta,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Explicit time-window TopN sample variance."""
    return _time_window_topn_unary(
        time,
        feature,
        sort_by,
        duration,
        top,
        ascending,
        ties_method,
        lambda values: _var_valid(values, ddof=1),
    )


def ts_time_varp_topn(
    time: DataProxy,
    feature: DataProxy,
    sort_by: DataProxy,
    duration: str | int | float | timedelta,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Explicit time-window TopN population variance."""
    return _time_window_topn_unary(
        time,
        feature,
        sort_by,
        duration,
        top,
        ascending,
        ties_method,
        lambda values: _var_valid(values, ddof=0),
    )


def ts_time_std_topn(
    time: DataProxy,
    feature: DataProxy,
    sort_by: DataProxy,
    duration: str | int | float | timedelta,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Explicit time-window TopN sample standard deviation."""
    return _time_window_topn_unary(
        time,
        feature,
        sort_by,
        duration,
        top,
        ascending,
        ties_method,
        lambda values: _std_valid(values, ddof=1),
    )


def ts_time_stdp_topn(
    time: DataProxy,
    feature: DataProxy,
    sort_by: DataProxy,
    duration: str | int | float | timedelta,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Explicit time-window TopN population standard deviation."""
    return _time_window_topn_unary(
        time,
        feature,
        sort_by,
        duration,
        top,
        ascending,
        ties_method,
        lambda values: _std_valid(values, ddof=0),
    )


def ts_time_cov_topn(
    time: DataProxy,
    feature1: DataProxy,
    feature2: DataProxy,
    sort_by: DataProxy,
    duration: str | int | float | timedelta,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Explicit time-window TopN covariance."""
    return _time_window_topn_pair(
        time, feature1, feature2, sort_by, duration, top, ascending, ties_method, _cov_population
    )


def ts_time_corr_topn(
    time: DataProxy,
    feature1: DataProxy,
    feature2: DataProxy,
    sort_by: DataProxy,
    duration: str | int | float | timedelta,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Explicit time-window TopN correlation."""
    return _time_window_topn_pair(
        time, feature1, feature2, sort_by, duration, top, ascending, ties_method, _corr_population
    )


def ts_time_beta_topn(
    time: DataProxy,
    feature_y: DataProxy,
    feature_x: DataProxy,
    sort_by: DataProxy,
    duration: str | int | float | timedelta,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Explicit time-window TopN beta."""
    return _time_window_topn_pair(
        time,
        feature_y,
        feature_x,
        sort_by,
        duration,
        top,
        ascending,
        ties_method,
        _beta_population,
    )


def ts_time_wsum_topn(
    time: DataProxy,
    feature: DataProxy,
    weight: DataProxy,
    sort_by: DataProxy,
    duration: str | int | float | timedelta,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Explicit time-window TopN weighted sum."""
    return _time_window_topn_pair(
        time, feature, weight, sort_by, duration, top, ascending, ties_method, _wsum_valid
    )


def _resolve_top_n(count: int, top: int | float) -> int:
    """Resolve integer or fractional top argument to an item count."""
    if count <= 0:
        return 0
    top_value = float(top)
    if top_value <= 0.0:
        return 0
    if top_value < 1.0:
        return max(1, min(count, int(math.ceil(count * top_value))))
    return max(1, min(count, int(round(top_value))))


def _top_indices(
    sort_values: np.ndarray,
    top: int | float,
    ascending: bool,
    ties_method: str,
) -> np.ndarray:
    """Select top-N row positions from a rolling sort vector."""
    valid_rows = np.flatnonzero(np.isfinite(sort_values))
    count = len(valid_rows)
    top_n = _resolve_top_n(count, top)
    if top_n == 0:
        return np.array([], dtype=int)

    scores = sort_values[valid_rows]
    positions = valid_rows.astype(float)
    method = ties_method.lower()

    if method == "all":
        primary = scores if ascending else -scores
        order = np.argsort(primary, kind="mergesort")
        cutoff = scores[order[top_n - 1]]
        if ascending:
            return valid_rows[scores <= cutoff]
        return valid_rows[scores >= cutoff]

    if method == "oldest":
        tie_key = positions
    elif method == "latest":
        tie_key = -positions
    else:
        raise ValueError("ties_method must be one of {'latest', 'oldest', 'all'}")

    primary = scores if ascending else -scores
    order = np.lexsort((tie_key, primary))
    return valid_rows[order[:top_n]]


def _rolling_topn_unary(
    values: np.ndarray,
    sort_values: np.ndarray,
    window: int,
    top: int | float,
    ascending: bool,
    ties_method: str,
    reducer: Callable[[np.ndarray], float],
) -> np.ndarray:
    """Rolling top-N reducer on one value vector."""
    out = np.full(len(values), np.nan, dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        local_sort = sort_values[start:index + 1]
        chosen = _top_indices(local_sort, top, ascending, ties_method)
        if len(chosen) == 0:
            continue
        out[index] = reducer(values[start:index + 1][chosen])
    return out


def _rolling_topn_pair(
    values1: np.ndarray,
    values2: np.ndarray,
    sort_values: np.ndarray,
    window: int,
    top: int | float,
    ascending: bool,
    ties_method: str,
    reducer: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Rolling top-N reducer on two value vectors."""
    out = np.full(len(values1), np.nan, dtype=float)
    for index in range(len(values1)):
        start = max(0, index - window + 1)
        local_sort = sort_values[start:index + 1]
        chosen = _top_indices(local_sort, top, ascending, ties_method)
        if len(chosen) == 0:
            continue
        out[index] = reducer(
            values1[start:index + 1][chosen],
            values2[start:index + 1][chosen],
        )
    return out


def _cumulative_topn_unary(
    values: np.ndarray,
    sort_values: np.ndarray,
    top: int | float,
    ascending: bool,
    ties_method: str,
    reducer: Callable[[np.ndarray], float],
) -> np.ndarray:
    """Cumulative TopN reducer on one value vector."""
    out = np.full(len(values), np.nan, dtype=float)
    for index in range(len(values)):
        chosen = _top_indices(sort_values[:index + 1], top, ascending, ties_method)
        if len(chosen) > 0:
            out[index] = reducer(values[:index + 1][chosen])
    return out


def _cumulative_topn_pair(
    values1: np.ndarray,
    values2: np.ndarray,
    sort_values: np.ndarray,
    top: int | float,
    ascending: bool,
    ties_method: str,
    reducer: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Cumulative TopN reducer on two value vectors."""
    out = np.full(len(values1), np.nan, dtype=float)
    for index in range(len(values1)):
        chosen = _top_indices(sort_values[:index + 1], top, ascending, ties_method)
        if len(chosen) > 0:
            out[index] = reducer(
                values1[:index + 1][chosen],
                values2[:index + 1][chosen],
            )
    return out


def ts_sum_topn(
    feature: DataProxy,
    sort_by: DataProxy,
    window: int | float,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Rolling sum of rows selected by top-N sort_by values."""
    size = _as_window(window)
    joined = _join_two(feature, sort_by)

    def reducer(values: np.ndarray) -> float:
        valid = _valid(values)
        if len(valid) == 0:
            return np.nan
        return float(np.sum(valid))

    return _per_symbol_joined(
        joined,
        ("x", "y"),
        lambda x, sort_values: _rolling_topn_unary(
            x, sort_values, size, top, ascending, ties_method, reducer
        ),
    )


def ts_mean_topn(
    feature: DataProxy,
    sort_by: DataProxy,
    window: int | float,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Rolling mean of rows selected by top-N sort_by values."""
    size = _as_window(window)
    joined = _join_two(feature, sort_by)

    def reducer(values: np.ndarray) -> float:
        valid = _valid(values)
        if len(valid) == 0:
            return np.nan
        return float(np.mean(valid))

    return _per_symbol_joined(
        joined,
        ("x", "y"),
        lambda x, sort_values: _rolling_topn_unary(
            x, sort_values, size, top, ascending, ties_method, reducer
        ),
    )


def ts_var_topn(
    feature: DataProxy,
    sort_by: DataProxy,
    window: int | float,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Rolling sample variance of rows selected by top-N sort_by values."""
    size = _as_window(window)
    joined = _join_two(feature, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y"),
        lambda x, sort_values: _rolling_topn_unary(
            x, sort_values, size, top, ascending, ties_method, lambda item: _var_valid(item, ddof=1)
        ),
    )


def ts_varp_topn(
    feature: DataProxy,
    sort_by: DataProxy,
    window: int | float,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Rolling population variance of rows selected by top-N sort_by values."""
    size = _as_window(window)
    joined = _join_two(feature, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y"),
        lambda x, sort_values: _rolling_topn_unary(
            x, sort_values, size, top, ascending, ties_method, lambda item: _var_valid(item, ddof=0)
        ),
    )


def ts_std_topn(
    feature: DataProxy,
    sort_by: DataProxy,
    window: int | float,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Rolling sample standard deviation of rows selected by top-N sort_by values."""
    size = _as_window(window)
    joined = _join_two(feature, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y"),
        lambda x, sort_values: _rolling_topn_unary(
            x, sort_values, size, top, ascending, ties_method, lambda item: _std_valid(item, ddof=1)
        ),
    )


def ts_stdp_topn(
    feature: DataProxy,
    sort_by: DataProxy,
    window: int | float,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Rolling population standard deviation of rows selected by top-N sort_by values."""
    size = _as_window(window)
    joined = _join_two(feature, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y"),
        lambda x, sort_values: _rolling_topn_unary(
            x, sort_values, size, top, ascending, ties_method, lambda item: _std_valid(item, ddof=0)
        ),
    )


def ts_skew_topn(
    feature: DataProxy,
    sort_by: DataProxy,
    window: int | float,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Rolling skewness of rows selected by top-N sort_by values."""
    size = _as_window(window)
    joined = _join_two(feature, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y"),
        lambda x, sort_values: _rolling_topn_unary(
            x, sort_values, size, top, ascending, ties_method, _skew_valid
        ),
    )


def ts_kurtosis_topn(
    feature: DataProxy,
    sort_by: DataProxy,
    window: int | float,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Rolling excess kurtosis of rows selected by top-N sort_by values."""
    size = _as_window(window)
    joined = _join_two(feature, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y"),
        lambda x, sort_values: _rolling_topn_unary(
            x, sort_values, size, top, ascending, ties_method, _kurtosis_valid
        ),
    )


def ts_wsum_topn(
    feature: DataProxy,
    weight: DataProxy,
    sort_by: DataProxy,
    window: int | float,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Rolling weighted sum of rows selected by top-N sort_by values."""
    size = _as_window(window)
    joined = _join_three(feature, weight, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y", "z"),
        lambda x, weight_values, sort_values: _rolling_topn_pair(
            x, weight_values, sort_values, size, top, ascending, ties_method, _wsum_valid
        ),
    )


def ts_wavg_topn(
    feature: DataProxy,
    weight: DataProxy,
    sort_by: DataProxy,
    window: int | float,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Rolling weighted average of rows selected by top-N sort_by values."""
    size = _as_window(window)
    joined = _join_three(feature, weight, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y", "z"),
        lambda x, weight_values, sort_values: _rolling_topn_pair(
            x, weight_values, sort_values, size, top, ascending, ties_method, _wavg_valid
        ),
    )


def ts_cov_topn(
    feature1: DataProxy,
    feature2: DataProxy,
    sort_by: DataProxy,
    window: int | float,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Rolling covariance of rows selected by top-N sort_by values."""
    size = _as_window(window)
    joined = _join_three(feature1, feature2, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y", "z"),
        lambda x, y, sort_values: _rolling_topn_pair(
            x, y, sort_values, size, top, ascending, ties_method, _cov_population
        ),
    )


def ts_beta_topn(
    feature_y: DataProxy,
    feature_x: DataProxy,
    sort_by: DataProxy,
    window: int | float,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Rolling beta of rows selected by top-N sort_by values."""
    size = _as_window(window)
    joined = _join_three(feature_y, feature_x, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y", "z"),
        lambda y, x, sort_values: _rolling_topn_pair(
            y, x, sort_values, size, top, ascending, ties_method, _beta_population
        ),
    )


def ts_corr_topn(
    feature1: DataProxy,
    feature2: DataProxy,
    sort_by: DataProxy,
    window: int | float,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Rolling correlation of rows selected by top-N sort_by values."""
    size = _as_window(window)
    joined = _join_three(feature1, feature2, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y", "z"),
        lambda x, y, sort_values: _rolling_topn_pair(
            x, y, sort_values, size, top, ascending, ties_method, _corr_population
        ),
    )


def ts_cumsum_topn(
    feature: DataProxy,
    sort_by: DataProxy,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Cumulative TopN sum."""
    joined = _join_two(feature, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y"),
        lambda x, sort_values: _cumulative_topn_unary(
            x, sort_values, top, ascending, ties_method, _sum_valid
        ),
    )


def ts_cummean_topn(
    feature: DataProxy,
    sort_by: DataProxy,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Cumulative TopN mean."""
    joined = _join_two(feature, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y"),
        lambda x, sort_values: _cumulative_topn_unary(
            x, sort_values, top, ascending, ties_method, _mean_valid
        ),
    )


def ts_cumvar_topn(
    feature: DataProxy,
    sort_by: DataProxy,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Cumulative TopN sample variance."""
    joined = _join_two(feature, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y"),
        lambda x, sort_values: _cumulative_topn_unary(
            x, sort_values, top, ascending, ties_method, lambda item: _var_valid(item, ddof=1)
        ),
    )


def ts_cumvarp_topn(
    feature: DataProxy,
    sort_by: DataProxy,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Cumulative TopN population variance."""
    joined = _join_two(feature, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y"),
        lambda x, sort_values: _cumulative_topn_unary(
            x, sort_values, top, ascending, ties_method, lambda item: _var_valid(item, ddof=0)
        ),
    )


def ts_cumstd_topn(
    feature: DataProxy,
    sort_by: DataProxy,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Cumulative TopN sample standard deviation."""
    joined = _join_two(feature, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y"),
        lambda x, sort_values: _cumulative_topn_unary(
            x, sort_values, top, ascending, ties_method, lambda item: _std_valid(item, ddof=1)
        ),
    )


def ts_cumstdp_topn(
    feature: DataProxy,
    sort_by: DataProxy,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Cumulative TopN population standard deviation."""
    joined = _join_two(feature, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y"),
        lambda x, sort_values: _cumulative_topn_unary(
            x, sort_values, top, ascending, ties_method, lambda item: _std_valid(item, ddof=0)
        ),
    )


def ts_cumcov_topn(
    feature1: DataProxy,
    feature2: DataProxy,
    sort_by: DataProxy,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Cumulative TopN covariance."""
    joined = _join_three(feature1, feature2, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y", "z"),
        lambda x, y, sort_values: _cumulative_topn_pair(
            x, y, sort_values, top, ascending, ties_method, _cov_population
        ),
    )


def ts_cumcorr_topn(
    feature1: DataProxy,
    feature2: DataProxy,
    sort_by: DataProxy,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Cumulative TopN correlation."""
    joined = _join_three(feature1, feature2, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y", "z"),
        lambda x, y, sort_values: _cumulative_topn_pair(
            x, y, sort_values, top, ascending, ties_method, _corr_population
        ),
    )


def ts_cumbeta_topn(
    feature_y: DataProxy,
    feature_x: DataProxy,
    sort_by: DataProxy,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Cumulative TopN beta."""
    joined = _join_three(feature_y, feature_x, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y", "z"),
        lambda y, x, sort_values: _cumulative_topn_pair(
            y, x, sort_values, top, ascending, ties_method, _beta_population
        ),
    )


def ts_cumwsum_topn(
    feature: DataProxy,
    weight: DataProxy,
    sort_by: DataProxy,
    top: int | float,
    ascending: bool = False,
    ties_method: str = "latest",
) -> DataProxy:
    """Cumulative TopN weighted sum."""
    joined = _join_three(feature, weight, sort_by)
    return _per_symbol_joined(
        joined,
        ("x", "y", "z"),
        lambda x, weight_values, sort_values: _cumulative_topn_pair(
            x, weight_values, sort_values, top, ascending, ties_method, _wsum_valid
        ),
    )


def _aligned_value_array(
    base: pl.DataFrame,
    value: DataProxy | int | float,
    name: str,
) -> tuple[pl.DataFrame, str]:
    """Attach a scalar or DataProxy value column to a base frame."""
    if isinstance(value, DataProxy):
        return (
            base.join(value.df.rename({"data": name}), on=["datetime", "vt_symbol"], how="inner"),
            name,
        )
    return base.with_columns(pl.lit(float(value)).alias(name)), name


def _iterate_previous(values: np.ndarray, method: str) -> float:
    """Aggregate previous output state."""
    valid = _valid(values)
    if len(valid) == 0:
        return np.nan
    method = method.lower()
    if method == "sum":
        return float(np.sum(valid))
    if method == "mean":
        return float(np.mean(valid))
    if method == "max":
        return float(np.max(valid))
    if method == "min":
        return float(np.min(valid))
    if method in {"last", "ffill"}:
        return float(valid[-1])
    raise ValueError("iterate method must be one of {'sum', 'mean', 'max', 'min', 'last', 'ffill'}")


def _linear_combine(
    value: float,
    state: float,
    coeff_x: float,
    coeff_state: float,
) -> float:
    """Combine current value and previous state while respecting NaN terms."""
    result = 0.0
    if coeff_x != 0.0:
        if np.isnan(value):
            return np.nan
        result += coeff_x * value
    if coeff_state != 0.0:
        if np.isnan(state):
            return np.nan
        result += coeff_state * state
    return result


def ts_state_iterate(
    feature: DataProxy,
    initial: DataProxy | int | float = 0.0,
    initial_window: int | float = 1,
    iterate: str = "last",
    iterate_window: int | float = 1,
    coeff_x: float = 1.0,
    coeff_state: float = 1.0,
) -> DataProxy:
    """Recursive state update from current feature and previous output state."""
    base = feature.df.with_row_index("_left_id").rename({"data": "x"})
    joined, initial_name = _aligned_value_array(base, initial, "_initial")
    joined = joined.sort("_left_id").with_row_index("_result_id")
    init_size = _as_window(initial_window)
    state_size = _as_window(iterate_window)

    def func(values: np.ndarray, initial_values: np.ndarray) -> np.ndarray:
        out = np.full(len(values), np.nan, dtype=float)
        for index, value in enumerate(values):
            if index < init_size:
                out[index] = initial_values[index]
                continue
            start = max(0, index - state_size)
            state = _iterate_previous(out[start:index], iterate)
            out[index] = _linear_combine(value, state, float(coeff_x), float(coeff_state))
        return out

    return _per_symbol_joined(joined, ("x", initial_name), func)


def ts_conditional_iterate(
    condition: DataProxy,
    true_value: DataProxy | int | float,
    false_iterate: str = "last",
    iterate_window: int | float = 1,
) -> DataProxy:
    """Recursive conditional update.

    If condition is true, output true_value. Otherwise output an aggregation of
    previous outputs.
    """
    base = condition.df.with_row_index("_left_id").rename({"data": "_condition"})
    joined, value_name = _aligned_value_array(base, true_value, "_true")
    joined = joined.sort("_left_id").with_row_index("_result_id")
    state_size = _as_window(iterate_window)
    out = np.full(joined.height, np.nan, dtype=float)

    for group in joined.partition_by("vt_symbol", maintain_order=True):
        rows = group["_result_id"].to_numpy()
        condition_values = _series_to_bool(group["_condition"])
        true_values = _series_to_float(group[value_name])
        local_out = np.full(len(condition_values), np.nan, dtype=float)
        for index, flag in enumerate(condition_values):
            if flag:
                local_out[index] = true_values[index]
            else:
                start = max(0, index - state_size)
                local_out[index] = _iterate_previous(local_out[start:index], false_iterate)
        out[rows] = local_out

    return _result_from_values(joined, out)
