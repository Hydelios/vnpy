"""
TA and EWM extension operators.

This module keeps TA/EWM additions separate from the original operator files.
All functions follow the DataProxy contract:
input/output frames contain ["datetime", "vt_symbol", "data"].
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl
import talib

from .utility import DataProxy


def _as_window(window: int | float) -> int:
    """Normalize window arguments."""
    result = int(round(float(window)))
    if result < 1:
        raise ValueError("window must be positive")
    return result


def _validate_alpha(alpha: float) -> float:
    """Validate alpha for EWM operators."""
    result = float(alpha)
    if result <= 0.0 or result > 1.0:
        raise ValueError("alpha must be in (0, 1]")
    return result


def _resolve_ewm_kwargs(
    alpha: float | None = None,
    *,
    com: float | None = None,
    span: float | None = None,
    halfLife: float | None = None,
    minPeriods: int | None = None,
    adjust: bool = False,
    ignoreNA: bool = False,
    times: Any | None = None,
) -> dict[str, Any]:
    """Normalize DolphinDB/pandas-style EWM parameters.

    The positional smoothing argument remains alpha for compatibility with
    existing alpha expressions.
    """
    if times is not None:
        raise NotImplementedError("EWM times parameter is not supported")

    decay_params = {
        "com": com,
        "span": span,
        "halflife": halfLife,
        "alpha": alpha,
    }
    specified = {name: value for name, value in decay_params.items() if value is not None}
    if len(specified) != 1:
        raise ValueError("exactly one of com/span/halfLife/alpha must be specified")

    normalized: dict[str, Any] = {
        "adjust": bool(adjust),
        "ignore_na": bool(ignoreNA),
    }
    if minPeriods is not None:
        periods = int(minPeriods)
        if periods < 0:
            raise ValueError("minPeriods must be non-negative")
        normalized["min_periods"] = periods

    name, value = next(iter(specified.items()))
    if name == "alpha":
        normalized["alpha"] = _validate_alpha(float(value))
    else:
        normalized[name] = float(value)

    return normalized


def _series_to_float(series: pl.Series) -> np.ndarray:
    """Convert a Polars series to a float numpy array."""
    values: list[float] = []
    for value in series.to_list():
        if value is None:
            values.append(np.nan)
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            values.append(np.nan)
            continue
        values.append(numeric if np.isfinite(numeric) else np.nan)
    return np.array(values, dtype=float)


def _result_from_values(
    base_df: pl.DataFrame,
    values: np.ndarray,
    interval: str | None = None,
) -> DataProxy:
    """Build a DataProxy result from numeric values."""
    result = base_df.select("datetime", "vt_symbol").with_columns(
        pl.Series("data", [float(value) if np.isfinite(value) else np.nan for value in values])
    )
    return DataProxy(result, interval=interval)


def _per_symbol_unary(
    feature: DataProxy,
    func: Callable[[np.ndarray], np.ndarray],
) -> DataProxy:
    """Apply a vector function per symbol while preserving input row order."""
    work = feature.df.with_row_index("_result_id")
    out = np.full(work.height, np.nan, dtype=float)

    for group in work.partition_by("vt_symbol", maintain_order=True):
        rows = group["_result_id"].to_numpy()
        values = _series_to_float(group["data"])
        result = np.asarray(func(values), dtype=float)
        out[rows] = result

    return _result_from_values(feature.df, out, interval=feature.interval)


def _join_two(feature1: DataProxy, feature2: DataProxy) -> pl.DataFrame:
    """Join two features and preserve feature1 row order."""
    left = feature1.df.with_row_index("_left_id").rename({"data": "x"})
    right = feature2.df.rename({"data": "y"})
    return (
        left.join(right, on=["datetime", "vt_symbol"], how="inner")
        .sort("_left_id")
        .with_row_index("_result_id")
    )


def _per_symbol_pair(
    feature1: DataProxy,
    feature2: DataProxy,
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> DataProxy:
    """Apply a pair vector function per symbol after datetime/vt_symbol alignment."""
    joined = _join_two(feature1, feature2)
    out = np.full(joined.height, np.nan, dtype=float)

    for group in joined.partition_by("vt_symbol", maintain_order=True):
        rows = group["_result_id"].to_numpy()
        x = _series_to_float(group["x"])
        y = _series_to_float(group["y"])
        result = np.asarray(func(x, y), dtype=float)
        out[rows] = result

    interval = feature1.interval if feature1.interval == feature2.interval else feature1.interval
    return _result_from_values(joined, out, interval=interval)


def _talib_unary(
    feature: DataProxy,
    func: Callable[..., np.ndarray],
    window: int | float,
    *args: float,
) -> DataProxy:
    """Apply a TA-Lib unary function per symbol."""
    size = _as_window(window)

    def apply(values: np.ndarray) -> np.ndarray:
        return np.asarray(func(values, timeperiod=size, *args), dtype=float)

    return _per_symbol_unary(feature, apply)


def _ewm_mean(
    feature: DataProxy,
    alpha: float | None = None,
    *,
    com: float | None = None,
    span: float | None = None,
    halfLife: float | None = None,
    minPeriods: int | None = None,
    adjust: bool = False,
    ignoreNA: bool = False,
    times: Any | None = None,
) -> DataProxy:
    """Exponentially weighted mean per symbol."""
    kwargs = _resolve_ewm_kwargs(
        alpha,
        com=com,
        span=span,
        halfLife=halfLife,
        minPeriods=minPeriods,
        adjust=adjust,
        ignoreNA=ignoreNA,
        times=times,
    )

    def apply(values: np.ndarray) -> np.ndarray:
        series = pd.Series(values, dtype="float64")
        return series.ewm(**kwargs).mean().to_numpy()

    return _per_symbol_unary(feature, apply)


def ewmMean(
    feature: DataProxy,
    alpha: float | None = None,
    *,
    com: float | None = None,
    span: float | None = None,
    halfLife: float | None = None,
    minPeriods: int | None = None,
    adjust: bool = False,
    ignoreNA: bool = False,
    times: Any | None = None,
) -> DataProxy:
    """DolphinDB-style EWM mean operator."""
    return _ewm_mean(
        feature,
        alpha,
        com=com,
        span=span,
        halfLife=halfLife,
        minPeriods=minPeriods,
        adjust=adjust,
        ignoreNA=ignoreNA,
        times=times,
    )


def ts_ewm_var(
    feature: DataProxy,
    alpha: float | None = None,
    *,
    com: float | None = None,
    span: float | None = None,
    halfLife: float | None = None,
    minPeriods: int | None = None,
    adjust: bool = False,
    ignoreNA: bool = False,
    bias: bool = False,
    times: Any | None = None,
) -> DataProxy:
    """Exponentially weighted variance per symbol."""
    kwargs = _resolve_ewm_kwargs(
        alpha,
        com=com,
        span=span,
        halfLife=halfLife,
        minPeriods=minPeriods,
        adjust=adjust,
        ignoreNA=ignoreNA,
        times=times,
    )

    def apply(values: np.ndarray) -> np.ndarray:
        series = pd.Series(values, dtype="float64")
        return series.ewm(**kwargs).var(bias=bias).to_numpy()

    return _per_symbol_unary(feature, apply)


def ts_ewm_std(
    feature: DataProxy,
    alpha: float | None = None,
    *,
    com: float | None = None,
    span: float | None = None,
    halfLife: float | None = None,
    minPeriods: int | None = None,
    adjust: bool = False,
    ignoreNA: bool = False,
    bias: bool = False,
    times: Any | None = None,
) -> DataProxy:
    """Exponentially weighted standard deviation per symbol."""
    variance = ts_ewm_var(
        feature,
        alpha,
        com=com,
        span=span,
        halfLife=halfLife,
        minPeriods=minPeriods,
        adjust=adjust,
        ignoreNA=ignoreNA,
        bias=bias,
        times=times,
    )
    result = variance.df.with_columns(pl.col("data").sqrt().alias("data"))
    return DataProxy(result, interval=feature.interval)


def ts_ewm_cov(
    feature1: DataProxy,
    feature2: DataProxy,
    alpha: float | None = None,
    *,
    com: float | None = None,
    span: float | None = None,
    halfLife: float | None = None,
    minPeriods: int | None = None,
    adjust: bool = False,
    ignoreNA: bool = False,
    bias: bool = False,
    times: Any | None = None,
) -> DataProxy:
    """Exponentially weighted covariance per symbol."""
    kwargs = _resolve_ewm_kwargs(
        alpha,
        com=com,
        span=span,
        halfLife=halfLife,
        minPeriods=minPeriods,
        adjust=adjust,
        ignoreNA=ignoreNA,
        times=times,
    )

    def apply(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        left = pd.Series(x, dtype="float64")
        right = pd.Series(y, dtype="float64")
        return left.ewm(**kwargs).cov(right, bias=bias).to_numpy()

    return _per_symbol_pair(feature1, feature2, apply)


def ts_ewm_corr(
    feature1: DataProxy,
    feature2: DataProxy,
    alpha: float | None = None,
    *,
    com: float | None = None,
    span: float | None = None,
    halfLife: float | None = None,
    minPeriods: int | None = None,
    adjust: bool = False,
    ignoreNA: bool = False,
    times: Any | None = None,
) -> DataProxy:
    """Exponentially weighted correlation per symbol."""
    kwargs = _resolve_ewm_kwargs(
        alpha,
        com=com,
        span=span,
        halfLife=halfLife,
        minPeriods=minPeriods,
        adjust=adjust,
        ignoreNA=ignoreNA,
        times=times,
    )

    def apply(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        left = pd.Series(x, dtype="float64")
        right = pd.Series(y, dtype="float64")
        return left.ewm(**kwargs).corr(right).to_numpy()

    return _per_symbol_pair(feature1, feature2, apply)


def ta_wma(feature: DataProxy, window: int | float) -> DataProxy:
    """TA-Lib weighted moving average."""
    return _talib_unary(feature, talib.WMA, window)


def ta_dema(feature: DataProxy, window: int | float) -> DataProxy:
    """TA-Lib double exponential moving average."""
    return _talib_unary(feature, talib.DEMA, window)


def ta_tema(feature: DataProxy, window: int | float) -> DataProxy:
    """TA-Lib triple exponential moving average."""
    return _talib_unary(feature, talib.TEMA, window)


def ta_trima(feature: DataProxy, window: int | float) -> DataProxy:
    """TA-Lib triangular moving average."""
    return _talib_unary(feature, talib.TRIMA, window)


def ta_kama(feature: DataProxy, window: int | float) -> DataProxy:
    """TA-Lib Kaufman adaptive moving average."""
    return _talib_unary(feature, talib.KAMA, window)


def ta_t3(
    feature: DataProxy,
    window: int | float,
    vfactor: float = 0.7,
    *,
    vFactor: float | None = None,
) -> DataProxy:
    """TA-Lib T3 moving average."""
    size = _as_window(window)
    factor = float(vfactor if vFactor is None else vFactor)

    def apply(values: np.ndarray) -> np.ndarray:
        return np.asarray(talib.T3(values, timeperiod=size, vfactor=factor), dtype=float)

    return _per_symbol_unary(feature, apply)


def ta_wilder(feature: DataProxy, window: int | float) -> DataProxy:
    """Welles Wilder smoothing, implemented as EWM alpha=1/window."""
    size = _as_window(window)
    return _ewm_mean(feature, alpha=1.0 / size, adjust=False)


def ta_gema(
    feature: DataProxy,
    window: int | float,
    alpha: float,
    adjust: bool = False,
) -> DataProxy:
    """Generalized EMA with explicit alpha.

    The window argument is kept for DolphinDB-style expression compatibility;
    smoothing is controlled by alpha.
    """
    _as_window(window)
    return _ewm_mean(feature, alpha=alpha, adjust=adjust)


def ta_ma(
    feature: DataProxy,
    window: int | float,
    ma_type: int = 0,
    *,
    maType: int | None = None,
) -> DataProxy:
    """Moving-average dispatcher.

    ma_type mapping follows the common TA-Lib/DolphinDB convention:
    0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 8=T3.
    MAMA (7) is intentionally not supported.
    """
    size = _as_window(window)
    normalized = int(ma_type if maType is None else maType)

    if normalized == 0:
        return _talib_unary(feature, talib.SMA, size)
    if normalized == 1:
        return _talib_unary(feature, talib.EMA, size)
    if normalized == 2:
        return ta_wma(feature, size)
    if normalized == 3:
        return ta_dema(feature, size)
    if normalized == 4:
        return ta_tema(feature, size)
    if normalized == 5:
        return ta_trima(feature, size)
    if normalized == 6:
        return ta_kama(feature, size)
    if normalized == 8:
        return ta_t3(feature, size)

    raise ValueError("ta_ma: ma_type must be one of {0,1,2,3,4,5,6,8}; 7=MAMA is not supported")
