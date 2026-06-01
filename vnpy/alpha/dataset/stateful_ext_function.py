"""
Extension stateful operators.

These operators were added for paper-factor implementations and are kept out of
stateful_function.py so the core rolling/time-series operator file stays focused.
"""

from __future__ import annotations

import math

import numpy as np

from .stateful_function import (
    _as_window,
    _cov_population,
    _join_three,
    _join_two,
    _per_symbol_joined,
    _valid_pair,
)
from .utility import DataProxy


def _as_min_periods(value: int | float | None, default: int) -> int:
    """Normalize optional min_periods arguments."""
    if value is None:
        return default
    try:
        if math.isnan(float(value)):
            return default
    except (TypeError, ValueError):
        return default
    result = int(round(float(value)))
    return max(1, result)


def _ols_design(x: np.ndarray, intercept: bool) -> np.ndarray:
    """Build a one-variable OLS design matrix."""
    if intercept:
        return np.column_stack([np.ones(len(x), dtype=float), x])
    return x.reshape(-1, 1)


def _rolling_ols_residual_values(
    y: np.ndarray,
    x: np.ndarray,
    window: int,
    intercept: bool,
    min_periods: int,
) -> np.ndarray:
    """Rolling current-row residual from y ~ x."""
    out = np.full(len(y), np.nan, dtype=float)
    for index in range(len(y)):
        if not np.isfinite(y[index]) or not np.isfinite(x[index]):
            continue

        start = max(0, index - window + 1)
        y_window = y[start:index + 1]
        x_window = x[start:index + 1]
        valid = np.isfinite(y_window) & np.isfinite(x_window)
        if int(valid.sum()) < min_periods:
            continue

        design = _ols_design(x_window[valid], intercept)
        try:
            coeffs = np.linalg.lstsq(design, y_window[valid], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        current = np.asarray([1.0, x[index]], dtype=float) if intercept else np.asarray([x[index]])
        out[index] = y[index] - float(current @ coeffs)
    return out


def _rolling_ols_residual_corr_values(
    y: np.ndarray,
    x: np.ndarray,
    target: np.ndarray,
    window: int,
    intercept: bool,
    min_periods: int,
) -> np.ndarray:
    """Rolling corr between current-window OLS residuals and target."""
    out = np.full(len(y), np.nan, dtype=float)
    for index in range(len(y)):
        start = max(0, index - window + 1)
        y_window = y[start:index + 1]
        x_window = x[start:index + 1]
        target_window = target[start:index + 1]
        valid = (
            np.isfinite(y_window)
            & np.isfinite(x_window)
            & np.isfinite(target_window)
        )
        if int(valid.sum()) < min_periods:
            continue

        design = _ols_design(x_window[valid], intercept)
        try:
            coeffs = np.linalg.lstsq(design, y_window[valid], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        residual = y_window[valid] - design @ coeffs
        out[index] = _corr_population_min(residual, target_window[valid], min_periods)
    return out


def _corr_population_min(x: np.ndarray, y: np.ndarray, min_periods: int) -> float:
    """Pearson correlation with a minimum aligned sample count."""
    x_valid, y_valid = _valid_pair(x, y)
    if len(x_valid) < min_periods:
        return np.nan
    x_std = float(np.std(x_valid, ddof=0))
    y_std = float(np.std(y_valid, ddof=0))
    if x_std == 0.0 or y_std == 0.0:
        return np.nan
    return _cov_population(x_valid, y_valid) / (x_std * y_std)


def _quantile_side(side: str) -> str:
    """Normalize quantile mask side names."""
    normalized = str(side).strip().lower()
    if normalized in {"high", "upper", "right", "top", "gte", ">="}:
        return "high"
    if normalized in {"low", "lower", "left", "bottom", "lte", "<="}:
        return "low"
    raise ValueError("side must be 'high' or 'low'")


def _rolling_corr_quantile_mask_values(
    x: np.ndarray,
    y: np.ndarray,
    mask_by: np.ndarray,
    window: int,
    quantile: float,
    side: str,
    min_periods: int,
) -> np.ndarray:
    """Rolling corr after selecting rows by a rolling quantile threshold."""
    q = float(quantile)
    if q < 0.0 or q > 1.0:
        raise ValueError("quantile must be between 0 and 1")

    side_name = _quantile_side(side)
    out = np.full(len(x), np.nan, dtype=float)
    for index in range(len(x)):
        start = max(0, index - window + 1)
        mask_window = mask_by[start:index + 1]
        finite_mask = np.isfinite(mask_window)
        if not np.any(finite_mask):
            continue

        threshold = float(np.quantile(mask_window[finite_mask], q))
        if side_name == "high":
            selected = mask_window >= threshold
        else:
            selected = mask_window <= threshold

        out[index] = _corr_population_min(
            x[start:index + 1][selected],
            y[start:index + 1][selected],
            min_periods,
        )
    return out


def ts_rolling_ols_residual(
    feature_y: DataProxy,
    feature_x: DataProxy,
    window: int | float,
    intercept: bool = True,
    min_periods: int | float | None = None,
) -> DataProxy:
    """Rolling current-row residual from y ~ x."""
    size = _as_window(window)
    default_min = 3 if bool(intercept) else 2
    min_count = _as_min_periods(min_periods, default_min)
    joined = _join_two(feature_y, feature_x)
    return _per_symbol_joined(
        joined,
        ("x", "y"),
        lambda y, x: _rolling_ols_residual_values(
            y, x, size, bool(intercept), min_count
        ),
    )


def ts_rolling_ols_residual_corr(
    feature_y: DataProxy,
    feature_x: DataProxy,
    target: DataProxy,
    window: int | float,
    intercept: bool = True,
    min_periods: int | float | None = None,
) -> DataProxy:
    """Rolling corr between y~x residuals and target in the same window."""
    size = _as_window(window)
    default_min = 3 if bool(intercept) else 2
    min_count = _as_min_periods(min_periods, default_min)
    joined = _join_three(feature_y, feature_x, target)
    return _per_symbol_joined(
        joined,
        ("x", "y", "z"),
        lambda y, x, target_values: _rolling_ols_residual_corr_values(
            y, x, target_values, size, bool(intercept), min_count
        ),
    )


def ts_corr_quantile_mask(
    feature1: DataProxy,
    feature2: DataProxy,
    mask_by: DataProxy,
    window: int | float,
    quantile: int | float,
    side: str = "high",
    min_periods: int | float | None = None,
) -> DataProxy:
    """Rolling corr after keeping rows above/below a rolling quantile."""
    size = _as_window(window)
    min_count = _as_min_periods(min_periods, 2)
    joined = _join_three(feature1, feature2, mask_by)
    return _per_symbol_joined(
        joined,
        ("x", "y", "z"),
        lambda x, y, mask_values: _rolling_corr_quantile_mask_values(
            x, y, mask_values, size, float(quantile), side, min_count
        ),
    )
