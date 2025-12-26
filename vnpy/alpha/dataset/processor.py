from datetime import datetime

import numpy as np
import polars as pl

from .neutralize_function import (
    cs_demean_by_category,
    cs_neutralize_ols1,
    cs_zscore_by_category,
)
from .utility import DataProxy, to_datetime


def process_drop_na(df: pl.DataFrame, names: list[str] | None = None) -> pl.DataFrame:
    """Remove rows with missing values"""
    if not names:
        names = df.columns[2:-1]

    for name in names:
        # 统一将 NaN/Inf 视为缺失，便于后续 drop_nulls 清理
        df = df.with_columns(
            pl.when(pl.col(name).is_infinite())
            .then(None)
            .otherwise(pl.col(name))
            .alias(name)
        )
        df = df.with_columns(pl.col(name).fill_nan(None))
    df = df.drop_nulls(subset=names)
    return df


def process_fill_na(
    df: pl.DataFrame,
    fill_value: float,
    fill_label: bool = True,
    names: list[str] | None = None,
) -> pl.DataFrame:
    """Fill missing values (explicit names when fill_label=False)."""
    if fill_label:
        df = df.fill_null(fill_value)
        df = df.fill_nan(fill_value)
    else:
        if names is None:
            names = df.columns[2:-1]
        if not names:
            return df
        df = df.with_columns(
            [pl.col(col).fill_null(fill_value).fill_nan(fill_value) for col in names]
        )
    return df


def process_cs_norm(
    df: pl.DataFrame,
    names: list[str],
    method: str         # robust/zscore
) -> pl.DataFrame:
    """Cross-sectional normalization"""
    _df: pl.DataFrame = df.fill_nan(None)

    # Median method
    if method == "robust":
        for col in names:
            df = df.with_columns(
                _df.select(
                    (pl.col(col) - pl.col(col).median()).over("datetime").alias(col),
                )
            )

            df = df.with_columns(
                df.select(
                    pl.col(col).abs().median().over("datetime").alias("mad"),
                )
            )

            df = df.with_columns(
                (pl.col(col) / pl.col("mad") / 1.4826).clip(-3, 3).alias(col)
            ).drop(["mad"])
    # Z-Score method
    else:
        for col in names:
            df = df.with_columns(
                _df.select(
                    pl.col(col).mean().over("datetime").alias("mean"),
                    pl.col(col).std().over("datetime").alias("std"),
                )
            )

            df = df.with_columns(
                (pl.col(col) - pl.col("mean")) / pl.col("std").alias(col)
            ).drop(["mean", "std"])

    return df


def process_robust_zscore_norm(
    df: pl.DataFrame,
    fit_start_time: datetime | str | None = None,
    fit_end_time: datetime | str | None = None,
    clip_outlier: bool = True,
    names: list[str] | None = None,
) -> pl.DataFrame:
    """Robust Z-Score normalization (explicit names override default slice)."""
    _df: pl.DataFrame = df.fill_nan(None)

    if fit_start_time and fit_end_time:
        fit_start_time = to_datetime(fit_start_time)
        fit_end_time = to_datetime(fit_end_time)
        _df = _df.filter((pl.col("datetime") >= fit_start_time) & (pl.col("datetime") <= fit_end_time))

    if names is None:
        names = df.columns[2:-1]
    if not names:
        return df
    cols = names
    X = _df.select(cols).to_numpy()

    mean_train = np.nanmedian(X, axis=0)
    std_train = np.nanmedian(np.abs(X - mean_train), axis=0)
    std_train += 1e-12
    std_train *= 1.4826

    for name in cols:
        normalized_col = (
            (pl.col(name) - mean_train[cols.index(name)]) / std_train[cols.index(name)]
        ).cast(pl.Float64)

        if clip_outlier:
            normalized_col = normalized_col.clip(-3, 3)

        df = df.with_columns(normalized_col.alias(name))

    return df


def process_cs_rank_norm(df: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    """Cross-sectional rank normalization"""
    _df: pl.DataFrame = df.fill_nan(None)

    _df = _df.with_columns([
        ((pl.col(col).rank("average").over("datetime") / pl.col("datetime").count().over("datetime")) - 0.5) * 3.46
        for col in names
    ])

    df = df.with_columns([
        _df[col].alias(col) for col in names
    ])

    return df


def process_cs_neutralize_ols(
    df: pl.DataFrame,
    exposure_col: str,
    names: list[str] | None = None,
    log_exposure: bool = False,
) -> pl.DataFrame:
    """Cross-sectional OLS neutralization for multiple columns."""
    if exposure_col not in df.columns:
        return df

    if names is None:
        exclude = {"datetime", "vt_symbol", "label", exposure_col}
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        names = [c for c in numeric_cols if c not in exclude]

    if not names:
        return df

    exposure_expr = (
        pl.when(pl.col(exposure_col).is_not_null() & (pl.col(exposure_col) > 0))
        .then(pl.col(exposure_col))
        .otherwise(None)
    )
    if log_exposure:
        exposure_expr = exposure_expr.log1p()

    exposure_df = df.select(
        "datetime",
        "vt_symbol",
        exposure_expr.cast(pl.Float64).alias(exposure_col),
    )
    exposure = DataProxy(exposure_df)

    results: list[pl.DataFrame] = []
    for name in names:
        feature_df = df.select(
            "datetime",
            "vt_symbol",
            pl.col(name).cast(pl.Float64, strict=False).alias(name),
        )
        feature = DataProxy(feature_df)
        adjusted = cs_neutralize_ols1(feature, exposure).df.rename({"data": name})
        results.append(adjusted)

    base = df.drop(names)
    for adjusted in results:
        base = base.join(adjusted, on=["datetime", "vt_symbol"], how="left")

    return base.select(df.columns)


def process_cap_neutralize(
    df: pl.DataFrame,
    cap_col: str = "cap",
    names: list[str] | None = None,
    log_cap: bool = True,
) -> pl.DataFrame:
    """Market-cap neutralization using log(cap) as exposure."""
    return process_cs_neutralize_ols(
        df=df,
        exposure_col=cap_col,
        names=names,
        log_exposure=log_cap,
    )


def process_cs_neutralize_by_category(
    df: pl.DataFrame,
    category_col: str,
    names: list[str] | None = None,
    method: str = "demean",
    fill_missing: str | None = None,
) -> pl.DataFrame:
    """Cross-sectional neutralization by category per date."""
    if category_col not in df.columns:
        return df

    if names is None:
        exclude = {"datetime", "vt_symbol", "label", category_col}
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        names = [c for c in numeric_cols if c not in exclude]

    if not names:
        return df

    category_expr = pl.col(category_col).cast(pl.Utf8)
    if fill_missing is not None:
        category_expr = category_expr.fill_null(fill_missing)

    category_df = df.select(
        "datetime",
        "vt_symbol",
        category_expr.alias(category_col),
    )
    category = DataProxy(category_df)

    results: list[pl.DataFrame] = []
    for name in names:
        feature_df = df.select(
            "datetime",
            "vt_symbol",
            pl.col(name).cast(pl.Float64, strict=False).alias(name),
        )
        feature = DataProxy(feature_df)
        if method == "zscore":
            adjusted = cs_zscore_by_category(feature, category)
        else:
            adjusted = cs_demean_by_category(feature, category)
        results.append(adjusted.df.rename({"data": name}))

    base = df.drop(names)
    for adjusted in results:
        base = base.join(adjusted, on=["datetime", "vt_symbol"], how="left")

    return base.select(df.columns)


def process_industry_neutralize(
    df: pl.DataFrame,
    industry_col: str = "industry",
    names: list[str] | None = None,
    method: str = "demean",
    fill_missing: str | None = None,
) -> pl.DataFrame:
    """Industry neutralization using category-based de-mean/zscore."""
    return process_cs_neutralize_by_category(
        df=df,
        category_col=industry_col,
        names=names,
        method=method,
        fill_missing=fill_missing,
    )
