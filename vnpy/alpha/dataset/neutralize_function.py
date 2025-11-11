"""
Neutralization Operators (Market/Industry/Baseline)

All functions return a DataProxy consistent with expression engine.

Notes:
- Category neutralization expects a categorical DataProxy (e.g., industry code)
  whose underlying df has columns: datetime, vt_symbol, data (string category).
- OLS neutralization (single exposure) is cross-sectional per-date regression:
  y_adj = (y - mean_y) - beta * (x - mean_x), beta = cov(y,x)/var(x).
"""

from __future__ import annotations

import polars as pl

from .utility import DataProxy


def cs_demean(feature: DataProxy) -> DataProxy:
    """Cross-sectional de-mean per date (market neutral)."""
    df = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        (pl.col("data") - pl.col("data").mean().over("datetime")).alias("data")
    )
    return DataProxy(df)


def cs_zscore(feature: DataProxy) -> DataProxy:
    """Cross-sectional z-score per date: (x - mean)/std."""
    df = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        (
            (pl.col("data") - pl.col("data").mean().over("datetime")) /
            (pl.col("data").std().over("datetime") + 1e-12)
        ).alias("data")
    )
    return DataProxy(df)


def cs_demean_by_category(feature: DataProxy, category: DataProxy) -> DataProxy:
    """De-mean by category per date: x - mean(x | date, cat)."""
    base = feature.df.join(category.df.rename({"data": "cat"}), on=["datetime", "vt_symbol"], how="inner")
    df = base.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        (pl.col("data") - pl.col("data").mean().over(["datetime", "cat"])).alias("data")
    )
    return DataProxy(df)


def cs_zscore_by_category(feature: DataProxy, category: DataProxy) -> DataProxy:
    """Z-score by category per date."""
    base = feature.df.join(category.df.rename({"data": "cat"}), on=["datetime", "vt_symbol"], how="inner")
    df = base.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        (
            (pl.col("data") - pl.col("data").mean().over(["datetime", "cat"])) /
            (pl.col("data").std().over(["datetime", "cat"]) + 1e-12)
        ).alias("data")
    )
    return DataProxy(df)


def cs_neutralize_ols1(feature: DataProxy, exposure: DataProxy) -> DataProxy:
    """Cross-sectional OLS neutralization with single exposure per date.

    y_adj = (y - mean_y) - beta * (x - mean_x), where beta = cov(y,x)/var(x).
    """
    merged = feature.df.join(exposure.df.rename({"data": "x"}), on=["datetime", "vt_symbol"], how="inner")
    df = merged.with_columns(
        pl.col("data").alias("y"),
        pl.col("x").alias("x"),
    ).with_columns(
        pl.col("y").mean().over("datetime").alias("mean_y"),
        pl.col("x").mean().over("datetime").alias("mean_x"),
    ).with_columns(
        (pl.col("x") - pl.col("mean_x")).pow(2).sum().over("datetime").alias("sum_x2"),
        ((pl.col("x") - pl.col("mean_x")) * (pl.col("y") - pl.col("mean_y"))).sum().over("datetime").alias("sum_xy"),
    ).with_columns(
        (pl.col("sum_xy") / (pl.col("sum_x2") + 1e-12)).alias("beta")
    ).with_columns(
        ((pl.col("y") - pl.col("mean_y")) - pl.col("beta") * (pl.col("x") - pl.col("mean_x"))).alias("data")
    ).select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data")
    )
    return DataProxy(df)

