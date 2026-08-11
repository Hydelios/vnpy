from __future__ import annotations

import polars as pl


def point_in_time_fill(
    df: pl.DataFrame,
    fields: list[str],
) -> pl.DataFrame:
    """Forward-fill known values per symbol; never back-fill future information."""
    required = {"datetime", "vt_symbol", *fields}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"financial input missing columns: {missing}")
    return df.sort(["vt_symbol", "datetime"]).with_columns(
        *[pl.col(field).forward_fill().over("vt_symbol").alias(field) for field in fields]
    )


def period_growth(
    df: pl.DataFrame,
    field: str,
    periods: int,
    name: str,
) -> pl.DataFrame:
    if periods <= 0:
        raise ValueError("periods must be positive")
    required = {"datetime", "vt_symbol", field}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"financial input missing columns: {missing}")
    return df.sort(["vt_symbol", "datetime"]).select(
        "datetime",
        "vt_symbol",
        (pl.col(field) / (pl.col(field).shift(periods).over("vt_symbol") + 1e-12) - 1).alias(name),
    )

