from __future__ import annotations

from typing import Literal

import polars as pl

from .neutralize_function import cs_demean
from .processor import process_cap_neutralize, process_industry_neutralize
from .utility import DataProxy

NeutralizeMode = Literal["market", "industry", "cap", "industry_cap", "industry_cap_vol"] | None

__all__ = ["neutralize_columns", "neutralize_label_columns"]


def _valid_targets(df: pl.DataFrame, target_cols: list[str]) -> list[str]:
    return [col for col in target_cols if col in df.columns]


def _exposure_frame(df: pl.DataFrame | None, column: str) -> pl.DataFrame | None:
    if df is None or df.is_empty() or column not in df.columns:
        return None
    return df.select(["datetime", "vt_symbol", column]).sort(["vt_symbol", "datetime"])


def _demean_by_date(df: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    results: list[pl.DataFrame] = []
    for name in names:
        values = df.select(
            "datetime",
            "vt_symbol",
            pl.col(name).cast(pl.Float64, strict=False).alias(name),
        )
        results.append(cs_demean(DataProxy(values)).df.rename({"data": name}))

    out = df.drop(names)
    for adjusted in results:
        out = out.join(adjusted, on=["datetime", "vt_symbol"], how="left")

    return out.select(df.columns)


def _weight_expr(weight_col: str) -> pl.Expr:
    raw_weight = pl.col(weight_col).cast(pl.Float64, strict=False)
    return (
        pl.when(raw_weight.is_finite() & (raw_weight > 1e-8))
        .then(raw_weight)
        .otherwise(1e-8)
    )


def _weighted_mean_expr(value_col: str, weight_col: str, over: str | list[str], alias: str) -> pl.Expr:
    valid = (
        pl.col(value_col).is_finite()
        & pl.col(weight_col).is_finite()
        & (pl.col(weight_col) > 0)
    )
    numerator = (
        pl.when(valid)
        .then(pl.col(value_col) * pl.col(weight_col))
        .otherwise(0.0)
        .sum()
        .over(over)
    )
    denominator = (
        pl.when(valid)
        .then(pl.col(weight_col))
        .otherwise(0.0)
        .sum()
        .over(over)
    )
    return (
        pl.when(denominator > 0)
        .then(numerator / denominator)
        .otherwise(None)
        .alias(alias)
    )


def _weighted_sum_product_expr(
    value_cols: list[str],
    weight_col: str,
    over: str | list[str],
    alias: str,
) -> pl.Expr:
    valid = pl.col(weight_col).is_finite() & (pl.col(weight_col) > 0)
    product = pl.col(weight_col)
    for col in value_cols:
        valid = valid & pl.col(col).is_finite()
        product = product * pl.col(col)

    return (
        pl.when(valid)
        .then(product)
        .otherwise(0.0)
        .sum()
        .over(over)
        .alias(alias)
    )


def _neutralize_by_industry_and_size(
    df: pl.DataFrame,
    names: list[str],
    *,
    industry_col: str,
    cap_col: str,
    cap_log: bool,
    fill_missing: str | None,
) -> pl.DataFrame:
    if industry_col not in df.columns or cap_col not in df.columns or not names:
        return df

    base = df.with_columns(pl.col(industry_col).cast(pl.Utf8).alias("_industry"))
    if fill_missing is not None:
        base = base.with_columns(pl.col("_industry").fill_null(fill_missing))

    size = (
        pl.when(pl.col(cap_col).is_not_null() & (pl.col(cap_col) > 0))
        .then(pl.col(cap_col))
        .otherwise(None)
    )
    if cap_log:
        size = size.log1p()

    base = base.with_columns(size.cast(pl.Float64).alias("_size"))

    results: list[pl.DataFrame] = []
    for name in names:
        adjusted = (
            base.with_columns(pl.col(name).cast(pl.Float64, strict=False).alias("_y"))
            .with_columns(
                (pl.col("_y").is_finite() & pl.col("_size").is_finite()).alias("_fit_valid")
            )
            .with_columns(
                pl.when(pl.col("_fit_valid"))
                .then(pl.col("_size"))
                .otherwise(None)
                .mean()
                .over(["datetime", "_industry"])
                .alias("_size_mean"),
                pl.when(pl.col("_fit_valid"))
                .then(pl.col("_y"))
                .otherwise(None)
                .mean()
                .over(["datetime", "_industry"])
                .alias("_y_mean"),
            )
            .with_columns(
                (pl.col("_size") - pl.col("_size_mean")).alias("_size_resid"),
                (pl.col("_y") - pl.col("_y_mean")).alias("_y_resid"),
            )
            .with_columns(
                pl.when(pl.col("_fit_valid"))
                .then(pl.col("_size_resid").pow(2))
                .otherwise(0.0)
                .sum()
                .over("datetime")
                .alias("_size_ss"),
                pl.when(pl.col("_fit_valid"))
                .then(pl.col("_size_resid") * pl.col("_y_resid"))
                .otherwise(0.0)
                .sum()
                .over("datetime")
                .alias("_size_y_cov")
            )
            .with_columns(
                pl.when(pl.col("_size_ss").abs() > 1e-12)
                .then(pl.col("_size_y_cov") / pl.col("_size_ss"))
                .otherwise(0.0)
                .alias("_beta_size")
            )
            .with_columns(
                pl.when(pl.col("_fit_valid"))
                .then(pl.col("_y_resid") - pl.col("_beta_size") * pl.col("_size_resid"))
                .otherwise(None)
                .alias(name)
            )
            .select(["datetime", "vt_symbol", name])
        )
        results.append(adjusted)

    out = df.drop(names)
    for adjusted in results:
        out = out.join(adjusted, on=["datetime", "vt_symbol"], how="left")

    return out.select(df.columns)


def _neutralize_by_industry_and_size_wls(
    df: pl.DataFrame,
    names: list[str],
    *,
    industry_col: str,
    cap_col: str,
    weight_col: str,
    cap_log: bool,
    fill_missing: str | None,
) -> pl.DataFrame:
    if (
        industry_col not in df.columns
        or cap_col not in df.columns
        or weight_col not in df.columns
        or not names
    ):
        return df

    base = df.with_columns(pl.col(industry_col).cast(pl.Utf8).alias("_industry"))
    if fill_missing is not None:
        base = base.with_columns(pl.col("_industry").fill_null(fill_missing))

    size = (
        pl.when(pl.col(cap_col).is_not_null() & (pl.col(cap_col) > 0))
        .then(pl.col(cap_col))
        .otherwise(None)
    )
    if cap_log:
        size = size.log1p()

    base = base.with_columns(
        size.cast(pl.Float64).alias("_size"),
        _weight_expr(weight_col).alias("_w"),
    )

    results: list[pl.DataFrame] = []
    for name in names:
        adjusted = (
            base.with_columns(pl.col(name).cast(pl.Float64, strict=False).alias("_y"))
            .with_columns(
                (
                    pl.col("_y").is_finite()
                    & pl.col("_size").is_finite()
                    & pl.col("_w").is_finite()
                    & (pl.col("_w") > 0)
                ).alias("_fit_valid")
            )
            .with_columns(
                pl.when(pl.col("_fit_valid"))
                .then(pl.col("_w"))
                .otherwise(0.0)
                .alias("_fit_w")
            )
            .with_columns(
                _weighted_mean_expr(
                    "_size", "_fit_w", ["datetime", "_industry"], "_size_wmean"
                ),
                _weighted_mean_expr(
                    "_y", "_fit_w", ["datetime", "_industry"], "_y_wmean"
                ),
            )
            .with_columns(
                (pl.col("_size") - pl.col("_size_wmean")).alias("_size_resid"),
                (pl.col("_y") - pl.col("_y_wmean")).alias("_y_resid"),
            )
            .with_columns(
                _weighted_sum_product_expr(
                    ["_size_resid", "_size_resid"],
                    "_fit_w",
                    "datetime",
                    "_size_ss",
                ),
                _weighted_sum_product_expr(
                    ["_size_resid", "_y_resid"],
                    "_fit_w",
                    "datetime",
                    "_size_y_cov",
                ),
            )
            .with_columns(
                pl.when(pl.col("_size_ss").abs() > 1e-12)
                .then(pl.col("_size_y_cov") / pl.col("_size_ss"))
                .otherwise(0.0)
                .alias("_beta_size")
            )
            .with_columns(
                pl.when(pl.col("_fit_valid"))
                .then(pl.col("_y_resid") - pl.col("_beta_size") * pl.col("_size_resid"))
                .otherwise(None)
                .alias(name)
            )
            .select(["datetime", "vt_symbol", name])
        )
        results.append(adjusted)

    out = df.drop(names)
    for adjusted in results:
        out = out.join(adjusted, on=["datetime", "vt_symbol"], how="left")

    return out.select(df.columns)


def _neutralize_by_industry_size_and_vol(
    df: pl.DataFrame,
    names: list[str],
    *,
    industry_col: str,
    cap_col: str,
    vol_col: str,
    cap_log: bool,
    fill_missing: str | None,
) -> pl.DataFrame:
    if (
        industry_col not in df.columns
        or cap_col not in df.columns
        or vol_col not in df.columns
        or not names
    ):
        return df

    base = df.with_columns(pl.col(industry_col).cast(pl.Utf8).alias("_industry"))
    if fill_missing is not None:
        base = base.with_columns(pl.col("_industry").fill_null(fill_missing))

    size = (
        pl.when(pl.col(cap_col).is_not_null() & (pl.col(cap_col) > 0))
        .then(pl.col(cap_col))
        .otherwise(None)
    )
    if cap_log:
        size = size.log1p()

    base = base.with_columns(
        size.cast(pl.Float64).alias("_size"),
        pl.col(vol_col).cast(pl.Float64, strict=False).alias("_vol"),
    )

    results: list[pl.DataFrame] = []
    for name in names:
        adjusted = (
            base.with_columns(pl.col(name).cast(pl.Float64, strict=False).alias("_y"))
            .with_columns(
                (
                    pl.col("_y").is_finite()
                    & pl.col("_size").is_finite()
                    & pl.col("_vol").is_finite()
                ).alias("_fit_valid")
            )
            .with_columns(
                pl.when(pl.col("_fit_valid"))
                .then(pl.col("_size"))
                .otherwise(None)
                .mean()
                .over(["datetime", "_industry"])
                .alias("_size_mean"),
                pl.when(pl.col("_fit_valid"))
                .then(pl.col("_vol"))
                .otherwise(None)
                .mean()
                .over(["datetime", "_industry"])
                .alias("_vol_mean"),
                pl.when(pl.col("_fit_valid"))
                .then(pl.col("_y"))
                .otherwise(None)
                .mean()
                .over(["datetime", "_industry"])
                .alias("_y_mean"),
            )
            .with_columns(
                (pl.col("_size") - pl.col("_size_mean")).alias("_size_resid"),
                (pl.col("_vol") - pl.col("_vol_mean")).alias("_vol_resid"),
                (pl.col("_y") - pl.col("_y_mean")).alias("_y_resid"),
            )
            .with_columns(
                pl.when(pl.col("_fit_valid"))
                .then(pl.col("_size_resid").pow(2))
                .otherwise(0.0)
                .sum()
                .over("datetime")
                .alias("_s_size_size"),
                pl.when(pl.col("_fit_valid"))
                .then(pl.col("_vol_resid").pow(2))
                .otherwise(0.0)
                .sum()
                .over("datetime")
                .alias("_s_vol_vol"),
                pl.when(pl.col("_fit_valid"))
                .then(pl.col("_size_resid") * pl.col("_vol_resid"))
                .otherwise(0.0)
                .sum()
                .over("datetime")
                .alias("_s_size_vol"),
                pl.when(pl.col("_fit_valid"))
                .then(pl.col("_size_resid") * pl.col("_y_resid"))
                .otherwise(0.0)
                .sum()
                .over("datetime")
                .alias("_s_size_y"),
                pl.when(pl.col("_fit_valid"))
                .then(pl.col("_vol_resid") * pl.col("_y_resid"))
                .otherwise(0.0)
                .sum()
                .over("datetime")
                .alias("_s_vol_y"),
            )
            .with_columns(
                (
                    pl.col("_s_size_size") * pl.col("_s_vol_vol")
                    - pl.col("_s_size_vol").pow(2)
                ).alias("_det")
            )
            .with_columns(
                pl.when(pl.col("_det").abs() > 1e-12)
                .then(
                    (
                        pl.col("_s_size_y") * pl.col("_s_vol_vol")
                        - pl.col("_s_vol_y") * pl.col("_s_size_vol")
                    )
                    / pl.col("_det")
                )
                .otherwise(0.0)
                .alias("_beta_size"),
                pl.when(pl.col("_det").abs() > 1e-12)
                .then(
                    (
                        pl.col("_s_vol_y") * pl.col("_s_size_size")
                        - pl.col("_s_size_y") * pl.col("_s_size_vol")
                    )
                    / pl.col("_det")
                )
                .otherwise(0.0)
                .alias("_beta_vol"),
            )
            .with_columns(
                pl.when(pl.col("_fit_valid"))
                .then(
                    pl.col("_y_resid")
                    - pl.col("_beta_size") * pl.col("_size_resid")
                    - pl.col("_beta_vol") * pl.col("_vol_resid")
                )
                .otherwise(None)
                .alias(name)
            )
            .select(["datetime", "vt_symbol", name])
        )
        results.append(adjusted)

    out = df.drop(names)
    for adjusted in results:
        out = out.join(adjusted, on=["datetime", "vt_symbol"], how="left")

    return out.select(df.columns)


def _neutralize_by_industry_size_and_vol_wls(
    df: pl.DataFrame,
    names: list[str],
    *,
    industry_col: str,
    cap_col: str,
    vol_col: str,
    weight_col: str,
    cap_log: bool,
    fill_missing: str | None,
) -> pl.DataFrame:
    if (
        industry_col not in df.columns
        or cap_col not in df.columns
        or vol_col not in df.columns
        or weight_col not in df.columns
        or not names
    ):
        return df

    base = df.with_columns(pl.col(industry_col).cast(pl.Utf8).alias("_industry"))
    if fill_missing is not None:
        base = base.with_columns(pl.col("_industry").fill_null(fill_missing))

    size = (
        pl.when(pl.col(cap_col).is_not_null() & (pl.col(cap_col) > 0))
        .then(pl.col(cap_col))
        .otherwise(None)
    )
    if cap_log:
        size = size.log1p()

    base = base.with_columns(
        size.cast(pl.Float64).alias("_size"),
        pl.col(vol_col).cast(pl.Float64, strict=False).alias("_vol"),
        _weight_expr(weight_col).alias("_w"),
    )

    results: list[pl.DataFrame] = []
    for name in names:
        adjusted = (
            base.with_columns(pl.col(name).cast(pl.Float64, strict=False).alias("_y"))
            .with_columns(
                (
                    pl.col("_y").is_finite()
                    & pl.col("_size").is_finite()
                    & pl.col("_vol").is_finite()
                    & pl.col("_w").is_finite()
                    & (pl.col("_w") > 0)
                ).alias("_fit_valid")
            )
            .with_columns(
                pl.when(pl.col("_fit_valid"))
                .then(pl.col("_w"))
                .otherwise(0.0)
                .alias("_fit_w")
            )
            .with_columns(
                _weighted_mean_expr(
                    "_size", "_fit_w", ["datetime", "_industry"], "_size_wmean"
                ),
                _weighted_mean_expr(
                    "_vol", "_fit_w", ["datetime", "_industry"], "_vol_wmean"
                ),
                _weighted_mean_expr(
                    "_y", "_fit_w", ["datetime", "_industry"], "_y_wmean"
                ),
            )
            .with_columns(
                (pl.col("_size") - pl.col("_size_wmean")).alias("_size_resid"),
                (pl.col("_vol") - pl.col("_vol_wmean")).alias("_vol_resid"),
                (pl.col("_y") - pl.col("_y_wmean")).alias("_y_resid"),
            )
            .with_columns(
                _weighted_sum_product_expr(
                    ["_size_resid", "_size_resid"],
                    "_fit_w",
                    "datetime",
                    "_s_size_size",
                ),
                _weighted_sum_product_expr(
                    ["_vol_resid", "_vol_resid"],
                    "_fit_w",
                    "datetime",
                    "_s_vol_vol",
                ),
                _weighted_sum_product_expr(
                    ["_size_resid", "_vol_resid"],
                    "_fit_w",
                    "datetime",
                    "_s_size_vol",
                ),
                _weighted_sum_product_expr(
                    ["_size_resid", "_y_resid"],
                    "_fit_w",
                    "datetime",
                    "_s_size_y",
                ),
                _weighted_sum_product_expr(
                    ["_vol_resid", "_y_resid"],
                    "_fit_w",
                    "datetime",
                    "_s_vol_y",
                ),
            )
            .with_columns(
                (
                    pl.col("_s_size_size") * pl.col("_s_vol_vol")
                    - pl.col("_s_size_vol").pow(2)
                ).alias("_det")
            )
            .with_columns(
                pl.when(pl.col("_det").abs() > 1e-12)
                .then(
                    (
                        pl.col("_s_size_y") * pl.col("_s_vol_vol")
                        - pl.col("_s_vol_y") * pl.col("_s_size_vol")
                    )
                    / pl.col("_det")
                )
                .otherwise(0.0)
                .alias("_beta_size"),
                pl.when(pl.col("_det").abs() > 1e-12)
                .then(
                    (
                        pl.col("_s_vol_y") * pl.col("_s_size_size")
                        - pl.col("_s_size_y") * pl.col("_s_size_vol")
                    )
                    / pl.col("_det")
                )
                .otherwise(0.0)
                .alias("_beta_vol"),
            )
            .with_columns(
                pl.when(pl.col("_fit_valid"))
                .then(
                    pl.col("_y_resid")
                    - pl.col("_beta_size") * pl.col("_size_resid")
                    - pl.col("_beta_vol") * pl.col("_vol_resid")
                )
                .otherwise(None)
                .alias(name)
            )
            .select(["datetime", "vt_symbol", name])
        )
        results.append(adjusted)

    out = df.drop(names)
    for adjusted in results:
        out = out.join(adjusted, on=["datetime", "vt_symbol"], how="left")

    return out.select(df.columns)


def neutralize_columns(
    result_df: pl.DataFrame,
    *,
    target_cols: list[str],
    industry_df: pl.DataFrame | None = None,
    cap_df: pl.DataFrame | None = None,
    vol_df: pl.DataFrame | None = None,
    mode: NeutralizeMode = None,
    industry_col: str = "industry",
    cap_col: str = "cap",
    vol_col: str = "hist_vol_60d",
    weight_col: str | None = None,
    industry_method: str = "demean",
    cap_log: bool = True,
    fill_missing: str | None = "UNKNOWN",
    suffix: str = "_neu",
) -> pl.DataFrame:
    """
    Neutralize selected columns before model training.

    mode=None applies available industry and size neutralization sequentially.
    Use mode="industry_cap" or "industry_cap_vol" for one-shot fixed-effect
    regressions that neutralize industry plus continuous exposures together.
    Pass weight_col with those two modes to run weighted least squares.
    Each target regression uses only rows where that target and all required
    exposures are finite, so labels with different horizons keep independent
    fitting samples.
    """
    if result_df is None or result_df.is_empty():
        return result_df

    names = _valid_targets(result_df, target_cols)
    if not names:
        return result_df
    if weight_col is not None and mode not in {"industry_cap", "industry_cap_vol"}:
        raise ValueError("weight_col only applies to mode='industry_cap' or mode='industry_cap_vol'")
    if weight_col is not None and weight_col not in result_df.columns:
        raise ValueError(f"weight_col not found in result_df: {weight_col}")

    industry = _exposure_frame(industry_df, industry_col)
    cap = _exposure_frame(cap_df, cap_col)
    vol = _exposure_frame(vol_df, vol_col)

    has_industry = industry is not None
    has_cap = cap is not None
    has_vol = vol is not None

    select_cols = ["datetime", "vt_symbol", *names]
    if weight_col is not None and weight_col not in select_cols:
        select_cols.append(weight_col)
    tmp = result_df.select(select_cols).sort(["vt_symbol", "datetime"])

    if mode == "market":
        tmp = _demean_by_date(tmp, names)
    elif mode == "industry":
        if not has_industry:
            return result_df
        tmp = tmp.join_asof(industry, on="datetime", by="vt_symbol", strategy="backward")
        tmp = process_industry_neutralize(
            tmp,
            industry_col=industry_col,
            method=industry_method,
            names=names,
            fill_missing=fill_missing,
        )
    elif mode == "cap":
        if not has_cap:
            return result_df
        tmp = tmp.join_asof(cap, on="datetime", by="vt_symbol", strategy="backward")
        tmp = process_cap_neutralize(tmp, cap_col=cap_col, log_cap=cap_log, names=names)
    elif mode == "industry_cap":
        if not (has_industry and has_cap):
            return result_df
        tmp = tmp.join_asof(industry, on="datetime", by="vt_symbol", strategy="backward")
        tmp = tmp.join_asof(cap, on="datetime", by="vt_symbol", strategy="backward")
        if weight_col is None:
            tmp = _neutralize_by_industry_and_size(
                tmp,
                names,
                industry_col=industry_col,
                cap_col=cap_col,
                cap_log=cap_log,
                fill_missing=fill_missing,
            )
        else:
            tmp = _neutralize_by_industry_and_size_wls(
                tmp,
                names,
                industry_col=industry_col,
                cap_col=cap_col,
                weight_col=weight_col,
                cap_log=cap_log,
                fill_missing=fill_missing,
            )
    elif mode == "industry_cap_vol":
        if not (has_industry and has_cap and has_vol):
            return result_df
        tmp = tmp.join_asof(industry, on="datetime", by="vt_symbol", strategy="backward")
        tmp = tmp.join_asof(cap, on="datetime", by="vt_symbol", strategy="backward")
        tmp = tmp.join_asof(vol, on="datetime", by="vt_symbol", strategy="backward")
        if weight_col is None:
            tmp = _neutralize_by_industry_size_and_vol(
                tmp,
                names,
                industry_col=industry_col,
                cap_col=cap_col,
                vol_col=vol_col,
                cap_log=cap_log,
                fill_missing=fill_missing,
            )
        else:
            tmp = _neutralize_by_industry_size_and_vol_wls(
                tmp,
                names,
                industry_col=industry_col,
                cap_col=cap_col,
                vol_col=vol_col,
                weight_col=weight_col,
                cap_log=cap_log,
                fill_missing=fill_missing,
            )
    elif mode is None:
        if has_industry:
            tmp = tmp.join_asof(industry, on="datetime", by="vt_symbol", strategy="backward")
            tmp = process_industry_neutralize(
                tmp,
                industry_col=industry_col,
                method=industry_method,
                names=names,
                fill_missing=fill_missing,
            )
        if has_cap:
            tmp = tmp.join_asof(cap, on="datetime", by="vt_symbol", strategy="backward")
            tmp = process_cap_neutralize(tmp, cap_col=cap_col, log_cap=cap_log, names=names)
    else:
        raise ValueError(f"unknown neutralize mode: {mode}")

    rename_map = {name: f"{name}{suffix}" for name in names}
    tmp = tmp.rename(rename_map).select(["datetime", "vt_symbol", *rename_map.values()])
    return result_df.join(tmp, on=["datetime", "vt_symbol"], how="left")


def neutralize_label_columns(
    result_df: pl.DataFrame,
    *,
    label_cols: list[str] | None = None,
    industry_df: pl.DataFrame | None = None,
    cap_df: pl.DataFrame | None = None,
    vol_df: pl.DataFrame | None = None,
    mode: NeutralizeMode = None,
    industry_col: str = "industry",
    cap_col: str = "cap",
    vol_col: str = "hist_vol_60d",
    weight_col: str | None = None,
    industry_method: str = "demean",
    cap_log: bool = True,
    fill_missing: str | None = "UNKNOWN",
    suffix: str = "_neu",
) -> pl.DataFrame:
    if result_df is None or result_df.is_empty():
        return result_df

    targets = label_cols or [col for col in result_df.columns if col.startswith("label_")]
    return neutralize_columns(
        result_df,
        target_cols=targets,
        industry_df=industry_df,
        cap_df=cap_df,
        vol_df=vol_df,
        mode=mode,
        industry_col=industry_col,
        cap_col=cap_col,
        vol_col=vol_col,
        weight_col=weight_col,
        industry_method=industry_method,
        cap_log=cap_log,
        fill_missing=fill_missing,
        suffix=suffix,
    )
