from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from vnpy.alpha import Segment
from vnpy.alpha.dataset import AlphaDataset, label_postprocess
from vnpy.alpha.dataset.processor_post import neutralize_label_columns

KEY_COLUMNS = ["datetime", "vt_symbol"]
PRICE_LIMIT_MASK_COL = "sample_weight_mask"
BARRA_CNE5_V1_STYLE_FACTORS = [
    "size",
    "non_linear_size",
    "momentum",
    "liquidity",
    "book_to_price",
    "leverage",
    "growth",
    "earnings_yield",
    "beta",
    "residual_volatility",
]
BARRA_EXCLUDE_COLUMNS = {
    "datetime",
    "trade_date",
    "date",
    "order_book_id",
    "vt_symbol",
    "factor",
    "exposure",
    "comovement",
}

__all__ = [
    "NeuLabel",
    "LabelStrategy",
    "BARRA_CNE5_V1_STYLE_FACTORS",
    "PRICE_LIMIT_MASK_COL",
    "apply_price_limit_mask",
    "build_price_limit_mask",
    "neutralize_label_barra_wls",
    "neutralize_label_size_wls",
]


def _resolve_label_cols(result_df: pl.DataFrame, label_cols: list[str] | None) -> list[str]:
    if label_cols is not None:
        missing = [col for col in label_cols if col not in result_df.columns]
        if missing:
            raise ValueError(f"missing label columns: {missing}")
        return label_cols
    return [col for col in result_df.columns if col.startswith("label_")]


def _valid_or_zero(expr: pl.Expr) -> pl.Expr:
    return pl.when(pl.col("_valid")).then(expr).otherwise(0.0)


def _pred_valid_or_null(expr: pl.Expr) -> pl.Expr:
    return pl.when(pl.col("_pred_valid")).then(expr).otherwise(None)


def _normalize_datetime_key(df: pl.DataFrame) -> pl.DataFrame:
    if "datetime" in df.columns:
        return df.with_columns(pl.col("datetime").cast(pl.Datetime, strict=False))
    if "trade_date" in df.columns:
        return df.with_columns(
            pl.col("trade_date")
            .cast(pl.Utf8, strict=False)
            .str.to_datetime(strict=False)
            .alias("datetime")
        )
    raise ValueError("missing datetime/trade_date column")


def build_price_limit_mask(
    price_df: pl.DataFrame,
    *,
    entry_open_shift: int = 1,
    round_decimals: int = 2,
) -> pl.DataFrame:
    """Build daily price-limit masks from true limit_up/limit_down prices."""
    required = ["vt_symbol", "open", "close", "limit_up", "limit_down"]
    missing = [col for col in required if col not in price_df.columns]
    if missing:
        raise ValueError(f"missing price limit columns: {missing}")
    if entry_open_shift < 1:
        raise ValueError("entry_open_shift must be >= 1")

    panel = (
        _normalize_datetime_key(price_df)
        .select([*KEY_COLUMNS, "open", "close", "limit_up", "limit_down"])
        .sort(["vt_symbol", "datetime"])
        .with_columns(
            [
                pl.col("open").cast(pl.Float64, strict=False).round(round_decimals).alias("_open"),
                pl.col("close").cast(pl.Float64, strict=False).round(round_decimals).alias("_close"),
                pl.col("limit_up")
                .cast(pl.Float64, strict=False)
                .round(round_decimals)
                .alias("_limit_up"),
                pl.col("limit_down")
                .cast(pl.Float64, strict=False)
                .round(round_decimals)
                .alias("_limit_down"),
            ]
        )
        .with_columns(
            [
                pl.col("_open").shift(-entry_open_shift).over("vt_symbol").alias("_entry_open"),
                pl.col("_limit_up")
                .shift(-entry_open_shift)
                .over("vt_symbol")
                .alias("_entry_limit_up"),
                pl.col("_limit_down")
                .shift(-entry_open_shift)
                .over("vt_symbol")
                .alias("_entry_limit_down"),
            ]
        )
    )

    entry_open_limit = (
        ((pl.col("_entry_open") == pl.col("_entry_limit_up")) & (pl.col("_entry_limit_up") != 0))
        | ((pl.col("_entry_open") == pl.col("_entry_limit_down")) & (pl.col("_entry_limit_down") != 0))
    )
    feature_close_limit = (
        ((pl.col("_close") == pl.col("_limit_up")) & (pl.col("_limit_up") != 0))
        | ((pl.col("_close") == pl.col("_limit_down")) & (pl.col("_limit_down") != 0))
    )
    return panel.with_columns(
        [
            entry_open_limit.fill_null(False).alias("is_entry_open_limit"),
            feature_close_limit.fill_null(False).alias("is_feature_close_limit"),
        ]
    ).select([*KEY_COLUMNS, "is_entry_open_limit", "is_feature_close_limit"])


def apply_price_limit_mask(
    result_df: pl.DataFrame,
    price_df: pl.DataFrame,
    *,
    label_cols: list[str] | None = None,
    sample_weight_col: str | None = PRICE_LIMIT_MASK_COL,
    entry_open_shift: int = 1,
    feature_day_close_mask: bool = True,
    round_decimals: int = 2,
    missing_threshold: float = 0.01,
) -> tuple[pl.DataFrame, dict[str, int | float]]:
    """Mask labels and sample weights for price-limit constrained samples."""
    if result_df is None or result_df.is_empty():
        return result_df, {
            "rows": 0,
            "dates": 0,
            "symbols": 0,
            "missing_mask_rows": 0,
            "missing_mask_rate": 0.0,
            "entry_open_limit_rows": 0,
            "entry_open_limit_rate": 0.0,
            "feature_close_limit_rows": 0,
            "feature_close_limit_rate": 0.0,
        }
    if price_df is None or price_df.is_empty():
        raise ValueError("price_df is empty")
    if sample_weight_col is not None and not str(sample_weight_col).strip():
        raise ValueError("sample_weight_col cannot be empty")

    targets = _resolve_label_cols(result_df, label_cols)
    missing_targets = [col for col in targets if col not in result_df.columns]
    if missing_targets:
        raise ValueError(f"missing label columns: {missing_targets}")

    base = _normalize_datetime_key(result_df)
    mask = build_price_limit_mask(
        price_df,
        entry_open_shift=entry_open_shift,
        round_decimals=round_decimals,
    )
    joined = base.join(mask, on=KEY_COLUMNS, how="left")
    missing_mask = joined["is_entry_open_limit"].is_null()
    joined = joined.with_columns(
        [
            pl.col("is_entry_open_limit").fill_null(False),
            pl.col("is_feature_close_limit").fill_null(False),
        ]
    )

    updates: list[pl.Expr] = [
        pl.when(pl.col("is_entry_open_limit"))
        .then(None)
        .otherwise(pl.col(label_col))
        .alias(label_col)
        for label_col in targets
    ]
    if sample_weight_col is not None:
        if sample_weight_col in base.columns:
            base_weight = pl.col(sample_weight_col).cast(pl.Float64, strict=False).fill_null(1.0)
        else:
            base_weight = pl.lit(1.0, dtype=pl.Float64)
        weight_multiplier = (
            pl.when(pl.col("is_feature_close_limit") & pl.lit(feature_day_close_mask))
            .then(0.0)
            .otherwise(1.0)
        )
        updates.append((base_weight * weight_multiplier).cast(pl.Float32).alias(sample_weight_col))

    out = joined.with_columns(updates).drop(["is_entry_open_limit", "is_feature_close_limit"], strict=False)

    stats = {
        "rows": int(base.height),
        "dates": int(base["datetime"].n_unique()),
        "symbols": int(base["vt_symbol"].n_unique()),
        "missing_mask_rows": int(missing_mask.sum()),
        "missing_mask_rate": float(missing_mask.mean()) if base.height else 0.0,
        "entry_open_limit_rows": int(joined["is_entry_open_limit"].sum()),
        "entry_open_limit_rate": float(joined["is_entry_open_limit"].mean()) if base.height else 0.0,
        "feature_close_limit_rows": int(joined["is_feature_close_limit"].sum()),
        "feature_close_limit_rate": float(joined["is_feature_close_limit"].mean()) if base.height else 0.0,
    }
    if stats["missing_mask_rate"] > missing_threshold:
        raise RuntimeError(
            "price-limit mask missing too many rows: "
            f"{stats['missing_mask_rate']:.4%} > {missing_threshold:.4%}"
        )
    return out, stats


def _neutralize_one_label_by_size_wls(
    prepared: pl.DataFrame,
    label_col: str,
    out_col: str,
    *,
    include_quadratic: bool,
) -> pl.DataFrame:
    work = prepared.with_columns(
        pl.col(label_col).cast(pl.Float64, strict=False).alias("_y")
    )
    if include_quadratic:
        fit_valid_expr = (
            pl.col("_y").is_finite()
            & pl.col("_x").is_finite()
            & pl.col("_z").is_finite()
            & pl.col("_w").is_finite()
            & (pl.col("_w") > 0)
        )
        pred_valid_expr = (
            pl.col("_y").is_finite()
            & pl.col("_x").is_finite()
            & pl.col("_z").is_finite()
        )
        return (
            work.with_columns(
                [
                    fit_valid_expr.alias("_valid"),
                    pred_valid_expr.alias("_pred_valid"),
                ]
            )
            .with_columns(
                [
                    _valid_or_zero(pl.col("_w")).sum().over("datetime").alias("_sum_w"),
                    _valid_or_zero(pl.col("_w") * pl.col("_x")).sum().over("datetime").alias("_sum_wx"),
                    _valid_or_zero(pl.col("_w") * pl.col("_z")).sum().over("datetime").alias("_sum_wz"),
                    _valid_or_zero(pl.col("_w") * pl.col("_y")).sum().over("datetime").alias("_sum_wy"),
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("_sum_w") > 0)
                    .then(pl.col("_sum_wx") / pl.col("_sum_w"))
                    .otherwise(None)
                    .alias("_x_wmean"),
                    pl.when(pl.col("_sum_w") > 0)
                    .then(pl.col("_sum_wz") / pl.col("_sum_w"))
                    .otherwise(None)
                    .alias("_z_wmean"),
                    pl.when(pl.col("_sum_w") > 0)
                    .then(pl.col("_sum_wy") / pl.col("_sum_w"))
                    .otherwise(None)
                    .alias("_y_wmean"),
                ]
            )
            .with_columns(
                [
                    _pred_valid_or_null(pl.col("_x") - pl.col("_x_wmean")).alias("_x_resid"),
                    _pred_valid_or_null(pl.col("_z") - pl.col("_z_wmean")).alias("_z_resid"),
                    _pred_valid_or_null(pl.col("_y") - pl.col("_y_wmean")).alias("_y_resid"),
                ]
            )
            .with_columns(
                [
                    _valid_or_zero(pl.col("_w") * pl.col("_x_resid").pow(2))
                    .sum()
                    .over("datetime")
                    .alias("_s_xx"),
                    _valid_or_zero(pl.col("_w") * pl.col("_x_resid") * pl.col("_z_resid"))
                    .sum()
                    .over("datetime")
                    .alias("_s_xz"),
                    _valid_or_zero(pl.col("_w") * pl.col("_z_resid").pow(2))
                    .sum()
                    .over("datetime")
                    .alias("_s_zz"),
                    _valid_or_zero(pl.col("_w") * pl.col("_x_resid") * pl.col("_y_resid"))
                    .sum()
                    .over("datetime")
                    .alias("_s_xy"),
                    _valid_or_zero(pl.col("_w") * pl.col("_z_resid") * pl.col("_y_resid"))
                    .sum()
                    .over("datetime")
                    .alias("_s_zy"),
                ]
            )
            .with_columns((pl.col("_s_xx") * pl.col("_s_zz") - pl.col("_s_xz").pow(2)).alias("_det"))
            .with_columns(
                pl.when(pl.col("_s_xx").abs() > 1e-12)
                .then(pl.col("_s_xy") / pl.col("_s_xx"))
                .otherwise(0.0)
                .alias("_beta_x_fallback")
            )
            .with_columns(
                [
                    pl.when(pl.col("_det").abs() > 1e-12)
                    .then(
                        (pl.col("_s_xy") * pl.col("_s_zz") - pl.col("_s_zy") * pl.col("_s_xz"))
                        / pl.col("_det")
                    )
                    .otherwise(pl.col("_beta_x_fallback"))
                    .alias("_beta_x"),
                    pl.when(pl.col("_det").abs() > 1e-12)
                    .then(
                        (pl.col("_s_xx") * pl.col("_s_zy") - pl.col("_s_xz") * pl.col("_s_xy"))
                        / pl.col("_det")
                    )
                    .otherwise(0.0)
                    .alias("_beta_z"),
                ]
            )
            .with_columns(
                pl.when(pl.col("_pred_valid"))
                .then(
                    pl.col("_y_resid")
                    - pl.col("_beta_x") * pl.col("_x_resid")
                    - pl.col("_beta_z") * pl.col("_z_resid")
                )
                .otherwise(None)
                .cast(pl.Float32)
                .alias(out_col)
            )
            .select([*KEY_COLUMNS, out_col])
        )

    fit_valid_expr = (
        pl.col("_y").is_finite()
        & pl.col("_x").is_finite()
        & pl.col("_w").is_finite()
        & (pl.col("_w") > 0)
    )
    pred_valid_expr = pl.col("_y").is_finite() & pl.col("_x").is_finite()
    return (
        work.with_columns(
            [
                fit_valid_expr.alias("_valid"),
                pred_valid_expr.alias("_pred_valid"),
            ]
        )
        .with_columns(
            [
                _valid_or_zero(pl.col("_w")).sum().over("datetime").alias("_sum_w"),
                _valid_or_zero(pl.col("_w") * pl.col("_x")).sum().over("datetime").alias("_sum_wx"),
                _valid_or_zero(pl.col("_w") * pl.col("_y")).sum().over("datetime").alias("_sum_wy"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("_sum_w") > 0)
                .then(pl.col("_sum_wx") / pl.col("_sum_w"))
                .otherwise(None)
                .alias("_x_wmean"),
                pl.when(pl.col("_sum_w") > 0)
                .then(pl.col("_sum_wy") / pl.col("_sum_w"))
                .otherwise(None)
                .alias("_y_wmean"),
            ]
        )
        .with_columns(
            [
                _pred_valid_or_null(pl.col("_x") - pl.col("_x_wmean")).alias("_x_resid"),
                _pred_valid_or_null(pl.col("_y") - pl.col("_y_wmean")).alias("_y_resid"),
            ]
        )
        .with_columns(
            [
                _valid_or_zero(pl.col("_w") * pl.col("_x_resid").pow(2))
                .sum()
                .over("datetime")
                .alias("_s_xx"),
                _valid_or_zero(pl.col("_w") * pl.col("_x_resid") * pl.col("_y_resid"))
                .sum()
                .over("datetime")
                .alias("_s_xy"),
            ]
        )
        .with_columns(
            pl.when(pl.col("_s_xx").abs() > 1e-12)
            .then(pl.col("_s_xy") / pl.col("_s_xx"))
            .otherwise(0.0)
            .alias("_beta_cap")
        )
        .with_columns(
            pl.when(pl.col("_pred_valid"))
            .then(pl.col("_y_resid") - pl.col("_beta_cap") * pl.col("_x_resid"))
            .otherwise(None)
            .cast(pl.Float32)
            .alias(out_col)
        )
        .select([*KEY_COLUMNS, out_col])
    )


def neutralize_label_size_wls(
    result_df: pl.DataFrame,
    cap_df: pl.DataFrame,
    *,
    label_cols: list[str] | None = None,
    cap_col: str = "cap",
    weight_col: str | None = None,
    include_quadratic: bool = False,
    suffix: str = "_size_neu",
    replace: bool = False,
    clip: float = 3.0,
) -> pl.DataFrame:
    """Neutralize labels by daily WLS on robust-zscored log market cap.

    Linear form: label ~ intercept + z(log1p(cap)).
    Nonlinear form adds z(log1p(cap))^2.
    """
    if result_df is None or result_df.is_empty():
        return result_df
    if cap_df is None or cap_df.is_empty():
        return result_df

    targets = _resolve_label_cols(result_df, label_cols)
    if not targets:
        return result_df
    if cap_col not in cap_df.columns:
        raise ValueError(f"missing cap column: {cap_col}")
    if weight_col is not None and weight_col not in result_df.columns:
        raise ValueError(f"missing weight column: {weight_col}")

    select_cols = [*KEY_COLUMNS, *targets]
    if weight_col is not None:
        select_cols.append(weight_col)
    cap = cap_df.select(KEY_COLUMNS + [cap_col]).sort(["vt_symbol", "datetime"])
    joined = (
        result_df.select(select_cols)
        .sort(["vt_symbol", "datetime"])
        .join_asof(cap, on="datetime", by="vt_symbol", strategy="backward")
    )

    if weight_col is not None:
        raw_weight_expr = pl.col(weight_col).cast(pl.Float64, strict=False)
        weight_expr = (
            pl.when(raw_weight_expr.is_finite() & (raw_weight_expr > 0))
            .then(raw_weight_expr)
            .otherwise(0.0)
        )
    else:
        weight_expr = pl.lit(1.0, dtype=pl.Float64)
    prepared = (
        joined.with_columns(
            [
                pl.when(pl.col(cap_col).is_finite() & (pl.col(cap_col) > 0))
                .then(pl.col(cap_col).log1p())
                .otherwise(None)
                .cast(pl.Float64)
                .alias("_log_cap"),
                weight_expr.alias("_w"),
            ]
        )
        .with_columns(pl.col("_log_cap").median().over("datetime").alias("_log_cap_median"))
        .with_columns((pl.col("_log_cap") - pl.col("_log_cap_median")).alias("_x_centered"))
        .with_columns(pl.col("_x_centered").abs().median().over("datetime").alias("_x_mad"))
        .with_columns(
            pl.when(pl.col("_x_mad").is_finite() & (pl.col("_x_mad") > 1e-12))
            .then((pl.col("_x_centered") / (pl.col("_x_mad") * 1.4826)).clip(-clip, clip))
            .when(pl.col("_log_cap").is_finite())
            .then(0.0)
            .otherwise(None)
            .cast(pl.Float64)
            .alias("_x")
        )
    )
    if include_quadratic:
        prepared = prepared.with_columns(
            pl.when(pl.col("_x").is_finite())
            .then(pl.col("_x").pow(2))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("_z")
        )

    out = result_df
    for target in targets:
        out_col = target if replace else f"{target}{suffix}"
        residual = _neutralize_one_label_by_size_wls(
            prepared,
            target,
            out_col,
            include_quadratic=include_quadratic,
        )
        out = out.drop(out_col, strict=False).join(residual, on=KEY_COLUMNS, how="left")
    return out


def _infer_barra_industry_cols(
    barra_df: pl.DataFrame,
    factor_cols: list[str],
    industry_cols: list[str] | None,
) -> list[str]:
    if industry_cols is not None:
        missing = [col for col in industry_cols if col not in barra_df.columns]
        if missing:
            raise ValueError(f"missing Barra industry columns: {missing}")
        return industry_cols

    factors = set(factor_cols)
    out = [
        col
        for col in barra_df.columns
        if col not in BARRA_EXCLUDE_COLUMNS and col not in factors
    ]
    if not out:
        raise ValueError("cannot infer Barra industry dummy columns")
    return out


def _prepare_barra_exposure_frame(
    barra_df: pl.DataFrame,
    *,
    factor_cols: list[str],
    industry_cols: list[str],
    clip: float,
) -> pl.DataFrame:
    missing_factors = [col for col in factor_cols if col not in barra_df.columns]
    if missing_factors:
        raise ValueError(f"missing Barra style factor columns: {missing_factors}")

    df = _normalize_datetime_key(barra_df)
    select_cols = [*KEY_COLUMNS, *factor_cols, *industry_cols]
    missing = [col for col in select_cols if col not in df.columns]
    if missing:
        raise ValueError(f"missing Barra exposure columns: {missing}")

    industry_exprs = [
        pl.col(col).cast(pl.Float64, strict=False).fill_null(0.0).alias(col)
        for col in industry_cols
    ]
    out = (
        df.select(select_cols)
        .with_columns(
            [pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in factor_cols]
            + industry_exprs
        )
        .unique(subset=KEY_COLUMNS, keep="last")
        .sort(["vt_symbol", "datetime"])
    )
    for col in factor_cols:
        median_col = f"__{col}_median"
        centered_col = f"__{col}_centered"
        mad_col = f"__{col}_mad"
        out = (
            out.with_columns(pl.col(col).median().over("datetime").alias(median_col))
            .with_columns((pl.col(col) - pl.col(median_col)).alias(centered_col))
            .with_columns(pl.col(centered_col).abs().median().over("datetime").alias(mad_col))
            .with_columns(
                pl.when(pl.col(col).is_finite() & pl.col(mad_col).is_finite() & (pl.col(mad_col) > 1e-12))
                .then((pl.col(centered_col) / (pl.col(mad_col) * 1.4826)).clip(-clip, clip))
                .when(pl.col(col).is_finite())
                .then(0.0)
                .otherwise(None)
                .cast(pl.Float64)
                .alias(col)
            )
            .drop([median_col, centered_col, mad_col], strict=False)
        )
    return out


def _neutralize_barra_group(
    group: pd.DataFrame,
    *,
    label_col: str,
    x_cols: list[str],
    weight_col: str,
    min_obs: int,
    rcond: float,
) -> pd.DataFrame:
    n = len(group)
    residual = np.full(n, np.nan, dtype=np.float64)
    y = group[label_col].to_numpy(dtype=np.float64)
    x_values = group[x_cols].to_numpy(dtype=np.float64)
    weights = group[weight_col].to_numpy(dtype=np.float64)

    x_valid = np.isfinite(x_values).all(axis=1)
    fit_mask = np.isfinite(y) & x_valid & np.isfinite(weights) & (weights > 0)
    pred_mask = np.isfinite(y) & x_valid
    if int(np.count_nonzero(fit_mask)) >= min_obs:
        x_fit = np.column_stack(
            [
                np.ones(int(np.count_nonzero(fit_mask)), dtype=np.float64),
                x_values[fit_mask],
            ]
        )
        y_fit = y[fit_mask]
        w_fit = weights[fit_mask]
        sqrt_w = np.sqrt(w_fit)
        try:
            beta, *_ = np.linalg.lstsq(x_fit * sqrt_w[:, None], y_fit * sqrt_w, rcond=rcond)
            x_pred = np.column_stack(
                [
                    np.ones(int(np.count_nonzero(pred_mask)), dtype=np.float64),
                    x_values[pred_mask],
                ]
            )
            residual[pred_mask] = y[pred_mask] - x_pred @ beta
        except np.linalg.LinAlgError:
            pass

    return pd.DataFrame(
        {
            "datetime": group["datetime"].to_numpy(),
            "vt_symbol": group["vt_symbol"].to_numpy(),
            "_residual": residual,
        }
    )


def neutralize_label_barra_wls(
    result_df: pl.DataFrame,
    barra_df: pl.DataFrame,
    *,
    label_cols: list[str] | None = None,
    factor_cols: list[str] | None = None,
    industry_cols: list[str] | None = None,
    weight_col: str | None = None,
    suffix: str = "_barra_cne5_neu",
    replace: bool = False,
    clip: float = 3.0,
    min_obs: int | None = None,
    rcond: float = 1e-10,
) -> pl.DataFrame:
    """Neutralize labels by daily WLS on Barra style factors and industries.

    The regression includes an intercept, robust-MAD standardized style
    exposures, and raw industry dummy columns. Rows with zero/non-finite
    weights do not affect the fitted plane but can still receive residuals.
    """
    if result_df is None or result_df.is_empty():
        return result_df
    if barra_df is None or barra_df.is_empty():
        return result_df

    targets = _resolve_label_cols(result_df, label_cols)
    if not targets:
        return result_df
    if weight_col is not None and weight_col not in result_df.columns:
        raise ValueError(f"missing weight column: {weight_col}")

    factors = list(factor_cols or BARRA_CNE5_V1_STYLE_FACTORS)
    industries = _infer_barra_industry_cols(barra_df, factors, industry_cols)
    x_cols = [*factors, *industries]
    effective_min_obs = int(min_obs or max(50, len(x_cols) + 2))

    barra = _prepare_barra_exposure_frame(
        barra_df,
        factor_cols=factors,
        industry_cols=industries,
        clip=clip,
    )

    select_cols = [*KEY_COLUMNS, *targets]
    if weight_col is not None:
        select_cols.append(weight_col)
    base = _normalize_datetime_key(result_df).select(select_cols)
    joined = base.join(barra, on=KEY_COLUMNS, how="left")
    work_weight_col = "__barra_w"
    if weight_col is not None:
        weight_expr = (
            pl.when(pl.col(weight_col).cast(pl.Float64, strict=False).is_finite() & (pl.col(weight_col).cast(pl.Float64, strict=False) > 0))
            .then(pl.col(weight_col).cast(pl.Float64, strict=False))
            .otherwise(0.0)
        )
    else:
        weight_expr = pl.lit(1.0, dtype=pl.Float64)
    joined = joined.with_columns(weight_expr.alias(work_weight_col)).sort(KEY_COLUMNS)

    pdf = joined.to_pandas()
    out = result_df
    for target in targets:
        out_col = target if replace else f"{target}{suffix}"
        residual_frames = [
            _neutralize_barra_group(
                group,
                label_col=target,
                x_cols=x_cols,
                weight_col=work_weight_col,
                min_obs=effective_min_obs,
                rcond=float(rcond),
            )
            for _, group in pdf.groupby("datetime", sort=True)
        ]
        residual_pdf = pd.concat(residual_frames, ignore_index=True) if residual_frames else pd.DataFrame(
            columns=[*KEY_COLUMNS, "_residual"]
        )
        residual = (
            pl.from_pandas(residual_pdf)
            .with_columns(
                [
                    pl.col("datetime").cast(pl.Datetime, strict=False),
                    pl.col("vt_symbol").cast(pl.Utf8),
                    pl.col("_residual").cast(pl.Float32).alias(out_col),
                ]
            )
            .select([*KEY_COLUMNS, out_col])
        )
        out = out.drop(out_col, strict=False).join(residual, on=KEY_COLUMNS, how="left")
    return out


class NeuLabel(AlphaDataset):
    """Dataset for label calculation and label post-processing."""

    def __init__(
        self,
        df: pl.DataFrame,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
        interval: str = "1d",
        enable_cache: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
            interval=interval,
            enable_cache=enable_cache,
            cache_dir=cache_dir,
        )

        self.add_feature("label_overnight_ret", "ts_delay(open, -1) / close - 1")

        o2o_h1 = "ts_delay(open, -2) / ts_delay(open, -1) - 1"
        o2o_h3 = "ts_delay(open, -4) / ts_delay(open, -1) - 1"
        o2o_h5 = "ts_delay(open, -6) / ts_delay(open, -1) - 1"
        o2o_h7 = "ts_delay(open, -8) / ts_delay(open, -1) - 1"
        self.add_feature("label_o2o_h1", o2o_h1)
        self.add_feature("label_o2o_h3", o2o_h3)
        self.add_feature("label_o2o_h5", o2o_h5)
        self.add_feature("label_o2o_h7", o2o_h7)

        self.add_feature("label_c2c_h3", "ts_delay(close, -3) / ts_delay(close, -1) - 1")

        self.add_feature("label_overnight_logret", "ts_log(ts_delay(open, -1)) - ts_log(close)")
        self.add_feature(
            "label_o2o_logret_h1",
            "ts_log(ts_delay(open, -2)) - ts_log(ts_delay(open, -1))",
        )
        self.add_feature(
            "label_o2o_logret_h3",
            "ts_log(ts_delay(open, -4)) - ts_log(ts_delay(open, -1))",
        )
        self.add_feature(
            "label_o2o_logret_h5",
            "ts_log(ts_delay(open, -6)) - ts_log(ts_delay(open, -1))",
        )

        hist_daily_ret = "(close / ts_delay(close, 1) - 1)"
        hist_vol_20d = f"ts_std({hist_daily_ret}, 20)"
        hist_vol_60d = f"ts_std({hist_daily_ret}, 60)"
        self.add_feature("hist_vol_20d", hist_vol_20d)
        self.add_feature("hist_vol_60d", hist_vol_60d)

        future_o2o_ret = "(ts_delay(open, -1) / open - 1)"
        self.add_feature("label_vol_h1", f"ts_std({future_o2o_ret}, 1)")
        self.add_feature("label_vol_h3", f"ts_std({future_o2o_ret}, 3)")
        self.add_feature("label_vol_h5", f"ts_std({future_o2o_ret}, 5)")
        self.add_feature(
            "label_sharpe_h1",
            f"ts_mean({future_o2o_ret}, 1) / (ts_std({future_o2o_ret}, 1) + 1e-12)",
        )
        self.add_feature(
            "label_sharpe_h3",
            f"ts_mean({future_o2o_ret}, 3) / (ts_std({future_o2o_ret}, 3) + 1e-12)",
        )
        self.add_feature(
            "label_sharpe_h5",
            f"ts_mean({future_o2o_ret}, 5) / (ts_std({future_o2o_ret}, 5) + 1e-12)",
        )

        clipped_hist_vol_20d = f"ts_clip({hist_vol_20d}, 1e-4)"
        self.add_feature("label_o2o_h1_vol_scaled", f"({o2o_h1}) / ({clipped_hist_vol_20d})")
        self.add_feature("label_o2o_h3_vol_scaled", f"({o2o_h3}) / ({clipped_hist_vol_20d})")
        self.add_feature("label_o2o_h5_vol_scaled", f"({o2o_h5}) / ({clipped_hist_vol_20d})")

        downside_ret = f"(({hist_daily_ret} - ts_abs({hist_daily_ret})) / 2)"
        downside_vol_20d = f"ts_std({downside_ret}, 20)"
        downside_vol_60d = f"ts_std({downside_ret}, 60)"
        self.add_feature("hist_downside_vol_20d", downside_vol_20d)
        self.add_feature("hist_downside_vol_60d", downside_vol_60d)

        clipped_downside_vol_20d = f"ts_clip({downside_vol_20d}, 1e-4)"
        self.add_feature("label_o2o_h1_sortino", f"({o2o_h1}) / ({clipped_downside_vol_20d})")
        self.add_feature("label_o2o_h3_sortino", f"({o2o_h3}) / ({clipped_downside_vol_20d})")
        self.add_feature("label_o2o_h5_sortino", f"({o2o_h5}) / ({clipped_downside_vol_20d})")

        drawdown_20d = "(ts_max(high, 20) - close) / ts_clip(ts_max(high, 20), 1e-4)"
        avg_drawdown_60d = f"ts_mean({drawdown_20d}, 60)"
        max_drawdown_60d = f"ts_max({drawdown_20d}, 60)"
        self.add_feature("hist_drawdown_20d", drawdown_20d)
        self.add_feature("hist_avg_drawdown_60d", avg_drawdown_60d)
        self.add_feature("hist_max_drawdown_60d", max_drawdown_60d)

        clipped_avg_drawdown_60d = f"ts_clip({avg_drawdown_60d}, 1e-3)"
        clipped_max_drawdown_60d = f"ts_clip({max_drawdown_60d}, 1e-3)"
        self.add_feature("label_o2o_h1_dd_scaled", f"({o2o_h1}) / ({clipped_avg_drawdown_60d})")
        self.add_feature("label_o2o_h3_dd_scaled", f"({o2o_h3}) / ({clipped_avg_drawdown_60d})")
        self.add_feature("label_o2o_h5_dd_scaled", f"({o2o_h5}) / ({clipped_avg_drawdown_60d})")
        self.add_feature("label_o2o_h1_mdd_scaled", f"({o2o_h1}) / ({clipped_max_drawdown_60d})")
        self.add_feature("label_o2o_h3_mdd_scaled", f"({o2o_h3}) / ({clipped_max_drawdown_60d})")
        self.add_feature("label_o2o_h5_mdd_scaled", f"({o2o_h5}) / ({clipped_max_drawdown_60d})")

    def neutralize_labels(
        self,
        industry_df: pl.DataFrame | None = None,
        cap_df: pl.DataFrame | None = None,
        vol_df: pl.DataFrame | None = None,
        *,
        mode: str | None = None,
        industry_col: str = "industry",
        cap_col: str = "cap",
        vol_col: str = "hist_vol_60d",
        weight_col: str | None = None,
        industry_method: str = "demean",
        cap_log: bool = True,
        fill_missing: str | None = "UNKNOWN",
        suffix: str = "_neu",
    ) -> None:
        self.result_df = neutralize_label_columns(
            result_df=self.result_df,
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

    def neutralize_size_labels(
        self,
        cap_df: pl.DataFrame,
        *,
        label_cols: list[str] | None = None,
        cap_col: str = "cap",
        weight_col: str | None = None,
        suffix: str = "_size_neu",
        replace: bool = False,
        clip: float = 3.0,
    ) -> None:
        self.result_df = neutralize_label_size_wls(
            result_df=self.result_df,
            cap_df=cap_df,
            label_cols=label_cols,
            cap_col=cap_col,
            weight_col=weight_col,
            include_quadratic=False,
            suffix=suffix,
            replace=replace,
            clip=clip,
        )

    def neutralize_size_nonlinear_labels(
        self,
        cap_df: pl.DataFrame,
        *,
        label_cols: list[str] | None = None,
        cap_col: str = "cap",
        weight_col: str | None = None,
        suffix: str = "_size_nonlinear_neu",
        replace: bool = False,
        clip: float = 3.0,
    ) -> None:
        self.result_df = neutralize_label_size_wls(
            result_df=self.result_df,
            cap_df=cap_df,
            label_cols=label_cols,
            cap_col=cap_col,
            weight_col=weight_col,
            include_quadratic=True,
            suffix=suffix,
            replace=replace,
            clip=clip,
        )

    def neutralize_barra_cne5_labels(
        self,
        barra_df: pl.DataFrame,
        *,
        label_cols: list[str] | None = None,
        factor_cols: list[str] | None = None,
        industry_cols: list[str] | None = None,
        weight_col: str | None = None,
        suffix: str = "_barra_cne5_neu",
        replace: bool = False,
        clip: float = 3.0,
        min_obs: int | None = None,
    ) -> None:
        self.result_df = neutralize_label_barra_wls(
            result_df=self.result_df,
            barra_df=barra_df,
            label_cols=label_cols,
            factor_cols=factor_cols,
            industry_cols=industry_cols,
            weight_col=weight_col,
            suffix=suffix,
            replace=replace,
            clip=clip,
            min_obs=min_obs,
        )

    def apply_price_limit_mask(
        self,
        price_df: pl.DataFrame,
        *,
        label_cols: list[str] | None = None,
        sample_weight_col: str | None = PRICE_LIMIT_MASK_COL,
        entry_open_shift: int = 1,
        feature_day_close_mask: bool = True,
        round_decimals: int = 2,
        missing_threshold: float = 0.01,
    ) -> dict[str, int | float]:
        self.result_df, stats = apply_price_limit_mask(
            result_df=self.result_df,
            price_df=price_df,
            label_cols=label_cols,
            sample_weight_col=sample_weight_col,
            entry_open_shift=entry_open_shift,
            feature_day_close_mask=feature_day_close_mask,
            round_decimals=round_decimals,
            missing_threshold=missing_threshold,
        )
        return stats

    def add_benchmark_excess_labels(
        self,
        bench_df: pl.DataFrame,
        *,
        suffix: str = "_excess",
        label_cols: list[str] | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.result_df = label_postprocess.add_benchmark_excess_labels(
            result_df=self.result_df,
            bench_df=bench_df,
            label_strategy_cls=self.__class__,
            train_period=self.data_periods[Segment.TRAIN],
            valid_period=self.data_periods[Segment.VALID],
            test_period=self.data_periods[Segment.TEST],
            interval=self.interval,
            enable_cache=getattr(self, "enable_cache", False),
            suffix=suffix,
            label_cols=label_cols,
            max_workers=max_workers,
        )


LabelStrategy = NeuLabel
