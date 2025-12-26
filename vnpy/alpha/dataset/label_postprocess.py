from __future__ import annotations

import polars as pl

from vnpy.alpha.dataset import process_cap_neutralize, process_industry_neutralize
from vnpy.alpha.dataset.neutralize_function import cs_demean
from vnpy.alpha.dataset.utility import DataProxy


def _neutralize_market(df: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    results: list[pl.DataFrame] = []
    for name in names:
        feature_df = df.select(
            "datetime",
            "vt_symbol",
            pl.col(name).cast(pl.Float64, strict=False).alias(name),
        )
        feature = DataProxy(feature_df)
        adjusted = cs_demean(feature).df.rename({"data": name})
        results.append(adjusted)

    base = df.drop(names)
    for adjusted in results:
        base = base.join(adjusted, on=["datetime", "vt_symbol"], how="left")

    return base.select(df.columns)


def _neutralize_industry_cap_fe(
    df: pl.DataFrame,
    names: list[str],
    industry_col: str,
    cap_col: str,
    cap_log: bool,
    fill_missing: str | None,
) -> pl.DataFrame:
    if industry_col not in df.columns or cap_col not in df.columns:
        return df
    if not names:
        return df

    base = df.with_columns(
        pl.col(industry_col).cast(pl.Utf8).alias("_industry"),
    )
    if fill_missing is not None:
        base = base.with_columns(pl.col("_industry").fill_null(fill_missing))

    cap_expr = (
        pl.when(pl.col(cap_col).is_not_null() & (pl.col(cap_col) > 0))
        .then(pl.col(cap_col))
        .otherwise(None)
    )
    if cap_log:
        cap_expr = cap_expr.log1p()

    base = base.with_columns(
        cap_expr.cast(pl.Float64).alias("_x"),
    ).with_columns(
        pl.col("_x").mean().over(["datetime", "_industry"]).alias("_x_mean"),
    ).with_columns(
        (pl.col("_x") - pl.col("_x_mean")).alias("_x_tilde"),
    ).with_columns(
        (pl.col("_x_tilde").pow(2)).sum().over("datetime").alias("_sum_x2"),
    )

    results: list[pl.DataFrame] = []
    for name in names:
        adjusted = base.with_columns(
            pl.col(name).cast(pl.Float64, strict=False).alias("_y"),
        ).with_columns(
            pl.col("_y").mean().over(["datetime", "_industry"]).alias("_y_mean"),
        ).with_columns(
            (pl.col("_y") - pl.col("_y_mean")).alias("_y_tilde"),
        ).with_columns(
            (pl.col("_x_tilde") * pl.col("_y_tilde"))
            .sum()
            .over("datetime")
            .alias("_sum_xy"),
        ).with_columns(
            (pl.col("_sum_xy") / (pl.col("_sum_x2") + 1e-12)).alias("_beta"),
        ).with_columns(
            (pl.col("_y_tilde") - pl.col("_beta") * pl.col("_x_tilde")).alias(name),
        ).select(["datetime", "vt_symbol", name])
        results.append(adjusted)

    base = df.drop(names)
    for adjusted in results:
        base = base.join(adjusted, on=["datetime", "vt_symbol"], how="left")

    return base.select(df.columns)


def neutralize_labels(
    result_df: pl.DataFrame,
    industry_df: pl.DataFrame | None = None,
    cap_df: pl.DataFrame | None = None,
    *,
    mode: str | None = None,
    industry_col: str = "industry",
    cap_col: str = "cap",
    industry_method: str = "demean",
    cap_log: bool = True,
    fill_missing: str | None = "UNKNOWN",
    suffix: str = "_neu",
) -> pl.DataFrame:
    if result_df is None or result_df.is_empty():
        return result_df

    label_cols = [c for c in result_df.columns if c.startswith("label_")]
    if not label_cols:
        return result_df

    has_industry = (
        industry_df is not None
        and not industry_df.is_empty()
        and industry_col in industry_df.columns
    )
    has_cap = (
        cap_df is not None
        and not cap_df.is_empty()
        and cap_col in cap_df.columns
    )
    if mode is None:
        apply_market = False
        apply_industry = has_industry
        apply_cap = has_cap
        apply_industry_cap_fe = False
    elif mode == "market":
        apply_market = True
        apply_industry = False
        apply_cap = False
        apply_industry_cap_fe = False
    elif mode == "industry":
        apply_market = False
        apply_industry = True
        apply_cap = False
        apply_industry_cap_fe = False
    elif mode == "industry_cap":
        apply_market = False
        apply_industry = True
        apply_cap = True
        apply_industry_cap_fe = True
    elif mode == "cap":
        apply_market = False
        apply_industry = False
        apply_cap = True
        apply_industry_cap_fe = False
    else:
        raise ValueError(f"unknown neutralize mode: {mode}")

    if not apply_market and not apply_industry and not apply_cap:
        return result_df

    tmp = result_df.select(["datetime", "vt_symbol", *label_cols]).sort(
        ["vt_symbol", "datetime"]
    )
    if apply_market:
        tmp = _neutralize_market(tmp, label_cols)

    if apply_industry_cap_fe:
        if not (has_industry and has_cap):
            return result_df
        ind = industry_df.select(
            ["datetime", "vt_symbol", industry_col]
        ).sort(["vt_symbol", "datetime"])
        cap = cap_df.select(
            ["datetime", "vt_symbol", cap_col]
        ).sort(["vt_symbol", "datetime"])
        tmp = tmp.join_asof(ind, on="datetime", by="vt_symbol", strategy="backward")
        tmp = tmp.join_asof(cap, on="datetime", by="vt_symbol", strategy="backward")
        tmp = _neutralize_industry_cap_fe(
            tmp,
            label_cols,
            industry_col=industry_col,
            cap_col=cap_col,
            cap_log=cap_log,
            fill_missing=fill_missing,
        )
    else:
        if apply_industry and has_industry:
            ind = industry_df.select(
                ["datetime", "vt_symbol", industry_col]
            ).sort(["vt_symbol", "datetime"])
            tmp = tmp.join_asof(ind, on="datetime", by="vt_symbol", strategy="backward")
            tmp = process_industry_neutralize(
                tmp,
                industry_col=industry_col,
                method=industry_method,
                names=label_cols,
                fill_missing=fill_missing,
            )

        if apply_cap and has_cap:
            cap = cap_df.select(
                ["datetime", "vt_symbol", cap_col]
            ).sort(["vt_symbol", "datetime"])
            tmp = tmp.join_asof(cap, on="datetime", by="vt_symbol", strategy="backward")
            tmp = process_cap_neutralize(
                tmp,
                cap_col=cap_col,
                log_cap=cap_log,
                names=label_cols,
            )

    rename_map = {c: f"{c}{suffix}" for c in label_cols}
    tmp = tmp.rename(rename_map).select(
        ["datetime", "vt_symbol", *rename_map.values()]
    )
    return result_df.join(tmp, on=["datetime", "vt_symbol"], how="left")


def add_benchmark_excess_labels(
    result_df: pl.DataFrame,
    bench_df: pl.DataFrame,
    *,
    label_strategy_cls: type,
    train_period: tuple[str, str],
    valid_period: tuple[str, str],
    test_period: tuple[str, str],
    interval: str,
    enable_cache: bool = False,
    suffix: str = "_excess",
    label_cols: list[str] | None = None,
    max_workers: int | None = None,
) -> pl.DataFrame:
    if result_df is None or result_df.is_empty():
        return result_df
    if bench_df is None or bench_df.is_empty():
        return result_df

    if label_cols is None:
        label_cols = [
            c
            for c in result_df.columns
            if c.startswith("label_") and not c.endswith(suffix)
        ]
    if not label_cols:
        return result_df

    bench = label_strategy_cls(
        df=bench_df,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        interval=interval,
        enable_cache=enable_cache,
    )
    bench.calculate_features(max_workers=max_workers)
    bench.merge_feature_results()

    available = [c for c in label_cols if c in bench.result_df.columns]
    if not available:
        return result_df

    bench_label = bench.result_df.select(
        ["datetime", *available]
    ).rename({c: f"bench_{c}" for c in available})

    base = result_df.join(bench_label, on="datetime", how="left")
    base = base.with_columns(
        [
            (pl.col(c) - pl.col(f"bench_{c}")).alias(f"{c}{suffix}")
            for c in available
        ]
    ).drop([f"bench_{c}" for c in available])

    return base
