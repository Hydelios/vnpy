from __future__ import annotations

import polars as pl

__all__ = ["add_benchmark_excess_labels"]


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
            col
            for col in result_df.columns
            if col.startswith("label_") and not col.endswith(suffix)
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

    available = [col for col in label_cols if col in bench.result_df.columns]
    if not available:
        return result_df

    bench_label = bench.result_df.select(["datetime", *available]).rename(
        {col: f"bench_{col}" for col in available}
    )
    bench_cols = [f"bench_{col}" for col in available]

    return (
        result_df.join(bench_label, on="datetime", how="left")
        .with_columns(
            [
                (pl.col(col) - pl.col(f"bench_{col}")).alias(f"{col}{suffix}")
                for col in available
            ]
        )
        .drop(bench_cols)
    )
