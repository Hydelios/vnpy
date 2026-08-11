"""AlphaInspect V1 因子评价命令行入口。

示例::

    python -m vnpy.alpha.analysis.run_factor_assessment \
        --input master.parquet --feature-prefix calendar \
        --output-dir runtime/factor_assessment/calendar
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from .factor_assessment import DEFAULT_LABEL_COLUMNS, run_factor_assessment


def _split_csv(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return list(dict.fromkeys(result))


def _resolve_feature_columns(
    columns: list[str],
    *,
    names: list[str],
    prefixes: list[str],
) -> list[str]:
    missing = [name for name in names if name not in columns]
    if missing:
        raise ValueError(f"输入数据缺少指定因子列: {missing[:20]}")
    selected = list(names)
    selected.extend(
        column
        for column in columns
        if any(column.startswith(prefix) for prefix in prefixes)
    )
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("没有选中因子列；请使用 --feature 或 --feature-prefix")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行正式 AlphaInspect V1 因子评价")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="包含行情、标签和因子的 master parquet")
    source.add_argument("--feature-input", type=Path, help="仅包含键和因子的 parquet")
    parser.add_argument("--label-input", type=Path, help="与 --feature-input 配套的行情标签 parquet")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="factor_assessment")
    parser.add_argument("--feature", action="append", default=[], help="因子列；可重复或逗号分隔")
    parser.add_argument("--feature-prefix", action="append", default=[], help="因子列名前缀；可重复")
    parser.add_argument("--label", action="append", default=[], help="标签列；默认 h1/h3/h5")
    parser.add_argument("--scope", choices=("full", "valid"), default="full")
    parser.add_argument("--valid-start")
    parser.add_argument("--valid-end")
    parser.add_argument("--sort-metric", default="IC_IR_5D")
    parser.add_argument("--quantiles", type=int, default=10)
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--min-valid-day-ratio", type=float, default=0.8)
    parser.add_argument("--sample-filter-column")
    parser.add_argument("--html-output", choices=("all", "none"), default="all")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.scope == "valid" and not (args.valid_start and args.valid_end):
        raise ValueError("scope=valid 时必须同时提供 --valid-start 和 --valid-end")

    if args.feature_input is not None:
        if args.label_input is None:
            raise ValueError("使用 --feature-input 时必须提供 --label-input")
        feature_df = pl.read_parquet(args.feature_input)
        label_df = pl.read_parquet(args.label_input)
        label_side_cols = [
            column
            for column in label_df.columns
            if column not in {"datetime", "vt_symbol"}
            and column not in feature_df.columns
        ]
        master_df = label_df.select(
            "datetime", "vt_symbol", *label_side_cols
        ).join(feature_df, on=["datetime", "vt_symbol"], how="inner")
    else:
        if args.label_input is not None:
            raise ValueError("--label-input 只能与 --feature-input 一起使用")
        master_df = pl.read_parquet(args.input)
    feature_cols = _resolve_feature_columns(
        master_df.columns,
        names=_split_csv(args.feature),
        prefixes=_split_csv(args.feature_prefix),
    )
    labels = _split_csv(args.label) or list(DEFAULT_LABEL_COLUMNS)
    valid_period = (
        (args.valid_start, args.valid_end)
        if args.valid_start and args.valid_end
        else None
    )
    artifacts = run_factor_assessment(
        master_df=master_df,
        feature_cols=feature_cols,
        valid_period=valid_period,
        output_dir=args.output_dir,
        run_name=args.run_name,
        config={
            "scope": args.scope,
            "label_columns": labels,
            "sort_metric": args.sort_metric,
            "quantiles": args.quantiles,
            "html_output": args.html_output,
            "min_coverage": args.min_coverage,
            "min_valid_day_ratio": args.min_valid_day_ratio,
            "sample_filter_column": args.sample_filter_column,
        },
    )
    print(f"summary={artifacts.summary_csv}")
    print(f"html={artifacts.index_html}")
    print(f"meta={artifacts.meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
