#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable Barra exposure analysis utilities.

The core daily exposure definition for each factor is:

    sum(valid_factor_weight * exposure) / sum(valid_factor_weight)

Missing exposures are filtered independently by factor. Daily outputs include
each factor's valid symbol count and retained raw weight for coverage auditing.

Inputs are local files only; this module does not depend on rqdatac.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


CNE5_FACTORS: list[str] = [
    "beta",
    "book_to_price",
    "size",
    "growth",
    "non_linear_size",
    "residual_volatility",
    "earnings_yield",
    "leverage",
    "liquidity",
    "momentum",
]

CNE5_FACTOR_LABELS: dict[str, str] = {
    "beta": "BETA",
    "book_to_price": "账面市值比",
    "size": "市值",
    "growth": "成长",
    "non_linear_size": "非线性市值",
    "residual_volatility": "残差波动率",
    "earnings_yield": "盈利预期",
    "leverage": "杠杆",
    "liquidity": "流动性",
    "momentum": "动量",
}

DATE_CANDIDATES = ("date", "trade_date", "datetime", "交易日")
SYMBOL_CANDIDATES = ("vt_symbol", "symbol", "order_book_id", "code", "万得代码")
SIGNAL_CANDIDATES = ("signal", "score", "prediction", "pred")


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv" or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"不支持的文件类型: {path}")


def _to_pandas_frame(data: Any, *, field_name: str) -> pd.DataFrame:
    """Convert a path, pandas frame, or frame exposing to_pandas()."""

    if isinstance(data, (str, Path)):
        return _read_table(data)
    if isinstance(data, pd.DataFrame):
        return data.copy()

    to_pandas = getattr(data, "to_pandas", None)
    if callable(to_pandas):
        frame = to_pandas()
        if isinstance(frame, pd.DataFrame):
            return frame
    raise TypeError(f"{field_name} 必须是 pandas/Polars DataFrame 或 parquet/csv 路径")


def _find_column(columns: Iterable[str], candidates: Iterable[str], *, field_name: str) -> str:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    raise ValueError(f"找不到 {field_name} 列，候选={list(candidates)}，实际={list(columns)}")


def normalize_vt_symbol(series: pd.Series) -> pd.Series:
    """Normalize common China stock code suffixes to vn.py vt_symbol style."""

    out = series.astype(str).str.strip()
    return (
        out.str.replace(r"\.XSHG$", ".SSE", regex=True)
        .str.replace(r"\.XSHE$", ".SZSE", regex=True)
        .str.replace(r"\.SH$", ".SSE", regex=True)
        .str.replace(r"\.SZ$", ".SZSE", regex=True)
    )


def _normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def load_barra_panel(path: str | Path, factors: list[str] | None = None) -> pd.DataFrame:
    """Load a wide Barra exposure panel as date/vt_symbol/factors.

    Expected input shape is one row per date and symbol, with factor columns.
    """

    factors = factors or CNE5_FACTORS
    df = _read_table(path)
    date_col = _find_column(df.columns, DATE_CANDIDATES, field_name="日期")
    symbol_col = _find_column(df.columns, SYMBOL_CANDIDATES, field_name="股票代码")

    missing = [factor for factor in factors if factor not in df.columns]
    if missing:
        raise ValueError(f"Barra exposure 缺少因子列: {missing}")

    out = df.loc[:, [date_col, symbol_col, *factors]].copy()
    out = out.rename(columns={date_col: "date", symbol_col: "vt_symbol"})
    out["date"] = _normalize_date(out["date"])
    out["vt_symbol"] = normalize_vt_symbol(out["vt_symbol"])
    for factor in factors:
        out[factor] = pd.to_numeric(out[factor], errors="coerce")
    out = out.dropna(subset=["date", "vt_symbol"])
    return out.sort_values(["date", "vt_symbol"]).drop_duplicates(["date", "vt_symbol"], keep="last")


def load_index_weight(path: str | Path) -> pd.DataFrame:
    """Load index or portfolio weights as date/vt_symbol/weight."""

    df = _read_table(path)
    date_col = _find_column(df.columns, DATE_CANDIDATES, field_name="日期")
    symbol_col = _find_column(df.columns, SYMBOL_CANDIDATES, field_name="股票代码")
    weight_col = _find_column(df.columns, ("weight", "指数权重", "持仓权重"), field_name="权重")

    out = df.loc[:, [date_col, symbol_col, weight_col]].copy()
    out = out.rename(columns={date_col: "date", symbol_col: "vt_symbol", weight_col: "weight"})
    out["date"] = _normalize_date(out["date"])
    out["vt_symbol"] = normalize_vt_symbol(out["vt_symbol"])
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out = out.dropna(subset=["date", "vt_symbol", "weight"])
    return out.sort_values(["date", "vt_symbol"]).drop_duplicates(["date", "vt_symbol"], keep="last")


def build_topk_equal_weights(
    signal: Any,
    *,
    top_k: int = 300,
    date_col: str | None = None,
    symbol_col: str | None = None,
    signal_col: str | None = None,
) -> pd.DataFrame:
    """Convert daily signal scores into deterministic TopK equal weights.

    Scores are sorted descending within each date. Ties are resolved by
    ``vt_symbol`` ascending so repeated runs produce identical constituents.
    If a date has fewer than ``top_k`` valid names, all valid names are kept
    and their weights still sum to one.
    """

    if top_k <= 0:
        raise ValueError(f"top_k 必须为正整数: {top_k}")

    df = _to_pandas_frame(signal, field_name="signal")
    date_col = date_col or _find_column(df.columns, DATE_CANDIDATES, field_name="日期")
    symbol_col = symbol_col or _find_column(df.columns, SYMBOL_CANDIDATES, field_name="股票代码")
    signal_col = signal_col or _find_column(df.columns, SIGNAL_CANDIDATES, field_name="信号")

    out = df.loc[:, [date_col, symbol_col, signal_col]].copy()
    out.columns = ["date", "vt_symbol", "signal"]
    out["date"] = _normalize_date(out["date"])
    out["vt_symbol"] = normalize_vt_symbol(out["vt_symbol"])
    out["signal"] = pd.to_numeric(out["signal"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["date", "vt_symbol", "signal"]
    )
    out = out.drop_duplicates(["date", "vt_symbol"], keep="last")
    out = out.sort_values(
        ["date", "signal", "vt_symbol"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    selected = out.groupby("date", sort=True, as_index=False).head(top_k).copy()
    selected["weight"] = 1.0 / selected.groupby("date")["vt_symbol"].transform("size")
    return selected.loc[:, ["date", "vt_symbol", "weight"]].reset_index(drop=True)


def _align_same_date(weights: pd.DataFrame, barra: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    return weights.merge(
        barra.loc[:, ["date", "vt_symbol", *factors]],
        on=["date", "vt_symbol"],
        how="inner",
    )


def _align_ffill_weight(weights: pd.DataFrame, barra: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    aligned_frames: list[pd.DataFrame] = []
    weight_cols = ["date", "vt_symbol", "weight"]
    barra_cols = ["date", "vt_symbol", *factors]

    # merge_asof requires sorting by the asof key first.
    for symbol, barra_day in barra.loc[:, barra_cols].groupby("vt_symbol", sort=False):
        weight_day = weights.loc[weights["vt_symbol"] == symbol, weight_cols]
        if weight_day.empty:
            continue
        merged = pd.merge_asof(
            barra_day.sort_values("date"),
            weight_day.sort_values("date"),
            on="date",
            direction="backward",
            allow_exact_matches=True,
        )
        merged["vt_symbol"] = symbol
        aligned_frames.append(merged)

    if not aligned_frames:
        return pd.DataFrame(columns=["date", "vt_symbol", "weight", *factors])
    return pd.concat(aligned_frames, ignore_index=True)


def compute_weighted_barra_exposure(
    weights: pd.DataFrame,
    barra: pd.DataFrame,
    factors: list[str] | None = None,
    *,
    date_align: str = "same_date",
) -> pd.DataFrame:
    """Compute daily Barra exposures with per-factor missing-value filters."""

    factors = factors or CNE5_FACTORS
    if date_align == "same_date":
        merged = _align_same_date(weights, barra, factors)
    elif date_align == "ffill_weight":
        merged = _align_ffill_weight(weights, barra, factors)
    else:
        raise ValueError(f"不支持的 date_align: {date_align}; 可选 same_date/ffill_weight")

    merged["weight"] = pd.to_numeric(merged["weight"], errors="coerce")
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=["date", "vt_symbol", "weight"])
    if merged.empty:
        return pd.DataFrame(columns=["date", "n_symbols", "raw_weight_sum", *factors])

    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for date, group in merged.groupby("date", sort=True):
        weight = group["weight"].to_numpy(dtype=float)
        weight_sum = float(np.nansum(weight))
        row: dict[str, float | int | pd.Timestamp] = {
            "date": date,
            "n_symbols": int(len(group)),
            "raw_weight_sum": weight_sum,
        }
        for factor in factors:
            exposure = pd.to_numeric(group[factor], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(exposure)
            factor_weight = weight[valid]
            factor_exposure = exposure[valid]
            factor_weight_sum = float(np.sum(factor_weight))

            row[f"{factor}_n_symbols"] = int(valid.sum())
            row[f"{factor}_raw_weight_sum"] = factor_weight_sum
            row[factor] = (
                float(np.sum(factor_weight * factor_exposure) / factor_weight_sum)
                if factor_weight_sum != 0
                else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def compute_annual_barra_summary(daily: pd.DataFrame, factors: list[str] | None = None) -> pd.DataFrame:
    """Summarize daily Barra exposure by calendar year."""

    factors = factors or CNE5_FACTORS
    if daily.empty:
        return pd.DataFrame(columns=["year", "n_dates", "avg_symbols", "avg_raw_weight_sum", *factors])

    df = daily.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    annual = (
        df.groupby("year", as_index=False)
        .agg(
            n_dates=("date", "nunique"),
            avg_symbols=("n_symbols", "mean"),
            avg_raw_weight_sum=("raw_weight_sum", "mean"),
            **{factor: (factor, "mean") for factor in factors},
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    return annual


def with_chinese_factor_labels(df: pd.DataFrame, factors: list[str] | None = None) -> pd.DataFrame:
    factors = factors or CNE5_FACTORS
    rename = {factor: CNE5_FACTOR_LABELS.get(factor, factor) for factor in factors}
    return df.rename(columns=rename)


def _format_markdown_table(df: pd.DataFrame, factors: list[str] | None = None) -> str:
    factors = factors or CNE5_FACTORS
    show = with_chinese_factor_labels(df.copy(), factors)
    float_cols = show.select_dtypes(include=["float", "float64", "float32"]).columns
    for col in float_cols:
        show[col] = show[col].map(lambda x: "" if pd.isna(x) else round(float(x), 4))
    return show.to_markdown(index=False)


def write_barra_report(
    daily: pd.DataFrame,
    annual: pd.DataFrame,
    latest: pd.DataFrame,
    output_dir: str | Path,
    *,
    factors: list[str] | None = None,
    title: str = "Barra CNE5 Exposure",
    date_align: str = "same_date",
) -> None:
    factors = factors or CNE5_FACTORS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily.to_csv(output_dir / "barra_cne5_daily.csv", index=False)
    annual.to_csv(output_dir / "barra_cne5_annual.csv", index=False)
    latest.to_csv(output_dir / "barra_cne5_latest.csv", index=False)

    latest_columns = [
        column
        for column in ["date", "n_symbols", "raw_weight_sum", *factors]
        if column in latest.columns
    ]
    latest_report = latest.loc[:, latest_columns]

    lines = [
        f"# {title}",
        "",
        f"- date_align: `{date_align}`",
        f"- n_daily_rows: `{len(daily)}`",
        "",
        "## Latest",
        "",
        _format_markdown_table(latest_report, factors) if not latest_report.empty else "No data.",
        "",
        "## Annual Mean",
        "",
        _format_markdown_table(annual, factors) if not annual.empty else "No data.",
        "",
        "## Outputs",
        "",
        "- `barra_cne5_daily.csv`",
        "- `barra_cne5_annual.csv`",
        "- `barra_cne5_latest.csv`",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_barra_exposure_analysis(
    *,
    barra_path: str | Path,
    weight_path: str | Path,
    output_dir: str | Path,
    factors: list[str] | None = None,
    date_align: str = "same_date",
    title: str = "Barra CNE5 Exposure",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factors = factors or CNE5_FACTORS
    barra = load_barra_panel(barra_path, factors=factors)
    weights = load_index_weight(weight_path)
    daily = compute_weighted_barra_exposure(weights, barra, factors=factors, date_align=date_align)
    annual = compute_annual_barra_summary(daily, factors=factors)
    latest = daily.tail(1).reset_index(drop=True)
    write_barra_report(
        daily=daily,
        annual=annual,
        latest=latest,
        output_dir=output_dir,
        factors=factors,
        title=title,
        date_align=date_align,
    )
    return daily, annual, latest


def run_topk_signal_barra_exposure_analysis(
    *,
    signal: Any,
    barra_path: str | Path,
    output_dir: str | Path,
    top_k: int = 300,
    factors: list[str] | None = None,
    date_align: str = "same_date",
    title: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analyze the Barra exposure of a daily TopK equal-weight signal.

    This is a target-portfolio analysis. It does not reconstruct positions
    after execution constraints such as n_drop, suspension, or lot sizing.
    """

    factors = factors or CNE5_FACTORS
    output_dir = Path(output_dir)
    weights = build_topk_equal_weights(signal, top_k=top_k)
    if weights.empty:
        raise ValueError("信号转换后的 TopK 权重为空")

    barra = load_barra_panel(barra_path, factors=factors)
    barra = barra[barra["date"].between(weights["date"].min(), weights["date"].max())]
    daily = compute_weighted_barra_exposure(
        weights,
        barra,
        factors=factors,
        date_align=date_align,
    )
    annual = compute_annual_barra_summary(daily, factors=factors)
    latest = daily.tail(1).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    weights.to_parquet(output_dir / "barra_topk_equal_weights.parquet", index=False)
    write_barra_report(
        daily=daily,
        annual=annual,
        latest=latest,
        output_dir=output_dir,
        factors=factors,
        title=title or f"Top{top_k} Equal Weight Barra CNE5 Exposure",
        date_align=date_align,
    )
    return daily, annual, latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze weighted Barra CNE5 exposure.")
    parser.add_argument("--barra-path", required=True, help="Barra wide panel parquet/csv path.")
    parser.add_argument("--weight-path", required=True, help="Weight parquet/csv path with date/vt_symbol/weight.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--date-align",
        choices=["same_date", "ffill_weight"],
        default="same_date",
        help="Date alignment rule between weights and Barra panel.",
    )
    parser.add_argument("--title", default="Barra CNE5 Exposure", help="Report title.")
    parser.add_argument(
        "--factors",
        nargs="*",
        default=None,
        help="Optional factor list. Defaults to CNE5 factors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, annual, latest = run_barra_exposure_analysis(
        barra_path=args.barra_path,
        weight_path=args.weight_path,
        output_dir=args.output_dir,
        factors=args.factors or CNE5_FACTORS,
        date_align=args.date_align,
        title=args.title,
    )
    print("LATEST")
    print(_format_markdown_table(latest, args.factors or CNE5_FACTORS))
    print()
    print("ANNUAL")
    print(_format_markdown_table(annual, args.factors or CNE5_FACTORS))


if __name__ == "__main__":
    main()
