#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable signal post-processing gates.

The main use case is daily cross-sectional filtering, such as removing the
lowest 30% by market cap or the lowest 10% by recent turnover.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:  # pragma: no cover - polars is optional for callers.
    pl = None


DATE_CANDIDATES = ("datetime", "date", "trade_date", "交易日")
SYMBOL_CANDIDATES = ("vt_symbol", "symbol", "order_book_id", "code", "万得代码")


@dataclass(frozen=True)
class QuantileDropRule:
    """Daily cross-sectional metric gate.

    Args:
        name: Rule name used in diagnostics.
        metric_col: Metric column used for ranking.
        drop_ratio: Fraction of valid metric rows to remove per date.
        ascending: True removes the lowest tail; False removes the highest tail.
        missing_policy: "drop" removes rows with missing metric; "keep" keeps them.
    """

    name: str
    metric_col: str
    drop_ratio: float
    ascending: bool = True
    missing_policy: str = "drop"


@dataclass
class SignalPostprocessResult:
    signal: pd.DataFrame
    daily: pd.DataFrame
    summary: pd.DataFrame


def load_table(path: str | Path) -> pd.DataFrame:
    """Load parquet/csv/csv.gz as pandas."""

    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv" or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"不支持的文件类型: {path}")


def _to_pandas(df: pd.DataFrame | Any) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df.copy()
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.to_pandas()
    raise TypeError(f"不支持的数据类型: {type(df)!r}")


def _find_column(columns: Iterable[str], candidates: Iterable[str], *, field_name: str) -> str:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    raise ValueError(f"找不到 {field_name} 列，候选={list(candidates)}，实际={list(columns)}")


def normalize_vt_symbol(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    out = (
        out.str.replace(r"\.XSHG$", ".SSE", regex=True)
        .str.replace(r"\.XSHE$", ".SZSE", regex=True)
        .str.replace(r"\.SH$", ".SSE", regex=True)
        .str.replace(r"\.SZ$", ".SZSE", regex=True)
    )
    invalid = out.str.lower().isin({"", "nan", "none", "null", "<na>"})
    return out.mask(invalid, pd.NA)


def normalize_signal_frame(
    signal: pd.DataFrame | Any,
    *,
    date_col: str | None = None,
    symbol_col: str | None = None,
    signal_col: str = "signal",
    output_date_col: str = "datetime",
) -> pd.DataFrame:
    """Normalize signal frame to datetime/vt_symbol/signal plus extra columns."""

    df = _to_pandas(signal)
    date_col = date_col or _find_column(df.columns, DATE_CANDIDATES, field_name="日期")
    symbol_col = symbol_col or _find_column(df.columns, SYMBOL_CANDIDATES, field_name="股票代码")
    if signal_col not in df.columns:
        raise ValueError(f"signal 缺少列: {signal_col}")

    out = df.copy()
    out[output_date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out["vt_symbol"] = normalize_vt_symbol(out[symbol_col])
    out[signal_col] = pd.to_numeric(out[signal_col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[output_date_col, "vt_symbol", signal_col])
    out = out.drop_duplicates([output_date_col, "vt_symbol"], keep="last")

    front_cols = [output_date_col, "vt_symbol", signal_col]
    other_cols = [col for col in out.columns if col not in set(front_cols)]
    return out.loc[:, [*front_cols, *other_cols]].sort_values([output_date_col, "vt_symbol"]).reset_index(drop=True)


def normalize_metric_frame(
    metrics: pd.DataFrame | Any,
    *,
    metric_col: str,
    date_col: str | None = None,
    symbol_col: str | None = None,
    output_date_col: str = "datetime",
) -> pd.DataFrame:
    """Normalize metric frame to datetime/vt_symbol/metric_col."""

    df = _to_pandas(metrics)
    date_col = date_col or _find_column(df.columns, DATE_CANDIDATES, field_name="日期")
    symbol_col = symbol_col or _find_column(df.columns, SYMBOL_CANDIDATES, field_name="股票代码")
    if metric_col not in df.columns:
        raise ValueError(f"metric frame 缺少列: {metric_col}")

    out = df.loc[:, [date_col, symbol_col, metric_col]].copy()
    out[output_date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out["vt_symbol"] = normalize_vt_symbol(out[symbol_col])
    out[metric_col] = pd.to_numeric(out[metric_col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[output_date_col, "vt_symbol"])
    return (
        out.loc[:, [output_date_col, "vt_symbol", metric_col]]
        .sort_values([output_date_col, "vt_symbol"])
        .drop_duplicates([output_date_col, "vt_symbol"], keep="last")
        .reset_index(drop=True)
    )


def _validate_rule(rule: QuantileDropRule) -> None:
    if not 0 <= rule.drop_ratio < 1:
        raise ValueError(f"{rule.name}: drop_ratio 必须在 [0, 1) 内，收到 {rule.drop_ratio}")
    if rule.missing_policy not in {"drop", "keep"}:
        raise ValueError(f"{rule.name}: missing_policy 只支持 drop/keep，收到 {rule.missing_policy}")


def apply_quantile_drop(
    signal: pd.DataFrame | Any,
    metrics: pd.DataFrame | Any,
    rule: QuantileDropRule,
    *,
    date_col: str = "datetime",
    signal_col: str = "signal",
) -> SignalPostprocessResult:
    """Apply one daily quantile drop rule to a signal frame."""

    _validate_rule(rule)
    signal_df = normalize_signal_frame(signal, signal_col=signal_col, output_date_col=date_col)
    metric_df = normalize_metric_frame(metrics, metric_col=rule.metric_col, output_date_col=date_col).rename(
        columns={rule.metric_col: "_gate_metric"}
    )

    merged = signal_df.merge(metric_df, on=[date_col, "vt_symbol"], how="left")
    rows: list[pd.DataFrame] = []
    daily_rows: list[dict[str, Any]] = []

    for dt, day in merged.groupby(date_col, sort=True):
        day = day.copy()
        candidate_count = int(len(day))
        missing_count = int(day["_gate_metric"].isna().sum())
        valid = day.loc[day["_gate_metric"].notna()].copy()
        valid_count = int(len(valid))
        drop_cutoff = int(valid_count * rule.drop_ratio)

        if valid_count:
            valid = valid.sort_values(
                ["_gate_metric", "vt_symbol"],
                ascending=[rule.ascending, True],
                kind="mergesort",
            )
            valid["_gate_rank"] = range(valid_count)
            kept_valid = valid.loc[valid["_gate_rank"] >= drop_cutoff].drop(columns=["_gate_rank"])
        else:
            kept_valid = valid

        if rule.missing_policy == "keep":
            kept_missing = day.loc[day["_gate_metric"].isna()].copy()
            kept = pd.concat([kept_valid, kept_missing], ignore_index=True)
        else:
            kept = kept_valid

        kept = kept.drop(columns=["_gate_metric"], errors="ignore")
        kept = kept.sort_values(["vt_symbol"], kind="mergesort")
        rows.append(kept)

        kept_count = int(len(kept))
        daily_rows.append(
            {
                "rule": rule.name,
                "datetime": dt,
                "candidate_count_before": candidate_count,
                "missing_metric_count": missing_count,
                "valid_metric_count": valid_count,
                "drop_cutoff": drop_cutoff,
                "dropped_by_rule_count": drop_cutoff,
                "kept_count_after": kept_count,
                "removed_total_count": candidate_count - kept_count,
                "kept_ratio": kept_count / candidate_count if candidate_count else 0.0,
                "drop_ratio_on_valid": drop_cutoff / valid_count if valid_count else pd.NA,
                "metric_col": rule.metric_col,
                "ascending": rule.ascending,
                "missing_policy": rule.missing_policy,
            }
        )

    filtered = pd.concat(rows, ignore_index=True) if rows else signal_df.iloc[0:0].copy()
    filtered = filtered.sort_values([date_col, "vt_symbol"]).reset_index(drop=True)
    daily = pd.DataFrame(daily_rows)
    summary = summarize_daily_stats(daily, rule)
    return SignalPostprocessResult(signal=filtered, daily=daily, summary=summary)


def summarize_daily_stats(daily: pd.DataFrame, rule: QuantileDropRule) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(
            [
                {
                    "rule": rule.name,
                    "metric_col": rule.metric_col,
                    "drop_ratio": rule.drop_ratio,
                    "signal_rows_before": 0,
                    "signal_rows_after": 0,
                    "n_dates": 0,
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "rule": rule.name,
                "metric_col": rule.metric_col,
                "drop_ratio": rule.drop_ratio,
                "ascending": rule.ascending,
                "missing_policy": rule.missing_policy,
                "signal_rows_before": int(daily["candidate_count_before"].sum()),
                "signal_rows_after": int(daily["kept_count_after"].sum()),
                "removed_total_count": int(daily["removed_total_count"].sum()),
                "dropped_by_rule_count": int(daily["dropped_by_rule_count"].sum()),
                "missing_metric_count": int(daily["missing_metric_count"].sum()),
                "n_dates": int(daily["datetime"].nunique()),
                "avg_candidate_count_before": float(daily["candidate_count_before"].mean()),
                "avg_valid_metric_count": float(daily["valid_metric_count"].mean()),
                "avg_kept_count_after": float(daily["kept_count_after"].mean()),
                "avg_kept_ratio": float(daily["kept_ratio"].mean()),
                "avg_drop_ratio_on_valid": float(pd.to_numeric(daily["drop_ratio_on_valid"], errors="coerce").mean()),
            }
        ]
    )


def apply_signal_postprocess(
    signal: pd.DataFrame | Any,
    metric_panels: dict[str, pd.DataFrame | Any],
    rules: list[QuantileDropRule],
    *,
    date_col: str = "datetime",
    signal_col: str = "signal",
) -> SignalPostprocessResult:
    """Apply multiple rules sequentially."""

    current = normalize_signal_frame(signal, signal_col=signal_col, output_date_col=date_col)
    daily_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []

    for rule in rules:
        if rule.metric_col not in metric_panels:
            raise ValueError(f"缺少 metric panel: {rule.metric_col}")
        result = apply_quantile_drop(
            current,
            metric_panels[rule.metric_col],
            rule,
            date_col=date_col,
            signal_col=signal_col,
        )
        current = result.signal
        daily_parts.append(result.daily)
        summary_parts.append(result.summary)

    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    return SignalPostprocessResult(signal=current, daily=daily, summary=summary)


def write_postprocess_outputs(result: SignalPostprocessResult, output_dir: str | Path, *, file_stem: str = "signal_filtered") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.signal.to_parquet(output_dir / f"{file_stem}.parquet", index=False)
    result.daily.to_csv(output_dir / "postprocess_daily.csv", index=False)
    result.summary.to_csv(output_dir / "postprocess_summary.csv", index=False)

    lines = [
        "# Signal Postprocess",
        "",
        "## Summary",
        "",
        result.summary.to_markdown(index=False) if not result.summary.empty else "No rules.",
        "",
        "## Outputs",
        "",
        f"- `{file_stem}.parquet`",
        "- `postprocess_daily.csv`",
        "- `postprocess_summary.csv`",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_rules(path: str | Path) -> list[QuantileDropRule]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("rules json 必须是 list")
    return [QuantileDropRule(**item) for item in payload]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply daily cross-sectional signal postprocess gates.")
    parser.add_argument("--signal-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rules-json", required=True, help="JSON list of QuantileDropRule fields.")
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Metric mapping in metric_col=path form. Repeat for multiple metrics.",
    )
    parser.add_argument("--file-stem", default="signal_filtered")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signal = load_table(args.signal_path)
    rules = _load_rules(args.rules_json)
    metric_panels: dict[str, pd.DataFrame] = {}
    for item in args.metric:
        if "=" not in item:
            raise ValueError(f"--metric 格式应为 metric_col=path，收到: {item}")
        metric_col, path = item.split("=", 1)
        metric_panels[metric_col] = load_table(path)

    result = apply_signal_postprocess(signal, metric_panels, rules)
    write_postprocess_outputs(result, args.output_dir, file_stem=args.file_stem)
    print(result.summary.to_markdown(index=False))


if __name__ == "__main__":
    main()
