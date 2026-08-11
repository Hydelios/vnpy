"""Filter full signal candidates before ranking and backtesting.

用途
----
模型完成预测后，对完整信号候选池执行每日市值/成交量尾部过滤，再由策略
重新排名并递补得到新的 Top-K。这里不是只删除原始 Top300 中的股票。

支持配置：

* ``cap10_volume10``：每日市值后 10% 或成交量后 10%，取并集；
* ``cap30``：每日市值后 30%，不应用成交量尾部过滤。

训练后过滤默认不处理 ST、停牌和一字涨跌停；这些交易状态由正式回测的
T+1 开盘可交易逻辑处理。缺失市值会被删除，``cap10_volume10`` 的缺失
成交量也会被删除。

最简命令行示例
--------------
使用 AlphaLab 自动加载市值和正式日线 ``volume``：

.. code-block:: bash

    conda run --no-capture-output -n python311 python \
      /home/hyd/research/vnpy_hub/vnpy/vnpy/alpha/signal/filter_signal_candidates.py \
      --input signal_stitched.parquet \
      --lab-path /home/hyd/research/vnpy_hub/playground/alpha_research/trade_real/lab/all_stocks \
      --profile cap10_volume10

切换为市值后 30%：

.. code-block:: bash

    conda run --no-capture-output -n python311 python \
      /home/hyd/research/vnpy_hub/vnpy/vnpy/alpha/signal/filter_signal_candidates.py \
      --input signal_stitched.parquet \
      --lab-path /home/hyd/research/vnpy_hub/playground/alpha_research/trade_real/lab/all_stocks \
      --profile cap30

也可以显式提供面板：

.. code-block:: bash

    python filter_signal_candidates.py \
      --input signal_stitched.parquet \
      --cap-data cap.parquet \
      --volume-data volume.parquet \
      --profile cap10_volume10

Python 示例
-----------

.. code-block:: python

    from vnpy.alpha.signal.filter_signal_candidates import filter_signal_candidates

    result = filter_signal_candidates(
        signal,
        cap_panel=cap,
        volume_panel=volume,
        profile="cap10_volume10",
    )
    filtered_signal = result.filtered

输入列
------
信号至少包含 ``datetime``、``vt_symbol``、``signal``。市值面板包含
``datetime``、``vt_symbol``、``cap``；成交量面板包含
``datetime``、``vt_symbol``、``volume``。

默认输出
--------
未指定输出路径时，在原始信号旁生成：

* ``<signal>.<profile>.filtered.parquet``：过滤后的完整候选信号；
* ``*_filter_daily.parquet``：全候选池逐日过滤审计；
* ``*_top300_impact.parquet``：原始 Top300 中被过滤的数量和比例；
* ``*.json``：过滤汇总和各输出路径。

正式回测应读取 ``filtered.parquet``，再对该候选池按 signal 排名选 Top300。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from vnpy.alpha.dataset.universe_filter import (
    FILTER_PROFILES,
    KEY_COLUMNS,
    UniverseFilterProfile,
    UniverseFilterResult,
    filter_universe,
)
from vnpy.alpha.lab import AlphaLab
from vnpy.trader.constant import Interval


def filter_signal_candidates(
    signal: pl.DataFrame,
    *,
    cap_panel: pl.DataFrame,
    volume_panel: pl.DataFrame | None,
    profile: str | UniverseFilterProfile,
) -> UniverseFilterResult:
    """Delete daily cap/volume tails from the full signal candidate panel."""
    return filter_universe(
        signal,
        cap_panel=cap_panel,
        volume_panel=volume_panel,
        status_panel=None,
        profile=profile,
        filter_status=False,
        stage="after prediction and before signal ranking/backtesting",
    )


def calculate_top_k_impact(
    original: pl.DataFrame,
    filtered: pl.DataFrame,
    *,
    top_k: int = 300,
) -> pl.DataFrame:
    """Count how many names from each original Top-K were made ineligible."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    required = {*KEY_COLUMNS, "signal"}
    missing = required.difference(original.columns)
    if missing:
        raise ValueError(f"signal missing required columns: {sorted(missing)}")
    top = (
        original.select(*KEY_COLUMNS, pl.col("signal").cast(pl.Float64))
        .filter(pl.col("signal").is_finite())
        .sort(["datetime", "signal", "vt_symbol"], descending=[False, True, False])
        .group_by("datetime", maintain_order=True)
        .head(top_k)
        .select(KEY_COLUMNS)
    )
    eligible = filtered.select(KEY_COLUMNS)
    removed = top.join(eligible, on=KEY_COLUMNS, how="anti")
    base = top.group_by("datetime").len(name="original_top_k_count")
    return (
        base.join(
            removed.group_by("datetime").len(name="removed_count"),
            on="datetime",
            how="left",
        )
        .with_columns(
            pl.col("removed_count").fill_null(0),
            (pl.col("removed_count").fill_null(0) / pl.col("original_top_k_count")).alias(
                "removed_rate"
            ),
        )
        .sort("datetime")
    )


def _read_table(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pl.read_csv(path, try_parse_dates=True, infer_schema_length=10_000)
    raise ValueError(f"unsupported table format: {path}")


def _normalize_cap_columns(frame: pl.DataFrame) -> pl.DataFrame:
    if "cap" in frame.columns:
        return frame
    available = [column for column in ["market_cap_2", "market_cap"] if column in frame.columns]
    if not available:
        raise ValueError("cap data must contain cap or market_cap_2/market_cap")
    return frame.with_columns(
        pl.coalesce([pl.col(column) for column in available]).cast(pl.Float64).alias("cap")
    )


def load_filter_panels_from_lab(
    signal: pl.DataFrame,
    lab_path: str | Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load cap and same-day daily volume from an AlphaLab store."""
    normalized = signal.with_columns(pl.col("datetime").cast(pl.Datetime))
    start = normalized["datetime"].min()
    end = normalized["datetime"].max()
    symbols = sorted(normalized["vt_symbol"].cast(pl.Utf8).unique().to_list())
    if start is None or end is None or not symbols:
        raise ValueError("signal has no usable datetime/vt_symbol keys")

    lab = AlphaLab(str(lab_path))
    raw_cap = lab.load_common_valuation_factors(
        columns=["market_cap_2", "market_cap"],
        start=start,
        end=end,
    )
    if raw_cap is None or raw_cap.is_empty():
        raise RuntimeError("AlphaLab returned an empty market-cap panel")
    cap = (
        _normalize_cap_columns(raw_cap)
        .select(*KEY_COLUMNS, "cap")
        .unique(subset=KEY_COLUMNS, keep="last")
    )

    bars = lab.load_bar_df(
        vt_symbols=symbols,
        interval=Interval.DAILY,
        start=start,
        end=end,
        extended_days=0,
    )
    if bars is None or bars.is_empty():
        raise RuntimeError("AlphaLab returned an empty daily volume panel")
    volume = bars.select(
        *KEY_COLUMNS,
        pl.col("volume").cast(pl.Float64),
    ).unique(subset=KEY_COLUMNS, keep="last")
    return cap, volume


def _default_output(input_path: Path, profile: str) -> Path:
    return input_path.with_name(f"{input_path.stem}.{profile}.filtered.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练后对完整信号候选池执行市值/成交量尾部过滤")
    parser.add_argument("--input", required=True, type=Path, help="完整信号 parquet/csv")
    parser.add_argument("--profile", required=True, choices=sorted(FILTER_PROFILES))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--lab-path", type=Path, help="自动加载市值和正式日线成交量的 AlphaLab 路径")
    source.add_argument("--cap-data", type=Path, help="含 cap 或 market_cap_2/market_cap 的数据")
    parser.add_argument("--volume-data", type=Path, default=None, help="含 volume 的数据；使用 --cap-data 时按配置需要")
    parser.add_argument("--output", type=Path, default=None, help="过滤后完整候选信号")
    parser.add_argument("--audit-output", type=Path, default=None, help="逐日过滤审计 parquet")
    parser.add_argument("--summary-output", type=Path, default=None, help="过滤汇总 JSON")
    parser.add_argument("--top-k", type=int, default=300, help="额外审计原始Top-K被剔除数量")
    parser.add_argument("--top-k-output", type=Path, default=None, help="Top-K影响审计 parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signal = _read_table(args.input)
    if args.lab_path is not None:
        cap, volume = load_filter_panels_from_lab(signal, args.lab_path)
        input_sources = {"lab_path": str(args.lab_path)}
    else:
        if args.cap_data is None:
            raise ValueError("--cap-data is required when --lab-path is not used")
        cap = _normalize_cap_columns(_read_table(args.cap_data))
        volume = _read_table(args.volume_data) if args.volume_data is not None else None
        input_sources = {
            "cap_data": str(args.cap_data),
            "volume_data": str(args.volume_data) if args.volume_data is not None else None,
        }

    result = filter_signal_candidates(
        signal,
        cap_panel=cap,
        volume_panel=volume,
        profile=args.profile,
    )
    impact = calculate_top_k_impact(signal, result.filtered, top_k=args.top_k)

    output = args.output or _default_output(args.input, args.profile)
    audit_output = args.audit_output or output.with_name(f"{output.stem}_filter_daily.parquet")
    summary_output = args.summary_output or output.with_suffix(".json")
    top_k_output = args.top_k_output or output.with_name(f"{output.stem}_top{args.top_k}_impact.parquet")
    for path in [output, audit_output, summary_output, top_k_output]:
        path.parent.mkdir(parents=True, exist_ok=True)
    result.filtered.write_parquet(output, compression="zstd")
    result.daily_audit.write_parquet(audit_output, compression="zstd")
    impact.write_parquet(top_k_output, compression="zstd")

    payload: dict[str, Any] = {
        **result.summary,
        "inputs": {"signal": str(args.input), **input_sources},
        "output": str(output),
        "daily_audit": str(audit_output),
        "top_k_impact": str(top_k_output),
        "top_k": args.top_k,
        "top_k_avg_removed_count": float(impact["removed_count"].mean()),
        "top_k_avg_removed_rate": float(impact["removed_rate"].mean()),
    }
    summary_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"信号候选池过滤完成：{result.summary['rows_before']:,} -> "
        f"{result.summary['rows_after']:,}，原始Top{args.top_k}日均剔除 "
        f"{payload['top_k_avg_removed_count']:.2f} 只，输出 {output}"
    )


if __name__ == "__main__":
    main()
