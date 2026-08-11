"""Physically filter training samples before label processing and model fit.

用途
----
在 label 中性化、归一化、CV 切分和模型训练之前，物理删除不合格样本：

1. T 日 ST、停牌、无行情、一字涨停、一字跌停；
2. ``cap10_volume10``：每日市值后 10% 或成交量后 10%，取并集；
3. ``cap30``：每日市值后 30%，不读取成交量尾部。

这里是物理删除行，不是把样本权重设为 0。市值和成交量分别在每日截面
独立排名，删除数量为 ``floor(valid_count * ratio)``。

命令行示例
----------
推荐使用仓库的 python311 环境：

.. code-block:: bash

    conda run --no-capture-output -n python311 python \
      /home/hyd/research/vnpy_hub/vnpy/vnpy/alpha/model/prepare_training_samples.py \
      --input raw_train.parquet \
      --cap-data cap.parquet \
      --market-data market_status.parquet \
      --profile cap10_volume10

切换为市值后 30%：

.. code-block:: bash

    conda run --no-capture-output -n python311 python \
      /home/hyd/research/vnpy_hub/vnpy/vnpy/alpha/model/prepare_training_samples.py \
      --input raw_train.parquet \
      --cap-data cap.parquet \
      --market-data market_status.parquet \
      --profile cap30

Python 示例
-----------
训练流水线中应直接调用函数，避免先写临时文件再读回：

.. code-block:: python

    from vnpy.alpha.model.prepare_training_samples import prepare_training_samples

    result = prepare_training_samples(
        train_raw,
        cap_panel=cap,
        market_panel=market,
        profile="cap10_volume10",
    )
    train_filtered = result.filtered
    daily_audit = result.daily_audit
    summary = result.summary

输入列
------
``--input`` 至少包含 ``datetime``、``vt_symbol``，其他特征和 label 原样保留。

``--cap-data`` 包含：

* ``datetime``、``vt_symbol``；
* ``cap``，或 ``market_cap_2``/``market_cap``。

``--market-data`` 包含：

* ``datetime``、``vt_symbol``、``volume``；
* ``is_st``、``is_suspended``、``has_bar``；
* ``is_one_word_limit_up``、``is_one_word_limit_down``。

默认输出
--------
未指定 ``--output`` 时，在输入文件旁生成：

* ``<input>.<profile>.filtered.parquet``：过滤后的训练样本；
* ``*_filter_daily.parquet``：逐日删除数量审计；
* ``*.json``：汇总、输入路径和输出路径。

本脚本只负责股票池/样本过滤，不执行 label OLS 中性化。正确流水线顺序是：
过滤 -> label 行业市值 OLS -> 归一化 -> CV5 训练。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from vnpy.alpha.dataset.universe_filter import (
    FILTER_PROFILES,
    UniverseFilterProfile,
    UniverseFilterResult,
    filter_universe,
)


def prepare_training_samples(
    frame: pl.DataFrame,
    *,
    cap_panel: pl.DataFrame,
    market_panel: pl.DataFrame,
    profile: str | UniverseFilterProfile,
) -> UniverseFilterResult:
    """Delete T-day invalid rows and daily cap/volume tails before training."""
    return filter_universe(
        frame,
        cap_panel=cap_panel,
        volume_panel=market_panel,
        status_panel=market_panel,
        profile=profile,
        filter_status=True,
        stage="before label neutralization, normalization, CV splitting, and model fit",
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


def _default_output(input_path: Path, profile: str) -> Path:
    return input_path.with_name(f"{input_path.stem}.{profile}.filtered.parquet")


def _write_outputs(
    result: UniverseFilterResult,
    *,
    output: Path,
    audit_output: Path,
    summary_output: Path,
    inputs: dict[str, str],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    result.filtered.write_parquet(output, compression="zstd")
    result.daily_audit.write_parquet(audit_output, compression="zstd")
    payload: dict[str, Any] = {
        **result.summary,
        "inputs": inputs,
        "output": str(output),
        "daily_audit": str(audit_output),
    }
    summary_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练前物理删除状态异常和市值/成交量尾部样本")
    parser.add_argument("--input", required=True, type=Path, help="原始训练样本 parquet/csv")
    parser.add_argument("--cap-data", required=True, type=Path, help="含 datetime/vt_symbol/cap 的数据")
    parser.add_argument(
        "--market-data",
        required=True,
        type=Path,
        help=(
            "含 datetime/vt_symbol/volume/is_st/is_suspended/has_bar/"
            "is_one_word_limit_up/is_one_word_limit_down 的数据"
        ),
    )
    parser.add_argument("--profile", required=True, choices=sorted(FILTER_PROFILES))
    parser.add_argument("--output", type=Path, default=None, help="过滤后训练样本，默认输出到输入文件旁")
    parser.add_argument("--audit-output", type=Path, default=None, help="逐日过滤审计 parquet")
    parser.add_argument("--summary-output", type=Path, default=None, help="过滤汇总 JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or _default_output(args.input, args.profile)
    audit_output = args.audit_output or output.with_name(f"{output.stem}_filter_daily.parquet")
    summary_output = args.summary_output or output.with_suffix(".json")
    result = prepare_training_samples(
        _read_table(args.input),
        cap_panel=_normalize_cap_columns(_read_table(args.cap_data)),
        market_panel=_read_table(args.market_data),
        profile=args.profile,
    )
    _write_outputs(
        result,
        output=output,
        audit_output=audit_output,
        summary_output=summary_output,
        inputs={
            "training_samples": str(args.input),
            "cap_data": str(args.cap_data),
            "market_data": str(args.market_data),
        },
    )
    print(
        f"训练样本过滤完成：{result.summary['rows_before']:,} -> "
        f"{result.summary['rows_after']:,}，输出 {output}"
    )


if __name__ == "__main__":
    main()
