from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from vnpy.alpha.rq_backtest.signal import normalize_signal, to_rq_order_book_id

from .config import RQOptimizerConfig


@dataclass
class RQOptimizerResult:
    output_dir: Path
    optimized_signal: pd.DataFrame
    diagnostics: pd.DataFrame
    exposure_comparison: pd.DataFrame
    summary: dict[str, Any]
    stdout: str


def _sample_rebalance_dates(frame: pd.DataFrame, interval: int) -> pd.DataFrame:
    dates = pd.Index(frame["datetime"].drop_duplicates().sort_values())
    selected = set(dates[::interval])
    return frame[frame["datetime"].isin(selected)].copy()


def prepare_optimizer_input(
    signal: Any,
    config: RQOptimizerConfig,
) -> pd.DataFrame:
    """Normalize a signal and retain the strategy's deterministic rebalance Top-K."""

    frame = normalize_signal(signal)
    if config.start_date is not None:
        frame = frame[frame["datetime"] >= pd.Timestamp(config.start_date)]
    if config.end_date is not None:
        frame = frame[frame["datetime"] <= pd.Timestamp(config.end_date)]
    if frame.empty:
        raise ValueError("日期过滤后的优化器输入为空")
    sampled = _sample_rebalance_dates(frame, config.rebalance_interval)
    sampled["rank"] = sampled.groupby("datetime", sort=True).cumcount() + 1
    sampled = sampled[sampled["rank"] <= config.top_k]
    counts = sampled.groupby("datetime")["order_book_id"].nunique()
    if (counts < config.top_k).any():
        bad = counts[counts < config.top_k]
        raise ValueError(f"以下调仓日候选股票不足 Top{config.top_k}: {bad.to_dict()}")
    return sampled[["datetime", "order_book_id", "signal", "rank"]].reset_index(drop=True)


def optimize_signal_with_rqoptimizer(
    signal: Any,
    output_dir: str | Path,
    config: RQOptimizerConfig | None = None,
) -> RQOptimizerResult:
    """Generate Top-K target weights through the isolated ``rqoptimizer`` env."""

    config = config or RQOptimizerConfig()
    config.validate()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer_input = prepare_optimizer_input(signal, config)
    input_path = output_dir / "optimizer_input.csv"
    optimizer_input.to_csv(input_path, index=False)
    config_path = output_dir / "optimizer_config.json"
    config_path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    worker_path = Path(__file__).with_name("worker.py")
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        config.conda_env,
        "python",
        str(worker_path),
        "--config",
        str(config_path),
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    stdout = completed.stdout + completed.stderr
    (output_dir / "rqoptimizer_stdout.log").write_text(stdout, encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join(stdout.splitlines()[-80:])
        raise RuntimeError(f"RQOptimizer 执行失败，exit={completed.returncode}:\n{tail}")

    optimized_path = output_dir / "optimized_signal.csv"
    diagnostics_path = output_dir / "diagnostics.csv"
    summary_path = output_dir / "summary.json"
    if not optimized_path.exists() or not diagnostics_path.exists() or not summary_path.exists():
        raise RuntimeError(f"RQOptimizer 输出不完整，请检查 {output_dir / 'rqoptimizer_stdout.log'}")

    optimized = pd.read_csv(optimized_path, parse_dates=["datetime", "execution_date"])
    optimized["vt_symbol"] = optimized["order_book_id"].map(_to_vt_symbol)
    diagnostics = pd.read_csv(diagnostics_path, parse_dates=["datetime", "execution_date"])
    exposure_path = output_dir / "exposure_comparison.csv"
    exposure = (
        pd.read_csv(exposure_path, parse_dates=["datetime", "execution_date"])
        if exposure_path.exists()
        else pd.DataFrame()
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return RQOptimizerResult(
        output_dir=output_dir,
        optimized_signal=optimized,
        diagnostics=diagnostics,
        exposure_comparison=exposure,
        summary=summary,
        stdout=stdout,
    )


def _to_vt_symbol(order_book_id: str) -> str:
    value = str(order_book_id)
    if value.endswith(".XSHG"):
        return value[:-5] + ".SSE"
    if value.endswith(".XSHE"):
        return value[:-5] + ".SZSE"
    return value


__all__ = [
    "RQOptimizerResult",
    "optimize_signal_with_rqoptimizer",
    "prepare_optimizer_input",
    "to_rq_order_book_id",
]
