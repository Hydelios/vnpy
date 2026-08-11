from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RQBacktestConfig
from .signal import prepare_ranked_signals, to_rq_order_book_id


@dataclass
class RQBacktestResult:
    output_dir: Path
    target_weights: pd.DataFrame
    rebalance_diagnostics: pd.DataFrame
    summary: dict[str, Any]
    annual_metrics: pd.DataFrame
    portfolio: pd.DataFrame
    trades: pd.DataFrame
    stdout: str


def _find_rqalpha_python(explicit: str | None = None) -> str:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)

    rqalpha_cli = shutil.which("rqalpha")
    if rqalpha_cli:
        try:
            first_line = Path(rqalpha_cli).read_text(encoding="utf-8").splitlines()[0]
            if first_line.startswith("#!"):
                candidates.append(first_line[2:].strip())
        except (OSError, UnicodeDecodeError, IndexError):
            pass

    candidates.extend(["/home/hyd/miniforge3/bin/python", shutil.which("python") or ""])
    checked: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in checked or not Path(candidate).exists():
            continue
        checked.add(candidate)
        probe = subprocess.run(
            [candidate, "-c", "import rqalpha"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            return candidate
    raise RuntimeError("找不到安装了 rqalpha 的 Python；请设置 python_executable")


def _read_csv_if_exists(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _probe_bundle_end_date(
    python_executable: str,
    bundle_path: str,
    order_book_id: str,
) -> pd.Timestamp | None:
    script = """
import sys
from pathlib import Path
import h5py

bundle = Path(sys.argv[1])
order_book_id = sys.argv[2]
for filename in ("indexes.h5", "stocks.h5", "funds.h5"):
    path = bundle / filename
    if not path.exists():
        continue
    with h5py.File(path, "r") as store:
        if order_book_id in store and len(store[order_book_id]):
            value = int(store[order_book_id][-1]["datetime"])
            print(str(value)[:8])
            raise SystemExit(0)
raise SystemExit(2)
"""
    completed = subprocess.run(
        [python_executable, "-c", script, bundle_path, order_book_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    return pd.Timestamp(completed.stdout.strip())


def run_rqalpha_signal_backtest(
    signal: Any,
    output_dir: str | Path,
    config: RQBacktestConfig | None = None,
) -> RQBacktestResult:
    """Run FixedCapitalTopKStrategy semantics in an isolated RQAlpha process."""

    config = config or RQBacktestConfig()
    config.validate()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rankings = prepare_ranked_signals(
        signal,
        start_date=config.start_date,
        end_date=config.end_date,
    )
    ranking_path = output_dir / "ranked_signals.parquet"
    rankings.to_parquet(ranking_path, index=False)

    start_date = config.start_date or rankings["datetime"].min().strftime("%Y-%m-%d")
    end_date = config.end_date or rankings["datetime"].max().strftime("%Y-%m-%d")
    benchmark = to_rq_order_book_id(config.benchmark)
    worker_config = config.to_dict() | {
        "start_date": start_date,
        "end_date": end_date,
        "benchmark": benchmark,
        "stock_commission_multiplier": config.stock_commission_multiplier,
        "tax_multiplier": config.tax_multiplier,
        "output_dir": str(output_dir),
    }
    config_path = output_dir / "rqalpha_config.json"
    config_path.write_text(
        json.dumps(worker_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    python_executable = _find_rqalpha_python(config.python_executable)
    bundle_end = _probe_bundle_end_date(
        python_executable,
        config.bundle_path,
        benchmark,
    )
    if bundle_end is not None and pd.Timestamp(end_date) > bundle_end:
        raise RuntimeError(
            "RQAlpha Bundle 不覆盖完整回测区间："
            f"benchmark={benchmark}, bundle_end={bundle_end.date()}, requested_end={end_date}。"
            f"请先执行 rqalpha update-bundle -d {config.bundle_path}，或显式缩短 end_date。"
        )
    worker_path = Path(__file__).with_name("worker.py")
    command = [
        python_executable,
        str(worker_path),
        "--config",
        str(config_path),
        "--rankings",
        str(ranking_path),
    ]
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
    completed = subprocess.run(
        command,
        cwd=str(output_dir),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout + completed.stderr
    (output_dir / "rqalpha_stdout.log").write_text(stdout, encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join(stdout.splitlines()[-50:])
        raise RuntimeError(f"RQAlpha 回测失败，exit={completed.returncode}:\n{tail}")

    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        log_path = output_dir / "rqalpha_stdout.log"
        raise RuntimeError(f"RQAlpha 未生成 summary.json，日志: {log_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    targets = _read_csv_if_exists(
        output_dir / "target_weights.csv",
        parse_dates=["datetime", "signal_datetime"],
    )
    diagnostics = _read_csv_if_exists(
        output_dir / "rebalance_diagnostics.csv",
        parse_dates=["datetime", "signal_datetime"],
    )
    return RQBacktestResult(
        output_dir=output_dir,
        target_weights=targets,
        rebalance_diagnostics=diagnostics,
        summary=summary,
        annual_metrics=_read_csv_if_exists(output_dir / "annual_metrics.csv"),
        portfolio=_read_csv_if_exists(output_dir / "portfolio.csv", parse_dates=["date"]),
        trades=_read_csv_if_exists(output_dir / "trades.csv"),
        stdout=stdout,
    )
