from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rqdatac
import requests
from rqoptimizer import (
    CovModel,
    MaxIndicator,
    OptimizationFailed,
    RiskModel,
    StyleConstraint,
    TurnoverLimit,
    portfolio_optimize,
)
from rqoptimizer.benchmark import IndexBenchmark
from rqoptimizer.helper import OptimizeHelper


STYLE_FACTORS = ["size", "liquidity", "beta"]
_ORIGINAL_SESSION_REQUEST = requests.sessions.Session.request


def _session_request_with_timeout(self: Any, method: str, url: str, **kwargs: Any) -> Any:
    """Bound RiceQuant license HTTP calls that otherwise have no timeout."""

    kwargs.setdefault("timeout", (10, 60))
    return _ORIGINAL_SESSION_REQUEST(self, method, url, **kwargs)


requests.sessions.Session.request = _session_request_with_timeout


def _retry_network_call(function: Any, label: str, attempts: int = 6) -> Any:
    """Retry transient RiceQuant HTTP/data connection failures."""

    for attempt in range(1, attempts + 1):
        try:
            return function()
        except (requests.RequestException, TimeoutError, OSError, ConnectionError) as exc:
            if attempt == attempts:
                raise
            delay = min(2 ** (attempt - 1), 20)
            print(
                f"{label} 网络失败，第{attempt}/{attempts}次，{delay}s后重试: "
                f"{type(exc).__name__}",
                flush=True,
            )
            time.sleep(delay)


def _adjust_score(signal: pd.Series, floor: float, ceiling: float) -> pd.Series:
    signal = signal.astype(float)
    span = float(signal.max() - signal.min())
    if not np.isfinite(span) or span <= 0:
        return pd.Series((floor + ceiling) / 2, index=signal.index, dtype=float)
    return (signal - signal.min()) / span * (ceiling - floor) + floor


def _risk_model(value: str) -> RiskModel:
    return RiskModel(value)


def _style_constraints(config: dict[str, Any], *, relaxed: bool = False) -> list[Any]:
    size_limit = float(config["size_active_limit"])
    other_limit = float(config["other_style_active_limit"])
    if relaxed:
        size_limit = max(size_limit, 0.5)
        other_limit = max(other_limit, 0.5)
    return [
        StyleConstraint(
            "size",
            lower_limit=-size_limit,
            upper_limit=size_limit,
            relative=True,
            hard=True,
        ),
        StyleConstraint(
            ["liquidity", "beta"],
            lower_limit=-other_limit,
            upper_limit=other_limit,
            relative=True,
            hard=True,
        ),
    ]


def _optimize_day(
    day: pd.DataFrame,
    execution_date: object,
    previous_weights: pd.Series | None,
    config: dict[str, Any],
) -> tuple[pd.Series, str, list[str]]:
    signal = day.set_index("order_book_id")["signal"]
    score = _adjust_score(signal, float(config["score_floor"]), float(config["score_ceiling"]))
    bounds = {"*": (float(config["min_weight"]), float(config["max_weight"]))}
    attempts: list[tuple[str, list[Any]]] = []
    strict_constraints = _style_constraints(config)
    if previous_weights is not None and not previous_weights.empty:
        strict_constraints += [
            TurnoverLimit(
                previous_weights,
                float(config["turnover_soft_limit"]),
                hard=False,
            ),
            TurnoverLimit(
                previous_weights,
                float(config["turnover_hard_limit"]),
                hard=True,
            ),
        ]
    attempts.append(("strict", strict_constraints))
    attempts.append(("style_only", _style_constraints(config)))
    attempts.append(("relaxed_style", _style_constraints(config, relaxed=True)))

    messages: list[str] = []
    for status, constraints in attempts:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                weights = _retry_network_call(
                    lambda: portfolio_optimize(
                        signal.index.tolist(),
                        execution_date,
                        objective=MaxIndicator(score),
                        bnds=bounds,
                        cons=constraints,
                        benchmark=str(config["benchmark"]),
                        risk_model=_risk_model(str(config["risk_model"])),
                    ),
                    f"{execution_date} portfolio_optimize/{status}",
                )
            messages.extend(str(item.message) for item in caught)
            weights = pd.Series(weights, dtype=float).reindex(signal.index)
            if weights.isna().any() or not np.isclose(weights.sum(), 1.0, atol=1e-5):
                raise ValueError("优化权重包含空值或权重和不为1")
            return weights, status, messages
        except (OptimizationFailed, ValueError, RuntimeError) as exc:
            messages.append(f"{status}: {type(exc).__name__}: {exc}")

    if not bool(config["fallback_equal_weight"]):
        raise RuntimeError("所有优化尝试均失败: " + " | ".join(messages[-6:]))
    equal = pd.Series(1.0 / len(signal), index=signal.index, dtype=float)
    return equal, "fallback_equal_weight", messages


def _exposure_rows(
    day: pd.DataFrame,
    weights: pd.Series,
    signal_date: object,
    execution_date: object,
    benchmark: str,
    risk_model: str,
) -> list[dict[str, Any]]:
    order_book_ids = day["order_book_id"].tolist()
    helper = OptimizeHelper(
        order_book_ids,
        pd.Timestamp(execution_date).date(),
        IndexBenchmark(benchmark),
        CovModel.FACTOR_MODEL_DAILY,
        _risk_model(risk_model),
    )
    exposure = helper.get_factor_exposure().reindex(order_book_ids)
    equal = pd.Series(1.0 / len(order_book_ids), index=order_book_ids)
    benchmark_style = helper.get_benchmark_style()
    rows: list[dict[str, Any]] = []
    for factor in STYLE_FACTORS:
        equal_value = float(exposure[factor].mul(equal).sum())
        optimized_value = float(exposure[factor].mul(weights).sum())
        benchmark_value = float(benchmark_style[factor])
        rows.append(
            {
                "datetime": signal_date,
                "execution_date": execution_date,
                "factor": factor,
                "equal_topk": equal_value,
                "optimized": optimized_value,
                "benchmark": benchmark_value,
                "equal_active": equal_value - benchmark_value,
                "optimized_active": optimized_value - benchmark_value,
            }
        )
    return rows


def run(config_path: Path, input_path: Path, output_dir: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(input_path, parse_dates=["datetime"])
    _retry_network_call(rqdatac.init, "rqdatac.init")

    checkpoint_dir = output_dir / ".checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_meta_path = checkpoint_dir / "meta.json"
    input_digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    checkpoint_meta = {"input_sha256": input_digest, "config_sha256": config_digest}
    existing_meta = (
        json.loads(checkpoint_meta_path.read_text(encoding="utf-8"))
        if checkpoint_meta_path.exists()
        else None
    )
    checkpoint_files = {
        "optimized": checkpoint_dir / "optimized_signal.csv",
        "diagnostics": checkpoint_dir / "diagnostics.csv",
        "exposures": checkpoint_dir / "exposure_comparison.csv",
    }
    if existing_meta != checkpoint_meta:
        for path in checkpoint_files.values():
            path.unlink(missing_ok=True)
        checkpoint_meta_path.write_text(
            json.dumps(checkpoint_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    completed_dates: set[pd.Timestamp] = set()
    previous_weights: pd.Series | None = None
    if checkpoint_files["optimized"].exists():
        previous_output = pd.read_csv(checkpoint_files["optimized"], parse_dates=["datetime"])
        completed_dates = set(previous_output["datetime"].drop_duplicates())
        if completed_dates:
            last_date = max(completed_dates)
            last = previous_output[previous_output["datetime"] == last_date]
            previous_weights = last.set_index("order_book_id")["weight"].astype(float)
            print(f"从 checkpoint 恢复，已完成 {len(completed_dates)} 个调仓日", flush=True)

    grouped = list(frame.groupby("datetime", sort=True))
    for index, (signal_date, day) in enumerate(grouped, start=1):
        if signal_date in completed_dates:
            continue
        execution_date = _retry_network_call(
            lambda: rqdatac.get_next_trading_date(signal_date.date()),
            f"{signal_date.date()} get_next_trading_date",
        )
        weights, status, messages = _optimize_day(day, execution_date, previous_weights, config)
        output = day.copy()
        output["execution_date"] = pd.Timestamp(execution_date)
        output["weight"] = output["order_book_id"].map(weights)
        output["status"] = status

        if previous_weights is None:
            theoretical_turnover = 0.5
        else:
            union = weights.index.union(previous_weights.index)
            theoretical_turnover = float(
                (
                    weights.reindex(union, fill_value=0.0)
                    - previous_weights.reindex(union, fill_value=0.0)
                ).abs().sum()
                / 2
            )
        warning_text = " | ".join(dict.fromkeys(messages))
        diagnostic = {
                "datetime": signal_date,
                "execution_date": execution_date,
                "status": status,
                "candidate_count": len(day),
                "weight_sum": float(weights.sum()),
                "min_weight": float(weights.min()),
                "max_weight": float(weights.max()),
                "effective_names": float(1.0 / weights.pow(2).sum()),
                "signal_weight_spearman": float(day.set_index("order_book_id")["signal"].corr(weights, method="spearman")),
                "theoretical_one_way_turnover": theoretical_turnover,
                "warnings": warning_text,
            }
        exposures = _retry_network_call(
            lambda: _exposure_rows(
                day,
                weights,
                signal_date.date(),
                execution_date,
                str(config["benchmark"]),
                str(config["risk_model"]),
            ),
            f"{execution_date} exposure_diagnostics",
        )
        output.to_csv(
            checkpoint_files["optimized"],
            mode="a",
            header=not checkpoint_files["optimized"].exists(),
            index=False,
        )
        pd.DataFrame([diagnostic]).to_csv(
            checkpoint_files["diagnostics"],
            mode="a",
            header=not checkpoint_files["diagnostics"].exists(),
            index=False,
        )
        pd.DataFrame(exposures).to_csv(
            checkpoint_files["exposures"],
            mode="a",
            header=not checkpoint_files["exposures"].exists(),
            index=False,
        )
        previous_weights = weights
        print(f"[{index}/{len(grouped)}] {signal_date.date()} -> {execution_date} {status}", flush=True)

    optimized = pd.read_csv(checkpoint_files["optimized"], parse_dates=["datetime", "execution_date"])
    diagnostics = pd.read_csv(checkpoint_files["diagnostics"], parse_dates=["datetime", "execution_date"])
    exposures = pd.read_csv(checkpoint_files["exposures"], parse_dates=["datetime", "execution_date"])
    final_names = {
        "optimized": "optimized_signal.csv",
        "diagnostics": "diagnostics.csv",
        "exposures": "exposure_comparison.csv",
    }
    for name, source in checkpoint_files.items():
        shutil.copyfile(source, output_dir / final_names[name])
    status_counts = diagnostics["status"].value_counts().to_dict()
    summary = {
        "start_date": diagnostics["datetime"].min().date().isoformat(),
        "end_date": diagnostics["datetime"].max().date().isoformat(),
        "rebalance_count": int(len(diagnostics)),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "fallback_count": int((diagnostics["status"] == "fallback_equal_weight").sum()),
        "avg_effective_names": float(diagnostics["effective_names"].mean()),
        "avg_theoretical_one_way_turnover": float(diagnostics["theoretical_one_way_turnover"].iloc[1:].mean()),
        "avg_signal_weight_spearman": float(diagnostics["signal_weight_spearman"].mean()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RQOptimizer Top-K reweighting")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.input, args.output_dir)


if __name__ == "__main__":
    main()
