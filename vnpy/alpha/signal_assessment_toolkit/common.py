"""信号评估流程共用的数据与统计函数。"""

from __future__ import annotations

import numpy as np
import polars as pl

from vnpy.alpha.dataset import to_datetime

from .config import AssessmentConfig


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def load_signal(config: AssessmentConfig, signal_variant: str) -> pl.DataFrame:
    if signal_variant not in config.signal_paths:
        raise ValueError(
            f"未知 signal_variant={signal_variant!r}，"
            f"可选值: {tuple(config.signal_paths)}"
        )
    start = to_datetime(config.start_date)
    end = to_datetime(config.end_date)
    return (
        pl.read_parquet(config.signal_paths[signal_variant])
        .select("datetime", "vt_symbol", "signal")
        .filter(
            (pl.col("datetime") >= pl.lit(start))
            & (pl.col("datetime") <= pl.lit(end))
            & pl.col("signal").is_finite()
        )
        .unique(subset=["datetime", "vt_symbol"], keep="last")
        .sort(["datetime", "vt_symbol"])
    )


def calculate_legacy_statistics(
    daily_result: pl.DataFrame,
    statistics: dict,
    capital: float,
) -> dict:
    legacy_daily = (
        daily_result.with_columns(balance=pl.col("net_pnl").cum_sum() + capital)
        .with_columns(
            legacy_return=pl.col("balance").pct_change().fill_null(0.0),
            highlevel=pl.col("balance").cum_max(),
        )
        .with_columns(
            legacy_ddpercent=(pl.col("balance") / pl.col("highlevel") - 1.0) * 100.0
        )
    )
    return_std = legacy_daily["legacy_return"].std()
    return {
        "total_return": statistics["total_return"],
        "annual_return": statistics["total_return"] / legacy_daily.height * 240,
        "sharpe_ratio": (
            legacy_daily["legacy_return"].mean() / return_std * (240**0.5)
            if return_std
            else 0.0
        ),
        "max_ddpercent": legacy_daily["legacy_ddpercent"].min(),
    }


def calculate_yearly_metrics(performance_df: pl.DataFrame) -> pl.DataFrame:
    """按自然年统计策略、基准、Alpha 和交易成本指标。"""
    required = {
        "date",
        "net_daily_return",
        "benchmark_daily_return",
        "commission",
        "turnover",
        "trade_count",
    }
    missing = required - set(performance_df.columns)
    if missing:
        raise ValueError(f"performance_df 缺少年度统计字段: {sorted(missing)}")

    def return_profile(values: np.ndarray) -> dict:
        values = np.asarray(values, dtype=float)
        total = float(values.sum())
        mean = float(values.mean()) if values.size else 0.0
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        curve = np.cumsum(values)
        high = np.maximum.accumulate(np.maximum(curve, 0.0))
        max_drawdown = float(np.min(curve - high)) if curve.size else 0.0
        return {
            "total_return": total * 100.0,
            "annual_return": mean * 240.0 * 100.0,
            "sharpe": mean / std * np.sqrt(240.0) if std else 0.0,
            "max_drawdown": max_drawdown * 100.0,
        }

    rows = []
    years = (
        performance_df.select(pl.col("date").dt.year().alias("year"))["year"]
        .unique()
        .sort()
    )
    for year in years.to_list():
        group = performance_df.filter(pl.col("date").dt.year() == year).sort("date")
        strategy_values = group["net_daily_return"].to_numpy()
        benchmark_values = group["benchmark_daily_return"].to_numpy()
        strategy = return_profile(strategy_values)
        benchmark = return_profile(benchmark_values)
        alpha = return_profile(strategy_values - benchmark_values)
        rows.append(
            {
                "year": int(year),
                "trading_days": group.height,
                "total_return": strategy["total_return"],
                "annual_return": strategy["annual_return"],
                "max_drawdown": strategy["max_drawdown"],
                "sharpe_ratio": strategy["sharpe"],
                "benchmark_return": benchmark["total_return"],
                "alpha_return": alpha["total_return"],
                "alpha_sharpe": alpha["sharpe"],
                "alpha_max_drawdown": alpha["max_drawdown"],
                "total_commission": float(group["commission"].sum()),
                "total_turnover": float(group["turnover"].sum()),
                "total_trade_count": int(group["trade_count"].sum()),
            }
        )
    return pl.DataFrame(rows)
