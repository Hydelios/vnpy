"""策略回测、年度统计与已有结果回看。"""

from __future__ import annotations

import json
from unittest.mock import patch

import plotly.graph_objects as go
import polars as pl

from vnpy.alpha import AlphaLab
from vnpy.alpha.dataset import to_datetime
from vnpy.alpha.strategy import BacktestingEngine
from vnpy.alpha.strategy.strategies.equity_demo_strategy_ori import EquityDemoStrategy
from vnpy.alpha.strategy.strategies.fixed_capital_topk_strategy import FixedCapitalTopKStrategy
from vnpy.trader.constant import Interval

from .common import (
    calculate_legacy_statistics,
    calculate_yearly_metrics,
    json_default,
    load_signal,
)
from .config import AssessmentConfig


BACKTEST_MODES = ("fixed_capital_topk", "equity_demo")


def run_backtest(
    config: AssessmentConfig,
    lab: AlphaLab,
    signal_variant: str,
    backtest_mode: str,
    *,
    show_chart: bool = False,
) -> dict:
    """运行一个“信号 × 策略”组合并持久化完整产物。"""
    signal = load_signal(config, signal_variant)
    if signal.is_empty():
        raise RuntimeError(f"区间内信号为空: {signal_variant}")
    start = to_datetime(config.start_date)
    end = to_datetime(config.end_date)

    if backtest_mode == "fixed_capital_topk":
        strategy_class = FixedCapitalTopKStrategy
        strategy_setting = dict(config.fixed_capital_setting)
        backtest_signal = signal
        backtest_symbols = sorted(set(signal["vt_symbol"].to_list()))
        engine_kwargs = {
            "open_rate": config.cost_setting["buy_commission"],
            "close_rate": (
                config.cost_setting["sell_commission"]
                + config.cost_setting["sell_tax"]
            ),
            "min_commission": config.cost_setting["min_commission"],
            "min_volume": config.fixed_capital_setting["min_volume"],
        }
        cost_setting = dict(config.cost_setting)
        cost_source = "显式费用设置，与 new/new_ols notebook 一致"
    elif backtest_mode == "equity_demo":
        strategy_class = EquityDemoStrategy
        strategy_setting = dict(config.equity_demo_setting)
        backtest_symbols = lab.load_component_symbols(
            config.universe_symbol, start, end
        )
        backtest_signal = signal.filter(pl.col("vt_symbol").is_in(backtest_symbols))
        engine_kwargs = {}
        cost_setting = None
        cost_source = "lab 合约费用，与 rqtech/rqtech_ols notebook 一致"
    else:
        raise ValueError(f"未知 backtest_mode={backtest_mode!r}，可选值: {BACKTEST_MODES}")

    print(
        f"开始回测: {signal_variant} × {backtest_mode} | "
        f"信号 {backtest_signal.height:,} 行 | 股票 {len(backtest_symbols):,} 只"
    )
    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=backtest_symbols,
        interval=Interval.DAILY,
        start=start,
        end=end,
        capital=config.capital,
        **engine_kwargs,
    )
    engine.add_strategy(strategy_class, strategy_setting, backtest_signal)
    engine.load_data()
    engine.run_backtesting()
    daily_result = engine.calculate_result()
    if daily_result is None or daily_result.is_empty():
        raise RuntimeError(f"回测结果为空: {signal_variant} × {backtest_mode}")
    statistics = engine.calculate_statistics()
    performance_df = engine.calculate_performance(config.benchmark_symbol)
    alpha_statistics = engine.calculate_alpha_statistics(performance_df=performance_df)
    legacy_statistics = calculate_legacy_statistics(
        daily_result, statistics, config.capital
    )

    output_dir = config.output_root / signal_variant / backtest_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_result.write_parquet(output_dir / "daily_result.parquet", compression="zstd")
    performance_df.write_parquet(output_dir / "performance.parquet", compression="zstd")
    yearly_metrics = calculate_yearly_metrics(performance_df)
    yearly_metrics_path = output_dir / "yearly_metrics.csv"
    yearly_metrics.write_csv(yearly_metrics_path)
    summary = {
        "signal_variant": signal_variant,
        "signal_path": str(config.signal_paths[signal_variant]),
        "backtest_mode": backtest_mode,
        "strategy_class": f"{strategy_class.__module__}.{strategy_class.__name__}",
        "period": [config.start_date, config.end_date],
        "signal_rows": backtest_signal.height,
        "symbol_count": len(backtest_symbols),
        "strategy_setting": strategy_setting,
        "cost_setting": cost_setting,
        "cost_source": cost_source,
        "statistics": statistics,
        "alpha_statistics": alpha_statistics,
        "legacy_statistics": legacy_statistics,
        "yearly_metrics_path": str(yearly_metrics_path),
        "yearly_metrics": yearly_metrics.to_dicts(),
        "output_dir": str(output_dir),
    }
    with patch.object(go.Figure, "show", return_value=None):
        performance_figure = engine.plot_performance(performance_df)
    performance_chart_path = output_dir / "performance.html"
    performance_figure.write_html(
        performance_chart_path, include_plotlyjs=True, full_html=True
    )
    summary["performance_chart"] = str(performance_chart_path)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    if show_chart:
        engine.show_chart()
        performance_figure.show()
    return summary


def backfill_yearly_metrics(config: AssessmentConfig) -> pl.DataFrame:
    """从已有 performance.parquet 重建逐年统计，不重新回测。"""
    combined = []
    for performance_path in sorted(config.output_root.glob("*/*/performance.parquet")):
        output_dir = performance_path.parent
        signal_variant = output_dir.parent.name
        backtest_mode = output_dir.name
        yearly_metrics = calculate_yearly_metrics(pl.read_parquet(performance_path))
        yearly_metrics_path = output_dir / "yearly_metrics.csv"
        yearly_metrics.write_csv(yearly_metrics_path)
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["yearly_metrics_path"] = str(yearly_metrics_path)
            summary["yearly_metrics"] = yearly_metrics.to_dicts()
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
                encoding="utf-8",
            )
        combined.append(
            yearly_metrics.with_columns(
                pl.lit(signal_variant).alias("signal_variant"),
                pl.lit(backtest_mode).alias("backtest_mode"),
            ).select(
                "signal_variant",
                "backtest_mode",
                pl.all().exclude("signal_variant", "backtest_mode"),
            )
        )
    if not combined:
        raise RuntimeError(f"没有可生成年度统计的绩效文件: {config.output_root}")
    yearly_all = pl.concat(combined).sort("signal_variant", "backtest_mode", "year")
    yearly_all.write_csv(config.output_root / "yearly_metrics_all.csv")
    return yearly_all


def review_saved_result(
    config: AssessmentConfig,
    lab: AlphaLab,
    signal_variant: str,
    backtest_mode: str,
) -> tuple[dict, pl.DataFrame, go.Figure]:
    """读取并绘制已有结果，不触发回测。"""
    output_dir = config.output_root / signal_variant / backtest_mode
    summary_path = output_dir / "summary.json"
    performance_path = output_dir / "performance.parquet"
    if not summary_path.exists() or not performance_path.exists():
        raise FileNotFoundError(f"尚未生成该组合的结果: {output_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    performance_df = pl.read_parquet(performance_path)
    engine = BacktestingEngine(lab)
    figure = engine.plot_performance(performance_df)
    return summary, performance_df, figure
