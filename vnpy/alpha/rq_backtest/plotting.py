from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from .runner import RQBacktestResult


def create_rqalpha_performance_chart(
    result: RQBacktestResult,
    *,
    title: str = "RQAlpha Performance",
    output_path: str | Path | None = None,
) -> Figure:
    """Create a notebook-friendly NAV, excess NAV, and drawdown chart."""

    portfolio = result.portfolio.copy()
    if portfolio.empty:
        raise ValueError("RQAlpha portfolio 为空，无法绘制绩效图")
    if "date" not in portfolio.columns:
        raise ValueError("RQAlpha portfolio 缺少 date 列")

    portfolio["date"] = pd.to_datetime(portfolio["date"], errors="coerce")
    portfolio = portfolio.dropna(subset=["date"]).sort_values("date")
    if portfolio.empty:
        raise ValueError("RQAlpha portfolio 不包含有效日期")

    if "unit_net_value" in portfolio.columns:
        strategy_nav = pd.to_numeric(
            portfolio["unit_net_value"], errors="coerce"
        )
    elif "total_value" in portfolio.columns:
        total_value = pd.to_numeric(portfolio["total_value"], errors="coerce")
        strategy_nav = total_value / total_value.dropna().iloc[0]
    else:
        raise ValueError("RQAlpha portfolio 缺少净值列")

    benchmark_nav = None
    if "benchmark_unit_net_value" in portfolio.columns:
        benchmark_nav = pd.to_numeric(
            portfolio["benchmark_unit_net_value"], errors="coerce"
        )
    relative_nav = strategy_nav / benchmark_nav if benchmark_nav is not None else None
    drawdown = strategy_nav / strategy_nav.cummax() - 1.0

    figure = Figure(
        figsize=(12, 10),
        layout="constrained",
    )
    FigureCanvasAgg(figure)
    axes = figure.subplots(
        3,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2, 1.2]},
    )
    dates = portfolio["date"]

    axes[0].plot(dates, strategy_nav, label="Strategy", linewidth=1.5)
    if benchmark_nav is not None:
        axes[0].plot(dates, benchmark_nav, label="Benchmark", linewidth=1.2)
    axes[0].set_ylabel("NAV")
    axes[0].set_title(title)
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    if relative_nav is not None:
        axes[1].plot(dates, relative_nav, color="tab:green", linewidth=1.3)
        axes[1].axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
        axes[1].set_ylabel("Strategy / Benchmark")
    else:
        axes[1].text(
            0.5,
            0.5,
            "Benchmark NAV unavailable",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_ylabel("Relative NAV")
    axes[1].grid(alpha=0.25)

    axes[2].fill_between(dates, drawdown, 0.0, color="tab:red", alpha=0.35)
    axes[2].plot(dates, drawdown, color="tab:red", linewidth=1.0)
    axes[2].set_ylabel("Drawdown")
    axes[2].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[2].grid(alpha=0.25)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160, bbox_inches="tight")
    return figure
