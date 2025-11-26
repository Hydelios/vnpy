"""
Alphalens 分析子包：提供清洗、绘图与汇总的职责分离实现。

对外主要导出以下稳定 API（保持与旧版兼容）：
 - run_factor_analysis
 - summarize_factor / summarize_factors
 - compute_summary_metrics
 - PlotConfig / PLOT_DEFAULTS
"""

from .runner import run_factor_analysis
from .summary import summarize_factor, summarize_factors
from .metrics import compute_summary_metrics, generate_multi_factor_summary
from .tears import PlotConfig, PLOT_DEFAULTS

__all__ = [
    "run_factor_analysis",
    "summarize_factor",
    "summarize_factors",
    "compute_summary_metrics",
    "generate_multi_factor_summary",
    "PlotConfig",
    "PLOT_DEFAULTS",
]
