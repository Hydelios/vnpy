"""可复用的信号分析、策略回测与 HTML 报告工具。"""

from .backtest_analysis import (
    backfill_yearly_metrics,
    calculate_yearly_metrics,
    review_saved_result,
    run_backtest,
)
from .config import AssessmentConfig
from .html_report import HtmlReportBuilder
from .signal_analysis import run_signal_analysis

__all__ = [
    "AssessmentConfig",
    "HtmlReportBuilder",
    "backfill_yearly_metrics",
    "calculate_yearly_metrics",
    "review_saved_result",
    "run_backtest",
    "run_signal_analysis",
]
