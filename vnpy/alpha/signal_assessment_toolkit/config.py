"""统一信号评估流程的配置对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from vnpy.alpha.signal import SimpleBacktestConfig


@dataclass(frozen=True)
class AssessmentConfig:
    """一次信号评估任务所需的全部显式配置。"""

    signal_paths: Mapping[str, Path]
    lab_path: Path
    output_root: Path
    barra_path: Path
    start_date: str
    end_date: str
    benchmark_symbol: str = "000852.SSE"
    universe_symbol: str = "all_stocks_3800_ex_real_estate_ex_ban"
    capital: float = 100_000_000.0
    fixed_capital_setting: Mapping[str, Any] = field(default_factory=dict)
    equity_demo_setting: Mapping[str, Any] = field(default_factory=dict)
    cost_setting: Mapping[str, float] = field(default_factory=dict)
    horizons: Sequence[int] = (1, 3, 5, 10, 20)
    quantiles: int = 9
    plot_forward_return: str = "FWD_RET_OO_1"
    simple_backtest_config: SimpleBacktestConfig | None = None

    def validate(self) -> None:
        """尽早报告路径或配置错误。"""
        missing = [str(path) for path in self.signal_paths.values() if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"以下信号文件不存在: {missing}")
        if not self.lab_path.exists():
            raise FileNotFoundError(f"AlphaLab 数据目录不存在: {self.lab_path}")
        if not self.barra_path.exists():
            raise FileNotFoundError(f"Barra 数据文件不存在: {self.barra_path}")
        if self.simple_backtest_config is None:
            raise ValueError("simple_backtest_config 不能为空")
        self.output_root.mkdir(parents=True, exist_ok=True)
