from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RQOptimizerConfig:
    """Configuration for Top-K reweighting in an isolated RQOptimizer process."""

    conda_env: str = "rqoptimizer"
    benchmark: str = "000852.XSHG"
    start_date: str | None = None
    end_date: str | None = None
    top_k: int = 300
    rebalance_interval: int = 5
    min_weight: float = 0.001
    max_weight: float = 0.005
    score_floor: float = 0.1
    score_ceiling: float = 1.1
    size_active_limit: float = 0.2
    other_style_active_limit: float = 0.3
    turnover_soft_limit: float = 0.3
    turnover_hard_limit: float = 0.5
    risk_model: str = "v1"
    fallback_equal_weight: bool = True

    def validate(self) -> None:
        for name, value in (
            ("top_k", self.top_k),
            ("rebalance_interval", self.rebalance_interval),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} 必须为正整数")
        for name, value in (
            ("min_weight", self.min_weight),
            ("max_weight", self.max_weight),
            ("score_floor", self.score_floor),
            ("score_ceiling", self.score_ceiling),
            ("size_active_limit", self.size_active_limit),
            ("other_style_active_limit", self.other_style_active_limit),
            ("turnover_soft_limit", self.turnover_soft_limit),
            ("turnover_hard_limit", self.turnover_hard_limit),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} 必须为有限数")
        if not 0 <= self.min_weight <= self.max_weight <= 1:
            raise ValueError("权重上下限必须满足 0 <= min_weight <= max_weight <= 1")
        if self.top_k * self.min_weight > 1 or self.top_k * self.max_weight < 1:
            raise ValueError("top_k 与权重上下限不能组成权重和为1的可行组合")
        if not 0 <= self.score_floor < self.score_ceiling:
            raise ValueError("score_floor/score_ceiling 必须满足 0 <= floor < ceiling")
        if self.size_active_limit <= 0 or self.other_style_active_limit <= 0:
            raise ValueError("风格偏离上限必须大于0")
        if not 0 <= self.turnover_soft_limit <= self.turnover_hard_limit:
            raise ValueError("换手约束必须满足 0 <= soft <= hard")
        if self.risk_model not in {"v1", "v2", "v2trd", "v2_bjse", "v2trd_bjse"}:
            raise ValueError(f"不支持的 risk_model: {self.risk_model}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
