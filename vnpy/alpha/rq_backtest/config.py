from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Self


@dataclass(frozen=True)
class RQBacktestConfig:
    """RQAlpha adapter for :class:`FixedCapitalTopKStrategy`.

    The adapter always executes the previous signal at the next open.  Its
    portfolio parameters intentionally mirror the vn.py strategy; the old
    ``n_drop`` and VWAP modes are not part of this backtest path.
    """

    start_date: str | None = None
    end_date: str | None = None
    capital: float = 100_000_000
    benchmark: str = "000852.SSE"
    frequency: str = "1d"
    top_k: int = 300
    cash_ratio: float = 0.99
    rebalance_interval: int = 5
    exit_rank_buffer: int = 0
    min_volume: int = 100
    open_rate: float = 0.0005
    close_rate: float = 0.0015
    min_commission: float = 5.0
    inactive_limit: bool = True
    bundle_path: str = "/home/hyd/.rqalpha-plus/bundle"
    python_executable: str | None = None
    log_level: str = "error"

    def validate(self) -> None:
        positive_ints = {
            "top_k": self.top_k,
            "rebalance_interval": self.rebalance_interval,
            "min_volume": self.min_volume,
        }
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in positive_ints.values()
        ):
            raise ValueError("top_k、rebalance_interval 和 min_volume 必须为正整数")
        if (
            not isinstance(self.exit_rank_buffer, int)
            or isinstance(self.exit_rank_buffer, bool)
            or self.exit_rank_buffer < 0
        ):
            raise ValueError("exit_rank_buffer 必须为非负整数")
        if not math.isfinite(self.capital) or self.capital <= 0:
            raise ValueError("capital 必须大于0")
        if not math.isfinite(self.cash_ratio) or not 0 < self.cash_ratio <= 1:
            raise ValueError("cash_ratio 必须在 (0, 1] 内")
        if self.open_rate < 0 or self.close_rate < self.open_rate:
            raise ValueError("手续费要求 0 <= open_rate <= close_rate")
        if self.min_commission < 0:
            raise ValueError("min_commission 不能为负数")
        if self.frequency != "1d":
            raise ValueError("FixedCapitalTopKStrategy 米筐适配器只支持 frequency='1d'")
        if not Path(self.bundle_path).exists():
            raise FileNotFoundError(f"RQAlpha bundle 不存在: {self.bundle_path}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_settings(
        cls,
        *,
        portfolio: Mapping[str, Any],
        cost: Mapping[str, Any],
        execution: Mapping[str, Any],
        **overrides: Any,
    ) -> Self:
        """Build a config from the shared FixedCapitalTopK settings layers."""

        legacy_keys = {
            "rebalance_mode",
            "n_drop",
            "min_days",
            "hold_thresh",
            "execution_mode",
            "vwap_start",
            "vwap_end",
            "h5_minbar_path",
            "slippage_model",
            "slippage",
            "volume_limit",
            "volume_percent",
        }
        supplied_legacy = sorted(legacy_keys & (set(portfolio) | set(execution)))
        if supplied_legacy:
            raise ValueError(
                "米筐回测已固定为 FixedCapitalTopKStrategy，"
                "不再接受旧版或改变策略语义的参数: "
                f"{supplied_legacy}"
            )

        required_cost = {
            "buy_commission",
            "sell_commission",
            "sell_tax",
            "min_commission",
        }
        missing = sorted(required_cost - set(cost))
        if missing:
            raise ValueError(f"cost 缺少参数: {missing}")
        buy_commission = float(cost["buy_commission"])
        sell_commission = float(cost["sell_commission"])
        if buy_commission != sell_commission:
            raise ValueError("RQAlpha 当前仅支持买卖使用相同佣金率")

        cost_config = {
            "open_rate": buy_commission,
            "close_rate": sell_commission + float(cost["sell_tax"]),
            "min_commission": float(cost["min_commission"]),
        }
        return cls(**(dict(portfolio) | cost_config | dict(execution) | overrides))

    @property
    def stock_commission_multiplier(self) -> float:
        """Map desired buy commission to RQAlpha's default 8 bps rate."""

        return self.open_rate / 0.0008

    @property
    def tax_multiplier(self) -> float:
        """Map the sell-only cost above commission to the default 10 bps tax."""

        return (self.close_rate - self.open_rate) / 0.001
