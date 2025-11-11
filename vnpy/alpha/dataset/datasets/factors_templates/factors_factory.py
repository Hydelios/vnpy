from __future__ import annotations
from .registry import FactorRegistry

# 导入各类模板
from .examples import RangePos, Alpha101_001, Alpha101_002, TA_RSI
from .my_factors import Alpha001
# 实例化注册器并注册所有模板类
registry = (
    FactorRegistry()
    .register(
        # 仅注册“因子模板”（继承自 FactorTemplate 的类实例），不要注册 AlphaDataset 类
        RangePos(),
        Alpha101_001(),
        Alpha101_002(),
        TA_RSI(),
        Alpha001(),
    )
)

def bind_all(dataset, runtime_overrides: dict | None = None) -> None:
    """
    快捷绑定所有注册模板到指定 AlphaDataset
    ------------------------------------------------
    等价于:
        from vnpy.alpha.factor_templates.factors_factory import registry
        registry.bind_to_dataset(dataset)
    """
    registry.bind_to_dataset(dataset, runtime_overrides=runtime_overrides)
