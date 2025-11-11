from .logger import logger
from .dataset import AlphaDataset, Segment, to_datetime
from .dataset.datasets.alpha_158 import Alpha158

# 使 Alpha101/Alpha191 为可选导入，避免缺失模块时报错
try:  # pragma: no cover - optional dataset families
    from .dataset.datasets.alpha_101 import Alpha101
except Exception:  # noqa: BLE001
    Alpha101 = None  # type: ignore

try:  # pragma: no cover - optional dataset families
    from .dataset.datasets.alpha_191 import Alpha191
except Exception:  # noqa: BLE001
    Alpha191 = None  # type: ignore

from .model import AlphaModel
from .strategy import AlphaStrategy, BacktestingEngine
from .lab import AlphaLab


__all__ = [
    "logger",
    "AlphaDataset",
    "Segment",
    "to_datetime",
    "Alpha158",
    "AlphaModel",
    "AlphaStrategy",
    "BacktestingEngine",
    "AlphaLab",
]

# 动态追加可用的可选数据集类
if Alpha101 is not None:
    __all__.append("Alpha101")
if Alpha191 is not None:
    __all__.append("Alpha191")
