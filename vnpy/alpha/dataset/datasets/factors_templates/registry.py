from __future__ import annotations
from typing import Iterable, Dict, Type
import polars as pl
from .factors_template import FactorTemplate, ExprLike

class FactorRegistry:
    """模板注册器：集中管理模板类与批量注册到 AlphaDataset。"""

    def __init__(self) -> None:
        self._templates: list[FactorTemplate] = []

    def register(self, *templates: FactorTemplate) -> "FactorRegistry":
        self._templates.extend(templates)
        return self

    def names(self) -> list[str]:
        return [t.factor_name for t in self._templates]

    def bind_to_dataset(self, dataset, runtime_overrides: dict | None = None) -> None:
        """将所有模板展开并 add_feature 到给定的 AlphaDataset 实例中。"""
        for t in self._templates:
            expr_map: Dict[str, ExprLike] = t.expand(runtime_overrides)
            # 这里不做列依赖检查，交由执行层（或自行在 dataset 处校验）
            for name, expr in expr_map.items():
                dataset.add_feature(name, expression=expr)