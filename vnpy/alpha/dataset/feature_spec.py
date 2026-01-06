from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence
from collections import OrderedDict
import itertools
import json

@dataclass(frozen=True)
class FactorDef:
    """
    特征定义元数据对象（所有列都是 feature）
    """
    name: str
    base_name: str
    expression: str
    category: str = "General"
    sub_category: str = ""
    doc: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    agg_method: str | list[str] | None = None

    @property
    def params_json(self) -> str:
        return json.dumps(self.params, sort_keys=True, ensure_ascii=False)

    def to_row(self) -> dict[str, Any]:
        return {
            "feature_name": self.name,
            "base_name": self.base_name,
            "category": self.category,
            "sub_category": self.sub_category,
            "doc": self.doc,
            "agg_method": self.agg_method,
            "params": self.params_json,
            "expression": self.expression,
        }

# ----------------------------
# Requirements
# ----------------------------

@dataclass
class InputRequirements:
    """
    输入依赖声明：
    - bars_cols：来自行情表（OHLCV/amount/vwap/...）
    - exposure_cols：来自暴露表（industry/log_cap/bench_ret/...）

    可选：per-feature 精细依赖（用于只计算子集 features 时收敛读列/拼接）。
    """
    bars_cols: set[str] = field(default_factory=set)
    exposure_cols: set[str] = field(default_factory=set)

    per_feature_bars: dict[str, set[str]] = field(default_factory=dict)
    per_feature_exposure: dict[str, set[str]] = field(default_factory=dict)

    def merge_inplace(self, other: "InputRequirements") -> None:
        self.bars_cols |= other.bars_cols
        self.exposure_cols |= other.exposure_cols

        for k, v in other.per_feature_bars.items():
            self.per_feature_bars.setdefault(k, set()).update(v)
        for k, v in other.per_feature_exposure.items():
            self.per_feature_exposure.setdefault(k, set()).update(v)

    def subset(self, *, features: Sequence[str] | None = None) -> "InputRequirements":
        """
        若存在 per-feature 依赖，且给定 features 子集，则返回更小的依赖集合；
        否则退化为全局 bars_cols / exposure_cols。
        """
        out = InputRequirements(bars_cols=set(self.bars_cols), exposure_cols=set(self.exposure_cols))

        if features is None:
            return out

        if not (self.per_feature_bars or self.per_feature_exposure):
            return out

        out.bars_cols = set()
        out.exposure_cols = set()
        for f in features:
            out.bars_cols |= self.per_feature_bars.get(f, set())
            out.exposure_cols |= self.per_feature_exposure.get(f, set())
        return out


class BaseFeatureSpec:
    """
    纯定义层（Spec）：
    - 注册 feature：name -> expression(str)
    - 参数化展开（网格）
    - 元数据 FactorDef
    - 输入依赖声明（bars/exposures）
    """
    # 子类可覆盖
    factor_name: str = "base_spec"
    category_default: str = "General"
    sub_category_default: str = ""

    def __init__(self) -> None:
        # 注册产物（强制有序：允许表达式引用先前注册的列时保持执行顺序一致）
        self.feature_expressions: "OrderedDict[str, str]" = OrderedDict()

        # 元数据
        self.definitions: list[FactorDef] = []

        # 输入依赖（全局 + 可选 per-feature）
        self.requirements = InputRequirements()

        # 防止重复 register
        self._registered: bool = False

    def register(self, **kwargs: Any) -> None:
        """
        子类必须实现：在这里调用 add_feature/add_parametric_feature/require_bars/require_exposures 等完成注册。
        """
        raise NotImplementedError

    def build(self, **kwargs: Any) -> "BaseFeatureSpec":
        """
        统一入口：保证 register 只执行一次。
        """
        if not self._registered:
            self.register(**kwargs)
            self._registered = True
        return self

    @staticmethod
    def _fmt_val(v: Any) -> str:
        if isinstance(v, bool): return "true" if v else "false"
        if isinstance(v, (int, float)): return f"{v:g}"
        return str(v)

    def _make_name(self, base: str, params: Mapping[str, Any]) -> str:
        """
        统一命名规则（不做压缩）：
        - 无参数：base
        - 有参数：base_{k1}{v1}_{k2}{v2}...
        - 参数顺序：按 params 插入顺序（dict 保序）
        """
        if not params:
            return base

        parts: list[str] = []
        for k, v in params.items():
            parts.append(f"{k}{self._fmt_val(v)}")

        return f"{base}_{'_'.join(parts)}"

    def _assert_unique(self, name: str) -> None:
        if name in self.feature_expressions:
            raise ValueError(f"Duplicate feature name: {name}")
          

    def _register_meta(
        self,
        *,
        name: str,
        base_name: str,
        expr: str,
        category: str,
        sub_category: str,
        doc: str,
        params: Mapping[str, Any],
        agg_method: str | list[str] | None,
    ) -> None:
        self.definitions.append(
            FactorDef(
                name=name,
                base_name=base_name,
                expression=expr,
                category=category,
                sub_category=sub_category,
                doc=doc,
                params=dict(params),
                agg_method=agg_method,
            )
        )


    # --------- deps declaration ---------

    def require_bars(self, *cols: str) -> None:
        self.requirements.bars_cols.update(cols)

    def require_exposures(self, *cols: str) -> None:
        self.requirements.exposure_cols.update(cols)

    def require_for_feature(
        self,
        name: str,
        *,
        bars_cols: Iterable[str] = (),
        exposure_cols: Iterable[str] = (),
    ) -> None:
        if bars_cols:
            self.requirements.per_feature_bars.setdefault(name, set()).update(bars_cols)
        if exposure_cols:
            self.requirements.per_feature_exposure.setdefault(name, set()).update(exposure_cols)

    # --------- register API ---------

    def add_feature(
        self,
        name: str,
        expression: str,
        *,
        base_name: str | None = None,
        category: str | None = None,
        sub_category: str | None = None,
        doc: str = "",
        params: Mapping[str, Any] | None = None,
        agg_method: str | list[str] | None = None,
        bars_cols: Iterable[str] = (),
        exposure_cols: Iterable[str] = (),
    ) -> None:
        """
        注册一个 feature（定义层，不执行计算）
        """
        self._assert_unique(name)
        self.feature_expressions[name] = expression

        # 依赖声明（可选：per-feature）
        if bars_cols:
            self.requirements.per_feature_bars.setdefault(name, set()).update(bars_cols)
        if exposure_cols:
            self.requirements.per_feature_exposure.setdefault(name, set()).update(exposure_cols)

        # 元数据
        self._register_meta(
            name=name,
            base_name=base_name or name,
            expr=expression,
            category=category or self.category_default,
            sub_category=sub_category or self.sub_category_default,
            doc=doc,
            params=params or {},
            agg_method=agg_method,
        )

    def add_parametric_feature(
        self,
        *,
        base_name: str,
        expr_tpl: str,
        param_grid: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Sequence[Any]]] | None = None,
        category: str | None = None,
        sub_category: str | None = None,
        doc: str = "",
        agg_method: str | list[str] | None = None,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
        dedup: bool = True,
        bars_cols: Iterable[str] = (),
        exposure_cols: Iterable[str] = (),
    ) -> None:
        """
        参数化展开注册 feature：
        - param_grid 支持 dict 或 list[dict]（多段网格）
        - 组合展开按网格 key 的插入顺序命名/format
        """
        if not param_grid:
            self.add_feature(
                name=base_name,
                base_name=base_name,
                expression=expr_tpl,
                category=category,
                sub_category=sub_category,
                doc=doc,
                params={},
                agg_method=agg_method,
                bars_cols=bars_cols,
                exposure_cols=exposure_cols,
            )
            return

        grids = [param_grid] if isinstance(param_grid, Mapping) else list(param_grid)
        seen: set[str] = set()

        for grid in grids:
            keys = list(grid.keys())
            values = [list(grid[k]) for k in keys]

            for combo in itertools.product(*values):
                params = {k: combo[i] for i, k in enumerate(keys)}
                if predicate is not None and not predicate(params):
                    continue

                name = self._make_name(base_name, params)
                if dedup and name in seen:
                    continue

                try:
                    expr = expr_tpl.format(**params)
                except KeyError as e:
                    raise ValueError(f"Param missing for expr_tpl: {e}") from e

                self.add_feature(
                    name=name,
                    base_name=base_name,
                    expression=expr,
                    category=category,
                    sub_category=sub_category,
                    doc=doc,
                    params=params,
                    agg_method=agg_method,
                    bars_cols=bars_cols,
                    exposure_cols=exposure_cols,
                )
                seen.add(name)

    # --------- export API ---------

    def export_features(self) -> Mapping[str, str]:
        return self.feature_expressions

    def export_definitions(self) -> Sequence[FactorDef]:
        return self.definitions

    def export_info_rows(self) -> list[dict[str, Any]]:
        return [d.to_row() for d in self.definitions]

    def required_bars_cols(self, *, features: Sequence[str] | None = None) -> set[str]:
        req = self.requirements.subset(features=features)
        return set(req.bars_cols)

    def required_exposure_cols(self, *, features: Sequence[str] | None = None) -> set[str]:
        req = self.requirements.subset(features=features)
        return set(req.exposure_cols)


__all__ = [
    "FactorDef",
    "InputRequirements",
    "BaseFeatureSpec",
]