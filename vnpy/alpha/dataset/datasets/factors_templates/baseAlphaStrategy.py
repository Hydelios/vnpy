from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Iterable
import json
import polars as pl
from vnpy.alpha.dataset import AlphaDataset  # 引用原生基类

# --- DTO 定义放这里 ---
@dataclass
class FactorDef:
    """
    因子定义元数据对象
    - 这里仅做字段承载，不做任何规则推断。
    - 聚合方法 agg_method 为可选，若为空将由 Synthesizer 在执行阶段映射默认规则。
    """
    name: str
    base_name: str
    expression: str
    category: str
    sub_category: str
    params: dict[str, Any]
    agg_method: str | list[str] | None = None  # 聚合方法：可为单个方法或方法列表；为空则由 Synthesizer 映射
    params_json: str = field(init=False)

    def __post_init__(self) -> None:
        self.params_json = json.dumps(self.params, sort_keys=True)

    def to_row(self) -> dict:
        return {
            "factor_name": self.name,
            "base_name": self.base_name,
            "category": self.category,
            "sub_category": self.sub_category,
            "agg_method": self.agg_method,
            "params": self.params_json,
            "expression": self.expression,
        }

# --- 中间层基类 ---
class BaseAlphaStrategy(AlphaDataset):
    """
    增强版的 AlphaDataset，支持元数据管理和自动参数展开
    """
    def __init__(self, *args, **kwargs):
        # 调用父类 (vnpy原版) 的初始化
        super().__init__(*args, **kwargs)
        
        # 初始化元数据列表
        self.definitions: list[FactorDef] = []

    # ============ 通用工具方法 (从 ApolloNo1 移过来) ============
    
    @staticmethod
    def _fmt_val(v: Any) -> str:
        if isinstance(v, bool): return "true" if v else "false"
        if isinstance(v, (int, float)): return f"{v:g}"
        return str(v)

    @classmethod
    def _make_name(cls, base: str, params: dict[str, Any]) -> str:
        if not params: return base
        parts = []
        for k in sorted(params):
            v_str = cls._fmt_val(params[k])
            if v_str in ["open", "high", "low", "close", "volume", "vwap", "amount"]:
                parts.append(v_str)
            else:
                parts.append(f"{k}{v_str}")
        return f"{base}_{'_'.join(parts)}"

    def _register_meta(
        self,
        name: str,
        base_name: str,
        expr: str,
        cat: str,
        sub_cat: str,
        params: dict[str, Any],
        agg_method: str | None = None,
    ) -> None:
        """
        注册单个因子元数据。
        仅保存 agg_method，不做默认规则推断；默认映射在 Synthesizer 阶段完成。
        """
        factor_def = FactorDef(
            name=name,
            base_name=base_name,
            expression=expr,
            category=cat,
            sub_category=sub_cat,
            params=params,
            agg_method=agg_method,
        )
        self.definitions.append(factor_def)

    def add_parametric_feature(  # 去掉下划线，作为公开 API
        self,
        base_name: str,
        expr_tpl: str,
        param_grid: dict | list[dict] | None = None,
        *,
        category: str = "General",
        sub_category: str = "",
        agg_method: str | None = None,
        predicate: Callable[[dict], bool] | None = None,
        dedup: bool = True,
    ) -> None:
        """
        核心方法：批量生成因子并自动注册元数据
        """
        if not param_grid:
            self.add_feature(base_name, expr_tpl)
            self._register_meta(
                base_name,
                base_name,
                expr_tpl,
                category,
                sub_category,
                {},
                agg_method,
            )
            return

        grids = [param_grid] if isinstance(param_grid, dict) else param_grid
        seen: set[str] = set()

        for grid in grids:
            keys = list(grid.keys())
            values = [list(v) for v in grid.values()]
            for combo in product(*values):
                params = {k: combo[i] for i, k in enumerate(keys)}
                if predicate and not predicate(params): continue
                
                name = self._make_name(base_name, params)
                if dedup and name in seen: continue
                
                try:
                    expr = expr_tpl.format(**params)
                    self.add_feature(name, expr)
                    seen.add(name)
                    self._register_meta(
                        name,
                        base_name,
                        expr,
                        category,
                        sub_category,
                        params,
                        agg_method,
                    )
                except KeyError as e:
                    raise ValueError(f"Param missing: {e}")

    def get_factor_info(self) -> pl.DataFrame:
        if not self.definitions: return pl.DataFrame()
        return pl.from_dicts([d.to_row() for d in self.definitions])
