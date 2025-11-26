from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Iterable
import json
import polars as pl
from vnpy.alpha.dataset import AlphaDataset # 引用原生基类

# --- DTO 定义放这里 ---
@dataclass
class FactorDef:
    name: str
    base_name: str
    expression: str
    category: str
    sub_category: str
    params: dict[str, Any]
    params_json: str = field(init=False)
    
    def __post_init__(self):
        self.params_json = json.dumps(self.params, sort_keys=True)

    def to_row(self) -> dict:
        return {
            "factor_name": self.name,
            "base_name": self.base_name,
            "category": self.category,
            "sub_category": self.sub_category,
            "params": self.params_json,
            "expression": self.expression
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

    def _register_meta(self, name, base_name, expr, cat, sub_cat, params):
        factor_def = FactorDef(name, base_name, expr, cat, sub_cat, params)
        self.definitions.append(factor_def)

    def add_parametric_feature(  # 去掉下划线，作为公开 API
        self,
        base_name: str,
        expr_tpl: str,
        param_grid: dict | list[dict] | None = None,
        *,
        category: str = "General",
        sub_category: str = "",
        predicate: Callable[[dict], bool] | None = None,
        dedup: bool = True,
    ) -> None:
        """
        核心方法：批量生成因子并自动注册元数据
        """
        if not param_grid:
            self.add_feature(base_name, expr_tpl)
            self._register_meta(base_name, base_name, expr_tpl, category, sub_category, {})
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
                    self._register_meta(name, base_name, expr, category, sub_category, params)
                except KeyError as e:
                    raise ValueError(f"Param missing: {e}")

    def get_factor_info(self) -> pl.DataFrame:
        if not self.definitions: return pl.DataFrame()
        return pl.from_dicts([d.to_row() for d in self.definitions])