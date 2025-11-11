# =============================================
# File: vnpy/alpha/factor_templates/polars_factor_template.py
# =============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Any, List, Tuple, Union, ClassVar
import polars as pl

ExprLike = Union[str, pl.Expr]

@dataclass
class FactorTemplate:
    """标准化的 Polars 因子模板基类

    目标：
    - 让每个因子以“类”的形式声明：名称/依赖列/参数网格/表达式生成器
    - 与 AlphaDataset 解耦，只负责“生成 (name → expression) 字典”
    - expression 既可为字符串（走 calculate_by_expression），也可为 pl.Expr（走 calculate_by_polars）
    """

    # 说明：以下属性作为“类级常量”使用，便于子类覆盖；
    # 使用 ClassVar 避免被 dataclass 当作实例字段，从而支持子类通过类属性覆盖。
    factor_name: ClassVar[str] = ""
    params: ClassVar[Dict[str, Iterable[Any]]] = {}
    required_columns: ClassVar[Tuple[str, ...]] = tuple()
    doc: ClassVar[str] = ""
    expr_kind: ClassVar[str] = "auto"  # "auto" | "str" | "pl"

    def build_expr(self, **p) -> ExprLike:
        """子类必须实现：返回一个因子的表达式（str 或 pl.Expr）。"""
        raise NotImplementedError

    # --------- 框架方法 ---------
    def expand(self, runtime_overrides: Dict[str, Any] | None = None) -> Dict[str, ExprLike]:
        """展开参数网格生成多个 (列名 → 表达式)。
        runtime_overrides 用于在运行时覆盖 params 中的部分键。
        """
        grid = self._param_grid(runtime_overrides or {})
        out: Dict[str, ExprLike] = {}
        for p in grid:
            expr = self.build_expr(**p)
            name = self._make_name(p)
            out[name] = expr
        return out

    # --------- 工具方法 ---------
    def _param_grid(self, runtime: Dict[str, Any]) -> List[Dict[str, Any]]:
        # 读取类级参数定义（子类可通过类属性覆盖）
        cls_params: Dict[str, Iterable[Any]] = getattr(type(self), "params", {}) or {}
        if not cls_params:
            return [runtime.copy()]
        keys = list(cls_params.keys())
        lists = [list(v) for v in cls_params.values()]
        # 笛卡尔积
        out: List[Dict[str, Any]] = []
        def rec(i: int, cur: Dict[str, Any]):
            if i == len(keys):
                # 覆盖运行时参数
                cur2 = {**cur, **runtime}
                out.append(cur2)
                return
            k = keys[i]
            for vv in lists[i]:
                cur[k] = vv
                rec(i + 1, cur)
        rec(0, {})
        return out

    def _make_name(self, p: Dict[str, Any]) -> str:
        if not p:
            return getattr(type(self), "factor_name", "")
        kv = ".".join([f"{k}={p[k]}" for k in sorted(p)])
        base = getattr(type(self), "factor_name", "")
        return f"{base}__{kv}"
