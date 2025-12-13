# file: core/synthesizer.py
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import polars as pl


class FactorSynthesizer:
    """
    通用因子重采样/合成器 (Production Ready)

    - 核心职责：将高频因子表（如 10m/60m）降频聚合为低频表（如 1h/1d）；
    - 性能优化：全面支持 LazyFrame，利用 Query Optimizer 优化内存和速度；
    - 配置驱动：支持通过 override、FactorDef、默认规则表等多级配置指定聚合方式；
    - 灵活性：支持任意频率（1m, 10m, 60m -> 1h, 1d），支持时间标签边缘对齐。
    """

    # 默认聚合规则（约定优于配置）
    DEFAULT_AGG_RULES: Dict[str, Union[str, List[str]]] = {
        # 趋势/状态类 -> 取收盘状态
        "Slope": "last",
        "R2": "last",
        "Bias": "last",
        "ROC": "last",
        "MA_Speed": "last",
        # 流量/累积类 -> 全周期累加
        "Volume": "sum",
        "MoneyFlow": "sum",
        # 情绪/均值类 -> 全周期平均
        "IBS": "mean",
        "Shadow": "mean",
        "Efficiency": "mean",
        # 隔夜类 -> 取开盘状态 (桶内第一个值)
        "Gap": "first",
        "GapBias": "first",
        # 结构类 -> 尾盘减早盘
        "SmartMoney": "diff_last_first",
    }

    def __init__(
        self,
        *,
        rules: Dict[str, str] | None = None,
        overrides: Dict[str, str] | None = None,
    ) -> None:
        self.logger = logging.getLogger("FactorSynthesizer")

        # 1. 初始化规则表
        self._rules: Dict[str, str] = dict(self.DEFAULT_AGG_RULES)
        if rules:
            self._rules.update(rules)
        
        # 2. 初始化覆盖表 (最高优先级)
        self._overrides: Dict[str, str] = overrides.copy() if overrides else {}

        # 3. 注册内置聚合算子
        self._aggregators: Dict[str, Callable[[str], pl.Expr]] = {
            # 基础统计
            "last": lambda c: pl.col(c).last(),
            "first": lambda c: pl.col(c).first(),
            "mean": lambda c: pl.col(c).mean(),
            "sum": lambda c: pl.col(c).sum(),
            "max": lambda c: pl.col(c).max(),
            "min": lambda c: pl.col(c).min(),
            "std": lambda c: pl.col(c).std(),
            "median": lambda c: pl.col(c).median(),
            
            # 逻辑别名 (Bypass)
            "identity": lambda c: pl.col(c).last(),
            "pass": lambda c: pl.col(c).last(),
            
            # 结构化逻辑 (依赖于桶内已按时间排序)
            "diff_last_first": lambda c: pl.col(c).last() - pl.col(c).first(),
            "range": lambda c: pl.col(c).max() - pl.col(c).min(),
            
            # 过滤空值
            "last_valid": lambda c: pl.col(c).filter(pl.col(c).is_not_null()).last(),
            "first_valid": lambda c: pl.col(c).filter(pl.col(c).is_not_null()).first(),
        }

    # ---------- 对外 API：动态注册 ----------
    def register_aggregator(self, name: str, func: Callable[[str], pl.Expr]) -> None:
        """注册新的聚合算法"""
        self._aggregators[name] = func

    def register_rule(self, sub_category: str, method: Union[str, List[str]]) -> None:
        """注册子类别的默认聚合规则"""
        self._rules[sub_category] = method

    def override_factor(self, factor_name: str, method: Union[str, List[str]]) -> None:
        """强制指定某因子的聚合方式"""
        self._overrides[factor_name] = method

    # ---------- 内部辅助 ----------
    @staticmethod
    def _normalize_interval(interval: str) -> str:
        """
        标准化时间频率，适配 Polars dt.truncate 标准。
        Input 支持: 1d, 1m, 10m, 30m, 60m (及 60min, daily 等变体)
        """
        s = interval.lower().strip()
        
        # 规范化 60m -> 1h
        if s in ("60m", "60min", "1h"):
            return "1h"
        
        # 规范化 daily -> 1d
        if s in ("daily", "day", "1d", "24h"):
            return "1d"
            
        # 其他透传 (1m, 10m, 30m, 1w 等)
        return s

    @staticmethod
    def _build_converted_alias(fname: str, target_interval: str, method: str) -> str:
        """
        生成带转换信息与聚合后缀的列名（新规）：
        - 输入: 原列名 fname（如 feat_60m_trend_slope_close_w4）
        - 输出: feat_{target}_{rest}_from_{src}__agg_{method}
        若 fname 未遵循 feat_* 命名，则直接附加 __agg_{method}。
        """
        import re as _re
        m = _re.match(r"^feat_([^_]+)_(.*)$", fname)
        if m:
            src, rest = m.group(1), m.group(2)
            return f"feat_{target_interval}_{rest}_from_{src}__agg_{method}"
        return f"{fname}__agg_{method}"

    def _choose_methods(self, f_def: Any, use_all_methods: bool = False) -> Tuple[List[str], str]:
        """
        决策聚合方式（可多种），优先级：
        Override > FactorDef.agg_method > Rule(by sub_category) > Fallback(['last'])
        - 支持方法为 str 或 list[str]
        - 统一清洗：去重、去空、过滤 'skip'/'ignore'
        """
        fname = getattr(f_def, "name", "")
        sub_cat = getattr(f_def, "sub_category", "")

        # 1) 覆盖
        if fname in self._overrides:
            spec = self._overrides[fname]
            source = "override"
        else:
            # 2) 因子定义
            m = getattr(f_def, "agg_method", None)
            if m is not None:
                spec = m
                source = "factor_def"
            else:
                # 3) 默认规则
                spec = self._rules.get(sub_cat)
                source = "rule" if spec is not None else "fallback"

        # 标准化为列表
        if spec is None:
            methods = list(self._aggregators.keys()) if use_all_methods else ["last"]
        elif isinstance(spec, str):
            methods = [spec]
        elif isinstance(spec, list):
            methods = list(spec)
        else:
            # 非法类型，回退
            methods = ["last"]
            source = "fallback"

        # 清洗
        out: List[str] = []
        seen: set[str] = set()
        for mm in methods:
            if not isinstance(mm, str):
                continue
            mm2 = mm.strip()
            if not mm2 or mm2.lower() in ("none", "null"):
                continue
            if mm2 in ("skip", "ignore"):
                continue
            if mm2 in seen:
                continue
            seen.add(mm2)
            out.append(mm2)

        if not out:
            out = ["last"]
            source = "fallback"

        return out, source

    # ---------- 核心执行引擎 ----------
    def synthesize(
        self,
        *,
        df: Union[pl.DataFrame, pl.LazyFrame],
        factor_defs: Iterable[Any],
        interval: str = "1d",           # 目标频率
        time_col: str = "datetime",
        symbol_col: str = "vt_symbol",
        label_edge: str = "left",       # 'left' or 'right'
        min_count: int | None = None,   # 最小样本数过滤
        return_rules: bool = False,
        use_all_methods: bool = False,
    ) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
        """
        执行重采样与聚合。
        
        :param df: 输入因子表 (支持 Eager/Lazy)
        :param interval: 目标频率字符串 (如 "1d", "10m")
        :param label_edge: 时间标签对齐方式。'left'为开始时间(10:00代表10:00-11:00)，'right'为结束时间。
        :return: 聚合后的 DataFrame
        """
        # 0. 快速空检查 (仅针对 DataFrame)
        if isinstance(df, pl.DataFrame) and df.is_empty():
            return (df, pl.DataFrame()) if return_rules else df

        # 1. 统一转为 Lazy 模式以优化性能
        lz = df.lazy() if isinstance(df, pl.DataFrame) else df
        target_interval = self._normalize_interval(interval)

        # 2. 生成时间桶 (Time Bucket)
        # dt.truncate 是核心降频算子
        lz = lz.with_columns(
            pl.col(time_col).dt.truncate(target_interval).alias("time_bucket")
        )

        # 处理右边缘标签 (Label Right)
        if label_edge == "right":
            lz = lz.with_columns(pl.col("time_bucket").dt.offset_by(target_interval))

        # 3. 排序
        # 确保 First/Last 逻辑正确。Lazy 模式下 Sort 会被优化推后执行。
        lz = lz.sort([symbol_col, "time_bucket", time_col])

        # 4. 获取列名 (Schema Check)
        try:
            # Polars >= 0.20
            existing_cols = set(lz.collect_schema().names())
        except AttributeError:
            existing_cols = set(lz.schema.keys())

        # 5. 构建聚合表达式
        uniq_defs = {getattr(x, "name", id(x)): x for x in factor_defs}.values()
        agg_exprs: List[pl.Expr] = []
        applied_rules: List[Dict[str, Any]] = []

        for f in uniq_defs:
            fname = getattr(f, "name", None)
            if not fname or fname not in existing_cols:
                continue

            methods, source = self._choose_methods(f, use_all_methods=use_all_methods)
            sub_cat = getattr(f, "sub_category", "")

            for m in methods:
                agg_func = self._aggregators.get(m)
                if not agg_func:
                    self.logger.warning("Unknown agg method '%s' for '%s'. Fallback to 'last'", m, fname)
                    m = "last"
                    agg_func = self._aggregators[m]

                alias = self._build_converted_alias(fname, target_interval, m)
                agg_exprs.append(agg_func(fname).alias(alias))
                if return_rules:
                    applied_rules.append({
                        "original_name": fname,
                        "factor_name": alias,
                        "sub_category": sub_cat,
                        "method_used": m,
                        "source": source,
                    })

        # 6. 执行聚合
        # 如果没有有效的聚合表达式，返回仅包含索引的空表
        if not agg_exprs:
            res_lz = lz.select([symbol_col, "time_bucket"]).unique().rename({"time_bucket": time_col})
            return (res_lz.collect(), pl.DataFrame(applied_rules)) if return_rules else res_lz.collect()

        # GroupBy 聚合
        lz_agg = lz.group_by(["time_bucket", symbol_col]).agg(agg_exprs)

        # 7. 应用 min_count (Lazy Join 过滤)
        if isinstance(min_count, int) and min_count > 0:
            counts = (
                lz.group_by(["time_bucket", symbol_col])
                .agg(pl.len().alias("_count"))
            )
            lz_agg = (
                lz_agg.join(counts, on=["time_bucket", symbol_col], how="left")
                .filter(pl.col("_count") >= min_count)
                .drop("_count")
            )

        # 8. 收尾：重命名
        lz_agg = lz_agg.rename({"time_bucket": time_col})

        # 最终排序
        lz_agg = lz_agg.sort([time_col, symbol_col])

        # 9. 触发计算 (Collect)
        final_df = lz_agg.collect()

        if return_rules:
            rules_df = pl.from_dicts(applied_rules) if applied_rules else pl.DataFrame()
            return final_df, rules_df
        
        return final_df
