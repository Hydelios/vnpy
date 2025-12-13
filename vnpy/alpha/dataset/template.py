import time
from datetime import datetime
from typing import cast
from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.context import BaseContext

import polars as pl
import pandas as pd
from tqdm import tqdm                                               # type: ignore
from alphalens.utils import get_clean_factor_and_forward_returns    # type: ignore
from alphalens.tears import create_full_tear_sheet                  # type: ignore

from ..logger import logger
from .utility import (
    to_datetime,
    Segment,
    calculate_by_expression,
    calculate_by_polars
)
from .cache_manager import FactorCacheManager


class AlphaDataset:
    """Alpha dataset template class"""

    def __init__(
        self,
        df: pl.DataFrame,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
        interval: str = "1d",
        process_type: str = "append",
        enable_cache: bool = False,
        cache_dir: str | None = None
    ) -> None:
        """Constructor"""
        self.df: pl.DataFrame = df
        self.interval: str = interval  # 数据频率标识: 1d, 1m, 10m, 30m, 60m

        # DataFrames for processed data
        self.result_df: pl.DataFrame
        self.raw_df: pl.DataFrame
        self.infer_df: pl.DataFrame
        self.learn_df: pl.DataFrame

        # New version
        self.data_periods: dict[Segment, tuple[str, str]] = {
            Segment.TRAIN: train_period,
            Segment.VALID: valid_period,
            Segment.TEST: test_period
        }

        self.feature_expressions: dict[str, str | pl.expr.expr.Expr] = {}
        self.feature_results: dict[str, pl.DataFrame] = {}
        self.label_expression: str = ""

        self.process_type: str = process_type
        self.infer_processors: list = []
        self.learn_processors: list = []
        
        # 用户自定义的运行参数（从JSON传入）；默认None，保存/加载时可挂载
        self.params: dict | None = None

        # Cache configuration
        self.enable_cache: bool = enable_cache
        if enable_cache:
            # 传递interval到缓存管理器
            self.cache_manager = FactorCacheManager(cache_dir, interval)
        else:
            self.cache_manager = None

    def add_feature(
        self,
        name: str,
        expression: str | pl.expr.expr.Expr | None = None,
        result: pl.DataFrame | None = None
    ) -> None:
        """
        Add a feature expression
        """
        if expression is not None and result is not None:
            raise ValueError("Only one of 'expression' or 'result' can be provided")

        if expression is not None:
            self.feature_expressions[name] = expression
        elif result is not None:
            self.feature_results[name] = result

    def set_label(self, expression: str) -> None:
        """
        Set the label expression
        """
        self.label_expression = expression

    def add_processor(self, task: str, processor: Callable[[pl.DataFrame], None]) -> None:
        """
        Add a feature preprocessor
        """
        if task == "infer":
            self.infer_processors.append(processor)
        else:
            self.learn_processors.append(processor)

    def calculate_features(self, max_workers: int | None = None) -> None:
        """
        仅计算特征（不计算标签、不构建视图）。

        - 支持缓存加速（enable_cache=True）。
        - 计算结果写入 self.result_df（包含原始 self.df 与新增特征列）。
        """
        results: list = []

        if self.enable_cache and self.cache_manager:
            logger.info("使用缓存管理器进行因子计算")

            context = get_context("spawn") if max_workers and max_workers > 1 else None
            n_jobs = max_workers if max_workers else 1

            self.result_df = self.cache_manager.calculate_factors_with_cache(
                df=self.df,
                expressions=self.feature_expressions,
                n_jobs=n_jobs,
                context=context,
            )
        else:
            # 表达式特征（不含 label）
            expressions: list[tuple[str, str | pl.expr.expr.Expr]] = list(self.feature_expressions.items())

            logger.info("开始计算表达式因子特征")

            args: list[tuple] = [(self.df, name, expression) for name, expression in expressions]

            if max_workers and max_workers > 0:
                context: BaseContext = get_context("spawn")
                with context.Pool(processes=max_workers) as pool:
                    it = pool.imap(calculate_feature, args)
                    for result in tqdm(it, total=len(args)):
                        results.append(result)
            else:
                for arg in tqdm(args):
                    result = calculate_feature(arg)
                    results.append(result)

            self.result_df = self.df.with_columns(results)

    def attach_label(self) -> None:
        """
        在 self.result_df 上按 label_expression 计算并附加标签列 "label"。
        若未设置 label_expression 则无操作。
        """
        if not self.label_expression:
            return

        if isinstance(self.label_expression, pl.expr.expr.Expr):
            label_result = calculate_by_polars(self.result_df, self.label_expression)["data"].alias("label")
        else:
            label_result = calculate_by_expression(self.result_df, self.label_expression)["data"].alias("label")
        self.result_df = self.result_df.with_columns(label_result)

    def merge_feature_results(self) -> None:
        """
        合并外部特征结果（feature_results）到 self.result_df。
        """
        if not self.feature_results:
            return
        logger.info("开始合并结果数据因子特征")
        for name, result in tqdm(self.feature_results.items()):
            result = result.rename({"data": name})
            self.result_df = self.result_df.join(result, on=["datetime", "vt_symbol"], how="inner")

    def build_raw(self, filters) -> None:
        """
        基于 self.result_df 生成 self.raw_df（可选过滤），并仅保留键与特征/标签列。
        """
        # rebuild_views(filters=...) 的常见诉求是：基于已存在的 raw_df 做成分股区间过滤，
        # 而不是每次都从宽表 result_df 重新过滤（会非常慢，且在加载 parquet 后 df.width 不一定可靠）。
        raw_df = self.raw_df.fill_null(float("nan"))

        if filters:
            logger.info("开始筛选成分股数据")
            # 先按 vt_symbol 粗过滤，减少后续分组开销
            symbols = list(filters.keys())
            if symbols:
                raw_df = raw_df.filter(pl.col("vt_symbol").is_in(symbols))

            # 将 DataFrame 按 vt_symbol 分组成多个小表，避免“每个 symbol 都扫全表”的 O(N_symbols*N_rows) 开销
            # 注意：Polars 的 partition_by(as_dict=True) 键通常是 tuple（即使只分一列也可能是 (value,)），
            # 因此这里显式用 ["vt_symbol"] 并按 (vt_symbol,) 取值，避免全量 miss 导致结果为空。
            grouped = raw_df.partition_by(["vt_symbol"], as_dict=True)

            parts: list[pl.DataFrame] = []
            for vt_symbol, ranges in tqdm(filters.items(), total=len(filters)):
                if not ranges:
                    continue
                df_symbol = grouped.get((vt_symbol,))
                if df_symbol is None or df_symbol.is_empty():
                    continue

                # 同一标的的多段区间合并为一个布尔表达式
                expr = None
                for start, end in ranges:
                    rng = (pl.col("datetime") >= pl.lit(start)) & (pl.col("datetime") <= pl.lit(end))
                    expr = rng if expr is None else (expr | rng)

                if expr is None:
                    continue

                parts.append(df_symbol.filter(expr))

            raw_df = pl.concat(parts) if parts else raw_df.head(0)

        self.raw_df = raw_df.sort(["datetime", "vt_symbol"])

    def build_infer(self) -> None:
        """
        基于 self.raw_df 生成 self.infer_df 并应用 infer_processors。
        """
        self.infer_df = self.raw_df
        for processor in self.infer_processors:
            self.infer_df = processor(df=self.infer_df)

    def build_learn(self) -> None:
        """
        基于 self.raw_df/self.infer_df 生成 self.learn_df 并应用 learn_processors。
        - process_type == "append" 时，以 infer_df 为基（允许此前清洗改变样本集合）。
        - 否则，以 raw_df 为基。
        """
        if self.process_type == "append":
            self.learn_df = self.infer_df
        else:
            self.learn_df = self.raw_df
        for processor in self.learn_processors:
            self.learn_df = processor(df=self.learn_df)

    def rebuild_views(self, filters: dict | None = None) -> None:
        """
        仅基于现有 self.result_df 重新构建视图：raw/infer/learn。
        适用于滚动回测时重复改 period/处理器，而不重复计算特征。
        """
        self.build_raw(filters=filters)
        self.build_infer()
        self.build_learn()

    def clear_processors(self, task: str | None = None) -> None:
        """
        清空处理器：task=None 清空全部；"infer" 仅清空推理处理器；其他清空学习处理器。
        """
        if task is None:
            self.infer_processors = []
            self.learn_processors = []
        elif task == "infer":
            self.infer_processors = []
        else:
            self.learn_processors = []

    def set_periods(
        self,
        train_period: tuple[str, str] | None = None,
        valid_period: tuple[str, str] | None = None,
        test_period: tuple[str, str] | None = None,
    ) -> None:
        """
        动态更新数据分段（便于滚动回测窗口推进）。
        """
        if train_period is not None:
            self.data_periods[Segment.TRAIN] = train_period
        if valid_period is not None:
            self.data_periods[Segment.VALID] = valid_period
        if test_period is not None:
            self.data_periods[Segment.TEST] = test_period

    def prepare_data(self, filters: dict | None = None, max_workers: int | None = None) -> None:
        """
        计算特征 →（可选）附加标签 → 合并外部结果 → 构建 raw/infer/learn 视图。
        保持向后兼容的一站式入口。
        """
        self.calculate_features(max_workers=max_workers)
        self.attach_label()
        self.merge_feature_results()
        self.rebuild_views(filters=filters)

    def fetch_raw(self, segment: Segment) -> pl.DataFrame:
        """
        Get raw data for a specific segment
        """
        start, end = self.data_periods[segment]
        return query_by_time(self.raw_df, start, end)

    def fetch_infer(self, segment: Segment) -> pl.DataFrame:
        """
        Get inference data for a specific segment
        """
        start, end = self.data_periods[segment]
        return query_by_time(self.infer_df, start, end)

    def fetch_learn(self, segment: Segment) -> pl.DataFrame:
        """
        Get learning data for a specific segment
        """
        start, end = self.data_periods[segment]
        return query_by_time(self.learn_df, start, end)

    def show_feature_performance(
        self,
        name: str,
        *,
        quantiles: int | None = None,
        periods: tuple[int, ...] | list[int] | None = None,
        max_loss: float | None = None,
        bins: list[float] | None = None,
        long_short: bool | None = None,
        group_neutral: bool | None = None,
        by_group: bool | None = None,
        allow_fallback: bool | None = None,
        quantile_order: str | None = None,  # "asc" 或 "desc"
        plots: dict | None = None,
    ) -> None:
        """
        单因子绩效分析：简化为调用 analysis.alphalens_backend 封装。

        - Notebook 调用保持简单：dataset.show_feature_performance("alpha101_003")。
        - 可选参数若传入，则覆盖 params["factors_analysis"] 中对应项。
        - 具体的 Alphalens 清洗/绘图/保存逻辑均在 wrapper 中实现。
        """
        # 组装配置：以 self.params.factors_analysis 为基，叠加调用方覆盖
        fa_cfg: dict = {}
        if isinstance(self.params, dict):
            fa_cfg = cast(dict, self.params.get("factors_analysis", {})).copy()

        # 调用参数覆盖（仅处理与 Alphalens 相关的项）
        if periods is not None:
            fa_cfg["alphalens_periods"] = list(periods)
        if quantiles is not None:
            fa_cfg["alphalens_quantiles"] = int(quantiles)
        if bins is not None:
            fa_cfg["alphalens_bins"] = list(bins)
        if max_loss is not None:
            fa_cfg["alphalens_max_loss"] = float(max_loss)
        if quantile_order is not None:
            fa_cfg["alphalens_quantile_order"] = str(quantile_order)

        # 交由 analysis 包处理
        from ..analysis.alphalens_backend import run_factor_analysis

        run_factor_analysis(
            self,
            name,
            cfg=fa_cfg,
            interval=getattr(self, "interval", "1d"),
            display=True,
            long_short=long_short,
            group_neutral=group_neutral,
            by_group=by_group,
            plots=plots,
        )

    def show_signal_performance(self, signal: pl.DataFrame) -> None:
        """
        Perform performance analysis for prediction signals
        """
        # Get signal start and end times
        start: datetime = cast(datetime, signal["datetime"].min())
        end: datetime = cast(datetime, signal["datetime"].max())

        # Select range
        df: pl.DataFrame = query_by_time(self.result_df, start, end)

        # Extract feature
        signal_df: pd.DataFrame = signal.to_pandas()
        signal_df.set_index(["datetime", "vt_symbol"], inplace=True)
        signal_s: pd.Series = signal_df["signal"]

        # Extract price
        price_df: pd.DataFrame = df.select(["datetime", "vt_symbol", "close"]).to_pandas()
        price_df = price_df.pivot(index="datetime", columns="vt_symbol", values="close")

        # Merge data
        clean_data: pd.DataFrame = get_clean_factor_and_forward_returns(
            signal_s,
            price_df,
            max_loss=1.0,
            quantiles=10
        )

        # Perform analysis
        create_full_tear_sheet(clean_data)


def query_by_time(df: pl.DataFrame, start: datetime | str = "", end: datetime | str = "") -> pl.DataFrame:
    """
    Filter DataFrame based on time range
    """
    if start:
        start = to_datetime(start)
        df = df.filter(pl.col("datetime") >= start)

    if end:
        end = to_datetime(end)
        df = df.filter(pl.col("datetime") <= end)

    return df.sort(["datetime", "vt_symbol"])


def calculate_feature(args: tuple[pl.DataFrame, str, str | pl.expr.expr.Expr]) -> pl.Series:
    """
    Calculate feature by expression
    """
    start = time.time()

    df, name, expression = args

    try:
        if isinstance(expression, pl.expr.expr.Expr):
            result = calculate_by_polars(df, expression)["data"].alias(name)
        else:
            result = calculate_by_expression(df, expression)["data"].alias(name)
    except Exception as e:
        # 打印出错因子的名称与表达式，便于快速定位
        expr_text = str(expression)
        print(f"[FeatureError] name={name} | expr={expr_text} | err={e}")
        # 同时将表达式附加到异常信息中抛出
        raise RuntimeError(f"Error calculating feature '{name}' with expression: {expr_text}") from e

    end = time.time()
    print(f"Feature calculation {name} took: {end - start} seconds | {expression}")

    return result
