from __future__ import annotations

from typing import Optional, Tuple
import pandas as pd
import polars as pl


def collect_period_range(dataset) -> Tuple[object, object]:
    """从 dataset.data_periods 汇总整体时间范围；失败时回退 result_df 范围。"""
    from ...dataset.template import to_datetime  # 延迟导入
    starts, ends = [], []
    for period in getattr(dataset, "data_periods", {}).values():
        try:
            starts.append(to_datetime(period[0]))
            ends.append(to_datetime(period[1]))
        except Exception:
            continue
    if not starts or not ends:
        try:
            df = getattr(dataset, "result_df")
            return df["datetime"].min(), df["datetime"].max()
        except Exception:
            return None, None
    return min(starts), max(ends)


def build_factor_and_price(
    dataset,
    factor_name: str,
    start,
    end,
    *,
    keys_from: str = "infer",
) -> tuple[pd.Series, pd.DataFrame]:
    """按 keys_from 对齐后，从 result_df 抽取因子与价格。"""
    return build_factor_and_price2(
        dataset,
        factor_name,
        start,
        end,
        keys_from=keys_from,
        values_from="result",
    )


def build_factor_and_price2(
    dataset,
    factor_name: str,
    start,
    end,
    *,
    keys_from: str = "infer",
    values_from: str = "result",
) -> tuple[pd.Series, pd.DataFrame]:
    """从 dataset 提取 (factor_s, price_df)。

    - keys_from: infer/raw → 决定样本键来源
    - values_from: result/infer/raw → 决定因子值视图
    """
    from ...dataset.template import query_by_time  # 避免循环导入

    src_keys_df: pl.DataFrame = getattr(dataset, "infer_df") if keys_from == "infer" else getattr(dataset, "raw_df")
    keys = query_by_time(src_keys_df, start, end).select(["datetime", "vt_symbol"]).unique()

    df_res: pl.DataFrame = query_by_time(getattr(dataset, "result_df"), start, end).join(
        keys, on=["datetime", "vt_symbol"], how="inner"
    )

    if values_from == "infer":
        values_view = getattr(dataset, "infer_df")
    elif values_from == "raw":
        values_view = getattr(dataset, "raw_df")
    else:
        values_view = getattr(dataset, "result_df")

    values_view = query_by_time(values_view, start, end)
    factor_df = values_view.select(["datetime", "vt_symbol", factor_name]).join(
        keys, on=["datetime", "vt_symbol"], how="inner"
    )

    factor_s = (
        factor_df.to_pandas().set_index(["datetime", "vt_symbol"])[factor_name].sort_index()
    )
    price_df = (
        df_res.select(["datetime", "vt_symbol", "close"]).to_pandas()
        .pivot(index="datetime", columns="vt_symbol", values="close").sort_index()
    )
    return factor_s, price_df


def prepare_shared_views(
    dataset,
    start,
    end,
    *,
    keys_from: str = "infer",
):
    """预构建共享视图：样本键（pl.DataFrame）与价格矩阵（pd.DataFrame）。

    - 返回 (keys_df_pl, price_df_pd)
    - 后续逐因子仅需从对应视图取列并与 keys join 构造 factor_s
    """
    from ...dataset.template import query_by_time

    # keys（pl.DataFrame, 唯一）
    src_keys_df: pl.DataFrame = getattr(dataset, "infer_df") if keys_from == "infer" else getattr(dataset, "raw_df")
    keys = query_by_time(src_keys_df, start, end).select(["datetime", "vt_symbol"]).unique()

    # price_df（pd.DataFrame）
    df_res: pl.DataFrame = query_by_time(getattr(dataset, "result_df"), start, end).join(
        keys, on=["datetime", "vt_symbol"], how="inner"
    )
    price_df = (
        df_res.select(["datetime", "vt_symbol", "close"]).to_pandas()
        .pivot(index="datetime", columns="vt_symbol", values="close").sort_index()
    )
    return keys, price_df


def build_factor_series(
    dataset,
    factor_name: str,
    start,
    end,
    *,
    values_from: str,
    keys_df: pl.DataFrame | None,
):
    """使用预构建的 keys_df，从指定视图提取因子序列 pd.Series（MultiIndex）。"""
    from ...dataset.template import query_by_time

    if values_from == "infer":
        values_view = getattr(dataset, "infer_df")
    elif values_from == "raw":
        values_view = getattr(dataset, "raw_df")
    else:
        values_view = getattr(dataset, "result_df")

    values_view = query_by_time(values_view, start, end)
    if keys_df is None:
        # 若未提供共享 keys，则直接使用 values_view 的键
        factor_df = values_view.select(["datetime", "vt_symbol", factor_name])
    else:
        factor_df = values_view.select(["datetime", "vt_symbol", factor_name]).join(
            keys_df, on=["datetime", "vt_symbol"], how="inner"
        )

    factor_s = (
        factor_df.to_pandas().set_index(["datetime", "vt_symbol"])[factor_name].sort_index()
    )
    return factor_s
