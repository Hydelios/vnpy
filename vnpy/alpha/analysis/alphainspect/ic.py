import itertools
from typing import Sequence, Literal

import numpy as np
import polars as pl
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from polars import Expr
from scipy import stats

from . import _DATE_
from .calc import calc_corr, calc_mean, calc_ir, calc_std
from .plotting import (plot_heatmap, get_row_col, plot_hist, plot_heatmap_monthly_mean, plot_qq, plot_ts,
                       plot_ic_decay_line, plot_ic_decay_heatmap, plot_ic_decay_bar, plot_summary_table)
from .utils import select_by_suffix, index_split_unstack


def rank_ic(a: str, b: str) -> Expr:
    """RankIC"""
    return pl.corr(a, b, method='spearman', ddof=0, propagate_nans=False)

def pearson_ic(a: str, b: str) -> Expr:
    """IC"""
    return pl.corr(a, b, method='pearson', ddof=0, propagate_nans=False)


def mutual_info(a: str, b: str) -> Expr:
    """互信息"""
    from sklearn.feature_selection import mutual_info_regression

    def mutual_info_func(xx) -> float:
        yx = np.vstack(xx).T
        mask = np.any(np.isnan(yx), axis=1)
        yx_ = yx[~mask, :]
        if len(yx_) <= 3:
            return np.nan
        mi = mutual_info_regression(yx_[:, 0].reshape(-1, 1), yx_[:, 1], n_neighbors=3)
        return float(mi[0])

    return pl.map_groups([a, b], lambda xx: mutual_info_func(xx))


def w_corr(a: str, b: str, w: str) -> pl.Expr:
    def _w_corr(xx):
        x, y, weights = xx
        cov_matrix = np.cov(x, y, aweights=weights)
        weighted_corr = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])
        return weighted_corr

    return pl.map_groups([a, b, w], lambda xx: _w_corr(xx))


def calc_ic(df: pl.DataFrame, factors: Sequence[str], forward_returns: Sequence[str],
            method: Literal['rank_ic', 'ic', 'mutual_info'] = 'rank_ic') -> pl.DataFrame:
    """多因子多收益的IC矩阵。方便部分用户统计大量因子信息"""
    if method == 'mutual_info':
        func = mutual_info
    elif method == 'ic':
        func = pearson_ic
    else:
        func = rank_ic
    return df.group_by(_DATE_).agg(
        [func(x, y).alias(f'{x}__{y}') for x, y in itertools.product(factors, forward_returns)]
    ).sort(_DATE_).fill_nan(None)


def calc_ic_stats(df_ic: pl.DataFrame, output_format: Literal['series', 'dict'] = 'series') -> dict:
    """计算IC序列的通用统计指标

    集中计算所有常用统计量，避免重复代码。
    优先使用polars进行计算以获得更好的性能，t-test使用scipy。

    Parameters
    ----------
    df_ic : pl.DataFrame
        IC序列数据框（包含date列和其他IC列）
    output_format : str
        'series' - 返回pd.Series格式（默认，内存友好）
        'dict' - 返回字典格式，包含原始DataFrame供后续使用

    Returns
    -------
    dict: 统计结果字典
        当 output_format='series':
            - ic_mean, ic_std, ic_ir, win_rate, t_stat, p_value: pd.Series
        当 output_format='dict':
            - 上述统计量 + df_ic: pl.DataFrame（原始数据，用于后续计算）
    """
    # 使用polars计算统计量（更高效）
    ic_mean_pl = calc_mean(df_ic)
    ic_std_pl = calc_std(df_ic)
    ic_ir_pl = calc_ir(df_ic)
    win_rate_pl = calc_win_rate(df_ic)

    # 转换为pandas（只做一次）
    df_pd = df_ic.to_pandas().set_index(_DATE_)
    ic_data = df_pd.reset_index(drop=True)

    # t-test 使用scipy
    t_stat, p_value = stats.ttest_1samp(ic_data, 0, nan_policy='omit')

    # 使用squeeze()将单行DataFrame转为Series
    ic_mean = ic_mean_pl.to_pandas().squeeze()
    ic_std = ic_std_pl.to_pandas().squeeze()
    ic_ir = ic_ir_pl.to_pandas().squeeze()
    win_rate = win_rate_pl.to_pandas().squeeze()

    result = {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ic_ir': ic_ir,
        'win_rate': win_rate,
        't_stat': pd.Series(t_stat, index=ic_data.columns),
        'p_value': pd.Series(p_value, index=ic_data.columns),
    }

    # 仅当需要后续处理时才保留原始数据引用
    if output_format == 'dict':
        result['df_ic'] = df_ic  # 保留polars DataFrame引用，不复制

    return result


def calc_win_rate(df: pl.DataFrame) -> pl.DataFrame:
    """计算IC>0的胜率"""
    return df.select(pl.exclude(_DATE_).gt(0).mean())


def _extract_period(col_name: str) -> int | None:
    """从列名中提取period数字

    支持格式: 'RETURN_OO_01', 'factor__RETURN_OO_01', 'delay_01'
    """
    try:
        period_str = col_name.rsplit('_', 1)[-1]
        return int(period_str)
    except (ValueError, IndexError):
        return None


def _sort_series_by_period(series: pd.Series) -> pd.Series:
    """按period对Series进行排序"""
    periods = [_extract_period(x) for x in series.index]
    valid_mask = [p is not None for p in periods]

    if not any(valid_mask):
        return series

    sorted_pairs = sorted(
        [(p, idx) for p, idx in zip(periods, series.index) if p is not None],
        key=lambda x: x[0]
    )
    sorted_index = [idx for _, idx in sorted_pairs]
    return series[sorted_index]


def _format_ic_stats(stats_dict: dict) -> dict:
    """将统计字典转换为标准输出格式（使用index_split_unstack）"""
    return {
        'df_ic': index_split_unstack(stats_dict['ic_mean']),
        'df_ir': index_split_unstack(stats_dict['ic_ir']),
        'df_win_rate': index_split_unstack(stats_dict['win_rate']),
        'df_tstat': index_split_unstack(stats_dict['t_stat']),
        'df_pvalue': index_split_unstack(stats_dict['p_value']),
    }


def conclude_ic_frame(df: pl.DataFrame,
                      factors: Sequence[str],
                      forward_returns: Sequence[str],
                      *,
                      method: Literal['rank_ic', 'ic', 'mutual_info'] = 'rank_ic',
                      include_decay: bool = False) -> dict:
    """统计多因子——多收益的IC、IR、胜率、t-stat、p-value 统计信息

    Parameters
    ----------
    df : pl.DataFrame
        因子数据
    factors : Sequence[str]
        因子列名列表
    forward_returns : Sequence[str]
        远期收益率列名列表
    method : str
        'rank_ic', 'ic' 或 'mutual_info'
    include_decay : bool
        是否包含IC衰减指标（半衰期），仅在单因子时有效

    Returns
    -------
    dict: 包含以下键的字典
        - df_ic: IC均值矩阵
        - df_ir: IR矩阵
        - df_win_rate: 胜率矩阵
        - df_tstat: t统计量矩阵
        - df_pvalue: p值矩阵
        - half_life: 半衰期（条件不满足时为np.nan）
    """
    df_ic = calc_ic(df, factors, forward_returns, method)
    stats = calc_ic_stats(df_ic)
    result = _format_ic_stats(stats)

    # 始终初始化 half_life，保持返回结构一致性
    result['half_life'] = np.nan

    if include_decay:
        if len(factors) == 1 and method in ('rank_ic', 'ic'):
            # 使用工具函数直接排序并计算半衰期
            sorted_ic_mean = _sort_series_by_period(stats['ic_mean'])
            if len(sorted_ic_mean) >= 2:
                result['half_life'] = _calc_half_life(sorted_ic_mean.to_numpy())
        elif len(factors) > 1:
            logger.warning("include_decay=True is ignored when multiple factors are provided. "
                           "Half-life calculation only supports single factor.")
        elif method == 'mutual_info':
            logger.warning("include_decay=True is ignored for method='mutual_info'. "
                           "Half-life calculation is not supported for mutual information.")

    return result


def create_ic1_sheet(df: pl.DataFrame, factor: str, forward_returns: Sequence[str],
                     *,
                     axvlines=(),
                     method: Literal['rank_ic','ic', 'mutual_info'] = 'rank_ic') -> pl.DataFrame:
    """生成IC图表系列。单因子多收益率"""
    df = calc_ic(df, [factor], forward_returns, method)

    for col in df.columns:
        if col == _DATE_:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        plot_ts(df, col, axvlines=axvlines, ax=axes[0, 0])
        plot_hist(df, col, ax=axes[0, 1])
        plot_qq(df, col, ax=axes[1, 0])
        plot_heatmap_monthly_mean(df, col, ax=axes[1, 1])

        fig.tight_layout()

    return df


def create_ic2_sheet(df: pl.DataFrame, factors: Sequence[str], forward_returns: Sequence[str],
                     *,
                     axvlines=(), ) -> pl.DataFrame:
    """生成IC图表。多因子多收益率。

    用于分析相似因子在不同持有期下的IC信息
    """
    df_ic = calc_ic(df, factors, forward_returns)

    # 使用统一的统计计算函数
    stats = calc_ic_stats(df_ic)

    # 转换为标准格式
    df_ic_formatted = index_split_unstack(stats['ic_mean'])
    df_ir = index_split_unstack(stats['ic_ir'])
    df_win_rate = index_split_unstack(stats['win_rate'])
    df_tstat = index_split_unstack(stats['t_stat'])
    df_pvalue = index_split_unstack(stats['p_value'])

    logger.info('Mean IC: {} \n{}', '=' * 60, df_ic_formatted)
    logger.info('IC_IR: {} \n{}', '=' * 60, df_ir)
    logger.info('Win Rate (IC>0): {} \n{}', '=' * 60, df_win_rate)
    logger.info('t-stat: {} \n{}', '=' * 60, df_tstat)
    logger.info('p-value: {} \n{}', '=' * 60, df_pvalue)

    # 画ic、ir的热力图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plot_heatmap(df_ic_formatted, title='Mean IC', ax=axes[0, 0])
    plot_heatmap(df_ir, title='IR', ax=axes[0, 1])
    fig.tight_layout()

    # IC之间相关性，可用于检查多重共线性
    corrs = {}
    for forward in forward_returns:
        corrs[forward] = calc_corr(select_by_suffix(df_ic, f'__{forward}'))

    row, col = get_row_col(len(corrs))
    fig, axes = plt.subplots(row, col, figsize=(12, 9), squeeze=False)
    axes = axes.flatten()
    for i, (k, v) in enumerate(corrs.items()):
        plot_heatmap(v, title=f'{k} IC Corr', ax=axes[i])
    fig.tight_layout()

    # 画ic时序图
    fig, axes = plt.subplots(len(factors), len(forward_returns), figsize=(12, 9), squeeze=False)
    axes = axes.flatten()
    logger.info('IC TimeSeries: {}', '=' * 60)
    for i, (x, y) in enumerate(itertools.product(factors, forward_returns)):
        plot_ts(df_ic, f'{x}__{y}', axvlines=axvlines, ax=axes[i])
    fig.tight_layout()

    return df_ic


def calc_ic_decay(df: pl.DataFrame,
                  factor: str,
                  forward_returns: Sequence[str],
                  *,
                  method: Literal['rank_ic', 'ic'] = 'rank_ic') -> pl.DataFrame:
    """计算单因子IC衰减序列

    计算因子与未来各期收益率的IC，观察因子预测能力随时间的衰减

    Parameters
    ----------
    df : pl.DataFrame
        因子数据，必须包含date、asset列
    factor : str
        因子列名
    forward_returns : Sequence[str]
        远期收益率列名列表，如 ['RETURN_OO_01', 'RETURN_OO_02', 'RETURN_OO_05']
    method : str
        'rank_ic' 或 'ic'

    Returns
    -------
    pl.DataFrame
        包含date和delay_01, delay_02, ... delay_n列的IC序列

    Examples
    --------
    >>> forward_returns = ['RETURN_OO_01', 'RETURN_OO_02', 'RETURN_OO_05', 'RETURN_OO_10']
    >>> df_ic_decay = calc_ic_decay(df, 'SMA_010', forward_returns)
    """
    available_cols = set(df.columns)
    valid_returns = [r for r in forward_returns if r in available_cols]

    if len(valid_returns) == 0:
        raise ValueError(f"No valid return columns found from: {forward_returns}")

    if len(valid_returns) < len(forward_returns):
        missing = set(forward_returns) - set(valid_returns)
        logger.warning(f"Missing columns: {missing}")

    df_ic = calc_ic(df, [factor], valid_returns, method=method)

    rename_map = {}
    for ret_col in valid_returns:
        old_name = f'{factor}__{ret_col}'
        if old_name in df_ic.columns:
            period = _extract_period(ret_col)
            if period is not None:
                new_name = f'delay_{period:02d}'
                rename_map[old_name] = new_name

    df_ic = df_ic.rename(rename_map)

    delay_cols = [c for c in df_ic.columns if c != _DATE_]
    sorted_cols = sorted(delay_cols, key=lambda x: _extract_period(x) or 0)
    df_ic = df_ic.select([_DATE_] + sorted_cols)

    return df_ic


def _calc_half_life(ic_values: np.ndarray) -> float:
    """计算IC半衰期

    假设IC呈指数衰减: IC(t) = IC(0) * exp(-lambda * t)
    半衰期 = ln(2) / lambda

    Parameters
    ----------
    ic_values : np.ndarray
        各延迟期的IC均值

    Returns
    -------
    float
        半衰期（期数），如果无法计算返回np.nan
    """
    if len(ic_values) < 2 or abs(ic_values[0]) < 0.001:
        return np.nan

    abs_ic = np.abs(ic_values)
    threshold = abs_ic[0] * 0.1
    valid_idx = np.where(abs_ic > threshold)[0]

    if len(valid_idx) < 2:
        return np.nan

    x = valid_idx.astype(float)
    y = np.log(abs_ic[valid_idx])

    try:
        coeffs = np.polyfit(x, y, 1)
        lambda_val = -coeffs[0]

        if lambda_val <= 0:
            return np.nan

        half_life = np.log(2) / lambda_val
        return half_life
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def calc_ic_decay_summary(df_ic_decay: pl.DataFrame) -> dict:
    """计算IC衰减的汇总统计（统一入口，包含可视化数据）

    Parameters
    ----------
    df_ic_decay : pl.DataFrame
        calc_ic_decay的输出结果

    Returns
    -------
    dict: 包含以下键的字典
        - summary: pd.DataFrame, 统计汇总表
        - half_life: float, 半衰期
        - periods: list, 延迟期数列表（用于可视化）
        - ic_mean: np.ndarray, 各期IC均值（用于可视化）
        - ic_std: np.ndarray, 各期IC标准差（用于可视化）
        - df_monthly: pd.DataFrame, 按月聚合数据（用于热力图）
        - delay_cols: list, 延迟列名
    """
    # 复用通用的统计计算函数
    stats = calc_ic_stats(df_ic_decay)

    # 按需计算numpy数组（避免重复存储）
    df_pd = df_ic_decay.to_pandas().set_index(_DATE_)
    ic_mean_np = df_pd.mean().to_numpy()
    ic_std_np = df_pd.std().to_numpy()

    delay_cols = df_pd.columns.tolist()
    periods = [_extract_period(col) for col in delay_cols]
    periods = [p for p in periods if p is not None]

    df_pd.index = pd.to_datetime(df_pd.index)
    df_monthly = df_pd.resample('ME').mean()

    # 确保按period排序后计算半衰期
    sorted_ic_mean = _sort_series_by_period(stats['ic_mean'])
    half_life = _calc_half_life(sorted_ic_mean.to_numpy())

    summary = pd.DataFrame({
        'IC_Mean': stats['ic_mean'].values,
        'IC_Std': stats['ic_std'].values,
        'IC_IR': stats['ic_ir'].values,
        'Win_Rate': stats['win_rate'].values,
        'T_Stat': stats['t_stat'].values,
        'P_Value': stats['p_value'].values,
    }, index=stats['ic_mean'].index)

    return {
        'summary': summary,
        'half_life': half_life,
        'periods': periods,
        'ic_mean': ic_mean_np,
        'ic_std': ic_std_np,
        'df_monthly': df_monthly,
        'delay_cols': delay_cols,
    }


def create_ic_decay_sheet(df: pl.DataFrame,
                          factor: str,
                          forward_returns: Sequence[str],
                          *,
                          method: Literal['rank_ic', 'ic'] = 'rank_ic',
                          figsize=(14, 10)) -> tuple[pl.DataFrame, dict]:
    """生成完整的IC衰减分析报告

    Parameters
    ----------
    df : pl.DataFrame
        因子数据
    factor : str
        因子列名
    forward_returns : Sequence[str]
        远期收益率列名列表
    method : str
        'rank_ic' 或 'ic'
    figsize : tuple

    Returns
    -------
    tuple[pl.DataFrame, dict]
        (IC衰减序列, 汇总统计字典)

    Examples
    --------
    >>> forward_returns = ['RETURN_OO_01', 'RETURN_OO_02', 'RETURN_OO_05', 'RETURN_OO_10']
    >>> df_ic_decay, result = create_ic_decay_sheet(df, 'SMA_010', forward_returns)
    """
    logger.info(f"Calculating IC decay for factor: {factor}")
    logger.info(f"Forward returns: {forward_returns}")

    df_ic_decay = calc_ic_decay(df, factor, forward_returns, method=method)
    result = calc_ic_decay_summary(df_ic_decay)

    hl = result['half_life']
    summary_df = result['summary']
    logger.info(f"IC Decay Summary for {factor}:")
    if not np.isnan(hl):
        logger.info(f"Half-life: {hl:.2f} periods")
    else:
        logger.info("Half-life: N/A (insufficient data or no decay pattern)")
    logger.info(f"\n{summary_df}")

    # 构建可视化数据字典
    decay_data = {
        'periods': result['periods'],
        'ic_mean': result['ic_mean'],
        'ic_std': result['ic_std'],
        'df_monthly': result['df_monthly'],
        'delay_cols': result['delay_cols'],
    }

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    plot_ic_decay_line(decay_data, summary_df, ax=axes[0, 0],
                       title=f'IC Decay Curve - {factor}')
    plot_ic_decay_heatmap(decay_data, ax=axes[0, 1],
                          title='IC Decay Heatmap (Monthly)')
    plot_ic_decay_bar(decay_data, ax=axes[1, 0],
                      title='Mean IC by Holding Period')
    plot_summary_table(summary_df, ax=axes[1, 1],
                       title='IC Decay Summary Statistics')

    plt.tight_layout()

    return df_ic_decay, result
