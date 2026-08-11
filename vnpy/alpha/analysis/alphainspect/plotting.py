import math
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt

from . import _DATE_, _QUANTILE_


def get_row_col(count: int):
    """通过图总数，得到二维数量。用于确定合适的子图数量"""
    len_sqrt = math.sqrt(count)
    row, col = math.ceil(len_sqrt), math.floor(len_sqrt)
    if row * col < count:
        col += 1
    return row, col


def plot_heatmap(df: pd.DataFrame,
                 *,
                 title='Mean IC',
                 ax=None,
                 annot=True,
                 cbar=False,
                 cbar_label: str = None) -> None:
    """热力图"""
    # https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
    cbar_kws = {'label': cbar_label} if cbar_label else {}
    ax = sns.heatmap(df, annot=annot, cmap='RdYlGn_r', cbar=cbar,
                     cbar_kws=cbar_kws, annot_kws={"size": 7}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('')


def plot_heatmap_monthly_mean(df: pl.DataFrame, col: str,
                              *,
                              ax=None) -> None:
    """月度平均热力图"""
    df = df.select([_DATE_, col,
                    pl.col(_DATE_).dt.year().alias('year'),
                    pl.col(_DATE_).dt.month().alias('month')
                    ])
    df = df.group_by('year', 'month').agg(pl.mean(col))
    df_pd = df.to_pandas().set_index(['year', 'month'])

    plot_heatmap(df_pd[col].unstack(), title=f"{col},Monthly Mean", ax=ax)


def plot_heatmap_monthly_diff(df: pd.DataFrame, col='G9',
                              *, ax=None) -> None:
    """月度热力图。月底减月初差值

    Parameters
    ----------
    df
    col
    ax

    """
    df = df.select([_DATE_, col,
                    pl.col(_DATE_).dt.year().alias('year'),
                    pl.col(_DATE_).dt.month().alias('month')
                    ]).sort(_DATE_)
    df = df.group_by('year', 'month').agg(pl.last(col) - pl.first(col))
    df_pd = df.to_pandas().set_index(['year', 'month'])

    plot_heatmap(df_pd[col].unstack(), title=f"{col},Monthly Last-First", ax=ax)

    # out = pd.DataFrame(index=df.index)
    # out['year'] = out.index.year
    # out['month'] = out.index.month
    # out['first'] = df[col]
    # out['last'] = df[col]
    # out = out.groupby(by=['year', 'month']).agg({'first': 'first', 'last': 'last'})
    # # 累计收益由累乘改成了累加，这里算法也需要改动
    # # out['cum_ret'] = out['last'] / out['first'] - 1
    # out['cum_ret'] = out['last'] - out['first']
    # plot_heatmap(out['cum_ret'].unstack(), title=f"{col},Monthly Return", ax=ax)


def plot_ts(df: pl.DataFrame, col: str,
            *,
            axvlines=(), ax=None) -> Dict[str, float]:
    """时序图

    Examples
    --------
    >>> plot_ts(df_pd, 'RETURN_OO_1')

    """
    from scipy import stats

    df = df.select([_DATE_, col])

    df = df.select([
        _DATE_,
        pl.col(col),
        pl.col(col).rolling_mean(20).alias('sma_20'),
        pl.col(col).fill_nan(0).cum_sum().alias('cum_sum'),
    ])
    df_pd = df.to_pandas().replace([-np.inf, np.inf], np.nan).dropna(subset=col)
    s: pd.Series = df_pd[col]

    mean = s.mean()
    zscore = s.mean() / s.std(ddof=0)
    ratio = s.abs().gt(0.02).mean()
    t_stat, p_value = stats.ttest_1samp(s, 0)

    title = f"{col},mean={mean:0.4f},mean/std={zscore:0.4f}, t_stat={t_stat:0.4f}, p_value={p_value:0.4f}"
    logger.info(title)
    ax1 = df_pd.plot.line(x=_DATE_, y=[col, 'sma_20'], alpha=0.5, lw=1,
                          title=title,
                          ax=ax)
    ax2 = df_pd.plot.line(x=_DATE_, y=['cum_sum'], alpha=0.9, lw=1,
                          secondary_y='cum_sum', c='r',
                          ax=ax1)
    ax1.axhline(y=mean, c="r", ls="--", lw=1)
    ax.set_xlabel('')
    for v in axvlines:
        ax1.axvline(x=v, c="b", ls="--", lw=1)

    return {'ic_mean': mean, 'ic_ir': zscore, 'ratio': ratio, 't_stat': t_stat, 'p_value': p_value}




def plot_hist(df: pl.DataFrame, col: str,
              *,
              kde: bool = False,
              ax=None) -> Dict[str, float]:
    """直方图

    Parameters
    ----------
    df
    col
        列名
    kde: bool
        是否启用kde。启用后速度慢了非常多
    ax
        子图

    Examples
    --------
    >>> plot_hist(df, 'RETURN_OO_1')
    """
    a = df[col].to_pandas().replace([-np.inf, np.inf], np.nan).dropna()

    mean = a.mean()
    std = a.std(ddof=0)
    skew = a.skew()
    kurt = a.kurt()

    ax = sns.histplot(a,
                      bins=50, kde=kde,
                      stat="density", kde_kws=dict(cut=3),
                      alpha=.4, edgecolor=(1, 1, 1, .4),
                      ax=ax)

    ax.axvline(x=mean, c="r", ls="--", lw=1)
    ax.axvline(x=mean + std * 3, c="r", ls="--", lw=1)
    ax.axvline(x=mean - std * 3, c="r", ls="--", lw=1)
    title = f"{col},std={std:0.4f},skew={skew:0.4f},kurt={kurt:0.4f}"
    logger.info(title)
    ax.set_title(title)
    ax.set_xlabel('')

    return {'std': std, 'skew': skew, 'kurt': kurt, 'mean_': mean}


def plot_qq(df: pl.DataFrame, col: str,
            *,
            ax=None) -> None:
    """QQ图

    Examples
    --------
    >>> plot_qq(df, 'RETURN_OO_1')
    """
    from statsmodels import api as sm

    a = df[col].to_pandas().replace([-np.inf, np.inf], np.nan).dropna()

    sm.qqplot(a, fit=True, line='45', ax=ax)


def plot_quantile_bar_mean(df: pl.DataFrame, cols: Sequence[str],
                           *,
                           factor_quantile: str = _QUANTILE_,
                           title: str = 'Mean By Quantile',
                           ax=None):
    """分组平均值柱状图

    Examples
    --------
    >>> plot_quantile_bar_mean(df, ['RETURN_OO_1', 'RETURN_OO_2', 'RETURN_CC_1'])

    """
    df = df.group_by(factor_quantile).agg([pl.mean(y) for y in cols]).sort(factor_quantile)
    df_pd = df.to_pandas().set_index(factor_quantile)
    ax = df_pd.plot.bar(ax=ax)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.bar_label(ax.containers[0])


def plot_quantile_bar_count(df: pl.DataFrame, cols: Sequence[str],
                            *,
                            factor_quantile: str = _QUANTILE_,
                            title: str = 'Count By Quantile',
                            ax=None):
    """分组数量柱状图

    Examples
    --------
    >>> plot_quantile_bar_mean(df, ['RETURN_OO_1', 'RETURN_OO_2', 'RETURN_CC_1'])
    """
    df = df.group_by(factor_quantile).agg([pl.count(y) for y in cols]).sort(factor_quantile)
    df_pd = df.to_pandas().set_index(factor_quantile)
    ax = df_pd.plot.bar(ax=ax)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.bar_label(ax.containers[0])


def plot_quantile_box(df: pl.DataFrame, forward_returns: Sequence[str],
                      *,
                      factor_quantile: str = _QUANTILE_,
                      title: str = 'Box By Quantile',
                      ax=None):
    """分组收益

    Examples
    --------
    >>> plot_quantile_box(df, ['RETURN_OO_1', 'RETURN_OO_2', 'RETURN_CC_1'])

    """
    df = df.select(factor_quantile, *forward_returns)
    df_pd = df.to_pandas().set_index(factor_quantile)

    df_pd = df_pd.stack().reset_index()
    df_pd.columns = ['x', '', 'y']
    df_pd = df_pd.sort_values(by=['x', ''])
    ax = sns.boxplot(data=df_pd, x='x', y='y', hue='', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('')


def create_describe1_sheet(df: pl.DataFrame, cols: Sequence[str], factor_quantile: str = _QUANTILE_):
    """单因子分组统计

    Parameters
    ----------
    df
    cols
    factor_quantile

    Returns
    -------

    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # 一定要过滤null才能用
    df = df.filter(pl.col(factor_quantile).is_not_null())

    plot_quantile_bar_count(df, cols, factor_quantile=factor_quantile, ax=axes[0])
    plot_quantile_bar_mean(df, cols, factor_quantile=factor_quantile, ax=axes[1])
    plot_quantile_box(df, cols, factor_quantile=factor_quantile, ax=axes[2])

    fig.tight_layout()


def create_describe2_sheet(df: pl.DataFrame,
                           col: str,
                           factor_quantiles: Sequence[str]):
    """双因子分组统计。灵活使用分组方法能实现独立双重排序和条件双重排序

    例如，将两个因子划分成3*5，查看两因子组合效果

    Parameters
    ----------
    df
    col
        可以是收益率，也可以是其他需要统计的值
    factor_quantiles

    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 9))

    df = df.filter([pl.col(q).is_not_null() for q in factor_quantiles])

    df_mean = df.group_by(*factor_quantiles).agg(pl.mean(col)).sort(*factor_quantiles).to_pandas()
    df_std = df.group_by(*factor_quantiles).agg(pl.std(col, ddof=0)).sort(*factor_quantiles).to_pandas()
    df_count = df.group_by(*factor_quantiles).agg(pl.count(col)).sort(*factor_quantiles).to_pandas()

    df_mean = df_mean.set_index(factor_quantiles)[col].unstack()
    df_std = df_std.set_index(factor_quantiles)[col].unstack()
    df_count = df_count.set_index(factor_quantiles)[col].unstack()

    plot_heatmap(df_mean, title='Mean', ax=axes[0])
    plot_heatmap(df_std, title='Std', ax=axes[1])
    plot_heatmap(df_count, title='Count', ax=axes[2])

    fig.tight_layout()


def plot_ic_decay_line(decay_data: dict,
                       summary: pd.DataFrame = None,
                       *,
                       ax=None,
                       title: str = 'IC Decay Curve') -> None:
    """绘制IC衰减曲线（线图）

    Parameters
    ----------
    decay_data : dict
        _get_ic_decay_data() 返回的数据字典
    summary : pd.DataFrame, optional
        calc_ic_decay_summary的输出结果，用于显示统计信息
    ax : matplotlib.axes, optional
    title : str

    Examples
    --------
    >>> decay_data = _get_ic_decay_data(df_ic_decay)
    >>> plot_ic_decay_line(decay_data, summary)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    periods = decay_data['periods']
    ic_mean = decay_data['ic_mean']
    ic_std = decay_data['ic_std']

    # 绘制IC均值线和标准差区域
    ax.plot(periods, ic_mean, 'b-o', linewidth=2, markersize=6, label='Mean IC')
    ax.fill_between(periods, ic_mean - ic_std, ic_mean + ic_std,
                    alpha=0.3, color='blue', label='±1 Std')

    # 添加零线和标签
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Holding Periods', fontsize=11)
    ax.set_ylabel('IC', fontsize=11)

    # 构建标题（包含半衰期信息）
    if summary is not None and 'half_life' in summary.attrs:
        hl = summary.attrs['half_life']
        if not np.isnan(hl):
            title = f"{title}\nHalf-life: {hl:.1f} periods"

    ax.set_title(title, fontsize=12)
    ax.set_xticks(periods)
    ax.set_xticklabels([f'{p}' for p in periods])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)


def plot_ic_decay_heatmap(decay_data: dict,
                          *,
                          ax=None,
                          title: str = 'IC Decay Heatmap') -> None:
    """绘制IC衰减热力图（时间 x 延迟期）

    Parameters
    ----------
    decay_data : dict
        _get_ic_decay_data() 返回的数据字典
    ax : matplotlib.axes, optional
    title : str

    Examples
    --------
    >>> decay_data = _get_ic_decay_data(df_ic_decay)
    >>> plot_ic_decay_heatmap(decay_data)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    df_monthly = decay_data['df_monthly']

    plot_heatmap(df_monthly.T, title=title, ax=ax, annot=False, cbar=True, cbar_label='IC')

    ax.set_ylabel('Holding Period', fontsize=11)

    # 调整y轴标签
    y_labels = [col.split("_")[1] for col in decay_data['delay_cols']]
    ax.set_yticklabels(y_labels, rotation=0)


def plot_ic_decay_bar(decay_data: dict,
                      *,
                      ax=None,
                      title: str = 'IC by Holding Period') -> None:
    """绘制各延迟期的IC均值柱状图

    Parameters
    ----------
    decay_data : dict
        _get_ic_decay_data() 返回的数据字典
    ax : matplotlib.axes, optional
    title : str

    Examples
    --------
    >>> decay_data = _get_ic_decay_data(df_ic_decay)
    >>> plot_ic_decay_bar(decay_data)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    periods = decay_data['periods']
    ic_mean = decay_data['ic_mean']
    half_width = 0.4

    colors = ['green' if v > 0 else 'red' for v in ic_mean]
    bars = ax.bar(periods, ic_mean, color=colors, alpha=0.7, edgecolor='black')

    for bar, val in zip(bars, ic_mean):
        height = bar.get_height()
        ax.text(bar.get_x() + half_width, height, f'{val:.3f}',
                ha='center', va='bottom' if val > 0 else 'top', fontsize=9)

    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Holding Period', fontsize=11)
    ax.set_ylabel('Mean IC', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(periods)
    ax.grid(True, alpha=0.3, axis='y')


def plot_summary_table(summary: pd.DataFrame, *, ax=None, title: str = 'Summary Statistics') -> None:
    """绘制DataFrame表格（通用函数）

    Parameters
    ----------
    summary : pd.DataFrame
        汇总统计表
    ax : matplotlib.axes, optional
    title : str
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.axis('off')
    table_data = summary.round(4).values.tolist()
    table = ax.table(
        cellText=table_data,
        rowLabels=summary.index.tolist(),
        colLabels=summary.columns.tolist(),
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title(title, fontsize=12, pad=20)
