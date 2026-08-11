"""AlphaInspect 的稳定调用门面。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd
import polars as pl
from matplotlib.figure import Figure

from .alphainspect import _ASSET_, _DATE_, _QUANTILE_
from .alphainspect.ic import calc_ic, calc_ic_stats
from .alphainspect.reports import create_3x2_sheet
from .alphainspect.utils import with_factor_quantile


@dataclass(slots=True)
class AlphaInspectSheetResult:
    """AlphaInspect 3x2 图表及其底层计算结果。"""

    figure: Figure
    quantile_frame: pl.DataFrame
    ic_stats: dict[str, Any]
    hist_stats: dict[str, Any]
    cumulative_return: pl.DataFrame
    average_return: pl.DataFrame
    return_std: pl.DataFrame


@dataclass(slots=True)
class AlphaInspectICReportResult:
    """IC 全样本汇总、日度序列和逐年统计结果。"""

    summary: pd.DataFrame
    daily: pl.DataFrame
    annual: pd.DataFrame


@dataclass(slots=True)
class AlphaInspectInputResult:
    """标准化信号及其 AlphaInspect 分析输入。"""

    signal: pl.DataFrame
    signal_summary: pl.DataFrame
    analysis_frame: pl.DataFrame
    coverage: pl.DataFrame
    forward_returns: tuple[str, ...]
    start: Any
    end: Any
    symbols: tuple[str, ...]


def _load_factor_frame(data: Any) -> pl.DataFrame:
    if isinstance(data, pl.DataFrame):
        return data.clone()
    if isinstance(data, pd.DataFrame):
        return pl.from_pandas(data)
    if isinstance(data, (str, Path)):
        path = Path(data)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pl.read_parquet(path)
        if suffix == ".csv":
            return pl.read_csv(path, try_parse_dates=True)
        if suffix in {".pkl", ".pickle"}:
            return pl.from_pandas(pd.read_pickle(path))
        raise ValueError(f"不支持的信号格式: {suffix}")
    raise TypeError(f"不支持的信号数据类型: {type(data)!r}")


def prepare_alphainspect_input(
    lab: Any,
    signal: pl.DataFrame | pd.DataFrame | str | Path,
    *,
    factor: str = "signal",
    horizons: Sequence[int] = (1, 3, 5, 10, 20),
    start_date: str | None = None,
    end_date: str | None = None,
    interval: Any = None,
    extended_days: int = 400,
) -> AlphaInspectInputResult:
    """加载并标准化信号，构造以 T+1 open 为起点的远期收益输入。"""

    periods = tuple(int(horizon) for horizon in horizons)
    if not periods or any(horizon <= 0 for horizon in periods):
        raise ValueError("horizons 必须包含正整数")
    if len(set(periods)) != len(periods):
        raise ValueError("horizons 不能重复")
    if extended_days < max(periods) + 1:
        raise ValueError("extended_days 必须覆盖最大 horizon 的退出价格")

    frame = _load_factor_frame(signal)
    required = {"datetime", "vt_symbol", factor}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"信号缺少必要列: {missing}")

    frame = (
        frame.select("datetime", "vt_symbol", factor)
        .with_columns(
            pl.col("datetime").cast(pl.Datetime, strict=False),
            pl.col("vt_symbol").cast(pl.Utf8),
            pl.col(factor).cast(pl.Float64, strict=False),
        )
        .drop_nulls(["datetime", "vt_symbol", factor])
        .filter(pl.col(factor).is_finite())
        .unique(["datetime", "vt_symbol"], keep="last")
        .sort(["datetime", "vt_symbol"])
    )
    if start_date is not None:
        frame = frame.filter(pl.col("datetime") >= pd.Timestamp(start_date).to_pydatetime())
    if end_date is not None:
        frame = frame.filter(pl.col("datetime") <= pd.Timestamp(end_date).to_pydatetime())
    if frame.is_empty():
        raise ValueError("日期过滤后信号为空")

    signal_summary = frame.select(
        pl.len().alias("rows"),
        pl.col("datetime").min().alias("start"),
        pl.col("datetime").max().alias("end"),
        pl.col("datetime").n_unique().alias("dates"),
        pl.col("vt_symbol").n_unique().alias("symbols"),
        pl.col(factor).mean().alias(f"{factor}_mean"),
        pl.col(factor).std().alias(f"{factor}_std"),
    )

    symbols = frame["vt_symbol"].unique().sort().to_list()
    load_kwargs = {
        "vt_symbols": symbols,
        "start": frame["datetime"].min(),
        "end": frame["datetime"].max(),
        "extended_days": extended_days,
    }
    if interval is not None:
        load_kwargs["interval"] = interval
    price_panel = lab.load_bar_df(**load_kwargs)
    if price_panel is None or price_panel.is_empty():
        raise ValueError("未从 AlphaLab 读取到日线价格")
    price_required = {"datetime", "vt_symbol", "open"}
    price_missing = sorted(price_required - set(price_panel.columns))
    if price_missing:
        raise ValueError(f"价格数据缺少必要列: {price_missing}")

    forward_returns = tuple(f"FWD_RET_OO_{horizon}" for horizon in periods)
    expressions = []
    for horizon, column in zip(periods, forward_returns):
        entry_open = pl.col("open").shift(-1).over("vt_symbol")
        exit_open = pl.col("open").shift(-(horizon + 1)).over("vt_symbol")
        expressions.append(
            ((exit_open / entry_open).pow(1.0 / horizon) - 1.0).alias(column)
        )
    price_with_returns = (
        price_panel.sort(["vt_symbol", "datetime"])
        .with_columns(expressions)
        .select("datetime", "vt_symbol", *forward_returns)
    )
    analysis_frame = (
        frame.join(price_with_returns, on=["datetime", "vt_symbol"], how="left")
        .rename({"datetime": _DATE_, "vt_symbol": _ASSET_})
        .with_columns(
            pl.col(factor).cast(pl.Float64),
            *[
                pl.col(column).cast(pl.Float64).fill_nan(None)
                for column in forward_returns
            ],
        )
        .sort([_DATE_, _ASSET_])
    )
    coverage = analysis_frame.select(
        pl.len().alias("signal_rows"),
        *[
            pl.col(column).is_not_null().sum().alias(f"{column}_rows")
            for column in forward_returns
        ],
    )
    return AlphaInspectInputResult(
        signal=frame,
        signal_summary=signal_summary,
        analysis_frame=analysis_frame,
        coverage=coverage,
        forward_returns=forward_returns,
        start=frame["datetime"].min(),
        end=frame["datetime"].max(),
        symbols=tuple(symbols),
    )


def create_ic_report(
    frame: pl.DataFrame,
    factor: str,
    *,
    forward_returns: Sequence[str],
    method: Literal["rank_ic", "ic", "mutual_info"] = "rank_ic",
    output_dir: str | Path | None = None,
    filename_prefix: str | None = None,
) -> AlphaInspectICReportResult:
    """一次生成单因子的 IC 汇总、日度序列和逐年统计。

    ``output_dir`` 非空时，同时保存 ``*_summary.csv``、``*_daily.parquet``
    和 ``*_annual.csv``。文件名前缀默认跟随计算方法。
    """

    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame 必须是 polars.DataFrame")
    returns = tuple(forward_returns)
    if not returns:
        raise ValueError("forward_returns 不能为空")

    required = {_DATE_, factor, *returns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"AlphaInspect IC 输入缺少必要列: {missing}")

    daily = calc_ic(
        frame,
        factors=[factor],
        forward_returns=returns,
        method=method,
    )
    stats = calc_ic_stats(daily)
    metric_names = {
        "rank_ic": ("RankIC", "RankICIR"),
        "ic": ("IC", "ICIR"),
        "mutual_info": ("mutual_info", "mutual_info_ir"),
    }
    mean_name, ir_name = metric_names[method]
    summary = pd.concat(
        {
            mean_name: stats["ic_mean"],
            ir_name: stats["ic_ir"],
            "win_rate": stats["win_rate"],
            "t_stat": stats["t_stat"],
            "p_value": stats["p_value"],
        },
        axis=1,
    )
    prefix = f"{factor}__"
    summary.index = summary.index.map(
        lambda column: column.removeprefix(prefix)
    )
    summary.index.name = "forward_return"

    daily_pd = daily.to_pandas().set_index(_DATE_)
    annual = daily_pd.groupby(daily_pd.index.year).agg(["mean", "std", "count"])
    for column in daily_pd.columns:
        annual[(column, "ir")] = (
            annual[(column, "mean")] / annual[(column, "std")]
        )
    annual = annual.sort_index(axis=1)
    annual.index.name = "year"

    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        file_prefix = filename_prefix or method
        summary.to_csv(output / f"{file_prefix}_summary.csv")
        daily.write_parquet(output / f"{file_prefix}_daily.parquet")
        annual.to_csv(output / f"{file_prefix}_annual.csv")

    return AlphaInspectICReportResult(
        summary=summary,
        daily=daily,
        annual=annual,
    )


def create_factor_sheet(
    frame: pl.DataFrame,
    factor: str,
    *,
    forward_return: str = "FWD_RET_OO_1",
    quantiles: int = 9,
    turnover_periods: Sequence[int] = (1, 3, 5, 10, 20),
    title: str | None = None,
    output_path: str | Path | None = None,
    dpi: int = 150,
    axvlines: Sequence[str] = (),
) -> AlphaInspectSheetResult:
    """生成包含分层收益、IC 时序和换手率的 AlphaInspect 3x2 图。

    输入 ``frame`` 应使用 AlphaInspect 标准列名 ``date`` 和 ``asset``，并且
    已包含待分析因子及未来收益。函数负责截面分组、绘图和可选保存，但不会
    调用 ``plt.show()``，由 notebook 或上层应用决定如何展示。
    """
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame 必须是 polars.DataFrame")

    required = {_DATE_, _ASSET_, factor, forward_return}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"AlphaInspect 输入缺少必要列: {missing}")
    if quantiles < 2:
        raise ValueError("quantiles 必须大于等于 2")

    periods = tuple(int(period) for period in turnover_periods)
    if not periods or any(period <= 0 for period in periods):
        raise ValueError("turnover_periods 必须包含正整数")

    prepared = (
        frame.with_columns(
            pl.col(factor).cast(pl.Float64, strict=False).fill_nan(None),
            pl.col(forward_return).cast(pl.Float64, strict=False).fill_nan(None),
        )
        .sort([_DATE_, _ASSET_])
    )
    quantile_frame = with_factor_quantile(
        prepared,
        factor=factor,
        quantiles=quantiles,
        by=[_DATE_],
        factor_quantile=_QUANTILE_,
    )

    figure, ic_stats, hist_stats, cumulative_return, average_return, return_std = (
        create_3x2_sheet(
            quantile_frame,
            factor=factor,
            fwd_ret_1=forward_return,
            periods=periods,
            axvlines=axvlines,
        )
    )

    if title is not None:
        figure.suptitle(title, fontsize=14, y=1.01)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")

    return AlphaInspectSheetResult(
        figure=figure,
        quantile_frame=quantile_frame,
        ic_stats=ic_stats,
        hist_stats=hist_stats,
        cumulative_return=cumulative_return,
        average_return=average_return,
        return_std=return_std,
    )


__all__ = [
    "AlphaInspectICReportResult",
    "AlphaInspectInputResult",
    "AlphaInspectSheetResult",
    "create_factor_sheet",
    "create_ic_report",
    "prepare_alphainspect_input",
]
