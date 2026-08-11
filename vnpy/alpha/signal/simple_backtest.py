#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple signal-to-return backtests.

This module intentionally avoids formal execution rules such as n_drop,
hold_thresh, commissions, order sizing, and limit checks. It is meant for
diagnosing signal predictiveness under clear O2O/C2C timing.

Rolling-sleeve interpretation:
    horizon=5 means the portfolio is split into five fixed-capital sleeves.
    Each day opens one new 1/5 sleeve from the latest signal, and each sleeve
    holds for five trading days. During ramp-up or signal gaps, unused capital
    remains cash. A Top300 sleeve gives each selected stock 1/5/300 portfolio
    weight. The rolling daily return is computed from real one-period O2O/C2C
    stock returns; it is not the five-day holding return divided by five.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:  # pragma: no cover - polars is optional for callers.
    pl = None


PriceMode = Literal["o2o", "c2c"]
SelectionMode = Literal["top_k", "top_frac"]


@dataclass(frozen=True)
class SimpleBacktestConfig:
    """Configuration for simple signal O2O/C2C backtests.

    ``horizons`` controls both the signal-date holding-period return and the
    rolling-sleeve portfolio. For horizon=5, rolling-sleeve mode opens one
    1/5-capital sleeve per day and holds each sleeve for five trading days.
    """

    price_mode: PriceMode = "o2o"
    horizons: tuple[int, ...] = (1, 3, 5, 10)
    selection_mode: SelectionMode = "top_k"
    top_k: int = 300
    top_frac: float = 0.1
    signal_col: str = "signal"
    date_col: str = "datetime"
    symbol_col: str = "vt_symbol"
    trading_days: int = 240
    benchmark_symbol: str | None = None


@dataclass
class SimpleBacktestResult:
    config: SimpleBacktestConfig
    selected: pd.DataFrame
    horizon_returns: pd.DataFrame
    rolling_daily: pd.DataFrame
    rolling_metrics: pd.DataFrame
    annual_metrics: pd.DataFrame
    turnover_daily: pd.DataFrame
    turnover_metrics: pd.DataFrame
    turnover_annual: pd.DataFrame
    nonoverlap_metrics: pd.DataFrame
    nonoverlap_offsets: pd.DataFrame
    summary: pd.DataFrame


def _to_pandas(df: pd.DataFrame | Any) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df.copy()
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.to_pandas()
    raise TypeError(f"不支持的数据类型: {type(df)!r}")


def normalize_vt_symbol(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    out = (
        out.str.replace(r"\.XSHG$", ".SSE", regex=True)
        .str.replace(r"\.XSHE$", ".SZSE", regex=True)
        .str.replace(r"\.SH$", ".SSE", regex=True)
        .str.replace(r"\.SZ$", ".SZSE", regex=True)
    )
    invalid = out.str.lower().isin({"", "nan", "none", "null", "<na>"})
    return out.mask(invalid, pd.NA)


def normalize_signal_frame(
    signal: pd.DataFrame | Any,
    *,
    date_col: str = "datetime",
    symbol_col: str = "vt_symbol",
    signal_col: str = "signal",
) -> pd.DataFrame:
    df = _to_pandas(signal)
    missing = [col for col in [date_col, symbol_col, signal_col] if col not in df.columns]
    if missing:
        raise ValueError(f"signal 缺少列: {missing}")
    out = df.loc[:, [date_col, symbol_col, signal_col]].copy()
    out.columns = ["datetime", "vt_symbol", "signal"]
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce").dt.normalize()
    out["vt_symbol"] = normalize_vt_symbol(out["vt_symbol"])
    out["signal"] = pd.to_numeric(out["signal"], errors="coerce")
    return (
        out.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["datetime", "vt_symbol", "signal"])
        .drop_duplicates(["datetime", "vt_symbol"], keep="last")
        .sort_values(["datetime", "signal", "vt_symbol"], ascending=[True, False, True], kind="mergesort")
        .reset_index(drop=True)
    )


def normalize_price_frame(
    prices: pd.DataFrame | Any,
    *,
    date_col: str = "datetime",
    symbol_col: str = "vt_symbol",
    required_price_cols: tuple[str, ...] = ("open", "close"),
) -> pd.DataFrame:
    df = _to_pandas(prices)
    missing = [col for col in [date_col, symbol_col, *required_price_cols] if col not in df.columns]
    if missing:
        raise ValueError(f"price frame 缺少列: {missing}")
    for col in ("open", "close"):
        if col not in df.columns:
            df[col] = np.nan
    out = df.loc[:, [date_col, symbol_col, "open", "close"]].copy()
    out.columns = ["datetime", "vt_symbol", "open", "close"]
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce").dt.normalize()
    out["vt_symbol"] = normalize_vt_symbol(out["vt_symbol"])
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return (
        out.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["datetime", "vt_symbol"])
        .sort_values(["vt_symbol", "datetime"], kind="mergesort")
        .drop_duplicates(["datetime", "vt_symbol"], keep="last")
        .reset_index(drop=True)
    )


def normalize_benchmark_frame(
    benchmark_prices: pd.DataFrame | Any,
    *,
    benchmark_symbol: str | None = None,
    date_col: str = "datetime",
    symbol_col: str = "vt_symbol",
    required_price_cols: tuple[str, ...] = ("open", "close"),
) -> pd.DataFrame:
    df = normalize_price_frame(
        benchmark_prices,
        date_col=date_col,
        symbol_col=symbol_col,
        required_price_cols=required_price_cols,
    )
    if benchmark_symbol is not None:
        symbol = normalize_vt_symbol(pd.Series([benchmark_symbol])).iloc[0]
        df = df[df["vt_symbol"] == symbol].copy()
    if df.empty:
        raise ValueError("benchmark price frame 为空")
    if df["vt_symbol"].nunique() > 1:
        raise ValueError("benchmark price frame 包含多个 vt_symbol，请传入 benchmark_symbol")
    return df.reset_index(drop=True)


def _price_column(price_mode: PriceMode) -> str:
    if price_mode == "o2o":
        return "open"
    if price_mode == "c2c":
        return "close"
    raise ValueError(f"不支持的 price_mode: {price_mode}")


def prepare_forward_returns(
    prices: pd.DataFrame | Any,
    *,
    price_mode: PriceMode = "o2o",
    horizons: tuple[int, ...] = (1, 3, 5, 10),
    date_col: str = "datetime",
    symbol_col: str = "vt_symbol",
) -> pd.DataFrame:
    """Build per-signal-date forward returns for each horizon.

    For both O2O and C2C, signal date t uses entry at t+1 and exits at
    t+1+horizon. Therefore horizon=1 is t+1 price to t+2 price. Here t+1
    intentionally means the symbol's next available observation; continuity
    against a shared market calendar is not enforced.
    """

    if any(int(h) <= 0 for h in horizons):
        raise ValueError(f"horizons 必须为正整数: {horizons}")
    price_col = _price_column(price_mode)
    df = normalize_price_frame(
        prices,
        date_col=date_col,
        symbol_col=symbol_col,
        required_price_cols=(price_col,),
    )
    df = df.dropna(subset=[price_col])
    df = df[df[price_col] > 0].copy()
    df = df.sort_values(["vt_symbol", "datetime"], kind="mergesort")

    out = df.loc[:, ["datetime", "vt_symbol"]].copy()
    grouped_price = df.groupby("vt_symbol", sort=False)[price_col]
    entry = grouped_price.shift(-1)
    for horizon in horizons:
        exit_price = grouped_price.shift(-(int(horizon) + 1))
        out[f"ret_h{int(horizon)}"] = exit_price / entry - 1.0
    return out.replace([np.inf, -np.inf], np.nan)


def prepare_one_period_returns(
    prices: pd.DataFrame | Any,
    *,
    price_mode: PriceMode = "o2o",
    date_col: str = "datetime",
    symbol_col: str = "vt_symbol",
) -> pd.DataFrame:
    """Build one-period returns keyed by the return start date.

    For O2O this is open(t+1) / open(t) - 1, keyed at t. For C2C this is
    close(t+1) / close(t) - 1, keyed at t. The shift follows each symbol's
    next available observation and does not enforce shared-calendar continuity.
    """

    price_col = _price_column(price_mode)
    df = normalize_price_frame(
        prices,
        date_col=date_col,
        symbol_col=symbol_col,
        required_price_cols=(price_col,),
    )
    df = df.dropna(subset=[price_col])
    df = df[df[price_col] > 0].copy()
    df = df.sort_values(["vt_symbol", "datetime"], kind="mergesort")
    grouped_price = df.groupby("vt_symbol", sort=False)[price_col]
    out = df.loc[:, ["datetime", "vt_symbol"]].copy()
    out["period_return"] = grouped_price.shift(-1) / df[price_col] - 1.0
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["period_return"]).reset_index(drop=True)


def prepare_benchmark_forward_returns(
    benchmark_prices: pd.DataFrame | Any,
    *,
    price_mode: PriceMode = "o2o",
    horizons: tuple[int, ...] = (1, 3, 5, 10),
    benchmark_symbol: str | None = None,
    date_col: str = "datetime",
    symbol_col: str = "vt_symbol",
) -> pd.DataFrame:
    benchmark = normalize_benchmark_frame(
        benchmark_prices,
        benchmark_symbol=benchmark_symbol,
        date_col=date_col,
        symbol_col=symbol_col,
        required_price_cols=(_price_column(price_mode),),
    )
    returns = prepare_forward_returns(
        benchmark,
        price_mode=price_mode,
        horizons=horizons,
        date_col="datetime",
        symbol_col="vt_symbol",
    )
    cols = ["datetime", *[f"ret_h{int(h)}" for h in horizons]]
    out = returns.loc[:, cols].copy()
    return out.rename(columns={f"ret_h{int(h)}": f"bench_ret_h{int(h)}" for h in horizons})


def prepare_benchmark_one_period_returns(
    benchmark_prices: pd.DataFrame | Any,
    *,
    price_mode: PriceMode = "o2o",
    benchmark_symbol: str | None = None,
    date_col: str = "datetime",
    symbol_col: str = "vt_symbol",
) -> pd.DataFrame:
    benchmark = normalize_benchmark_frame(
        benchmark_prices,
        benchmark_symbol=benchmark_symbol,
        date_col=date_col,
        symbol_col=symbol_col,
        required_price_cols=(_price_column(price_mode),),
    )
    returns = prepare_one_period_returns(
        benchmark,
        price_mode=price_mode,
        date_col="datetime",
        symbol_col="vt_symbol",
    )
    return returns.loc[:, ["datetime", "period_return"]].rename(columns={"period_return": "benchmark_return"})


def select_signal(
    signal: pd.DataFrame | Any,
    *,
    selection_mode: SelectionMode = "top_k",
    top_k: int = 300,
    top_frac: float = 0.1,
    date_col: str = "datetime",
    symbol_col: str = "vt_symbol",
    signal_col: str = "signal",
) -> pd.DataFrame:
    """Select daily TopK or TopFraction signal names."""

    df = normalize_signal_frame(signal, date_col=date_col, symbol_col=symbol_col, signal_col=signal_col)
    if selection_mode == "top_k" and int(top_k) <= 0:
        raise ValueError(f"top_k 必须为正数: {top_k}")
    if selection_mode == "top_frac" and not 0 < float(top_frac) <= 1:
        raise ValueError(f"top_frac 必须在 (0, 1] 内: {top_frac}")
    if selection_mode not in {"top_k", "top_frac"}:
        raise ValueError(f"不支持的 selection_mode: {selection_mode}")

    parts: list[pd.DataFrame] = []
    for dt, group in df.groupby("datetime", sort=True):
        group = group.sort_values(["signal", "vt_symbol"], ascending=[False, True], kind="mergesort")
        eligible_count = int(len(group))
        if selection_mode == "top_k":
            selected_count = min(int(top_k), eligible_count)
        else:
            selected_count = max(1, int(math.ceil(eligible_count * float(top_frac))))
        selected = group.head(selected_count).copy()
        selected["rank"] = np.arange(1, selected_count + 1)
        selected["eligible_count"] = eligible_count
        selected["selected_count"] = selected_count
        selected["weight"] = 1.0 / selected_count
        parts.append(selected)

    if not parts:
        return pd.DataFrame(
            columns=["datetime", "vt_symbol", "signal", "rank", "eligible_count", "selected_count", "weight"]
        )
    return pd.concat(parts, ignore_index=True)


def build_horizon_returns(
    selected: pd.DataFrame,
    forward_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame | None,
    *,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Compute one row per signal date and horizon.

    Missing constituent returns contribute zero and keep their original
    portfolio weight. A signal date is omitted only when every selected name
    lacks the requested forward return.
    """

    joined = selected.merge(forward_returns, on=["datetime", "vt_symbol"], how="left")
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        ret_col = f"ret_h{int(horizon)}"
        for dt, group in joined.groupby("datetime", sort=True):
            weight = pd.to_numeric(group["weight"], errors="coerce").to_numpy(dtype=float)
            ret = pd.to_numeric(group[ret_col], errors="coerce").to_numpy(dtype=float)
            valid_weight = np.isfinite(weight) & (weight >= 0)
            valid_return = valid_weight & np.isfinite(ret)
            if not valid_return.any():
                continue
            weight_sum = float(weight[valid_weight].sum())
            if weight_sum <= 0:
                continue
            filled_return = np.where(valid_return, ret, 0.0)
            rows.append(
                {
                    "datetime": pd.Timestamp(dt),
                    "horizon": int(horizon),
                    "strategy_return": float(np.dot(weight[valid_weight], filled_return[valid_weight]) / weight_sum),
                    "selected_count": int(len(group)),
                    "return_count": int(valid_return.sum()),
                    "missing_return_count": int(valid_weight.sum() - valid_return.sum()),
                    "weight_sum": weight_sum,
                    "return_coverage_weight": float(weight[valid_return].sum() / weight_sum),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "horizon",
                "strategy_return",
                "benchmark_return",
                "excess_return",
                "selected_count",
                "return_count",
                "missing_return_count",
                "weight_sum",
                "return_coverage_weight",
            ]
        )

    if benchmark_returns is not None and not benchmark_returns.empty:
        bench_parts = []
        for horizon in horizons:
            col = f"bench_ret_h{int(horizon)}"
            if col in benchmark_returns.columns:
                part = benchmark_returns.loc[:, ["datetime", col]].rename(columns={col: "benchmark_return"})
                part["horizon"] = int(horizon)
                bench_parts.append(part)
        if bench_parts:
            bench = pd.concat(bench_parts, ignore_index=True)
            out = out.merge(bench, on=["datetime", "horizon"], how="left")
    if "benchmark_return" not in out.columns:
        out["benchmark_return"] = np.nan
    out["excess_return"] = out["strategy_return"] - out["benchmark_return"]
    return out.sort_values(["horizon", "datetime"]).reset_index(drop=True)


def compute_return_profile(returns: pd.Series, *, trading_days: int = 240) -> dict[str, float | int | None]:
    series = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return {
            "n_dates": 0,
            "total_return": None,
            "annual_return": None,
            "annual_volatility": None,
            "sharpe_ratio": None,
            "max_drawdown": None,
            "win_rate": None,
            "daily_mean": None,
        }
    curve = (1.0 + series).cumprod()
    total_return = float(curve.iloc[-1] - 1.0)
    ending_wealth = 1.0 + total_return
    annual_return = ending_wealth ** (trading_days / len(series)) - 1.0 if ending_wealth >= 0 else np.nan
    vol = float(series.std(ddof=0))
    annual_vol = vol * math.sqrt(trading_days)
    sharpe = float(series.mean()) / vol * math.sqrt(trading_days) if vol > 0 else None
    wealth = pd.concat([pd.Series([1.0]), curve.reset_index(drop=True)], ignore_index=True)
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "n_dates": int(len(series)),
        "total_return": total_return,
        "annual_return": float(annual_return),
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((series > 0).mean()),
        "daily_mean": float(series.mean()),
    }


def compute_relative_total_return(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float | None:
    """Return strategy wealth divided by benchmark wealth, minus one."""

    aligned = pd.concat(
        [
            pd.to_numeric(strategy_returns, errors="coerce").rename("strategy"),
            pd.to_numeric(benchmark_returns, errors="coerce").rename("benchmark"),
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        return None
    strategy_wealth = float((1.0 + aligned["strategy"]).prod())
    benchmark_wealth = float((1.0 + aligned["benchmark"]).prod())
    if benchmark_wealth == 0:
        return None
    return strategy_wealth / benchmark_wealth - 1.0


def build_rolling_sleeve_daily(
    selected: pd.DataFrame,
    one_period_returns: pd.DataFrame,
    benchmark_one_period_returns: pd.DataFrame | None,
    *,
    horizons: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build daily rolling-sleeve returns from actual one-period stock returns.

    A horizon=5 sleeve always owns exactly 1/5 of total capital. During ramp-up,
    signal gaps, and wind-down, unused capital remains cash with zero return.
    Missing stock returns are also treated as zero without reallocating their
    weights. Turnover is calculated from the resulting daily target weights
    before return-driven drift.
    """

    selected = selected.copy()
    selected["datetime"] = pd.to_datetime(selected["datetime"]).dt.normalize()
    one_period = one_period_returns.copy()
    one_period["datetime"] = pd.to_datetime(one_period["datetime"]).dt.normalize()
    one_period = one_period.drop_duplicates(["datetime", "vt_symbol"], keep="last")
    calendar = pd.DatetimeIndex(sorted(one_period["datetime"].dropna().unique()))
    date_to_idx = {pd.Timestamp(dt): idx for idx, dt in enumerate(calendar)}
    selected["_signal_idx"] = selected["datetime"].map(date_to_idx)
    selected = selected.dropna(subset=["_signal_idx"]).copy()
    selected["_signal_idx"] = selected["_signal_idx"].astype(int)

    return_lookup = one_period.set_index(["datetime", "vt_symbol"])["period_return"]
    daily_parts: list[pd.DataFrame] = []
    turnover_parts: list[pd.DataFrame] = []
    for horizon in horizons:
        horizon = int(horizon)
        base = selected.loc[selected["_signal_idx"] + horizon < len(calendar)].copy()
        if base.empty:
            continue
        holdings_parts: list[pd.DataFrame] = []
        for offset in range(horizon):
            part = base.loc[:, ["datetime", "vt_symbol", "weight", "_signal_idx"]].copy()
            part = part.rename(columns={"datetime": "signal_date"})
            return_idx = part["_signal_idx"].to_numpy(dtype=int) + 1 + offset
            part["datetime"] = calendar.take(return_idx).to_numpy()
            part["target_weight"] = pd.to_numeric(part["weight"], errors="coerce") / horizon
            holdings_parts.append(part.loc[:, ["datetime", "signal_date", "vt_symbol", "target_weight"]])
        holdings = pd.concat(holdings_parts, ignore_index=True)

        lookup_index = pd.MultiIndex.from_frame(holdings.loc[:, ["datetime", "vt_symbol"]])
        raw_return = return_lookup.reindex(lookup_index).to_numpy(dtype=float)
        holdings["has_return"] = np.isfinite(raw_return)
        holdings["period_return"] = np.where(holdings["has_return"], raw_return, 0.0)
        holdings["return_contribution"] = holdings["target_weight"] * holdings["period_return"]

        daily = (
            holdings.groupby("datetime", as_index=False, sort=True)
            .agg(
                strategy_return=("return_contribution", "sum"),
                active_sleeves=("signal_date", "nunique"),
                selected_position_count=("vt_symbol", "size"),
                return_count=("has_return", "sum"),
                invested_weight=("target_weight", "sum"),
            )
            .sort_values("datetime")
            .reset_index(drop=True)
        )
        active_calendar = calendar[(calendar >= holdings["datetime"].min()) & (calendar <= holdings["datetime"].max())]
        daily = daily.set_index("datetime").reindex(active_calendar).rename_axis("datetime").reset_index()
        count_cols = ["active_sleeves", "selected_position_count", "return_count"]
        daily[count_cols] = daily[count_cols].fillna(0).astype(int)
        daily[["strategy_return", "invested_weight"]] = daily[["strategy_return", "invested_weight"]].fillna(0.0)
        daily["horizon"] = horizon
        daily["missing_return_count"] = daily["selected_position_count"] - daily["return_count"]
        daily["return_coverage_ratio"] = np.where(
            daily["selected_position_count"] > 0,
            daily["return_count"] / daily["selected_position_count"],
            np.nan,
        )
        daily["cash_weight"] = (1.0 - daily["invested_weight"]).clip(lower=0.0)
        daily["avg_symbols_per_sleeve"] = np.where(
            daily["active_sleeves"] > 0,
            daily["selected_position_count"] / daily["active_sleeves"],
            0.0,
        )
        daily_parts.append(daily)

        target_weights = (
            holdings.groupby(["datetime", "vt_symbol"], as_index=False, sort=True)["target_weight"].sum()
        )
        turnover_parts.append(
            compute_target_weight_turnover(target_weights, horizon=horizon, calendar=active_calendar)
        )

    if not daily_parts:
        empty_daily = pd.DataFrame(
            columns=["datetime", "horizon", "strategy_return", "benchmark_return", "excess_return", "active_sleeves"]
        )
        return empty_daily, pd.DataFrame()

    daily = pd.concat(daily_parts, ignore_index=True)
    if benchmark_one_period_returns is not None and not benchmark_one_period_returns.empty:
        benchmark = benchmark_one_period_returns.copy()
        benchmark["datetime"] = pd.to_datetime(benchmark["datetime"]).dt.normalize()
        benchmark = benchmark.drop_duplicates("datetime", keep="last")
        daily = daily.merge(benchmark.loc[:, ["datetime", "benchmark_return"]], on="datetime", how="left")
    else:
        daily["benchmark_return"] = np.nan
    daily["excess_return"] = daily["strategy_return"] - daily["benchmark_return"]
    daily = daily.sort_values(["horizon", "datetime"]).reset_index(drop=True)
    daily["strategy_curve"] = daily.groupby("horizon")["strategy_return"].transform(lambda s: (1.0 + s).cumprod())
    daily["benchmark_curve"] = daily.groupby("horizon")["benchmark_return"].transform(lambda s: (1.0 + s).cumprod())
    daily["excess_curve"] = daily.groupby("horizon")["excess_return"].transform(lambda s: (1.0 + s).cumprod())
    turnover = pd.concat(turnover_parts, ignore_index=True) if turnover_parts else pd.DataFrame()
    return daily, turnover


def compute_target_weight_turnover(
    target_weights: pd.DataFrame,
    *,
    horizon: int,
    calendar: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Calculate one-way turnover from daily stock and cash target weights."""

    if target_weights.empty:
        return pd.DataFrame()
    matrix = target_weights.pivot(index="datetime", columns="vt_symbol", values="target_weight").fillna(0.0)
    matrix = matrix.sort_index()
    if calendar is not None:
        matrix = matrix.reindex(pd.DatetimeIndex(calendar), fill_value=0.0)
    invested = matrix.sum(axis=1)
    cash = (1.0 - invested).clip(lower=0.0)

    stock_delta = matrix.diff()
    stock_delta.iloc[0] = matrix.iloc[0]
    previous_cash = cash.shift(1)
    previous_cash.iloc[0] = 1.0
    cash_delta = cash - previous_cash

    stock_buy = stock_delta.clip(lower=0.0).sum(axis=1)
    stock_sell = -stock_delta.clip(upper=0.0).sum(axis=1)
    one_way = 0.5 * (stock_delta.abs().sum(axis=1) + cash_delta.abs())
    overlap_weight = matrix.combine(matrix.shift(1).fillna(0.0), np.minimum).sum(axis=1)
    overlap_weight.iloc[0] = 0.0

    return pd.DataFrame(
        {
            "datetime": matrix.index,
            "horizon": int(horizon),
            "one_way_turnover": one_way.to_numpy(dtype=float),
            "gross_stock_turnover": (stock_buy + stock_sell).to_numpy(dtype=float),
            "stock_buy_weight": stock_buy.to_numpy(dtype=float),
            "stock_sell_weight": stock_sell.to_numpy(dtype=float),
            "invested_weight": invested.to_numpy(dtype=float),
            "cash_weight": cash.to_numpy(dtype=float),
            "overlap_weight": overlap_weight.to_numpy(dtype=float),
            "position_count": (matrix > 0).sum(axis=1).to_numpy(dtype=int),
        }
    )


def compute_rolling_metrics(rolling_daily: pd.DataFrame, *, trading_days: int = 240) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, group in rolling_daily.groupby("horizon", sort=True):
        strategy = compute_return_profile(group["strategy_return"], trading_days=trading_days)
        benchmark = compute_return_profile(group["benchmark_return"], trading_days=trading_days)
        excess = compute_return_profile(group["excess_return"], trading_days=trading_days)
        rows.append(
            {
                "horizon": int(horizon),
                "mode": "rolling_sleeve",
                "start": group["datetime"].min(),
                "end": group["datetime"].max(),
                "n_dates": strategy["n_dates"],
                "total_return": strategy["total_return"],
                "annual_return": strategy["annual_return"],
                "sharpe_ratio": strategy["sharpe_ratio"],
                "max_drawdown": strategy["max_drawdown"],
                "win_rate": strategy["win_rate"],
                "daily_mean": strategy["daily_mean"],
                "benchmark_total_return": benchmark["total_return"],
                "excess_total_return": excess["total_return"],
                "excess_annual_return": excess["annual_return"],
                "excess_sharpe_ratio": excess["sharpe_ratio"],
                "excess_max_drawdown": excess["max_drawdown"],
                "relative_total_return": compute_relative_total_return(
                    group["strategy_return"], group["benchmark_return"]
                ),
                "avg_active_sleeves": float(group["active_sleeves"].mean()),
                "avg_invested_weight": float(group["invested_weight"].mean()),
                "avg_return_coverage_ratio": float(group["return_coverage_ratio"].mean()),
            }
        )
    return pd.DataFrame(rows)


def compute_annual_metrics(rolling_daily: pd.DataFrame, *, trading_days: int = 240) -> pd.DataFrame:
    """Compute calendar-year rolling-sleeve strategy and excess metrics."""

    if rolling_daily.empty:
        return pd.DataFrame()
    data = rolling_daily.copy()
    data["year"] = pd.to_datetime(data["datetime"]).dt.year
    rows: list[dict[str, Any]] = []
    for (horizon, year), group in data.groupby(["horizon", "year"], sort=True):
        strategy = compute_return_profile(group["strategy_return"], trading_days=trading_days)
        benchmark = compute_return_profile(group["benchmark_return"], trading_days=trading_days)
        excess = compute_return_profile(group["excess_return"], trading_days=trading_days)
        rows.append(
            {
                "horizon": int(horizon),
                "year": int(year),
                "start": group["datetime"].min(),
                "end": group["datetime"].max(),
                "n_dates": strategy["n_dates"],
                "total_return": strategy["total_return"],
                "annualized_return": strategy["annual_return"],
                "sharpe_ratio": strategy["sharpe_ratio"],
                "max_drawdown": strategy["max_drawdown"],
                "win_rate": strategy["win_rate"],
                "benchmark_total_return": benchmark["total_return"],
                "excess_total_return": excess["total_return"],
                "excess_annualized_return": excess["annual_return"],
                "excess_sharpe_ratio": excess["sharpe_ratio"],
                "excess_max_drawdown": excess["max_drawdown"],
                "relative_total_return": compute_relative_total_return(
                    group["strategy_return"], group["benchmark_return"]
                ),
                "avg_active_sleeves": float(group["active_sleeves"].mean()),
                "avg_invested_weight": float(group["invested_weight"].mean()),
                "avg_return_coverage_ratio": float(group["return_coverage_ratio"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _summarize_turnover_group(group: pd.DataFrame, *, trading_days: int) -> dict[str, float | int]:
    turnover = pd.to_numeric(group["one_way_turnover"], errors="coerce").dropna()
    return {
        "n_dates": int(len(turnover)),
        "avg_one_way_turnover": float(turnover.mean()),
        "median_one_way_turnover": float(turnover.median()),
        "total_one_way_turnover": float(turnover.sum()),
        "annualized_one_way_turnover": float(turnover.mean() * trading_days),
        "avg_gross_stock_turnover": float(group["gross_stock_turnover"].mean()),
        "avg_stock_buy_weight": float(group["stock_buy_weight"].mean()),
        "avg_stock_sell_weight": float(group["stock_sell_weight"].mean()),
        "avg_overlap_weight": float(group["overlap_weight"].mean()),
        "avg_invested_weight": float(group["invested_weight"].mean()),
        "avg_cash_weight": float(group["cash_weight"].mean()),
        "avg_position_count": float(group["position_count"].mean()),
    }


def compute_turnover_metrics(
    turnover_daily: pd.DataFrame,
    *,
    trading_days: int = 240,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize target-weight turnover for the full sample and by year."""

    if turnover_daily.empty:
        return pd.DataFrame(), pd.DataFrame()
    data = turnover_daily.copy()
    data["datetime"] = pd.to_datetime(data["datetime"]).dt.normalize()
    data["year"] = data["datetime"].dt.year

    summary_rows: list[dict[str, Any]] = []
    for horizon, group in data.groupby("horizon", sort=True):
        summary_rows.append(
            {
                "horizon": int(horizon),
                "start": group["datetime"].min(),
                "end": group["datetime"].max(),
                **_summarize_turnover_group(group, trading_days=trading_days),
            }
        )

    annual_rows: list[dict[str, Any]] = []
    for (horizon, year), group in data.groupby(["horizon", "year"], sort=True):
        annual_rows.append(
            {
                "horizon": int(horizon),
                "year": int(year),
                "start": group["datetime"].min(),
                "end": group["datetime"].max(),
                **_summarize_turnover_group(group, trading_days=trading_days),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(annual_rows)


def compute_nonoverlap_cohort_metrics(
    horizon_returns: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    trading_days: int = 240,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute non-overlapping cohort metrics for each horizon."""

    offset_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        part = horizon_returns[horizon_returns["horizon"] == int(horizon)].copy()
        if part.empty:
            continue
        part = part.sort_values("datetime").reset_index(drop=True)
        for offset in range(int(horizon)):
            cohort = part.iloc[offset:: int(horizon)].copy()
            if cohort.empty:
                continue
            strategy = compute_return_profile(cohort["strategy_return"], trading_days=trading_days / int(horizon))
            benchmark = compute_return_profile(cohort["benchmark_return"], trading_days=trading_days / int(horizon))
            excess = compute_return_profile(cohort["excess_return"], trading_days=trading_days / int(horizon))
            offset_rows.append(
                {
                    "horizon": int(horizon),
                    "offset": int(offset),
                    "start": cohort["datetime"].min(),
                    "end": cohort["datetime"].max(),
                    "n_periods": strategy["n_dates"],
                    "total_return": strategy["total_return"],
                    "annual_return": strategy["annual_return"],
                    "sharpe_ratio": strategy["sharpe_ratio"],
                    "max_drawdown": strategy["max_drawdown"],
                    "win_rate": strategy["win_rate"],
                    "period_mean": strategy["daily_mean"],
                    "benchmark_total_return": benchmark["total_return"],
                    "excess_total_return": excess["total_return"],
                    "excess_annual_return": excess["annual_return"],
                    "excess_sharpe_ratio": excess["sharpe_ratio"],
                    "excess_max_drawdown": excess["max_drawdown"],
                }
            )

        offsets = pd.DataFrame([row for row in offset_rows if row["horizon"] == int(horizon)])
        if offsets.empty:
            continue
        summary_rows.append(
            {
                "horizon": int(horizon),
                "mode": "nonoverlap_cohort",
                "start": offsets["start"].min(),
                "end": offsets["end"].max(),
                "offset_count": int(offsets["offset"].nunique()),
                "avg_n_periods": float(offsets["n_periods"].mean()),
                "total_return": float(offsets["total_return"].mean()),
                "annual_return": float(offsets["annual_return"].mean()),
                "sharpe_ratio": float(offsets["sharpe_ratio"].mean()),
                "max_drawdown": float(offsets["max_drawdown"].mean()),
                "win_rate": float(offsets["win_rate"].mean()),
                "period_mean": float(offsets["period_mean"].mean()),
                "benchmark_total_return": float(offsets["benchmark_total_return"].mean()),
                "excess_total_return": float(offsets["excess_total_return"].mean()),
                "excess_annual_return": float(offsets["excess_annual_return"].mean()),
                "excess_sharpe_ratio": float(offsets["excess_sharpe_ratio"].mean()),
                "excess_max_drawdown": float(offsets["excess_max_drawdown"].mean()),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(offset_rows)


def summarize_signal_date_returns(
    horizon_returns: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        part = horizon_returns[horizon_returns["horizon"] == int(horizon)].copy()
        if part.empty:
            continue
        ret = pd.to_numeric(part["strategy_return"], errors="coerce")
        excess = pd.to_numeric(part["excess_return"], errors="coerce")
        rows.append(
            {
                "horizon": int(horizon),
                "mode": "signal_date_horizon",
                "n_dates": int(ret.notna().sum()),
                "avg_names": float(part["return_count"].mean()),
                "avg_selected_names": float(part["selected_count"].mean()),
                "avg_return_coverage_weight": float(part["return_coverage_weight"].mean()),
                "avg_return": float(ret.mean()),
                "median_return": float(ret.median()),
                "win_rate": float((ret > 0).mean()),
                "avg_benchmark_return": float(pd.to_numeric(part["benchmark_return"], errors="coerce").mean()),
                "avg_excess_return": float(excess.mean()),
                "excess_win_rate": float((excess > 0).mean()),
                "return_t": float(ret.mean() / ret.std(ddof=0) * math.sqrt(ret.notna().sum()))
                if ret.std(ddof=0) > 0
                else np.nan,
                "excess_t": float(excess.mean() / excess.std(ddof=0) * math.sqrt(excess.notna().sum()))
                if excess.std(ddof=0) > 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_simple_signal_backtest(
    *,
    signal: pd.DataFrame | Any,
    prices: pd.DataFrame | Any,
    config: SimpleBacktestConfig | None = None,
    benchmark_prices: pd.DataFrame | Any | None = None,
) -> SimpleBacktestResult:
    """Run simple O2O/C2C signal diagnostics."""

    config = config or SimpleBacktestConfig()
    horizons = tuple(sorted({int(h) for h in config.horizons}))
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError(f"horizons 必须包含正整数: {config.horizons}")
    if config.trading_days <= 0:
        raise ValueError(f"trading_days 必须为正数: {config.trading_days}")
    selected = select_signal(
        signal,
        selection_mode=config.selection_mode,
        top_k=config.top_k,
        top_frac=config.top_frac,
        date_col=config.date_col,
        symbol_col=config.symbol_col,
        signal_col=config.signal_col,
    )
    forward_returns = prepare_forward_returns(
        prices,
        price_mode=config.price_mode,
        horizons=horizons,
        date_col=config.date_col,
        symbol_col=config.symbol_col,
    )
    one_period_returns = prepare_one_period_returns(
        prices,
        price_mode=config.price_mode,
        date_col=config.date_col,
        symbol_col=config.symbol_col,
    )
    benchmark_returns = None
    benchmark_one_period_returns = None
    if benchmark_prices is not None:
        benchmark_returns = prepare_benchmark_forward_returns(
            benchmark_prices,
            price_mode=config.price_mode,
            horizons=horizons,
            benchmark_symbol=config.benchmark_symbol,
            date_col=config.date_col,
            symbol_col=config.symbol_col,
        )
        benchmark_one_period_returns = prepare_benchmark_one_period_returns(
            benchmark_prices,
            price_mode=config.price_mode,
            benchmark_symbol=config.benchmark_symbol,
            date_col=config.date_col,
            symbol_col=config.symbol_col,
        )
    horizon_returns = build_horizon_returns(selected, forward_returns, benchmark_returns, horizons=horizons)
    rolling_daily, turnover_daily = build_rolling_sleeve_daily(
        selected,
        one_period_returns,
        benchmark_one_period_returns,
        horizons=horizons,
    )
    rolling_metrics = compute_rolling_metrics(rolling_daily, trading_days=config.trading_days)
    annual_metrics = compute_annual_metrics(rolling_daily, trading_days=config.trading_days)
    turnover_metrics, turnover_annual = compute_turnover_metrics(
        turnover_daily,
        trading_days=config.trading_days,
    )
    nonoverlap_metrics, nonoverlap_offsets = compute_nonoverlap_cohort_metrics(
        horizon_returns,
        horizons=horizons,
        trading_days=config.trading_days,
    )
    signal_date_summary = summarize_signal_date_returns(horizon_returns, horizons=horizons)
    summary = pd.concat([signal_date_summary, rolling_metrics, nonoverlap_metrics], ignore_index=True, sort=False)
    return SimpleBacktestResult(
        config=config,
        selected=selected,
        horizon_returns=horizon_returns,
        rolling_daily=rolling_daily,
        rolling_metrics=rolling_metrics,
        annual_metrics=annual_metrics,
        turnover_daily=turnover_daily,
        turnover_metrics=turnover_metrics,
        turnover_annual=turnover_annual,
        nonoverlap_metrics=nonoverlap_metrics,
        nonoverlap_offsets=nonoverlap_offsets,
        summary=summary,
    )


def run_simple_signal_backtest_from_lab(
    *,
    signal: pd.DataFrame | Any,
    lab: Any,
    config: SimpleBacktestConfig | None = None,
    output_dir: str | Path | None = None,
    interval: Any = None,
    extended_days: int = 400,
    parquet: bool = True,
) -> SimpleBacktestResult:
    """Load prices from ``AlphaLab`` and run a simple signal backtest.

    This is the notebook-facing entrypoint. It keeps price loading, benchmark
    loading, calculation, and result persistence out of notebook cells.
    Signal dates define the requested backtest range; ``extended_days`` gives
    the price loader enough trailing observations for the longest horizon.
    """

    config = config or SimpleBacktestConfig()
    normalized_signal = normalize_signal_frame(
        signal,
        date_col=config.date_col,
        symbol_col=config.symbol_col,
        signal_col=config.signal_col,
    )
    if normalized_signal.empty:
        raise ValueError("signal 标准化后为空")
    if extended_days < 0:
        raise ValueError("extended_days 不能为负数")

    load_kwargs: dict[str, Any] = {
        "start": normalized_signal["datetime"].min().to_pydatetime(),
        "end": normalized_signal["datetime"].max().to_pydatetime(),
        "extended_days": int(extended_days),
    }
    if interval is not None:
        load_kwargs["interval"] = interval

    symbols = normalized_signal["vt_symbol"].dropna().unique().tolist()
    prices = lab.load_bar_df(vt_symbols=symbols, **load_kwargs)
    if prices is None or len(prices) == 0:
        raise ValueError("未从 AlphaLab 读取到股票价格")

    benchmark_prices = None
    if config.benchmark_symbol:
        benchmark_symbol = normalize_vt_symbol(
            pd.Series([config.benchmark_symbol])
        ).iloc[0]
        benchmark_prices = lab.load_bar_df(
            vt_symbols=[str(benchmark_symbol)],
            **load_kwargs,
        )
        if benchmark_prices is None or len(benchmark_prices) == 0:
            raise ValueError(
                f"未从 AlphaLab 读取到基准价格: {config.benchmark_symbol}"
            )

    normalized_config = SimpleBacktestConfig(
        **{
            **asdict(config),
            "date_col": "datetime",
            "symbol_col": "vt_symbol",
            "signal_col": "signal",
        }
    )
    result = run_simple_signal_backtest(
        signal=normalized_signal,
        prices=prices,
        config=normalized_config,
        benchmark_prices=benchmark_prices,
    )
    if output_dir is not None:
        write_simple_backtest_outputs(result, output_dir, parquet=parquet)
    return result


def write_simple_backtest_outputs(
    result: SimpleBacktestResult,
    output_dir: str | Path,
    *,
    parquet: bool = True,
) -> None:
    """Write metrics, diagnostics, config, and optional parquet details."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(output / "summary.csv", index=False)
    result.rolling_metrics.to_csv(output / "rolling_metrics.csv", index=False)
    result.annual_metrics.to_csv(output / "annual_metrics.csv", index=False)
    result.turnover_daily.to_csv(output / "turnover_daily.csv", index=False)
    result.turnover_metrics.to_csv(output / "turnover_metrics.csv", index=False)
    result.turnover_annual.to_csv(output / "turnover_annual.csv", index=False)
    result.nonoverlap_metrics.to_csv(output / "nonoverlap_cohort_metrics.csv", index=False)
    result.nonoverlap_offsets.to_csv(output / "nonoverlap_cohort_offsets.csv", index=False)
    result.horizon_returns.to_csv(output / "signal_date_horizon_returns.csv", index=False)
    result.rolling_daily.to_csv(output / "rolling_daily_returns.csv", index=False)
    (output / "config.json").write_text(
        json.dumps(asdict(result.config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if parquet:
        result.selected.to_parquet(output / "selected.parquet", index=False)
        result.horizon_returns.to_parquet(output / "signal_date_horizon_returns.parquet", index=False)
        result.rolling_daily.to_parquet(output / "rolling_daily_returns.parquet", index=False)
        if not result.turnover_daily.empty:
            result.turnover_daily.to_parquet(output / "turnover_daily.parquet", index=False)

    lines = [
        "# Simple Signal Backtest",
        "",
        "该结果是不含手续费、涨跌停、n_drop 和 hold_thresh 的简单信号回测。",
        "",
        "- 信号日 T，在该股票下一条有效行情价格进入；当前不强制校验全市场交易日连续性。",
        "- horizon=h 时，每个 sleeve 固定占总资金 1/h，空缺资金保留现金。",
        "- 股票缺失单期收益按 0 处理，不向其他股票重新分配权重。",
        "- 换手是基于每日目标权重、包含现金变化的单边换手，不包含收益造成的盘中权重漂移。",
        "- excess 指逐日 strategy_return - benchmark_return 后复利；relative_total_return 指策略净值/基准净值-1。",
        "",
        "## Overall",
        "",
        result.rolling_metrics.to_markdown(index=False) if not result.rolling_metrics.empty else "No rolling results.",
        "",
        "## Annual",
        "",
        result.annual_metrics.to_markdown(index=False) if not result.annual_metrics.empty else "No annual results.",
        "",
        "## Turnover",
        "",
        result.turnover_metrics.to_markdown(index=False) if not result.turnover_metrics.empty else "No turnover results.",
        "",
        "## Annual Turnover",
        "",
        result.turnover_annual.to_markdown(index=False) if not result.turnover_annual.empty else "No annual turnover results.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a parquet or CSV table for the command-line entrypoint."""

    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv" or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"不支持的文件类型: {path}")


def _parse_horizons(value: str) -> tuple[int, ...]:
    horizons = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise argparse.ArgumentTypeError("horizons 必须是逗号分隔的正整数")
    return horizons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple O2O/C2C signal backtest.")
    parser.add_argument("--signal-path", required=True)
    parser.add_argument("--price-path", required=True)
    parser.add_argument("--benchmark-price-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--price-mode", choices=["o2o", "c2c"], default="o2o")
    parser.add_argument("--horizons", type=_parse_horizons, default=(1, 3, 5, 10))
    parser.add_argument("--selection-mode", choices=["top_k", "top_frac"], default="top_k")
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--top-frac", type=float, default=0.1)
    parser.add_argument("--signal-col", default="signal")
    parser.add_argument("--date-col", default="datetime")
    parser.add_argument("--symbol-col", default="vt_symbol")
    parser.add_argument("--trading-days", type=int, default=240)
    parser.add_argument("--benchmark-symbol")
    parser.add_argument("--no-parquet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimpleBacktestConfig(
        price_mode=args.price_mode,
        horizons=args.horizons,
        selection_mode=args.selection_mode,
        top_k=args.top_k,
        top_frac=args.top_frac,
        signal_col=args.signal_col,
        date_col=args.date_col,
        symbol_col=args.symbol_col,
        trading_days=args.trading_days,
        benchmark_symbol=args.benchmark_symbol,
    )
    result = run_simple_signal_backtest(
        signal=load_table(args.signal_path),
        prices=load_table(args.price_path),
        benchmark_prices=load_table(args.benchmark_price_path) if args.benchmark_price_path else None,
        config=config,
    )
    write_simple_backtest_outputs(result, args.output_dir, parquet=not args.no_parquet)
    print(result.rolling_metrics.to_markdown(index=False))
    print(result.turnover_metrics.to_markdown(index=False))


if __name__ == "__main__":
    main()
