from __future__ import annotations

from datetime import datetime, time
from math import gamma, pi

import numpy as np
import polars as pl


KEYS = ["trade_date", "vt_symbol"]


def _prepare(df: pl.DataFrame, fields: tuple[str, ...]) -> pl.DataFrame:
    required = {"datetime", "vt_symbol", *fields}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"intraday data missing columns: {missing}")
    return (
        df.select("datetime", "vt_symbol", *fields)
        .with_columns(
            pl.col("datetime").cast(pl.Datetime, strict=False),
            pl.col("datetime").cast(pl.Date).alias("trade_date"),
            *[pl.col(field).cast(pl.Float64, strict=False) for field in fields],
        )
        .sort(["vt_symbol", "datetime"])
    )


def _finish(df: pl.DataFrame, name: str) -> pl.DataFrame:
    return df.select(
        pl.col("trade_date").cast(pl.Datetime).alias("datetime"),
        "vt_symbol",
        pl.col(name).cast(pl.Float64, strict=False).fill_nan(None),
    ).sort(["datetime", "vt_symbol"])


def intraday_reduce(df: pl.DataFrame, field: str, stat: str, name: str) -> pl.DataFrame:
    work = _prepare(df, (field,))
    exprs = {
        "sum": pl.col(field).sum(),
        "mean": pl.col(field).mean(),
        "std": pl.col(field).std(ddof=0),
        "var": pl.col(field).var(ddof=0),
        "skew": pl.col(field).skew(bias=True),
        "kurtosis": pl.col(field).kurtosis(fisher=True, bias=True),
        "min": pl.col(field).min(),
        "max": pl.col(field).max(),
        "first": pl.col(field).first(),
        "last": pl.col(field).last(),
        "median": pl.col(field).median(),
    }
    if stat not in exprs:
        raise ValueError(f"unsupported intraday stat: {stat}")
    return _finish(work.group_by(KEYS).agg(exprs[stat].alias(name)), name)


def intraday_return_stat(df: pl.DataFrame, stat: str, name: str) -> pl.DataFrame:
    work = _prepare(df, ("close",)).with_columns(
        (pl.col("close").log() - pl.col("close").log().shift(1))
        .over(KEYS)
        .alias("_return")
    )
    return intraday_reduce(work, "_return", stat, name)


def intraday_return_measure(df: pl.DataFrame, measure: str, name: str) -> pl.DataFrame:
    """Compute exact daily measures from within-day one-bar log returns."""
    work = _prepare(df, ("close",)).with_columns(
        (pl.col("close").log() - pl.col("close").log().shift(1))
        .over(KEYS)
        .alias("_return")
    )
    squared = pl.col("_return").pow(2)
    total = squared.sum()
    up = pl.when(pl.col("_return") > 0).then(squared).otherwise(0.0).sum()
    down = pl.when(pl.col("_return") < 0).then(squared).otherwise(0.0).sum()
    expressions = {
        "realized_variance": total,
        "up_share": up / (total + 1e-12),
        "down_share": down / (total + 1e-12),
        "asymmetry": (up - down) / (total + 1e-12),
    }
    if measure not in expressions:
        raise ValueError(f"unsupported intraday return measure: {measure}")
    return _finish(work.group_by(KEYS).agg(expressions[measure].alias(name)), name)


def intraday_period_return(df: pl.DataFrame, minutes: int | None, name: str) -> pl.DataFrame:
    work = _prepare(df, ("close",))
    if minutes is not None:
        if minutes <= 0:
            raise ValueError("minutes must be positive")
        work = work.with_columns(pl.col("datetime").max().over(KEYS).alias("_day_end")).filter(
            pl.col("datetime") >= pl.col("_day_end") - pl.duration(minutes=minutes)
        )
    result = work.group_by(KEYS).agg(
        (pl.col("close").last() / (pl.col("close").first() + 1e-12) - 1).alias(name)
    )
    return _finish(result, name)


def intraday_max_drawdown(df: pl.DataFrame, name: str) -> pl.DataFrame:
    work = _prepare(df, ("close",)).with_columns(
        pl.col("close").cum_max().over(KEYS).alias("_peak")
    )
    result = work.group_by(KEYS).agg(
        (pl.col("close") / (pl.col("_peak") + 1e-12) - 1).min().alias(name)
    )
    return _finish(result, name)


def intraday_volume_share_stat(df: pl.DataFrame, stat: str, name: str) -> pl.DataFrame:
    work = _prepare(df, ("volume",)).with_columns(
        (pl.col("volume") / (pl.col("volume").sum().over(KEYS) + 1e-12)).alias("_share")
    )
    return intraday_reduce(work, "_share", stat, name)


def intraday_return_turnover_corr(
    df: pl.DataFrame,
    *,
    return_lag: int,
    turnover_lag: int,
    name: str,
) -> pl.DataFrame:
    work = _prepare(df, ("close", "turnover")).with_columns(
        (pl.col("close").log() - pl.col("close").log().shift(1))
        .over(KEYS)
        .abs()
        .alias("_abs_return"),
    ).with_columns(
        pl.col("_abs_return").shift(return_lag).over(KEYS).alias("_abs_return"),
        pl.col("turnover").shift(turnover_lag).over(KEYS).alias("_turnover"),
    )
    return _finish(work.group_by(KEYS).agg(pl.corr("_abs_return", "_turnover").alias(name)), name)


def intraday_autocorr(df: pl.DataFrame, field: str, lag: int, name: str) -> pl.DataFrame:
    if lag <= 0:
        raise ValueError("lag must be positive")
    work = _prepare(df, (field,)).with_columns(
        pl.col(field).shift(lag).over(KEYS).alias("_lagged")
    )
    return _finish(work.group_by(KEYS).agg(pl.corr(field, "_lagged").alias(name)), name)


def intraday_semivariance(df: pl.DataFrame, side: str, name: str) -> pl.DataFrame:
    work = _prepare(df, ("close",)).with_columns(
        (pl.col("close").log() - pl.col("close").log().shift(1))
        .over(KEYS)
        .alias("_return")
    )
    if side == "up":
        selected = pl.col("_return") > 0
    elif side == "down":
        selected = pl.col("_return") <= 0
    else:
        raise ValueError("side must be 'up' or 'down'")
    result = work.group_by(KEYS).agg(
        pl.when(selected).then(pl.col("_return").pow(2)).otherwise(0.0).sum().alias(name)
    )
    return _finish(result, name)


def intraday_corr(df: pl.DataFrame, left: str, right: str, name: str) -> pl.DataFrame:
    work = _prepare(df, (left, right))
    result = work.group_by(KEYS).agg(pl.corr(left, right).alias(name))
    return _finish(result, name)


def intraday_quantile(df: pl.DataFrame, field: str, quantile: float, name: str) -> pl.DataFrame:
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    work = _prepare(df, (field,))
    result = work.group_by(KEYS).agg(pl.col(field).quantile(quantile).alias(name))
    return _finish(result, name)


def intraday_time_slice(
    df: pl.DataFrame,
    start: time,
    end: time,
) -> pl.DataFrame:
    if start > end:
        raise ValueError("intraday slice cannot cross midnight")
    return df.filter(pl.col("datetime").dt.time().is_between(start, end, closed="both"))


def intraday_tail_share(
    df: pl.DataFrame,
    field: str,
    minutes: int,
    name: str,
) -> pl.DataFrame:
    if minutes <= 0:
        raise ValueError("minutes must be positive")
    work = _prepare(df, (field,)).with_columns(
        pl.col("datetime").max().over(KEYS).alias("_day_end")
    )
    tail = pl.col("datetime") > (pl.col("_day_end") - pl.duration(minutes=minutes))
    result = work.group_by(KEYS).agg(
        (
            pl.when(tail).then(pl.col(field)).otherwise(0.0).sum()
            / (pl.col(field).sum() + 1e-12)
        ).alias(name)
    )
    return _finish(result, name)


def intraday_peak_count(
    df: pl.DataFrame,
    field: str,
    rolling_window: int,
    threshold: float,
    name: str,
) -> pl.DataFrame:
    """Count local rolling z-score peaks independently within each trading day."""
    if rolling_window < 3 or threshold <= 0:
        raise ValueError("rolling_window >= 3 and threshold > 0 are required")
    work = _prepare(df, (field,))
    mean = pl.col(field).rolling_mean(rolling_window, min_samples=rolling_window).over(KEYS)
    std = pl.col(field).rolling_std(rolling_window, min_samples=rolling_window, ddof=0).over(KEYS)
    work = work.with_columns(((pl.col(field) - mean) / (std + 1e-12)).alias("_z"))
    result = work.group_by(KEYS).agg((pl.col("_z") >= threshold).sum().alias(name))
    return _finish(result, name)


def intraday_path_metric(df: pl.DataFrame, metric: str, name: str) -> pl.DataFrame:
    """Path-dependent daily metrics whose definitions are clearest in array form."""
    work = _prepare(df, ("open", "high", "low", "close", "volume", "turnover"))

    def calculate(group: pl.DataFrame) -> pl.DataFrame:
        close = group["close"].to_numpy()
        open_ = group["open"].to_numpy()
        high = group["high"].to_numpy()
        low = group["low"].to_numpy()
        volume = group["volume"].to_numpy()
        turnover = group["turnover"].to_numpy()
        log_return = np.diff(np.log(close), prepend=np.nan)
        simple_return = np.diff(close, prepend=np.nan) / np.roll(close, 1)
        simple_return[0] = np.nan
        valid_r = log_return[np.isfinite(log_return)]

        if metric == "bipower_variation":
            value = pi / 2 * np.sum(np.abs(valid_r[1:]) * np.abs(valid_r[:-1]))
        elif metric in {
            "tripower_variation", "jump_variation", "up_jump", "down_jump",
            "large_up_jump", "large_down_jump", "small_up_jump", "small_down_jump",
            "jump_asymmetry", "large_jump_asymmetry", "small_jump_asymmetry",
        }:
            mu = 2 ** (1 / 3) * gamma(5 / 6) / gamma(1 / 2)
            tripower = 0.0
            if valid_r.size >= 3:
                tripower = float(
                    np.sum(
                        np.abs(valid_r[2:]) ** (2 / 3)
                        * np.abs(valid_r[1:-1]) ** (2 / 3)
                        * np.abs(valid_r[:-2]) ** (2 / 3)
                    )
                    / mu**3
                )
            value = tripower
            if metric == "jump_variation":
                value = max(float(np.sum(valid_r**2)) - tripower, 0.0)
            elif metric != "tripower_variation":
                up = max(float(np.sum(valid_r[valid_r > 0] ** 2)) - 0.5 * tripower, 0.0)
                down = max(float(np.sum(valid_r[valid_r < 0] ** 2)) - 0.5 * tripower, 0.0)
                threshold = 4 * (1 / max(valid_r.size, 1)) ** 0.49 * np.sqrt(max(tripower, 0.0))
                large_up = min(up, float(np.sum(valid_r[valid_r > threshold] ** 2)))
                large_down = min(down, float(np.sum(valid_r[valid_r < -threshold] ** 2)))
                components = {
                    "up_jump": up,
                    "down_jump": down,
                    "large_up_jump": large_up,
                    "large_down_jump": large_down,
                    "small_up_jump": up - large_up,
                    "small_down_jump": down - large_down,
                    "jump_asymmetry": up - down,
                    "large_jump_asymmetry": large_up - large_down,
                    "small_jump_asymmetry": (up - large_up) - (down - large_down),
                }
                value = components[metric]
        elif metric == "jump_degree":
            valid = np.isfinite(simple_return) & np.isfinite(log_return)
            value = float(np.mean(2 * (simple_return[valid] - log_return[valid]) - log_return[valid] ** 2))
        elif metric == "max_rise":
            valid = simple_return[np.isfinite(simple_return)]
            count = max(1, int(np.ceil(valid.size * 0.1)))
            value = float(np.prod(1 + np.sort(valid)[-count:]))
        elif metric == "turnover_cv":
            value = float(np.std(turnover) / (np.mean(turnover) + 1e-12))
        elif metric == "price_elasticity":
            value = float(np.mean((high - low) / (turnover + 1e-12)))
        elif metric == "amount_entropy":
            share = turnover / (np.sum(turnover) + 1e-12)
            share = share[share > 0]
            value = float(-np.sum(share * np.log(share)))
        elif metric == "single_amount_entropy":
            probability = volume / (np.sum(volume) + 1e-12) * close / (np.sum(close) + 1e-12)
            probability = probability[probability > 0]
            value = float(-np.sum(probability * np.log(probability)))
        elif metric == "weighted_skew":
            weight = volume / (np.sum(volume) + 1e-12)
            centered = close - np.mean(close)
            value = float(np.sum(weight * centered**3) / (np.std(close) ** 3 + 1e-12))
        elif metric == "volume_ratio":
            times = group["datetime"].dt.time().to_list()
            morning = sum(v for t, v in zip(times, volume) if time(9, 30) <= t <= time(10, 0))
            afternoon = sum(v for t, v in zip(times, volume) if time(13, 0) <= t <= time(13, 30))
            value = float(morning / (afternoon + 1e-12))
        elif metric == "intraday_cvar":
            valid = np.isfinite(log_return)
            weighted = log_return[valid] * volume[valid] / (np.sum(volume[valid]) + 1e-12)
            cutoff = np.quantile(weighted, 0.05)
            value = float(np.mean(weighted[weighted <= cutoff]))
        elif metric == "trend_ratio":
            value = float((close[-1] - close[0]) / (np.sum(np.abs(np.diff(close))) + 1e-12))
        elif metric == "weighted_close_ratio":
            vwap = np.sum(close * volume) / (np.sum(volume) + 1e-12)
            value = float(vwap / (np.mean(close) + 1e-12))
        elif metric == "price_volume_corr":
            value = float(np.corrcoef(close, volume)[0, 1]) if np.std(close) and np.std(volume) else np.nan
        elif metric == "time_weighted_relative_price":
            price = (open_ + high + low + close) / 4
            value = float((np.mean(price) - np.min(low)) / (np.max(high) - np.min(low) + 1e-12))
        elif metric == "structured_reversal":
            count = max(1, int(np.ceil(volume.size * 0.1)))
            selected = np.argsort(volume)[:count]
            valid = selected[np.isfinite(simple_return[selected])]
            weights = volume[valid] / (np.sum(volume[valid]) + 1e-12)
            value = float(np.sum(simple_return[valid] * weights))
        elif metric == "golden_reversal":
            times = group["datetime"].dt.time().to_list()
            candidates = [i for i, t in enumerate(times) if t <= time(10, 0)]
            index = candidates[-1] if candidates else 0
            value = float(np.log(close[-1] / (close[index] + 1e-12)))
        elif metric == "volume_bucket_entropy":
            bins = min(20, volume.size)
            counts = np.histogram(volume, bins=bins)[0].astype(float)
            probability = counts[counts > 0] / (np.sum(counts) + 1e-12)
            value = float(-np.sum(probability * np.log(probability)))
        elif metric == "shortcut_illiquidity":
            value = float(np.sum((2 * (high - low) - np.abs(close - open_)) / (turnover + 1e-12)))
        elif metric == "tide_rate":
            neighborhood = np.convolve(volume, np.ones(9), mode="same")
            if volume.size <= 10:
                value = np.nan
            else:
                peak = int(np.argmax(neighborhood[4:-4])) + 4
                left = int(np.argmin(neighborhood[4:peak])) + 4 if peak > 4 else 0
                right = int(np.argmin(neighborhood[peak + 1 : -4])) + peak + 1 if peak + 5 < volume.size else volume.size - 1
                value = float((close[right] - close[left]) / (close[left] + 1e-12) / max(right - left, 1))
        elif metric in {"fuzzy_amount", "fuzzy_volume"}:
            volatility = np.full(close.size, np.nan)
            ambiguity = np.full(close.size, np.nan)
            for index in range(4, close.size):
                volatility[index] = np.nanstd(log_return[index - 4 : index + 1])
            for index in range(8, close.size):
                ambiguity[index] = np.nanstd(volatility[index - 4 : index + 1])
            valid = np.isfinite(ambiguity)
            fog = valid & (ambiguity > np.nanmean(ambiguity))
            measure = turnover if metric == "fuzzy_amount" else volume
            value = float(np.mean(measure[fog]) / (np.mean(measure[valid]) + 1e-12)) if fog.any() else np.nan
        elif metric == "calendar_cci":
            typical = (high + low + close) / 3
            center = float(np.mean(typical))
            avedev = float(np.mean(np.abs(typical - center)))
            value = float((typical[-1] - center) / (0.015 * avedev + 1e-12))
        elif metric == "calendar_rsi":
            delta = np.diff(close)
            value = float(100 * np.sum(np.maximum(delta, 0)) / (np.sum(np.abs(delta)) + 1e-12))
        elif metric == "calendar_br":
            previous_close = close[:-1]
            buy = np.maximum(high[1:] - previous_close, 0)
            # The source formula explicitly uses the previous minute's low in the denominator.
            sell = np.maximum(previous_close - low[:-1], 0)
            value = float(100 * np.sum(buy) / (np.sum(sell) + 1e-12))
        elif metric == "calendar_elder":
            alpha = 2 / 241
            ema = close[0]
            for item in close[1:]:
                ema = alpha * item + (1 - alpha) * ema
            bull = high[-1] - ema
            bear = low[-1] - ema
            value = float((bull - bear) / (close[-1] + 1e-12))
        elif metric == "calendar_psy":
            value = float(np.mean(np.diff(close) > 0))
        elif metric == "calendar_volume_ratio":
            delta = np.diff(close)
            minute_volume = volume[1:]
            up = np.sum(minute_volume[delta > 0])
            non_up = np.sum(minute_volume[delta <= 0])
            value = float(up / (non_up + 1e-12))
        elif metric == "calendar_chande":
            delta = np.diff(close)
            up = np.sum(np.maximum(delta, 0))
            down = np.sum(np.maximum(-delta, 0))
            value = float(100 * (up - down) / (up + down + 1e-12))
        elif metric == "calendar_srmi":
            delta = np.diff(close)
            scale = np.maximum(close[1:], close[:-1])
            value = float(np.mean(delta / (scale + 1e-12)))
        elif metric == "calendar_hurst_rs":
            centered = close - np.mean(close)
            cumulative = np.cumsum(centered)
            value = float((np.max(cumulative) - np.min(cumulative)) / (np.std(close) + 1e-12))
        elif metric == "calendar_price_mad":
            value = float(np.mean(np.abs(close - np.mean(close))))
        elif metric == "calendar_mfi":
            typical = (high + low + close) / 3
            money_flow = typical * volume
            direction = np.diff(typical)
            positive = np.sum(money_flow[1:][direction > 0])
            negative = np.sum(money_flow[1:][direction <= 0])
            ratio = positive / (negative + 1e-12)
            value = float(100 - 100 / (1 + ratio))
        elif metric == "calendar_money_flow":
            value = float(np.sum((high + low + close) / 3 * volume))
        elif metric == "calendar_pvt":
            value = float(np.nansum(simple_return * volume))
        elif metric == "calendar_emv":
            midpoint = (high + low) / 2
            midpoint_move = np.diff(midpoint)
            box_ratio = volume[1:] / (high[1:] - low[1:] + 1e-12)
            value = float(np.mean(midpoint_move / (box_ratio + 1e-12)))
        else:
            raise ValueError(f"unsupported intraday path metric: {metric}")
        return pl.DataFrame(
            {
                "trade_date": [group["trade_date"][0]],
                "vt_symbol": [group["vt_symbol"][0]],
                name: [value],
            }
        )

    return _finish(work.group_by(*KEYS).map_groups(calculate), name)


def intraday_semibeta(df: pl.DataFrame, stock_side: str, market_side: str, name: str) -> pl.DataFrame:
    """Daily realized semibeta from aligned stock and benchmark minute returns."""
    work = _prepare(df, ("close", "market_minute_ret")).with_columns(
        (pl.col("close") / (pl.col("close").shift(1).over(KEYS) + 1e-12) - 1).alias("_stock_ret")
    )
    stock = pl.col("_stock_ret")
    market = pl.col("market_minute_ret")
    stock_part = stock.clip(lower_bound=0) if stock_side == "positive" else stock.clip(upper_bound=0)
    market_part = market.clip(lower_bound=0) if market_side == "positive" else market.clip(upper_bound=0)
    sign = -1.0 if stock_side != market_side else 1.0
    result = work.group_by(KEYS).agg(
        (sign * (stock_part * market_part).sum() / (market.pow(2).sum() + 1e-12)).alias(name)
    )
    return _finish(result, name)


def intraday_autocorrelation_composite(
    minute: pl.DataFrame,
    daily: pl.DataFrame,
    mode: str,
    name: str,
    *,
    window: int = 20,
) -> pl.DataFrame:
    """Conditional intraday price/volume autocorrelation calendar factors."""
    field = "close" if mode.startswith("price") else "volume"
    double_difference = mode.endswith("double")
    work = _prepare(minute, (field,))

    def calculate(group: pl.DataFrame) -> pl.DataFrame:
        values = group[field].to_numpy()
        delta = np.diff(values)
        right = np.diff(values)[1:] if double_difference else values[2:]
        left = delta[:-1]
        positive = (left > 0) & (right > 0) if double_difference else left > 0
        negative = (left < 0) & (right < 0) if double_difference else left < 0

        def corr(mask: np.ndarray) -> float:
            return float(np.corrcoef(left[mask], right[mask])[0, 1]) if mask.sum() >= 3 else np.nan

        return pl.DataFrame({
            "trade_date": [group["trade_date"][0]], "vt_symbol": [group["vt_symbol"][0]],
            "_positive": [corr(positive)], "_negative": [corr(negative)],
        })

    raw = work.group_by(*KEYS).map_groups(calculate).with_columns(
        pl.col("trade_date").cast(pl.Datetime).alias("datetime")
    ).drop("trade_date").sort(["vt_symbol", "datetime"]).with_columns(
        pl.col("_positive").rolling_mean(window, min_samples=window).over("vt_symbol"),
        pl.col("_negative").rolling_mean(window, min_samples=window).over("vt_symbol"),
    )
    cap = "a_share_market_val_in_circulation"
    raw = raw.join(daily.select("datetime", "vt_symbol", cap), on=["datetime", "vt_symbol"]).with_columns(
        pl.col(cap).log().alias("_x")
    )
    for column in ("_positive", "_negative"):
        raw = raw.with_columns(
            pl.col("_x").mean().over("datetime").alias("_xm"),
            pl.col(column).mean().over("datetime").alias("_ym"),
        ).with_columns(
            (pl.col("_x") - pl.col("_xm")).alias("_xc"),
            (pl.col(column) - pl.col("_ym")).alias("_yc"),
        ).with_columns(
            ((pl.col("_xc") * pl.col("_yc")).mean().over("datetime") / (pl.col("_xc").pow(2).mean().over("datetime") + 1e-12)).alias("_beta")
        ).with_columns((pl.col("_yc") - pl.col("_beta") * pl.col("_xc")).alias(column))
    sign = 1.0 if double_difference else -1.0
    return raw.select(
        "datetime", "vt_symbol",
        (
            (pl.col("_positive") - pl.col("_positive").mean().over("datetime")) / (pl.col("_positive").std(ddof=0).over("datetime") + 1e-12)
            + sign * (pl.col("_negative") - pl.col("_negative").mean().over("datetime")) / (pl.col("_negative").std(ddof=0).over("datetime") + 1e-12)
        ).alias(name),
    ).sort(["datetime", "vt_symbol"])


def intraday_price_volume_composite(
    minute: pl.DataFrame,
    daily: pl.DataFrame,
    mode: str,
    name: str,
    *,
    window: int = 20,
) -> pl.DataFrame:
    """Composite price-volume correlation factors with stated cross-sectional residualization."""
    raw = _prepare(minute, ("close", "volume")).group_by(KEYS).agg(
        pl.corr("close", "volume").alias("_corr")
    ).with_columns(pl.col("trade_date").cast(pl.Datetime).alias("datetime")).drop("trade_date")
    raw = raw.sort(["vt_symbol", "datetime"]).with_columns(
        pl.col("_corr").rolling_mean(window, min_samples=window).over("vt_symbol").alias("_avg"),
        pl.col("_corr").rolling_std(window, min_samples=window, ddof=0).over("vt_symbol").alias("_std"),
        pl.col("_corr").rolling_map(
            lambda s: float(np.polyfit(np.arange(s.len()), s.to_numpy(), 1)[0]),
            window_size=window, min_samples=window,
        ).over("vt_symbol").alias("_trend"),
    ).join(
        daily.select("datetime", "vt_symbol", "close", "a_share_market_val_in_circulation"),
        on=["datetime", "vt_symbol"], how="left",
    ).sort(["vt_symbol", "datetime"]).with_columns(
        pl.col("a_share_market_val_in_circulation").log().alias("_cap"),
        (pl.col("close") / (pl.col("close").shift(window).over("vt_symbol") + 1e-12) - 1).alias("_reversal"),
    )

    def residualize(frame: pl.DataFrame, column: str, control: str) -> pl.DataFrame:
        return frame.with_columns(
            pl.col(column).mean().over("datetime").alias("_ym"),
            pl.col(control).mean().over("datetime").alias("_xm"),
        ).with_columns(
            (pl.col(column) - pl.col("_ym")).alias("_yc"),
            (pl.col(control) - pl.col("_xm")).alias("_xc"),
        ).with_columns(
            ((pl.col("_xc") * pl.col("_yc")).mean().over("datetime") / (pl.col("_xc").pow(2).mean().over("datetime") + 1e-12)).alias("_beta")
        ).with_columns((pl.col("_yc") - pl.col("_beta") * pl.col("_xc")).alias(column))

    raw = residualize(raw, "_avg", "_cap")
    raw = residualize(raw, "_std", "_cap")
    if mode == "cpv":
        raw = residualize(raw, "_avg", "_reversal")
        raw = residualize(raw, "_std", "_reversal")
    components = ["_avg", "_std"] if mode == "composite" else ["_avg", "_std", "_trend"]
    zscores = [
        (pl.col(column) - pl.col(column).mean().over("datetime"))
        / (pl.col(column).std(ddof=0).over("datetime") + 1e-12)
        for column in components
    ]
    value = zscores[0]
    for component in zscores[1:]:
        value = value + component
    return raw.select("datetime", "vt_symbol", value.alias(name)).sort(["datetime", "vt_symbol"])


def intraday_turnover_uniformity(
    minute: pl.DataFrame,
    daily: pl.DataFrame,
    name: str,
    *,
    window: int = 20,
) -> pl.DataFrame:
    raw = _prepare(minute, ("volume",)).group_by(KEYS).agg(
        pl.col("volume").std(ddof=0).alias("_volume_std")
    ).with_columns(pl.col("trade_date").cast(pl.Datetime).alias("datetime")).drop("trade_date")
    raw = raw.join(
        daily.select("datetime", "vt_symbol", "close", "a_share_market_val_in_circulation"),
        on=["datetime", "vt_symbol"], how="left",
    ).sort(["vt_symbol", "datetime"]).with_columns(
        (pl.col("_volume_std") * pl.col("close") / (pl.col("a_share_market_val_in_circulation") + 1e-12)).alias("_turnvol")
    )
    return raw.select(
        "datetime", "vt_symbol",
        (
            pl.col("_turnvol").rolling_std(window, min_samples=window, ddof=0).over("vt_symbol")
            / (pl.col("_turnvol").rolling_mean(window, min_samples=window).over("vt_symbol") + 1e-12)
        ).alias(name),
    ).sort(["datetime", "vt_symbol"])


def intraday_apm(minute: pl.DataFrame, daily: pl.DataFrame, name: str, *, window: int = 20) -> pl.DataFrame:
    """Morning-minus-afternoon residual t-stat, neutralized against 20-day momentum."""
    work = _prepare(minute, ("close", "market_minute_ret"))

    def daily_periods(group: pl.DataFrame) -> pl.DataFrame:
        times = group["datetime"].dt.time().to_list()
        morning = np.array([value <= time(11, 30) for value in times])
        afternoon = np.array([value >= time(13, 0) for value in times])
        close = group["close"].to_numpy()
        market = group["market_minute_ret"].to_numpy()

        def period(mask: np.ndarray) -> tuple[float, float]:
            indices = np.flatnonzero(mask)
            if indices.size < 2:
                return np.nan, np.nan
            stock_ret = close[indices[-1]] / (close[indices[0]] + 1e-12) - 1
            market_ret = np.prod(1 + market[indices][np.isfinite(market[indices])]) - 1
            return float(stock_ret), float(market_ret)

        am, market_am = period(morning)
        pm, market_pm = period(afternoon)
        return pl.DataFrame({
            "trade_date": [group["trade_date"][0]], "vt_symbol": [group["vt_symbol"][0]],
            "_am": [am], "_pm": [pm], "_market_am": [market_am], "_market_pm": [market_pm],
        })

    periods = work.group_by(*KEYS).map_groups(daily_periods).with_columns(
        pl.col("trade_date").cast(pl.Datetime).alias("datetime")
    ).drop("trade_date").sort(["vt_symbol", "datetime"])

    def rolling_stat(group: pl.DataFrame) -> pl.DataFrame:
        values = np.full(group.height, np.nan)
        for end in range(window - 1, group.height):
            sample = group.slice(end - window + 1, window)
            stock = np.column_stack([sample["_am"].to_numpy(), sample["_pm"].to_numpy()]).reshape(-1)
            market = np.column_stack([sample["_market_am"].to_numpy(), sample["_market_pm"].to_numpy()]).reshape(-1)
            valid = np.isfinite(stock) & np.isfinite(market)
            if valid.sum() < window:
                continue
            design = np.column_stack([np.ones(valid.sum()), market[valid]])
            residual = stock[valid] - design @ np.linalg.lstsq(design, stock[valid], rcond=None)[0]
            paired = residual.reshape(-1, 2)
            delta = paired[:, 0] - paired[:, 1]
            values[end] = np.mean(delta) / (np.std(delta, ddof=0) / np.sqrt(delta.size) + 1e-12)
        return group.select("datetime", "vt_symbol").with_columns(pl.Series("_stat", values).fill_nan(None))

    raw = periods.group_by("vt_symbol").map_groups(rolling_stat).join(
        daily.select("datetime", "vt_symbol", "close").sort(["vt_symbol", "datetime"]).with_columns(
            (pl.col("close") / (pl.col("close").shift(window).over("vt_symbol") + 1e-12) - 1).alias("_momentum")
        ).select("datetime", "vt_symbol", "_momentum"),
        on=["datetime", "vt_symbol"], how="left",
    ).with_columns(
        pl.col("_stat").mean().over("datetime").alias("_ym"),
        pl.col("_momentum").mean().over("datetime").alias("_xm"),
    ).with_columns(
        (pl.col("_stat") - pl.col("_ym")).alias("_yc"),
        (pl.col("_momentum") - pl.col("_xm")).alias("_xc"),
    ).with_columns(
        ((pl.col("_xc") * pl.col("_yc")).mean().over("datetime") / (pl.col("_xc").pow(2).mean().over("datetime") + 1e-12)).alias("_beta")
    )
    return raw.select("datetime", "vt_symbol", (pl.col("_yc") - pl.col("_beta") * pl.col("_xc")).alias(name)).sort(["datetime", "vt_symbol"])


def intraday_smart_money(minute: pl.DataFrame, name: str, *, window: int = 10) -> pl.DataFrame:
    work = _prepare(minute, ("close", "volume"))
    parts: list[pl.DataFrame] = []
    for group in work.partition_by("vt_symbol", maintain_order=True):
        dates = group["trade_date"].unique(maintain_order=True).to_list()
        values = []
        for end, date in enumerate(dates):
            start = max(0, end - window + 1)
            sample = group.filter(pl.col("trade_date").is_in(dates[start : end + 1]))
            close = sample["close"].to_numpy()
            volume = sample["volume"].to_numpy()
            returns = np.diff(close, prepend=np.nan) / np.roll(close, 1)
            returns[0] = np.nan
            score = np.abs(returns) / (volume**0.25 + 1e-12)
            order = np.argsort(np.nan_to_num(score, nan=-np.inf))[::-1]
            cumulative = np.cumsum(volume[order]) / (np.sum(volume) + 1e-12)
            selected = order[cumulative <= 0.2]
            if selected.size == 0:
                selected = order[:1]
            smart_vwap = np.sum(close[selected] * volume[selected]) / (np.sum(volume[selected]) + 1e-12)
            all_vwap = np.sum(close * volume) / (np.sum(volume) + 1e-12)
            values.append(smart_vwap / (all_vwap + 1e-12))
        parts.append(pl.DataFrame({
            "datetime": [datetime.combine(d, time.min) for d in dates],
            "vt_symbol": [group["vt_symbol"][0]] * len(dates), name: values,
        }).with_columns(pl.col("datetime").cast(pl.Datetime)))
    return pl.concat(parts).sort(["datetime", "vt_symbol"]) if parts else pl.DataFrame()


def intraday_fuzzy_spread(minute: pl.DataFrame, name: str) -> pl.DataFrame:
    amount = intraday_path_metric(minute, "fuzzy_amount", "_amount")
    volume = intraday_path_metric(minute, "fuzzy_volume", "_volume")
    work = amount.join(volume, on=["datetime", "vt_symbol"]).sort(["vt_symbol", "datetime"]).with_columns(
        (pl.col("_amount") - pl.col("_volume")).alias("_spread")
    ).with_columns(
        pl.col("_spread").rolling_std(10, min_samples=10, ddof=0).over("vt_symbol").alias("_std10")
    ).with_columns(
        pl.when(pl.col("_spread") < 0).then(pl.col("_spread")).otherwise(0.0).sum().over("datetime").alias("_s1"),
        pl.when(pl.col("_spread") < 0).then(pl.col("_spread") / (pl.col("_std10") + 1e-12)).otherwise(pl.col("_spread")).alias("_corrected"),
    ).with_columns(
        pl.when(pl.col("_corrected") < 0).then(pl.col("_corrected")).otherwise(0.0).sum().over("datetime").alias("_s2")
    ).with_columns(
        pl.when(pl.col("_corrected") < 0).then(pl.col("_corrected") / (pl.col("_s2") + 1e-12) * pl.col("_s1")).otherwise(pl.col("_corrected")).alias("_adjusted")
    )
    value = pl.col("_adjusted")
    return work.select(
        "datetime", "vt_symbol",
        (0.5 * (value.rolling_mean(20, min_samples=20).over("vt_symbol") + value.rolling_std(20, min_samples=20, ddof=0).over("vt_symbol"))).alias(name),
    ).sort(["datetime", "vt_symbol"])


def intraday_extreme_reversal(minute: pl.DataFrame, name: str, *, window: int = 20) -> pl.DataFrame:
    work = _prepare(minute, ("close",))

    def calculate(group: pl.DataFrame) -> pl.DataFrame:
        close = group["close"].to_numpy()
        returns = np.diff(np.log(close), prepend=np.nan)
        center = np.nanmedian(returns)
        index = int(np.nanargmax(np.abs(returns - center)))
        previous = returns[index - 1] if index > 0 else np.nan
        return pl.DataFrame({
            "trade_date": [group["trade_date"][0]], "vt_symbol": [group["vt_symbol"][0]],
            "_extreme": [returns[index]], "_previous": [previous],
        })

    raw = work.group_by(*KEYS).map_groups(calculate).with_columns(
        pl.col("trade_date").cast(pl.Datetime).alias("datetime")
    ).drop("trade_date").sort(["vt_symbol", "datetime"]).with_columns(
        pl.col("_extreme").rolling_mean(window, min_samples=window).over("vt_symbol"),
        pl.col("_previous").rolling_mean(window, min_samples=window).over("vt_symbol"),
    )
    return raw.select(
        "datetime", "vt_symbol",
        (pl.col("_extreme").rank(method="average").over("datetime") / pl.len().over("datetime")
         + pl.col("_previous").rank(method="average").over("datetime") / pl.len().over("datetime")).alias(name),
    ).sort(["datetime", "vt_symbol"])


def intraday_market_event_metric(minute: pl.DataFrame, metric: str, side: str, name: str) -> pl.DataFrame:
    work = _prepare(minute, ("close", "turnover")).with_columns(
        (pl.col("close") / (pl.col("close").shift(1).over(KEYS) + 1e-12) - 1).alias("_ret")
    ).with_columns(
        pl.col("_ret").mean().over("datetime").alias("_market_mean"),
        pl.col("_ret").std(ddof=0).over("datetime").alias("_market_std"),
        (pl.col("turnover") / (pl.col("turnover").sum().over("datetime") + 1e-12)).alias("_amount_share"),
    ).with_columns(pl.col("_amount_share").shift(1).over(KEYS).alias("_lag_amount_share"))

    def calculate(group: pl.DataFrame) -> pl.DataFrame:
        returns = group["_ret"].to_numpy()
        count = 5 if metric == "resonance" else 30
        order = np.argsort(np.nan_to_num(returns, nan=(-np.inf if side == "high" else np.inf)))
        selected = order[-count:] if side == "high" else order[:count]
        if metric == "resonance":
            first = float(np.nanmean(group["_market_mean"].to_numpy()[selected]))
            second = float(np.nanmean(group["_market_std"].to_numpy()[selected]))
        else:
            first = float(np.nansum(group["_lag_amount_share"].to_numpy()[selected]))
            second = float(np.nanmean(group["_amount_share"].to_numpy()))
        return pl.DataFrame({
            "trade_date": [group["trade_date"][0]], "vt_symbol": [group["vt_symbol"][0]],
            "_first": [first], "_second": [second],
        })

    raw = work.group_by(*KEYS).map_groups(calculate).with_columns(
        pl.col("trade_date").cast(pl.Datetime).alias("datetime")
    ).drop("trade_date")
    if metric == "resonance":
        rank = pl.col("_first").rank(method="average", descending=(side == "low")).over("datetime") / pl.len().over("datetime")
        value = rank / (pl.col("_second") + 1e-12)
    else:
        value = pl.col("_first") / (pl.col("_second") + 1e-12)
    return raw.select("datetime", "vt_symbol", value.alias(name)).sort(["datetime", "vt_symbol"])


def intraday_volume_diff_regression(minute: pl.DataFrame, metric: str, name: str) -> pl.DataFrame:
    work = _prepare(minute, ("close", "volume"))

    def calculate(group: pl.DataFrame) -> pl.DataFrame:
        close = group["close"].to_numpy()
        diff = np.diff(group["volume"].to_numpy(), prepend=np.nan)
        returns = np.diff(close, prepend=np.nan) / np.roll(close, 1)
        rows = []
        y = []
        for index in range(6, group.height):
            row = [1.0, *[diff[index - lag] for lag in range(6)]]
            if np.isfinite(row).all() and np.isfinite(returns[index]):
                rows.append(row)
                y.append(returns[index])
        t_intercept = lag_t_std = f_value = np.nan
        if len(y) > 10:
            design = np.asarray(rows)
            response = np.asarray(y)
            coef = np.linalg.lstsq(design, response, rcond=None)[0]
            residual = response - design @ coef
            dof = max(response.size - design.shape[1], 1)
            covariance = np.linalg.pinv(design.T @ design) * np.sum(residual**2) / dof
            tvalues = coef / (np.sqrt(np.diag(covariance)) + 1e-12)
            t_intercept = float(tvalues[0])
            lag_t_std = float(np.std(tvalues[2:7], ddof=0))
            total_ss = np.sum((response - np.mean(response)) ** 2)
            explained = max(total_ss - np.sum(residual**2), 0.0)
            f_value = float((explained / 6) / (np.sum(residual**2) / dof + 1e-12))
        return pl.DataFrame({
            "trade_date": [group["trade_date"][0]], "vt_symbol": [group["vt_symbol"][0]],
            "_t_intercept": [t_intercept], "_lag_t_std": [lag_t_std], "_f": [f_value],
        })

    raw = work.group_by(*KEYS).map_groups(calculate).with_columns(
        pl.col("trade_date").cast(pl.Datetime).alias("datetime")
    ).drop("trade_date").sort(["datetime", "vt_symbol"])
    if metric == "morning":
        return raw.sort(["vt_symbol", "datetime"]).select(
            "datetime", "vt_symbol",
            pl.col("_lag_t_std").rolling_mean(20, min_samples=20).over("vt_symbol").alias(name),
        )
    if metric == "noon":
        return raw.select(
            "datetime", "vt_symbol",
            pl.when(pl.col("_f") < pl.col("_f").mean().over("datetime"))
            .then(-pl.col("_t_intercept").abs()).otherwise(pl.col("_t_intercept").abs()).alias(name),
        )
    if metric == "night":
        dates = raw["datetime"].unique().sort().to_list()
        symbols = raw["vt_symbol"].unique().sort().to_list()
        lookup = {(r[0], r[1]): r[2] for r in raw.select("datetime", "vt_symbol", "_t_intercept").iter_rows()}
        rows = []
        for end, date in enumerate(dates):
            if end < 19:
                continue
            period = dates[end - 19 : end + 1]
            for symbol in symbols:
                own = np.array([lookup.get((day, symbol), np.nan) for day in period])
                correlations = []
                for peer in symbols:
                    if peer == symbol:
                        continue
                    other = np.array([lookup.get((day, peer), np.nan) for day in period])
                    valid = np.isfinite(own) & np.isfinite(other)
                    if valid.sum() >= 10 and np.std(own[valid]) and np.std(other[valid]):
                        correlations.append(abs(np.corrcoef(own[valid], other[valid])[0, 1]))
                rows.append((date, symbol, np.mean(correlations) if correlations else np.nan))
        return pl.DataFrame(rows, schema=["datetime", "vt_symbol", name], orient="row")
    raise ValueError(f"unsupported volume-diff regression metric: {metric}")


def intraday_volume_valley_price(minute: pl.DataFrame, name: str) -> pl.DataFrame:
    """Volume-valley VWAP ratio using the previous 20 days at the same minute slot."""
    work = _prepare(minute, ("close", "volume"))
    rows: list[tuple[datetime, str, float]] = []
    for symbol in work["vt_symbol"].unique().sort().to_list():
        stock = work.filter(pl.col("vt_symbol") == symbol)
        dates = stock["trade_date"].unique().sort().to_list()
        daily_rows = {date: stock.filter(pl.col("trade_date") == date) for date in dates}
        slot_history: dict[time, list[float]] = {}
        ratios: list[float] = []
        for date in dates:
            day = daily_rows[date]
            prices = day["close"].to_numpy()
            volumes = day["volume"].to_numpy()
            slots = [value.time() for value in day["datetime"].to_list()]
            valley = np.zeros(day.height, dtype=bool)
            for index, (slot, volume) in enumerate(zip(slots, volumes, strict=True)):
                history = slot_history.get(slot, [])
                if len(history) >= 20:
                    sample = np.asarray(history[-20:], dtype=float)
                    valley[index] = volume <= np.nanmean(sample) + np.nanstd(sample)
            total_volume = np.nansum(volumes)
            valley_volume = np.nansum(volumes[valley])
            ratio = np.nan
            if total_volume > 0 and valley_volume > 0:
                total_vwap = np.nansum(prices * volumes) / total_volume
                valley_vwap = np.nansum(prices[valley] * volumes[valley]) / valley_volume
                ratio = valley_vwap / (total_vwap + 1e-12)
            ratios.append(float(ratio))
            value = float(np.nanmean(ratios[-20:])) if len(ratios) >= 20 and np.isfinite(ratios[-20:]).all() else np.nan
            rows.append((datetime.combine(date, time()), symbol, value))
            for slot, volume in zip(slots, volumes, strict=True):
                slot_history.setdefault(slot, []).append(float(volume))
    return pl.DataFrame(rows, schema=["datetime", "vt_symbol", name], orient="row").with_columns(
        pl.col(name).fill_nan(None)
    ).sort(["datetime", "vt_symbol"])


def _neutralize_cap_industry(frame: pl.DataFrame, name: str) -> pl.DataFrame:
    """Cross-sectional OLS residual against log market cap and industry fixed effects."""
    def neutralize(group: pl.DataFrame) -> pl.DataFrame:
        values = group[name].to_numpy()
        cap = group["_cap"].to_numpy()
        industries = [str(value) if value is not None else "__missing__" for value in group["_industry"]]
        valid = np.isfinite(values) & np.isfinite(cap) & (cap > 0)
        residual = np.full(group.height, np.nan)
        if valid.sum() >= 3:
            levels = sorted(set(np.asarray(industries)[valid]))
            columns = [np.ones(valid.sum()), np.log(cap[valid])]
            columns.extend((np.asarray(industries)[valid] == level).astype(float) for level in levels[1:])
            design = np.column_stack(columns)
            coefficient = np.linalg.lstsq(design, values[valid], rcond=None)[0]
            residual[valid] = values[valid] - design @ coefficient
        return group.select("datetime", "vt_symbol").with_columns(
            pl.Series(name, residual).fill_nan(None)
        )

    return frame.group_by("datetime", maintain_order=True).map_groups(neutralize).sort(
        ["datetime", "vt_symbol"]
    )


def intraday_trend_fund_metric(
    minute: pl.DataFrame,
    daily: pl.DataFrame,
    metric: str,
    name: str,
) -> pl.DataFrame:
    """Monthly trend-fund, net-support, and extreme-following behavior factors."""
    work = _prepare(minute, ("close", "volume", "turnover"))
    required_daily = {
        "datetime", "vt_symbol", "close", "a_share_market_val_in_circulation",
        "first_industry_code",
    }
    missing = sorted(required_daily.difference(daily.columns))
    if missing:
        raise ValueError(f"trend-fund factor daily data missing columns: {missing}")
    metadata = daily.select(
        pl.col("datetime").cast(pl.Date).alias("trade_date"), "vt_symbol",
        pl.col("close").cast(pl.Float64).alias("_daily_close"),
        pl.col("a_share_market_val_in_circulation").cast(pl.Float64).alias("_cap"),
        pl.col("first_industry_code").alias("_industry"),
    ).unique(["trade_date", "vt_symbol"], keep="last")

    raw_rows: list[tuple[datetime, str, float]] = []
    for symbol in work["vt_symbol"].unique().sort().to_list():
        stock = work.filter(pl.col("vt_symbol") == symbol)
        dates = stock["trade_date"].unique().sort().to_list()
        histories: list[np.ndarray] = []
        for date in dates:
            day = stock.filter(pl.col("trade_date") == date)
            prices = day["close"].to_numpy()
            volumes = day["volume"].to_numpy()
            amounts = day["turnover"].to_numpy()
            threshold = np.nan
            if len(histories) >= 5:
                threshold = float(np.nanquantile(np.concatenate(histories[-5:]), 0.9))
            trend = volumes > threshold
            value = np.nan
            if metric == "relative_vwap" and trend.any():
                trend_vwap = np.nansum(prices[trend] * volumes[trend]) / (np.nansum(volumes[trend]) + 1e-12)
                all_vwap = np.nansum(prices * volumes) / (np.nansum(volumes) + 1e-12)
                value = trend_vwap / (all_vwap + 1e-12) - 1
            elif metric in {"net_support", "trend_net_support"}:
                selected = trend if metric == "trend_net_support" else np.ones(day.height, dtype=bool)
                if selected.any():
                    center = np.nanmean(prices[selected])
                    support = np.nansum(volumes[selected & (prices < center)])
                    resistance = np.nansum(volumes[selected & (prices > center)])
                    meta = metadata.filter(
                        (pl.col("trade_date") == date) & (pl.col("vt_symbol") == symbol)
                    )
                    if meta.height:
                        float_shares = meta["_cap"][0] / (meta["_daily_close"][0] + 1e-12)
                        value = (support - resistance) / (float_shares + 1e-12)
            elif metric in {"follower_ratio", "follower_corr"} and trend.any():
                leaders: list[float] = []
                followers: list[float] = []
                for index in np.flatnonzero(trend):
                    following = amounts[index + 1 : index + 6]
                    if following.size and np.isfinite(following).any() and amounts[index] > 0:
                        leaders.append(float(amounts[index]))
                        followers.append(float(np.nanmax(following)))
                if metric == "follower_ratio" and leaders:
                    value = float(np.mean(np.asarray(followers) / (np.asarray(leaders) + 1e-12)))
                elif metric == "follower_corr" and len(leaders) >= 2:
                    if np.std(leaders) > 1e-15 and np.std(followers) > 1e-15:
                        value = float(np.corrcoef(leaders, followers)[0, 1])
            elif metric not in {
                "relative_vwap", "net_support", "trend_net_support", "follower_ratio", "follower_corr"
            }:
                raise ValueError(f"unsupported trend-fund metric: {metric}")
            raw_rows.append((datetime.combine(date, time()), symbol, float(value)))
            histories.append(volumes.astype(float))

    raw = pl.DataFrame(raw_rows, schema=["datetime", "vt_symbol", "_raw"], orient="row").sort(
        ["vt_symbol", "datetime"]
    ).with_columns(
        pl.col("_raw").fill_nan(None).rolling_mean(20, min_samples=20).over("vt_symbol").alias(name)
    ).with_columns(
        pl.col("datetime").dt.year().alias("_year"),
        pl.col("datetime").dt.month().alias("_month"),
    ).filter(
        pl.col("datetime") == pl.col("datetime").max().over("_year", "_month")
    ).join(
        metadata.with_columns(pl.col("trade_date").cast(pl.Datetime).alias("datetime")).select(
            "datetime", "vt_symbol", "_cap", "_industry"
        ),
        on=["datetime", "vt_symbol"], how="left",
    )
    return _neutralize_cap_industry(raw, name)
