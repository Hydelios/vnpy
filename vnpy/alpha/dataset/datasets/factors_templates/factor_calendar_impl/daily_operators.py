from __future__ import annotations

import polars as pl
import numpy as np


KEYS = ["datetime", "vt_symbol"]


def _prepare(df: pl.DataFrame, fields: tuple[str, ...]) -> pl.DataFrame:
    missing = sorted({"datetime", "vt_symbol", *fields}.difference(df.columns))
    if missing:
        raise ValueError(f"daily data missing columns: {missing}")
    return df.sort(["vt_symbol", "datetime"])


def daily_liquidity_metric(
    df: pl.DataFrame,
    metric: str,
    name: str,
    *,
    window: int = 20,
    long_window: int = 250,
) -> pl.DataFrame:
    """Daily OHLCV liquidity measures used by the calendar definitions."""
    work = _prepare(
        df,
        ("close", "volume", "turnover", "a_share_market_val_in_circulation"),
    ).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("vt_symbol") - 1).alias("_ret"),
        (
            pl.col("volume") * pl.col("close")
            / (pl.col("a_share_market_val_in_circulation") + 1e-12)
        ).alias("_turnover_rate"),
    ).with_columns(
        (pl.col("_ret").abs() / (pl.col("turnover") + 1e-12)).alias("_illiq")
    )

    def mean(column: str, n: int = window) -> pl.Expr:
        return pl.col(column).rolling_mean(n, min_samples=n).over("vt_symbol")

    def std(column: str, n: int = window) -> pl.Expr:
        return pl.col(column).rolling_std(n, min_samples=n, ddof=0).over("vt_symbol")

    expressions = {
        "turnover_mean": mean("_turnover_rate"),
        "turnover_std": std("_turnover_rate"),
        "turnover_cv": std("_turnover_rate") / (mean("_turnover_rate") + 1e-12),
        "turnover_abnormal": mean("_turnover_rate") / (mean("_turnover_rate", long_window) + 1e-12),
        "amount_mean": mean("turnover"),
        "amount_std": std("turnover"),
        "amount_mean_over_std": mean("turnover") / (std("turnover") + 1e-12),
        "amount_change": pl.col("turnover") / (pl.col("turnover").shift(window).over("vt_symbol") + 1e-12) - 1,
        "amihud": mean("_illiq"),
        "amihud_cv": std("_illiq") / (mean("_illiq") + 1e-12),
        "negative_amihud": (
            pl.when(pl.col("_ret") < 0).then(pl.col("_illiq")).otherwise(None)
            .rolling_mean(window, min_samples=max(5, window // 2))
            .over("vt_symbol")
        ),
    }
    if metric not in expressions:
        raise ValueError(f"unsupported daily liquidity metric: {metric}")
    return work.select("datetime", "vt_symbol", expressions[metric].alias(name))


def daily_rolling_risk(
    df: pl.DataFrame,
    metric: str,
    name: str,
    *,
    window: int = 252,
    min_samples: int | None = None,
) -> pl.DataFrame:
    """Rolling return-risk and single-market-factor regression measures."""
    work = _prepare(df, ("close", "market_ret"))
    min_samples = min_samples or max(20, window // 2)

    def calculate(group: pl.DataFrame) -> pl.DataFrame:
        close = group["close"].to_numpy()
        stock = np.diff(np.log(close), prepend=np.nan)
        market = group["market_ret"].to_numpy()
        values = np.full(group.height, np.nan)
        for end in range(group.height):
            start = max(0, end - window + 1)
            y = stock[start : end + 1]
            x = market[start : end + 1]
            valid = np.isfinite(y) & np.isfinite(x)
            if valid.sum() < min_samples:
                continue
            y = y[valid]
            x = x[valid]
            if metric in {"var", "cvar"}:
                cutoff = np.quantile(y, 0.05)
                values[end] = cutoff if metric == "var" else np.mean(y[y <= cutoff])
                continue
            if metric == "tail_beta":
                cutoff = np.quantile(x, 0.1)
                tail = x <= cutoff
                y, x = y[tail], x[tail]
            elif metric == "downside_beta":
                tail = x < 0
                y, x = y[tail], x[tail]
            if y.size < 5 or np.var(x) <= 1e-18:
                continue
            if metric == "coskew_zhu":
                yc = y - np.mean(y)
                xc = x - np.mean(x)
                values[end] = np.sum(yc * xc**2) / (np.sum(xc**3) + 1e-18)
                continue
            design = np.column_stack([np.ones(x.size), x])
            coef = np.linalg.lstsq(design, y, rcond=None)[0]
            residual = y - design @ coef
            if metric in {"beta", "downside_beta", "tail_beta"}:
                values[end] = coef[1]
            elif metric == "idio_vol":
                values[end] = np.std(residual)
            elif metric == "idio_skew":
                sigma = np.std(residual)
                values[end] = np.mean((residual - np.mean(residual)) ** 3) / (sigma**3 + 1e-18)
            elif metric == "idiosyncrasy":
                # Calendar definition is 1-R^2, i.e. unexplained variance share.
                values[end] = np.var(residual) / (np.var(y) + 1e-18)
            elif metric == "coskew":
                yc = y - np.mean(y)
                xc = x - np.mean(x)
                values[end] = np.mean(yc * xc**2) / (
                    np.std(yc) * np.var(xc) + 1e-18
                )
            else:
                raise ValueError(f"unsupported daily rolling risk metric: {metric}")
        return group.select("datetime", "vt_symbol").with_columns(
            pl.Series(name, values).fill_nan(None)
        )

    return work.group_by("vt_symbol").map_groups(calculate).sort(KEYS)


def daily_price_acceleration(
    df: pl.DataFrame,
    name: str,
    *,
    window: int = 60,
) -> pl.DataFrame:
    """Quadratic coefficient from rolling price ~ 1 + t + t^2 regressions."""
    work = _prepare(df, ("close",))

    def calculate(group: pl.DataFrame) -> pl.DataFrame:
        close = group["close"].to_numpy()
        values = np.full(group.height, np.nan)
        design = np.column_stack([np.ones(window), np.arange(1, window + 1), np.arange(1, window + 1) ** 2])
        for end in range(window - 1, group.height):
            y = close[end - window + 1 : end + 1]
            if np.isfinite(y).all():
                values[end] = np.linalg.lstsq(design, y, rcond=None)[0][2]
        return group.select("datetime", "vt_symbol").with_columns(
            pl.Series(name, values).fill_nan(None)
        )

    return work.group_by("vt_symbol").map_groups(calculate).sort(KEYS)


def daily_ideal_amplitude(
    df: pl.DataFrame,
    name: str,
    *,
    window: int = 20,
    fraction: float = 0.25,
) -> pl.DataFrame:
    """Mean amplitude of high-price days minus that of low-price days."""
    work = _prepare(df, ("high", "low", "close"))

    def calculate(group: pl.DataFrame) -> pl.DataFrame:
        close = group["close"].to_numpy()
        amplitude = group["high"].to_numpy() / (group["low"].to_numpy() + 1e-12) - 1
        values = np.full(group.height, np.nan)
        count = max(1, int(np.ceil(window * fraction)))
        for end in range(window - 1, group.height):
            start = end - window + 1
            order = np.argsort(close[start : end + 1])
            sample = amplitude[start : end + 1]
            values[end] = np.mean(sample[order[-count:]]) - np.mean(sample[order[:count]])
        return group.select("datetime", "vt_symbol").with_columns(
            pl.Series(name, values).fill_nan(None)
        )

    return work.group_by("vt_symbol").map_groups(calculate).sort(KEYS)


def daily_technical_metric(
    df: pl.DataFrame,
    metric: str,
    name: str,
    *,
    window: int = 20,
) -> pl.DataFrame:
    """Technical indicators that require state or array-based rolling selection."""
    work = _prepare(df, ("open", "high", "low", "close", "volume"))

    def calculate(group: pl.DataFrame) -> pl.DataFrame:
        high = group["high"].to_numpy()
        low = group["low"].to_numpy()
        close = group["close"].to_numpy()
        volume = group["volume"].to_numpy()
        returns = np.diff(np.log(close), prepend=np.nan)
        values = np.full(group.height, np.nan)
        if metric in {"pvi", "nvi"}:
            values[0] = 1000.0
            for index in range(1, group.height):
                selected = volume[index] > volume[index - 1] if metric == "pvi" else volume[index] < volume[index - 1]
                change = (close[index] - close[index - 1]) / (close[index - 1] + 1e-12)
                # Preserve the sign printed in each source definition.
                values[index] = values[index - 1] * (1 + change if metric == "pvi" else 1 - change) if selected else values[index - 1]
        elif metric == "asi":
            si = np.full(group.height, np.nan)
            open_ = group["open"].to_numpy()
            for index in range(1, group.height):
                a = abs(high[index] - close[index - 1])
                b = abs(low[index] - close[index - 1])
                c = abs(high[index] - low[index - 1])
                d = abs(close[index - 1] - open_[index - 1])
                x = close[index] - close[index - 1] + 0.5 * (close[index] - open_[index]) + close[index - 1] - open_[index - 1]
                k = max(a, b)
                r = a + 0.5 * b + 0.25 * d if a > b and a > c else (b + 0.5 * a + 0.25 * d if b > a and b > c else c + 0.25 * d)
                si[index] = 16 * x / (r + 1e-12) * k
            for end in range(window - 1, group.height):
                values[end] = np.nansum(si[end - window + 1 : end + 1])
        elif metric == "kvo":
            trend = np.ones(group.height)
            typical = high + low + close
            trend[1:] = np.where(typical[1:] > typical[:-1], 1.0, -1.0)
            dm = high - low
            cm = np.copy(dm)
            for index in range(1, group.height):
                cm[index] = cm[index - 1] + dm[index] if trend[index] == trend[index - 1] else dm[index - 1] + dm[index]
            vf = volume * np.abs(2 * (dm / (cm + 1e-12) - 1)) * trend * 100

            def ema(values_: np.ndarray, span: int) -> np.ndarray:
                out = np.full(values_.size, np.nan)
                alpha = 2 / (span + 1)
                out[0] = values_[0]
                for idx in range(1, values_.size):
                    out[idx] = alpha * values_[idx] + (1 - alpha) * out[idx - 1]
                return out

            values = ema(vf, 34) - ema(vf, 55)
        else:
            for end in range(window - 1, group.height):
                start = end - window + 1
                if metric == "duvol":
                    sample = returns[start : end + 1]
                    sample = sample[np.isfinite(sample)]
                    mean = np.mean(sample)
                    up, down = sample[sample > mean], sample[sample < mean]
                    if up.size > 1 and down.size > 1:
                        values[end] = np.log(
                            ((up.size - 1) * np.sum((down - mean) ** 2))
                            / ((down.size - 1) * np.sum((up - mean) ** 2) + 1e-18)
                        )
                elif metric == "aroon":
                    high_pos = int(np.argmax(high[start : end + 1]))
                    low_pos = int(np.argmin(low[start : end + 1]))
                    values[end] = (high_pos - low_pos) / window * 100
                else:
                    raise ValueError(f"unsupported daily technical metric: {metric}")
        return group.select("datetime", "vt_symbol").with_columns(pl.Series(name, values).fill_nan(None))

    return work.group_by("vt_symbol").map_groups(calculate).sort(KEYS)


def daily_frazzini_pedersen_beta(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """FP beta: 5-year 3-day correlation times 12-month volatility ratio."""
    work = _prepare(df, ("close", "market_ret"))

    def calculate(group: pl.DataFrame) -> pl.DataFrame:
        stock = np.diff(np.log(group["close"].to_numpy()), prepend=np.nan)
        market = group["market_ret"].to_numpy()
        stock3 = np.full(stock.size, np.nan)
        market3 = np.full(market.size, np.nan)
        for index in range(2, stock.size):
            if np.isfinite(stock[index - 2 : index + 1]).all():
                stock3[index] = np.sum(stock[index - 2 : index + 1])
            if np.isfinite(market[index - 2 : index + 1]).all():
                market3[index] = np.sum(market[index - 2 : index + 1])
        values = np.full(group.height, np.nan)
        for end in range(group.height):
            corr_start = max(0, end - 1260 + 1)
            vol_start = max(0, end - 252 + 1)
            corr_valid = np.isfinite(stock3[corr_start : end + 1]) & np.isfinite(market3[corr_start : end + 1])
            vol_valid = np.isfinite(stock[vol_start : end + 1]) & np.isfinite(market[vol_start : end + 1])
            if corr_valid.sum() < 750 or vol_valid.sum() < 120:
                continue
            rho = np.corrcoef(stock3[corr_start : end + 1][corr_valid], market3[corr_start : end + 1][corr_valid])[0, 1]
            sigma_stock = np.std(stock[vol_start : end + 1][vol_valid])
            sigma_market = np.std(market[vol_start : end + 1][vol_valid])
            values[end] = rho * sigma_stock / (sigma_market + 1e-18)
        return group.select("datetime", "vt_symbol").with_columns(pl.Series(name, values).fill_nan(None))

    return work.group_by("vt_symbol").map_groups(calculate).sort(KEYS)


def daily_capital_gain_overhang(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Reference-price CGO using historical turnover survival weights only."""
    work = _prepare(df, ("close", "volume", "a_share_market_val_in_circulation"))

    def calculate(group: pl.DataFrame) -> pl.DataFrame:
        close = group["close"].to_numpy()
        volume = group["volume"].to_numpy()
        cap = group["a_share_market_val_in_circulation"].to_numpy()
        rates = np.clip(volume * close / (cap + 1e-12), 0, 1)
        values = np.full(group.height, np.nan)
        weighted_price = weight = 0.0
        for index in range(group.height):
            if weight > 1e-18 and index > 0:
                reference = weighted_price / weight
                values[index] = (close[index - 1] - reference) / (close[index - 1] + 1e-12)
            weighted_price = weighted_price * (1 - rates[index]) + rates[index] * close[index]
            weight = weight * (1 - rates[index]) + rates[index]
        return group.select("datetime", "vt_symbol").with_columns(pl.Series(name, values).fill_nan(None))

    return work.group_by("vt_symbol").map_groups(calculate).sort(KEYS)


def daily_liquidity_shock(df: pl.DataFrame, metric: str, name: str) -> pl.DataFrame:
    """Monthly Amihud liquidity shock, simple or rolling ARMA(1,1)-conditional."""
    work = _prepare(df, ("close", "turnover")).with_columns(
        (pl.col("close").log() - pl.col("close").log().shift(1).over("vt_symbol")).abs().alias("_abs_ret"),
        pl.col("datetime").dt.strftime("%Y-%m").alias("_month"),
    ).with_columns((pl.col("_abs_ret") / (pl.col("turnover") + 1e-12)).alias("_illiq"))
    monthly = work.group_by("vt_symbol", "_month").agg(
        pl.col("datetime").max().alias("datetime"), pl.col("_illiq").mean().alias("_illiq")
    ).sort(["vt_symbol", "datetime"])

    def calculate(group: pl.DataFrame) -> pl.DataFrame:
        series = group["_illiq"].to_numpy()
        values = np.full(group.height, np.nan)
        if metric == "simple":
            for end in range(12, group.height):
                values[end] = -(series[end] - np.mean(series[end - 12 : end]))
        elif metric == "conditional":
            from scipy.optimize import minimize

            for end in range(23, group.height):
                sample = series[max(0, end - 59) : end + 1]
                if not np.isfinite(sample).all():
                    continue

                def residuals(params: np.ndarray) -> np.ndarray:
                    intercept, phi, theta = params
                    errors = np.zeros(sample.size)
                    for index in range(1, sample.size):
                        errors[index] = sample[index] - intercept - phi * sample[index - 1] - theta * errors[index - 1]
                    return errors

                initial = np.array([np.mean(sample) * 0.1, 0.5, 0.0])
                fitted = minimize(
                    lambda params: float(np.sum(residuals(params)[1:] ** 2)), initial,
                    method="L-BFGS-B", bounds=[(None, None), (-0.99, 0.99), (-0.99, 0.99)],
                )
                values[end] = -residuals(fitted.x)[-1]
        else:
            raise ValueError(f"unsupported liquidity shock metric: {metric}")
        return group.select("datetime", "vt_symbol").with_columns(pl.Series(name, values).fill_nan(None))

    return monthly.group_by("vt_symbol").map_groups(calculate).sort(KEYS)


def daily_panic_metric(
    daily: pl.DataFrame,
    minute: pl.DataFrame | None,
    metric: str,
    name: str,
    *,
    numerator_offset: bool = False,
) -> pl.DataFrame:
    work = _prepare(daily, ("close", "market_ret")).with_columns(
        (pl.col("close") / (pl.col("close").shift(1).over("vt_symbol") + 1e-12) - 1).alias("_ret")
    )
    numerator = (pl.col("_ret") - pl.col("market_ret")).abs() + (0.1 if numerator_offset else 0.0)
    denominator = pl.col("_ret").abs() + pl.col("market_ret").abs() + (0.0 if numerator_offset else 0.1)
    work = work.with_columns((numerator / (denominator + 1e-12)).alias("_panic"))
    if metric == "volatility":
        if minute is None:
            raise ValueError(f"{name}: minute data required")
        intraday = _prepare(minute, ("close",)).with_columns(
            pl.col("datetime").cast(pl.Date).alias("trade_date"),
        ).sort(["vt_symbol", "datetime"]).with_columns(
            (pl.col("close").log() - pl.col("close").log().shift(1).over("trade_date", "vt_symbol"))
            .alias("_minute_ret")
        ).group_by("trade_date", "vt_symbol").agg(
            pl.col("_minute_ret").std(ddof=0).alias("_intraday_vol")
        ).with_columns(
            pl.col("trade_date").cast(pl.Datetime).alias("datetime")
        ).drop("trade_date")
        work = work.join(intraday, on=["datetime", "vt_symbol"], how="left").with_columns(
            (pl.col("_panic") * pl.col("_ret") * pl.col("_intraday_vol")).alias("_weighted")
        )
    elif metric == "decay":
        decayed = pl.col("_panic") - (
            pl.col("_panic").shift(1).over("vt_symbol") + pl.col("_panic").shift(2).over("vt_symbol")
        ) / 2
        work = work.with_columns(
            pl.when(decayed >= 0).then(decayed * pl.col("_ret")).otherwise(None).alias("_weighted")
        )
    elif metric == "base":
        work = work.with_columns((pl.col("_panic") * pl.col("_ret")).alias("_weighted"))
    else:
        raise ValueError(f"unsupported panic metric: {metric}")
    weighted = pl.col("_weighted")
    return work.select(
        "datetime", "vt_symbol",
        (0.5 * (
            weighted.rolling_mean(20, min_samples=20).over("vt_symbol")
            + weighted.rolling_std(20, min_samples=20, ddof=0).over("vt_symbol")
        )).alias(name),
    )


def daily_reversal_frequency(df: pl.DataFrame, direction: str, abnormal: bool, name: str) -> pl.DataFrame:
    work = _prepare(df, ("open", "close")).with_columns(
        (pl.col("open") / (pl.col("close").shift(1).over("vt_symbol") + 1e-12) - 1).alias("_overnight"),
        (pl.col("close") / (pl.col("open") + 1e-12) - 1).alias("_intraday"),
        pl.col("datetime").dt.strftime("%Y-%m").alias("_month"),
    )
    condition = (
        (pl.col("_overnight") < 0) & (pl.col("_intraday") > 0)
        if direction == "positive"
        else (pl.col("_overnight") > 0) & (pl.col("_intraday") < 0)
    )
    monthly = work.group_by("vt_symbol", "_month").agg(
        pl.col("datetime").max().alias("datetime"), condition.cast(pl.Float64).mean().alias("_frequency")
    ).sort(["vt_symbol", "datetime"])
    value = pl.col("_frequency")
    if abnormal:
        value = value / (value.rolling_mean(12, min_samples=12).over("vt_symbol") + 1e-12)
    return monthly.select("datetime", "vt_symbol", value.alias(name)).sort(KEYS)


def daily_gamma_illiquidity(df: pl.DataFrame, name: str) -> pl.DataFrame:
    work = _prepare(df, ("close", "turnover", "market_ret")).with_columns(
        (pl.col("close") / (pl.col("close").shift(1).over("vt_symbol") + 1e-12) - 1).alias("_ret"),
        pl.col("datetime").dt.strftime("%Y-%m").alias("_month"),
    ).with_columns((pl.col("_ret") - pl.col("market_ret")).alias("_excess"))

    def calculate(group: pl.DataFrame) -> pl.DataFrame:
        ret = group["_ret"].to_numpy()
        excess = group["_excess"].to_numpy()
        amount = group["turnover"].to_numpy()
        y = excess[1:]
        design = np.column_stack([np.ones(max(group.height - 1, 0)), ret[:-1], np.sign(excess[:-1]) * amount[:-1]])
        valid = np.isfinite(y) & np.isfinite(design).all(axis=1)
        value = np.nan
        if valid.sum() >= 10:
            gamma_value = np.linalg.lstsq(design[valid], y[valid], rcond=None)[0][2]
            value = -abs(float(gamma_value))
        return pl.DataFrame({"datetime": [group["datetime"].max()], "vt_symbol": [group["vt_symbol"][0]], name: [value]})

    return work.group_by("vt_symbol", "_month").map_groups(calculate).sort(KEYS)


def daily_salience_return(df: pl.DataFrame, name: str, *, variant: str = "standard") -> pl.DataFrame:
    work = _prepare(df, ("close",)).with_columns(
        (pl.col("close") / (pl.col("close").shift(1).over("vt_symbol") + 1e-12) - 1).alias("_ret"),
        pl.col("datetime").dt.strftime("%Y-%m").alias("_month"),
    ).with_columns(pl.col("_ret").median().over("datetime").alias("_market_median"))
    if variant == "standard":
        salience = (pl.col("_ret") - pl.col("_market_median")).abs() / (
            pl.col("_ret").abs() + pl.col("_market_median").abs() + 0.1
        )
    elif variant == "2026":
        salience = (
            pl.col("_ret").abs() + pl.col("_market_median").abs()
            - pl.col("_ret") - pl.col("_market_median")
        ) / 0.1
    else:
        raise ValueError(f"unsupported salience variant: {variant}")
    work = work.with_columns(salience.alias("_salience"))

    def calculate(group: pl.DataFrame) -> pl.DataFrame:
        returns = group["_ret"].to_numpy()
        salient = group["_salience"].to_numpy()
        valid = np.isfinite(returns) & np.isfinite(salient)
        value = np.nan
        if valid.sum() >= 5:
            order = np.argsort(-salient[valid])
            ranks = np.empty(order.size, dtype=float)
            ranks[order] = np.arange(1, order.size + 1)
            weights = 0.7**ranks
            weights = weights / (np.sum(weights) / weights.size + 1e-12)
            value = float(np.mean((weights - np.mean(weights)) * (returns[valid] - np.mean(returns[valid]))))
        return pl.DataFrame({"datetime": [group["datetime"].max()], "vt_symbol": [group["vt_symbol"][0]], name: [value]})

    return work.group_by("vt_symbol", "_month").map_groups(calculate).sort(KEYS)


def daily_trend_clarity(df: pl.DataFrame, metric: str, name: str) -> pl.DataFrame:
    work = _prepare(df, ("close",))

    def calculate(group: pl.DataFrame) -> pl.DataFrame:
        close = group["close"].to_numpy()
        clarity = np.full(group.height, np.nan)
        momentum = np.full(group.height, np.nan)
        x = np.arange(221, dtype=float)
        design = np.column_stack([np.ones(221), x])
        for end in range(240, group.height):
            y = close[end - 240 : end - 19]
            fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
            clarity[end] = 1 - np.sum((y - fitted) ** 2) / (np.sum((y - np.mean(y)) ** 2) + 1e-18)
            momentum[end] = close[end - 20] / (close[end - 240] + 1e-12) - 1
        return group.select("datetime", "vt_symbol").with_columns(
            pl.Series("_clarity", clarity).fill_nan(None), pl.Series("_momentum", momentum).fill_nan(None)
        )

    raw = work.group_by("vt_symbol").map_groups(calculate).sort(KEYS)
    if metric == "clarity":
        return raw.select("datetime", "vt_symbol", pl.col("_clarity").alias(name))
    if metric == "momentum":
        z_clarity = (pl.col("_clarity") - pl.col("_clarity").mean().over("datetime")) / (pl.col("_clarity").std(ddof=0).over("datetime") + 1e-12)
        z_momentum = (pl.col("_momentum") - pl.col("_momentum").mean().over("datetime")) / (pl.col("_momentum").std(ddof=0).over("datetime") + 1e-12)
        return raw.select("datetime", "vt_symbol", (-(z_momentum - z_clarity).abs()).alias(name))
    raise ValueError(f"unsupported trend clarity metric: {metric}")


def daily_magnitude_information_dispersion(df: pl.DataFrame, name: str) -> pl.DataFrame:
    work = _prepare(df, ("close",)).with_columns(
        (pl.col("close") / (pl.col("close").shift(1).over("vt_symbol") + 1e-12) - 1).alias("_ret")
    ).with_columns(
        (pl.col("_ret").abs().rank(method="average").over("datetime") / pl.len().over("datetime")).alias("_pct")
    ).with_columns(
        pl.when(pl.col("_pct") <= 0.2).then(5 / 15)
        .when(pl.col("_pct") <= 0.4).then(4 / 15)
        .when(pl.col("_pct") <= 0.6).then(3 / 15)
        .when(pl.col("_pct") <= 0.8).then(2 / 15)
        .otherwise(1 / 15).alias("_weight")
    )
    work = work.with_columns(
        (
            pl.col("_ret").shift(21).over("vt_symbol").sign()
            * pl.col("_weight").shift(21).over("vt_symbol")
        ).alias("_signed"),
        (
            pl.col("close").shift(21).over("vt_symbol")
            / (pl.col("close").shift(252).over("vt_symbol") + 1e-12) - 1
        ).alias("_pret"),
    )
    return work.select(
        "datetime", "vt_symbol",
        (
            -pl.col("_pret").sign()
            * pl.col("_signed").rolling_mean(231, min_samples=231).over("vt_symbol")
        ).alias(name),
    )


def daily_attention_metric(
    df: pl.DataFrame,
    metric: str,
    name: str,
    *,
    window: int = 20,
    baseline_window: int = 250,
) -> pl.DataFrame:
    work = _prepare(df, ("close", "volume")).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("vt_symbol") - 1).alias("_return"),
        pl.col("volume").rolling_mean(baseline_window, min_samples=baseline_window).over("vt_symbol").alias("_volume_base"),
    ).with_columns(
        (pl.col("volume") / (pl.col("_volume_base") + 1e-12)).alias("_abnormal_volume"),
        pl.col("_return").median().over("datetime").alias("_market_median_return"),
    ).with_columns(
        (pl.col("_return") - pl.col("_market_median_return")).pow(2).alias("_squared_abnormal_return")
    )
    expressions = {
        "max_abnormal_return": pl.col("_return").abs().rolling_max(window, min_samples=window).over("vt_symbol"),
        "max_abnormal_volume": pl.col("_abnormal_volume").rolling_max(window, min_samples=window).over("vt_symbol"),
        "mean_abnormal_volume": pl.col("_abnormal_volume").rolling_mean(window, min_samples=window).over("vt_symbol"),
        "decay_volume": pl.col("volume").rolling_mean(window, weights=list(range(1, window + 1)), min_samples=window).over("vt_symbol"),
        "mean_squared_abnormal_return": pl.col("_squared_abnormal_return").rolling_mean(
            window, min_samples=window
        ).over("vt_symbol"),
    }
    if metric not in expressions:
        raise ValueError(f"unsupported daily attention metric: {metric}")
    return work.select("datetime", "vt_symbol", expressions[metric].alias(name))


def daily_prospect_value(
    df: pl.DataFrame,
    source: str,
    name: str,
    *,
    window: int = 20,
) -> pl.DataFrame:
    required = ("close",) if source == "return" else ("pb_ratio_ttm",)
    work = _prepare(df, required)

    def weight(probability: np.ndarray, exponent: float) -> np.ndarray:
        return probability**exponent / (
            (probability**exponent + (1 - probability) ** exponent) ** (1 / exponent) + 1e-18
        )

    def calculate(group: pl.DataFrame) -> pl.DataFrame:
        base = group["close"].to_numpy() if source == "return" else group["pb_ratio_ttm"].to_numpy()
        changes = np.diff(np.log(base), prepend=np.nan)
        values = np.full(group.height, np.nan)
        alpha, loss_aversion = 0.88, 2.25
        for end in range(window, group.height):
            sample = changes[end - window + 1 : end + 1]
            sample = sample[np.isfinite(sample)]
            if sample.size < window:
                continue
            gains = np.sort(sample[sample >= 0])
            losses = np.sort(sample[sample < 0])
            score = 0.0
            if gains.size:
                tail = np.arange(gains.size, 0, -1) / sample.size
                next_tail = np.arange(gains.size - 1, -1, -1) / sample.size
                score += float(np.sum(gains**alpha * (weight(tail, 0.61) - weight(next_tail, 0.61))))
            if losses.size:
                cumulative = np.arange(1, losses.size + 1) / sample.size
                previous = np.arange(0, losses.size) / sample.size
                score += float(
                    np.sum(-loss_aversion * (-losses) ** alpha * (weight(cumulative, 0.69) - weight(previous, 0.69)))
                )
            values[end] = score
        return group.select("datetime", "vt_symbol").with_columns(pl.Series(name, values).fill_nan(None))

    return work.group_by("vt_symbol").map_groups(calculate).sort(KEYS)


def daily_peer_csad(df: pl.DataFrame, metric: str, name: str) -> pl.DataFrame:
    """Monthly small-peer CSAD factors built from the trailing 120 daily returns."""
    work = _prepare(df, ("close",)).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("vt_symbol") - 1).alias("_return")
    )
    dates = work["datetime"].unique().sort().to_list()
    symbols = work["vt_symbol"].unique().sort().to_list()
    date_index = {value: index for index, value in enumerate(dates)}
    symbol_index = {value: index for index, value in enumerate(symbols)}
    returns = np.full((len(dates), len(symbols)), np.nan)
    for date, symbol, value in work.select("datetime", "vt_symbol", "_return").iter_rows():
        if value is not None:
            returns[date_index[date], symbol_index[symbol]] = float(value)

    month_ends = [
        index for index, date in enumerate(dates)
        if index == len(dates) - 1
        or (dates[index + 1].year, dates[index + 1].month) != (date.year, date.month)
    ]
    rows: list[tuple[object, str, float]] = []
    for end in month_ends:
        if end < 119:
            continue
        sample = returns[end - 119 : end + 1]
        for focal_index, symbol in enumerate(symbols):
            focal = sample[:, focal_index]
            if np.isfinite(focal).sum() < 100:
                continue
            correlations: list[tuple[float, int]] = []
            for peer_index in range(len(symbols)):
                if peer_index == focal_index or np.isfinite(sample[:, peer_index]).sum() < 100:
                    continue
                valid = np.isfinite(focal) & np.isfinite(sample[:, peer_index])
                if valid.sum() < 100:
                    continue
                left, right = focal[valid], sample[valid, peer_index]
                if np.std(left) <= 1e-15 or np.std(right) <= 1e-15:
                    continue
                correlations.append((float(np.corrcoef(left, right)[0, 1]), peer_index))
            if len(correlations) < 9:
                continue
            peer_indices = [item[1] for item in sorted(correlations, reverse=True)[:9]]
            panel = sample[:, [focal_index, *peer_indices]]
            if metric == "small":
                center = np.nanmean(panel, axis=1)
            elif metric in {"position", "ratio"}:
                center = focal
            else:
                raise ValueError(f"unsupported peer CSAD metric: {metric}")
            csad = np.nanmean(np.abs(panel - center[:, None]), axis=1)
            valid = np.isfinite(csad) & np.isfinite(focal)
            if valid.sum() < 100:
                continue
            value = np.nan
            if metric == "ratio":
                short_csad = np.nanstd(csad[-20:])
                short_focal = np.nanstd(focal[-20:])
                long_csad = np.nanstd(csad)
                long_focal = np.nanstd(focal)
                if short_focal > 1e-15 and long_csad > 1e-15 and long_focal > 1e-15:
                    value = -(short_csad / short_focal) / (long_csad / long_focal)
            else:
                deviation = np.nanstd(csad)
                if deviation > 1e-15:
                    standardized = (csad - np.nanmean(csad)) / deviation
                    value = -float(np.nanmean(standardized[-20:]))
            rows.append((dates[end], symbol, float(value)))
    return pl.DataFrame(rows, schema=["datetime", "vt_symbol", name], orient="row")


def daily_attention_spillover(df: pl.DataFrame, metric: str, name: str) -> pl.DataFrame:
    """Monthly attention spillover among same-industry and same-size peers."""
    work = _prepare(
        df,
        ("close", "volume", "a_share_market_val_in_circulation", "first_industry_code"),
    ).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("vt_symbol") - 1).alias("_return"),
        (
            pl.col("volume") * pl.col("close")
            / (pl.col("a_share_market_val_in_circulation") + 1e-12)
        ).alias("_turnover_rate"),
        pl.col("datetime").dt.year().alias("_year"),
        pl.col("datetime").dt.month().alias("_month"),
    ).with_columns(pl.col("_return").mean().over("datetime").alias("_market_return"))
    if metric == "turnover":
        attention = pl.col("_turnover_rate")
    elif metric == "abnormal_return":
        attention = (pl.col("_return") - pl.col("_market_return")).pow(2)
    else:
        raise ValueError(f"unsupported attention-spillover metric: {metric}")
    monthly = work.sort(["vt_symbol", "datetime"]).group_by(
        "_year", "_month", "vt_symbol", maintain_order=True
    ).agg(
        pl.col("datetime").max(),
        attention.mean().alias("_attention"),
        pl.col("a_share_market_val_in_circulation").last().alias("_cap"),
        pl.col("first_industry_code").drop_nulls().last().alias("_industry"),
    ).with_columns(
        (
            pl.col("_cap").rank(method="average").over("_year", "_month", "_industry")
            / pl.len().over("_year", "_month", "_industry")
        ).alias("_size_pct")
    ).with_columns(
        pl.when(pl.col("_size_pct") <= 0.3).then(pl.lit("small"))
        .when(pl.col("_size_pct") <= 0.7).then(pl.lit("middle"))
        .otherwise(pl.lit("large")).alias("_size_bucket")
    )
    peer_keys = ["_year", "_month", "_industry", "_size_bucket"]
    peer_count = pl.col("_attention").count().over(peer_keys)
    peer_mean = (
        (pl.col("_attention").sum().over(peer_keys) - pl.col("_attention"))
        / (peer_count - 1)
    )
    return monthly.select(
        "datetime", "vt_symbol",
        pl.when(peer_count > 1).then(peer_mean - pl.col("_attention")).otherwise(None).alias(name),
    ).sort(KEYS)
