from __future__ import annotations

import numpy as np
import polars as pl


def chip_reference_price(
    df: pl.DataFrame,
    *,
    price_col: str = "close",
    turnover_rate_col: str = "turnover_rate",
    name: str = "chip_reference_price",
) -> pl.DataFrame:
    """Recursive average holder cost implied by a daily turnover survival model.

    RP[t] = (1-turnover[t]) * RP[t-1] + turnover[t] * price[t].
    Missing rows do not invent a new state; invalid turnover is clipped to [0, 1]
    and explicitly represents the modelling assumption used by calendar factors.
    """
    required = {"datetime", "vt_symbol", price_col, turnover_rate_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"chip input missing columns: {missing}")

    def per_symbol(group: pl.DataFrame) -> pl.DataFrame:
        group = group.sort("datetime")
        price = group[price_col].cast(pl.Float64).to_numpy()
        turnover = group[turnover_rate_col].cast(pl.Float64).to_numpy()
        out = np.full(len(group), np.nan, dtype=float)
        state = np.nan
        for i, (p, rate) in enumerate(zip(price, turnover)):
            if not np.isfinite(p):
                out[i] = state
                continue
            r = float(np.clip(rate, 0.0, 1.0)) if np.isfinite(rate) else 0.0
            state = p if not np.isfinite(state) else (1.0 - r) * state + r * p
            out[i] = state
        return group.select("datetime", "vt_symbol").with_columns(pl.Series(name, out))

    parts = [per_symbol(group) for group in df.partition_by("vt_symbol", maintain_order=True)]
    if not parts:
        return pl.DataFrame(schema={"datetime": pl.Datetime, "vt_symbol": pl.String, name: pl.Float64})
    return pl.concat(parts).sort(["datetime", "vt_symbol"])


def chip_profit_ratio(
    df: pl.DataFrame,
    reference_price_col: str,
    price_col: str = "close",
    name: str = "chip_profit_ratio",
) -> pl.DataFrame:
    required = {"datetime", "vt_symbol", reference_price_col, price_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"chip input missing columns: {missing}")
    return df.select(
        "datetime",
        "vt_symbol",
        (pl.col(price_col) / (pl.col(reference_price_col) + 1e-12) - 1).alias(name),
    )


def chip_distribution_metric(
    daily: pl.DataFrame,
    minute: pl.DataFrame,
    metric: str,
    name: str,
    *,
    price_decimals: int = 4,
) -> pl.DataFrame:
    """Reconstruct a surviving holder-cost distribution from minute volume.

    Each day's minute volume distribution receives the daily free-float turnover
    weight; the previous distribution survives with weight ``1-turnover``.
    Prices are merged on a fixed decimal grid solely to keep the state bounded.
    """
    daily_required = {"datetime", "vt_symbol", "high", "low", "close", "volume", "a_share_market_val_in_circulation"}
    minute_required = {"datetime", "vt_symbol", "close", "volume"}
    missing_daily = sorted(daily_required.difference(daily.columns))
    missing_minute = sorted(minute_required.difference(minute.columns))
    if missing_daily or missing_minute:
        raise ValueError(f"chip distribution missing daily={missing_daily}, minute={missing_minute}")

    minute_days: dict[tuple[str, object], tuple[np.ndarray, np.ndarray]] = {}
    prepared_minute = minute.with_columns(pl.col("datetime").cast(pl.Date).alias("_date"))
    for group in prepared_minute.partition_by(["vt_symbol", "_date"], maintain_order=True):
        minute_days[(group["vt_symbol"][0], group["_date"][0])] = (
            group["close"].cast(pl.Float64).to_numpy(),
            group["volume"].cast(pl.Float64).to_numpy(),
        )

    parts: list[pl.DataFrame] = []
    for group in daily.partition_by("vt_symbol", maintain_order=True):
        group = group.sort("datetime")
        symbol = group["vt_symbol"][0]
        distribution: dict[float, float] = {}
        bias_state = np.nan
        previous_winner = np.nan
        values = np.full(group.height, np.nan)
        for row_index, row in enumerate(group.iter_rows(named=True)):
            close = float(row["close"])
            float_value = float(row["a_share_market_val_in_circulation"])
            rate = float(np.clip(float(row["volume"]) * close / (float_value + 1e-12), 0, 1))
            date = row["datetime"].date()
            prices, volumes = minute_days.get((symbol, date), (np.array([close]), np.array([1.0])))
            valid = np.isfinite(prices) & np.isfinite(volumes) & (volumes > 0)
            prices, volumes = prices[valid], volumes[valid]
            if prices.size == 0:
                prices, volumes = np.array([close]), np.array([1.0])

            previous = dict(distribution)
            distribution = {price: weight * (1 - rate) for price, weight in distribution.items() if weight * (1 - rate) > 1e-10}
            for price, weight in zip(prices, volumes / volumes.sum() * rate):
                bucket = round(float(price), price_decimals)
                distribution[bucket] = distribution.get(bucket, 0.0) + float(weight)
            total = sum(distribution.values())
            if total <= 0:
                distribution = {round(close, price_decimals): 1.0}
                total = 1.0
            p = np.array(sorted(distribution), dtype=float)
            w = np.array([distribution[value] / total for value in p], dtype=float)
            mean = float(np.sum(p * w))
            std = float(np.sqrt(np.sum((p - mean) ** 2 * w)))
            cumulative = np.cumsum(w)
            q05 = float(p[np.searchsorted(cumulative, 0.05)])
            q95 = float(p[min(np.searchsorted(cumulative, 0.95), p.size - 1)])
            winner = float(w[p <= close].sum())
            new_weight = volumes / volumes.sum()
            new_gain = float(np.sum(np.maximum(close - prices, 0) * new_weight * rate))
            new_loss = float(np.sum(np.maximum(prices - close, 0) * new_weight * rate))

            prev_total = sum(previous.values())
            realized_gain = realized_loss = 0.0
            if prev_total > 0:
                for price, weight in previous.items():
                    sold = weight / prev_total * rate
                    realized_gain += max(close - price, 0) * sold
                    realized_loss += max(price - close, 0) * sold
            paper_gain = float(np.sum(np.maximum(close - p, 0) * w))
            paper_loss = float(np.sum(np.maximum(p - close, 0) * w))
            cpgr = realized_gain / (realized_gain + paper_gain + 1e-12)
            cplr = realized_loss / (realized_loss + paper_loss + 1e-12)
            gain = float(np.sum(np.maximum((close - p) / (close + 1e-12), 0) * w))
            loss = float(np.sum(np.minimum((close - p) / (close + 1e-12), 0) * w))

            if metric == "reference_price":
                value = mean
            elif metric == "capital_gain_overhang":
                value = (close - mean) / (close + 1e-12)
            elif metric == "unrealized_profit":
                value = (close - mean) / (mean + 1e-12)
            elif metric == "winner_ratio":
                value = winner
            elif metric == "concentration":
                value = 2 * (q95 - q05) / (q95 + q05 + 1e-12)
            elif metric == "relative_position":
                value = (close - mean) / (p[-1] - p[0] + 1e-12)
            elif metric == "coefficient_of_variation":
                value = std / (mean + 1e-12)
            elif metric == "cost_bandwidth":
                value = (p[-1] - p[0]) / (p[0] + 1e-12)
            elif metric == "kurtosis":
                value = float(np.sum(((p - mean) / (std + 1e-12)) ** 4 * w))
            elif metric == "active_share":
                limit = 0.2 if symbol.startswith(("300", "688")) else 0.1
                value = float(w[(p >= close * (1 - limit)) & (p <= close * (1 + limit))].sum())
            elif metric == "chip_bias":
                bias_state = winner if not np.isfinite(bias_state) else winner * rate + bias_state * (1 - rate)
                value = bias_state
            elif metric == "gain_tendency":
                value = gain
            elif metric == "loss_tendency":
                value = loss
            elif metric == "v_tendency":
                value = gain + 0.23 * abs(loss)
            elif metric == "realized_profit_ratio":
                value = cpgr
            elif metric == "realized_loss_ratio":
                value = cplr
            elif metric == "disposition_effect":
                value = cpgr - cplr
            elif metric == "v_disposition_effect":
                value = cpgr + 0.23 * cplr
            elif metric == "chip_turnover":
                in_range = float(w[(p >= float(row["low"])) & (p <= float(row["high"]))].sum())
                value = in_range / (rate + 1e-12)
            elif metric == "chip_penetration":
                value = (winner - previous_winner) / (rate + 1e-12) if np.isfinite(previous_winner) else np.nan
            elif metric == "new_gain_ratio":
                value = new_gain / (paper_gain + 1e-12)
            elif metric == "new_loss_ratio":
                value = new_loss / (paper_loss + 1e-12)
            else:
                raise ValueError(f"unsupported chip distribution metric: {metric}")
            values[row_index] = value
            previous_winner = winner
        parts.append(group.select("datetime", "vt_symbol").with_columns(pl.Series(name, values)))
    if not parts:
        return pl.DataFrame(schema={"datetime": pl.Datetime, "vt_symbol": pl.String, name: pl.Float64})
    return pl.concat(parts).sort(["datetime", "vt_symbol"])
