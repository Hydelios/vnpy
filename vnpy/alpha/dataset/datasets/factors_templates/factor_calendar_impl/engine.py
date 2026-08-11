from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import polars as pl

from .chip_operators import chip_distribution_metric
from .daily_operators import (
    daily_attention_metric,
    daily_ideal_amplitude,
    daily_frazzini_pedersen_beta,
    daily_capital_gain_overhang,
    daily_liquidity_shock,
    daily_panic_metric,
    daily_reversal_frequency,
    daily_gamma_illiquidity,
    daily_salience_return,
    daily_trend_clarity,
    daily_magnitude_information_dispersion,
    daily_peer_csad,
    daily_attention_spillover,
    daily_liquidity_metric,
    daily_price_acceleration,
    daily_prospect_value,
    daily_rolling_risk,
    daily_technical_metric,
)
from .intraday_operators import (
    intraday_autocorr,
    intraday_autocorrelation_composite,
    intraday_apm,
    intraday_max_drawdown,
    intraday_path_metric,
    intraday_period_return,
    intraday_price_volume_composite,
    intraday_reduce,
    intraday_return_measure,
    intraday_return_stat,
    intraday_return_turnover_corr,
    intraday_semivariance,
    intraday_semibeta,
    intraday_smart_money,
    intraday_fuzzy_spread,
    intraday_extreme_reversal,
    intraday_market_event_metric,
    intraday_volume_diff_regression,
    intraday_volume_valley_price,
    intraday_trend_fund_metric,
    intraday_tail_share,
    intraday_turnover_uniformity,
    intraday_volume_share_stat,
)
from .metadata import CalendarFactorDef, ImplementationStatus


@dataclass
class CalendarDataBundle:
    daily: pl.DataFrame
    minute: pl.DataFrame | None = None
    exposures: pl.DataFrame | None = None
    external: dict[str, pl.DataFrame] = field(default_factory=dict)


def _keys(df: pl.DataFrame) -> pl.DataFrame:
    required = {"datetime", "vt_symbol"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"daily data missing keys: {missing}")
    return df.sort(["vt_symbol", "datetime"])


class CalendarFactorEngine:
    """Standalone executor; it does not alter the legacy expression namespace."""

    def compute(
        self,
        definitions: Iterable[CalendarFactorDef],
        data: CalendarDataBundle,
        *,
        strict: bool = True,
    ) -> pl.DataFrame:
        daily = _keys(data.daily)
        if data.exposures is not None:
            add_cols = [c for c in data.exposures.columns if c not in {"datetime", "vt_symbol"}]
            daily = daily.join(
                data.exposures.select("datetime", "vt_symbol", *add_cols),
                on=["datetime", "vt_symbol"],
                how="left",
            )
        result = daily.select("datetime", "vt_symbol").unique().sort(["datetime", "vt_symbol"])
        for definition in definitions:
            if definition.implementation_status not in {ImplementationStatus.EXACT, ImplementationStatus.PROXY}:
                continue
            try:
                computed = self._compute_one(definition, daily, data.minute)
                computed = self._postprocess(computed, definition, daily)
            except Exception:
                if strict:
                    raise
                continue
            result = result.join(computed, on=["datetime", "vt_symbol"], how="left")
        return result

    @staticmethod
    def _postprocess(
        computed: pl.DataFrame,
        definition: CalendarFactorDef,
        daily: pl.DataFrame,
    ) -> pl.DataFrame:
        """Apply explicitly declared daily aggregation to a raw daily metric."""
        params = definition.default_params
        if "post_window" in params:
            window = int(params["post_window"])
            stat = str(params.get("post_stat", "mean"))
            value = pl.col(definition.name)
            expressions = {
                "mean": value.rolling_mean(window, min_samples=window).over("vt_symbol"),
                "std": value.rolling_std(window, min_samples=window, ddof=0).over("vt_symbol"),
                "sum": value.rolling_sum(window, min_samples=window).over("vt_symbol"),
                "linear_decay": value.rolling_mean(
                    window,
                    weights=list(range(1, window + 1)),
                    min_samples=window,
                ).over("vt_symbol"),
                "mean_std": 0.5 * (
                    value.rolling_mean(window, min_samples=window).over("vt_symbol")
                    + value.rolling_std(window, min_samples=window, ddof=0).over("vt_symbol")
                ),
                "slope": value.rolling_map(
                    lambda series: float(np.polyfit(np.arange(series.len()), series.to_numpy(), 1)[0]),
                    window_size=window,
                    min_samples=window,
                ).over("vt_symbol"),
            }
            if stat not in expressions:
                raise ValueError(f"unsupported post_stat: {stat}")
            computed = computed.sort(["vt_symbol", "datetime"]).select(
                "datetime", "vt_symbol", expressions[stat].alias(definition.name)
            )
        if params.get("post_neutralize_market_cap"):
            cap = "a_share_market_val_in_circulation"
            work = computed.join(daily.select("datetime", "vt_symbol", cap), on=["datetime", "vt_symbol"])
            work = work.with_columns(pl.col(cap).log().alias("_x")).with_columns(
                pl.col("_x").mean().over("datetime").alias("_xm"),
                pl.col(definition.name).mean().over("datetime").alias("_ym"),
            ).with_columns(
                (pl.col("_x") - pl.col("_xm")).alias("_xc"),
                (pl.col(definition.name) - pl.col("_ym")).alias("_yc"),
            ).with_columns(
                (
                    (pl.col("_xc") * pl.col("_yc")).mean().over("datetime")
                    / (pl.col("_xc").pow(2).mean().over("datetime") + 1e-12)
                ).alias("_beta")
            )
            computed = work.select(
                "datetime", "vt_symbol", (pl.col("_yc") - pl.col("_beta") * pl.col("_xc")).alias(definition.name)
            )
        if params.get("post_zscore"):
            y = pl.col(definition.name)
            computed = computed.with_columns(
                ((y - y.mean().over("datetime")) / (y.std(ddof=0).over("datetime") + 1e-12)).alias(definition.name)
            )
        return computed

    def _compute_one(
        self,
        definition: CalendarFactorDef,
        daily: pl.DataFrame,
        minute: pl.DataFrame | None,
    ) -> pl.DataFrame:
        key = definition.implementation_key
        name = definition.name
        params = definition.default_params
        if key == "chip_distribution_metric":
            if minute is None:
                raise ValueError(f"{name}: minute data required")
            return chip_distribution_metric(daily, minute, str(params["metric"]), name)
        missing = [field for field in definition.required_fields if field not in daily.columns]
        if definition.input_frequency == "1m":
            if minute is None:
                raise ValueError(f"{name}: minute data required")
            missing = [
                field for field in definition.required_fields
                if field not in minute.columns and field not in daily.columns
            ]
        if missing:
            raise ValueError(f"{name}: missing fields {missing}")

        if key.startswith("alias:"):
            field = key.split(":", 1)[1]
            return daily.select("datetime", "vt_symbol", pl.col(field).cast(pl.Float64).alias(name))
        if key == "log":
            field = definition.required_fields[0]
            return daily.select("datetime", "vt_symbol", pl.col(field).cast(pl.Float64).log().alias(name))
        if key == "turnover_rate":
            return daily.select(
                "datetime", "vt_symbol",
                (
                    pl.col("volume") * pl.col("close")
                    / (pl.col("a_share_market_val_in_circulation") + 1e-12)
                ).alias(name),
            )
        if key == "daily_liquidity_metric":
            return daily_liquidity_metric(
                daily,
                str(params["metric"]),
                name,
                window=int(params.get("window", 20)),
                long_window=int(params.get("long_window", 250)),
            )
        if key == "daily_rolling_risk":
            return daily_rolling_risk(
                daily,
                str(params["metric"]),
                name,
                window=int(params.get("window", 252)),
                min_samples=int(params["min_samples"]) if "min_samples" in params else None,
            )
        if key == "momentum_acceleration_halves":
            window = int(params.get("window", 126))
            prior = pl.col("close").shift(window).over("vt_symbol")
            prior_twice = pl.col("close").shift(2 * window).over("vt_symbol")
            return daily.select(
                "datetime", "vt_symbol",
                (
                    (pl.col("close") / (prior + 1e-12) - 1)
                    - (prior / (prior_twice + 1e-12) - 1)
                ).alias(name),
            )
        if key == "daily_price_acceleration":
            return daily_price_acceleration(daily, name, window=int(params.get("window", 60)))
        if key == "daily_ideal_amplitude":
            return daily_ideal_amplitude(
                daily, name, window=int(params.get("window", 20)),
                fraction=float(params.get("fraction", 0.25)),
            )
        if key == "daily_technical_metric":
            return daily_technical_metric(
                daily, str(params["metric"]), name, window=int(params.get("window", 20))
            )
        if key == "daily_frazzini_pedersen_beta":
            return daily_frazzini_pedersen_beta(daily, name)
        if key == "daily_capital_gain_overhang":
            return daily_capital_gain_overhang(daily, name)
        if key == "daily_liquidity_shock":
            return daily_liquidity_shock(daily, str(params["metric"]), name)
        if key == "daily_panic_metric":
            return daily_panic_metric(
                daily, minute, str(params["metric"]), name,
                numerator_offset=bool(params.get("numerator_offset", False)),
            )
        if key == "daily_reversal_frequency":
            return daily_reversal_frequency(
                daily, str(params["direction"]), bool(params.get("abnormal", False)), name
            )
        if key == "daily_gamma_illiquidity":
            return daily_gamma_illiquidity(daily, name)
        if key == "daily_salience_return":
            return daily_salience_return(daily, name, variant=str(params.get("variant", "standard")))
        if key == "daily_trend_clarity":
            return daily_trend_clarity(daily, str(params["metric"]), name)
        if key == "daily_magnitude_information_dispersion":
            return daily_magnitude_information_dispersion(daily, name)
        if key == "daily_peer_csad":
            return daily_peer_csad(daily, str(params["metric"]), name)
        if key == "daily_attention_spillover":
            return daily_attention_spillover(daily, str(params["metric"]), name)
        if key == "daily_attention_metric":
            return daily_attention_metric(
                daily, str(params["metric"]), name,
                window=int(params.get("window", 20)),
                baseline_window=int(params.get("baseline_window", 250)),
            )
        if key == "daily_prospect_value":
            return daily_prospect_value(
                daily, str(params["source"]), name, window=int(params.get("window", 20))
            )
        if key == "rolling_high_ratio":
            window = int(params["window"])
            return daily.select(
                "datetime", "vt_symbol",
                (pl.col("close") / (pl.col("close").rolling_max(window).over("vt_symbol") + 1e-12)).alias(name),
            )
        if key == "return":
            window = int(params["window"])
            sign = float(params.get("sign", 1.0))
            return daily.select(
                "datetime", "vt_symbol",
                (sign * (pl.col("close") / (pl.col("close").shift(window).over("vt_symbol") + 1e-12) - 1)).alias(name),
            )
        if key == "daily_intraday_return":
            return daily.select(
                "datetime", "vt_symbol", (pl.col("close") / (pl.col("open") + 1e-12) - 1).alias(name)
            )
        if key == "overnight_return":
            return daily.select(
                "datetime", "vt_symbol",
                (pl.col("open") / (pl.col("close").shift(1).over("vt_symbol") + 1e-12) - 1).alias(name),
            )
        if key == "rolling_return_stat":
            return self._rolling_return_stat(daily, name, params)
        if key == "legacy_expression":
            from ....utility import calculate_by_expression

            expression = str(params.get("base_expression", definition.implemented_formula))
            calculated = calculate_by_expression(daily, expression, interval="1d")
            return calculated.select("datetime", "vt_symbol", pl.col("data").alias(name))
        if key == "market_residual_stat":
            return self._market_residual_stat(daily, name, params)
        if key == "intraday_return_stat":
            return intraday_return_stat(minute, str(params["stat"]), name)  # type: ignore[arg-type]
        if key == "intraday_return_measure":
            return intraday_return_measure(minute, str(params["measure"]), name)  # type: ignore[arg-type]
        if key == "intraday_period_return":
            minutes = params.get("minutes")
            return intraday_period_return(minute, int(minutes) if minutes is not None else None, name)  # type: ignore[arg-type]
        if key == "intraday_max_drawdown":
            return intraday_max_drawdown(minute, name)  # type: ignore[arg-type]
        if key == "intraday_path_metric":
            return intraday_path_metric(minute, str(params["metric"]), name)  # type: ignore[arg-type]
        if key == "intraday_semivariance":
            return intraday_semivariance(minute, str(params["side"]), name)  # type: ignore[arg-type]
        if key == "intraday_semibeta":
            return intraday_semibeta(
                minute, str(params["stock_side"]), str(params["market_side"]), name
            )  # type: ignore[arg-type]
        if key == "intraday_tail_share":
            return intraday_tail_share(
                minute, str(params["field"]), int(params["minutes"]), name  # type: ignore[arg-type]
            )
        if key == "intraday_reduce":
            return intraday_reduce(minute, str(params["field"]), str(params["stat"]), name)  # type: ignore[arg-type]
        if key == "intraday_volume_share_stat":
            return intraday_volume_share_stat(minute, str(params["stat"]), name)  # type: ignore[arg-type]
        if key == "intraday_return_turnover_corr":
            return intraday_return_turnover_corr(
                minute, return_lag=int(params["return_lag"]),
                turnover_lag=int(params["turnover_lag"]), name=name,
            )  # type: ignore[arg-type]
        if key == "intraday_autocorr":
            return intraday_autocorr(
                minute, str(params["field"]), int(params["lag"]), name,
            )  # type: ignore[arg-type]
        if key == "intraday_autocorrelation_composite":
            return intraday_autocorrelation_composite(
                minute, daily, str(params["mode"]), name, window=int(params.get("window", 20))
            )  # type: ignore[arg-type]
        if key == "intraday_price_volume_composite":
            return intraday_price_volume_composite(
                minute, daily, str(params["mode"]), name, window=int(params.get("window", 20))
            )  # type: ignore[arg-type]
        if key == "intraday_turnover_uniformity":
            return intraday_turnover_uniformity(
                minute, daily, name, window=int(params.get("window", 20))
            )  # type: ignore[arg-type]
        if key == "intraday_apm":
            return intraday_apm(minute, daily, name, window=int(params.get("window", 20)))  # type: ignore[arg-type]
        if key == "intraday_smart_money":
            return intraday_smart_money(minute, name, window=int(params.get("window", 10)))  # type: ignore[arg-type]
        if key == "intraday_fuzzy_spread":
            return intraday_fuzzy_spread(minute, name)  # type: ignore[arg-type]
        if key == "intraday_extreme_reversal":
            return intraday_extreme_reversal(minute, name, window=int(params.get("window", 20)))  # type: ignore[arg-type]
        if key == "intraday_market_event_metric":
            return intraday_market_event_metric(
                minute, str(params["metric"]), str(params["side"]), name
            )  # type: ignore[arg-type]
        if key == "intraday_volume_diff_regression":
            return intraday_volume_diff_regression(minute, str(params["metric"]), name)  # type: ignore[arg-type]
        if key == "intraday_volume_valley_price":
            return intraday_volume_valley_price(minute, name)  # type: ignore[arg-type]
        if key == "intraday_trend_fund_metric":
            return intraday_trend_fund_metric(
                minute, daily, str(params["metric"]), name
            )  # type: ignore[arg-type]
        raise ValueError(f"{name}: unknown implementation key {key!r}")

    @staticmethod
    def _rolling_return_stat(df: pl.DataFrame, name: str, params: dict) -> pl.DataFrame:
        window = int(params["window"])
        stat = str(params["stat"])
        sign = float(params.get("sign", 1.0))
        work = df.with_columns(
            (pl.col("close").log() - pl.col("close").log().shift(1))
            .over("vt_symbol")
            .alias("_return")
        )
        ret = pl.col("_return")
        expressions = {
            "std": ret.rolling_std(window, min_samples=window, ddof=0).over("vt_symbol"),
            "skew": ret.rolling_skew(window, bias=True, min_samples=window).over("vt_symbol"),
            "max": ret.rolling_max(window, min_samples=window).over("vt_symbol"),
            "min": ret.rolling_min(window, min_samples=window).over("vt_symbol"),
        }
        if stat not in expressions:
            raise ValueError(f"unsupported rolling return stat: {stat}")
        return work.select("datetime", "vt_symbol", (sign * expressions[stat]).alias(name))

    @staticmethod
    def _market_residual_stat(df: pl.DataFrame, name: str, params: dict) -> pl.DataFrame:
        if "market_ret" not in df.columns:
            raise ValueError(f"{name}: market_ret is required")
        window = int(params["window"])
        work = df.with_columns(
            (pl.col("close").log() - pl.col("close").log().shift(1))
            .over("vt_symbol")
            .alias("_stock_ret")
        )
        ret = pl.col("_stock_ret")
        market = pl.col("market_ret")
        ret_mean = ret.rolling_mean(window, min_samples=window).over("vt_symbol")
        market_mean = market.rolling_mean(window, min_samples=window).over("vt_symbol")
        covariance = (
            (ret * market).rolling_mean(window, min_samples=window).over("vt_symbol")
            - ret_mean * market_mean
        )
        market_variance = (
            market.pow(2).rolling_mean(window, min_samples=window).over("vt_symbol")
            - market_mean.pow(2)
        )
        work = work.with_columns((covariance / (market_variance + 1e-12)).alias("_beta"))
        work = work.with_columns((ret - pl.col("_beta") * market).alias("_residual"))
        return work.select(
            "datetime", "vt_symbol",
            pl.col("_residual").rolling_std(window, min_samples=window, ddof=0).over("vt_symbol").alias(name),
        )
