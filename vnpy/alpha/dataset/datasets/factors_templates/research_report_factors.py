"""Research-report factor families.

This is the fixed Python landing file for paper/report-derived factor classes.
Shared expression operators should be added in stateful_ext_function.py.
"""

from __future__ import annotations

import polars as pl

from .baseAlphaStrategy import BaseAlphaStrategy


class Fangzheng72DailyPV(BaseAlphaStrategy):
    """Daily price-volume factors from Founder Securities "Factor 72" reports."""

    EPS_EXPR = "1e-12"
    JUMP_WINDOWS = [5, 10, 20, 40, 60, 120]
    BIAS_WINDOWS = [20, 40, 60, 120]
    MOMENTUM_WINDOWS = [20, 40, 60, 120]
    PV_WINDOW = 20
    FLY_CUT_PRICE_Q = 0.4
    FR_CUT_RET_Q = 0.9
    REQUIRED_COLUMNS = {"vt_symbol", "volume", "turnover", "market_cap_2"}

    def __init__(
        self,
        df: pl.DataFrame,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
        interval: str = "1d",
        enable_cache: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        df = self._validate_required_columns(df)
        super().__init__(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
            interval=interval,
            enable_cache=enable_cache,
            cache_dir=cache_dir,
        )
        self.set_label("ts_delay(close, -3) / ts_delay(close, -1) - 1")
        self.register_all()

    @classmethod
    def _validate_required_columns(cls, df: pl.DataFrame) -> pl.DataFrame:
        missing = cls.REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Fangzheng72DailyPV缺少必要字段: {missing_text}")

        return df

    @classmethod
    def turnover_rate_expr(cls) -> str:
        return f"(turnover / (market_cap_2 + {cls.EPS_EXPR}))"

    @classmethod
    def rel_volume_20_expr(cls) -> str:
        return f"(volume / (ts_mean(volume, 20) + {cls.EPS_EXPR}))"

    def register_all(self) -> None:
        self.register_jump_factors()
        self.register_bias_factors()
        self.register_bias_momentum_avg_factors()
        self.register_momentum_factors()
        self.register_fly_factors()
        self.register_fly_cut_factors()
        self.register_front_runner_factors()
        self.register_front_runner_cut_factors()
        self.register_front_runner_pure_factors()

    def register_jump_factors(self) -> None:
        self.add_parametric_feature(
            base_name="report_fz72_jump",
            expr_tpl=(
                "-1 * ts_mean("
                "ts_abs(open / (ts_delay(close, 1) + 1e-12) - 1), "
                "{w}"
                ")"
            ),
            param_grid={"w": self.JUMP_WINDOWS},
            category="Fangzheng72",
            sub_category="OvernightJump",
        )

    def register_bias_factors(self) -> None:
        self.add_parametric_feature(
            base_name="report_fz72_bias",
            expr_tpl="-1 * (close / (ts_mean(close, {w}) + 1e-12) - 1)",
            param_grid={"w": self.BIAS_WINDOWS},
            category="Fangzheng72",
            sub_category="MABias",
        )

    def register_bias_momentum_avg_factors(self) -> None:
        self.add_parametric_feature(
            base_name="report_fz72_bias_momentum_avg",
            expr_tpl="ts_mean(close, {w}) / (close + 1e-12)",
            param_grid={"w": self.BIAS_WINDOWS},
            category="Fangzheng72",
            sub_category="MABiasMomentumAverage",
        )

    def register_momentum_factors(self) -> None:
        self.add_parametric_feature(
            base_name="report_fz72_momentum",
            expr_tpl="-1 * (close / (ts_delay(close, {w}) + 1e-12) - 1)",
            param_grid={"w": self.MOMENTUM_WINDOWS},
            category="Fangzheng72",
            sub_category="MomentumReversal",
        )

    def register_fly_factors(self) -> None:
        turnover_rate = self.turnover_rate_expr()
        rel_volume_20 = self.rel_volume_20_expr()
        self.add_parametric_feature(
            base_name="report_fz72_fly_turnover",
            expr_tpl=f"-1 * ts_corr(close, {turnover_rate}, {{w}})",
            param_grid={"w": [self.PV_WINDOW]},
            category="Fangzheng72",
            sub_category="PriceVolumeFly",
        )
        self.add_parametric_feature(
            base_name="report_fz72_fly_relvol",
            expr_tpl=f"-1 * ts_corr(close, {rel_volume_20}, {{w}})",
            param_grid={"w": [self.PV_WINDOW]},
            category="Fangzheng72",
            sub_category="PriceVolumeFly",
        )
        self.add_parametric_feature(
            base_name="report_fz72_fly_r2_turnover",
            expr_tpl=f"-1 * (ts_corr(close, {turnover_rate}, {{w}}) ** 2)",
            param_grid={"w": [self.PV_WINDOW]},
            category="Fangzheng72",
            sub_category="PriceVolumeFlyR2",
        )
        self.add_parametric_feature(
            base_name="report_fz72_fly_r2_relvol",
            expr_tpl=f"-1 * (ts_corr(close, {rel_volume_20}, {{w}}) ** 2)",
            param_grid={"w": [self.PV_WINDOW]},
            category="Fangzheng72",
            sub_category="PriceVolumeFlyR2",
        )

    def register_fly_cut_factors(self) -> None:
        turnover_rate = self.turnover_rate_expr()
        rel_volume_20 = self.rel_volume_20_expr()
        self.add_parametric_feature(
            base_name="report_fz72_fly_cut_price_high60_turnover",
            expr_tpl=(
                "-1 * ts_corr_quantile_mask("
                f"close, {turnover_rate}, close, "
                f"{{w}}, {self.FLY_CUT_PRICE_Q:g}, \"high\""
                ")"
            ),
            param_grid={"w": [self.PV_WINDOW]},
            category="Fangzheng72",
            sub_category="PriceVolumeFlyCut",
        )
        self.add_parametric_feature(
            base_name="report_fz72_fly_cut_price_high60_relvol",
            expr_tpl=(
                "-1 * ts_corr_quantile_mask("
                f"close, {rel_volume_20}, close, "
                f"{{w}}, {self.FLY_CUT_PRICE_Q:g}, \"high\""
                ")"
            ),
            param_grid={"w": [self.PV_WINDOW]},
            category="Fangzheng72",
            sub_category="PriceVolumeFlyCut",
        )

    def register_front_runner_factors(self) -> None:
        ret = "(close / (ts_delay(close, 1) + 1e-12) - 1)"
        turnover_rate = self.turnover_rate_expr()
        rel_volume_20 = self.rel_volume_20_expr()
        self.add_parametric_feature(
            base_name="report_fz72_fr_turnover",
            expr_tpl=f"ts_corr(ts_delay({turnover_rate}, 1), {ret}, {{w}})",
            param_grid={"w": [self.PV_WINDOW]},
            category="Fangzheng72",
            sub_category="FrontRunner",
        )
        self.add_parametric_feature(
            base_name="report_fz72_fr_relvol",
            expr_tpl=f"ts_corr(ts_delay({rel_volume_20}, 1), {ret}, {{w}})",
            param_grid={"w": [self.PV_WINDOW]},
            category="Fangzheng72",
            sub_category="FrontRunner",
        )

    def register_front_runner_cut_factors(self) -> None:
        ret = "(close / (ts_delay(close, 1) + 1e-12) - 1)"
        prior_ret = f"ts_delay({ret}, 1)"
        turnover_rate = self.turnover_rate_expr()
        rel_volume_20 = self.rel_volume_20_expr()
        self.add_parametric_feature(
            base_name="report_fz72_fr_cut_turnover",
            expr_tpl=(
                "ts_corr_quantile_mask("
                f"ts_delay({turnover_rate}, 1), {ret}, {prior_ret}, "
                f"{{w}}, {self.FR_CUT_RET_Q:g}, \"low\""
                ")"
            ),
            param_grid={"w": [self.PV_WINDOW]},
            category="Fangzheng72",
            sub_category="FrontRunnerCut",
        )
        self.add_parametric_feature(
            base_name="report_fz72_fr_cut_relvol",
            expr_tpl=(
                "ts_corr_quantile_mask("
                f"ts_delay({rel_volume_20}, 1), {ret}, {prior_ret}, "
                f"{{w}}, {self.FR_CUT_RET_Q:g}, \"low\""
                ")"
            ),
            param_grid={"w": [self.PV_WINDOW]},
            category="Fangzheng72",
            sub_category="FrontRunnerCut",
        )

    def register_front_runner_pure_factors(self) -> None:
        ret = "(close / (ts_delay(close, 1) + 1e-12) - 1)"
        turnover_rate = self.turnover_rate_expr()
        rel_volume_20 = self.rel_volume_20_expr()
        self.add_parametric_feature(
            base_name="report_fz72_fr_pure_turnover",
            expr_tpl=(
                "ts_rolling_ols_residual_corr("
                f"ts_delay({turnover_rate}, 1), "
                f"{turnover_rate}, {ret}, {{w}}"
                ")"
            ),
            param_grid={"w": [self.PV_WINDOW]},
            category="Fangzheng72",
            sub_category="FrontRunnerPure",
        )
        self.add_parametric_feature(
            base_name="report_fz72_fr_pure_relvol",
            expr_tpl=(
                "ts_rolling_ols_residual_corr("
                f"ts_delay({rel_volume_20}, 1), "
                f"{rel_volume_20}, {ret}, {{w}}"
                ")"
            ),
            param_grid={"w": [self.PV_WINDOW]},
            category="Fangzheng72",
            sub_category="FrontRunnerPure",
        )


class ReportDailyOHLCVBatch1(BaseAlphaStrategy):
    """First daily-OHLCV batch from the report replication queue."""

    EPS = 1e-12
    EPS_EXPR = "1e-12"
    LIQUIDITY_WINDOWS = [5, 20, 60, 120]
    AMPLITUDE_WINDOWS = [21, 42, 63, 126]
    MOMENTUM_WINDOWS = [60, 120, 231]
    MOMENTUM_SKIP = 21
    REQUIRED_COLUMNS = {
        "vt_symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "market_cap_2",
    }

    def __init__(
        self,
        df: pl.DataFrame,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
        interval: str = "1d",
        enable_cache: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        df = self._validate_required_columns(df)
        super().__init__(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
            interval=interval,
            enable_cache=enable_cache,
            cache_dir=cache_dir,
        )
        self.set_label("ts_delay(close, -3) / ts_delay(close, -1) - 1")
        self.register_all()

    @classmethod
    def _validate_required_columns(cls, df: pl.DataFrame) -> pl.DataFrame:
        missing = cls.REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"ReportDailyOHLCVBatch1缺少必要字段: {missing_text}")

        return df

    @classmethod
    def turnover_rate_expr(cls) -> str:
        return f"(turnover / (market_cap_2 + {cls.EPS_EXPR}))"

    @classmethod
    def rel_volume_20_expr(cls) -> str:
        return f"(volume / (ts_mean(volume, 20) + {cls.EPS_EXPR}))"

    @classmethod
    def shortcut_expr(cls) -> str:
        return "((2 * (high - low)) - ts_abs(open - close))"

    def register_all(self) -> None:
        self.register_shortcut_illiquidity_factors()
        self.register_amihud_illiquidity_factors()
        self.register_price_volume_amplitude_factors()
        self.register_abs_return_weighted_momentum_factors()

    def register_shortcut_illiquidity_factors(self) -> None:
        shortcut = self.shortcut_expr()
        self.add_parametric_feature(
            base_name="report_shortcut_illiq",
            expr_tpl=(
                f"ts_mean({shortcut} / (turnover + {self.EPS_EXPR}), "
                "{w})"
            ),
            param_grid={"w": self.LIQUIDITY_WINDOWS},
            category="ReportDailyOHLCV",
            sub_category="KLineShortcutIlliquidity",
        )

    def register_amihud_illiquidity_factors(self) -> None:
        ret = f"(close / (ts_delay(close, 1) + {self.EPS_EXPR}) - 1)"
        oc_ret = f"(close / (open + {self.EPS_EXPR}) - 1)"
        illiq = f"ts_abs({ret}) / (turnover + {self.EPS_EXPR})"
        oc_illiq = f"ts_abs({oc_ret}) / (turnover + {self.EPS_EXPR})"

        self.add_parametric_feature(
            base_name="report_amihud_log",
            expr_tpl=f"ts_log(1 + ts_mean({illiq}, {{w}}))",
            param_grid={"w": self.LIQUIDITY_WINDOWS},
            category="ReportDailyOHLCV",
            sub_category="AmihudIlliquidity",
        )
        self.add_parametric_feature(
            base_name="report_amihud_up_log",
            expr_tpl=(
                "ts_log(1 + "
                f"ts_sum(ts_where(({ret}) > 0, {illiq}, 0), {{w}}) "
                f"/ (ts_sum(ts_where(({ret}) > 0, 1, 0), {{w}}) + {self.EPS_EXPR}))"
            ),
            param_grid={"w": self.LIQUIDITY_WINDOWS},
            category="ReportDailyOHLCV",
            sub_category="SemiIlliquidity",
        )
        self.add_parametric_feature(
            base_name="report_amihud_down_log",
            expr_tpl=(
                "ts_log(1 + "
                f"ts_sum(ts_where(({ret}) < 0, {illiq}, 0), {{w}}) "
                f"/ (ts_sum(ts_where(({ret}) < 0, 1, 0), {{w}}) + {self.EPS_EXPR}))"
            ),
            param_grid={"w": self.LIQUIDITY_WINDOWS},
            category="ReportDailyOHLCV",
            sub_category="SemiIlliquidity",
        )
        self.add_parametric_feature(
            base_name="report_oc_amihud_log",
            expr_tpl=f"ts_log(1 + ts_mean({oc_illiq}, {{w}}))",
            param_grid={"w": self.LIQUIDITY_WINDOWS},
            category="ReportDailyOHLCV",
            sub_category="OpenCloseIlliquidity",
        )
        self.add_parametric_feature(
            base_name="report_oc_amihud_up_log",
            expr_tpl=(
                "ts_log(1 + "
                f"ts_sum(ts_where(({oc_ret}) > 0, {oc_illiq}, 0), {{w}}) "
                f"/ (ts_sum(ts_where(({oc_ret}) > 0, 1, 0), {{w}}) + {self.EPS_EXPR}))"
            ),
            param_grid={"w": self.LIQUIDITY_WINDOWS},
            category="ReportDailyOHLCV",
            sub_category="OpenCloseSemiIlliquidity",
        )
        self.add_parametric_feature(
            base_name="report_oc_amihud_down_log",
            expr_tpl=(
                "ts_log(1 + "
                f"ts_sum(ts_where(({oc_ret}) < 0, {oc_illiq}, 0), {{w}}) "
                f"/ (ts_sum(ts_where(({oc_ret}) < 0, 1, 0), {{w}}) + {self.EPS_EXPR}))"
            ),
            param_grid={"w": self.LIQUIDITY_WINDOWS},
            category="ReportDailyOHLCV",
            sub_category="OpenCloseSemiIlliquidity",
        )

    def register_price_volume_amplitude_factors(self) -> None:
        turnover_rate = self.turnover_rate_expr()
        self.add_parametric_feature(
            base_name="report_price_amp",
            expr_tpl=f"ts_max(high, {{w}}) / (ts_min(low, {{w}}) + {self.EPS_EXPR}) - 1",
            param_grid={"w": self.AMPLITUDE_WINDOWS},
            category="ReportDailyOHLCV",
            sub_category="PriceVolumeAmplitude",
        )
        self.add_parametric_feature(
            base_name="report_turnover_amp_score",
            expr_tpl=(
                f"-1 * (ts_max({turnover_rate}, {{w}}) "
                f"- ts_min({turnover_rate}, {{w}}))"
            ),
            param_grid={"w": self.AMPLITUDE_WINDOWS},
            category="ReportDailyOHLCV",
            sub_category="PriceVolumeAmplitude",
        )

    def register_abs_return_weighted_momentum_factors(self) -> None:
        log_ret = f"ts_log(close / (ts_delay(close, 1) + {self.EPS_EXPR}))"
        delayed_ret = f"ts_delay({log_ret}, {{skip}})"
        abs_delayed_ret = f"ts_abs({delayed_ret})"

        self.add_parametric_feature(
            base_name="report_standard_mom",
            expr_tpl=f"ts_mean({delayed_ret}, {{w}})",
            param_grid={"skip": [self.MOMENTUM_SKIP], "w": self.MOMENTUM_WINDOWS},
            category="ReportDailyOHLCV",
            sub_category="WeightedMomentumProxy",
        )
        self.add_parametric_feature(
            base_name="report_absret_weighted_mom",
            expr_tpl=(
                f"ts_wsum({delayed_ret}, {abs_delayed_ret}, {{w}}) "
                f"/ (ts_sum({abs_delayed_ret}, {{w}}) + {self.EPS_EXPR})"
            ),
            param_grid={"skip": [self.MOMENTUM_SKIP], "w": self.MOMENTUM_WINDOWS},
            category="ReportDailyOHLCV",
            sub_category="WeightedMomentumProxy",
        )


class ReportDailyOHLCVQueue(ReportDailyOHLCVBatch1):
    """Daily-OHLCV implementation queue for the full report list."""

    SHORT_WINDOWS = [5, 10, 20]
    QUEUE_WINDOWS = [20, 60, 120]
    LONG_WINDOWS = [60, 120, 252]

    @classmethod
    def vwap_proxy_expr(cls) -> str:
        return f"(turnover / (volume + {cls.EPS_EXPR}))"

    def register_all(self) -> None:
        super().register_all()
        self.register_salience_and_reversal_proxy_factors()
        self.register_beta_proxy_factors()
        self.register_volatility_distribution_factors()
        self.register_trend_momentum_proxy_factors()
        self.register_price_volume_proxy_factors()
        self.register_turnover_liquidity_proxy_factors()
        self.register_chip_cgo_proxy_factors()
        self.register_price_shape_proxy_factors()

    def register_salience_and_reversal_proxy_factors(self) -> None:
        ret = f"(close / (ts_delay(close, 1) + {self.EPS_EXPR}) - 1)"
        log_ret = f"ts_log(close / (ts_delay(close, 1) + {self.EPS_EXPR}))"
        mkt_ret = f"cs_mean({ret})"
        salience = (
            f"ts_abs({ret} - {mkt_ret}) "
            f"/ (ts_abs({ret}) + ts_abs({mkt_ret}) + {self.EPS_EXPR})"
        )

        self.add_parametric_feature(
            base_name="report_salience_weighted_ret",
            expr_tpl=f"ts_wavg({ret}, {salience}, {{w}})",
            param_grid={"w": [20, 60]},
            category="ReportDailyOHLCVQueue",
            sub_category="SalienceReturnProxy",
        )
        self.add_parametric_feature(
            base_name="report_salient_loss_score",
            expr_tpl=(
                f"-1 * ts_wavg(ts_where({ret} < 0, ts_abs({ret}), 0), "
                f"{salience}, {{w}})"
            ),
            param_grid={"w": [20, 60]},
            category="ReportDailyOHLCVQueue",
            sub_category="SalienceReturnProxy",
        )
        self.add_parametric_feature(
            base_name="report_short_reversal",
            expr_tpl=f"-1 * ts_sum({log_ret}, {{w}})",
            param_grid={"w": self.SHORT_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="ShortReversalProxy",
        )
        self.add_parametric_feature(
            base_name="report_extreme_reversal",
            expr_tpl=f"-1 * ts_wavg({log_ret}, ts_abs({log_ret}), {{w}})",
            param_grid={"w": self.SHORT_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="ShortReversalProxy",
        )
        self.add_parametric_feature(
            base_name="report_market_adjusted_reversal",
            expr_tpl=f"-1 * (ts_sum({log_ret}, {{w}}) - ts_sum(cs_mean({log_ret}), {{w}}))",
            param_grid={"w": self.SHORT_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="ShortReversalProxy",
        )

    def register_beta_proxy_factors(self) -> None:
        ret = f"(close / (ts_delay(close, 1) + {self.EPS_EXPR}) - 1)"
        mkt_ret = f"cs_mean({ret})"
        up_ret = f"ts_where({mkt_ret} > 0, {ret}, 0)"
        up_mkt_ret = f"ts_where({mkt_ret} > 0, {mkt_ret}, 0)"
        down_ret = f"ts_where({mkt_ret} < 0, {ret}, 0)"
        down_mkt_ret = f"ts_where({mkt_ret} < 0, {mkt_ret}, 0)"

        self.add_parametric_feature(
            base_name="report_low_beta_score",
            expr_tpl=f"-1 * ts_beta({ret}, {mkt_ret}, {{w}})",
            param_grid={"w": self.LONG_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="BetaProxy",
        )
        self.add_parametric_feature(
            base_name="report_low_up_beta_score",
            expr_tpl=f"-1 * ts_beta({up_ret}, {up_mkt_ret}, {{w}})",
            param_grid={"w": self.LONG_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="BetaProxy",
        )
        self.add_parametric_feature(
            base_name="report_low_down_beta_score",
            expr_tpl=f"-1 * ts_beta({down_ret}, {down_mkt_ret}, {{w}})",
            param_grid={"w": self.LONG_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="BetaProxy",
        )

    def register_volatility_distribution_factors(self) -> None:
        ret = f"(close / (ts_delay(close, 1) + {self.EPS_EXPR}) - 1)"
        good_ret = f"ts_where({ret} > 0, {ret}, 0)"
        bad_ret = f"ts_where({ret} < 0, {ret}, 0)"

        self.add_parametric_feature(
            base_name="report_low_vol_score",
            expr_tpl=f"-1 * ts_std({ret}, {{w}})",
            param_grid={"w": self.QUEUE_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="VolatilityDistribution",
        )
        self.add_parametric_feature(
            base_name="report_bad_vol_score",
            expr_tpl=f"-1 * ts_std({bad_ret}, {{w}})",
            param_grid={"w": self.QUEUE_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="VolatilityDistribution",
        )
        self.add_parametric_feature(
            base_name="report_good_minus_bad_vol",
            expr_tpl=f"ts_std({good_ret}, {{w}}) - ts_std({bad_ret}, {{w}})",
            param_grid={"w": self.QUEUE_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="VolatilityDistribution",
        )
        self.add_parametric_feature(
            base_name="report_ret_skew",
            expr_tpl=f"ts_skew({ret}, {{w}})",
            param_grid={"w": self.QUEUE_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="ReturnAsymmetry",
        )
        self.add_parametric_feature(
            base_name="report_lottery_maxret_score",
            expr_tpl=f"-1 * ts_max({ret}, {{w}})",
            param_grid={"w": self.QUEUE_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="LotteryPreference",
        )
        self.add_parametric_feature(
            base_name="report_vol_of_vol_score",
            expr_tpl=f"-1 * ts_std(ts_std({ret}, 20), {{w}})",
            param_grid={"w": self.QUEUE_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="VolatilityOfVolatilityProxy",
        )

    def register_trend_momentum_proxy_factors(self) -> None:
        log_ret = f"ts_log(close / (ts_delay(close, 1) + {self.EPS_EXPR}))"
        mkt_log_ret = f"cs_mean({log_ret})"
        turnover_rate = self.turnover_rate_expr()
        rel_volume_20 = self.rel_volume_20_expr()

        self.add_parametric_feature(
            base_name="report_trend_sharpe",
            expr_tpl=f"ts_mean({log_ret}, {{w}}) / (ts_std({log_ret}, {{w}}) + {self.EPS_EXPR})",
            param_grid={"w": self.LONG_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="TrendMomentumProxy",
        )
        self.add_parametric_feature(
            base_name="report_turnover_weighted_mom",
            expr_tpl=f"ts_wavg({log_ret}, {turnover_rate}, {{w}})",
            param_grid={"w": self.LONG_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="TurnoverWeightedMomentum",
        )
        self.add_parametric_feature(
            base_name="report_relvol_weighted_mom",
            expr_tpl=f"ts_wavg({log_ret}, {rel_volume_20}, {{w}})",
            param_grid={"w": self.LONG_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="TurnoverWeightedMomentum",
        )
        self.add_parametric_feature(
            base_name="report_idio_mom_proxy",
            expr_tpl=f"ts_mean(ts_rolling_ols_residual({log_ret}, {mkt_log_ret}, {{w}}), {{w}})",
            param_grid={"w": self.LONG_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="IdiosyncraticMomentumProxy",
        )

    def register_price_volume_proxy_factors(self) -> None:
        ret = f"(close / (ts_delay(close, 1) + {self.EPS_EXPR}) - 1)"
        amount_mkt = "cs_mean(turnover)"
        turnover_rate = self.turnover_rate_expr()
        rel_volume_20 = self.rel_volume_20_expr()
        price_position = (
            f"(close - ts_min(close, {{w}})) "
            f"/ (ts_max(close, {{w}}) - ts_min(close, {{w}}) + {self.EPS_EXPR})"
        )

        self.add_parametric_feature(
            base_name="report_amount_market_follow",
            expr_tpl=f"ts_corr(turnover, {amount_mkt}, {{w}})",
            param_grid={"w": self.QUEUE_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="AmountMarketFollowing",
        )
        self.add_parametric_feature(
            base_name="report_amount_follow_lowpos_score",
            expr_tpl=f"-1 * ts_corr(turnover, {amount_mkt}, {{w}}) * (1 - {price_position})",
            param_grid={"w": self.QUEUE_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="AmountMarketFollowing",
        )
        self.add_parametric_feature(
            base_name="report_price_volume_corr",
            expr_tpl=f"ts_corr({ret}, {turnover_rate}, {{w}})",
            param_grid={"w": self.QUEUE_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="PriceVolumeRelation",
        )
        self.add_parametric_feature(
            base_name="report_volume_volatility_corr",
            expr_tpl=f"ts_corr(ts_abs({ret}), {rel_volume_20}, {{w}})",
            param_grid={"w": self.QUEUE_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="VolumeVolatilityRelation",
        )

    def register_turnover_liquidity_proxy_factors(self) -> None:
        turnover_rate = self.turnover_rate_expr()
        self.add_parametric_feature(
            base_name="report_turnover_mean_score",
            expr_tpl=f"-1 * ts_mean({turnover_rate}, {{w}})",
            param_grid={"w": self.QUEUE_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="TurnoverLiquidity",
        )
        self.add_parametric_feature(
            base_name="report_turnover_bias_score",
            expr_tpl=(
                f"-1 * (ts_mean({turnover_rate}, {{w}}) "
                f"/ (ts_mean({turnover_rate}, 252) + {self.EPS_EXPR}) - 1)"
            ),
            param_grid={"w": self.QUEUE_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="TurnoverLiquidity",
        )
        self.add_parametric_feature(
            base_name="report_turnover_vol_score",
            expr_tpl=f"-1 * ts_std({turnover_rate}, {{w}})",
            param_grid={"w": self.QUEUE_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="TurnoverLiquidity",
        )
        self.add_parametric_feature(
            base_name="report_turnover_volume_disagree_score",
            expr_tpl=f"-1 * ts_std({turnover_rate} - cs_mean({turnover_rate}), {{w}})",
            param_grid={"w": self.QUEUE_WINDOWS},
            category="ReportDailyOHLCVQueue",
            sub_category="HeterogeneousBeliefProxy",
        )

    def register_chip_cgo_proxy_factors(self) -> None:
        turnover_rate = self.turnover_rate_expr()
        cost = f"ts_wavg(close, {turnover_rate}, {{w}})"

        self.add_parametric_feature(
            base_name="report_cgo_proxy",
            expr_tpl=f"(close - {cost}) / (close + {self.EPS_EXPR})",
            param_grid={"w": [60, 120]},
            category="ReportDailyOHLCVQueue",
            sub_category="CGOChipProxy",
        )
        self.add_parametric_feature(
            base_name="report_chip_cost_gap_score",
            expr_tpl=f"-1 * ts_abs(close / ({cost} + {self.EPS_EXPR}) - 1)",
            param_grid={"w": [60, 120]},
            category="ReportDailyOHLCVQueue",
            sub_category="CGOChipProxy",
        )
        self.add_parametric_feature(
            base_name="report_chip_concentration_score",
            expr_tpl=(
                f"-1 * ts_wavg(ts_abs(close / ({cost} + {self.EPS_EXPR}) - 1), "
                f"{turnover_rate}, {{w}})"
            ),
            param_grid={"w": [60, 120]},
            category="ReportDailyOHLCVQueue",
            sub_category="ChipDistributionProxy",
        )

    def register_price_shape_proxy_factors(self) -> None:
        vwap_proxy = self.vwap_proxy_expr()
        upper_shadow = (
            f"(high - ts_where(open > close, open, close)) "
            f"/ (high - low + {self.EPS_EXPR})"
        )
        lower_shadow = (
            f"(ts_where(open < close, open, close) - low) "
            f"/ (high - low + {self.EPS_EXPR})"
        )

        self.add_parametric_feature(
            base_name="report_open_rush_score",
            expr_tpl=f"-1 * ts_mean(high / (open + {self.EPS_EXPR}) - 1, {{w}})",
            param_grid={"w": [20, 60]},
            category="ReportDailyOHLCVQueue",
            sub_category="PriceShapeProxy",
        )
        self.add_parametric_feature(
            base_name="report_low_recovery",
            expr_tpl=f"ts_mean((close - low) / (low + {self.EPS_EXPR}), {{w}})",
            param_grid={"w": [20, 60]},
            category="ReportDailyOHLCVQueue",
            sub_category="PriceShapeProxy",
        )
        self.add_parametric_feature(
            base_name="report_upper_shadow_score",
            expr_tpl=f"-1 * ts_mean({upper_shadow}, {{w}})",
            param_grid={"w": [20, 60]},
            category="ReportDailyOHLCVQueue",
            sub_category="PriceShapeProxy",
        )
        self.add_parametric_feature(
            base_name="report_lower_shadow_score",
            expr_tpl=f"ts_mean({lower_shadow}, {{w}})",
            param_grid={"w": [20, 60]},
            category="ReportDailyOHLCVQueue",
            sub_category="PriceShapeProxy",
        )
        self.add_parametric_feature(
            base_name="report_vwap_deviation",
            expr_tpl=f"ts_mean(close / ({vwap_proxy} + {self.EPS_EXPR}) - 1, {{w}})",
            param_grid={"w": [20, 60]},
            category="ReportDailyOHLCVQueue",
            sub_category="PriceShapeProxy",
        )
