# file: strategies/aapollo_etf_bar_pack_v1.py
from .baseAlphaStrategy import BaseAlphaStrategy

class a1hBarFeature(BaseAlphaStrategy):
    """
    只依赖 1h bar 的因子补全集：
      - 趋势/震荡: KDJ-K/J, RSI, CCI, PPO, KO, OBV, PriceOsc, MA差, TrendIndex, NR
      - 波动/方向分解: Range, RSJ, RVI, UpVolRatio
      - 历史统计: 累计收益、其 rolling std/skew/kurt、香农熵
      - 量能结构: Volume Concentration/Ratio/Imbalance 系
    说明：
      * 仅使用 ts_* / cs_* / extra_* 表达式；无 L2 依赖。
      * 公式尽量贴近你原文件定义；窗口集合与命名保持可读。
    """

    def __init__(
        self,
        df,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
        interval: str = "60m",
        enable_cache: bool = False,
        cache_dir: str | None = None,
        label_horizon: int = 5,
    ):
        super().__init__(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
            interval=interval,
            enable_cache=enable_cache,
            cache_dir=cache_dir,
        )

        # Label：前视收益（需别的标签可自行替换）
        self.set_label(f"ts_delay(close, -{label_horizon}) / close - 1")

        # 注册各家族（均只用 OHLCV/turnover）
        self.register_trend_oscillators()
        self.register_volatility_family()
        self.register_history_stats_family()
        self.register_volume_structure_family()

    # ===================== 趋势 / 震荡 =====================
    def register_trend_oscillators(self):
        W = [50, 100, 200, 300, 600]

        # 1) Price Oscillator ~ (快速EMA - 慢速EMA) / rolling range
        self.add_parametric_feature(
            base_name="feat1h_oscillator",
            expr_tpl=(
                "(ts_ema(close, {s_fast}) - ts_ema(close, {s_slow})) / "
                "(ts_max(close, {window}) - ts_min(close, {window}) + 1e-12)"
            ),
            param_grid={"window": W, "s_fast": [5, 10, 20], "s_slow": [50, 100, 200, 300, 600]},
            category="Trend", sub_category="PriceOsc",
        )

        # 2) range.pos（区间位置，居中到 [-0.5, 0.5]）
        self.add_parametric_feature(
            base_name="feat1h_range_pos",
            expr_tpl=(
                "(close - ts_min(close, {window})) / "
                "(ts_max(close, {window}) - ts_min(close, {window}) + 1e-12) - 0.5"
            ),
            param_grid={"window": W},
            category="Trend", sub_category="RangePos",
        )

        # 3) KDJ-K（按原式：RSV→平滑→中心化放大）
        self.add_parametric_feature(
            base_name="feat1h_kdj",
            expr_tpl=(
                "ts_ema(("
                "  (close - ts_min(close, {window})) / (ts_max(close, {window}) - ts_min(close, {window}) + 1e-12)"
                "  - 0.5"
                ") * 2, {smooth})"
            ),
            param_grid={"window": W, "smooth": [5, 10, 20]},
            category="Trend", sub_category="KDJ",
        )

        # 4) KDJ-J（再平滑一次）
        self.add_parametric_feature(
            base_name="feat1h_kdj_j",
            expr_tpl=(
                "ts_ema(ts_ema(("
                "  (close - ts_min(close, {window})) / (ts_max(close, {window}) - ts_min(close, {window}) + 1e-12)"
                "  - 0.5"
                ") * 2, {smooth}), {smooth})"
            ),
            param_grid={"window": W, "smooth": [5, 10, 20]},
            category="Trend", sub_category="KDJ",
        )

        # 5) RSI（EMA 版）
        self.add_parametric_feature(
            base_name="feat1h_rsi",
            expr_tpl=(
                "1 - 1 / (1 + ("
                "  ts_ema(ts_where(ts_delta(close,1)>0,  ts_delta(close,1), 0), {span})"
                "  / (ts_ema(ts_where(ts_delta(close,1)<=0, -ts_delta(close,1), 0), {span}) + 1e-12)"
                "))"
            ),
            param_grid={"span": [64, 128, 256, 512]},
            category="Trend", sub_category="RSI",
        )

        # 6) CCI（用 HLC/3 典型价）
        self.add_parametric_feature(
            base_name="feat1h_cci",
            expr_tpl=(
                "((high + low + close) / 3 - ts_mean((high + low + close) / 3, {window})) / "
                "(0.015 * ts_sum(abs((high + low + close) / 3 - ts_mean((high + low + close) / 3, {window})), {window})"
                " + 1e-12)"
            ),
            param_grid={"window": W},
            category="Trend", sub_category="CCI",
        )

        # 7) PPO： (EMA_fast - EMA_slow) / EMA_slow
        self.add_parametric_feature(
            base_name="feat1h_ppo",
            expr_tpl=(
                "(ts_ema(close, {s_fast}) - ts_ema(close, {s_slow})) / "
                "(ts_ema(close, {s_slow}) + 1e-12)"
            ),
            param_grid={"s_fast": [10, 20, 30], "s_slow": [50, 100, 200, 300, 600]},
            category="Trend", sub_category="PPO",
        )

        # 8) KO（Klinger-like，基于 TP 变动方向 × 量）
        self.add_parametric_feature(
            base_name="feat1h_ko",
            expr_tpl=(
                "ts_ema(ts_delay(volume,1) * ts_sign(ts_delta((high+low+close)/3, 1)), {s_slow})"
                " - ts_ema(ts_delay(volume,1) * ts_sign(ts_delta((high+low+close)/3, 1)), {s_fast})"
            ),
            param_grid={"s_fast": [10, 20], "s_slow": [50, 100, 200]},
            category="Trend", sub_category="KO",
        )

        # 9) OBV（平滑差）
        self.add_parametric_feature(
            base_name="feat1h_obv",
            expr_tpl=(
                "ts_ema(ts_delay(volume,1) * ts_sign(ts_delta(close,1)), {s_slow})"
                " - ts_ema(ts_delay(volume,1) * ts_sign(ts_delta(close,1)), {s_fast})"
            ),
            param_grid={"s_fast": [10, 20], "s_slow": [50, 100, 200]},
            category="Trend", sub_category="OBV",
        )

        # 10) MA 差（近似原 “ma.dif.10.period”）
        self.add_parametric_feature(
            base_name="feat1h_ma_dif",
            expr_tpl=(
                "(ts_ema(close, {s_short}) - ts_ema(close, {s_long})) / (close + 1e-12)"
            ),
            param_grid={"s_short": [5, 10, 20], "s_long": [50, 100, 200, 300, 600]},
            category="Trend", sub_category="MADiff",
        )

        # 11) 趋势指数 TrendIndex：|Δ_{W}| / range_{W}
        self.add_parametric_feature(
            base_name="feat1h_trend_index",
            expr_tpl=(
                "abs(close - ts_delay(close, {window})) / "
                "(ts_max(close, {window}) - ts_min(close, {window}) + 1e-12)"
            ),
            param_grid={"window": W},
            category="Trend", sub_category="TrendIndex",
        )

        # 12) MA Price Power：MA / lag(MA, n) - 1
        self.add_parametric_feature(
            base_name="feat1h_ma_price",
            expr_tpl=(
                "ts_ema(close, {ma_span}) / (ts_delay(ts_ema(close, {ma_span}), {lag}) + 1e-12) - 1"
            ),
            param_grid={"ma_span": [100, 300], "lag": [100, 300]},
            category="Trend", sub_category="MAPower",
        )

        # 13) NR：E[ret] / E[|ret|]
        self.add_parametric_feature(
            base_name="feat1h_nr",
            expr_tpl=(
                "ts_ema(ts_delta(ts_log(close),1), {window}) / "
                "(ts_ema(abs(ts_delta(ts_log(close),1)), {window}) + 1e-12)"
            ),
            param_grid={"window": W},
            category="Trend", sub_category="NR",
        )

    # ===================== 波动 / 区间 / 方向分解 =====================
    def register_volatility_family(self):
        W = [50, 100, 200, 300, 600]

        # 1) 区间 Range
        self.add_parametric_feature(
            base_name="feat1h_range_period",
            expr_tpl="ts_max(close, {window}) - ts_min(close, {window})",
            param_grid={"window": W},
            category="Volatility", sub_category="Range",
        )

        # 2) RSJ： (RV_pos - RV_neg) / RV_total
        # self.add_parametric_feature(
        #     base_name="feat1h_rsj_volatility",
        #     expr_tpl=(
        #         "("
        #         "  ts_sum((ts_delta(ts_log(close),1)*ts_delta(close,1>0)) * ts_delta(ts_log(close),1), {window})"
        #         " - ts_sum((ts_delta(ts_log(close),1)*ts_delta(close,1<=0)) * ts_delta(ts_log(close),1), {window})"
        #         ") / (ts_sum((ts_delta(ts_log(close),1))*(ts_delta(ts_log(close),1)), {window}) + 1e-12)"
        #     ),
        #     # 为避免表达式里 >0 / <=0 歧义，改用 where 版本更稳：
        #     # 最终我们用 where 版本：
        # )
        # 用 ts_where 的版本（覆盖上面的注册，避免解析器对布尔表达式不友好）
        self.add_parametric_feature(
            base_name="feat1h_rsj_volatility_where",
            expr_tpl=(
                " ("
                "   ts_sum(ts_where(ts_delta(close,1)>0,  (ts_delta(ts_log(close),1)*ts_delta(ts_log(close),1)), 0), {window})"
                " - ts_sum(ts_where(ts_delta(close,1)<=0, (ts_delta(ts_log(close),1)*ts_delta(ts_log(close),1)), 0), {window})"
                " ) / (ts_sum((ts_delta(ts_log(close),1)*ts_delta(ts_log(close),1)), {window}) + 1e-12)"
            ),
            param_grid={"window": W},
            category="Volatility", sub_category="RSJ",
        )

        # 3) RVI：对波动变化做 RSI
        self.add_parametric_feature(
            base_name="feat1h_rvi_period",
            expr_tpl=(
                "1 - 1 / (1 + ("
                "  ts_ema(ts_where(ts_delta(ts_std(close, {window}),1)>0,  ts_delta(ts_std(close, {window}),1), 0), {window})"
                "  / (ts_ema(ts_where(ts_delta(ts_std(close, {window}),1)<=0, -ts_delta(ts_std(close, {window}),1), 0), {window}) + 1e-12)"
                "))"
            ),
            param_grid={"window": W},
            category="Volatility", sub_category="RVI",
        )

        # 4) Up Volatility Ratio：∑(ret^2 仅正) / ∑(ret^2)
        self.add_parametric_feature(
            base_name="feat1h_up_volatility_ratio",
            expr_tpl=(
                "ts_sum(ts_where(ts_delta(ts_log(close),1)>0, (ts_delta(ts_log(close),1)*ts_delta(ts_log(close),1)), 0), {window})"
                " / (ts_sum((ts_delta(ts_log(close),1)*ts_delta(ts_log(close),1)), {window}) + 1e-12)"
            ),
            param_grid={"window": W},
            category="Volatility", sub_category="UpVolRatio",
        )

    # ===================== 历史统计类 =====================
    def register_history_stats_family(self):
        W = [50, 100, 200, 300, 600]

        # 1) 累计收益（rolling sum of log-ret）
        self.add_parametric_feature(
            base_name="feat1h_ret_history",
            expr_tpl="ts_sum(ts_delta(ts_log(close),1), {window})",
            param_grid={"window": W},
            category="Stats", sub_category="CumRet",
        )

        # 2) 累计收益的 rolling std（等价于对 cumret 再做 std）
        self.add_parametric_feature(
            base_name="feat1h_ret_history_std",
            expr_tpl="ts_std(ts_sum(ts_delta(ts_log(close),1), {window}), {window})",
            param_grid={"window": W},
            category="Stats", sub_category="CumRetStd",
        )

        # 3) 累计收益的 rolling skew
        self.add_parametric_feature(
            base_name="feat1h_ret_history_skew",
            expr_tpl=(
                "ts_mean( (ts_sum(ts_delta(ts_log(close),1), {window}) - ts_mean(ts_sum(ts_delta(ts_log(close),1), {window}), {window}))"
                "        * (ts_sum(ts_delta(ts_log(close),1), {window}) - ts_mean(ts_sum(ts_delta(ts_log(close),1), {window}), {window}))"
                "        * (ts_sum(ts_delta(ts_log(close),1), {window}) - ts_mean(ts_sum(ts_delta(ts_log(close),1), {window}), {window}))"
                ", {window})"
                " / ((ts_std(ts_sum(ts_delta(ts_log(close),1), {window}), {window}) + 1e-12)"
                "    * (ts_std(ts_sum(ts_delta(ts_log(close),1), {window}), {window}) + 1e-12)"
                "    * (ts_std(ts_sum(ts_delta(ts_log(close),1), {window}), {window}) + 1e-12))"
            ),
            param_grid={"window": W},
            category="Stats", sub_category="CumRetSkew",
        )

        # 4) 累计收益的 rolling kurt（Fisher/非过度统一不做减3）
        self.add_parametric_feature(
            base_name="feat1h_ret_history_kurt",
            expr_tpl=(
                "ts_mean( (ts_sum(ts_delta(ts_log(close),1), {window}) - ts_mean(ts_sum(ts_delta(ts_log(close),1), {window}), {window}))"
                "        * (ts_sum(ts_delta(ts_log(close),1), {window}) - ts_mean(ts_sum(ts_delta(ts_log(close),1), {window}), {window}))"
                "        * (ts_sum(ts_delta(ts_log(close),1), {window}) - ts_mean(ts_sum(ts_delta(ts_log(close),1), {window}), {window}))"
                "        * (ts_sum(ts_delta(ts_log(close),1), {window}) - ts_mean(ts_sum(ts_delta(ts_log(close),1), {window}), {window}))"
                ", {window})"
                " / ((ts_std(ts_sum(ts_delta(ts_log(close),1), {window}), {window}) + 1e-12)"
                "    * (ts_std(ts_sum(ts_delta(ts_log(close),1), {window}), {window}) + 1e-12)"
                "    * (ts_std(ts_sum(ts_delta(ts_log(close),1), {window}), {window}) + 1e-12)"
                "    * (ts_std(ts_sum(ts_delta(ts_log(close),1), {window}), {window}) + 1e-12))"
            ),
            param_grid={"window": W},
            category="Stats", sub_category="CumRetKurt",
        )

        # 5) 香农熵（按方向）
        self.add_parametric_feature(
            base_name="feat1h_shannon_entropy",
            expr_tpl=(
                # p = E[1{ret>=0}]
                "(-1.4426950408889634) * ("
                "  (ts_mean(ts_where(ts_delta(ts_log(close),1) >= 0, 1, 0), {window})"
                "   * ts_log(ts_mean(ts_where(ts_delta(ts_log(close),1) >= 0, 1, 0), {window}) + 1e-12))"
                "  + ((1 - ts_mean(ts_where(ts_delta(ts_log(close),1) >= 0, 1, 0), {window}))"
                "     * ts_log(1 - ts_mean(ts_where(ts_delta(ts_log(close),1) >= 0, 1, 0), {window}) + 1e-12))"
                ")"
            ),
            param_grid={"window": [100, 300, 600]},
            category="Stats", sub_category="ShannonEntropy",
        )

        # 6) 昨收/现价（对数比或比值-1，保留一个实现）
        self.add_parametric_feature(
            base_name="feat1h_prev_close_divide_latest",
            expr_tpl="ts_delay(close, 1) / (close + 1e-12) - 1",
            param_grid={},
            category="Stats", sub_category="PrevCloseVsNow",
        )

    # ===================== 量能结构与方向分解 =====================
    def register_volume_structure_family(self):
        W = [100, 300, 600]

        # 通用定义
        # vol_chg = Δvolume, px_chg = Δclose
        vol_chg = "ts_delta(volume, 1)"
        px_chg  = "ts_delta(close, 1)"

        # 0) Volume Concentration（平方和 / 和的平方）
        self.add_parametric_feature(
            base_name="feat1h_volume_concentration_period",
            expr_tpl=(
                f"ts_sum(({vol_chg})*({vol_chg}), {{window}})"
                " / ((ts_sum(" + vol_chg + ", {window}) * ts_sum(" + vol_chg + ", {window})) + 1e-12)"
            ),
            param_grid={"window": W},
            category="Volume", sub_category="Concentration",
        )

        # 1) Buy Volume Concentration
        self.add_parametric_feature(
            base_name="feat1h_buy_volume_concentration",
            expr_tpl=(
                "ts_sum(ts_where(" + px_chg + ">0, " + vol_chg + ", 0) * ts_where(" + px_chg + ">0, " + vol_chg + ", 0), {window})"
                " / ((ts_sum(ts_where(" + px_chg + ">0, " + vol_chg + ", 0), {window})"
                "     * ts_sum(ts_where(" + px_chg + ">0, " + vol_chg + ", 0), {window})) + 1e-12)"
            ),
            param_grid={"window": W},
            category="Volume", sub_category="BuyConcentration",
        )

        # 2) Sell Volume Concentration
        self.add_parametric_feature(
            base_name="feat1h_sell_volume_concentration",
            expr_tpl=(
                "ts_sum(ts_where(" + px_chg + "<0, " + vol_chg + ", 0) * ts_where(" + px_chg + "<0, " + vol_chg + ", 0), {window})"
                " / ((ts_sum(ts_where(" + px_chg + "<0, " + vol_chg + ", 0), {window})"
                "     * ts_sum(ts_where(" + px_chg + "<0, " + vol_chg + ", 0), {window})) + 1e-12)"
            ),
            param_grid={"window": W},
            category="Volume", sub_category="SellConcentration",
        )

        # 3) Buy Volume Ratio：∑buy/∑全体
        self.add_parametric_feature(
            base_name="feat1h_buy_volume_ratio",
            expr_tpl=(
                "ts_sum(ts_where(" + px_chg + ">0, " + vol_chg + ", 0), {window})"
                " / (ts_sum(" + vol_chg + ", {window}) + 1e-12)"
            ),
            param_grid={"window": W},
            category="Volume", sub_category="BuyRatio",
        )

        # 4) Sell Volume Ratio
        self.add_parametric_feature(
            base_name="feat1h_sell_volume_ratio",
            expr_tpl=(
                "ts_sum(ts_where(" + px_chg + "<0, " + vol_chg + ", 0), {window})"
                " / (ts_sum(" + vol_chg + ", {window}) + 1e-12)"
            ),
            param_grid={"window": W},
            category="Volume", sub_category="SellRatio",
        )

        # 5) Buy-Sell Volume Imbalance： (∑buy - ∑sell) / ∑全体
        self.add_parametric_feature(
            base_name="feat1h_imb_buy_sell_volume.",
            expr_tpl=(
                "("
                "  ts_sum(ts_where(" + px_chg + ">0, " + vol_chg + ", 0), {window})"
                "  - ts_sum(ts_where(" + px_chg + "<0, " + vol_chg + ", 0), {window})"
                ") / (ts_sum(" + vol_chg + ", {window}) + 1e-12)"
            ),
            param_grid={"window": W},
            category="Volume", sub_category="Imbalance",
        )

        # 6) Buy/Sell Concentration Imbalance： (BC - SC)/(BC + SC)
        self.add_parametric_feature(
            base_name="feat1h_imb_buy_sell_volume_concentration",
            expr_tpl=(
                "("
                "  (ts_sum(ts_where(" + px_chg + ">0, " + vol_chg + ", 0) * ts_where(" + px_chg + ">0, " + vol_chg + ", 0), {window})"
                "   / (ts_sum(ts_where(" + px_chg + ">0, " + vol_chg + ", 0), {window})"
                "      * ts_sum(ts_where(" + px_chg + ">0, " + vol_chg + ", 0), {window}) + 1e-12))"
                " - (ts_sum(ts_where(" + px_chg + "<0, " + vol_chg + ", 0) * ts_where(" + px_chg + "<0, " + vol_chg + ", 0), {window})"
                "   / (ts_sum(ts_where(" + px_chg + "<0, " + vol_chg + ", 0), {window})"
                "      * ts_sum(ts_where(" + px_chg + "<0, " + vol_chg + ", 0), {window}) + 1e-12))"
                ") / ("
                "  (ts_sum(ts_where(" + px_chg + ">0, " + vol_chg + ", 0) * ts_where(" + px_chg + ">0, " + vol_chg + ", 0), {window})"
                "   / (ts_sum(ts_where(" + px_chg + ">0, " + vol_chg + ", 0), {window})"
                "      * ts_sum(ts_where(" + px_chg + ">0, " + vol_chg + ", 0), {window}) + 1e-12))"
                " + (ts_sum(ts_where(" + px_chg + "<0, " + vol_chg + ", 0) * ts_where(" + px_chg + "<0, " + vol_chg + ", 0), {window})"
                "   / (ts_sum(ts_where(" + px_chg + "<0, " + vol_chg + ", 0), {window})"
                "      * ts_sum(ts_where(" + px_chg + "<0, " + vol_chg + ", 0), {window}) + 1e-12))"
                " + 1e-12)"
            ),
            param_grid={"window": W},
            category="Volume", sub_category="ConcImb",
        )
