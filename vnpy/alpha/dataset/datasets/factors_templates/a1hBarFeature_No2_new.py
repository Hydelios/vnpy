# file: strategies/a1hBarFeature_helios.py
from vnpy.alpha.dataset.feature_spec import BaseFeatureSpec

class a60mBarFeature(BaseFeatureSpec):
    """
    只依赖 1h bar 的因子补全集：
      - 趋势/震荡: Slope, R2, Stable Momentum
    """

    def register(
        self,
        *,
        interval: str = "60m",
        windows: list[int] | None = None,
    ) -> None:
        self.interval = interval
        self._default_windows = windows or [4, 8, 24, 48]  # 稍微补充了一个24小时的周期
        self.require_bars("open", "high", "low", "close", "volume", "vwap")
        self.register_trend()
        self.register_intraday_pattern()
        self.register_gap()

    # ===================== 趋势 / 震荡 =====================
    def register_trend(self):
        W = self._default_windows
        # 定义需要计算的字段 (OHLCV + VWAP)
        fields = ["open", "high", "low", "close", "volume", "vwap"]

        # ----------------------------------------------------
        # 1. 基础因子：斜率 (Slope) - 衡量趋势的速度/强度
        # ----------------------------------------------------
        self.add_parametric_feature(
            base_name=f"feat_{self.interval}_trend_slope",
            expr_tpl="ts_slope(ts_log({value}), {window})",
            param_grid={
                "window": W,
                "value": fields
            },
            category="Trend",
            sub_category="Slope"
        )

        # ----------------------------------------------------
        # 2. 基础因子：拟合度 (R-Squared) - 衡量趋势的稳定性/线性度
        # ----------------------------------------------------
        self.add_parametric_feature(
            base_name=f"feat_{self.interval}_trend_rsquare",
            expr_tpl="ts_rsquare(ts_log({value}), {window})",
            param_grid={
                "window": W,
                "value": fields
            },
            category="Trend",
            sub_category="Stability"
        )

        # ----------------------------------------------------
        # 3. 组合因子：稳健动量 (Stable Momentum) - 原始值相乘
        # 逻辑：涨得快(Slope大) 且 涨得稳(R2高) = 最好的趋势
        # ----------------------------------------------------
        self.add_parametric_feature(
            base_name=f"feat_{self.interval}_mom_stable_raw",
            expr_tpl="ts_slope(ts_log({value}), {window}) * ts_rsquare(ts_log({value}), {window})",
            param_grid={
                "window": W,
                "value": fields 
            },
            category="Momentum",
            sub_category="Stable_Raw"
        )

        # ----------------------------------------------------
        # 4. 组合因子：归一化稳健动量 (Rank * Rank)
        # 逻辑：先对 Slope 和 R2 分别做截面排名(0~1)，再相乘。
        # 优势：消除了 Slope 和 R2 量纲不一致的问题，对离群值更鲁棒。
        # ----------------------------------------------------
        self.add_parametric_feature(
            base_name=f"feat_{self.interval}_mom_stable_rank",
            expr_tpl="cs_rank(ts_slope(ts_log({value}), {window})) * cs_rank(ts_rsquare(ts_log({value}), {window}))",
            param_grid={
                "window": W,
                "value": fields
            },
            category="Momentum",
            sub_category="Stable_Rank"
        )

        # ----------------------------------------------------
        # 5. 值的变化 (Delta)
        # 含义：当前值相对于 window 前的变化量 (x_t - x_{t-w})
        # 注意：对于价格列，建议用 log 处理以消除高低价股差异
        # ----------------------------------------------------
        self.add_parametric_feature(
            base_name=f"feat_{self.interval}_trend_delta",
            expr_tpl="ts_delta({value}, {window})",  # 或者 ts_delta(ts_log({value}), {window})
            param_grid={
                "window": W,
                "value": fields
            },
            category="Trend",
            sub_category="Delta"
        )

        # ----------------------------------------------------
        # 6. 值的变化率 (ROC - Rate of Change)
        # 含义：(当前值 / 过去值) - 1，即 N周期收益率
        # 这是最朴素但也最有效的动量因子
        # ----------------------------------------------------
        self.add_parametric_feature(
            base_name=f"feat_{self.interval}_trend_roc",
            expr_tpl="{value} / ts_delay({value}, {window}) - 1",
            param_grid={
                "window": W,
                "value": fields
            },
            category="Trend",
            sub_category="ROC"
        )

        # ----------------------------------------------------
        # 7. 移动平均值 (MA) & 乖离率 (Bias)
        # 注意：单纯的 MA (如3000点) 不是平稳序列，直接入模效果不好。
        # 我们这里计算 "Bias" (当前价 / 均价 - 1)，衡量价格偏离趋势的程度
        # 如果你确实需要原始 MA 值，就把 expr_tpl 改为 "ts_mean({value}, {window})"
        # ----------------------------------------------------
        self.add_parametric_feature(
            base_name=f"feat_{self.interval}_trend_bias",
            expr_tpl="{value} / ts_mean({value}, {window}) - 1",
            param_grid={
                "window": W,
                "value": fields
            },
            category="Trend",
            sub_category="Bias" # 乖离率
        )

        # ----------------------------------------------------
        # 8. 均值的变化率 (MA ROC / MA Speed)
        # 含义：均线本身是在变大还是变小？(MA_t / MA_{t-1} - 1)
        # 逻辑：均线向上 = 趋势向上；均线走平 = 震荡。这是对趋势方向最平滑的度量。
        # 这里我们考察 MA 在过去 1 个单位时间的变化率 (瞬时速度)
        # ----------------------------------------------------
        self.add_parametric_feature(
            base_name=f"feat_{self.interval}_trend_ma_speed",
            expr_tpl="ts_mean({value}, {window}) / ts_delay(ts_mean({value}, {window}), 1) - 1",
            param_grid={
                "window": W,
                "value": fields
            },
            category="Trend",
            sub_category="MA_Speed"
        )

    def register_intraday_pattern(self):
        # 假设 daily_window = 4 (对应一天4小时)
        DW = 4 
        
        # 1. 日内趋势一致性 (Intraday Correlation/Slope)
        # 如果是正值，说明这4小时一路向上；负值说明一路向下
        # ts_rank(price, DW) 实际上是在看当前的 price 在过去 4 个小时里的位置
        # 更好的办法是直接算 slope
        self.add_parametric_feature(
            base_name=f"feat_{self.interval}_intraday_trend_consistency",
            expr_tpl="ts_slope({value}, {window})", 
            param_grid={"window": [DW], "value": ["vwap", "close"]},
            category="Pattern", sub_category="IntradayTrend"
        )

        # 2. 日内路径效率 (Efficiency Ratio)
        # 净涨幅 / 波动总和。 1 = 直线，0 = 疯狂震荡
        # abs(close - close_t-4) / sum(abs(close - close_t-1), 4)
        self.add_parametric_feature(
            base_name=f"feat_{self.interval}_intraday_efficiency",
            expr_tpl="ts_abs(ts_delta({value}, {window})) / ts_sum(ts_abs(ts_delta({value}, 1)), {window})",
            param_grid={"window": [DW], "value": ["close"]},
            category="Pattern", sub_category="Efficiency"
        )

        # 3. 日内反转力度 (Intraday Reversal)
        # (最高价 - 收盘价) / (最高价 - 最低价)
        # 如果接近 1，说明收在最低，日内冲高回落（墓碑线）
        # 这里需要用 rolling_max/min 模拟日内的 High/Low
        self.add_parametric_feature(
            base_name=f"feat_{self.interval}_intraday_selling_pressure",
            expr_tpl="(ts_max(high, {window}) - close) / (ts_max(high, {window}) - ts_min(low, {window}) + 1e-9)",
            param_grid={"window": [DW]},
            category="Pattern", sub_category="SellingPressure"
        )

    # ===================== 隔夜 / 缺口 (Gap) =====================
    def register_gap(self):
        # 1天, 2天, 1周的窗口 (Bar为单位)
        W = [4, 8, 24] 
        
        # 基础组件：
        # Gap Return (当前开盘 / 上一根收盘 - 1)
        # Intraday Return (当前收盘 / 当前开盘 - 1)
        expr_gap = "(open / ts_delay(close, 1) - 1)"
        expr_intraday = "(close / open - 1)"

        # ----------------------------------------------------
        # 思路 A: 隔夜乖离 / 超买超卖 (Gap Bias)
        # 逻辑：跳空幅度 / 波动率。如果大幅高开且超过了正常波动范围 -> 看空
        # ----------------------------------------------------
        self.add_parametric_feature(
            base_name=f"feat_{self.interval}_gap_bias_zscore",
            expr_tpl=f"{expr_gap} / ts_std(close, {{window}})", 
            param_grid={"window": W},
            category="Overnight", sub_category="GapBias"
        )

        # ----------------------------------------------------
        # 思路 B: 隔夜与日内反转 (Gap Reversal)
        # 逻辑：隔夜收益 * 日内收益。
        # 预期为负 (高开低走 或 低开高走)。因子值越小(越负)，反转效应越强。
        # ----------------------------------------------------
        self.add_parametric_feature(
            base_name=f"feat_{self.interval}_gap_reversal_prod",
            expr_tpl=f"{expr_gap} * {expr_intraday}",
            param_grid=None, # 这是一个瞬时状态，通常不需要 rolling mean，或者短窗口平滑
            category="Overnight", sub_category="GapReversal"
        )
        
        # ----------------------------------------------------
        # 思路 C: 聪明钱博弈 (Smart Money Proxy) - 滚动版
        # 逻辑：(开盘成交量 / 昨日收盘成交量) * (开盘涨幅 / 昨日收盘涨幅)
        # 这里的实现是“瞬时”的，真正的“早盘 vs 尾盘”对比需要在合成阶段做。
        # 这里我们计算：当前 Bar 的量价动能相对于上一根 Bar 的突变程度。
        # ----------------------------------------------------
        # 定义：Bar 动能 = (Close/Open - 1) * Volume
        expr_bar_power = f"({expr_intraday} * volume)"
        
        self.add_parametric_feature(
            base_name=f"feat_{self.interval}_gap_smart_money_flow",
            expr_tpl=f"{expr_bar_power} - ts_delay({expr_bar_power}, 1)",
            param_grid=None,
            category="Overnight", sub_category="SmartMoneyFlow"
        )
