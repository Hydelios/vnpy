import polars as pl

from vnpy.alpha import Segment
from vnpy.alpha.dataset import label_postprocess
from vnpy.alpha.dataset.processor_post import neutralize_label_columns

from .baseAlphaStrategy import BaseAlphaStrategy

class LabelStrategy(BaseAlphaStrategy):
    """
    Label 专用策略
    逻辑：专门用来计算成交逻辑与各类训练目标（收益率、波动率缩放收益率）
    """

    def __init__(
        self,
        df,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
        interval: str = "1d",
        enable_cache: bool = False,
        cache_dir: str | None = None,
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

        # =========================================================
        # overnight return 收益率计算
        # 设置 Label：当日收盘买入，明日开盘卖出
        # =========================================================
        self.add_feature("label_overnight_ret", "ts_delay(open, -1) / close - 1")

        # =========================================================
        # open to open 收益率计算
        # 设置 Label：T+1 开盘买入，持有 H 天
        # 注意：ts_delay(..., -N) 在表达式引擎中通常代表获取未来第 N 天的数据
        # =========================================================
        o2o_h1 = "ts_delay(open, -2) / ts_delay(open, -1) - 1"
        o2o_h3 = "ts_delay(open, -4) / ts_delay(open, -1) - 1"
        o2o_h5 = "ts_delay(open, -6) / ts_delay(open, -1) - 1"
        o2o_h7 = "ts_delay(open, -8) / ts_delay(open, -1) - 1"
        self.add_feature("label_o2o_h1", o2o_h1)
        self.add_feature("label_o2o_h3", o2o_h3)
        self.add_feature("label_o2o_h5", o2o_h5)
        self.add_feature("label_o2o_h7", o2o_h7)

        # =========================================================
        # close to close 收益率计算
        # =========================================================
        self.add_feature("label_c2c_h3", "ts_delay(close, -3) / ts_delay(close, -1) - 1")
        
        # =========================================================
        # 对数收益（与上述窗口保持一致）
        # =========================================================
        self.add_feature("label_overnight_logret", "ts_log(ts_delay(open, -1)) - ts_log(close)")
        self.add_feature("label_o2o_logret_h1", "ts_log(ts_delay(open, -2)) - ts_log(ts_delay(open, -1))")
        self.add_feature("label_o2o_logret_h3", "ts_log(ts_delay(open, -4)) - ts_log(ts_delay(open, -1))")
        self.add_feature("label_o2o_logret_h5", "ts_log(ts_delay(open, -6)) - ts_log(ts_delay(open, -1))")

        # =========================================================
        # 波动率评估 (历史风险指标)
        # 逻辑：使用过去 N 天的真实收益率来评估股票自身的固有波动风险
        # 注意：ts_delay(..., 1) 代表获取历史 T-1 的数据
        # =========================================================
        hist_daily_ret = "(close / ts_delay(close, 1) - 1)"
        hist_vol_20d = f"ts_std({hist_daily_ret}, 20)"
        hist_vol_60d = f"ts_std({hist_daily_ret}, 60)"
        
        # 计算过去 20 天和 60 天的日度收益率标准差（历史波动率）
        self.add_feature("hist_vol_20d", hist_vol_20d)
        self.add_feature("hist_vol_60d", hist_vol_60d)

        # =========================================================
        # 兼容旧版未来波动率 / Sharpe 标签
        # =========================================================
        future_o2o_ret = "(ts_delay(open, -1) / open - 1)"
        self.add_feature("label_vol_h1", f"ts_std({future_o2o_ret}, 1)")
        self.add_feature("label_vol_h3", f"ts_std({future_o2o_ret}, 3)")
        self.add_feature("label_vol_h5", f"ts_std({future_o2o_ret}, 5)")
        self.add_feature("label_sharpe_h1", f"ts_mean({future_o2o_ret}, 1) / (ts_std({future_o2o_ret}, 1) + 1e-12)")
        self.add_feature("label_sharpe_h3", f"ts_mean({future_o2o_ret}, 3) / (ts_std({future_o2o_ret}, 3) + 1e-12)")
        self.add_feature("label_sharpe_h5", f"ts_mean({future_o2o_ret}, 5) / (ts_std({future_o2o_ret}, 5) + 1e-12)")

        # =========================================================
        # Volatility Scaled Labels (波动率缩放标签)
        # 逻辑：未来收益率 / 历史波动率。逼迫模型寻找高收益且历史走势平稳的标的。
        # 对历史波动率做 1e-4 hard floor，避免长期停牌或死水股导致标签极端放大。
        # =========================================================
        clipped_hist_vol_20d = f"ts_clip({hist_vol_20d}, 1e-4)"
        self.add_feature("label_o2o_h1_vol_scaled", f"({o2o_h1}) / ({clipped_hist_vol_20d})")
        self.add_feature("label_o2o_h3_vol_scaled", f"({o2o_h3}) / ({clipped_hist_vol_20d})")
        self.add_feature("label_o2o_h5_vol_scaled", f"({o2o_h5}) / ({clipped_hist_vol_20d})")

        # =========================================================
        # Asymmetric Scaled Labels (非对称风险缩放标签)
        # 逻辑：用下行波动率或历史回撤惩罚未来收益，减少普通 vol_scaled
        # 对上涨波动的惩罚。所有分母都做 hard floor，避免停牌/一字板样本
        # 把 label 放大到极端值。
        # =========================================================
        downside_ret = f"(({hist_daily_ret} - ts_abs({hist_daily_ret})) / 2)"
        downside_vol_20d = f"ts_std({downside_ret}, 20)"
        downside_vol_60d = f"ts_std({downside_ret}, 60)"
        self.add_feature("hist_downside_vol_20d", downside_vol_20d)
        self.add_feature("hist_downside_vol_60d", downside_vol_60d)

        clipped_downside_vol_20d = f"ts_clip({downside_vol_20d}, 1e-4)"
        self.add_feature("label_o2o_h1_sortino", f"({o2o_h1}) / ({clipped_downside_vol_20d})")
        self.add_feature("label_o2o_h3_sortino", f"({o2o_h3}) / ({clipped_downside_vol_20d})")
        self.add_feature("label_o2o_h5_sortino", f"({o2o_h5}) / ({clipped_downside_vol_20d})")

        drawdown_20d = "(ts_max(high, 20) - close) / ts_clip(ts_max(high, 20), 1e-4)"
        avg_drawdown_60d = f"ts_mean({drawdown_20d}, 60)"
        max_drawdown_60d = f"ts_max({drawdown_20d}, 60)"
        self.add_feature("hist_drawdown_20d", drawdown_20d)
        self.add_feature("hist_avg_drawdown_60d", avg_drawdown_60d)
        self.add_feature("hist_max_drawdown_60d", max_drawdown_60d)

        clipped_avg_drawdown_60d = f"ts_clip({avg_drawdown_60d}, 1e-3)"
        clipped_max_drawdown_60d = f"ts_clip({max_drawdown_60d}, 1e-3)"
        self.add_feature("label_o2o_h1_dd_scaled", f"({o2o_h1}) / ({clipped_avg_drawdown_60d})")
        self.add_feature("label_o2o_h3_dd_scaled", f"({o2o_h3}) / ({clipped_avg_drawdown_60d})")
        self.add_feature("label_o2o_h5_dd_scaled", f"({o2o_h5}) / ({clipped_avg_drawdown_60d})")
        self.add_feature("label_o2o_h1_mdd_scaled", f"({o2o_h1}) / ({clipped_max_drawdown_60d})")
        self.add_feature("label_o2o_h3_mdd_scaled", f"({o2o_h3}) / ({clipped_max_drawdown_60d})")
        self.add_feature("label_o2o_h5_mdd_scaled", f"({o2o_h5}) / ({clipped_max_drawdown_60d})")

    def neutralize_labels(
        self,
        industry_df: pl.DataFrame | None = None,
        cap_df: pl.DataFrame | None = None,
        vol_df: pl.DataFrame | None = None,
        *,
        mode: str | None = None,
        industry_col: str = "industry",
        cap_col: str = "cap",
        vol_col: str = "hist_vol_60d",
        weight_col: str | None = None,
        industry_method: str = "demean",
        cap_log: bool = True,
        fill_missing: str = "UNKNOWN",
        suffix: str = "_neu",
    ) -> None:
        self.result_df = neutralize_label_columns(
            result_df=self.result_df,
            industry_df=industry_df,
            cap_df=cap_df,
            vol_df=vol_df,
            mode=mode,
            industry_col=industry_col,
            cap_col=cap_col,
            vol_col=vol_col,
            weight_col=weight_col,
            industry_method=industry_method,
            cap_log=cap_log,
            fill_missing=fill_missing,
            suffix=suffix,
        )

    def add_benchmark_excess_labels(
        self,
        bench_df: pl.DataFrame,
        *,
        suffix: str = "_excess",
        label_cols: list[str] | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.result_df = label_postprocess.add_benchmark_excess_labels(
            result_df=self.result_df,
            bench_df=bench_df,
            label_strategy_cls=self.__class__,
            train_period=self.data_periods[Segment.TRAIN],
            valid_period=self.data_periods[Segment.VALID],
            test_period=self.data_periods[Segment.TEST],
            interval=self.interval,
            enable_cache=getattr(self, "enable_cache", False),
            suffix=suffix,
            label_cols=label_cols,
            max_workers=max_workers,
        )
