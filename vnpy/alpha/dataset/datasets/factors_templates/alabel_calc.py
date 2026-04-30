import polars as pl

from vnpy.alpha import Segment
from vnpy.alpha.dataset import label_postprocess

from .baseAlphaStrategy import BaseAlphaStrategy

class LabelStrategy(BaseAlphaStrategy):
    """
    Label 专用策略
    逻辑：专门用来计算成交逻辑
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
        self.add_feature("label_overnight_ret", f"ts_delay(open, -1) / close - 1")

        # =========================================================
        # open to open 收益率计算
        # 设置 Label：T+1 开盘买入，持有 H 天
        # =========================================================
        # 辅助 Label (1天收益)：T+1买，T+2卖 (偏移 -1)
        self.add_feature("label_o2o_h1", f"ts_delay(open, -2) / ts_delay(open, -1) - 1")
        # 辅助 Label (3天收益)：T+1买，T+4卖 (偏移 -4)
        self.add_feature("label_o2o_h3", f"ts_delay(open, -4) / ts_delay(open, -1) - 1")
        # 辅助 Label (5天收益)：T+1买，T+6卖 (偏移 -6)
        self.add_feature("label_o2o_h5", f"ts_delay(open, -6) / ts_delay(open, -1) - 1")
        # 辅助 Label (7天收益)：T+1买，T+8卖 (偏移 -8)
        self.add_feature("label_o2o_h7", f"ts_delay(open, -8) / ts_delay(open, -1) - 1")

        # =========================================================
        # close to close 收益率计算
        # =========================================================
        # 辅助 Label (1天收益)：T+1买，T+2卖 (偏移 -1)
        self.add_feature("label_c2c_h3", "ts_delay(close, -3) / ts_delay(close, -1) - 1")
        
        # =========================================================
        # 对数收益（与上述窗口保持一致）
        # =========================================================
        self.add_feature("label_overnight_logret", "ts_log(ts_delay(open, -1)) - ts_log(close)")
        self.add_feature("label_o2o_logret_h1", "ts_log(ts_delay(open, -2)) - ts_log(ts_delay(open, -1))")
        self.add_feature("label_o2o_logret_h3", "ts_log(ts_delay(open, -4)) - ts_log(ts_delay(open, -1))")
        self.add_feature("label_o2o_logret_h5", "ts_log(ts_delay(open, -6)) - ts_log(ts_delay(open, -1))")

        # =========================================================
        # 波动率 / 风险指标（未来 open-to-open 收益序列的滚动指标）
        # =========================================================
        future_o2o_ret = "(ts_delay(open, -1) / open - 1)"
        self.add_feature("label_vol_h1", f"ts_std({future_o2o_ret}, 1)")
        self.add_feature("label_vol_h3", f"ts_std({future_o2o_ret}, 3)")
        self.add_feature("label_vol_h5", f"ts_std({future_o2o_ret}, 5)")

        self.add_feature(
            "label_sharpe_h1",
            f"ts_mean({future_o2o_ret}, 1) / (ts_std({future_o2o_ret}, 1) + 1e-12)"
        )
        self.add_feature(
            "label_sharpe_h3",
            f"ts_mean({future_o2o_ret}, 3) / (ts_std({future_o2o_ret}, 3) + 1e-12)"
        )
        self.add_feature(
            "label_sharpe_h5",
            f"ts_mean({future_o2o_ret}, 5) / (ts_std({future_o2o_ret}, 5) + 1e-12)"
        )

    def neutralize_labels(
        self,
        industry_df: pl.DataFrame | None = None,
        cap_df: pl.DataFrame | None = None,
        *,
        mode: str | None = None,
        industry_col: str = "industry",
        cap_col: str = "cap",
        industry_method: str = "demean",
        cap_log: bool = True,
        fill_missing: str = "UNKNOWN",
        suffix: str = "_neu",
    ) -> None:
        self.result_df = label_postprocess.neutralize_labels(
            result_df=self.result_df,
            industry_df=industry_df,
            cap_df=cap_df,
            mode=mode,
            industry_col=industry_col,
            cap_col=cap_col,
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
