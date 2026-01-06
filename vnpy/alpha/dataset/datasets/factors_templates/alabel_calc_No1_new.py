from vnpy.alpha.dataset.feature_spec import BaseFeatureSpec


class LabelStrategy(BaseFeatureSpec):
    """
    Label 专用特征集合
    """

    def register(self) -> None:
        self.require_bars("open", "close")

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
        # 辅助 Label (3天收益)：T+1买，T+4卖 (偏移 -3)
        self.add_feature("label_o2o_h3", f"ts_delay(open, -4) / ts_delay(open, -1) - 1")
        # 辅助 Label (5天收益)：T+1买，T+6卖 (偏移 -5)
        self.add_feature("label_o2o_h5", f"ts_delay(open, -6) / ts_delay(open, -1) - 1")
        # 辅助 Label (7天收益)：T+1买，T+8卖 (偏移 -7)
        self.add_feature("label_o2o_h7", f"ts_delay(open, -8) / ts_delay(open, -1) - 1")
        # 辅助 Label (10天收益)：T+1买，T+11卖 (偏移 -10)
        self.add_feature("label_o2o_h10", f"ts_delay(open, -11) / ts_delay(open, -1) - 1")

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
