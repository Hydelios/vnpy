import polars as pl

from vnpy.alpha.dataset import AlphaDataset


class Alpha101(AlphaDataset):
    """Alpha101 factors implemented in AlphaDataset style (expression-based).

    注：先迁移基础与常用子集，后续可按需扩展其余条目；
    所有表达式均基于 ts_*/cs_*/ta_* 算子，可与 Alpha158 一致使用。
    """

    def __init__(
        self,
        df: pl.DataFrame,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
        interval: str = "1d",
        enable_cache: bool = False,
        cache_dir: str | None = None,
        select: list[str] | None = None,
        skip: list[str] | None = None,
        include_extras: bool = True,
    ) -> None:
        super().__init__(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
            interval=interval,
            enable_cache=enable_cache,
            cache_dir=cache_dir,
        )

        # Helpers
        ret = "close / ts_delay(close, 1) - 1"
        adv10 = "ts_mean(volume, 10)"
        adv20 = "ts_mean(volume, 20)"
        adv30 = "ts_mean(volume, 30)"
        adv40 = "ts_mean(volume, 40)"
        adv50 = "ts_mean(volume, 50)"
        adv60 = "ts_mean(volume, 60)"
        adv180 = "ts_mean(volume, 180)"

        # ---- Core subset（附用途注释）----
        # 001: (rank(ts_argmax((ret<0? stddev(ret,20):close)^2,5)) - 0.5)
        # 用途：同 Alpha191_010，捕捉“极端强弱”的位置（近5天最大项的截面分位），减0.5居中
        self.add_feature(
            "alpha101_001",
            "cs_rank(ts_argmax((ts_where((%s) < 0, ts_std(%s, 20), close) * ts_where((%s) < 0, ts_std(%s, 20), close)), 5)) - 0.5"
            % (ret, ret, ret, ret),
        )

        # 002: -corr(rank(delta(log(volume),2)), rank((close-open)/open), 6)
        # 用途：量变（对数量变化）与日内涨跌幅的相关性，取负寻找“价升量缩/价跌量缩”
        self.add_feature(
            "alpha101_002",
            "-1 * ts_corr(cs_rank(ts_log(volume) - ts_delay(ts_log(volume), 2)), cs_rank((close - open) / open), 6)",
        )

        # 003: -corr(rank(open), rank(volume), 10)
        # 用途：开盘价与成交量的相关性（截面秩后），取负择强
        self.add_feature("alpha101_003", "-1 * ts_corr(cs_rank(open), cs_rank(volume), 10)")

        # 004: -ts_rank(rank(low), 9)
        # 用途：低价的时序分位，越靠近近期低点越小，取负选择“低位”
        self.add_feature("alpha101_004", "-1 * ts_rank(cs_rank(low), 9)")

        # 005: rank(open - mean(vwap,10)) * (-abs(rank(close - vwap)))
        # 用途：开盘相对10日VWAP 的偏离越大越强，同时收盘离VWAP越近越稳健
        self.add_feature(
            "alpha101_005",
            "cs_rank(open - ts_sum(vwap, 10) / 10) * (-1 * ts_abs(cs_rank(close - vwap)))",
        )

        # 006: -corr(open, volume, 10)
        # 用途：开盘价与量的相关性，取负作为反转维度
        self.add_feature("alpha101_006", "-1 * ts_corr(open, volume, 10)")

        # 007: (adv20<volume) ? (-ts_rank(abs(delta(close,7)),60)*sign(delta(close,7))) : -1
        # 用途：量能超过20日均量时，放大7日动量方向；否则恒为-1
        self.add_feature(
            "alpha101_007",
            f"ts_where({adv20} < volume, (-1 * ts_rank(ts_abs(close - ts_delay(close, 7)), 60)) * ts_sign(close - ts_delay(close, 7)), -1)",
        )

        # 008: -rank((sum(open,5)*sum(ret,5)) - delay(sum(open,5)*sum(ret,5),10))
        # 用途：价量动量项与10期前差值，取负择强
        self.add_feature(
            "alpha101_008",
            "-1 * cs_rank((ts_sum(open, 5) * ts_sum(%s, 5)) - ts_delay(ts_sum(open, 5) * ts_sum(%s, 5), 10))" % (ret, ret),
        )

        # 010: -rank( sign(close-close1) + sign(close1-close2) + sign(close2-close3) )
        # 用途：三日方向一致性（连续上涨/下跌越强，值越极端）
        self.add_feature(
            "alpha101_010",
            "-1 * cs_rank(ts_sign(close - ts_delay(close,1)) + ts_sign(ts_delay(close,1) - ts_delay(close,2)) + ts_sign(ts_delay(close,2) - ts_delay(close,3)))",
        )

        # 072: rank(corr((high+low)/2, adv40, 9)) / rank(corr(ts_rank(vwap,4), ts_rank(volume,19),7))
        # 用途：均价与均量的相关强弱比，分子价格-量，分母成交价-量的时序秩相关
        self.add_feature(
            "alpha101_072",
            f"cs_rank(ts_corr((high + low) / 2, {adv40}, 9)) / cs_rank(ts_corr(ts_rank(vwap, 4), ts_rank(volume, 19), 7))",
        )
        # 071: max(ts_rank(decay_linear(corr(ts_rank(close,3), ts_rank(adv180,12), 18), 4), 16),
        #           ts_rank(decay_linear((cs_rank(((low+open)-(vwap+vwap))) ** 2), 16), 4))
        # 用途：价格-长期均量的相关强度与形态偏差强度的择强
        self.add_feature(
            "alpha101_071",
            f"ts_greater(ts_rank(ts_decay_linear(ts_corr(ts_rank(close, 3), ts_rank({adv180}, 12), 18), 4), 16), ts_rank(ts_decay_linear((cs_rank(((low + open) - (vwap + vwap))) ** 2), 16), 4))",
        )

        # 075: rank(corr(vwap, volume, 4)) < rank(corr(rank(low), rank(adv50), 12))
        # 用途：价量相关性（VWAP-量）弱而低价-量相关性强时为真（布尔型因子）
        self.add_feature(
            "alpha101_075",
            f"(cs_rank(ts_corr(vwap, volume, 4)) < cs_rank(ts_corr(cs_rank(low), cs_rank({adv50}), 12)))",
        )
        # 074: rank(corr((high+low)/2, adv40, 9)) < rank(corr(cs_rank(high*w + vwap*(1-w)), cs_rank(volume), 11))
        # 用途：形态-量相关 vs 价量秩相关的强弱比较（布尔）
        self.add_feature(
            "alpha101_074",
            f"(cs_rank(ts_corr((high + low) / 2, {adv40}, 9)) < cs_rank(ts_corr(cs_rank(high * 0.0261661 + vwap * (1 - 0.0261661)), cs_rank(volume), 11)))",
        )

        # 073: max(rank(decay_linear(delta(vwap, ~5), 3)), ts_rank(decay_linear(((delta((open*w1 + low*(1-w1)),2)/(open*w1 + low*(1-w1))) * -1), 3), 17))
        # 用途：VWAP 动量与开低价组合的相对变化，择强
        mix = "open * 0.147155 + low * (1 - 0.147155)"
        self.add_feature(
            "alpha101_073",
            f"ts_greater(cs_rank(ts_decay_linear(vwap - ts_delay(vwap, 5), 3)), ts_rank(ts_decay_linear(((%s - ts_delay(%s, 2)) / (%s + 1e-12) * -1), 3), 17))" % (mix, mix, mix),
        )

        # 077: min(rank(decay_linear((((high+low)/2 + high) - (vwap + high)), 20)), rank(decay_linear(corr(((high+low)/2), adv40, 3), 6)))
        # 用途：价差结构与价量相关的保守强度（取较小）
        self.add_feature(
            "alpha101_077",
            f"ts_less(cs_rank(ts_decay_linear((((high + low) / 2 + high) - (vwap + high)), 20)), cs_rank(ts_decay_linear(ts_corr(((high + low) / 2), {adv40}, 3), 6)))",
        )

        # 078: rank(corr(sum(low*w + vwap*(1-w), 20), sum(adv40,20), 7)) ^ rank(corr(rank(vwap), rank(volume), 6))
        # 用途：低价-成交价的累积量相关 × VWAP-量 的秩相关强度（乘性组合）
        self.add_feature(
            "alpha101_078",
            f"(cs_rank(ts_corr(ts_sum(low * 0.352233 + vwap * (1 - 0.352233), 20), ts_sum({adv40}, 20), 7)) ** cs_rank(ts_corr(cs_rank(vwap), cs_rank(volume), 6)))",
        )

        # 018: -rank( corr(close, sum(adv20,20), 8) )
        # 用途：价格与累计均量的相关性，取负作为反转维度
        self.add_feature(
            "alpha101_018",
            f"-1 * cs_rank(ts_corr(close, ts_sum({adv20}, 20), 8))",
        )

        # 020: -rank( open - delay(high,1) )
        # 用途：开盘相对昨日高点的缺口越大（向上），值越小（反转）
        self.add_feature(
            "alpha101_020",
            "-1 * cs_rank(open - ts_delay(high, 1))",
        )

        # 021: (((mean(close,8) + stddev(close,8)) < mean(close,2)) ? -1 : (((mean(close,2) < (mean(close,8) - stddev(close,8))) ? 1 : ((volume/adv20) >= 1 ? 1 : -1))))
        # 用途：基于均值与波动的双阈值判断，叠加量能条件
        avg8 = "ts_sum(close, 8) / 8"
        avg2 = "ts_sum(close, 2) / 2"
        sd8 = "ts_std(close, 8)"
        self.add_feature(
            "alpha101_021",
            f"ts_where(({avg8} + {sd8}) < {avg2}, -1, ts_where({avg2} < ({avg8} - {sd8}), 1, ts_where(volume / ({adv20}) >= 1, 1, -1)))",
        )

        # ---- 继续补齐（11-17、19、22、23、25、26、28、30、34-45）----
        # 011: (rank(ts_max(vwap - close,3)) + rank(ts_min(vwap - close,3))) * rank(delta(volume,3))
        self.add_feature(
            "alpha101_011",
            "(cs_rank(ts_max(vwap - close, 3)) + cs_rank(ts_min(vwap - close, 3))) * cs_rank(volume - ts_delay(volume, 3))",
        )

        # 012: sign(delta(volume,1)) * (-1 * delta(close,1))
        self.add_feature(
            "alpha101_012",
            "ts_sign(volume - ts_delay(volume, 1)) * (-1 * (close - ts_delay(close, 1)))",
        )

        # 013: -rank(covariance(rank(close), rank(volume),5))
        self.add_feature(
            "alpha101_013",
            "-1 * cs_rank(ts_cov(cs_rank(close), cs_rank(volume), 5))",
        )

        # 014: (-1 * rank(delta(returns,3))) * corr(open, volume,10)
        self.add_feature(
            "alpha101_014",
            f"-1 * cs_rank(({ret}) - ts_delay(({ret}), 3)) * ts_corr(open, volume, 10)",
        )

        # 015: -sum(rank(corr(rank(high), rank(volume),3)), 3)
        self.add_feature(
            "alpha101_015",
            "-1 * ts_sum(cs_rank(ts_corr(cs_rank(high), cs_rank(volume), 3)), 3)",
        )

        # 016: -rank(covariance(rank(high), rank(volume),5))
        self.add_feature(
            "alpha101_016",
            "-1 * cs_rank(ts_cov(cs_rank(high), cs_rank(volume), 5))",
        )

        # 017: (-1 * rank(ts_rank(close,10))) * rank(delta(delta(close,1),1)) * rank(ts_rank(volume/adv20,5))
        self.add_feature(
            "alpha101_017",
            f"-1 * cs_rank(ts_rank(close, 10)) * cs_rank((close - ts_delay(close,1)) - (ts_delay(close,1) - ts_delay(close,2))) * cs_rank(ts_rank(volume / ({adv20}), 5))",
        )

        # 019: (-1 * sign((close - delay(close,7)) + delta(close,7))) * (1 + rank(1 + sum(returns,250)))
        self.add_feature(
            "alpha101_019",
            f"(-1 * ts_sign(close - ts_delay(close, 7))) * (cs_rank(ts_sum({ret}, 250)) + 1)",
        )

        # 022: - (delta(corr(high, volume,5),5) * rank(stddev(close,20)))
        self.add_feature(
            "alpha101_022",
            "-1 * ( (ts_corr(high, volume, 5) - ts_delay(ts_corr(high, volume, 5), 5)) * cs_rank(ts_std(close, 20)) )",
        )

        # 023: ((mean(high,20) < high) ? (-delta(high,2)) : 0)
        self.add_feature(
            "alpha101_023",
            "ts_where(ts_sum(high, 20)/20 < high, -1 * (high - ts_delay(high, 2)), 0)",
        )

        # 024: (((delta(mean(close,100),100) / delay(close,100)) <= 0.05) ? (-1 * (close - ts_min(close,100))) : (-1 * delta(close,3)))
        # 用途：长期均值相对稳定时使用“回落到通道下沿”，否则使用短期动量（取负）
        self.add_feature(
            "alpha101_024",
            "ts_where(((ts_sum(close, 100) / 100) - ts_delay((ts_sum(close, 100) / 100), 100)) / (ts_delay(close, 100) + 1e-12) <= 0.05, -1 * (close - ts_min(close, 100)), -1 * (close - ts_delay(close, 3)))",
        )

        # 025: rank(((-1 * returns) * adv20) * vwap * (high - close))
        self.add_feature(
            "alpha101_025",
            f"cs_rank(((-1 * ({ret})) * ({adv20})) * vwap * (high - close))",
        )

        # 026: -ts_max(corr(ts_rank(volume,5), ts_rank(high,5),5), 3)
        self.add_feature(
            "alpha101_026",
            "-1 * ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)",
        )

        # 027: (0.5 < rank( (sum(corr(rank(volume), rank(vwap),6), 2) / 2.0) )) ? -1 : 1
        # 用途：量价秩相关的近期强度，超过分位阈值则取 -1
        self.add_feature(
            "alpha101_027",
            "ts_where(cs_rank(ts_sum(ts_corr(cs_rank(volume), cs_rank(vwap), 6), 2) / 2.0) > 0.5, -1, 1)",
        )

        # 028: scale(corr(adv20, low, 5) + ((high + low)/2) - close)
        self.add_feature(
            "alpha101_028",
            f"cs_scale(ts_corr({adv20}, low, 5) + ((high + low) / 2) - close)",
        )

        # 030: ((1 - rank(sign(delta_close + sign(delta_close[-1]) + sign(delta_close[-2])))) * sum(volume,5)) / sum(volume,20)
        # 用途：三日方向一致性与量比的组合
        self.add_feature(
            "alpha101_030",
            "((-1 * cs_rank(ts_sign(close - ts_delay(close,1)) + ts_sign(ts_delay(close,1) - ts_delay(close,2)) + ts_sign(ts_delay(close,2) - ts_delay(close,3)))) + 1) * ts_sum(volume,5) / (ts_sum(volume,20) + 1e-12)",
        )

        # 034: rank(2 - rank(stddev(returns,2)/stddev(returns,5)) - rank(delta(close,1)))
        self.add_feature(
            "alpha101_034",
            f"cs_rank(((-1 * cs_rank(ts_std({ret}, 2) / (ts_std({ret}, 5) + 1e-12))) - cs_rank(close - ts_delay(close, 1)) + 2))",
        )

        # 035: (ts_rank(volume,32) * (1 - ts_rank((close + high - low), 16))) * (1 - ts_rank(returns,32))
        self.add_feature(
            "alpha101_035",
            f"ts_rank(volume, 32) * ((-1 * ts_rank((close + high - low), 16)) + 1) * ((-1 * ts_rank({ret}, 32)) + 1)",
        )

        # 036: 组合多项（开收价差-量相关、价量秩时序、长均价与开收差、VWAP-adv20 相关绝对值）
        self.add_feature(
            "alpha101_036",
            f"(2.21 * cs_rank(ts_corr(close - open, ts_delay(volume, 1), 15)) + 0.7 * cs_rank(open - close) + 0.73 * cs_rank(ts_rank(ts_delay(-1 * {ret}, 6), 5)) + cs_rank(ts_abs(ts_corr(vwap, {adv20}, 6))) + 0.6 * cs_rank(((ts_mean(close, 200) - open) * (close - open))))",
        )

        # 037: rank(corr(delay(open - close,1), close,200)) + rank(open - close)
        self.add_feature(
            "alpha101_037",
            "cs_rank(ts_corr(ts_delay(open - close, 1), close, 200)) + cs_rank(open - close)",
        )

        # 038: (-1 * rank(ts_rank(open,10))) * rank(close/open)
        self.add_feature(
            "alpha101_038",
            "-1 * cs_rank(ts_rank(open, 10)) * cs_rank(close / (open + 1e-12))",
        )

        # 039: (-1 * rank(delta(close,7) * (1 - rank(decay_linear(volume/adv20,9))))) * (1 + rank(sum(returns,250)))
        self.add_feature(
            "alpha101_039",
            f"(-1 * cs_rank((close - ts_delay(close, 7)) * ((-1 * cs_rank(ts_decay_linear(volume / ({adv20}), 9))) + 1))) * (cs_rank(ts_sum({ret}, 250)) + 1)",
        )

        # 040: (-1 * rank(stddev(high,10))) * corr(high, volume,10)
        self.add_feature(
            "alpha101_040",
            "(-1 * cs_rank(ts_std(high, 10))) * ts_corr(high, volume, 10)",
        )

        # 041: sqrt(high*low) - vwap
        self.add_feature("alpha101_041", "(high * low) ** 0.5 - vwap")

        # 042: rank(vwap - close) / rank(vwap + close)
        self.add_feature("alpha101_042", "cs_rank(vwap - close) / (cs_rank(vwap + close) + 1e-12)")

        # 043: ts_rank(volume/adv20,20) * ts_rank(-delta(close,7), 8)
        self.add_feature(
            "alpha101_043",
            f"ts_rank(volume / ({adv20}), 20) * ts_rank(-1 * (close - ts_delay(close, 7)), 8)",
        )

        # 044: -corr(high, rank(volume), 5)
        self.add_feature(
            "alpha101_044",
            "-1 * ts_corr(high, cs_rank(volume), 5)",
        )

        # 045: - (rank(mean(delay(close,5),20)) * corr(close, volume,2) * rank(corr(sum(close,5), sum(close,20),2)))
        self.add_feature(
            "alpha101_045",
            "-1 * ( cs_rank(ts_sum(ts_delay(close, 5), 20) / 20) * ts_corr(close, volume, 2) * cs_rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2)) )",
        )

        # 046: piecewise on x=((delay20-close10)/10 - (delay10-close)/10)
        # 用途：基于三段阈值的动量/均值回归判断（输出 +1 / -1）
        x = "(ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10"
        self.add_feature(
            "alpha101_046",
            f"ts_where(({x}) > 0.25, -1, ts_where(({x}) < 0, 1, ts_where(volume / ({adv20}) >= 1, 1, -1)))",
        )

        # 047: ((((rank(1/close) * volume)/adv20) * ((high * rank(high - close)) / (sum(high,5)/5))) - rank(vwap - delay(vwap,5)))
        # 用途：价量综合强度（收盘偏低、量大、形态偏强）减去 VWAP 近期动量
        self.add_feature(
            "alpha101_047",
            f"(((cs_rank(1 / (close + 1e-12)) * volume) / ({adv20})) * (high * cs_rank(high - close) / (ts_sum(high, 5) / 5))) - cs_rank(vwap - ts_delay(vwap, 5))",
        )

        # 049: x < -0.1 ? 1 : -delta(close,1)
        # 用途：阈值型的动量/均值回归判断
        self.add_feature(
            "alpha101_049",
            f"ts_where(({x}) < -0.1, 1, -1 * (close - ts_delay(close, 1)))",
        )

        # 050: -ts_max(rank(corr(rank(volume), rank(vwap),5)), 5)
        # 用途：量-成交价秩相关的时序极值（取负择强）
        self.add_feature(
            "alpha101_050",
            "-1 * ts_max(cs_rank(ts_corr(cs_rank(volume), cs_rank(vwap), 5)), 5)",
        )

        # 051: x < -0.05 ? 1 : -delta(close,1)
        # 用途：阈值更宽松的动量/均值回归判断
        self.add_feature(
            "alpha101_051",
            f"ts_where(({x}) < -0.05, 1, -1 * (close - ts_delay(close, 1)))",
        )

        # 052: ((-delta(ts_min(low,5),5)) * rank((sum(ret,240)-sum(ret,20))/220)) * ts_rank(volume,5)
        # 用途：低点抬升 × 长短期累计收益差 × 量能时序分位
        self.add_feature(
            "alpha101_052",
            f"(-1 * ((ts_min(low, 5) - ts_delay(ts_min(low, 5), 5))) ) * cs_rank((ts_sum({ret}, 240) - ts_sum({ret}, 20)) / 220) * ts_rank(volume, 5)",
        )

        # 053: -delta( (((close-low)-(high-close)) / (close-low)), 9 )
        # 用途：实体在日内区间中的相对位置的时序变化（取负）
        self.add_feature(
            "alpha101_053",
            "-1 * ( ((close - low) - (high - close)) / (close - low + 1e-12) - ts_delay(((close - low) - (high - close)) / (close - low + 1e-12), 9) )",
        )

        # 054: - ((low - close) * open^5) / ((low - high) * close^5)
        # 用途：形态极端项的比例关系（取负）
        self.add_feature(
            "alpha101_054",
            "-1 * ((low - close) * (open ** 5)) / (((low - high) + 1e-12) * (close ** 5) + 1e-12)",
        )

        # 055: -corr(rank((close - ts_min(low,12)) / (ts_max(high,12) - ts_min(low,12))), rank(volume), 6)
        # 用途：通道位置与量秩的负相关
        self.add_feature(
            "alpha101_055",
            "-1 * ts_corr(cs_rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12) + 1e-12)), cs_rank(volume), 6)",
        )
        # 057: - ( (close - vwap) / decay_linear(rank(ts_argmax(close,30)), 2) )
        # 用途：实体偏离 VWAP 的强度，用“近30日极值位置”的线性权直接平衡
        self.add_feature(
            "alpha101_057",
            "-1 * ((close - vwap) / (ts_decay_linear(cs_rank(ts_argmax(close, 30)), 2) + 1e-12))",
        )

        # 076: max(rank(decay_linear(delta(vwap,1), ~12)), ts_rank(decay_linear(ts_argmax(corr(ts_rank(close,7), ts_rank(adv60,4), 4), 13), 14), 13)) * -1
        # 用途：VWAP 动量与“价-量秩相关的峰值位置”的保守择强（取负）
        self.add_feature(
            "alpha101_076",
            f"ts_greater(cs_rank(ts_decay_linear(vwap - ts_delay(vwap, 1), 12)), ts_rank(ts_decay_linear(ts_argmax(ts_corr(ts_rank(close, 7), ts_rank({adv60}, 4), 4), 13), 14), 13)) * -1",
        )

        # 常用动量/波动类补充（示例）
        for w in (5, 10, 20):
            self.add_feature(f"mom_{w}", f"close / ts_delay(close, {w}) - 1")
            self.add_feature(f"vol_{w}", f"ts_std({ret}, {w})")

        # Label（与 Alpha158 一致）
        self.set_label("ts_delay(close, -3) / ts_delay(close, -1) - 1")

        # ---- 更多扩展（附用途注释）----
        # 081: (rank(log(product(rank((rank(corr(vwap, sum(adv10,50),8))^4)),15))) < rank(corr(rank(vwap), rank(volume), 5))) * -1
        # 用途：成交价-累计均量相关的“持续强度”（滚动连乘） vs VWAP-量的秩相关；为真则-1
        left_081 = f"cs_rank(ts_log(ts_prod(cs_rank((cs_rank(ts_corr(vwap, ts_sum({adv10}, 50), 8)) ** 4)), 15)))"
        right_081 = "cs_rank(ts_corr(cs_rank(vwap), cs_rank(volume), 5))"
        self.add_feature(
            "alpha101_081",
            f"(({left_081}) < ({right_081})) * -1",
        )

        # 083: (rank(delay(((high-low)/(sum(close,5)/5)),2)) * rank(rank(volume))) / (((high-low)/(sum(close,5)/5)) / (vwap - close))
        # 用途：波动率（相对均价）与量强度的配比因子
        self.add_feature(
            "alpha101_083",
            "cs_rank(ts_delay(((high - low) / (ts_sum(close, 5) / 5)), 2)) * cs_rank(cs_rank(volume)) / ((((high - low) / (ts_sum(close, 5) / 5))) / (vwap - close + 1e-12))",
        )

        # 086: (ts_rank(corr(close, sum(adv20,15),6), 20) < rank((open+close) - (vwap + open))) * -1
        # 用途：价格-累计均量的时序相关弱，且当日收盘高于 VWAP 越多越强，为真则-1
        self.add_feature(
            "alpha101_086",
            f"(ts_rank(ts_corr(close, ts_sum({adv20}, 15), 6), 20) < cs_rank((open + close) - (vwap + open))) * -1",
        )

        # 088: min(rank(decay_linear(((rank(open)+rank(low))-(rank(high)+rank(close))),8)), ts_rank(decay_linear(corr(ts_rank(close,8), ts_rank(adv60,21),8),7),3))
        # 用途：价差结构与价量相关的保守强度
        self.add_feature(
            "alpha101_088",
            f"ts_less(cs_rank(ts_decay_linear((cs_rank(open) + cs_rank(low) - cs_rank(high) - cs_rank(close)), 8)), ts_rank(ts_decay_linear(ts_corr(ts_rank(close, 8), ts_rank({adv60}, 21), 8), 7), 3))",
        )

        # 092: min(ts_rank(decay_linear(((((high+low)/2)+close) < (low+open)), 15),19), ts_rank(decay_linear(corr(rank(low), rank(adv30), 8),7),7))
        # 用途：形态条件与低价-量相关的保守择强
        self.add_feature(
            "alpha101_092",
            f"ts_less(ts_rank(ts_decay_linear((((high + low) / 2 + close) < (low + open)), 15), 19), ts_rank(ts_decay_linear(ts_corr(cs_rank(low), cs_rank({adv30}), 8), 7), 7))",
        )

        # 094: (rank(vwap - ts_min(vwap,12)) ^ ts_rank(corr(ts_rank(vwap,20), ts_rank(adv60,4), 18),3)) * -1
        # 用途：VWAP 上行强度 × 价量相关时序强度；取负
        self.add_feature(
            "alpha101_094",
            f"(cs_rank(vwap - ts_min(vwap, 12)) ** ts_rank(ts_corr(ts_rank(vwap, 20), ts_rank({adv60}, 4), 18), 3)) * -1",
        )

        # 095: rank(open - ts_min(open,12)) < ts_rank((rank(corr(sum((high+low)/2,19), sum(adv40,19), 13))^5), 12)
        # 用途：开盘位置偏离 vs 价量相关高次强度
        self.add_feature(
            "alpha101_095",
            f"(cs_rank(open - ts_min(open, 12)) < ts_rank((cs_rank(ts_corr(ts_sum((high + low) / 2, 19), ts_sum({adv40}, 19), 13)) ** 5), 12))",
        )

        # 096: max(ts_rank(decay_linear(corr(rank(vwap), rank(volume),3),4),8), ts_rank(decay_linear(ts_argmax(corr(ts_rank(close,7), ts_rank(adv60,4), 4), 13),14),13)) * -1
        # 用途：两路价量时序强度的择强后取反
        self.add_feature(
            "alpha101_096",
            f"(ts_greater(ts_rank(ts_decay_linear(ts_corr(cs_rank(vwap), cs_rank(volume), 3), 4), 8), ts_rank(ts_decay_linear(ts_argmax(ts_corr(ts_rank(close, 7), ts_rank({adv60}, 4), 4), 13), 14), 13))) * -1",
        )

        # 098: rank(decay_linear(corr(vwap, sum(adv5,26),5),7)) - rank(decay_linear(ts_rank(ts_argmin(corr(rank(open), rank(adv15),21),9),7),8))
        # 用途：成交价-短均量相关与开盘-量相关低谷位置的对比
        self.add_feature(
            "alpha101_098",
            "cs_rank(ts_decay_linear(ts_corr(vwap, ts_sum(ts_mean(volume, 5), 26), 5), 7)) - cs_rank(ts_decay_linear(ts_rank(ts_argmin(ts_corr(cs_rank(open), cs_rank(ts_mean(volume, 15)), 21), 9), 7), 8))",
        )

        # 099: (rank(corr(sum((high+low)/2,20), sum(adv60,20), 9)) < rank(corr(low, volume, 6))) * -1
        # 用途：形态价量相关弱而低价-量相关强时为真，取负
        self.add_feature(
            "alpha101_099",
            f"(cs_rank(ts_corr(ts_sum((high + low) / 2, 20), ts_sum({adv60}, 20), 9)) < cs_rank(ts_corr(low, volume, 6))) * -1",
        )

        # 101: (close - open) / ((high - low) + 0.001)
        # 用途：实体占比（相对当日振幅），形态强弱
        self.add_feature(
            "alpha101_101",
            "(close - open) / ((high - low) + 0.001)",
        )

        # ---- 更多 101 因子（附用途注释）----
        # 060: - (2*scale(rank(((close-low)-(high-close))*volume/(high-low))) - scale(rank(ts_argmax(close,10))))
        # 用途：K线位置（上/下影差）× 量 的强弱对比，并与“近10日极值位置”做尺度对冲
        self.add_feature(
            "alpha101_060",
            "-1 * (2 * cs_scale(cs_rank(((close - low) - (high - close)) * volume / (high - low + 1e-12))) - cs_scale(cs_rank(ts_argmax(close, 10))))",
        )

        # 061: rank(vwap - ts_min(vwap,16)) < rank(corr(vwap, adv180, 18))
        # 用途：成交价上行幅度 vs 价格-长期均量相关的强弱比较（布尔）
        self.add_feature(
            "alpha101_061",
            f"(cs_rank(vwap - ts_min(vwap, 16)) < cs_rank(ts_corr(vwap, {adv180}, 18)))",
        )

        # 062: rank(corr(vwap, sum(adv20,22),10)) < rank(((rank(open)+rank(open)) < (rank((high+low)/2)+rank(high)))) 取负
        # 用途：成交价-累计均量相关弱，且开盘秩之和低于高/均价秩之和（形态偏弱）
        self.add_feature(
            "alpha101_062",
            f"(cs_rank(ts_corr(vwap, ts_sum({adv20}, 22), 10)) < cs_rank((cs_rank(open) + cs_rank(open)) < (cs_rank((high + low) / 2) + cs_rank(high)))) * -1",
        )

        # 064: rank(corr(sum(open*w+low*(1-w),13), sum(adv120,13),17)) < rank(delta(((high+low)/2*w + vwap*(1-w)),4)) 取负
        # 用途：形态-长期均量相关弱，同时形态平滑价正向变动为强
        self.add_feature(
            "alpha101_064",
            f"(cs_rank(ts_corr(ts_sum(open * 0.178404 + low * (1 - 0.178404), 13), ts_sum(ts_mean(volume, 120), 13), 17)) < cs_rank(((high + low) / 2 * 0.178404 + vwap * (1 - 0.178404)) - ts_delay(((high + low) / 2 * 0.178404 + vwap * (1 - 0.178404)), 4))) * -1",
        )

        # 065: rank(corr(open*w + vwap*(1-w), sum(adv60,9),6)) < rank(open - ts_min(open,14)) 取负
        # 用途：开盘/成交价组合与累计均量相关弱，但开盘位置偏上
        self.add_feature(
            "alpha101_065",
            f"(cs_rank(ts_corr(open * 0.00817205 + vwap * (1 - 0.00817205), ts_sum({adv60}, 9), 6)) < cs_rank(open - ts_min(open, 14))) * -1",
        )

        # 066: (rank(decay_linear(delta(vwap,4),7)) + ts_rank(decay_linear(((low - vwap)/(open - (high+low)/2)),11),7)) * -1
        # 用途：VWAP 动量（衰减）与相对位置（低/开-中位）强弱综合（取负）
        self.add_feature(
            "alpha101_066",
            "(cs_rank(ts_decay_linear(vwap - ts_delay(vwap, 4), 7)) + ts_rank(ts_decay_linear(((low - vwap) / (open - ((high + low) / 2) + 1e-12)), 11), 7)) * -1",
        )

        # 068: ts_rank(corr(rank(high), rank(adv15),9),14) < rank(delta(close*w + low*(1-w),1)) 取负
        # 用途：高价-短期均量秩相关弱且价形组合上行
        self.add_feature(
            "alpha101_068",
            f"(ts_rank(ts_corr(cs_rank(high), cs_rank(ts_mean(volume, 15)), 9), 14) < cs_rank((close * 0.518371 + low * (1 - 0.518371)) - ts_delay((close * 0.518371 + low * (1 - 0.518371)), 1))) * -1",
        )

        # 084: (ts_rank(vwap - ts_max(vwap,15), 21) ^ delta(close,5))
        # 用途：成交价通道下沿的时序强度，按价格动量放大（幂次）
        self.add_feature(
            "alpha101_084",
            "(ts_rank(vwap - ts_max(vwap, 15), 21) ** (close - ts_delay(close, 5)))",
        )

        # 085: rank(corr(high*w + close*(1-w), adv30,10)) ^ rank(corr(ts_rank((high+low)/2,4), ts_rank(volume,10),7))
        # 用途：高收价-均量相关强度 与 形态-量的秩相关强度 的乘性组合
        self.add_feature(
            "alpha101_085",
            f"(cs_rank(ts_corr(high * 0.876703 + close * (1 - 0.876703), {adv30}, 10)) ** cs_rank(ts_corr(ts_rank((high + low) / 2, 4), ts_rank(volume, 10), 7)))",
        )

        # 选择/排序：按编号升序，仅针对 alpha101_xxx 数值因子；额外因子可选包含
        self._apply_feature_selection_101(select=select, skip=skip, include_extras=include_extras)

    def _apply_feature_selection_101(self, select: list[str] | None, skip: list[str] | None, include_extras: bool) -> None:
        """Reorder and optionally filter Alpha101 features.

        - 将 alpha101_XXX 因子按编号升序重建。
        - 若提供 select：仅保留 select 中的条目；若提供 skip：剔除 skip 中的条目（两者都给时以 select 为准）。
        - 额外因子（非 alpha101_XXX）通过 include_extras 控制是否保留，默认保留，放在末尾。
        """
        items = list(self.feature_expressions.items())
        numeric: list[tuple[str, object]] = []
        extras: list[tuple[str, object]] = []

        for name, expr in items:
            if name.startswith("alpha101_"):
                numeric.append((name, expr))
            else:
                extras.append((name, expr))

        def num_id(name: str) -> int:
            try:
                return int(name.split("_")[-1])
            except Exception:
                return 999999

        # 过滤
        if select:
            select_set = set(select)
            numeric = [(n, e) for n, e in numeric if n in select_set]
        elif skip:
            skip_set = set(skip)
            numeric = [(n, e) for n, e in numeric if n not in skip_set]

        # 排序
        numeric.sort(key=lambda x: num_id(x[0]))

        # 重建
        ordered = numeric + (extras if include_extras else [])
        self.feature_expressions.clear()
        for name, expr in ordered:
            self.feature_expressions[name] = expr

        # 仅负责排序/筛选，不再在此追加新因子
