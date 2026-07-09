from __future__ import annotations

import polars as pl

from vnpy.alpha.dataset import AlphaDataset


class RQAlpha101Factor(AlphaDataset):
    """RiceQuant WorldQuant Alpha101 formulas.

    This template follows ``rq_alpha101.md`` and registers factors using the
    same names returned by ``rqdatac.get_factor``:
    ``WorldQuant_alpha001`` ... ``WorldQuant_alpha101``.
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
        amount_col: str = "turnover",
        cap_col: str = "market_cap",
        industry_col: str | None = None,
    ) -> None:
        self.select = set(select) if select else None
        self.skip = set(skip) if skip else set()
        prepared = self._prepare_input(df, amount_col, cap_col, industry_col)
        super().__init__(
            df=prepared,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
            interval=interval,
            enable_cache=enable_cache,
            cache_dir=cache_dir,
        )
        try:
            self.set_label("ts_delay(close, -3) / ts_delay(close, -1) - 1")
        except Exception:
            pass

    @staticmethod
    def _prepare_input(df: pl.DataFrame, amount_col: str, cap_col: str, industry_col: str | None) -> pl.DataFrame:
        out = df
        if "vwap" not in out.columns:
            source = amount_col if amount_col in out.columns else "total_turnover" if "total_turnover" in out.columns else None
            if source is not None and "volume" in out.columns:
                out = out.with_columns(
                    pl.when(pl.col("volume") != 0)
                    .then(pl.col(source).cast(pl.Float64) / pl.col("volume").cast(pl.Float64))
                    .otherwise(None)
                    .alias("vwap")
                )

        if "cap" not in out.columns:
            cap_source = None
            for candidate in [cap_col, "market_cap", "a_share_market_val", "capital"]:
                if candidate in out.columns:
                    cap_source = candidate
                    break
            if cap_source is None:
                out = out.with_columns(pl.lit(1.0).alias("cap"))
            else:
                out = out.with_columns(pl.col(cap_source).cast(pl.Float64).alias("cap"))

        if "industry" not in out.columns:
            ind_source = None
            for candidate in [industry_col, "industry_code", "indclass", "industry_name", "sector_code"]:
                if candidate and candidate in out.columns:
                    ind_source = candidate
                    break
            if ind_source is None:
                out = out.with_columns(pl.lit("ALL").alias("industry"))
            else:
                out = out.with_columns(pl.col(ind_source).cast(pl.Utf8).alias("industry"))

        return out

    @staticmethod
    def name(number: int) -> str:
        return f"WorldQuant_alpha{number:03d}"

    def _add(self, number: int, expression: str) -> None:
        name = self.name(number)
        if self.select is not None and name not in self.select and f"alpha{number:03d}" not in self.select:
            return
        if name in self.skip or f"alpha{number:03d}" in self.skip:
            return
        self.add_feature(name, expression)

    def register_all(self, windows: list[int] | None = None) -> None:
        r = "(close / ts_delay(close, 1) - 1)"
        v = "vwap"

        self._add(1, f"rq_rank(rq_ts_argmax(signed_power(ts_where(({r}) < 0, rq_ts_std({r}, 20), close), 2), 5)) - 0.5")
        self._add(2, "-1 * ts_corr(rq_rank(ts_log(volume) - ts_delay(ts_log(volume), 2)), rq_rank((close - open) / open), 6)")
        self._add(3, "-1 * ts_corr(rq_rank(open), rq_rank(volume), 10)")
        self._add(4, "-1 * rq_ts_rank(rq_rank(low), 9)")
        self._add(5, f"rq_rank(open - ts_sum({v}, 10) / 10) * (-1 * abs(rq_rank(close - {v})))")
        self._add(6, "-1 * ts_corr(open, volume, 10)")
        self._add(7, "ts_where(ts_mean(volume, 20) < volume, (-1 * rq_ts_rank(abs(close - ts_delay(close, 7)), 60)) * rq_sign(close - ts_delay(close, 7)), -1)")
        self._add(8, f"-1 * rq_rank((ts_sum(open, 5) * ts_sum({r}, 5)) - ts_delay((ts_sum(open, 5) * ts_sum({r}, 5)), 10))")
        self._add(9, "ts_where(0 < ts_min(close - ts_delay(close, 1), 5), close - ts_delay(close, 1), ts_where(ts_max(close - ts_delay(close, 1), 5) < 0, close - ts_delay(close, 1), -1 * (close - ts_delay(close, 1))))")
        self._add(10, "rq_rank(ts_where(0 < ts_min(close - ts_delay(close, 1), 4), close - ts_delay(close, 1), ts_where(ts_max(close - ts_delay(close, 1), 4) < 0, close - ts_delay(close, 1), -1 * (close - ts_delay(close, 1)))))")
        self._add(11, f"(rq_rank(ts_max({v} - close, 3)) + rq_rank(ts_min({v} - close, 3))) * rq_rank(volume - ts_delay(volume, 3))")
        self._add(12, "rq_sign(volume - ts_delay(volume, 1)) * (-1 * (close - ts_delay(close, 1)))")
        self._add(13, "-1 * rq_rank(ts_cov(rq_rank(close), rq_rank(volume), 5))")
        self._add(14, f"(-1 * rq_rank(({r}) - ts_delay(({r}), 3))) * ts_corr(open, volume, 10)")
        self._add(15, "-1 * ts_sum(rq_rank(ts_corr(rq_rank(high), rq_rank(volume), 3)), 3)")
        self._add(16, "-1 * rq_rank(ts_cov(rq_rank(high), rq_rank(volume), 5))")
        self._add(17, f"((-1 * rq_rank(rq_ts_rank(close, 10))) * rq_rank((close - ts_delay(close, 1)) - ts_delay(close - ts_delay(close, 1), 1))) * rq_rank(rq_ts_rank(volume / ts_mean(volume, 20), 5))")
        self._add(18, "-1 * rq_rank(rq_ts_std(abs(close - open), 5) + (close - open) + ts_corr(close, open, 10))")
        self._add(19, f"(-1 * rq_sign((close - ts_delay(close, 7)) + (close - ts_delay(close, 7)))) * (1 + rq_rank(1 + ts_sum({r}, 250)))")
        self._add(20, "-1 * rq_rank(open - ts_delay(high, 1)) * rq_rank(open - ts_delay(close, 1)) * rq_rank(open - ts_delay(low, 1))")
        self._add(21, "ts_where((ts_sum(close, 8) / 8 + rq_ts_std(close, 8)) < (ts_sum(close, 2) / 2), -1, ts_where((ts_sum(close, 2) / 2) < (ts_sum(close, 8) / 8 - rq_ts_std(close, 8)), 1, ts_where(volume / ts_mean(volume, 20) >= 1, 1, -1)))")
        self._add(22, "-1 * ((ts_corr(high, volume, 5) - ts_delay(ts_corr(high, volume, 5), 5)) * rq_rank(rq_ts_std(close, 20)))")
        self._add(23, "ts_where(ts_sum(high, 20) / 20 < high, -1 * (high - ts_delay(high, 2)), 0)")
        self._add(24, "ts_where(((ts_sum(close, 100) / 100) - ts_delay(ts_sum(close, 100) / 100, 100)) / ts_delay(close, 100) <= 0.05, -1 * (close - ts_min(close, 100)), -1 * (close - ts_delay(close, 3)))")
        self._add(25, f"rq_rank(((-1 * {r}) * ts_mean(volume, 20) * {v}) * (high - close))")
        self._add(26, "-1 * ts_max(ts_corr(rq_ts_rank(volume, 5), rq_ts_rank(high, 5), 5), 3)")
        self._add(27, f"ts_where(0.5 < rq_rank(ts_sum(ts_corr(rq_rank(volume), rq_rank({v}), 6), 2) / 2.0), -1, 1)")
        self._add(28, "cs_scale(ts_corr(ts_mean(volume, 20), low, 5) + ((high + low) / 2) - close)")
        self._add(29, f"rq_min(ts_prod(rq_rank(rq_rank(cs_scale(ts_log(ts_sum(ts_min(rq_rank(rq_rank(-1 * rq_rank((close - 1) - ts_delay(close - 1, 5)))), 2), 1))))), 1), 5) + rq_ts_rank(ts_delay(-1 * ({r}), 6), 5)")
        self._add(30, "(1.0 - rq_rank(rq_sign(close - ts_delay(close, 1)) + rq_sign(ts_delay(close, 1) - ts_delay(close, 2)) + rq_sign(ts_delay(close, 2) - ts_delay(close, 3)))) * ts_sum(volume, 5) / ts_sum(volume, 20)")
        self._add(31, "rq_rank(rq_rank(rq_rank(ts_decay_linear(-1 * rq_rank(rq_rank(close - ts_delay(close, 10))), 10)))) + rq_rank(-1 * (close - ts_delay(close, 3))) + rq_sign(cs_scale(ts_corr(ts_mean(volume, 20), low, 12)))")
        self._add(32, f"cs_scale(ts_sum(close, 7) / 7 - close) + 20 * cs_scale(ts_corr({v}, ts_delay(close, 5), 230))")
        self._add(33, "rq_rank(-1 * ((1 - (open / close)) ** 1))")
        self._add(34, f"rq_rank((1 - rq_rank(rq_ts_std({r}, 2) / rq_ts_std({r}, 5))) + (1 - rq_rank(close - ts_delay(close, 1))))")
        self._add(35, f"(rq_ts_rank(volume, 32) * (1 - rq_ts_rank((close + high) - low, 16))) * (1 - rq_ts_rank({r}, 32))")
        self._add(36, f"(2.21 * rq_rank(ts_corr(close - open, ts_delay(volume, 1), 15))) + (0.7 * rq_rank(open - close)) + (0.73 * rq_rank(rq_ts_rank(ts_delay(-1 * ({r}), 6), 5))) + rq_rank(abs(ts_corr({v}, ts_mean(volume, 20), 6))) + (0.6 * rq_rank((ts_sum(close, 200) / 200 - open) * (close - open)))")
        self._add(37, "rq_rank(ts_corr(ts_delay(open - close, 1), close, 200)) + rq_rank(open - close)")
        self._add(38, "-1 * rq_rank(rq_ts_rank(close, 10)) * rq_rank(close / open)")
        self._add(39, f"(-1 * rq_rank((close - ts_delay(close, 7)) * (1 - rq_rank(ts_decay_linear(volume / ts_mean(volume, 20), 9))))) * (1 + rq_rank(ts_sum({r}, 250)))")
        self._add(40, "-1 * rq_rank(rq_ts_std(high, 10)) * ts_corr(high, volume, 10)")
        self._add(41, f"((high * low) ** 0.5) - {v}")
        self._add(42, f"rq_rank({v} - close) / rq_rank({v} + close)")
        self._add(43, "rq_ts_rank(volume / ts_mean(volume, 20), 20) * rq_ts_rank(-1 * (close - ts_delay(close, 7)), 8)")
        self._add(44, "-1 * ts_corr(high, rq_rank(volume), 5)")
        self._add(45, "-1 * (rq_rank(ts_sum(ts_delay(close, 5), 20) / 20) * ts_corr(close, volume, 2) * rq_rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2)))")
        inner = "(((ts_delay(close, 20) - ts_delay(close, 10)) / 10) - ((ts_delay(close, 10) - close) / 10))"
        self._add(46, f"ts_where(0.25 < {inner}, -1, ts_where({inner} < 0, 1, -1 * (close - ts_delay(close, 1))))")
        self._add(47, f"(((rq_rank(1 / close) * volume) / ts_mean(volume, 20)) * ((high * rq_rank(high - close)) / (ts_sum(high, 5) / 5))) - rq_rank({v} - ts_delay({v}, 5))")
        self._add(48, "rq_indneutralize((ts_corr(close - ts_delay(close, 1), ts_delay(close, 1) - ts_delay(close, 2), 250) * (close - ts_delay(close, 1)) / close), industry) / ts_sum(((close - ts_delay(close, 1)) / ts_delay(close, 1)) ** 2, 250)")
        self._add(49, f"ts_where({inner} < -0.1, 1, -1 * (close - ts_delay(close, 1)))")
        self._add(50, f"-1 * ts_max(rq_rank(ts_corr(rq_rank(volume), rq_rank({v}), 5)), 5)")
        self._add(51, f"ts_where({inner} < -0.05, 1, -1 * (close - ts_delay(close, 1)))")
        self._add(52, f"((-1 * ts_min(low, 5)) + ts_delay(ts_min(low, 5), 5)) * rq_rank((ts_sum({r}, 240) - ts_sum({r}, 20)) / 220) * rq_ts_rank(volume, 5)")
        self._add(53, "-1 * ((((close - low) - (high - close)) / (close - low)) - ts_delay(((close - low) - (high - close)) / (close - low), 9))")
        self._add(54, "-1 * ((low - close) * (open ** 5)) / ((low - high) * (close ** 5))")
        self._add(55, "-1 * ts_corr(rq_rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), rq_rank(volume), 6)")
        self._add(56, f"0 - (rq_rank(ts_sum({r}, 10) / ts_sum(ts_sum({r}, 2), 3)) * rq_rank({r} * cap))")
        self._add(57, f"0 - ((close - {v}) / ts_decay_linear(rq_rank(rq_ts_argmax(close, 30)), 2))")
        self._add(58, f"-1 * rq_ts_rank(ts_decay_linear(ts_corr(rq_indneutralize({v}, industry), volume, 4), 8), 6)")
        self._add(59, f"-1 * rq_ts_rank(ts_decay_linear(ts_corr(rq_indneutralize(({v} * 0.728317) + ({v} * (1 - 0.728317)), industry), volume, 4), 16), 8)")
        self._add(60, "0 - (2 * cs_scale(rq_rank((((close - low) - (high - close)) / (high - low)) * volume)) - cs_scale(rq_rank(rq_ts_argmax(close, 10))))")
        self._add(61, f"rq_rank({v} - ts_min({v}, 16)) < rq_rank(ts_corr({v}, ts_mean(volume, 180), 18))")
        self._add(62, f"-1 * (rq_rank(ts_corr({v}, ts_sum(ts_mean(volume, 20), 22), 10)) < rq_rank((rq_rank(open) + rq_rank(open)) < (rq_rank((high + low) / 2) + rq_rank(high))))")
        self._add(63, f"(rq_rank(ts_decay_linear((rq_indneutralize(close, industry) - ts_delay(rq_indneutralize(close, industry), 2)), 8)) - rq_rank(ts_decay_linear(ts_corr(({v} * 0.318108) + (open * (1 - 0.318108)), ts_sum(ts_mean(volume, 180), 37), 14), 12))) * -1")
        self._add(64, f"-1 * (rq_rank(ts_corr(ts_sum((open * 0.178404) + (low * (1 - 0.178404)), 13), ts_sum(ts_mean(volume, 120), 13), 17)) < rq_rank((((high + low) / 2) * 0.178404 + {v} * (1 - 0.178404)) - ts_delay(((high + low) / 2) * 0.178404 + {v} * (1 - 0.178404), 4)))")
        self._add(65, f"-1 * (rq_rank(ts_corr((open * 0.00817205) + ({v} * (1 - 0.00817205)), ts_sum(ts_mean(volume, 60), 9), 6)) < rq_rank(open - ts_min(open, 14)))")
        self._add(66, f"(rq_rank(ts_decay_linear({v} - ts_delay({v}, 4), 7)) + rq_ts_rank(ts_decay_linear((((low * 0.96633) + (low * (1 - 0.96633))) - {v}) / (open - ((high + low) / 2)), 11), 7)) * -1")
        self._add(67, f"(rq_rank(high - ts_min(high, 2)) ** rq_rank(ts_corr(rq_indneutralize({v}, industry), rq_indneutralize(ts_mean(volume, 20), industry), 6))) * -1")
        self._add(68, "-1 * (rq_ts_rank(ts_corr(rq_rank(high), rq_rank(ts_mean(volume, 15)), 9), 14) < rq_rank((close * 0.518371 + low * (1 - 0.518371)) - ts_delay(close * 0.518371 + low * (1 - 0.518371), 1)))")
        self._add(69, f"(rq_rank(ts_max(rq_indneutralize({v}, industry) - ts_delay(rq_indneutralize({v}, industry), 3), 5)) ** rq_ts_rank(ts_corr((close * 0.490655) + ({v} * (1 - 0.490655)), ts_mean(volume, 20), 5), 9)) * -1")
        self._add(70, f"(rq_rank({v} - ts_delay({v}, 1)) ** rq_ts_rank(ts_corr(rq_indneutralize(close, industry), ts_mean(volume, 50), 18), 18)) * -1")
        self._add(71, f"rq_max(rq_ts_rank(ts_decay_linear(ts_corr(rq_ts_rank(close, 3), rq_ts_rank(ts_mean(volume, 180), 12), 18), 4), 16), rq_ts_rank(ts_decay_linear(rq_rank((low + open) - ({v} + {v})) ** 2, 16), 4))")
        self._add(72, f"rq_rank(ts_decay_linear(ts_corr((high + low) / 2, ts_mean(volume, 40), 9), 10)) / rq_rank(ts_decay_linear(ts_corr(rq_ts_rank({v}, 4), rq_ts_rank(volume, 19), 7), 3))")
        self._add(73, f"rq_max(rq_rank(ts_decay_linear({v} - ts_delay({v}, 5), 3)), rq_ts_rank(ts_decay_linear((((open * 0.147155 + low * (1 - 0.147155)) - ts_delay(open * 0.147155 + low * (1 - 0.147155), 2)) / (open * 0.147155 + low * (1 - 0.147155))) * -1, 3), 17)) * -1")
        self._add(74, f"-1 * (rq_rank(ts_corr(close, ts_sum(ts_mean(volume, 30), 37), 15)) < rq_rank(ts_corr(rq_rank(high * 0.0261661 + {v} * (1 - 0.0261661)), rq_rank(volume), 11)))")
        self._add(75, f"rq_rank(ts_corr({v}, volume, 4)) < rq_rank(ts_corr(rq_rank(low), rq_rank(ts_mean(volume, 50)), 12))")
        self._add(76, f"rq_max(rq_rank(ts_decay_linear({v} - ts_delay({v}, 1), 12)), rq_ts_rank(ts_decay_linear(rq_ts_rank(ts_corr(rq_indneutralize(low, industry), ts_mean(volume, 81), 8), 20), 17), 19)) * -1")
        self._add(77, f"rq_min(rq_rank(ts_decay_linear((((high + low) / 2) + high) - ({v} + high), 20)), rq_rank(ts_decay_linear(ts_corr((high + low) / 2, ts_mean(volume, 40), 3), 6)))")
        self._add(78, f"rq_rank(ts_corr(ts_sum(low * 0.352233 + {v} * (1 - 0.352233), 20), ts_sum(ts_mean(volume, 40), 20), 7)) ** rq_rank(ts_corr(rq_rank({v}), rq_rank(volume), 6))")
        self._add(79, f"rq_rank((rq_indneutralize(close * 0.60733 + open * (1 - 0.60733), industry)) - ts_delay(rq_indneutralize(close * 0.60733 + open * (1 - 0.60733), industry), 1)) < rq_rank(ts_corr(rq_ts_rank({v}, 4), rq_ts_rank(ts_mean(volume, 150), 9), 15))")
        self._add(80, "(rq_rank(rq_sign((rq_indneutralize(open * 0.868128 + high * (1 - 0.868128), industry)) - ts_delay(rq_indneutralize(open * 0.868128 + high * (1 - 0.868128), industry), 4))) ** rq_ts_rank(ts_corr(high, ts_mean(volume, 10), 5), 6)) * -1")
        self._add(81, f"(rq_rank(ts_log(ts_prod(rq_rank(rq_rank(ts_corr({v}, ts_sum(ts_mean(volume, 10), 50), 8)) ** 4), 15))) < rq_rank(ts_corr(rq_rank({v}), rq_rank(volume), 5))) * -1")
        self._add(82, "-1 * rq_min(rq_rank(ts_decay_linear(open - ts_delay(open, 1), 15)), rq_ts_rank(ts_decay_linear(ts_corr(rq_indneutralize(volume, industry), open, 17), 7), 13))")
        self._add(83, f"(rq_rank(ts_delay((high - low) / (ts_sum(close, 5) / 5), 2)) * rq_rank(rq_rank(volume))) / (((high - low) / (ts_sum(close, 5) / 5)) / ({v} - close))")
        self._add(84, f"signed_power(rq_ts_rank({v} - ts_max({v}, 15), 21), close - ts_delay(close, 5))")
        self._add(85, "rq_rank(ts_corr((high * 0.876703) + (close * (1 - 0.876703)), ts_mean(volume, 30), 10)) ** rq_rank(ts_corr(rq_ts_rank((high + low) / 2, 4), rq_ts_rank(volume, 10), 7))")
        self._add(86, f"((rq_ts_rank(ts_corr(close, ts_sum(ts_mean(volume, 20), 15), 6), 20) < rq_rank((open + close) - ({v} + open))) * -1)")
        self._add(87, f"rq_max(rq_rank(ts_decay_linear((close * 0.369701 + {v} * (1 - 0.369701)) - ts_delay(close * 0.369701 + {v} * (1 - 0.369701), 2), 3)), rq_ts_rank(ts_decay_linear(abs(ts_corr(rq_indneutralize(ts_mean(volume, 81), industry), close, 13)), 5), 14)) * -1")
        self._add(88, "rq_min(rq_rank(ts_decay_linear((rq_rank(open) + rq_rank(low)) - (rq_rank(high) + rq_rank(close)), 8)), rq_ts_rank(ts_decay_linear(ts_corr(rq_ts_rank(close, 8), rq_ts_rank(ts_mean(volume, 60), 21), 8), 7), 3))")
        self._add(89, f"rq_ts_rank(ts_decay_linear(ts_corr(low, ts_mean(volume, 10), 7), 6), 4) - rq_ts_rank(ts_decay_linear(rq_indneutralize({v}, industry) - ts_delay(rq_indneutralize({v}, industry), 3), 10), 15)")
        self._add(90, "(rq_rank(close - ts_max(close, 5)) ** rq_ts_rank(ts_corr(rq_indneutralize(ts_mean(volume, 40), industry), low, 5), 3)) * -1")
        self._add(91, f"(rq_ts_rank(ts_decay_linear(ts_decay_linear(ts_corr(rq_indneutralize(close, industry), volume, 10), 16), 4), 5) - rq_rank(ts_decay_linear(ts_corr({v}, ts_mean(volume, 30), 4), 3))) * -1")
        self._add(92, "rq_min(rq_ts_rank(ts_decay_linear(rq_as_float((((high + low) / 2) + close) < (low + open)), 15), 19), rq_ts_rank(ts_decay_linear(ts_corr(rq_rank(low), rq_rank(ts_mean(volume, 30)), 8), 7), 7))")
        self._add(93, f"rq_ts_rank(ts_decay_linear(ts_corr(rq_indneutralize({v}, industry), ts_mean(volume, 81), 17), 20), 8) / rq_rank(ts_decay_linear((close * 0.524434 + {v} * (1 - 0.524434)) - ts_delay(close * 0.524434 + {v} * (1 - 0.524434), 3), 16))")
        self._add(94, f"(rq_rank({v} - ts_min({v}, 12)) ** rq_ts_rank(ts_corr(rq_ts_rank({v}, 20), rq_ts_rank(ts_mean(volume, 60), 4), 18), 3)) * -1")
        self._add(95, "rq_rank(open - ts_min(open, 12)) < rq_ts_rank(rq_rank(ts_corr(ts_sum((high + low) / 2, 19), ts_sum(ts_mean(volume, 40), 19), 13)) ** 5, 12)")
        self._add(96, f"rq_max(rq_ts_rank(ts_decay_linear(ts_corr(rq_rank({v}), rq_rank(volume), 4), 4), 8), rq_ts_rank(ts_decay_linear(rq_ts_argmax(ts_corr(rq_ts_rank(close, 7), rq_ts_rank(ts_mean(volume, 60), 4), 4), 13), 14), 13)) * -1")
        self._add(97, f"(rq_rank(ts_decay_linear((rq_indneutralize(low * 0.721001 + {v} * (1 - 0.721001), industry)) - ts_delay(rq_indneutralize(low * 0.721001 + {v} * (1 - 0.721001), industry), 3), 20)) - rq_ts_rank(ts_decay_linear(rq_ts_rank(ts_corr(rq_ts_rank(low, 8), rq_ts_rank(ts_mean(volume, 60), 17), 5), 19), 16), 7)) * -1")
        self._add(98, f"rq_rank(ts_decay_linear(ts_corr({v}, ts_sum(ts_mean(volume, 5), 26), 6), 7)) - rq_rank(ts_decay_linear(rq_ts_rank(rq_ts_argmin(ts_corr(rq_rank(open), rq_rank(ts_mean(volume, 15)), 21), 9), 7), 8))")
        self._add(99, "-1 * (rq_rank(ts_corr(ts_sum((high + low) / 2, 20), ts_sum(ts_mean(volume, 60), 20), 9)) < rq_rank(ts_corr(low, volume, 6)))")
        self._add(100, "0 - (((1.5 * cs_scale(rq_indneutralize(rq_indneutralize(rq_rank((((close - low) - (high - close)) / (high - low)) * volume), industry), industry))) - cs_scale(rq_indneutralize(ts_corr(close, rq_rank(ts_mean(volume, 20)), 5) - rq_rank(rq_ts_argmin(close, 30)), industry))) * (volume / ts_mean(volume, 20)))")
        self._add(101, "(close - open) / ((high - low) + 0.001)")
