from __future__ import annotations

import polars as pl

from vnpy.alpha.dataset import AlphaDataset


class Alpha101DolphinDB(AlphaDataset):
    """
    DolphinDB WQ101 派生模板。

    表达式尽量保留 DolphinDB `wq101alpha.dos` 的算子命名与窗口语义，
    依赖后续兼容入口提供 rowRank/mrank/mavg/mfirst/msum/mcorr 等函数。
    """

    TODO_WQALPHA: tuple[int, ...] = ()

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
        super().__init__(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
            interval=interval,
            enable_cache=enable_cache,
            cache_dir=cache_dir,
        )
        try:
            self.set_label("mfirst(close, -3) / mfirst(close, -1) - 1")
        except Exception:
            pass

    def register_all(self) -> None:
        self.register_001_010()
        self.register_011_020()
        self.register_021_030()
        self.register_031_040()
        self.register_041_060()
        self.register_061_080()
        self.register_081_101()

    def register_001_010(self) -> None:
        self.add_feature(
            "ddb_wq101_001",
            "rowRank(mimax(signedPower(iif(ratios(close) - 1 < 0, mstd(ratios(close) - 1, 20), close), 2.0), 5), percent=true) - 0.5",
        )
        self.add_feature(
            "ddb_wq101_002",
            "-mcorr(rowRank(log(volume) - log(mfirst(volume, 3)), percent=true), rowRank((close - open) / open, percent=true), 6)",
        )
        self.add_feature(
            "ddb_wq101_003",
            "-mcorr(rowRank(open, percent=true), rowRank(volume, percent=true), 10)",
        )
        self.add_feature(
            "ddb_wq101_004",
            "-mrank(rowRank(low, percent=true), true, 9)",
        )
        self.add_feature(
            "ddb_wq101_005",
            "rowRank(open - (msum(vwap, 10) / 10), percent=true) * (-1 * abs(rowRank(close - vwap, percent=true)))",
        )
        self.add_feature("ddb_wq101_006", "-mcorr(open, volume, 10)")
        self.add_feature(
            "ddb_wq101_007",
            "iif(mavg(volume, 20) < volume, -mrank(abs(close - mfirst(close, 8)), true, 60) * sign(close - mfirst(close, 8)), -1)",
        )
        self.add_feature(
            "ddb_wq101_008",
            "-rowRank((msum(open, 5) * msum(ratios(close) - 1, 5)) - mfirst(msum(open, 5) * msum(ratios(close) - 1, 5), 11), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_009",
            "iif(0 < mmin(close - mfirst(close, 2), 5), close - mfirst(close, 2), iif(mmax(close - mfirst(close, 2), 5) < 0, close - mfirst(close, 2), -(close - mfirst(close, 2))))",
        )
        self.add_feature(
            "ddb_wq101_010",
            "rowRank(iif(0 < mmin(close - mfirst(close, 2), 4), close - mfirst(close, 2), iif(mmax(close - mfirst(close, 2), 4) < 0, close - mfirst(close, 2), -(close - mfirst(close, 2)))), percent=true)",
        )

    def register_011_020(self) -> None:
        self.add_feature(
            "ddb_wq101_011",
            "(rowRank(mmax(vwap - close, 3), percent=true) + rowRank(mmin(vwap - close, 3), percent=true)) * rowRank(volume - mfirst(volume, 4), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_012",
            "sign(volume - mfirst(volume, 2)) * (-1 * (close - mfirst(close, 2)))",
        )
        self.add_feature(
            "ddb_wq101_013",
            "-rowRank(mcovar(rowRank(close, percent=true), rowRank(volume, percent=true), 5), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_014",
            "-rowRank((ratios(close) - 1) - mfirst(ratios(close) - 1, 4), percent=true) * mcovar(open, volume, 10)",
        )
        self.add_feature(
            "ddb_wq101_015",
            "-msum(rowRank(mcorr(rowRank(high, percent=true), rowRank(volume, percent=true), 3), percent=true), 3)",
        )
        self.add_feature(
            "ddb_wq101_016",
            "-rowRank(mcovar(rowRank(high, percent=true), rowRank(volume, percent=true), 5), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_017",
            "-rowRank(mrank(close, true, 10), percent=true) * rowRank((close - mfirst(close, 2)) - mfirst(close - mfirst(close, 2), 2), percent=true) * rowRank(mrank(volume / mavg(volume, 20), true, 5), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_018",
            "-rowRank(mstd(abs(close - open), 5) + close - open + mcorr(close, open, 10), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_019",
            "-sign(close - mfirst(close, 8) + close - mfirst(close, 8)) * (1 + rowRank(1 + msum(ratios(close) - 1, 250), percent=true))",
        )
        self.add_feature(
            "ddb_wq101_020",
            "-rowRank(open - mfirst(high, 2), percent=true) * rowRank(open - mfirst(close, 2), percent=true) * rowRank(open - mfirst(low, 2), percent=true)",
        )

    def register_021_030(self) -> None:
        self.add_feature(
            "ddb_wq101_021",
            "iif((msum(close, 8) / 8 + mstd(close, 8)) < (msum(close, 2) / 2), -1, iif((msum(close, 2) / 2) < (msum(close, 8) / 8 - mstd(close, 8)), 1, iif((volume / mavg(volume, 20)) >= 1, 1, -1)))",
        )
        self.add_feature(
            "ddb_wq101_022",
            "-(mcorr(high, volume, 5) - mfirst(mcorr(high, volume, 5), 6)) * rowRank(mstd(close, 20), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_023",
            "iif(msum(high, 20) / 20 < high, -(high - mfirst(high, 3)), 0)",
        )
        self.add_feature(
            "ddb_wq101_024",
            "iif((msum(close, 100) / 100 - mfirst(msum(close, 100) / 100, 101)) / mfirst(close, 101) <= 0.05, -(close - mmin(close, 100)), -(close - mfirst(close, 4)))",
        )
        self.add_feature(
            "ddb_wq101_025",
            "rowRank((-(ratios(close) - 1) * mavg(volume, 20) * vwap * (high - close)), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_026",
            "-mmax(mcorr(mrank(volume, true, 5), mrank(high, true, 5), 5), 3)",
        )
        self.add_feature(
            "ddb_wq101_027",
            "iif(0.5 < rowRank(msum(mcorr(rowRank(volume, percent=true), rowRank(vwap, percent=true), 6), 2) / 2.0, percent=true), -1, 1)",
        )
        self.add_feature(
            "ddb_wq101_028",
            "cs_scale(mcorr(mavg(volume, 20), low, 5) + ((high + low) / 2) - close)",
        )
        self.add_feature(
            "ddb_wq101_029",
            "mmin(rowRank(rowRank(cs_scale(log(mmin(rowRank(rowRank(-rowRank(close - 1 - mfirst(close - 1, 6), percent=true), percent=true), percent=true), 2))), percent=true), percent=true), 5) + mrank(mfirst(-(ratios(close) - 1), 7), true, 5)",
        )
        self.add_feature(
            "ddb_wq101_030",
            "(1.0 - rowRank(sign(close - mfirst(close, 2)) + sign(mfirst(close, 2) - mfirst(close, 3)) + sign(mfirst(close, 3) - mfirst(close, 4)), percent=true)) * msum(volume, 5) / msum(volume, 20)",
        )

    def register_031_040(self) -> None:
        self.add_feature(
            "ddb_wq101_031",
            "rowRank(rowRank(rowRank(ts_decay_linear(-rowRank(rowRank(close - mfirst(close, 11), percent=true), percent=true), 10), percent=true), percent=true), percent=true) + rowRank(-(close - mfirst(close, 4)), percent=true) + sign(cs_scale(mcorr(mavg(volume, 20), low, 12)))",
        )
        self.add_feature(
            "ddb_wq101_032",
            "cs_scale(msum(close, 7) / 7 - close) + 20 * cs_scale(mcorr(vwap, mfirst(close, 6), 230))",
        )
        self.add_feature("ddb_wq101_033", "rowRank(open / close - 1, percent=true)")
        self.add_feature(
            "ddb_wq101_034",
            "rowRank(1 - rowRank(mstd(ratios(close) - 1, 2) / mstd(ratios(close) - 1, 5), percent=true) + 1 - rowRank(close - mfirst(close, 2), percent=true), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_035",
            "mrank(volume, true, 32) * (1 - mrank(close + high - low, true, 16)) * (1 - mrank(ratios(close) - 1, true, 32))",
        )
        self.add_feature(
            "ddb_wq101_036",
            "2.21 * rowRank(mcorr(close - open, mfirst(volume, 2), 15), percent=true) + 0.7 * rowRank(open - close, percent=true) + 0.73 * rowRank(mrank(mfirst(-(ratios(close) - 1), 7), true, 5), percent=true) + rowRank(abs(mcorr(vwap, mavg(volume, 20), 6)), percent=true) + 0.6 * rowRank((msum(close, 200) / 200 - open) * (close - open), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_037",
            "rowRank(mcorr(mfirst(open - close, 2), close, 200), percent=true) + rowRank(open - close, percent=true)",
        )
        self.add_feature(
            "ddb_wq101_038",
            "-rowRank(mrank(close, true, 10), percent=true) * rowRank(close / open, percent=true)",
        )
        self.add_feature(
            "ddb_wq101_039",
            "-rowRank((close - mfirst(close, 8)) * (1 - rowRank(ts_decay_linear(volume / mavg(volume, 20), 9), percent=true)), percent=true) * (1 + rowRank(msum(ratios(close) - 1, 250), percent=true))",
        )
        self.add_feature(
            "ddb_wq101_040",
            "-rowRank(mstd(high, 10), percent=true) * mcorr(high, volume, 10)",
        )

    def register_041_060(self) -> None:
        self.add_feature("ddb_wq101_041", "(high * low) ** 0.5 - vwap")
        self.add_feature(
            "ddb_wq101_042",
            "rowRank(vwap - close, percent=true) / rowRank(vwap + close, percent=true)",
        )
        self.add_feature(
            "ddb_wq101_043",
            "mrank(volume / mavg(volume, 20), true, 20) * mrank(-(close - mfirst(close, 8)), true, 8)",
        )
        self.add_feature(
            "ddb_wq101_044",
            "-mcorr(high, rowRank(volume, percent=true), 5)",
        )
        self.add_feature(
            "ddb_wq101_045",
            "-rowRank(msum(mfirst(close, 6), 20) / 20, percent=true) * mcorr(close, volume, 2) * rowRank(mcorr(msum(close, 5), msum(close, 20), 2), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_046",
            "iif(0.25 < ((mfirst(close, 21) - mfirst(close, 11)) / 10 - (mfirst(close, 11) - close) / 10), -1, iif(((mfirst(close, 21) - mfirst(close, 11)) / 10 - (mfirst(close, 11) - close) / 10) < 0, 1, mfirst(close, 2) - close))",
        )
        self.add_feature(
            "ddb_wq101_047",
            "rowRank(1 / close, percent=true) * volume / mavg(volume, 20) * (high * rowRank(high - close, percent=true) / (msum(high, 5) / 5)) - rowRank(vwap - mfirst(vwap, 6), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_048",
            "cs_demean_by_category(mcorr(close - mfirst(close, 2), mfirst(close, 2) - mfirst(mfirst(close, 2), 2), 250) * (close - mfirst(close, 2)) / close, indclass) / msum(((close - mfirst(close, 2)) / mfirst(close, 2)) ** 2, 250)",
        )
        self.add_feature(
            "ddb_wq101_049",
            "iif(((mfirst(close, 21) - mfirst(close, 11)) / 10 - (mfirst(close, 11) - close) / 10) < -0.1, 1, mfirst(close, 2) - close)",
        )
        self.add_feature(
            "ddb_wq101_050",
            "-mmax(rowRank(mcorr(rowRank(volume, percent=true), rowRank(vwap, percent=true), 5), percent=true), 5)",
        )
        self.add_feature(
            "ddb_wq101_051",
            "iif(((mfirst(close, 21) - mfirst(close, 11)) / 10 - (mfirst(close, 11) - close) / 10) < -0.05, 1, -(close - mfirst(close, 2)))",
        )
        self.add_feature(
            "ddb_wq101_052",
            "(-mmin(low, 5) + mfirst(mmin(low, 5), 6)) * rowRank((msum(ratios(close) - 1, 240) - msum(ratios(close) - 1, 20)) / 220, percent=true) * mrank(volume, true, 5)",
        )
        self.add_feature(
            "ddb_wq101_053",
            "-(((close - low) - (high - close)) / (close - low) - mfirst(((close - low) - (high - close)) / (close - low), 10))",
        )
        self.add_feature(
            "ddb_wq101_054",
            "-(low - close) * (open ** 5) / ((low - high) * (close ** 5))",
        )
        self.add_feature(
            "ddb_wq101_055",
            "-mcorr(rowRank((close - mmin(low, 12)) / (mmax(high, 12) - mmin(low, 12)), percent=true), rowRank(volume, percent=true), 6)",
        )
        self.add_feature(
            "ddb_wq101_056",
            "-rowRank(msum(ratios(close) - 1, 10) / msum(msum(ratios(close) - 1, 2), 3), percent=true) * rowRank((ratios(close) - 1) * cap, percent=true)",
        )
        self.add_feature(
            "ddb_wq101_057",
            "-(close - vwap) / ts_decay_linear(rowRank(mimax(close, 30), percent=true), 2)",
        )
        self.add_feature(
            "ddb_wq101_058",
            "-mrank(ts_decay_linear(mcorr(cs_demean_by_category(vwap, indclass), volume, 4), 8), true, 6)",
        )
        self.add_feature(
            "ddb_wq101_059",
            "-mrank(ts_decay_linear(mcorr(cs_demean_by_category(vwap * 0.728317 + vwap * (1 - 0.728317), indclass), volume, 4), 16), true, 8)",
        )
        self.add_feature(
            "ddb_wq101_060",
            "-(2 * cs_scale(rowRank(((close - low) - (high - close)) / (high - low) * volume, percent=true)) - cs_scale(rowRank(mimax(close, 10), percent=true)))",
        )

    def register_061_080(self) -> None:
        self.add_feature(
            "ddb_wq101_061",
            "rowRank(vwap - mmin(vwap, 16), percent=true) < rowRank(mcorr(vwap, mavg(volume, 180), 18), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_062",
            "(rowRank(mcorr(vwap, msum(mavg(volume, 20), 22), 10), percent=true) < rowRank(((rowRank(open, percent=true) + rowRank(open, percent=true)) < (rowRank((high + low) / 2, percent=true) + rowRank(high, percent=true))) * 1, percent=true)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_063",
            "(rowRank(ts_decay_linear(cs_demean_by_category(close, indclass) - mfirst(cs_demean_by_category(close, indclass), 3), 8), percent=true) - rowRank(ts_decay_linear(mcorr(vwap * 0.318108 + open * (1 - 0.318108), msum(mavg(volume, 180), 37), 14), 12), percent=true)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_064",
            "(rowRank(mcorr(msum(open * 0.178404 + low * (1 - 0.178404), 13), msum(mavg(volume, 120), 13), 17), percent=true) < rowRank(((high + low) / 2 * 0.178404 + vwap * (1 - 0.178404)) - mfirst(((high + low) / 2 * 0.178404 + vwap * (1 - 0.178404)), 5), percent=true)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_065",
            "(rowRank(mcorr(open * 0.00817205 + vwap * (1 - 0.00817205), msum(mavg(volume, 60), 9), 6), percent=true) < rowRank(open - mmin(open, 14), percent=true)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_066",
            "(rowRank(ts_decay_linear(vwap - mfirst(vwap, 5), 7), percent=true) + mrank(ts_decay_linear((low - vwap) / (open - ((high + low) / 2)), 11), true, 7)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_067",
            "(rowRank(high - mmin(high, 2), percent=true) ** rowRank(mcorr(cs_demean_by_category(vwap, indclass), cs_demean_by_category(mavg(volume, 20), indclass), 6), percent=true)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_068",
            "(mrank(mcorr(rowRank(high, percent=true), rowRank(mavg(volume, 15), percent=true), 9), true, 14) < rowRank((close * 0.518371 + low * (1 - 0.518371)) - mfirst((close * 0.518371 + low * (1 - 0.518371)), 2), percent=true)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_069",
            "(rowRank(mmax(cs_demean_by_category(vwap, indclass) - mfirst(cs_demean_by_category(vwap, indclass), 4), 5), percent=true) ** mrank(mcorr(close * 0.490655 + vwap * (1 - 0.490655), mavg(volume, 20), 5), true, 9)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_070",
            "(rowRank(vwap - mfirst(vwap, 2), percent=true) ** mrank(mcorr(cs_demean_by_category(close, indclass), mavg(volume, 50), 18), true, 18)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_071",
            "ddb_max(mrank(ts_decay_linear(mcorr(mrank(close, true, 3), mrank(mavg(volume, 180), true, 12), 18), 4), true, 16), mrank(ts_decay_linear(rowRank(low + open - (vwap + vwap), percent=true) ** 2, 16), true, 4))",
        )
        self.add_feature(
            "ddb_wq101_072",
            "rowRank(ts_decay_linear(mcorr((high + low) / 2, mavg(volume, 40), 9), 10), percent=true) / rowRank(ts_decay_linear(mcorr(mrank(vwap, true, 4), mrank(volume, true, 19), 7), 3), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_073",
            "ddb_max(rowRank(ts_decay_linear(vwap - mfirst(vwap, 6), 3), percent=true), mrank(ts_decay_linear(((open * 0.147155 + low * (1 - 0.147155)) - mfirst((open * 0.147155 + low * (1 - 0.147155)), 3)) / (open * 0.147155 + low * (1 - 0.147155)) * (-1), 3), true, 17)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_074",
            "(rowRank(mcorr(close, msum(mavg(volume, 30), 37), 15), percent=true) < rowRank(mcorr(rowRank(rowRank(high * 0.0261661 + vwap * (1 - 0.0261661), percent=true), percent=true), rowRank(volume, percent=true), 11), percent=true)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_075",
            "rowRank(mcorr(vwap, volume, 4), percent=true) < rowRank(mcorr(rowRank(low, percent=true), rowRank(mavg(volume, 50), percent=true), 12), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_076",
            "ddb_max(rowRank(ts_decay_linear(vwap - mfirst(vwap, 2), 12), percent=true), mrank(ts_decay_linear(mrank(mcorr(cs_demean_by_category(low, indclass), mavg(volume, 81), 8), true, 20), 17), true, 19)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_077",
            "ddb_min(rowRank(ts_decay_linear((high + low) / 2 + high - (vwap + high), 20), percent=true), rowRank(ts_decay_linear(mcorr((high + low) / 2, mavg(volume, 40), 3), 6), percent=true))",
        )
        self.add_feature(
            "ddb_wq101_078",
            "rowRank(mcorr(msum(low * 0.352233 + vwap * (1 - 0.352233), 20), msum(mavg(volume, 40), 20), 7), percent=true) ** rowRank(mcorr(rowRank(vwap, percent=true), rowRank(volume, percent=true), 6), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_079",
            "rowRank((close * 0.60733 + open * (1 - 0.60733) - cs_demean_by_category(close * 0.60733 + open * (1 - 0.60733), indclass)) - mfirst(close * 0.60733 + open * (1 - 0.60733) - cs_demean_by_category(close * 0.60733 + open * (1 - 0.60733), indclass), 2), percent=true) < rowRank(mcorr(mrank(vwap, true, 4), mrank(mavg(volume, 150), true, 9), 15), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_080",
            "rowRank(sign((open * 0.868128 + high * (1 - 0.868128) - cs_demean_by_category(open * 0.868128 + high * (1 - 0.868128), indclass)) - mfirst(open * 0.868128 + high * (1 - 0.868128) - cs_demean_by_category(open * 0.868128 + high * (1 - 0.868128), indclass), 5)), percent=true) ** mrank(mcorr(high, mavg(volume, 10), 5), true, 6)",
        )

    def register_081_101(self) -> None:
        self.add_feature(
            "ddb_wq101_081",
            "(rowRank(log(mprod(rowRank(rowRank(mcorr(vwap, msum(mavg(volume, 10), 49), 8), percent=true) ** 4, percent=true), 15)), percent=true) < rowRank(mcorr(rowRank(vwap, percent=true), rowRank(volume, percent=true), 5), percent=true)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_082",
            "ddb_min(rowRank(ts_decay_linear(open - mfirst(open, 2), 15), percent=true), mrank(ts_decay_linear(mcorr(cs_demean_by_category(volume, indclass), open * 0.634196 + open * (1 - 0.634196), 17), 7), true, 13)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_083",
            "rowRank(mfirst((high - low) / (msum(close, 5) / 5), 3), percent=true) * rowRank(rowRank(volume, percent=true), percent=true) / (((high - low) / (msum(close, 5) / 5)) / (vwap - close))",
        )
        self.add_feature(
            "ddb_wq101_084",
            "signedPower(mrank(vwap - mmax(vwap, 15), true, 20), close - mfirst(close, 6))",
        )
        self.add_feature(
            "ddb_wq101_085",
            "rowRank(mcorr(high * 0.876703 + close * (1 - 0.876703), mavg(volume, 30), 10), percent=true) ** rowRank(mcorr(mrank((high + low) / 2, true, 4), mrank(volume, true, 10), 7), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_086",
            "(mrank(mcorr(close, msum(mavg(volume, 20), 15), 6), true, 20) < rowRank(open + close - (vwap + open), percent=true)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_087",
            "ddb_max(rowRank(ts_decay_linear(close * 0.369701 + vwap * (1 - 0.369701) - mfirst(close * 0.369701 + vwap * (1 - 0.369701), 3), 3), percent=true), mrank(ts_decay_linear(abs(mcorr(cs_demean_by_category(mavg(volume, 81), indclass), close, 13)), 5), true, 14)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_088",
            "ddb_min(rowRank(ts_decay_linear(rowRank(open, percent=true) + rowRank(low, percent=true) - (rowRank(high, percent=true) + rowRank(close, percent=true)), 8), percent=true), mrank(ts_decay_linear(mcorr(mrank(close, true, 8), mrank(mavg(volume, 60), true, 21), 8), 7), true, 3))",
        )
        self.add_feature(
            "ddb_wq101_089",
            "mrank(ts_decay_linear(mcorr(low * 0.967285 + low * (1 - 0.967285), mavg(volume, 10), 7), 6), true, 4) - mrank(ts_decay_linear(cs_demean_by_category(vwap, indclass) - mfirst(cs_demean_by_category(vwap, indclass), 4), 10), true, 15)",
        )
        self.add_feature(
            "ddb_wq101_090",
            "(rowRank(close - mmax(close, 5), percent=true) ** mrank(mcorr(cs_demean_by_category(mavg(volume, 40), indclass), low, 5), true, 3)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_091",
            "mrank(ts_decay_linear(ts_decay_linear(mcorr(cs_demean_by_category(close, indclass), volume, 10), 16), 4), true, 5) - rowRank(ts_decay_linear(mcorr(vwap, mavg(volume, 30), 4), 3), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_092",
            "ddb_min(mrank(ts_decay_linear(((high + low) / 2 + close) < (low + open), 15), true, 19), mrank(ts_decay_linear(mcorr(rowRank(low, percent=true), rowRank(mavg(volume, 30), percent=true), 8), 7), true, 7))",
        )
        self.add_feature(
            "ddb_wq101_093",
            "mrank(ts_decay_linear(mcorr(cs_demean_by_category(vwap, indclass), mavg(volume, 81), 17), 20), true, 8) / rowRank(ts_decay_linear(close * 0.524434 + vwap * (1 - 0.524434) - mfirst(close * 0.524434 + vwap * (1 - 0.524434), 4), 16), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_094",
            "(rowRank(vwap - mmin(vwap, 12), percent=true) ** mrank(mcorr(mrank(vwap, true, 20), mrank(mavg(volume, 60), true, 4), 18), true, 3)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_095",
            "rowRank(open - mmin(open, 12), percent=true) < mrank(rowRank(mcorr(msum((high + low) / 2, 19), msum(mavg(volume, 40), 19), 13), percent=true) ** 5, true, 12)",
        )
        self.add_feature(
            "ddb_wq101_096",
            "ddb_max(mrank(ts_decay_linear(mcorr(rowRank(vwap, percent=true), rowRank(volume, percent=true), 4), 4), true, 8), mrank(ts_decay_linear(mimax(mcorr(mrank(close, true, 7), mrank(mavg(volume, 60), true, 4), 4), 13), 14), true, 13)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_097",
            "(rowRank(ts_decay_linear(cs_demean_by_category(low * 0.721001 + vwap * (1 - 0.721001), indclass) - mfirst(cs_demean_by_category(low * 0.721001 + vwap * (1 - 0.721001), indclass), 4), 20), percent=true) - mrank(ts_decay_linear(mrank(mcorr(mrank(low, true, 8), mrank(mavg(volume, 60), true, 17), 5), true, 19), 16), true, 7)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_098",
            "rowRank(ts_decay_linear(mcorr(vwap, msum(mavg(volume, 5), 26), 5), 7), percent=true) - rowRank(ts_decay_linear(mrank(9 - mimin(mcorr(rowRank(open, percent=true), rowRank(mavg(volume, 15), percent=true), 21), 9), true, 7), 8), percent=true)",
        )
        self.add_feature(
            "ddb_wq101_099",
            "(rowRank(mcorr(msum((high + low) / 2, 20), msum(mavg(volume, 60), 20), 9), percent=true) < rowRank(mcorr(low, volume, 6), percent=true)) * (-1)",
        )
        self.add_feature(
            "ddb_wq101_100",
            "-(1.5 * cs_scale(cs_demean_by_category(cs_demean_by_category(rowRank(((close - low - (high - close)) / (high - low) * volume), percent=true), indclass), indclass)) - cs_scale(cs_demean_by_category(mcorr(close, rowRank(mavg(volume, 20), percent=true), 5) - rowRank(mimin(close, 30), percent=true), indclass))) * (volume / mavg(volume, 20))",
        )
        self.add_feature(
            "ddb_wq101_101",
            "(close - open) / (high - low + 0.001)",
        )
