from __future__ import annotations

import polars as pl

from vnpy.alpha.dataset import AlphaDataset


class Alpha191DolphinDB(AlphaDataset):
    """DolphinDB GTJA191-style expression template."""

    TODO_GTJA: tuple[int, ...] = ()
    BLOCKED_GTJA: tuple[int, ...] = ()

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

    def _add_ddb(self, number: int, expression: str) -> None:
        self.add_feature(f"ddb_gtja191_{number:03d}", expression)

    def register_001_080(self) -> None:
        ret = "ratios(close) - 1"
        amount = "volume * vwap"
        hlc3 = "(high + low + close) / 3"

        self._add_ddb(1, "-1 * mcorr(rowRank(log(volume) - mfirst(log(volume), 2), percent=true), rowRank((close - open) / open, percent=true), 6)")
        self._add_ddb(2, "-1 * (((close - low - (high - close)) / (high - low)) - mfirst(((close - low - (high - close)) / (high - low)), 2))")
        self._add_ddb(3, "msum(iif(close == mfirst(close, 2), 0, close - iif(close > mfirst(close, 2), iif(low < mfirst(close, 2), low, mfirst(close, 2)), iif(high > mfirst(close, 2), high, mfirst(close, 2)))), 6)")
        self._add_ddb(4, "iif((msum(close, 8) / 8 + mstd(close, 8)) < (msum(close, 2) / 2), -1, iif((msum(close, 2) / 2) < (msum(close, 8) / 8 - mstd(close, 8)), 1, iif(volume / mavg(volume, 20) >= 1, 1, -1)))")
        self._add_ddb(5, "-1 * mmax(mcorr(mrank(volume, true, 5), mrank(high, true, 5), 5), 3)")
        self._add_ddb(6, "-1 * rowRank(sign(open * 0.85 + high * 0.15 - mfirst(open * 0.85 + high * 0.15, 5)), percent=true)")
        self._add_ddb(7, "(rowRank(mmax(vwap - close, 3), percent=true) + rowRank(mmin(vwap - close, 3), percent=true)) * rowRank(volume - mfirst(volume, 4), percent=true)")
        self._add_ddb(8, "rowRank(((high + low) / 2 * 0.2 + vwap * 0.8 - mfirst((high + low) / 2 * 0.2 + vwap * 0.8, 5)) * -1, percent=true)")
        self._add_ddb(9, "ewmMean(((high + low) / 2 - (mfirst(high, 2) + mfirst(low, 2)) / 2) * (high - low) / volume, alpha=2/7)")
        self._add_ddb(10, f"rowRank(mmax(pow(iif(({ret}) < 0, mstd({ret}, 20), close), 2), 5), percent=true)")
        self._add_ddb(11, "msum((close - low - (high - close)) / (high - low) * volume, 6)")
        self._add_ddb(12, "rowRank(open - msum(vwap, 10) / 10, percent=true) * (-1 * rowRank(abs(close - vwap), percent=true))")
        self._add_ddb(13, "pow(high * low, 0.5) - vwap")
        self._add_ddb(14, "close - mfirst(close, 6)")
        self._add_ddb(15, "open / mfirst(close, 2) - 1")
        self._add_ddb(16, "-1 * mmax(rowRank(mcorr(rowRank(volume, percent=true), rowRank(vwap, percent=true), 5), percent=true), 5)")
        self._add_ddb(17, "pow(rowRank(vwap - mmax(vwap, 15), percent=true), close - mfirst(close, 6))")
        self._add_ddb(18, "close / mfirst(close, 6)")
        self._add_ddb(19, "iif(close < mfirst(close, 6), (close - mfirst(close, 6)) / mfirst(close, 6), iif(close == mfirst(close, 6), 0, (close - mfirst(close, 6)) / close))")
        self._add_ddb(20, "(close - mfirst(close, 7)) / mfirst(close, 7) * 100")
        self._add_ddb(21, "linearTimeTrend(close, 6)")

        self._add_ddb(22, "ewmMean((close - mavg(close, 6)) / mavg(close, 6) - move((close - mavg(close, 6)) / mavg(close, 6), 3), alpha=1/12)")
        self._add_ddb(23, "ewmMean(iif(close > move(close, 1), mstd(close, 20), 0), alpha=1/20) / (ewmMean(iif(close > move(close, 1), mstd(close, 20), 0), alpha=1/20) + ewmMean(iif(close <= move(close, 1), mstd(close, 20), 0), alpha=1/20)) * 100")
        self._add_ddb(24, "ewmMean(close - move(close, 5), alpha=1/5)")
        self._add_ddb(25, "-1 * rowRank((close - mfirst(close, 8)) * (1 - rowRank(ts_decay_linear(volume / mavg(volume, 20), 9), percent=true)), percent=true) * (1 + rowRank(msum(ratios(close) - 1, 250), percent=true))")
        self._add_ddb(26, "msum(close, 7) / 7 - close + mcorr(vwap, mfirst(close, 6), 230)")
        self._add_ddb(27, "ts_decay_linear((close - move(close, 3)) / move(close, 3) * 100 + (close - move(close, 6)) / move(close, 6) * 100, 12)")
        self._add_ddb(28, "3 * ewmMean((close - mmin(low, 9)) / (mmax(high, 9) - mmin(low, 9)) * 100, alpha=1/3) - 2 * ewmMean(ewmMean((close - mmin(low, 9)) / (mmax(high, 9) - mmin(low, 9)) * 100, alpha=1/3), alpha=1/3)")
        self._add_ddb(29, "(close - mfirst(close, 7)) / mfirst(close, 7) * volume")
        self._add_ddb(30, "ts_decay_linear(rollingOlsResidual3(close / move(close, 1), MKT, SMB, HML, 60) ** 2, 20)")

        self._add_ddb(31, "(close - mavg(close, 12)) / mavg(close, 12) * 100")
        self._add_ddb(32, "-1 * msum(rowRank(mcorr(rowRank(high, percent=true), rowRank(volume, percent=true), 3), percent=true), 3)")
        self._add_ddb(33, f"(-1 * mmin(low, 5) + mfirst(mmin(low, 5), 6)) * rowRank((msum({ret}, 240) - msum({ret}, 20)) / 220, percent=true) * mrank(volume, true, 5)")
        self._add_ddb(34, "mavg(close, 12) / close")
        self._add_ddb(35, "ddb_min(rowRank(ts_decay_linear(open - mfirst(open, 2), 15), percent=true), rowRank(ts_decay_linear(mcorr(volume, open, 17), 7), percent=true)) * -1")
        self._add_ddb(36, "rowRank(msum(mcorr(rowRank(volume, percent=true), rowRank(vwap, percent=true), 6), 2), percent=true)")
        self._add_ddb(37, f"-1 * rowRank(msum(open, 5) * msum({ret}, 5) - mfirst(msum(open, 5) * msum({ret}, 5), 11), percent=true)")
        self._add_ddb(38, "iif((msum(high, 20) / 20) < high, -1 * (high - mfirst(high, 3)), 0)")
        self._add_ddb(39, "(rowRank(ts_decay_linear(close - mfirst(close, 3), 8), percent=true) - rowRank(ts_decay_linear(mcorr(vwap * 0.3 + open * 0.7, msum(mavg(volume, 180), 37), 14), 12), percent=true)) * -1")
        self._add_ddb(40, "msum(iif(close > mfirst(close, 2), volume, 0), 26) / msum(iif(close <= mfirst(close, 2), volume, 0), 26) * 100")
        self._add_ddb(41, "rowRank(mmax(vwap - mfirst(vwap, 4), 5), percent=true) * -1")
        self._add_ddb(42, "-1 * rowRank(mstd(high, 10), percent=true) * mcorr(high, volume, 10)")
        self._add_ddb(43, "msum(iif(close > mfirst(close, 2), volume, iif(close < mfirst(close, 2), -volume, 0)), 6)")
        self._add_ddb(44, "mrank(ts_decay_linear(mcorr(low, mavg(volume, 10), 7), 6), true, 4) + mrank(ts_decay_linear(vwap - mfirst(vwap, 4), 10), true, 15)")

        self._add_ddb(45, "rowRank(close * 0.6 + open * 0.4 - mfirst(close * 0.6 + open * 0.4, 2), percent=true) * rowRank(mcorr(vwap, mavg(volume, 150), 15), percent=true)")
        self._add_ddb(46, "(mavg(close, 3) + mavg(close, 6) + mavg(close, 12) + mavg(close, 24)) / (4 * close)")
        self._add_ddb(47, "ewmMean((mmax(high, 6) - close) / (mmax(high, 6) - mmin(low, 6)) * 100, alpha=1/9)")
        self._add_ddb(48, "-1 * rowRank(sign(close - mfirst(close, 2)) + sign(mfirst(close, 2) - mfirst(close, 3)) + sign(mfirst(close, 3) - mfirst(close, 4)), percent=true) * msum(volume, 5) / msum(volume, 20)")
        self._add_ddb(49, "msum(iif((high + low) >= (mfirst(high, 2) + mfirst(low, 2)), 0, iif(abs(high - mfirst(high, 2)) > abs(low - mfirst(low, 2)), abs(high - mfirst(high, 2)), abs(low - mfirst(low, 2)))), 12) / (msum(iif((high + low) >= (mfirst(high, 2) + mfirst(low, 2)), 0, iif(abs(high - mfirst(high, 2)) > abs(low - mfirst(low, 2)), abs(high - mfirst(high, 2)), abs(low - mfirst(low, 2)))), 12) + msum(iif((high + low) <= (mfirst(high, 2) + mfirst(low, 2)), 0, iif(abs(high - mfirst(high, 2)) > abs(low - mfirst(low, 2)), abs(high - mfirst(high, 2)), abs(low - mfirst(low, 2)))), 12))")
        self._add_ddb(50, "msum(iif((high + low) <= (mfirst(high, 2) + mfirst(low, 2)), 0, iif(abs(high - mfirst(high, 2)) > abs(low - mfirst(low, 2)), abs(high - mfirst(high, 2)), abs(low - mfirst(low, 2)))), 12) / (msum(iif((high + low) <= (mfirst(high, 2) + mfirst(low, 2)), 0, iif(abs(high - mfirst(high, 2)) > abs(low - mfirst(low, 2)), abs(high - mfirst(high, 2)), abs(low - mfirst(low, 2)))), 12) + msum(iif((high + low) >= (mfirst(high, 2) + mfirst(low, 2)), 0, iif(abs(high - mfirst(high, 2)) > abs(low - mfirst(low, 2)), abs(high - mfirst(high, 2)), abs(low - mfirst(low, 2)))), 12)) - msum(iif((high + low) >= (mfirst(high, 2) + mfirst(low, 2)), 0, iif(abs(high - mfirst(high, 2)) > abs(low - mfirst(low, 2)), abs(high - mfirst(high, 2)), abs(low - mfirst(low, 2)))), 12) / (msum(iif((high + low) <= (mfirst(high, 2) + mfirst(low, 2)), 0, iif(abs(high - mfirst(high, 2)) > abs(low - mfirst(low, 2)), abs(high - mfirst(high, 2)), abs(low - mfirst(low, 2)))), 12) + msum(iif((high + low) >= (mfirst(high, 2) + mfirst(low, 2)), 0, iif(abs(high - mfirst(high, 2)) > abs(low - mfirst(low, 2)), abs(high - mfirst(high, 2)), abs(low - mfirst(low, 2)))), 12))")
        self._add_ddb(51, "msum(iif((high + low) <= (mfirst(high, 2) + mfirst(low, 2)), 0, iif(abs(high - mfirst(high, 2)) > abs(low - mfirst(low, 2)), abs(high - mfirst(high, 2)), abs(low - mfirst(low, 2)))), 12) / (msum(iif((high + low) <= (mfirst(high, 2) + mfirst(low, 2)), 0, iif(abs(high - mfirst(high, 2)) > abs(low - mfirst(low, 2)), abs(high - mfirst(high, 2)), abs(low - mfirst(low, 2)))), 12) + msum(iif((high + low) >= (mfirst(high, 2) + mfirst(low, 2)), 0, iif(abs(high - mfirst(high, 2)) > abs(low - mfirst(low, 2)), abs(high - mfirst(high, 2)), abs(low - mfirst(low, 2)))), 12))")
        self._add_ddb(52, f"msum(iif(0 > high - mfirst({hlc3}, 2), 0, high - mfirst({hlc3}, 2)), 26) / msum(iif(0 > mfirst({hlc3}, 2) - low, 0, mfirst({hlc3}, 2) - low), 26) * 100")
        self._add_ddb(53, "msum(iif(close > mfirst(close, 2), 1, 0), 12) / 12 * 100")
        self._add_ddb(54, "-1 * rowRank(mstd(abs(close - open), 10) + close - open + mcorr(close, open, 10), percent=true)")
        self._add_ddb(55, "msum(16 * (close - mfirst(close, 2) + (close - open) / 2 + mfirst(close, 2) - mfirst(open, 2)) / iif((abs(high - mfirst(close, 2)) > abs(low - mfirst(close, 2))) & (abs(high - mfirst(close, 2)) > abs(high - mfirst(low, 2))), abs(high - mfirst(close, 2)) + abs(low - mfirst(close, 2)) / 2 + abs(mfirst(close, 2) - mfirst(open, 2)) / 4, iif((abs(low - mfirst(close, 2)) > abs(high - mfirst(low, 2))) & (abs(low - mfirst(close, 2)) > abs(high - mfirst(close, 2))), abs(low - mfirst(close, 2)) + abs(high - mfirst(low, 2)) / 2 + abs(mfirst(close, 2) - mfirst(open, 2)) / 4, abs(high - mfirst(low, 2)) + abs(mfirst(close, 2) - mfirst(open, 2)) / 4)) * ddb_max(abs(high - mfirst(close, 2)), abs(low - mfirst(close, 2))), 20)")
        self._add_ddb(56, "rowRank(open - mmin(open, 12), percent=true) < rowRank(pow(rowRank(mcorr(msum((high + low) / 2, 19), msum(mavg(volume, 40), 19), 13), percent=true), 5), percent=true)")
        self._add_ddb(57, "ewmMean((close - mmin(low, 9)) / (mmax(high, 9) - mmin(low, 9)) * 100, alpha=1/3)")
        self._add_ddb(58, "msum(iif(close > mfirst(close, 2), 1, 0), 20) / 20 * 100")
        self._add_ddb(59, "msum(iif(close == mfirst(close, 2), 0, close - iif(close > mfirst(close, 2), iif(low < mfirst(close, 2), low, mfirst(close, 2)), iif(high > mfirst(close, 2), high, mfirst(close, 2)))), 20)")
        self._add_ddb(60, "msum((close - low - (high - close)) / (high - low) * volume, 20)")

        self._add_ddb(61, "ddb_max(rowRank(ts_decay_linear(vwap - mfirst(vwap, 2), 12), percent=true), rowRank(ts_decay_linear(rowRank(mcorr(low, mavg(volume, 80), 8), percent=true), 17), percent=true)) * -1")
        self._add_ddb(62, "-1 * mcorr(high, rowRank(volume, percent=true), 5)")
        self._add_ddb(63, "ewmMean(iif(close - move(close, 1) > 0, close - move(close, 1), 0), alpha=1/6) / ewmMean(abs(close - move(close, 1)), alpha=1/6) * 100")
        self._add_ddb(64, "ddb_max(rowRank(ts_decay_linear(mcorr(rowRank(vwap, percent=true), rowRank(volume, percent=true), 4), 4), percent=true), rowRank(ts_decay_linear(mmax(mcorr(rowRank(close, percent=true), rowRank(mavg(volume, 60), percent=true), 4), 13), 14), percent=true)) * -1")
        self._add_ddb(65, "mavg(close, 6) / close")
        self._add_ddb(66, "(close - mavg(close, 6)) / mavg(close, 6) * 100")
        self._add_ddb(67, "ewmMean(iif(close - move(close, 1) > 0, close - move(close, 1), 0), alpha=1/24) / ewmMean(abs(close - move(close, 1)), alpha=1/24) * 100")
        self._add_ddb(68, "ewmMean(((high + low) / 2 - (move(high, 1) + move(low, 1)) / 2) * (high - low) / volume, alpha=2/15)")
        self._add_ddb(69, "iif(msum(iif(open <= mfirst(open, 2), 0, iif(high - open > open - mfirst(open, 2), high - open, open - mfirst(open, 2))), 20) > msum(iif(open >= mfirst(open, 2), 0, iif(open - low > open - mfirst(open, 2), open - low, open - mfirst(open, 2))), 20), (msum(iif(open <= mfirst(open, 2), 0, iif(high - open > open - mfirst(open, 2), high - open, open - mfirst(open, 2))), 20) - msum(iif(open >= mfirst(open, 2), 0, iif(open - low > open - mfirst(open, 2), open - low, open - mfirst(open, 2))), 20)) / msum(iif(open <= mfirst(open, 2), 0, iif(high - open > open - mfirst(open, 2), high - open, open - mfirst(open, 2))), 20), iif(msum(iif(open <= mfirst(open, 2), 0, iif(high - open > open - mfirst(open, 2), high - open, open - mfirst(open, 2))), 20) == msum(iif(open >= mfirst(open, 2), 0, iif(open - low > open - mfirst(open, 2), open - low, open - mfirst(open, 2))), 20), 0, (msum(iif(open <= mfirst(open, 2), 0, iif(high - open > open - mfirst(open, 2), high - open, open - mfirst(open, 2))), 20) - msum(iif(open >= mfirst(open, 2), 0, iif(open - low > open - mfirst(open, 2), open - low, open - mfirst(open, 2))), 20)) / msum(iif(open >= mfirst(open, 2), 0, iif(open - low > open - mfirst(open, 2), open - low, open - mfirst(open, 2))), 20)))")
        self._add_ddb(70, f"mstd({amount}, 6)")
        self._add_ddb(71, "(close - mavg(close, 24)) / mavg(close, 24) * 100")
        self._add_ddb(72, "ewmMean((mmax(high, 6) - close) / (mmax(high, 6) - mmin(low, 6)) * 100, alpha=1/15)")
        self._add_ddb(73, "(mrank(ts_decay_linear(ts_decay_linear(mcorr(close, volume, 10), 16), 4), true, 5) - rowRank(ts_decay_linear(mcorr(vwap, mavg(volume, 30), 4), 3), percent=true)) * -1")
        self._add_ddb(74, "rowRank(mcorr(msum(low * 0.35 + vwap * 0.65, 20), msum(mavg(volume, 40), 20), 7), percent=true) + rowRank(mcorr(rowRank(vwap, percent=true), rowRank(volume, percent=true), 6), percent=true)")
        self._add_ddb(75, "msum(iif((close > open) & (index_close < index_open), 1, 0), 50) / msum(iif(index_close < index_open, 1, 0), 50)")
        self._add_ddb(76, "mstd(abs(close / mfirst(close, 2) - 1) / volume, 20) / mavg(abs(close / mfirst(close, 2) - 1) / volume, 20)")
        self._add_ddb(77, "ddb_min(rowRank(ts_decay_linear((high + low) / 2 + high - (vwap + high), 20), percent=true), rowRank(ts_decay_linear(mcorr((high + low) / 2, mavg(volume, 40), 3), 6), percent=true))")
        self._add_ddb(78, f"(({hlc3}) - mavg({hlc3}, 12)) / (0.015 * mavg(abs(close - mavg({hlc3}, 12)), 12))")
        self._add_ddb(79, "ewmMean(iif(close - move(close, 1) > 0, close - move(close, 1), 0), alpha=1/12) / ewmMean(abs(close - move(close, 1)), alpha=1/12) * 100")
        self._add_ddb(80, "(volume - move(volume, 5)) / move(volume, 5) * 100")

    def register_081_191(self) -> None:
        amount = "volume * vwap"
        hlc3 = "(high + low + close) / 3"
        ret = "ratios(close) - 1"
        up12 = "ewmMean(iif(close - move(close, 1) > 0, close - move(close, 1), 0), alpha=1/12)"
        abs12 = "ewmMean(abs(close - move(close, 1)), alpha=1/12)"
        rsi12 = f"({up12}) / ({abs12}) * 100"
        tr = "ddb_max(ddb_max(high - low, abs(high - mfirst(close, 2))), abs(low - mfirst(close, 2)))"
        hd = "high - mfirst(high, 2)"
        ld = "mfirst(low, 2) - low"
        dmi = f"mavg(abs(msum(iif(({ld} > 0) & ({ld} > {hd}), {ld}, 0), 14) * 100 / msum({tr}, 14) - msum(iif(({hd} > 0) & ({hd} > {ld}), {hd}, 0), 14) * 100 / msum({tr}, 14)) / (msum(iif(({ld} > 0) & ({ld} > {hd}), {ld}, 0), 14) * 100 / msum({tr}, 14) + msum(iif(({hd} > 0) & ({hd} > {ld}), {hd}, 0), 14) * 100 / msum({tr}, 14)) * 100, 6)"

        self._add_ddb(81, "ewmMean(volume, alpha=1/21)")
        self._add_ddb(82, "ewmMean((mmax(high, 6) - close) / (mmax(high, 6) - mmin(low, 6)) * 100, alpha=1/20)")
        self._add_ddb(83, "-1 * rowRank(mcovar(rowRank(high, percent=true), rowRank(volume, percent=true), 5), percent=true)")
        self._add_ddb(84, "msum(iif(close > mfirst(close, 2), volume, iif(close < mfirst(close, 2), -volume, 0)), 20)")
        self._add_ddb(85, "mrank(volume / mavg(volume, 20), true, 20) * mrank(-1 * (close - mfirst(close, 8)), true, 8)")
        self._add_ddb(86, "iif(0.25 < ((mfirst(close, 21) - mfirst(close, 11)) / 10 - (mfirst(close, 11) - close) / 10), -1, iif(((mfirst(close, 21) - mfirst(close, 11)) / 10 - (mfirst(close, 11) - close) / 10) < 0, 1, -1 * (close - mfirst(close, 2))))")
        self._add_ddb(87, "(rowRank(ts_decay_linear(vwap - mfirst(vwap, 5), 7), percent=true) + mrank(ts_decay_linear((low - vwap) / (open - (high + low) / 2), 11), true, 7)) * -1")
        self._add_ddb(88, "(close - mfirst(close, 21)) / mfirst(close, 21) * 100")
        self._add_ddb(89, "2 * (ewmMean(close, alpha=2/13) - ewmMean(ewmMean(close, alpha=2/13), alpha=2/27) - ewmMean(ewmMean(close, alpha=2/13) - ewmMean(ewmMean(close, alpha=2/13), alpha=2/27), alpha=2/10))")
        self._add_ddb(90, "rowRank(mcorr(rowRank(vwap, percent=true), rowRank(volume, percent=true), 5), percent=true) * -1")
        self._add_ddb(91, "rowRank(close - mmax(close, 5), percent=true) * rowRank(mcorr(mavg(volume, 40), low, 5), percent=true) * -1")
        self._add_ddb(92, "ddb_max(rowRank(ts_decay_linear(close * 0.35 + vwap * 0.65 - mfirst(close * 0.35 + vwap * 0.65, 3), 3), percent=true), mrank(ts_decay_linear(abs(mcorr(mavg(volume, 180), close, 13)), 5), true, 15)) * -1")
        self._add_ddb(93, "msum(iif(open >= mfirst(open, 2), 0, ddb_max(open - low, open - mfirst(open, 2))), 20)")
        self._add_ddb(94, "msum(iif(close > mfirst(close, 2), volume, iif(close < mfirst(close, 2), -volume, 0)), 30)")
        self._add_ddb(95, f"mstd({amount}, 20)")
        self._add_ddb(96, "ewmMean(ewmMean((close - mmin(low, 9)) / (mmax(high, 9) - mmin(low, 9)) * 100, alpha=1/3), alpha=1/3)")
        self._add_ddb(97, "mstd(volume, 10)")
        self._add_ddb(98, "iif(((msum(close, 100) / 100 - mfirst(msum(close, 100) / 100, 101)) / mfirst(close, 101) < 0.05) | ((msum(close, 100) / 100 - mfirst(msum(close, 100) / 100, 101)) / mfirst(close, 101) == 0.05), -1 * (close - mmin(close, 100)), -1 * (close - mfirst(close, 4)))")
        self._add_ddb(99, "-1 * rowRank(mcovar(rowRank(close, percent=true), rowRank(volume, percent=true), 5), percent=true)")
        self._add_ddb(100, "mstd(volume, 20)")
        self._add_ddb(101, "(rowRank(mcorr(close, msum(mavg(volume, 30), 37), 15), percent=true) < rowRank(mcorr(rowRank(high * 0.1 + vwap * 0.9, percent=true), rowRank(volume, percent=true), 11), percent=true)) * -1")
        self._add_ddb(102, "ewmMean(iif(volume - move(volume, 1) > 0, volume - move(volume, 1), 0), alpha=1/6) / ewmMean(abs(volume - move(volume, 1)), alpha=1/6) * 100")
        self._add_ddb(103, "(20 - (19 - mimin(low, 20))) / 20 * 100")
        self._add_ddb(104, "-1 * (mcorr(high, volume, 5) - mfirst(mcorr(high, volume, 5), 6)) * rowRank(mstd(close, 20), percent=true)")
        self._add_ddb(105, "-1 * mcorr(rowRank(open, percent=true), rowRank(volume, percent=true), 10)")
        self._add_ddb(106, "close - mfirst(close, 21)")
        self._add_ddb(107, "-1 * rowRank(open - mfirst(high, 2), percent=true) * rowRank(open - mfirst(close, 2), percent=true) * rowRank(open - mfirst(low, 2), percent=true)")
        self._add_ddb(108, "pow(rowRank(high - mmin(high, 2), percent=true), rowRank(mcorr(vwap, mavg(volume, 120), 6), percent=true)) * -1")
        self._add_ddb(109, "ewmMean(high - low, alpha=2/10) / ewmMean(ewmMean(high - low, alpha=2/10), alpha=2/10)")
        self._add_ddb(110, "msum(iif(high - mfirst(close, 2) > 0, high - mfirst(close, 2), 0), 20) / msum(iif(mfirst(close, 2) - low > 0, mfirst(close, 2) - low, 0), 20) * 100")
        self._add_ddb(111, "ewmMean(volume * ((close - low) - (high - close)) / (high - low), alpha=2/11) - ewmMean(volume * ((close - low) - (high - close)) / (high - low), alpha=2/4)")
        self._add_ddb(112, "((msum(iif(close - mfirst(close, 2) > 0, close - mfirst(close, 2), 0), 12)) - (msum(iif(close - mfirst(close, 2) < 0, abs(close - mfirst(close, 2)), 0), 12))) / ((msum(iif(close - mfirst(close, 2) > 0, close - mfirst(close, 2), 0), 12)) + (msum(iif(close - mfirst(close, 2) < 0, abs(close - mfirst(close, 2)), 0), 12))) * 100")
        self._add_ddb(113, "-1 * rowRank(msum(mfirst(close, 6), 20) / 20, percent=true) * mcorr(close, volume, 2) * rowRank(mcorr(msum(close, 5), msum(close, 20), 2), percent=true)")
        self._add_ddb(114, "rowRank(mfirst((high - low) / (msum(close, 5) / 5), 3), percent=true) * rowRank(rowRank(volume, percent=true), percent=true) / (((high - low) / (msum(close, 5) / 5)) / (vwap - close))")
        self._add_ddb(115, "pow(rowRank(mcorr(high * 0.9 + close * 0.1, mavg(volume, 30), 10), percent=true), rowRank(mcorr(mrank((high + low) / 2, true, 4), mrank(volume, true, 10), 7), percent=true))")
        self._add_ddb(116, "linearTimeTrend(close, 20)")
        self._add_ddb(117, f"mrank(volume, true, 32) * (1 - mrank(close + high - low, true, 16)) * (1 - mrank({ret}, true, 32))")
        self._add_ddb(118, "msum(high - open, 20) / msum(open - low, 20) * 100")
        self._add_ddb(119, "rowRank(ts_decay_linear(mcorr(vwap, msum(mavg(volume, 5), 26), 5), 7), percent=true) - rowRank(ts_decay_linear(mrank(mmin(mcorr(rowRank(open, percent=true), rowRank(mavg(volume, 15), percent=true), 21), 9), true, 7), 8), percent=true)")
        self._add_ddb(120, "rowRank(vwap - close, percent=true) / rowRank(vwap + close, percent=true)")
        self._add_ddb(121, "pow(rowRank(vwap - mmin(vwap, 12), percent=true), mrank(mcorr(mrank(vwap, true, 20), mrank(mavg(volume, 60), true, 2), 18), true, 3)) * -1")
        self._add_ddb(122, "(ewmMean(ewmMean(ewmMean(log(close), alpha=2/13), alpha=2/13), alpha=2/13) - move(ewmMean(ewmMean(ewmMean(log(close), alpha=2/13), alpha=2/13), alpha=2/13), 1)) / move(ewmMean(ewmMean(ewmMean(log(close), alpha=2/13), alpha=2/13), alpha=2/13), 1)")
        self._add_ddb(123, "(rowRank(mcorr(msum((high + low) / 2, 20), msum(mavg(volume, 60), 20), 9), percent=true) < rowRank(mcorr(low, volume, 6), percent=true)) * -1")
        self._add_ddb(124, "(close - vwap) / ts_decay_linear(rowRank(mmax(close, 30), percent=true), 2)")
        self._add_ddb(125, "rowRank(ts_decay_linear(mcorr(vwap, mavg(volume, 80), 17), 20), percent=true) / rowRank(ts_decay_linear(close * 0.5 + vwap * 0.5 - mfirst(close * 0.5 + vwap * 0.5, 4), 16), percent=true)")
        self._add_ddb(126, hlc3)
        self._add_ddb(127, "pow(mavg(pow(100 * (close - mmax(close, 12)) / mmax(close, 12), 2), 12), 0.5)")
        self._add_ddb(128, f"100 - (100 / (1 + msum(iif(({hlc3}) > mfirst({hlc3}, 2), ({hlc3}) * volume, 0), 14) / msum(iif(({hlc3}) < mfirst({hlc3}, 2), ({hlc3}) * volume, 0), 14)))")
        self._add_ddb(129, "msum(iif(close - move(close, 1) < 0, abs(close - move(close, 1)), 0), 12)")
        self._add_ddb(130, "rowRank(ts_decay_linear(mcorr((high + low) / 2, mavg(volume, 40), 9), 10), percent=true) / rowRank(ts_decay_linear(mcorr(rowRank(vwap, percent=true), rowRank(volume, percent=true), 7), 3), percent=true)")
        self._add_ddb(131, "pow(rowRank(vwap - mfirst(vwap, 2), percent=true), mrank(mcorr(close, mavg(volume, 50), 18), true, 18))")
        self._add_ddb(132, f"mavg({amount}, 20)")
        self._add_ddb(133, "(20 - (19 - mimax(high, 20))) / 20 * 100 - (20 - (19 - mimin(low, 20))) / 20 * 100")
        self._add_ddb(134, "(close - mfirst(close, 13)) / mfirst(close, 13) * volume")
        self._add_ddb(135, "ewmMean(move(close / move(close, 20), 1), alpha=1/20)")
        self._add_ddb(136, f"-1 * rowRank({ret} - mfirst({ret}, 4), percent=true) * mcorr(open, volume, 10)")
        self._add_ddb(137, "16 * (close - mfirst(close, 2) + (close - open) / 2 + mfirst(close, 2) - mfirst(open, 2)) / iif((abs(high - mfirst(close, 2)) > abs(low - mfirst(close, 2))) & (abs(high - mfirst(close, 2)) > abs(high - mfirst(low, 2))), abs(high - mfirst(close, 2)) + abs(low - mfirst(close, 2)) / 2 + abs(mfirst(close, 2) - mfirst(open, 2)) / 4, iif((abs(low - mfirst(close, 2)) > abs(high - mfirst(low, 2))) & (abs(low - mfirst(close, 2)) > abs(high - mfirst(close, 2))), abs(low - mfirst(close, 2)) + abs(high - mfirst(close, 2)) / 2 + abs(mfirst(close, 2) - mfirst(open, 2)) / 4, abs(high - mfirst(low, 2)) + abs(mfirst(close, 2) - mfirst(open, 2)) / 4)) * ddb_max(abs(high - mfirst(close, 2)), abs(low - mfirst(close, 2)))")
        self._add_ddb(138, "(rowRank(ts_decay_linear(low * 0.7 + vwap * 0.3 - mfirst(low * 0.7 + vwap * 0.3, 4), 20), percent=true) - mrank(ts_decay_linear(mrank(mcorr(mrank(low, true, 8), mrank(mavg(volume, 60), true, 17), 5), true, 19), 16), true, 7)) * -1")
        self._add_ddb(139, "-1 * mcorr(open, volume, 10)")
        self._add_ddb(140, "ddb_min(rowRank(ts_decay_linear(rowRank(open, percent=true) + rowRank(low, percent=true) - (rowRank(high, percent=true) + rowRank(close, percent=true)), 8), percent=true), mrank(ts_decay_linear(mcorr(mrank(close, true, 8), mrank(mavg(volume, 60), true, 20), 8), 7), true, 3))")
        self._add_ddb(141, "rowRank(mcorr(rowRank(high, percent=true), rowRank(mavg(volume, 15), percent=true), 9), percent=true) * -1")
        self._add_ddb(142, "-1 * rowRank(mrank(close, true, 10), percent=true) * rowRank(close - mfirst(close, 2) - mfirst(close - mfirst(close, 2), 2), percent=true) * rowRank(mrank(volume / mavg(volume, 20), true, 5), percent=true)")
        self._add_ddb(143, "conditionalCumprod(close / move(close, 1) > 1, close / move(close, 1) - 1, 1)")
        self._add_ddb(144, f"msum(iif(close < mfirst(close, 2), abs(close / mfirst(close, 2) - 1) / ({amount}), 0), 20) / msum(iif(close < mfirst(close, 2), 1, 0), 20)")
        self._add_ddb(145, "(mavg(volume, 9) - mavg(volume, 26)) / mavg(volume, 12) * 100")
        self._add_ddb(146, "mavg((close - move(close, 1)) / move(close, 1) - ewmMean((close - move(close, 1)) / move(close, 1), alpha=2/61), 20) * ((close - move(close, 1)) / move(close, 1) - ewmMean((close - move(close, 1)) / move(close, 1), alpha=2/61)) / mavg(pow(ewmMean((close - move(close, 1)) / move(close, 1), alpha=2/61), 2), 60)")
        self._add_ddb(147, "linearTimeTrend(mavg(close, 12), 12)")
        self._add_ddb(148, "(rowRank(mcorr(open, msum(mavg(volume, 60), 9), 6), percent=true) < rowRank(open - mmin(open, 14), percent=true)) * -1")
        self._add_ddb(149, "mbeta(iif(index_close < move(index_close, 1), close / move(close, 1) - 1, NULL), iif(index_close < move(index_close, 1), index_close / move(index_close, 1) - 1, NULL), 252)")
        self._add_ddb(150, f"({hlc3}) * volume")
        self._add_ddb(151, "ewmMean(close - move(close, 20), alpha=1/20)")
        self._add_ddb(152, "ewmMean(mavg(move(ewmMean(move(close / move(close, 9), 1), alpha=1/9), 1), 12) - mavg(move(ewmMean(move(close / move(close, 9), 1), alpha=1/9), 1), 26), alpha=1/9)")
        self._add_ddb(153, "(mavg(close, 3) + mavg(close, 6) + mavg(close, 12) + mavg(close, 24)) / 4")
        self._add_ddb(154, "(vwap - mmin(vwap, 16)) < mcorr(vwap, mavg(volume, 180), 18)")
        self._add_ddb(155, "ewmMean(volume, alpha=2/13) - ewmMean(ewmMean(volume, alpha=2/13), alpha=2/27) - ewmMean(ewmMean(volume, alpha=2/13) - ewmMean(ewmMean(volume, alpha=2/13), alpha=2/27), alpha=2/10)")
        self._add_ddb(156, "ddb_max(rowRank(ts_decay_linear(vwap - mfirst(vwap, 6), 3), percent=true), rowRank(ts_decay_linear((open * 0.15 + low * 0.85 - mfirst(open * 0.15 + low * 0.85, 3)) / (open * 0.15 + low * 0.85) * -1, 3), percent=true)) * -1")
        self._add_ddb(157, f"ddb_min(rowRank(rowRank(log(mmin(rowRank(rowRank(-1 * rowRank(close - 1 - mfirst(close - 1, 6), percent=true), percent=true), percent=true), 2)), percent=true), percent=true), 5) + mrank(mfirst(-1 * ({ret}), 7), true, 5)")
        self._add_ddb(158, "((high - ewmMean(close, alpha=2/15)) - (low - ewmMean(close, alpha=2/15))) / close")
        self._add_ddb(159, "((close - msum(ddb_min(low, mfirst(close, 2)), 6)) / msum(ddb_max(high, mfirst(close, 2)) - ddb_min(low, mfirst(close, 2)), 6) * 12 * 24 + (close - msum(ddb_min(low, mfirst(close, 2)), 12)) / msum(ddb_max(high, mfirst(close, 2)) - ddb_min(low, mfirst(close, 2)), 12) * 6 * 24 + (close - msum(ddb_min(low, mfirst(close, 2)), 24)) / msum(ddb_max(high, mfirst(close, 2)) - ddb_min(low, mfirst(close, 2)), 24) * 6 * 24) * 100 / (6 * 12 + 6 * 24 + 12 * 24)")
        self._add_ddb(160, "ewmMean(iif(close <= move(close, 1), mstd(close, 20), 0), alpha=1/20)")
        self._add_ddb(161, "mavg(ddb_max(ddb_max(high - low, abs(mfirst(close, 2) - high)), abs(mfirst(close, 2) - low)), 12)")
        self._add_ddb(162, f"(({rsi12}) - mmin({rsi12}, 12)) / (mmax({rsi12}, 12) - mmin({rsi12}, 12))")
        self._add_ddb(163, f"rowRank(-1 * ({ret}) * mavg(volume, 20) * vwap * (high - close), percent=true)")
        self._add_ddb(164, "ewmMean((iif(close > move(close, 1), 1 / (close - move(close, 1)), 1) - mmin(iif(close > move(close, 1), 1 / (close - move(close, 1)), 1), 12)) / (high - low) * 100, alpha=2/13)")
        self._add_ddb(165, "rowMax(msum(close - mavg(close, 48), 48)) - rowMin(msum(close - mavg(close, 48), 48)) / mstd(close, 48)")
        self._add_ddb(166, "(-20 * pow(20 - 1, 1.5) * msum(close / mfirst(close, 2) - 1 - mavg(close / mfirst(close, 2) - 1, 20), 20)) / ((20 - 1) * (20 - 2) * pow(msum(pow(mavg(close / mfirst(close, 2), 20), 2), 20), 1.5))")
        self._add_ddb(167, "msum(iif(close - mfirst(close, 2) > 0, close - mfirst(close, 2), 0), 12)")
        self._add_ddb(168, "-1 * volume / mavg(volume, 20)")
        self._add_ddb(169, "ewmMean(mavg(move(ewmMean(close - move(close, 1), alpha=1/9), 1), 12) - mavg(move(ewmMean(close - move(close, 1), alpha=1/9), 1), 26), alpha=1/10)")
        self._add_ddb(170, "rowRank(1 / close, percent=true) * volume / mavg(volume, 20) * high * rowRank(high - close, percent=true) / (msum(high, 5) / 5) - rowRank(vwap - mfirst(vwap, 6), percent=true)")
        self._add_ddb(171, "-1 * (low - close) * pow(open, 5) / ((close - high) * pow(close, 5))")
        self._add_ddb(172, dmi)
        self._add_ddb(173, "3 * ewmMean(close, alpha=2/13) - 2 * ewmMean(ewmMean(close, alpha=2/13), alpha=2/13) + ewmMean(ewmMean(ewmMean(close, alpha=2/13), alpha=2/13), alpha=2/13)")
        self._add_ddb(174, "ewmMean(iif(close > move(close, 1), mstd(close, 20), 0), alpha=1/20)")
        self._add_ddb(175, "mavg(ddb_max(ddb_max(high - low, abs(mfirst(close, 2) - high)), abs(mfirst(close, 2) - low)), 6)")
        self._add_ddb(176, "mcorr(rowRank((close - mmin(low, 12)) / (mmax(high, 12) - mmin(low, 12)), percent=true), rowRank(volume, percent=true), 6)")
        self._add_ddb(177, "(20 - (19 - mimax(high, 20))) / 20 * 100")
        self._add_ddb(178, "(close - mfirst(close, 2)) / mfirst(close, 2) * volume")
        self._add_ddb(179, "rowRank(mcorr(vwap, volume, 4), percent=true) * rowRank(mcorr(rowRank(low, percent=true), rowRank(mavg(volume, 50), percent=true), 12), percent=true)")
        self._add_ddb(180, "iif(mavg(volume, 20) < volume, -1 * mrank(abs(close - mfirst(close, 8)), true, 60) * sign(close - mfirst(close, 8)), -1 * volume)")
        self._add_ddb(181, "msum(((close / move(close, 1) - 1) - mavg(close / move(close, 1) - 1, 20)) - pow(index_close - mavg(index_close, 20), 2), 20) / msum(pow(index_close - mavg(index_close, 20), 3), 20)")
        self._add_ddb(182, "mcount(iif(((close > open) & (index_close > index_open)) | ((close < open) & (index_close < index_open)), 1, NULL), 20) / 20")
        self._add_ddb(183, "rowMax(msum(close - mavg(close, 24), 24)) - rowMin(msum(close - mavg(close, 24), 24)) / mstd(close, 24)")
        self._add_ddb(184, "rowRank(mcorr(mfirst(open - close, 2), close, 200), percent=true) + rowRank(open - close, percent=true)")
        self._add_ddb(185, "rowRank(-1 * pow(1 - open / close, 2), percent=true)")
        self._add_ddb(186, f"({dmi} + mfirst({dmi}, 7)) / 2")
        self._add_ddb(187, "msum(iif(open <= mfirst(open, 2), 0, ddb_max(high - open, open - mfirst(open, 2))), 20)")
        self._add_ddb(188, "(high - low - ewmMean(high - low, alpha=2/11)) / ewmMean(high - low, alpha=2/11) * 100")
        self._add_ddb(189, "mavg(abs(close - mavg(close, 6)), 6)")
        self._add_ddb(190, "log(((msum(iif(close / move(close, 1) - 1 > pow(close / move(close, 19), 1/20) - 1, 1, 0), 20) - 1) * msum(pow(close / move(close, 1) - 1 - (pow(close / move(close, 19), 1/20) - 1), 2) * iif(close / move(close, 1) - 1 < pow(close / move(close, 19), 1/20) - 1, 1, 0), 20)) / (msum(iif(close / move(close, 1) - 1 < pow(close / move(close, 19), 1/20) - 1, 1, 0), 20) * msum(pow(close / move(close, 1) - 1 - (pow(close / move(close, 19), 1/20) - 1), 2) * iif(close / move(close, 1) - 1 > pow(close / move(close, 19), 1/20) - 1, 1, 0), 20)))")
        self._add_ddb(191, "mcorr(mavg(volume, 20), low, 5) + (high + low) / 2 - close")

    def register_amount_extras(self) -> None:
        amount = "volume * vwap"

        self._add_ddb(95, f"mstd({amount}, 20)")
        self._add_ddb(132, f"mavg({amount}, 20)")

    def register_all(self) -> None:
        self.register_001_080()
        self.register_081_191()
        self.register_amount_extras()
