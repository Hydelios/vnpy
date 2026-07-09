from __future__ import annotations

from typing import Iterable

import polars as pl

from vnpy.alpha.dataset import AlphaDataset


class RQTechFactor(AlphaDataset):
    """Register Ricequant technical factor expressions.

    The formulas are translated from rq_tech.md.  Every feature expression is
    self-contained because AlphaDataset evaluates registered expressions
    independently against the source dataframe.
    """

    MA_WINDOWS: tuple[int, ...] = (3, 5, 10, 20, 30, 55, 60, 120, 250)
    WMA_WINDOWS: tuple[int, ...] = (3, 5, 10, 20, 60, 120, 250)
    AMP_WINDOWS: tuple[int, ...] = (1, 3, 5, 10, 20, 60)
    EPS = "1e-12"

    def __init__(
        self,
        df: pl.DataFrame,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
        interval: str = "1d",
        process_type: str = "append",
        enable_cache: bool = False,
        cache_dir: str | None = None,
        select: Iterable[str] | None = None,
        skip: Iterable[str] | None = None,
        amount_col: str = "turnover",
        capital_col: str = "capital",
    ) -> None:
        super().__init__(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
            interval=interval,
            process_type=process_type,
            enable_cache=enable_cache,
            cache_dir=cache_dir,
        )
        self.select = set(select) if select is not None else None
        self.skip = set(skip or [])
        self.amount_col = amount_col
        self.capital_col = capital_col

    # -------------------- expression helpers --------------------

    def _enabled(self, name: str) -> bool:
        if name in self.skip:
            return False
        return self.select is None or name in self.select

    def _add(self, name: str, expr: str) -> None:
        if self._enabled(name):
            self.add_feature(name, expr)

    @classmethod
    def _delay(cls, expr: str, n: int) -> str:
        return f"ts_delay(({expr}), {n})"

    @classmethod
    def _ma(cls, expr: str, n: int) -> str:
        return f"ts_mean(({expr}), {n})"

    @classmethod
    def _ema(cls, expr: str, n: int) -> str:
        return f"ts_ema(({expr}), {n})"

    @classmethod
    def _std(cls, expr: str, n: int) -> str:
        return f"ts_std(({expr}), {n})"

    @classmethod
    def _sum(cls, expr: str, n: int) -> str:
        return f"ts_sum(({expr}), {n})"

    @classmethod
    def _hhv(cls, expr: str, n: int) -> str:
        return f"ts_max(({expr}), {n})"

    @classmethod
    def _llv(cls, expr: str, n: int) -> str:
        return f"ts_min(({expr}), {n})"

    @classmethod
    def _abs(cls, expr: str) -> str:
        return f"ts_abs(({expr}))"

    @classmethod
    def _max(cls, left: str, right: str) -> str:
        return f"ts_greater(({left}), ({right}))"

    @classmethod
    def _min(cls, left: str, right: str) -> str:
        return f"ts_less(({left}), ({right}))"

    @classmethod
    def _if(cls, cond: str, true_expr: str, false_expr: str) -> str:
        return f"ts_where(({cond}), ({true_expr}), ({false_expr}))"

    @classmethod
    def _div(cls, numerator: str, denominator: str) -> str:
        return f"(({numerator}) / (({denominator}) + {cls.EPS}))"

    @property
    def amount(self) -> str:
        return self.amount_col

    @property
    def capital(self) -> str:
        return self.capital_col

    @property
    def hsl(self) -> str:
        return self._div("100 * volume", self.capital)

    @property
    def mid(self) -> str:
        return "((3 * close + low + open + high) / 6)"

    def _true_range(self) -> str:
        lc = self._delay("close", 1)
        return self._max(self._max("high - low", self._abs(f"high - {lc}")), self._abs(f"low - {lc}"))

    # -------------------- register groups --------------------

    def register_all(self) -> None:
        self.register_ma_indicators()
        self.register_overbought_oversold_indicators()
        self.register_energy_indicators()

    def register_ma_indicators(self) -> None:
        self._register_macd()
        self._register_trix()
        self._register_boll()
        self._register_asi()
        self._register_price_ma_family()
        self._register_turnover_ma_family()
        self._register_bbiboll()
        self._register_dpo()
        self._register_mcst()

    def register_overbought_oversold_indicators(self) -> None:
        self._register_obos()
        self._register_kdj()
        self._register_rsi()
        self._register_wr_lwr()
        self._register_bias()
        self._register_accer_cyf_fsl()
        self._register_adtm()
        self._register_atr()
        self._register_dkx()
        self._register_tapi()
        self._register_osc_cci_roc_mfi()
        self._register_mtm_marsi_skd_udl()
        self._register_dmi()

    def register_energy_indicators(self) -> None:
        self._register_arbr()
        self._register_vr()
        self._register_cr()
        self._register_mass()
        self._register_sy_pcnt_cyr()
        self._register_amp_wma_volatility()
        self._register_aroon_qtyr_obv()

    # -------------------- moving average indicators --------------------

    def _register_macd(self) -> None:
        diff = f"({self._ema('close', 12)} - {self._ema('close', 26)})"
        dea = self._ema(diff, 9)
        self._add("MACD_DIFF", diff)
        self._add("MACD_DEA", dea)
        self._add("MACD_HIST", f"(({diff}) - ({dea})) * 2")

    def _register_trix(self) -> None:
        tr = self._ema(self._ema(self._ema("close", 12), 12), 12)
        trix = self._div(f"({tr}) - {self._delay(tr, 1)}", self._delay(tr, 1)) + " * 100"
        self._add("TRIX", trix)
        self._add("MATRIX", self._ma(trix, 20))

    def _register_boll(self) -> None:
        boll = self._ma("close", 20)
        width = f"{self._std('close', 20)} * 2"
        self._add("BOLL", boll)
        self._add("BOLL_UP", f"({boll}) + ({width})")
        self._add("BOLL_DOWN", f"({boll}) - ({width})")

    def _register_asi(self) -> None:
        lc = self._delay("close", 1)
        lo1 = self._delay("low", 1)
        op1 = self._delay("open", 1)
        aa = self._abs(f"high - {lc}")
        bb = self._abs(f"low - {lc}")
        cc = self._abs(f"high - {lo1}")
        dd = self._abs(f"{lc} - {op1}")
        r = self._if(
            f"({aa} > {bb}) & ({aa} > {cc})",
            f"{aa} + {bb} / 2 + {dd} / 4",
            self._if(
                f"({bb} > {cc}) & ({bb} > {aa})",
                f"{bb} + {aa} / 2 + {dd} / 4",
                f"{cc} + {dd} / 4",
            ),
        )
        x = f"(close - {lc} + (close - open) / 2 + {lc} - {op1})"
        si = f"{self._div(f'({x}) * 16 * {self._max(aa, bb)}', r)}"
        asi = self._sum(si, 26)
        self._add("ASI", asi)
        self._add("ASIT", self._ma(asi, 10))

    def _register_price_ma_family(self) -> None:
        vv = "((high + open + low + close) / 4)"
        amov = "(volume * (open + close) / 2)"
        for n in self.MA_WINDOWS:
            self._add(f"MA{n}", self._ma("close", n))
            self._add(f"EMA{n}", self._ema("close", n))
            self._add(f"HMA{n}", self._ma("high", n))
            self._add(f"LMA{n}", self._ma("low", n))
            self._add(f"VMA{n}", self._ma(vv, n))
            self._add(f"AMV{n}", self._div(self._sum(amov, n), self._sum("volume", n)))

    def _register_turnover_ma_family(self) -> None:
        hsl = self.hsl
        vol120 = self._ma(hsl, 120)
        for n in self.MA_WINDOWS:
            self._add(f"VOL{n}", self._ma(hsl, n))
        for n in (5, 10, 20):
            self._add(f"DAVOL{n}", self._div(self._ma(hsl, n), vol120))

    def _register_bbiboll(self) -> None:
        bbi = (
            f"({self._ma('close', 3)} + {self._ma('close', 6)} + "
            f"{self._ma('close', 12)} + {self._ma('close', 24)}) / 4"
        )
        width = f"6 * {self._std(bbi, 11)}"
        self._add("BBI", bbi)
        self._add("BBIBOLL_UP", f"({bbi}) + ({width})")
        self._add("BBIBOLL_DOWN", f"({bbi}) - ({width})")

    def _register_dpo(self) -> None:
        dpo = f"(close - {self._delay(self._ma('close', 20), 10)})"
        self._add("DPO", dpo)
        self._add("MADPO", self._ma(dpo, 6))

    def _register_mcst(self) -> None:
        avg_price = self._div(self.amount, "volume")
        self._add("MCST", f"rq_dma(({avg_price}), ({self.hsl}))")

    # -------------------- overbought/oversold indicators --------------------

    def _register_obos(self) -> None:
        lc = self._delay("close", 1)
        up_count = f"cs_sum(close > {lc})"
        down_count = f"cs_sum(close < {lc})"
        self._add("OBOS", self._sum(f"({up_count}) - ({down_count})", 10))

    def _register_kdj(self) -> None:
        lowv = self._llv("low", 9)
        highv = self._hhv("high", 9)
        rsv = f"{self._div(f'close - {lowv}', f'{highv} - {lowv}')} * 100"
        k = self._ema(rsv, 5)
        d = self._ema(k, 5)
        self._add("KDJ_K", k)
        self._add("KDJ_D", d)
        self._add("KDJ_J", f"({k}) * 3 - ({d}) * 2")

    def _register_rsi(self) -> None:
        lc = self._delay("close", 1)
        delta = f"close - {lc}"
        for n in (6, 10):
            rsi = self._div(self._ma(self._max(delta, "0"), n), self._ma(self._abs(delta), n)) + " * 100"
            self._add(f"RSI{n}", rsi)

    def _register_wr_lwr(self) -> None:
        hhv = self._hhv("high", 10)
        llv = self._llv("low", 10)
        self._add("WR", self._div(f"{hhv} - close", f"{hhv} - {llv}") + " * 100")

        hhv9 = self._hhv("high", 9)
        llv9 = self._llv("low", 9)
        rsv = self._div(f"{hhv9} - close", f"{hhv9} - {llv9}") + " * 100"
        lwr1 = f"ta_sma(({rsv}), 3, 1)"
        lwr2 = f"ta_sma(({lwr1}), 3, 1)"
        self._add("LWR1", lwr1)
        self._add("LWR2", lwr2)

    def _register_bias(self) -> None:
        for n in (5, 10, 20):
            ma = self._ma("close", n)
            self._add(f"BIAS{n}", self._div(f"close - {ma}", ma) + " * 100")
        bias36 = f"({self._ma('close', 3)} - {self._ma('close', 6)})"
        bias612 = f"({self._ma('close', 6)} - {self._ma('close', 12)})"
        self._add("BIAS36", bias36)
        self._add("BIAS612", bias612)
        self._add("MABIAS", self._ma(bias36, 6))

    def _register_accer_cyf_fsl(self) -> None:
        self._add("ACCER", self._div("ts_slope(close, 8)", "close"))
        self._add("CYF", f"100 - {self._div('100', f'1 + {self._ema(self.hsl, 21)}')}")
        self._add("SWL", f"({self._ema('close', 5)} * 7 + {self._ema('close', 10)} * 3) / 10")
        sws_alpha = self._max(self._div("100 * ts_sum(volume, 5)", f"3 * {self.capital}"), "1")
        self._add("SWS", f"rq_dma(({self._ema('close', 12)}), ({sws_alpha}))")

    def _register_adtm(self) -> None:
        op1 = self._delay("open", 1)
        dtm = self._if(
            f"open <= {op1}",
            "0",
            self._max("high - open", f"open - {op1}"),
        )
        dbm = self._if(
            f"open >= {op1}",
            "0",
            self._max("open - low", f"open - {op1}"),
        )
        stm = self._sum(dtm, 23)
        sbm = self._sum(dbm, 23)
        adtm = self._if(
            f"{stm} > {sbm}",
            self._div(f"{stm} - {sbm}", stm),
            self._if(f"{stm} == {sbm}", "0", self._div(f"{stm} - {sbm}", sbm)),
        )
        self._add("ADTM", adtm)
        self._add("MAADTM", self._ma(adtm, 8))

    def _register_atr(self) -> None:
        tr = self._sum(self._true_range(), 9)
        self._add("TR", tr)
        self._add("ATR", self._ma(tr, 14))

    def _register_dkx(self) -> None:
        mid = self.mid
        terms = [f"{20 - i} * {self._delay(mid, i)}" if i else f"20 * ({mid})" for i in range(20)]
        dkx = f"(({'+'.join(terms)}) / 210)"
        self._add("DKX", dkx)
        self._add("MADKX", self._ma(dkx, 10))

    def _register_tapi(self) -> None:
        tapi = self._div(self.amount, "close")
        self._add("TAPI", tapi)
        self._add("MATAPI", self._ma(tapi, 6))

    def _register_osc_cci_roc_mfi(self) -> None:
        self._add("OSC", f"100 * (close - {self._ma('close', 10)})")
        typ = "((high + low + close) / 3)"
        typ_ma = self._ma(typ, 14)
        avedev = f"ts_avedev(({typ}), 14)"
        self._add("CCI", self._div(f"{typ} - {typ_ma}", f"0.015 * {avedev}"))
        self._add("ROC", self._div(f"close - {self._delay('close', 12)}", self._delay("close", 12)) + " * 100")

        typ1 = self._delay(typ, 1)
        up_flow = self._if(f"{typ} > {typ1}", f"{typ} * volume", "0")
        down_flow = self._if(f"{typ} < {typ1}", f"{typ} * volume", "0")
        v1 = self._div(self._sum(up_flow, 14), self._sum(down_flow, 14))
        self._add("MFI", f"100 - {self._div('100', f'1 + {v1}')}")

    def _register_mtm_marsi_skd_udl(self) -> None:
        mtm = f"(close - {self._delay('close', 14)})"
        self._add("MTM", mtm)
        self._add("MAMTM", self._ma(mtm, 6))

        lc = self._delay("close", 1)
        delta = f"close - {lc}"
        for n in (6, 10):
            rsi = self._div(f"ta_sma(({self._max(delta, '0')}), {n}, 1)", f"ta_sma(({self._abs(delta)}), {n}, 1)") + " * 100"
            self._add(f"MARSI{n}", self._ma(rsi, n))

        lowv = self._llv("low", 9)
        highv = self._hhv("high", 9)
        skd_rsv = self._ema(self._div(f"close - {lowv}", f"{highv} - {lowv}") + " * 100", 3)
        skd_k = self._ema(skd_rsv, 3)
        self._add("SKD_K", skd_k)
        self._add("SKD_D", self._ma(skd_k, 3))

        udl = f"({self._ma('close', 3)} + {self._ma('close', 5)} + {self._ma('close', 10)} + {self._ma('close', 20)}) / 4"
        self._add("UDL", udl)
        self._add("MAUDL", self._ma(udl, 6))

    def _register_dmi(self) -> None:
        tr = self._sum(self._true_range(), 14)
        hd = f"(high - {self._delay('high', 1)})"
        ld = f"({self._delay('low', 1)} - low)"
        dmp = self._sum(self._if(f"({hd} > 0) & ({hd} > {ld})", hd, "0"), 14)
        dmm = self._sum(self._if(f"({ld} > 0) & ({ld} > {hd})", ld, "0"), 14)
        di1 = self._div(f"{dmp} * 100", tr)
        di2 = self._div(f"{dmm} * 100", tr)
        adx = self._ma(self._div(f"{self._abs(f'{di2} - {di1}')} * 100", f"{di1} + {di2}"), 6)
        self._add("DI1", di1)
        self._add("DI2", di2)
        self._add("ADX", adx)
        self._add("ADXR", f"(({adx}) + {self._delay(adx, 6)}) / 2")

    # -------------------- energy indicators --------------------

    def _register_arbr(self) -> None:
        self._add("AR", self._div(self._sum("high - open", 26), self._sum("open - low", 26)) + " * 100")
        lc = self._delay("close", 1)
        self._add("BR", self._div(self._sum(self._max(f"high - {lc}", "0"), 26), self._sum(self._max(f"{lc} - low", "0"), 26)) + " * 100")

    def _register_vr(self) -> None:
        lc = self._delay("close", 1)
        vr = self._div(
            self._sum(self._if(f"close > {lc}", "volume", "0"), 26),
            self._sum(self._if(f"close <= {lc}", "volume", "0"), 26),
        ) + " * 100"
        self._add("VR", vr)
        self._add("MAVR", self._ma(vr, 6))

    def _register_cr(self) -> None:
        mid = f"({self._delay('high + low', 1)} / 2)"
        cr = self._div(self._sum(self._max(f"high - {mid}", "0"), 26), self._sum(self._max(f"{mid} - low", "0"), 26)) + " * 100"
        self._add("CR", cr)
        for name, ma_window, delay_window in (
            ("MACR1", 10, 5),
            ("MACR2", 20, 9),
            ("MACR3", 40, 17),
            ("MACR4", 62, 26),
        ):
            self._add(name, self._delay(self._ma(cr, ma_window), delay_window))

    def _register_mass(self) -> None:
        hl_ma = self._ma("high - low", 9)
        mass = self._sum(self._div(hl_ma, self._ma(hl_ma, 9)), 25)
        self._add("MASS", mass)
        self._add("MAMASS", self._ma(mass, 6))

    def _register_sy_pcnt_cyr(self) -> None:
        lc = self._delay("close", 1)
        self._add("SY", f"{self._div(self._sum(self._if(f'close > {lc}', '1', '0'), 9), '9')} * 100")
        self._add("PCNT", self._div(f"close - {lc}", "close") + " * 100")

        dive = self._div(f"0.01 * {self._ema(self.amount, 13)}", self._ema("volume", 13))
        cyr = f"({self._div(dive, self._delay(dive, 1))} - 1) * 100"
        self._add("CYR", cyr)
        self._add("MACYR", self._ma(cyr, 5))

    def _register_amp_wma_volatility(self) -> None:
        for n in self.AMP_WINDOWS:
            self._add(f"AMP{n}", self._div(f"{self._hhv('high', n)} - {self._llv('low', n)}", self._delay("close", n)))
        for n in self.WMA_WINDOWS:
            self._add(f"WMA{n}", f"ta_wma(close, {n})")
        for n in (20, 60):
            self._add(f"VOLT{n}", self._std("close", n))
            self._add(f"MDD{n}", f"ts_mdd(close, {n})")

    def _register_aroon_qtyr_obv(self) -> None:
        n = 14
        self._add("AROON_UP", f"(({n} - (ts_highday(high, {n}) - 1)) / {n}) * 100")
        self._add("AROON_DOWN", f"(({n} - (ts_lowday(low, {n}) - 1)) / {n}) * 100")
        self._add("QTYR_5_20", self._div(self._ma("volume", 5), self._ma("volume", 20)))
        sign = self._if(
            f"close > {self._delay('close', 1)}",
            "1",
            self._if(f"close < {self._delay('close', 1)}", "-1", "0"),
        )
        self._add("OBV", f"ts_cumsum(({sign}) * volume)")
