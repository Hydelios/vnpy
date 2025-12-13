# alpha_191_new.py
# 参考 alpha_101_new.py，把 Alpha191 改造成参数化表达式注册版

from itertools import product
from typing import Any, Callable, Iterable
import polars as pl
from vnpy.alpha.dataset import AlphaDataset


class Alpha191NEW(AlphaDataset):
    """
    在当前 Dataset 内注册 Alpha191 的表达式（字符串），
    通过 self.add_feature(name: str, expr: str) 写入到本对象的 feature_expressions 中。

    用法示例：
      ds = Alpha191NEW(df, train_period, valid_period, test_period, interval="1d")
      ds.register_001_010()
      # 后续可以增加 register_011_020 等
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
    DEFAULT_WINDOWS: list[int] = [5, 10, 20, 30]

    # ========= 工具：多参数展开 =========
    @staticmethod
    def _fmt_val(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            return f"{v:.12g}"
        return str(v)

    @classmethod
    def _make_name(cls, base: str, params: dict[str, Any]) -> str:
        if not params:
            return base
        kv = "__".join(f"{k}={cls._fmt_val(params[k])}" for k in sorted(params))
        return f"{base}__{kv}"

    def _add_parametric_feature(
        self,
        base_name: str,
        expr_tpl: str,
        param_grid: dict[str, Iterable[Any]] | None = None,
        *,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
        dedup: bool = True,
    ) -> None:
        """
        完全照搬 alpha_101_new.py 的注册逻辑：
        - 无 param_grid：直接 add_feature(base_name, expr_tpl)
        - 有 param_grid：做笛卡尔积，名字带上参数后缀
        """
        if not param_grid:
            self.add_feature(base_name, expr_tpl)
            return
        keys = list(param_grid.keys())
        values = [list(v) for v in param_grid.values()]
        seen: set[str] = set()
        for combo in product(*values):
            params = {k: combo[i] for i, k in enumerate(keys)}
            if predicate and not predicate(params):
                continue
            name = self._make_name(base_name, params)
            if dedup and name in seen:
                continue
            expr = expr_tpl.format(**params)
            self.add_feature(name, expr)
            seen.add(name)

    # ========= 公共片段 =========
    @property
    def ret(self) -> str:
        """与原 self.returns 一致：日收益率 close_t / close_{t-1} - 1"""
        return "(close / ts_delay(close, 1) - 1)"

    @property
    def adv20(self) -> str:
        return "ts_mean(volume, 20)"

    # ========= 001 ～ 010 =========
    def register_001_010(self, windows: list[int] | None = None) -> None:
        """
        对应 alphas191.py 中 alpha001 ~ alpha010
        逻辑尽量按原始实现翻译到 ts_*/cs_* 表达式：
        - Corr → ts_corr
        - Rank → cs_rank / ts_rank
        - Delta → ts_delta / (x - ts_delay(x, n))
        - Std / Mean / Sum / Tsmax / Tsmin → ts_std / ts_mean / ts_sum / ts_max / ts_min
        - 条件 → ts_where
        """
        W = windows or self.DEFAULT_WINDOWS
        ret = self.ret
        vwap = "vwap"
        ret = self.ret

        # 001: (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))
        # 原式：Corr(Rank(Delta(log(volume),1)), Rank((close-open)/open), 6) * -1【191#1】
        self._add_parametric_feature(
            base_name="alpha191_001",
            expr_tpl=(
                "-1 * ts_corr("
                "  cs_rank(ts_log(volume) - ts_delay(ts_log(volume), {w_dv})),"
                "  cs_rank((close - open) / open),"
                "  {w_corr}"
                ")"
            ),
            param_grid={"w_dv": [1], "w_corr": [6]},
        )

        # 002: -1 * delta((((close-low)-(high-close))/(high-low)),1)
        # 原式完全对应【191#2】
        self._add_parametric_feature(
            base_name="alpha191_002",
            expr_tpl=(
                "-1 * ts_delta("
                "  (((close - low) - (high - close)) / ((high - low) + 1e-12)),"
                "  {w_d}"
                ")"
            ),
            param_grid={"w_d": [1]},
        )

        # 003: SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?
        #              MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
        # 这里用 ts_where + 内置 min/max 近似（需要表达式引擎支持 min/max 元素级运算）
        #【191#3】
        self._add_parametric_feature(
            base_name="alpha191_003",
            expr_tpl=(
                "ts_sum("
                "  ts_where("
                "    close == ts_delay(close, {w_d}),"
                "    0,"
                "    ts_where("
                "      close > ts_delay(close, {w_d}),"
                "      close - ts_less(low, ts_delay(close, {w_d})),"
                "      close - ts_greater(high, ts_delay(close, {w_d}))"
                "    )"
                "  ),"
                "  {w_sum}"
                ")"
            ),
            param_grid={"w_d": [1], "w_sum": [6]},
        )

        # 004:
        # ((((SUM(CLOSE, 8)/8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2)/2)) ? (-1) :
        #   (((SUM(CLOSE, 2)/2) < ((SUM(CLOSE, 8)/8) - STD(CLOSE, 8))) ? 1 :
        #    (((1 < (VOLUME/MEAN(VOLUME,20))) || ((VOLUME/MEAN(VOLUME,20)) == 1)) ? 1 : -1)))
        # 用 ts_mean/ts_std + ts_where 实现【191#4】
        self._add_parametric_feature(
            base_name="alpha191_004",
            expr_tpl=(
                "ts_where("
                "  (ts_mean(close, {w_long}) + ts_std(close, {w_std})) < ts_mean(close, {w_short}),"
                "  -1,"
                "  ts_where("
                "    ts_mean(close, {w_short}) < (ts_mean(close, {w_long}) - ts_std(close, {w_std})),"
                "    1,"
                "    ts_where("
                "      volume / (ts_mean(volume, {w_vol}) + 1e-12) >= 1,"
                "      1,"
                "      -1"
                "    )"
                "  )"
                ")"
            ),
            param_grid={"w_long": [8], "w_short": [2], "w_std": [8], "w_vol": [20]},
        )

        # 005: (-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))
        #【191#5】
        self._add_parametric_feature(
            base_name="alpha191_005",
            expr_tpl=(
                "-1 * ts_max("
                "  ts_corr(ts_rank(volume, {w_rank}), ts_rank(high, {w_rank}), {w_corr}),"
                "  {w_tsm}"
                ")"
            ),
            param_grid={"w_rank": [5], "w_corr": [5], "w_tsm": [3]},
        )

        # 006: (RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)
        # Sign → ts_sign；Delta → ts_delta【191#6】
        self._add_parametric_feature(
            base_name="alpha191_006",
            expr_tpl=(
                "-1 * cs_rank( ts_sign( ts_delta(open*0.85 + high*0.15, {w_d}) ) )"
            ),
            param_grid={"w_d": [4]},
        )

        # 007: ((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))
        # Tsmax/Tsmin → ts_max/ts_min【191#7】
        self._add_parametric_feature(
            base_name="alpha191_007",
            expr_tpl=(
                "( cs_rank(ts_max({vwap} - close, {w_ex}))"
                "  + cs_rank(ts_min({vwap} - close, {w_ex})) )"
                " * cs_rank(ts_delta(volume, {w_d}))"
            ).format(vwap=vwap, w_ex="{w_ex}", w_d="{w_d}"),
            param_grid={"w_ex": [3], "w_d": [3]},
        )

        # 008: RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)
        #【191#8】
        self._add_parametric_feature(
            base_name="alpha191_008",
            expr_tpl=(
                "cs_rank(-1 * ts_delta(((high + low) / 2 * 0.2 + {vwap} * 0.8), {w_d}))"
            ).format(vwap=vwap, w_d="{w_d}"),
            param_grid={"w_d": [4]},
        )

        # 009: SMA(((HIGH+LOW)/2 - (DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME, 7, 2)
        # 这里没有直接 ta_sma 的表达式算子，先用 ts_mean 作一个“等权 Sma”近似
        #【191#9】
        self._add_parametric_feature(
            base_name="alpha191_009",
            expr_tpl=(
                "ts_mean("
                "  ((high + low) / 2 - (ts_delay(high, 1) + ts_delay(low, 1)) / 2)"
                "  * (high - low) / (volume + 1e-12),"
                "  {w_ma}"
                ")"
            ),
            param_grid={"w_ma": [7]},  # 原 Sma(…, 7, 2) 的 n=7
        )

        # 010: (RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))
        # 原实现：用 returns<0 时用 Std(returns,20)，否则 close，再平方后 Tsmax + Rank【191#10】
        self._add_parametric_feature(
            base_name="alpha191_010",
            expr_tpl=(
                "cs_rank("
                "  ts_max("
                "    ( ts_where( ({ret}) < 0, ts_std({ret}, {w_std}), close ) ** 2 ),"
                "    {w_tsm}"
                "  )"
                ")"
            ).format(ret=ret, w_std="{w_std}", w_tsm="{w_tsm}"),
            param_grid={"w_std": [20], "w_tsm": [5]},
        )

    # ========= 011 ～ 020 =========
    def register_011_020(self, windows: list[int] | None = None) -> None:
        """
        对应 alphas191.py 中 alpha011 ~ alpha020
        """
        vwap = "vwap"

        # 011: SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,6)
        # 直接用 ts_sum 实现滚动求和
        self._add_parametric_feature(
            base_name="alpha191_011",
            expr_tpl=(
                "ts_sum("
                "  ((close - low) - (high - close)) / ((high - low) + 1e-12) * volume,"
                "  {w_sum}"
                ")"
            ),
            param_grid={"w_sum": [6]},
        )

        # 012: (RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))
        # Rank 是截面排序 → cs_rank
        self._add_parametric_feature(
            base_name="alpha191_012",
            expr_tpl=(
                "cs_rank(open - ts_mean({vwap}, {w_v}))*"
                "(-1 * cs_rank(abs(close - {vwap})))"
            ).format(vwap=vwap, w_v="{w_v}"),
            param_grid={"w_v": [10]},
        )

        # 013: (((HIGH * LOW)^0.5) - VWAP)
        self._add_parametric_feature(
            base_name="alpha191_013",
            expr_tpl=(
                "(high * low) ** 0.5 - {vwap}"
            ).format(vwap=vwap),
        )

        # 014: CLOSE - DELAY(CLOSE,5)
        self._add_parametric_feature(
            base_name="alpha191_014",
            expr_tpl="close - ts_delay(close, {w_d})",
            param_grid={"w_d": [5]},
        )

        # 015: OPEN/DELAY(CLOSE,1) - 1
        self._add_parametric_feature(
            base_name="alpha191_015",
            expr_tpl="open / ts_delay(close, {w_d}) - 1",
            param_grid={"w_d": [1]},
        )

        # 016: (-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))
        # Corr 里先做截面 Rank，再在时间维度做 Corr+Tsmax
        self._add_parametric_feature(
            base_name="alpha191_016",
            expr_tpl=(
                "-1 * ts_max("
                "  cs_rank("
                "    ts_corr(cs_rank(volume), cs_rank({vwap}), {w_corr})"
                "  ),"
                "  {w_tsm}"
                ")"
            ).format(vwap=vwap, w_corr="{w_corr}", w_tsm="{w_tsm}"),
            param_grid={"w_corr": [5], "w_tsm": [5]},
        )

        # 017: RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)
        # Rank((vwap - ts_max(vwap,15))) ** ts_delta(close,5)
        self._add_parametric_feature(
            base_name="alpha191_017",
            expr_tpl=(
                "cs_rank({vwap} - ts_max({vwap}, {w_max}))"
                " ** ts_delta(close, {w_d})"
            ).format(vwap=vwap, w_max="{w_max}", w_d="{w_d}"),
            param_grid={"w_max": [15], "w_d": [5]},
        )

        # 018: CLOSE/DELAY(CLOSE,5)
        self._add_parametric_feature(
            base_name="alpha191_018",
            expr_tpl="close / ts_delay(close, {w_d})",
            param_grid={"w_d": [5]},
        )

        # 019:
        # (CLOSE<DELAY(CLOSE,5)?
        #    (CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5) :
        #    (CLOSE=DELAY(CLOSE,5)? 0 :
        #      (CLOSE-DELAY(CLOSE,5))/CLOSE))
        # 用 ts_where 链式实现条件
        self._add_parametric_feature(
            base_name="alpha191_019",
            expr_tpl=(
                "ts_where("
                "  close < ts_delay(close, {w_d}),"
                "  (close - ts_delay(close, {w_d})) / (ts_delay(close, {w_d}) + 1e-12),"
                "  ts_where("
                "    close == ts_delay(close, {w_d}),"
                "    0,"
                "    (close - ts_delay(close, {w_d})) / (close + 1e-12)"
                "  )"
                ")"
            ),
            param_grid={"w_d": [5]},
        )

        # 020: (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
        self._add_parametric_feature(
            base_name="alpha191_020",
            expr_tpl=(
                "(close - ts_delay(close, {w_d}))"
                " / (ts_delay(close, {w_d}) + 1e-12) * 100"
            ),
            param_grid={"w_d": [6]},
        )

    # ========= 021 ～ 030 =========
    def register_021_030(self, windows: list[int] | None = None) -> None:
        """
        对应 alphas191.py / 研报 / BigQuant / zybuluo 中的 Alpha21~Alpha30
        """
        W = windows or self.DEFAULT_WINDOWS
        ret = self.ret
        vwap = "vwap"

        # 021: REGBETA(MEAN(CLOSE,6), SEQUENCE(6))
        # 参考：用 TA 的 LINEARREG_SLOPE 做 6 日线性回归斜率
        # 需要你在 ta_function.py 里实现 ta_linearreg_slope(series, window)
        self._add_parametric_feature(
            base_name="alpha191_021",
            expr_tpl=(
                "ta_linearreg_slope("
                "  ts_mean(close, {w_mean}),"
                "  {w_reg}"
                ")"
            ),
            param_grid={"w_mean": [6], "w_reg": [6]},
        )

        # 022: SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)
        #           - DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
        # 用 ts_mean 近似 Sma
        self._add_parametric_feature(
            base_name="alpha191_022",
            expr_tpl=(
                "ts_mean("
                "  ( (close - ts_mean(close, {w_mean})) / ts_mean(close, {w_mean})"
                "    - ts_delay((close - ts_mean(close, {w_mean})) / ts_mean(close, {w_mean}), {w_d})"
                "  ),"
                "  {w_sma}"
                ")"
            ),
            param_grid={"w_mean": [6], "w_d": [3], "w_sma": [12]},
        )

        # 023:
        # SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
        #   / (SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
        #      + SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)) * 100
        self._add_parametric_feature(
            base_name="alpha191_023",
            expr_tpl=(
                "ts_mean("
                "  ts_where("
                "    close > ts_delay(close, 1),"
                "    ts_std(close, {w_std}),"
                "    0"
                "  ),"
                "  {w_sma}"
                ")"
                " / ("
                "  ts_mean("
                "    ts_where("
                "      close > ts_delay(close, 1),"
                "      ts_std(close, {w_std}),"
                "      0"
                "    ),"
                "    {w_sma}"
                "  )"
                "  + ts_mean("
                "    ts_where("
                "      close <= ts_delay(close, 1),"
                "      ts_std(close, {w_std}),"
                "      0"
                "    ),"
                "    {w_sma}"
                "  ) + 1e-12"
                ") * 100"
            ),
            param_grid={"w_std": [20], "w_sma": [20]},
        )

        # 024: SMA(CLOSE-DELAY(CLOSE,5),5,1)
        self._add_parametric_feature(
            base_name="alpha191_024",
            expr_tpl=(
                "ts_mean(close - ts_delay(close, {w_d}), {w_sma})"
            ),
            param_grid={"w_d": [5], "w_sma": [5]},
        )

        # 025:
        # ((-1 * RANK(DELTA(CLOSE,7) * (1 - RANK(DECAYLINEAR((VOLUME/MEAN(VOLUME,20)),9))))) *
        #  (1 + RANK(SUM(RET, 250))))
        self._add_parametric_feature(
            base_name="alpha191_025",
            expr_tpl=(
                "(-1 * cs_rank("
                "   ts_delta(close, {w_d})"
                "   * (1 - cs_rank(ts_decay_linear(volume / (ts_mean(volume, {w_adv}) + 1e-12), {w_dec})))"
                "))"
                " * (1 + cs_rank(ts_sum({ret}, {w_ret})))"
            ).format(ret=ret, w_d="{w_d}", w_adv="{w_adv}", w_dec="{w_dec}", w_ret="{w_ret}"),
            param_grid={"w_d": [7], "w_adv": [20], "w_dec": [9], "w_ret": [250]},
        )

        # 026: (((SUM(CLOSE,7) / 7) - CLOSE) + CORR(VWAP, DELAY(CLOSE,5),230))
        self._add_parametric_feature(
            base_name="alpha191_026",
            expr_tpl=(
                "(ts_mean(close, {w_mean}) - close)"
                " + ts_corr({vwap}, ts_delay(close, {w_d}), {w_corr})"
            ).format(vwap=vwap, w_mean="{w_mean}", w_d="{w_d}", w_corr="{w_corr}"),
            param_grid={"w_mean": [7], "w_d": [5], "w_corr": [230]},
        )

        # 027: WMA( (CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100
        #          + (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100, 12)
        # 用 ts_mean 近似 WMA
        self._add_parametric_feature(
            base_name="alpha191_027",
            expr_tpl=(
                "ts_mean("
                "  (close - ts_delay(close, 3)) / (ts_delay(close, 3) + 1e-12) * 100"
                "  + (close - ts_delay(close, 6)) / (ts_delay(close, 6) + 1e-12) * 100,"
                "  {w_ma}"
                ")"
            ),
            param_grid={"w_ma": [12]},
        )

        # 028:
        # 3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
        #   - 2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
        # 这里第二段按 zybuluo 的表达，用同一 denominators 近似
        self._add_parametric_feature(
            base_name="alpha191_028",
            expr_tpl=(
                "3 * ts_mean("
                "      (close - ts_min(low, {w_win})) / (ts_max(high, {w_win}) - ts_min(low, {w_win}) + 1e-12) * 100,"
                "      {w_sma1}"
                "  )"
                " - 2 * ts_mean("
                "      ts_mean("
                "        (close - ts_min(low, {w_win})) / (ts_max(high, {w_win}) - ts_min(low, {w_win}) + 1e-12) * 100,"
                "        {w_sma1}"
                "      ),"
                "      {w_sma2}"
                "  )"
            ),
            param_grid={"w_win": [9], "w_sma1": [3], "w_sma2": [3]},
        )

        # 029: (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
        self._add_parametric_feature(
            base_name="alpha191_029",
            expr_tpl=(
                "(close - ts_delay(close, {w_d}))"
                " / (ts_delay(close, {w_d}) + 1e-12) * volume"
            ),
            param_grid={"w_d": [6]},
        )
        
        # 未实现
        # 030: WMA((REGRESI(CLOSE/DELAY(CLOSE)-1,MKT,SMB,HML,60))^2,20)
        # 需要指数/风格因子 & 多元回归，这里先占位为常数 0，后续你可以单独接入 Barra/Fama-French 回归再改掉
        self._add_parametric_feature(
            base_name="alpha191_030",
            expr_tpl="close-close",
            param_grid={},
        )

    # ========= 031 ～ 040 =========
    def register_031_040(self, windows: list[int] | None = None) -> None:
        """
        对应 Alpha191 中的 Alpha31 ~ Alpha40
        """
        W = windows or self.DEFAULT_WINDOWS
        ret = self.ret
        vwap = "vwap"

        # 031: (CLOSE - MEAN(CLOSE,12)) / MEAN(CLOSE,12) * 100
        self._add_parametric_feature(
            base_name="alpha191_031",
            expr_tpl=(
                "(close - ts_mean(close, {w_mean}))"
                " / (ts_mean(close, {w_mean}) + 1e-12) * 100"
            ),
            param_grid={"w_mean": [12]},
        )

        # 032: (-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))
        self._add_parametric_feature(
            base_name="alpha191_032",
            expr_tpl=(
                "-1 * ts_sum("
                "  cs_rank(ts_corr(cs_rank(high), cs_rank(volume), {w_corr})),"
                "  {w_sum}"
                ")"
            ),
            param_grid={"w_corr": [3], "w_sum": [3]},
        )

        # 033:
        # ((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5))
        #   * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220)))
        #   * TSRANK(VOLUME, 5))
        self._add_parametric_feature(
            base_name="alpha191_033",
            expr_tpl=(
                "(((-1 * ts_min(low, {w_low}))"
                "  + ts_delay(ts_min(low, {w_low}), {w_d}))"
                " * cs_rank((ts_sum({ret}, {w_long})"
                "           - ts_sum({ret}, {w_short})) / 220)"
                " * ts_rank(volume, {w_tsr}))"
            ).format(ret=ret, w_low="{w_low}", w_d="{w_d}",
                     w_long="{w_long}", w_short="{w_short}", w_tsr="{w_tsr}"),
            param_grid={
                "w_low": [5],
                "w_d": [5],
                "w_long": [240],
                "w_short": [20],
                "w_tsr": [5],
            },
        )

        # 034: MEAN(CLOSE,12) / CLOSE
        self._add_parametric_feature(
            base_name="alpha191_034",
            expr_tpl="ts_mean(close, {w_mean}) / (close + 1e-12)",
            param_grid={"w_mean": [12]},
        )

        # 035:
        # (MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)),
        #      RANK(DECAYLINEAR(CORR(VOLUME, (OPEN*0.65+OPEN*0.35), 17), 7))) * -1)
        # 注意：OPEN*0.65 + OPEN*0.35 在研报中本身就这么写，这里保持一致
        self._add_parametric_feature(
            base_name="alpha191_035",
            expr_tpl=(
                "-1 * ts_less("
                "  cs_rank(ts_decay_linear(ts_delta(open, {w_d}), {w_dec1})),"
                "  cs_rank(ts_decay_linear("
                "    ts_corr(volume, (open*0.65 + open*0.35), {w_corr}),"
                "    {w_dec2}"
                "  ))"
                ")"
            ),
            param_grid={"w_d": [1], "w_dec1": [15], "w_corr": [17], "w_dec2": [7]},
        )

        # 036: RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2))
        self._add_parametric_feature(
            base_name="alpha191_036",
            expr_tpl=(
                "cs_rank("
                "  ts_sum(ts_corr(cs_rank(volume), cs_rank({vwap}), {w_corr}), {w_sum})"
                ")"
            ).format(vwap=vwap, w_corr="{w_corr}", w_sum="{w_sum}"),
            param_grid={"w_corr": [6], "w_sum": [2]},
        )

        # 037: (-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5))
        #                  - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))
        self._add_parametric_feature(
            base_name="alpha191_037",
            expr_tpl=(
                "-1 * cs_rank("
                "  (ts_sum(open, {w_open}) * ts_sum({ret}, {w_ret})"
                "   - ts_delay(ts_sum(open, {w_open}) * ts_sum({ret}, {w_ret}), {w_d}))"
                ")"
            ).format(ret=ret, w_open="{w_open}", w_ret="{w_ret}", w_d="{w_d}"),
            param_grid={"w_open": [5], "w_ret": [5], "w_d": [10]},
        )

        # 038: (((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)
        self._add_parametric_feature(
            base_name="alpha191_038",
            expr_tpl=(
                "ts_where("
                "  ts_sum(high, {w_mean}) / {w_mean} < high,"
                "  -1 * ts_delta(high, {w_d}),"
                "  0"
                ")"
            ),
            param_grid={"w_mean": [20], "w_d": [2]},
        )

        # 039:
        # ((RANK(DECAYLINEAR(DELTA(CLOSE, 2),8))
        #   - RANK(DECAYLINEAR(CORR((VWAP*0.3 + OPEN*0.7),
        #                           SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)
        self._add_parametric_feature(
            base_name="alpha191_039",
            expr_tpl=(
                "-1 * ("
                "  cs_rank(ts_decay_linear(ts_delta(close, {w_d}), {w_dec1}))"
                "  - cs_rank(ts_decay_linear("
                "      ts_corr({vwap}*0.3 + open*0.7,"
                "              ts_sum(ts_mean(volume, {w_mv}), {w_sum_mv}),"
                "              {w_corr}"
                "      ),"
                "      {w_dec2}"
                "    ))"
                ")"
            ).format(vwap=vwap,
                     w_d="{w_d}", w_dec1="{w_dec1}",
                     w_mv="{w_mv}", w_sum_mv="{w_sum_mv}",
                     w_corr="{w_corr}", w_dec2="{w_dec2}"),
            param_grid={
                "w_d": [2],
                "w_dec1": [8],
                "w_mv": [180],
                "w_sum_mv": [37],
                "w_corr": [14],
                "w_dec2": [12],
            },
        )

        # 040:
        # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)
        #   / SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26) * 100
        self._add_parametric_feature(
            base_name="alpha191_040",
            expr_tpl=(
                "ts_sum("
                "  ts_where(close > ts_delay(close, 1), volume, 0),"
                "  {w_win}"
                ")"
                " / ("
                "  ts_sum("
                "    ts_where(close <= ts_delay(close, 1), volume, 0),"
                "    {w_win}"
                "  ) + 1e-12"
                ") * 100"
            ),
            param_grid={"w_win": [26]},
        )

    def register_041_050(self, windows: list[int] | None = None) -> None:
        """Register Alpha191 factors 041-050"""
        W = windows or self.DEFAULT_WINDOWS
        vwap = "vwap"

        # 041: (RANK(MAX(DELTA((VWAP), 3), 5)) * -1)
        self._add_parametric_feature(
            base_name="alpha191_041",
            expr_tpl="-1 * cs_rank(ts_max(ts_delta(vwap, 3), 5))",
            param_grid=None,
        )

        # 042: ((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))
        self._add_parametric_feature(
            base_name="alpha191_042",
            expr_tpl="(-1 * cs_rank(ts_std(high, 10))) * ts_corr(high, volume, 10)",
            param_grid=None,
        )

        # 043: SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)
        self._add_parametric_feature(
            base_name="alpha191_043",
            expr_tpl=(
                "ts_sum("
                "  ts_where("
                "    close > ts_delay(close, 1),"
                "    volume,"
                "    ts_where(close < ts_delay(close, 1), -volume, 0)"
                "  ),"
                "  6"
                ")"
            ),
            param_grid=None,
        )

        # 044: TSRANK(DECAYLINEAR(CORR(LOW, MEAN(VOLUME,10),7),6),4)
        #       + TSRANK(DECAYLINEAR(DELTA(VWAP,3),10),15)
        self._add_parametric_feature(
            base_name="alpha191_044",
            expr_tpl=(
                "ts_rank(ts_decay_linear(ts_corr(low, ts_mean(volume, 10), 7), 6), 4)"
                " + ts_rank(ts_decay_linear(ts_delta(vwap, 3), 10), 15)"
            ),
            param_grid=None,
        )

        # 045: RANK(DELTA(CLOSE*0.6+OPEN*0.4,1)) * RANK(CORR(VWAP, MEAN(VOLUME,150),15))
        self._add_parametric_feature(
            base_name="alpha191_045",
            expr_tpl=(
                "cs_rank(ts_delta(close * 0.6 + open * 0.4, 1))"
                " * cs_rank(ts_corr(vwap, ts_mean(volume, 150), 15))"
            ),
            param_grid=None,
        )

        # 046: (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24)) / (4*CLOSE)
        self._add_parametric_feature(
            base_name="alpha191_046",
            expr_tpl=(
                "(ts_mean(close, 3) + ts_mean(close, 6)"
                " + ts_mean(close, 12) + ts_mean(close, 24))"
                " / (4 * close)"
            ),
            param_grid=None,
        )

        # 047: SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100, 9, 1)
        self._add_parametric_feature(
            base_name="alpha191_047",
            expr_tpl=(
                "ts_mean("
                "  (ts_max(high, 6) - close) / (ts_max(high, 6) - ts_min(low, 6) + 1e-12) * 100,"
                "  9"
                ")"
            ),
            param_grid=None,
        )

        # 048:
        # -1 * ( RANK( SIGN(CLOSE - DELAY(CLOSE,1))
        #             + SIGN(DELAY(CLOSE,1) - DELAY(CLOSE,2))
        #             + SIGN(DELAY(CLOSE,2) - DELAY(CLOSE,3)) )
        #        * SUM(VOLUME,5) ) / SUM(VOLUME,20)
        self._add_parametric_feature(
            base_name="alpha191_048",
            expr_tpl=(
                "-1 * ("
                "  cs_rank("
                "    ts_sign(close - ts_delay(close, 1))"
                "    + ts_sign(ts_delay(close, 1) - ts_delay(close, 2))"
                "    + ts_sign(ts_delay(close, 2) - ts_delay(close, 3))"
                "  )"
                "  * ts_sum(volume, 5)"
                ") / ts_sum(volume, 20)"
            ),
            param_grid=None,
        )

        # 049:
        # A = SUM( cond_up ? 0 : amp, 12 )
        # B = SUM( cond_down ? 0 : amp, 12 )
        # where amp = max(|HIGH-DELAY(HIGH,1)|, |LOW-DELAY(LOW,1)|)
        # alpha49 = A / (A + B)
        expr_49 = (
            "ts_sum("
            "  ts_where("
            "    (high + low) >= (ts_delay(high, 1) + ts_delay(low, 1)),"
            "    0,"
            "    ts_where("
            "      abs(high - ts_delay(high, 1)) >= abs(low - ts_delay(low, 1)),"
            "      abs(high - ts_delay(high, 1)),"
            "      abs(low - ts_delay(low, 1))"
            "    )"
            "  ),"
            "  12"
            ")"
            " / "
            "("
            "  ts_sum("
            "    ts_where("
            "      (high + low) >= (ts_delay(high, 1) + ts_delay(low, 1)),"
            "      0,"
            "      ts_where("
            "        abs(high - ts_delay(high, 1)) >= abs(low - ts_delay(low, 1)),"
            "        abs(high - ts_delay(high, 1)),"
            "        abs(low - ts_delay(low, 1))"
            "      )"
            "    ),"
            "    12"
            "  )"
            "  + "
            "  ts_sum("
            "    ts_where("
            "      (high + low) <= (ts_delay(high, 1) + ts_delay(low, 1)),"
            "      0,"
            "      ts_where("
            "        abs(high - ts_delay(high, 1)) >= abs(low - ts_delay(low, 1)),"
            "        abs(high - ts_delay(high, 1)),"
            "        abs(low - ts_delay(low, 1))"
            "      )"
            "    ),"
            "    12"
            "  )"
            ")"
        )
        self._add_parametric_feature(
            base_name="alpha191_049",
            expr_tpl=expr_49,
            param_grid=None,
        )

        # 050:
        # same A, B as 049
        # alpha50 = A/(A+B) - B/(A+B)
        den_50 = (
            "  ts_sum("
            "    ts_where("
            "      (high + low) >= (ts_delay(high, 1) + ts_delay(low, 1)),"
            "      0,"
            "      ts_where("
            "        abs(high - ts_delay(high, 1)) >= abs(low - ts_delay(low, 1)),"
            "        abs(high - ts_delay(high, 1)),"
            "        abs(low - ts_delay(low, 1))"
            "      )"
            "    ),"
            "    12"
            "  )"
            "  + "
            "  ts_sum("
            "    ts_where("
            "      (high + low) <= (ts_delay(high, 1) + ts_delay(low, 1)),"
            "      0,"
            "      ts_where("
            "        abs(high - ts_delay(high, 1)) >= abs(low - ts_delay(low, 1)),"
            "        abs(high - ts_delay(high, 1)),"
            "        abs(low - ts_delay(low, 1))"
            "      )"
            "    ),"
            "    12"
            "  )"
        )
        expr_50 = (
            "ts_sum("
            "  ts_where("
            "    (high + low) <= (ts_delay(high, 1) + ts_delay(low, 1)),"
            "    0,"
            "    ts_where("
            "      abs(high - ts_delay(high, 1)) >= abs(low - ts_delay(low, 1)),"
            "      abs(high - ts_delay(high, 1)),"
            "      abs(low - ts_delay(low, 1))"
            "    )"
            "  ),"
            "  12"
            ")"
            " / "
            "(" + den_50 + ")"
            " - "
            "ts_sum("
            "  ts_where("
            "    (high + low) >= (ts_delay(high, 1) + ts_delay(low, 1)),"
            "    0,"
            "    ts_where("
            "      abs(high - ts_delay(high, 1)) >= abs(low - ts_delay(low, 1)),"
            "      abs(high - ts_delay(high, 1)),"
            "      abs(low - ts_delay(low, 1))"
            "    )"
            "  ),"
            "  12"
            ")"
            " / "
            "(" + den_50 + ")"
        )
        self._add_parametric_feature(
            base_name="alpha191_050",
            expr_tpl=expr_50,
            param_grid=None,
        )

    # ========= 051 ～ 060 =========
    def register_051_060(self, windows: list[int] | None = None) -> None:
        """
        对应 Alpha191 中的 Alpha51 ~ Alpha60
        """
        W = windows or self.DEFAULT_WINDOWS

        # 051:
        # SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),
        #                                                     ABS(LOW-DELAY(LOW,1)))),12)
        # / [同样 window 内 up+down 之和]
        amp_expr = (
            "ts_where("
            "  abs(high - ts_delay(high, 1)) >= abs(low - ts_delay(low, 1)),"
            "  abs(high - ts_delay(high, 1)),"
            "  abs(low - ts_delay(low, 1))"
            ")"
        )
        up_expr = (
            "ts_where("
            "  high + low <= ts_delay(high, 1) + ts_delay(low, 1),"
            "  0,"
            f"  {amp_expr}"
            ")"
        )
        down_expr = (
            "ts_where("
            "  high + low <= ts_delay(high, 1) + ts_delay(low, 1),"
            f"  {amp_expr},"
            "  0"
            ")"
        )
        self._add_parametric_feature(
            base_name="alpha191_051",
            expr_tpl=(
                "ts_sum(" + up_expr + ", {w_win})"
                " / (ts_sum(" + up_expr + ", {w_win})"
                "    + ts_sum(" + down_expr + ", {w_win}) + 1e-12)"
            ),
            param_grid={"w_win": [12]},
        )

        # 052:
        # SUM(MAX(0, HIGH - DELAY((HIGH+LOW+CLOSE)/3,1)),26)
        # / SUM(MAX(0, DELAY((HIGH+LOW+CLOSE)/3,1) - LOW),26) * 100
        hlc1 = "(high + low + close) / 3"
        delay_hlc1 = f"ts_delay({hlc1}, 1)"
        self._add_parametric_feature(
            base_name="alpha191_052",
            expr_tpl=(
                "ts_sum("
                "  ts_where("
                f"    high - {delay_hlc1} > 0,"
                f"    high - {delay_hlc1},"
                "    0"
                "  ),"
                "  {w_win}"
                ")"
                " / ts_sum("
                "  ts_where("
                f"    {delay_hlc1} - low > 0,"
                f"    {delay_hlc1} - low,"
                "    0"
                "  ),"
                "  {w_win}"
                ") * 100"
            ),
            param_grid={"w_win": [26]},
        )

        # 053: COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
        self._add_parametric_feature(
            base_name="alpha191_053",
            expr_tpl=(
                "ts_sum("
                "  ts_where(close > ts_delay(close, 1), 1, 0),"
                "  {w_win}"
                ") / {w_win} * 100"
            ),
            param_grid={"w_win": [12]},
        )

        # 054:
        # (-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))
        # 这里 STD 没给窗口，按常见实现取 20 日
        self._add_parametric_feature(
            base_name="alpha191_054",
            expr_tpl=(
                "-1 * cs_rank("
                "  (ts_std(abs(close - open), {w_std}) + (close - open))"
                "  + ts_corr(close, open, {w_corr})"
                ")"
            ),
            param_grid={"w_std": [20], "w_corr": [10]},
        )
        
        # 未实现
        # 055: 原文件注释“公式有问题”，源码中也有省略号，无法可靠还原，这里先占位 0
        self._add_parametric_feature(
            base_name="alpha191_055",
            expr_tpl="close-close",
            param_grid={},
        )

        # 056:
        # (RANK((OPEN - TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19),
        #                                                      SUM(MEAN(VOLUME,40), 19), 13))^5)))
        # ? 1 : 0
        sum_mid = "ts_sum((high + low) / 2, {w_mid})"
        sum_mv = "ts_sum(ts_mean(volume, {w_mv}), {w_mid})"
        corr_big = (
            f"ts_corr({sum_mid}, {sum_mv}, {{w_corr}})"
        )
        self._add_parametric_feature(
            base_name="alpha191_056",
            expr_tpl=(
                "ts_where("
                "  cs_rank(open - ts_min(open, {w_min}))"
                "  < cs_rank(cs_rank(" + corr_big + ") ** 5),"
                "  1,"
                "  0"
                ")"
            ),
            param_grid={"w_min": [12], "w_mid": [19], "w_mv": [40], "w_corr": [13]},
        )

        # 057: SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
        # 这里还是用 ts_mean 近似 Sma
        self._add_parametric_feature(
            base_name="alpha191_057",
            expr_tpl=(
                "ts_mean("
                "  (close - ts_min(low, {w_win}))"
                "  / (ts_max(high, {w_win}) - ts_min(low, {w_win}) + 1e-12) * 100,"
                "  {w_sma}"
                ")"
            ),
            param_grid={"w_win": [9], "w_sma": [3]},
        )

        # 058: COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
        self._add_parametric_feature(
            base_name="alpha191_058",
            expr_tpl=(
                "ts_sum("
                "  ts_where(close > ts_delay(close, 1), 1, 0),"
                "  {w_win}"
                ") / {w_win} * 100"
            ),
            param_grid={"w_win": [20]},
        )

        # 059:
        # SUM((CLOSE=DELAY(CLOSE,1)?0:
        #      CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)
        self._add_parametric_feature(
            base_name="alpha191_059",
            expr_tpl=(
                "ts_sum("
                "  ts_where("
                "    close == ts_delay(close, 1),"
                "    0,"
                "    ts_where("
                "      close > ts_delay(close, 1),"
                "      close - ts_less(low, ts_delay(close, 1)),"
                "      close - ts_greater(high, ts_delay(close, 1))"
                "    )"
                "  ),"
                "  {w_win}"
                ")"
            ),
            param_grid={"w_win": [20]},
        )

        # 060: SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,20)
        self._add_parametric_feature(
            base_name="alpha191_060",
            expr_tpl=(
                "ts_sum("
                "  ((close - low) - (high - close))"
                "  / ((high - low) + 1e-12) * volume,"
                "  {w_win}"
                ")"
            ),
            param_grid={"w_win": [20]},
        )

    # ========= 061 ～ 070 =========
    def register_061_070(self, windows: list[int] | None = None) -> None:
        """
        对应 Alpha191 中的 Alpha61 ~ Alpha70
        """
        W = windows or self.DEFAULT_WINDOWS
        vwap = "vwap"

        # 061:
        # (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),
        #      RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80), 8)), 17))) * -1)
        self._add_parametric_feature(
            base_name="alpha191_061",
            expr_tpl=(
                "-1 * ts_greater("
                "  cs_rank(ts_decay_linear(ts_delta({vwap}, {w_d}), {w_dec1})),"
                "  cs_rank(ts_decay_linear("
                "    cs_rank(ts_corr(low, ts_mean(volume, {w_mv}), {w_corr})),"
                "    {w_dec2}"
                "  ))"
                ")"
            ).format(
                vwap=vwap,
                w_d="{w_d}",
                w_dec1="{w_dec1}",
                w_mv="{w_mv}",
                w_corr="{w_corr}",
                w_dec2="{w_dec2}",
            ),
            param_grid={"w_d": [1], "w_dec1": [12], "w_mv": [80], "w_corr": [8], "w_dec2": [17]},
        )

        # 062: (-1 * CORR(HIGH, RANK(VOLUME), 5))
        self._add_parametric_feature(
            base_name="alpha191_062",
            expr_tpl=(
                "-1 * ts_corr(high, cs_rank(volume), {w_corr})"
            ),
            param_grid={"w_corr": [5]},
        )

        # 063:
        # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)
        #   / SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1) * 100
        # 这里用 ts_mean 近似 Sma
        self._add_parametric_feature(
            base_name="alpha191_063",
            expr_tpl=(
                "ts_mean("
                "  ts_where(close - ts_delay(close, 1) > 0,"
                "           close - ts_delay(close, 1),"
                "           0),"
                "  {w_win}"
                ")"
                " / ("
                "  ts_mean(abs(close - ts_delay(close, 1)), {w_win})"
                "  + 1e-12"
                ") * 100"
            ),
            param_grid={"w_win": [6]},
        )

        # 064:
        # (MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),4),4)),
        #      RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE),RANK(MEAN(VOLUME,60)),4),13),14))) * -1)
        self._add_parametric_feature(
            base_name="alpha191_064",
            expr_tpl=(
                "-1 * ts_greater("
                "  cs_rank("
                "    ts_decay_linear("
                "      ts_corr(cs_rank({vwap}), cs_rank(volume), {w_corr1}),"
                "      {w_dec1}"
                "    )"
                "  ),"
                "  cs_rank("
                "    ts_decay_linear("
                "      ts_max("
                "        ts_corr(cs_rank(close), cs_rank(ts_mean(volume, {w_mv})), {w_corr2}),"
                "        {w_tsm}"
                "      ),"
                "      {w_dec2}"
                "    )"
                "  )"
                ")"
            ).format(
                vwap=vwap,
                w_corr1="{w_corr1}",
                w_dec1="{w_dec1}",
                w_mv="{w_mv}",
                w_corr2="{w_corr2}",
                w_tsm="{w_tsm}",
                w_dec2="{w_dec2}",
            ),
            param_grid={
                "w_corr1": [4],
                "w_dec1": [4],
                "w_mv": [60],
                "w_corr2": [4],
                "w_tsm": [13],
                "w_dec2": [14],
            },
        )

        # 065: MEAN(CLOSE,6)/CLOSE
        self._add_parametric_feature(
            base_name="alpha191_065",
            expr_tpl=(
                "ts_mean(close, {w_mean}) / (close + 1e-12)"
            ),
            param_grid={"w_mean": [6]},
        )

        # 066: (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
        self._add_parametric_feature(
            base_name="alpha191_066",
            expr_tpl=(
                "(close - ts_mean(close, {w_mean}))"
                " / (ts_mean(close, {w_mean}) + 1e-12) * 100"
            ),
            param_grid={"w_mean": [6]},
        )

        # 067:
        # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)
        #   / SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1) * 100
        self._add_parametric_feature(
            base_name="alpha191_067",
            expr_tpl=(
                "ts_mean("
                "  ts_where(close - ts_delay(close, 1) > 0,"
                "           close - ts_delay(close, 1),"
                "           0),"
                "  {w_win}"
                ")"
                " / ("
                "  ts_mean(abs(close - ts_delay(close, 1)), {w_win})"
                "  + 1e-12"
                ") * 100"
            ),
            param_grid={"w_win": [24]},
        )

        # 068:
        # SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)
        # 这里也用 ts_mean 近似 Sma
        self._add_parametric_feature(
            base_name="alpha191_068",
            expr_tpl=(
                "ts_mean("
                "  ((high + low) / 2"
                "   - (ts_delay(high, 1) + ts_delay(low, 1)) / 2)"
                "  * (high - low) / (volume + 1e-12),"
                "  {w_win}"
                ")"
            ),
            param_grid={"w_win": [15]},
        )

        # 069:
        # DTM = (OPEN<=DELAY(OPEN,1)?0:MAX(HIGH-OPEN, OPEN-DELAY(OPEN,1)))
        # DBM = (OPEN>=DELAY(OPEN,1)?0:MAX(OPEN-LOW, OPEN-DELAY(OPEN,1)))
        # if SUM(DTM,20) > SUM(DBM,20):
        #     (SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20)
        # elif ==: 0
        # else: (SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)
        dtm_expr = (
            "ts_where("
            "  open <= ts_delay(open, 1),"
            "  0,"
            "  ts_greater(high - open, open - ts_delay(open, 1))"
            ")"
        )
        dbm_expr = (
            "ts_where("
            "  open >= ts_delay(open, 1),"
            "  0,"
            "  ts_greater(open - low, open - ts_delay(open, 1))"
            ")"
        )
        sum_dtm = f"ts_sum({dtm_expr}, {{w_win}})"
        sum_dbm = f"ts_sum({dbm_expr}, {{w_win}})"
        self._add_parametric_feature(
            base_name="alpha191_069",
            expr_tpl=(
                "ts_where("
                f"  {sum_dtm} > {sum_dbm},"
                f"  ({sum_dtm} - {sum_dbm}) / ({sum_dtm} + 1e-12),"
                "  ts_where("
                f"    {sum_dtm} == {sum_dbm},"
                "    0,"
                f"    ({sum_dtm} - {sum_dbm}) / ({sum_dbm} + 1e-12)"
                "  )"
                ")"
            ),
            param_grid={"w_win": [20]},
        )

        # 070: STD(turnover,6)
        self._add_parametric_feature(
            base_name="alpha191_070",
            expr_tpl="ts_std(turnover, {w_win})",
            param_grid={"w_win": [6]},
        )

    # ========= 071 ～ 080 =========
    def register_071_080(self, windows: list[int] | None = None) -> None:
        """
        对应 Alpha191 中的 Alpha71 ~ Alpha80
        """
        W = windows or self.DEFAULT_WINDOWS
        vwap = "vwap"

        # 071: (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
        self._add_parametric_feature(
            base_name="alpha191_071",
            expr_tpl=(
                "(close - ts_mean(close, {w_mean}))"
                " / (ts_mean(close, {w_mean}) + 1e-12) * 100"
            ),
            param_grid={"w_mean": [24]},
        )

        # 072: SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
        # 这里用 ts_mean 近似 Sma
        self._add_parametric_feature(
            base_name="alpha191_072",
            expr_tpl=(
                "ts_mean("
                "  (ts_max(high, {w_hl}) - close)"
                "  / (ts_max(high, {w_hl}) - ts_min(low, {w_hl}) + 1e-12) * 100,"
                "  {w_sma}"
                ")"
            ),
            param_grid={"w_hl": [6], "w_sma": [15]},
        )

        # 073:
        # ((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR(CLOSE, VOLUME,10),16),4),5)
        #   - RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30),4),3))) * -1)
        self._add_parametric_feature(
            base_name="alpha191_073",
            expr_tpl=(
                "-1 * ("
                "  ts_rank("
                "    ts_decay_linear("
                "      ts_decay_linear(ts_corr(close, volume, {w_corr1}), {w_dec1}),"
                "      {w_dec2}"
                "    ),"
                "    {w_tsr}"
                "  )"
                "  - cs_rank("
                "      ts_decay_linear("
                "        ts_corr({vwap}, ts_mean(volume, {w_mv}), {w_corr2}),"
                "        {w_dec3}"
                "      )"
                "    )"
                ")"
            ).format(
                vwap=vwap,
                w_corr1="{w_corr1}",
                w_dec1="{w_dec1}",
                w_dec2="{w_dec2}",
                w_tsr="{w_tsr}",
                w_mv="{w_mv}",
                w_corr2="{w_corr2}",
                w_dec3="{w_dec3}",
            ),
            param_grid={
                "w_corr1": [10],
                "w_dec1": [16],
                "w_dec2": [4],
                "w_tsr": [5],
                "w_mv": [30],
                "w_corr2": [4],
                "w_dec3": [3],
            },
        )

        # 074:
        # (RANK(CORR(SUM(LOW*0.35 + VWAP*0.65, 20),
        #            SUM(MEAN(VOLUME,40), 20), 7))
        #  + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))
        self._add_parametric_feature(
            base_name="alpha191_074",
            expr_tpl=(
                "cs_rank("
                "  ts_corr("
                "    ts_sum(low*0.35 + {vwap}*0.65, {w_sum1}),"
                "    ts_sum(ts_mean(volume, {w_mv}), {w_sum1}),"
                "    {w_corr1}"
                "  )"
                ")"
                " + cs_rank("
                "     ts_corr(cs_rank({vwap}), cs_rank(volume), {w_corr2})"
                "   )"
            ).format(
                vwap=vwap,
                w_sum1="{w_sum1}",
                w_mv="{w_mv}",
                w_corr1="{w_corr1}",
                w_corr2="{w_corr2}",
            ),
            param_grid={"w_sum1": [20], "w_mv": [40], "w_corr1": [7], "w_corr2": [6]},
        )

        # benchmark_close未实现
        # 075:
        # COUNT(CLOSE>OPEN & BENCHMARKCLOSE<BENCHMARKOPEN,50)
        # / COUNT(BENCHMARKCLOSE<BENCHMARKOPEN,50)
        # 这里假设基准指数列名为 benchmark_close / benchmark_open
        # self._add_parametric_feature(
        #     base_name="alpha191_075",
        #     expr_tpl=(
        #         "ts_sum("
        #         "  ts_where("
        #         "    (close > open) & (benchmark_close < benchmark_open),"
        #         "    1,"
        #         "    0"
        #         "  ),"
        #         "  {w_win}"
        #         ")"
        #         " / ("
        #         "  ts_sum("
        #         "    ts_where(benchmark_close < benchmark_open, 1, 0),"
        #         "    {w_win}"
        #         "  )"
        #         "  + 1e-12"
        #         ")"
        #     ),
        #     param_grid={"w_win": [50]},
        # )

        # 076:
        # STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
        # / MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
        self._add_parametric_feature(
            base_name="alpha191_076",
            expr_tpl=(
                "ts_std("
                "  abs(close / (ts_delay(close, 1) + 1e-12) - 1) / (volume + 1e-12),"
                "  {w_win}"
                ")"
                " / ("
                "  ts_mean("
                "    abs(close / (ts_delay(close, 1) + 1e-12) - 1) / (volume + 1e-12),"
                "    {w_win}"
                "  )"
                "  + 1e-12"
                ")"
            ),
            param_grid={"w_win": [20]},
        )

        # 077:
        # MIN( RANK(DECAYLINEAR(((((HIGH + LOW)/2) + HIGH) - (VWAP + HIGH)), 20)),
        #      RANK(DECAYLINEAR(CORR(((HIGH + LOW)/2), MEAN(VOLUME,40), 3), 6)) )
        mid = "(high + low) / 2"
        expr_077 = (
            "ts_less("
            "  cs_rank( ts_decay_linear( ({mid} + high) - (vwap + high), {w_dec1}) ),"
            "  cs_rank( ts_decay_linear( ts_corr({mid}, ts_mean(volume, {w_mv}), {w_corr}), {w_dec2}) )"
            ")"
        ).replace("{mid}", mid)
        self._add_parametric_feature(
            base_name="alpha191_077",
            expr_tpl=expr_077,
            param_grid={"w_dec1": [20], "w_mv": [40], "w_corr": [3], "w_dec2": [6]},
        )

        # 078:
        # ((HIGH+LOW+CLOSE)/3 - MA((HIGH+LOW+CLOSE)/3,12))
        # / (0.015 * MEAN(ABS(CLOSE - MEAN((HIGH+LOW+CLOSE)/3,12)),12))
        hlc3 = "(high + low + close) / 3"
        ma_hlc3_12 = f"ts_mean({hlc3}, {{w_ma}})"
        self._add_parametric_feature(
            base_name="alpha191_078",
            expr_tpl=(
                "(({hlc3}) - {ma})"
                " / (0.015 * ts_mean("
                "        abs(close - {ma}),"
                "        {w_ma}"
                "      )"
                "    + 1e-12)"
            ).format(
                hlc3=hlc3,
                ma=ma_hlc3_12,
                w_ma="{w_ma}",
            ),
            param_grid={"w_ma": [12]},
        )

        # 079:
        # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)
        # / SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1) * 100
        self._add_parametric_feature(
            base_name="alpha191_079",
            expr_tpl=(
                "ts_mean("
                "  ts_where(close - ts_delay(close, 1) > 0,"
                "           close - ts_delay(close, 1),"
                "           0),"
                "  {w_win}"
                ")"
                " / ("
                "  ts_mean(abs(close - ts_delay(close, 1)), {w_win})"
                "  + 1e-12"
                ") * 100"
            ),
            param_grid={"w_win": [12]},
        )

        # 080: (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
        self._add_parametric_feature(
            base_name="alpha191_080",
            expr_tpl=(
                "(volume - ts_delay(volume, {w_d}))"
                " / (ts_delay(volume, {w_d}) + 1e-12) * 100"
            ),
            param_grid={"w_d": [5]},
        )

    # ========= 081 ～ 090 =========
    def register_081_090(self, windows: list[int] | None = None) -> None:
        """
        对应 Alpha191 中的 Alpha81 ~ Alpha90
        """
        W = windows or self.DEFAULT_WINDOWS
        vwap = "vwap"

        # 081: SMA(VOLUME,21,2)
        # 这里按前面习惯用 ts_mean 近似 Sma
        self._add_parametric_feature(
            base_name="alpha191_081",
            expr_tpl=(
                "ts_mean(volume, {w_win})"
            ),
            param_grid={"w_win": [21]},
        )

        # 082: SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)
        self._add_parametric_feature(
            base_name="alpha191_082",
            expr_tpl=(
                "ts_mean("
                "  (ts_max(high, {w_hl}) - close)"
                "  / (ts_max(high, {w_hl}) - ts_min(low, {w_hl}) + 1e-12) * 100,"
                "  {w_sma}"
                ")"
            ),
            param_grid={"w_hl": [6], "w_sma": [20]},
        )

        # 083: (-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))
        self._add_parametric_feature(
            base_name="alpha191_083",
            expr_tpl=(
                "-1 * cs_rank("
                "  ts_cov(cs_rank(high), cs_rank(volume), {w_cov})"
                ")"
            ),
            param_grid={"w_cov": [5]},
        )

        # 084: SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)
        # 按公式实现（涨：+vol，跌：-vol，平：0）
        self._add_parametric_feature(
            base_name="alpha191_084",
            expr_tpl=(
                "ts_sum("
                "  ts_where("
                "    close > ts_delay(close, 1),"
                "    volume,"
                "    ts_where(close < ts_delay(close, 1), -volume, 0)"
                "  ),"
                "  {w_win}"
                ")"
            ),
            param_grid={"w_win": [20]},
        )

        # 085: (TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))
        self._add_parametric_feature(
            base_name="alpha191_085",
            expr_tpl=(
                "ts_rank(volume / (ts_mean(volume, {w_mv}) + 1e-12), {w_tsr1})"
                " * ts_rank(-1 * ts_delta(close, {w_d}), {w_tsr2})"
            ),
            param_grid={"w_mv": [20], "w_d": [7], "w_tsr1": [20], "w_tsr2": [8]},
        )

        # 086:
        # A = ((DELAY(CLOSE,20)-DELAY(CLOSE,10))/10 - (DELAY(CLOSE,10)-CLOSE)/10)
        # A > 0.25 -> -1
        # A < 0     ->  1
        # else      -> -1*(CLOSE - DELAY(CLOSE,1))
        self._add_parametric_feature(
            base_name="alpha191_086",
            expr_tpl=(
                "ts_where("
                "  ( (ts_delay(close, 20) - ts_delay(close, 10)) / 10"
                "    - (ts_delay(close, 10) - close) / 10"
                "  ) > 0.25,"
                "  -1,"
                "  ts_where("
                "    ( (ts_delay(close, 20) - ts_delay(close, 10)) / 10"
                "      - (ts_delay(close, 10) - close) / 10"
                "    ) < 0,"
                "    1,"
                "    -1 * (close - ts_delay(close, 1))"
                "  )"
                ")"
            ),
            param_grid={},
        )

        # 087:
        # ((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) +
        #   TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP)
        #                       / (OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1)
        self._add_parametric_feature(
            base_name="alpha191_087",
            expr_tpl=(
                "-1 * ("
                "  cs_rank(ts_decay_linear(ts_delta({vwap}, 4), 7))"
                "  + ts_rank("
                "      ts_decay_linear("
                "        (((low * 0.9) + (low * 0.1)) - {vwap})"
                "        / (open - ((high + low) / 2) + 1e-12),"
                "        11"
                "      ),"
                "      7"
                "    )"
                ")"
            ).format(vwap=vwap),
            param_grid={},
        )

        # 088: (CLOSE - DELAY(CLOSE,20)) / DELAY(CLOSE,20) * 100
        self._add_parametric_feature(
            base_name="alpha191_088",
            expr_tpl=(
                "(close - ts_delay(close, {w_d}))"
                " / (ts_delay(close, {w_d}) + 1e-12) * 100"
            ),
            param_grid={"w_d": [20]},
        )

        # 089:
        # 2 * (SMA(CLOSE,13,2) - SMA(CLOSE,27,2)
        #      - SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
        # 这里所有 Sma 用 ts_mean 近似
        self._add_parametric_feature(
            base_name="alpha191_089",
            expr_tpl=(
                "2 * ("
                "  ts_mean(close, {w_fast})"
                "  - ts_mean(close, {w_slow})"
                "  - ts_mean("
                "      ts_mean(close, {w_fast}) - ts_mean(close, {w_slow}),"
                "      {w_sma}"
                "    )"
                ")"
            ),
            param_grid={"w_fast": [13], "w_slow": [27], "w_sma": [10]},
        )

        # 090: (RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)
        self._add_parametric_feature(
            base_name="alpha191_090",
            expr_tpl=(
                "-1 * cs_rank("
                "  ts_corr(cs_rank({vwap}), cs_rank(volume), {w_corr})"
                ")"
            ).format(vwap=vwap, w_corr="{w_corr}"),
            param_grid={"w_corr": [5]},
        )

    # ========= 091 ～ 100 =========
    def register_091_100(self, windows: list[int] | None = None) -> None:
        """
        对应 Alpha191 中的 Alpha91 ~ Alpha100
        """
        W = windows or self.DEFAULT_WINDOWS
        vwap = "vwap"

        # 091:
        # ((RANK((CLOSE - MAX(CLOSE, 5))) * RANK(CORR(MEAN(VOLUME,40), LOW, 5))) * -1)
        self._add_parametric_feature(
            base_name="alpha191_091",
            expr_tpl=(
                "-1 * cs_rank(close - ts_max(close, {w_max}))"
                "    * cs_rank(ts_corr(ts_mean(volume, {w_mv}), low, {w_corr}))"
            ),
            param_grid={"w_max": [5], "w_mv": [40], "w_corr": [5]},
        )

        # 092:
        # (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE*0.35)+(VWAP*0.65)),2),3)),\
        #      TSRANK(DECAYLINEAR(ABS(CORR(MEAN(VOLUME,180), CLOSE, 13)), 5), 15)) * -1)
        self._add_parametric_feature(
            base_name="alpha191_092",
            expr_tpl=(
                "-1 * ts_greater("
                "  cs_rank("
                "    ts_decay_linear("
                "      ts_delta(close*0.35 + {vwap}*0.65, {w_d1}),"
                "      {w_dec1}"
                "    )"
                "  ),"
                "  ts_rank("
                "    ts_decay_linear("
                "      abs(ts_corr(ts_mean(volume, {w_mv}), close, {w_corr})),"
                "      {w_dec2}"
                "    ),"
                "    {w_tsr}"
                "  )"
                ")"
            ).replace("{vwap}", vwap),
            param_grid={
                "w_d1": [2],
                "w_dec1": [3],
                "w_mv": [180],
                "w_corr": [13],
                "w_dec2": [5],
                "w_tsr": [15],
            },
        )

        # 093:
        # SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)
        dt_expr = (
            "ts_where("
            "  open >= ts_delay(open, 1),"
            "  0,"
            "  ts_greater(open - low, open - ts_delay(open, 1))"
            ")"
        )
        self._add_parametric_feature(
            base_name="alpha191_093",
            expr_tpl=f"ts_sum({dt_expr}, {{w_win}})",
            param_grid={"w_win": [20]},
        )

        # 094:
        # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)
        self._add_parametric_feature(
            base_name="alpha191_094",
            expr_tpl=(
                "ts_sum("
                "  ts_where("
                "    close > ts_delay(close, 1),"
                "    volume,"
                "    ts_where(close < ts_delay(close, 1), -volume, 0)"
                "  ),"
                "  {w_win}"
                ")"
            ),
            param_grid={"w_win": [30]},
        )

        # 095: STD(turnover,20)
        self._add_parametric_feature(
            base_name="alpha191_095",
            expr_tpl="ts_std(turnover, {w_win})",
            param_grid={"w_win": [20]},
        )

        # 096:
        # SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
        # 这里两层 SMA 都用 ts_mean 近似
        inner = (
            "(close - ts_min(low, {w_win}))"
            " / (ts_max(high, {w_win}) - ts_min(low, {w_win}) + 1e-12) * 100"
        )
        self._add_parametric_feature(
            base_name="alpha191_096",
            expr_tpl=(
                "ts_mean("
                "  ts_mean("
                f"    {inner},"
                "    {w_sma1}"
                "  ),"
                "  {w_sma2}"
                ")"
            ),
            param_grid={"w_win": [9], "w_sma1": [3], "w_sma2": [3]},
        )

        # 097: STD(VOLUME,10)
        self._add_parametric_feature(
            base_name="alpha191_097",
            expr_tpl="ts_std(volume, {w_win})",
            param_grid={"w_win": [10]},
        )

        # 098:
        # ((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) <= 0.05)
        #   ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3))))
        self._add_parametric_feature(
            base_name="alpha191_098",
            expr_tpl=(
                "ts_where("
                "  ts_delta(ts_mean(close, {w_mean}), {w_d_long})"
                "    / (ts_delay(close, {w_d_long}) + 1e-12) <= 0.05,"
                "  -1 * (close - ts_min(close, {w_mean})),"
                "  -1 * ts_delta(close, {w_d_short})"
                ")"
            ),
            param_grid={"w_mean": [100], "w_d_long": [100], "w_d_short": [3]},
        )

        # 099: (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5)))
        self._add_parametric_feature(
            base_name="alpha191_099",
            expr_tpl=(
                "-1 * cs_rank("
                "  ts_cov(cs_rank(close), cs_rank(volume), {w_cov})"
                ")"
            ),
            param_grid={"w_cov": [5]},
        )

        # 100: STD(VOLUME,20)
        self._add_parametric_feature(
            base_name="alpha191_100",
            expr_tpl="ts_std(volume, {w_win})",
            param_grid={"w_win": [20]},
        )

    # ========= 101 ～ 110 =========
    def register_101_110(self, windows: list[int] | None = None) -> None:
        """
        对应 Alpha191 中的 Alpha101 ~ Alpha110
        """
        W = windows or self.DEFAULT_WINDOWS
        vwap = "vwap"

        # 101:
        # ((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30), 37), 15))
        #   < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))),
        #               RANK(VOLUME), 11))) ? 1 : 0)
        # 注意：原注释最后有 * -1，但你这份实现返回的是 1/0，这里按代码实现来。
        self._add_parametric_feature(
            base_name="alpha191_101",
            expr_tpl=(
                "ts_where("
                "  cs_rank(ts_corr(close, ts_sum(ts_mean(volume, 30), 37), 15))"
                "  < cs_rank(ts_corr(cs_rank(high*0.1 + {vwap}*0.9),"
                "                     cs_rank(volume), 11)),"
                "  1,"
                "  0"
                ")"
            ).format(vwap=vwap),
            param_grid={},
        )

        # 102:
        # SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)
        # / SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1) * 100
        # 这里还是用 ts_mean 近似 Sma
        self._add_parametric_feature(
            base_name="alpha191_102",
            expr_tpl=(
                "ts_mean("
                "  ts_where(volume - ts_delay(volume, 1) > 0,"
                "           volume - ts_delay(volume, 1),"
                "           0),"
                "  {w_win}"
                ")"
                " / ("
                "  ts_mean(ts_abs(volume - ts_delay(volume, 1)), {w_win})"
                "  + 1e-12"
                ") * 100"
            ),
            param_grid={"w_win": [6]},
        )

        # 103:
        # ((20 - LOWDAY(LOW,20)) / 20) * 100
        # 这里假设后面会在 ts_function/extra_function 里实现 ts_lowday(low, window)
        self._add_parametric_feature(
            base_name="alpha191_103",
            expr_tpl=(
                "({w_win} - ts_lowday(low, {w_win})) / {w_win} * 100"
            ),
            param_grid={"w_win": [20]},
        )

        # 104:
        # (-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))
        self._add_parametric_feature(
            base_name="alpha191_104",
            expr_tpl=(
                "-1 * ("
                "  ts_delta(ts_corr(high, volume, {w_corr}), {w_d})"
                "  * cs_rank(ts_std(close, {w_std}))"
                ")"
            ),
            param_grid={"w_corr": [5], "w_d": [5], "w_std": [20]},
        )

        # 105:
        # (-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))
        self._add_parametric_feature(
            base_name="alpha191_105",
            expr_tpl=(
                "-1 * ts_corr(cs_rank(open), cs_rank(volume), {w_corr})"
            ),
            param_grid={"w_corr": [10]},
        )

        # 106:
        # CLOSE - DELAY(CLOSE,20)
        self._add_parametric_feature(
            base_name="alpha191_106",
            expr_tpl=(
                "close - ts_delay(close, {w_d})"
            ),
            param_grid={"w_d": [20]},
        )

        # 107:
        # (((-1 * RANK(OPEN - DELAY(HIGH,1))) *
        #   RANK(OPEN - DELAY(CLOSE,1))) *
        #   RANK(OPEN - DELAY(LOW,1)))
        self._add_parametric_feature(
            base_name="alpha191_107",
            expr_tpl=(
                "(-1 * cs_rank(open - ts_delay(high, {w_d})))"
                " * cs_rank(open - ts_delay(close, {w_d}))"
                " * cs_rank(open - ts_delay(low, {w_d}))"
            ),
            param_grid={"w_d": [1]},
        )

        # 108:
        # ((RANK(HIGH - TSMIN(HIGH,2)) ^ RANK(CORR(VWAP, MEAN(VOLUME,120), 6))) * -1)
        self._add_parametric_feature(
            base_name="alpha191_108",
            expr_tpl=(
                "-1 * ("
                "  cs_rank(high - ts_min(high, {w_min}))"
                "  ** cs_rank(ts_corr({vwap}, ts_mean(volume, {w_mv}), {w_corr}))"
                ")"
            ).replace("{vwap}", vwap),
            param_grid={"w_min": [2], "w_mv": [120], "w_corr": [6]},
        )

        # 109:
        # SMA(HIGH-LOW,10,2) / SMA(SMA(HIGH-LOW,10,2),10,2)
        # 两层 Sma 都用 ts_mean 近似
        inner_109 = "ts_mean(high - low, {w_sma1})"
        self._add_parametric_feature(
            base_name="alpha191_109",
            expr_tpl=(
                "{inner}"
                " / ts_mean({inner}, {w_sma2})"
            ).replace("{inner}", inner_109),
            param_grid={"w_sma1": [10], "w_sma2": [10]},
        )

        # 110:
        # SUM(MAX(0, HIGH - DELAY(CLOSE,1)),20)
        # / SUM(MAX(0, DELAY(CLOSE,1) - LOW),20) * 100
        self._add_parametric_feature(
            base_name="alpha191_110",
            expr_tpl=(
                "ts_sum("
                "  ts_where("
                "    high - ts_delay(close, {w_d}) > 0,"
                "    high - ts_delay(close, {w_d}),"
                "    0"
                "  ),"
                "  {w_win}"
                ")"
                " / ("
                "  ts_sum("
                "    ts_where("
                "      ts_delay(close, {w_d}) - low > 0,"
                "      ts_delay(close, {w_d}) - low,"
                "      0"
                "    ),"
                "    {w_win}"
                "  )"
                "  + 1e-12"
                ") * 100"
            ),
            param_grid={"w_d": [1], "w_win": [20]},
        )

    # ========= 111 ～ 120 =========
    def register_111_120(self, windows: list[int] | None = None) -> None:
        """
        对应 Alpha191 中的 Alpha111 ~ Alpha120
        公式参考 JoinQuant alpha191 文档。
        """
        W = windows or self.DEFAULT_WINDOWS
        vwap = "vwap"
        ret = self.ret

        # 111:
        # SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)
        #   - SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)
        # 这里 Sma(n, m) 仍然用 ts_mean(..., n) 近似
        base_111 = (
            "volume * ((close - low) - (high - close))"
            " / ((high - low) + 1e-12)"
        )
        self._add_parametric_feature(
            base_name="alpha191_111",
            expr_tpl=(
                "ts_mean("
                f"  {base_111},"
                "  {w_long}"
                ")"
                " - ts_mean("
                f"  {base_111},"
                "  {w_short}"
                ")"
            ),
            param_grid={"w_long": [11], "w_short": [4]},
        )

        # 112:
        # (SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)
        #  - SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))
        # / (同样两部分和的加总) * 100
        self._add_parametric_feature(
            base_name="alpha191_112",
            expr_tpl=(
                " ("
                "   ts_sum("
                "     ts_where(close - ts_delay(close, 1) > 0,"
                "              close - ts_delay(close, 1),"
                "              0),"
                "     {w_win}"
                "   )"
                "   - ts_sum("
                "       ts_where(close - ts_delay(close, 1) < 0,"
                "                abs(close - ts_delay(close, 1)),"
                "                0),"
                "       {w_win}"
                "     )"
                " )"
                " / ("
                "   ts_sum("
                "     ts_where(close - ts_delay(close, 1) > 0,"
                "              close - ts_delay(close, 1),"
                "              0),"
                "     {w_win}"
                "   )"
                "   + ts_sum("
                "       ts_where(close - ts_delay(close, 1) < 0,"
                "                abs(close - ts_delay(close, 1)),"
                "                0),"
                "       {w_win}"
                "     )"
                "   + 1e-12"
                " ) * 100"
            ),
            param_grid={"w_win": [12]},
        )

        # 113:
        # -1 * ((RANK((SUM(DELAY(CLOSE,5),20)/20))
        #        * CORR(CLOSE,VOLUME,2))
        #       * RANK(CORR(SUM(CLOSE,5),SUM(CLOSE,20),2)))
        self._add_parametric_feature(
            base_name="alpha191_113",
            expr_tpl=(
                "-1 * ("
                "  cs_rank("
                "    ts_sum(ts_delay(close, {w_d}), {w_sum}) / {w_sum}"
                "  )"
                "  * ts_corr(close, volume, {w_corr1})"
                "  * cs_rank("
                "      ts_corr("
                "        ts_sum(close, {w_sum5}),"
                "        ts_sum(close, {w_sum20}),"
                "        {w_corr2}"
                "      )"
                "    )"
                ")"
            ),
            param_grid={
                "w_d": [5],
                "w_sum": [20],
                "w_corr1": [2],
                "w_sum5": [5],
                "w_sum20": [20],
                "w_corr2": [2],
            },
        )

        # 114:
        # ((RANK(DELAY(((HIGH-LOW)/(SUM(CLOSE,5)/5)),2))
        #   * RANK(RANK(VOLUME)))
        #  / (((HIGH-LOW)/(SUM(CLOSE,5)/5)) / (VWAP-CLOSE)))
        num114_inner = (
            "(high - low) / (ts_sum(close, {w_sumc}) / {w_sumc} + 1e-12)"
        )
        self._add_parametric_feature(
            base_name="alpha191_114",
            expr_tpl=(
                "("
                "  cs_rank("
                f"    ts_delay({num114_inner}, {{w_d}})"
                "  )"
                "  * cs_rank(cs_rank(volume))"
                ")"
                " / ("
                f"  {num114_inner}"
                "  / ({vwap} - close + 1e-12)"
                "  + 1e-12"
                ")"
            ).format(vwap=vwap, w_sumc="{w_sumc}", w_d="{w_d}"),
            param_grid={"w_sumc": [5], "w_d": [2]},
        )

        # 115:
        # (RANK(CORR(((HIGH*0.9)+(CLOSE*0.1)),MEAN(VOLUME,30),10))
        #  ^ RANK(CORR(TSRANK(((HIGH+LOW)/2),4),TSRANK(VOLUME,10),7)))
        mid = "(high + low) / 2"
        self._add_parametric_feature(
            base_name="alpha191_115",
            expr_tpl=(
                "cs_rank("
                "  ts_corr(high*0.9 + close*0.1,"
                "          ts_mean(volume, {w_mv}),"
                "          {w_corr1})"
                ")"
                " ** cs_rank("
                "      ts_corr("
                "        ts_rank(" + mid + ", {w_tsr1}),"
                "        ts_rank(volume, {w_tsr2}),"
                "        {w_corr2}"
                "      )"
                "    )"
            ),
            param_grid={"w_mv": [30], "w_corr1": [10], "w_tsr1": [4], "w_tsr2": [10], "w_corr2": [7]},
        )

        # 116:
        # REGBETA(CLOSE, SEQUENCE, 20)
        # 用 ta_linearreg_slope(close, 20) 近似回归 beta（需要在 ta_function.py 实现）
        self._add_parametric_feature(
            base_name="alpha191_116",
            expr_tpl="ta_linearreg_slope(close, {w_reg})",
            param_grid={"w_reg": [20]},
        )

        # 117:
        # ((TSRANK(VOLUME,32) * (1 - TSRANK(((CLOSE+HIGH)-LOW),16)))
        #   * (1 - TSRANK(RET,32)))
        self._add_parametric_feature(
            base_name="alpha191_117",
            expr_tpl=(
                "ts_rank(volume, {w_vrank})"
                " * (1 - ts_rank(close + high - low, {w_th}))"
                " * (1 - ts_rank({ret}, {w_r}))"
            ).format(ret=ret, w_vrank="{w_vrank}", w_th="{w_th}", w_r="{w_r}"),
            param_grid={"w_vrank": [32], "w_th": [16], "w_r": [32]},
        )

        # 118:
        # SUM(HIGH-OPEN,20) / SUM(OPEN-LOW,20) * 100
        self._add_parametric_feature(
            base_name="alpha191_118",
            expr_tpl=(
                "ts_sum(high - open, {w_win})"
                " / (ts_sum(open - low, {w_win}) + 1e-12) * 100"
            ),
            param_grid={"w_win": [20]},
        )

        # 119:
        # (RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOLUME,5),26),5),7))
        #  - RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN),
        #                         RANK(MEAN(VOLUME,15)),21),9),7),8)))
        sum_mv5 = "ts_sum(ts_mean(volume, {w_mv5}), {w_sum5})"
        corr1 = f"ts_corr({vwap}, {sum_mv5}, {{w_corr1}})"
        self._add_parametric_feature(
            base_name="alpha191_119",
            expr_tpl=(
                "cs_rank("
                "  ts_decay_linear("
                f"    {corr1},"
                "    {w_dec1}"
                "  )"
                ")"
                " - cs_rank("
                "     ts_decay_linear("
                "       ts_rank("
                "         ts_min("
                "           ts_corr(cs_rank(open),"
                "                   cs_rank(ts_mean(volume, {w_mv15})),"
                "                   {w_corr2}"
                "           ),"
                "           {w_min}"
                "         ),"
                "         {w_tsr}"
                "       ),"
                "       {w_dec2}"
                "     )"
                "   )"
            ),
            param_grid={
                "w_mv5": [5],
                "w_sum5": [26],
                "w_corr1": [5],
                "w_dec1": [7],
                "w_mv15": [15],
                "w_corr2": [21],
                "w_min": [9],
                "w_tsr": [7],
                "w_dec2": [8],
            },
        )

        # 120:
        # (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))
        self._add_parametric_feature(
            base_name="alpha191_120",
            expr_tpl=(
                "cs_rank({vwap} - close)"
                " / (cs_rank({vwap} + close) + 1e-12)"
            ).format(vwap=vwap),
            param_grid={},
        )

    # ========= 121 ～ 130 =========
    def register_121_130(self, windows: list[int] | None = None) -> None:
        """
        Alpha191: 121 ~ 130
        """
        vwap = "vwap"

        # 121: ((RANK(VWAP - MIN(VWAP,12)) ^ TSRANK(CORR(TSRANK(VWAP,20),
        #             TSRANK(MEAN(VOLUME,60),2),18),3)) * -1)
        self._add_parametric_feature(
            base_name="alpha191_121",
            expr_tpl=(
                "-1*("
                "cs_rank({v}-ts_min({v},{w_min}))"
                "**ts_rank(ts_corr(ts_rank({v},20),"
                "                  ts_rank(ts_mean(volume,60),2),"
                "                  {w_corr}),"
                "        {w_tsr})"
                ")"
            ).format(v=vwap, w_min="{w_min}", w_corr="{w_corr}", w_tsr="{w_tsr}"),
            param_grid={"w_min": [12], "w_corr": [18], "w_tsr": [3]},
        )

        # 122: (SMA(SMA(SMA(LOG(C),13,2),13,2),13,2)-DELAY(...,1))/DELAY(...,1)
        base_122 = "ts_mean(ts_mean(ts_mean(ts_log(close),{w_s}),{w_s}),{w_s})"
        self._add_parametric_feature(
            base_name="alpha191_122",
            expr_tpl=(
                f"({base_122}-ts_delay({base_122},{'{w_d}'}))"
                f"/(ts_delay({base_122},{'{w_d}'})+1e-12)"
            ),
            param_grid={"w_s": [13], "w_d": [1]},
        )

        # 123: ((RANK(CORR(SUM((H+L)/2,20),SUM(MEAN(V,60),20),9)) <
        #        RANK(CORR(LOW,V,6))) * -1)  → 输出 {1,0}
        self._add_parametric_feature(
            base_name="alpha191_123",
            expr_tpl=(
                "ts_where("
                "  cs_rank(ts_corr(ts_sum((high+low)/2,{w_sum}),"
                "                  ts_sum(ts_mean(volume,60),{w_sum}),"
                "                  {w_corr1}))"
                "  < cs_rank(ts_corr(low,volume,{w_corr2})),"
                "  -1, 0)"
            ),
            param_grid={"w_sum": [20], "w_corr1": [9], "w_corr2": [6]},
        )

        # 124: (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE,30)),2)
        self._add_parametric_feature(
            base_name="alpha191_124",
            expr_tpl=(
                "(close-{v})/"
                "(ts_decay_linear(cs_rank(ts_max(close,{w_max})),{w_dec})+1e-12)"
            ).format(v=vwap, w_max="{w_max}", w_dec="{w_dec}"),
            param_grid={"w_max": [30], "w_dec": [2]},
        )

        # 125: RANK(DECAYLINEAR(CORR(VWAP,MEAN(V,80),17),20))
        #      / RANK(DECAYLINEAR(DELTA(C*0.5+V*0.5,3),16))
        self._add_parametric_feature(
            base_name="alpha191_125",
            expr_tpl=(
                "cs_rank(ts_decay_linear(ts_corr({v},ts_mean(volume,{w_mv}),{w_corr1}),{w_dec1}))"
                "/(cs_rank(ts_decay_linear(ts_delta(close*0.5+{v}*0.5,{w_d}),{w_dec2}))+1e-12)"
            ).format(v=vwap, w_mv="{w_mv}", w_corr1="{w_corr1}", w_dec1="{w_dec1}",
                     w_d="{w_d}", w_dec2="{w_dec2}"),
            param_grid={
                "w_mv": [80],
                "w_corr1": [17],
                "w_dec1": [20],
                "w_d": [3],
                "w_dec2": [16],
            },
        )

        # 126: (C+H+L)/3   —— 无窗口，直接用
        self._add_parametric_feature(
            base_name="alpha191_126",
            expr_tpl="(close+high+low)/3",
            param_grid=None,
        )

        # 127: (MEAN((100*(C-MAX(C,12))/MAX(C,12))^2,12))^(1/2)
        inner_127 = "100*(close-ts_max(close,{w_max}))/(ts_max(close,{w_max})+1e-12)"
        self._add_parametric_feature(
            base_name="alpha191_127",
            expr_tpl=f"(ts_mean(({inner_127})*({inner_127}),{{w_mean}}))**0.5",
            param_grid={"w_max": [12], "w_mean": [12]},
        )

        # 128: 100 - 100/(1 + SUM(up,14)/SUM(down,14)), up/down 按 hlc3 与 delay 比较
        hlc3 = "(high+low+close)/3"
        up = f"ts_where({hlc3}>ts_delay({hlc3},1),{hlc3}*volume,0)"
        down = f"ts_where({hlc3}<ts_delay({hlc3},1),{hlc3}*volume,0)"
        self._add_parametric_feature(
            base_name="alpha191_128",
            expr_tpl=(
                "100-100/(1+"
                f"ts_sum({up},{{w_win}})/(ts_sum({down},{{w_win}})+1e-12)"
                ")"
            ),
            param_grid={"w_win": [14]},
        )

        # 129: SUM((C-DELTA(C,1)<0 ? ABS(DELTA(C,1)) : 0),12)
        d129 = "close-ts_delay(close,1)"
        self._add_parametric_feature(
            base_name="alpha191_129",
            expr_tpl=(
                "ts_sum(ts_where({d}<0,abs({d}),0),{w_win})"
            ).format(d=d129, w_win="{w_win}"),
            param_grid={"w_win": [12]},
        )

        # 130: RANK(DECAYLINEAR(CORR((H+L)/2,MEAN(V,40),9),10))
        #      / RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(V),7),3))
        mid = "(high+low)/2"
        self._add_parametric_feature(
            base_name="alpha191_130",
            expr_tpl=(
                "cs_rank(ts_decay_linear(ts_corr({m},ts_mean(volume,{w_mv}),{w_corr1}),{w_dec1}))"
                "/(cs_rank(ts_decay_linear(ts_corr(cs_rank({v}),cs_rank(volume),{w_corr2}),{w_dec2}))+1e-12)"
            ).format(m=mid, v=vwap,
                     w_mv="{w_mv}", w_corr1="{w_corr1}", w_dec1="{w_dec1}",
                     w_corr2="{w_corr2}", w_dec2="{w_dec2}"),
            param_grid={
                "w_mv": [40],
                "w_corr1": [9],
                "w_dec1": [10],
                "w_corr2": [7],
                "w_dec2": [3],
            },
        )

    # ========= 131 ～ 140 =========
    def register_131_140(self, windows: list[int] | None = None) -> None:
        """
        Alpha191: 131 ~ 140
        137 / 138 / 140 在原始实现中公式不完整，这里暂时占位为 0
        """
        vwap = "vwap"
        ret = self.ret

        # 131: (RANK(DELTA(VWAP,1)) ^ TSRANK(CORR(CLOSE,MEAN(V,50),18),18))
        self._add_parametric_feature(
            base_name="alpha191_131",
            expr_tpl=(
                "cs_rank(ts_delta({v},{w_d}))"
                "**ts_rank(ts_corr(close,ts_mean(volume,{w_mv}),{w_corr}),{w_tsr})"
            ).format(v=vwap, w_d="{w_d}", w_mv="{w_mv}", w_corr="{w_corr}", w_tsr="{w_tsr}"),
            param_grid={"w_d": [1], "w_mv": [50], "w_corr": [18], "w_tsr": [18]},
        )

        # 132: MEAN(turnover,20)
        self._add_parametric_feature(
            base_name="alpha191_132",
            expr_tpl="ts_mean(turnover,{w_win})",
            param_grid={"w_win": [20]},
        )

        # 133: ((20-HIGHDAY(H,20))/20*100 - (20-LOWDAY(L,20))/20*100)
        # 需要 ts_highday/ts_lowday
        self._add_parametric_feature(
            base_name="alpha191_133",
            expr_tpl=(
                "(({w_win}-ts_highday(high,{w_win}))/{w_win}*100"
                " - ({w_win}-ts_lowday(low,{w_win}))/{w_win}*100)"
            ),
            param_grid={"w_win": [20]},
        )

        # 134: (C-DELAY(C,12))/DELAY(C,12)*VOLUME
        self._add_parametric_feature(
            base_name="alpha191_134",
            expr_tpl=(
                "(close-ts_delay(close,{w_d}))"
                "/(ts_delay(close,{w_d})+1e-12)*volume"
            ),
            param_grid={"w_d": [12]},
        )

        # 135: SMA(DELAY(C/DELAY(C,20),1),20,1) ≈ ts_mean(...)
        r135 = "close/(ts_delay(close,{w_dl})+1e-12)"
        self._add_parametric_feature(
            base_name="alpha191_135",
            expr_tpl=(
                "ts_mean(ts_delay({r},{w_dd}),{w_sma})"
            ).format(r=r135, w_dl="{w_dl}", w_dd="{w_dd}", w_sma="{w_sma}"),
            param_grid={"w_dl": [20], "w_dd": [1], "w_sma": [20]},
        )

        # 136: (-1 * RANK(DELTA(RET,3))) * CORR(OPEN,V,10)
        self._add_parametric_feature(
            base_name="alpha191_136",
            expr_tpl=(
                "(-1*cs_rank(ts_delta({r},{w_d})))"
                "*ts_corr(open,volume,{w_corr})"
            ).format(r=ret, w_d="{w_d}", w_corr="{w_corr}"),
            param_grid={"w_d": [3], "w_corr": [10]},
        )
        
        # 未实现
        # 137: 原文件公式残缺，占位 0
        self._add_parametric_feature(
            base_name="alpha191_137",
            expr_tpl="close-close",
            param_grid={},
        )
        
        # 未实现
        # 138: 原文件公式残缺，占位 0
        self._add_parametric_feature(
            base_name="alpha191_138",
            expr_tpl="close-close",
            param_grid={},
        )

        # 139: (-1 * CORR(OPEN, VOLUME, 10))
        self._add_parametric_feature(
            base_name="alpha191_139",
            expr_tpl="-1*ts_corr(open,volume,{w_corr})",
            param_grid={"w_corr": [10]},
        )
        
        # 未实现
        # 140: 原文件公式残缺，占位 0
        self._add_parametric_feature(
            base_name="alpha191_140",
            expr_tpl="close-close",
            param_grid={},
        )

    # ========= 141 ～ 150 =========
    def register_141_150(self, windows: list[int] | None = None) -> None:
        """
        Alpha191: 141 ~ 150
        143 / 149 在原 alphas191.py 中就是 0，这里也占位 0
        """
        vwap = "vwap"
        ret = self.ret

        # 141: (RANK(CORR(RANK(H), RANK(MEAN(V,15)), 9)) * -1)
        self._add_parametric_feature(
            base_name="alpha191_141",
            expr_tpl=(
                "-1*cs_rank("
                "ts_corr(cs_rank(high),cs_rank(ts_mean(volume,{w_mv})),{w_corr})"
                ")"
            ),
            param_grid={"w_mv": [15], "w_corr": [9]},
        )

        # 142: (-1*RANK(TSRANK(C,10))*RANK(DELTA(DELTA(C,1),1))*RANK(TSRANK(V/MEAN(V,20),5)))
        self._add_parametric_feature(
            base_name="alpha191_142",
            expr_tpl=(
                "(-1*cs_rank(ts_rank(close,{w_tsr_c})))"
                "*cs_rank(ts_delta(ts_delta(close,{w_d1}),{w_d2}))"
                "*cs_rank(ts_rank(volume/(ts_mean(volume,{w_mv})+1e-12),{w_tsr_v}))"
            ),
            param_grid={
                "w_tsr_c": [10],
                "w_d1": [1],
                "w_d2": [1],
                "w_mv": [20],
                "w_tsr_v": [5],
            },
        )
        
        # 未实现
        # 143: 原实现依赖 SELF/FILTER，且代码中直接返回 0，这里占位 0
        self._add_parametric_feature(
            base_name="alpha191_143",
            expr_tpl="close-close",
            param_grid={},
        )

        # 144: SUMIF(val,20, C<DELAY(C,1)) / COUNT(C<DELAY(C,1),20)
        # val = ABS(C/DELAY(C,1)-1)/turnover
        val144 = "abs(close/(ts_delay(close,1)+1e-12)-1)/(turnover+1e-12)"
        self._add_parametric_feature(
            base_name="alpha191_144",
            expr_tpl=(
                "ts_sum(ts_where(close<ts_delay(close,1),{v},0),{w_win})"
                "/(ts_sum(ts_where(close<ts_delay(close,1),1,0),{w_win})+1e-12)"
            ).format(v=val144, w_win="{w_win}"),
            param_grid={"w_win": [20]},
        )

        # 145: (MEAN(V,9)-MEAN(V,26))/MEAN(V,12)*100
        self._add_parametric_feature(
            base_name="alpha191_145",
            expr_tpl=(
                "(ts_mean(volume,{w1})-ts_mean(volume,{w2}))"
                "/(ts_mean(volume,{w3})+1e-12)*100"
            ),
            param_grid={"w1": [9], "w2": [26], "w3": [12]},
        )

        # 146: 标准化动量类，按原实现参数化
        # r = C/DELAY(C,1)-1, m=SMA(r,61,2)≈ts_mean(r,61)
        r146 = "close/(ts_delay(close,1)+1e-12)-1"
        m146 = "ts_mean({r}, {w_m})".format(r=r146, w_m="{w_m}")
        dev146 = f"{r146}-{m146}"
        self._add_parametric_feature(
            base_name="alpha191_146",
            expr_tpl=(
                f"ts_mean({dev146},{'{w_mean}'})*({dev146})"
                f"/(ts_mean({m146}*{m146},{'{w_var}'})+1e-12)"
            ),
            param_grid={"w_m": [61], "w_mean": [20], "w_var": [60]},
        )

        # ta_linearreg_slope未实现
        # 147: REGBETA(MEAN(C,12),SEQUENCE(12)) ≈ ta_linearreg_slope(mean(C,12),12)
        self._add_parametric_feature(
            base_name="alpha191_147",
            expr_tpl="ta_linearreg_slope(ts_mean(close,{w_mean}),{w_reg})",
            param_grid={"w_mean": [12], "w_reg": [12]},
        )

        # 148: (RANK(CORR(O, SUM(MEAN(V,60),9),6)) < RANK(O-TSMIN(O,14)))*-1
        self._add_parametric_feature(
            base_name="alpha191_148",
            expr_tpl=(
                "ts_where("
                "  cs_rank(ts_corr(open,ts_sum(ts_mean(volume,{w_mv}),{w_sum}),{w_corr}))"
                "  < cs_rank(open-ts_min(open,{w_min})),"
                "  -1, 0)"
            ),
            param_grid={"w_mv": [60], "w_sum": [9], "w_corr": [6], "w_min": [14]},
        )
        
        # 未实现
        # 149: 原始实现是 REGBETA(FILTER(...),FILTER(...),252) 且直接 return 0
        self._add_parametric_feature(
            base_name="alpha191_149",
            expr_tpl="close-close",
            param_grid={},
        )

        # 150: (C+H+L)/3 * VOLUME
        self._add_parametric_feature(
            base_name="alpha191_150",
            expr_tpl="(close+high+low)/3*volume",
            param_grid=None,
        )

    # ========= 151 ～ 160 =========
    def register_151_160(self, windows: list[int] | None = None) -> None:
        """
        Alpha191: 151 ~ 160（紧凑 + 参数化）
        """
        vwap = "vwap"
        ret = self.ret

        # 151: SMA(C-DELAY(C,20),20,1)
        self._add_parametric_feature(
            base_name="alpha191_151",
            expr_tpl=(
                "ts_mean(close-ts_delay(close,{w_d}),{w_sma})"
            ),
            param_grid={"w_d": [20], "w_sma": [20]},
        )

        # 152: SMA( MEAN(DELAY(SMA(DELAY(C/DELAY(C,9),1),9,1),1),12)
        #           - MEAN(DELAY(SMA(DELAY(C/DELAY(C,9),1),9,1),1),26), 9,1)
        r = "close/(ts_delay(close,{w_d_c})+1e-12)"
        sma_r = f"ts_mean(ts_delay({r},{{w_delay1}}),{{w_sma_in}})"
        d_sma_r = f"ts_delay({sma_r},{{w_delay2}})"
        self._add_parametric_feature(
            base_name="alpha191_152",
            expr_tpl=(
                "ts_mean("
                f"  ts_mean({d_sma_r},{{w_mean1}})"
                f"  - ts_mean({d_sma_r},{{w_mean2}}),"
                "  {w_sma_out}"
                ")"
            ),
            param_grid={
                "w_d_c": [9],
                "w_delay1": [1],
                "w_sma_in": [9],
                "w_delay2": [1],
                "w_mean1": [12],
                "w_mean2": [26],
                "w_sma_out": [9],
            },
        )

        # 153: (MEAN(C,3)+MEAN(C,6)+MEAN(C,12)+MEAN(C,24))/4
        self._add_parametric_feature(
            base_name="alpha191_153",
            expr_tpl=(
                "(ts_mean(close,{w1})+ts_mean(close,{w2})"
                "+ts_mean(close,{w3})+ts_mean(close,{w4}))/4"
            ),
            param_grid={"w1": [3], "w2": [6], "w3": [12], "w4": [24]},
        )

        # 154: ((VWAP-MIN(VWAP,16)) < CORR(VWAP,MEAN(V,180),18)) → 1/0
        self._add_parametric_feature(
            base_name="alpha191_154",
            expr_tpl=(
                "ts_where("
                "  {v}-ts_min({v},{w_min})"
                "    < ts_corr({v},ts_mean(volume,{w_mv}),{w_corr}),"
                "  1, 0)"
            ).replace("{v}", vwap),
            param_grid={"w_min": [16], "w_mv": [180], "w_corr": [18]},
        )

        # 155: SMA(V,13,2)-SMA(V,27,2)-SMA(SMA(V,13,2)-SMA(V,27,2),10,2)
        self._add_parametric_feature(
            base_name="alpha191_155",
            expr_tpl=(
                "ts_mean(volume,{w_fast})-ts_mean(volume,{w_slow})"
                "-ts_mean(ts_mean(volume,{w_fast})-ts_mean(volume,{w_slow}),{w_sma})"
            ),
            param_grid={"w_fast": [13], "w_slow": [27], "w_sma": [10]},
        )

        # 156: -MAX( RANK(DECAYLINEAR(DELTA(VWAP,5),3)),
        #            RANK(DECAYLINEAR(-DELTA(0.15*O+0.85*L,2)/(0.15*O+0.85*L),3)) )
        mix_ol = "open*0.15+low*0.85"
        self._add_parametric_feature(
            base_name="alpha191_156",
            expr_tpl=(
                "-1*ts_greater("
                "  cs_rank(ts_decay_linear(ts_delta({v},{w_dv}),{w_decv})),"
                "  cs_rank(ts_decay_linear("
                "    -1*ts_delta({m},{w_dm})/({m}+1e-12),"
                "    {w_decm}"
                "  ))"
                ")"
            ).replace("{v}", vwap).replace("{m}", mix_ol),
            param_grid={"w_dv": [5], "w_decv": [3], "w_dm": [2], "w_decm": [3]},
        )

        # 157: MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK(-RANK(DELTA(C-1,5)))),2),1)))),1),5)
        #      + TSRANK(DELAY(-RET,6),5)
        u = "-1*cs_rank(ts_delta(close-1,{w_d_rc}))"
        r2 = f"cs_rank(cs_rank(ts_log(ts_sum(ts_min({u},{{w_tmin}}),{{w_tsum}}))))"
        prod = f"ts_prod({r2},{{w_prod}})"
        self._add_parametric_feature(
            base_name="alpha191_157",
            expr_tpl=(
                "ts_min("
                f"  {prod},"
                "  {w_minp}"
                ")"
                "+ts_rank(ts_delay(-1*{ret},{w_d_ret}),{w_tsr})"
            ).replace("{ret}", ret),
            param_grid={
                "w_d_rc": [5],
                "w_tmin": [2],
                "w_tsum": [1],
                "w_prod": [1],
                "w_minp": [5],
                "w_d_ret": [6],
                "w_tsr": [5],
            },
        )

        # 158: ((H-SMA(C,15,2))-(L-SMA(C,15,2)))/C
        self._add_parametric_feature(
            base_name="alpha191_158",
            expr_tpl=(
                "((high-ts_mean(close,{w_sma}))-(low-ts_mean(close,{w_sma})))"
                "/(close+1e-12)"
            ),
            param_grid={"w_sma": [15]},
        )

        # 159: 三个不同窗长的 K 线振幅加权组合（原始 6/12/24）
        min_ld = "ts_less(low,ts_delay(close,1))"
        rng = "ts_greater(high,ts_delay(close,1))-ts_less(low,ts_delay(close,1))"
        term6 = "(close-ts_sum({m},{w1}))/(ts_sum({r},{w1})+1e-12)*12*24"
        term12 = "(close-ts_sum({m},{w2}))/(ts_sum({r},{w2})+1e-12)*6*24"
        term24 = "(close-ts_sum({m},{w3}))/(ts_sum({r},{w3})+1e-12)*6*24"
        expr_159 = (
            "("
            + term6.format(m=min_ld, r=rng, w1="{w1}", w2="{w2}", w3="{w3}")
            + "+"
            + term12.format(m=min_ld, r=rng, w1="{w1}", w2="{w2}", w3="{w3}")
            + "+"
            + term24.format(m=min_ld, r=rng, w1="{w1}", w2="{w2}", w3="{w3}")
            + ")*100/(6*12+6*24+12*24)"
        )
        self._add_parametric_feature(
            base_name="alpha191_159",
            expr_tpl=expr_159,
            param_grid={"w1": [6], "w2": [12], "w3": [24]},
        )

        # 160: SMA((C<=DELAY(C,1)?STD(C,20):0),20,1)
        self._add_parametric_feature(
            base_name="alpha191_160",
            expr_tpl=(
                "ts_mean(ts_where(close<=ts_delay(close,{w_d}),ts_std(close,{w_std}),0),{w_sma})"
            ),
            param_grid={"w_d": [1], "w_std": [20], "w_sma": [20]},
        )

    # ========= 161 ～ 170 =========
    def register_161_170(self, windows: list[int] | None = None) -> None:
        """
        Alpha191: 161 ~ 170（紧凑 + 参数化）
        """
        vwap = "vwap"
        ret = self.ret

        # 161: MEAN(MAX(MAX(H-L,ABS(DELAY(C,1)-H)),ABS(DELAY(C,1)-L)),12)
        self._add_parametric_feature(
            base_name="alpha191_161",
            expr_tpl=(
                "ts_mean("
                "  ts_greater(ts_greater(high-low,abs(ts_delay(close,{w_d})-high)),"
                "             abs(ts_delay(close,{w_d})-low)),"
                "  {w_win}"
                ")"
            ),
            param_grid={"w_d": [1], "w_win": [12]},
        )

        # 162: RSV-K 类 (RSV=(SMA(up,12)/SMA(abs,12))*100)，再做 (RSV-min)/(max-min)
        up = (
            "ts_mean(ts_where(close-ts_delay(close,1)>0,"
            "                  close-ts_delay(close,1),0),"
            "{w_rwin})"
        )
        dn = "ts_mean(abs(close-ts_delay(close,1)),{w_rwin})+1e-12"
        rsv = f"{up}/{dn}*100"
        self._add_parametric_feature(
            base_name="alpha191_162",
            expr_tpl=(
                f"({rsv}-ts_min({rsv},{{w_norm}}))"
                f"/(ts_max({rsv},{{w_norm}})-ts_min({rsv},{{w_norm}})+1e-12)"
            ),
            param_grid={"w_rwin": [12], "w_norm": [12]},
        )

        # 163: RANK(((-RET)*MEAN(V,20))*VWAP*(H-C))
        self._add_parametric_feature(
            base_name="alpha191_163",
            expr_tpl=(
                "cs_rank(((-1*{r})*ts_mean(volume,{w_mv}))*{v}*(high-close))"
            ).replace("{r}", ret).replace("{v}", vwap),
            param_grid={"w_mv": [20]},
        )

        # 164: RSV 型：x=(C>DELAY(C,1)?1/(C-DELTA(C,1)):1),
        #       之后 ((x-MIN(x,12))/(H-L)*100) 再 13 日 SMA
        x = (
            "ts_where(close>ts_delay(close,1),"
            "         1/(close-ts_delay(close,1)+1e-12),"
            "         1)"
        )
        self._add_parametric_feature(
            base_name="alpha191_164",
            expr_tpl=(
                "ts_mean((({x}-ts_min({x},{w_min}))"
                "/(high-low+1e-12)*100),{w_sma})"
            ).replace("{x}", x),
            param_grid={"w_min": [12], "w_sma": [13]},
        )
        
        # 未实现
        # 165: 聚宽原文“尚未实现”，这里占位 0
        self._add_parametric_feature(
            base_name="alpha191_165",
            expr_tpl="close-close",
            param_grid={},
        )

        # 166: 高阶标准化收益，原始参数 20
        r166 = "close/(ts_delay(close,1)+1e-12)-1"
        m166 = "ts_mean({r},{w_m})".format(r=r166, w_m="{w_m}")
        num166 = f"-20*(20-1)**1.5*ts_sum({r166}-{m166},{{w_win}})"
        den166 = (
            "( (20-1)*(20-2)"
            "  * (ts_sum(ts_mean(close/(ts_delay(close,1)+1e-12),{w_mean})**2,{w_win}))**1.5"
            ")"
        )
        self._add_parametric_feature(
            base_name="alpha191_166",
            expr_tpl=f"{num166}/({den166}+1e-12)",
            param_grid={"w_m": [20], "w_win": [20], "w_mean": [20]},
        )

        # 167: SUM(C>DELAY(C,1)?C-DELTA(C,1):0,12)
        self._add_parametric_feature(
            base_name="alpha191_167",
            expr_tpl=(
                "ts_sum(ts_where(close>ts_delay(close,1),"
                "                 close-ts_delay(close,1),0),"
                "       {w_win})"
            ),
            param_grid={"w_win": [12]},
        )

        # 168: (-1*V/MEAN(V,20))
        self._add_parametric_feature(
            base_name="alpha191_168",
            expr_tpl="-1*volume/(ts_mean(volume,{w_mv})+1e-12)",
            param_grid={"w_mv": [20]},
        )

        # 169: SMA( mean(delay(SMA(C-DELTA(C,1),9,1),1),12) - mean(...,26), 10,1)
        r169 = "close-ts_delay(close,1)"
        sma9 = "ts_mean({r},{w_in})".format(r=r169, w_in="{w_in}")
        d_sma9 = "ts_delay({s},{w_delay})".format(s=sma9, w_delay="{w_delay}")
        self._add_parametric_feature(
            base_name="alpha191_169",
            expr_tpl=(
                "ts_mean("
                f"  ts_mean({d_sma9},{{w_mean1}})"
                f"  - ts_mean({d_sma9},{{w_mean2}}),"
                "  {w_sma}"
                ")"
            ),
            param_grid={
                "w_in": [9],
                "w_delay": [1],
                "w_mean1": [12],
                "w_mean2": [26],
                "w_sma": [10],
            },
        )

        # 170: ((RANK(1/C)*V/MEAN(V,20)) * ((H*RANK(H-C))/(SUM(H,5)/5))) - RANK(VWAP-DELTA(VWAP,5))
        self._add_parametric_feature(
            base_name="alpha191_170",
            expr_tpl=(
                "(cs_rank(1/close)*volume/(ts_mean(volume,{w_mv})+1e-12)"
                "* (high*cs_rank(high-close)/(ts_sum(high,{w_hwin})/{w_hwin}+1e-12)))"
                " - cs_rank({v}-ts_delay({v},{w_dv}))"
            ).replace("{v}", vwap),
            param_grid={"w_mv": [20], "w_hwin": [5], "w_dv": [5]},
        )


    # ========= 171 ～ 180 =========
    def register_171_180(self, windows: list[int] | None = None) -> None:
        """
        Alpha191: 171 ~ 180（紧凑 + 参数化）
        """
        vwap = "vwap"

        # 171: (-1 * ((L-C)*(O^5)) / ((C-H)*(C^5)))
        self._add_parametric_feature(
            base_name="alpha191_171",
            expr_tpl="-1*((low-close)*(open**5))/((close-high)*(close**5)+1e-12)",
            param_grid=None,
        )

        # 172:
        # TR = max(max(H-L, |delay(C,1)-H|), |delay(C,1)-L|)
        # LD = delay(L,1)-L, HD = H-delay(H,1)
        # mean( | sum(LD>0 & LD>HD ? LD:0,14)*100/sum(TR,14)
        #        -sum(HD>0 & HD>LD ? HD:0,14)*100/sum(TR,14) | *100, 6)
        TR = (
            "ts_greater(ts_greater(high-low,abs(ts_delay(close,{w_d_c})-high)),"
            "           abs(low-ts_delay(close,{w_d_c})))"
        )
        LD = "ts_delay(low,{w_d_l})-low"
        HD = "high-ts_delay(high,{w_d_h})"
        up = f"ts_where(({LD}>0)&({LD}>{HD}),{LD},0)"
        dn = f"ts_where(({HD}>0)&({HD}>{LD}),{HD},0)"
        num_up = f"ts_sum({up},{{w_tr}})*100/ts_sum({TR},{{w_tr}})"
        num_dn = f"ts_sum({dn},{{w_tr}})*100/ts_sum({TR},{{w_tr}})"
        self._add_parametric_feature(
            base_name="alpha191_172",
            expr_tpl=(
                f"ts_mean(abs({num_up}-{num_dn})*100,{{w_mean}})"
            ),
            param_grid={"w_d_c": [1], "w_d_l": [1], "w_d_h": [1], "w_tr": [14], "w_mean": [6]},
        )

        # 173: 3*SMA(C,13,2) - 2*SMA(SMA(C,13,2),13,2) + SMA(SMA(SMA(LOG(C),13,2),13,2),13,2)
        base_c = "ts_mean(close,{w_s})"
        base_l = "ts_mean(ts_log(close),{w_s})"
        self._add_parametric_feature(
            base_name="alpha191_173",
            expr_tpl=(
                f"3*{base_c}"
                f"-2*ts_mean({base_c},{'{w_s2}'})"
                f"+ts_mean(ts_mean(ts_mean({base_l},{'{w_s2}'}),{{w_s2}}),{{w_s2}})"
            ),
            param_grid={"w_s": [13], "w_s2": [13]},
        )

        # 174: SMA(C<=DELAY(C,1)?STD(C,20):0,20,1)
        self._add_parametric_feature(
            base_name="alpha191_174",
            expr_tpl=(
                "ts_mean("
                "  ts_where(close<=ts_delay(close,{w_d}),ts_std(close,{w_std}),0),"
                "  {w_sma}"
                ")"
            ),
            param_grid={"w_d": [1], "w_std": [20], "w_sma": [20]},
        )

        # 175: MEAN(MAX(MAX(H-L,ABS(DELAY(C,1)-H)),ABS(DELAY(C,1)-L)),6)
        self._add_parametric_feature(
            base_name="alpha191_175",
            expr_tpl=(
                "ts_mean("
                "  ts_greater(ts_greater(high-low,abs(ts_delay(close,{w_d})-high)),"
                "             abs(ts_delay(close,{w_d})-low)),"
                "  {w_win}"
                ")"
            ),
            param_grid={"w_d": [1], "w_win": [6]},
        )

        # 176: CORR(RANK((C-TSMIN(L,12))/(TSMAX(H,12)-TSMIN(L,12))), RANK(V), 6)
        self._add_parametric_feature(
            base_name="alpha191_176",
            expr_tpl=(
                "ts_corr("
                "  cs_rank((close-ts_min(low,{w_hl}))/(ts_max(high,{w_hl})-ts_min(low,{w_hl})+1e-12)),"
                "  cs_rank(volume),"
                "  {w_corr}"
                ")"
            ),
            param_grid={"w_hl": [12], "w_corr": [6]},
        )

        # 177: ((20-HIGHDAY(H,20))/20)*100
        self._add_parametric_feature(
            base_name="alpha191_177",
            expr_tpl="(20-ts_highday(high,{w_win}))/20*100",
            param_grid={"w_win": [20]},
        )

        # 178: (C-DELAY(C,1))/DELAY(C,1)*V
        self._add_parametric_feature(
            base_name="alpha191_178",
            expr_tpl=(
                "(close-ts_delay(close,{w_d}))"
                "/(ts_delay(close,{w_d})+1e-12)*volume"
            ),
            param_grid={"w_d": [1]},
        )

        # 179: RANK(CORR(VWAP,V,4))*RANK(CORR(RANK(L),RANK(MEAN(V,50)),12))
        self._add_parametric_feature(
            base_name="alpha191_179",
            expr_tpl=(
                "cs_rank(ts_corr({v},volume,{w_c1}))"
                "*cs_rank(ts_corr(cs_rank(low),cs_rank(ts_mean(volume,{w_mv})),{w_c2}))"
            ).replace("{v}", vwap),
            param_grid={"w_c1": [4], "w_mv": [50], "w_c2": [12]},
        )

        # 180: (MEAN(V,20)<V ? -TSRANK(ABS(DELTA(C,7)),60)*SIGN(DELTA(C,7)) : -V)
        self._add_parametric_feature(
            base_name="alpha191_180",
            expr_tpl=(
                "ts_where("
                "  ts_mean(volume,{w_mv})<volume,"
                "  -1*ts_rank(abs(ts_delta(close,{w_d})),{w_tsr})*ts_sign(ts_delta(close,{w_d})),"
                "  -1*volume)"
            ),
            param_grid={"w_mv": [20], "w_d": [7], "w_tsr": [60]},
        )

    # ========= 181 ～ 191 =========
    def register_181_191(self, windows: list[int] | None = None) -> None:
        """
        Alpha191: 181 ~ 191
        181 / 182 / 186 / 190 原文件公式残缺或直接 return 0，这里占位 0。
        """
        
        # 未实现
        # 181: 占位 0
        self._add_parametric_feature(
            base_name="alpha191_181",
            expr_tpl="close-close",
            param_grid={},
        )
        
        # 未实现
        # 182: 原注释/实现均不完整，保守起见占位 0
        self._add_parametric_feature(
            base_name="alpha191_182",
            expr_tpl="close-close",
            param_grid={},
        )

        # 183: (MAX(cum)-MIN(cum))/STD(C,24), 其中 cum=SUM(C-MEAN(C,24),24)
        expr_183 = (
            "(ts_max(ts_sum(close-ts_mean(close,{w_mean}),{w_win}),{w_win})"
            " - ts_min(ts_sum(close-ts_mean(close,{w_mean}),{w_win}),{w_win}))"
            " / (ts_std(close,{w_win})+1e-12)"
        )
        self._add_parametric_feature(
            base_name="alpha191_183",
            expr_tpl=expr_183,
            param_grid={"w_mean": [24], "w_win": [24]},
        )

        # 184: RANK(CORR(DELAY(O-C,1),C,200)) + RANK(O-C)
        self._add_parametric_feature(
            base_name="alpha191_184",
            expr_tpl=(
                "cs_rank(ts_corr(ts_delay(open-close,{w_d}),close,{w_corr}))"
                "+cs_rank(open-close)"
            ),
            param_grid={"w_d": [1], "w_corr": [200]},
        )

        # 185: RANK(-1*(1-O/C)^2)
        self._add_parametric_feature(
            base_name="alpha191_185",
            expr_tpl="cs_rank(-1*((1-open/(close+1e-12))**2))",
            param_grid=None,
        )
        
        # 未实现
        # 186: 原文件公式残缺，占位 0
        self._add_parametric_feature(
            base_name="alpha191_186",
            expr_tpl="close-close",
            param_grid={},
        )

        # 187: SUM(O<=DELAY(O,1)?0:MAX(H-O,O-DELAY(O,1)),20)
        self._add_parametric_feature(
            base_name="alpha191_187",
            expr_tpl=(
                "ts_sum("
                "  ts_where(open<=ts_delay(open,{w_d}),"
                "           0,"
                "           ts_greater(high-open,open-ts_delay(open,{w_d}))),"
                "  {w_win}"
                ")"
            ),
            param_grid={"w_d": [1], "w_win": [20]},
        )

        # 188: ((H-L)-SMA(H-L,11,2))/SMA(H-L,11,2)*100
        sma_hl = "ts_mean(high-low,{w_sma})"
        self._add_parametric_feature(
            base_name="alpha191_188",
            expr_tpl=f"((high-low)-{sma_hl})/({sma_hl}+1e-12)*100",
            param_grid={"w_sma": [11]},
        )

        # 189: MEAN(ABS(C-MEAN(C,6)),6)
        self._add_parametric_feature(
            base_name="alpha191_189",
            expr_tpl="ts_mean(abs(close-ts_mean(close,{w_mean})),{w_win})",
            param_grid={"w_mean": [6], "w_win": [6]},
        )
        
        # 未实现
        # 190: 原 alphas191.py 中就是 return 0
        self._add_parametric_feature(
            base_name="alpha191_190",
            expr_tpl="close-close",
            param_grid={},
        )

        # 191: (CORR(MEAN(V,20),L,5) + (H+L)/2 - C)
        self._add_parametric_feature(
            base_name="alpha191_191",
            expr_tpl=(
                "ts_corr(ts_mean(volume,{w_mv}),low,{w_corr})"
                "+(high+low)/2-close"
            ),
            param_grid={"w_mv": [20], "w_corr": [5]},
        )


    def register_all(self, windows: list[int] | None = None) -> None:
        self.register_001_010(windows)
        self.register_011_020(windows)
        self.register_021_030(windows)
        self.register_031_040(windows)
        self.register_041_050(windows)
        self.register_051_060(windows)
        self.register_061_070(windows)
        self.register_071_080(windows)
        self.register_081_090(windows)
        self.register_091_100(windows)
        self.register_101_110(windows)
        self.register_111_120(windows)
        self.register_121_130(windows)
        self.register_131_140(windows)
        self.register_141_150(windows)
        self.register_151_160(windows)
        self.register_161_170(windows)
        self.register_171_180(windows)
        self.register_181_191(windows)
