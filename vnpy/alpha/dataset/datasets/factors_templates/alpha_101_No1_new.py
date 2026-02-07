# alpha_101_dataset_parametric.py
# 新版 Alpha101：参数化（多参数）注册版
# 说明：
# - 仅在“注册阶段”做参数展开；执行仍依赖你现有的表达式引擎与算子（ts_*/cs_*/ta_*、extra.ts_where 等）
# - 命名规范：<base>__k1=v1__k2=v2，键按字典序，值做合适格式化
# - 本文件实现 #001～#020，后续可按同样模式继续补齐

from itertools import product
from typing import Any, Callable, Iterable

from vnpy.alpha.dataset.feature_spec import BaseFeatureSpec


class Alpha101NEW(BaseFeatureSpec):
    """
    用于注册 Alpha101 的表达式（字符串），
    通过 self.add_feature(name: str, expr: str) 写入到本对象的 feature_expressions 中。

    使用方法（示例）：
      spec = Alpha101NEW().build(windows=[5, 10, 20, 30])
    """

    DEFAULT_WINDOWS: list[int] = [5, 10, 20, 30]

    def register(self, *, windows: list[int] | None = None) -> None:
        self.require_bars("open", "high", "low", "close", "volume", "vwap", "turnover")
        self.register_all(windows=windows)


    # ============ 工具：多参数展开 ============
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
        bars_cols: Iterable[str] = (),
        exposure_cols: Iterable[str] = (),
    ) -> None:
        if not param_grid:
            self.add_feature(
                base_name,
                expr_tpl,
                bars_cols=bars_cols,
                exposure_cols=exposure_cols,
            )
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
            self.add_feature(
                name,
                expr,
                bars_cols=bars_cols,
                exposure_cols=exposure_cols,
            )
            seen.add(name)


    # ============ 公共片段（便于复用） ============
    @property
    def ret(self) -> str:
        return "(close / ts_delay(close, 1) - 1)"

    @property
    def adv20(self) -> str:
        return "ts_mean(volume, 20)"

    # ============ 001 ～ 010 ============
    def register_001_010(self, windows: list[int] | None = None) -> None:
        W = windows or self.DEFAULT_WINDOWS
        ret = self.ret
        vwap = "vwap"

        # 001: (rank(Ts_ArgMax(SignedPower(((ret<0)? stddev(ret, 20) : close), 2.), 5)) - 0.5)
        self._add_parametric_feature(
            base_name="alpha101_001",
            expr_tpl=(
                "cs_rank("
                "  ts_argmax("
                "    (ts_where(({ret}) < 0, ts_std({ret}, {w_std}), close)) ** 2,"
                "    {w_arg}"
                "  )"
                ") - 0.5"
            ).format(ret=ret, w_std="{w_std}", w_arg="{w_arg}"),
            param_grid={"w_std": [10, 20, 30], "w_arg": [5, 10]},
        )

        # 002: -corr(rank(delta(log(volume),2)), rank((close-open)/open), w)
        self._add_parametric_feature(
            base_name="alpha101_002",
            expr_tpl=(
                "-1 * ts_corr("
                "  cs_rank(ts_log(volume) - ts_delay(ts_log(volume), 2)),"
                "  cs_rank((close - open) / open),"
                "  {w}"
                ")"
            ),
            param_grid={"w": [6, 10, 20]},
        )

        # 003: -corr(rank(open), rank(volume), w)
        self._add_parametric_feature(
            base_name="alpha101_003",
            expr_tpl="-1 * ts_corr(cs_rank(open), cs_rank(volume), {w})",
            param_grid={"w": [10, 20, 30]},
        )

        # 004: -ts_rank(rank(low), w)
        self._add_parametric_feature(
            base_name="alpha101_004",
            expr_tpl="-1 * ts_rank(cs_rank(low), {w})",
            param_grid={"w": W},
        )

        # 005: rank(open - sum(vwap,10)/10) * (-1 * abs(rank(close - vwap)))
        self._add_parametric_feature(
            base_name="alpha101_005",
            expr_tpl=(
                "cs_rank(open - ts_mean({vwap}, {w_ma}))"
                " * (-1 * abs(cs_rank(close - {vwap})))"
            ).format(vwap=vwap, w_ma="{w_ma}"),
            param_grid={"w_ma": [10, 20, 30]},
        )

        # 006: -corr(open, volume, w)
        self._add_parametric_feature(
            base_name="alpha101_006",
            expr_tpl="-1 * ts_corr(open, volume, {w})",
            param_grid={"w": [10, 20, 30]},
        )

        # 007: (adv20 < volume) ? ((-1 * ts_rank(abs(delta(close,7)),60)) * sign(delta(close,7))) : -1
        # 使用 ts_where 实现条件
        self._add_parametric_feature(
            base_name="alpha101_007",
            expr_tpl=(
                "ts_where("
                "  volume > ts_mean(volume, {w_adv}),"
                "  (-1 * ts_rank(abs(close - ts_delay(close, 7)), {w_rank})) * ts_sign(close - ts_delay(close, 7)),"
                "  -1"
                ")"
            ),
            param_grid={"w_adv": [20], "w_rank": [60]},
        )

        # 008: -rank(((sum(open,5) * sum(returns,5)) - delay((sum(open,5) * sum(returns,5)),10)))
        self._add_parametric_feature(
            base_name="alpha101_008",
            expr_tpl=(
                "-1 * cs_rank( (ts_sum(open, {w1}) * ts_sum({ret}, {w1})) - ts_delay((ts_sum(open, {w1}) * ts_sum({ret}, {w1})), {w2}) )"
            ).format(ret=ret, w1="{w1}", w2="{w2}"),
            param_grid={"w1": [5], "w2": [10]},
        )

        # 009: 条件翻转：
        # ((0 < ts_min(delta(close,1),5)) ? delta(close,1) : ((ts_max(delta(close,1),5) < 0) ? delta(close,1) : (-1 * delta(close,1))))
        self._add_parametric_feature(
            base_name="alpha101_009",
            expr_tpl=(
                "ts_where("
                "  ts_min(close - ts_delay(close,1), {w}) > 0,"
                "  (close - ts_delay(close,1)),"
                "  ts_where("
                "    ts_max(close - ts_delay(close,1), {w}) < 0,"
                "    (close - ts_delay(close,1)),"
                "    (-1 * (close - ts_delay(close,1)))"
                "  )"
                ")"
            ),
            param_grid={"w": [5]},
        )

        # 010: rank(条件翻转 with window=4)
        self._add_parametric_feature(
            base_name="alpha101_010",
            expr_tpl=(
                "cs_rank("
                "  ts_where("
                "    ts_min(close - ts_delay(close,1), {w}) > 0,"
                "    (close - ts_delay(close,1)),"
                "    ts_where("
                "      ts_max(close - ts_delay(close,1), {w}) < 0,"
                "      (close - ts_delay(close,1)),"
                "      (-1 * (close - ts_delay(close,1)))"
                "    )"
                "  )"
                ")"
            ),
            param_grid={"w": [4]},
        )

    def register_011_020(self, windows: list[int] | None = None) -> None:
        W = windows or self.DEFAULT_WINDOWS
        ret = self.ret
        vwap = "vwap"

        # 011: ((rank(ts_max(vwap-close,3)) + rank(ts_min(vwap-close,3))) * rank(delta(volume,3)))
        self._add_parametric_feature(
            base_name="alpha101_011",
            expr_tpl=(
                "( cs_rank(ts_max({vwap} - close, {w_ex})) + cs_rank(ts_min({vwap} - close, {w_ex})) )"
                " * cs_rank(volume - ts_delay(volume, {w_d}))"
            ).format(vwap=vwap, w_ex="{w_ex}", w_d="{w_d}"),
            param_grid={"w_ex": [3], "w_d": [3]},
        )

        # 012: sign(delta(volume,1)) * (-1 * delta(close,1))
        self._add_parametric_feature(
            base_name="alpha101_012",
            expr_tpl="ts_sign(volume - ts_delay(volume,1)) * (-1 * (close - ts_delay(close,1)))",
            param_grid=None,
        )

        # 013: -rank(covariance(rank(close), rank(volume), 5))
        self._add_parametric_feature(
            base_name="alpha101_013",
            expr_tpl="-1 * cs_rank(ts_cov(cs_rank(close), cs_rank(volume), {w}))",
            param_grid={"w": [5, 10]},
        )

        # 014: ((-1 * rank(delta(returns,3))) * corr(open, volume, 10))
        self._add_parametric_feature(
            base_name="alpha101_014",
            expr_tpl=(
                "(-1 * cs_rank({ret} - ts_delay({ret}, 3))) * ts_corr(open, volume, {w})"
            ).format(ret=ret, w="{w}"),
            param_grid={"w": [10, 20]},
        )

        # 015: -sum(rank(corr(rank(high), rank(volume), 3)), 3)
        self._add_parametric_feature(
            base_name="alpha101_015",
            expr_tpl=(
                "-1 * ts_sum( cs_rank( ts_corr(cs_rank(high), cs_rank(volume), {w_c}) ), {w_s})"
            ),
            param_grid={"w_c": [3], "w_s": [3]},
        )

        # 016: -rank(covariance(rank(high), rank(volume), 5))
        self._add_parametric_feature(
            base_name="alpha101_016",
            expr_tpl="-1 * cs_rank(ts_cov(cs_rank(high), cs_rank(volume), {w}))",
            param_grid={"w": [5, 10]},
        )

        # 017: ((-1 * rank(ts_rank(close,10))) * rank(delta(delta(close,1),1))) * rank(ts_rank(volume/adv20,5))
        self._add_parametric_feature(
            base_name="alpha101_017",
            expr_tpl=(
                "(-1 * cs_rank(ts_rank(close, {w1}))) * cs_rank((close - 2*ts_delay(close,1) + ts_delay(close,2)))"
                " * cs_rank(ts_rank(volume / ts_mean(volume, {w_adv}), {w2}))"
            ),
            param_grid={"w1": [10], "w2": [5], "w_adv": [20, 60]},
        )

        # 018: -rank( stddev(abs(close-open),5) + (close-open) + corr(close, open, 10) )
        self._add_parametric_feature(
            base_name="alpha101_018",
            expr_tpl=(
                "-1 * cs_rank( ts_std(abs(close - open), {w_std}) + (close - open) + ts_corr(close, open, {w_corr}) )"
            ),
            param_grid={"w_std": [5, 10], "w_corr": [10, 20]},
        )

        # 019: (-1 * sign((close - delay(close,7)) + delta(close,7))) * (1 + rank(1 + sum(returns,250)))
        self._add_parametric_feature(
            base_name="alpha101_019",
            expr_tpl=(
                "(-1 * ts_sign( (close - ts_delay(close, {w_d})) + (close - ts_delay(close, {w_d})) )) * (1 + cs_rank(1 + ts_sum({ret}, {w_sum})))"
            ).format(ret=ret, w_d="{w_d}", w_sum="{w_sum}"),
            param_grid={"w_d": [7], "w_sum": [250]},
        )

        # 020: (-1) * rank(open - delay(high,1)) * rank(open - delay(close,1)) * rank(open - delay(low,1))
        self._add_parametric_feature(
            base_name="alpha101_020",
            expr_tpl=(
                "-1 * cs_rank(open - ts_delay(high,1)) * cs_rank(open - ts_delay(close,1)) * cs_rank(open - ts_delay(low,1))"
            ),
            param_grid=None,
        )

    def register_021_030(self, windows: list[int] | None = None) -> None:
        W = windows or self.DEFAULT_WINDOWS
        ret = self.ret
        vwap = "vwap"

        # 021: 三条件布尔 → {-1, 1}
        self._add_parametric_feature(
            base_name="alpha101_021",
            expr_tpl=(
                "( ts_where("
                "    (ts_mean(close, {w_slow}) + ts_std(close, {w_std}) < ts_mean(close, {w_fast})),"
                "    -1,"
                "    ts_where("
                "      (ts_mean(close, {w_fast}) < ts_mean(close, {w_slow}) - ts_std(close, {w_std})) | (ts_mean(volume, {w_adv}) / volume >= 1),"
                "      1,"
                "      -1"
                "    )"
                "  ) )"
            ),
            param_grid={"w_fast": [2], "w_slow": [8], "w_std": [8], "w_adv": [20]},
        )

        # 022: - delta(corr(high,volume,wc), wd) * rank(stddev(close, ws))
        self._add_parametric_feature(
            base_name="alpha101_022",
            expr_tpl=(
                "-1 * ( ts_delta( ts_corr(high, volume, {w_corr}), {w_delta}) * cs_rank(ts_std(close, {w_std})) )"
            ),
            param_grid={"w_corr": [5, 10], "w_delta": [5], "w_std": [20, 30]},
        )

        # 023: (sma(high, w_ma) < high) ? (-delta(high, w_d)) : 0
        self._add_parametric_feature(
            base_name="alpha101_023",
            expr_tpl=(
                "ts_where( ts_mean(high, {w_ma}) < high, -1 * (high - ts_delay(high, {w_d})), 0 )"
            ),
            param_grid={"w_ma": [20, 60], "w_d": [2]},
        )

        # 024: regime 切换（阈值可调）
        self._add_parametric_feature(
            base_name="alpha101_024",
            expr_tpl=(
                "ts_where( (ts_mean(close, {w_long}) - ts_delay(ts_mean(close, {w_long}), {w_long})) / ts_delay(close, {w_long}) <= {th},"
                "         -1 * (close - ts_min(close, {w_long})),"
                "         -1 * (close - ts_delay(close, {w_short})) )"
            ),
            param_grid={"w_long": [100], "w_short": [3], "th": [0.05]},
        )

        # 025: rank(((((-1 * returns) * adv_w) * vwap) * (high - close)))
        self._add_parametric_feature(
            base_name="alpha101_025",
            expr_tpl=(
                "cs_rank(((-1 * {ret}) * ts_mean(volume, {w_adv}) * {vwap}) * (high - close))"
            ).format(ret=ret, w_adv="{w_adv}", vwap=vwap),
            param_grid={"w_adv": [20, 60]},
        )

        # 026: (-1 * ts_max(corr(ts_rank(volume, w_r), ts_rank(high, w_r), w_c), w_m))
        self._add_parametric_feature(
            base_name="alpha101_026",
            expr_tpl=(
                "-1 * ts_max( ts_corr( ts_rank(volume, {w_r}), ts_rank(high, {w_r}), {w_c}), {w_m})"
            ),
            param_grid={"w_r": [5, 10], "w_c": [5, 10], "w_m": [3, 5]},
        )

        # 027: ((0.5 < rank(mean(corr(rank(volume), rank(vwap), w_c), w_s))) ? -1 : 1)
        self._add_parametric_feature(
            base_name="alpha101_027",
            expr_tpl=(
                "ts_sign( ( cs_rank( ts_mean( ts_corr(cs_rank(volume), cs_rank({vwap}), {w_c}), {w_s}) ) - 0.5 ) * (-2) )"
            ).format(vwap=vwap, w_c="{w_c}", w_s="{w_s}"),
            param_grid={"w_c": [6, 10], "w_s": [2, 3]},
        )

        # 028: scale(((corr(adv_w, low, w_c) + ((high + low) / 2)) - close))
        self._add_parametric_feature(
            base_name="alpha101_028",
            expr_tpl=(
                "cs_scale( ( ts_corr(ts_mean(volume, {w_adv}), low, {w_c}) + ((high + low) / 2) ) - close )"
            ),
            param_grid={"w_adv": [20, 60], "w_c": [5, 10]},
        )

        # 029: 复杂式子，保持窗口可调（核心 5 与 6）
        self._add_parametric_feature(
            base_name="alpha101_029",
            expr_tpl=(
                # 近似实现：用 -delta(close,1) 的截面秩，经滚动求和后取 log、scale、再做两次截面秩，最后取时序最小
                "ts_min("
                "  cs_rank( cs_rank( cs_scale( ts_log( ts_sum( cs_rank(-1 * (close - ts_delay(close, 1))), {w1}) ) ) ) ),"
                "  {w1}"
                ") + ts_rank( ts_delay( -1 * ({ret}), {w2}), {w1})"
            ).format(ret=ret, w1="{w1}", w2="{w2}"),
            param_grid={"w1": [5], "w2": [6]},
        )

        # 030: ((1 - rank(sum of signs)) * sum(volume, w1)) / sum(volume, w2)
        self._add_parametric_feature(
            base_name="alpha101_030",
            expr_tpl=(
                "( (1 - cs_rank( ts_sign(close - ts_delay(close,1)) + ts_sign(ts_delay(close,1) - ts_delay(close,2)) + ts_sign(ts_delay(close,2) - ts_delay(close,3)) ))"
                " * ts_sum(volume, {w1}) ) / ts_sum(volume, {w2})"
            ),
            param_grid={"w1": [5, 10], "w2": [20]},
        )

    def register_031_040(self, windows: list[int] | None = None) -> None:
        W = windows or self.DEFAULT_WINDOWS
        ret = self.ret
        vwap = "vwap"

        # 031: 组合项（三段式）
        self._add_parametric_feature(
            base_name="alpha101_031",
            expr_tpl=(
                "cs_rank(cs_rank(cs_rank(ts_decay_linear((-1 * cs_rank(cs_rank(close - ts_delay(close,10)))), {w_d}))))"
                " + cs_rank(-1 * (close - ts_delay(close, 3)))"
                " + ts_sign( cs_scale( ts_corr(ts_mean(volume, {w_adv}), low, {w_c}) ) )"
            ),
            param_grid={"w_d": [10], "w_adv": [20, 60], "w_c": [12]},
        )

        # 032: scale(((sum(close, w_ma) / w_ma) - close)) + k * scale(corr(vwap, delay(close, w_d), w_c))
        self._add_parametric_feature(
            base_name="alpha101_032",
            expr_tpl=(
                "cs_scale( (ts_mean(close, {w_ma}) / {w_ma}) - close ) + {k} * cs_scale( ts_corr({vwap}, ts_delay(close, {w_d}), {w_c}) )"
            ).format(vwap=vwap, k="{k}", w_ma="{w_ma}", w_d="{w_d}", w_c="{w_c}"),
            param_grid={"w_ma": [7], "w_d": [5], "w_c": [230], "k": [20]},
        )

        # 033: rank(-1 + open/close)
        self._add_parametric_feature(
            base_name="alpha101_033",
            expr_tpl="cs_rank(-1 + (open / close))",
            param_grid=None,
        )

        # 034: rank(2 - rank(std(returns,w1)/std(returns,w2)) - rank(delta(close,1)))
        self._add_parametric_feature(
            base_name="alpha101_034",
            expr_tpl=(
                "cs_rank( 2 - cs_rank( ts_std({ret}, {w1}) / (ts_std({ret}, {w2}) + 1e-12) ) - cs_rank(close - ts_delay(close,1)) )"
            ).format(ret=ret, w1="{w1}", w2="{w2}"),
            param_grid={"w1": [2], "w2": [5]},
        )

        # 035: (Ts_Rank(volume, wv) * (1 - Ts_Rank(close+high-low, wh))) * (1 - Ts_Rank(returns, wv))
        self._add_parametric_feature(
            base_name="alpha101_035",
            expr_tpl=(
                "( ts_rank(volume, {wv}) * (1 - ts_rank(close + high - low, {wh})) ) * (1 - ts_rank({ret}, {wv}))"
            ).format(ret=ret, wv="{wv}", wh="{wh}"),
            param_grid={"wv": [32], "wh": [16]},
        )

        # 037: rank(corr(delay(open-close,1), close, wc)) + rank(open-close)
        self._add_parametric_feature(
            base_name="alpha101_037",
            expr_tpl=(
                "cs_rank( ts_corr(ts_delay(open - close, 1), close, {w_c}) ) + cs_rank(open - close)"
            ),
            param_grid={"w_c": [200]},
        )

        # 038: (-1 * rank(Ts_Rank(open, w))) * rank(close/open)
        self._add_parametric_feature(
            base_name="alpha101_038",
            expr_tpl=(
                "-1 * cs_rank( ts_rank(open, {w}) ) * cs_rank( (close / open) )"
            ),
            param_grid={"w": [10]},
        )

        # 039: ((-1 * rank(delta(close,7) * (1 - rank(decay_linear(volume/adv20,9))))) * (1 + rank(sum(returns,250))))
        self._add_parametric_feature(
            base_name="alpha101_039",
            expr_tpl=(
                "-1 * cs_rank( (close - ts_delay(close, {w_d})) * (1 - cs_rank( ts_decay_linear( volume / ts_mean(volume, {w_adv}), {w_dec}) )) )"
                " * (1 + cs_rank( ts_sum({ret}, {w_sum}) ))"
            ).format(ret=ret, w_d="{w_d}", w_adv="{w_adv}", w_dec="{w_dec}", w_sum="{w_sum}"),
            param_grid={"w_d": [7], "w_adv": [20, 60], "w_dec": [9], "w_sum": [250]},
        )

        # 040: （参考公开版本，给出常见实现）
        self._add_parametric_feature(
            base_name="alpha101_040",
            expr_tpl=(
                "-1 * cs_rank( ts_std(high, {w1}) ) * cs_rank( (high / low) ) * (1 + cs_rank(volume))"
            ),
            param_grid={"w1": [10, 20]},
        )

        # 041: (((high * low)^0.5) - vwap)
        self._add_parametric_feature(
            base_name="alpha101_041",
            expr_tpl="((high * low) ** 0.5) - {vwap}".replace("{vwap}", vwap),
            param_grid=None,
        )

        # 042: (rank((vwap - close)) / rank((vwap + close)))
        self._add_parametric_feature(
            base_name="alpha101_042",
            expr_tpl="cs_rank({vwap} - close) / cs_rank({vwap} + close)".replace("{vwap}", vwap),
            param_grid=None,
        )

        # 043: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
        self._add_parametric_feature(
            base_name="alpha101_043",
            expr_tpl=(
                "ts_rank(volume / ts_mean(volume, {w_adv}), {w_tr1}) * "
                "ts_rank(-1 * (close - ts_delay(close, {w_d})), {w_tr2})"
            ),
            param_grid={
                "w_adv": [20],   # adv20
                "w_tr1": [20],   # ts_rank window 1
                "w_d": [7],      # delta window
                "w_tr2": [8],    # ts_rank window 2
            },
        )

        # 044: (-1 * correlation(high, rank(volume), 5))
        self._add_parametric_feature(
            base_name="alpha101_044",
            expr_tpl="-1 * ts_corr(high, cs_rank(volume), {w_corr})",
            param_grid={"w_corr": [5]},
        )

        # 045: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *
        #             rank(correlation(sum(close, 5), sum(close, 20), 2))))
        self._add_parametric_feature(
            base_name="alpha101_045",
            expr_tpl=(
                "-1 * ( cs_rank(ts_mean(ts_delay(close, {w_d1}), {w_ma}))"
                " * ts_corr(close, volume, {w_corr1})"
                " * cs_rank(ts_corr(ts_sum(close, {w_s1}), ts_sum(close, {w_s2}), {w_corr2})) )"
            ),
            param_grid={
                "w_d1": [5],
                "w_ma": [20],
                "w_corr1": [2],
                "w_s1": [5],
                "w_s2": [20],
                "w_corr2": [2],
            },
        )

        # 公用 inner 表达式（用于 046 / 049 / 051）
        inner = (
            "((ts_delay(close, {w1}) - ts_delay(close, {w2})) / {w3} - "
            " ((ts_delay(close, {w2}) - close) / {w3}))"
        )

        # 046: 0.25 上阈值三段式
        self._add_parametric_feature(
            base_name="alpha101_046",
            expr_tpl=(
                "ts_where("
                f"  {inner} > {{th1}},"
                "  -1,"
                "  ts_where("
                f"    {inner} < 0,"
                "    1,"
                "    -1 * (close - ts_delay(close, 1))"
                "  )"
                ")"
            ),
            param_grid={"w1": [20], "w2": [10], "w3": [10], "th1": [0.25]},
        )

        # 047: ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) /
        #        (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))
        self._add_parametric_feature(
            base_name="alpha101_047",
            expr_tpl=(
                "((cs_rank(1 / close) * volume) / ts_mean(volume, {w_adv}))"
                " * (high * cs_rank(high - close) / (ts_sum(high, {w_sh}) / {w_sh}))"
                " - cs_rank({vwap} - ts_delay({vwap}, {w_d}))"
            ).replace("{vwap}", vwap),
            param_grid={"w_adv": [20], "w_sh": [5], "w_d": [5]},
        )

        # 048: 近似实现：直接用原公式结构，但忽略行业中性 indneutralize
        # (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) *
        #   delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
        self._add_parametric_feature(
            base_name="alpha101_048",
            expr_tpl=(
                "( ts_corr(close - ts_delay(close, 1),"
                "          ts_delay(close, 1) - ts_delay(close, 2), {w_corr})"
                "  * (close - ts_delay(close, 1)) / close )"
                " / ts_sum(((close - ts_delay(close, 1)) / ts_delay(close, 1)) ** 2, {w_sum})"
            ),
            param_grid={"w_corr": [250], "w_sum": [250]},
        )

        # 049: 同 inner，但阈值 -0.1
        self._add_parametric_feature(
            base_name="alpha101_049",
            expr_tpl=(
                "ts_where("
                f"  {inner} < {{th}},"
                "  1,"
                "  -1 * (close - ts_delay(close, 1))"
                ")"
            ),
            param_grid={"w1": [20], "w2": [10], "w3": [10], "th": [-0.1]},
        )

        # 050: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
        self._add_parametric_feature(
            base_name="alpha101_050",
            expr_tpl=(
                "-1 * ts_max( cs_rank(ts_corr(cs_rank(volume), cs_rank({vwap}), {w_corr})), {w_max})"
            ).replace("{vwap}", vwap),
            param_grid={"w_corr": [5], "w_max": [5]},
        )

        # 051: 同 inner，但阈值 -0.05
        self._add_parametric_feature(
            base_name="alpha101_051",
            expr_tpl=(
                "ts_where("
                f"  {inner} < {{th}},"
                "  1,"
                "  -1 * (close - ts_delay(close, 1))"
                ")"
            ),
            param_grid={"w1": [20], "w2": [10], "w3": [10], "th": [-0.05]},
        )

        # 052: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -
        #        sum(returns, 20)) / 220))) * ts_rank(volume, 5))
        self._add_parametric_feature(
            base_name="alpha101_052",
            expr_tpl=(
                "(-1 * ts_delta(ts_min(low, {w_min}), {w_delta}))"
                " * cs_rank((ts_sum({ret}, {w_long}) - ts_sum({ret}, {w_short})) / {w_div})"
                " * ts_rank(volume, {w_tr})"
            ).replace("{ret}", ret),
            param_grid={
                "w_min": [5],
                "w_delta": [5],
                "w_long": [240],
                "w_short": [20],
                "w_div": [220],
                "w_tr": [5],
            },
        )

        # 053: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
        self._add_parametric_feature(
            base_name="alpha101_053",
            expr_tpl=(
                "-1 * ts_delta((((close - low) - (high - close)) / ((close - low) + 1e-12)), {w_d})"
            ),
            param_grid={"w_d": [9]},
        )

        # 054: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
        self._add_parametric_feature(
            base_name="alpha101_054",
            expr_tpl=(
                "-1 * ((low - close) * (open ** 5)) / (((low - high) * (close ** 5)) + 1e-12)"
            ),
            param_grid=None,
        )

        # 055: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,
        #        12)))), rank(volume), 6))
        self._add_parametric_feature(
            base_name="alpha101_055",
            expr_tpl=(
                "-1 * ts_corr("
                "  cs_rank((close - ts_min(low, {w_l})) / ((ts_max(high, {w_l}) - ts_min(low, {w_l})) + 1e-12)),"
                "  cs_rank(volume),"
                "  {w_corr}"
                ")"
            ),
            param_grid={"w_l": [12], "w_corr": [6]},
        )

        # 056: (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
        # 需要暴露列 cap，由调用侧确保提供
        self._add_parametric_feature(
            base_name="alpha101_056",
            expr_tpl=(
                "0 - 1 * ( cs_rank(ts_sum({ret}, {w_long}) / ts_sum(ts_sum({ret}, {w_short}), {w_s2}))"
                "          * cs_rank({ret} * cap) )"
            ).replace("{ret}", ret),
            param_grid={"w_long": [10], "w_short": [2], "w_s2": [3]},
            exposure_cols=("cap",),
        )

        # 057: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
        self._add_parametric_feature(
            base_name="alpha101_057",
            expr_tpl=(
                "0 - 1 * ((close - {vwap}) / ts_decay_linear(cs_rank(ts_argmax(close, {w_arg})), {w_dec}))"
            ).replace("{vwap}", vwap),
            param_grid={"w_arg": [30], "w_dec": [2]},
        )

        # 058: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, sector), volume,3.92795), 7.89291), 5.50322))
        # 这里同样忽略行业中性，直接用 vwap
        self._add_parametric_feature(
            base_name="alpha101_058",
            expr_tpl=(
                "-1 * ts_rank( ts_decay_linear(ts_corr({vwap}, volume, {w_corr}), {w_dec}), {w_tr})"
            ).replace("{vwap}", vwap),
            param_grid={"w_corr": [3.92795], "w_dec": [7.89291], "w_tr": [5.50322]},
        )

        # 059: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.7283) + (vwap * (1 - 0.7283))), industry), volume, 4.25197), 16.2289), 8.19648))
        # 注意 ((vwap * a) + (vwap * (1-a))) == vwap
        self._add_parametric_feature(
            base_name="alpha101_059",
            expr_tpl=(
                "-1 * ts_rank( ts_decay_linear(ts_corr({vwap}, volume, {w_corr}), {w_dec}), {w_tr})"
            ).replace("{vwap}", vwap),
            param_grid={"w_corr": [4.25197], "w_dec": [16.2289], "w_tr": [8.19648]},
        )

        # 060: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -
        #        scale(rank(ts_argmax(close, 10))))))
        self._add_parametric_feature(
            base_name="alpha101_060",
            expr_tpl=(
                "0 - 1 * ( 2 * cs_scale(cs_rank( (((close - low) - (high - close)) / ((high - low) + 1e-12)) * volume ))"
                "          - cs_scale(cs_rank(ts_argmax(close, {w_arg}))) )"
            ),
            param_grid={"w_arg": [10]},
        )


        # 061: (rank(vwap - ts_min(vwap, 16.1219)) < rank(correlation(vwap, adv180, 17.9282)))
        self._add_parametric_feature(
            base_name="alpha101_061",
            expr_tpl=(
                "( cs_rank({vwap} - ts_min({vwap}, {w_min})) < "
                "  cs_rank(ts_corr({vwap}, ts_mean(volume, {w_adv}), {w_corr})) )"
            ).replace("{vwap}", vwap),
            param_grid={"w_min": [16], "w_adv": [180], "w_corr": [18]},
        )

        # 062: ((rank(corr(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open)+rank(open)) < (rank((high+low)/2)+rank(high))))) * -1)
        self._add_parametric_feature(
            base_name="alpha101_062",
            expr_tpl=(
                "-1 * ( cs_rank(ts_corr({vwap}, ts_sum(ts_mean(volume, 20), {w_s}), {w_corr})) < "
                "       cs_rank( (cs_rank(open) + cs_rank(open)) < (cs_rank((high+low)/2) + cs_rank(high)) ) )"
            ).replace("{vwap}", vwap),
            param_grid={"w_s": [22], "w_corr": [10]},
        )

        # 063: ((rank(decay_linear(delta(IndNeutralize(close,...)) - rank(decay_linear(corr((vwap*0.318108+open*(1-..)), sum(adv180,...), ...))) ) * -1
        # 忽略行业中性，只对 close 使用 ts_delta
        self._add_parametric_feature(
            base_name="alpha101_063",
            expr_tpl=(
                "-1 * ( cs_rank( ts_decay_linear( ts_delta(close, {w_d1}), {w_dec1}) )"
                "      - cs_rank( ts_decay_linear( ts_corr({vwap}*{a} + open*(1-{a}), ts_sum(ts_mean(volume, {w_adv}), {w_sadv}), {w_corr}), {w_dec2}) ) )"
            ).replace("{vwap}", vwap).replace("{a}", "0.318108"),
            param_grid={"w_d1": [2], "w_dec1": [8], "w_adv": [180], "w_sadv": [37], "w_corr": [14], "w_dec2": [12]},
        )

        # 064: ((rank(corr(sum(open*0.178404+low*(1-..), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((high+low)/2*0.178404+vwap*(1-..)), 3.69741))) * -1)
        self._add_parametric_feature(
            base_name="alpha101_064",
            expr_tpl=(
                "-1 * ( cs_rank( ts_corr( ts_sum(open*{a} + low*(1-{a}), {w_s}), ts_sum(ts_mean(volume, 120), {w_s}), {w_corr}) )"
                "      < cs_rank( ts_delta( ((high+low)/2*{a} + {vwap}*(1-{a})), {w_d}) ) )"
            ).replace("{vwap}", vwap).replace("{a}", "0.178404"),
            param_grid={"w_s": [13], "w_corr": [17], "w_d": [4]},
        )

        # 065: ((rank(corr(open*0.00817205+vwap*(1-..), sum(adv60, 8.6911), 6.40374)) < rank(open - ts_min(open,13.635))) * -1)
        self._add_parametric_feature(
            base_name="alpha101_065",
            expr_tpl=(
                "-1 * ( cs_rank( ts_corr( open*{a} + {vwap}*(1-{a}), ts_sum(ts_mean(volume, 60), {w_s}), {w_corr}) )"
                "      < cs_rank( open - ts_min(open, {w_min}) ) )"
            ).replace("{vwap}", vwap).replace("{a}", "0.00817205"),
            param_grid={"w_s": [9], "w_corr": [6], "w_min": [14]},
        )

        # 066: ((rank(decay_linear(delta(vwap,3.51013),7.23052)) + Ts_Rank(decay_linear( (((low*0.96633+low*(1-..))-vwap)/(open-(high+low)/2)),11.4157),6.72611)) * -1)
        self._add_parametric_feature(
            base_name="alpha101_066",
            expr_tpl=(
                "-1 * ( cs_rank( ts_decay_linear( ts_delta({vwap}, {w_dv}), {w_decv}) )"
                "      + ts_rank( ts_decay_linear( (((low*{a} + low*(1-{a})) - {vwap}) / (open - ((high+low)/2) + 1e-12)), {w_dec2}), {w_tr}) )"
            ).replace("{vwap}", vwap).replace("{a}", "0.96633"),
            param_grid={"w_dv": [4], "w_decv": [7], "w_dec2": [11], "w_tr": [7]},
        )

        # 067: ((rank(high - ts_min(high,2.14593))^rank(corr(IndNeutralize(vwap,sector), IndNeutralize(adv20,subindustry),6.02936))) * -1)
        # 忽略行业中性，直接用 vwap 和 adv20
        self._add_parametric_feature(
            base_name="alpha101_067",
            expr_tpl=(
                "-1 * ( cs_rank( high - ts_min(high, {w_min}) ) ** "
                "       cs_rank( ts_corr({vwap}, ts_mean(volume, 20), {w_corr}) ) )"
            ).replace("{vwap}", vwap),
            param_grid={"w_min": [2], "w_corr": [6]},
        )

        # 068: ((Ts_Rank(corr(rank(high), rank(adv15),8.91644),13.9333) < rank(delta(close*0.518371+low*(1-..),1.06157))) * -1)
        self._add_parametric_feature(
            base_name="alpha101_068",
            expr_tpl=(
                "-1 * ( ts_rank( ts_corr(cs_rank(high), cs_rank(ts_mean(volume,15)), {w_corr}), {w_tr})"
                "      < cs_rank( ts_delta( (close*{a} + low*(1-{a})), {w_d}) ) )"
            ).replace("{a}", "0.518371"),
            param_grid={"w_corr": [9], "w_tr": [14], "w_d": [1]},
        )

        # 069: ((rank(ts_max(delta(IndNeutralize(vwap,industry),2.72412),4.79344))^Ts_Rank(corr(close*0.490655+vwap*(1-..), adv20,4.92416),9.0615)) * -1)
        # 仍然忽略行业中性
        self._add_parametric_feature(
            base_name="alpha101_069",
            expr_tpl=(
                "-1 * ( cs_rank( ts_max( ts_delta({vwap}, {w_d}), {w_m}) ) ** "
                "       ts_rank( ts_corr(close*{a} + {vwap}*(1-{a}), ts_mean(volume, 20), {w_corr}), {w_tr}) )"
            ).replace("{vwap}", vwap).replace("{a}", "0.490655"),
            param_grid={"w_d": [3], "w_m": [5], "w_corr": [5], "w_tr": [9]},
        )

        # 070: ((rank(delta(vwap,1.29456))^Ts_Rank(corr(IndNeutralize(close,industry), adv50,17.8256),17.9171)) * -1)
        self._add_parametric_feature(
            base_name="alpha101_070",
            expr_tpl=(
                "-1 * ( cs_rank( ts_delta({vwap}, {w_d}) ) ** "
                "       ts_rank( ts_corr(close, ts_mean(volume, 50), {w_corr}), {w_tr}) )"
            ).replace("{vwap}", vwap),
            param_grid={"w_d": [1], "w_corr": [18], "w_tr": [18]},
        )

        # 071: max(Ts_Rank(decay_linear(corr(Ts_Rank(close,3.43976), Ts_Rank(adv180,12.0647),18.0175),4.20501),15.6948),
        #           Ts_Rank(decay_linear((rank((low+open-(vwap+vwap)))^2),16.4662),4.4388))
        self._add_parametric_feature(
            base_name="alpha101_071",
            expr_tpl=(
                "ts_greater("
                "  ts_rank( ts_decay_linear( ts_corr(ts_rank(close, 3), ts_rank(ts_mean(volume, 180), 12), {w_corr1}), {w_dec1}), {w_tr1}),"
                "  ts_rank( ts_decay_linear( cs_rank((low + open - ({vwap} + {vwap})) ) ** 2, {w_dec2}), {w_tr2})"
                ")"
            ).replace("{vwap}", vwap),
            param_grid={"w_corr1": [18], "w_dec1": [4], "w_tr1": [16], "w_dec2": [16], "w_tr2": [4]},
        )

        # 072: (rank(decay_linear(corr((high+low)/2, adv40,8.93345),10.1519)) / rank(decay_linear(corr(Ts_Rank(vwap,3.72469), Ts_Rank(volume,18.5188),6.86671),2.95011)))
        self._add_parametric_feature(
            base_name="alpha101_072",
            expr_tpl=(
                "cs_rank( ts_decay_linear( ts_corr((high+low)/2, ts_mean(volume, 40), {w_corr1}), {w_dec1}) ) / "
                "cs_rank( ts_decay_linear( ts_corr(ts_rank({vwap}, 4), ts_rank(volume, 19), {w_corr2}), {w_dec2}) )"
            ).replace("{vwap}", vwap),
            param_grid={"w_corr1": [9], "w_dec1": [10], "w_corr2": [7], "w_dec2": [3]},
        )

        # 073: (max(rank(decay_linear(delta(vwap,4.72775),2.91864)), Ts_Rank(decay_linear(((delta(open*0.147155+low*(1-..),2.03608)/(open*0.147155+low*(1-..)))*-1),3.33829),16.7411)) * -1)
        self._add_parametric_feature(
            base_name="alpha101_073",
            expr_tpl=(
                "-1 * ts_greater("
                "  cs_rank( ts_decay_linear( ts_delta({vwap}, {w_dv}), {w_decv}) ),"
                "  ts_rank( ts_decay_linear( ( ts_delta(open*{a} + low*(1-{a}), {w_do}) / (open*{a} + low*(1-{a}) + 1e-12) * -1 ), {w_dec2}), {w_tr})"
                ")"
            ).replace("{vwap}", vwap).replace("{a}", "0.147155"),
            param_grid={"w_dv": [5], "w_decv": [3], "w_do": [2], "w_dec2": [3], "w_tr": [17]},
        )

        # 074: ((rank(corr(close, sum(adv30,37.4843),15.1365)) < rank(corr(rank(high*0.0261661+vwap*(1-..)), rank(volume),11.4791))) * -1)
        self._add_parametric_feature(
            base_name="alpha101_074",
            expr_tpl=(
                "-1 * ( cs_rank( ts_corr(close, ts_sum(ts_mean(volume, 30), {w_sadv}), {w_corr1}) ) < "
                "       cs_rank( ts_corr(cs_rank(high*{a} + {vwap}*(1-{a})), cs_rank(volume), {w_corr2}) ) )"
            ).replace("{vwap}", vwap).replace("{a}", "0.0261661"),
            param_grid={"w_sadv": [37], "w_corr1": [15], "w_corr2": [11]},
        )

        # 075: (rank(corr(vwap, volume,4.24304)) < rank(corr(rank(low), rank(adv50),12.4413)))
        self._add_parametric_feature(
            base_name="alpha101_075",
            expr_tpl=(
                "( cs_rank( ts_corr({vwap}, volume, {w_corr1}) ) < "
                "  cs_rank( ts_corr(cs_rank(low), cs_rank(ts_mean(volume, 50)), {w_corr2}) ) )"
            ).replace("{vwap}", vwap),
            param_grid={"w_corr1": [4], "w_corr2": [12]},
        )

        # 076: (max(rank(decay_linear(delta(vwap,1.24383),11.8259)), Ts_Rank(decay_linear(Ts_Rank(corr(IndNeutralize(low,sector), adv81,8.14941),19.569),17.1543),19.383)) * -1)
        # 忽略行业中性
        self._add_parametric_feature(
            base_name="alpha101_076",
            expr_tpl=(
                "-1 * ts_greater("
                "  cs_rank( ts_decay_linear( ts_delta({vwap}, {w_dv}), {w_decv}) ),"
                "  ts_rank( ts_decay_linear( ts_rank(ts_corr(low, ts_mean(volume, 81), {w_corr}), {w_tr1}), {w_dec2}), {w_tr2})"
                ")"
            ).replace("{vwap}", vwap),
            param_grid={"w_dv": [1], "w_decv": [12], "w_corr": [8], "w_tr1": [20], "w_dec2": [17], "w_tr2": [19]},
        )

        # 077: min(rank(decay_linear((((high+low)/2 + high) - (vwap + high)),20.0451)), rank(decay_linear(corr((high+low)/2, adv40,3.1614),5.64125)))
        self._add_parametric_feature(
            base_name="alpha101_077",
            expr_tpl=(
                "ts_less("
                "  cs_rank( ts_decay_linear( (((high+low)/2 + high) - ({vwap} + high)), {w_dec1}) ),"
                "  cs_rank( ts_decay_linear( ts_corr((high+low)/2, ts_mean(volume, 40), {w_corr}), {w_dec2}) )"
                ")"
            ).replace("{vwap}", vwap),
            param_grid={"w_dec1": [20], "w_corr": [3], "w_dec2": [6]},
        )

        # 078: (rank(corr(sum(low*0.352233+vwap*(1-..),19.7428), sum(adv40,19.7428),6.83313))^rank(corr(rank(vwap), rank(volume),5.77492)))
        self._add_parametric_feature(
            base_name="alpha101_078",
            expr_tpl=(
                "cs_rank( ts_corr( ts_sum(low*{a} + {vwap}*(1-{a}), {w_s}), ts_sum(ts_mean(volume, 40), {w_s}), {w_corr1}) ) ** "
                "cs_rank( ts_corr(cs_rank({vwap}), cs_rank(volume), {w_corr2}) )"
            ).replace("{vwap}", vwap).replace("{a}", "0.352233"),
            param_grid={"w_s": [20], "w_corr1": [7], "w_corr2": [6]},
        )

        # 079: (rank(delta(IndNeutralize(close*0.60733+open*(1-..),sector),1.23438)) < rank(corr(Ts_Rank(vwap,3.60973), Ts_Rank(adv150,9.18637),14.6644)))
        # 忽略行业中性
        self._add_parametric_feature(
            base_name="alpha101_079",
            expr_tpl=(
                "( cs_rank( ts_delta(close*{a} + open*(1-{a}), {w_d}) ) < "
                "  cs_rank( ts_corr( ts_rank({vwap}, 4), ts_rank(ts_mean(volume, 150), 9), {w_corr}) ) )"
            ).replace("{vwap}", vwap).replace("{a}", "0.60733"),
            param_grid={"w_d": [1], "w_corr": [15]},
        )

        # 080: ((rank(Sign(delta(IndNeutralize(open*0.868128+high*(1-..),industry),4.04545)))^Ts_Rank(corr(high, adv10,5.11456),5.53756)) * -1)
        # 忽略行业中性，Sign → ts_sign
        self._add_parametric_feature(
            base_name="alpha101_080",
            expr_tpl=(
                "-1 * ( cs_rank( ts_sign( ts_delta(open*{a} + high*(1-{a}), {w_d}) ) ) ** "
                "       ts_rank( ts_corr(high, ts_mean(volume, 10), {w_corr}), {w_tr}) )"
            ).replace("{a}", "0.868128"),
            param_grid={"w_d": [4], "w_corr": [5], "w_tr": [6]},
        )


    # ============ 041 ～ 060 ============
    # def register_test(self, windows: list[int] | None = None) -> None:
    #     """Alpha#41～#60：全部改为可参数化形式（但默认网格只包含原始窗口值，不再额外扩散）。"""
    #     W = windows or self.DEFAULT_WINDOWS
    #     ret = self.ret
    #     vwap = "vwap"


        """Register alpha101 #81-#101 with parameterized (but single-valued) windows.

        Notes
        -----
        * 对应 Kakushadze(2016) 里 Alpha#81-#101 的公式，全部做了整数窗口近似。
        * 所有窗口参数都进入 param_grid，但默认只给一个值，后续如需多窗口可以在外面改 param_grid。
        * 行业中性 IndNeutralize/indneutralize 继续简化为原序列本身（行业中性放到回归层做）。
        """

        # -------------- Alpha#81 --------------
        # Alpha#81: ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655)))
        #            < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
        expr_081 = (
            "-1 * cs_where("
            "    cs_rank(log(ts_prod(cs_rank(cs_rank(ts_corr(vwap, ts_sum(ts_mean(volume, {w_adv10}), {w_sum}), {w_corr1})) ** 4), {w_prod}))) "
            "    < cs_rank(ts_corr(cs_rank(vwap), cs_rank(volume), {w_corr2})),"
            "    1, -1)"
        )
        self._add_parametric_feature(
            base_name="alpha101_081",
            expr_tpl=expr_081,
            param_grid={
                "w_adv10": [10],
                "w_sum": [50],
                "w_corr1": [8],
                "w_prod": [15],
                "w_corr2": [5],
            },
        )

        # -------------- Alpha#82 --------------
        # Alpha#82: (min(rank(decay_linear(delta(open, 1.46063), 14.8717)),
        #                 Ts_Rank(decay_linear(correlation(IndNeutralize(volume,...), open, 17.4842), 6.92131), 13.4283)) * -1)
        expr_082 = (
            "-1 * ts_less("
            "    cs_rank(ts_decay_linear(ts_delta(open, {w_d_open}), {w_dec1})), "
            "    ts_rank(ts_decay_linear(ts_corr(volume, open, {w_corr2}), {w_dec2}), {w_tr})"
            ")"
        )
        self._add_parametric_feature(
            base_name="alpha101_082",
            expr_tpl=expr_082,
            param_grid={
                "w_d_open": [1],
                "w_dec1": [15],
                "w_corr2": [17],
                "w_dec2": [7],
                "w_tr": [13],
            },
        )

        # -------------- Alpha#83 --------------
        # Alpha#83: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) /
        #            (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))
        expr_083 = (
            "(cs_rank(ts_delay(((high - low) / (ts_sum(close, {w_sum}) / {w_sum})), {w_delay})) "
            " * cs_rank(cs_rank(volume))) / ( ((high - low) / (ts_sum(close, {w_sum}) / {w_sum})) / (vwap - close + 1e-12) + 1e-12)"
        )
        self._add_parametric_feature(
            base_name="alpha101_083",
            expr_tpl=expr_083,
            param_grid={
                "w_sum": [5],
                "w_delay": [2],
            },
        )

        # -------------- Alpha#84 --------------
        # Alpha#84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))
        # 这里用 "(x ** y)" 近似 SignedPower(x, y)
        expr_084 = (
            "(ts_rank(vwap - ts_max(vwap, {w_max}), {w_tr}) ** ts_delta(close, {w_d}))"
        )
        self._add_parametric_feature(
            base_name="alpha101_084",
            expr_tpl=expr_084,
            param_grid={
                "w_max": [15],
                "w_tr": [21],
                "w_d": [5],
            },
        )

        # -------------- Alpha#85 --------------
        # Alpha#85: (rank(correlation(0.876703*high + 0.123297*close, adv30, 9.61331))^
        #            rank(correlation(Ts_Rank((high+low)/2, 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))
        expr_085 = (
            "(cs_rank(ts_corr(0.876703*high + (1-0.876703)*close, ts_mean(volume, {w_adv30}), {w_corr1})) "
            " ** cs_rank(ts_corr(ts_rank((high + low) / 2, {w_tr_price}), ts_rank(volume, {w_tr_vol}), {w_corr2})))"
        )
        self._add_parametric_feature(
            base_name="alpha101_085",
            expr_tpl=expr_085,
            param_grid={
                "w_adv30": [30],
                "w_corr1": [10],
                "w_tr_price": [4],
                "w_tr_vol": [10],
                "w_corr2": [7],
            },
        )

        # -------------- Alpha#86 --------------
        # Alpha#86: ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195)
        #             < rank(((open + close) - (vwap + open)))) * -1)
        expr_086 = (
            "-1 * cs_where("
            "    ts_rank(ts_corr(close, ts_sum(ts_mean(volume, {w_adv20}), {w_sum_adv}), {w_corr1}), {w_tr}) "
            "    < cs_rank((open + close) - (vwap + open)), 1, -1)"
        )
        self._add_parametric_feature(
            base_name="alpha101_086",
            expr_tpl=expr_086,
            param_grid={
                "w_adv20": [20],
                "w_sum_adv": [15],
                "w_corr1": [6],
                "w_tr": [20],
            },
        )

        # -------------- Alpha#87 --------------
        # Alpha#87: max(rank(decay_linear(delta(0.369701*close + 0.630299*vwap, 1.91233), 2.65461)),
        #               Ts_Rank(decay_linear(abs(correlation(adv81, close, 13.4132)), 4.89768), 14.4535)) * -1
        expr_087 = (
            "-1 * ts_greater("
            "    cs_rank(ts_decay_linear(ts_delta(0.369701*close + (1-0.369701)*vwap, {w_d1}), {w_dec1})), "
            "    ts_rank(ts_decay_linear(abs(ts_corr(ts_mean(volume, {w_adv81}), close, {w_corr2})), {w_dec2}), {w_tr2})"
            ")"
        )
        self._add_parametric_feature(
            base_name="alpha101_087",
            expr_tpl=expr_087,
            param_grid={
                "w_d1": [2],
                "w_dec1": [3],
                "w_adv81": [81],
                "w_corr2": [13],
                "w_dec2": [5],
                "w_tr2": [14],
            },
        )

        # -------------- Alpha#88 --------------
        # Alpha#88: min(rank(decay_linear((rank(open)+rank(low)-rank(high)-rank(close)), 8.06882)),
        #               Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))
        expr_088 = (
            "ts_less("
            "    cs_rank(ts_decay_linear((cs_rank(open) + cs_rank(low) - cs_rank(high) - cs_rank(close)), {w_dec1})), "
            "    ts_rank(ts_decay_linear(ts_corr(ts_rank(close, {w_tr_close}), ts_rank(ts_mean(volume, {w_adv60}), {w_tr_adv}), {w_corr2}), {w_dec2}), {w_tr2})"
            ")"
        )
        self._add_parametric_feature(
            base_name="alpha101_088",
            expr_tpl=expr_088,
            param_grid={
                "w_dec1": [8],
                "w_tr_close": [8],
                "w_adv60": [60],
                "w_tr_adv": [21],
                "w_corr2": [8],
                "w_dec2": [7],
                "w_tr2": [3],
            },
        )

        # -------------- Alpha#89 --------------
        # Alpha#89: Ts_Rank(decay_linear(correlation(low, adv10, 6.94279), 5.51607), 3.79744)
        #            - Ts_Rank(decay_linear(delta(vwap, 3.48158), 10.1466), 15.3012)
        expr_089 = (
            "ts_rank(ts_decay_linear(ts_corr(low, ts_mean(volume, {w_adv10}), {w_corr1}), {w_dec1}), {w_tr1}) "
            "- ts_rank(ts_decay_linear(ts_delta(vwap, {w_d2}), {w_dec2}), {w_tr2})"
        )
        self._add_parametric_feature(
            base_name="alpha101_089",
            expr_tpl=expr_089,
            param_grid={
                "w_adv10": [10],
                "w_corr1": [7],
                "w_dec1": [6],
                "w_tr1": [4],
                "w_d2": [3],
                "w_dec2": [10],
                "w_tr2": [15],
            },
        )

        # -------------- Alpha#90 --------------
        # Alpha#90: (rank(close - ts_max(close, 4.66719))^
        #            Ts_Rank(correlation(adv40, low, 5.38375), 3.21856)) * -1
        expr_090 = (
            "-1 * (cs_rank(close - ts_max(close, {w_max})) ** "
            "        ts_rank(ts_corr(ts_mean(volume, {w_adv40}), low, {w_corr2}), {w_tr2}))"
        )
        self._add_parametric_feature(
            base_name="alpha101_090",
            expr_tpl=expr_090,
            param_grid={
                "w_max": [5],
                "w_adv40": [40],
                "w_corr2": [5],
                "w_tr2": [3],
            },
        )

        # -------------- Alpha#91 --------------
        # Alpha#91: (Ts_Rank(decay_linear(decay_linear(correlation(close, volume, 9.74928), 16.398), 3.83219), 4.8667)
        #            - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1
        expr_091 = (
            "-1 * ("
            "    ts_rank(ts_decay_linear(ts_decay_linear(ts_corr(close, volume, {w_corr1}), {w_dec1}), {w_dec2}), {w_tr1}) "
            "    - cs_rank(ts_decay_linear(ts_corr(vwap, ts_mean(volume, {w_adv30}), {w_corr2}), {w_dec3}))"
            ")"
        )
        self._add_parametric_feature(
            base_name="alpha101_091",
            expr_tpl=expr_091,
            param_grid={
                "w_corr1": [10],
                "w_dec1": [16],
                "w_dec2": [4],
                "w_tr1": [5],
                "w_adv30": [30],
                "w_corr2": [4],
                "w_dec3": [3],
            },
        )

        # -------------- Alpha#92 --------------
        # Alpha#92: min(Ts_Rank(decay_linear(((((high+low)/2)+close) < (low+open)), 14.7221), 18.8683),
        #               Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))
        # 这里用差值替代布尔比较，避免显式 < 运算：使用 ((high+low)/2+close - (low+open))
        expr_092 = (
            "ts_less("
            "    ts_rank(ts_decay_linear((((high + low) / 2 + close) - (low + open)), {w_dec1}), {w_tr1}), "
            "    ts_rank(ts_decay_linear(ts_corr(cs_rank(low), cs_rank(ts_mean(volume, {w_adv30})), {w_corr2}), {w_dec2}), {w_tr2})"
            ")"
        )
        self._add_parametric_feature(
            base_name="alpha101_092",
            expr_tpl=expr_092,
            param_grid={
                "w_dec1": [15],
                "w_tr1": [19],
                "w_adv30": [30],
                "w_corr2": [8],
                "w_dec2": [7],
                "w_tr2": [7],
            },
        )

        # -------------- Alpha#93 --------------
        # Alpha#93: Ts_Rank(decay_linear(correlation(vwap, adv81, 17.4193), 19.848), 7.54455)
        #            / rank(decay_linear(delta(0.524434*close + 0.475566*vwap, 2.77377), 16.2664))
        expr_093 = (
            "ts_rank(ts_decay_linear(ts_corr(vwap, ts_mean(volume, {w_adv81}), {w_corr1}), {w_dec1}), {w_tr1}) "
            " / (cs_rank(ts_decay_linear(ts_delta(0.524434*close + (1-0.524434)*vwap, {w_d2}), {w_dec2})) + 1e-12)"
        )
        self._add_parametric_feature(
            base_name="alpha101_093",
            expr_tpl=expr_093,
            param_grid={
                "w_adv81": [81],
                "w_corr1": [17],
                "w_dec1": [20],
                "w_tr1": [8],
                "w_d2": [3],
                "w_dec2": [16],
            },
        )

        # -------------- Alpha#94 --------------
        # Alpha#94: (rank(vwap - ts_min(vwap, 11.5783))^
        #            Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1
        expr_094 = (
            "-1 * (cs_rank(vwap - ts_min(vwap, {w_min})) ** "
            "        ts_rank(ts_corr(ts_rank(vwap, {w_tr_vwap}), ts_rank(ts_mean(volume, {w_adv60}), {w_tr_adv}), {w_corr1}), {w_tr1}))"
        )
        self._add_parametric_feature(
            base_name="alpha101_094",
            expr_tpl=expr_094,
            param_grid={
                "w_min": [12],
                "w_tr_vwap": [20],
                "w_adv60": [60],
                "w_tr_adv": [4],
                "w_corr1": [18],
                "w_tr1": [3],
            },
        )

        # -------------- Alpha#95 --------------
        # Alpha#95: (rank(open - ts_min(open, 12.4105)) <
        #            Ts_Rank((rank(correlation(sum((high+low)/2, 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
        # 用差值 + ts_sign 近似比较符号
        expr_095 = (
            "ts_sign( ts_rank((cs_rank(ts_corr(ts_sum((high + low) / 2, {w_sum_price}), ts_sum(ts_mean(volume, {w_adv40}), {w_sum_adv}), {w_corr1})) ** 5), {w_tr1}) "
            "        - cs_rank(open - ts_min(open, {w_min_open})) )"
        )
        self._add_parametric_feature(
            base_name="alpha101_095",
            expr_tpl=expr_095,
            param_grid={
                "w_min_open": [12],
                "w_sum_price": [19],
                "w_adv40": [40],
                "w_sum_adv": [19],
                "w_corr1": [13],
                "w_tr1": [12],
            },
        )

        # -------------- Alpha#96 --------------
        # Alpha#96: max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151),
        #               Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1
        expr_096 = (
            "-1 * ts_greater("
            "    ts_rank(ts_decay_linear(ts_corr(cs_rank(vwap), cs_rank(volume), {w_corr1}), {w_dec1}), {w_tr1}), "
            "    ts_rank(ts_decay_linear(ts_argmax(ts_corr(ts_rank(close, {w_tr_close}), ts_rank(ts_mean(volume, {w_adv60}), {w_tr_adv}), {w_corr2}), {w_arg}), {w_dec2}), {w_tr2})"
            ")"
        )
        self._add_parametric_feature(
            base_name="alpha101_096",
            expr_tpl=expr_096,
            param_grid={
                "w_corr1": [4],
                "w_dec1": [4],
                "w_tr1": [8],
                "w_tr_close": [7],
                "w_adv60": [60],
                "w_tr_adv": [4],
                "w_corr2": [4],
                "w_arg": [13],
                "w_dec2": [14],
                "w_tr2": [13],
            },
        )

        # -------------- Alpha#97 --------------
        # Alpha#97: ((rank(decay_linear(delta(0.721001*low + 0.278999*vwap, 3.3705), 20.4523))
        #             - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1
        expr_097 = (
            "-1 * ("
            "    cs_rank(ts_decay_linear(ts_delta(0.721001*low + (1-0.721001)*vwap, {w_d1}), {w_dec1})) "
            "    - ts_rank(ts_decay_linear(ts_rank(ts_corr(ts_rank(low, {w_tr_low}), ts_rank(ts_mean(volume, {w_adv60}), {w_tr_adv}), {w_corr2}), {w_corr3}), {w_dec2}), {w_tr2})"
            ")"
        )
        self._add_parametric_feature(
            base_name="alpha101_097",
            expr_tpl=expr_097,
            param_grid={
                "w_d1": [3],
                "w_dec1": [20],
                "w_tr_low": [8],
                "w_adv60": [60],
                "w_tr_adv": [17],
                "w_corr2": [5],
                "w_corr3": [19],
                "w_dec2": [16],
                "w_tr2": [7],
            },
        )

        # -------------- Alpha#98 --------------
        # Alpha#98: rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088))
        #            - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206))
        expr_098 = (
            "cs_rank(ts_decay_linear(ts_corr(vwap, ts_sum(ts_mean(volume, {w_adv5}), {w_sum_adv5}), {w_corr1}), {w_dec1})) "
            "- cs_rank(ts_decay_linear(ts_rank(ts_argmin(ts_corr(cs_rank(open), cs_rank(ts_mean(volume, {w_adv15})), {w_corr2}), {w_arg}), {w_tr_inner}), {w_dec2}))"
        )
        self._add_parametric_feature(
            base_name="alpha101_098",
            expr_tpl=expr_098,
            param_grid={
                "w_adv5": [5],
                "w_sum_adv5": [26],
                "w_corr1": [5],
                "w_dec1": [7],
                "w_adv15": [15],
                "w_corr2": [21],
                "w_arg": [9],
                "w_tr_inner": [7],
                "w_dec2": [8],
            },
        )

        # -------------- Alpha#99 --------------
        # Alpha#99: (rank(correlation(sum((high+low)/2, 19.8975), sum(adv60, 19.8975), 8.8136))
        #            < rank(correlation(low, volume, 6.28259))) * -1
        expr_099 = (
            "-1 * cs_where("
            "    cs_rank(ts_corr(ts_sum((high + low) / 2, {w_sum_price}), ts_sum(ts_mean(volume, {w_adv60}), {w_sum_adv}), {w_corr1})) "
            "    < cs_rank(ts_corr(low, volume, {w_corr2})), 1, -1)"
        )
        self._add_parametric_feature(
            base_name="alpha101_099",
            expr_tpl=expr_099,
            param_grid={
                "w_sum_price": [20],
                "w_adv60": [60],
                "w_sum_adv": [20],
                "w_corr1": [9],
                "w_corr2": [6],
            },
        )

        # -------------- Alpha#100 --------------
        # Alpha#100: 复杂行业中性版本，这里近似为：
        # 1. 先构造经典 (close-low-high-close)/range * volume 结构并做 cs_rank/scale
        # 2. 再减去 "correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))" 的行业中性版本
        # 仍然保持整体 mean-reversion 结构和 volume/adv20 放大因子
        expr_100 = (
            "-1 * ("
            "    1.5 * cs_scale(cs_rank((((close - low) - (high - close)) / (high - low + 1e-12) * volume))) "
            "    - cs_scale(cs_rank(ts_corr(close, cs_rank(ts_mean(volume, {w_adv20})), {w_corr1}) - cs_rank(ts_argmin(close, {w_arg}))))"
            ") * (volume / (ts_mean(volume, {w_adv20}) + 1e-12))"
        )
        self._add_parametric_feature(
            base_name="alpha101_100",
            expr_tpl=expr_100,
            param_grid={
                "w_adv20": [20],
                "w_corr1": [5],
                "w_arg": [30],
            },
        )

        # -------------- Alpha#101 --------------
        # Alpha#101: (close - open) / ((high - low) + 0.001)
        expr_101 = "(close - open) / ((high - low) + 0.001)"
        self._add_parametric_feature(
            base_name="alpha101_101",
            expr_tpl=expr_101,
            param_grid={},
        )

    # 便捷一次性注册前 40 个
    def register_all(self, windows: list[int] | None = None) -> None:
        self.register_001_010(windows)
        self.register_011_020(windows)
        self.register_021_030(windows)
        self.register_031_040(windows)
        # self.register_test(windows)
