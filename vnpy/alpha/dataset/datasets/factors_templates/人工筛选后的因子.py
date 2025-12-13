from itertools import product
from typing import Any, Callable, Iterable
from vnpy.alpha.dataset import AlphaDataset
import polars as pl


class Selected(AlphaDataset):
    """
    用于在当前 Dataset 内注册 Alpha101 的表达式（字符串），
    通过 self.add_feature(name: str, expr: str) 写入到本对象的 feature_expressions 中。

    使用方法（示例）：
      ds = Select额的(df, train_period, valid_period, test_period, interval="1d")
      ds.register_all(windows=[5,10,20,30])
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
        # 默认 Label：与 Alpha101 一致（未来3天相对1天的收益）
        try:
            self.set_label("ts_delay(close, -3) / ts_delay(close, -1) - 1")
        except Exception:
            pass

        # 注意：不注入外部 add_feature_fn，统一使用 AlphaDataset.add_feature


    DEFAULT_WINDOWS: list[int] = [1, 3, 5, 10, 20]


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
    ) -> None:
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


    # ============ 公共片段（便于复用） ============
    @property
    def ret(self) -> str:
        return "(close / ts_delay(close, 1) - 1)"

    def opensource_factors(self, windows: list[int] | None = None) -> None:
        windows = self.DEFAULT_WINDOWS
        ret = self.ret
        vwap = "vwap"

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

        # 042: ((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))
        self._add_parametric_feature(
            base_name="alpha191_042",
            expr_tpl="(-1 * cs_rank(ts_std(high, 10))) * ts_corr(high, volume, 10)",
            param_grid=None,
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

        for w in windows:
          self.add_feature("alpha158_klow_2", "((ts_less(open, close) - low) / (high - low + 1e-12))")

        # 139: (-1 * CORR(OPEN, VOLUME, 10))
        self._add_parametric_feature(
            base_name="alpha191_139",
            expr_tpl="-1*ts_corr(open,volume,{w_corr})",
            param_grid={"w_corr": [10]},
        )
        

        # 006: -corr(open, volume, w)
        self._add_parametric_feature(
            base_name="alpha101_006",
            expr_tpl="-1 * ts_corr(open, volume, {w})",
            param_grid={"w": [10, 20, 30]},
        )


        # 026: (-1 * ts_max(corr(ts_rank(volume, w_r), ts_rank(high, w_r), w_c), w_m))
        self._add_parametric_feature(
            base_name="alpha101_026",
            expr_tpl=(
                "-1 * ts_max( ts_corr( ts_rank(volume, {w_r}), ts_rank(high, {w_r}), {w_c}), {w_m})"
            ),
            param_grid={"w_r": [5, 10], "w_c": [5, 10], "w_m": [3, 5]},
        )

        # 040: （参考公开版本，给出常见实现）
        self._add_parametric_feature(
            base_name="alpha101_040",
            expr_tpl=(
                "-1 * cs_rank( ts_std(high, {w1}) ) * cs_rank( (high / low) ) * (1 + cs_rank(volume))"
            ),
            param_grid={"w1": [10, 20]},
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

                # 136: (-1 * RANK(DELTA(RET,3))) * CORR(OPEN,V,10)
        self._add_parametric_feature(
            base_name="alpha191_136",
            expr_tpl=(
                "(-1*cs_rank(ts_delta({r},{w_d})))"
                "*ts_corr(open,volume,{w_corr})"
            ).format(r=ret, w_d="{w_d}", w_corr="{w_corr}"),
            param_grid={"w_d": [3], "w_corr": [10]},
        )


                # 014: ((-1 * rank(delta(returns,3))) * corr(open, volume, 10))
        self._add_parametric_feature(
            base_name="alpha101_014",
            expr_tpl=(
                "(-1 * cs_rank({ret} - ts_delay({ret}, 3))) * ts_corr(open, volume, {w})"
            ).format(ret=ret, w="{w}"),
            param_grid={"w": [10, 20]},
        )


                # 016: -rank(covariance(rank(high), rank(volume), 5))
        self._add_parametric_feature(
            base_name="alpha101_016",
            expr_tpl="-1 * cs_rank(ts_cov(cs_rank(high), cs_rank(volume), {w}))",
            param_grid={"w": [5, 10]},
        )

        # 062: (-1 * CORR(HIGH, RANK(VOLUME), 5))
        self._add_parametric_feature(
            base_name="alpha191_062",
            expr_tpl=(
                "-1 * ts_corr(high, cs_rank(volume), {w_corr})"
            ),
            param_grid={"w_corr": [5]},
        )

                # 044: (-1 * correlation(high, rank(volume), 5))
        self._add_parametric_feature(
            base_name="alpha101_044",
            expr_tpl="-1 * ts_corr(high, cs_rank(volume), {w_corr})",
            param_grid={"w_corr": [5]},
        )
        for w in windows:
          self.add_feature(f"alpha158_min_{w}", f"ts_min(low, {w}) / close")


                # 023: (sma(high, w_ma) < high) ? (-delta(high, w_d)) : 0
        self._add_parametric_feature(
            base_name="alpha101_023",
            expr_tpl=(
                "ts_where( ts_mean(high, {w_ma}) < high, -1 * (high - ts_delay(high, {w_d})), 0 )"
            ),
            param_grid={"w_ma": [20, 60], "w_d": [2]},
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

                # 013: -rank(covariance(rank(close), rank(volume), 5))
        self._add_parametric_feature(
            base_name="alpha101_013",
            expr_tpl="-1 * cs_rank(ts_cov(cs_rank(close), cs_rank(volume), {w}))",
            param_grid={"w": [5, 10]},
        )

        for w in windows:
            self.add_feature(f"alpha158_qtld_{w}", f"ts_quantile(close, {w}, 0.2) / close")

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

    def register_all(self, windows: list[int] | None = None) -> None:
        self.opensource_factors(windows)