from itertools import product
from typing import Any, Callable, Iterable
from vnpy.alpha.dataset import AlphaDataset
import polars as pl

class ApolloNo1(AlphaDataset):
    """
    最牛逼的当然是阿波罗一号了！
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
        # 训练用 Label：未来3天相对1天的收益
        self.set_label("ts_delay(close, -3) / ts_delay(close, -1) - 1")

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

    @property
    def adv20(self) -> str:
        return "ts_mean(volume, 20)"

    def opensource_factor_families(self, windows: list[int] | None = None) -> None:
        """
        Open-source 因子的 family 版实现：
        - 价量相关 family（corr / cov / spike）
        - 回撤 / 低位 / 分位 family
        - K 线 / 反转 family
        """
        windows = windows or self.DEFAULT_WINDOWS
        ret = self.ret
        vwap = "vwap"

        # ------------------------------------------------------------------
        # 1) 价量相关 family
        # ------------------------------------------------------------------

        # 1.1 统一的 Price-Volume Corr family
        # 覆盖：
        # - alpha101_006 / alpha191_139（open-volume corr）
        # - alpha191_062 / alpha101_044（high-volume corr）
        # - alpha191_042 中的 corr(high, volume, 10) 那条腿
        self._add_parametric_feature(
            base_name="os_pv_corr",
            expr_tpl=(
                "-1 * ts_corr("
                "  {price_col},"
                "  {vol_expr},"
                "  {w_corr}"
                ")"
            ),
            param_grid={
                # 不同价：open / high / close
                "price_col": ["open", "high", "close"],
                # 直接用量 or 横截面排名后的量
                "vol_expr": ["volume", "cs_rank(volume)"],
                "w_corr": [3, 5, 10, 20],
            },
        )

        # 1.2 Price-Volume Cov family
        # 覆盖：
        # - alpha101_016（high × volume 协方差）
        # - alpha101_013（close × volume 协方差）
        # - alpha191_083（实质与 016 同质）
        self._add_parametric_feature(
            base_name="os_pv_cov",
            expr_tpl=(
                "-1 * cs_rank("
                "  ts_cov("
                "    cs_rank({price_col}),"
                "    cs_rank(volume),"
                "    {w_cov}"
                "  )"
                ")"
            ),
            param_grid={
                "price_col": ["high", "close"],
                "w_cov": [5, 10, 20],
            },
        )

        # 1.3 Price-Volume Spike family（量价共振尖峰）
        # 覆盖：
        # - alpha101_026（多参数 ts_max(corr(...))）
        # - alpha191_005（026 的一个特例）
        # - alpha191_032（sum/cs_rank 版本，思路类似）
        self._add_parametric_feature(
            base_name="os_pv_spike",
            expr_tpl=(
                "-1 * {agg_fn}("
                "  ts_corr("
                "    ts_rank({price_col}, {w_r}),"
                "    ts_rank(volume, {w_r}),"
                "    {w_corr}"
                "  ),"
                "  {w_agg}"
                ")"
            ),
            param_grid={
                "price_col": ["high"],      # 如需扩展可加 "close"
                "w_r": [5, 10],
                "w_corr": [5, 10],
                "w_agg": [3, 5],
                "agg_fn": ["ts_max", "ts_mean"],   # 032 可视作 sum/decay 变种，回测有需要再开
            },
        )

        # 1.4 高位波动风险（从 191_042 / 101_040 拆出的单腿）
        self._add_parametric_feature(
            base_name="os_high_vol_risk",
            expr_tpl="-1 * cs_rank(ts_std(high, {w_vol}))",
            param_grid={"w_vol": [5, 10, 20]},
        )

        # 1.5 振幅 & 量的单腿（从 alpha101_040 拆出）
        self._add_parametric_feature(
            base_name="os_range_hl",
            expr_tpl="-1 * cs_rank(high / low)",
            param_grid=None,
        )
        self._add_parametric_feature(
            base_name="os_vol_cs_rank",
            expr_tpl="-1 * cs_rank(volume)",
            param_grid=None,
        )

        # ------------------------------------------------------------------
        # 2) 回撤 / 低位 / 分位 family
        # ------------------------------------------------------------------

        # 2.1 Regime + Drawdown family
        # 覆盖：
        # - alpha191_098
        # - alpha101_024
        # long_term_returm ~ (mean(close, w_ma) 与 w_reg 期前均价比)
        self._add_parametric_feature(
            base_name="os_regime_drawdown",
            expr_tpl=(
                "ts_where("
                # 长期趋势 regime：类似 ((Δ MA) / delay(close, w_reg)) <= theta
                "  (ts_mean(close, {w_ma}) - ts_delay(ts_mean(close, {w_ma}), {w_reg}))"
                "    / ts_delay(close, {w_reg}) <= {theta},"
                # 弱趋势：用长期回撤腿
                "  -1 * (close - ts_min(close, {w_dd})),"
                # 强趋势：用短期反转腿
                "  -1 * ts_delta(close, {w_short})"
                ")"
            ),
            param_grid={
                "w_ma": [60, 100],
                "w_reg": [60, 100],
                "w_dd": [60, 100],
                "w_short": [3, 5],
                "theta": [0.03, 0.05],
            },
        )

        # 2.2 低点类：过去 w 日最低价 / 当前 close
        # 覆盖：
        # - alpha158_min_w
        self._add_parametric_feature(
            base_name="os_low_min_pos",
            expr_tpl="ts_min(low, {w}) / close",
            param_grid={"w": windows},
        )

        # 2.3 分位数低位类：过去 w 日分位数 / 当前 close
        # 覆盖：
        # - alpha158_qtld_w（q=0.2）
        self._add_parametric_feature(
            base_name="os_price_quantile_pos",
            expr_tpl="ts_quantile(close, {w}, {q}) / close",
            param_grid={
                "w": windows,
                "q": [0.1, 0.2, 0.3],
            },
        )

        # ------------------------------------------------------------------
        # 3) K 线形态 / 反转 family
        # ------------------------------------------------------------------

        # 3.1 下影线 / 上影线 / 实体比例（K 线结构）
        # 覆盖 & 扩展：
        # - alpha158_klow_2：下影线 (ts_less(open, close) - low)/(high-low)
        self._add_parametric_feature(
            base_name="os_candle_shadow_lower",
            expr_tpl="(ts_less(open, close) - low) / (high - low + 1e-12)",
            param_grid=None,
        )
        self._add_parametric_feature(
            base_name="os_candle_shadow_upper",
            expr_tpl="(high - ts_greater(open, close)) / (high - low + 1e-12)",
            param_grid=None,
        )
        self._add_parametric_feature(
            base_name="os_candle_body",
            expr_tpl="ts_abs(close - open) / (high - low + 1e-12)",
            param_grid=None,
        )

        # 3.2 新高回落（Breakout + Pullback）
        # 覆盖：
        # - alpha101_023
        self._add_parametric_feature(
            base_name="os_breakout_pullback",
            expr_tpl=(
                "ts_where("
                "  high > ts_mean(high, {w_ma}),"
                "  -1 * (high - ts_delay(high, {w_d})),"
                "  0"
                ")"
            ),
            param_grid={
                "w_ma": [20, 60],
                "w_d": [1, 2, 3],
            },
        )

        # 3.3 放量中期反转（Volume Filter Reversal）
        # 覆盖并改良：
        # - alpha101_007：条件 volume > adv20，使用 7 日价差反转
        self._add_parametric_feature(
            base_name="os_volume_filter_reversal",
            expr_tpl=(
                "ts_where("
                # 放量条件（你也可以改成基于 quantile 的条件）
                "  volume > ts_mean(volume, {w_adv}),"
                "  (-1 * ts_rank(abs(close - ts_delay(close, {w_d})), {w_rank}))"
                "    * ts_sign(close - ts_delay(close, {w_d})),"
                # else: 无信号时给 0，避免硬编码 -1
                "  0"
                ")"
            ),
            param_grid={
                "w_adv": [20],
                "w_d": [5, 7],
                "w_rank": [30, 60],
            },
        )

        # ------------------------------------------------------------------
        # 4) 其它：从 alpha101_071 拆出的两条腿（可选）
        # ------------------------------------------------------------------

        # 4.1 close vs adv180 corr（价量相关长期腿）
        # 左路：corr(ts_rank(close,3), ts_rank(adv180,12), 18) 的简化 family 版
        self._add_parametric_feature(
            base_name="os_close_adv_corr",
            expr_tpl=(
                "ts_corr("
                "  ts_rank(close, {w_r_price}),"
                "  ts_rank(ts_mean(volume, {w_adv}), {w_r_vol}),"
                "  {w_corr}"
                ")"
            ),
            param_grid={
                "w_r_price": [3],
                "w_adv": [180],
                "w_r_vol": [12],
                "w_corr": [10, 18],
            },
        )

        # 4.2 (low + open - 2*vwap) 形态偏离腿
        # 右路：((low+open-2*vwap)^2) 经过 cs_rank + decay + ts_rank
        self._add_parametric_feature(
            base_name="os_candle_vwap_deviation",
            expr_tpl=(
                "ts_rank("
                "  ts_decay_linear("
                "    cs_rank((low + open - ({vwap} + {vwap})) ** 2),"
                "    {w_dec}"
                "  ),"
                "  {w_tr}"
                ")"
            ).replace("{vwap}", vwap),
            param_grid={
                "w_dec": [8, 16],
                "w_tr": [4, 8],
            },
        )


        # ------------------------------------------------------------------
        # 5) 收益反转 family：os_ret_reversal
        #    核心思想：-rank(Δ_w ret)，对“最近跌得多”的票给正向信号
        #    覆盖 / 抽象自：
        #      - alpha191_136 / alpha101_014 中的 ret 变化那条腿
        # ------------------------------------------------------------------
        self._add_parametric_feature(
            base_name="os_ret_reversal",
            expr_tpl=(
                "-1 * cs_rank("
                "  ts_delta({ret}, {w_d})"
                ")"
            ).format(ret=ret, w_d="{w_d}"),
            param_grid={
                # 这里可以按你策略节奏再调，比如 [3, 5, 10]
                "w_d": [1, 3, 5, 10],
            },
        )

    def register_all(self, windows: list[int] | None = None) -> None:
        self.opensource_factor_families(windows)