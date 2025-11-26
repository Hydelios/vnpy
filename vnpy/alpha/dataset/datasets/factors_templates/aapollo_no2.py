from .baseAlphaStrategy import BaseAlphaStrategy
import polars as pl

class ApolloNo2(BaseAlphaStrategy):
    """
    阿波罗一号：只关注因子长什么样
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
        # 1. 设定 Label
        self.set_label("ts_delay(close, -3) / ts_delay(close, -1) - 1")
        
        # 2. 定义因子
        self.register_pv_families()
        self.register_trend_families()

    def register_pv_families(self):
        # 直接调用基类的方法
        self.add_parametric_feature(
            base_name="os_pv_corr",
            expr_tpl="-1 * ts_corr({price_col}, {vol_expr}, {w_corr})",
            param_grid={
                "price_col": ["open", "close"],
                "vol_expr": ["volume"],
                "w_corr": [10, 20]
            },
            category="PriceVolume",
            sub_category="Correlation"
        )

    def register_trend_families(self):
         self.add_parametric_feature(
            base_name="os_ma_trend",
            expr_tpl="ts_mean(close, {window}) > ts_mean(close, {window} * 2)",
            param_grid={"window": [10, 20]},
            category="Trend",
            sub_category="MA_Cross"
         )