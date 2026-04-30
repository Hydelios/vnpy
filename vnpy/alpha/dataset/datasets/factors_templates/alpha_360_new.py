from __future__ import annotations

from typing import Final

import polars as pl

from vnpy.alpha import AlphaDataset


DEFAULT_ALPHA360_WINDOWS: Final[tuple[int, ...]] = (5, 10, 20, 30, 60)
DEFAULT_ALPHA360_LABEL: Final[str] = "ts_delay(close, -3) / ts_delay(close, -1) - 1"


class Alpha360NEW(AlphaDataset):
    """Fixed-window OHLCV/VWAP, Amihud, and shadow-asymmetry features."""

    DEFAULT_WINDOWS: tuple[int, ...] = DEFAULT_ALPHA360_WINDOWS

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

        self.windows = tuple(int(window) for window in self.DEFAULT_WINDOWS)
        if not self.windows:
            raise ValueError("alpha360.DEFAULT_WINDOWS 不能为空")
        if any(window <= 0 for window in self.windows):
            raise ValueError("alpha360.DEFAULT_WINDOWS 必须全部大于 0")

        self._ensure_vwap_column()
        self._ensure_turnover_column()
        self.set_label(DEFAULT_ALPHA360_LABEL)

        ret1 = "close / (ts_delay(close, 1) + 1e-12) - 1"
        amihud = f"abs({ret1}) / (turnover + 1e-12)"
        upper_shadow = "(high - ts_greater(open, close)) / (close + 1e-12)"
        lower_shadow = "(ts_less(open, close) - low) / (close + 1e-12)"
        shadow_asym = f"(({lower_shadow}) - ({upper_shadow})) / (high - low + 1e-12)"

        self.add_feature("alpha360_amihud_raw", amihud)
        self.add_feature("alpha360_amihud_cs_rank", f"cs_rank({amihud})")
        self.add_feature("alpha360_upper_shadow_raw", upper_shadow)
        self.add_feature("alpha360_lower_shadow_raw", lower_shadow)
        self.add_feature("alpha360_shadow_asym_raw", shadow_asym)

        for window in self.windows:
            self.add_feature(f"alpha360_close_{window}", f"ts_delay(close, {window}) / close")
            self.add_feature(f"alpha360_open_{window}", f"ts_delay(open, {window}) / close")
            self.add_feature(f"alpha360_high_{window}", f"ts_delay(high, {window}) / close")
            self.add_feature(f"alpha360_low_{window}", f"ts_delay(low, {window}) / close")
            self.add_feature(f"alpha360_vwap_{window}", f"ts_delay(vwap, {window}) / close")
            self.add_feature(f"alpha360_volume_{window}", f"ts_delay(volume, {window}) / (volume + 1e-12)")

            self.add_feature(
                f"alpha360_close_cs_pct_rank_{window}",
                f"cs_pct_rank(ts_delay(close, {window}) / close)",
            )
            self.add_feature(
                f"alpha360_open_cs_pct_rank_{window}",
                f"cs_pct_rank(ts_delay(open, {window}) / close)",
            )
            self.add_feature(
                f"alpha360_high_cs_pct_rank_{window}",
                f"cs_pct_rank(ts_delay(high, {window}) / close)",
            )
            self.add_feature(
                f"alpha360_low_cs_pct_rank_{window}",
                f"cs_pct_rank(ts_delay(low, {window}) / close)",
            )
            self.add_feature(
                f"alpha360_vwap_cs_pct_rank_{window}",
                f"cs_pct_rank(ts_delay(vwap, {window}) / close)",
            )
            self.add_feature(
                f"alpha360_volume_cs_pct_rank_{window}",
                f"cs_pct_rank(ts_delay(volume, {window}) / (volume + 1e-12))",
            )

        for window in self.windows:
            self.add_feature(f"alpha360_amihud_mean_{window}", f"ts_mean({amihud}, {window})")
            self.add_feature(f"alpha360_amihud_std_{window}", f"ts_std({amihud}, {window})")
            self.add_feature(f"alpha360_amihud_slope_{window}", f"ts_slope({amihud}, {window})")

            self.add_feature(f"alpha360_shadow_asym_mean_{window}", f"ts_mean({shadow_asym}, {window})")
            self.add_feature(f"alpha360_shadow_asym_std_{window}", f"ts_std({shadow_asym}, {window})")
            self.add_feature(
                f"alpha360_shadow_asym_pos_ratio_{window}",
                f"ts_mean(({shadow_asym}) > 0, {window})",
            )
            self.add_feature(
                f"alpha360_shadow_asym_q20_{window}",
                f"ts_quantile({shadow_asym}, {window}, 0.2)",
            )
            self.add_feature(
                f"alpha360_shadow_asym_q80_{window}",
                f"ts_quantile({shadow_asym}, {window}, 0.8)",
            )

    def _ensure_vwap_column(self) -> None:
        if "vwap" in self.df.columns:
            return

        if "turnover" not in self.df.columns or "volume" not in self.df.columns:
            raise ValueError("alpha360 需要 vwap 列；当前数据缺少 vwap，且无法从 turnover/volume 推导")

        self.df = self.df.with_columns(
            pl.when(pl.col("volume").abs() > 1e-12)
            .then((pl.col("turnover") / pl.col("volume")).cast(pl.Float64))
            .otherwise(float("nan"))
            .alias("vwap")
        )

    def _ensure_turnover_column(self) -> None:
        if "turnover" in self.df.columns:
            return
        raise ValueError("alpha360 需要 turnover 列；当前数据缺少 turnover，无法计算 Amihud 因子")
