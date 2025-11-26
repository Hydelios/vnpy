import polars as pl

from vnpy.alpha import AlphaDataset


class Alpha158NEW(AlphaDataset):
    """158 basic factors from Qlib"""

    def __init__(
        self,
        df: pl.DataFrame,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
        interval: str = "1d",
        enable_cache: bool = False,
        cache_dir: str | None = None
    ) -> None:
        """Constructor"""
        super().__init__(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
            interval=interval,
            enable_cache=enable_cache,
            cache_dir=cache_dir
        )

        # Candlestick pattern features
        self.add_feature("alpha158_kmid", "(close - open) / open")
        self.add_feature("alpha158_klen", "(high - low) / open")
        self.add_feature("alpha158_kmid_2", "(close - open) / (high - low + 1e-12)")
        self.add_feature("alpha158_kup", "(high - ts_greater(open, close)) / open")
        self.add_feature("alpha158_kup_2", "(high - ts_greater(open, close)) / (high - low + 1e-12)")
        self.add_feature("alpha158_klow", "(ts_less(open, close) - low) / open")
        self.add_feature("alpha158_klow_2", "((ts_less(open, close) - low) / (high - low + 1e-12))")
        self.add_feature("alpha158_ksft", "(close * 2 - high - low) / open")
        self.add_feature("alpha158_ksft_2", "(close * 2 - high - low) / (high - low + 1e-12)")

        # Price change features
        for field in ["open", "high", "low", "vwap"]:
            self.add_feature(f"alpha158_{field}_0", f"{field} / close")

        # Time series features
        windows: list[int] = [5, 10, 20, 30, 60]

        for w in windows:
            self.add_feature(f"alpha158_roc_{w}", f"ts_delay(close, {w}) / close")

        for w in windows:
            self.add_feature(f"alpha158_ma_{w}", f"ts_mean(close, {w}) / close")

        for w in windows:
            self.add_feature(f"alpha158_std_{w}", f"ts_std(close, {w}) / close")

        for w in windows:
            self.add_feature(f"alpha158_beta_{w}", f"ts_slope(close, {w}) / close")

        for w in windows:
            self.add_feature(f"alpha158_rsqr_{w}", f"ts_rsquare(close, {w})")

        for w in windows:
            self.add_feature(f"alpha158_resi_{w}", f"ts_resi(close, {w}) / close")

        for w in windows:
            self.add_feature(f"alpha158_max_{w}", f"ts_max(high, {w}) / close")

        for w in windows:
            self.add_feature(f"alpha158_min_{w}", f"ts_min(low, {w}) / close")

        for w in windows:
            self.add_feature(f"alpha158_qtlu_{w}", f"ts_quantile(close, {w}, 0.8) / close")

        for w in windows:
            self.add_feature(f"alpha158_qtld_{w}", f"ts_quantile(close, {w}, 0.2) / close")

        for w in windows:
            self.add_feature(f"alpha158_rank_{w}", f"ts_rank(close, {w})")

        for w in windows:
            self.add_feature(f"alpha158_rsv_{w}", f"(close - ts_min(low, {w})) / (ts_max(high, {w}) - ts_min(low, {w}) + 1e-12)")

        for w in windows:
            self.add_feature(f"alpha158_imax_{w}", f"ts_argmax(high, {w}) / {w}")

        for w in windows:
            self.add_feature(f"alpha158_imin_{w}", f"ts_argmin(low, {w}) / {w}")

        for w in windows:
            self.add_feature(f"alpha158_imxd_{w}", f"(ts_argmax(high, {w}) - ts_argmin(low, {w})) / {w}")

        for w in windows:
            self.add_feature(f"alpha158_corr_{w}", f"ts_corr(close, ts_log(volume + 1), {w})")

        for w in windows:
            self.add_feature(f"alpha158_cord_{w}", f"ts_corr(close / ts_delay(close, 1), ts_log(volume / ts_delay(volume, 1) + 1), {w})")

        for w in windows:
            self.add_feature(f"alpha158_cntp_{w}", f"ts_mean(close > ts_delay(close, 1), {w})")

        for w in windows:
            self.add_feature(f"alpha158_cntn_{w}", f"ts_mean(close < ts_delay(close, 1), {w})")

        for w in windows:
            self.add_feature(f"alpha158_cntd_{w}", f"ts_mean(close > ts_delay(close, 1), {w}) - ts_mean(close < ts_delay(close, 1), {w})")

        for w in windows:
            self.add_feature(f"alpha158_sump_{w}", f"ts_sum(ts_greater(close - ts_delay(close, 1), 0), {w}) / (ts_sum(ts_abs(close - ts_delay(close, 1)), {w}) + 1e-12)")

        for w in windows:
            self.add_feature(f"alpha158_sumn_{w}", f"ts_sum(ts_greater(ts_delay(close, 1) - close, 0), {w}) / (ts_sum(ts_abs(close - ts_delay(close, 1)), {w}) + 1e-12)")

        for w in windows:
            self.add_feature(f"alpha158_sumd_{w}", f"(ts_sum(ts_greater(close - ts_delay(close, 1), 0), {w}) - ts_sum(ts_greater(ts_delay(close, 1) - close, 0), {w})) / (ts_sum(ts_abs(close - ts_delay(close, 1)), {w}) + 1e-12)")

        for w in windows:
            self.add_feature(f"alpha158_vma_{w}", f"ts_mean(volume, {w}) / (volume + 1e-12)")

        for w in windows:
            self.add_feature(f"alpha158_vstd_{w}", f"ts_std(volume, {w}) / (volume + 1e-12)")

        for w in windows:
            self.add_feature(f"alpha158_wvma_{w}", f"ts_std(ts_abs(close / ts_delay(close, 1) - 1) * volume, {w}) / (ts_mean(ts_abs(close / ts_delay(close, 1) - 1) * volume, {w}) + 1e-12)")

        for w in windows:
            self.add_feature(f"alpha158_vsump_{w}", f"ts_sum(ts_greater(volume - ts_delay(volume, 1), 0), {w}) / (ts_sum(ts_abs(volume - ts_delay(volume, 1)), {w}) + 1e-12)")

        for w in windows:
            self.add_feature(f"alpha158_vsumn_{w}", f"ts_sum(ts_greater(ts_delay(volume, 1) - volume, 0), {w}) / (ts_sum(ts_abs(volume - ts_delay(volume, 1)), {w}) + 1e-12)")

        for w in windows:
            self.add_feature(f"alpha158_vsumd_{w}", f"(ts_sum(ts_greater(volume - ts_delay(volume, 1), 0), {w}) - ts_sum(ts_greater(ts_delay(volume, 1) - volume, 0), {w})) / (ts_sum(ts_abs(volume - ts_delay(volume, 1)), {w}) + 1e-12)")

        # Set label
        self.set_label("ts_delay(close, -3) / ts_delay(close, -1) - 1")
