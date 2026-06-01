"""Curated time-series supreme factor library.

Only factors that pass the factor-ingestion SOP should be added here.
Research candidates should stay in exploratory files such as
``ai_ts_factors.py`` until they are admitted.
"""

from __future__ import annotations

import polars as pl

from .baseAlphaStrategy import BaseAlphaStrategy


class AlphaTsSupremeFactors(BaseAlphaStrategy):
    """正式入库的时序精选因子。"""

    CATEGORY = "AlphaTSSupreme"
    VERSION = "v20260515_00"

    CANDIDATE_SPECS: list[dict[str, object]] = []

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
        self.set_label("ts_delay(close, -1) / close - 1")
        self.register_all()

    def register_all(self) -> None:
        self._add_specs(self.CANDIDATE_SPECS)

    @staticmethod
    def _normalize_base_name(base_name: str) -> str:
        if base_name.startswith("alpha_ts_supreme_"):
            return base_name
        if base_name.startswith("alpha_ts_"):
            return f"alpha_ts_supreme_{base_name[9:]}"
        return f"alpha_ts_supreme_{base_name}"

    def _add_specs(self, specs: list[dict[str, object]]) -> None:
        for spec in specs:
            self.add_parametric_feature(
                base_name=self._normalize_base_name(str(spec["base_name"])),
                expr_tpl=str(spec["expr_tpl"]),
                param_grid=spec.get("param_grid") or {},
                category=self.CATEGORY,
                sub_category=str(spec["sub_category"]),
            )
