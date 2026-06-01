"""AI-generated time-series factor candidates.

This module is the dedicated entry point for CTA/T0/timing-style AI factors.
Keep pure time-series ideas here and leave cross-section stock-selection
ideas in ``ai_cs_factors.py``.
"""

from __future__ import annotations

import polars as pl

from .baseAlphaStrategy import BaseAlphaStrategy


class AiTsFactors(BaseAlphaStrategy):
    """AI 发散构思的时序候选因子。"""

    CATEGORY = "AITS"
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
        # The final TS label is evaluation-specific; this default is only a
        # placeholder for expression-library extraction.
        self.set_label("ts_delay(close, -1) / close - 1")
        self.register_all()

    def register_all(self) -> None:
        self._add_specs(self.CANDIDATE_SPECS)

    @staticmethod
    def _normalize_base_name(base_name: str) -> str:
        if base_name.startswith("ai_ts_"):
            return base_name
        if base_name.startswith("ai_"):
            return f"ai_ts_{base_name[3:]}"
        return f"ai_ts_{base_name}"

    def _add_specs(self, specs: list[dict[str, object]]) -> None:
        for spec in specs:
            self.add_parametric_feature(
                base_name=self._normalize_base_name(str(spec["base_name"])),
                expr_tpl=str(spec["expr_tpl"]),
                param_grid=spec.get("param_grid") or {},
                category=self.CATEGORY,
                sub_category=str(spec["sub_category"]),
            )


AiThinkTsFactors = AiTsFactors
