from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable


@dataclass
class AlphalensConfig:
    """Alphalens 配置对象（仅承载常用键）。"""

    alphalens_periods: Optional[tuple[int, ...]] = None
    alphalens_quantiles: Optional[int] = 10
    alphalens_bins: Optional[list[float]] = None
    alphalens_max_loss: float = 1.0

    alphalens_keys_from: str = "infer"      # infer/raw
    alphalens_values_from: str = "result"    # result/infer/raw
    alphalens_quantile_order: str = "asc"    # asc/desc

    alphalens_save_figs: bool = False
    alphalens_figs_dpi: int = 150
    alphalens_save_analysis: bool = False
    alphalens_analysis_dir: Optional[str] = None


def _as_tuple_int(seq) -> Optional[tuple[int, ...]]:
    if seq is None:
        return None
    try:
        return tuple(int(x) for x in seq)
    except Exception:
        return None


def _as_bins(seq) -> Optional[list[float]]:
    if seq is None:
        return None
    try:
        return [float(x) for x in seq]
    except Exception:
        return None


def merge_config(base: dict | None, overrides: dict | None) -> AlphalensConfig:
    """合并配置（调用时覆盖 > base），并做轻量合法化转换。"""
    b = dict(base or {})
    o = dict(overrides or {})
    cfg = {**b, **o}

    return AlphalensConfig(
        alphalens_periods=_as_tuple_int(cfg.get("alphalens_periods")),
        alphalens_quantiles=int(cfg.get("alphalens_quantiles")) if cfg.get("alphalens_quantiles") is not None else None,
        alphalens_bins=_as_bins(cfg.get("alphalens_bins")),
        alphalens_max_loss=float(cfg.get("alphalens_max_loss", 1.0)),
        alphalens_keys_from=str(cfg.get("alphalens_keys_from", "infer")).lower(),
        alphalens_values_from=str(cfg.get("alphalens_values_from", "result")).lower(),
        alphalens_quantile_order=str(cfg.get("alphalens_quantile_order", "asc")).lower(),
        alphalens_save_figs=bool(cfg.get("alphalens_save_figs", False)),
        alphalens_figs_dpi=int(cfg.get("alphalens_figs_dpi", 150)),
        alphalens_save_analysis=bool(cfg.get("alphalens_save_analysis", False)),
        alphalens_analysis_dir=cfg.get("alphalens_analysis_dir"),
    )

