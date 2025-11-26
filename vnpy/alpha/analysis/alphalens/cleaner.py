from __future__ import annotations

from typing import Optional, Tuple, Iterable
import pandas as pd


def derive_default_periods(interval: str) -> tuple[int, ...]:
    itv = (interval or "1d").lower()
    if itv in {"1d", "d", "daily"}:
        return (1, 5, 10, 20)
    if itv == "60m":
        return (1, 4, 12)
    if itv == "30m":
        return (1, 8, 24)
    if itv == "10m":
        return (1, 24, 72)
    if itv == "1m":
        return (1, 30, 120)
    return (1, 3, 5)


def build_clean(
    factor_s: pd.Series,
    price_df: pd.DataFrame,
    *,
    periods: Tuple[int, ...],
    quantiles: Optional[int],
    bins: Optional[Iterable[float]],
    max_loss: float,
) -> pd.DataFrame:
    from alphalens.utils import get_clean_factor_and_forward_returns
    return get_clean_factor_and_forward_returns(
        factor=factor_s,
        prices=price_df,
        periods=periods,
        quantiles=None if bins is not None else quantiles,
        bins=bins,
        max_loss=max_loss,
    )


def build_clean_with_fallback(
    factor_s: pd.Series,
    price_df: pd.DataFrame,
    *,
    interval: str,
    cfg: dict,
) -> pd.DataFrame:
    from alphalens.utils import get_clean_factor_and_forward_returns, get_forward_returns_columns

    default_periods = derive_default_periods(interval)

    def _as_int_tuple(x, default: tuple[int, ...]) -> tuple[int, ...]:
        try:
            if isinstance(x, (list, tuple)):
                t = tuple(int(p) for p in x)
                return t or default
        except Exception:
            pass
        return default

    def _as_int(x, default: int) -> int:
        try:
            return int(x)
        except Exception:
            return default

    def _as_float(x, default: float) -> float:
        try:
            return float(x)
        except Exception:
            return default

    periods = _as_int_tuple(cfg.get("alphalens_periods"), default_periods)
    quantiles = _as_int(cfg.get("alphalens_quantiles", 10), 10)
    bins_cfg = cfg.get("alphalens_bins") if "alphalens_bins" in cfg else None
    try:
        if bins_cfg is not None:
            bins_cfg = [float(b) for b in bins_cfg]
    except Exception:
        bins_cfg = None
    max_loss = _as_float(cfg.get("alphalens_max_loss", 1.0), 1.0)

    if factor_s.empty:
        raise ValueError("因子序列为空：没有可用的 (datetime, vt_symbol) 样本对")
    if price_df.empty:
        raise ValueError("价格矩阵为空：没有可用的价格数据")
    if not periods:
        periods = default_periods

    def _try(periods_tuple: tuple[int, ...], qs: int) -> pd.DataFrame:
        return get_clean_factor_and_forward_returns(
            factor=factor_s,
            prices=price_df,
            periods=periods_tuple,
            quantiles=None if bins_cfg is not None else qs,
            bins=bins_cfg,
            max_loss=max_loss,
        )

    clean_data = _try(periods, quantiles)
    if clean_data.empty and bins_cfg is None:
        for qs in ({5, 3} - {quantiles}):
            tmp = _try(periods, qs)
            if not tmp.empty:
                clean_data, quantiles = tmp, qs
                break
    if clean_data.empty:
        for p in [(periods[0],), (1,)]:
            tmp = _try(tuple(p), quantiles)
            if not tmp.empty:
                clean_data, periods = tmp, tuple(p)
                break

    fwd_cols = get_forward_returns_columns(clean_data.columns)
    if len(fwd_cols) == 0:
        raise ValueError(
            "Alphalens 未生成远期收益列，请检查价格数据与周期设置："
            f"prices.shape={price_df.shape}, factor.len={len(factor_s)}, periods={periods}."
        )
    if clean_data.empty:
        raise ValueError(
            "清洗后的因子数据为空，无法绘制绩效报告；"
            "请检查样本期、过滤条件与 periods 是否过大导致对齐后无有效数据。"
        )
    return clean_data


def flip_quantile_order(clean_data: pd.DataFrame, *, order: str | None) -> pd.DataFrame:
    if not isinstance(clean_data, pd.DataFrame):
        return clean_data
    if (order or "asc").lower() != "desc":
        return clean_data
    if "factor_quantile" not in clean_data.columns:
        return clean_data
    try:
        q_col = pd.to_numeric(clean_data["factor_quantile"], errors="coerce").astype(int)
        if q_col.empty:
            return clean_data
        n_bins = int(q_col.max())
        if n_bins <= 0:
            return clean_data
        cd = clean_data.copy()
        cd["factor_quantile"] = (n_bins + 1 - q_col).astype(int)
        return cd
    except Exception:
        return clean_data

