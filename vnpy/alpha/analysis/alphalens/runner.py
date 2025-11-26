from __future__ import annotations

from typing import Optional
from pathlib import Path
import pandas as pd

from .extract import collect_period_range, build_factor_and_price2, build_factor_series
from .cleaner import build_clean_with_fallback, flip_quantile_order
from .tears import display_tearsheet, PlotConfig
from ...logger import logger


def run_factor_analysis(
    dataset,
    factor_name: str,
    *,
    cfg: Optional[dict] = None,
    interval: Optional[str] = None,
    display: bool = True,
    long_short: Optional[bool] = None,
    group_neutral: Optional[bool] = None,
    by_group: Optional[bool] = None,
    plots: Optional[dict] = None,
    shared_keys=None,
    shared_price=None,
    no_save: bool = False,
) -> pd.DataFrame:
    """抽取→清洗（带兜底）→分位翻转→可选展示→可选保存，返回 clean_data。"""
    fa = dict(cfg or {})
    itv = (interval or getattr(dataset, "interval", "1d")).lower()

    keys_from = str(fa.get("alphalens_keys_from", "infer")).lower()
    values_from = str(fa.get("alphalens_values_from", "result")).lower()
    q_order = str(fa.get("alphalens_quantile_order", "asc")).lower()

    start, end = collect_period_range(dataset)

    if shared_keys is not None and shared_price is not None:
        # 若 shared_keys 非 Polars DataFrame（如通过 spawn 传入的 pandas），在此转换回 Polars
        try:
            import polars as pl  # type: ignore
            if not isinstance(shared_keys, pl.DataFrame):
                try:
                    shared_keys = pl.from_pandas(shared_keys)  # type: ignore
                except Exception:
                    # 回退：不使用共享键
                    shared_keys = None
        except Exception:
            shared_keys = None
        factor_s = build_factor_series(
            dataset,
            factor_name,
            start,
            end,
            values_from=values_from,
            keys_df=shared_keys if shared_keys is not None else None,
        )
        price_df = shared_price
    else:
        factor_s, price_df = build_factor_and_price2(
            dataset,
            factor_name,
            start,
            end,
            keys_from=keys_from,
            values_from=values_from,
        )

    clean = build_clean_with_fallback(
        factor_s,
        price_df,
        interval=itv,
        cfg=fa,
    )

    clean = flip_quantile_order(clean, order=q_order)

    if display:
        try:
            display_tearsheet(
                clean,
                long_short=long_short,
                group_neutral=group_neutral,
                by_group=by_group,
            )
        except Exception:
            pass

    # 取消图片保存逻辑：无论配置如何，均不再落盘图片
    save_analysis = False if no_save else bool(fa.get("alphalens_save_analysis", False))
    if save_analysis:
        base_dir = fa.get("alphalens_analysis_dir")
        out_dir = (Path(base_dir) if base_dir else Path.cwd() / "analysis") / itv / factor_name
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"创建分析目录失败：{out_dir} | {e}")
            return clean

        # 尽量保证 CSV 一定写出（即使 Parquet 可用也同时写 CSV）
        # 为提高兼容性，重置索引后保存
        try:
            clean.reset_index().to_csv(out_dir / "clean.csv", index=False, encoding="utf-8-sig")
        except Exception as e:
            logger.error(f"保存 clean.csv 失败：{out_dir} | {e}")

        # 可选：若环境具备 Parquet 依赖，则同时写出 parquet 便于后续读取
        try:
            clean.to_parquet(out_dir / "clean.parquet")
        except Exception:
            # parquet 失败不影响整体流程
            pass

    return clean
