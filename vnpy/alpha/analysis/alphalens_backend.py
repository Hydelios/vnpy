from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
import polars as pl
import warnings


def build_clean(
    factor_s: pd.Series,
    price_df: pd.DataFrame,
    *,
    periods: Tuple[int, ...],
    quantiles: Optional[int],
    bins: Optional[Iterable[float]],
    max_loss: float,
) -> pd.DataFrame:
    """Build cleaned factor data via Alphalens.

    Minimal、纯函数式封装，避免在上层模板中夹杂 Alphalens 细节。
    """
    from alphalens.utils import get_clean_factor_and_forward_returns

    clean = get_clean_factor_and_forward_returns(
        factor_s,
        price_df,
        periods=periods,
        quantiles=None if bins is not None else quantiles,
        bins=bins,
        max_loss=max_loss,
    )
    return clean


def render_and_save(
    clean_data: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 150,
) -> None:
    """Render Alphalens tear sheets and save as PNG + a combined PDF.

    为提高稳定性：分项生成（returns/information/quantile/turnover），生成后立即保存并关闭。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
    from alphalens import tears as al_tears  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)
    # 非交互模式 + 清理残留图形
    try:
        import matplotlib.pyplot as _plt
        _plt.ioff()
        _plt.close('all')
    except Exception:
        pass

    fig_idx = 1
    def _save_new(prefix: str, pdf: PdfPages, before: set[int]) -> int:
        nonlocal fig_idx
        new_nums = sorted(set(plt.get_fignums()) - before)
        for num in new_nums:
            fig = plt.figure(num)
            try:
                fig.patch.set_facecolor('white')
            except Exception:
                pass
            try:
                fig.canvas.draw()
                try:
                    fig.tight_layout()
                except Exception:
                    pass
            except Exception:
                pass
            fig.savefig(out_dir / f"{prefix}_fig_{fig_idx:02d}.png", dpi=dpi, bbox_inches="tight", transparent=False)
            pdf.savefig(fig, bbox_inches="tight")
            fig_idx += 1
        for num in new_nums:
            plt.close(plt.figure(num))
        return len(new_nums)

    # 屏蔽 Agg 的 show 警告，并暂时禁用 plt.show
    old_show = plt.show
    plt.show = (lambda *a, **k: None)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*FigureCanvasAgg is non-interactive, and thus cannot be shown.*",
                category=UserWarning,
            )
            with PdfPages(out_dir / "tear_sheet.pdf") as pdf:
                before = set(plt.get_fignums())
                al_tears.create_returns_tear_sheet(clean_data)
                _save_new("returns", pdf, before)

                before = set(plt.get_fignums())
                al_tears.create_information_tear_sheet(clean_data)
                _save_new("information", pdf, before)

                # 某些版本可能缺失 quantile 报告，故保留注释/跳过
                # before = set(plt.get_fignums())
                # al_tears.create_quantile_tear_sheet(clean_data)
                # _save_new("quantile", pdf, before)

                before = set(plt.get_fignums())
                al_tears.create_turnover_tear_sheet(clean_data)
                captured = _save_new("turnover", pdf, before)

                # 兜底：若未捕获到任何图，尝试 full tear sheet
                if fig_idx == 1 and captured == 0:
                    try:
                        before = set(plt.get_fignums())
                        al_tears.create_full_tear_sheet(clean_data)
                        _save_new("full", pdf, before)
                    except Exception:
                        pass
    finally:
        plt.show = old_show

    # 若仍无图片，生成一张说明图，避免空目录
    try:
        import os as _os
        if not any(str(p).endswith('.png') for p in out_dir.glob('*.png')):
            import matplotlib.pyplot as _plt2
            from matplotlib.backends.backend_pdf import PdfPages as _Pdf2
            fig = _plt2.figure(figsize=(8, 3))
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.text(0.01, 0.7, "未生成 Alphalens 图表", fontsize=14, fontweight='bold')
            ax.text(0.01, 0.45, "可能原因：数据样本不足/periods 不匹配/版本 API 差异", fontsize=11)
            ax.text(0.01, 0.25, "建议：缩短 periods，增大 max_loss，或使用 quantiles=5/3", fontsize=11)
            fig.savefig(out_dir / "fallback_info.png", dpi=dpi, bbox_inches='tight', transparent=False)
            with _Pdf2(out_dir / "tear_sheet.pdf") as _pdf:
                _pdf.savefig(fig, bbox_inches='tight')
            _plt2.close(fig)
    except Exception:
        pass


def display_tearsheet(clean_data: pd.DataFrame) -> None:
    """仅展示，不保存。"""
    from alphalens.tears import create_full_tear_sheet  # type: ignore
    create_full_tear_sheet(clean_data)


def derive_default_periods(interval: str) -> tuple[int, ...]:
    """根据数据频率给出默认 forward 周期集合。"""
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


def build_clean_with_fallback(
    factor_s: pd.Series,
    price_df: pd.DataFrame,
    *,
    interval: str,
    cfg: dict,
) -> pd.DataFrame:
    """根据 cfg 构建 clean_data，并在必要时做兜底（调整 quantiles/periods）。

    cfg 支持键：alphalens_periods, alphalens_quantiles, alphalens_bins, alphalens_max_loss。
    """
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

    # 基本健壮性检查
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
    # 若为空且使用 quantiles，尝试 5/3 兜底
    if clean_data.empty and bins_cfg is None:
        for qs in ({5, 3} - {quantiles}):
            tmp = _try(periods, qs)
            if not tmp.empty:
                clean_data, quantiles = tmp, qs
                break
    # 仍为空则缩短周期
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


def build_factor_and_price(
    dataset,
    factor_name: str,
    start,
    end,
    *,
    keys_from: str = "infer",
) -> tuple[pd.Series, pd.DataFrame]:
    """从 dataset 中按 keys_from（infer/raw）对齐并提取 (factor_s, price_df)。"""
    from ..dataset.template import query_by_time  # 避免顶层循环导入

    # 选择键来源
    src_df: pl.DataFrame = getattr(dataset, "infer_df") if keys_from == "infer" else getattr(dataset, "raw_df")
    keys = query_by_time(src_df, start, end).select(["datetime", "vt_symbol"]).unique()

    df_res: pl.DataFrame = query_by_time(dataset.result_df, start, end).join(
        keys, on=["datetime", "vt_symbol"], how="inner"
    )

    factor_s = (
        df_res.select(["datetime", "vt_symbol", factor_name])
        .to_pandas()
        .set_index(["datetime", "vt_symbol"])[factor_name]
        .sort_index()
    )
    price_df = (
        df_res.select(["datetime", "vt_symbol", "close"]).to_pandas()
        .pivot(index="datetime", columns="vt_symbol", values="close").sort_index()
    )
    return factor_s, price_df
