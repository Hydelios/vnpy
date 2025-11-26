from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import warnings
import pandas as pd


@dataclass
class PlotConfig:
    returns: bool = True
    information: bool = True
    turnover: bool = True
    full_if_empty: bool = True

    @staticmethod
    def from_any(obj: dict | None) -> "PlotConfig":
        if not isinstance(obj, dict):
            return PlotConfig()
        cfg = PlotConfig()
        for k in ("returns", "information", "turnover", "full_if_empty"):
            if k in obj:
                setattr(cfg, k, bool(obj[k]))
        return cfg


PLOT_DEFAULTS = PlotConfig()


def display_tearsheet(
    clean_data: pd.DataFrame,
    *,
    long_short: Optional[bool] = None,
    group_neutral: Optional[bool] = None,
    by_group: Optional[bool] = None,
) -> None:
    from alphalens.tears import create_full_tear_sheet  # type: ignore
    create_full_tear_sheet(
        clean_data,
        long_short=bool(long_short) if long_short is not None else False,
        group_neutral=bool(group_neutral) if group_neutral is not None else False,
        by_group=bool(by_group) if by_group is not None else False,
    )


def render_and_save(
    clean_data: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 150,
    plots: PlotConfig | dict | None = None,
) -> None:
    """分项绘制并保存（PNG+合并PDF），带图开关。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
    from alphalens import tears as al_tears  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)
    # 非交互 + 清理残留图
    try:
        import matplotlib.pyplot as _plt
        _plt.ioff()
        _plt.close("all")
    except Exception:
        pass

    fig_idx = 1

    def _save_new(prefix: str, pdf: PdfPages, before: set[int]) -> int:
        nonlocal fig_idx
        new_nums = sorted(set(plt.get_fignums()) - before)
        for num in new_nums:
            fig = plt.figure(num)
            try:
                fig.patch.set_facecolor("white")
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

    flags = PlotConfig.from_any(plots or PLOT_DEFAULTS.__dict__)

    old_show = plt.show
    plt.show = (lambda *a, **k: None)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*FigureCanvasAgg is non-interactive.*",
                category=UserWarning,
            )
            with PdfPages(out_dir / "tear_sheet.pdf") as pdf:
                captured = 0
                if flags.returns:
                    before = set(plt.get_fignums())
                    al_tears.create_returns_tear_sheet(clean_data)
                    _save_new("returns", pdf, before)

                if flags.information:
                    before = set(plt.get_fignums())
                    al_tears.create_information_tear_sheet(clean_data)
                    _save_new("information", pdf, before)

                if flags.turnover:
                    before = set(plt.get_fignums())
                    al_tears.create_turnover_tear_sheet(clean_data)
                    captured = _save_new("turnover", pdf, before)

                if flags.full_if_empty and fig_idx == 1 and captured == 0:
                    try:
                        before = set(plt.get_fignums())
                        al_tears.create_full_tear_sheet(clean_data)
                        _save_new("full", pdf, before)
                    except Exception:
                        pass
    finally:
        plt.show = old_show

    # 若无图片，输出说明图避免空目录
    try:
        if fig_idx == 1 and not list((out_dir).glob("*.png")):
            import matplotlib.pyplot as _plt2  # type: ignore
            from matplotlib.backends.backend_pdf import PdfPages as _Pdf2  # type: ignore
            fig = _plt2.figure(figsize=(6, 4), dpi=dpi)
            fig.patch.set_facecolor("white")
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.01, 0.7, "未生成 Alphalens 图表", fontsize=14, fontweight="bold")
            ax.text(0.01, 0.45, "可能原因：样本不足/periods 不匹配/版本差异", fontsize=11)
            ax.text(0.01, 0.25, "建议：缩短 periods，增大 max_loss，或使用 quantiles=5/3", fontsize=11)
            fig.savefig(out_dir / "fallback_info.png", dpi=dpi, bbox_inches="tight", transparent=False)
            with _Pdf2(out_dir / "tear_sheet.pdf") as _pdf:
                _pdf.savefig(fig, bbox_inches="tight")
            _plt2.close(fig)
    except Exception:
        pass

