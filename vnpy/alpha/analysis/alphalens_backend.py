"""
门面模块：向后兼容旧路径导入，提供惰性封装，避免导入时依赖链问题。
"""

from __future__ import annotations

__all__ = [
    "run_factor_analysis",
    "summarize_factor",
    "summarize_factors",
    "compute_summary_metrics",
    "PlotConfig",
    "PLOT_DEFAULTS",
]


def run_factor_analysis(*args, **kwargs):
    from importlib import import_module, reload
    mod = import_module('vnpy.alpha.analysis.alphalens.runner')
    try:
        mod = reload(mod)
    except Exception:
        pass
    return mod.run_factor_analysis(*args, **kwargs)


def summarize_factor(*args, **kwargs):
    from importlib import import_module, reload
    mod = import_module('vnpy.alpha.analysis.alphalens.summary')
    try:
        mod = reload(mod)
    except Exception:
        pass
    return mod.summarize_factor(*args, **kwargs)


def summarize_factors(*args, **kwargs):
    from importlib import import_module, reload
    mod = import_module('vnpy.alpha.analysis.alphalens.summary')
    try:
        mod = reload(mod)
    except Exception:
        pass
    return mod.summarize_factors(*args, **kwargs)


def compute_summary_metrics(*args, **kwargs):
    from importlib import import_module, reload
    mod = import_module('vnpy.alpha.analysis.alphalens.metrics')
    try:
        mod = reload(mod)
    except Exception:
        pass
    return mod.compute_summary_metrics(*args, **kwargs)


try:
    from .alphalens.tears import PlotConfig as _PlotConfig, PLOT_DEFAULTS as _PLOT_DEFAULTS
    PlotConfig = _PlotConfig
    PLOT_DEFAULTS = _PLOT_DEFAULTS
except Exception:  # pragma: no cover - 若依赖缺失，延迟到实际使用时报错
    PlotConfig = None  # type: ignore
    PLOT_DEFAULTS = None  # type: ignore
