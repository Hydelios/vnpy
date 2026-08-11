"""AlphaInspect、SimpleBacktest 与 Barra 信号分析。"""

from __future__ import annotations

import gc
import json

from vnpy.alpha import AlphaLab
from vnpy.alpha.analysis.alphainspect_backend import (
    create_factor_sheet,
    create_ic_report,
    prepare_alphainspect_input,
)
from vnpy.alpha.barra.barra_exposure_analysis import (
    CNE5_FACTORS,
    run_topk_signal_barra_exposure_analysis,
)
from vnpy.alpha.signal import run_simple_signal_backtest_from_lab
from vnpy.trader.constant import Interval

from .common import json_default
from .config import AssessmentConfig


def run_signal_analysis(
    config: AssessmentConfig,
    lab: AlphaLab,
    signal_variant: str,
    *,
    force: bool = False,
) -> dict:
    """运行一个信号的三类分析，并返回可序列化摘要。"""
    if signal_variant not in config.signal_paths:
        raise ValueError(f"未知 signal_variant={signal_variant!r}")
    output_dir = config.output_root / "signal_analysis" / signal_variant
    summary_path = output_dir / "signal_analysis_summary.json"
    required = [
        output_dir / "rank_ic_summary.csv",
        output_dir / "rank_ic_annual.csv",
        output_dir / "alphainspect_3x2.png",
        output_dir / "simple_backtest_o2o/rolling_metrics.csv",
        output_dir / "simple_backtest_o2o/annual_metrics.csv",
        output_dir / "simple_backtest_o2o/turnover_metrics.csv",
        output_dir / "barra_cne5_top300_equal_weight/barra_cne5_annual.csv",
        output_dir / "barra_cne5_top300_equal_weight/barra_cne5_latest.csv",
    ]
    if not force and summary_path.exists() and all(path.exists() for path in required):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("period") == [config.start_date, config.end_date]:
            print(f"复用信号分析: {signal_variant}")
            return summary

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"开始信号分析: {signal_variant}")
    prepared = prepare_alphainspect_input(
        lab,
        config.signal_paths[signal_variant],
        factor="signal",
        horizons=config.horizons,
        start_date=config.start_date,
        end_date=config.end_date,
        interval=Interval.DAILY,
        extended_days=400,
    )
    signal = prepared.signal
    analysis_df = prepared.analysis_frame
    forward_returns = list(prepared.forward_returns)
    create_ic_report(
        analysis_df,
        factor="signal",
        forward_returns=forward_returns,
        method="rank_ic",
        output_dir=output_dir,
    )
    create_factor_sheet(
        analysis_df,
        factor="signal",
        forward_return=config.plot_forward_return,
        quantiles=config.quantiles,
        turnover_periods=config.horizons,
        title=f"{signal_variant} | {config.plot_forward_return}",
        output_path=output_dir / "alphainspect_3x2.png",
    )
    simple_output_dir = output_dir / "simple_backtest_o2o"
    run_simple_signal_backtest_from_lab(
        signal=signal,
        lab=lab,
        config=config.simple_backtest_config,
        output_dir=simple_output_dir,
        interval=Interval.DAILY,
    )
    barra_output_dir = output_dir / "barra_cne5_top300_equal_weight"
    run_topk_signal_barra_exposure_analysis(
        signal=signal,
        barra_path=config.barra_path,
        output_dir=barra_output_dir,
        top_k=300,
        factors=CNE5_FACTORS,
        date_align="same_date",
        title=f"{signal_variant} Top300 Equal Weight Barra CNE5",
    )
    summary = {
        "signal_variant": signal_variant,
        "signal_path": str(config.signal_paths[signal_variant]),
        "period": [config.start_date, config.end_date],
        "signal_start": str(prepared.start),
        "signal_end": str(prepared.end),
        "signal_rows": signal.height,
        "signal_dates": signal["datetime"].n_unique(),
        "signal_symbols": signal["vt_symbol"].n_unique(),
        "forward_returns": forward_returns,
        "output_dir": str(output_dir),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    del prepared, signal, analysis_df
    gc.collect()
    return summary
