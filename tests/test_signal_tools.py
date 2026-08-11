import numpy as np
import pandas as pd
import pytest

from vnpy.alpha.signal.postprocess import normalize_signal_frame as normalize_postprocess_signal
from vnpy.alpha.signal.simple_backtest import (
    SimpleBacktestConfig,
    build_horizon_returns,
    build_rolling_sleeve_daily,
    compute_annual_metrics,
    compute_return_profile,
    compute_turnover_metrics,
    normalize_signal_frame,
    run_simple_signal_backtest,
    run_simple_signal_backtest_from_lab,
    write_simple_backtest_outputs,
)


def test_signal_normalization_drops_invalid_and_deduplicates() -> None:
    signal = pd.DataFrame(
        {
            "datetime": ["2026-01-01"] * 4,
            "vt_symbol": [None, "", "000001.SZ", "000001.SZ"],
            "signal": [1.0, 2.0, 3.0, 4.0],
        }
    )

    postprocessed = normalize_postprocess_signal(signal)
    backtest = normalize_signal_frame(signal)

    assert postprocessed[["vt_symbol", "signal"]].to_dict("records") == [
        {"vt_symbol": "000001.SZSE", "signal": 4.0}
    ]
    assert backtest[["vt_symbol", "signal"]].to_dict("records") == [
        {"vt_symbol": "000001.SZSE", "signal": 4.0}
    ]


def test_return_profile_includes_initial_nav_in_drawdown() -> None:
    profile = compute_return_profile(pd.Series([-0.1, 0.0]))

    assert profile["max_drawdown"] == pytest.approx(-0.1)


def test_rolling_sleeves_keep_cash_and_missing_return_weight() -> None:
    selected = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]),
            "vt_symbol": ["A.SSE", "B.SSE", "A.SSE"],
            "weight": [0.5, 0.5, 1.0],
        }
    )
    one_period = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
            "vt_symbol": ["A.SSE"] * 4,
            "period_return": [0.1] * 4,
        }
    )

    daily, turnover = build_rolling_sleeve_daily(selected, one_period, None, horizons=(2,))

    np.testing.assert_allclose(daily["strategy_return"], [0.025, 0.075, 0.05])
    np.testing.assert_allclose(daily["invested_weight"], [0.5, 1.0, 0.5])
    assert daily["active_sleeves"].tolist() == [1, 2, 1]
    assert daily["missing_return_count"].tolist() == [1, 1, 0]
    np.testing.assert_allclose(turnover["one_way_turnover"], [0.5, 0.5, 0.5])


def test_horizon_return_keeps_missing_constituent_weight() -> None:
    selected = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "vt_symbol": ["A.SSE", "B.SSE"],
            "weight": [0.5, 0.5],
        }
    )
    forward = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "vt_symbol": ["A.SSE", "B.SSE"],
            "ret_h1": [0.1, np.nan],
        }
    )

    result = build_horizon_returns(selected, forward, None, horizons=(1,))

    assert result.loc[0, "strategy_return"] == pytest.approx(0.05)
    assert result.loc[0, "selected_count"] == 2
    assert result.loc[0, "return_count"] == 1
    assert result.loc[0, "return_coverage_weight"] == pytest.approx(0.5)


def test_signal_gap_is_kept_as_cash_and_turnover_counts_exit_and_reentry() -> None:
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    selected = pd.DataFrame(
        {
            "datetime": [dates[0], dates[4]],
            "vt_symbol": ["A.SSE", "A.SSE"],
            "weight": [1.0, 1.0],
        }
    )
    one_period = pd.DataFrame(
        {
            "datetime": dates,
            "vt_symbol": ["A.SSE"] * len(dates),
            "period_return": [0.01] * len(dates),
        }
    )

    daily, turnover = build_rolling_sleeve_daily(selected, one_period, None, horizons=(1,))

    np.testing.assert_allclose(daily["invested_weight"], [1.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(daily["strategy_return"], [0.01, 0.0, 0.0, 0.0, 0.01])
    np.testing.assert_allclose(turnover["one_way_turnover"], [1.0, 1.0, 0.0, 0.0, 1.0])


def test_annual_and_turnover_summaries() -> None:
    rolling = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-12-31", "2026-01-02"]),
            "horizon": [1, 1],
            "strategy_return": [0.1, -0.1],
            "benchmark_return": [0.0, 0.0],
            "excess_return": [0.1, -0.1],
            "active_sleeves": [1, 1],
            "invested_weight": [1.0, 1.0],
            "return_coverage_ratio": [1.0, 1.0],
        }
    )
    turnover = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-12-31", "2026-01-02"]),
            "horizon": [1, 1],
            "one_way_turnover": [1.0, 0.2],
            "gross_stock_turnover": [1.0, 0.4],
            "stock_buy_weight": [1.0, 0.2],
            "stock_sell_weight": [0.0, 0.2],
            "invested_weight": [1.0, 1.0],
            "cash_weight": [0.0, 0.0],
            "overlap_weight": [0.0, 0.8],
            "position_count": [2, 2],
        }
    )

    annual = compute_annual_metrics(rolling)
    turnover_summary, turnover_annual = compute_turnover_metrics(turnover)

    assert annual["year"].tolist() == [2025, 2026]
    assert turnover_summary.loc[0, "avg_one_way_turnover"] == 0.6
    assert turnover_annual["year"].tolist() == [2025, 2026]


def test_full_backtest_accepts_single_required_price_column_and_writes_outputs(tmp_path) -> None:
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    signal = pd.DataFrame(
        {
            "datetime": dates[:3],
            "vt_symbol": ["A.SSE"] * 3,
            "signal": [3.0, 2.0, 1.0],
        }
    )
    prices = pd.DataFrame(
        {
            "datetime": dates,
            "vt_symbol": ["A.SSE"] * 5,
            "open": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )
    config = SimpleBacktestConfig(price_mode="o2o", horizons=(1, 2), top_k=1)

    result = run_simple_signal_backtest(signal=signal, prices=prices, config=config)
    write_simple_backtest_outputs(result, tmp_path, parquet=False)

    assert not result.annual_metrics.empty
    assert not result.turnover_metrics.empty
    assert (tmp_path / "annual_metrics.csv").exists()
    assert (tmp_path / "turnover_metrics.csv").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "report.md").exists()


def test_lab_entrypoint_loads_prices_and_writes_outputs(tmp_path) -> None:
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    signal = pd.DataFrame(
        {
            "datetime": dates[:3],
            "vt_symbol": ["A.SSE"] * 3,
            "signal": [3.0, 2.0, 1.0],
        }
    )

    class FakeLab:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def load_bar_df(self, *, vt_symbols, **kwargs):
            self.calls.append(vt_symbols)
            symbol = vt_symbols[0]
            return pd.DataFrame(
                {
                    "datetime": dates,
                    "vt_symbol": [symbol] * len(dates),
                    "open": [10.0, 11.0, 12.0, 13.0, 14.0],
                }
            )

    lab = FakeLab()
    config = SimpleBacktestConfig(
        price_mode="o2o",
        horizons=(1,),
        top_k=1,
        benchmark_symbol="000852.SSE",
    )

    result = run_simple_signal_backtest_from_lab(
        signal=signal,
        lab=lab,
        config=config,
        output_dir=tmp_path,
        parquet=False,
    )

    assert lab.calls == [["A.SSE"], ["000852.SSE"]]
    assert not result.rolling_metrics.empty
    assert (tmp_path / "rolling_metrics.csv").exists()
    assert (tmp_path / "report.md").exists()
