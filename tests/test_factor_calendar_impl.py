from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from vnpy.alpha.dataset.datasets.factors_templates.factor_calendar_impl.catalog import (
    CalendarFactorCatalog,
    EXPECTED_COUNTS,
    YEARS,
)
from vnpy.alpha.dataset.datasets.factors_templates.factor_calendar_impl.chip_operators import (
    chip_reference_price,
)
from vnpy.alpha.dataset.datasets.factors_templates.factor_calendar_impl.engine import (
    CalendarDataBundle,
    CalendarFactorEngine,
)
from vnpy.alpha.dataset.datasets.factors_templates.factor_calendar_impl.intraday_operators import (
    intraday_autocorr,
    intraday_max_drawdown,
    intraday_return_measure,
    intraday_return_stat,
    intraday_return_turnover_corr,
    intraday_semivariance,
    intraday_tail_share,
    intraday_path_metric,
    intraday_volume_valley_price,
)
from vnpy.alpha.dataset.datasets.factors_templates.factor_calendar_impl.metadata import (
    CalendarFactorDef,
    ImplementationStatus,
)


def test_catalog_counts_names_and_recovered_daily_ids() -> None:
    catalog = CalendarFactorCatalog()
    all_names: list[str] = []
    for year in YEARS:
        definitions = catalog.build_year(year)
        assert len(definitions) == EXPECTED_COUNTS[year]
        assert all(item.calendar_year == year for item in definitions)
        assert all(item.source_file == f"factors{year}.md" for item in definitions)
        assert all(item.title and not item.title.startswith("未恢复标题_") for item in definitions)
        all_names.extend(item.name for item in definitions)
    assert len(all_names) == 1815
    assert len(all_names) == len(set(all_names))
    assert 2025 in YEARS
    definitions = CalendarFactorCatalog().build_all()
    assert not any(
        item.implementation_status == ImplementationStatus.PENDING_IMPLEMENTATION
        for item in definitions if item.calendar_year != 2025
    )


def test_2025_supported_price_volume_batch_and_attention_event_blocker() -> None:
    definitions = CalendarFactorCatalog().build_year(2025)
    expected_exact_ids = {
        "26", "38", "58", "88", "104", "118", "132", "156", "170",
        "219", "240", "251", "259", "268", "297", "305", "358",
    }
    selected = [item for item in definitions if item.calendar_id in expected_exact_ids]
    assert {item.calendar_id for item in selected} == expected_exact_ids
    assert all(
        item.implementation_status == ImplementationStatus.EXACT
        and item.implementation_key
        for item in selected
    )
    attention_event = next(item for item in definitions if item.calendar_id == "263")
    assert attention_event.implementation_status == ImplementationStatus.BLOCKED_DATA
    assert attention_event.required_fields == ("attention_events",)


def test_2026_newly_mapped_factors_are_exact_or_explicit_proxy() -> None:
    expected = {
        "量谷相对加权价格",
        "CSAD模型(基于时序回归变形)",
        "基于换手率的注意力溢出",
        "基于异常收益的注意力溢出",
        "CSAD 模型（基于“小型”板块）",
        "CSAD 模型（基于板块中地位）",
        "趋势资金相对均价因子",
        "净支撑成交量因子",
        "趋势资金净支撑量因子",
        "极端跟随行为_比值因子",
        "极端跟随行为_相关性因子",
    }
    selected = [item for item in CalendarFactorCatalog().build_year(2026) if item.title in expected]
    assert {item.title for item in selected} == expected
    assert all(
        item.implementation_status in {ImplementationStatus.EXACT, ImplementationStatus.PROXY}
        and item.implementation_key
        for item in selected
    )
    proxies = [item for item in selected if item.implementation_status == ImplementationStatus.PROXY]
    assert len(proxies) == 7
    assert all("申万一级" in item.implementation_notes and "中信一级" in item.implementation_notes for item in proxies)


def test_proxy_has_separate_name_and_source_link() -> None:
    catalog = CalendarFactorCatalog()
    original = next(
        item for item in catalog.build_year(2026)
        if item.implementation_status == ImplementationStatus.BLOCKED_DATA
    )
    proxy = catalog.with_proxy(
        original,
        key="alias:close",
        formula="close",
        required_fields=["close"],
        notes="test proxy",
    )
    assert proxy.name.endswith("_proxy")
    assert proxy.proxy_of == original.name
    assert proxy.implementation_status == ImplementationStatus.PROXY
    assert original.implementation_status == ImplementationStatus.BLOCKED_DATA


def _minute_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": [
                datetime(2024, 1, 2, 14, 29), datetime(2024, 1, 2, 14, 30),
                datetime(2024, 1, 2, 14, 59), datetime(2024, 1, 2, 15, 0),
                datetime(2024, 1, 3, 9, 30), datetime(2024, 1, 3, 9, 31),
            ],
            "vt_symbol": ["000001.SZSE"] * 6,
            "close": [10.0, 11.0, 10.0, 12.0, 20.0, 18.0],
            "volume": [1.0, 2.0, 3.0, 4.0, 5.0, 5.0],
            "turnover": [10.0, 22.0, 30.0, 48.0, 100.0, 90.0],
        }
    )


def test_intraday_operators_reset_at_trade_date() -> None:
    minute = _minute_fixture()
    skew = intraday_return_stat(minute, "skew", "value")
    assert skew.height == 2
    # Day two has only one within-day return; the overnight 12 -> 20 move is excluded.
    assert skew.filter(pl.col("datetime") == datetime(2024, 1, 3))["value"][0] is None

    down = intraday_semivariance(minute, "down", "value")
    day_two = down.filter(pl.col("datetime") == datetime(2024, 1, 3))["value"][0]
    assert day_two == pytest.approx(np.log(18.0 / 20.0) ** 2)


def test_intraday_tail_share_uses_each_day_end() -> None:
    result = intraday_tail_share(_minute_fixture(), "turnover", 30, "tail")
    day_one = result.filter(pl.col("datetime") == datetime(2024, 1, 2))["tail"][0]
    assert day_one == pytest.approx((30.0 + 48.0) / 110.0)


def test_intraday_realized_measures_drawdown_and_correlations() -> None:
    minute = _minute_fixture()
    realized = intraday_return_measure(minute, "realized_variance", "value")
    day_one = realized.filter(pl.col("datetime") == datetime(2024, 1, 2))["value"][0]
    expected = sum(np.diff(np.log([10.0, 11.0, 10.0, 12.0])) ** 2)
    assert day_one == pytest.approx(expected)

    drawdown = intraday_max_drawdown(minute, "value")
    assert drawdown.filter(pl.col("datetime") == datetime(2024, 1, 2))["value"][0] == pytest.approx(10 / 11 - 1)
    assert intraday_autocorr(minute, "turnover", 1, "value").height == 2
    assert intraday_return_turnover_corr(
        minute, return_lag=0, turnover_lag=1, name="value"
    ).height == 2


def test_2025_intraday_technical_metrics_are_finite() -> None:
    rows = []
    for day in range(2):
        base = datetime(2024, 1, 2 + day, 9, 30)
        for minute in range(30):
            close = 10 + day + 0.01 * minute + 0.03 * np.sin(minute / 3)
            rows.append(
                (
                    base + timedelta(minutes=minute), "000001.SZSE",
                    close - 0.01, close + 0.02, close - 0.03, close,
                    1000 + 10 * minute, close * (1000 + 10 * minute),
                )
            )
    minute = pl.DataFrame(
        rows,
        schema=[
            "datetime", "vt_symbol", "open", "high", "low", "close",
            "volume", "turnover",
        ],
        orient="row",
    )
    metrics = {
        "calendar_cci", "calendar_rsi", "calendar_br", "calendar_elder",
        "calendar_psy", "calendar_volume_ratio", "calendar_chande",
        "calendar_srmi", "calendar_hurst_rs", "calendar_price_mad",
        "calendar_mfi", "calendar_money_flow", "calendar_pvt", "calendar_emv",
    }
    for metric in metrics:
        result = intraday_path_metric(minute, metric, "value")
        assert result.height == 2
        assert result["value"].is_finite().all()


def test_volume_valley_threshold_uses_only_prior_same_slot_days() -> None:
    rows = []
    start = datetime(2024, 1, 1)
    for day in range(40):
        date = start + timedelta(days=day)
        rows.append((date.replace(hour=9, minute=30), "000001.SZSE", 10.0, 100.0))
        high_volume = 100.0 if day < 20 else float(1000 * 2 ** (day - 20))
        rows.append((date.replace(hour=9, minute=31), "000001.SZSE", 20.0, high_volume))
    minute = pl.DataFrame(
        rows, schema=["datetime", "vt_symbol", "close", "volume"], orient="row"
    )
    result = intraday_volume_valley_price(minute, "value")
    assert result.height == 40
    assert result["value"].is_not_null().sum() == 1
    assert 0 < result["value"].drop_nulls()[0] < 1


def test_chip_reference_price_is_stateful_per_symbol() -> None:
    frame = pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 2), datetime(2024, 1, 3), datetime(2024, 1, 4)],
            "vt_symbol": ["000001.SZSE"] * 3,
            "close": [10.0, 20.0, 30.0],
            "turnover_rate": [0.2, 0.5, 1.2],
        }
    )
    result = chip_reference_price(frame)
    assert result["chip_reference_price"].to_list() == pytest.approx([10.0, 15.0, 30.0])


def test_engine_computes_daily_and_intraday_without_legacy_registration() -> None:
    daily = pl.DataFrame(
        {
            "datetime": [datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "vt_symbol": ["000001.SZSE"] * 2,
            "close": [10.0, 11.0],
            "market_cap_2": [100.0, 110.0],
        }
    )
    definitions = [
        CalendarFactorDef(
            name="calendar_test_cap",
            calendar_year=2023,
            calendar_id="x",
            title="总市值",
            category="规模",
            required_fields=("market_cap_2",),
            implemented_formula="market_cap_2",
            implementation_status=ImplementationStatus.EXACT,
            implementation_key="alias:market_cap_2",
        ),
        CalendarFactorDef(
            name="calendar_test_down",
            calendar_year=2026,
            calendar_id="y",
            title="下行已实现波动率",
            category="高频",
            input_frequency="1m",
            required_fields=("close",),
            implemented_formula="sum(r**2 where r<=0)",
            implementation_status=ImplementationStatus.EXACT,
            implementation_key="intraday_semivariance",
            default_params={"side": "down"},
        ),
    ]
    result = CalendarFactorEngine().compute(
        definitions,
        CalendarDataBundle(daily=daily, minute=_minute_fixture()),
    )
    assert {"calendar_test_cap", "calendar_test_down"}.issubset(result.columns)
    assert result["calendar_test_cap"].to_list() == [100.0, 110.0]


def test_market_residual_operator_uses_rolling_covariance_without_lookahead() -> None:
    dates = [datetime(2024, 1, day) for day in range(1, 9)]
    daily = pl.DataFrame(
        {
            "datetime": dates,
            "vt_symbol": ["000001.SZSE"] * len(dates),
            "close": [10.0, 10.2, 10.1, 10.4, 10.5, 10.3, 10.7, 10.8],
            "market_ret": [None, 0.01, -0.005, 0.015, 0.002, -0.01, 0.012, 0.004],
        }
    )
    definition = CalendarFactorDef(
        name="calendar_test_residual_vol",
        calendar_year=2023,
        calendar_id="z",
        title="月度特质波动率",
        category="波动率",
        required_fields=("close", "market_ret"),
        implemented_formula="std(residual(stock_ret ~ market_ret),3)",
        implementation_status=ImplementationStatus.EXACT,
        implementation_key="market_residual_stat",
        default_params={"window": 3, "stat": "std"},
    )
    result = CalendarFactorEngine().compute([definition], CalendarDataBundle(daily=daily))
    assert result[definition.name].is_not_null().sum() > 0
