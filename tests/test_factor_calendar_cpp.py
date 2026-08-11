from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from vnpy.alpha.dataset.datasets.factors_templates.factor_calendar_impl.catalog import (
    CalendarFactorCatalog,
)
from vnpy.alpha.dataset.datasets.factors_templates.factor_calendar_impl.cpp_backend.backend import (
    CALENDAR_2025_SPECIAL_IDS,
    CalendarCppBackend,
)
from vnpy.alpha.dataset.datasets.factors_templates.factor_calendar_impl.cpp_backend.library import (
    classify_cpp_coverage,
)
from vnpy.alpha.dataset.datasets.factors_templates.factor_calendar_impl.engine import (
    CalendarDataBundle,
    CalendarFactorEngine,
)


SPECIAL_TITLES = {
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


def _business_dates(count: int) -> list[datetime]:
    values = []
    value = datetime(2023, 1, 2)
    while len(values) < count:
        if value.weekday() < 5:
            values.append(value)
        value += timedelta(days=1)
    return values


def _special_fixture() -> CalendarDataBundle:
    rng = np.random.default_rng(7)
    dates = _business_dates(140)
    symbols = [f"{index:06d}.SZSE" for index in range(1, 13)]
    daily_rows = []
    minute_rows = []
    for symbol_index, symbol in enumerate(symbols):
        price = 10.0 + symbol_index
        for day_index, date in enumerate(dates):
            price *= np.exp(
                0.0002 + 0.008 * rng.normal() + 0.003 * np.sin(day_index / 8 + symbol_index)
            )
            cap = 1e9 * (1 + symbol_index / 3) * (1 + day_index / 1000)
            daily_rows.append((
                date, symbol, price, 1e6 * (1 + rng.random()), cap,
                1.2 + 0.01 * symbol_index + 0.001 * day_index,
                f"I{symbol_index % 2}",
            ))
            for minute_index in range(40):
                stamp = date.replace(
                    hour=9 + (30 + minute_index) // 60,
                    minute=(30 + minute_index) % 60,
                )
                minute_price = price * (
                    1 + 0.002 * np.sin(minute_index / 4 + symbol_index) + 0.0005 * rng.normal()
                )
                minute_volume = 1000 * (
                    1 + 0.5 * rng.random() + 0.3 * np.sin(minute_index + day_index)
                )
                minute_rows.append((
                    stamp, symbol, minute_price * 0.9998, minute_price * 1.001,
                    minute_price * 0.999, minute_price, minute_volume,
                    minute_price * minute_volume,
                ))
    daily = pl.DataFrame(
        daily_rows,
        schema=[
            "datetime", "vt_symbol", "close", "volume",
            "a_share_market_val_in_circulation", "pb_ratio_ttm", "first_industry_code",
        ],
        orient="row",
    )
    minute = pl.DataFrame(
        minute_rows,
        schema=[
            "datetime", "vt_symbol", "open", "high", "low", "close", "volume", "turnover",
        ],
        orient="row",
    )
    return CalendarDataBundle(daily=daily, minute=minute)


def test_cpp_daily_library_contains_286_compilable_rows() -> None:
    coverage = classify_cpp_coverage()
    assert len(coverage.expressions) == 286
    assert len(coverage.unsupported) == 407


def test_titan_calendar_registry_owns_314_prefixed_factors() -> None:
    backend = CalendarCppBackend()
    assert len(backend.available_daily_expressions()) == 286
    assert len(backend.available_custom_factors()) == 28
    assert len(backend.available_factors()) == 314
    assert all(name.startswith("calendar_") for name in backend.available_factors())
    assert backend.custom_kernels.names() == (
        "calendar_special_2025",
        "calendar_special_2026",
    )


def test_cpp_daily_jit_matches_python_for_alias_return_and_rolling() -> None:
    catalog = CalendarFactorCatalog()
    candidates = catalog.build_all()
    definitions = [
        next(item for item in candidates if item.implementation_key == "alias:market_cap_2"),
        next(item for item in candidates if item.implementation_key == "return"),
        next(item for item in candidates if item.implementation_key == "rolling_return_stat"),
    ]
    rows = []
    for symbol_index, symbol in enumerate(["000001.SZSE", "000002.SZSE", "600000.SSE"]):
        close = 10.0 + symbol_index
        for day_index, date in enumerate(_business_dates(160)):
            close *= np.exp(0.001 + 0.002 * np.sin(day_index / 7 + symbol_index))
            rows.append(
                (
                    date, symbol, close * 0.999, close * 1.01, close * 0.99, close,
                    1e6 + day_index, close * (1e6 + day_index), 1e9 + symbol_index * 1e8,
                )
            )
    daily = pl.DataFrame(
        rows,
        schema=[
            "datetime", "vt_symbol", "open", "high", "low", "close", "volume",
            "turnover", "market_cap_2",
        ],
        orient="row",
    ).with_columns((pl.col("turnover") / pl.col("volume")).alias("vwap"))
    python_result = CalendarFactorEngine().compute(definitions, CalendarDataBundle(daily=daily))
    backend = CalendarCppBackend(num_threads=2)
    cpp_result = backend.compute_daily(
        daily, [item.name for item in definitions]
    )
    merged = python_result.join(cpp_result, on=["datetime", "vt_symbol"])
    for definition in definitions:
        left = merged[definition.name].to_numpy()
        right = merged[backend.titan_name(definition.name)].to_numpy()
        valid = np.isfinite(left) & np.isfinite(right)
        assert valid.sum() > 0
        np.testing.assert_allclose(left[valid], right[valid], rtol=1e-10, atol=1e-12)
        assert np.array_equal(np.isfinite(left), np.isfinite(right))


def test_cpp_bundle_injects_daily_pit_fundamental_fields(tmp_path) -> None:
    lab = tmp_path / "lab"
    (lab / "daily").mkdir(parents=True)
    (lab / "common").mkdir()
    symbol = "000001.SZSE"
    dates = [datetime(2024, 3, 28), datetime(2024, 3, 29), datetime(2024, 4, 1)]
    pl.DataFrame(
        {
            "datetime": dates,
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1],
            "close": [10.1, 10.2, 10.3],
            "volume": [100.0, 110.0, 120.0],
            "turnover": [1010.0, 1122.0, 1236.0],
        }
    ).write_parquet(lab / "daily" / f"{symbol}.parquet")
    (lab / "common" / "factors_financial.csv").write_text(
        "trade_date,vt_symbol,return_on_equity_ttm\n"
        f"2024-03-28,{symbol},0.10\n"
        f"2024-03-29,{symbol},0.10\n"
        f"2024-04-01,{symbol},0.12\n",
        encoding="utf-8",
    )
    definition = next(
        item for item in CalendarFactorCatalog().build_all()
        if item.implementation_key == "alias:return_on_equity_ttm"
    )
    bundle = CalendarCppBackend().prepare_pit_bundle(
        lab,
        [symbol],
        start="2024-03-28",
        end="2024-04-01",
        definitions=[definition],
    )
    assert bundle.daily["return_on_equity_ttm"].to_list() == [0.10, 0.10, 0.12]


def test_cpp_special_factors_match_python_reference() -> None:
    bundle = _special_fixture()
    definitions = [
        item for item in CalendarFactorCatalog().build_year(2026)
        if item.title in SPECIAL_TITLES
    ]
    python_result = CalendarFactorEngine().compute(definitions, bundle, strict=True)
    backend = CalendarCppBackend(num_threads=4)
    cpp_result = backend.compute_special(bundle)
    merged = python_result.join(
        cpp_result,
        on=["datetime", "vt_symbol"],
        how="inner",
    )
    for definition in definitions:
        left = merged[definition.name].to_numpy()
        right = merged[backend.titan_name(definition.name)].to_numpy()
        valid = np.isfinite(left) & np.isfinite(right)
        assert valid.sum() > 0, definition.title
        np.testing.assert_allclose(left[valid], right[valid], rtol=1e-8, atol=1e-10)
        assert np.array_equal(np.isfinite(left), np.isfinite(right)), definition.title


def test_cpp_special_incremental_warmup_matches_full_last_day() -> None:
    bundle = _special_fixture()
    backend = CalendarCppBackend(num_threads=4)
    full = backend.compute_special(bundle)
    dates = bundle.daily["datetime"].unique().sort().to_list()
    start = dates[-121]
    incremental_bundle = CalendarDataBundle(
        daily=bundle.daily.filter(pl.col("datetime") >= start),
        minute=bundle.minute.filter(pl.col("datetime").dt.date() >= start.date()),
    )
    incremental = backend.compute_special(incremental_bundle)
    last = dates[-1]
    left = full.filter(pl.col("datetime") == last).sort("vt_symbol")
    right = incremental.filter(pl.col("datetime") == last).sort("vt_symbol")
    assert left["vt_symbol"].to_list() == right["vt_symbol"].to_list()
    for column in full.columns[2:]:
        full_values = left[column].to_numpy()
        incremental_values = right[column].to_numpy()
        valid = np.isfinite(full_values) & np.isfinite(incremental_values)
        np.testing.assert_allclose(
            full_values[valid], incremental_values[valid], rtol=1e-8, atol=1e-10
        )
        assert np.array_equal(np.isfinite(full_values), np.isfinite(incremental_values)), column


def test_cpp_2025_special_factors_match_python_reference() -> None:
    bundle = _special_fixture()
    definitions = [
        item for item in CalendarFactorCatalog().build_year(2025)
        if item.calendar_id in CALENDAR_2025_SPECIAL_IDS
    ]
    python_result = CalendarFactorEngine().compute(definitions, bundle, strict=True)
    backend = CalendarCppBackend(num_threads=4)
    cpp_result = backend.compute_2025_special(bundle)
    merged = python_result.join(cpp_result, on=["datetime", "vt_symbol"])
    for definition in definitions:
        left = merged[definition.name].to_numpy()
        right = merged[backend.titan_name(definition.name)].to_numpy()
        valid = np.isfinite(left) & np.isfinite(right)
        assert valid.sum() > 0, f"{definition.calendar_id} {definition.title}"
        np.testing.assert_allclose(left[valid], right[valid], rtol=1e-8, atol=1e-10)
        assert np.array_equal(np.isfinite(left), np.isfinite(right)), definition.title


def test_cpp_2025_special_threads_and_incremental_are_deterministic() -> None:
    bundle = _special_fixture()
    single = CalendarCppBackend(num_threads=1).compute_2025_special(bundle)
    parallel = CalendarCppBackend(num_threads=4).compute_2025_special(bundle)
    for column in single.columns[2:]:
        np.testing.assert_array_equal(single[column].to_numpy(), parallel[column].to_numpy())

    dates = bundle.daily["datetime"].unique().sort().to_list()
    start = dates[-121]
    incremental_bundle = CalendarDataBundle(
        daily=bundle.daily.filter(pl.col("datetime") >= start),
        minute=bundle.minute.filter(pl.col("datetime").dt.date() >= start.date()),
    )
    incremental = CalendarCppBackend(num_threads=4).compute_2025_special(incremental_bundle)
    last = dates[-1]
    left = single.filter(pl.col("datetime") == last).sort("vt_symbol")
    right = incremental.filter(pl.col("datetime") == last).sort("vt_symbol")
    for column in single.columns[2:]:
        left_values = left[column].to_numpy()
        right_values = right[column].to_numpy()
        valid = np.isfinite(left_values) & np.isfinite(right_values)
        np.testing.assert_allclose(left_values[valid], right_values[valid], rtol=1e-8, atol=1e-10)
        assert np.array_equal(np.isfinite(left_values), np.isfinite(right_values)), column
