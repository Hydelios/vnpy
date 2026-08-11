from __future__ import annotations

import os
from datetime import datetime, timedelta

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib
import polars as pl
import pytest

matplotlib.use("Agg")

from vnpy.alpha.analysis.alphainspect_backend import (
    create_factor_sheet,
    create_ic_report,
    prepare_alphainspect_input,
)


def make_analysis_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2024, 1, 2)
    for day_index in range(8):
        date = start + timedelta(days=day_index)
        for asset_index in range(18):
            signal = float(asset_index + (day_index % 3) * ((asset_index % 4) - 1.5))
            rows.append(
                {
                    "date": date,
                    "asset": f"asset_{asset_index:02d}",
                    "signal": signal,
                    "FWD_RET_OO_1": signal * 0.001 + (asset_index % 3) * 0.0001,
                }
            )
    return pl.DataFrame(rows)


class FakeLab:
    def __init__(self, price_panel: pl.DataFrame) -> None:
        self.price_panel = price_panel
        self.load_kwargs: dict[str, object] = {}

    def load_bar_df(self, **kwargs) -> pl.DataFrame:
        self.load_kwargs = kwargs
        return self.price_panel


def test_create_factor_sheet_returns_and_saves(tmp_path) -> None:
    frame = make_analysis_frame()
    output_path = tmp_path / "alphainspect.png"

    result = create_factor_sheet(
        frame,
        factor="signal",
        forward_return="FWD_RET_OO_1",
        quantiles=9,
        turnover_periods=(1, 2),
        title="测试信号",
        output_path=output_path,
    )

    assert output_path.is_file()
    assert "factor_quantile" in result.quantile_frame.columns
    assert "factor_quantile" not in frame.columns
    assert result.figure._suptitle is not None
    assert result.figure._suptitle.get_text() == "测试信号"
    assert result.cumulative_return.height == 8


def test_create_factor_sheet_validates_required_columns() -> None:
    frame = make_analysis_frame().drop("asset")

    with pytest.raises(ValueError, match="asset"):
        create_factor_sheet(frame, factor="signal")


def test_create_ic_report_returns_and_saves_all_tables(tmp_path) -> None:
    frame = make_analysis_frame().with_columns(
        (pl.col("FWD_RET_OO_1") * 2).alias("FWD_RET_OO_2")
    )

    result = create_ic_report(
        frame,
        factor="signal",
        forward_returns=["FWD_RET_OO_1", "FWD_RET_OO_2"],
        method="rank_ic",
        output_dir=tmp_path,
    )

    assert result.summary.index.tolist() == ["FWD_RET_OO_1", "FWD_RET_OO_2"]
    assert result.summary.columns.tolist() == [
        "RankIC",
        "RankICIR",
        "win_rate",
        "t_stat",
        "p_value",
    ]
    assert result.daily.columns == [
        "date",
        "signal__FWD_RET_OO_1",
        "signal__FWD_RET_OO_2",
    ]
    assert result.annual.index.name == "year"
    assert (tmp_path / "rank_ic_summary.csv").is_file()
    assert (tmp_path / "rank_ic_daily.parquet").is_file()
    assert (tmp_path / "rank_ic_annual.csv").is_file()


def test_create_ic_report_validates_required_columns() -> None:
    with pytest.raises(ValueError, match="FWD_RET_OO_2"):
        create_ic_report(
            make_analysis_frame(),
            factor="signal",
            forward_returns=["FWD_RET_OO_2"],
        )


def test_prepare_alphainspect_input_builds_t1_open_returns() -> None:
    dates = [datetime(2024, 1, 2) + timedelta(days=day) for day in range(4)]
    signal = pl.DataFrame(
        {
            "datetime": dates[:2],
            "vt_symbol": ["A.SSE", "A.SSE"],
            "signal": [1.0, 2.0],
        }
    )
    prices = pl.DataFrame(
        {
            "datetime": dates,
            "vt_symbol": ["A.SSE"] * 4,
            "open": [10.0, 11.0, 12.1, 13.31],
        }
    )
    lab = FakeLab(prices)

    result = prepare_alphainspect_input(
        lab,
        signal,
        horizons=(1, 2),
        extended_days=3,
    )

    assert result.forward_returns == ("FWD_RET_OO_1", "FWD_RET_OO_2")
    assert result.analysis_frame.columns == [
        "date",
        "asset",
        "signal",
        "FWD_RET_OO_1",
        "FWD_RET_OO_2",
    ]
    first = result.analysis_frame.row(0, named=True)
    assert first["FWD_RET_OO_1"] == pytest.approx(0.1)
    assert first["FWD_RET_OO_2"] == pytest.approx(0.1)
    assert result.coverage["FWD_RET_OO_1_rows"][0] == 2
    assert lab.load_kwargs["vt_symbols"] == ["A.SSE"]
