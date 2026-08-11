from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from vnpy.alpha.dataset.processor_post import neutralize_columns


def make_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    dt = datetime(2024, 1, 2)
    symbols = [f"S{index}" for index in range(8)]
    keys = {"datetime": [dt] * len(symbols), "vt_symbol": symbols}
    result = pl.DataFrame(
        {
            **keys,
            "label_h1": [1.0, None, 2.5, 4.0, 3.0, 6.0, 5.5, 8.0],
            "label_h5": [2.0, 1.0, 3.0, 4.5, None, 5.0, 7.0, 9.0],
            "weight": [1.0, 2.0, 0.8, 1.5, 3.0, 1.2, 2.5, 0.7],
        }
    )
    industry = pl.DataFrame(
        {**keys, "industry": ["A", "A", "A", "A", "B", "B", "B", "B"]}
    )
    cap = pl.DataFrame(
        {**keys, "cap": [10.0, 13.0, 18.0, 27.0, 11.0, 16.0, 25.0, 38.0]}
    )
    volume = pl.DataFrame(
        {**keys, "hist_vol_60d": [0.10, 0.18, 0.14, 0.31, 0.21, None, 0.37, 0.28]}
    )
    return result, industry, cap, volume


def expected_residuals(
    result: pl.DataFrame,
    industry: pl.DataFrame,
    cap: pl.DataFrame,
    volume: pl.DataFrame,
    *,
    target: str,
    include_volume: bool,
    weighted: bool,
) -> dict[str, float | None]:
    joined = result.join(industry, on=["datetime", "vt_symbol"]).join(
        cap, on=["datetime", "vt_symbol"]
    )
    if include_volume:
        joined = joined.join(volume, on=["datetime", "vt_symbol"])

    rows = joined.sort("vt_symbol").to_dicts()
    valid_rows: list[dict[str, object]] = []
    for row in rows:
        values = [row[target], row["cap"]]
        if include_volume:
            values.append(row["hist_vol_60d"])
        if all(value is not None and np.isfinite(float(value)) for value in values):
            valid_rows.append(row)

    industries = sorted({str(row["industry"]) for row in valid_rows})
    design_rows: list[list[float]] = []
    targets: list[float] = []
    weights: list[float] = []
    for row in valid_rows:
        design = [float(row["industry"] == name) for name in industries]
        design.append(float(np.log1p(float(row["cap"]))))
        if include_volume:
            design.append(float(row["hist_vol_60d"]))
        design_rows.append(design)
        targets.append(float(row[target]))
        weights.append(float(row["weight"]) if weighted else 1.0)

    design_array = np.asarray(design_rows)
    target_array = np.asarray(targets)
    sqrt_weight = np.sqrt(np.asarray(weights))
    beta = np.linalg.lstsq(
        design_array * sqrt_weight[:, None],
        target_array * sqrt_weight,
        rcond=None,
    )[0]
    residual = target_array - design_array @ beta

    expected: dict[str, float | None] = {str(row["vt_symbol"]): None for row in rows}
    for row, value in zip(valid_rows, residual, strict=True):
        expected[str(row["vt_symbol"])] = float(value)
    return expected


@pytest.mark.parametrize("include_volume", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_neutralize_uses_target_specific_valid_rows(
    include_volume: bool,
    weighted: bool,
) -> None:
    result, industry, cap, volume = make_frames()
    mode = "industry_cap_vol" if include_volume else "industry_cap"
    output = neutralize_columns(
        result,
        target_cols=["label_h1", "label_h5"],
        industry_df=industry,
        cap_df=cap,
        vol_df=volume,
        mode=mode,
        weight_col="weight" if weighted else None,
        fill_missing=None,
        suffix="_neu",
    ).sort("vt_symbol")

    for target in ["label_h1", "label_h5"]:
        expected = expected_residuals(
            result,
            industry,
            cap,
            volume,
            target=target,
            include_volume=include_volume,
            weighted=weighted,
        )
        actual = dict(
            zip(
                output["vt_symbol"].to_list(),
                output[f"{target}_neu"].to_list(),
                strict=True,
            )
        )
        for symbol, expected_value in expected.items():
            if expected_value is None:
                assert actual[symbol] is None
            else:
                assert actual[symbol] == pytest.approx(expected_value, abs=1e-10)

