import pandas as pd
import pytest

from vnpy.alpha.barra.barra_exposure_analysis import (
    build_topk_equal_weights,
    compute_weighted_barra_exposure,
    run_topk_signal_barra_exposure_analysis,
)


def test_build_topk_equal_weights_is_deterministic_and_normalized() -> None:
    signal = pd.DataFrame(
        {
            "datetime": ["2026-01-02"] * 4 + ["2026-01-05"] * 2,
            "vt_symbol": [
                "000003.SZ",
                "000002.SZ",
                "000001.SZ",
                "000004.SZ",
                "600001.SH",
                "600002.SH",
            ],
            "signal": [1.0, 2.0, 2.0, float("nan"), 3.0, 1.0],
        }
    )

    weights = build_topk_equal_weights(signal, top_k=2)

    assert weights["vt_symbol"].tolist() == [
        "000001.SZSE",
        "000002.SZSE",
        "600001.SSE",
        "600002.SSE",
    ]
    assert weights.groupby("date")["weight"].sum().tolist() == pytest.approx([1.0, 1.0])
    assert weights["weight"].tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_run_topk_signal_barra_exposure_analysis_writes_outputs(tmp_path) -> None:
    signal = pd.DataFrame(
        {
            "datetime": ["2025-12-31"] * 2 + ["2026-01-02"] * 2,
            "vt_symbol": ["A.SSE", "B.SSE"] * 2,
            "signal": [2.0, 1.0, 1.0, 2.0],
        }
    )
    barra = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-12-31"] * 2 + ["2026-01-02"] * 2),
            "vt_symbol": ["A.SSE", "B.SSE"] * 2,
            "beta": [1.0, -1.0, 2.0, -2.0],
            "size": [0.5, -0.5, 1.0, -1.0],
        }
    )
    barra_path = tmp_path / "barra.parquet"
    output_dir = tmp_path / "report"
    barra.to_parquet(barra_path, index=False)

    daily, annual, latest = run_topk_signal_barra_exposure_analysis(
        signal=signal,
        barra_path=barra_path,
        output_dir=output_dir,
        top_k=1,
        factors=["beta", "size"],
    )

    assert daily["beta"].tolist() == pytest.approx([1.0, -2.0])
    assert annual["year"].tolist() == [2025, 2026]
    assert latest.loc[0, "size"] == pytest.approx(-1.0)
    assert (output_dir / "barra_topk_equal_weights.parquet").exists()
    assert (output_dir / "barra_cne5_daily.csv").exists()
    assert (output_dir / "barra_cne5_annual.csv").exists()
    assert (output_dir / "report.md").exists()


def test_barra_missing_values_are_filtered_per_factor() -> None:
    weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
            "vt_symbol": ["A.SSE", "B.SSE"],
            "weight": [0.5, 0.5],
        }
    )
    barra = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
            "vt_symbol": ["A.SSE", "B.SSE"],
            "beta": [1.0, float("nan")],
            "size": [3.0, 5.0],
        }
    )

    daily = compute_weighted_barra_exposure(
        weights,
        barra,
        factors=["beta", "size"],
    )

    assert daily.loc[0, "n_symbols"] == 2
    assert daily.loc[0, "raw_weight_sum"] == pytest.approx(1.0)
    assert daily.loc[0, "beta"] == pytest.approx(1.0)
    assert daily.loc[0, "beta_n_symbols"] == 1
    assert daily.loc[0, "beta_raw_weight_sum"] == pytest.approx(0.5)
    assert daily.loc[0, "size"] == pytest.approx(4.0)
    assert daily.loc[0, "size_n_symbols"] == 2
    assert daily.loc[0, "size_raw_weight_sum"] == pytest.approx(1.0)
