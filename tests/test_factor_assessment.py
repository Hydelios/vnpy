from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from vnpy.alpha.analysis.factor_assessment import (
    PREPROCESS_VERSION,
    AssessmentConfig,
    prepare_assessment_frame,
    run_factor_assessment,
)


def make_master_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2024, 1, 2)
    for day in range(5):
        for asset in range(4):
            value = float(asset + day / 10)
            rows.append(
                {
                    "datetime": start + timedelta(days=day),
                    "vt_symbol": f"asset_{asset}",
                    "close": 10.0 + asset,
                    "turnover": 1000.0 + asset,
                    "factor_good": value,
                    "factor_sparse": value if day == 0 else None,
                    "label_o2o_h1": value * 0.01,
                    "label_o2o_h3": value * 0.02,
                    "label_o2o_h5": value * 0.03,
                    "eligible_sample": asset != 3,
                }
            )
    return pl.DataFrame(rows)


def test_v1_defaults_are_full_and_quality_gated() -> None:
    config = AssessmentConfig.from_mapping(None)

    assert config.scope == "full"
    assert config.min_coverage == 0.8
    assert config.min_valid_day_ratio == 0.8


def test_prepare_assessment_frame_casts_float64_and_nulls_non_finite() -> None:
    master = make_master_frame().with_columns(
        pl.when(pl.col("vt_symbol") == "asset_0")
        .then(float("inf"))
        .otherwise(pl.col("factor_good"))
        .alias("factor_good")
    )
    config = AssessmentConfig.from_mapping({"sample_filter_column": "eligible_sample"})

    frame = prepare_assessment_frame(
        master_df=master,
        feature_cols=["factor_good"],
        label_cols=list(config.label_columns),
        valid_period=None,
        config=config,
    )

    assert frame.height == 15
    assert frame.schema["factor_good"] == pl.Float64
    assert frame["factor_good"].null_count() == 5
    assert not frame["factor_good"].is_infinite().any()


def test_run_factor_assessment_marks_sparse_factors(tmp_path) -> None:
    result = run_factor_assessment(
        master_df=make_master_frame(),
        feature_cols=["factor_good", "factor_sparse"],
        valid_period=None,
        output_dir=tmp_path,
        config={"scope": "full", "html_output": "none"},
        run_name="quality_gate",
    )

    summary = result.summary_df.sort("feature")
    good = summary.filter(pl.col("feature") == "factor_good").row(0, named=True)
    sparse = summary.filter(pl.col("feature") == "factor_sparse").row(0, named=True)
    assert good["eligible"] is True
    assert good["assessment_group"] == "regular"
    assert sparse["eligible"] is False
    assert sparse["assessment_group"] == "sparse"
    assert "coverage<80%" in sparse["exclusion_reason"]
    assert result.artifacts_meta["preprocess_version"] == PREPROCESS_VERSION
    assert result.artifacts_meta["eligible_feature_count"] == 1
    assert result.artifacts_meta["sparse_feature_count"] == 1
