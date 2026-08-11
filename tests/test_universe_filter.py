from datetime import datetime

import polars as pl

from vnpy.alpha.model.prepare_training_samples import prepare_training_samples
from vnpy.alpha.signal.filter_signal_candidates import (
    calculate_top_k_impact,
    filter_signal_candidates,
)


DATE = datetime(2026, 1, 5)


def make_frame(count: int = 10) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": [DATE] * count,
            "vt_symbol": [f"S{i:02d}.SSE" for i in range(count)],
            "signal": [float(count - i) for i in range(count)],
            "label": [float(i) for i in range(count)],
        }
    )


def make_cap(count: int = 10) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": [DATE] * count,
            "vt_symbol": [f"S{i:02d}.SSE" for i in range(count)],
            "cap": [float(i + 1) for i in range(count)],
        }
    )


def make_market(count: int = 10) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": [DATE] * count,
            "vt_symbol": [f"S{i:02d}.SSE" for i in range(count)],
            "volume": [float(count - i) for i in range(count)],
            "is_st": [False] * count,
            "is_suspended": [False] * count,
            "has_bar": [True] * count,
            "is_one_word_limit_up": [False] * count,
            "is_one_word_limit_down": [False] * count,
        }
    )


def test_cap_volume_tails_are_ranked_independently_then_unioned() -> None:
    result = prepare_training_samples(
        make_frame(),
        cap_panel=make_cap(),
        market_panel=make_market(),
        profile="cap10_volume10",
    )

    assert result.filtered["vt_symbol"].to_list() == [f"S{i:02d}.SSE" for i in range(1, 9)]
    assert result.summary["cap_tail_rows"] == 1
    assert result.summary["volume_tail_rows"] == 1
    assert result.summary["tail_overlap_rows"] == 0
    assert result.summary["removed_total_rows"] == 2


def test_training_status_is_deleted_before_tail_rank() -> None:
    frame = make_frame(11)
    cap = make_cap(11)
    market = make_market(11).with_columns(
        pl.when(pl.col("vt_symbol") == "S00.SSE")
        .then(True)
        .otherwise(pl.col("is_st"))
        .alias("is_st")
    )
    result = prepare_training_samples(
        frame,
        cap_panel=cap,
        market_panel=market,
        profile="cap10_volume10",
    )

    assert "S00.SSE" not in result.filtered["vt_symbol"].to_list()
    assert "S01.SSE" not in result.filtered["vt_symbol"].to_list()
    assert result.summary["status_union_rows"] == 1
    assert result.summary["cap_tail_rows"] == 1


def test_cap30_does_not_require_or_filter_volume() -> None:
    result = filter_signal_candidates(
        make_frame(),
        cap_panel=make_cap(),
        volume_panel=None,
        profile="cap30",
    )

    assert result.filtered["vt_symbol"].to_list() == [f"S{i:02d}.SSE" for i in range(3, 10)]
    assert result.summary["cap_tail_rows"] == 3
    assert result.summary["missing_volume_rows"] == 0
    assert result.summary["volume_tail_rows"] == 0


def test_signal_filter_does_not_apply_status_flags() -> None:
    market = make_market().with_columns(pl.lit(True).alias("is_st"))
    result = filter_signal_candidates(
        make_frame(),
        cap_panel=make_cap(),
        volume_panel=market,
        profile="cap10_volume10",
    )

    assert result.summary["status_filter_enabled"] is False
    assert result.summary["status_union_rows"] == 0
    assert result.filtered.height == 8


def test_top_k_impact_counts_original_names_made_ineligible() -> None:
    original = make_frame()
    filtered = original.filter(~pl.col("vt_symbol").is_in(["S00.SSE", "S02.SSE"]))

    impact = calculate_top_k_impact(original, filtered, top_k=5)

    assert impact["original_top_k_count"].item() == 5
    assert impact["removed_count"].item() == 2
    assert impact["removed_rate"].item() == 0.4
